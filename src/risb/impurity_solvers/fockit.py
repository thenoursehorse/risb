# Copyright (c) 2026 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Impurity solver using exact diagonalization from fockit."""

import warnings
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, override

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from fockit.algebra import (
    Ladder,
    OpSum,
    c,
    cdag,
    commutator,
    hc,
    nc,
    normal_order,
    zero,
)
from fockit.hilbert import (
    Basis,
    FermionSpace,
    build_basis,
    expand_state,
    expectation,
    symmetry_components,
    to_matrix,
)
from fockit.hilbert.symmetries import (
    OnsiteSymmetry,
    PermutationSymmetry,
    symmetry_phases,
)
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence, aslinearoperator, eigsh

from risb.helpers import get_entropy_impurity
from risb.types import BlockMatrix, BlockMatrixLike, GfStruct, ImpuritySolver

# Boltzmann weight, relative to the ground state, below which a state is dropped
_WEIGHT_TOL = 1e-14

# Largest sector that a thermal solve diagonalizes as a dense matrix
_DENSE_DIM = 2000

# Times eigsh is retried with a doubled ncv when it does not converge
_ARPACK_RETRIES = 2

# Energy below the highest state found by eigsh at which a state was missed
_MISSED_TOL = 1e-10

# A symmetry sector: the values of the diagonal symmetries, None where a
# sector does not fix one, and the eigenvalues of the permutation and on-site
# symmetries, or None without them
type _Key = tuple[tuple[float | None, ...], tuple[complex, ...] | None]

# The annihilators of the modes into one sector they reach: the indices of the
# modes and the matrices
type _Annihilators = tuple[list[int], list[Any]]


class _FockitBase(ImpuritySolver):
    # The impurity Hamiltonian, density matrices and expectation values of the
    # fockit solvers. A subclass finds the states in _find_states

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        conserved: Sequence[OpSum],
        bath_size: Mapping[str, int] | None,
        permutation_symmetries: Sequence[PermutationSymmetry] = (),
        onsite_symmetries: Sequence[OnsiteSymmetry] = (),
    ) -> None:
        #: fockit.algebra.OpSum : Interaction Hamiltonian.
        self.h_int = h_int

        #: risb.types.GfStruct : Block matrix structure of the physical orbitals
        #: (c-electrons).
        self.gf_struct = gf_struct

        #: risb.types.GfStruct : Block matrix structure of the bath orbitals
        #: (f-electrons).
        self.gf_struct_bath = [
            (_bath(bl), (bath_size or {}).get(bl, bl_size)) for bl, bl_size in gf_struct
        ]

        modes = [
            (bl, i)
            for bl, bl_size in self.gf_struct + self.gf_struct_bath
            for i in range(bl_size)
        ]
        if len(modes) % 2:
            msg = f"The impurity problem has {len(modes)} orbitals, so it has no half-filled sector."
            raise ValueError(msg)
        N = zero()
        for mode in modes:
            N += nc(*mode)
        self._modes = modes
        self._index = {mode: j for j, mode in enumerate(modes)}
        self._N = N
        self._permutation = list(permutation_symmetries)
        self._onsite = list(onsite_symmetries)

        # The diagonal symmetries, the particle number first, and how much the
        # annihilator of each mode lowers each, or None if the sectors it
        # reaches do not have one value of it
        self._diagonal = [N, *conserved]
        self._shifts = [self._shift(Q) for Q in self._diagonal]

        # Built once: the bases of the sectors, the annihilators from them,
        # and the matrices of h_int in them
        self._bases: dict[_Key, Basis] = {}
        self._annihilators: dict[_Key, list[_Annihilators] | None] = {}
        self._h_int_matrices: dict[_Key, Any] = {}

        #: float : Lowest energy of the impurity problem, set by :meth:`solve`.
        self.gs_energy: float

        #: numpy.ndarray : Weights of the states in the expectation values, in
        #: the order of :attr:`ensemble`, set by :meth:`solve`.
        self.weights: NDArray[np.float64]

        #: list of tuple : The states that the expectation values average
        #: over, as ``(basis, states, weights)`` with the states as columns in
        #: the basis, set by :meth:`solve`.
        self.ensemble: list[tuple[Basis, NDArray[np.complex128], NDArray[np.float64]]]

        #: float : Entropy ``-sum(weights * log(weights))`` of the states of
        #: the impurity problem, set by :meth:`solve`. It is zero for the
        #: ground state.
        self.entropy: float

        #: fockit.algebra.OpSum : Impurity Hamiltonian. It is the sum of
        #: :attr:`h0_loc`, :attr:`h_int`, :attr:`h_hybr`, and :attr:`h_bath`.
        self.h_imp: OpSum = zero()

        #: fockit.algebra.OpSum : Single-particle quadratic couplings of
        #: c-electron terms in :attr:`h_imp`.
        self.h0_loc: OpSum = zero()

        #: fockit.algebra.OpSum : Bath terms in :attr:`h_imp`.
        self.h_bath: OpSum = zero()

        #: fockit.algebra.OpSum : Hybridization terms in :attr:`h_imp`.
        self.h_hybr: OpSum = zero()

        #: risb.types.BlockMatrix : f-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_f: BlockMatrix = {}

        #: risb.types.BlockMatrix : c-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_c: BlockMatrix = {}

        #: risb.types.BlockMatrix : Density matrix of hybridization terms
        #: (c- and f-electrons), set by :meth:`solve`.
        self.rho_cf: BlockMatrix = {}

    def set_h0_loc(self, h0_loc_matrix: BlockMatrixLike) -> None:
        """
        Set the single-particle quadratic couplings of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h0_loc_matrix : risb.types.BlockMatrixLike
            Quadratic terms as a matrix. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        self.h0_loc = zero()
        for bl, bl_size in self.gf_struct:
            self.h0_loc += _bilinear(
                [cdag(bl, i) for i in range(bl_size)],
                h0_loc_matrix[bl],
                [c(bl, i) for i in range(bl_size)],
            )

    def set_h_int(self, h_int: OpSum) -> None:
        """
        Set the interaction terms of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h_int : fockit.algebra.OpSum
            Interaction Hamiltonian of the physical orbitals.

        """
        self.h_int = h_int
        self._h_int_matrices = {}

    def set_h_bath(self, Lambda_c: BlockMatrixLike) -> None:
        """
        Set the bath terms in the impurity Hamiltonian.

        Parameters
        ----------
        Lambda_c : risb.types.BlockMatrixLike
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        self.h_bath = zero()
        for (bl, _), (bl_bath, n_bath) in zip(
            self.gf_struct, self.gf_struct_bath, strict=True
        ):
            self.h_bath += _bilinear(
                [c(bl_bath, a) for a in range(n_bath)],
                Lambda_c[bl],
                [cdag(bl_bath, a) for a in range(n_bath)],
            )

    def set_h_hybr(self, D: BlockMatrixLike) -> None:
        """
        Set the hybridization terms in the impurity Hamiltonian.

        Parameters
        ----------
        D : risb.types.BlockMatrixLike
            Hybridization coupling, indexed as ``D[bl][a, alpha]`` with ``a`` a
            bath orbital and ``alpha`` a physical orbital. The term is
            ``sum D[bl][a, alpha] c^dag(bl, alpha) c(bath, a) + h.c.``. Each key
            in dictionary must follow :attr:`gf_struct`.

        """
        self.h_hybr = zero()
        for (bl, bl_size), (bl_bath, n_bath) in zip(
            self.gf_struct, self.gf_struct_bath, strict=True
        ):
            tmp = _bilinear(
                [cdag(bl, alpha) for alpha in range(bl_size)],
                np.asarray(D[bl]).T,
                [c(bl_bath, a) for a in range(n_bath)],
            )
            self.h_hybr += tmp + hc(tmp)

    @override
    def set_h_imp(
        self,
        Lambda_c: BlockMatrixLike,
        D: BlockMatrixLike,
        h0_loc_matrix: BlockMatrixLike | None = None,
        mu: float | None = None,
    ) -> None:
        """
        Set the terms of the impurity Hamiltonian.

        Parameters
        ----------
        Lambda_c : risb.types.BlockMatrixLike
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        D : risb.types.BlockMatrixLike
            Hybridization coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        h0_loc_matrix : risb.types.BlockMatrixLike, optional
            Single-particle quadratic couplings of the c-electrons. Each key
            in dictionary must follow :attr:`gf_struct`.
        mu : float, optional
            Chemical potential, subtracted as ``mu * n`` on the physical and
            bath orbitals.

        """
        if h0_loc_matrix is not None:
            self.set_h0_loc(h0_loc_matrix)
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        self.h_imp = self.h0_loc + self.h_int + self.h_bath + self.h_hybr
        if mu is not None:
            self.h_imp -= mu * self._N

    @override
    def solve(self) -> None:
        """
        Solve the impurity problem.

        Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block.

        Raises
        ------
        ValueError
            If :attr:`h_imp` is not Hermitian, or does not have the symmetries
            of the basis.

        """
        for Q in self._diagonal[1:]:
            terms = normal_order(commutator(self.h_imp, Q)).terms.values()
            if any(abs(complex(coeff)) > 1e-10 for coeff in terms):
                msg = "h_imp does not conserve a diagonal symmetry."
                raise ValueError(msg)
        keys = self._find_states()
        self.entropy = get_entropy_impurity(self.weights)

        # <c_i^dag c_j> of every pair of modes, as the sum over the sectors
        # c_j reaches of (c_i psi)^dag (c_j psi)
        G = np.zeros((len(self._modes), len(self._modes)), dtype=complex)
        for key, (_, states, weights) in zip(keys, self.ensemble, strict=True):
            scaled = states * np.sqrt(weights)
            annihilators = self._annihilators_of(key)
            assert annihilators is not None  # the states are in a basis with them
            for modes, matrices in annihilators:
                A = np.stack([(C @ scaled).ravel() for C in matrices])
                G[np.ix_(modes, modes)] += A.conj() @ A.T

        for (bl, bl_size), (bl_bath, n_bath) in zip(
            self.gf_struct, self.gf_struct_bath, strict=True
        ):
            cs = [self._index[(bl, alpha)] for alpha in range(bl_size)]
            fs = [self._index[(bl_bath, a)] for a in range(n_bath)]
            self.rho_f[bl] = np.eye(n_bath) - G[np.ix_(fs, fs)]
            self.rho_cf[bl] = G[np.ix_(cs, fs)]
            self.rho_c[bl] = G[np.ix_(cs, cs)]

    @abstractmethod
    def _find_states(self) -> list[_Key]:
        """Set gs_energy, weights and ensemble, and return the sector of each entry of ensemble."""

    def _shift(self, Q: OpSum) -> NDArray[np.float64] | None:
        # q_j with [Q, c_j] = -q_j c_j, if Q is a sum of number operators that
        # the permutation and on-site symmetries keep
        T, _, rest = _one_body(Q, self._index)
        if rest.terms or np.any(T - np.diag(np.diag(T))):
            return None
        if self._permutation or self._onsite:
            try:
                phases = symmetry_phases(Q, self._permutation, self._onsite)
            except ValueError:
                return None
            if not np.allclose(phases, 1):
                return None
        return np.diag(T).real

    def _basis(self, key: _Key) -> Basis:
        if key not in self._bases:
            values, eigenvalues = key
            n_permutation = len(self._permutation)
            permutation, onsite = [], []
            if eigenvalues is not None:
                permutation = [
                    (m, e)
                    for (m, _), e in zip(
                        self._permutation, eigenvalues[:n_permutation], strict=True
                    )
                ]
                onsite = [
                    (f, e)
                    for (f, _), e in zip(
                        self._onsite, eigenvalues[n_permutation:], strict=True
                    )
                ]
            diagonal = {
                Q: v
                for Q, v in zip(self._diagonal, values, strict=True)
                if v is not None
            }
            self._bases[key] = build_basis(
                {FermionSpace(): self._modes},
                diagonal_symmetries=diagonal,
                permutation_symmetries=permutation,
                onsite_symmetries=onsite,
            )
        return self._bases[key]

    def _annihilators_of(self, key: _Key) -> list[_Annihilators] | None:
        # The matrices of the annihilators of the modes from the sector into
        # each sector they reach, or None if the permutation and on-site
        # symmetries do not keep the particle number, or do not split them
        # into parts that reach one sector each
        if key not in self._annihilators:
            values, eigenvalues = key
            if eigenvalues is not None and self._shifts[0] is None:
                self._annihilators[key] = None
                return None
            source = self._basis(key)
            reached: dict[_Key, _Annihilators] = {}
            for j, mode in enumerate(self._modes):
                target_values = tuple(
                    None if v is None or shift is None else round(v - shift[j], 8)
                    for v, shift in zip(values, self._shifts, strict=True)
                )
                parts: dict[tuple[complex, ...] | None, OpSum] = {None: c(*mode)}
                if eigenvalues is not None:
                    try:
                        parts = dict(symmetry_components(c(*mode), source).items())
                    except ValueError:
                        self._annihilators[key] = None
                        return None
                for target_eigenvalues, part in parts.items():
                    target = (target_values, target_eigenvalues)
                    modes, matrices = reached.setdefault(target, ([], []))
                    modes.append(j)
                    matrices.append(
                        to_matrix(part, self._basis(target), source=source).tocsr()
                    )
            self._annihilators[key] = list(reached.values())
        return self._annihilators[key]

    def _hamiltonian(
        self, key: _Key, T: NDArray[np.inexact], const: float, rest: OpSum
    ) -> Any:
        # h_int + const + sum_ij T_ij c_i^dag c_j + rest in the sector, with
        # the one-body terms from the annihilators if it has them
        basis = self._basis(key)
        annihilators = self._annihilators_of(key)
        if annihilators is None:
            return _hermitian_matrix(self.h_imp, basis)
        if key not in self._h_int_matrices:
            self._h_int_matrices[key] = _hermitian_matrix(self.h_int, basis)
        H = self._h_int_matrices[key] + const * sp.identity(basis.dim, format="csr")
        if rest.terms:
            H = H + _hermitian_matrix(rest, basis)
        for modes, matrices in annihilators:
            block = T[np.ix_(modes, modes)]
            for a, C in enumerate(matrices):
                nonzero = np.flatnonzero(block[a])
                if len(nonzero):
                    Y = sum(block[a, b] * matrices[b] for b in nonzero)
                    H = H + C.conj().T @ Y
        return H.tocsr()

    @override
    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        """
        Return f-electron density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The f-electron density matrix :attr:`rho_f` from the impurity problem.

        """
        return self.rho_f[bl]

    @override
    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        """
        Return c-electron density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The c-electron density matrix :attr:`rho_c` from the impurity problem.

        """
        return self.rho_c[bl]

    @override
    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        """
        Return the cf hybridization (off-diagonal) density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The c,f-electron hybridization density matrix :attr:`rho_cf` from
            the impurity problem.

        """
        return self.rho_cf[bl]

    def overlap(self, Op: OpSum, force_real: bool = True) -> float | complex:
        """
        Calculate the expectation value of an operator in the states of the impurity problem.

        Parameters
        ----------
        Op : fockit.algebra.OpSum
            Operator to take expectation of. The bath modes are
            ``("bath_" + block, index)``.
        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        float or complex
            Expectation value.

        """
        res = 0j
        for basis, states, weights in self.ensemble:
            # The parts of Op that leave the sector have no expectation value
            part = Op
            if len(basis.group) > 1 or basis.onsite_symmetries:
                eigenvalues = [
                    e
                    for _, e in (
                        *basis.permutation_symmetries,
                        *basis.onsite_symmetries,
                    )
                ]
                part = zero()
                for key, piece in symmetry_components(Op, basis).items():
                    if np.allclose(key, eigenvalues):
                        part = piece
            Op_matrix = to_matrix(part, basis)
            res += complex(
                sum(
                    weight * expectation(Op_matrix, state)
                    for weight, state in zip(weights, states.T, strict=True)
                )
            )
        if force_real:
            return res.real
        return res


class FockitImpuritySolver(_FockitBase):
    """
    Impurity solver using exact diagonalization from fockit.

    Parameters
    ----------
    h_int : fockit.algebra.OpSum
        Interaction Hamiltonian of the physical orbitals. Its modes are
        ``(block, index)``, e.g. ``nc("up", 0) * nc("dn", 0)``.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    method : {"eigsh", "eigh"}, optional
        Find the ground state with :func:`scipy.sparse.linalg.eigsh` or with
        :func:`scipy.linalg.eigh` on the dense matrix. Default is ``"eigsh"``.
    method_options : mapping, optional
        Keyword arguments passed to ``method``, e.g. ``{"ncv": 60}``.
    diagonal_symmetries : mapping, optional
        Passed to ``fockit.hilbert.build_basis``, e.g. ``{Sz: 0}``. The
        half-filled particle number is always added. Modes of the bath orbitals
        are ``("bath_" + block, index)``.
    permutation_symmetries : sequence, optional
        Passed to ``fockit.hilbert.build_basis``.
    onsite_symmetries : sequence, optional
        Passed to ``fockit.hilbert.build_basis``.
    degenerate : {"ignore", "warn"}, optional
        With ``"warn"``, find two states and warn if their energies are within
        ``degenerate_tol``. Default is ``"ignore"``.
    degenerate_tol : float, optional
        Energy difference below which the ground state is degenerate.
    beta : float, optional
        Inverse temperature of the impurity problem. With the default ``None``
        the density matrices are from its ground state. With a float they are
        averages over the eigenstates of the half-filled sector with Boltzmann
        weights, dropping states with a weight below ``1e-14`` relative to the
        ground state. It is a convergence aid for the zero-temperature
        solution. A sector of at most 2000 states is diagonalized as a dense
        matrix.
    bath_size : mapping, optional
        Number of bath orbitals of each block, ``B`` times its size for ghost
        RISB with ``B`` quasiparticle orbitals per orbital, e.g.,
        ``{"up": 3, "dn": 3}`` for ``B = 3`` and one orbital per spin. Blocks
        that are not given have as many bath orbitals as physical orbitals
        (``B = 1``). The total number of physical and bath orbitals must be
        even.

    """

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        method: Literal["eigsh", "eigh"] = "eigsh",
        method_options: Mapping[str, Any] | None = None,
        diagonal_symmetries: Mapping[OpSum, complex] | None = None,
        permutation_symmetries: Sequence[PermutationSymmetry] = (),
        onsite_symmetries: Sequence[OnsiteSymmetry] = (),
        degenerate: Literal["ignore", "warn"] = "ignore",
        degenerate_tol: float = 1e-8,
        beta: float | None = None,
        *,
        bath_size: Mapping[str, int] | None = None,
    ) -> None:
        if method not in ("eigsh", "eigh"):
            msg = f"Unknown method {method!r}."
            raise ValueError(msg)
        if degenerate not in ("ignore", "warn"):
            msg = f"Unknown degenerate {degenerate!r}."
            raise ValueError(msg)
        super().__init__(
            h_int,
            gf_struct,
            list(diagonal_symmetries or {}),
            bath_size,
            permutation_symmetries,
            onsite_symmetries,
        )

        #: str : Method that finds the ground state.
        self.method = method

        #: dict : Keyword arguments passed to :attr:`method`.
        self.method_options = dict(method_options or {})

        #: dict : Diagonal symmetries of the basis, without the particle number.
        self.diagonal_symmetries = dict(diagonal_symmetries or {})

        #: str : What to do when the ground state is degenerate.
        self.degenerate = degenerate

        #: float : Energy difference below which the ground state is degenerate.
        self.degenerate_tol = degenerate_tol

        #: float or None : Inverse temperature of the impurity problem, or
        #: ``None`` for its ground state.
        self.beta = beta

        values = (len(self._modes) // 2, *self.diagonal_symmetries.values())
        eigenvalues = [e for _, e in (*permutation_symmetries, *onsite_symmetries)]
        self._key: _Key = (
            tuple(round(float(np.real(v)), 8) for v in values),
            tuple(complex(e) for e in eigenvalues) if eigenvalues else None,
        )

        # The states are found in the sector of _key. If its symmetries do not
        # split the annihilators into parts that reach one sector each, they
        # are written in the sector without the permutation and on-site
        # symmetries
        self._states_key = self._key
        if self._annihilators_of(self._key) is None:
            self._states_key = (self._key[0], None)

        #: fockit.hilbert.Basis : Half-filled particle sector of the impurity
        #: problem that the states are in, restricted by the symmetries.
        self.basis: Basis = self._basis(self._states_key)

        #: numpy.ndarray : Ground state of the impurity problem in
        #: :attr:`basis`, set by :meth:`solve`.
        self.gs_vector: NDArray[np.complex128]

        #: numpy.ndarray : Eigenstates of the impurity problem in :attr:`basis`
        #: that the expectation values average over, as columns, set by
        #: :meth:`solve`.
        self.states: NDArray[np.complex128]

    @override
    def solve(self) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the impurity problem.

        With :attr:`beta` set, also find the low-lying eigenstates for the
        averages with Boltzmann weights. Sets the density matrices
        :attr:`rho_f`, :attr:`rho_cf`, and :attr:`rho_c` of every block.

        Raises
        ------
        ValueError
            If :attr:`h_imp` is not Hermitian, or does not have the symmetries
            of the basis.

        Warns
        -----
        RuntimeWarning
            If :attr:`degenerate` is ``"warn"`` and the ground state is
            degenerate.

        """
        super().solve()

    @override
    def _find_states(self) -> list[_Key]:
        if self._permutation or self._onsite:
            try:
                phases = symmetry_phases(self.h_imp, self._permutation, self._onsite)
            except ValueError:
                phases = [np.nan]
            if not np.allclose(phases, 1):
                msg = "h_imp does not have the permutation and onsite symmetries."
                raise ValueError(msg)
        H = self._hamiltonian(
            self._key, *_one_body(self.h_imp - self.h_int, self._index)
        )
        energies, states = _lowest_states(
            H,
            self.beta,
            2 if self.degenerate == "warn" else 1,
            self.method,
            self.method_options,
        )
        if (
            self.degenerate == "warn"
            and len(energies) > 1
            and energies[1] - energies[0] < self.degenerate_tol
        ):
            msg = (
                "The ground state of h_imp is degenerate (E1 - E0 = "
                f"{energies[1] - energies[0]:.1e}). The density matrices are from "
                "one state in the degenerate subspace."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=4)

        if self.beta is None:
            states = states[:, :1]
            self.weights = np.ones(1)
        else:
            weights = _weight(energies, self.beta, energies[0])
            keep = weights > _WEIGHT_TOL
            states = states[:, keep]
            self.weights = weights[keep] / np.sum(weights[keep])
        if self._states_key != self._key:
            states = np.column_stack(
                [expand_state(v, self._basis(self._key), self.basis) for v in states.T]
            )
        self.states = np.asarray(states, dtype=complex)
        self.gs_energy = float(energies[0])
        self.ensemble = [(self.basis, self.states, self.weights)]
        self.gs_vector = self.states[:, 0]
        return [self._states_key]


class FockitThermalImpuritySolver(_FockitBase):
    """
    Grand-canonical impurity solver at finite temperature using exact diagonalization from fockit.

    The density matrices are thermal averages over every particle number of
    the impurity problem, with the chemical potential of :meth:`set_h_imp`, as
    :class:`risb.ThermalLatticeSolver` needs.

    Parameters
    ----------
    h_int : fockit.algebra.OpSum
        Interaction Hamiltonian of the physical orbitals. Its modes are
        ``(block, index)``, e.g. ``nc("up", 0) * nc("dn", 0)``.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    beta : float
        Inverse temperature of the impurity problem. States with a weight
        below ``1e-14`` relative to the lowest state are dropped.
    method : {"eigsh", "eigh"}, optional
        Find the states with :func:`scipy.sparse.linalg.eigsh` or with
        :func:`scipy.linalg.eigh` on the dense matrix. Default is ``"eigsh"``.
        A block of at most 2000 states is always diagonalized as a dense
        matrix.
    method_options : mapping, optional
        Keyword arguments passed to ``method``, e.g. ``{"ncv": 60}``.
    diagonal_symmetries : sequence, optional
        Sums of number operators that commute with the impurity Hamiltonian,
        e.g. ``[Sz]``. They split the Hamiltonian into blocks, and every block
        is in the average. Modes of the bath orbitals are
        ``("bath_" + block, index)``.
    permutation_symmetries : sequence, optional
        Mode relabellings that commute with the impurity Hamiltonian, as the
        maps of ``fockit.hilbert.build_basis`` without eigenvalues, e.g.
        ``[layer_swap]``. They split the blocks further, and every eigenvalue
        is in the average.
    onsite_symmetries : sequence, optional
        Products of single-mode operators that commute with the impurity
        Hamiltonian, as the factors of ``fockit.hilbert.build_basis`` without
        eigenvalues.
    bath_size : mapping, optional
        Number of bath orbitals of each block, ``B`` times its size for ghost
        RISB with ``B`` quasiparticle orbitals per orbital, e.g.,
        ``{"up": 3, "dn": 3}`` for ``B = 3`` and one orbital per spin. Blocks
        that are not given have as many bath orbitals as physical orbitals
        (``B = 1``). The total number of physical and bath orbitals must be
        even.

    Raises
    ------
    ValueError
        If a diagonal symmetry is not a sum of number operators, if a
        permutation or onsite symmetry changes a diagonal symmetry or the
        particle number, or if the permutation and onsite symmetries do not
        commute with each other.

    """

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        beta: float,
        method: Literal["eigsh", "eigh"] = "eigsh",
        method_options: Mapping[str, Any] | None = None,
        diagonal_symmetries: Sequence[OpSum] = (),
        permutation_symmetries: Sequence[Mapping[Any, Any]] = (),
        onsite_symmetries: Sequence[Sequence[OpSum]] = (),
        *,
        bath_size: Mapping[str, int] | None = None,
    ) -> None:
        if method not in ("eigsh", "eigh"):
            msg = f"Unknown method {method!r}."
            raise ValueError(msg)
        super().__init__(
            h_int,
            gf_struct,
            diagonal_symmetries,
            bath_size,
            [(m, 1) for m in permutation_symmetries],
            [(f, 1) for f in onsite_symmetries],
        )
        if any(shift is None for shift in self._shifts):
            msg = (
                "Each diagonal symmetry must be a sum of number operators, like "
                "Sz, and the permutation and onsite symmetries must not change it "
                "or the particle number. Use one or the other, e.g., Sz or the "
                "spin swap."
            )
            raise ValueError(msg)

        #: float : Inverse temperature of the impurity problem.
        self.beta = beta

        #: str : Method that finds the states.
        self.method = method

        #: dict : Keyword arguments passed to :attr:`method`.
        self.method_options = dict(method_options or {})

        #: list : Operators that split the Hamiltonian into blocks.
        self.diagonal_symmetries = list(diagonal_symmetries)

        #: float : ``-log(sum(exp(-beta * E))) / beta`` over the states, set by
        #: :meth:`solve`.
        self.grand_potential: float

        # The number of states with each set of values of the diagonal
        # symmetries, adding the modes one at a time
        vacuum = tuple(round(_constant(Q).real, 8) for Q in self._diagonal)
        self._counts: dict[tuple[float | None, ...], int] = {vacuum: 1}
        for j in range(len(self._modes)):
            counts = dict(self._counts)
            for vs, count in self._counts.items():
                vs_j = self._raised(vs, j)
                counts[vs_j] = counts.get(vs_j, 0) + count
            self._counts = counts

        # How the parts of the creation operator of each mode change the
        # eigenvalues of the permutation and onsite symmetries
        self._phases: list[list[tuple[complex, ...]]] = [[] for _ in self._modes]
        eigenvalues: tuple[complex, ...] | None = None
        if self._permutation or self._onsite:
            reference = self._basis(
                (vacuum, (1,) * len(self._permutation + self._onsite))
            )
            for j, mode in enumerate(self._modes):
                try:
                    self._phases[j] = list(symmetry_components(cdag(*mode), reference))
                except ValueError:
                    msg = "The permutation and onsite symmetries must commute with each other."
                    raise ValueError(msg) from None
            eigenvalues = tuple(
                [1] * len(self._permutation)
                + [
                    np.prod([_constant(f) for f in factors])
                    for factors, _ in self._onsite
                ]
            )

        # Every sector, from the vacuum: each state is a product of creation
        # operators on it, so the sectors their parts reach are all of them
        found: set[_Key] = {(vacuum, eigenvalues)}
        frontier = list(found)
        while frontier:
            vs, eigs = frontier.pop()
            for j in range(len(self._modes)):
                targets: list[_Key] = [(self._raised(vs, j), None)]
                if eigs is not None:
                    targets = [
                        (
                            self._raised(vs, j),
                            _rounded([a * b for a, b in zip(eigs, phase, strict=True)]),
                        )
                        for phase in self._phases[j]
                    ]
                for target in targets:
                    if target[0] in self._counts and target not in found:
                        found.add(target)
                        frontier.append(target)
        self._sectors: dict[int, list[_Key]] = {}
        for key in sorted(found, key=str):
            n = key[0][0]
            assert n is not None  # every sector has a particle number
            self._sectors.setdefault(round(n), []).append(key)

    def _raised(self, values: tuple[float | None, ...], j: int) -> tuple[float, ...]:
        # The values of the diagonal symmetries after adding a particle in mode j
        return tuple(
            round(v + shift[j], 8)
            for v, shift in zip(values, self._shifts, strict=True)
            if v is not None and shift is not None
        )

    @override
    def _find_states(self) -> list[_Key]:
        beta = self.beta
        one_body = _one_body(self.h_imp - self.h_int, self._index)
        hamiltonians = {}
        lowest = {}

        def lowest_of(n: int) -> float:
            dims: dict[tuple[float | None, ...], int] = {}
            for key in self._sectors[n]:
                dim = self._basis(key).dim
                dims[key[0]] = dims.get(key[0], 0) + dim
                if dim:
                    hamiltonians[key] = self._hamiltonian(key, *one_body)
                    lowest[key] = _lowest_energy(hamiltonians[key], self.method_options)
            if any(dim != self._counts[vs] for vs, dim in dims.items()):
                msg = (
                    "Some states are in no sector of the symmetries. The "
                    "permutation and onsite symmetries must commute with each other."
                )
                raise ValueError(msg)
            return min(lowest[key] for key in self._sectors[n] if key in lowest)

        # The lowest energy of each particle number, from half filling
        # outwards until it has a negligible weight. This assumes the lowest
        # energy has no other minimum as a function of the particle number
        half = len(self._modes) // 2
        lowest_n = {half: lowest_of(half)}
        for step in (1, -1):
            n = half + step
            while n in self._sectors:
                lowest_n[n] = lowest_of(n)
                if (
                    np.exp(-beta * (lowest_n[n] - min(lowest_n.values())))
                    <= _WEIGHT_TOL
                ):
                    break
                n += step
        e0 = min(lowest_n.values())

        keys = []
        ensemble = []
        for key in sorted(lowest, key=lambda key: lowest[key]):
            if np.exp(-beta * (lowest[key] - e0)) <= _WEIGHT_TOL:
                continue
            energies, states = _lowest_states(
                hamiltonians[key], beta, 1, self.method, self.method_options, e0
            )
            keep = _weight(energies, beta, e0) > _WEIGHT_TOL
            keys.append(key)
            ensemble.append((self._basis(key), states[:, keep], energies[keep]))
        weights = _weight(np.concatenate([e for _, _, e in ensemble]), beta, e0)
        self.weights = weights / np.sum(weights)
        self.ensemble = [
            (basis, states, _weight(energies, beta, e0) / np.sum(weights))
            for basis, states, energies in ensemble
        ]
        self.gs_energy = e0
        self.grand_potential = e0 - float(np.log(np.sum(weights))) / beta
        return keys


def _eigsh(
    H: Any, k: int, method_options: Mapping[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    dim = H.shape[0]
    options: dict[str, Any] = {
        "which": "SA",
        "v0": np.random.default_rng(0).normal(size=dim),
        "ncv": min(dim, max(40, 2 * k + 1)),
    } | dict(method_options)
    # ARPACK can run out of iterations when eigenvalues are close, and a
    # larger Krylov space usually fixes it
    for _ in range(_ARPACK_RETRIES):
        try:
            energies, states = eigsh(H, k=k, **options)
            break
        except ArpackNoConvergence:
            options["ncv"] = min(dim, 2 * options["ncv"])
    else:
        energies, states = eigsh(H, k=k, **options)
    order = np.argsort(energies)
    return energies[order], states[:, order]


def _add_missed_states(
    H: Any, states: NDArray[np.complex128], method_options: Mapping[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    # Lanczos finds one state of a degenerate level in exact arithmetic, so
    # eigsh can skip the others. Shift the states found above the highest of
    # them, and add the lowest state of the shifted H while it is below it
    dim = H.shape[0]
    while True:
        # As when eigsh finds more states, diagonalize the whole sector once
        # the states would be more than a quarter of it
        if dim <= 2 * (2 * states.shape[1] + 1):
            energies, states = np.linalg.eigh(H.toarray())
            return energies, states
        # eigsh's vectors within a degenerate level need not be orthogonal
        states = scipy.linalg.orth(states)
        energies, rotation = np.linalg.eigh(states.conj().T @ (H @ states))
        states = states @ rotation
        shift = energies[-1] - energies[0] + 1
        shifted = aslinearoperator(H) + shift * (
            aslinearoperator(states) @ aslinearoperator(states.conj().T)
        )
        energy, state = _eigsh(shifted, 1, method_options)
        if energy[0] > energies[-1] - _MISSED_TOL:
            return energies, states
        states = np.column_stack([states, state])


def _weight(
    energies: NDArray[np.float64], beta: float, reference: float
) -> NDArray[np.float64]:
    weights: NDArray[np.float64] = np.exp(-beta * (energies - reference))
    return weights


def _hermitian_matrix(op: OpSum, basis: Basis) -> Any:
    H = to_matrix(op, basis)
    if abs(H - H.conj().T).max() > 1e-10:
        msg = "h_imp is not Hermitian."
        raise ValueError(msg)
    return H


def _lowest_energy(H: Any, method_options: Mapping[str, Any]) -> float:
    if H.shape[0] <= _DENSE_DIM:
        return float(np.linalg.eigvalsh(H.toarray())[0])
    return float(_eigsh(H, 1, method_options)[0][0])


def _bath(bl: str) -> str:
    return "bath_" + bl


def _bilinear(
    left: Sequence[OpSum], A: NDArray[np.inexact], right: Sequence[OpSum]
) -> OpSum:
    out = zero()
    for i, j in np.ndindex(A.shape):
        out += complex(A[i, j]) * left[i] * right[j]
    return out


def _lowest_states(
    H: Any,
    beta: float | None,
    k: int,
    method: str,
    method_options: Mapping[str, Any],
    reference: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    # The lowest k states, or with beta every state with a weight above
    # _WEIGHT_TOL relative to the energy reference (default the lowest)
    dim = H.shape[0]
    if method == "eigh":
        energies, states = scipy.linalg.eigh(H.toarray(), **method_options)
        return energies, states
    if dim <= k + 1 or (beta is not None and dim <= _DENSE_DIM):
        energies, states = np.linalg.eigh(H.toarray())
        return energies, states
    energies, states = _eigsh(H, k, method_options)
    # Find more states until the highest one has a negligible weight
    while beta is not None and (
        _weight(energies, beta, energies[0] if reference is None else reference)[-1]
        > _WEIGHT_TOL
    ):
        k *= 2
        # Diagonalize the whole sector once the Krylov space of eigsh
        # would be more than half of it
        if dim <= 2 * (2 * k + 1):
            energies, states = np.linalg.eigh(H.toarray())
            return energies, states
        energies, states = _eigsh(H, k, method_options)
    if 1 < len(energies) < dim:
        energies, states = _add_missed_states(H, states, method_options)
    return energies, states


def _one_body(
    op: OpSum, index: Mapping[Any, int]
) -> tuple[NDArray[np.inexact], float, OpSum]:
    # op as const + sum_ij T_ij c_i^dag c_j + rest, with rest the other terms
    T = np.zeros((len(index), len(index)), dtype=complex)
    const = 0j
    rest = {}
    for mono, coeff in normal_order(op).terms.items():
        ops = mono.ops
        if not ops:
            const += complex(coeff)
        elif (
            len(ops) == 2
            and ops[0].kind == Ladder.CREATE
            and ops[1].kind == Ladder.ANNIHILATE
        ):
            T[index[ops[0].mode], index[ops[1].mode]] += complex(coeff)
        else:
            rest[mono] = coeff
    if abs(T - T.conj().T).max(initial=0) > 1e-10 or abs(const.imag) > 1e-10:
        msg = "h_imp is not Hermitian."
        raise ValueError(msg)
    return (T if np.any(T.imag) else T.real), const.real, OpSum(rest)


def _constant(op: OpSum) -> complex:
    # The coefficient of the identity in op
    return sum(
        (
            complex(coeff)
            for mono, coeff in normal_order(op).terms.items()
            if not mono.ops
        ),
        start=0j,
    )


def _rounded(eigenvalues: Sequence[complex]) -> tuple[complex, ...]:
    # As fockit rounds the eigenvalues of the sectors it reaches
    return tuple(
        complex(round(complex(e).real, 10) + 0.0, round(complex(e).imag, 10) + 0.0)
        for e in eigenvalues
    )

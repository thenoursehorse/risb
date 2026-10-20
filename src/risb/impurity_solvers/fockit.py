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
from fockit.algebra import OpSum, c, cdag, commutator, hc, nc, normal_order, zero
from fockit.hilbert import (
    Basis,
    FermionSpace,
    build_basis,
    expand_state,
    expectation,
    to_matrix,
)
from fockit.hilbert.symmetries import OnsiteSymmetry, PermutationSymmetry
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


class _FockitBase(ImpuritySolver):
    # The impurity Hamiltonian, density matrices and expectation values of the
    # fockit solvers. A subclass finds the states in _find_states

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        conserved: Sequence[OpSum],
        bath_size: Mapping[str, int] | None,
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
        self._N = N
        self._conserved = list(conserved)

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
        for Q in self._conserved:
            terms = normal_order(commutator(self.h_imp, Q)).terms.values()
            if any(abs(complex(coeff)) > 1e-10 for coeff in terms):
                msg = "h_imp does not conserve a diagonal symmetry."
                raise ValueError(msg)
        self._find_states()
        self.entropy = get_entropy_impurity(self.weights)
        for (bl, bl_size), (bl_bath, n_bath) in zip(
            self.gf_struct, self.gf_struct_bath, strict=True
        ):
            sites = range(bl_size)
            baths = range(n_bath)
            self.rho_f[bl] = np.array(
                [
                    [
                        self.overlap(c(bl_bath, b) * cdag(bl_bath, a), False)
                        for b in baths
                    ]
                    for a in baths
                ]
            )
            self.rho_cf[bl] = np.array(
                [
                    [
                        self.overlap(cdag(bl, alpha) * c(bl_bath, a), False)
                        for a in baths
                    ]
                    for alpha in sites
                ]
            )
            self.rho_c[bl] = np.array(
                [
                    [
                        self.overlap(cdag(bl, alpha) * c(bl, beta), False)
                        for beta in sites
                    ]
                    for alpha in sites
                ]
            )

    @abstractmethod
    def _find_states(self) -> None:
        """Set gs_energy, weights and ensemble."""

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
            Op_matrix = to_matrix(Op, basis)
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
        super().__init__(h_int, gf_struct, list(diagonal_symmetries or {}), bath_size)

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

        diagonal = {self._N: len(self._modes) // 2, **self.diagonal_symmetries}

        #: fockit.hilbert.Basis : Half-filled particle sector of the impurity
        #: problem, restricted by :attr:`diagonal_symmetries`.
        self.basis: Basis = build_basis(
            {FermionSpace(): self._modes}, diagonal_symmetries=diagonal
        )

        # The states are found in this basis, which also has the permutation
        # and on-site symmetries, and then written in basis
        self._symmetric_basis = self.basis
        if permutation_symmetries or onsite_symmetries:
            self._symmetric_basis = build_basis(
                {FermionSpace(): self._modes},
                diagonal_symmetries=diagonal,
                permutation_symmetries=permutation_symmetries,
                onsite_symmetries=onsite_symmetries,
            )

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
    def _find_states(self) -> None:
        H = _hermitian_matrix(self.h_imp, self._symmetric_basis)
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
        if self._symmetric_basis is not self.basis:
            states = np.column_stack(
                [expand_state(v, self._symmetric_basis, self.basis) for v in states.T]
            )
        self.states = np.asarray(states, dtype=complex)
        self.gs_energy = float(energies[0])
        self.ensemble = [(self.basis, self.states, self.weights)]
        self.gs_vector = self.states[:, 0]


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
        Operators that commute with the impurity Hamiltonian, e.g. ``[Sz]``.
        They split the Hamiltonian into blocks, and every block is in the
        average. Modes of the bath orbitals are ``("bath_" + block, index)``.
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
        beta: float,
        method: Literal["eigsh", "eigh"] = "eigsh",
        method_options: Mapping[str, Any] | None = None,
        diagonal_symmetries: Sequence[OpSum] = (),
        *,
        bath_size: Mapping[str, int] | None = None,
    ) -> None:
        if method not in ("eigsh", "eigh"):
            msg = f"Unknown method {method!r}."
            raise ValueError(msg)
        super().__init__(h_int, gf_struct, diagonal_symmetries, bath_size)

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

        # Bases of the particle numbers that a solve needs
        self._sector_bases: dict[int, Basis] = {}

    @override
    def _find_states(self) -> None:
        beta = self.beta
        n_modes = len(self._modes)
        half = n_modes // 2

        # The Hamiltonian of a particle number in blocks, one for each set of
        # values of the diagonal symmetries on the basis states
        def blocks(n: int) -> list[tuple[NDArray[np.intp], Any]]:
            if n not in self._sector_bases:
                self._sector_bases[n] = build_basis(
                    {FermionSpace(): self._modes}, diagonal_symmetries={self._N: n}
                )
            basis = self._sector_bases[n]
            H = _hermitian_matrix(self.h_imp, basis).tocsr()
            values = [
                np.round(to_matrix(Q, basis).diagonal().real, 8)
                for Q in self._conserved
            ]
            labels = np.stack(values, axis=1) if values else np.zeros((basis.dim, 0))
            groups = np.unique(labels, axis=0, return_inverse=True)[1].ravel()
            return [
                (idx, H[idx][:, idx])
                for idx in (np.flatnonzero(groups == g) for g in np.unique(groups))
            ]

        # The lowest energy of each particle number, from half filling
        # outwards until it has a negligible weight. This assumes the lowest
        # energy has no other minimum as a function of the particle number
        lowest: dict[int, float] = {}
        sectors = {half: blocks(half)}
        lowest[half] = min(
            _lowest_energy(H, self.method_options) for _, H in sectors[half]
        )
        for step in (1, -1):
            n = half + step
            while 0 <= n <= n_modes:
                sectors[n] = blocks(n)
                lowest[n] = min(
                    _lowest_energy(H, self.method_options) for _, H in sectors[n]
                )
                if np.exp(-beta * (lowest[n] - min(lowest.values()))) <= _WEIGHT_TOL:
                    break
                n += step
        e0 = min(lowest.values())

        ensemble = []
        all_energies = []
        for n in sorted(sectors, key=lambda n: lowest[n]):
            if np.exp(-beta * (lowest[n] - e0)) <= _WEIGHT_TOL:
                continue
            dim = self._sector_bases[n].dim
            energies_n = []
            states_n = []
            for idx, H in sectors[n]:
                energies, states = _lowest_states(
                    H, beta, 1, self.method, self.method_options, e0
                )
                keep = _weight(energies, beta, e0) > _WEIGHT_TOL
                full = np.zeros((dim, int(np.sum(keep))), dtype=complex)
                full[idx] = states[:, keep]
                energies_n.append(energies[keep])
                states_n.append(full)
            energies = np.concatenate(energies_n)
            order = np.argsort(energies)
            ensemble.append(
                (self._sector_bases[n], np.hstack(states_n)[:, order], energies[order])
            )
            all_energies.append(energies[order])
        weights = _weight(np.concatenate(all_energies), beta, e0)
        self.weights = weights / np.sum(weights)
        self.ensemble = [
            (basis, states, _weight(energies, beta, e0) / np.sum(weights))
            for basis, states, energies in ensemble
        ]
        self.gs_energy = e0
        self.grand_potential = e0 - float(np.log(np.sum(weights))) / beta


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

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
from scipy.sparse.linalg import eigsh

from risb.helpers import get_entropy_impurity
from risb.types import BlockMatrix, BlockMatrixLike, GfStruct, ImpuritySolver

# Boltzmann weight, relative to the ground state, below which a state is dropped
_WEIGHT_TOL = 1e-14


class FockitImpuritySolver(ImpuritySolver):
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
        thermal averages over the eigenstates of the half-filled sector,
        dropping states with a Boltzmann weight below ``1e-14`` relative to the
        ground state.
    bath_size : mapping, optional
        Number of bath orbitals of each block, e.g. ``{"up": 3, "dn": 3}``.
        Blocks that are not given have as many bath orbitals as physical
        orbitals.

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
        N = zero()
        for mode in modes:
            N += nc(*mode)

        diagonal = {N: len(modes) // 2, **self.diagonal_symmetries}

        #: fockit.hilbert.Basis : Half-filled particle sector of the impurity
        #: problem, restricted by :attr:`diagonal_symmetries`.
        self.basis: Basis = build_basis(
            {FermionSpace(): modes}, diagonal_symmetries=diagonal
        )

        # The states are found in this basis, which also has the permutation
        # and on-site symmetries, and then written in basis
        self._symmetric_basis = self.basis
        if permutation_symmetries or onsite_symmetries:
            self._symmetric_basis = build_basis(
                {FermionSpace(): modes},
                diagonal_symmetries=diagonal,
                permutation_symmetries=permutation_symmetries,
                onsite_symmetries=onsite_symmetries,
            )

        #: numpy.ndarray : Ground state of the impurity problem in
        #: :attr:`basis`, set by :meth:`solve`.
        self.gs_vector: NDArray[np.complex128]

        #: float : Ground state energy of the impurity problem, set by
        #: :meth:`solve`.
        self.gs_energy: float

        #: numpy.ndarray : Eigenstates of the impurity problem in :attr:`basis`
        #: that the expectation values average over, as columns, set by
        #: :meth:`solve`.
        self.states: NDArray[np.complex128]

        #: numpy.ndarray : Weights of :attr:`states` in the expectation values,
        #: set by :meth:`solve`.
        self.weights: NDArray[np.float64]

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
            Chemical potential, subtracted as ``mu * n`` on the physical
            orbitals.

        """
        if h0_loc_matrix is not None:
            self.set_h0_loc(h0_loc_matrix)
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        self.h_imp = self.h0_loc + self.h_int + self.h_bath + self.h_hybr
        if mu is not None:
            for bl, bl_size in self.gf_struct:
                for alpha in range(bl_size):
                    self.h_imp -= mu * nc(bl, alpha)

    @override
    def solve(self) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the impurity problem.

        With :attr:`beta` set, also find the low-lying eigenstates for the
        thermal averages. Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block.

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
        H = to_matrix(self.h_imp, self._symmetric_basis)
        if abs(H - H.conj().T).max() > 1e-10:
            msg = "h_imp is not Hermitian."
            raise ValueError(msg)
        for Q in self.diagonal_symmetries:
            terms = normal_order(commutator(self.h_imp, Q)).terms.values()
            if any(abs(complex(coeff)) > 1e-10 for coeff in terms):
                msg = "h_imp does not conserve a diagonal symmetry."
                raise ValueError(msg)

        k = 2 if self.degenerate == "warn" else 1
        dim = H.shape[0]
        if self.method == "eigh":
            energies, states = scipy.linalg.eigh(H.toarray(), **self.method_options)
        elif dim <= k + 1:
            energies, states = np.linalg.eigh(H.toarray())
        else:
            energies, states = _eigsh(H, k, self.method_options)
            # Find more states until the highest one has a negligible weight
            while (
                self.beta is not None and _weight(energies, self.beta)[-1] > _WEIGHT_TOL
            ):
                k *= 2
                # Diagonalize the whole sector once the Krylov space of eigsh
                # would be more than half of it
                if dim <= 2 * (2 * k + 1):
                    energies, states = np.linalg.eigh(H.toarray())
                    break
                energies, states = _eigsh(H, k, self.method_options)
            if self.beta is not None:
                # eigsh's vectors within a degenerate level need not be orthogonal
                subspace = scipy.linalg.orth(states)
                energies, rotation = np.linalg.eigh(subspace.conj().T @ (H @ subspace))
                states = subspace @ rotation

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
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        if self.beta is None:
            states = states[:, :1]
            self.weights = np.ones(1)
        else:
            weights = _weight(energies, self.beta)
            keep = weights > _WEIGHT_TOL
            states = states[:, keep]
            self.weights = weights[keep] / np.sum(weights[keep])
        if self._symmetric_basis is not self.basis:
            states = np.column_stack(
                [expand_state(v, self._symmetric_basis, self.basis) for v in states.T]
            )
        self.states = np.asarray(states, dtype=complex)
        self.entropy = get_entropy_impurity(self.weights)
        self.gs_energy = float(energies[0])
        self.gs_vector = self.states[:, 0]
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
        Calculate the expectation value of an operator in the impurity problem, in its ground state or thermal state (see ``beta``).

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
        Op_matrix = to_matrix(Op, self.basis)
        res = complex(
            sum(
                weight * expectation(Op_matrix, state)
                for weight, state in zip(self.weights, self.states.T, strict=True)
            )
        )
        if force_real:
            return res.real
        return res


def _eigsh(
    H: Any, k: int, method_options: Mapping[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    dim = H.shape[0]
    options: dict[str, Any] = {
        "which": "SA",
        "v0": np.random.default_rng(0).normal(size=dim),
        "ncv": min(dim, max(40, 2 * k + 1)),
    } | dict(method_options)
    energies, states = eigsh(H, k=k, **options)
    order = np.argsort(energies)
    return energies[order], states[:, order]


def _weight(energies: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
    weights: NDArray[np.float64] = np.exp(-beta * (energies - energies[0]))
    return weights


def _bath(bl: str) -> str:
    return "bath_" + bl


def _bilinear(
    left: Sequence[OpSum], A: NDArray[np.inexact], right: Sequence[OpSum]
) -> OpSum:
    out = zero()
    for i, j in np.ndindex(A.shape):
        out += complex(A[i, j]) * left[i] * right[j]
    return out

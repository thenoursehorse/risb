# Copyright (c) 2023 H. L. Nourse
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

"""Solver for rotationally invariant slave-boson mean-field theory on a lattice."""

from collections.abc import Callable, Mapping, Sequence
from typing import override

import numpy as np
from numpy.typing import NDArray

from risb import helpers
from risb.solve_lattice_base import BaseLatticeSolver
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    BlockRealArray,
    GfStruct,
    ImpuritySolver,
    Symmetrize,
    UpdateWeights,
)


class LatticeSolver(BaseLatticeSolver):
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with a local interaction on each cluster.

    It solves the zero-temperature RISB equations.

    Parameters
    ----------
    update_weights : risb.types.UpdateWeights
        The function that gives the integral weights at each k-point on the
        lattice. It is called as ``update_weights(energies)``,
        where the energies are a dictionary with each key a list.
        See class :class:`.SmearingKWeight`.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, check_projectors, double_counting, bath_size
        As in :class:`.BaseLatticeSolver`.

    """

    def __init__(
        self,
        h0_k: BlockMatrixLike,
        gf_struct: GfStruct | list[GfStruct],
        impurity_solvers: ImpuritySolver | Sequence[ImpuritySolver],
        update_weights: UpdateWeights,
        projectors: Sequence[BlockMatrixLike] | None = None,
        gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
        symmetrizers: Sequence[Symmetrize] = (),
        force_real: bool = False,
        check_projectors: bool = False,
        double_counting: Callable[[int, BlockMatrix], BlockMatrixLike] | None = None,
        *,
        bath_size: Sequence[Mapping[str, int]] | None = None,
    ) -> None:
        super().__init__(
            h0_k,
            gf_struct,
            impurity_solvers,
            projectors=projectors,
            gf_struct_mapping=gf_struct_mapping,
            symmetrizers=symmetrizers,
            force_real=force_real,
            check_projectors=check_projectors,
            double_counting=double_counting,
            bath_size=bath_size,
        )
        self._update_weights = update_weights

    @override
    def update_weights(self, energies: BlockRealArray) -> BlockRealArray:
        return self._update_weights(energies)

    @override
    def update_bath(self) -> None:
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.D[i][bl] = helpers.get_d(
                        self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                    ).real
                    # self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl]).real
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(
                        self.rho_qp[i][bl],
                        self.R[i][bl],
                        self.Lambda[i][bl],
                        self.D[i][bl],
                    ).real
                else:
                    self.D[i][bl] = helpers.get_d(
                        self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                    )
                    # self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl])
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(
                        self.rho_qp[i][bl],
                        self.R[i][bl],
                        self.Lambda[i][bl],
                        self.D[i][bl],
                    )

        # Enforce matrix structure from symmetries
        self.Lambda_c = self._project(self.Lambda_c, is_coeff_real=True)
        self.D = self._project(self.D, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.Lambda_c = symmetrize(self.Lambda_c)
            self.D = symmetrize(self.D)

        # A completely full (empty) quasiparticle orbital has no bath partner.
        # Its D is zero, and the limit of the gradient term that would put its
        # bath orbital empty (full) is not finite, so put the bath level far
        # away instead. Otherwise the bath orbital competes for electrons and
        # the impurity problem can end at R = 0 with the wrong filling
        core_scale = self._core_scale()
        for i in range(self.n_clusters):
            for bl in self.Lambda_c[i]:
                self.Lambda_c[i][bl] = _place_core_baths(
                    self.Lambda_c[i][bl], self.rho_qp[i][bl], core_scale
                )

    @override
    def update_qp(self) -> None:
        core_scale = self._core_scale()
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                self.f1[i][bl] = helpers.get_f1(
                    self.rho_cf[i][bl], self.rho_qp[i][bl], self.R[i][bl]
                )
                self.f2[i][bl] = helpers.get_f2(self.rho_f[i][bl], self.rho_qp[i][bl])

        # Enforce matrix structure from symmetries
        self.f2 = self._project(self.f2, is_coeff_real=True)
        self.f1 = self._project(self.f1, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.f1 = symmetrize(self.f1)
            self.f2 = symmetrize(self.f2)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.Lambda[i][bl] = helpers.get_lambda(
                        self.R[i][bl],
                        self.D[i][bl],
                        self.Lambda_c[i][bl],
                        self.rho_f[i][bl],
                    ).real
                    self.R[i][bl] = helpers.get_r(
                        self.rho_cf[i][bl], self.rho_f[i][bl]
                    ).real
                else:
                    self.Lambda[i][bl] = helpers.get_lambda(
                        self.R[i][bl],
                        self.D[i][bl],
                        self.Lambda_c[i][bl],
                        self.rho_f[i][bl],
                    )
                    self.R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl])

        # Enforce matrix structure from symmetries
        self.Lambda = self._project(self.Lambda, is_coeff_real=True)
        self.R = self._project(self.R, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.Lambda = symmetrize(self.Lambda)
            self.R = symmetrize(self.R)

        # The quasiparticle level of a full (empty) orbital is only fixed to be
        # below (above) the chemical potential, so keep it below (above) every
        # other quasiparticle band. This finds the solution where the orbital
        # stays decoupled. A solution with a lower energy, where the
        # quasiparticle orbital is not exactly full (empty) and is an extra
        # channel for the other orbitals, can also exist
        for i in range(self.n_clusters):
            for bl in self.Lambda[i]:
                self.Lambda[i][bl] = _place_core_levels(
                    self.Lambda[i][bl],
                    self.rho_f[i][bl],
                    self._kinetic_half_bandwidth + core_scale,
                )

    def _core_scale(self) -> float:
        # An energy far from every energy of the model, leaving out the
        # levels of full or empty orbitals, which are themselves set far away
        largest = max(
            float(np.max(np.abs(A[i][bl]), initial=0))
            for A in (self.h0_loc_matrix,)
            for i in range(self.n_clusters)
            for bl in A[i]
        )
        for i in range(self.n_clusters):
            for bl in self.Lambda[i]:
                _, vectors, edge = _edge_orbitals(self.rho_qp[i][bl])
                active = vectors[:, ~edge].conj()
                if active.size:
                    levels = np.linalg.eigvalsh(
                        active.conj().T @ self.Lambda[i][bl] @ active
                    )
                    largest = max(largest, float(np.max(np.abs(levels))))
        return 10 * (self._kinetic_half_bandwidth + largest)


def _edge_orbitals(
    rho: NDArray[np.inexact],
) -> tuple[NDArray[np.float64], NDArray[np.inexact], NDArray[np.bool_]]:
    # Eigenvalues, eigenvectors, and which eigenvalues are 0 or 1
    eigenvalues, vectors = np.linalg.eigh(rho)
    return eigenvalues, vectors, eigenvalues * (1 - eigenvalues) < helpers.EDGE_TOL


def _place_core_baths(
    Lambda_c: NDArray[np.inexact], rho_qp: NDArray[np.inexact], scale: float
) -> NDArray[np.inexact]:
    eigenvalues, vectors, edge = _edge_orbitals(rho_qp)
    # The bath term is f Lambda_c f^dag, so filling the bath orbital costs
    # -Lambda_c: empty for a full quasiparticle orbital, full for an empty one
    for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
        level = -scale if delta > 0.5 else scale
        Lambda_c = Lambda_c + (level - np.real(v.conj() @ Lambda_c @ v)) * np.outer(
            v, v.conj()
        )
    return Lambda_c


def _place_core_levels(
    Lambda: NDArray[np.inexact], rho_f: NDArray[np.inexact], margin: float
) -> NDArray[np.inexact]:
    eigenvalues, vectors, edge = _edge_orbitals(rho_f)
    if not edge.any():
        return Lambda
    # rho_qp is the transpose of n_F(lambda - mu), so these are the
    # eigenvectors of lambda
    vectors = vectors.conj()
    active = vectors[:, ~edge]
    others = (
        np.linalg.eigvalsh(active.conj().T @ Lambda @ active)
        if active.size
        else np.zeros(1)
    )
    for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
        level = others.min() - margin if delta > 0.5 else others.max() + margin
        Lambda = Lambda + (level - np.real(v.conj() @ Lambda @ v)) * np.outer(
            v, v.conj()
        )
    return Lambda

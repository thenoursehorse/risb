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

from risb import helpers
from risb.diagnostics import DiagnosticObserver
from risb.pinning import FullEmptyPinning
from risb.solve_lattice_base import BaseLatticeSolver
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    BlockRealArray,
    GfStruct,
    ImpuritySolver,
    PinFullEmpty,
    Symmetrize,
    UpdateWeights,
)

_DEFAULT_PINNING = FullEmptyPinning()


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
    pin_full_empty : risb.types.PinFullEmpty or None, optional
        Sets the levels of full or empty auxiliary orbitals. Default is
        :class:`risb.pinning.FullEmptyPinning` with its default parameters.
        ``None`` does no pinning.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Use :class:`~risb.diagnostics.LatticeDiagnostics` for progress reports.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, force_real_atol, force_real_rtol, check_projectors, double_counting, bath_size
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
        pin_full_empty: PinFullEmpty | None = _DEFAULT_PINNING,
        force_real_atol: float = 1e-12,
        force_real_rtol: float = 1e-10,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        super().__init__(
            h0_k,
            gf_struct,
            impurity_solvers,
            projectors=projectors,
            gf_struct_mapping=gf_struct_mapping,
            symmetrizers=symmetrizers,
            force_real=force_real,
            force_real_atol=force_real_atol,
            force_real_rtol=force_real_rtol,
            diagnostics=diagnostics,
            check_projectors=check_projectors,
            double_counting=double_counting,
            bath_size=bath_size,
        )
        self._update_weights = update_weights
        self.pin_full_empty = pin_full_empty

    @override
    def update_weights(self, energies: BlockRealArray) -> BlockRealArray:
        return self._update_weights(energies)

    @override
    def update_bath(self) -> None:
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                self.D[i][bl] = helpers.get_d(
                    self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                )
                if self.force_real:
                    self._warn_if_complex(self.D[i][bl], "D", i, bl)
                    self.D[i][bl] = self.D[i][bl].real
                self.Lambda_c[i][bl] = helpers.get_lambda_c(
                    self.rho_qp[i][bl],
                    self.R[i][bl],
                    self.Lambda[i][bl],
                    self.D[i][bl],
                )

        # Enforce matrix structure from symmetries
        self.Lambda_c = self._project(self.Lambda_c, is_hermitian=True, name="Lambda_c")
        self.D = self._project(self.D, is_hermitian=False, name="D")
        for symmetrize in self.symmetrizers:
            self.Lambda_c = symmetrize(self.Lambda_c)
            self.D = symmetrize(self.D)

        if self.pin_full_empty is not None:
            energy_scale = self._energy_scale()
            for i in range(self.n_clusters):
                for bl in self.Lambda_c[i]:
                    self.Lambda_c[i][bl] = self.pin_full_empty.bath_levels(
                        self.Lambda_c[i][bl], self.rho_qp[i][bl], energy_scale
                    )

    @override
    def update_residuals(self) -> None:
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                self.f1[i][bl] = helpers.get_f1(
                    self.rho_cf[i][bl], self.rho_qp[i][bl], self.R[i][bl]
                )
                self.f2[i][bl] = helpers.get_f2(self.rho_f[i][bl], self.rho_qp[i][bl])

        # Enforce matrix structure from symmetries
        self.f2 = self._project(self.f2, is_hermitian=True, name="f2")
        self.f1 = self._project(self.f1, is_hermitian=False, name="f1")
        for symmetrize in self.symmetrizers:
            self.f1 = symmetrize(self.f1)
            self.f2 = symmetrize(self.f2)

    @override
    def next_r_lambda(self) -> tuple[list[BlockMatrix], list[BlockMatrix]]:
        Lambda: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        R: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                Lambda[i][bl] = helpers.get_lambda(
                    self.R[i][bl],
                    self.D[i][bl],
                    self.Lambda_c[i][bl],
                    self.rho_f[i][bl],
                )
                R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl])

        # Enforce matrix structure from symmetries
        Lambda = self._project(Lambda, is_hermitian=True, name="Lambda")
        R = self._project(R, is_hermitian=False, name="R")
        for symmetrize in self.symmetrizers:
            Lambda = symmetrize(Lambda)
            R = symmetrize(R)

        if self.pin_full_empty is not None:
            energy_scale = self._energy_scale()
            for i in range(self.n_clusters):
                for bl in Lambda[i]:
                    Lambda[i][bl] = self.pin_full_empty.qp_levels(
                        Lambda[i][bl],
                        self.Lambda[i][bl],
                        self.rho_f[i][bl],
                        energy_scale,
                        self._kinetic_half_bandwidth,
                    )
        return R, Lambda

    def _energy_scale(self) -> float:
        # The largest energy of the model. It is set by the model and not by
        # Lambda, which can drift together with the chemical potential, so
        # that the pinned levels do not grow from cycle to cycle
        largest = max(
            float(np.max(np.abs(A[bl]), initial=0))
            for A in self.h0_loc_matrix
            for bl in A
        )
        for solver in self.impurity_solvers:
            terms = solver.h_int.terms.values()
            largest = max([largest, *(abs(complex(coeff)) for coeff in terms)])
        return self._kinetic_half_bandwidth + largest

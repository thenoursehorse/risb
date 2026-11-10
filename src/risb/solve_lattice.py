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
from risb.kintegration import (
    BandKIntegration,
    BandKIntegrationResult,
    BandObservables,
    Hamiltonian,
    KPoints,
    max_occupation_slope,
)
from risb.solve_lattice_base import BaseLatticeSolver
from risb.stabilization import (
    FullEmptyPinning,
    LinearWeightedDensityFeedback,
    MuNewtonStep,
    MuUnitStep,
)
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    ChemicalPotentialStep,
    DensityFeedback,
    GaugeFix,
    GfStruct,
    ImpurityChargeSusceptibility,
    ImpuritySolver,
    PinFullEmpty,
    Symmetrize,
)

_DEFAULT_PINNING = FullEmptyPinning()
_DEFAULT_FEEDBACK = LinearWeightedDensityFeedback()
_DEFAULT_MU_STEP = MuUnitStep()
_DEFAULT_REFIT_STEP = MuNewtonStep(min_slope=1e-3)


class ZeroTemperatureLatticeSolver(BaseLatticeSolver):
    """
    Base for the zero-temperature RISB lattice solvers.

    It computes the bath, the residuals and the next :attr:`R` and
    :attr:`Lambda` from the zero-temperature RISB equations, and can pin the
    levels of full and empty orbitals and shift the quasiparticle levels
    against the density mismatch.

    Parameters
    ----------
    band_integrator : risb.kintegration.BandKIntegration
        Integrates the quasiparticle densities and kinetic matrices.
    pin_full_empty : risb.types.PinFullEmpty or None, optional
        Sets the levels of full or empty auxiliary orbitals. Default is
        :class:`risb.stabilization.FullEmptyPinning` with its default parameters.
        ``None`` does no pinning.
    density_feedback : risb.types.DensityFeedback or None, optional
        Shifts the quasiparticle levels against the density mismatch, before
        pinning, e.g.,
        :class:`risb.stabilization.LinearWeightedDensityFeedback`. Default is
        no feedback.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Use :class:`~risb.diagnostics.LatticeDiagnostics` for progress reports.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, force_real_atol, force_real_rtol, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, gauge_fix, clip_r
        As in :class:`.BaseLatticeSolver`.

    """

    def __init__(
        self,
        h0_k: BlockMatrixLike,
        gf_struct: GfStruct | list[GfStruct],
        impurity_solvers: ImpuritySolver | Sequence[ImpuritySolver],
        band_integrator: BandKIntegration,
        projectors: Sequence[BlockMatrixLike] | None = None,
        gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
        symmetrizers: Sequence[Symmetrize] = (),
        force_real: bool = False,
        check_projectors: bool = False,
        double_counting: Callable[[int, BlockMatrix], BlockMatrixLike] | None = None,
        *,
        bath_size: Sequence[Mapping[str, int]] | None = None,
        h0_k_evaluator: Callable[[KPoints], BlockMatrixLike] | None = None,
        projectors_evaluator: Callable[[KPoints], Sequence[BlockMatrixLike]]
        | None = None,
        pin_full_empty: PinFullEmpty | None = _DEFAULT_PINNING,
        density_feedback: DensityFeedback | None = None,
        gauge_fix: GaugeFix | None = None,
        clip_r: bool = True,
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
            band_integrator=band_integrator,
            h0_k_evaluator=h0_k_evaluator,
            projectors_evaluator=projectors_evaluator,
            gauge_fix=gauge_fix,
            clip_r=clip_r,
        )
        self.pin_full_empty = pin_full_empty
        self.density_feedback = density_feedback

    @override
    def update_bath(self) -> None:
        for i in range(self.n_clusters):
            if i in self._copies:
                continue
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

        self.D = self._copy_clusters(self.D)
        self.Lambda_c = self._copy_clusters(self.Lambda_c)

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
            if i in self._copies:
                continue
            for bl, _ in self.gf_struct[i]:
                self.f1[i][bl] = helpers.get_f1(
                    self.rho_cf[i][bl], self.rho_qp[i][bl], self.R[i][bl]
                )
                self.f2[i][bl] = helpers.get_f2(self.rho_f[i][bl], self.rho_qp[i][bl])
        self.f1 = self._copy_clusters(self.f1)
        self.f2 = self._copy_clusters(self.f2)

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
            if i in self._copies:
                continue
            for bl, _ in self.gf_struct[i]:
                Lambda[i][bl] = helpers.get_lambda(
                    self.R[i][bl],
                    self.D[i][bl],
                    self.Lambda_c[i][bl],
                    self.rho_f[i][bl],
                )
                R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl])
        Lambda = self._copy_clusters(Lambda)
        R = self._copy_clusters(R)

        # Enforce matrix structure from symmetries
        Lambda = self._project(Lambda, is_hermitian=True, name="Lambda")
        R = self._project(R, is_hermitian=False, name="R")
        for symmetrize in self.symmetrizers:
            Lambda = symmetrize(Lambda)
            R = symmetrize(R)

        if self.density_feedback is not None:
            slope = self._occupation_slope()
            for i in range(self.n_clusters):
                for bl in Lambda[i]:
                    Lambda[i][bl] = self.density_feedback.qp_levels(
                        Lambda[i][bl], self.R[i][bl], self.f2[i][bl], slope
                    )

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

    def _occupation_slope(self) -> float:
        smearing = getattr(self.band_integrator, "smearing", None)
        if smearing is None:
            return np.inf
        return max_occupation_slope(
            smearing, getattr(self.band_integrator, "beta", np.inf)
        )

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


class LatticeSolver(ZeroTemperatureLatticeSolver):
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with a local interaction on each cluster.

    It solves the zero-temperature RISB equations, with the chemical potential
    of the quasiparticle bands set by their filling on `band_integrator`.

    Parameters
    ----------
    band_integrator : risb.kintegration.BandKIntegration
        Integrates the quasiparticle densities and kinetic matrices.
        Set beta and the band filling on this integrator.
    pin_full_empty, density_feedback, diagnostics
        As in :class:`.ZeroTemperatureLatticeSolver`.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, force_real_atol, force_real_rtol, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, gauge_fix, clip_r
        As in :class:`.BaseLatticeSolver`.

    """


class PhysicalMuLatticeSolver(ZeroTemperatureLatticeSolver):
    r"""
    Zero-temperature RISB lattice solver at a fixed physical chemical potential or filling.

    The quasiparticle bands are filled at the chemical potential :attr:`mu`.
    With `n_target`, :attr:`mu` is solved for with :attr:`Lambda` and
    :attr:`R` so that :attr:`n_electrons` is `n_target`. Each step of
    :attr:`mu` also shifts :attr:`Lambda` by the same amount and
    :attr:`Lambda_c` the other way, so the quasiparticle levels of the
    clusters keep their place relative to :attr:`mu`. Their occupations then change only through hybridization with
    the orbitals in no cluster. A solution is a solution of
    :class:`LatticeSolver` with the same :attr:`Lambda` and :attr:`R`.

    At a solution the filling of the quasiparticle bands is `n_target` plus
    :attr:`n_ghost_electrons`. While solving, :attr:`mu` is kept where the
    filling is within `max_qp_filling_deviation` of this, so that a step of
    :attr:`mu` cannot fill or empty every band.

    Parameters
    ----------
    band_integrator : risb.kintegration.BandKIntegration
        Integrates the quasiparticle densities and kinetic matrices. Set beta
        on this integrator. Its mu and filling are not used.
    mu : float, optional
        Chemical potential, or with `n_target` the starting guess for it. The
        starting :attr:`Lambda` is shifted by `mu`. Default (only with
        `n_target`) is the chemical potential at which the quasiparticle bands
        of :attr:`Lambda` and :attr:`R` at the first :meth:`solve` without
        ``x0`` have filling `n_target` plus :attr:`n_ghost_electrons`.
    n_target : float, optional
        Number of electrons per unit cell. The chemical potential :attr:`mu`
        is then solved for with :attr:`Lambda` and :attr:`R`, and the root
        functions include ``n_electrons - n_target``. Default is a fixed `mu`.
    max_qp_filling_deviation : float or None, optional
        Largest deviation of the quasiparticle filling from its value at a
        solution, in electrons per unit cell. ``None`` does not limit it.
    mu_step : risb.types.ChemicalPotentialStep, optional
        The step of :attr:`mu` after each cycle, with `n_target`. Default is
        :class:`risb.stabilization.MuUnitStep`; see also
        :class:`risb.stabilization.MuNewtonStep`.
    impurity_charge_susceptibility : callable or sequence of callable, optional
        Charge susceptibility of the physical orbitals of an impurity, e.g.,
        :func:`risb.impurity_solvers.fockit.charge_susceptibility`, for the
        :math:`dn/d\mu` that `mu_step` may use. One for all clusters, or one
        per cluster in the order of `impurity_solvers`; ``None`` leaves an
        impurity out. A cluster with a :class:`.DummyImpuritySolver` takes
        that of the cluster it copies. Default leaves every impurity out.
    max_filling_error : float or None, optional
        With `n_target`, the chemical potential of the impurities is refitted
        in each cycle until ``|n_electrons - n_target|`` is at most this. A
        refit moves :attr:`mu` and :attr:`Lambda` by :math:`\delta\mu` and
        :attr:`Lambda_c` by :math:`-\delta\mu`, with :attr:`D` and the
        double counting held, and solves the impurities again. At the fixed
        particle number of each impurity this adds :math:`-\delta\mu N_c`,
        with :math:`N_c` the number of electrons on its physical orbitals. The
        step comes from `mu_impurity_refit_step`, kept between the chemical
        potentials found with too few and too many electrons. The rest of the
        cycle runs at the last refit. ``None`` does not refit.
    mu_impurity_refit_step : risb.types.ChemicalPotentialStep, optional
        The step of each refit. Default is
        :class:`risb.stabilization.MuNewtonStep` with ``min_slope = 1e-3``,
        which needs `impurity_charge_susceptibility`; see also
        :class:`risb.stabilization.MuSecantStep`, which does not.
    max_mu_refits : int, optional
        Largest number of refits in a cycle. When they run out, the cycle
        carries on.
    density_feedback : risb.types.DensityFeedback or None, optional
        Shifts the quasiparticle levels against the density mismatch, before
        pinning. Default is
        :class:`risb.stabilization.LinearWeightedDensityFeedback` with its
        default parameters. ``None`` does no feedback.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, pin_full_empty, gauge_fix, clip_r, force_real_atol, force_real_rtol, diagnostics
        As in :class:`.ZeroTemperatureLatticeSolver`.

    """

    def __init__(
        self,
        h0_k: BlockMatrixLike,
        gf_struct: GfStruct | list[GfStruct],
        impurity_solvers: ImpuritySolver | Sequence[ImpuritySolver],
        band_integrator: BandKIntegration,
        projectors: Sequence[BlockMatrixLike] | None = None,
        gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
        symmetrizers: Sequence[Symmetrize] = (),
        force_real: bool = False,
        check_projectors: bool = False,
        double_counting: Callable[[int, BlockMatrix], BlockMatrixLike] | None = None,
        *,
        mu: float | None = None,
        n_target: float | None = None,
        max_qp_filling_deviation: float | None = 0.75,
        mu_step: ChemicalPotentialStep = _DEFAULT_MU_STEP,
        impurity_charge_susceptibility: ImpurityChargeSusceptibility
        | Sequence[ImpurityChargeSusceptibility | None]
        | None = None,
        max_filling_error: float | None = None,
        mu_impurity_refit_step: ChemicalPotentialStep = _DEFAULT_REFIT_STEP,
        max_mu_refits: int = 3,
        bath_size: Sequence[Mapping[str, int]] | None = None,
        h0_k_evaluator: Callable[[KPoints], BlockMatrixLike] | None = None,
        projectors_evaluator: Callable[[KPoints], Sequence[BlockMatrixLike]]
        | None = None,
        pin_full_empty: PinFullEmpty | None = _DEFAULT_PINNING,
        density_feedback: DensityFeedback | None = _DEFAULT_FEEDBACK,
        gauge_fix: GaugeFix | None = None,
        clip_r: bool = True,
        force_real_atol: float = 1e-12,
        force_real_rtol: float = 1e-10,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        super().__init__(
            h0_k,
            gf_struct,
            impurity_solvers,
            band_integrator,
            projectors=projectors,
            gf_struct_mapping=gf_struct_mapping,
            symmetrizers=symmetrizers,
            force_real=force_real,
            check_projectors=check_projectors,
            double_counting=double_counting,
            bath_size=bath_size,
            h0_k_evaluator=h0_k_evaluator,
            projectors_evaluator=projectors_evaluator,
            pin_full_empty=pin_full_empty,
            density_feedback=density_feedback,
            gauge_fix=gauge_fix,
            clip_r=clip_r,
            force_real_atol=force_real_atol,
            force_real_rtol=force_real_rtol,
            diagnostics=diagnostics,
        )

        self.max_qp_filling_deviation = max_qp_filling_deviation
        self._set_chemical_potential(
            mu,
            n_target,
            mu_step,
            impurity_charge_susceptibility,
            max_filling_error,
            mu_impurity_refit_step,
            max_mu_refits,
        )

    @override
    def integrate_bands(
        self, hamiltonian: Hamiltonian, observables: BandObservables
    ) -> BandKIntegrationResult:
        """Integrate the quasiparticle bands at :attr:`mu`, with the filling kept within `max_qp_filling_deviation` of its value at a solution."""
        result = super().integrate_bands(hamiltonian, observables)
        if self.n_target is None or self.max_qp_filling_deviation is None:
            return result
        filling = self.n_target + self.n_ghost_electrons
        excess = result.n_electrons - filling
        if abs(excess) <= self.max_qp_filling_deviation:
            return result
        result = self.band_integrator.integrate(
            hamiltonian,
            observables=observables,
            n_target=filling + np.copysign(self.max_qp_filling_deviation, excess),
        )
        self.mu = float(result.mu)
        return result

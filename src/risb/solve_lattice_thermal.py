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

"""Solver for finite-temperature rotationally invariant slave-boson mean-field theory on a lattice."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import override

import numpy as np
from numpy.typing import NDArray

from risb import helpers
from risb.diagnostics import DiagnosticObserver
from risb.kintegration import (
    BandKIntegration,
    BandKIntegrationResult,
    BandKIntegrator,
    BandObservables,
    Hamiltonian,
    KPoints,
    WeightedKPointIntegrator,
)
from risb.optimize import DIIS, NewtonSolver, OnPlateau, ScipyRoot, SolveResult
from risb.optimize.fixed_point import Callback
from risb.solve_lattice_base import BaseLatticeSolver
from risb.stabilization import MuUnitStep
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    Checkpoint,
    ChemicalPotentialStep,
    DensityMatchingJacobian,
    DensityMatchingRegularization,
    GaugeFix,
    GfStruct,
    ImpurityChargeSusceptibility,
    ImpuritySolver,
    Symmetrize,
)

_DEFAULT_MU_STEP = MuUnitStep()
# Step of mu for the slope of the filling of the bands
_SLOPE_DMU = 1e-4


class ThermalLatticeSolver(BaseLatticeSolver):
    """
    Finite-temperature RISB lattice solver at a fixed chemical potential or filling.

    It solves the finite-temperature ghost RISB equations of `arXiv:2603.20559
    <https://arxiv.org/abs/2603.20559>`__ at inverse temperature `beta`. The
    impurity problems are thermal states over every particle number, and the
    quadratic embedding Hamiltonian of :func:`risb.helpers.get_h0_emb` links
    them to the quasiparticles. Each cycle fits :attr:`D` and
    :attr:`Lambda_c` so that the quadratic embedding Hamiltonian has the
    density matrices :attr:`rho_qp` and :attr:`lopsided_ke_qp` of the
    quasiparticles, solves the impurity problems, and then fits
    :attr:`Lambda` and :attr:`R` so that it has their :attr:`rho_f` and
    :attr:`rho_cf`.

    Parameters
    ----------
    impurity_solvers : risb.types.ImpuritySolver or sequence of risb.types.ImpuritySolver
        As in :class:`.BaseLatticeSolver`, but each must solve the thermal
        state at `beta` over every particle number, e.g.,
        :class:`.FockitThermalImpuritySolver`.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential, or with `n_target` the starting guess for it.
        Default (only with `n_target`) is the chemical potential at which the
        quasiparticle bands of :attr:`Lambda` and :attr:`R` at the first
        :meth:`solve` without ``x0`` have filling `n_target` plus
        :attr:`n_ghost_electrons`.
    n_target : float, optional
        Number of electrons per unit cell. The chemical potential :attr:`mu`
        is then solved for with :attr:`Lambda` and :attr:`R`, and the root
        functions include ``n_electrons - n_target``. Default is a fixed `mu`.
    mu_step : risb.types.ChemicalPotentialStep, optional
        The step of :attr:`mu` after each cycle, with `n_target`. Default is
        :class:`risb.stabilization.MuUnitStep`; see also
        :class:`risb.stabilization.MuNewtonStep`.
    impurity_charge_susceptibility : callable or sequence of callable, optional
        As in :class:`.PhysicalMuLatticeSolver`, but of the thermal state of
        each impurity.
    band_integrator : risb.kintegration.BandKIntegration, optional
        Band integration method. If omitted, use Fermi smearing on the
        supplied equally weighted mesh. Each cycle evaluates the band
        integrals at the physical beta and the current chemical potential.
    density_matching_jacobian : {"2-point", "analytic"}, optional
        Use finite differences (``"2-point"``) or analytic
        derivatives (``"analytic"``) for both density-matching steps.
    density_matching_regularization : {"uniform", "response-weighted"}, optional
        Generalized Tikhonov regularization penalizes the squared distance
        from the reference values set by :meth:`anchor`.
        ``"uniform"`` gives all parameter directions the same weight.
        ``"response-weighted"`` gives more weight to directions in which
        changes have little effect on the density matrices. The penalty
        applies only after :meth:`anchor` is called, for example by
        :class:`~risb.optimize.OnPlateau`.
    density_matching_regularization_scale : float, optional
        Positive scale for weighting the singular values from an SVD of
        the matching Jacobian. Directions with singular values much smaller
        than this scale receive nearly the full anchor penalty; directions
        with larger values receive less. The value depends on the energy units.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Use :class:`~risb.diagnostics.ThermalLatticeDiagnostics` for progress
        and fit reports.

    Other Parameters
    ----------------
    h0_k, gf_struct, projectors, gf_struct_mapping, symmetrizers, force_real, force_real_atol, force_real_rtol, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, gauge_fix, clip_r
        As in :class:`.BaseLatticeSolver`.

    """

    def __init__(
        self,
        h0_k: BlockMatrixLike,
        gf_struct: GfStruct | list[GfStruct],
        impurity_solvers: ImpuritySolver | Sequence[ImpuritySolver],
        beta: float,
        mu: float | None = None,
        projectors: Sequence[BlockMatrixLike] | None = None,
        gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
        symmetrizers: Sequence[Symmetrize] = (),
        force_real: bool = False,
        check_projectors: bool = False,
        double_counting: Callable[[int, BlockMatrix], BlockMatrixLike] | None = None,
        *,
        bath_size: Sequence[Mapping[str, int]] | None = None,
        n_target: float | None = None,
        band_integrator: BandKIntegration | None = None,
        h0_k_evaluator: Callable[[KPoints], BlockMatrixLike] | None = None,
        projectors_evaluator: Callable[[KPoints], Sequence[BlockMatrixLike]]
        | None = None,
        density_matching_jacobian: DensityMatchingJacobian = "2-point",
        density_matching_regularization: DensityMatchingRegularization = "uniform",
        density_matching_regularization_scale: float = 1e-3,
        mu_step: ChemicalPotentialStep = _DEFAULT_MU_STEP,
        impurity_charge_susceptibility: ImpurityChargeSusceptibility
        | Sequence[ImpurityChargeSusceptibility | None]
        | None = None,
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
            band_integrator=BandKIntegrator(WeightedKPointIntegrator())
            if band_integrator is None
            else band_integrator,
            h0_k_evaluator=h0_k_evaluator,
            projectors_evaluator=projectors_evaluator,
            gauge_fix=gauge_fix,
            clip_r=clip_r,
            mu=mu,
        )
        #: float : Inverse temperature.
        self.beta = beta
        self.density_matching_jacobian = density_matching_jacobian
        self.density_matching_regularization = density_matching_regularization
        self.density_matching_regularization_scale = (
            density_matching_regularization_scale
        )

        #: float or None : Number of electrons per unit cell to solve for.
        self.n_target = n_target
        self.mu_step = mu_step
        self._impurity_charge_susceptibility = self._per_cluster(
            impurity_charge_susceptibility
        )

        #: float : Largest absolute entry of the bath density and kinetic
        #: matching residuals of the current cycle, excluding the anchor penalty.
        self.bath_fit_error: float = np.inf

        # The strength and the reference parameters of each fit after anchor()
        self._anchor_alpha: float | None = None
        self._anchors: dict[
            tuple[str, int, str], tuple[NDArray[np.inexact], NDArray[np.inexact]]
        ] = {}

        self._mu_from_bands = mu is None
        self.mu: float
        if mu is not None:
            self.mu = mu
            # The quasiparticle levels start at the chemical potential
            for i in range(self.n_clusters):
                for bl in self.Lambda[i]:
                    self.Lambda[i][bl] = self.Lambda[i][bl] + mu * np.eye(
                        self.Lambda[i][bl].shape[0]
                    )
        elif n_target is not None:
            self.mu = self._band_mu(n_target)
        else:
            msg = "Need mu, or n_target to solve for it !"
            raise ValueError(msg)

    @override
    def _diagnostic_configuration(self) -> dict[str, object]:
        return super()._diagnostic_configuration() | {
            "beta": self.beta,
            "mu": self.mu,
            "n_target": self.n_target,
            "density_matching_jacobian": self.density_matching_jacobian,
            "density_matching_regularization": self.density_matching_regularization,
            "density_matching_regularization_scale": self.density_matching_regularization_scale,
        }

    @override
    def integrate_bands(
        self, hamiltonian: Hamiltonian, observables: BandObservables
    ) -> BandKIntegrationResult:
        """Integrate at the physical beta and the current outer-loop mu."""
        return self.band_integrator.integrate(
            hamiltonian, observables=observables, beta=self.beta, mu=self.mu
        )

    @override
    def update_bath(self) -> None:
        for i in range(self.n_clusters):
            if i in self._copies:
                continue
            for bl, _ in self.gf_struct[i]:
                R, Lambda = self.R[i][bl], self.Lambda[i][bl]
                rho_qp, ke = self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                # The fit starts from the previous cycle, or else from the
                # zero-temperature D and Lambda_c
                if bl in self.D[i]:
                    D, Lambda_c = self.D[i][bl], self.Lambda_c[i][bl]
                else:
                    D = helpers.get_d(rho_qp, ke)
                    Lambda_c = helpers.get_lambda_c(rho_qp, R, Lambda, D)
                self._warn_if_complex(D, "D", i, bl)
                self._warn_if_complex(Lambda_c, "Lambda_c", i, bl)
                with self._stage("bath_fit", cluster=i, block=bl):
                    self.D[i][bl], self.Lambda_c[i][bl] = (
                        helpers.get_thermal_d_lambda_c(
                            rho_qp,
                            ke,
                            R,
                            Lambda,
                            self.beta,
                            self.mu,
                            D,
                            Lambda_c,
                            anchor=self._anchor_of(("bath", i, bl), (D, Lambda_c)),
                            alpha=self._anchor_alpha or 0.0,
                            jacobian=self.density_matching_jacobian,
                            regularization=self.density_matching_regularization,
                            regularization_scale=self.density_matching_regularization_scale,
                            force_real=self.force_real,
                            diagnostics=partial(
                                self._emit,
                                cluster=i,
                                block=bl,
                                fit="bath",
                                iteration=self.iteration + 1,
                            )
                            if self.diagnostics is not None
                            else None,
                        )
                    )

        self.D = self._copy_clusters(self.D)
        self.Lambda_c = self._copy_clusters(self.Lambda_c)

        # Enforce matrix structure from symmetries
        self.Lambda_c = self._project(self.Lambda_c, is_hermitian=True, name="Lambda_c")
        self.D = self._project(self.D, is_hermitian=False, name="D")
        for symmetrize in self.symmetrizers:
            self.Lambda_c = symmetrize(self.Lambda_c)
            self.D = symmetrize(self.D)

    @override
    def update_residuals(self) -> None:
        self.bath_fit_error = 0.0
        for i in range(self.n_clusters):
            if i in self._copies:
                continue
            for bl, _ in self.gf_struct[i]:
                R = self.R[i][bl]
                rho_qp_emb, rho_qpf, rho_f_emb = helpers.get_h0_emb_densities(
                    R,
                    self.Lambda[i][bl],
                    self.D[i][bl],
                    self.Lambda_c[i][bl],
                    self.beta,
                    self.mu,
                )
                self.bath_fit_error = max(
                    self.bath_fit_error,
                    float(np.max(np.abs(rho_qp_emb - self.rho_qp[i][bl]))),
                    float(
                        np.max(
                            np.abs(
                                rho_qpf @ self.D[i][bl] - self.lopsided_ke_qp[i][bl].T
                            )
                        )
                    ),
                )
                self.f1[i][bl] = helpers.get_thermal_f1(self.rho_cf[i][bl], rho_qpf, R)
                self.f2[i][bl] = helpers.get_thermal_f2(self.rho_f[i][bl], rho_f_emb)
        self.f1 = self._copy_clusters(self.f1)
        self.f2 = self._copy_clusters(self.f2)

        # Enforce matrix structure from symmetries
        self.f2 = self._project(self.f2, is_hermitian=True, name="f2")
        self.f1 = self._project(self.f1, is_hermitian=False, name="f1")
        for symmetrize in self.symmetrizers:
            self.f1 = symmetrize(self.f1)
            self.f2 = symmetrize(self.f2)

        self._emit(
            "filling",
            iteration=self.iteration + 1,
            mu=self.mu,
            filling=self.n_electrons,
            target=self.n_target,
        )

    @override
    def next_r_lambda(self) -> tuple[list[BlockMatrix], list[BlockMatrix]]:
        R_next: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        Lambda_next: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        for i in range(self.n_clusters):
            if i in self._copies:
                continue
            for bl, _ in self.gf_struct[i]:
                R, Lambda = self.R[i][bl], self.Lambda[i][bl]
                self._warn_if_complex(R, "R", i, bl)
                self._warn_if_complex(Lambda, "Lambda", i, bl)
                with self._stage("quasiparticles_fit", cluster=i, block=bl):
                    R_next[i][bl], Lambda_next[i][bl] = helpers.get_thermal_r_lambda(
                        self.rho_f[i][bl],
                        self.rho_cf[i][bl],
                        self.D[i][bl],
                        self.Lambda_c[i][bl],
                        self.beta,
                        self.mu,
                        R,
                        Lambda,
                        anchor=self._anchor_of(("qp", i, bl), (R, Lambda)),
                        alpha=self._anchor_alpha or 0.0,
                        jacobian=self.density_matching_jacobian,
                        regularization=self.density_matching_regularization,
                        regularization_scale=self.density_matching_regularization_scale,
                        force_real=self.force_real,
                        diagnostics=partial(
                            self._emit,
                            cluster=i,
                            block=bl,
                            fit="quasiparticles",
                            iteration=self.iteration + 1,
                        )
                        if self.diagnostics is not None
                        else None,
                    )

        R_next = self._copy_clusters(R_next)
        Lambda_next = self._copy_clusters(Lambda_next)

        # Enforce matrix structure from symmetries
        R_next = self._project(R_next, is_hermitian=False, name="R")
        Lambda_next = self._project(Lambda_next, is_hermitian=True, name="Lambda")
        for symmetrize in self.symmetrizers:
            R_next = symmetrize(R_next)
            Lambda_next = symmetrize(Lambda_next)
        return R_next, Lambda_next

    @override
    def get_x(self) -> NDArray[np.float64]:
        """
        Return :attr:`Lambda`, :attr:`R`, and with ``n_target`` :attr:`mu`, as the vector that :meth:`cycle` takes.

        Returns
        -------
        numpy.ndarray

        """
        x = super().get_x()
        if self.n_target is None:
            return x
        return np.append(x, self.mu)

    @override
    def cycle(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run one cycle using the parameters packed in x.

        The vector contains Lambda and R, and also mu when n_target is set.

        Parameters
        ----------
        x : numpy.ndarray
            As from :meth:`get_x`.

        Returns
        -------
        x_new : numpy.ndarray
            Next Lambda and R, followed by ``mu`` plus the step of :attr:`mu_step`
            when solving at fixed filling.
        residual : numpy.ndarray
            Packed f2 and f1, followed by ``n_electrons - n_target``
            when n_target is set.

        """
        D, Lambda_c = deepcopy(self.D), deepcopy(self.Lambda_c)
        alpha, anchors = self._anchor_alpha, deepcopy(self._anchors)
        if self.n_target is None:
            x_new, residual = super().cycle(x)
        else:
            self.mu = float(x[-1])
            x_new, residual = super().cycle(x[:-1])
            mu_new = self.mu + self.mu_step.step(
                self.n_electrons - self.n_target,
                self._filling_slope,
                [(self.mu, self.n_electrons)],
            )
            x_new = np.append(x_new, mu_new)
            residual = np.append(residual, self.n_electrons - self.n_target)
        base_checkpoint = super().checkpoint()

        def restore() -> None:
            self.D, self.Lambda_c = deepcopy(D), deepcopy(Lambda_c)
            self._anchor_alpha, self._anchors = alpha, deepcopy(anchors)
            base_checkpoint.restore()
            self._checkpoint = saved

        saved = Checkpoint(restore, evaluations=base_checkpoint.evaluations)
        self._checkpoint = saved
        return x_new, residual

    def _filling_slope(self) -> float:
        # dn_electrons/dmu at the R and Lambda of the cycle: the bands of the
        # orbitals in no cluster, and the impurities given
        bands = (
            self._filling_outside_clusters(_SLOPE_DMU)
            - self._filling_outside_clusters(0.0)
        ) / _SLOPE_DMU
        return bands + self._impurity_filling_slope(
            self._impurity_charge_susceptibility
        )

    def _filling_outside_clusters(self, dmu: float) -> float:
        # Electrons in the bands of the orbitals in no cluster after a step dmu
        # of mu, at the R and Lambda of the last cycle
        hamiltonian, observables = self._get_band_integrands()
        result = self.band_integrator.integrate(
            hamiltonian, observables=observables, beta=self.beta, mu=self.mu + dmu
        )
        return float(result.n_electrons) - sum(
            float(np.trace(result.values["rho_qp", i, bl]).real)
            for i, cluster in enumerate(self.gf_struct)
            for bl, _ in cluster
        )

    @override
    def default_optimizer(self) -> NewtonSolver:
        """
        Return the optimizer that :meth:`solve` uses by default.

        Returns
        -------
        risb.optimize.NewtonSolver
            :class:`.DIIS` with the default :class:`.Pulay` policy, using the
            matching residuals. At high temperature, Lambda and R can keep
            changing even when the matching equations are satisfied.

        """
        return DIIS()

    @override
    def default_callbacks(self) -> list[Callback]:
        """
        Return the callbacks that :meth:`solve` uses by default.

        Returns
        -------
        list of callable
            :class:`.OnPlateau` with its default options.

        """
        return [OnPlateau()]

    @override
    def solve(
        self,
        optimizer: NewtonSolver | ScipyRoot | None = None,
        callbacks: Sequence[Callback] | None = None,
        tol: float = 1e-8,
        maxiter: int = 1000,
        self_consistency_tol: float = 1e-6,
        x0: NDArray[np.float64] | None = None,
        rho_c0: Sequence[BlockMatrixLike] | None = None,
    ) -> SolveResult:
        """
        Solve as :meth:`.BaseLatticeSolver.solve`, after :meth:`release`.

        On the first solve without `x0`, a chemical potential that was not
        given is set from the quasiparticle bands of the current
        :attr:`Lambda` and :attr:`R`.

        Parameters
        ----------
        optimizer, callbacks, tol, maxiter, self_consistency_tol, x0, rho_c0
            As in :meth:`.BaseLatticeSolver.solve`.

        Returns
        -------
        risb.optimize.SolveResult

        """
        self.release()
        if self._mu_from_bands and x0 is None:
            assert self.n_target is not None
            self.mu = self._band_mu(self.n_target)
        self._mu_from_bands = False
        return super().solve(
            optimizer=optimizer,
            callbacks=callbacks,
            tol=tol,
            maxiter=maxiter,
            self_consistency_tol=self_consistency_tol,
            x0=x0,
            rho_c0=rho_c0,
        )

    def _band_mu(self, n_target: float) -> float:
        # The chemical potential at which the quasiparticle bands have the
        # filling n_target plus that of the ghosts
        self.R_full, self.Lambda_full = helpers.upfold_r_lambda(
            self.R,
            self.Lambda,
            self.h0_kin_k,
            self._projectors_qp,
            self.gf_struct_mapping,
        )
        hamiltonian, _ = self._get_band_integrands()
        return self.band_integrator.integrate(
            hamiltonian, beta=self.beta, n_target=n_target + self.n_ghost_electrons
        ).mu

    def anchor(self, alpha: float = 1e-8) -> None:
        """
        Keep :attr:`D`, :attr:`Lambda_c`, :attr:`R` and :attr:`Lambda` close to fixed reference values.

        Each matching fit saves its starting parameters on its first call
        after this method. Later fits penalize deviations from those values.
        At high temperature, some combinations of parameters can move
        without changing the density matrices much.
        The anchor limits that motion until :meth:`release` or the next
        :meth:`solve`. Use ``density_matching_regularization`` to choose
        which parameter changes to penalize.

        Parameters
        ----------
        alpha : float, optional
            Strength of the anchor.

        """
        self._anchor_alpha = alpha
        self._anchors = {}
        self._emit("anchor.set", alpha=alpha)

    def release(self) -> None:
        """Remove the anchor of :meth:`anchor`."""
        was_anchored = self._anchor_alpha is not None
        self._anchor_alpha = None
        self._anchors = {}
        if was_anchored:
            self._emit("anchor.released")

    def _anchor_of(
        self,
        key: tuple[str, int, str],
        start: tuple[NDArray[np.inexact], NDArray[np.inexact]],
    ) -> tuple[NDArray[np.inexact], NDArray[np.inexact]] | None:
        if self._anchor_alpha is None:
            return None
        return self._anchors.setdefault(key, (start[0].copy(), start[1].copy()))

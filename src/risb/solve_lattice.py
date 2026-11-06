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
from risb.diagnostics import DiagnosticObserver
from risb.kintegration import (
    BandKIntegration,
    BandKIntegrationResult,
    BandObservables,
    Hamiltonian,
    KPoints,
    max_occupation_slope,
)
from risb.optimize import NewtonSolver, ScipyRoot, SolveResult
from risb.optimize.fixed_point import Callback
from risb.solve_lattice_base import BaseLatticeSolver
from risb.stabilization import FullEmptyPinning, LinearWeightedDensityFeedback
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    DensityFeedback,
    GaugeFix,
    GfStruct,
    ImpuritySolver,
    PinFullEmpty,
    Symmetrize,
)

_DEFAULT_PINNING = FullEmptyPinning()
_DEFAULT_FEEDBACK = LinearWeightedDensityFeedback()


class LatticeSolver(BaseLatticeSolver):
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with a local interaction on each cluster.

    It solves the zero-temperature RISB equations.

    Parameters
    ----------
    band_integrator : risb.kintegration.BandKIntegration
        Integrates the quasiparticle densities and kinetic matrices.
        Set beta and the band filling on this integrator.
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
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, force_real_atol, force_real_rtol, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, gauge_fix
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
        )
        self.pin_full_empty = pin_full_empty
        self.density_feedback = density_feedback

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


class PhysicalMuLatticeSolver(LatticeSolver):
    """
    Zero-temperature RISB lattice solver at a fixed physical chemical potential or filling.

    The quasiparticle bands are filled at the chemical potential :attr:`mu`.
    With `n_target`, :attr:`mu` is solved for with :attr:`Lambda` and
    :attr:`R` so that :attr:`n_electrons` is `n_target`. Each step of
    :attr:`mu` also shifts :attr:`Lambda` by the same amount, so it does not
    change the occupations of the quasiparticles. A solution is a solution of
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
    density_feedback : risb.types.DensityFeedback or None, optional
        Shifts the quasiparticle levels against the density mismatch, before
        pinning. Default is
        :class:`risb.stabilization.LinearWeightedDensityFeedback` with its
        default parameters. ``None`` does no feedback.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, pin_full_empty, gauge_fix, force_real_atol, force_real_rtol, diagnostics
        As in :class:`.LatticeSolver`.

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
        bath_size: Sequence[Mapping[str, int]] | None = None,
        h0_k_evaluator: Callable[[KPoints], BlockMatrixLike] | None = None,
        projectors_evaluator: Callable[[KPoints], Sequence[BlockMatrixLike]]
        | None = None,
        pin_full_empty: PinFullEmpty | None = _DEFAULT_PINNING,
        density_feedback: DensityFeedback | None = _DEFAULT_FEEDBACK,
        gauge_fix: GaugeFix | None = None,
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
            force_real_atol=force_real_atol,
            force_real_rtol=force_real_rtol,
            diagnostics=diagnostics,
        )

        #: float or None : Number of electrons per unit cell to solve for.
        self.n_target = n_target
        self.max_qp_filling_deviation = max_qp_filling_deviation

        self._mu_from_bands = mu is None
        self.mu: float
        if mu is not None:
            self.mu = mu
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
            "mu": self.mu,
            "n_target": self.n_target,
        }

    @override
    def integrate_bands(
        self, hamiltonian: Hamiltonian, observables: BandObservables
    ) -> BandKIntegrationResult:
        """Integrate the quasiparticle bands at :attr:`mu`, with the filling kept within `max_qp_filling_deviation` of its value at a solution."""
        result = self.band_integrator.integrate(
            hamiltonian, observables=observables, mu=self.mu
        )
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

    @override
    def update_residuals(self) -> None:
        super().update_residuals()
        self._emit(
            "filling",
            iteration=self.iteration + 1,
            mu=self.mu,
            filling=self.n_electrons,
            target=self.n_target,
        )

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

        Parameters
        ----------
        x : numpy.ndarray
            As from :meth:`get_x`.

        Returns
        -------
        x_new : numpy.ndarray
            Next Lambda and R, followed by ``mu + n_target - n_electrons``
            when solving at fixed filling.
        residual : numpy.ndarray
            Packed f2 and f1, followed by ``n_electrons - n_target``
            when n_target is set.

        """
        if self.n_target is None:
            return super().cycle(x)
        self.mu = float(x[-1])
        x_new, residual = super().cycle(x[:-1])
        step = self.n_target - self.n_electrons
        identity: list[BlockMatrix] = [
            {bl: np.eye(L.shape[0]) for bl, L in L_i.items()} for L_i in self.Lambda
        ]
        shift = self._flatten_matrix(identity, is_hermitian=True, name="Lambda")
        x_new[: len(shift)] += step * np.asarray(shift)
        return np.append(x_new, self.mu + step), np.append(residual, -step)

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
        Solve as :meth:`.BaseLatticeSolver.solve`.

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
            hamiltonian, n_target=n_target + self.n_ghost_electrons
        ).mu

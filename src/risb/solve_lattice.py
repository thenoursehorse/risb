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
)
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
        pinning. Default is
        :class:`risb.stabilization.LinearWeightedDensityFeedback` with its
        default parameters. ``None`` does no feedback.
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
            for i in range(self.n_clusters):
                for bl in Lambda[i]:
                    Lambda[i][bl] = self.density_feedback.qp_levels(
                        Lambda[i][bl], self.R[i][bl], self.f2[i][bl]
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

    The chemical potential :attr:`mu` is subtracted on the physical orbitals
    of the impurity problems and on the bands of the orbitals in no cluster.
    The chemical potential of the quasiparticle bands is fixed at zero, so
    the bands of the ghosts do not depend on :attr:`mu` directly.

    Parameters
    ----------
    band_integrator : risb.kintegration.BandKIntegration
        Integrates the quasiparticle densities and kinetic matrices. Set beta
        on this integrator. Its mu and filling are not used.
    mu : float, optional
        Chemical potential, or with `n_target` the starting guess for it.
        Default (only with `n_target`) is the chemical potential at which the
        starting quasiparticle bands have filling `n_target` plus
        :attr:`n_ghost_electrons`.
    n_target : float, optional
        Number of electrons per unit cell. The chemical potential :attr:`mu`
        is then solved for with :attr:`Lambda` and :attr:`R`, and the root
        functions include ``n_electrons - n_target``. Default is a fixed `mu`.

    Other Parameters
    ----------------
    h0_k, gf_struct, impurity_solvers, projectors, gf_struct_mapping, symmetrizers, force_real, check_projectors, double_counting, bath_size, h0_k_evaluator, projectors_evaluator, pin_full_empty, density_feedback, gauge_fix, force_real_atol, force_real_rtol, diagnostics
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
        self.mu_on_c_only = True

        #: float or None : Number of electrons per unit cell to solve for.
        self.n_target = n_target

        self.mu: float
        if mu is not None:
            self.mu = mu
        elif n_target is not None:
            self.mu = 0.0
            self.R_full, self.Lambda_full = helpers.upfold_r_lambda(
                self.R,
                self.Lambda,
                self.h0_kin_k,
                self._projectors_qp,
                self.gf_struct_mapping,
            )
            hamiltonian, _ = self._get_band_integrands()
            self.mu = self.band_integrator.integrate(
                hamiltonian, n_target=n_target + self.n_ghost_electrons
            ).mu
            # Shift Lambda so that the starting bands have this filling at zero
            for i in range(self.n_clusters):
                for bl in self.Lambda[i]:
                    self.Lambda[i][bl] = self.Lambda[i][bl] - self.mu * np.eye(
                        self.Lambda[i][bl].shape[0]
                    )
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
    def _get_band_integrands(self) -> tuple[Hamiltonian, BandObservables]:
        hamiltonian, observables = super()._get_band_integrands()
        if not callable(hamiltonian):
            uncorrelated = self._uncorrelated_qp(None)
            return {
                bl: h - self.mu * uncorrelated[bl] for bl, h in hamiltonian.items()
            }, observables

        def get_hamiltonian(kpoints: KPoints) -> BlockMatrix:
            uncorrelated = self._uncorrelated_qp(kpoints)
            return {
                bl: h - self.mu * uncorrelated[bl]
                for bl, h in hamiltonian(kpoints).items()
            }

        return get_hamiltonian, observables

    def _uncorrelated_qp(self, kpoints: KPoints | None) -> BlockMatrix:
        # The projector onto the quasiparticles of the orbitals in no cluster
        if kpoints is None:
            h0, projectors = self.h0_k, self._projectors_qp
        else:
            h0 = self._get_h0_at(kpoints)
            projectors = self._get_projectors_at(kpoints)
            if self.n_ghost_electrons:
                projectors = helpers.get_ghost_projectors(
                    h0, self.bath_size, projectors, self.gf_struct_mapping
                )
        zero = [{bl: np.zeros_like(L) for bl, L in L_i.items()} for L_i in self.Lambda]
        return helpers.upfold(zero, h0, projectors, self.gf_struct_mapping, outside=1)

    @override
    def integrate_bands(
        self, hamiltonian: Hamiltonian, observables: BandObservables
    ) -> BandKIntegrationResult:
        """Integrate the quasiparticle bands at a chemical potential of zero."""
        return self.band_integrator.integrate(
            hamiltonian, observables=observables, mu=0.0
        )

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
        return (
            np.append(x_new, self.mu + self.n_target - self.n_electrons),
            np.append(residual, self.n_electrons - self.n_target),
        )

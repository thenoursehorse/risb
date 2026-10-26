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
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit

from risb import helpers
from risb.optimize import DIIS, NewtonSolver, OnPlateau, SolveResult
from risb.optimize.driver import Callback
from risb.solve_lattice_base import BaseLatticeSolver
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    BlockRealArray,
    GfStruct,
    ImpuritySolver,
    Symmetrize,
)


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
        starting quasiparticle bands hold `n_target` plus
        :attr:`n_ghost_electrons` electrons.
    n_target : float, optional
        Number of electrons per unit cell. The chemical potential :attr:`mu`
        is then solved for with :attr:`Lambda` and :attr:`R`, and the root
        functions include ``n_electrons - n_target``. Default is a fixed `mu`.

    Other Parameters
    ----------------
    h0_k, gf_struct, projectors, gf_struct_mapping, symmetrizers, force_real, check_projectors, double_counting, bath_size
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
            mu=mu,
        )
        #: float : Inverse temperature.
        self.beta = beta

        #: float or None : Number of electrons per unit cell to solve for.
        self.n_target = n_target

        #: float : Number of electrons per unit cell, those of the impurity
        #: problems (:attr:`rho_c`) and of the quasiparticle bands of the
        #: orbitals in no cluster, set each cycle.
        self.n_electrons: float

        # The strength and the reference parameters of each fit after anchor()
        self._anchor_alpha: float | None = None
        self._anchors: dict[
            tuple[str, int, str], tuple[NDArray[np.inexact], NDArray[np.inexact]]
        ] = {}

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
            R_full, Lambda_full = helpers.upfold_r_lambda(
                self.R,
                self.Lambda,
                self.h0_kin_k,
                self._projectors_qp,
                self.gf_struct_mapping,
            )
            energies = {
                bl: np.linalg.eigvalsh(
                    helpers.get_h_qp(R_full[bl], Lambda_full[bl], h0_kin)
                )
                for bl, h0_kin in self.h0_kin_k.items()
            }
            self.mu = _mu_of_filling(energies, beta, n_target + self.n_ghost_electrons)
        else:
            msg = "Need mu, or n_target to solve for it !"
            raise ValueError(msg)

    @override
    def update_weights(self, energies: BlockRealArray) -> BlockRealArray:
        """
        Return the Fermi-Dirac occupation of each band at each k-point, divided by the number of k-points.

        Parameters
        ----------
        energies : risb.types.BlockRealArray
            Energies at each k-point in each block.

        Returns
        -------
        risb.types.BlockRealArray
            Integration weights at each k-point in each block.

        """
        return {
            bl: expit(-self.beta * (e - self.mu)) / e.shape[0]
            for bl, e in energies.items()
        }

    @override
    def update_bath(self) -> None:
        for i in range(self.n_clusters):
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
                self.D[i][bl], self.Lambda_c[i][bl] = helpers.get_thermal_d_lambda_c(
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
                    force_real=self.force_real,
                )

        # Enforce matrix structure from symmetries
        self.Lambda_c = self._project(self.Lambda_c, is_hermitian=True)
        self.D = self._project(self.D, is_hermitian=False)
        for symmetrize in self.symmetrizers:
            self.Lambda_c = symmetrize(self.Lambda_c)
            self.D = symmetrize(self.D)

    @override
    def update_residuals(self) -> None:
        # The electrons of the bands of the orbitals in no cluster are those of
        # all bands minus those of the clusters
        self.n_electrons = float(
            sum(np.sum(w) for w in self.kweights.values())
            + sum(
                np.trace(self.rho_c[i][bl]).real - np.trace(self.rho_qp[i][bl]).real
                for i in range(self.n_clusters)
                for bl in self.rho_c[i]
            )
        )
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                R = self.R[i][bl]
                _, rho_qpf, rho_f_emb = helpers.get_h0_emb_densities(
                    R,
                    self.Lambda[i][bl],
                    self.D[i][bl],
                    self.Lambda_c[i][bl],
                    self.beta,
                    self.mu,
                )
                self.f1[i][bl] = helpers.get_thermal_f1(self.rho_cf[i][bl], rho_qpf, R)
                self.f2[i][bl] = helpers.get_thermal_f2(self.rho_f[i][bl], rho_f_emb)

        # Enforce matrix structure from symmetries
        self.f2 = self._project(self.f2, is_hermitian=True)
        self.f1 = self._project(self.f1, is_hermitian=False)
        for symmetrize in self.symmetrizers:
            self.f1 = symmetrize(self.f1)
            self.f2 = symmetrize(self.f2)

    @override
    def next_r_lambda(self) -> tuple[list[BlockMatrix], list[BlockMatrix]]:
        R_next: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        Lambda_next: list[BlockMatrix] = [{} for _ in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                R, Lambda = self.R[i][bl], self.Lambda[i][bl]
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
                    force_real=self.force_real,
                )

        # Enforce matrix structure from symmetries
        R_next = self._project(R_next, is_hermitian=False)
        Lambda_next = self._project(Lambda_next, is_hermitian=True)
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
        Run :meth:`one_cycle` from the :attr:`Lambda`, :attr:`R`, and with ``n_target`` :attr:`mu`, in ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            As from :meth:`get_x`.

        Returns
        -------
        x_new : numpy.ndarray
            The new :attr:`Lambda` and :attr:`R`, and with ``n_target`` the
            next guess for :attr:`mu`, moved by the error of the filling.
        residual : numpy.ndarray
            The root functions :attr:`f2` and :attr:`f1`, and with
            ``n_target`` ``n_electrons - n_target``, zero at a solution.

        """
        if self.n_target is None:
            return super().cycle(x)
        self.mu = float(x[-1])
        x_new, residual = super().cycle(x[:-1])
        mu_new = self.mu + self.n_target - self.n_electrons
        return np.append(x_new, mu_new), np.append(
            residual, self.n_electrons - self.n_target
        )

    @override
    def default_optimizer(self) -> NewtonSolver:
        """
        Return the optimizer that :meth:`solve` uses by default.

        Returns
        -------
        risb.optimize.NewtonSolver
            :class:`.DIIS` with ``error_fun='root'``. At high temperature
            :attr:`Lambda` and :attr:`R` can keep changing at a solution, so
            their change is not a good error.

        """
        return DIIS(error_fun="root")

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
    def solve(self, *args: Any, **kwargs: Any) -> SolveResult:
        """
        Solve as :meth:`.BaseLatticeSolver.solve`, after :meth:`release`.

        Returns
        -------
        risb.optimize.SolveResult

        """
        self.release()
        return super().solve(*args, **kwargs)

    def anchor(self, alpha: float = 1e-8) -> None:
        """
        Anchor the fits of :attr:`D`, :attr:`Lambda_c`, :attr:`Lambda`, and :attr:`R` to their current values.

        Each fit then also minimizes ``alpha`` times the squared distance of
        its parameters from where its first fit after this starts. At high
        temperature the equations barely fix some combinations of the
        parameters, and the anchor pins them. The anchor holds until
        :meth:`release` or the next :meth:`solve`.

        Parameters
        ----------
        alpha : float, optional
            Strength of the anchor. Default is ``1e-8``.

        """
        self._anchor_alpha = alpha
        self._anchors = {}

    def release(self) -> None:
        """Remove the anchor of :meth:`anchor`."""
        self._anchor_alpha = None
        self._anchors = {}

    def _anchor_of(
        self,
        key: tuple[str, int, str],
        start: tuple[NDArray[np.inexact], NDArray[np.inexact]],
    ) -> tuple[NDArray[np.inexact], NDArray[np.inexact]] | None:
        # Where the first fit of key after anchor started, if anchored
        if self._anchor_alpha is None:
            return None
        return self._anchors.setdefault(key, (start[0].copy(), start[1].copy()))


def _mu_of_filling(
    energies: Mapping[str, NDArray[np.floating]], beta: float, n: float
) -> float:
    # The chemical potential at which the bands hold n electrons
    def excess(mu: float) -> float:
        return float(
            sum(np.sum(expit(-beta * (e - mu))) / e.shape[0] for e in energies.values())
            - n
        )

    low = min(float(np.min(e)) for e in energies.values()) - 50 / beta
    high = max(float(np.max(e)) for e in energies.values()) + 50 / beta
    return float(brentq(excess, low, high))

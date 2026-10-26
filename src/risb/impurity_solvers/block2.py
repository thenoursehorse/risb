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

"""Impurity solvers using DMRG from block2."""

import logging
import math
import tempfile
import warnings
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations, product
from typing import Any, override

import numpy as np
from fockit.algebra import OpSum, c, cdag, commutator, nc, normal_order, zero
from numpy.typing import NDArray
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from risb.helpers_fockit import Op_to_tensors
from risb.helpers_mpi import Comm, SerialComm, gather_tasks, round_robin
from risb.types import BlockMatrix, BlockMatrixLike, GfStruct, ImpuritySolver

logger = logging.getLogger(__name__)

# Energy by which a sector must be lower than the previous one to replace it
_SECTOR_TOL = 1e-8

# Largest difference between the spin-up and spin-down one-body terms of a
# spin-rotation invariant impurity Hamiltonian
_SU2_TOL = 1e-6

# Default of mix_schedule: (strength, sweeps, noise) of each stage of the
# symmetry-mixing sweeps
_MIX_SCHEDULE = ((0.1, 6, 1e-5), (0.01, 6, 1e-5), (0.001, 6, 1e-5), (1e-4, 6, 1e-5))

# Sweeps of each imaginary time step. block2's default of two leaves the
# targeted states of a step unconverged
_SUB_SWEEPS = 4

# Energy tolerance of state-averaged DMRG, which converges slowly in its
# highest states, and the most sweeps of the refinement of each of its states
_ROOTS_TOL = 1e-8
_REFINE_SWEEPS = 20

# Largest weight of the found states outside the window of the lowest states
# without a warning. It changes the density matrices by up to twice as much
_EXCLUDED_WEIGHT_TOL = 1e-4

# Imaginary time and order of the Taylor series of the first step without a
# conserved spin
_SEED_TIME = 0.01
_SEED_ORDER = 6

# block2's matrix product states and operators, which it gives no types for
type _MPS = Any
type _MPO = Any

# A sector: two times the total spin and two times its projection, None where
# it is not fixed
type _Sector = tuple[int | None, int | None]

# The energies, <c_p^dag c_q> of the spin orbitals and tags of the lowest
# states of a sector, the discarded weight, the directory with the states, and
# their energy variances
type _Run = tuple[
    list[float],
    list[NDArray[np.float64]],
    list[str],
    float,
    tempfile.TemporaryDirectory[str],
    list[float],
]

# A pure state of a thermal state: its weight, directory, tag, sector (None
# for a purified state) and particle number
type _ThermalState = tuple[float, str, str, _Sector | None, int]

# Density matrix, energy, log of the partition function, directories, and pure
# states of a thermal state
type _ThermalResult = tuple[
    NDArray[np.float64],
    float,
    float,
    list[tempfile.TemporaryDirectory[str]],
    list[_ThermalState],
]


class _Block2Base(ImpuritySolver):
    # The chain of spatial orbitals, the impurity Hamiltonian as tensors, and
    # the density matrices, shared by the solvers

    #: Conserved quantities besides the particle number.
    symmetries: Collection[str]

    def __init__(
        self,
        gf_struct: GfStruct,
        bond_dim: int,
        spin_blocks: Sequence[tuple[str, str]],
        bath_size: Mapping[str, int] | None,
        comm: Comm | None,
        mix_schedule: Sequence[tuple[float, int, float]],
    ) -> None:
        #: risb.types.GfStruct : Block matrix structure of the physical orbitals
        #: (c-electrons).
        self.gf_struct = gf_struct

        #: risb.helpers_mpi.Comm : Communicator of the ranks the sectors are
        #: split over.
        self.comm = SerialComm() if comm is None else comm

        #: list of tuple : ``(strength, sweeps, noise)`` of each stage of the
        #: symmetry-mixing sweeps that start each solve.
        self.mix_schedule = list(mix_schedule)

        #: int : Bond dimension of the matrix product state.
        self.bond_dim = bond_dim

        sizes = dict(gf_struct)
        bath_sizes = {bl: (bath_size or {}).get(bl, n) for bl, n in gf_struct}
        if sorted(bl for pair in spin_blocks for bl in pair) != sorted(sizes):
            msg = "spin_blocks must pair every block of gf_struct once."
            raise ValueError(msg)

        # The spatial orbitals on the DMRG chain, as their spin-up and
        # spin-down modes: each physical orbital, then its bath orbitals
        # a = k * n + alpha
        self._sites: list[tuple[tuple[str, int], tuple[str, int]]] = []
        self._physical: list[int] = []
        for up, dn in spin_blocks:
            n, n_bath = sizes[up], bath_sizes[up]
            if (sizes[dn], bath_sizes[dn]) != (n, n_bath):
                msg = f"Blocks {up!r} and {dn!r} have different sizes."
                raise ValueError(msg)
            for alpha in range(n):
                self._physical.append(len(self._sites))
                self._sites.append(((up, alpha), (dn, alpha)))
                for a in range(alpha, n_bath, n):
                    self._sites.append(((_bath(up), a), (_bath(dn), a)))
        # The spin orbitals (site j, spin s) are 2 * j + s
        self._index = {
            mode: 2 * j + s
            for j, modes in enumerate(self._sites)
            for s, mode in enumerate(modes)
        }
        self._bath_sizes = bath_sizes
        self._h0_loc: BlockMatrixLike = {bl: np.zeros((n, n)) for bl, n in gf_struct}

        #: risb.types.BlockMatrix : f-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_f: BlockMatrix = {}

        #: risb.types.BlockMatrix : c-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_c: BlockMatrix = {}

        #: risb.types.BlockMatrix : Density matrix of hybridization terms
        #: (c- and f-electrons), set by :meth:`solve`.
        self.rho_cf: BlockMatrix = {}

    def set_h_int(self, h_int: OpSum) -> None:
        """
        Set the interaction terms of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h_int : fockit.algebra.OpSum
            Interaction Hamiltonian of the physical orbitals.

        Raises
        ------
        ValueError
            If ``h_int`` does not conserve the quantities in
            :attr:`symmetries`, has an imaginary constant term, or is
            complex.

        """
        if self.symmetries:
            S_plus, Sz = zero(), zero()
            for j in self._physical:
                up_mode, dn_mode = self._sites[j]
                S_plus += cdag(*up_mode) * c(*dn_mode)
                Sz += 0.5 * (nc(*up_mode) - nc(*dn_mode))
            Q = S_plus if "s" in self.symmetries else Sz
            terms = normal_order(commutator(h_int, Q)).terms.values()
            if any(abs(complex(coeff)) > 1e-10 for coeff in terms):
                msg = "h_int does not conserve the symmetries."
                raise ValueError(msg)
        #: fockit.algebra.OpSum : Interaction Hamiltonian.
        self.h_int = h_int
        const, h1, V = Op_to_tensors(h_int, self._index)
        if abs(np.imag(const)) > 1e-10:
            msg = "h_int is not Hermitian."
            raise ValueError(msg)
        self._const_int, self._h1_int, self._V = (
            float(np.real(const)),
            _real(h1),
            _real(V),
        )
        self._const, self._h1 = self._const_int, self._h1_int

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
            Hybridization coupling, indexed as ``D[bl][a, alpha]`` with ``a`` a
            bath orbital and ``alpha`` a physical orbital. Each key in
            dictionary must follow :attr:`gf_struct`.
        h0_loc_matrix : risb.types.BlockMatrixLike, optional
            Single-particle quadratic couplings of the c-electrons. Each key
            in dictionary must follow :attr:`gf_struct`.
        mu : float, optional
            Chemical potential, subtracted as ``mu * n`` on the physical and
            bath orbitals.

        Raises
        ------
        ValueError
            If a term is complex.

        """
        if h0_loc_matrix is not None:
            self._h0_loc = h0_loc_matrix
        h1 = np.array(self._h1_int, dtype=complex)
        const = complex(self._const_int)
        for bl, n in self.gf_struct:
            fl = _bath(bl)
            cs = [self._index[bl, i] for i in range(n)]
            fs = [self._index[fl, a] for a in range(self._bath_sizes[bl])]
            Lambda_c_bl = np.asarray(Lambda_c[bl])
            D_bl = np.asarray(D[bl])
            h1[np.ix_(cs, cs)] += np.asarray(self._h0_loc[bl])
            # sum Lambda_c[a, b] f_a f_b^dag = tr Lambda_c - sum Lambda_c[a, b] f_b^dag f_a
            h1[np.ix_(fs, fs)] -= Lambda_c_bl.T
            const += np.trace(Lambda_c_bl)
            h1[np.ix_(cs, fs)] += D_bl.T
            h1[np.ix_(fs, cs)] += D_bl.conj()
        if mu is not None:
            h1 -= mu * np.eye(len(h1))
        self._const, self._h1 = _real(np.array(const)).item(), _real(h1)

    @override
    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_f[bl]

    @override
    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_cf[bl]

    @override
    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_c[bl]

    def _check_h_imp(self) -> None:
        h1, V = self._h1, self._V
        if (
            np.abs(h1 - h1.T).max() > 1e-10
            or np.abs(V - V.transpose(3, 2, 1, 0)).max(initial=0) > 1e-10
        ):
            msg = "h_imp is not Hermitian."
            raise ValueError(msg)
        if "s" in self.symmetries:
            diff = np.abs(h1[0::2, 0::2] - h1[1::2, 1::2]).max()
            if diff > _SU2_TOL:
                msg = f"The spin-up and spin-down terms of h_imp differ by {diff:.1e}."
                raise ValueError(msg)

    def _set_rho(self, G: NDArray[np.float64]) -> None:
        # The density matrices from <c_p^dag c_q> of the spin orbitals
        for bl, n in self.gf_struct:
            cs = [self._index[bl, i] for i in range(n)]
            fs = [self._index[_bath(bl), a] for a in range(self._bath_sizes[bl])]
            self.rho_f[bl] = np.eye(len(fs)) - G[np.ix_(fs, fs)]
            self.rho_cf[bl] = G[np.ix_(cs, fs)]
            self.rho_c[bl] = G[np.ix_(cs, cs)]

    def _driver(
        self, scratch: str, sector: _Sector, n_elec: int
    ) -> tuple[DMRGDriver, SymmetryTypes]:
        # A driver of the sector with n_elec electrons, and its symmetry type
        twos, twosz = sector
        L = len(self._sites)
        if twos is not None:
            symm_type, n_sites, spin = SymmetryTypes.SU2, L, twos
        elif twosz is not None:
            symm_type, n_sites, spin = SymmetryTypes.SZ, L, twosz
        else:
            symm_type, n_sites, spin = SymmetryTypes.SGF, 2 * L, 0
        driver = DMRGDriver(scratch=scratch, symm_type=symm_type)
        driver.initialize_system(n_sites=n_sites, n_elec=n_elec, spin=spin)
        return driver, symm_type

    def _run(self, sector: _Sector, n_elec: int, nroots: int) -> _Run:
        # The lowest nroots states in sector with n_elec electrons, by
        # state-averaged DMRG (block Davidson) and a refinement of each for
        # more than one
        twosz = sector[1]
        L = len(self._sites)
        scratch = tempfile.TemporaryDirectory()
        driver, symm_type = self._driver(scratch.name, sector, n_elec)
        tol = 1e-12 if nroots == 1 else _ROOTS_TOL
        ket, mpo = self._optimize(driver, symm_type, nroots, self.bond_dim, tol)
        discarded_weight = float(driver.get_dmrg_results()[1][-1])
        kets = [ket]
        if nroots > 1:
            kets, discarded_weight = self._refine(
                driver,
                mpo,
                [driver.split_mps(ket, i, f"KET{i}") for i in range(nroots)],
                scratch.name,
            )
        # The energies of the stored states, of which the density matrices
        # are. block2's are of the two-site states before their truncation
        energies = [float(driver.expectation(k, mpo, k)) for k in kets]

        Gs = []
        for k in kets:
            G = np.zeros((2 * L, 2 * L))
            if symm_type == SymmetryTypes.SGF:
                G[:] = driver.get_1pdm(k)
            elif symm_type == SymmetryTypes.SZ:
                G[0::2, 0::2], G[1::2, 1::2] = driver.get_1pdm(k)
            elif twosz is None:
                # Spin summed: the average over the multiplet has half in each
                # spin
                G[0::2, 0::2] = G[1::2, 1::2] = np.asarray(driver.get_1pdm(k)) / 2
            else:
                zket, norm = self._sz_state(driver, k, twosz, n_elec)
                dm_up, dm_dn = driver.get_1pdm(zket)
                G[0::2, 0::2], G[1::2, 1::2] = dm_up / norm, dm_dn / norm
            Gs.append(G)
        tags = [k.info.tag for k in kets]
        # After the density matrices, which change kets with a conserved total
        # spin and a spin projection as the variances do
        variances = [self._variance(scratch.name, tag, sector, n_elec) for tag in tags]
        return energies, Gs, tags, discarded_weight, scratch, variances

    def _refine(
        self, driver: DMRGDriver, mpo: _MPO, kets: Sequence[_MPS], scratch: str
    ) -> tuple[list[_MPS], float]:
        # Each state of block Davidson, refined by DMRG of that state alone
        # from it, orthogonal to the refined lower ones, and the largest weight
        # discarded in the last sweeps. The states of block Davidson share one
        # basis and are much less accurate than a single state at the same bond
        # dimension. The penalty of the lower states exceeds every energy
        # difference between the states
        energies = [float(driver.expectation(k, mpo, k)) for k in kets]
        weight = 1 + 2 * (max(energies) - min(energies))
        refined: list[_MPS] = []
        discarded_weight = 0.0
        for ket in kets:
            driver.dmrg(
                mpo,
                ket,
                n_sweeps=_REFINE_SWEEPS,
                bond_dims=[self.bond_dim],
                noises=[1e-6] * 2 + [0.0],
                thrds=[1e-15],
                tol=1e-12,
                iprint=0,
                proj_mpss=refined,
                proj_weights=[weight] * len(refined),
            )
            ket.save_data()
            ket.info.save_data(f"{scratch}/{ket.info.tag}-mps_info.bin")
            refined.append(ket)
            discarded_weight = max(
                discarded_weight, float(driver.get_dmrg_results()[1][-1])
            )
        return refined, discarded_weight

    def _optimize(
        self,
        driver: DMRGDriver,
        symm_type: SymmetryTypes,
        nroots: int,
        M: int,
        tol: float,
    ) -> tuple[_MPS, _MPO]:
        # The lowest nroots states of the sector of driver at bond dimension
        # M, converged to tol, and the MPO of the impurity Hamiltonian
        m1, m2 = max(M // 4, min(M, 25)), max(M // 2, min(M, 50))
        ket = driver.get_random_mps(
            tag="KET", bond_dim=M if self.mix_schedule else m1, nroots=nroots
        )
        if self.mix_schedule:
            mixing = self._mixing(symm_type)
            for strength, sweeps, noise in self.mix_schedule:
                mpo = self._mpo(
                    driver,
                    symm_type,
                    self._const,
                    self._h1 + strength * mixing,
                    self._V,
                )
                driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=sweeps,
                    bond_dims=[M],
                    noises=[noise],
                    thrds=[1e-12],
                    tol=0.0,
                    iprint=0,
                )
            schedule: dict[str, Any] = {
                "bond_dims": [M],
                "noises": [1e-6] * 2 + [0.0],
            }
        else:
            schedule = {
                "bond_dims": [m1] * 2 + [m2] * 2 + [M],
                "noises": [1e-4] * 2 + [1e-5] * 2 + [1e-6] * 2 + [0.0],
            }
        mpo = self._mpo(driver, symm_type, self._const, self._h1, self._V)
        driver.dmrg(mpo, ket, n_sweeps=40, thrds=[1e-15], tol=tol, iprint=0, **schedule)
        return ket, mpo

    def _sz_state(
        self, driver: DMRGDriver, ket: _MPS, twosz: int, n_elec: int
    ) -> tuple[_MPS, float]:
        # The state of the spin multiplet ket with spin projection twosz / 2
        # and n_elec electrons, and its norm, with driver changed to that
        # sector without a conserved total spin
        L = len(self._sites)
        driver.align_mps_center(ket, ref=0)
        zket = driver.mps_change_to_sz(ket, "ZKET", sz=twosz)
        driver.symm_type = SymmetryTypes.SZ
        driver.initialize_system(n_sites=L, n_elec=n_elec, spin=twosz)
        return zket, float(driver.expectation(zket, driver.get_identity_mpo(), zket))

    def _variance(self, scratch: str, tag: str, sector: _Sector, n_elec: int) -> float:
        # <H^2> - <H>^2 of the state tag of sector with n_elec electrons. With a
        # conserved total spin, of its state of a fixed spin projection, since
        # block2's <H^2> does not run with one
        driver, symm_type = self._driver(scratch, sector, n_elec)
        ket, norm = driver.load_mps(tag), 1.0
        twos, twosz = sector
        if twos is not None:
            ket, norm = self._sz_state(
                driver, ket, twos if twosz is None else twosz, n_elec
            )
            symm_type = SymmetryTypes.SZ
        # Without the constant term, which does not change the variance
        mpo = self._mpo(driver, symm_type, 0.0, self._h1, self._V)
        h = float(driver.expectation(ket, mpo, ket)) / norm
        h2 = float(driver.expectation(ket, mpo, ket, stacked_mpo=mpo)) / norm
        return h2 - h * h

    def _state_expectation(
        self,
        scratch: str,
        tag: str,
        sector: _Sector,
        n_elec: int,
        const: float,
        h1: NDArray[np.float64],
        V: NDArray[np.float64],
    ) -> float:
        # <const + sum h1 c^dag c + sum V c^dag c^dag c c> in the state tag of
        # sector with n_elec electrons, averaged over the spin multiplet with a
        # conserved total spin and no spin projection
        twos, twosz = sector
        if twos is None:
            driver, symm_type = self._driver(scratch, sector, n_elec)
            ket = driver.load_mps(tag)
            mpo = self._mpo(driver, symm_type, const, h1, V)
            return float(driver.expectation(ket, mpo, ket))
        twoszs = [twosz] if twosz is not None else list(range(-twos, twos + 1, 2))
        total = 0.0
        for m in twoszs:
            driver, _ = self._driver(scratch, (twos, None), n_elec)
            zket, norm = self._sz_state(driver, driver.load_mps(tag), m, n_elec)
            mpo = self._mpo(driver, SymmetryTypes.SZ, const, h1, V)
            total += float(driver.expectation(zket, mpo, zket)) / norm
        return total / len(twoszs)

    def _mixing(self, symm_type: SymmetryTypes) -> NDArray[np.float64]:
        # Random hoppings between every pair of physical orbitals, which can
        # be far apart on the chain, and between neighbours on the chain, as a
        # one-body matrix of the spin orbitals. Spin flips without a conserved
        # spin, the same in both spins with a conserved total spin
        off = self._h1.copy()
        np.fill_diagonal(off, 0)
        scale = np.abs(off).max()
        L = len(self._sites)
        pairs = set(combinations(self._physical, 2)) | {
            (j, j + 1) for j in range(L - 1)
        }
        rng = np.random.default_rng(0)
        mixing = np.zeros((2 * L, 2 * L))
        for i, j in sorted(pairs):
            r = rng.uniform(-1, 1, size=(2, 2))
            if symm_type == SymmetryTypes.SU2:
                r[1, 1] = r[0, 0]
            if symm_type != SymmetryTypes.SGF:
                r[0, 1] = r[1, 0] = 0
            mixing[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = scale * r
        if symm_type == SymmetryTypes.SGF:
            for j in range(L):
                mixing[2 * j, 2 * j + 1] = scale * rng.uniform(-1, 1)
        return mixing + mixing.T

    def _mpo(
        self,
        driver: DMRGDriver,
        symm_type: SymmetryTypes,
        const: float,
        h1: NDArray[np.float64],
        V: NDArray[np.float64],
        ancilla: bool = False,
    ) -> _MPO:
        # The MPO of const + sum h1 c^dag c + sum V c^dag c^dag c c, with an
        # identity on an ancilla next to each site if ancilla. (ij|kl) is the
        # coefficient of c_i^dag c_k^dag c_l c_j / 2, summed over every i, j,
        # k, l (and spins with a conserved spin). With a conserved spin
        # projection the terms that change it are dropped, which have no
        # expectation value in its sectors
        if symm_type == SymmetryTypes.SGF:
            return driver.get_qc_mpo(
                h1e=h1,
                g2e=2 * V.transpose(0, 3, 1, 2),
                ecore=const,
                ancilla=ancilla,
                iprint=0,
            )
        up, dn = slice(0, None, 2), slice(1, None, 2)
        g_ud = 4 * V[up, dn, dn, up].transpose(0, 3, 1, 2)
        if symm_type == SymmetryTypes.SU2:
            h1e = (h1[up, up] + h1[dn, dn]) / 2
            return driver.get_qc_mpo(
                h1e=h1e, g2e=g_ud, ecore=const, ancilla=ancilla, iprint=0
            )
        g_uu = 2 * V[up, up, up, up].transpose(0, 3, 1, 2)
        g_dd = 2 * V[dn, dn, dn, dn].transpose(0, 3, 1, 2)
        return driver.get_qc_mpo(
            h1e=(h1[up, up], h1[dn, dn]),
            g2e=(g_uu, g_ud, g_dd),
            ecore=const,
            ancilla=ancilla,
            iprint=0,
        )


class Block2ImpuritySolver(_Block2Base):
    """
    Impurity solver using the density matrix renormalization group (DMRG) from block2.

    It finds the ground state of the impurity problem at half filling as a
    matrix product state, for a collinear (spin-diagonal) impurity problem
    with real terms. It needs the optional dependency
    `block2 <https://block2.readthedocs.io>`__.

    Each physical orbital is on the DMRG chain next to its bath orbitals. Each
    solve starts with the sweeps of Appendix A1 of F. A. Wolf, A. Go, I. P.
    McCulloch, A. J. Millis, and U. Schollwoeck, Phys. Rev. X 5, 041032 (2015)
    (`arXiv:1507.08650 <https://arxiv.org/abs/1507.08650>`__), with extra
    random hoppings that fall in the stages of ``mix_schedule``, at the full
    bond dimension. DMRG otherwise does not leave the sectors of quantities
    that the Hamiltonian conserves and the solver does not, e.g. the total
    spin and the parity of each orbital with its bath orbitals in a Kanamori
    impurity, and can end in an excited state. The hoppings keep the
    symmetries in ``symmetries``. At a smaller bond dimension the order of
    nearly degenerate states can differ from that at the full one, and the
    state can end in the sector of a higher one.

    Parameters
    ----------
    h_int : fockit.algebra.OpSum
        Interaction Hamiltonian of the physical orbitals. Its modes are
        ``(block, index)``, e.g. ``nc("up", 0) * nc("dn", 0)``.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('dn', 3) ]``.
    bond_dim : int, optional
        Bond dimension of the matrix product state.
    symmetries : mapping, optional
        Conserved quantities besides the particle number, ``"sz"`` (total
        spin projection) and ``"s"`` (total spin), and the sector of each. A
        number, e.g. ``{"sz": 0}``, is the sector of the ground state. With
        ``None``, e.g. ``{"s": None}``, every sector is solved and the lowest
        is kept. With ``"s"``, ``"sz"`` picks the state of the spin multiplet,
        and without it the density matrices average over the multiplet. The
        default ``{}`` conserves only the particle number. A spin-rotation
        invariant impurity problem has its ground state in ``{"sz": 0}`` (or
        ``1/2``), and is fastest with ``{"s": 0}`` when its ground state is a
        singlet.
    spin_blocks : sequence of tuple, optional
        The names of the spin-up and spin-down block of each pair of blocks
        in ``gf_struct`` with the same orbitals. Default is
        ``[("up", "dn")]``.
    mix_schedule : sequence of tuple, optional
        The symmetry-mixing sweeps that start each solve, as ``(strength,
        sweeps, noise)`` of each stage: random hoppings of ``strength`` times
        the largest one-body coupling, for ``sweeps`` sweeps at the full bond
        dimension with ``noise``. Default is ``((0.1, 6, 1e-5), (0.01, 6,
        1e-5), (0.001, 6, 1e-5), (1e-4, 6, 1e-5))``, and ``()`` turns them
        off.
    bath_size : mapping, optional
        Number of bath orbitals of each block, ``B`` times its size for ghost
        RISB with ``B`` quasiparticle orbitals per orbital, e.g.,
        ``{"up": 3, "dn": 3}`` for ``B = 3`` and one orbital per spin. Blocks
        that are not given have as many bath orbitals as physical orbitals
        (``B = 1``).
    comm : risb.helpers_mpi.Comm, optional
        MPI communicator, e.g. ``mpi4py.MPI.COMM_WORLD``, whose ranks solve
        the sectors in ``symmetries`` in parallel. Every rank must call
        :meth:`solve` and :meth:`overlap`. Default is a single process.

    Raises
    ------
    ValueError
        If ``spin_blocks`` does not pair every block of ``gf_struct`` with one
        of the same size, if ``symmetries`` has an unknown key or a sector
        with the wrong half-integer, or if ``h_int`` does not conserve them.

    """

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        bond_dim: int = 500,
        symmetries: Mapping[str, float | None] | None = None,
        spin_blocks: Sequence[tuple[str, str]] = (("up", "dn"),),
        mix_schedule: Sequence[tuple[float, int, float]] = _MIX_SCHEDULE,
        *,
        bath_size: Mapping[str, int] | None = None,
        comm: Comm | None = None,
    ) -> None:
        super().__init__(
            gf_struct, bond_dim, spin_blocks, bath_size, comm, mix_schedule
        )

        #: dict : Conserved quantities and their sectors.
        self.symmetries = dict(symmetries or {})

        L = len(self._sites)
        unknown = set(self.symmetries) - {"s", "sz"}
        if unknown:
            msg = f"Unknown symmetries {sorted(unknown)}."
            raise ValueError(msg)
        twice = {
            key: None if v is None else round(2 * v)
            for key, v in self.symmetries.items()
        }
        if any(v is not None and (v - L) % 2 for v in twice.values()):
            msg = f"The sectors of {L} electrons have {'half-' if L % 2 else ''}integer spin."
            raise ValueError(msg)
        twos, twosz = twice.get("s"), twice.get("sz")
        if "s" in twice:
            S_values = [twos] if twos is not None else list(range(L % 2, L + 1, 2))
            Sz_values = [twosz]
            self._sectors: list[_Sector] = [
                (s, sz)
                for s, sz in product(S_values, Sz_values)
                if sz is None or abs(sz) <= s
            ]
        elif "sz" in twice:
            Sz_values = [twosz] if twosz is not None else list(range(-L, L + 1, 2))
            self._sectors = [(None, sz) for sz in Sz_values]
        else:
            self._sectors = [(None, None)]
        if not self._sectors:
            msg = "No sector has the spin projection in symmetries."
            raise ValueError(msg)

        self._previous: _Sector | None = None
        # The directory with the matrix product state of the ground state, and
        # its sector
        self._state: tuple[tempfile.TemporaryDirectory[str], _Sector] | None = None
        self.set_h_int(h_int)

        #: float : Lowest energy of the impurity problem, set by :meth:`solve`.
        self.gs_energy: float

        #: float : Largest weight discarded in the last sweep, set by
        #: :meth:`solve`.
        self.discarded_weight: float

        #: float : Energy variance ``<H^2> - <H>^2`` of the ground state, set
        #: by :meth:`solve`.
        self.variance: float

        #: dict : Sector of the ground state, with ``"s"`` and ``"sz"`` where
        #: they are conserved, set by :meth:`solve`.
        self.sector: dict[str, float] = {}

        #: float : Entropy of the ground state, zero.
        self.entropy = 0.0

    @override
    def solve(self) -> None:
        """
        Solve for the ground state in the half-filled particle sector of the impurity problem.

        Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block.

        Raises
        ------
        ValueError
            If the impurity Hamiltonian is not Hermitian, or if ``symmetries``
            has ``"s"`` and its spin-up and spin-down terms differ.

        """
        self._check_h_imp()
        L = len(self._sites)
        runs = {
            i: self._run(self._sectors[i], L, 1)
            for i in round_robin(self.comm, len(self._sectors))
        }
        gathered = gather_tasks(self.comm, {i: run[0][0] for i, run in runs.items()})
        energies = {self._sectors[i]: e for i, e in gathered.items()}
        lowest = min(energies, key=energies.__getitem__)
        if (
            self._previous in energies
            and energies[self._previous] < energies[lowest] + _SECTOR_TOL
        ):
            lowest = self._previous
        self._previous = lowest
        if len(energies) > 1:
            logger.info(
                "Lowest energies by sector: %s",
                ", ".join(
                    f"{s}: {e:.10f}"
                    for s, e in sorted(energies.items(), key=lambda x: x[1])
                ),
            )
        if self._state is not None:
            self._state[0].cleanup()
        self._state = None
        # The rank of the lowest sector has its state
        G, discarded_weight, variance = np.zeros((2 * L, 2 * L)), 0.0, 0.0
        for i, run in runs.items():
            if self._sectors[i] == lowest:
                _, (G,), _, discarded_weight, scratch, (variance,) = run
                self._state = (scratch, lowest)
            else:
                run[4].cleanup()
        self.gs_energy = energies[lowest]
        self.discarded_weight = self.comm.allreduce(discarded_weight)
        self.variance = self.comm.allreduce(variance)
        G = self.comm.allreduce(G)
        twos, twosz = lowest
        self.sector = {}
        if twos is not None:
            self.sector["s"] = twos / 2
        if twosz is not None:
            self.sector["sz"] = twosz / 2
        self._set_rho(G)

    def overlap(self, Op: OpSum, force_real: bool = True) -> float | complex:
        """
        Calculate the expectation value of an operator in the ground state of the impurity problem.

        With ``"s"`` in :attr:`symmetries` and no ``"sz"``, it is the average
        over the states of the spin multiplet, as the density matrices are.

        Parameters
        ----------
        Op : fockit.algebra.OpSum
            Operator to take expectation of, with at most two-body terms. The
            bath modes are ``("bath_" + block, index)``.
        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        float or complex
            Expectation value.

        """
        const, h1, V = Op_to_tensors(Op, self._index)
        res = complex(self._expectation(np.real(const), h1.real, V.real))
        if np.iscomplexobj(h1):
            res += 1j * self._expectation(np.imag(const), h1.imag, V.imag)
        if force_real:
            return res.real
        return res

    def _expectation(
        self, const: float, h1: NDArray[np.float64], V: NDArray[np.float64]
    ) -> float:
        # <const + sum h1 c^dag c + sum V c^dag c^dag c c> in the ground state,
        # averaged over the spin multiplet with a conserved total spin and no
        # spin projection, from the rank with the state
        total = 0.0
        if self._state is not None:
            scratch, sector = self._state
            total = self._state_expectation(
                scratch.name, "KET", sector, len(self._sites), const, h1, V
            )
        return float(self.comm.allreduce(total))


class Block2ThermalImpuritySolver(_Block2Base):
    """
    Grand-canonical impurity solver at finite temperature using DMRG from block2.

    The density matrices are thermal averages over every particle number of
    the impurity problem, with the chemical potential of :meth:`set_h_imp`, as
    :class:`risb.ThermalLatticeSolver` needs. It is for a collinear
    (spin-diagonal) impurity problem with real terms, and needs the optional
    dependency `block2 <https://block2.readthedocs.io>`__.

    With ``method="purification"``, the thermal state is a matrix product
    state of the orbitals and an ancilla next to each, evolved in imaginary
    time from infinite temperature, as in A. E. Feiguin and S. R. White, Phys.
    Rev. B 72, 220401(R) (2005)
    (`arXiv:cond-mat/0510124 <https://arxiv.org/abs/cond-mat/0510124>`__),
    with time-step targeting fourth-order Runge-Kutta steps.

    With ``method="lowest"``, the thermal average is over the lowest states of
    each sector within ``delta`` particles of half filling, found together by
    state-averaged DMRG and then each refined alone, and keeps the states with
    ``beta * (E - E0) <= eta`` above the lowest energy ``E0``, as in L.
    Giordano, Y. Tan, Z.-H. Cui, and C. Sun,
    `arXiv:2601.01641 <https://arxiv.org/abs/2601.01641>`__. It is for low
    temperatures, where few states have a weight.

    Parameters
    ----------
    h_int : fockit.algebra.OpSum
        Interaction Hamiltonian of the physical orbitals. Its modes are
        ``(block, index)``, e.g. ``nc("up", 0) * nc("dn", 0)``.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('dn', 3) ]``.
    beta : float
        Inverse temperature of the impurity problem.
    bond_dim : int, optional
        Bond dimension of the matrix product states.
    time_step : float, optional
        Largest step in imaginary time of ``"purification"``. The state
        evolves to ``beta / 2`` in equal steps.
    symmetries : sequence of str, optional
        Conserved quantity besides the particle number, ``"sz"``
        (:math:`S_z`) or ``"s"`` (:math:`S`), or empty for the particle
        number only. Every sector is in the average. Default is ``["sz"]``.
    spin_blocks : sequence of tuple, optional
        The names of the spin-up and spin-down block of each pair of blocks
        in ``gf_struct`` with the same orbitals. Default is
        ``[("up", "dn")]``.
    method : str, optional
        ``"purification"`` (the default) or ``"lowest"``.
    eta : float, optional
        Largest ``beta * (E - E0)`` of a state in ``"lowest"``. Default is 8.
    delta : int, optional
        Largest difference of the particle number from half filling of a
        sector in ``"lowest"``. Default is 2.
    mix_schedule : sequence of tuple, optional
        The symmetry-mixing sweeps that start each solve of ``"lowest"``, as
        in :class:`Block2ImpuritySolver`.
    bath_size : mapping, optional
        Number of bath orbitals of each block, ``B`` times its size for ghost
        RISB with ``B`` quasiparticle orbitals per orbital, e.g.,
        ``{"up": 3, "dn": 3}`` for ``B = 3`` and one orbital per spin. Blocks
        that are not given have as many bath orbitals as physical orbitals
        (``B = 1``).
    comm : risb.helpers_mpi.Comm, optional
        MPI communicator, e.g. ``mpi4py.MPI.COMM_WORLD``, whose ranks solve
        the sectors of ``"lowest"`` in parallel. ``"purification"`` runs on
        rank 0. Every rank must call :meth:`solve` and :meth:`overlap`.
        Default is a single process.

    Raises
    ------
    ValueError
        If ``spin_blocks`` does not pair every block of ``gf_struct`` with one
        of the same size, if ``symmetries`` is not empty, ``"sz"`` or ``"s"``,
        if ``method`` is unknown, or if ``h_int`` does not conserve the
        symmetries.

    """

    def __init__(
        self,
        h_int: OpSum,
        gf_struct: GfStruct,
        beta: float,
        bond_dim: int = 500,
        time_step: float = 0.1,
        symmetries: Sequence[str] = ("sz",),
        spin_blocks: Sequence[tuple[str, str]] = (("up", "dn"),),
        method: str = "purification",
        eta: float = 8.0,
        delta: int = 2,
        mix_schedule: Sequence[tuple[float, int, float]] = _MIX_SCHEDULE,
        *,
        bath_size: Mapping[str, int] | None = None,
        comm: Comm | None = None,
    ) -> None:
        super().__init__(
            gf_struct, bond_dim, spin_blocks, bath_size, comm, mix_schedule
        )

        #: float : Inverse temperature of the impurity problem.
        self.beta = beta

        #: float : Largest step in imaginary time.
        self.time_step = time_step

        #: list : Conserved quantities besides the particle number.
        self.symmetries = list(symmetries)

        if self.symmetries not in ([], ["sz"], ["s"]):
            msg = 'symmetries must be [], ["sz"] or ["s"].'
            raise ValueError(msg)

        #: str : How the thermal state is found.
        self.method = method

        #: float : Largest ``beta * (E - E0)`` of a state in ``"lowest"``.
        self.eta = eta

        #: int : Largest difference of the particle number from half filling
        #: of a sector in ``"lowest"``.
        self.delta = delta

        if method not in ("purification", "lowest"):
            msg = 'method must be "purification" or "lowest".'
            raise ValueError(msg)

        # The directories of the thermal state, and its pure states with their
        # weights: the purified state, without a sector, or the lowest states
        self._dirs: list[tempfile.TemporaryDirectory[str]] = []
        self._states: list[_ThermalState] = []
        self.set_h_int(h_int)

        #: float : Thermal average of the impurity Hamiltonian, set by
        #: :meth:`solve`.
        self.energy: float

        #: float : ``-log(Z) / beta`` with ``Z`` the partition function, set by
        #: :meth:`solve`.
        self.grand_potential: float

        #: float : Entropy ``beta * (energy - grand_potential)`` of the impurity
        #: problem, set by :meth:`solve`.
        self.entropy: float

    @override
    def solve(self) -> None:
        """
        Solve for the thermal state of the impurity problem.

        Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block, :attr:`energy`, :attr:`grand_potential`,
        and :attr:`entropy`.

        Warns
        -----
        RuntimeWarning
            With ``"lowest"``, if the sectors ``delta`` particles from half
            filling have states with ``beta * (E - E0) <= eta``, or if the
            states found with ``beta * (E - E0) > eta`` have a weight above
            ``1e-4``.

        Raises
        ------
        ValueError
            If the impurity Hamiltonian is not Hermitian, or if ``symmetries``
            has ``"s"`` and its spin-up and spin-down terms differ.

        """
        self._check_h_imp()
        for d in self._dirs:
            d.cleanup()
        solve = self._lowest if self.method == "lowest" else self._purify
        G, self.energy, log_Z, self._dirs, self._states = solve()
        self.grand_potential = -log_Z / self.beta
        self.entropy = self.beta * (self.energy - self.grand_potential)
        self._set_rho(G)

    def overlap(self, Op: OpSum, force_real: bool = True) -> float | complex:
        """
        Calculate the thermal average of an operator in the impurity problem.

        Parameters
        ----------
        Op : fockit.algebra.OpSum
            Operator to take expectation of, with at most two-body terms. The
            bath modes are ``("bath_" + block, index)``.
        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        float or complex
            Expectation value.

        """
        const, h1, V = Op_to_tensors(Op, self._index)
        res = complex(self._expectation(np.real(const), h1.real, V.real))
        if np.iscomplexobj(h1):
            res += 1j * self._expectation(np.imag(const), h1.imag, V.imag)
        if force_real:
            return res.real
        return res

    def _expectation(
        self, const: float, h1: NDArray[np.float64], V: NDArray[np.float64]
    ) -> float:
        # <const + sum h1 c^dag c + sum V c^dag c^dag c c> in the thermal state,
        # summed over the states of every rank
        total = 0.0
        for weight, scratch, tag, sector, n_elec in self._states:
            if sector is None:
                total += weight * self._purified_expectation(scratch, tag, const, h1, V)
            else:
                total += weight * self._state_expectation(
                    scratch, tag, sector, n_elec, const, h1, V
                )
        return float(self.comm.allreduce(total))

    def _purify(self) -> _ThermalResult:
        # The thermal state by purification, on rank 0
        L = len(self._sites)
        G, energy, log_Z = np.zeros((2 * L, 2 * L)), 0.0, 0.0
        dirs: list[tempfile.TemporaryDirectory[str]] = []
        states: list[_ThermalState] = []
        if self.comm.rank == 0:
            G, energy, log_Z, dirs, states = self._purified_state()
        return (
            self.comm.allreduce(G),
            self.comm.allreduce(energy),
            self.comm.allreduce(log_Z),
            dirs,
            states,
        )

    def _purified_state(self) -> _ThermalResult:
        # The thermal state by purification
        L = len(self._sites)
        scratch = tempfile.TemporaryDirectory()
        symm_type = SymmetryTypes.SU2 if "s" in self.symmetries else SymmetryTypes.SZ
        if self.symmetries:
            driver = DMRGDriver(scratch=scratch.name, symm_type=symm_type)
            driver.initialize_system(n_sites=L, n_elec=L, spin=0)
            mpo = self._mpo(
                driver, symm_type, self._const, self._h1, self._V, ancilla=True
            )
            ket, t0, log_norm = driver.get_ancilla_mps("KET0"), 0.0, 0.0
        else:
            # block2's infinite-temperature state has one particle number at
            # each bond between two pairs of a spin orbital and its ancilla,
            # and td_dmrg keeps no state without weight, so particles never
            # move between pairs that no two neighbouring sites connect. A
            # first step exp(-t0 H) by a Taylor series of MPO applications
            # gives weight to every particle number
            n = 2 * L
            driver = DMRGDriver(scratch=scratch.name, symm_type=SymmetryTypes.SGF)
            driver.initialize_system(n_sites=n, n_elec=n, spin=0)
            driver.get_ancilla_mps("KET0", target=driver.bw.SX(n, 0, 0))
            driver.initialize_system(n_sites=2 * n, n_elec=n, spin=0)
            ket0 = ket = driver.load_mps("KET0")
            identity = driver.get_identity_mpo()
            t0 = min(_SEED_TIME, self.beta / 2)
            for k in range(_SEED_ORDER, 0, -1):
                # ket0 - t0 / k H ket
                s = -t0 / k
                bra = driver.get_random_mps(f"SEED{k}", bond_dim=self.bond_dim)
                driver.addition(
                    bra,
                    ket0,
                    ket,
                    mpo_a=identity,
                    mpo_b=self._doubled_mpo(
                        driver, s * self._const, s * self._h1, s * self._V
                    ),
                    bra_bond_dims=[self.bond_dim],
                    n_sweeps=20,
                    tol=1e-12,
                    iprint=0,
                )
                ket = bra
            log_norm = float(np.log(driver.expectation(ket, identity, ket)))
            mpo = self._doubled_mpo(driver, self._const, self._h1, self._V)
        # exp(-beta H / 2) on the infinite-temperature state, normalized
        # after each step
        ket = driver.td_dmrg(
            mpo,
            ket,
            target_t=self.beta / 2 - t0,
            n_steps=max(1, math.ceil((self.beta / 2 - t0) / self.time_step)),
            bond_dims=[self.bond_dim],
            te_type="rk4",
            n_sub_sweeps=_SUB_SWEEPS,
            normalize_mps=True,
            hermitian=True,
            final_mps_tag="KET",
            iprint=0,
        )
        # Z is 4^L times the product of the squared norms before each
        # normalization
        log_Z = (
            L * np.log(4)
            + log_norm
            + float(np.sum(np.log(np.real(driver._te.normsqs))))
        )
        energy = float(driver.expectation(ket, mpo, ket))

        # The thermal state as a state of the orbitals and the ancillas, with
        # at most a conserved spin projection, and its 1-RDM of the orbitals
        if self.symmetries:
            driver.initialize_system(n_sites=2 * L, n_elec=2 * L, spin=0)
            if symm_type == SymmetryTypes.SU2:
                driver.align_mps_center(ket, ref=0)
                ket = driver.mps_change_to_sz(ket, "ZKET", sz=0)
                driver.symm_type = SymmetryTypes.SZ
                driver.initialize_system(n_sites=2 * L, n_elec=2 * L, spin=0)
        ket.save_data()
        ket.info.save_data(f"{scratch.name}/{ket.info.tag}-mps_info.bin")
        if self.symmetries:
            dm_up, dm_dn = driver.get_1pdm(ket)
            G = np.zeros((2 * L, 2 * L))
            G[0::2, 0::2], G[1::2, 1::2] = dm_up[0::2, 0::2], dm_dn[0::2, 0::2]
        else:
            G = np.asarray(driver.get_1pdm(ket))[0::2, 0::2]
        return G, energy, log_Z, [scratch], [(1.0, scratch.name, ket.info.tag, None, L)]

    def _purified_expectation(
        self,
        scratch: str,
        tag: str,
        const: float,
        h1: NDArray[np.float64],
        V: NDArray[np.float64],
    ) -> float:
        # <const + sum h1 c^dag c + sum V c^dag c^dag c c> in the purified
        # state tag
        L = len(self._sites)
        if self.symmetries:
            driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SZ)
            driver.initialize_system(n_sites=L, n_elec=L, spin=0)
            mpo = self._mpo(driver, SymmetryTypes.SZ, const, h1, V, ancilla=True)
        else:
            driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SGF)
            driver.initialize_system(n_sites=4 * L, n_elec=2 * L, spin=0)
            mpo = self._doubled_mpo(driver, const, h1, V)
        ket = driver.load_mps(tag)
        return float(driver.expectation(ket, mpo, ket))

    def _lowest(self) -> _ThermalResult:
        # The thermal average over the lowest states of the sectors within
        # delta particles of half filling, with beta (E - E0) <= eta. Each
        # sector solves for its lowest state, and for twice as many states
        # until the highest is above the window or every state of the sector
        # is found. The ranks split the sectors
        L = len(self._sites)
        window = self.eta / self.beta
        tasks = [
            (n_elec, sector)
            for n_elec in range(max(0, L - self.delta), min(2 * L, L + self.delta) + 1)
            for sector in self._spin_sectors(min(n_elec, 2 * L - n_elec))
        ]
        runs = {
            i: self._run(tasks[i][1], tasks[i][0], 1)
            for i in round_robin(self.comm, len(tasks))
        }
        E0 = min(
            gather_tasks(self.comm, {i: run[0][0] for i, run in runs.items()}).values()
        )
        for i in runs:
            n_elec, sector = tasks[i]
            size = _sector_size(L, n_elec, sector)
            while max(runs[i][0]) - E0 <= window and len(runs[i][0]) < size:
                nroots = min(2 * len(runs[i][0]), size)
                runs[i][4].cleanup()
                runs[i] = self._run(sector, n_elec, nroots)
            logger.debug(
                "Sector %d, %s: energies above the lowest %s",
                n_elec,
                sector,
                np.round(np.array(runs[i][0]) - E0, 6),
            )
        energies = gather_tasks(self.comm, {i: run[0] for i, run in runs.items()})

        weights: dict[tuple[int, int], float] = {}
        n_states, excluded = 0, 0.0
        for i, es in energies.items():
            # Every state of a spin multiplet has the same weight
            twos = tasks[i][1][0]
            multiplicity = 1 if twos is None else twos + 1
            for k, e in enumerate(es):
                weight = multiplicity * np.exp(-self.beta * (e - E0))
                if e - E0 <= window:
                    weights[i, k] = weight
                    n_states += multiplicity
                else:
                    excluded += weight
        Z = float(sum(weights.values()))
        energy = float(sum(w * energies[i][k] for (i, k), w in weights.items())) / Z

        G = np.zeros((2 * L, 2 * L))
        dirs: list[tempfile.TemporaryDirectory[str]] = []
        states: list[_ThermalState] = []
        for i, (_, run_Gs, tags, _, scratch, _) in runs.items():
            n_elec, sector = tasks[i]
            kept = [(k, w / Z) for (j, k), w in weights.items() if j == i]
            if not kept:
                scratch.cleanup()
                continue
            dirs.append(scratch)
            for k, p in kept:
                G += p * run_Gs[k]
                states.append((float(p), scratch.name, tags[k], sector, n_elec))
        G = self.comm.allreduce(G)
        # The largest discarded weight and energy variance of the states in
        # the average
        in_average = {i: [k for j, k in weights if j == i] for i in runs}
        largest = gather_tasks(
            self.comm,
            {
                i: (runs[i][3], max(runs[i][5][k] for k in ks))
                for i, ks in in_average.items()
                if ks
            },
        )
        if self.comm.rank == 0:
            self._report(
                tasks,
                energies,
                E0,
                n_states,
                excluded / Z,
                max(d for d, _ in largest.values()),
                max(v for _, v in largest.values()),
            )
        return G, energy, -self.beta * E0 + np.log(Z), dirs, states

    def _report(
        self,
        tasks: Sequence[tuple[int, _Sector]],
        energies: Mapping[int, Sequence[float]],
        E0: float,
        n_states: int,
        excluded: float,
        discarded: float,
        variance: float,
    ) -> None:
        # Log the lowest states, and warn if delta or eta is too small
        L = len(self._sites)
        lowest: dict[int, float] = {}
        for i, es in energies.items():
            n_elec = tasks[i][0]
            lowest[n_elec] = min(lowest.get(n_elec, np.inf), self.beta * (es[0] - E0))
        n0, sector0 = tasks[min(energies, key=lambda i: energies[i][0])]
        logger.info("Lowest energy %.10f, sector %d, %s", E0, n0, sector0)
        logger.info(
            "Lowest beta (E - E0) by particle number: %s",
            ", ".join(f"{n}: {b:.2f}" for n, b in sorted(lowest.items())),
        )
        logger.info(
            "%d states in the thermal average. The found states outside it have "
            "a weight of %.1e",
            n_states,
            excluded,
        )
        logger.info(
            "Largest discarded weight of the sectors in it %.1e, and largest "
            "energy variance of its states %.1e",
            discarded,
            variance,
        )
        edges = [n for n in (L - self.delta, L + self.delta) if 0 < n < 2 * L]
        inside = [n for n in edges if lowest[n] <= self.eta]
        if inside:
            msg = (
                f"Sectors with {inside} particles, delta from half filling, have "
                f"states with beta (E - E0) <= eta. Increase delta."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=4)
        if excluded > _EXCLUDED_WEIGHT_TOL:
            msg = (
                f"The found states with beta (E - E0) > eta have a weight of "
                f"{excluded:.1e}, which changes the density matrices by up to "
                f"twice as much. Increase eta."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=4)

    def _spin_sectors(self, top: int) -> list[_Sector]:
        # The spin sectors of a particle number with top unpaired electrons at
        # most
        if "s" in self.symmetries:
            return [(twos, None) for twos in range(top % 2, top + 1, 2)]
        if "sz" in self.symmetries:
            return [(None, twosz) for twosz in range(-top, top + 1, 2)]
        return [(None, None)]

    def _doubled_mpo(
        self,
        driver: DMRGDriver,
        const: float,
        h1: NDArray[np.float64],
        V: NDArray[np.float64],
    ) -> _MPO:
        # The MPO of the particle-number chain of each spin orbital and its
        # ancilla next to it, with h1 and V on the spin orbitals
        n = len(h1)
        p = np.arange(0, 2 * n, 2)
        h1_doubled = np.zeros((2 * n, 2 * n))
        h1_doubled[np.ix_(p, p)] = h1
        V_doubled = np.zeros((2 * n,) * 4)
        V_doubled[np.ix_(p, p, p, p)] = V
        return self._mpo(driver, SymmetryTypes.SGF, const, h1_doubled, V_doubled)


def _bath(bl: str) -> str:
    return "bath_" + bl


def _real(A: NDArray[np.inexact]) -> NDArray[np.float64]:
    if np.abs(A.imag).max(initial=0) > 1e-12:
        msg = "Block2ImpuritySolver needs a real impurity Hamiltonian."
        raise ValueError(msg)
    return np.asarray(A.real, dtype=np.float64)


def _sector_size(L: int, n_elec: int, sector: _Sector) -> int:
    # Number of states (spin multiplets with a conserved total spin) of
    # n_elec electrons in L spatial orbitals in sector
    twos, twosz = sector
    if twos is not None:
        # Weyl-Paldus formula
        return (
            (twos + 1)
            * math.comb(L + 1, (n_elec - twos) // 2)
            * math.comb(L + 1, (n_elec + twos) // 2 + 1)
            // (L + 1)
        )
    if twosz is not None:
        return math.comb(L, (n_elec + twosz) // 2) * math.comb(L, (n_elec - twosz) // 2)
    return math.comb(2 * L, n_elec)

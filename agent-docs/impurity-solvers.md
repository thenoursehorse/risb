# Impurity solvers for H_emb (literature survey, 2026-10-25)

What H_emb is: finite; interacting physical orbitals plus a non-interacting
ghost bath (Lambda_c f f^dag + D c^dag f + h.c.). The RISB loop only needs static 1-RDMs
(T=0: ground state; finite T: grand-canonical averages over all N, plus Omega
and S for the free energy after convergence). Tolerances of 1e-6..1e-8, so
stochastic noise sets the reachable RISB tol.

| Family | T=0 | Finite T | d/f with B=3? | Fit to our problem |
|---|---|---|---|---|
| ED / Lanczos (fockit) | yes | full ED; lowest states (Capone et al.); FTLM/LTLM; TPQ typicality | no (d B=1; f B=1 borderline) | have it |
| Truncated / selected CI: RASCI (Zgid, Gull, Chan), ATHS (Go, Millis), ASCI (Mejuto-Zaera et al.), SCI/QSCI | yes | lowest states or Lanczos in the truncated space | yes if natural occupations near 0/1 | gGA ground states get sparser in the CI basis as B grows (arXiv 2506.21431) |
| DMRG (block2) | have | purification (have): breaks down at low T | T=0 yes | - |
| Purification + disentangler (Hauschild et al. 2018) | - | deterministic | open | biggest gain at low T, our failure mode; own code |
| XTRG / tanTRG (density-matrix MPO, beta doubles each step) | - | deterministic | open | 2D Hubbard done; not in block2 |
| METTS | - | stochastic | likely (each sample ~ M_GS) | Cao, Stoudenmire, Parcollet 2024: multi-orbital impurity, low T, SOC |
| TPQ-MPS (Iwaki, Shimizu, Hotta) | - | stochastic | open | random-sample average like METTS |
| Lowest states ("LT-DMRG", Giordano et al. 2026) | - | low T only | while few states | = our route A with a looser cutoff |
| Tree TN (FTPS, Bauernfeind et al.) | yes | via METTS (Cao et al.) | yes | geometry = backbone of physical spin orbitals + ghost branches; ITensor, not block2 |
| CT-HYB (TRIQS/cthyb, w2dynamics) | - | stochastic, cost ~ beta | yes | discrete bath fine (below); sign problem with full Slater / off-diagonal D; TRIQS dep |
| NQS (Zhouyin, Lee, Chen, Lanata, Guo 2026, gGA) | yes | - | yes | stochastic; bottleneck = sampling accuracy |
| CC / AFQMC | yes | finite-T versions exist | yes | weak-correlation / phaseless bias; poor for Mott |
| NRG | - | - | - | needs a continuous bath; no |

## Notes per method
- LT-DMRG (arXiv 2601.01641 App. A): sectors (N_up, N_dn) within +-delta of the
  ground state's, kept if beta Dc(k) - mu k <= eta1, Dc(k) = E0(N0 + k) - E0(N0);
  in each, states with beta (E_i - E0_sector) <= eta2, found one at a time with
  the previous ones projected out (memory) or by block Davidson (more stable);
  then the FT-FCI sums over the kept states. eta1 = eta2 = 8 in their runs,
  "5-10" recommended. Canonical (delta = 0) is poor even at low T. eta = 8 drops
  weights of order e^-8 per state, so rho errors ~1e-3-1e-4 (2 orb B=1 beta=10:
  weight beyond eta=8 is 7e-4). Route A was the same with a e^-32 cutoff.
- METTS + FTPS (Cao et al.): collapse in the occupation basis; the N_p bath
  sites with the smallest |eps| are purified instead (ancilla pairs), so the
  rest can be collapsed; first steps by global basis extension with
  (1 - dtau H)^k |psi> truncated to m' ~ 50, then TDVP; 120-1080 samples; m
  80-240; statistical error ~ sigma/sqrt(N_S). block2 has
  sample_csf_coefficients (collapse) and get_mps_from_csf_coefficients
  (product-state MPS).
- CT-HYB on a discrete bath: integrating out any quadratic bath gives
  Delta(iw) = D (iw - eps_f)^-1 D^dag (sum of B n poles; the hole convention
  Lambda_c f f^dag is -Lambda_c^T f^dag f + const). From G_cc: G_fc = g D^dag G_cc,
  G_ff = g + g D^dag G_cc D g with g = (iw - eps_f)^-1; equal-time sums give
  rho_cf, rho_f. Noise of G_cc enters all three.
- Binder, Barthel 2015: purification beats METTS at equal cost except far below
  the gap.
- Ghost GA at finite T (Giuli et al. arXiv 2603.20559): finite-T gGA = DMFT as
  B -> inf; needs only static thermal averages; lists FT-ED/Lanczos, QMC,
  thermal TN, TPQ as solvers. GEM code (Giuli et al., TRIQS-based) has zero and
  finite T.
- block2-mpi: near-ideal scaling 448 -> 2800 cores for ground-state DMRG (Zhai,
  Chan 2021); sum-of-MPO parallelism non-spin-adapted only.
- DIIS/Broyden with noisy solvers: standard in DFT+DMFT with CT-QMC (Zitko
  2009); converges to the noise floor.

## Sources
- Giuli et al., finite-T ghost GA: https://arxiv.org/html/2603.20559v1
- GEM: https://www.researchgate.net/publication/414926092_GEM_An_implementation_of_the_ghost-Gutzwiller_approximation_for_simulating_interacting_quantum_systems
- Giordano, Tan, Cui, Sun, FT-DMET (LT-DMRG, pFT-DMRG): https://arxiv.org/html/2601.01641v1
- Zhouyin et al., NQS for gGA, PRB 113, 155123 (2026): https://journals.aps.org/prb/abstract/10.1103/rkd8-q6yl
- Quantum-assisted ghost GA (SCI/QSCI): https://arxiv.org/html/2506.21431v1
- Binder, Barthel, PRB 92, 125119 (2015): https://arxiv.org/abs/1411.3033
- Hauschild et al., PRB 98, 235163 (2018): https://arxiv.org/abs/1711.01288
- Chen et al., XTRG, PRX 8, 031082 (2018): https://arxiv.org/pdf/1801.00142.pdf
- XTRG for fermions, PRB 103, L041107 (2021): https://repository.hku.hk/bitstream/10722/296304/1/content.pdf
- Cao, Stoudenmire, Parcollet, METTS impurity solver, PRB 109, 245113 (2024): https://arxiv.org/abs/2312.13668
- Bauernfeind et al., FTPS, PRX 7, 031013 (2017): https://ar5iv.arxiv.org/html/1612.05587
- Iwaki, Shimizu, Hotta, TPQ-MPS, PRR 3, L022015 (2021): https://ar5iv.labs.arxiv.org/html/2005.06829
- Capone, de' Medici, Georges, low-T Lanczos: https://arxiv.org/abs/cond-mat/0512484
- Zgid, Gull, Chan, PRB 86, 165128 (2012): https://ar5iv.arxiv.org/html/1203.1914
- Go, Millis, PRB 96, 085139 (2017): https://ar5iv.org/html/1703.04928
- Mejuto-Zaera, Tubman, Whaley, ASCI, PRB 100, 125165 (2019): https://arxiv.org/pdf/1711.04771
- TRIQS/CTHYB: https://arxiv.org/pdf/1507.00175
- Zitko, Broyden for DMFT, PRB 80, 125125 (2009): https://ar5iv.arxiv.org/html/0908.0613
- Zhai, Chan, block2 parallel DMRG, JCP (2021): https://authors.library.caltech.edu/108531

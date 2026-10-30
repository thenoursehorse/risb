# Preconditioning the self-consistency

How DFT codes precondition their SCF mixing, and what the same idea would look
like for `DIIS` in RISB.

## Theory

Anderson/Pulay convergence is set by the spectrum of `P(1 − J)`, where
`J = ∂g/∂x` and `P` is the preconditioner.

- In DFT, `1 − J` is the dielectric operator `ε = 1 − v_c χ₀`. Near a stable
  ground state it is positive with real eigenvalues, but badly conditioned: in
  metals `ε(G) ≈ 1 + q_TF²/G²`, which blows up at small G (charge sloshing).
- The preconditioner `P` is a cheap model of `ε⁻¹`, so that `P(1 − J) ≈ 1`.

## What the codes do

- Kerker: `δρ(G) = A · G²/(G² + q₀²) · r(G)`, the inverse of the
  Thomas-Fermi dielectric function.
- VASP:
  - A Pulay mixer that starts from Kerker as its initial guess for the inverse
    Jacobian (`AMIX` ≈ 0.4, `BMIX` ≈ 1.0).
  - Charge and magnetization get separate parameters (`AMIX_MAG` ≈ 1.6).
  - The DIIS inner product is weighted toward long wavelengths.
  - It prints eigenvalues of the preconditioned dielectric matrix, estimated
    from the Pulay history, to tune `AMIX`.
- QE:
  - Modified Broyden with `mixing_beta` 0.7 and `mixing_ndim` 8.
  - `mixing_mode` can be `TF` (Kerker-like) or `local-TF` for inhomogeneous
    systems.
  - The residual norm uses a Coulomb metric.
  - In DFT+U, the occupations `ns` are mixed separately.
- DFTK (Herbst & Levitt): LDOS-based preconditioning, plus adaptive damping
  that uses an energy model as a line search.

The SCF in DFT gets hard exactly where `1 − J` stops being positive: when XC
terms add negative parts, e.g., magnetism, symmetry breaking, DFT+U. RISB near
a Mott transition or with competing solutions is in that regime. The
positive-spectrum assumption that makes DFT easy is probably the real gap.

## For RISB

`f2` is a density mismatch and Λ is the field that couples to density. That is
the same pairing as residual density and potential in DFT, so the natural
preconditioner is an inverse susceptibility:

- Simplest: separate step sizes for Λ and R, like VASP's charge vs
  magnetization.
- Kerker analogue: `δΛ ≈ −χ_qp⁻¹ f2`, where `χ_qp = ∂ρ_qp/∂Λ` is the local
  quasiparticle susceptibility. It comes from the lattice side only
  (Lindhard-like, from the bands and k-weights), so it is cheap and has no
  impurity noise. It ignores how the impurity responds, just as Kerker ignores
  detailed χ₀. R and `f1` would be treated the same way.
- Diagnostic first: estimate the eigenvalues of J from the DIIS history
  (`J ≈ ΔG ΔX⁺`), as VASP does. If some are complex, close to 1 or above 1,
  that is what makes it hard.

## The `'root'` error mode

`x + βf` does not make sense dimensionally. Λ is an energy and `f2` is a
density, so β would have to be an inverse susceptibility. That is why a
preconditioner is physically required there. Right now the map
`f → g(x) − x` built into RISB's own update formula is acting as P without
anyone choosing it. It was never designed to give `P(1 − J) ≈ 1`.

What to try first: Anderson with an explicit P,
`x_{n+1} = Σ cᵢ xᵢ + P Σ cᵢ fᵢ`, with P set to separate β_Λ and β_R or the
χ_qp⁻¹ preconditioner. That gives a knob like `mixing_beta`/`AMIX` per
variable type, and it is the main thing DFT has that we do not.

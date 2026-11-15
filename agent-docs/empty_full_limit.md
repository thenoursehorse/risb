# D and Λc of nearly full or empty quasiparticle orbitals

The lattice half of the zero-temperature cycle builds the impurity's hybridization D and bath
levels Λc from the quasiparticle bands (PRX 5, 011008, Eqs. 35–36). As a quasiparticle orbital
fills or empties, both formulas divide by √(n(1 − n)). This note shows that D and Λc have finite
limits, writes Λc in a form that does not lose digits near the edge, and records the checks.

## Notation

For one block of one cluster, with the quasiparticle Hamiltonian h_k = R ε_k R† + Λ and occupations
Γ_k = g(h_k), where g is any function (a step at T = 0, a smearing otherwise):

| symbol | definition | in risb |
|---|---|---|
| ρ | Σ_k w_k Γ_k | `rho_qp` is ρᵀ |
| σ | Σ_k w_k (1 − Γ_k) = 1 − ρ | not computed |
| ke | Σ_k w_k ε_k R† Γ_k | `lopsided_ke_qp` |
| T | R ke = Σ_k w_k R ε_k R† Γ_k | |
| H | Σ_k w_k h_k Γ_k | not computed |
| H^Q | Σ_k w_k h_k (1 − Γ_k) | not computed |

With projectors, every sum is projected onto the quasiparticles of the cluster.

The kinetic term has no local part, Σ_k w_k ε_k = 0, so Σ_k w_k h_k = Λ. This gives two identities:

    H + H^Q = Λ,        T = H − Λρ.

Neither needs T = 0: Γ_k only has to be a function of h_k.

Eigenbasis of ρ: occupations n_a, holes s_a = 1 − n_a, and f_a = √(n_a s_a).

## The PRX formulas

    D = S⁻¹ keᵀ,  S = √(ρ(1 − ρ)),
    Λc = −∂/∂ρ (Tr[D Rᵀ S(ρ)] + c.c.) − Λ.

The derivative of S(ρ) (Daleckii–Krein) has, in the eigenbasis of ρ, the divided differences

    Δ_ab = (f_a − f_b) / (n_a − n_b) = (1 − n_a − n_b) / (f_a + f_b),    Δ_aa = (1 − 2n_a) / (2 f_a).

With X = D Rᵀ = S⁻¹ Tᵀ, i.e. X_ab = T_ba / f_a, the gradient term is G_ab = Δ_ab (X_ab + X_ba), and
Λc = −G − Λ. This is what `helpers.get_lambda_c` computes. Below, the matrices are real symmetric;
complex Hermitian matrices are checked numerically.

## The closed form

Insert T_ab = H_ab − Λ_ab n_b:

    G_ab = Δ_ab [ H_ab (1/f_a + 1/f_b) − Λ_ab (n_a/f_b + n_b/f_a) ].

The first term gives (1 − n_a − n_b) H_ab / (f_a f_b), since Δ_ab (f_a + f_b) = 1 − n_a − n_b. In
the Λ terms of −G − Λ, the numerator over (f_a + f_b) f_a f_b is

    (1 − n_a − n_b)(n_a f_b + n_b f_a) − (f_a + f_b) f_a f_b
      = f_b [(1 − n_a − n_b) n_a − f_a²] + f_a [(1 − n_a − n_b) n_b − f_b²]
      = −n_a n_b (f_a + f_b),

so

    Λc_ab = [ (n_a + n_b − 1) H_ab − n_a n_b Λ_ab ] / (f_a f_b)          (particle form)
          = [ (s_a + s_b − 1) H^Q_ab − s_a s_b Λ_ab ] / (f_a f_b)        (hole form)

The hole form follows from the particle form with H = Λ − H^Q and n = 1 − s. Without a basis,

    Λc = S⁻¹ (ρH + Hρ − H − ρΛρ) S⁻¹ = S⁻¹ (σH^Q + H^Qσ − H^Q − σΛσ) S⁻¹,

and risb's `Lambda_c` is the transpose of this. Against `get_lambda_c`, with complex random R, Λ and
ε_k and B = 3, the difference is 3e-13 (`scratch/edge_conv.py`).

D has no cancellation, but needs accurate n and s: in the eigenbasis of ρ, row a of D is column a
of ke divided by f_a.

## Why the limits are finite

From 0 ≤ Γ_k ≤ 1, Σ_{b≠a} |(Γ_k)_ab|² ≤ (Γ_k)_aa (1 − (Γ_k)_aa). So the correlations of an
orbital that fills (or empties) with all the others vanish at least as √s (√n). Hence:

- **D and R:** the column a of ke vanishes as √s_a, so row a of D has a finite limit. The same
  holds for the R row ρ_cf/√s.
- **The quasiparticle level:** Λ_aa diverges, as it must for a full (empty) orbital.
- **The bath level:** in the particle form for a filling orbital, H_aa and Λ_aa are both of the
  size of the diverging level, and they cancel. In the hole form, H^Q_aa = Σ_k Σ_m (1 − g(e_km))
  e_km |V_am|² is a sum over empty states only. It is of order s_a, with finite energies e_km, so it
  is computed without cancellation. The term s_a² Λ_aa / (f_a²) → s_a Λ_aa → 0.

Limits as orbitals fill (s → 0):

| pair | limit of Λc_ab |
|---|---|
| a filling (diagonal) | −H^Q_aa / s_a: minus the mean energy of the empty states, weighted by the weight of a in them |
| a, b both filling | −H^Q_ab / √(s_a s_b) |
| a filling, b emptying | 0 |
| a filling, b regular | finite; the PRX form is accurate |

For emptying orbitals the same holds with the particle form, H for H^Q, and n for s.

## Which form to use

Both forms are exact, so for each pair of eigenvectors of ρ:

- use the hole form when n_a + n_b > 1, the particle form otherwise;
- there is no threshold, and the switch has no jump: the two differ only in round-off.

Three numerical requirements:

1. **n and s:** take them from the weight of each eigenvector in the full and empty states,
   n_a = u_a† ρ u_a and s_a = u_a† σ u_a. Do not use the eigenvalues of ρ: they are accurate only to
   ~1e-16 in absolute terms, which is noise for a nearly empty orbital.
2. **σ and H^Q:** build them from 1 − g computed directly (for Fermi smearing, g(−x)), not from
   1 − ρ and Λ − H. These subtractions lose exactly the digits the hole form needs.
3. **The floor:** the eigenvector components are accurate to ~1e-16, so weights below ~1e-14 are
   not resolved. Such an orbital is decoupled to machine precision, and only the side its bath level
   is on (full or empty bath) carries information.

## The current code loses digits well above 1e-12

`get_lambda_c` computes Λc = −G − Λ: two numbers that grow as 1/√s are subtracted.

- With one orbital filling, the result is lost by s ≈ 1e-9.
- With two filling together, it is lost by s ≈ 1e-6.

Chain checks (`scratch/edge_limit*.py`), B = 2 and 3, R and Λ set by hand:

| case | s | code | closed form |
|---|---|---|---|
| one orbital filling, B = 2 | 1e-8 | −0.6166 | −0.6192 |
| | 1e-10 | +1.96 | −0.6189 |
| | 1e-12 | +2475 | −0.604 |
| two filling, B = 3, second orbital | ~1e-6 | +1.40 | −0.636 |
| | ~3e-9 | +1.9e3 | −0.633 |

Below 1e-12, `EDGE_TOL` sets D's row and the gradient term to zero, so the bath level becomes −Λ_aa,
far from the limit. The bath half of `FullEmptyPinning` then puts it at ±margin × the energy scale.

## The impurity half

**Numerics** (`scratch/imp_limit.py`): one orbital per spin, U = 2, h0_loc = −U/2, B = 3, exact
ground state, densities averaged over spin. D = (0.5, 0.3, 0.2)ᵀ, Λc = diag(0.2, −0.3, −E) + 0.1
off-diagonal. Pushing the bath level −E down fills auxiliary orbital 3:

| E | s_f | R row | gradient (aa) | Λ_aa | Λ_ab, regular b |
|---|---|---|---|---|---|
| 1 | 3.3e-4 | 0.330 | 1.59 | −0.577 | −0.172 |
| 64 | 1.4e-6 | 0.386 | 65.2 | −1.233 | −0.191 |
| 1024 | 5.7e-9 | 0.3865 | 1025 | −1.251 | −0.1915 |
| 4096 | 3.6e-10 | 0.3866 | 4097 | −1.252 | −0.1915 |

This is the mirror image of the lattice half. The gradient term grows as E and cancels −Λcᵀ_aa, so
the quasiparticle level has a finite limit, and so does the R row.

| side | its input level | its output level |
|---|---|---|
| lattice | Λ_aa → −∞ fills the orbital | Λc_aa finite |
| impurity | Λc_aa → −∞ fills the orbital | Λ_aa finite |

So at a solution, s cannot go to zero through a level going to infinity:

- **On the lattice,** s_qp → 0 needs Λ_aa → −∞ (or the R row → 0).
- **On the impurity,** s_f → 0 needs Λc_aa → −∞ (or the D row → 0).
- **But** each side returns a finite level when its input level diverges.

This suggests that an exactly full (empty) orbital at a solution is a decoupled ghost, with its R
and D rows zero. Its levels are then undetermined (0/0), which is the case the pinning actually
handles. The suite cases that need the pinning all look like that: the saturated ferromagnet, the
Néel state, the non-interacting B = 3 run (ghosts with R = 0), the Kanamori Mott insulator (R = 0),
and the decoupled ghosts of a metal on ligands. Not proven.

With two orbitals filling (Λc = diag(0.2, −E, −2E)), the deeper one has s_f exactly 0 in floating
point from E = 1024, so the helpers treat it as an edge (Λ_aa = −Λc_aa). Not understood yet.

**Closed form on the impurity side** (`scratch/imp_closed.py`). With the bath operators f and the
ground state of energy E0, [H_emb, f_b†] = Σ_α D_bα c_α† − Σ_d Λc_bd f_d† (H_loc does not act on the
bath). Define

    P_ab = ⟨f_a (H − E0) f_b†⟩    addition energies, weighted by ρ_f
    Q_ab = ⟨f_b† (E0 − H) f_a⟩    removal energies, weighted by the bath electrons σ_f = 1 − ρ_f

Then P + Q = ⟨{f_a, [H, f_b†]}⟩ = −Λcᵀ, the impurity counterpart of Σ_k h_k = Λ. With
Y = D ρ_cf (so that D Rᵀ = Y S_f⁻¹), Yᵀ = −P − Λc ρ_f, the counterpart of T = H − Λρ. The algebra of
the lattice half then gives, in the eigenbasis of ρ_f,

    Λ_ab = [ (n_a + n_b − 1)(−P)_ab − n_a n_b (Λcᵀ)_ab ] / (f_a f_b)
         = [ (s_a + s_b − 1)(−Q)_ab − s_a s_b (Λcᵀ)_ab ] / (f_a f_b).

Checked against `get_lambda` to 3e-13 where the latter is accurate. One orbital filling:

| E | s_f | `get_lambda` | closed form |
|---|---|---|---|
| 1e3 | 5.7e-9 | −1.250934 | −1.250926 |
| 1e5 | 6.0e-13 | +1e5 (edge treatment) | −1.252105 |
| 1e7 | 6.0e-17 | +1e7 | −1.252117 |

**Two orbitals filling** (Λc = diag(0.2, −E, −2E)): one combination of the two deep bath orbitals has
no direct hybridization with c. Its electron weight falls as about E⁻⁶ (4.5e-8 at E = 4, 1.4e-15 at
E = 64, 8e-23 at E = 1000), which is why it looked exactly empty. For it, the closed form gives a
level that follows the bath level (−66 at E = 64, −1008 at E = 1000), on the correct side (full on
both sides). `get_lambda` treats it as an edge and gives +1.9E, the wrong side, which the
quasiparticle half of the pinning repairs. Below s ≈ 1e-20 the weight is round-off.

**The PRX formulas with accurate n and s** (`scratch/prx_accurate_s.py`): keep the formulas, but take
n and s from the weights in ρ and in σ (computed directly), not from the eigenvalues of ρ.

- **Impurity side:** enough down to s ≈ 1e-15 (−1.252106 at 6e-13, −1.2734 at 6e-17).
- **Lattice side,** two orbitals filling: not enough (−0.632 against −0.636 at s = 1.6e-12, −2212 at
  2.9e-17).

On the impurity side, the accuracy of the solver's s_f (DMRG) is far worse than 1e-15, so there
accurate s may suffice without the energy matrices P and Q, which a DMRG solver would have to
compute specially. On the lattice side, the closed form is needed.

**Earlier text (wrong):**

Λ = −G(ρ_f, R, D) − Λcᵀ. With Λc finite, Λ_aa would grow as 1/√s_f. But on the impurity side it is Λc
that makes the orbital fill, and then the two cancel (above). R = (ρ_cf S_f⁻¹)ᵀ has a finite row. The
limitation is the input: at a solution s_f = s_qp, so the impurity solver must give s_f to the
accuracy wanted for Λ_aa. In NiO, s ≈ 1e-9 for the full t2g orbital, and block2 reproduces it only
to a relative 2e-3 (bond dimension 500).

Test from the converged NiO solution, 10 cycles of linear mixing (`scratch/clip/nio10`): at bond
dimension 500, 1 − n_f of the full t2g orbital jitters by 1–3 % between cycles, and the next Λ of that
orbital jumps by 0.05–0.07 with it. At bond dimension 1500 both are smooth. The error is the same in
both (8e-6 to 9e-6), so the impurity noise does not set the NiO error floor. In both, that orbital
keeps filling (1 − n_qp from 1.297e-9 to 1.29e-9) and its level keeps sinking (4.575 to 4.53, μ =
7.63): the iteration drifts towards the full, decoupled ghost, along the t2g Λ that the residual
does not see.

`EDGE_TOL` still zeroes the impurity half below 1e-12, and the quasiparticle half of
`FullEmptyPinning` still repairs that.

## Prototype (`scratch/edge_plugin.py`)

| mode | lattice-solver tests (71) | full suite (285) |
|---|---|---|
| closed form, bath pin only below 1e-14, quasiparticle pin and step limit kept | 71 pass | 285 pass |
| closed form, no pinning at all | 5 fail: saturated ferromagnet, Néel (`LatticeSolver`), Kanamori Mott (B = 1), decoupled ghosts of a metal on ligands, the pinning's own test | — |
| closed form, quasiparticle pin and step limit, no bath pin at all | 5 fail: ferromagnet, non-interacting B = 3, Néel (`PhysicalMuLatticeSolver`), Kanamori Mott, the pinning's own test | — |
| closed form, bath pin only below 1e-14, no quasiparticle pin or step limit | 4 fail: ferromagnet, Néel (`LatticeSolver`), decoupled ghosts of a metal on ligands, the pinning's own test | — |

So both halves are still needed: the bath half only where the weights are below the 1e-14 floor, the
quasiparticle half for the impurity side.

NiO with the closed form is unchanged: fresh starts converge as before (error 8e-6,
|g(x) − x| = 0.152 in the t2g Λ), and the cycle-19 start still fails. No NiO orbital is below 1e-14.

## Open

- [ ] The impurity half: a form of Λ and R that does not divide by s_f, e.g. from ground-state
  identities of the embedding Hamiltonian. Noise in s_f matters at bond dimension 500 (NiO), not at 1500.
- [ ] NiO: the full t2g ghost drifts towards the edge. Is the solution there (decoupled ghost)?
- [ ] The thermal bath fit: ρ_qp near 0 or 1 barely depends on the level, so the fit may have a flat
  direction there. Not checked.
- [ ] AutoBZIntegrator does not evaluate the observables per k in Python; the closed form falls back
  there.

## Decoupled orbitals

Two ways an auxiliary orbital decouples (its R and D rows go to zero):

- **At half filling:** a flat band at μ, n ≈ 1/2. Nothing divides by zero, but the gradient term
  vanishes, so Λ_aa = −Λc_aa and Λc_aa = −Λ_aa: the map is the identity along the level, a neutral
  direction. `LinearWeightedDensityFeedback` adds −c M f2ᵀ M, with M = (1 − RR†)^{1/2}, a linear
  restoring force that vanishes at a solution. The occupation fixes the level because
  n = f(level − μ) can be inverted at half filling. No limit is needed.
- **Full or empty:** f saturates, so every deep enough level gives n = 1. f2 gives no restoring
  force, and the level is free. Along the line through the decoupled U = 0 ghost
  (`scratch/smoothness.py`, MODEL=nonint), the pin fixes the level (|dg|/|dt| ≈ 2e-10); without the
  pin, the next level follows the input 1:1 (|dg|/|dt| → √2, two spin blocks).

**The jump** (`scratch/smoothness.py`, LINE=decouple): Hubbard, U = 2.4, B = 3, at the solution. The
lowest quasiparticle level is pushed down by 5, and its R row is scaled by λ from 1 to 1e-10.

| λ | lattice s | current code: Λc_vv / next Λ_vv / impurity s_f | closed forms, pin placement below 1e-14 |
|---|---|---|---|
| 1.5e-4 | 1.9e-11 | −1.039 / 1.033 / 6.1e-3 | −1.039 / 1.033 / 6.1e-3 |
| 1.5e-5 | 1.9e-13 | −33.74 / −34.72 / 4e-14 | −1.039 / 1.033 / 6.1e-3 |
| 1.5e-6 | 1.9e-15 | −33.74 / −34.72 / 1e-16 | −33.74 / −8.01 / 1e-16 |

The jump is where the treatment switches from the formula to the far placement, not in the physics:
along the line the levels have finite limits. The closed forms move the jump from 1e-12 to the 1e-14
floor. Here the impurity side stays regular (s_f = 6e-3), because the lattice gives finite D and Λc
in the limit.

**First-order limit** (`scratch/decouple_limit.py`). Let v be the orbital, G the others, R_v = λ r,
and the coupling of v to G λ c_k with c_k = R_G ε_k rᵀ (plus Λ_Gv / λ if Λ couples them). With
|m⟩, e_km the eigenstates of the G block and a_km = ⟨m|c_k⟩ / (e_km − Λ_vv), first order in λ gives:

    σ_vv = λ² S₂,  S₂ = Σ_k w_k Σ_m (1 − f_km) a_km²,       H^Q_vv = λ² H₂ (same, with e_km)
    σ_Gv = λ σ₁,   σ₁ = Σ_k w_k Σ_m (1 − f_km) a_km |m⟩,    H^Q_Gv = λ h₁ (same, with e_km)
    σ_GG, H^Q_GG   at λ = 0

The filling orbital is the eigenvector of ρ, u = e_v + λ w with w = −σ_GG⁻¹ σ₁. This rotation
matters at leading order, because σ_Gv is of order λ while σ_vv is of order λ². Along u:

    s_u   = λ² [S₂ − σ₁ᵀ σ_GG⁻¹ σ₁]
    H^Q_uu = λ² [H₂ − 2 σ₁ᵀ σ_GG⁻¹ h₁ + σ₁ᵀ σ_GG⁻¹ H^Q_GG σ_GG⁻¹ σ₁]
    Λc_uu → −H^Q_uu / s_u

Check: −1.03928 against −1.03929 from the closed form at λ → 0. Without the rotation, −H₂/S₂ =
−0.97857, which is wrong. Everything is computed from the G block and the direction r, with no tiny
numbers, so the limit holds at λ = 0 itself, given r. The D row and the off-diagonal elements still
have to be done the same way.

What remains a choice: at an R row that is exactly zero there is no direction r. Either the direction
is carried over from the last cycle that had one, or, for an orbital that is decoupled from the start
(the full-shell model), something like the far placement has to be chosen.

Impurity side: in this line it stays regular. When the D row goes to zero, the bath orbital's weight is
second order in D, and the first-order limit would need the embedding's excited states (a resolvent,
⟨f (E0 − H)⁻¹ c†⟩-type terms) rather than ground-state expectation values. That is more expensive for
DMRG. Not done.

**Where the closed form loses precision** (`scratch/decouple_floor.py`): along the decoupling line, it
holds to six digits down to s ≈ 1e-13 (λ = 1e-5) and drifts at 1e-15 (λ = 1e-6). Below that, s is
noise at the 1e-17 level: the weights come from the band eigenvectors in the basis of the block, and
lose precision there.

**First-order band sums** (`scratch/edge2_plugin.py`, mode pert). For the orbitals F with
min(n, s) < 1e-10, the band sums are computed in the basis (F, G), from first-order states: the
G-block eigenstates with F components Φ a_m, and the F-block eigenstates with G components −M a_j. The
same closed form is then evaluated on them. Along the decoupling line:

| λ | closed form only | closed form + first order |
|---|---|---|
| 1e-5 | −1.039402 | −1.039283 |
| 1e-6 | −33.74 (pin placement) | −1.039283 |
| 1e-10 | −33.74 | −1.039283 |
| 1e-12 | −33.74 | −1.040219 |
| 0 | −33.74 | −0.176 (noise) |

The D row (−0.29378) and the off-diagonal elements are just as flat. At a relative coupling of 1e-12,
the round-off of forming the (F, G) blocks (~1e-16 × ‖h‖) starts to show; at zero, the coupling is
round-off and its direction is arbitrary. So an orbital whose coupling is below 1e-13 × the largest
element of h is decoupled, and gets zero D rows and the pinning's placement.

On the 71 lattice-solver tests (impurity side as in side_step):

| test | result |
|---|---|
| bilayer Krylov root | 2e-10 off against a tolerance of 1e-10; benign |
| `[not pinned]` | finds 3 electrons, not the trap's 2; correct |
| Kanamori Mott | passes (failed in side_step) |
| saturated ferromagnet | stalls at 0.17 |
| non-interacting B = 3 | stalls at 1.4e-4 |

**Why the last two stall: the decoupled manifold.** At a solution with a decoupled orbital, its level
is free, and so are its couplings: the off-diagonal Λ and Λc elements to the other orbitals, and its R
and D rows near zero. The iteration lets them wander (1e-8 to 2e-5 in the non-interacting test), above
round-off. The first-order limit follows the direction of the wandering coupling and puts the
orbital's bath level inside the band (about ±0.48), where it mixes with the active bath. The pinning
avoids this by collapsing all of these free directions: far levels make the couplings irrelevant.

So smoothness alone does not settle the decoupled case. The solutions form a manifold of equivalent
points, and a choice on the free components is needed: a gauge fix. It should be continuous, so the
map stays smooth, which is the purpose of 3(a) (remove them) or 3(b) (a smooth restoring force on the
free components).

**The pin switched on continuously** (`scratch/edge2_plugin.py`, mode smooth). Any restoring force that
vanishes at every solution cannot pick a point on the decoupled family, since every point there is a
solution. So the choice has a scale: a range of s over which it acts, and in which it biases the
levels. Here the weight is w = smoothstep in log10 s, w = 0 at s = 1e-10 and w = 1 at 1e-12 (the
existing pin threshold). For each orbital with w > 0:

- **Lattice side:** after the closed form plus first order, its D row and its off-diagonal bath levels
  are scaled by 1 − w, and its bath level moves a fraction w towards the pin's far value.
- **Impurity side:** the same with the R row, the off-diagonal Λ and the quasiparticle level, using
  s_f from a directly computed σ_f.
- **Step limit:** kept.
- **Exactly zero weight:** zero rows.

At w = 1 this is the current pin. Along the decoupling line the bath level now moves continuously
(−1.04, −1.75, −6.3, −13.5, −21.6, −28.8, −33.2 as s goes from 1e-10 to 1e-12), but steeply:
|dg| ≈ 43 per decade at the centre of the switch, against a jump of 300 between neighbouring points
for the current pin. A second steep step follows when moving the bath away empties the impurity's
bath orbital (s_f from 7e-7 to 2e-13 between neighbouring points), and the impurity-side switch
passes quickly.

71 lattice-solver tests: only the Krylov tolerance and `[not pinned]` (which avoids the trap) fail.
The ferromagnet, the non-interacting B = 3 run, both Néel tests and the Kanamori Mott insulator pass.

**Suites and NiO in smooth mode** (`scratch/edge/suite`, cycles counted per test):

| | current code | smooth |
|---|---|---|
| full suite | 285 pass | fails only the Krylov tolerance and `[not pinned]` (avoids the trap) |
| full suite, cycles of the 127 tests passing in both | 4992 | 4981 |
| swapped suite | 9 fail | the same 9 fail |
| swapped suite, cycles of the 120 tests passing in both | 3941 | 3994 |

Largest changes, smooth against the current code:

- **Fewer cycles:** decoupled ghosts of a metal on ligands (−50 in both suites), Néel with
  `LatticeSolver` in the full suite (−14), the zero-temperature parts of the low-temperature tests in
  the swapped suite (−18, −15), double-counting restarts (−7, −14).
- **More cycles:** Néel with `PhysicalMuLatticeSolver` (+56), Néel with `LatticeSolver` under the swap
  (+57), Kanamori Mott under the swap (+30).
- **The block2 thermal tests** vary by about ±5 cycles between runs of the same code: the thermal
  Kanamori lattice test took 28 cycles in the suite and 54 alone, with the current code.

NiO: fresh from DFT, linear mixing, 50 iterations: 8.0e-6 (current code 7.9e-6). From CSC cycle 19:
plateau at 0.306 (current code 0.31).

**Why the Néel test (`PhysicalMuLatticeSolver`) takes 56 more cycles in smooth mode**
(`scratch/edge/neel_*.txt`): 21 cycles with the current code, 77 in smooth mode.

| cycles | smooth mode |
|---|---|
| 4–10 | orbitals enter and leave the window (s from 1e-13 to 1e-11), as the pinned ones do in the current code |
| 12–28 | the residual bounces between 0.3 and 0.75 while one orbital sits partly in the window (w ≈ 0.27, s ≈ 2e-11) |
| 29–77 | no orbital in the window; the residual falls slowly from 1e-3 to 8e-10, with the smallest impurity weight settling at s_f = 1.9e-8 |

The two modes converge to different solutions:

| | current code: pinned ghost | smooth: coupled ghost |
|---|---|---|
| self-consistency | 7.7e-11 | 4.9e-10 |
| μ (inside the gap) | 1.293 | 0.978 |
| m | ±0.966874 | ±0.966762 |
| smallest eigenvalue of Z | 0.98545 | 0.99982 |
| energy (band + local) | −0.124899 | −0.124926 |

The smooth solution is lower by 2.7e-5. It is the case the `FullEmptyPinning` docstring describes: the
pin holds the ghost exactly decoupled at a higher energy, while the lower solution has it slightly
coupled (s_f = 1.9e-8) as an extra channel. The extra cycles are the iteration converging along that
weakly determined ghost, not the window.

## The impurity half of a decoupling orbital

`scratch/imp_decouple.py`: one orbital per spin, U = 2, B = 3, exact ground state by a numpy Jordan-Wigner
diagonalization (8 modes, half filling). Bath orbital v at level −5; its D row and its bath couplings
scaled by λ. The closed form (Q_uu / s_u along the eigenvector u of ρ_f):

| λ | s_u | Λ_uu |
|---|---|---|
| 1e-1 | 1.2e-6 | −1.024980 |
| 1e-3 | 1.2e-10 | −1.024868 |
| 1e-5 | 1.2e-14 | −1.024868 |
| 1e-7 | 1.2e-18 | −1.024868 |

**First-order limit.** H = H0 + λW is linear in λ, so the ground state is Φ = Φ0 + λΦ1 with

    Φ1 = (E0 − H0)⁻¹ W Φ0, projected off Φ0.

With σ_ab = ⟨f_a† f_b⟩ and Q_ab = ⟨f_b† (E0 − H) f_a⟩, the leading orders are σ_vv = λ² s₂, σ_Gv = λ s₁,
Q_vv = λ² q₂ and Q_Gv = λ q₁, with σ_GG and Q_GG at λ = 0. Along u = e_v + λw, w = −σ_GG⁻¹ s₁:

    s_u  = λ² [s₂ − s₁ᵀ σ_GG⁻¹ s₁]
    Q_uu = λ² [q₂ − 2 s₁ᵀ σ_GG⁻¹ q₁ + s₁ᵀ σ_GG⁻¹ Q_GG σ_GG⁻¹ s₁]
    Λ_uu → Q_uu / s_u = −1.024868

This agrees with the closed form at λ → 0. Without the rotation, q₂/s₂ = −0.598, which is wrong. The
R row and the off-diagonal elements follow from Φ1 the same way (not done).

**Cost and accuracy.**
- Exact diagonalization (fockit): Q and σ come from operators applied to the exact state, so the
  closed form already holds down to s ≈ 1e-18 here. The first-order form is only needed at λ = 0.
- DMRG (block2): the parts of the ground state with weight below the truncation error are lost, so s_f
  below roughly the discarded weight is not resolved. This is likely the 1–3 % noise of the t2g
  s_f ≈ 1e-9 in NiO at bond dimension 500. The first-order form computes Φ1 as a separate state, one
  correction-vector solve per decoupling orbital, which keeps the small part from being truncated
  against Φ0. The cost is about one extra DMRG solve per such orbital.

**\(\sum_i\operatorname{Tr}f_{2,i}=0\) expresses the required charge condition under the existing quasiparticle filling constraint. It is not, by itself, an algorithm for enforcing it.**

Let
\[
\mathcal T(A)=\sum_{k,\sigma}w_k\operatorname{Tr}A_{k\sigma},
\qquad
\epsilon=\mathcal T(\Delta N).
\]
Currently, \(\epsilon=\sum_i\operatorname{Tr}f_{2,i}\), including equivalent-site multiplicities. This follows from the [gRISB density and constraint equations](https://arxiv.org/html/2406.04636v2#S2).

There are four useful approaches. These are proposed changes; none is implemented here yet.

| Approach | Makes calculated RISB density neutral? | Additional impurity solves? |
|---|---|---|
| 1. Subtract charge error from exported density | Exported density only | No |
| 2. Bounded projection of exported density | Exported density only | No |
| 3. Solve a charge component of `Lambda` | Yes, retaining quasiparticle filling | Usually several |
| 4. Search \(\mu\) using physical filling | Yes, allowing quasiparticle filling to drift temporarily | Usually several |

1. **Subtract the charge error using a chosen distribution**

   Construct
   \[
   \Delta N_{\rm neutral}=\Delta N-\epsilon W,
   \qquad \mathcal T(W)=1.
   \]
   This guarantees \(\mathcal T(\Delta N_{\rm neutral})=0\).

   **In practice:** calculate the density normally, measure its charge error, distribute the compensating charge through \(W\), and export the adjusted density. Keep the original impurity matrices and residuals.

   Choices for \(W\) include uniform weight over the band window, weight restricted to correlated orbitals, or weight near the Fermi level.

   **Pros:** cheapest implementation; no additional impurity calculations; the adjustment vanishes whenever the original charge error vanishes.

   **Cons:** neutrality does not determine where the compensating charge belongs. Different choices change intermediate charge transfer. Occupations can become negative or exceed 1. Restricting compensation to nearly full ligand bands can exhaust their available capacity.

   This leaves the one-shot RISB iteration unchanged. It affects CSC through the next DFT potential.

2. **Project onto a density with fixed charge and valid occupations**

   Diagonalize the calculated physical density:
   \[
   \rho_{k\sigma}=U_{k\sigma}\operatorname{diag}(n_{k\sigma j})U_{k\sigma}^\dagger.
   \]
   Find a single scalar \(\eta\) such that
   \[
   \widetilde n_{k\sigma j}=\operatorname{clip}(n_{k\sigma j}+\eta,0,1)
   \]
   sums to the required electron count. Reconstruct \(\widetilde\rho\) and export \(\widetilde\rho-\rho_{\rm DFT}\).

   **In practice:** diagonalizations plus a cheap scalar root search. This is the nearest fixed-charge density satisfying occupation bounds under a weighted Frobenius-norm criterion.

   **Pros:** guarantees neutrality, Hermiticity, and occupations within \([0,1]\). No additional impurity solves. It avoids choosing an arbitrary subset of bands to receive charge.

   **Cons:** it still changes the density independently of the impurity solution. The distance metric is a numerical choice. Clipping can remain active even when the total charge is correct, so we must check that it leaves converged RISB densities unchanged.

   Like option 1, it changes the exported density and subsequent CSC trajectory, without directly correcting RISB’s internal residuals.

3. **Solve the global charge component of `Lambda` each iteration**

   Retain the existing quasiparticle filling condition. Split a common scalar component from the correlated quasiparticle potentials:
   \[
   \Lambda_i=\Lambda_i^\perp+\lambda_{\rm charge}I_i.
   \]
   For trial \(\lambda_{\rm charge}\):

   - Recalculate quasiparticles and their chemical potential.
   - Rebuild the impurity baths and solve the impurities.
   - Evaluate \(S=\sum_i\operatorname{Tr}f_{2,i}\).
   - Adjust \(\lambda_{\rm charge}\) until \(S=0\).

   The other components of `Lambda` and `R` continue through ordinary RISB iterations. Only the **global** charge condition is enforced; individual impurity occupations remain free.

   **Pros:** the calculated density itself has the correct charge. The density, impurity solution, and residuals come from the same calculation. Full RISB convergence remains unnecessary before each DFT update.

   **Cons:** each root evaluation generally requires impurity solves. Level crossings or disconnected orbitals can make the charge response discontinuous or flat. The optimizer must handle the separately solved scalar consistently.

   **I tested this on the small B=1 and B=3 examples:** charge became correct while other residuals remained around \(0.05\). The first scalar search took roughly 11–12 evaluations. Predicting the scalar from the previous iteration could reduce subsequent cost, but that needs testing.

4. **Choose chemical potential from the final physical filling**

   Replace the quasiparticle number search with
   \[
   N_{\rm physical}(\mu)=N_{\rm target}.
   \]
   Every trial \(\mu\) must recalculate quasiparticle occupations, baths, impurity solutions, and physical density.

   **Pros:** directly enforces the requested physical count. When all `f2` constraints converge, the standard quasiparticle filling condition is recovered.

   **Cons:** it also requires repeated impurity solves. It can achieve physical neutrality by driving quasiparticle filling far from its eventual target, increasing other constraint errors. I observed that behavior in the toy example. A usable continuous root is not guaranteed at every intermediate configuration.

   Adjusting only the final quasiparticle occupations while freezing impurity densities is insufficient generally: the remaining uncorrelated orbitals may have too little capacity to absorb the discrepancy.

**For your short-loop objective, I would compare option 2 as inexpensive CSC stabilization against option 3 as a solver-level solution.** Option 3 is the stronger physical approach, but its extra impurity work could outweigh the benefit of short loops.

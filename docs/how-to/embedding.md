---
title: Solve embedding
parent: How-to guides
---

# Table of Contents
{: .no_toc .text-delta }

- TOC
{:toc}

It is kept this way because this is the most computationally expensive part. The simplest thing 
to do is blindly use exact diagonalization. There is also a specialized 
exact diagonalization implementation for the specific structure of the 
embedding Hamiltonian, and an outdated solver using DMRG. There are so 
many avenues to go down for this: 
[Pomerol](https://aeantipov.github.io/pomerol/), 
[QuTiP](https://qutip.org/)/[QuSpin](https://weinbe58.github.io/QuSpin/), 
QMC, NISQ devices, etc.

## Make an embedding class

## Use `EmbeddingEd`

This embedding solver is specialized to only solve for the embedding state
that RISB requires. A big difference is that it needs the local Hamiltonian 
to initialize because it automatically finds and enforces many symmetries.

```python
from embedding_ed import *

emb_solver = EmbeddingEd(h_loc, gf_struct)
```

Compared to `AtomDiag` the only difference in using it is the local 
Hamiltonian is not passed to the method that constructs the embedding 
Hamiltonian.

```python
emb_solver.set_h_emb(Lambda_c, D)
emb_solver.solve()
```

There are some additional parameters that one can use in solve to do with 
the Lanczos algorithm.

```python
psiS_size = emb_solver.get_psiS_size()
emb_solver.solve(ncv = min(30,psiS_size), max_iter = 1000, tolerance = 0)
```

Here `ncv` is to do with how many vectors it keeps in its iterative solver, 
and the rest should be self explanatory. You usually want to keep as many 
vectors as possible, but for large problems it is not possible. I find that 
30 often works well, but this is all problem dependent.
# risb
Rotationally invariant slave-bosons.

### Todo

- Add kweights tests, DIIS tests, multiple cluster tests, complex SOC tests, 
all helpers functions (make random inputs and store, because current tests
have too much structure)
- Fix sensitive to initial R, Lambda guess, make more robust
- Get static type hints working for mypy
- Add verbose output to `LatticeSolver`
- Add verbose output to `DIIS`
- Maybe? Refactor `DIIS` and `NewtonSolver`
- Explanation for why the root finders kind of suck.
- Remove h_emb to real if real in `EmbeddingAtom` when TRIQS bumps from 
version 3.2.0
- Sort out intersphinx linking in docs
- Helper plot functions
- Helper functions for calculating free/total energy
- add linting actions
- pipy packaging

### In Progress

- symmetry representation tutorial
- Kagome/projectors tutorial
- single-orbital Hubbard tutorial 
- Finish TBmodels in tight-binding how-to
- Sort out cross-referencing to API in docs
- action for hatchling version bump

### Done ✓
- Setup github actions
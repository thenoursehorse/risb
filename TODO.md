# risb

Rotationally invariant slave-bosons.

### Todo

- Add kweights tests, DIIS tests, multiple cluster tests, complex SOC tests,
  all helpers functions (make random inputs and store, because current tests
  have too much structure)
- Add helpers_triqs tests
- Fix sensitive to initial R, Lambda guess, make more robust
- Implement initial guess as R, pdensity, as this will likely be more robust.
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
- dimer insulator to explain some different insulating states
- change references in docs to use sphinxcontrib-bibtex
- pytest filterwarnings are not working. They need to be for risb package only and not promote warnings to errors in third-party packages.

### In Progress

- symmetry representation tutorial
- Kagome/projectors tutorial
- Finish TBmodels in tight-binding how-to
- Sort out cross-referencing to API in docs

### Done ✓

- Setup github actions
- single-orbital Hubbard tutorial
- action for hatchling version bump (done with tag update action)
- pipy packaging
- Green's function helper functions

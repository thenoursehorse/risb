# risb

Rotationally invariant slave-bosons.

### Todo

- Add multiple cluster tests, complex SOC tests,
  all helpers functions (make random inputs and store, because current tests
  have too much structure)
- Fix sensitive to initial R, Lambda guess, make more robust
- Implement initial guess as R, pdensity, as this will likely be more robust.
- Add verbose output to `LatticeSolver`
- Add verbose output to `DIIS`
- Maybe? Refactor `DIIS` and `NewtonSolver`
- Explanation for why the root finders kind of suck.
- Helper plot functions
- Helper functions for calculating free/total energy
- dimer insulator to explain some different insulating states
- change references in docs to use sphinxcontrib-bibtex
- pytest filterwarnings are not working. They need to be for risb package only and not promote warnings to errors in third-party packages.

### In Progress

- symmetry representation tutorial
- Kagome/projectors tutorial
- Finish TBmodels in tight-binding how-to

### Done ✓

- Static type hints (strict mypy, checked in CI)
- Linting in CI (pre-commit)
- Intersphinx linking in docs
- Cross-referencing to API in docs
- kweights and DIIS tests
- Setup github actions
- single-orbital Hubbard tutorial
- action for hatchling version bump (done with tag update action)
- pipy packaging
- Green's function helper functions

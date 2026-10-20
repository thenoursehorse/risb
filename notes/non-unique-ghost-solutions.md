# Non-unique ghost solutions at finite temperature

At high temperature, `ThermalLatticeSolver` can find many slightly different
solutions that have the same physics.

## What happens

For one band on the Bethe lattice (U = 2, B = 3, half filling) at T = 1, two
solutions have the same double occupancy, energy and grand potential, but
their ghost parameters differ, and Σ(iω₁) differs by 2%.

Besides the usual gauge freedom (rotating the quasiparticle or bath orbitals),
there are a few directions in which the parameters can move while the
equations barely change. There are 2(2B − 3) of them, and they get flatter as
the temperature rises. Moving along them changes the double occupancy by about
1e-10 and Σ by about 1e-4.

## Why

The equations match thermal density matrices. At high temperature these hardly
depend on the details of the ghost poles, only on a few averages of them, so
the rest of the ghost parameters are barely pinned down.

## What it means

- The thermodynamics is reliable. The ghost parameters, and Σ in detail, are
  not.
- Plain DIIS drifts along these directions and can take hundreds of cycles, or
  fail.
- `OnPlateau`, the default in `ThermalLatticeSolver`, anchors the fits once the
  solve is close, which fixes this.

It is still open why there are exactly 2(2B − 3) such directions, and whether
the spread in Σ matters for spectra.

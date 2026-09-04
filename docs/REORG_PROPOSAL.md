# dmipy-sim layout proposal

Measured 2026-09-04 against `dmipy_sim/` @ 6f8e83b.

## What is actually there

**Correction to the first draft of this document:** it claimed a flat 32-module namespace.
That came from globbing `dmipy_sim/*.py`, which does not see directories. `dmipy_sim`
**already has four subpackages** — `substrate/` (the biophysical tissue spec), `sequences/`
(physical acquisition), `math/`, `io/` — organised by ownership and role ("sim-owned forward
truth"). Two consequences: the names `substrate/` and `acquisition/` were already taken and
mean something different, and the claim that `dmipy_sim.substrate` / `dmipy_sim.sequences`
were broken downstream imports was wrong (they are packages; only `canonical` is missing).

- **19,845 lines in the top-level namespace**, plus those four packages. `geometries.py`
  alone was 3,548 lines and 12 classes.
- **88 test files / 14,483 lines**, flat except `tests/physics/` and `tests/validation/`.
- Five modules have **no direct test**: `mesh_axon`, `mesh_shapes`, `susceptibility`,
  `bank_cactus`, `_gpu_config`. Twelve test files import only the top-level package, so
  nothing names the module they cover.

## The key finding: the structure already exists, the filesystem just doesn't say so

The internal import graph is **acyclic and cleanly layered**:

| layer | modules |
|---|---|
| 0 | `_boundary`, `constants`, `gpu`, `_gpu_config`, `gaunt`, `noise`, `susceptibility_field` |
| 1 | `geometries`, `physics`, `waveforms`, `rf`, `compression`, `trajectories`, `susceptibility` |
| 2 | `core`, `bloch`, `mesh`, `curved_tube`, `mt`, `mt_walk`, `replay`, `mesh_shapes`, `viz`, `pedagogy` |
| 3 | `bank`, `phantom`, `pulse_sequence`, `mesh_axon` |
| 4 | `bank_cactus`, `mesh_bundle`, `sh_convolution` |

Nothing needs re-architecting. The layering is latent and correct; expressing it as
directories is a *rename*, not a redesign. That is what makes this worth doing and also
what caps the risk.

## Proposed layout

```
dmipy_sim/
  __init__.py                 # public API — unchanged, this is the contract
  constants.py

  geometry/                   # WHERE a walker may be  (NOT substrate/ -- that name is taken)
    _boundary.py              #   the shared rules (every geometry calls these)
    base.py                   #   Geometry ABC, FreeDiffusion, Box1D          ~130 l
    analytic.py               #   Sphere, Cylinder, Ellipsoid, Slab1D, Shell  ~1400 l
    packed.py                 #   PackedCylinders, PackedSpheres              ~1300 l
    myelin.py                 #   MyelinatedCylinder, PackedMyelinated…        ~650 l
    packing.py                #   pack_cylinders / pack_spheres helpers
    curved_tube.py
    mesh.py                   #   Mesh + load_ply
    mesh_shapes.py

  engine/                     # HOW it moves and accrues phase
    physics.py                #   scan bodies
    core.py                   #   simulate / simulate_trajectories entry points
    bloch.py                  #   vector-Bloch forward
    pulse_sequence.py         #   bloch-specific, belongs beside it
    mt.py  mt_walk.py
    gpu.py  _gpu_config.py

  # NOTE: no acquisition/ -- `sequences/` already occupies that role.
  # waveforms/rf/noise fold into it in a later step.

  replay/                     # walk once, replay many
    trajectories.py  compression.py  replay.py  bank.py  phantom.py
    sh_convolution.py  gaunt.py
    builders/                 #   substrate-specific master-walk assembly
      bank_cactus.py  mesh_bundle.py  mesh_axon.py

  fields/                     # off-resonance
    susceptibility.py  susceptibility_field.py

  viz/
    viz.py  pedagogy.py
```

Package-level dependencies then read `fields ← substrate ← engine ← replay`, with
`acquisition` a leaf. The one back-edge (`core → trajectories`) is *already* a deferred
function-level import, so it does not bind at module load.

## Tests mirror the package, one file per module

```
tests/geometry/test_packed.py       <-> dmipy_sim/geometry/packed.py
tests/geometry/test_boundary.py     <-> dmipy_sim/geometry/_boundary.py
tests/engine/test_core.py           <-> dmipy_sim/engine/core.py
tests/validation/                    (kept: cross-cutting physics vs analytic references)
tests/physics/                       (kept)
```

The rule that buys the most: **a test file's path is derivable from the module it covers.**
Today it is not, so "what do I re-run after editing X" is a guess. `tests/validation/` stays
as the deliberate exception — those assert physics across modules and should not be forced
into a 1:1 mapping.

## Migration, without breaking downstream

`dmipy-fit` and `dmipy-design` import **submodules**, not just the package:
`dmipy_sim.replay` (12x), `constants` (8x), `waveforms` (5x), `sh_convolution` (5x),
`gaunt` (4x), `geometries` (3x), `compression` (3x), `rf`, `bank`, `pulse_sequence`.

So the move needs shims. Ten one-line re-export modules at the old flat paths cover every
observed consumer:

```python
# dmipy_sim/geometries.py  (shim)
from .substrate.analytic import *      # noqa: F401,F403
from .substrate.packed import *        # noqa: F401,F403
```

**Related evidence that this is overdue:** three downstream imports are already broken —
`dmipy_sim.sequences`, `dmipy_sim.canonical`, `dmipy_sim.substrate` are imported by
dmipy-fit and *do not exist*. The flat namespace has already rotted silently. An
`__init__` surface test would have caught it and should land with this.

## Order of work

1. **Split `geometries.py`** into `geometry/` (highest value, self-contained). — done, #89
   1b. **Collapse `PackedCylinders`/`PackedSpheres`** onto a shared radial traversal: the
       two are 64-73% identical after erasing the 2-D/3-D naming, and ~300 lines of the
       duplication is scaffold rather than physics. Changes executed code, so its own PR.
2. Add `tests/geometry/` mirroring it; move the matching tests.
3. Add the public-API surface test + fix the three rotted downstream imports.
4. Move the remaining groups one package at a time, each with its shim.

Each step is independently green-able. Do not do all four in one commit: CI is ~80 min,
so a bad batch costs a full cycle to bisect.

## Honest cost

This is ~34k lines of pure motion with no behaviour change. The payoff is real but it is
*ergonomic*, not correctness — nothing here fixes a bug. It competes for time against
actual physics work, so it is worth doing when a change is about to touch many files
anyway (e.g. the next boundary or replay change), not as a standalone project.

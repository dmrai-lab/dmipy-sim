# The pack

A replay pack is a walk stored once: for every walker, its positions over the walk as a bridge (the exact start
and end, and `K` sine bands between them), the pool it was in, its accumulated wall contact, and, when the substrate
has a field source, the field along its path as a few cosine modes. The pack also carries the substrate spec it was
walked from, the walk's parameters, and a certificate: the codec's error and the Monte-Carlo floor, measured over a
battery of acquisitions. Nothing in it is a signal, and not one T2, relaxivity or susceptibility value is applied
in it; those are what a replay brings.

## Making one

The guide's pages build a small pack on the CPU, a cylinder of 2 µm radius walked for 10 ms by 300 walkers; every
later page starts from it the same way.

```python
import numpy as np
import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack

walk = d.simulate_trajectories(300, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="guide/cylinder", K=8, license="CC-BY-4.0", citation="the guide")
print(pack.n_walkers, pack.K, pack.n_t, pack.dt)                 # 300 walkers, 8 bands, the save grid
```

A walk can also keep the bands instead of its saves: `simulate_trajectories(..., accumulate=K)` accumulates them while
it runs, block by block, and returns a `CoefficientWalk` whose memory does not grow with the save interval. Its pack
is on the **continuous** basis (`compression.basis`, the projections on `sqrt(2 / T) sin(pi k t / T)` rather than the
DST-I of the saves), certified without the raw walk: the truncation from the free band variances, the block
quadrature's constant, the container's quantisation and the split-half floor read from the coefficients.

```python
cw = d.simulate_trajectories(300, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False, accumulate=8)
acc = build_replay_pack(cw, id="guide/cylinder-accumulated", license="CC-BY-4.0", citation="the guide")
print(acc.meta["compression"]["basis"], acc.fidelity["certified"], acc.meta["walk_params"]["block"]["m"])   # continuous derived 1
```

`K` is the number of sine bands kept per axis (per window, below): the pack's temporal band, `K / (2 T)` in hertz with `T` a window's duration, which is what an
acquisition's gradient content is checked against (`pack.waveform_band(seq)` says which band a sequence needs, and a
replay beyond the pack's band is refused; see [the sequence](sequence.md)). Left unstated, `K` is the scanner class's:
`build_replay_pack(walk, scanner="prisma")` stores at the class's certified band times the storage margin (the band the
paper measured for each class; a walk from `walk_spec` carries its class, so it packs with no argument), and a walk with
no class and no `K` is refused rather than searched. The walk's length `T` is the longest echo time the pack can
replay; a shorter one is a prefix.

A pack stores its walk in **storage windows** of one duration, 100 ms by default (`build_replay_pack(window_T=)`), consecutive
windows sharing their boundary save: a walk within one window is stored as one, a longer walk as a whole number of them,
each encoded with the same `K` and certified on its own. Window 0's tensors sit under the channel names and window
`i` under `s{i}/`, so `pack.window(i)` is a pack of that window, `pack.truncate(k)` the first `k` windows without
re-encoding, and an acquisition reads only the windows it reaches; a replay across windows is the sum of their
contractions. `pack.windows` is the table. A walk is continued rather than re-walked: `pack.extend(0.1, seed=17)`
resumes every walker from the last window's exact endpoint in the pool it is in, walks one more window on the
pack's own substrate with the fresh seed, and appends it with its own certificate (`pack.end_state()` is where the
walkers are; `replay.continuation` holds the two steps). A pack written as one long window is stored in shorter ones
by `pack.rewindow(0.1, out_path=...)`: every window is the parent's channels decoded on its saves and re-encoded at
the parent's bands per second, certified against them, and a trailing remainder of at most `max_dropped_fraction`
of the saves that no whole window covers is dropped and recorded (`provenance.rewindowed`).

## What it holds

```python
print(sorted(pack.arrays))                                        # the stored channels
print(pack.has_relaxation, pack.diffusivity)                     # the occupancy channel is there; the walk's D
print(round(pack.meta["fidelity"]["floor_max"], 2))                # the certificate: the Monte-Carlo floor of this walk, coarse at 300 walkers
```

- `pos_*`: the positions, as ends and bands. Bare diffusion needs only these.
- `comp_static` (or a run-length form): the pool of every walker over the walk, for T2 and T1 per pool.
- `blt_*`: the boundary local time, for surface relaxivity.
- `susc_path_*`, when the substrate has a field source: the field's channels along the path, for the sheath's
  susceptibility at a scanner's field.

A pack's susceptibility field, `pack.susceptibility_field`, is its substrate's pools' own: `"present"` when a pool
carries a `susceptibility` block (it is magnetic), `"absent"` when none does and the field is identically zero. Three
things say the same and must agree -- a magnetic pool, the `"field"` tier in the spec's `validity.tiers`, and the
field channel (`susc_path_*` or the field grid) stored in the pack -- and a pack, layout or spec in which they
disagree is refused by name when it is written and when it is read
(`dmipy_sim.spec.susceptibility_field_of`), as is a pack that embeds no spec. What a tissue's chi and a scanner may
then do is the [scanner](scanner.md) page's rule.
- `voxel_ijk`, `voxel_certificate`, when the pack was built on a voxel grid: a floor per voxel and pool.

## The nominal tissue

The spec the pack embeds names the material it was declared with, and `pack.nominal` is that material as a
`Tissue`; `pack.nominal_field_T` is the field the spec's susceptibilities were calibrated at.

```python
print(pack.nominal)                                               # the spec's T2, T1 by pool name, rho2, chi; D is the walk's
print(pack.nominal_field_T)
```

A published pack therefore reproduces its paper with one call, `pack.replay(seq, tissue=pack.nominal,
scanner=pack.nominal_field_T)`, and never by default.

## The band container

A pack stores a walker's path as the endpoints plus sine bands of the Brownian bridge; band `k` falls as `1/k`, so
the bits a band needs fall as `log2 k`. The default container is derived per pack: one quantisation step common to all bands, set so that the quantisation
phase noise costs a tenth of the floor the pack is ordered at, and band `k` takes `ceil(log2(a_k / step)) + 1` bits
for its measured maximum `a_k` (3 to 16 bits). Equal-bit runs form ranges of a whole number of bytes.
`band_curve(K, b1)` (`round(b1 - log2 k)` bits) and the two-level container (16 bands at 16 bits, the rest at 8) are
explicit choices. Each
range is one tensor per axis: `int8` / `int16` at 8 / 16 bits (`pos_x_b0`), otherwise bit-packed rows of
offset-binary values, most significant bit first, under `pos_x_b2u7` (`u<bits>` names the depth). The metadata lists
the ranges (`compression.container`, `{"bands": [k0, k1], "bits": b}`); a reader reads every container a pack names,
so packs written with the two-level container (16 bands at 16 bits, the rest at 8) read unchanged. The contact
channel takes the same container; the field path channel takes ranges over its modes (`{"modes": [k0, k1], ...}`)
and defaults to one 8-bit range.

## Saving, loading, opening

```python
import os, tempfile
tmp = tempfile.mkdtemp()
pack.save(os.path.join(tmp, "cylinder.rpk"))
from dmipy_sim.replay import ReplayPack
same = ReplayPack.load(os.path.join(tmp, "cylinder.rpk"))
assert same.n_walkers == pack.n_walkers
```

`ReplayPack.load` reads one `.rpk` file whole, from a local path or from a hub (`hf://owner/name/path.rpk`, checked
against the dataset's manifest; see [publishing](README.md#publishing-a-pack)). `ReplayPack.open(uri)` opens the columnar layout of a large pack
by reference, on a directory or on the Hub, and reads only the rows and bands an acquisition needs; that is the
subject of [images](images.md).

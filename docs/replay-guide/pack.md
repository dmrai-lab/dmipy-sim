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

walk = d.simulate_trajectories(300, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.01, seed=0, require_gpu=False, K=8)
pack = build_replay_pack(walk, id="guide/cylinder", license="CC-BY-4.0", citation="the guide")
print(pack.n_walkers, pack.K, pack.fidelity["certified"], pack.block["m"])   # 300 walkers, 8 bands, derived, 3
```

The walk keeps the bands, not the path: `simulate_trajectories` accumulates them while it runs, block by block (the
band's own block, `omega_K h = 0.8` at the top band, `pack.block["h"]`), and returns a `CoefficientWalk` whose memory
does not grow with the block count. The bands are the projections of each walker's bridge on `sqrt(2 / T) sin(pi k t /
T)`, and the pack is certified without a raw walk: the truncation from the free band variances, the block quadrature
(`rel(m, h)` times each channel's measured quadratic variation), the container's quantisation and the split-half floor
read from the coefficients. `raw_subset=n` keeps the first `n` walkers' block ends beside the bands
(`walk.saves()`), for a certificate measured on them (`fidelity="measured"`) or a picture of the paths.

`K` is the number of sine bands kept per axis (per window, below): the pack's temporal band, `K / (2 T)` in hertz with `T` a window's duration, which is what an
acquisition's gradient content is checked against (`pack.waveform_band(seq)` says which band a sequence needs, and a
replay beyond the pack's band is refused; see [the sequence](sequence.md)). The walk's `K` is the storage band by
default (256 bands per 100 ms); `walk_spec` walks the scanner class's certified band times the storage margin, and
`build_replay_pack(walk, scanner="prisma")` or `K=` stores a smaller band of a walk. The walk's length `T` is the longest echo time the pack can
replay; a shorter one is a prefix.

A walk is accumulated in **storage windows** of one duration, 100 ms by default (`simulate_trajectories(window_T=)`),
consecutive windows sharing their boundary instant: a walk within one window is one, a longer walk a whole number of
them, each with its own bands at the same `K`, stored and certified on its own. Window 0's tensors sit under the channel names and window
`i` under `s{i}/`, so `pack.window(i)` is a pack of that window, `pack.truncate(k)` the first `k` windows without
re-encoding, and an acquisition reads only the windows it reaches; a replay across windows is the sum of their
contractions. `pack.windows` is the table. A walk is continued rather than re-walked: `pack.extend(0.1, seed=17)`
resumes every walker from the last window's exact endpoint in the pool it is in, walks one more window on the
pack's own substrate with the fresh seed, and appends it with its own certificate (`pack.end_state()` is where the
walkers are; `replay.continuation` holds the two steps). A pack written as one long window is stored in shorter ones
by `pack.rewindow(0.1, out_path=...)`: every window is the parent's channels decoded at its block ends and re-encoded
at the parent's bands per second, certified against them, and a trailing remainder of at most `max_dropped_fraction`
of the blocks that no whole window covers is dropped and recorded (`provenance.rewindowed`).

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

A pack whose header carries a save grid (`n_t` / `dt_traj`), a declared `compression.basis` or a
`walk_params.cell_size` predates the continuous basis and is refused by name on load (RPK.md 9.5.1): its bands are
another basis's numbers, so its substrate is walked again; a pack is never converted.

`ReplayPack.load` reads one `.rpk` file whole, from a local path or from a hub (`hf://owner/name/path.rpk`, checked
against the dataset's manifest; see [publishing](README.md#publishing-a-pack)). `ReplayPack.open(uri)` opens the columnar layout of a large pack
by reference, on a directory or on the Hub, and reads only the rows and bands an acquisition needs; that is the
subject of [images](images.md).

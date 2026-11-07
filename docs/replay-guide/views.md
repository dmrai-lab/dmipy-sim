# Views: the same walk at another diffusivity or another size

A pack is walked once, at one diffusivity, one permeability and one size. Diffusion has two similarities that let
that one walk be read as a walk of a different situation without decoding anything: in time (`pack.at_diffusivity`,
dmipy-sim#289) and in space (`pack.at_length_scale`, dmipy-sim#542). Both return an ordinary `ReplayPack` that
shares the walk's arrays, says what it realises (its diffusivity, permeability and embedded spec are the view's),
and records the pair in `provenance`. This page states what each view keeps, what it moves, and where it stops,
because the quantities are intertwined: a diffusivity, a permeability, a relaxivity and a size cannot be changed one
at a time.

## The invariant

A walk is the same walk when its dimensionless numbers are the same. With `L` the smallest feature, `dt` the step,
`T` the duration, `D` the diffusivity, `kappa` a permeability and `rho` a surface relaxivity (both m/s):

| number | what it decides |
|---|---|
| `sqrt(D dt) / L` | how finely the walk resolves the wall |
| `kappa L / D`, `rho L / D` | how often a wall encounter crosses or relaxes (`kappa sqrt(dt / D)` per encounter) |
| `D T / L^2` | how far the walk reaches across the structure |

Each view changes one dimensional quantity and moves the others so that these stay what the walk realised. That is
why a view is exact: it is not an approximation of another walk, it IS that walk, read from the same arrays.

| view | what you ask for | what moves with it | what stays | direction |
|---|---|---|---|---|
| `at_diffusivity(D)`, `a = D / D_walk` | another diffusivity | the block (`h / a`, `T / a`), the band in hertz and `G_max` (`x a`), every permeability and MT reactivity (`x a`), a slower view's field scale (`chi_max x a`) | positions, every wall, every length | both, the envelope moved by `a` |
| `at_length_scale(s)` | another size | every length (`x s`: positions, walls, box, contact), the diffusivity (`x s^2`), every wall rate (`x s`), the certificate's battery (`b_max / s^2`, `rho2_over_D_max / s`) | the block, the duration, the band in hertz, the field values, T2 and T1 | both, bounded by the battery |

## In time: `at_diffusivity`

A path walked at `D` and read on a time axis divided by `a` (the block `h / a`, the window `T / a`) is a path walked at `a D`: the stored coefficients are
duration-agnostic. The same path crosses a permeable wall with the walked probability per encounter, which goes as
`kappa sqrt(dt / D)`, so the view is the walk at `(a D, a kappa)` and at no other pair.

```python
import numpy as np
import dmipy_sim as d
from dmipy_replay import Envelope, sequences
from dmipy_replay.spec.tissue import Tissue

env = Envelope(name="guide", band_hz=400.0, G_max=50.0, B0_list=(3.0, 7.0), chi_max=None, rho2_max=1e-4,
               max_refocusing_pulses=None)          # the guide's: a 400 Hz band, its short lobes' amplitude
walk = d.walk(d.Cylinder(2e-6, (0, 0, 1), permeability=1e-5), 300, T_max=0.01, envelope=env, diffusivity=2e-9,
              seed=0, require_gpu=False)
pack = d.build_replay_pack(walk, id="guide/cylinder", license="CC-BY-4.0", citation="the guide")

fast = pack.at_diffusivity(4e-9)                                  # a = 2: the same walk, twice as fast
print(fast.diffusivity, fast.permeability, fast.dt, fast.n_t)    # 4e-9, 2e-5 (the pair), h / 2, the same blocks
print(fast.meta["provenance"]["diffusivity_scaled"]["a"])        # 2.0, with walked and replayed values beside it
slow = pack.at_diffusivity(0.5e-9)                                # a = 1/4: the same walk over four times the time
print(pack.temporal_bandwidth_hz, slow.temporal_bandwidth_hz)    # the same bands over 4 T: a quarter of the band
print(pack.envelope.G_max, slow.envelope.G_max)                  # and a quarter of the amplitude
lobes = sequences.pgse([[1, 0, 0]], 0.001, 0.003, bvalues=[1e9], TE=0.009, slew_rate=np.inf)
pack.replay(lobes)                                                # inside the walk's 400 Hz
try:
    slow.replay(lobes)                                            # beyond the slower view's 100 Hz: refused by name
except ValueError as e:
    print(str(e)[:60])
```

Why both directions, and what each gives up. With `t' = a t` the walk's clock, a waveform `G(t)` on the view is the
waveform `G(t' / a) / a` on the walk, a field scale `B0 chi` on the view the walk's `B0 chi / a`, and a surface
relaxivity `rho2` the walk's `rho2 / a`. So the view's envelope is the walk's moved by `a` -- its band in hertz and its
`G_max` times `a`, and a slower view's `chi_max` times `a` -- and a request inside it IS a request inside the walk's:
every term of the certificate is the walk's own, at the walk's request. A faster view serves a wider band and a
stronger gradient over a shorter walk, `T / a`, which an acquisition must fit. A slower view serves a longer walk at a
narrower band: the walk stored `K` bands over `T` and the view spreads them over `T / a`, so a waveform beyond `a`
times the walked band, or beyond `a` times its `G_max`, is refused by name, the refusal naming the view.
`Tissue(D=...)` on any replay route takes this view; `Tissue(kappa=...)` takes it by the permeability, and the two
stated together must sit on the walk's line `kappa / D`.

## In space: `at_length_scale`

Multiply every length by `s` at fixed times. The path `s x(t)` is a walk of the substrate at `s` times the size
only if the step `sqrt(2 D dt)` also scales by `s`, so the diffusivity becomes `s^2 D`, and the per-encounter
probabilities `kappa sqrt(dt / D)` stay, so every wall rate becomes `s kappa`. The position coefficients and the
contact channel's boundary local time are lengths, linear in the stored arrays, so the view multiplies those arrays
by `s` in float64 and shares everything else; the embedded spec is scaled the same way
(`SubstrateSpec.scaled`), and the dipole field at corresponding points is unchanged, a dipole field depending on
shape, not size.

```python
half = pack.at_length_scale(0.5)                                  # the 1 um cylinder at 0.5e-9, same 10 ms
print(half.substrate.walls[0].surface.radius, half.diffusivity, half.permeability)   # 1e-6, 5e-10, 5e-6
seq = sequences.pgse([[1, 0, 0]], 0.001, 0.003, bvalues=[1e9], TE=0.006, slew_rate=np.inf)
seq_s = sequences.pgse([[1, 0, 0]], 0.001, 0.003, bvalues=[0.25e9], TE=0.006, slew_rate=np.inf)   # b goes as G^2
print(np.allclose(half.replay(seq), pack.replay(seq_s), rtol=1e-12))       # the view at G IS the pack at s G
rho2 = 2e-6
print(np.allclose(half.replay(seq, tissue=Tissue(rho2=rho2)), pack.replay(seq_s, tissue=Tissue(rho2=rho2 / 0.5)), rtol=1e-12))
```

Each channel follows in its own space. The gradient phase `gamma G . s x` makes the view at `G` the pack at `s G`.
The surface term `rho2 ell / D` with `ell -> s ell` and `D -> s^2 D` makes the view at `rho2` the pack at `rho2 / s`.
The relaxation gates, the compartment labels and the field path channel are unchanged.

Where it stops: the battery. The codec error and the floor were measured over a battery of acquisitions up to
`replay_envelope.acquisition.b_max`, the pack's own, not a scanner's. The view at `b` reads the walk at `s^2 b`, so
the view serves `b <= b_max / s^2`: more than the pack for `s < 1`, less for `s > 1`, and a replay past it is
refused rather than returned uncertified. The contact tier's `rho2_over_D_max` moves to `rho2_over_D_max / s` for
the same reason.

```python
big = pack.at_length_scale(2.0)                                   # the 4 um cylinder at 8e-9
print(pack.replay_envelope["acquisition"]["b_max"], big.replay_envelope["acquisition"]["b_max"])   # b_max, b_max / 4
try:
    big.replay(seq)                                               # b = 1e9 was fine on the pack: it is beyond b_max / 4
except ValueError as e:
    print(str(e)[:40])                                            # "b = 1e+09 s/m^2 is beyond the 7.5e+08 .."
```

## Together: one walk per shape

A target `(L, D, T)` is reached from a walk `(L0, D0, T0)` by the size view `s = L / L0` and then the time view
`a = D / (s^2 D0)`, which needs `T <= T0 / a`, and serves the band `a` times the walk's:

```
D T / L^2  <=  D0 T0 / L0^2        band(L, D)  =  (D / L^2) / (D0 / L0^2)  x  band(L0, D0)
```

So one walk per shape over the longest `D T / L^2` serves every size and every diffusivity, read with `pack.prefix`
where it is longer than needed. A walk at the slowest `D / L^2` the study needs (`a >= 1` at every target) holds its
band at every target; a target slower than the walk holds `a` times it. It saves walks, not steps: the walk must be
as long in blocks as the smallest target's walk would be.

```python
one_um_at_2e9 = pack.at_length_scale(0.5).at_diffusivity(2e-9)   # the 1 um cylinder at 2e-9, over 2.5 ms
print(one_um_at_2e9.diffusivity, one_um_at_2e9.permeability, one_um_at_2e9.dt, (one_um_at_2e9.n_t - 1) * one_um_at_2e9.dt)
```

## What a view does not do

* **It does not change the material.** T2, T1 and chi are not lengths and are not touched; a size view at the same
  tissue is the same water in a bigger structure. A permeability off the walk's line, or a wall made permeable, needs
  a walk: a crossing is a per-event draw and only its rate scales.
* **It does not re-certify.** Each view moves the envelope the walk was certified over and holds every replay to the
  moved one: the time view's band and `G_max` by `a`, the size view's battery by `s`; beyond either it refuses.
  An unscaled pack replayed beyond its own `b_max` is not refused today (dmipy-sim#706 tracks that gap against the
  format's §11), so the view's refusal is the stricter rule, not a weaker one.
* **It does not extend the walk.** The time view shortens it to `T / a`; the size view keeps it; neither reaches
  further across the structure than `D T / L^2` walked.
* **A saved view is a bigger file.** Its scaled arrays are float64, and it keeps the walk's `id`: a view you publish
  needs its own id, with `provenance.diffusivity_scaled` / `length_scaled` saying what it is.
* **A study takes the pack already viewed.** `Study` contracts the bands once on one grid, so a tissue at another
  diffusivity inside it is refused: `pack.at_diffusivity(D).study(...)`.

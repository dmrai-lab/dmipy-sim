# The scanner model

## What this is

`ScannerLimits` is not a table of deliverability limits with some extra fields bolted on. It is a
**parameterised forward model of an MRI machine**, and the catalogue is a set of parameter vectors
for it — one per machine we have citable numbers for.

That distinction is the whole design. A consumer does not ask "is this a Swoop?" and branch. It asks
the model for a field, a gradient, a transmit scale, a drift, at a point, and the model answers from
whatever parameters it was given. `ScannerLimits.of("swoop")` and a `ScannerLimits` you construct by
hand with your own numbers go down the same code path, because there is only one path.

So the catalogue's entries are not special. They are the values of the model that happen to be
published.

## A catalogue of twins

A **digital twin**, in the sense Hall et al. (MAGMA 2025) give the term, is a copy of a *specific
device*, kept *updated* as that device changes. That is a stronger claim than "a model of a scanner
of this type", and most of what is called a digital twin in MRI does not meet it.

This catalogue is a catalogue of twins in the weak sense and is built to support the strong one:

- **Weak (what we ship).** Each entry is the published characterisation of a *model* of machine —
  every Swoop, not a particular Swoop. Where a number is a population figure rather than a unit
  figure, the leaf says so, and where a fit is one admissible member of a family the catalogue
  records *how wide the family is* rather than presenting the member as the answer.
- **Strong (what the shape allows).** A site with its own field map, its own gradient
  nonlinearity fit, its own `f0` log, can fill in the same parameter vector for *their* serial
  number and get a twin of that unit. Nothing in the model needs to change; the values do. That is
  the point of keeping the machine's numbers out of Python entirely (`test_api_surface` enforces it)
  and out of the consumer's branches.

What we do **not** claim is the update loop. A twin that is not re-estimated as the magnet ages and
is re-shimmed is a snapshot, and the honest word for a snapshot is a characterisation. The
temperature and `f0`-recentering parameters exist so that the *within-session* part of that drift is
modelled rather than assumed away; the between-session part is the site's to measure.

## The frames

Three frames, and the model refuses to conflate them:

| frame | +z is | what lives here |
|---|---|---|
| substrate | the substrate's own axis | the walk, the pack |
| patient | RAS | the phantom grid, the prescription |
| magnet | `b0_axis` | the field laws, the concomitant expansion |

`b0_axis` and `b1_axis` are catalogue parameters, not constants, because on a Halbach magnet B0 is
*not* along the bore. `magnet_frame()` is the rotation into the frame where the field maths is
written, and `_check_frame` refuses a catalogue entry whose transmit axis is not perpendicular to
its B0 — a physically impossible machine should not be expressible.

## What a field law is allowed to be

`B_z` obeys Laplace's equation in a current-free region, so it is harmonic, and the model will only
evaluate laws that are. `solid_harmonics` is the standard basis (Romeo & Hoult 1984 nomenclature,
orders 1–4), every term verified harmonic with an analytic gradient.

This is a real constraint and it has already caught a wrong law: a `c·r²` inhomogeneity is not a
magnetic field at all (`∇²r² = 6`), and fitting one places a *forbidden interior minimum* at
isocentre. The published scalars are reproduced instead by an admissible `Z2 + Z2X` combination.

Two published scalars against fifteen coefficients is underdetermined, and the catalogue says so
rather than hiding it: the **scale** is pinned (every admissible law has the same steepest
gradient) while the **shape** is not (sampled laws disagree about the background gradient's
direction by a median 71°). A quantity driven by the worst gradient is therefore trustworthy; one
driven by where the gradient points is not. That split is recorded in the leaf, not in a comment.

## Reading it

- `dmipy_sim/acquisition/scanner_constants.json` — the catalogue. Every leaf is
  `value, unit, source_key, confidence`, and `source_key` resolves to a citation whose DOI is
  itself checked (#361).
- `dmipy_sim/acquisition/scanners.py` — the model.
- `dmipy_sim/acquisition/solid_harmonics.py` — the basis the field laws are written in.
- `dmipy_sim/phantom/bore.py` — the model evaluated over a phantom grid.

Gap analysis against other simulators' scanner inputs: #359.

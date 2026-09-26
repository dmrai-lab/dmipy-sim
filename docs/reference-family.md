# Producing a reference family

You have been given three things: a **family name**, its **queue row** in
[#459](https://github.com/dmrai-lab/dmipy-sim/issues/459), and this document. That is the whole brief.

A reference family is a set of replay packs of one released substrate that reproduces a published quantity, and
it is produced by running one pipeline: `dmipy_sim.replay.reference.ReferenceFamily`, the reference-pack
protocol of [#482](https://github.com/dmrai-lab/dmipy-sim/issues/482) with the card producer of
[#480](https://github.com/dmrai-lab/dmipy-sim/issues/480) as its eighth stage. You do not design a family; you
fill its first two records from primary sources and then operate the pipeline.

**Your judgement stops at the reference record.** Stages 1 and 2 are where you read, decide and write: which
files the family rests on, what their licence says, which published quantity it reproduces, what was measured on
what, and whose every free parameter is. From stage 3 on you operate: the spec comes from a producer, the walker
count from a measured pilot, the certificate from the pack, the verdict from the gate, the card from the
records, and none of them is yours to argue with. If a stage refuses, the refusal names the field; fix the
field, not the stage. If you find yourself wanting to widen a threshold, edit the gate, add a coverage factor,
re-run a measurement "to get a cleaner number", or write a number into a card by hand, you have left the
protocol — say so in your report instead.

## The nine stages

Each writes one record into the family's `records/`. A stage reads the record before it and nothing else,
refuses when that record is missing or has changed since it was read, and returns the record it already wrote
when its inputs are unchanged. No flag skips a stage and no flag reorders them.

| # | stage | what you do |
|---|---|---|
| 1 | `source` | Every input file: its URL, the host's licence text **verbatim** (a title is not a text), its sha256, its size, and the commit or version of the host record. A file that is not on disk cannot be digested and is refused. |
| 2 | `reference` | The published quantity as data: a DOI **resolved through Crossref** with its title compared; the quantity, its value, its relative uncertainty (`0.0` with a reason if the publication states none); the verbatim line of the document that PRINTS the number; the sample and whether it is the same object, the same material or a matched statistic; our own recorded direct measurement of the same quantity with its **analytic** standard error (an ensemble mean's, or a derived statistic's by the delta method — never a fold spread or a split half, which are seed-dependent), its grid and its solver; and every free parameter marked `theirs`, `literature` or `ours`. You never state a grade: `grade_of` assigns it. A family with no reference records `none`, may publish, and its card says so. |
| 3 | `spec` | A producer (`spec.label_volume_spec`, `winther_spec`, `disco_spec`, …) emits the substrate spec per pack; the stage round-trips it through `spec_of(geometry_from_spec(spec))` and checks that every file the spec cites is one the source record digested. Write a producer if none fits; do not construct a geometry. |
| 4 | `design` | Declare what the packs serve (the envelope, the waveforms, `K`, the window and save grid with the reason for it), the target floor per tier, the memory budget, the tolerance as the record **paths** its terms come from, and any measured systematic with its evidence. The stage then runs a **pilot** on the real window and sets the walker count from the measured floor and the measured resident bytes per walker; when the floor and the budget cannot both hold it records the trade and which tier holds. There is no way to supply an estimate. |
| 5 | `walk` | The walk at the design, with the engine's own counters, the memory budget checked before it starts. A family whose packs already exist declares a `RecordedWalk` instead and nothing is walked again. |
| 6 | `pack` | `build_replay_pack` on the declared envelope; the certificate per tier with `meets_target`; the served-vs-decoded difference; every declared waveform against `waveform_band`; and the reproduced quantity, computed on the grid the reference record states. |
| 7 | `gate` | Deterministic. It reads only records and compares only numbers other stages measured, and it has no tolerance in it: the budget is the design record's terms in quadrature. It fails by name. |
| 8 | `card` | The card, rendered from the records: the manifest table, three orthogonal cross-sections per substrate (`spec.preview`), the reproduction with its grade and the disagreement stated in the first paragraph when the comparison does not hold, the gate's verdict per pack, and a "Use me" snippet **executed** with a one-minute ceiling. A card whose snippet fails is not written. |
| 9 | `publish` | The packs the gate passed — the list comes from the gate's record, never from a glob — the records, the previews and the card. The code commit must be clean and pushed; creating the dataset is an explicit, logged step; the organisation page is regenerated from the families' own records afterwards. |

## What a family looks like

`examples/substrate_bank/build_imperial_rocks.py` is the worked example: the declaration (sources, reference,
design, build, publication) and three callables — a spec producer, a walk, and the reproduction of the quantity.
Everything else is the protocol. Run it stage by stage:

```bash
HF_HUB_CACHE=~/.cache/hf-session DMIPY_SIM_SURFACE_DIR=<the images> \
  python examples/substrate_bank/build_imperial_rocks.py --data <the images> --work <the family dir> \
  --stage source          # then reference, spec, design, walk, pack, gate, card
python examples/substrate_bank/build_imperial_rocks.py ... --dry       # everything, uploading nothing
```

## Your report is the records

Not a summary of them. Say which stage refused and why if one did, what the gate's verdict was per pack, and
what the card states in its first paragraph. If the family's packs do not reproduce their reference, that is a
result and the card says it; it is not a reason to change the tolerance.

## The things that went wrong before this existed

Each is now a refusal, and each cost a build:

* an acceptance test that checked monotonicity and the band but never compared the replayed number with the
  direct one, so a pack 2.2× off passed;
* a coverage factor of 2 introduced *after* a gate run, load-bearing for one pack's verdict;
* a tolerance whose dominant term was a six-fold spread that swung 3× over ten seeds;
* a walker count sized from an estimate that put the pack stage at 153 GB against a 60 GB budget;
* an inversion whose sampling grid changed the answer by a factor of two without the grid being in the record;
* a grade of `A` written by hand for a paper that prints no number;
* a "Use me" snippet that was a fourth copy of an estimator, was never executed, and did not return in 15 min;
* a card that never named its own gate, its `meets_target`, or the 17.7 % it sat from the measurement.

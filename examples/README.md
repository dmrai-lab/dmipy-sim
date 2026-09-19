# The cookbook

A ladder. Each rung runs, adds one idea, and says which rung it assumes. Start at the top if you are new;
jump to a part if you know what you need.

The order is not arbitrary. **A substrate is a spec, and the spec fixes what every later step can do**: which
pools exist and what water is in them, what the walls between them do, which pools this walk samples, and
which tiers the situation supports. A pack cannot carry a channel the walk did not record, a replay cannot
apply a tier the pack does not carry, and none of it recovers information the spec did not state. So the
contract comes first and everything else is downstream of it.

## Part I -- the substrate is a spec  (`spec/`)

| | |
|---|---|
| [01 the spec is the situation](spec/01_the_spec_is_the_situation.py) | every field of a spec, and that a geometry is one spelling of it |
| [02 pools](spec/02_pools.py) | where the water is, the pool-id convention, and what "nominal" means |
| [03 walls](spec/03_walls.py) | the surface between two pools, per-side relaxivity, per-direction permeability |
| [04 domain and boundary](spec/04_domain_and_boundary.py) | periodic, reflecting, open -- and why the recorded path is not wrapped |
| [05 seeding](spec/05_seeding.py) | which pools this walk samples, and how walkers are weighted |
| [06 validity and provenance](spec/06_validity_and_provenance.py) | what the spec claims it supports, and where it came from |
| [07 the fixed point](spec/07_the_fixed_point.py) | geometry to spec to geometry, walking to the same positions bit for bit |
| [08 write your own](spec/08_write_your_own.py) | a spec by hand, and every way it is refused |

## Part II -- what the walk fixes, and what replay leaves free

Planned. The division the whole library rests on: the walk bakes in the geometry, the diffusivity and the
permeability, because those change the paths; the tissue, the field, the pose and the acquisition are applied
afterwards to the same paths. Rungs: what a walk fixes; what a replay leaves free; the channels that carry
the free part and what refuses without them.

## Part III -- one walk, many acquisitions

Planned. Walk once and replay a scheme; sweep a tissue, a pose, a field without re-walking; save a pack,
load it elsewhere, read what it is certified for.

## Part IV -- substrates in practice

Analytic packings; calibrated white matter ([canonical_wm_flagship.ipynb](canonical_wm_flagship.ipynb));
meshes ([mesh_ply_and_viz.ipynb](mesh_ply_and_viz.ipynb)); the published producers (CACTUS, DiSCo,
CATERPillar, Winther).

## Part V -- the physics, one tier at a time

| | |
|---|---|
| [surface relaxivity, 1-D to 3-D](validation/surface_relaxivity_1d_to_3d.py) | against exact eigenvalues |
| [permeability, 1-D to 3-D](validation/permeability_1d_to_3d.py) | and what an idealised formula leaves out |
| [susceptibility](susceptibility_field.py) | a field at any strength and orientation |
| [magnetization transfer](mt_zspectrum.py) | a Z-spectrum |
| [coherence gating](pgste_t1_mixing_time.py) | a stimulated echo runs its mixing time on T1 |
| [exchange](fexi_axr_demo.py) | FEXI and the AXR |
| [extra-axonal tortuosity](validation/extra_axonal_tortuosity_scale.py) | the scale sweep |

## Part VI -- acquisitions

Planned. Realising an exact b and what a builder refuses; a multi-TE protocol from one walk; a pulse train
and what its crusher selects; Pulseq in and out; the same scheme through the analytical model and this engine.

## Part VII -- voxels and phantoms

[Composing a phantom from packs](rph/circular_wm_phantom.py); [a brain from a measured
FOD](rph/brain_from_csd.py) and [its round trip](rph/brain_from_csd_check.py). Planned: partitioning one walk
into voxels and regridding it; adding noise and fitting the result.

## Part VIII -- at scale

Planned. Opening a pack by reference and reading one voxel; planning a transfer before making it; imaging a
grid in one pass; running and consolidating a distributed fill.

## Part IX -- rigour

Planned. Reading a certificate; sizing a run by the floor you need; validating against a published dataset.

## Part X -- extending

Planned, and the part whose absence costs most. Adding a sequence family (the assembler, and the invariant
that the schedule decides where the echoes are); adding a scanner to the catalogue; adding a channel. The
geometry contract is already written, in `CLAUDE.md`, and is the model for the other three.

---

Rungs marked *planned* are tracked in dmipy-sim#317. Every file here runs in the test suite
(`tests/test_examples.py`); one that stops running is a rung that lies.

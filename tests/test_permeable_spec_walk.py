"""A spec with a permeable wall that maps to one geometry is walked by :func:`~dmipy_sim.walk` as that geometry is
walked, and a permeable multi-surface bundle is refused by name.

Exact, not statistical:

* the spec-level walk of a packed cell whose permeability is per object and over time is the geometry-level walk of
  the same substrate, bit for bit, in float32 (production) and float64 (the reference), with and without a
  :class:`~dmipy_sim.spec.walk.WalkContext`; so is the impermeable cell's, which states the impermeable rule;
* the permeability per object and over time is the spec's wall table, read back into the geometry and the walk's
  record, and the pack states it beside the rule its walkers' pools followed;
* a walker keeps the pool it was seeded in as its label, walkers do change pool, and none is excluded for it;
* a backend is handed the geometry's permeability table and the permeable step; one that does not declare it walks a
  table is refused by name;
* the step a pinned ``sub_steps`` is held to is the permeable law's.
"""
import numpy as np
import pytest

from dmipy_replay.spec.substrate import SpecError, SubstrateSpec
from dmipy_sim import walk
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry import PackedCylinders, PermeabilitySchedule, pack_cylinders
from dmipy_sim.spec import geometry_from_spec
from dmipy_sim.spec.walk import (DROP_POOL_CHANGE_RULE, IMPERMEABLE_POOL_RULE, PERMEABLE_POOL_RULE, WalkContext,
                                 _refuse_permeable_bundle)
from tests._envelope import stated, walk_envelope

D = 2e-9
RADII = [1.0e-6, 0.8e-6, 1.2e-6, 0.9e-6]
KAPPA = 4e-3                          # m/s: kappa R / D = 2 at R = 1 um, so walkers cross within the walk
T_MAX = 1e-3                          # four blocks of the walk envelope's band
T_B = 0.5e-3                          # the schedule's breakpoint: two blocks, on a step boundary
PER_OBJECT = [KAPPA, 0.0, KAPPA, 0.5 * KAPPA]       # object 1 sealed
LATER = [0.0, KAPPA, 2.0 * KAPPA, KAPPA]            # after the breakpoint: object 0 sealed, object 1 open
N = 64                                # an exact equality diverges on the first walker that reaches the branch
SEED = 5
ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])


def _cell(permeability):
    c, L, _ = pack_cylinders(RADII, target_vf=0.45, seed=3)
    return PackedCylinders(RADII, c, L, permeability=permeability)


def _scheduled():
    return _cell(PermeabilitySchedule([T_B], [PER_OBJECT, LATER]))


@pytest.fixture(params=["float32", "float64"])
def precision(request):
    """float32, the production arithmetic, and float64 (``jax_enable_x64``), the reference."""
    if request.param == "float64":
        request.getfixturevalue("x64")
    return request.param


def _spec_walk(spec, **kw):
    return walk(spec, N, T_MAX, envelope=walk_envelope(of=spec), diffusivity=D, seed=SEED, require_gpu=False, **kw)


def _geometry_walk(g, like):
    """The geometry-level walk of ``g`` at the spec-level walk ``like``'s band."""
    return _walk_geometry(N, D, g, T_MAX, K=like.K, f_band=float(walk_envelope(of=g.spec).band_hz), seed=SEED,
                          require_gpu=False)


def _assert_same_walk(a, b):
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    np.testing.assert_array_equal(a.boundary_endpoint, b.boundary_endpoint)
    np.testing.assert_array_equal(a.boundary_bands, b.boundary_bands)
    np.testing.assert_array_equal(a.compartment, b.compartment)
    assert (a.comp_events is None) == (b.comp_events is None)
    if a.comp_events is not None:
        np.testing.assert_array_equal(a.comp_events.offsets, b.comp_events.offsets)
        np.testing.assert_array_equal(a.comp_events.t, b.comp_events.t)
        np.testing.assert_array_equal(a.comp_events.label, b.comp_events.label)
    assert (a.sub_steps, a.dt_sim, a.illegal_crossings) == (b.sub_steps, b.dt_sim, b.illegal_crossings)
    assert a.work["permeability"] == b.work["permeability"]


@pytest.mark.parametrize("make, rule", [(_scheduled, PERMEABLE_POOL_RULE), (lambda: _cell(None), IMPERMEABLE_POOL_RULE)],
                         ids=["per_object_scheduled", "impermeable"])
def test_the_spec_walk_is_the_geometry_walk_to_the_bit(make, rule, precision):
    g = make()
    spec = g.spec
    plain = _spec_walk(spec)
    reference = _geometry_walk(g, plain)
    _assert_same_walk(plain, reference)
    _assert_same_walk(_spec_walk(spec, context=WalkContext(spec)), reference)
    assert plain.work["pool_rule"] == rule and "pool_rule" not in reference.work
    if rule == PERMEABLE_POOL_RULE:
        assert plain.comp_events.n_events > 0 and plain.work["permeability"]["breakpoints_s"] == [T_B]
    else:
        assert plain.comp_events is None and plain.work["permeability"] is None


def test_the_permeability_per_object_and_over_time_round_trips_through_the_spec():
    g = _scheduled()
    spec = SubstrateSpec.from_json(g.spec.to_json())
    (wall,) = spec.walls
    assert wall.permeability_table.breakpoints == [T_B]
    assert wall.permeability_table.in_to_out == [PER_OBJECT, LATER] == wall.permeability_table.out_to_in
    assert wall.permeability.in_to_out == wall.permeability.out_to_in == 2.0 * KAPPA      # the table's largest
    back = geometry_from_spec(spec)
    np.testing.assert_array_equal(back.permeability_table.breakpoints, g.permeability_table.breakpoints)
    np.testing.assert_array_equal(back.permeability_table.values, g.permeability_table.values)
    assert back.permeability == g.permeability == 2.0 * KAPPA
    one = _cell(PER_OBJECT).spec                                       # per object, constant in time
    assert one.walls[0].permeability_table.breakpoints == [] and one.walls[0].permeability_table.in_to_out == [PER_OBJECT]
    np.testing.assert_array_equal(geometry_from_spec(one).permeability_table.values, [PER_OBJECT])


def test_a_walker_keeps_its_seeded_pool_and_none_is_excluded_for_crossing():
    """The walk's label is the pool of the walker's start (the geometry's own classification), walkers change pool by
    the events of their occupancy, every walker is kept, and the pack states the permeability it walked and the rule."""
    from dmipy_sim.replay.bank import build_replay_pack
    g = _scheduled()
    spec = g.spec
    w = _spec_walk(spec, context=WalkContext(spec))
    assert w.n_walkers == N and "dropped_pool_change" not in w.work and w.pool_changed is None
    start = np.asarray(w.position_coeffs[:, 0, 0, :])
    seeded = np.asarray(g.pool_of(g.classify_positions_exact(start)))
    np.testing.assert_array_equal(w.compartment, seeded)
    assert 0 < (seeded == 1).sum() < N                                 # both pools seeded
    changed = np.diff(w.comp_events.offsets) > 0
    assert changed.sum() > 0, "no walker crossed a permeable wall"
    assert w.work["pool_rule"] == PERMEABLE_POOL_RULE != DROP_POOL_CHANGE_RULE
    with pytest.raises(SpecError, match="a pool change is an event of the walker's occupancy and excludes no walker"):
        _spec_walk(spec, drop_pool_changes=0.01)
    pk = build_replay_pack(w, id="test/permeable-spec", license="x", citation="x", K=w.K, **stated(ENV, of=w))
    walked = pk.meta["provenance"]["walk"]
    assert walked["pool_rule"] == PERMEABLE_POOL_RULE
    assert walked["permeability"]["breakpoints_s"] == [T_B] and walked["permeability"]["values_m_per_s"] == [PER_OBJECT, LATER]
    assert pk.arrays.get("comp_events_t") is not None


def test_the_pinned_step_is_held_to_the_permeable_law():
    """The membrane adds to the step law (``physics.MEMBRANE_COEFFICIENT``): a count the impermeable cell's law accepts is
    coarser than the permeable cell's, and refused there."""
    from dmipy_sim.engine.core import walk_cost
    perm, imp = _scheduled(), _cell(None)
    env = walk_envelope(of=perm.spec)
    kw = dict(K=env.mode_count(T_MAX), f_band=float(env.band_hz))
    law_perm, law_imp = (walk_cost(1, D, g, T_MAX, **kw).sub_steps for g in (perm, imp))
    assert law_perm > law_imp
    with pytest.raises(SpecError, match="coarser than the step law"):
        _spec_walk(perm.spec, sub_steps=int(law_imp))


class _Recording(backends.Backend):
    """A backend that keeps every request it is handed and whose walkers never move; it declares it walks a
    permeability table."""
    name = "recording"
    capabilities = frozenset({backends.PERMEABILITY_TABLE})

    def __init__(self):
        self.requests = []

    def walk_batch(self, request, r0, keys, *, out=None):
        from tests.backends.test_backend_hook import coefficient_batch
        self.requests.append(request)
        return coefficient_batch(request, r0)


class _Undeclared(_Recording):
    name = "undeclared"
    capabilities = frozenset()


def test_a_backend_is_handed_the_geometrys_permeability_table_and_the_permeable_step():
    spec = _scheduled().spec
    reference = _spec_walk(spec)                                        # the JAX walk of the same spec
    be = _Recording()
    w = _spec_walk(spec, context=WalkContext(spec), backend=be)
    assert w.work["backend"] == "recording" and w.work["pool_rule"] == PERMEABLE_POOL_RULE and be.requests
    for req in be.requests:
        table = req.geometry.permeability_table
        np.testing.assert_array_equal(table.breakpoints, [T_B])
        np.testing.assert_array_equal(table.values, [PER_OBJECT, LATER])
        assert req.kappa_over_D == pytest.approx(2.0 * KAPPA / D, rel=1e-12)       # the largest, which sizes the events
        assert req.record and req.events > 0 and req.diffusivity == D
        assert (req.sub_steps, req.dt_sim) == (reference.sub_steps, reference.dt_sim)
    assert w.work["permeability"] == reference.work["permeability"]
    with pytest.raises(ValueError, match="backend 'undeclared': this walk uses a permeability per object or over time"):
        _spec_walk(spec, backend=_Undeclared())


def _bundle(kappa):
    """A sphere-union spec seeding both sides of its wall: walked pool by pool."""
    from dmipy_sim.geometry.sphere_union import SphereUnion
    lo, hi = np.full(3, -4e-6), np.full(3, 4e-6)
    d = SphereUnion(np.array([[0.0, 0, 0], [1.5e-6, 0, 0]]), np.array([1e-6, 1e-6]), pool="intra", box=(lo, hi)).spec.to_dict()
    d["seeding"]["pools"] = [0, 1]
    for p in d["pools"]:
        p["water_fraction"], p["D"] = 1.0, D
    d["walls"][0]["permeability"] = {"in_to_out": kappa, "out_to_in": kappa}
    d["validity"]["tiers"] = ["gradient", "relaxation", "surface"] + (["exchange"] if kappa else [])
    return SubstrateSpec.from_dict(d)


def test_a_permeable_multi_surface_bundle_is_refused_by_name():
    spec = _bundle(1e-5)
    spec.validate()
    for call in (lambda: _spec_walk(spec), lambda: WalkContext(spec)):
        with pytest.raises(SpecError, match=r"wall \['union'\] is permeable, and the spec is walked pool by pool.*"
                                            r"a permeable multi-surface walk is not implemented"):
            call()
    _refuse_permeable_bundle(_bundle(0.0))                              # the impermeable bundle is walked
    assert WalkContext(_bundle(0.0)).tests.seeded == [0, 1]

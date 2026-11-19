"""A permeability per walker (``_walk_geometry(permeability_factor=)``): each walker of a walk meets the walls the
geometry scales at ``kappa * factor[w]``.

Every comparison is exact (``array_equal``), on the whole walk: the bands, the contact, the labels and their events,
the raw subset's positions and labels, and the trial record's ``n``, ``N`` and ``S``. The two cells that take the
argument: the packed cell of sheathed and plain cylinders (its plain cylinders' ``kappa_outer`` scaled, the sheathed
walls as stated) and the packed cylinders (the ``kappa / D`` each step reads, scaled).

1. an array of ones is the walk without the argument, bit for bit, in float32 and float64 accumulation, on two seeds;
2. an array of two values is two walks: walker ``w`` with factor ``f`` is walker ``w`` of the walk of the same cell
   whose scaled walls are stated at ``kappa * f``, walked without the argument, on the same seed (the same walker
   keys, so the same steps, words and uniforms). The factors are ``1/8`` and ``1/2``: a power of two scales a
   float32 or a float64 exactly, so the reference cell's stated values are the walker's own products bit for bit,
   whether the factor multiplies ``kappa`` (the plain cylinders) or ``kappa / D`` (the packed cylinders);
3. a walker of factor 0 never crosses a scaled wall: the walk is the walk of the cell whose plain cylinders are
   stated sealed; the sheathed walls are untouched;
4. a trial's outcome is its word against the threshold of the walker's own probability
   (:func:`~dmipy_sim.geometry._boundary.realised_acceptance` at ``kappa * factor / D`` and the logged overshoot);
5. the walk returns the array and states the rule; a walk without it states neither;
6. what the walk cannot take is refused by name.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine.core import PERMEABILITY_FACTOR_RULE, _walk_geometry
from dmipy_sim.geometry import (Cylinder, MyelinatedCylinder, PackedCylinders, PackedMyelinatedCylinders, PackedSpheres,
                                pack_cylinders)
from dmipy_sim.geometry._boundary import realised_acceptance
from dmipy_sim.geometry.base import PermeabilitySchedule
from dmipy_sim.geometry.myelin import concentric_wall_kernel
from dmipy_sim.geometry.packing import periodic_min_gap
from dmipy_sim.math.step_law import LONGEST_STEP
from dmipy_sim.trial_record import TrialRecordSpec

D = 2e-9
T = 4e-3
BLOCK = 2.5e-4
N_W = 32
SPEC = TrialRecordSpec(modes=3, band_edges=(0.0, 1e-3, 2.5e-3, T))
F_A, F_B = 0.125, 0.5                     # the two factors of criterion 2, powers of two

R = 2e-6
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
G = np.array([0.7, 1.0, 0.7, 1.0])                                     # cylinders 1 and 3 plain
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
COMPS = Compartments(intra=Pool(D=D), myelin=Pool(D=0.5e-9), extra=Pool(D=D))
KAPPA_OUTER = np.array([1e-5, 4e-3, 1e-5, 2e-3])                       # sheathed 0 and 2, plain 1 and 3
PLAIN = G == 1.0

RADII = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2]) * 1e-6
KAPPA = np.array([4e-3, 0.0, 4e-3, 4e-3, 4e-3, 4e-3])                  # object 1 sealed


def _mixed(plain_scale=1.0):
    """The mixed cell; ``plain_scale`` multiplies the plain cylinders' stated ``kappa_outer`` (the sheathed walls as
    stated)."""
    k_out = np.where(PLAIN, KAPPA_OUTER * plain_scale, KAPPA_OUTER)
    return PackedMyelinatedCylinders(OUTER * G, G, CENTRES, 4.4 * R, N_max=4, compartments=COMPS, kappa_inner=1e-5,
                                     kappa_outer=list(k_out))


def _packed(scale=1.0):
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    return PackedCylinders(RADII, c, Lc, permeability=KAPPA * scale)


CELLS = {"mixed": _mixed, "packed": _packed}


def _walk(geo, seed=3, acc="float32", **kw):
    return _walk_geometry(N_W, D, geo, T, BLOCK, seed=seed, tiers="all", require_gpu=False, K=8, raw_subset=N_W,
                          record_trials=SPEC, accumulation_dtype=acc, **kw)


def _rows(w, idx=None):
    """Every per-walker row of the walk (of the walkers ``idx``)."""
    idx = np.arange(w.n_walkers) if idx is None else np.asarray(idx)
    ev = w.comp_events
    off = np.asarray(ev.offsets)
    events = [(np.asarray(ev.t)[off[i]:off[i + 1]], np.asarray(ev.label)[off[i]:off[i + 1]]) for i in idx]
    out = dict(position_coeffs=w.position_coeffs[idx], boundary_endpoint=w.boundary_endpoint[idx],
               boundary_bands=w.boundary_bands[idx], qv_positions=w.qv_positions[idx], qv_boundary=w.qv_boundary[idx],
               compartment=w.compartment[idx], raw_positions=w.raw_positions[idx],
               raw_compartment=w.raw_compartment[idx], raw_boundary=w.raw_boundary[idx], n=w.trial_record.n[idx],
               N=w.trial_record.N[idx], S=w.trial_record.S[idx])
    out = {k: np.asarray(v) for k, v in out.items()}
    out["events_t"] = [t for t, _ in events]
    out["events_label"] = [lab for _, lab in events]
    return out


def _assert_rows_equal(a, b, what):
    assert a.keys() == b.keys()
    for k in a:
        if k.startswith("events"):
            assert len(a[k]) == len(b[k])
            for i, (x, y) in enumerate(zip(a[k], b[k])):
                np.testing.assert_array_equal(x, y, err_msg=f"{what}: {k} of walker {i}")
        else:
            assert a[k].dtype == b[k].dtype, f"{what}: {k}"
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{what}: {k}")


def _two_values():
    return np.where(np.arange(N_W) % 2 == 0, F_A, F_B).astype(np.float32)


# ── 1. ones are the walk ───────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("acc", ["float32", "float64"])
@pytest.mark.parametrize("cell", ["mixed", "packed"])
def test_an_array_of_ones_is_the_walk_without_it(cell, acc, request):
    """Two seeds, one geometry: the walk with ``permeability_factor = ones`` and the walk without it, every row
    ``array_equal`` and the work record the same but for the stated rule."""
    if acc == "float64":
        request.getfixturevalue("x64")
    geo = CELLS[cell]()
    for seed in (3, 4):
        bare = _walk(geo, seed, acc)
        ones = _walk(geo, seed, acc, permeability_factor=np.ones(N_W, np.float32))
        _assert_rows_equal(_rows(bare), _rows(ones), f"{cell} seed {seed}")
        assert int(bare.trial_record.N.sum()) > 0
        assert bare.illegal_crossings == ones.illegal_crossings
        wa = {k: v for k, v in bare.work.items() if k != "block"}
        wb = {k: v for k, v in ones.work.items() if k not in ("block", "permeability_factor")}
        assert wa == wb


def test_an_array_of_ones_is_the_position_only_walk_without_it():
    """The packed cylinders' position-only walk (``tiers=()``) with an array of ones and without it: the same bands."""
    geo = _packed()
    kw = dict(seed=3, tiers=(), require_gpu=False, K=8)
    bare = _walk_geometry(16, D, geo, T, BLOCK, **kw)
    ones = _walk_geometry(16, D, geo, T, BLOCK, permeability_factor=np.ones(16, np.float32), **kw)
    np.testing.assert_array_equal(bare.position_coeffs, ones.position_coeffs)
    np.testing.assert_array_equal(bare.qv_positions, ones.qv_positions)


# ── 2. two values are two walks ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", ["mixed", "packed"])
def test_an_array_of_two_values_is_two_walks_walker_by_walker(cell):
    """Even walkers at ``F_A``, odd at ``F_B``: each half's rows are the rows of the walk of the cell stated at the
    scaled permeability, walked without the argument on the same seed; the two factors grant different counts."""
    mixed = _walk(CELLS[cell](), permeability_factor=_two_values())
    got = _rows(mixed)
    even, odd = np.arange(0, N_W, 2), np.arange(1, N_W, 2)
    refs = {}
    for f, idx in ((F_A, even), (F_B, odd)):
        refs[f] = _walk(CELLS[cell](f))
        _assert_rows_equal(_rows(mixed, idx), _rows(refs[f], idx), f"{cell} factor {f}")
    # the same walkers at the two factors: different granted counts, so a factor read from the wrong walker shows
    N_A, N_B = (int(refs[f].trial_record.N.sum()) for f in (F_A, F_B))
    assert 0 < N_A < N_B, (N_A, N_B)
    assert int(got["N"][even].sum()) != int(refs[F_B].trial_record.N[even].sum())


# ── 3. zero never crosses ──────────────────────────────────────────────────────────────────────────────────────────

def test_a_walker_of_factor_zero_never_crosses_a_plain_wall_and_the_sheaths_are_untouched():
    """The mixed cell at factor 0 is the cell whose plain cylinders are stated sealed (the sheathed walls as stated),
    walked without the argument: every row equal, no plain trial granted, plain trials met, and sheathed walls
    crossed (a label change into or out of the sheath, pool 2)."""
    zero = _walk(_mixed(), permeability_factor=np.zeros(N_W, np.float32))
    ref = _walk(_mixed(0.0))
    a, b = _rows(zero), _rows(ref)
    _assert_rows_equal(a, b, "factor 0")
    assert int(a["N"].sum()) == 0 and int(a["n"].sum()) > 0
    assert any((lab == 2).any() for lab in a["events_label"]) or (a["compartment"] == 2).any()
    assert any(lab.size for lab in a["events_label"])


def test_a_packed_cylinder_walker_of_factor_zero_never_crosses():
    """The packed cylinders at factor 0: walls met, no trial granted, no label changed."""
    w = _walk(_packed(), permeability_factor=np.zeros(N_W, np.float32))
    assert int(w.trial_record.N.sum()) == 0 and int(w.trial_record.n.sum()) > 0
    assert int(np.asarray(w.comp_events.offsets)[-1]) == 0


# ── 4. the walker's own probability ────────────────────────────────────────────────────────────────────────────────

def test_a_trials_outcome_is_its_word_against_the_walkers_own_threshold():
    """One step of the mixed kernel per walker (steps up to six times the longest, so a step meets a plain wall
    several times), factors 1/8, 1/2 and 1 in turn, its plain-wall trials logged: every
    granted trial's word is below :func:`realised_acceptance` at the walker's ``kappa_outer * factor / D`` and the
    logged overshoot (``d_perp = u R / 2``, bracketed by its float32 rounding), every refused trial's word is not; the
    cell's stated ``kappa_outer`` misjudges some of them."""
    f32 = np.float32
    inner = np.where(PLAIN, OUTER, OUTER * G)
    outer = OUTER
    Lc = 4.4 * R
    step_max = LONGEST_STEP * inner.min() / 6
    Dj = jnp.full((4,), f32(D))
    k_out = KAPPA_OUTER.astype(f32)
    wall = concentric_wall_kernel(jnp.asarray(CENTRES, f32), jnp.asarray(inner, f32), jnp.asarray(outer, f32), f32(Lc),
                                  Dj, Dj, Dj, jnp.full(4, f32(1e-5)), jnp.asarray(k_out), jnp.zeros((4, 4), f32),
                                  f32(1e-7 * inner.min()), f32(1e-4 * inner.min()), step_max,
                                  periodic_min_gap(CENTRES, outer, Lc), trials=True)
    rng = np.random.default_rng(7)
    n = 6000
    k = rng.choice(np.flatnonzero(PLAIN), n)
    off = rng.uniform(-1.0, 1.0, n) * step_max
    ang = rng.uniform(0, 2 * np.pi, n)
    pts = (CENTRES[k] + (outer[k] + off)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(f32)
    th = rng.uniform(0, 2 * np.pi, n)
    dirs = np.stack([np.cos(th), np.sin(th)], 1).astype(f32)
    ls = (step_max * rng.uniform(0.2, 6.0, n)).astype(f32)                    # long: several encounters a step
    pool = np.where(off < 0, 1, 0).astype(np.int32)
    own = np.where(off < 0, k, 0).astype(np.int32)
    fac = np.array([F_A, F_B, 1.0], f32)[np.arange(n) % 3]
    keys = jax.random.split(jax.random.PRNGKey(7), n)
    words = jax.vmap(lambda kk: jax.random.bits(kk, (wall.bounce_loop.budget,), dtype=jnp.uint32))(keys)
    us = jax.vmap(lambda kk: jax.random.uniform(kk, dtype=jnp.float32))(keys)
    h = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, w, f: wall(p, d, l, pl, o, u, w, f)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(pool), jnp.asarray(own), us, words,
        jnp.asarray(fac))
    v, g, o, u = (np.asarray(x) for x in (h.trials.valid, h.trials.granted, h.trials.obj, h.trials.u))
    W = np.asarray(words).astype(np.uint64)
    walker = np.broadcast_to(np.arange(n)[:, None], v.shape)[v]
    obj, ov, granted, word = o[v], u[v].astype(np.float64), g[v], W[v]
    assert obj.size > 500 and granted.sum() > 50 and (~granted).sum() > 50
    assert v[:, 1:].sum() > 100                                                 # later encounters, each its own word
    kod_w = (k_out[obj] * fac[walker]).astype(f32) / f32(D)                  # the walker's own kappa / D, float32
    d_perp = ov * outer[obj].astype(f32).astype(np.float64) / 2.0
    lo = realised_acceptance(kod_w, (d_perp * (1 - 4e-7)).astype(f32)).astype(np.uint64)
    hi = realised_acceptance(kod_w, (d_perp * (1 + 4e-7)).astype(f32)).astype(np.uint64)
    assert np.all(np.where(granted, word < hi, word >= lo))
    assert np.all((word < lo) | (word >= hi))                                  # no trial within the bracket
    stated = realised_acceptance((k_out[obj]).astype(f32) / f32(D), d_perp.astype(f32)).astype(np.uint64)
    assert np.any(granted != (word < stated))                                  # the stated value misjudges some


# ── 5. the record ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_walk_returns_each_walkers_factor_and_states_the_rule(tmp_path):
    """The array, float32 in walker order, on the walk and through its walk file, a take and a join; the rule and the
    dtype the caller gave in the work record; a walk without the argument carries neither."""
    from dmipy_sim.coefficient_walk import CoefficientWalk, join_walks
    geo = _mixed()
    fac64 = np.linspace(0.0, 1.0, N_W)
    w = _walk(geo, permeability_factor=fac64)
    np.testing.assert_array_equal(w.permeability_factor, fac64.astype(np.float32))
    assert w.permeability_factor.dtype == np.float32 and w.permeability_factor.shape == (N_W,)
    rec = w.work["permeability_factor"]
    assert rec == dict(rule=PERMEABILITY_FACTOR_RULE, values="CoefficientWalk.permeability_factor", dtype="float32",
                       given_dtype="float64")
    w.save(tmp_path / "w.safetensors")
    back = CoefficientWalk.load(tmp_path / "w.safetensors")
    np.testing.assert_array_equal(back.permeability_factor, w.permeability_factor)
    assert back.work["permeability_factor"] == rec
    np.testing.assert_array_equal(w.take(np.arange(5, 9)).permeability_factor, fac64[5:9].astype(np.float32))
    a = _walk_geometry(16, D, geo, T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8,
                       permeability_factor=fac64[:16], events_capacity=64)
    b = _walk_geometry(16, D, geo, T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, first_walker=16,
                       permeability_factor=fac64[16:], events_capacity=64)
    np.testing.assert_array_equal(join_walks([a, b]).permeability_factor, fac64.astype(np.float32))
    bare = _walk(geo)
    assert bare.permeability_factor is None and "permeability_factor" not in bare.work


def test_walk_forwards_the_factor_on_a_spec_of_one_geometry():
    """``walk`` on the packed cylinders' spec with an array of ones is the walk without it."""
    from dmipy_replay import envelope_of
    from dmipy_sim.spec.walk import walk
    geo = _packed()
    env = envelope_of("prisma")
    kw = dict(envelope=env, diffusivity=D, seed=3, K=8, record_trials=SPEC, require_gpu=False)
    a = walk(geo.spec, 8, T, BLOCK, **kw)
    b = walk(geo.spec, 8, T, BLOCK, permeability_factor=np.ones(8, np.float32), **kw)
    for name in ("position_coeffs", "compartment"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.trial_record.N, b.trial_record.N)
    np.testing.assert_array_equal(b.permeability_factor, np.ones(8, np.float32))


# ── 6. refusals ───────────────────────────────────────────────────────────────────────────────────────────────────

def _refused(geo, fac, match, exc=ValueError, n=8, **kw):
    with pytest.raises(exc, match=match):
        _walk_geometry(n, D, geo, T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, permeability_factor=fac, **kw)


@pytest.mark.parametrize("make", [
    lambda: Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5),
    lambda: PackedSpheres(np.array([1e-6, 1.2e-6]), np.array([[0.0, 0.0, 0.0], [3e-6, 3e-6, 3e-6]]), 8e-6,
                          permeability=1e-5),
    lambda: MyelinatedCylinder(1.4e-6, 2e-6, (0, 0, 1), kappa_inner=1e-5, kappa_outer=1e-5, compartments=COMPS),
], ids=["cylinder", "packed_spheres", "myelinated_cylinder"])
def test_every_other_geometry_refuses_the_factor_by_name(make):
    geo = make()
    _refused(geo, np.ones(8, np.float32), type(geo).__name__ + " walks every walker at its stated permeability")


def test_a_cell_with_nothing_to_scale_refuses_the_factor():
    sheathed = PackedMyelinatedCylinders(OUTER * 0.7, np.full(4, 0.7), CENTRES, 4.4 * R, N_max=4, compartments=COMPS,
                                         kappa_inner=1e-5, kappa_outer=1e-5)
    _refused(sheathed, np.ones(8, np.float32), "has no plain cylinder")
    _refused(_mixed(0.0), np.ones(8, np.float32), "stated sealed")
    _refused(_packed(0.0), np.ones(8, np.float32), "impermeable")
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    sched = PackedCylinders(RADII, c, Lc, permeability=PermeabilitySchedule([1e-3], [4e-3, 2e-3]))
    _refused(sched, np.ones(8, np.float32), "schedule")


@pytest.mark.parametrize("fac, match, exc", [
    (np.ones(7, np.float32), r"shape \(8,\)", ValueError),
    (np.ones((8, 1), np.float32), r"shape \(8,\)", ValueError),
    (np.ones(8, np.int64), "float array", TypeError),
    (np.r_[np.ones(7), np.nan], "finite", ValueError),
    (np.r_[np.ones(7), np.inf], "finite", ValueError),
    (np.r_[np.ones(7), -0.5], r"in \[0, 1\]", ValueError),
    (np.r_[np.ones(7), 1.5], "event capacity are sized at the geometry's stated permeability", ValueError),
], ids=["short", "2d", "int", "nan", "inf", "negative", "above_one"])
def test_a_malformed_factor_is_refused(fac, match, exc):
    _refused(_mixed(), fac, match, exc)


def test_mt_binding_and_another_backend_refuse_the_factor():
    _refused(_mixed(), np.ones(8, np.float32), "MT binding", kappa_MT=1e-5, dwell_time=1e-3,
             equilibrate_binding="off")

    class Other:
        name = "other"
        capabilities = ()

        def refuses(self, request):
            return None

        def walk_batch(self, request, r0, keys):
            raise AssertionError("not walked")

    _refused(_mixed(), np.ones(8, np.float32), "JAX kernels", backend=Other())

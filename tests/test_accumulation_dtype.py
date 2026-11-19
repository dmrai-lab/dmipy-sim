"""The walk's accumulation dtype: float32 is the production precision and float64 the reference precision of the SAME
walk, so the float32 accumulation's rounding is the difference of two calls.

Exact, not statistical:

* the accumulators hold the dtype asked for, and the walk's ``block["precision"]`` is the dtype of the arrays it
  returned, in both precisions; the spec-level walk passes the dtype to the geometry walk and is that walk to the bit;
* a float64 accumulation walks the same walkers as the float32 one (the starts, the raw subset's positions, contact
  and pools, the start pools, the refused steps, all equal to the bit);
* a fixed sequence whose float32 running sum loses what the float64 running sum keeps (2^24 followed by a thousand
  ones) comes out of the walk's own accumulator code as each precision sums it: the quadratic variation through
  :func:`~dmipy_sim.engine.accumulate.moment_scan` and :func:`~dmipy_sim.engine.accumulate.ring_scan` from a float32
  path, and the level-2 cosine accumulator through :func:`~dmipy_sim.engine.accumulate.ring_scan` from fixed block
  moments (the smallest function that owns that sum);
* a request the walk cannot honour is refused by name: an unknown dtype, float64 with JAX's x64 mode off, float64 on
  a backend, on the adaptive walk and on the fused oracle.

The size of the float32 rounding is a measurement (``tools/accumulation_dtype.py``), asserted nowhere.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import walk
from dmipy_sim.engine.core import _walk_geometry, walk_cost
from dmipy_sim.geometry.base import Box1D
from tests._envelope import walk_envelope

D = 2e-9
BIG = 2.0 ** 24                       # float32's last integer with a unit spacing below it: BIG + 1 rounds to BIG
ONES = 1000


@pytest.fixture(params=["float32", "float64"])
def precision(request):
    """float32, the production precision, and float64 (``jax_enable_x64``), the reference."""
    if request.param == "float64":
        request.getfixturevalue("x64")
    return request.param


def _slab(dtype):
    return _walk_geometry(200, D, Box1D(length=10e-6), 0.1, seed=7, K=60, tiers=(), require_gpu=False,
                          accumulation_dtype=dtype)


def _cell_geometry():
    return d.PackedCylinders([2e-6], [[0.0, 0.0]], 8e-6)


def _cell(dtype):
    return _walk_geometry(200, D, _cell_geometry(), 0.01, seed=3, K=20, tiers="all", raw_subset=20, require_gpu=False,
                          accumulation_dtype=dtype)


ACCUMULATED = ("position_coeffs", "qv_positions", "boundary_endpoint", "boundary_bands", "qv_boundary")


def test_the_accumulators_hold_the_dtype_asked_for_and_the_record_says_so(precision):
    w = _cell(precision)
    for name in ACCUMULATED:
        assert np.asarray(getattr(w, name)).dtype == np.dtype(precision), name
    assert w.raw_positions.dtype == np.float32 and w.raw_boundary.dtype == np.float32      # the walk itself is float32
    assert w.block["precision"] == np.asarray(w.position_coeffs).dtype.name == precision
    assert f"a {precision} level-2 accumulator" in w.block["accumulation"]


def _spy_on_the_projection(monkeypatch):
    """Record the dtype name of every ring projection the walk's program traces."""
    import dmipy_sim.engine.accumulate as acc
    seen, real = [], acc.ring_projection

    def spy(tab, mom):
        out = real(tab, mom)
        seen.append(out.dtype.name)
        return out
    monkeypatch.setattr(acc, "ring_projection", spy)
    return seen


@pytest.mark.parametrize("make", [_cell, _slab], ids=["cell", "slab"])
def test_the_block_states_the_dtype_of_the_projection_product(make, precision, monkeypatch):
    seen = _spy_on_the_projection(monkeypatch)
    w = make(precision)                                          # a fresh geometry: the compiled program is cached on it
    assert seen and set(seen) == {w.block["projection_precision"]} == {precision}


def test_projection_precision_follows_the_product(x64, monkeypatch):
    """The field reads the dtype of what ``ring_projection`` returns, not the dtype it was asked for."""
    import jax.numpy as jnp
    import dmipy_sim.engine.accumulate as acc
    assert acc.projection_precision("float64", 4, 3) == "float64"
    real = acc.ring_projection
    monkeypatch.setattr(acc, "ring_projection", lambda tab, mom: real(tab, mom).astype(jnp.float32))
    assert acc.projection_precision("float64", 4, 3) == "float32"


def test_a_joined_walk_refuses_pieces_that_differ_in_the_projection_precision():
    from dataclasses import replace
    from dmipy_sim.coefficient_walk import join_walks
    g = Box1D(length=10e-6)
    a = _walk_geometry(32, D, g, 0.01, K=6, seed=1, require_gpu=False, tiers=())
    b = _walk_geometry(32, D, g, 0.01, K=6, seed=1, require_gpu=False, tiers=(), first_walker=32)
    assert join_walks([a, b]).block["projection_precision"] == "float32"
    other = replace(b, block=dict(b.block, projection_precision="float64"))
    with pytest.raises(ValueError, match="projection_precision"):
        join_walks([a, other])


@pytest.mark.parametrize("states, joined", [(("float32", "float32"), "float32"), (("float32", "float64"), "mixed")],
                         ids=["agree", "differ"])
def test_a_bundle_states_the_projection_precision_its_pools_agree_on_else_mixed(states, joined, tmp_path, monkeypatch):
    """Pool walks with equal array dtypes and given ``projection_precision`` metadata: the joined block states their
    common value, ``"mixed"`` when they differ, never the first pool's."""
    from dataclasses import replace
    import dmipy_sim.spec.walk as spec_walk
    from dmipy_sim.engine.accumulate import FIELD_RULE_CHI
    from tests.test_bundle_raw_subset import H, K, N_PROBE, T_MAX, _spec
    real, kept = spec_walk._join_pool_walks, []

    def join(parts, *a, **k):
        it = iter(states)
        parts = [(pid, w if isinstance(w, np.ndarray) else replace(w, block=dict(w.block, projection_precision=next(it))))
                 for pid, w in parts]
        kept.append(parts)
        return real(parts, *a, **k)
    monkeypatch.setattr(spec_walk, "_join_pool_walks", join)
    w = walk(_spec(tmp_path), 12, T_max=T_MAX, block_h=H, K=K, seed=3, require_gpu=False, n_probe=N_PROBE,
             envelope=walk_envelope(chi_max=FIELD_RULE_CHI))
    assert w.block["projection_precision"] == joined
    assert w.block["precision"] == "float32"


def test_the_default_is_the_float32_accumulation():
    a = _walk_geometry(64, D, Box1D(length=10e-6), 0.01, seed=7, K=12, tiers=(), require_gpu=False)
    b = _walk_geometry(64, D, Box1D(length=10e-6), 0.01, seed=7, K=12, tiers=(), require_gpu=False,
                       accumulation_dtype="float32")
    assert a.block["precision"] == "float32" and a.position_coeffs.dtype == np.float32
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    np.testing.assert_array_equal(a.qv_positions, b.qv_positions)


@pytest.mark.parametrize("make", [_slab, _cell], ids=["slab", "cell"])
def test_float64_accumulates_the_walkers_float32_walks(make, x64):
    """Both calls in one process (x64 on): the walk is float32 in either, so its walkers are the same to the bit and
    only the sums differ."""
    w32, w64 = make("float32"), make("float64")
    np.testing.assert_array_equal(w32.position_coeffs[:, :, 0, :], w64.position_coeffs[:, :, 0, :])     # the starts
    for name in ("raw_positions", "raw_boundary", "raw_compartment", "compartment"):
        a, b = getattr(w32, name), getattr(w64, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert a.dtype == b.dtype, name
            np.testing.assert_array_equal(a, b)
    assert w32.illegal_crossings == w64.illegal_crossings
    assert w32.work["exhausted_steps"] == w64.work["exhausted_steps"]
    assert (w32.block["precision"], w64.block["precision"]) == ("float32", "float64")


def test_the_spec_walk_is_the_geometry_walk_in_either_dtype(precision):
    g = _cell_geometry()
    env = walk_envelope(of=g.spec)
    plain = walk(g.spec, 64, 2e-3, envelope=env, diffusivity=D, seed=5, require_gpu=False, accumulation_dtype=precision)
    ref = _walk_geometry(64, D, g, 2e-3, K=plain.K, f_band=float(env.band_hz), seed=5, require_gpu=False,
                         accumulation_dtype=precision)
    for name in ACCUMULATED:
        a, b = getattr(plain, name), getattr(ref, name)
        assert a.dtype == b.dtype == np.dtype(precision), name
        np.testing.assert_array_equal(a, b)
    assert plain.block["precision"] == ref.block["precision"] == precision


def _rings_with_one(n_rings):
    """Per block: ``BIG`` at block 0, 1 at the first block of every later ring of
    :data:`~dmipy_sim.engine.accumulate.RING` blocks, 0 elsewhere -- so every ring's own sum is exact in float32 and
    the precision shows in the chain of rings alone, whatever order a ring is summed in."""
    from dmipy_sim.engine.accumulate import RING
    v = np.zeros(n_rings * RING)
    v[0] = BIG
    v[RING::RING] = 1.0
    return v


def test_the_quadratic_variation_is_summed_in_the_dtype_from_a_float32_path(precision):
    """The walk's own accumulation (moment_scan inside ring_scan, as ``_walk_geometry`` runs them) of a walker whose
    float32 path steps ``sqrt(BIG) = 4096`` once and then 1 once per ring: its quadratic variation is ``BIG + ONES``
    summed in float64 and ``BIG`` in float32, where every ``BIG + 1`` rounds back to ``BIG``."""
    import jax.numpy as jnp
    from dmipy_sim.engine.accumulate import Kind, moment_scan, ring_scan
    dtype = jnp.dtype(precision)
    steps = np.sqrt(_rings_with_one(ONES + 1))                      # one step per block
    x = np.concatenate([[0.0], np.cumsum(steps)])
    assert np.array_equal(x.astype(np.float32).astype(np.float64), x)          # a float32 path, every increment exact
    P = jnp.asarray(np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1), jnp.float32)
    kind = Kind(inner=lambda c, _: ((c[0] + 1,), None), pos_of=lambda c: P[c[0]], read_pos_of=lambda c: P[c[0]])

    def interval(c, b):
        c, mom = moment_scan(kind, c, 1, 0, dtype=dtype, step0=b)
        return c, None, mom

    c0 = ((jnp.int32(0),), None, None, None)
    _, ap, _, _, _, _, (qp, _, _, _, _), _ = ring_scan(interval, c0, len(steps), 0, 0, has_blt=False, dtype=dtype)
    assert qp.dtype == dtype and ap.dtype == dtype
    assert float(qp[0]) == (BIG + ONES if precision == "float64" else BIG)
    assert float(qp[1]) == float(qp[2]) == 0.0


def test_the_level2_accumulator_is_summed_in_the_dtype(precision):
    """ring_scan, the function that owns the level-2 cosine accumulator, given the block moments ``M_0`` of
    :func:`_rings_with_one` in the dtype: the ``k = 0`` mode (the table's row 1) is ``BIG + ONES`` in float64 and
    ``BIG`` in float32."""
    import jax.numpy as jnp
    from dmipy_sim.engine.accumulate import ring_scan
    dtype = jnp.dtype(precision)
    M0 = jnp.asarray(_rings_with_one(ONES + 1), dtype)

    def interval(c, b):
        M = jnp.zeros((1, 3), dtype).at[0, 0].set(M0[b])
        return c, None, (M, None, jnp.zeros((3,), dtype), None, None, None, None, None, None)

    _, ap, _, _, _, _, _, _ = ring_scan(interval, jnp.int32(0), int(M0.shape[0]), 0, 0, has_blt=False, dtype=dtype)
    assert ap.dtype == dtype
    assert float(ap[0, 0]) == (BIG + ONES if precision == "float64" else BIG)


def test_a_bundle_joins_its_pools_in_the_dtype_and_refuses_a_pool_kept_in_another(tmp_path, x64):
    """A multi-surface spec's pools, each walked by the geometry walk, joined in their dtype (the frozen sheath's
    zeros and field modes too); a pool checkpoint accumulated in float32 is refused by name for a float64 walk."""
    from dmipy_replay.spec.substrate import SpecError
    from dmipy_sim.engine.accumulate import FIELD_RULE_CHI
    from tests.test_bundle_raw_subset import H, K, N_PROBE, T_MAX, _spec
    spec = _spec(tmp_path)
    kw = dict(T_max=T_MAX, block_h=H, K=K, seed=3, require_gpu=False, n_probe=N_PROBE,
              envelope=walk_envelope(chi_max=FIELD_RULE_CHI))
    kept = str(tmp_path / "pools")
    w32 = walk(spec, 12, pool_checkpoint=kept, **kw)
    w64 = walk(spec, 12, accumulation_dtype="float64", **kw)
    for w, dt in ((w32, np.float32), (w64, np.float64)):
        for name in ("position_coeffs", "qv_positions", "field_coeffs", "qv_field", "qv_field_increments"):
            assert np.asarray(getattr(w, name)).dtype == dt, name
        assert w.block["precision"] == np.dtype(dt).name
    np.testing.assert_array_equal(w32.position_coeffs[:, :, 0, :], w64.position_coeffs[:, :, 0, :])
    np.testing.assert_array_equal(w32.compartment, w64.compartment)
    with pytest.raises(SpecError, match="accumulated in 'float32', not this walk's accumulation_dtype='float64'"):
        walk(spec, 12, pool_checkpoint=kept, accumulation_dtype="float64", **kw)


def test_the_cost_counts_the_accumulators_in_their_dtype(x64):
    """walk_cost's accumulator and device bytes are the dtype's: twice the float32 count in float64 for a walk with
    no events buffer (whose entries are bytes of their own)."""
    c32 = walk_cost(8, D, Box1D(length=10e-6), 0.1, K=60, tiers=())
    c64 = walk_cost(8, D, Box1D(length=10e-6), 0.1, K=60, tiers=(), accumulation_dtype="float64")
    assert c64.accumulator_bytes_per_walker == 2 * c32.accumulator_bytes_per_walker
    assert c64.bytes_per_walker - 20 == 2 * (c32.bytes_per_walker - 20)          # the start and key stay float32


# -------------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("bad", ["float16", "f64", "Float64", None, np.float64])
def test_an_unknown_dtype_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="accumulation_dtype is one of"):
        _walk_geometry(8, D, Box1D(length=10e-6), 1e-3, accumulation_dtype=bad)
    with pytest.raises(ValueError, match="accumulation_dtype is one of"):
        walk_cost(8, D, Box1D(length=10e-6), 1e-3, accumulation_dtype=bad)


def test_float64_without_x64_is_refused_by_name():
    from dmipy_replay import pgse
    with pytest.raises(ValueError, match="needs JAX's x64 mode"):
        _walk_geometry(8, D, Box1D(length=10e-6), 1e-3, accumulation_dtype="float64")
    with pytest.raises(ValueError, match="needs JAX's x64 mode"):
        walk_cost(8, D, Box1D(length=10e-6), 1e-3, accumulation_dtype="float64")
    g = _cell_geometry()
    with pytest.raises(ValueError, match="needs JAX's x64 mode"):
        walk(g.spec, 8, 2e-3, envelope=walk_envelope(of=g.spec), diffusivity=D, require_gpu=False,
             accumulation_dtype="float64")
    wf = pgse([[1, 0, 0]], 0.01, 0.03, bvalues=[1e9], n_t=300)
    with pytest.raises(ValueError, match="needs JAX's x64 mode"):
        d.simulate(8, D, wf, Box1D(length=10e-6), accumulation_dtype="float64", require_gpu=False)


class _Backend:
    """A backend object that would walk anything: what refuses the float64 request is the producer."""
    name = "elsewhere"

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys):
        raise AssertionError("a refused walk walks nothing")


def test_float64_on_a_backend_is_refused_by_name(x64):
    with pytest.raises(ValueError, match="accumulated by the JAX kernels; the backend 'elsewhere'"):
        _walk_geometry(8, D, Box1D(length=10e-6), 1e-3, backend=_Backend(), accumulation_dtype="float64")
    with pytest.raises(ValueError, match="accumulated by the JAX kernels; the backend 'elsewhere'"):
        walk_cost(8, D, Box1D(length=10e-6), 1e-3, backend=_Backend(), accumulation_dtype="float64")


def test_float64_on_the_adaptive_walk_is_refused_by_name(x64):
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
    with pytest.raises(ValueError, match="the adaptive walk accumulates in float32 only"):
        simulate_trajectories_adaptive(8, D, Box1D(length=10e-6), 1e-3, accumulation_dtype="float64")


def test_float64_on_the_fused_oracle_is_refused_by_name(x64):
    """A run without a diffusivity is the fused oracle's, which accumulates no coefficients."""
    from dmipy_replay import pgse
    wf = pgse([[1, 0, 0]], 0.01, 0.03, bvalues=[1e9], n_t=300)
    with pytest.raises(ValueError, match="the fused oracle's"):
        d.simulate(8, None, wf, d.Cylinder(radius=5e-6, orientation=(0, 0, 1)), accumulation_dtype="float64",
                   require_gpu=False)

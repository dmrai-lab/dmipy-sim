"""The near / far read of a periodic cell of parallel hollow cylinders (:class:`HollowCylinderField` with
``read="near_far"``): the near cylinders' closed form plus a per-cell Taylor series of the rest, against today's exact
read (``read="exact"``) and against its own derived remainder bound.

The cells are the repository's own packings: ``densify_packing`` at volume fraction 0.6 with a gap floor of 0.03 of
the mean radius, outer radii Gamma-distributed (mean 1 um, coefficient of variation 0.25, ``default_rng(0)``),
placement seed 0, 70 % of the cylinders sheathed at g-ratio 0.7, the rest plain (g-ratio 1) and carrying no field.
150, 1 000 and 10 000 cylinders are 105, 700 and 7 000 field sources.

1. Outside its outer wall one cylinder's twelve channels are, in float64 and against the closed form,
   ``(0, tP Re f, -tP Re f, -tP Im f, 0, 0, tA Re f, -tA Re f, 0, -tA Im f, 0, 0)`` with ``f = 1 / z^2``, ``z = x +
   i y`` from the axis, ``tP = -(b^2 - a^2) / 2`` and ``tA = (b^2 - a^2) / 12``: six channels zero, six the real or
   imaginary part of one analytic function times a weight (the structure of the lattice term).
2. The remainder bound holds as a bound and is not vacuous: at uniform points and on every cell corner (where the
   series is worst), in float64, ``|near_far - exact|`` is at most :meth:`HollowCylinderField.remainder_bound` per
   point and channel, and the largest bound is at most 100 times the largest observed error. The error is systematic
   (the truncated series' ``|zeta|^(order + 1)``): the corners' largest error exceeds the uniform points' and its
   log-log slope against ``|zeta|`` is reported.
3. At the working tolerance the largest bound is at most 1e-6 of the exact field's rms in its largest channel over the
   test points, and the float32 near / far read's largest deviation from the float64 exact read is at most twice the
   float32 exact read's.
4. Planted faults: the order lowered by 4 violates 3; the near set narrowed by one cell at the order chosen for two
   exceeds the tolerance.
5. Points in the lumen and the sheath of near cylinders and within one gap of a wall (either side) agree with the
   exact read within the bound.
The 10^5-point checks and the 10 000-cylinder cell (an exact read of 7 000 sources at every point) are tier 2
(``gpu``), the 150- and 1 000-cylinder cells at 10^4 points the CPU tier.

6. The rule and the record: below 512 cylinders and without a cell the exact read; a forced ``near_far`` without a
   cell refused by name; ``summation`` and the new keys recorded, a record rebuilt as it was written, the other read
   refused by name, a record whose stated bound the rebuild does not reproduce refused; the exact read's
   ``device_arrays`` refused for a near / far field. A walk reads the near / far field and records it.
"""
import functools
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim.fields.hollow_cylinder as hollow_cylinder
from dmipy_sim.fields.hollow_cylinder import (HollowCylinderField, NEAR_FAR_MIN_CYLINDERS, NEAR_FAR_TOLERANCE,
                                              NEAR_FAR_WIDTH, clear_near_far_cache)
from dmipy_sim.geometry.densify import densify_packing
from tests.test_seed_keys import _with_x64

R_MEAN = 1e-6
CHUNK = 128
logger = logging.getLogger("dmipy_sim.fields.hollow_cylinder")


@functools.lru_cache(maxsize=None)
def _packing(n):
    """``(outer, g, centres, L)`` of the ``n``-cylinder cell (module docstring)."""
    rng = np.random.default_rng(0)
    outer = rng.gamma(16.0, R_MEAN / 16.0, n)
    (centres, L), = densify_packing(outer, (0.6,), 2, seed=0, min_gap=0.03 * R_MEAN)
    g = np.full(n, 0.7)
    g[rng.permutation(n)[:int(round(0.3 * n))]] = 1.0
    return outer, g, centres, L


def _sources(n):
    outer, g, centres, L = _packing(n)
    sh = g < 1.0
    return centres[sh], (outer * g)[sh], outer[sh], L


@functools.lru_cache(maxsize=None)
def _field(n, read, order=None, near_cells=None):
    c, a, b, L = _sources(n)
    return HollowCylinderField(c, a, b, L=L, read=read, order=order, near_cells=near_cells)


def _read(field, P, dtype):
    """The field's read at ``P`` ``(n, 3)`` in ``dtype``, in chunks of :data:`CHUNK` points (the exact read at 7 000
    sources holds ``CHUNK x N`` closed forms at once)."""
    arrays, read, _ = field.device_read(dtype)
    f = jax.jit(jax.vmap(lambda r: read(arrays, r)))
    P = np.asarray(P, dtype)
    return np.concatenate([np.asarray(f(jnp.asarray(P[i:i + CHUNK])), np.float64) for i in range(0, len(P), CHUNK)])


def _corners(field):
    nf = field.near_far()
    G, cs, L = nf["cells_per_side"], nf["cell_size"], field.L
    x = -0.5 * L + np.arange(G) * cs
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)


def _uniform(L, n, seed=1):
    P = np.random.default_rng(seed).uniform(-0.5 * L, 0.5 * L, (n, 2))
    return np.concatenate([P, np.zeros((n, 1))], 1)


def _zeta(field, P):
    nf = field.near_far()
    G, cs, L = nf["cells_per_side"], nf["cell_size"], field.L
    p = np.mod(P[:, :2] + 0.5 * L, L)
    ij = np.clip(np.floor(p / cs).astype(np.int64), 0, G - 1)
    return np.hypot(*(p - (ij + 0.5) * cs).T)


# ---------------------------------------------------------------------------------------------------------- 1

def test_each_channel_outside_the_wall_is_the_stated_function_of_z(x64):
    """Claim 1, in float64, against the closed form (``dmipy_analytic.jax.fields.hollow_cylinder_basis``) at 20 000
    points from just outside the outer wall to 50 outer radii, every direction."""
    from dmipy_analytic.jax.fields import hollow_cylinder_basis
    rng = np.random.default_rng(0)
    a, b, n = 0.7e-6, 1e-6, 20_000
    rho = b * np.exp(rng.uniform(0.0, np.log(50.0), n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = rho * np.cos(th), rho * np.sin(th)
    P = np.stack([x, y, np.zeros(n)], 1)
    F = np.asarray(hollow_cylinder_basis(jnp.asarray(P), jnp.asarray(np.tile([0.0, 0.0, 1.0], (n, 1))), a, b))
    f = 1.0 / (x + 1j * y) ** 2
    tP, tA, z = -(b * b - a * a) / 2.0, (b * b - a * a) / 12.0, np.zeros(n)
    E = np.stack([z, tP * f.real, -tP * f.real, -tP * f.imag, z, z, tA * f.real, -tA * f.real, z, -tA * f.imag, z, z], 1)
    zero = [0, 4, 5, 8, 10, 11]
    assert np.all(F[:, zero] == 0.0)
    scale = np.abs(E).max(0)
    rel = np.abs(F - E).max(0)[[1, 2, 3, 6, 7, 9]] / scale[[1, 2, 3, 6, 7, 9]]
    assert np.all(rel < 64 * 2.0 ** -53), rel


# ---------------------------------------------------------------------------------------------------------- 2, 3

def _bound_check(n, n_points):
    f, e = _field(n, "near_far"), _field(n, "exact")
    nf = f.near_far()
    U, C = _uniform(f.L, n_points), _corners(f)
    P = np.concatenate([U, C])
    err = np.abs(_read(f, P, jnp.float64) - _read(e, P, jnp.float64))
    bound = f.remainder_bound(P)
    over = err > bound
    assert not over.any(), f"{int(over.sum())} (point, channel) pairs above the bound; worst {(err / np.where(bound > 0, bound, 1))[over].max():.3g}"
    ratio = bound.max() / err.max()
    assert ratio <= 100.0, ratio
    # systematic: the corners hold the largest error, and it grows as a power of |zeta|
    eu, ec = err[:len(U)].max(), err[len(U):].max()
    z = _zeta(f, U)
    eU = err[:len(U)].max(1)
    sel = eU > 0
    slope = np.polyfit(np.log(z[sel]), np.log(eU[sel]), 1)[0]
    print(f"\n{n} cylinders ({f.centers.shape[0]} sources), {n_points} uniform + {len(C)} corners: order {nf['order']}, "
          f"near width {nf['near_cells']} cells, {nf['cells_per_side']} cells per side, list {nf['list_length']}; "
          f"largest error uniform {eu:.3e} corners {ec:.3e}; largest bound {bound.max():.3e} = {ratio:.2f} x the largest "
          f"error; log-log slope of the error against |zeta| {slope:.2f}")
    assert ec >= eu
    return f, e, P, err


@pytest.mark.parametrize("n", [150, 1000])
def test_the_bound_holds_and_is_not_vacuous(n, x64):
    _bound_check(n, 10_000)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000, 10_000])
def test_the_bound_holds_at_1e5_points(n, x64):
    _bound_check(n, 100_000)


def _tolerance_check(n, n_points):
    f, e = _field(n, "near_far"), _field(n, "exact")
    P = np.concatenate([_uniform(f.L, n_points, seed=2), _corners(f)])
    ex64 = _read(e, P, jnp.float64)
    rms = np.sqrt(np.mean(ex64 ** 2, axis=0)).max()
    bmax = f.remainder_bound(P).max()
    d_new = np.abs(_read(f, P, jnp.float32) - ex64).max()
    d_old = np.abs(_read(e, P, jnp.float32) - ex64).max()
    print(f"\n{n} cylinders: largest bound {bmax:.3e} = {bmax / rms:.3e} of the rms {rms:.4e}; float32 against the "
          f"float64 exact read: near / far {d_new:.3e}, exact {d_old:.3e} (ratio {d_new / d_old:.3f})")
    assert bmax <= NEAR_FAR_TOLERANCE * rms
    assert d_new <= 2.0 * d_old
    return f, rms


@pytest.mark.parametrize("n", [150, 1000])
def test_the_working_tolerance_and_float32(n, x64):
    _tolerance_check(n, 10_000)


@pytest.mark.gpu
def test_the_working_tolerance_and_float32_at_10000(x64):
    _tolerance_check(10_000, 100_000)


# ---------------------------------------------------------------------------------------------------------- 4

@pytest.mark.parametrize("n", [150, 1000])
def test_planted_faults_fail(n, x64):
    """The order lowered by 4 and the near set narrowed by one cell (order held): both the largest bound and the
    largest observed error exceed the working tolerance."""
    good = _field(n, "near_far")
    nf = good.near_far()
    assert nf["near_cells"] == NEAR_FAR_WIDTH and nf["order"] >= 4
    e = _field(n, "exact")
    P = np.concatenate([_uniform(good.L, 10_000, seed=3), _corners(good)])
    ex64 = _read(e, P, jnp.float64)
    tol = NEAR_FAR_TOLERANCE * np.sqrt(np.mean(ex64 ** 2, axis=0)).max()
    for fault in (dict(order=nf["order"] - 4, near_cells=NEAR_FAR_WIDTH),
                  dict(order=nf["order"], near_cells=NEAR_FAR_WIDTH - 1)):
        bad = _field(n, "near_far", **fault)
        observed = np.abs(_read(bad, P, jnp.float64) - ex64).max()
        print(f"\n{n} cylinders, {fault}: largest bound {bad.near_far()['bound_max']:.3e}, observed {observed:.3e}, "
              f"tolerance {tol:.3e}")
        assert bad.near_far()["bound_max"] > tol
        assert observed > tol
        assert np.all(np.abs(_read(bad, P, jnp.float64) - ex64) <= bad.remainder_bound(P))


# ---------------------------------------------------------------------------------------------------------- 5

@pytest.mark.parametrize("n", [150, 1000])
def test_interior_and_wall_points_agree_within_the_bound(n, x64):
    """In the lumen, in the sheath and within one gap (0.03 of the mean radius) of either wall, both sides, of 50
    cylinders: the near cylinder's closed form carries the interior terms, so the six local channels agree to
    rounding and the rest within the bound."""
    f, e = _field(n, "near_far"), _field(n, "exact")
    rng = np.random.default_rng(4)
    k = rng.choice(f.centers.shape[0], 50, replace=False)
    c, a, b = f.centers[k], f.inner_radii[k], f.outer_radii[k]
    gap = 0.03 * R_MEAN
    radii = [rng.uniform(0, 1, (50, 20)) * a[:, None],                               # lumen
             a[:, None] + rng.uniform(0, 1, (50, 20)) * (b - a)[:, None],          # sheath
             a[:, None] + rng.uniform(-1, 1, (50, 20)) * gap,                      # within a gap of the inner wall
             b[:, None] + rng.uniform(-1, 1, (50, 20)) * gap]                      # within a gap of the outer wall
    th = rng.uniform(0, 2 * np.pi, (50, 20))
    P = np.concatenate([np.stack([c[:, 0:1] + r * np.cos(th), c[:, 1:2] + r * np.sin(th)], -1).reshape(-1, 2)
                        for r in radii])
    P = np.concatenate([P, np.zeros((len(P), 1))], 1)
    A, B = _read(f, P, jnp.float64), _read(e, P, jnp.float64)
    local = [0, 4, 5, 8, 10, 11]
    assert np.abs(A - B)[:, local].max() <= 1e-12 * np.abs(B).max()
    assert np.all(np.abs(A - B) <= f.remainder_bound(P))


# ---------------------------------------------------------------------------------------------------------- 6

def test_the_rule_and_the_refusals():
    c, a, b, L = _sources(150)
    assert HollowCylinderField(c, a, b, L=L).read == "exact"                       # 105 sources, below the rule
    assert HollowCylinderField(c, a, b).read == "exact"
    big = _sources(1000)
    assert big[0].shape[0] >= NEAR_FAR_MIN_CYLINDERS
    assert HollowCylinderField(*big[:3], L=big[3]).read == "near_far"
    assert HollowCylinderField(*big[:3], L=big[3]).summation == "lattice_near_far"
    with pytest.raises(ValueError, match="no cell"):
        HollowCylinderField(c, a, b, read="near_far")
    with pytest.raises(ValueError, match="read is None"):
        HollowCylinderField(c, a, b, L=L, read="fast")
    with pytest.raises(ValueError, match="exact read takes"):
        HollowCylinderField(c, a, b, L=L, read="exact", order=3)
    with pytest.raises(ValueError, match="device_arrays is the exact read"):
        _field(150, "near_far").device_arrays()


def test_the_record_round_trips_and_refuses_the_other_read():
    f = _field(150, "near_far")
    m = f.meta
    nf = f.near_far()
    assert m["summation"] == "lattice_near_far"
    assert (m["series_order"], m["near_cells"], m["series_cells_per_side"], m["series_list_length"]) == (
        nf["order"], nf["near_cells"], nf["cells_per_side"], nf["list_length"])
    assert m["series_tolerance"] == NEAR_FAR_TOLERANCE and m["series_bound_max"] == nf["bound_max"]
    assert m["series_table_bytes"] == nf["near"].size * 4 + nf["coefficients"].size * 8
    back = HollowCylinderField.from_meta(m)
    assert back.read == "near_far" and back.meta == m
    with pytest.raises(ValueError, match="written with the 'near_far' read"):
        HollowCylinderField.from_meta(m, read="exact")
    me = _field(150, "exact").meta
    assert me["summation"] == "lattice" and "series_order" not in me
    with pytest.raises(ValueError, match="written with the 'exact' read"):
        HollowCylinderField.from_meta(me, read="near_far")
    assert HollowCylinderField.from_meta(me).read == "exact"
    with pytest.raises(ValueError, match="this revision builds"):
        HollowCylinderField.from_meta(dict(m, series_bound_max=0.5 * m["series_bound_max"]))
    with pytest.raises(ValueError, match="states summation"):
        HollowCylinderField.from_meta(dict(m, summation="lattice_window"))


def _walk_geometry_150():
    from dmipy_sim.compartments import Compartments, Pool
    from dmipy_sim.geometry import PackedMyelinatedCylinders
    outer, g, centres, L = _packing(150)
    D = 2e-9
    compartments = Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D))
    return PackedMyelinatedCylinders(outer * g, g, centres, L, N_max=150, compartments=compartments), D


def _short_field_walk(geo, D):
    from dmipy_sim.engine.core import _walk_geometry
    f = HollowCylinderField.of(geo, read="near_far")
    assert f.read == "near_far"
    return _walk_geometry(16, D, geo, 0.001, seed=0, K=4, window_T=0.001, field_basis=f, K_c3=2,
                          field_read_interval=5e-4, raw_subset=16, require_gpu=False)


def test_a_walk_reads_the_near_far_field_and_records_it(x64):
    """A short walk of the 150-cylinder cell with the near / far field forced: the walk runs, its field record states
    the read, and its float32 reads of the walk's own positions agree with the exact read within the bound plus the
    float32 deviation the exact read itself has there."""
    geo, D = _walk_geometry_150()
    w = _short_field_walk(geo, D)
    f = w.field_basis
    assert w.field_basis.meta["summation"] == "lattice_near_far"
    P = np.asarray(w.raw_positions, np.float64).reshape(-1, 3)
    e = HollowCylinderField.of(geo, read="exact")
    ex = _read(e, P, jnp.float64)
    d_new = np.abs(_read(f, P, jnp.float32) - ex)
    d_old = np.abs(_read(e, P, jnp.float32) - ex)
    assert np.all(d_new <= f.remainder_bound(P) + 2.0 * d_old.max())


# ---------------------------------------------------------------------------------------------------------- Cache

@pytest.fixture
def empty_near_far_cache():
    clear_near_far_cache()
    yield
    clear_near_far_cache()


@pytest.fixture
def build_counts(empty_near_far_cache, monkeypatch):
    counts = dict(build=0, rms=0)

    def count(name, function):
        @functools.wraps(function)
        def counted(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)

        return counted

    monkeypatch.setattr(hollow_cylinder, "_build_near_far", count("build", hollow_cylinder._build_near_far))
    monkeypatch.setattr(hollow_cylinder, "_exact_rms", count("rms", hollow_cylinder._exact_rms))
    return counts


def _new_near_far_field(n=150, **options):
    c, a, b, L = _sources(n)
    return HollowCylinderField(c, a, b, **(dict(L=L, read="near_far") | options))


def _assert_same_bits(reference_tables, *comparisons):
    hashes = []
    for tables in (reference_tables, *comparisons):
        assert tables.keys() == reference_tables.keys()
        digest = sha256()
        for name, value in sorted(tables.items()):
            reference = reference_tables[name]
            if isinstance(reference, np.ndarray):
                assert isinstance(value, np.ndarray), name
                assert (value.dtype, value.shape) == (reference.dtype, reference.shape), name
                payload = value.tobytes()
                assert payload == reference.tobytes(), name
                digest.update(repr((name, value.dtype.str, value.shape, len(payload))).encode("ascii"))
                digest.update(payload)
            else:
                assert type(value) is type(reference) and value == reference, name
                digest.update(repr((name, value)).encode("ascii"))
        hashes.append(digest.hexdigest())
    assert all(digest == hashes[0] for digest in hashes)
    logger.info("Matching SHA-256: %s", hashes[0])


def _write_independent_tables(path):
    path = Path(path)
    centers, inner, outer, L = pickle.loads(path.read_bytes())
    with jax.default_device(jax.devices("cpu")[1]):
        field = HollowCylinderField(centers, inner, outer, L=L, read="near_far")
        tables = field._near_far_uncached()
    path.write_bytes(pickle.dumps(tables))


def _assert_cached_table_bits(cold, warm, bypassed):
    for name, value in bypassed.items():
        if isinstance(value, np.ndarray):
            assert np.shares_memory(cold[name], warm[name]), name
            assert not np.shares_memory(cold[name], value), name
    _assert_same_bits(cold, warm, bypassed)


def test_cached_tables_match_across_precision_and_default_device(empty_near_far_cache, tmp_path, monkeypatch):
    assert not jax.config.x64_enabled  # The suite's regular mode.
    cold = _new_near_far_field(1000).near_far()
    warm = _new_near_far_field(1000).near_far()
    exchange = tmp_path / "geometry-and-tables.pkl"
    exchange.write_bytes(pickle.dumps(_sources(1000)))
    flags = os.environ.get("XLA_FLAGS", "")
    monkeypatch.setenv("XLA_FLAGS", f"{flags} --xla_force_host_platform_device_count=2")
    _with_x64(
        "import tests.conftest; "
        "from tests.test_hollow_cylinder_near_far import _write_independent_tables; "
        f"_write_independent_tables({str(exchange)!r})"
    )
    _assert_cached_table_bits(cold, warm, pickle.loads(exchange.read_bytes()))


def test_cached_tables_preserve_walk_bits(x64, build_counts, monkeypatch):
    geo, D = _walk_geometry_150()
    HollowCylinderField.of(geo, read="near_far").near_far()
    warm = _short_field_walk(geo, D)
    assert build_counts == dict(build=1, rms=1)
    monkeypatch.setattr(HollowCylinderField, "near_far", HollowCylinderField._near_far_uncached)
    bypassed = _short_field_walk(geo, D)
    assert build_counts == dict(build=2, rms=2)
    names = ("field_coeffs", "position_coeffs", "qv_field", "qv_field_increments", "raw_field", "raw_positions")
    rows = []
    for walk in (warm, bypassed):
        record = {}
        for name in names:
            value = getattr(walk, name)
            assert value is not None, name
            record[name] = np.asarray(value)
        rows.append(record)
    _assert_same_bits(*rows)


def test_matching_keys_reuse_tables_and_a_changed_key_rebuilds(build_counts):
    cold = _new_near_far_field().near_far()
    assert build_counts == dict(build=1, rms=1)
    new = _new_near_far_field()
    warm = new.near_far()
    start = perf_counter()
    arrays, _, _ = new.device_read()
    jax.block_until_ready(arrays)
    seconds = perf_counter() - start
    logger.info("Warm device_read: %.6f s, cumulative counts %s", seconds, build_counts)
    assert build_counts == dict(build=1, rms=1)
    assert seconds < 0.5
    changed_tolerance = float(np.nextafter(NEAR_FAR_TOLERANCE, np.inf))
    _new_near_far_field(tolerance=changed_tolerance).near_far()
    assert build_counts == dict(build=2, rms=2)
    _new_near_far_field(tolerance=changed_tolerance).near_far()
    assert build_counts == dict(build=2, rms=2)
    bypassed = _new_near_far_field()._near_far_uncached()
    assert build_counts == dict(build=3, rms=3)
    _assert_cached_table_bits(cold, warm, bypassed)


@pytest.mark.parametrize("component", [
    "centers", "inner_radii", "outer_radii", "L", "tolerance", "order", "near_cells",
    "NEAR_FAR_MAX_ORDER", "NEAR_FAR_WIDTH", "NEAR_FAR_MAX_WIDTH", "NEAR_FAR_EXPLICIT_RADIUS",
    "NEAR_FAR_SAMPLE_RADIUS", "NEAR_FAR_SAMPLES", "NEAR_FAR_BOUND_RINGS", "NEAR_FAR_RMS_POINTS",
    "LATTICE_ROWS", "SERIES", "SERIES_SWITCH",
])
def test_each_table_input_changes_the_key(component, monkeypatch):
    original = _new_near_far_field()
    changed = _new_near_far_field()
    key = hollow_cylinder._near_far_key(original)
    assert hollow_cylinder._near_far_key(changed) == key
    if component in ("centers", "inner_radii", "outer_radii"):
        value = getattr(changed, component).reshape(-1)
        value[0] = np.nextafter(value[0], np.inf)
    elif component in ("L", "tolerance"):
        value = float(np.nextafter(getattr(changed, component), np.inf))
        changed = _new_near_far_field(**{component: value})
    elif component in ("order", "near_cells"):
        changed = _new_near_far_field(**{component: 10 if component == "order" else 2})
    else:
        value = getattr(hollow_cylinder, component)
        # Constants bound as function defaults still belong to the lookup fingerprint.
        if component == "SERIES":
            value = (value[0] * 1.001, *value[1:])
        elif component == "SERIES_SWITCH":
            monkeypatch.setitem(value, "float64", value["float64"] * 0.999)
        elif component == "NEAR_FAR_WIDTH":
            value -= 1
        elif isinstance(value, float):
            value = float(np.nextafter(value, -np.inf))
        else:
            value += 1
        monkeypatch.setattr(hollow_cylinder, component, value)
    assert hollow_cylinder._near_far_key(changed) != key


def test_returned_tables_protect_the_cached_entry(empty_near_far_cache):
    field = _new_near_far_field()
    tables = field.near_far()
    widths = deepcopy(tables["widths"])
    keys = set(tables)
    coefficient_shape = tables["coefficients"].shape
    for value in tables.values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = 0
            with pytest.raises(ValueError, match="WRITEABLE"):
                value.setflags(write=True)
    tables["coefficients"].shape = (-1,)
    tables["widths"][0]["bound_max"][0] = -1
    tables["widths"].clear()
    tables.clear()
    warm = field.near_far()
    assert warm["widths"] == widths
    assert warm.keys() == keys
    assert warm["coefficients"].shape == coefficient_shape


def test_the_fifth_key_evicts_the_least_recently_used(empty_near_far_cache, monkeypatch):
    builds = []

    def build(field):
        builds.append(field.tolerance)
        return {"tolerance": field.tolerance}

    monkeypatch.setattr(hollow_cylinder, "_build_near_far", build)
    fields = [_new_near_far_field(tolerance=NEAR_FAR_TOLERANCE * (1 + i / 10)) for i in range(5)]
    for field in fields[:4]:
        assert field.near_far()["tolerance"] == field.tolerance
    assert builds == [field.tolerance for field in fields[:4]]
    fields[0].near_far()
    fields[4].near_far()
    for i in (0, 2, 3, 4):
        fields[i].near_far()
    assert builds == [field.tolerance for field in fields]
    fields[1].near_far()
    assert builds == [field.tolerance for field in fields] + [fields[1].tolerance]


def test_concurrent_fields_share_one_cold_build(build_counts):
    ready = Barrier(8)

    def lookup(_):
        field = _new_near_far_field()
        ready.wait()
        return field.near_far()

    _packing(150)
    with ThreadPoolExecutor(max_workers=8) as pool:
        tables = list(pool.map(lookup, range(8)))
    assert build_counts == dict(build=1, rms=1)
    assert all(np.shares_memory(table["coefficients"], tables[0]["coefficients"]) for table in tables)
    logger.info("Eight concurrent lookups: cumulative counts %s", build_counts)

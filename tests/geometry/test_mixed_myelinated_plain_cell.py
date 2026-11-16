"""A periodic cell of myelinated and plain cylinders: :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` with a
g-ratio of 1 for a cylinder without a sheath.

Exact, not statistical:

* a cell whose every cylinder is myelinated walks as it always has: every array a walk returns, with and without the
  field, and the forward signal, hash to the values recorded for it;
* a plain cylinder's wall is the plain packing's wall, step for step: the same start, step and uniforms fed to
  :func:`~dmipy_sim.geometry.packed.packed_wall_kernel` and to the cell's kernel end at the same position, with the
  same crossings, encounters and contact, bit for bit, on sparse and dense cells, near walls and across gaps, at
  every prefix of the bounce loop; also where a step meets a (reflecting) sheathed cylinder and a plain wall;
* negating the axial component of a step leaves every in-plane quantity of the step bit-identical and negates the
  axial displacement; a step whose in-plane part is sheared by its axial part is the control that fails it;
* the walk accumulates the continuous position: a walker that crosses the cell's boundary has the displacement its
  steps add up to, not the difference of the folded positions;
* the spec of a mixed cell walks back to the same cell; refusals are by name.
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.geometry import PackedMyelinatedCylinders, pack_cylinders
from dmipy_sim.geometry._boundary import bounce_budget
from dmipy_sim.geometry.myelin import concentric_wall_kernel
from dmipy_sim.geometry.packed import packed_wall_kernel
from dmipy_sim.geometry.packing import periodic_min_gap
from dmipy_sim.math.step_law import LONGEST_STEP

D = 2e-9
R = 2e-6
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
L = 4.4 * R
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
COMPS = Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D))


def _cell(g, order=None, **kw):
    g = np.asarray(g, float)
    o = np.arange(4) if order is None else np.asarray(order)
    return PackedMyelinatedCylinders(OUTER[o] * g, g, CENTRES[o], L, N_max=4, compartments=kw.pop("compartments", COMPS),
                                     **kw)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def _events_sha(ev):
    return _sha(np.concatenate([np.asarray(x).ravel().view(np.uint8) for x in (ev.offsets, ev.t, ev.label)]))


# ── refusals ────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [np.nan, 0.0, -0.5, 1.2, np.inf])
def test_a_g_ratio_outside_zero_one_is_refused_by_name(bad):
    with pytest.raises(ValueError, match=r"every g-ratio is finite and in \(0, 1\]"):
        _cell([0.7, bad, 0.7, 1.0])


def test_a_myelinated_cylinder_without_a_sheath_in_float32_is_refused_by_name():
    with pytest.raises(ValueError, match="sheath of positive thickness"):
        _cell([0.7, 1.0 - 1e-12, 0.7, 1.0])


def test_overlapping_outer_walls_are_refused_by_name():
    c = CENTRES.copy(); c[1] = [1.5 * R, 0.0]                       # cylinder 1's wall inside cylinder 0's
    with pytest.raises(ValueError, match="overlap"):
        PackedMyelinatedCylinders(OUTER * [0.7, 1.0, 0.7, 1.0], [0.7, 1.0, 0.7, 1.0], c, L, N_max=4, compartments=COMPS)


def test_an_axolemma_property_on_a_plain_cylinder_is_refused_by_name():
    with pytest.raises(ValueError, match="kappa_inner is the axolemma of a myelinated cylinder"):
        _cell([0.7, 1.0, 0.7, 1.0], kappa_inner=[1e-5, 1e-5, 1e-5, 1e-5])
    with pytest.raises(ValueError, match="rho_inner is the axolemma"):
        _cell([0.7, 1.0, 0.7, 1.0], rho_inner=[1e-5, 2e-5, 1e-5, 0.0])
    g = _cell([0.7, 1.0, 0.7, 1.0], kappa_inner=1e-5)                 # a scalar is the myelinated cylinders' axolemma
    np.testing.assert_array_equal(np.asarray(g._kappa_inner_jax), np.float32([1e-5, 0.0, 1e-5, 0.0]))


@pytest.mark.parametrize("bad", ["sheath", ("extra", "extra"), (), 3])
def test_an_unknown_seed_pool_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="seed_pools is None"):
        _cell([0.7, 1.0, 0.7, 1.0], seed_pools=bad)


def test_a_cell_without_a_myelinated_cylinder_has_no_field_source():
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    plain = _cell([1.0] * 4)
    for f in (HollowCylinderField.of, field_grid_of):
        with pytest.raises(ValueError, match="no myelinated cylinder"):
            f(plain)
    mixed = _cell([0.7, 1.0, 0.7, 1.0])
    fb = HollowCylinderField.of(mixed)                                  # the sheaths of cylinders 0 and 2 only
    np.testing.assert_array_equal(fb.centers, CENTRES[[0, 2]])
    np.testing.assert_allclose(fb.inner_radii, 0.7 * OUTER[[0, 2]])


# ── the spec ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_spec_of_a_mixed_cell_walks_back_to_the_same_cell():
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.spec import geometry_from_spec
    g = _cell([0.7, 0.7, 1.0, 1.0], order=[0, 2, 1, 3], kappa_inner=1e-5, kappa_outer=[0.0, 0.0, 4e-5, 2e-5],
              rho_outer=1e-6, seed_pools=("extra", "intra"))           # myelinated first: the spec's order
    spec = g.spec
    assert [(w.name, w.inside_pool, w.outside_pool, len(w.surface.instances["radii"])) for w in spec.walls] == [
        ("axolemma", 1, 2, 2), ("sheath", 2, 0, 2), ("membrane", 1, 0, 2)]
    assert spec.validity.smallest_feature == pytest.approx(0.3 * R)                    # the thinnest sheath
    assert spec.validity.thinnest_shell == pytest.approx(0.3 * R) and spec.seeding.pools == [0, 1]
    back = geometry_from_spec(spec)
    assert type(back) is PackedMyelinatedCylinders
    for name in ("_inner_radii_np", "_outer_radii_np", "_centers_np"):
        np.testing.assert_allclose(getattr(back, name)[:4], getattr(g, name)[:4], rtol=1e-15)
    for name in ("_kappa_inner_jax", "_kappa_outer_jax", "_rho_inner_jax", "_rho_outer_jax", "plain_mask"):
        np.testing.assert_array_equal(np.asarray(getattr(back, name))[:4], np.asarray(getattr(g, name))[:4])
    assert back.seed_pools == g.seed_pools
    dry = _cell([0.7, 1.0, 0.7, 1.0], compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0, water_fraction=0.0),
                                                                 extra=Pool(D=D)), seed_pools=("extra", "intra"))
    assert geometry_from_spec(dry.spec).seed_pools == ("extra", "intra")      # a dry sheath stays unseeded
    assert back.spec.to_dict() == spec.to_dict()
    a = _walk_geometry(64, D, g, 1e-3, 2.5e-4, seed=1, K=4, require_gpu=False)
    b = _walk_geometry(64, D, back, 1e-3, 2.5e-4, seed=1, K=4, require_gpu=False)
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    np.testing.assert_array_equal(a.boundary_bands, b.boundary_bands)
    # a cell in any order comes back with its myelinated cylinders first
    mixed = geometry_from_spec(_cell([0.7, 1.0, 0.7, 1.0]).spec)
    np.testing.assert_array_equal(mixed.plain_mask[:4], [False, False, True, True])
    np.testing.assert_allclose(mixed._centers_np[:4], CENTRES[[0, 2, 1, 3]])


def test_a_cell_of_plain_cylinders_is_one_wall_and_walks_back_to_the_same_class():
    from dmipy_sim.spec import geometry_from_spec
    g = _cell([1.0] * 4, kappa_outer=[4e-5, 0.0, 4e-5, 2e-5])
    spec = g.spec
    (w,) = spec.walls
    assert (w.inside_pool, w.outside_pool, len(spec.pools)) == (1, 0, 3) and spec.validity.thinnest_shell is None
    back = geometry_from_spec(spec)
    assert type(back) is PackedMyelinatedCylinders and back.plain_mask[:4].all()
    np.testing.assert_array_equal(np.asarray(back._kappa_outer_jax), np.asarray(g._kappa_outer_jax))
    assert back.spec.to_dict() == spec.to_dict()


# ── the all-myelinated cell is today's walk ─────────────────────────────────────────────────────────────────────────

#: The arrays of the all-myelinated reference walks, recorded on the class before plain cylinders existed (CPU,
#: ``JAX_PLATFORMS=cpu``, :data:`RECORDED_WITH`). A walk of the same cell, seed and settings returns these bytes.
REFERENCE = {
    "field": {
        "position_coeffs": "cf784afe6f1372399844bcab7cb93f574ba8a88c778f5d7fd0f2bcaaa1c43351",
        "boundary_endpoint": "d3cc0706df8367d39925c199520600e49f0fc0fcb20553b454fd1f2ca2e34d76",
        "boundary_bands": "5e8da891248e713afa01dfd0cb836a41b84b5337bf47eaa2f64714e8ebed68d9",
        "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
        "comp_events": "8e63dde602a3ea9378abbb8dd463aaebc6238cc6e953d967d2966ef2ae3ec8d6",
        "bound_start": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66",
        "bound_events": "5fceeabde86a91a42fdcd3e4387e8ad12dc05e256ba127dcdead4e397e1b86ef",
        "field_coeffs": "c341c30675e8d5a62fed537aaa88da89f7787e36b43a3f7efa64649197fcc6b6",
        "qv_positions": "24b73a3becd1be85e234e41d7e3499964c0ac3281be51779a953841ff6065480",
        "qv_boundary": "35896dec343b87e45003e85c66e66c4dfe8f236a58904f8b1b39a2059f978dd8",
        "qv_field": "6337d5e2eb87fba167b869b26a14420bf1fc1cb50386c739961e976479a82d04",
        "qv_field_increments": "f961b2ba95e33cce2ea06312c71d968bb1d11db828b837eca3199b898e93f373",
    },
    "no_field": {
        "position_coeffs": "a0b73bfba5cf61b8a45af3b108f1dbdf14218e8f381f33259ad75ef8bd3f6735",
        "boundary_endpoint": "4fb5140ff86c9562da177c02646822b49bdf944854ba7604d8634a9c934d8123",
        "boundary_bands": "e5e86fd4ab9d6972470ba4e0c93adbc6d1b1567b257c76136c1f86b11c8e6882",
        "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
        "comp_events": "2f56bc4c3598c6f0629ca864e727aa46b27c10d550146be5795ae23bc25caea0",
        "qv_positions": "85c42a002f7dffe5e7961621d7893c417ffb0636759eaf93e39594a867b59db3",
        "qv_boundary": "63b3a85e00fba12436b9bda927206485a1adb4c454f996cf76813d3b6a897fc7",
    },
    "forward": ["97145e59d848907a4f0c22bafa96a722fb6b91d609ef3dcfabaf79c75d613caf",
                "af197e46ba0cfc590830fd5cae9f874da496969454db10331188111b7a2c07a7",
                "b43b8c74066db5a8cfb4c87116414e3dab2f8e7ea1fae1a62645aae9d4a1b0ea",
                "7ac4047d43a83524622f202a99027af4df10ebcbf607c8079b8401c0a0405ac4",
                "e5f307e5bd533a68a1eabbe7c01d8f203f188f40e5d1b7fe51245d446cd12b2d"],
}
G_ALL = [0.7, 0.8, 0.7, 0.75]
#: the JAX build the hashes were recorded with: another build compiles the same walk to other float32 bits (measured:
#: jaxlib 0.12 differs from 0.10 on every float array), so the hashes are an expectation of this build alone
RECORDED_WITH = ("0.10.2", "0.10.2")


@pytest.fixture
def recorded_build():
    import jax
    import jaxlib
    if (jax.__version__, jaxlib.__version__) != RECORDED_WITH:
        pytest.skip(f"the reference hashes were recorded with jax / jaxlib {RECORDED_WITH}; this is "
                    f"{(jax.__version__, jaxlib.__version__)}, which compiles the walk to other float32 bits")


def _hashes(w, names):
    return {n: (_events_sha(getattr(w, n)) if n.endswith("events") else _sha(getattr(w, n))) for n in names}


def test_the_all_myelinated_walk_with_the_field_is_the_recorded_walk(recorded_build):
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    geo = _cell(G_ALL, kappa_inner=4e-5, kappa_outer=4e-5)
    w = _walk_geometry(300, D, geo, 4e-3, 5e-5, tiers="all", seed=3, require_gpu=False, K=8, kappa_MT=4e-5,
                       dwell_time=1e-3, equilibrate_binding="off", field_basis=HollowCylinderField.of(geo), K_c3=8,
                       field_read_interval=1.25e-5)
    assert _hashes(w, REFERENCE["field"]) == REFERENCE["field"]


def test_the_all_myelinated_walk_without_the_field_is_the_recorded_walk(recorded_build):
    from dmipy_sim.engine.core import _walk_geometry
    geo = _cell(G_ALL, kappa_inner=4e-5, kappa_outer=4e-5)
    w = _walk_geometry(300, D, geo, 4e-3, 5e-5, tiers="all", seed=3, require_gpu=False, K=8)
    assert _hashes(w, REFERENCE["no_field"]) == REFERENCE["no_field"]


def test_the_all_myelinated_forward_signal_is_the_recorded_signal(recorded_build):
    from dmipy_replay import pgse
    from dmipy_sim.engine._fused import simulate_fused
    geo = _cell(G_ALL, compartments=Compartments(intra=Pool(D=D, T2=0.08), myelin=Pool(D=0.0, T2=0.015),
                                                 extra=Pool(D=D, T2=0.06)),
                kappa_inner=4e-5, kappa_outer=4e-5, rho_inner=1e-5, rho_outer=2e-5)
    wf = pgse([[1, 0, 0], [0, 1, 0]], 2e-3, 8e-3, bvalues=[1e9, 1e9], n_t=100, slew_rate=np.inf)
    out = simulate_fused(300, waveform=wf, geometry=geo, seed=3, require_gpu=False, return_walker_signals=True,
                         return_compartments="full")
    assert [_sha(a) for a in out] == REFERENCE["forward"]


# ── a plain wall is the plain packing's wall, step for step ─────────────────────────────────────────────────────────

def _dense_cell():
    radii = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2]) * 1e-6
    c, Lc, _ = pack_cylinders(radii, target_vf=0.45, seed=1)
    return radii / 0.999, c, Lc                                          # walls almost touching: min gap ~ 2 nm


def _gap_cell():
    r = 1e-6
    return np.array([r, r]), np.array([[0.0, 0.0], [2 * r + 0.05 * r, 0.0]]), 20 * r


def _starts(rng, radii, centres, Lc, step_max, n, sides):
    """``n`` starts and steps: half uniform in the cell, half at a stated offset from a wall (the nudge, 1e-3 and 0.1
    of a step, up to two steps) on the side ``sides`` allows, each with a step up to ``step_max`` in any direction."""
    nudge = 1e-4 * radii.min()
    pts = rng.uniform(-Lc / 2, Lc / 2, (n // 2, 2))
    k = rng.integers(0, len(radii), n - n // 2)
    off = rng.choice([nudge * 1.5, 1e-3 * step_max, 0.1 * step_max, 0.5 * step_max, step_max, 2 * step_max], len(k))
    sign = rng.choice(sides, len(k))
    ang = rng.uniform(0, 2 * np.pi, len(k))
    rad = np.maximum(radii[k] + sign * off, 0.1 * radii[k])
    near = centres[k] + rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = np.concatenate([pts, near])
    pts = (pts + Lc / 2) % Lc - Lc / 2
    th = rng.uniform(0, 2 * np.pi, n)
    length = step_max * rng.uniform(0.0, 1.0, n) ** 0.5
    return pts.astype(np.float32), np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32), length.astype(np.float32)


def _owner(pts, radii, centres, Lc):
    q = pts[:, None, :] - centres[None]
    q -= Lc * np.round(q / Lc)
    d2 = np.sum(q * q, -1)
    inside = d2 < radii[None] ** 2
    return inside.any(1), np.argmax(inside, 1)


def _pair(radii, centres, Lc, kappa, plain, n, seed, *, budget=None, sheath=None, coins=None):
    """Feed ``n`` steps to the plain packing's kernel and to the cell's kernel; return both results and the inputs.
    ``plain`` marks the cylinders that are plain in the cell (the others are myelinated with inner radius
    ``sheath * radius`` and impermeable, so the plain kernel sees their outer wall as a reflecting disk)."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    N = len(radii)
    nudge = f32(1e-4 * min(radii.min(), (radii * (sheath if sheath is not None else 1.0)).min()))
    eps = f32(1e-7 * radii.min())
    step_max = LONGEST_STEP * radii.min() / 6
    min_gap = periodic_min_gap(centres, radii, Lc)
    kap = np.where(plain, kappa, 0.0).astype(f32)
    inner = np.where(plain, radii, radii * (sheath if sheath is not None else 1.0))
    cj, rj = jnp.asarray(centres, f32), jnp.asarray(radii, f32)
    Dj = jnp.full((N,), f32(D))
    kod = jnp.asarray(kap) / jnp.maximum(Dj, 1e-30)                     # the cell kernel's own arithmetic
    rho_over_D = f32(1e5)
    w_p = packed_wall_kernel(cj, rj, f32(Lc), eps, nudge, step_max, min_gap, count=True, budget=budget)
    w_u = concentric_wall_kernel(cj, jnp.asarray(inner, f32), rj, f32(Lc), Dj, Dj, Dj, jnp.zeros(N, f32), jnp.asarray(kap),
                                 jnp.full((N, 4), rho_over_D), eps, nudge, step_max, min_gap, count=True, budget=budget)
    B = w_p.bounce_loop.budget if coins is None else coins          # ``coins``: the length the uniforms are drawn at
    sides = [1.0, -1.0] if sheath is None else [1.0]
    pts, dirs, ls = _starts(rng, radii, centres, Lc, step_max, n, sides)
    inside, own = _owner(pts, radii, centres, Lc)
    keep = ~inside | plain[own]                                          # never inside a myelinated cylinder
    pts, dirs, ls, inside, own = pts[keep], dirs[keep], ls[keep], inside[keep], own[keep]
    keys = jax.random.split(jax.random.PRNGKey(seed), len(pts))
    us = jax.vmap(lambda k: jax.random.uniform(k, (B,), dtype=jnp.float32))(keys)
    pool = jnp.asarray(inside.astype(np.int32)); k0 = jnp.asarray(own.astype(np.int32))
    hp = jax.jit(jax.vmap(lambda p, d, l, s, key: w_p(p, d, l, s, kod, rho_over_D, key)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    hu = jax.jit(jax.vmap(lambda p, d, l, pl, k, u: w_u(p, d, l, pl, k, f32(0.5), u)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), pool, k0, us[:, :w_u.bounce_loop.budget])
    return w_p, w_u, hp, hu, pool


def _assert_pairs_equal(hp, hu, pool, what):
    np.testing.assert_array_equal(np.asarray(hu.xy), np.asarray(hp.r), err_msg=f"{what}: end position")
    np.testing.assert_array_equal(np.asarray(hu.pool != pool), np.asarray(hp.crossed), err_msg=f"{what}: crossing")
    np.testing.assert_array_equal(np.asarray(hu.n_hits), np.asarray(hp.n_hits), err_msg=f"{what}: encounters")
    np.testing.assert_array_equal(np.asarray(hu.dlog_rho), np.asarray(hp.dlog_w), err_msg=f"{what}: contact")
    np.testing.assert_array_equal(np.asarray(hu.exhausted), np.asarray(hp.exhausted), err_msg=f"{what}: exhausted")


CELLS = {"sparse": lambda: (OUTER, CENTRES, L), "dense": _dense_cell, "gap": _gap_cell}


@pytest.mark.parametrize("kappa", [4e-3, 4e-2], ids=["kappa_moderate", "kappa_large"])
def test_a_plain_wall_is_the_plain_packings_wall_step_for_step(kappa):
    """6,000 paired steps over three cells of plain cylinders (sparse, dense with a 2 nm gap, two walls 0.05 R apart),
    the starts within two steps of a wall on both sides and uniform in the cell: the end position, the crossing, the
    encounters (reflections plus crossings) and the contact are bit-identical, with the derived budget (compiled) and at
    every prefix of the loop (a budget of one, two and three encounters, each primitive evaluated on its own), each
    prefix with its own uniforms. At the large
    permeability some steps reject their first encounter and transmit at a later one, which a one-trial-per-step
    rule cannot do."""
    n_steps = n_later = n_cross = n_reject = 0
    for i, (name, make) in enumerate(CELLS.items()):
        radii, centres, Lc = make()
        plain = np.ones(len(radii), bool)
        w_p, w_u, hp, hu, pool = _pair(radii, centres, Lc, kappa, plain, 2000, seed=10 + i)
        assert w_u.bounce_loop.budget == w_p.bounce_loop.budget, f"{name}: the plain cell takes the plain packing's budget"
        assert w_u.n_cand == w_p.n_cand
        _assert_pairs_equal(hp, hu, pool, name)
        n_steps += len(pool)
        crossed = np.asarray(hp.crossed); n_cross += int(crossed.sum())
        for b in (1, 2, 3):
            with jax.disable_jit():                                      # each primitive as IEEE arithmetic: a loop of
                _, _, hp_b, hu_b, pool_b = _pair(radii, centres, Lc, kappa, plain, 2000, seed=10 + i, budget=b)
            # one iteration compiled whole lets the compiler fuse the two kernels' arithmetic differently (1 ulp)
            _assert_pairs_equal(hp_b, hu_b, pool_b, f"{name}, budget {b}")
        # the first encounter's own decision, on the full walk's first uniform: the loop stopped after one encounter
        with jax.disable_jit():
            _, _, _, hu_1, pool_1 = _pair(radii, centres, Lc, kappa, plain, 2000, seed=10 + i, budget=1,
                                          coins=w_p.bounce_loop.budget)
        hit1 = np.asarray(hu_1.n_hits) > 0
        first_in = np.asarray(hu_1.pool != pool_1)
        n_reject += int((hit1 & ~first_in).sum())
        n_later += int((hit1 & ~first_in & crossed).sum())          # rejected first, crossed after it
    assert n_steps >= 6000
    assert n_cross > 0 and n_reject > 0
    if kappa == 4e-2:
        assert n_later > 0, "no step transmitted at a later encounter after rejecting its first"


def test_a_plain_wall_beside_a_myelinated_cylinder_is_the_plain_packings_wall():
    """In a cell of myelinated and plain cylinders, a walker outside the sheaths meets a myelinated cylinder's
    outer wall, impermeable, as a reflecting disk: the steps that meet it and a plain wall, in either order, are the
    plain packing's steps on the same disks (the myelinated ones at zero permeability), bit for bit."""
    radii, centres, Lc = _dense_cell()
    plain = np.array([True, False, True, False, True, False])
    total = 0
    for kappa, seed in ((4e-3, 21), (4e-2, 22)):
        w_p, w_u, hp, hu, pool = _pair(radii, centres, Lc, kappa, plain, 3000, seed=seed, sheath=0.7,
                                       budget=None)
        assert w_u.bounce_loop.budget >= w_p.bounce_loop.budget
        _, _, hp, hu, pool = _pair(radii, centres, Lc, kappa, plain, 3000, seed=seed, sheath=0.7,
                                   budget=w_u.bounce_loop.budget)
        _assert_pairs_equal(hp, hu, pool, f"mixed, kappa {kappa}")
        total += len(pool)
        assert np.asarray(hp.crossed).any()
    assert total >= 4000


# ── the axial component of a step ───────────────────────────────────────────────────────────────────────────────────

DT_AXIAL = 1e-5


def _move(geo):
    from dmipy_sim.engine.physics import make_myelin_substep
    geo.count_walls = True
    try:
        sub = make_myelin_substep(geo, DT_AXIAL, rho_weights=jnp.full((geo.N_max, 4), 1e5, jnp.float32))
    finally:
        geo.count_walls = False
    return sub, jax.jit(jax.vmap(sub.move))


def test_negating_the_axial_step_changes_nothing_in_plane_and_negates_the_axial_displacement():
    """6,000 paired single steps on a cell of myelinated and plain cylinders, every pool started in, a diffusing
    sheath: the step ``(x, y, z)`` and ``(x, y, -z)`` give the same in-plane end, pool, contact channels, surface
    weight and encounters, bit for bit, and each moves the axial coordinate by its own ``z``. The control shears the
    in-plane step by the axial component; the same comparison then fails on most pairs."""
    geo = _cell([0.7, 1.0, 0.7, 1.0], compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=D), extra=Pool(D=D)),
                kappa_inner=4e-3, kappa_outer=4e-3)
    sub, move = _move(geo)
    n = 6000
    rng = np.random.default_rng(3)
    r = np.zeros((n, 3), np.float32)
    r[:, :2] = rng.uniform(-L / 2, L / 2, (n, 2))
    r[: n // 2, 2] = rng.normal(0.0, 1e-6, n // 2)
    comp = jnp.asarray(geo.classify_positions_exact(r)).astype(jnp.int32)
    h = float(np.sqrt(6 * D * DT_AXIAL))                                # the longest step the kernel is built for
    th = rng.uniform(0, 2 * np.pi, n)
    s = np.stack([np.cos(th), np.sin(th), rng.choice([-1.0, 1.0], n)], 1) * (h * np.sqrt(rng.uniform(0, 0.5, (n, 1))))
    s = s.astype(np.float32)
    u = jnp.asarray(rng.uniform(0, 1, n), jnp.float32)
    us = jnp.asarray(rng.uniform(0, 1, (n, sub.bounce_loop.budget)), jnp.float32)
    flip = np.array([1, 1, -1], np.float32)

    def run(steps):
        return move(jnp.asarray(r), jnp.asarray(steps), comp, u, us)

    a, b = run(s), run(s * flip)
    pools = np.asarray(geo.pool_of(comp))
    assert set(np.unique(pools)) == {0, 1, 2} and (np.asarray(a[4][0]) > 0).sum() > 500   # every pool; walls met
    np.testing.assert_array_equal(np.asarray(a[0])[:, :2], np.asarray(b[0])[:, :2])
    for x, y in zip(a[1:4], b[1:4]):                                     # pool, contact channels, surface weight
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(a[4], b[4]):                                         # encounters, crossings, exhaustion
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    np.testing.assert_array_equal(np.asarray(a[0])[:, 2], r[:, 2] + s[:, 2])
    np.testing.assert_array_equal(np.asarray(b[0])[:, 2], r[:, 2] + (s * flip)[:, 2])
    z0 = slice(n // 2, n)                                                # from z = 0 the displacement negates exactly
    np.testing.assert_array_equal(np.asarray(a[0])[z0, 2], -np.asarray(b[0])[z0, 2])

    def sheared(steps):
        out = steps.copy()
        out[:, 0] += 0.5 * steps[:, 2]
        return out

    c, e = run(sheared(s)), run(sheared(s * flip))
    differs = np.any(np.asarray(c[0])[:, :2] != np.asarray(e[0])[:, :2], axis=1)
    assert differs.mean() > 0.9, "the control (a sheared in-plane step) must fail the comparison"


# ── the walk accumulates the continuous position ────────────────────────────────────────────────────────────────────

def test_a_walker_crossing_the_cell_boundary_keeps_the_displacement_of_its_steps():
    """The step folds the in-plane position into the cell; the walk accumulates the continuous one. Every recorded
    block increment is a short continuous move (never a jump across the cell), the displacement is their sum, and
    walkers that crossed the boundary have a displacement the folded difference of their in-cell positions misses."""
    from dmipy_sim.engine.core import _walk_geometry
    geo = _cell([0.7, 1.0, 0.7, 1.0], kappa_outer=[0.0, 4e-5, 0.0, 4e-5], seed_pools=("extra",))
    n = 256
    w = _walk_geometry(n, D, geo, 20e-3, 5e-5, seed=2, K=4, require_gpu=False, raw_subset=n)
    raw = np.asarray(w.raw_positions, np.float64)                        # (n, blocks + 1, 3)
    inc = np.diff(raw, axis=1)[..., :2]                                  # 0.45 um rms per axis; a fold would be ~ L
    assert np.abs(inc).max() < 0.5 * L, "a block increment jumped across the cell"
    disp = np.asarray(w.position_coeffs, np.float64)[:, 0, 1, :2]        # the walk's exact displacement
    np.testing.assert_allclose(disp, inc.sum(axis=1), rtol=0, atol=1e-10)       # float32 rounding of ~1e-12
    np.testing.assert_allclose(np.asarray(w.position_coeffs, np.float64)[:, 0, 0, :2], raw[:, 0, :2], atol=1e-10)
    fold = lambda x: (x + L / 2) % L - L / 2
    folded = fold(raw[:, -1, :2]) - fold(raw[:, 0, :2])
    assert (np.abs(raw[:, 0, :2]) <= L / 2).all()                       # the walk starts in the cell
    crossed = (np.abs(raw[:, -1, :2]) > L / 2).any(axis=1)              # and these end outside it
    assert crossed.sum() > 20, "too few walkers crossed the cell boundary for the check to resolve"
    assert (np.abs(folded[crossed] - disp[crossed]).max(axis=1) > 0.5 * L).all()


# ── seeding without the sheaths ─────────────────────────────────────────────────────────────────────────────────────

def test_no_walker_starts_in_a_sheath_when_the_sheath_is_not_seeded():
    """``seed_pools=("extra", "intra")`` places the walkers by area among the lumens and the extra pool; with the
    myelinated cylinders impermeable no walker is ever in a sheath, in the walk and in the forward signal's path."""
    from dmipy_replay import pgse
    from dmipy_sim.engine._fused import simulate_fused
    from dmipy_sim.engine.core import _walk_geometry
    geo = _cell([0.7, 1.0, 0.7, 1.0], kappa_outer=[0.0, 4e-5, 0.0, 4e-5], seed_pools=("extra", "intra"))
    r0 = np.asarray(geo.init_positions(4000, jax.random.PRNGKey(0)))
    pools = np.asarray(geo.pool_of(geo._init_compartments))
    assert (pools != 2).all() and (pools == 1).sum() > 500 and (pools == 0).sum() > 500
    np.testing.assert_array_equal(np.asarray(geo.pool_of(geo.classify_positions_exact(r0))), pools)
    w = _walk_geometry(500, D, geo, 4e-3, 5e-5, seed=3, K=8, require_gpu=False, raw_subset=500)
    assert (np.asarray(w.raw_compartment) != 2).all()
    assert (np.asarray(w.raw_compartment) != np.asarray(w.raw_compartment)[:, :1]).any(), "the plain walls exchange"
    wf = pgse([[1, 0, 0]], 2e-3, 8e-3, bvalues=[1e9], n_t=50, slew_rate=np.inf)
    out = simulate_fused(500, waveform=wf, geometry=geo, seed=3, require_gpu=False, return_compartments="full")
    assert (np.asarray(out[-1]) != 2).all() and (np.asarray(out[-2]) != 2).all()
    many = PackedMyelinatedCylinders(np.full(300, 0.1e-6), 1.0, np.stack(np.meshgrid(np.arange(15), np.arange(20)), -1)
                                     .reshape(-1, 2) * 0.5e-6 - 4.75e-6, 10e-6, N_max=300, compartments=COMPS,
                                     seed_pools=("intra",))
    many.init_positions(4096, jax.random.PRNGKey(1))                     # 300 equal lumens: whole counts, none negative
    assert (np.asarray(many.pool_of(many._init_compartments)) == 1).all() and len(many._init_compartments) == 4096
    default = _cell([0.7, 1.0, 0.7, 1.0])                                # the default seeds every pool by area
    default.init_positions(4000, jax.random.PRNGKey(0))
    assert (np.asarray(default.pool_of(default._init_compartments)) == 2).sum() > 100


def test_a_position_is_classified_into_the_cylinder_whose_wall_holds_it():
    """Cylinders of unequal radii: a point inside a small cylinder can lie nearer the centre of a larger neighbour;
    the label is the cylinder whose outer wall holds the point (the wall kernel's rule), never the nearest centre."""
    radii = np.array([1.0e-6, 0.8e-6, 1.2e-6, 0.9e-6])
    c, Lc, _ = pack_cylinders(radii, target_vf=0.45, seed=3)
    for g in (np.ones(4), np.full(4, 0.7)):
        geo = PackedMyelinatedCylinders(radii * g, g, c, Lc, N_max=4, compartments=COMPS)
        p = np.zeros((200_000, 3), np.float32)
        p[:, :2] = np.random.default_rng(0).uniform(-Lc / 2, Lc / 2, (200_000, 2))
        q = p[:, None, :2] - c[None]
        q -= Lc * np.round(q / Lc)
        inside = np.hypot(q[..., 0], q[..., 1]) < radii
        truth = np.where(inside.any(1), np.argmax(inside, 1) + 1, 0)
        got = np.asarray(geo.classify_positions_exact(p))
        lumen_or_sheath = np.where(got > 4, got - 4, got)                   # the cylinder, either pool
        near_wall = (np.abs(np.hypot(q[..., 0], q[..., 1]) - radii) < 1e-12).any(1)
        assert (lumen_or_sheath[~near_wall] == truth[~near_wall]).all()

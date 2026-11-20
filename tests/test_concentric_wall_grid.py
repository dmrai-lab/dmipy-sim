"""The wall search of a packed cell of myelinated and plain cylinders reads a grid and returns the dense search's
candidates exactly.

:func:`~dmipy_sim.geometry.myelin.concentric_wall_kernel` searches the cylinders twice per step: at the start, for
the nearest cylinder ``k0`` and the ``n_cand`` candidates ``idx`` (``top_k`` order), and at the end, for the owner
``k_new``. ``search="grid"`` reads the list of the grid cell holding the position, ``search="dense"`` every slot;
``wall.nearest`` and ``wall.owner`` are those two searches, the ones ``wall`` calls.

The cells are the repository's own packings: ``densify_packing`` at volume fraction 0.6 with a gap floor of 0.03 of
the mean radius, outer radii Gamma-distributed (mean 1 um, coefficient of variation 0.25, ``default_rng(0)``),
placement seed 0, 30 % of the cylinders plain. A plain cylinder is marked as ``main`` marks it: g-ratio 1 (its inner
radius equals its outer radius); the others have g-ratio 0.7.

1. The integers ``idx``, ``k0`` and ``k_new`` of the grid equal the dense search's at 10^5 uniformly drawn
   positions plus positions on the grid's cell edges and corners (and one float32 ulp either side) and on the
   periodic seam (and beyond it by up to a step, where a step's end lies), at 150, 1 000 and 10 000 cylinders and at
   two steps: a short one (``step_max`` 0.07 of the mean radius) and a long one (``step_max`` the smallest lumen
   radius).
2. A planted fault -- the inclusion margin ``2 (h + pad)`` set to 0, so a cell lists only the cylinders of wall
   distance at most ``D`` at its centre -- fails 1 at 10 000 cylinders.
3. The search rule: the grid is chosen at all three sizes, at 10 000 cylinders with ``M < N / 10``; the dense search
   without a periodic cell; a forced grid that cannot be built is refused by name.
4. A walk is bit-identical under either search (positions, pools, the boundary channels, the stored coefficients,
   the occupancy events, the counts and the trial record), in a permeable mixed cell at 150 and 1 000 cylinders.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry import PackedMyelinatedCylinders
from dmipy_sim.geometry import myelin
from dmipy_sim.geometry.densify import densify_packing
from dmipy_sim.geometry.myelin import concentric_wall_kernel
from dmipy_sim.trial_record import TrialRecordSpec

f32 = np.float32
R_MEAN = 1e-6
D = 2e-9
COMPS = Compartments(intra=Pool(D=D), myelin=Pool(D=1e-9), extra=Pool(D=D))
N_UNIFORM = 100_000
CHUNK = 2048


@functools.lru_cache(maxsize=None)
def _packing(n):
    """``(outer, g, centres, L)`` of the ``n``-cylinder cell (module docstring)."""
    rng = np.random.default_rng(0)
    outer = rng.gamma(16.0, R_MEAN / 16.0, n)
    (centres, L), = densify_packing(outer, (0.6,), 2, seed=0, min_gap=0.03 * R_MEAN)
    g = np.full(n, 0.7)
    g[rng.permutation(n)[:int(round(0.3 * n))]] = 1.0
    return outer, g, centres, L


def _cell(n, permeable=False):
    outer, g, centres, L = _packing(n)
    kw = dict(kappa_inner=1e-5, kappa_outer=np.where(g == 1.0, 4e-5, 1e-5)) if permeable else {}
    return PackedMyelinatedCylinders(outer * g, g, centres, L, N_max=n, compartments=COMPS, **kw)


def _short_step(geo):
    return 0.07 * R_MEAN


def _long_step(geo):
    c = geo.cell_arrays()
    return float(np.min(c.inner_radii[c.outer_radii > 0]))


STEPS = {"short": _short_step, "long": _long_step}


@functools.lru_cache(maxsize=None)
def _kernels(n, step):
    """The grid and the dense kernel of the ``n``-cylinder cell at ``step`` ("short" / "long"), built from the
    geometry's own arrays and constants (its ``_eps``, ``_nudge`` and ``min_gap``, as ``make_myelin_substep`` passes
    them)."""
    geo = _cell(n)
    c = geo.cell_arrays()
    step_max = STEPS[step](geo)
    args = (jnp.asarray(c.centers), jnp.asarray(c.inner_radii), jnp.asarray(c.outer_radii), f32(c.cell_size),
            jnp.asarray(c.D_intra), jnp.asarray(c.D_myelin), jnp.asarray(c.D_extra), jnp.asarray(c.kappa_inner),
            jnp.asarray(c.kappa_outer), jnp.zeros((c.N_max, 4), f32), c.eps, c.nudge, step_max, c.min_gap)
    return (concentric_wall_kernel(*args, search="grid"), concentric_wall_kernel(*args, search="dense"),
            step_max, float(c.cell_size))


def _positions(wall, L, step_max):
    """``(n, 2)`` float32: the uniform draws, the grid's edges and corners (and one ulp either side), the seam."""
    rng = np.random.default_rng(1)
    uni = rng.uniform(-L / 2, L / 2, (N_UNIFORM, 2)).astype(f32)
    G, cs = wall.grid_report["cells_per_side"], f32(wall.grid_report["cell_size_m"])
    edges = (f32(-0.5) * f32(L) + np.arange(G + 1, dtype=f32) * cs).astype(f32)
    near = np.concatenate([edges, np.nextafter(edges, f32(np.inf)), np.nextafter(edges, f32(-np.inf))])
    pick = rng.choice(near.size, size=min(near.size, 400), replace=False)
    ex, ey = np.meshgrid(near[pick], near[pick][::-1])                      # corners of the cells
    corners = np.stack([ex.ravel(), ey.ravel()], 1)
    free = rng.uniform(-L / 2, L / 2, near.size).astype(f32)
    on_edges = np.concatenate([np.stack([near, free], 1), np.stack([free, near], 1)])
    half = f32(0.5) * f32(L)
    seam_x = np.array([half, -half, np.nextafter(half, f32(0)), np.nextafter(-half, f32(0)),
                       np.nextafter(half, f32(np.inf)), np.nextafter(-half, f32(-np.inf))], f32)
    beyond = (half + rng.uniform(0.0, step_max, 2000)).astype(f32) * rng.choice([-1, 1], 2000).astype(f32)
    s = np.concatenate([seam_x, beyond])
    t = rng.uniform(-L / 2, L / 2, s.size).astype(f32)
    sx, sy = np.meshgrid(seam_x, seam_x)
    seam = np.concatenate([np.stack([s, t], 1), np.stack([t, s], 1), np.stack([sx.ravel(), sy.ravel()], 1)])
    return np.concatenate([uni, on_edges, corners, seam]).astype(f32)


def _search(wall, xy):
    """``(k0, idx, k_new)`` of ``wall``'s two searches at every row of ``xy``, as numpy integers."""
    fn = jax.jit(jax.vmap(lambda p: (*wall.nearest(p), wall.owner(p))))
    outs = [fn(jnp.asarray(xy[i:i + CHUNK])) for i in range(0, len(xy), CHUNK)]
    return tuple(np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(3))


@functools.lru_cache(maxsize=None)
def _xy(n, step):
    """The test positions of the ``n``-cylinder cell at ``step``, on the edges of the grid the rule chose."""
    grid, _, step_max, L = _kernels(n, step)
    return _positions(grid, L, step_max)


@functools.lru_cache(maxsize=None)
def _dense_ints(n, step):
    """The dense search's integers at :func:`_xy` (computed once per module: the 10 000-cylinder case is the costly
    one, read by tests 1 and 2)."""
    return _search(_kernels(n, step)[1], _xy(n, step))


def _disagreements(grid, n, step):
    """Per integer, the number of positions of :func:`_xy` where ``grid``'s searches differ from the dense ones."""
    out = {}
    for name, a, b in zip(("k0", "idx", "k_new"), _search(grid, _xy(n, step)), _dense_ints(n, step)):
        diff = a != b
        out[name] = int(np.sum(diff.any(-1) if diff.ndim > 1 else diff))
    return out


# ── 1. the integers ──────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("step", ["short", "long"])
@pytest.mark.parametrize("n", [150, 1000, 10_000])
def test_the_grid_returns_the_dense_searchs_integers(n, step):
    """``idx`` (in ``top_k``'s order), ``k0`` and ``k_new`` are the same integers under both searches at every
    position. The short step's ``step_max`` is 0.07 um (``n_cand`` 8 at every size); the long step's is the smallest
    lumen radius (``n_cand`` 9)."""
    grid, dense, step_max, L = _kernels(n, step)
    assert grid.search == "grid" and dense.search == "dense"
    assert grid.n_cand == dense.n_cand == {"short": 8, "long": 9}[step]
    assert len(_xy(n, step)) > N_UNIFORM
    assert _disagreements(grid, n, step) == {"k0": 0, "idx": 0, "k_new": 0}


# ── 2. the planted fault ─────────────────────────────────────────────────────────────────────────────────────────────

def test_the_inclusion_margin_is_needed(monkeypatch):
    """With the margin ``2 (h + pad)`` removed (a cell lists only the cylinders whose wall distance at its centre is
    at most the ``n_cand``-th smallest there), the candidates disagree at some of the same positions: the check of
    test 1 resolves the margin."""
    step_max = _kernels(10_000, "short")[2]
    c = _cell(10_000).cell_arrays()
    monkeypatch.setattr(myelin, "_grid_reach", lambda half_diagonal, pad: 0.0)
    faulty = concentric_wall_kernel(jnp.asarray(c.centers), jnp.asarray(c.inner_radii), jnp.asarray(c.outer_radii),
                                    f32(c.cell_size), jnp.asarray(c.D_intra), jnp.asarray(c.D_myelin),
                                    jnp.asarray(c.D_extra), jnp.asarray(c.kappa_inner), jnp.asarray(c.kappa_outer),
                                    jnp.zeros((c.N_max, 4), f32), c.eps, c.nudge, step_max, c.min_gap, search="grid")
    bad = _disagreements(faulty, 10_000, "short")
    assert bad["idx"] > 0, bad


# ── 3. the rule ──────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [150, 1000, 10_000])
def test_the_rule_chooses_the_grid_and_reports_it(n):
    """By default (``search=None``) the packed cell's kernel runs the grid; its report states the cells per side,
    the list length ``M`` (below the slot count; below a tenth of it at 10 000 cylinders) and the bytes."""
    geo = _cell(n)
    from dmipy_sim.engine.physics import make_myelin_substep
    sub = make_myelin_substep(geo, 1e-6)
    assert sub.wall_search == "grid"
    rep = sub.grid_report
    assert rep["list_length"] < n and rep["bytes"] <= rep["byte_cap"]
    assert rep["bytes"] == 4 * rep["cells_per_side"] ** 2 * rep["list_length"]
    assert min(o["list_length"] for o in rep["options"] if o["bytes"] <= rep["byte_cap"]) == rep["list_length"]
    if n == 10_000:
        assert rep["list_length"] < n / 10
    geo.wall_search = "dense"
    sub = make_myelin_substep(geo, 1e-6)
    assert sub.wall_search == "dense" and sub.grid_report is None


def test_without_a_periodic_cell_the_search_is_dense_and_a_forced_grid_is_refused():
    """No periodic cell (``L`` None, the isolated cylinder's kernel) is the dense search; ``search="grid"`` there, or
    where fewer cylinders are real than candidates, is refused naming why; an unknown search is refused."""
    c = _cell(150).cell_arrays()
    a = [jnp.asarray(c.centers), jnp.asarray(c.inner_radii), jnp.asarray(c.outer_radii), None, jnp.asarray(c.D_intra),
         jnp.asarray(c.D_myelin), jnp.asarray(c.D_extra), jnp.asarray(c.kappa_inner), jnp.asarray(c.kappa_outer),
         jnp.zeros((c.N_max, 4), f32), c.eps, c.nudge, 0.07 * R_MEAN, c.min_gap]
    w = concentric_wall_kernel(*a)
    assert w.search == "dense" and w.grid_report is None and "periodic" in w.search_reason
    with pytest.raises(ValueError, match="L is None"):
        concentric_wall_kernel(*a, search="grid")
    a[3] = f32(c.cell_size)
    with pytest.raises(ValueError, match="search is None"):
        concentric_wall_kernel(*a, search="nearest")
    padded = list(a)
    outer = np.array(c.outer_radii)
    outer[5:] = 0.0                                                         # five real cylinders, n_cand 8
    padded[2] = jnp.asarray(outer)
    w = concentric_wall_kernel(*padded)
    assert w.search == "dense" and "padding" in w.search_reason
    with pytest.raises(ValueError, match="padding"):
        concentric_wall_kernel(*padded, search="grid")


# ── 4. the walk ──────────────────────────────────────────────────────────────────────────────────────────────────────

T = 2e-3
BLOCK = 2.5e-4
SUB_STEPS = 40                       # 320 steps of 6.25 us (the derived rule would take 6 715 per block here)
SPEC = TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, T))
ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "qv_positions", "qv_boundary",
          "raw_positions", "raw_boundary", "raw_compartment")


def _walk(n, search):
    geo = _cell(n, permeable=True)
    geo.wall_search = search
    geo.count_walls = True
    return _walk_geometry(64, D, geo, T, BLOCK, seed=5, K=8, sub_steps=SUB_STEPS, raw_subset=64, require_gpu=False,
                          record_trials=SPEC)


@pytest.mark.parametrize("n", [150, 1000])
def test_a_walk_is_bit_identical_under_either_search(n):
    """64 walkers of the permeable mixed cell (plain walls 4e-5 m/s, sheaths 1e-5 m/s on both walls), 320 steps of
    6.25 us (a step of 0.27 um rms, beyond the smallest lumen radius, so multi-bounce steps are common):
    every array of the walk, the occupancy events, the work record (counts included) and the trial record are the
    same bits under ``wall_search="grid"`` and ``"dense"``."""
    a, b = _walk(n, "grid"), _walk(n, "dense")
    assert a.work["walker_steps"] == 64 * 8 * SUB_STEPS
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert a.comp_events is not None and len(a.comp_events.t) > 0
    for f in ("offsets", "t", "label"):
        np.testing.assert_array_equal(getattr(a.comp_events, f), getattr(b.comp_events, f), err_msg=f)
    wa = {k: v for k, v in a.work.items() if k != "block"}
    wb = {k: v for k, v in b.work.items() if k != "block"}
    assert wa == wb
    assert wa["n_crossings"] > 0
    assert int(a.trial_record.n.sum()) > 0
    for f in ("n", "N", "S"):
        np.testing.assert_array_equal(getattr(a.trial_record, f), getattr(b.trial_record, f), err_msg=f)

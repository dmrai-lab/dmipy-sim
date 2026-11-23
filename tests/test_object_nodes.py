"""A permeability node per (walker, object) (``_walk_geometry(object_nodes=)``): a walker under the mask meets object
``k``'s permeable wall at ``factors[node(seed, w, k)]``, the node a keyed hash of the seed, the walker's stable index
``w = first_walker + row`` and the object's index.

1. the hash is the stated uint64 rule bit for bit, with JAX's 64-bit mode off and on, and gives the golden values;
2. the node index is ``searchsorted`` of the uniform on the cumulative fractions, an integer comparison exact at
   every threshold;
3. the nodes of many (walker, object) pairs have the declared law;
4. two objects of one walker draw independent nodes;
5. at kernel level, each trial reads its node's threshold by the cell's factor rule, the sheathed walls as stated;
6. a walk cut into pieces at another ``first_walker`` draws the same nodes, bit for bit;
7. one node is the per-walker factor, and nodes of factor 1 are the walk without either keyword;
8. the walkers outside the mask walk as without the keyword;
9. what the walk cannot take is refused by name.

The cells, the walk and the kernels' threshold search are those of ``tests/test_permeability_factor.py``.
"""
import math
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.engine.physics import (ObjectNodes, object_node_bits, object_node_index, object_node_keys,
                                      object_node_table, object_node_thresholds)
from dmipy_sim.geometry import Cylinder, PackedCylinders, pack_cylinders
from dmipy_sim.geometry._boundary import realised_acceptance
from dmipy_sim.geometry.base import PermeabilitySchedule
from dmipy_sim.geometry.packing import periodic_min_gap
from dmipy_sim.geometry.myelin import OBJECT_NODES_RULE as MIXED_NODES_RULE, concentric_wall_kernel
from dmipy_sim.geometry.packed import OBJECT_NODES_RULE as PACKED_NODES_RULE, packed_wall_kernel
from dmipy_sim.math.step_law import LONGEST_STEP
from dmipy_sim.trial_record import TrialRecordSpec

from tests.test_permeability_factor import (BLOCK, CENTRES, D, KAPPA, N_W, OUTER, PLAIN, RADII, SPEC, T,
                                      _assert_rows_equal, _mixed, _packed, _plain_starts, _rows, _thresholds, _walk)

FRACTIONS = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
FACTORS = np.array([0.05, 0.2, 0.4, 0.7, 1.0], np.float32)
SEED = 20261123
CELLS = {"mixed": _mixed, "packed": _packed}
_MASK64 = (1 << 64) - 1


# ── the NumPy uint64 reference of the stated rule ─────────────────────────────────────────────────────────────────

def _mix_ref(z):
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _m_ref(seed, w, k):
    """``m = z >> 11`` of the stated rule, on uint64 arrays (NumPy's uint64 arithmetic is mod 2^64)."""
    G = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = _mix_ref(np.asarray(seed, np.uint64))
        z = _mix_ref(z + G * (np.asarray(w, np.uint64) + np.uint64(1)))
        z = _mix_ref(z + G * (np.asarray(k, np.uint64) + np.uint64(1)))
    return z >> np.uint64(11)


def _prefix_u(seed, w):
    """``u`` after the seed and the walker only (the first golden value), in Python integers."""
    def mix(z):
        z &= _MASK64
        z ^= z >> 30; z = (z * 0xBF58476D1CE4E5B9) & _MASK64
        z ^= z >> 27; z = (z * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)
    z = mix(seed)
    z = mix(z + 0x9E3779B97F4A7C15 * (w + 1))
    return (z >> 11) * 2.0 ** -53


def _keys(seed, w):
    """``(n, 4)`` uint32 node keys of uint64 seeds and walker indices."""
    seed, w = np.asarray(seed, np.uint64), np.asarray(w, np.uint64)
    lo32 = np.uint64(0xFFFFFFFF)
    return np.stack([(seed >> np.uint64(32)), seed & lo32, (w >> np.uint64(32)), w & lo32], 1).astype(np.uint32)


def _device_m(keys, k):
    h, l = jax.jit(jax.vmap(object_node_bits))(jnp.asarray(keys), jnp.asarray(np.asarray(k, np.uint32)))
    return (np.asarray(h).astype(np.uint64) << np.uint64(32)) | np.asarray(l).astype(np.uint64)


def _triples(n, rng):
    seed = (rng.integers(0, 2 ** 32, n, dtype=np.uint64) << np.uint64(32)) | rng.integers(0, 2 ** 32, n, dtype=np.uint64)
    w = rng.integers(0, 2 ** 40, n, dtype=np.uint64)
    k = rng.integers(0, 2 ** 31, n, dtype=np.uint64)
    edge = np.array([0, 1, 2 ** 32 - 1, 2 ** 32, 2 ** 63, 2 ** 64 - 1], np.uint64)
    seed[:6], w[:6], k[:6] = edge, edge[::-1], np.minimum(edge, np.uint64(2 ** 32 - 1))
    return seed, w, k


def _hash_check():
    """The device hash against the reference on 10^5 triples and the golden values; run in a subprocess per mode."""
    rng = np.random.default_rng(619)
    seed, w, k = _triples(100_000, rng)
    got, want = _device_m(_keys(seed, w), k), _m_ref(seed, w, k)
    assert got.dtype == want.dtype == np.uint64
    assert np.array_equal(got, want), int(np.sum(got != want))
    assert _prefix_u(0, 0) == 0.8833108082136426
    m = _device_m(_keys(np.array([12345], np.uint64), np.array([7], np.uint64)), np.array([3]))
    assert float(m[0]) * 2.0 ** -53 == 0.33570001687179485
    np.testing.assert_array_equal(object_node_keys(2 ** 64 - 1, 2 ** 32 - 1, 3),
                                  _keys(np.full(3, 2 ** 64 - 1, np.uint64), np.arange(2 ** 32 - 1, 2 ** 32 + 2, dtype=np.uint64)))
    print("hash ok", bool(jax.config.jax_enable_x64))


# ── 1. the hash ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_hash_golden_and_reference():
    """The device hash on uint32 pairs equals the NumPy uint64 reference on 10^5 random triples (and the edge words),
    and gives the golden values, in two processes: JAX's 64-bit mode off and on."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"import sys; sys.path.insert(0, {root!r}); import tests.test_object_nodes as t; t._hash_check()"
    for x64 in ("0", "1"):
        env = dict(os.environ, JAX_ENABLE_X64=x64, JAX_PLATFORMS="cpu")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600,
                             cwd=root)
        assert out.returncode == 0, out.stderr[-4000:]
        assert f"hash ok {x64 == '1'}" in out.stdout, out.stdout


# ── 2. the node index ─────────────────────────────────────────────────────────────────────────────────────────────

def _fraction_sets():
    t = np.exp(np.log(1.45) + (np.arange(201) + 0.5) * np.log(1160 / 1.45) / 201)
    f = t ** -0.15
    return {"K1": np.array([1.0]), "K5": FRACTIONS.copy(), "K201": f / f.sum()}


def _tau_ref(fractions):
    """``tau_k = ceil(c_k 2^53)``, ``c_k`` accumulated in float64 in order, as Python integers."""
    c, out = 0.0, []
    for x in fractions[:-1]:
        c = c + float(x)
        out.append(math.ceil(c * 2.0 ** 53))
    return out


def _index(m, tau_hi, tau_lo):
    m = np.asarray(m, np.uint64)
    return np.asarray(jax.jit(object_node_index)(jnp.asarray((m >> np.uint64(32)).astype(np.uint32)),
                                                 jnp.asarray((m & np.uint64(0xFFFFFFFF)).astype(np.uint32)),
                                                 tau_hi, tau_lo))


@pytest.mark.parametrize("name", ["K1", "K5", "K201"])
def test_node_index_is_searchsorted(name):
    """(a) the device node of 10^5 random triples is ``searchsorted(cumsum(f), u, 'right')`` with the last cumulative
    fraction ``1 + 1e-12``; (b) at every threshold, ``m = tau_k - 1`` gives node ``k`` and ``m = tau_k`` node
    ``k + 1``."""
    f = _fraction_sets()[name]
    K = f.shape[0]
    table = object_node_table(np.ones(K, np.float32), f)
    tau = _tau_ref(f)
    assert object_node_thresholds(f) == tau
    rng = np.random.default_rng(K)
    seed, w, k = _triples(100_000, rng)
    keys = _keys(seed, w)
    node = np.asarray(jax.jit(jax.vmap(lambda kk, ob: object_node_index(*object_node_bits(kk, ob), table.tau_hi,
                                                                         table.tau_lo)))(jnp.asarray(keys),
                                                                                         jnp.asarray(k.astype(np.uint32))))
    cum = np.cumsum(f)
    cum[-1] = 1.0 + 1e-12
    u = _m_ref(seed, w, k).astype(np.float64) * 2.0 ** -53
    np.testing.assert_array_equal(node, np.searchsorted(cum, u, side="right"))
    if K > 1:
        assert len(np.unique(node)) == K
    m = np.array([[t - 1, t] for t in tau], np.uint64).reshape(-1)
    assert m.size == 2 * (K - 1)
    want = np.array([[i, i + 1] for i in range(K - 1)]).reshape(-1)
    np.testing.assert_array_equal(_index(m, table.tau_hi, table.tau_lo), want)


# ── 3. the marginal law ───────────────────────────────────────────────────────────────────────────────────────────

def _nodes(walkers, objects):
    table = object_node_table(FACTORS, FRACTIONS)
    keys = jnp.asarray(object_node_keys(SEED, 0, walkers))
    ob = jnp.asarray(np.asarray(objects, np.uint32))
    f = jax.jit(jax.vmap(lambda kk: object_node_index(*object_node_bits(kk, ob), table.tau_hi, table.tau_lo)))
    return np.asarray(f(keys))


def test_marginal_law_is_the_declared_mixture():
    """Walkers 0..4095 x objects 0..63: the node counts' Pearson chi^2 against ``fractions`` is below chi^2(4)'s 0.999
    quantile; the hash is deterministic, so the counts are recorded."""
    counts = np.bincount(_nodes(4096, np.arange(64)).ravel(), minlength=5)
    expected = FRACTIONS * counts.sum()
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    np.testing.assert_array_equal(counts, [26347, 52256, 78719, 65577, 39245])
    assert chi2 < 18.467, chi2
    assert abs(chi2 - 1.4882) < 1e-4, chi2


# ── 4. two objects of one walker ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair, chi2_want, corr_want", [((0, 1), 13.7743, 0.00068), ((3, 40), 17.1935, 0.00573),
                                                        ((7, 8), 7.8937, -0.00075)])
def test_two_objects_of_one_walker_are_independent(pair, chi2_want, corr_want):
    """Walkers 0..16383 at objects ``pair``: the 5 x 5 contingency table's chi^2 (expected from its margins) is below
    chi^2(16)'s 0.999 quantile and the two factors' correlation is under ``4 / sqrt(16384)``; the values are
    recorded."""
    n = _nodes(16384, np.asarray(pair))
    table = np.zeros((5, 5))
    np.add.at(table, (n[:, 0], n[:, 1]), 1.0)
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    chi2 = float(np.sum((table - expected) ** 2 / expected))
    corr = float(np.corrcoef(FACTORS[n[:, 0]].astype(np.float64), FACTORS[n[:, 1]].astype(np.float64))[0, 1])
    assert chi2 < 39.252, chi2
    assert abs(corr) < 4.0 / np.sqrt(16384), corr
    assert abs(chi2 - chi2_want) < 1e-4 and abs(corr - corr_want) < 1e-5, (chi2, corr)


# ── 5. each trial reads its node's threshold ──────────────────────────────────────────────────────────────────────

FIRST = 32                                  # the stable walker indices start here: a batch row is not the index


def _node_factor(w, obj):
    """``factors[node(SEED, w, obj)]`` by the NumPy reference."""
    cum = np.cumsum(FRACTIONS)
    cum[-1] = 1.0 + 1e-12
    u = _m_ref(np.uint64(SEED), np.asarray(w, np.uint64), np.asarray(obj, np.uint64)).astype(np.float64) * 2.0 ** -53
    return FACTORS[np.searchsorted(cum, u, side="right")]


def _mixed_kernel(cell, k_out, Ds, nodes=None, step_max=None):
    f32 = np.float32
    return concentric_wall_kernel(jnp.asarray(cell.centers), jnp.asarray(cell.inner_radii),
                                  jnp.asarray(cell.outer_radii), f32(cell.cell_size), *(jnp.asarray(x) for x in Ds),
                                  jnp.asarray(cell.kappa_inner), jnp.asarray(k_out, f32), jnp.zeros((4, 4), f32),
                                  f32(cell.eps), f32(cell.nudge), cell.step_max(1e-6) if step_max is None else step_max,
                                  cell.min_gap, trials=True,
                                  nodes=nodes)


def test_each_trial_reads_its_nodes_threshold():
    """The mixed cell's kernel and the packed cylinders' kernel with the node table, the walkers' stable indices
    ``32 .. 32 + n - 1``:

    - every logged plain-wall trial's outcome is its word against :func:`realised_acceptance` at the walker's node
      factor at the object met, by the cell's rule (the reference hash), its overshoot bracketed by float32 rounding;
    - in the mixed cell, the threshold of each step's first plain trial, recovered exactly by the word search, is the
      threshold of the kernel without nodes whose plain ``kappa_outer`` is the rule's ``kappa / D`` at that walker's
      node (diffusivities 1);
    - a step of the mixed cell at a sheathed cylinder is the step of the kernel without nodes, bit for bit."""
    f32 = np.float32
    table = object_node_table(FACTORS, FRACTIONS)
    cell = _mixed().cell_arrays()
    step_max = cell.step_max(1e-6)
    Dw = (cell.D_intra, cell.D_myelin, cell.D_extra)
    walk_kernel = _mixed_kernel(cell, cell.kappa_outer, Dw, nodes=table)
    n = 6000
    keys = jnp.asarray(object_node_keys(SEED, FIRST, n))
    on = jnp.ones(n, bool)
    w_idx = FIRST + np.arange(n)

    # the outcome of every trial, long steps (several encounters each), at the step scale of the factor's own test
    rng = np.random.default_rng(5)
    long_max = LONGEST_STEP * float(cell.inner_radii.min()) / 6
    long_kernel = _mixed_kernel(cell, cell.kappa_outer, Dw, nodes=table, step_max=long_max)
    pts, dirs, _, pool, own, _ = (np.asarray(a) for a in _plain_starts(n, long_max, 5))
    ls = (long_max * rng.uniform(0.2, 6.0, n)).astype(f32)
    rk = jax.random.split(jax.random.PRNGKey(5), n)
    words = jax.vmap(lambda kk: jax.random.bits(kk, (long_kernel.bounce_loop.budget,), dtype=jnp.uint32))(rk)
    us = jax.vmap(lambda kk: jax.random.uniform(kk, dtype=jnp.float32))(rk)
    h = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd, nk, nn: long_kernel(p, d, l, pl, o, u, wd, None, nk, nn)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(pool), jnp.asarray(own), us, words, keys, on)
    v, g, o, uo = (np.asarray(x) for x in (h.trials.valid, h.trials.granted, h.trials.obj, h.trials.u))
    walker = np.broadcast_to(np.arange(n)[:, None], v.shape)[v]
    obj, granted, word = o[v], g[v], np.asarray(words).astype(np.uint64)[v]
    assert obj.size > 500 and granted.sum() > 50 and (~granted).sum() > 50 and v[:, 1:].sum() > 100
    fac = _node_factor(w_idx[walker], obj)
    assert len(np.unique(fac)) == 5
    kod = (cell.kappa_outer[obj] * fac).astype(f32) / f32(D)                  # the mixed rule
    d_perp = uo[v].astype(np.float64) * cell.outer_radii[obj].astype(np.float64) / 2.0
    lo = realised_acceptance(kod, (d_perp * (1 - 4e-7)).astype(f32)).astype(np.uint64)
    hi = realised_acceptance(kod, (d_perp * (1 + 4e-7)).astype(f32)).astype(np.uint64)
    assert np.all(np.where(granted, word < hi, word >= lo))
    assert np.all((word < lo) | (word >= hi))

    # the threshold of every valid first trial, exactly
    args = _plain_starts(n, step_max, 11)
    def keyed(p, d, l, pl, o, u, wd, nk):                    # the node key in the factor's operand slot
        return walk_kernel(p, d, l, pl, o, u, wd, None, nk)
    keyed.bounce_loop = walk_kernel.bounce_loop
    t_walk, valid = _thresholds(keyed, args, factor=keys)
    first_obj = np.asarray(jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd, nk, nn: walk_kernel(
        p, d, l, pl, o, u, wd, None, nk, nn).trials.obj[0]))(*args, jnp.zeros((n, walk_kernel.bounce_loop.budget),
                                                                              jnp.uint32), keys, on))
    want = np.zeros(n, np.uint64)
    fac_first = _node_factor(w_idx, first_obj)
    ones = (np.ones(4, f32),) * 3
    for f in np.unique(fac_first[valid]):
        stated = np.where(cell.plain_mask, (cell.kappa_outer * f).astype(f32) / f32(D), cell.kappa_outer)
        t_ref, v_ref = _thresholds(_mixed_kernel(cell, stated, ones), args)
        np.testing.assert_array_equal(v_ref, valid)
        sel = fac_first == f
        want[sel] = t_ref[sel]
    assert valid.sum() > 800 and len(np.unique(fac_first[valid])) == 5
    assert np.sum((t_walk[valid] > 0) & (t_walk[valid] < 2 ** 32)) > 800
    np.testing.assert_array_equal(t_walk[valid], want[valid])

    # the sheathed walls as stated: the kernel with nodes and without, at steps by the sheathed cylinders
    sheathed = np.flatnonzero(~PLAIN)
    ks = rng.choice(sheathed, n)
    off = rng.uniform(-1.0, 1.0, n) * step_max
    ang = rng.uniform(0, 2 * np.pi, n)
    sp = (CENTRES[ks] + (OUTER[ks] + off)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(f32)
    sd = np.stack([np.cos(ang + 2.0), np.sin(ang + 2.0)], 1).astype(f32)
    sl = (step_max * rng.uniform(0.2, 1.0, n)).astype(f32)
    spool = np.where(off < 0, 2, 0).astype(np.int32)
    sown = np.where(off < 0, ks, 0).astype(np.int32)
    su = rng.uniform(0, 1e-3, n).astype(f32)                                 # small: the sheathed trials grant
    plain_kernel = _mixed_kernel(cell, cell.kappa_outer, Dw)
    words = jax.vmap(lambda kk: jax.random.bits(kk, (walk_kernel.bounce_loop.budget,), dtype=jnp.uint32))(rk)
    a = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd, nk, nn: walk_kernel(p, d, l, pl, o, u, wd, None, nk, nn)))(
        *(jnp.asarray(x) for x in (sp, sd, sl, spool, sown, su)), words, keys, on)
    b = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd: plain_kernel(p, d, l, pl, o, u, wd)))(
        *(jnp.asarray(x) for x in (sp, sd, sl, spool, sown, su)), words)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    crossed_sheath = int(np.sum(np.asarray(a.pool) != spool))
    assert crossed_sheath > 50, crossed_sheath

    # the packed cylinders: every trial's outcome against its node's threshold, kappa / D per cylinder
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    r_min = float(RADII.min())
    step_p = LONGEST_STEP * r_min / 6.0
    pk = packed_wall_kernel(jnp.asarray(c, f32), jnp.asarray(RADII, f32), f32(Lc), f32(1e-7 * r_min),
                            f32(1e-4 * r_min), step_p, periodic_min_gap(c, RADII, Lc), trials=True, nodes=table)
    kp = rng.integers(0, len(RADII), n)
    offp = rng.uniform(-1.0, 1.0, n) * step_p
    angp = rng.uniform(0, 2 * np.pi, n)
    pp = (c[kp] + (RADII[kp] + offp)[:, None] * np.stack([np.cos(angp), np.sin(angp)], 1)).astype(f32)
    thp = rng.uniform(0, 2 * np.pi, n)
    dp = np.stack([np.cos(thp), np.sin(thp)], 1).astype(f32)
    lp = (step_p * rng.uniform(0.2, 6.0, n)).astype(f32)
    inside = offp < 0
    kod_obj = np.asarray(KAPPA / D, f32)                                       # float32(kappa / D), per cylinder
    hp = jax.jit(jax.vmap(lambda p, d, l, i0, kk, nk, nn: pk(p, d, l, i0, jnp.asarray(kod_obj), f32(0.0), kk, nk, nn)))(
        jnp.asarray(pp), jnp.asarray(dp), jnp.asarray(lp), jnp.asarray(inside), rk, keys, on)
    wp = np.asarray(jax.vmap(lambda kk: jax.random.bits(kk, (pk.bounce_loop.budget,), dtype=jnp.uint32))(rk))
    v, g, o, uo = (np.asarray(x) for x in (hp.trials.valid, hp.trials.granted, hp.trials.obj, hp.trials.u))
    walker = np.broadcast_to(np.arange(n)[:, None], v.shape)[v]
    obj, granted, word = o[v], g[v], wp.astype(np.uint64)[v]
    obj_perm = KAPPA[obj] > 0
    assert obj_perm.sum() > 500 and granted.sum() > 50 and (~granted[obj_perm]).sum() > 50
    kod = (kod_obj[obj] * _node_factor(w_idx[walker], obj)).astype(f32)        # the packed rule
    d_perp = uo[v].astype(np.float64) * RADII[obj].astype(f32).astype(np.float64) / 2.0
    lo = realised_acceptance(kod, (d_perp * (1 - 4e-7)).astype(f32)).astype(np.uint64)
    hi = realised_acceptance(kod, (d_perp * (1 + 4e-7)).astype(f32)).astype(np.uint64)
    assert np.all(np.where(granted, word < hi, word >= lo))
    assert np.all((word < lo) | (word >= hi))


# ── 6. pieces ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _nodes_of(n, walkers=None, factors=FACTORS, fractions=FRACTIONS, seed=SEED):
    return ObjectNodes(np.asarray(factors, np.float32), np.asarray(fractions, np.float64), seed,
                       np.ones(n, bool) if walkers is None else walkers)


@pytest.mark.parametrize("cell", ["mixed", "packed"])
def test_pieces_give_the_same_nodes(cell):
    """64 walkers as one walk and as two pieces (``first_walker`` 0 and 32), joined: equal positions (bands, raw
    subset) and an equal trial record (``n``, ``N``, ``S``), bit for bit; the joined walk states the object nodes of
    the whole."""
    from dmipy_sim.coefficient_walk import join_walks
    geo = CELLS[cell]()
    spec = TrialRecordSpec(modes=3, band_edges=SPEC.band_edges, slots=48)
    kw = dict(seed=3, tiers="all", require_gpu=False, K=8, record_trials=spec, events_capacity=64)
    whole = _walk_geometry(64, D, geo, T, BLOCK, raw_subset=64, object_nodes=_nodes_of(64), **kw)
    a = _walk_geometry(32, D, geo, T, BLOCK, raw_subset=32, object_nodes=_nodes_of(32), **kw)
    b = _walk_geometry(32, D, geo, T, BLOCK, first_walker=32, object_nodes=_nodes_of(32), **kw)
    joined = join_walks([a, b])
    np.testing.assert_array_equal(joined.position_coeffs, whole.position_coeffs)
    np.testing.assert_array_equal(a.raw_positions, np.asarray(whole.raw_positions)[:32])
    tw, tj = whole.trial_record.to_dense(), joined.trial_record.to_dense()
    for name in ("n", "N", "S"):
        np.testing.assert_array_equal(getattr(tj, name), getattr(tw, name), err_msg=name)
    assert int(tw.N.sum()) > 0
    assert joined.work["object_nodes"] == whole.work["object_nodes"]
    np.testing.assert_array_equal(joined.object_nodes.walkers, whole.object_nodes.walkers)
    # the second piece's own raw subset is the whole walk's walkers 32..63
    b_raw = _walk_geometry(32, D, geo, T, BLOCK, first_walker=32, raw_subset=32, object_nodes=_nodes_of(32), **kw)
    _assert_rows_equal(_rows(b_raw), _rows(whole, np.arange(32, 64)), "piece at first_walker 32")


def test_the_walk_returns_and_states_its_object_nodes(tmp_path):
    """The echo (float32 / float64 / int / bool) and the work record, through the walk file and a take; a walk file
    whose echo disagrees with its record is refused on load; a walk without object nodes carries neither."""
    import dataclasses
    from dmipy_sim.coefficient_walk import CoefficientWalk
    mask = np.arange(N_W) % 3 != 0
    w = _walk(_mixed(), object_nodes=_nodes_of(N_W, mask, seed=2 ** 64 - 1))
    on = w.object_nodes
    assert on.factors.dtype == np.float32 and on.fractions.dtype == np.float64 and on.walkers.dtype == np.bool_
    assert on.seed == 2 ** 64 - 1 and isinstance(on.seed, int)
    rec = w.work["object_nodes"]
    assert rec == dict(rule=MIXED_NODES_RULE, seed=2 ** 64 - 1, K=5, n=N_W, n_per_object=int(mask.sum()),
                       realised_fractions=[float(x) for x in on.realised_fractions])
    np.testing.assert_array_equal(w.object_node_index, np.arange(N_W, dtype=np.int64))
    w.save(tmp_path / "w.safetensors")
    back = CoefficientWalk.load(tmp_path / "w.safetensors")
    assert back.work["object_nodes"] == rec and back.object_nodes.seed == on.seed
    assert np.array_equal(np.asarray(back.work["object_nodes"]["realised_fractions"], np.float64), on.realised_fractions)
    assert back.object_node_index.dtype == np.int64 and np.array_equal(back.object_node_index, w.object_node_index)
    for name in ("factors", "fractions", "walkers"):
        x, y = getattr(back.object_nodes, name), getattr(on, name)
        assert x.dtype == y.dtype and np.array_equal(x, y)
    part = w.take(np.arange(0, 6))
    assert part.work["object_nodes"]["n"] == 6 and part.work["object_nodes"]["n_per_object"] == 4
    np.testing.assert_array_equal(part.object_nodes.walkers, mask[:6])
    dataclasses.replace(w, object_nodes=on._replace(walkers=mask[:5])).save(tmp_path / "broken.safetensors")
    with pytest.raises(ValueError, match="object-node mask"):
        CoefficientWalk.load(tmp_path / "broken.safetensors")
    pw = _walk(_packed(), object_nodes=_nodes_of(N_W))
    assert pw.work["object_nodes"]["rule"] == PACKED_NODES_RULE
    bare = _walk(_mixed())
    assert bare.object_nodes is None and "object_nodes" not in bare.work


# ── 7. one node is the factor ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", ["mixed", "packed"])
def test_one_node_is_the_per_walker_factor(cell):
    """K = 1 of factor 0.3 on every walker is ``permeability_factor = full(n, 0.3)`` bit for bit; K = 3 of factor 1.0 is
    the walk without either keyword."""
    geo = CELLS[cell]()
    f = np.float32(0.3)
    one = _walk(geo, object_nodes=_nodes_of(N_W, factors=[f], fractions=[1.0], seed=7))
    fac = _walk(geo, permeability_factor=np.full(N_W, f, np.float32))
    _assert_rows_equal(_rows(one), _rows(fac), f"{cell} K = 1")
    assert int(one.trial_record.N.sum()) > 0
    three = _walk(geo, object_nodes=_nodes_of(N_W, factors=[1.0, 1.0, 1.0], fractions=[0.2, 0.3, 0.5], seed=7))
    _assert_rows_equal(_rows(three), _rows(_walk(geo)), f"{cell} K = 3 of ones")


# ── 8. the other walkers ──────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", ["mixed", "packed"])
def test_the_other_walkers_are_untouched(cell):
    """``walkers`` = the second half: the first half's rows equal those of the walk without object nodes, bit for bit
    (exact as an array of ones is the walk without the factor: the threshold a walker outside the mask reads is the
    same number); the second half's differ."""
    geo = CELLS[cell]()
    half = np.arange(N_W) >= N_W // 2
    nodes = _walk(geo, object_nodes=_nodes_of(N_W, half))
    bare = _walk(geo)
    first = np.arange(N_W // 2)
    _assert_rows_equal(_rows(nodes, first), _rows(bare, first), f"{cell}: walkers outside the mask")
    second = np.arange(N_W // 2, N_W)
    assert not np.array_equal(_rows(nodes, second)["position_coeffs"], _rows(bare, second)["position_coeffs"])


def test_a_factor_beside_the_nodes_keeps_its_walkers():
    """Both keywords: the walkers outside the mask walk at their factor, those under it at their nodes; the first
    half equals the walk of the factor alone."""
    geo = _mixed()
    half = np.arange(N_W) >= N_W // 2
    fac = np.where(half, 1.0, 0.5).astype(np.float32)
    both = _walk(geo, permeability_factor=fac, object_nodes=_nodes_of(N_W, half))
    only = _walk(geo, permeability_factor=fac)
    first = np.arange(N_W // 2)
    _assert_rows_equal(_rows(both, first), _rows(only, first), "factor beside nodes")


# ── 9. refusals ───────────────────────────────────────────────────────────────────────────────────────────────────

def _refused(geo, nodes, match, exc=ValueError, n=8, **kw):
    with pytest.raises(exc, match=match):
        _walk_geometry(n, D, geo, T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, object_nodes=nodes, **kw)


def test_refusals():
    """Each refusal by its name, before any walk."""
    good = _nodes_of(8)
    _refused(Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5), good, "object_nodes_geometry")
    _refused(_mixed(0.0), good, "object_nodes_geometry")
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    _refused(PackedCylinders(RADII, c, Lc, permeability=PermeabilitySchedule([1e-3], [4e-3, 2e-3])), good,
             "object_nodes_geometry")
    _refused(_mixed(), good, "object_nodes_mt_binding", kappa_MT=1e-5, dwell_time=1e-3, equilibrate_binding="off")

    class Other:
        name = "other"
        capabilities = ()

        def refuses(self, request):
            return None

        def walk_batch(self, request, r0, keys):
            raise AssertionError("not walked")

    _refused(_mixed(), good, "object_nodes_backend", backend=Other())
    for bad, match in (
            (good._replace(factors=FACTORS.astype(np.float64)), "object_nodes_factors"),
            (good._replace(factors=FACTORS[None]), "object_nodes_factors"),
            (good._replace(factors=np.zeros(0, np.float32), fractions=np.zeros(0)), "object_nodes_factors"),
            (_nodes_of(8, factors=np.ones(4097, np.float32), fractions=np.full(4097, 1 / 4097)), "object_nodes_factors"),
            (good._replace(factors=np.r_[FACTORS[:4], 1.5].astype(np.float32)), "object_nodes_factors"),
            (good._replace(factors=np.r_[FACTORS[:4], -0.1].astype(np.float32)), "object_nodes_factors"),
            (good._replace(factors=np.r_[FACTORS[:4], np.nan].astype(np.float32)), "object_nodes_factors"),
            (good._replace(fractions=FRACTIONS.astype(np.float32)), "object_nodes_fractions"),
            (good._replace(fractions=FRACTIONS[:4] / FRACTIONS[:4].sum()), "object_nodes_fractions"),
            (good._replace(fractions=np.r_[0.0, 0.3, 0.3, 0.25, 0.15]), "object_nodes_fractions"),
            (good._replace(fractions=FRACTIONS * (1 + 2e-9)), "object_nodes_fractions"),
            (good._replace(seed=-1), "object_nodes_seed"),
            (good._replace(seed=2 ** 64), "object_nodes_seed"),
            (good._replace(seed=1.0), "object_nodes_seed"),
            (good._replace(seed=True), "object_nodes_seed"),
            (good._replace(walkers=np.ones(7, bool)), "object_nodes_walkers"),
            (good._replace(walkers=np.ones(8, np.int64)), "object_nodes_walkers")):
        _refused(_mixed(), bad, match)
    _refused(_mixed(), dict(good._asdict()), "object_nodes_type", exc=TypeError)
    _refused(_mixed(), good, "object_nodes_factor_conflict",
             permeability_factor=np.r_[np.ones(7), np.float32(0.999)].astype(np.float32))
    near_one = np.r_[np.ones(7), 1.0 - 2.0 ** -26]                           # float64: float32 rounds it to 1.0
    assert np.float32(near_one[-1]) == np.float32(1.0)
    _refused(_mixed(), good, "object_nodes_factor_conflict", permeability_factor=near_one)
    # a factor of exactly 1.0 under the mask and any factor outside it are taken
    half = np.arange(8) >= 4
    w = _walk_geometry(8, D, _mixed(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8,
                       object_nodes=_nodes_of(8, half), permeability_factor=np.where(half, 1.0, 0.25).astype(np.float32))
    assert w.work["object_nodes"]["n_per_object"] == 4
    from dmipy_sim.spec.walk import walk
    from dmipy_replay import envelope_of
    with pytest.raises(ValueError, match="object_nodes_geometry"):
        walk(_packed().spec, 8, T, BLOCK, envelope=envelope_of("prisma"), diffusivity=D, seed=3, K=8,
             require_gpu=False, adaptive_steps=True, object_nodes=good)


def test_walk_forwards_the_nodes_on_a_spec_of_one_geometry():
    """``walk`` on the packed cylinders' spec with nodes of factor 1.0 is the walk without them, and returns them."""
    from dmipy_replay import envelope_of
    from dmipy_sim.spec.walk import walk
    geo = _packed()
    kw = dict(envelope=envelope_of("prisma"), diffusivity=D, seed=3, K=8, record_trials=SPEC, require_gpu=False)
    a = walk(geo.spec, 8, T, BLOCK, **kw)
    b = walk(geo.spec, 8, T, BLOCK, object_nodes=_nodes_of(8, factors=[1.0, 1.0], fractions=[0.5, 0.5]), **kw)
    for name in ("position_coeffs", "compartment"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.trial_record.N, b.trial_record.N)
    assert b.work["object_nodes"]["K"] == 2


# ── review: the realised masses, the stable index, pieces with an empty mask, the seed, the thresholds ────────────

def _node_ref(seed, w, obj, masses):
    """The node of the stated rule from the RECORD's quantities: the seed, the stable index and the realised masses,
    whose running sums times ``2^53`` are the thresholds (NumPy reference hash)."""
    tau = np.cumsum(np.round(np.ldexp(np.asarray(masses, np.float64), 53)).astype(np.uint64))[:-1]
    m = _m_ref(np.uint64(seed), np.asarray(w, np.uint64), np.asarray(obj, np.uint64))
    return np.sum(m[..., None] >= tau, axis=-1)


@pytest.mark.parametrize("name", ["K1", "K5", "K201"])
def test_the_realised_masses_are_the_thresholds(name):
    """The masses the nodes are drawn with: the thresholds' differences times ``2^-53``, exactly; they sum to 1
    exactly. Against the input: each mass of ``i < K - 1`` is within ``2^-53`` plus the float64 rounding of the
    cumulative ``c_i``, the last within ``2^-53`` of ``1 - c_{K-2}`` (all in exact rationals); for the ticket's five
    fractions every mass is within ``2^-53`` of its fraction. The device node index reads the same thresholds."""
    from fractions import Fraction
    from dmipy_sim.engine.physics import object_node_masses, object_node_thresholds_of_masses
    f = _fraction_sets()[name]
    nodes = ObjectNodes(np.ones(len(f), np.float32), f, SEED, np.ones(1, bool))
    m = nodes.realised_fractions
    tau = object_node_thresholds(f)
    assert m.dtype == np.float64 and m.shape == f.shape
    assert float(np.sum(m)) == 1.0 and sum(Fraction(float(x)) for x in m) == 1
    edges = [0] + tau + [2 ** 53]
    assert [Fraction(float(x)) for x in m] == [Fraction(b - a, 2 ** 53) for a, b in zip(edges[:-1], edges[1:])]
    assert object_node_thresholds_of_masses(m) == tau
    np.testing.assert_array_equal(object_node_masses(tau), m)
    ulp = Fraction(1, 2 ** 53)
    c = 0.0
    cs = []
    for x in f[:-1]:
        c = c + float(x)
        cs.append(Fraction(c))
    prev = Fraction(0)
    for i in range(len(f) - 1):
        rounding = abs(cs[i] - (prev + Fraction(float(f[i]))))
        assert abs(Fraction(float(m[i])) - Fraction(float(f[i]))) < ulp + rounding, i
        prev = cs[i]
    assert abs(Fraction(float(m[-1])) - (1 - prev)) < ulp
    if name == "K5":
        assert all(abs(Fraction(float(a)) - Fraction(float(b))) < ulp for a, b in zip(m, f))
        assert m[0] == 900719925474100 / 2 ** 53 and m[0] != f[0]


def test_take_keeps_the_stable_index_and_the_record_recomputes_the_node(tmp_path):
    """``take([1])``: row 0 is walker 1, its stable index 1 is kept, and its node at object 0 recomputed from the
    returned record alone (the seed, the realised masses, the stable index) is walker 1's node, 3 for the ticket's
    seed and fractions; through the walk file too. Row 0 of the full walk has another node there."""
    from dmipy_sim.coefficient_walk import CoefficientWalk
    w = _walk(_packed(), object_nodes=_nodes_of(N_W))
    part = w.take([1])
    np.testing.assert_array_equal(part.object_node_index, [1])
    part.save(tmp_path / "part.safetensors")
    back = CoefficientWalk.load(tmp_path / "part.safetensors")
    for walk in (part, back):
        rec = walk.work["object_nodes"]
        node = _node_ref(rec["seed"], walk.object_node_index[0], 0, rec["realised_fractions"])
        assert int(node) == 3
    assert int(_node_ref(SEED, 0, 0, w.work["object_nodes"]["realised_fractions"])) != 3
    # the device's node, from the record's quantities, is the same
    from dmipy_sim.engine.physics import object_node_key_rows, object_node_thresholds_of_masses
    tau = object_node_thresholds_of_masses(part.work["object_nodes"]["realised_fractions"])
    table = object_node_table(np.ones(5, np.float32), FRACTIONS)
    assert list(table.key[1]) == tau
    k = jnp.asarray(object_node_key_rows(part.work["object_nodes"]["seed"], part.object_node_index)[0])
    assert int(table.node_of(k, jnp.asarray(0))) == 3


def test_pieces_with_an_empty_mask_join():
    """A 64-walker mask selecting only the last 32: walked in two pieces (the first piece's mask selects none of its
    walkers, a walk with object nodes all the same), joined, equals the one-piece walk -- the records and the
    positions bit for bit; a piece walked without object nodes does not join one with them."""
    from dmipy_sim.coefficient_walk import join_walks
    geo = _packed()
    mask = np.arange(64) >= 32
    spec = TrialRecordSpec(modes=3, band_edges=SPEC.band_edges, slots=48)
    kw = dict(seed=3, tiers="all", require_gpu=False, K=8, record_trials=spec, events_capacity=64)
    whole = _walk_geometry(64, D, geo, T, BLOCK, object_nodes=_nodes_of(64, mask), **kw)
    a = _walk_geometry(32, D, geo, T, BLOCK, object_nodes=_nodes_of(32, mask[:32]), **kw)
    b = _walk_geometry(32, D, geo, T, BLOCK, first_walker=32, object_nodes=_nodes_of(32, mask[32:]), **kw)
    assert a.work["object_nodes"]["n_per_object"] == 0
    joined = join_walks([a, b])
    assert joined.work["object_nodes"] == whole.work["object_nodes"]
    assert {k: v for k, v in joined.work.items() if k not in ("block",)} == \
        {k: v for k, v in whole.work.items() if k not in ("block",)}
    for name in ("position_coeffs", "boundary_bands", "compartment", "object_node_index"):
        np.testing.assert_array_equal(getattr(joined, name), getattr(whole, name), err_msg=name)
    np.testing.assert_array_equal(joined.object_nodes.walkers, mask)
    tw, tj = whole.trial_record.to_dense(), joined.trial_record.to_dense()
    for name in ("n", "N", "S"):
        np.testing.assert_array_equal(getattr(tj, name), getattr(tw, name), err_msg=name)
    bare = _walk_geometry(32, D, geo, T, BLOCK, **kw)
    with pytest.raises(ValueError, match="object_nodes: walks"):
        join_walks([bare, b])


SEED_MAX = 2 ** 64 - 1


def _seed_check(w, seed):
    """Factors ``[0, 1]``: every (walker, object) pair whose node is 0 at ``seed`` (the reference hash on the
    walker's stable index) grants nothing, and pairs whose node is 1 grant; returns the granted pairs."""
    rec = w.work["object_nodes"]
    N = w.trial_record.to_dense().N.sum(axis=2)                                 # (walker, object)
    n = w.trial_record.to_dense().n.sum(axis=2)
    walker, obj = np.nonzero(n)
    node = _node_ref(seed, np.asarray(w.object_node_index)[walker], obj, rec["realised_fractions"])
    granted = N[walker, obj] > 0
    assert not np.any(granted & (node == 0)), "a pair of node 0 (factor 0) was granted"
    assert np.sum(node == 0) > 20 and np.sum(granted & (node == 1)) > 20, (np.sum(node == 0), np.sum(granted))
    return int(np.sum(granted))


def test_the_seed_reaches_the_kernels_on_both_walk_paths():
    """Factors ``[0, 1]`` at equal fractions and the seed ``2^64 - 1``: through ``_walk_geometry`` and through
    ``walk`` on the spec, every pair the hash gives node 0 at THAT seed is never granted while pairs of node 1 are, so
    the kernels read the walk's seed (a key built with any other seed puts about half of the granted pairs on node 0
    by the stated rule)."""
    from dmipy_replay import envelope_of
    from dmipy_sim.spec.walk import walk
    geo = _packed()
    nodes = _nodes_of(N_W, factors=[0.0, 1.0], fractions=[0.5, 0.5], seed=SEED_MAX)
    a = _walk(geo, object_nodes=nodes)
    b = walk(geo.spec, N_W, T, BLOCK, envelope=envelope_of("prisma"), diffusivity=D, seed=3, K=8, record_trials=SPEC,
             require_gpu=False, object_nodes=nodes)
    assert a.work["object_nodes"]["seed"] == b.work["object_nodes"]["seed"] == SEED_MAX
    for w in (a, b):
        _seed_check(w, SEED_MAX)
    with pytest.raises(AssertionError):
        _seed_check(a, SEED)                                                   # the check tells the seeds apart


TWO = (np.array([0.1, 1.0], np.float32), np.array([0.5, 0.5]))


def _packed_nodes_kernel(n, seed_rng, factors, fractions):
    """The packed cylinders' kernel with the node table of ``factors`` and ``fractions``, long steps from starts by the walls: per logged trial
    its word, outcome, object, node (reference), and the bracket of every node's threshold."""
    f32 = np.float32
    table = object_node_table(factors, fractions)
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    r_min = float(RADII.min())
    step = LONGEST_STEP * r_min / 6.0
    pk = packed_wall_kernel(jnp.asarray(c, f32), jnp.asarray(RADII, f32), f32(Lc), f32(1e-7 * r_min),
                            f32(1e-4 * r_min), step, periodic_min_gap(c, RADII, Lc), trials=True, nodes=table)
    rng = np.random.default_rng(seed_rng)
    k = rng.integers(0, len(RADII), n)
    off = rng.uniform(-1.0, 1.0, n) * step
    ang = rng.uniform(0, 2 * np.pi, n)
    p = (c[k] + (RADII[k] + off)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(f32)
    th = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([np.cos(th), np.sin(th)], 1).astype(f32)
    ln = (step * rng.uniform(0.2, 6.0, n)).astype(f32)
    keys = jax.random.split(jax.random.PRNGKey(seed_rng), n)
    nk = jnp.asarray(object_node_keys(SEED, FIRST, n))
    kod = jnp.asarray(np.asarray(KAPPA / D, f32))
    run = jax.jit(jax.vmap(lambda a, b, l, i0, kk, key: pk(a, b, l, i0, kod, f32(0.0), kk, key)))
    h = run(jnp.asarray(p), jnp.asarray(d), jnp.asarray(ln), jnp.asarray(off < 0), keys, nk)
    words = np.asarray(jax.vmap(lambda kk: jax.random.bits(kk, (pk.bounce_loop.budget,), dtype=jnp.uint32))(keys))
    v, g, o, u = (np.asarray(x) for x in (h.trials.valid, h.trials.granted, h.trials.obj, h.trials.u))
    walker = np.broadcast_to(np.arange(n)[:, None], v.shape)[v]
    obj, granted, word = o[v], g[v], words.astype(np.uint64)[v]
    node = _node_ref(SEED, FIRST + walker, obj, object_node_masses_of(fractions))
    d_perp = u[v].astype(np.float64) * RADII[obj].astype(f32).astype(np.float64) / 2.0
    brackets = []
    for fac in factors:
        kod_n = (np.asarray(KAPPA / D, f32)[obj] * fac).astype(f32)
        lo = realised_acceptance(kod_n, (d_perp * (1 - 4e-7)).astype(f32)).astype(np.uint64)
        hi = realised_acceptance(kod_n, (d_perp * (1 + 4e-7)).astype(f32)).astype(np.uint64)
        brackets.append((lo, hi))
    return dict(granted=granted, word=word, node=node, obj=obj, brackets=brackets, h=h)


def object_node_masses_of(fractions):
    from dmipy_sim.engine.physics import object_node_masses
    return object_node_masses(object_node_thresholds(fractions))


def test_the_packed_kernel_reads_each_nodes_own_threshold():
    """Two nodes of factors 0.1 and 1 (thresholds an order of magnitude apart) on the permeable cylinders: every logged
    trial's outcome is its word against its OWN node's threshold; each node has granted and refused trials (both
    sides of its threshold populated); and for each node at least 100 trials would have the other outcome under the
    other node's threshold, so a kernel reading the wrong node's threshold fails here whatever the draws."""
    t = _packed_nodes_kernel(20000, 13, *TWO)
    perm = KAPPA[t["obj"]] > 0
    granted, word, node = t["granted"][perm], t["word"][perm], t["node"][perm]
    own_lo = np.where(node == 0, t["brackets"][0][0][perm], t["brackets"][1][0][perm])
    own_hi = np.where(node == 0, t["brackets"][0][1][perm], t["brackets"][1][1][perm])
    assert np.all(np.where(granted, word < own_hi, word >= own_lo))
    assert np.all((word < own_lo) | (word >= own_hi))
    pops = {}
    for i in (0, 1):
        sel = node == i
        lo_o, hi_o = t["brackets"][1 - i][0][perm][sel], t["brackets"][1 - i][1][perm][sel]
        wrong_under_other = np.where(granted[sel], word[sel] >= hi_o, word[sel] < lo_o)
        pops[i] = (int(granted[sel].sum()), int((~granted[sel]).sum()), int(wrong_under_other.sum()))
        assert pops[i][0] >= 100 and pops[i][1] >= 100 and pops[i][2] >= 100, pops
    print("packed two-node populations (granted, refused, wrong under the other node):", pops)


def _mixed_long(table, n=6000, seed=5):
    """The mixed cell's kernel with ``table`` at the factor test's long steps from starts by the plain walls."""
    f32 = np.float32
    cell = _mixed().cell_arrays()
    long_max = LONGEST_STEP * float(cell.inner_radii.min()) / 6
    kern = _mixed_kernel(cell, cell.kappa_outer, (cell.D_intra, cell.D_myelin, cell.D_extra), nodes=table,
                         step_max=long_max)
    rng = np.random.default_rng(seed)
    pts, dirs, _, pool, own, _ = (np.asarray(a) for a in _plain_starts(n, long_max, seed))
    ls = (long_max * rng.uniform(0.2, 6.0, n)).astype(f32)
    rk = jax.random.split(jax.random.PRNGKey(seed), n)
    words = jax.vmap(lambda kk: jax.random.bits(kk, (kern.bounce_loop.budget,), dtype=jnp.uint32))(rk)
    us = jax.vmap(lambda kk: jax.random.uniform(kk, dtype=jnp.float32))(rk)
    keys = jnp.asarray(object_node_keys(SEED, FIRST, n))
    return jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd, nk: kern(p, d, l, pl, o, u, wd, None, nk)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(pool), jnp.asarray(own), us, words, keys)


@pytest.mark.parametrize("kernel", ["mixed", "packed"])
def test_changing_one_nodes_factor_changes_only_its_trials(kernel):
    """Node 2's factor changed (0.4 to 0.9), every other the same: a step with no trial at a (walker, object) pair of
    node 2 (reference hash) is the same step bit for bit, and at least 100 steps that met node 2 change."""
    changed = FACTORS.copy()
    changed[2] = np.float32(0.9)
    if kernel == "mixed":
        a, b = _mixed_long(object_node_table(FACTORS, FRACTIONS)), _mixed_long(object_node_table(changed, FRACTIONS))
        walker_of = np.arange(a.trials.valid.shape[0])
    else:
        a = _packed_nodes_kernel(6000, 17, FACTORS, FRACTIONS)["h"]
        b = _packed_nodes_kernel(6000, 17, changed, FRACTIONS)["h"]
        walker_of = np.arange(a.trials.valid.shape[0])
    v, o = np.asarray(a.trials.valid), np.asarray(a.trials.obj)
    node = _node_ref(SEED, FIRST + walker_of[:, None], np.where(v, o, 0), object_node_masses_of(FRACTIONS))
    at_two = np.any(v & (node == 2), axis=1)
    same = np.ones(v.shape[0], bool)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        same &= np.all((x == y).reshape(x.shape[0], -1), axis=1)
    assert np.all(same[~at_two]), int(np.sum(~same[~at_two]))
    assert int(np.sum(~same[at_two])) >= 100, int(np.sum(~same[at_two]))


def test_the_sheathed_walls_read_no_node():
    """Steps by the sheathed cylinders with small uniforms (their trials grant): the kernel with nodes is the kernel
    without, bit for bit, while the kernel whose sheathed ``kappa_outer`` is scaled by a node's factor (0.05) gives at
    least 20 other outcomes -- the equality is not vacuous."""
    f32 = np.float32
    cell = _mixed().cell_arrays()
    Dw = (cell.D_intra, cell.D_myelin, cell.D_extra)
    step_max = cell.step_max(1e-6)
    with_nodes = _mixed_kernel(cell, cell.kappa_outer, Dw, nodes=object_node_table(FACTORS, FRACTIONS))
    plain = _mixed_kernel(cell, cell.kappa_outer, Dw)
    scaled = _mixed_kernel(cell, np.where(cell.plain_mask, cell.kappa_outer, cell.kappa_outer * f32(0.05)), Dw)
    n = 6000
    rng = np.random.default_rng(23)
    ks = rng.choice(np.flatnonzero(~PLAIN), n)
    off = rng.uniform(-1.0, 1.0, n) * step_max
    ang = rng.uniform(0, 2 * np.pi, n)
    sp = (CENTRES[ks] + (OUTER[ks] + off)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(f32)
    sd = np.stack([np.cos(ang + 2.0), np.sin(ang + 2.0)], 1).astype(f32)
    sl = (step_max * rng.uniform(0.2, 1.0, n)).astype(f32)
    spool = np.where(off < 0, 2, 0).astype(np.int32)
    sown = np.where(off < 0, ks, 0).astype(np.int32)
    su = rng.uniform(0, 1e-3, n).astype(f32)
    rk = jax.random.split(jax.random.PRNGKey(23), n)
    words = jax.vmap(lambda kk: jax.random.bits(kk, (with_nodes.bounce_loop.budget,), dtype=jnp.uint32))(rk)
    keys = jnp.asarray(object_node_keys(SEED, FIRST, n))
    args = tuple(jnp.asarray(x) for x in (sp, sd, sl, spool, sown, su))
    a = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd, nk: with_nodes(p, d, l, pl, o, u, wd, None, nk)))(*args, words, keys)
    b = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd: plain(p, d, l, pl, o, u, wd)))(*args, words)
    c = jax.jit(jax.vmap(lambda p, d, l, pl, o, u, wd: scaled(p, d, l, pl, o, u, wd)))(*args, words)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    crossed_a = np.asarray(a.pool) != spool
    crossed_c = np.asarray(c.pool) != spool
    assert crossed_a.sum() > 50 and int(np.sum(crossed_a != crossed_c)) >= 20, (crossed_a.sum(), np.sum(crossed_a != crossed_c))


def test_refuses_reads_the_nodes_the_batches_carry():
    """A backend declaring ``OBJECT_NODES`` is asked ``refuses`` with the whole walk's object nodes and stable indices,
    which its batch requests then carry row by row."""
    from dmipy_sim.engine import backends
    seen = {}

    class Rec:
        name = "rec"
        capabilities = backends.JAX_CAPABILITIES

        def refuses(self, request):
            seen["nodes"], seen["index"] = request.object_nodes, request.object_node_index
            return "stop here"

        def walk_batch(self, request, r0, keys):
            raise AssertionError("not walked")

    mask = np.arange(8) % 2 == 0
    with pytest.raises(ValueError, match="stop here"):
        _walk_geometry(8, D, _packed(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, first_walker=5,
                       object_nodes=_nodes_of(8, mask), backend=Rec())
    assert seen["nodes"] is not None and int(seen["nodes"].seed) == SEED
    np.testing.assert_array_equal(seen["nodes"].walkers, mask)
    np.testing.assert_array_equal(seen["index"], 5 + np.arange(8))

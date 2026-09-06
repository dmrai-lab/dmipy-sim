"""The myelinated substrates walk the same wall physics as every other geometry.

`geometry.myelin.concentric_wall_kernel` is the one wall interaction behind `MyelinatedCylinder`
and `PackedMyelinatedCylinders`. The fast tests here are impact tables on that kernel -- the
adversarial hits (grazing, spanning, in a gap narrower than the step) for each of the three pools
-- and the worst-case derivation of its bounce budget. The slow tests are the consequences on
signals: exchange through the axon membrane matches `PermeableShell`, the extra-axonal walk of a
bare pack matches `PackedCylinders`, the boundary local time on the lumen wall is the
Brownstein-Tarr `S/V` of a `Cylinder`, and per-axon properties are per axon.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry.myelin import concentric_wall_kernel

D = 2e-9
R = 1e-6                       # lumen radius of the impact-table axon
G = 1.4                        # its sheath boundary, in units of R
OFFSETS = [("1e-3R", 1e-3), ("1e-2R", 1e-2), ("0.1R", 0.1), ("0.3R", 0.3)]
DISTANCES = [("0.05R", 0.05), ("0.3R", 0.3), ("1R", 1.0), ("3R", 3.0)]
ANGLES = [("head-on", 0.0), ("30deg", 30.0), ("60deg", 60.0), ("85deg", 85.0), ("89.9deg", 89.9)]


def _pack(inner_radii, g, packing, seed):
    """A non-overlapping pack of sheaths at outer packing fraction `packing`: (centers, L)."""
    inner_radii = np.asarray(inner_radii, float)
    L = float(np.sqrt(np.pi * np.sum((inner_radii / g) ** 2) / packing))
    _, _, c = d.pack_myelinated_cylinders(inner_radii, g, None, cell_size=L, seed=seed)
    return c, L


def _kernel(centers, inner, outer, L, kappa_inner=0.0, step_max=3 * R, min_gap=np.inf):
    n = len(inner)
    f = lambda v: jnp.asarray(np.broadcast_to(np.asarray(v, np.float64), (n,)), jnp.float32)
    return concentric_wall_kernel(
        jnp.asarray(np.asarray(centers, np.float64), jnp.float32), f(inner), f(outer), L,
        f(D), f(D), f(D), f(kappa_inner), f(0.0), jnp.zeros((n, 4), jnp.float32),
        jnp.float32(1e-7 * min(inner)), jnp.float32(1e-4 * min(inner)), step_max, min_gap)


def _impacts(r_wall, sign):
    """(label, start, step): a walker `off` from the wall at radius `r_wall`, on the side `sign`
    (+1 outside, -1 inside), aimed at the wall over `dist` at `angle` from the normal."""
    out = []
    for oname, off in OFFSETS:
        for dname, dist in DISTANCES:
            for aname, deg in ANGLES:
                th = np.deg2rad(deg)
                start = np.array([r_wall + sign * off * R, 0.0])
                dv = np.array([-sign * np.cos(th), np.sin(th)]) * dist * R
                out.append((f"{oname}/{dname}/{aname}", start, dv))
    return out


def _run(wall, cases, pool, u=1.0):
    starts = jnp.asarray(np.stack([c[1] for c in cases]), jnp.float32)
    steps = jnp.asarray(np.stack([c[2] for c in cases]), jnp.float32)
    lens = jnp.linalg.norm(steps, axis=1)
    dirs = steps / lens[:, None]

    def one(p, dh, l):
        xy, pool_new, k, chan, dlog, crossed = wall(p, dh, l, jnp.int32(pool), jnp.int32(0), jnp.float32(u))
        return xy, pool_new, crossed
    xy, pool_new, crossed = jax.jit(jax.vmap(one))(starts, dirs, lens)
    return np.asarray(xy), np.asarray(pool_new), np.asarray(crossed), np.asarray(starts), np.asarray(lens)


def _report(cases, bad, xy, what):
    rows = "\n".join(f"      {cases[i][0]:26} -> |q| = {np.linalg.norm(xy[i]) / R:7.4f} R"
                     for i in np.flatnonzero(bad)[:12])
    return f"{bad.sum()}/{len(cases)} impacts {what}:\n{rows}"


# ── impact tables on the kernel ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("pool,r_wall,sign,lo,hi", [
    (1, R, -1, 0.0, 1.0),                 # lumen walker aimed at the membrane from inside
    (2, R, +1, 1.0, G),                   # sheath walker aimed at the membrane from outside
    (2, G * R, -1, 1.0, G),               # sheath walker aimed at the outer boundary
    (0, G * R, +1, G, np.inf),            # extra walker aimed at the sheath from outside
], ids=["lumen->membrane", "sheath->membrane", "sheath->outer", "extra->outer"])
def test_impermeable_walls_keep_every_pool_in_place(pool, r_wall, sign, lo, hi):
    wall = _kernel([[0.0, 0.0]], [R], [G * R], None)
    cases = _impacts(r_wall, sign)
    xy, pool_new, crossed, starts, lens = _run(wall, cases, pool)
    rad = np.linalg.norm(xy, axis=1) / R
    left = (rad <= lo * (1 - 1e-6)) | (rad >= hi * (1 - 1e-6) if np.isfinite(hi) else False)
    assert not left.any(), _report(cases, left, xy, f"left pool {pool}")
    assert (pool_new == pool).all() and not crossed.any()
    moved = np.linalg.norm(xy - starts, axis=1)
    assert not (moved > lens * (1 + 1e-4) + 1e-12).any(), "a reflection added path length"


def test_a_granted_crossing_moves_the_walker_exactly_one_pool_over():
    """kappa -> infinity with u = 0 grants the first hit: a lumen walker ends in the sheath, never
    beyond it (the decision is made once; the outer boundary then reflects)."""
    wall = _kernel([[0.0, 0.0]], [R], [G * R], None, kappa_inner=1e9)
    cases = _impacts(R, -1)
    xy, pool_new, crossed, starts, lens = _run(wall, cases, 1, u=0.0)
    # which impacts reach the membrane at all: the exit root of the straight ray within the step
    dirs = np.stack([c[2] for c in cases]) / lens[:, None]
    dp = np.sum(dirs * starts, axis=1)
    t_exit = -dp + np.sqrt(dp ** 2 - (np.sum(starts ** 2, axis=1) - R ** 2))
    reaches = t_exit < lens
    assert reaches.sum() > len(cases) // 2
    rad = np.linalg.norm(xy, axis=1) / R
    assert (crossed == reaches).all(), "every hit must be granted at kappa -> infinity, and only hits"
    assert (pool_new[reaches] == 2).all() and (pool_new[~reaches] == 1).all()
    inside_sheath = (rad > 1.0) & (rad < G)
    assert inside_sheath[reaches].all(), _report(cases, ~inside_sheath & reaches, xy, "did not end in the sheath")
    assert (rad[~reaches] < 1.0).all()


def test_a_step_spanning_the_gap_between_two_axons_stays_in_the_gap():
    """The adversarial packed case: two sheaths 0.05 R apart, a walker in the gap, steps up to
    ten times the gap width at every angle. Zig-zag reflection keeps it in the gap; a single-hit
    rule would end it inside the neighbour."""
    Ro = G * R
    gap = 0.05 * R
    centers = [[0.0, 0.0], [2 * Ro + gap, 0.0]]
    wall = _kernel(centers, [R, R], [Ro, Ro], None, step_max=10 * gap, min_gap=gap)
    assert wall.max_bounces >= 11, "the budget must cover a step of ten gap widths"
    mid = np.array([Ro + gap / 2, 0.0])
    cases = []
    for f in (0.5, 1.0, 3.0, 10.0):
        for deg in (0.0, 20.0, 45.0, 70.0, 89.0, 90.0, 135.0, 180.0):
            th = np.deg2rad(deg)
            cases.append((f"{f}gap/{deg}deg", mid + np.array([0.0, 1e-3 * R]),
                          np.array([np.cos(th), np.sin(th)]) * f * gap))
    xy, pool_new, crossed, starts, lens = _run(wall, cases, 0)
    d0 = np.linalg.norm(xy - np.array(centers[0]), axis=1)
    d1 = np.linalg.norm(xy - np.array(centers[1]), axis=1)
    entered = (d0 <= Ro * (1 - 1e-6)) | (d1 <= Ro * (1 - 1e-6))
    assert not entered.any(), _report(cases, entered, xy, "ended inside a sheath")
    assert (pool_new == 0).all()
    moved = np.linalg.norm(xy - starts, axis=1)
    assert not (moved > lens * (1 + 1e-4) + 1e-12).any()


def test_bounce_budget_and_candidates_follow_the_worst_case():
    """The budget is `step / narrowest passage + 1`, the passage being the smallest of the gap, the
    sheath thickness and the nudged grazing chord; the candidate count covers the disjoint disks
    that can have a wall within one step."""
    Ro = G * R
    w_small = _kernel([[0.0, 0.0]], [R], [Ro], None, step_max=R / 6)
    w_big = _kernel([[0.0, 0.0]], [R], [Ro], None, step_max=2 * R)
    assert 2 <= w_small.max_bounces < w_big.max_bounces <= 32
    chord = 2 * np.sqrt(2 * 1e-4)                       # in units of R
    assert w_small.max_bounces == int(np.ceil((1 / 6) / min(G - 1, chord)) + 1)
    w_gap = _kernel([[0.0, 0.0], [3 * Ro, 0.0]], [R, R], [Ro, Ro], None, step_max=R / 6,
                    min_gap=0.01 * R)
    assert w_gap.max_bounces == int(np.ceil((1 / 6) / 0.01) + 1)
    assert w_gap.n_cand == 2 and w_small.n_cand == 1
    c, L = _pack([R] * 30, 0.7, 0.5, seed=0)
    pm = d.PackedMyelinatedCylinders([R] * 30, 0.7, c, L, N_max=64)
    from dmipy_sim.physics import make_myelin_substep
    sub = make_myelin_substep(pm, 1e-6)
    assert 8 <= sub.n_cand <= 64


# ── the consequences on signals ───────────────────────────────────────────────────────────
@pytest.mark.slow
def test_membrane_exchange_matches_permeable_shell():
    """`MyelinatedCylinder(kappa_inner=k, kappa_outer=None)` with one diffusivity is the closed
    two-compartment cylinder `PermeableShell(kind='cylinder')`: the lumen fraction after a
    diffusion time agrees to the MC floor."""
    R_in, R_out, kappa, N = 3e-6, 6e-6, 2e-5, 30_000
    wf = d.set_b(d.pgse(delta=2e-3, DELTA=18e-3, G_magnitude=0.01, bvecs=[[0, 0, 1]], n_t=200,
                        slew_rate=np.inf), 1e6)
    mc = d.MyelinatedCylinder(R_in, R_out, (0, 0, 1), D, D, D_myelin=D, kappa_inner=kappa,
                              kappa_outer=None, water_fractions=(1.0, 1.0, 0.0))
    _, o_m, f_m = d.simulate(N, None, wf, mc, seed=3, return_compartments="final", require_gpu=False)
    keep = o_m == 1                                     # the shell seeds its lumen; compare that population
    sh = d.geometry.PermeableShell(R_in, R_out, kappa, kind="cylinder", orientation=(0, 0, 1))
    _, o_s, f_s = d.simulate(N, D, wf, sh, seed=3, return_compartments="final", require_gpu=False)
    assert (o_s == 1).all()
    fT_m, fT_s = (f_m[keep] == 1).mean(), (f_s == 1).mean()
    assert fT_s < 0.95, "no exchange happened"
    tol = max(0.015, 4.0 * np.sqrt(fT_s * (1 - fT_s) * (1.0 / keep.sum() + 1.0 / N)))
    assert abs(fT_m - fT_s) < tol, f"lumen fraction after TE: myelin kernel {fT_m:.4f} vs shell {fT_s:.4f}"


def _extra_only_signal(pm, wf, N, seed):
    """The extra-axonal pool's own signal: intra and myelin water are removed by a vanishing T2, and
    the spin-density normalisation of `simulate` is undone with the seeded pool counts."""
    s, origin, _ = d.simulate(N, None, wf, pm, seed=seed, return_compartments="final", require_gpu=False)
    n = np.bincount(origin, minlength=3)
    w_total = n[0] + n[1] + pm._myelin_proton_density * n[2]
    return np.asarray(s).ravel() * w_total / n[0], int(n[0])


@pytest.mark.slow
def test_bare_pack_extra_axonal_signal_matches_packed_cylinders():
    """g -> 1 removes the sheath: the extra-axonal walk of `PackedMyelinatedCylinders` is the walk
    of `PackedCylinders` with the same centres and (outer) radii."""
    radii = [1.0e-6, 1.3e-6, 0.8e-6, 1.1e-6, 0.9e-6, 1.2e-6]
    g = 0.999
    c, L = _pack(radii, g, 0.35, seed=1)
    pm = d.PackedMyelinatedCylinders(radii, g, c, L, N_max=8, T2_intra=1e-6, T2_myelin=1e-6)
    pc = d.PackedCylinders(np.asarray(radii) / g, c, L)
    wf = d.set_b(d.pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=200,
                        slew_rate=np.inf), 1.5e9)
    N = 60_000
    s_pm, n_extra = _extra_only_signal(pm, wf, N, seed=4)
    s_pc, _, f_pc = d.simulate(N, D, wf, pc, seed=4, return_compartments="final", require_gpu=False)
    s_pc = np.asarray(s_pc).ravel()
    # the reference is only a reference while its single-hit rule keeps the walkers outside (#94 A6):
    # a leaked walker is restricted for the rest of the walk, so the leak must stay far below the tolerance
    leaked = (np.asarray(f_pc) > 0).mean()
    assert leaked < 1e-3, f"PackedCylinders let {leaked:.2%} of its walkers into the cylinders"
    tol = 4.0 * np.sqrt(1.0 / n_extra + 1.0 / N)
    assert abs(s_pm[0] - s_pc[0]) < tol, f"extra-axonal: myelin kernel {s_pm[0]:.4f} vs PackedCylinders {s_pc[0]:.4f}"


@pytest.mark.slow
def test_lumen_boundary_local_time_is_brownstein_tarr():
    """The unit boundary local time recorded on the lumen wall (`rho/D = 1`) accrues at
    `2 D / R` per unit time -- Brownstein-Tarr `rho S/V` for a cylinder -- and at the same rate the
    `Cylinder` producer records, since they now share one estimator."""
    R_l, N, T_max, dt_save = 2e-6, 20_000, 4e-3, 2e-4
    c, L = _pack([R_l] * 4, 0.7, 0.5, seed=0)
    pm = d.PackedMyelinatedCylinders([R_l] * 4, 0.7, c, L, N_max=4)
    out = d.simulate_trajectories(N, D, pm, T_max, dt_save, seed=5, save_relaxation_data=True,
                                  require_gpu=False)
    dlog, comp = np.asarray(out.boundary_local_time), np.asarray(out.compartment)
    intra = (comp == 1).all(axis=1)
    rate_pm = -dlog[intra].mean() / dt_save
    cyl = d.Cylinder(R_l, (0, 0, 1))
    out_c = d.simulate_trajectories(N, D, cyl, T_max, dt_save, seed=5, save_relaxation_data=True,
                                    require_gpu=False)
    rate_c = -np.asarray(out_c.boundary_local_time).mean() / dt_save
    theory = 2 * D / R_l
    for name, rate in (("packed myelin lumen", rate_pm), ("Cylinder", rate_c)):
        assert abs(rate / theory - 1) < 0.05, f"{name}: rho S/V = {rate:.4g} vs 2D/R = {theory:.4g}"
    assert abs(rate_pm / rate_c - 1) < 0.03, f"estimators differ: {rate_pm:.4g} vs {rate_c:.4g}"


@pytest.mark.slow
def test_per_axon_properties_are_per_axon():
    """Two distinct intra T2 values decay the b = 0 signal by their mixture, not by the first
    axon's value; a relaxivity on the inner wall only leaves the extra-axonal signal untouched."""
    radii = [1e-6] * 6
    c, L = _pack(radii, 0.7, 0.5, seed=2)
    wf = d.set_b(d.pgse(delta=2e-3, DELTA=28e-3, G_magnitude=0.01, bvecs=[[1, 0, 0]], n_t=300,
                        slew_rate=np.inf), 1e6)
    N = 20_000
    mixed = d.PackedMyelinatedCylinders(radii, 0.7, c, L, N_max=8, T2_intra=[0.01] * 3 + [1.0] * 3,
                                        T2_extra=1e-6, T2_myelin=1e-6)
    first = d.PackedMyelinatedCylinders(radii, 0.7, c, L, N_max=8, T2_intra=0.01,
                                        T2_extra=1e-6, T2_myelin=1e-6)
    s_mixed, o, _ = d.simulate(N, None, wf, mixed, seed=6, return_compartments="final", require_gpu=False)
    s_first, _, _ = d.simulate(N, None, wf, first, seed=6, return_compartments="final", require_gpu=False)
    n = np.bincount(o, minlength=3)
    w_total = n[0] + n[1] + mixed._myelin_proton_density * n[2]
    TE = wf.echo_idx * wf.dt
    intra_first = float(s_first[0]) * w_total / n[1]
    intra_mixed = float(s_mixed[0]) * w_total / n[1]
    assert abs(intra_first - np.exp(-TE / 0.01)) < 0.03
    assert abs(intra_mixed - 0.5 * (np.exp(-TE / 0.01) + np.exp(-TE / 1.0))) < 0.03, \
        f"intra b0 signal {intra_mixed:.4f} does not decay by the per-axon T2 mixture"

    rho_in = d.PackedMyelinatedCylinders(radii, 0.7, c, L, N_max=8, rho_inner=2e-6,
                                         T2_intra=1e-6, T2_myelin=1e-6)
    plain = d.PackedMyelinatedCylinders(radii, 0.7, c, L, N_max=8, T2_intra=1e-6, T2_myelin=1e-6)
    s_rho, n_e = _extra_only_signal(rho_in, wf, N, seed=7)
    s_plain, _ = _extra_only_signal(plain, wf, N, seed=7)
    assert abs(s_rho[0] - s_plain[0]) < 4.0 * np.sqrt(2.0 / n_e), \
        f"an inner-wall relaxivity changed the extra-axonal signal: {s_rho[0]:.4f} vs {s_plain[0]:.4f}"
    only_intra = d.PackedMyelinatedCylinders(radii, 0.7, c, L, N_max=8, rho_inner=2e-6,
                                             T2_extra=1e-6, T2_myelin=1e-6)
    s_i, o_i, _ = d.simulate(N, None, wf, only_intra, seed=7, return_compartments="final", require_gpu=False)
    n_i = np.bincount(o_i, minlength=3)
    intra_rho = float(s_i[0]) * (n_i[0] + n_i[1] + only_intra._myelin_proton_density * n_i[2]) / n_i[1]
    assert intra_rho < 1.0 - 0.05, "the inner-wall relaxivity did not act on the lumen water"

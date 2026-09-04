"""Single designed events, checked against closed forms. No walks, no statistics.

The principle, after #90 and #91: **a run is only about statistics at scale; the correctness
of an individual event should be isolated and small.** Every physical effect in this
simulator is a deterministic function of one event's geometry, so it can be fired directly
and compared to the formula it is supposed to implement -- exactly, in milliseconds, with a
failure that names the case.

Contrast with what these replace. Surface relaxation was only ever checked through an
aggregate signal over 10^5 walkers: if `d_perp` had the wrong angular dependence, the test
would report "signal off by 3%" after minutes, and would not say at which incidence angle,
or whether the fault was the angle, the path length, the factor of 2 or the walk itself.

Geometries are built inside each test, never at collection time (see test_wall_impacts).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry import Sphere, PermeableSlab1D

R = 5.0e-6
D = 2.0e-9


def test_surface_log_weight_follows_cos_alpha_at_every_incidence_angle():
    """`dlog_w = -2 (rho/D) * d_perp`, with `d_perp = remaining * cos(alpha)`.

    Fire at a known angle from the surface normal and the answer is arithmetic. Sweeping the
    angle checks the ANGULAR DEPENDENCE, which is the part an aggregate signal cannot
    localise: a grazing hit must deposit less than a head-on one, by exactly cos(alpha).
    """
    geom = Sphere(radius=R, permeability=0.0)
    rho_over_D = 1.0e3
    reach = 0.02 * R                       # remaining path after the hit

    # Hold `remaining` CONSTANT while varying the angle, or two things change at once.
    # A walker `d0` from the wall aiming at angle `th` travels d0/cos(th) to reach it, so
    # the step must be d0/cos(th) + reach for the post-hit path to be `reach` every time.
    d0 = 0.01 * R
    angles = np.deg2rad([0.0, 15.0, 30.0, 45.0, 60.0])
    start, steps = [], []
    for th in angles:
        start.append([R - d0, 0.0, 0.0])
        travel = d0 / np.cos(th) + reach
        steps.append([travel * np.cos(th), travel * np.sin(th), 0.0])
    starts = jnp.asarray(np.array(start, np.float32))
    stps = jnp.asarray(np.array(steps, np.float32))

    hits = jax.jit(jax.vmap(lambda p, s: geom.interact(
        p, s, rho_over_D=jnp.float32(rho_over_D))))(starts, stps)
    dlw = np.asarray(hits.dlog_w)

    # every hit must deposit weight, and a grazing hit strictly less than a head-on one
    assert (dlw < 0).all(), f"no surface weight deposited at some angle: {dlw}"
    assert np.all(np.diff(np.abs(dlw)) < 1e-12), (
        f"|dlog_w| must fall monotonically as incidence becomes grazing, got "
        f"{np.abs(dlw)} at {np.rad2deg(angles)} deg")
    # and the ratio to head-on must follow cos(alpha), not some other angular law
    ratio = np.abs(dlw) / abs(dlw[0])
    npred = np.cos(angles)
    assert np.allclose(ratio, npred, atol=0.08), (
        f"angular dependence is not cos(alpha):\n  measured {np.round(ratio, 4)}\n"
        f"  cos(alpha) {np.round(npred, 4)}")


def test_surface_log_weight_is_linear_in_rho_and_in_path_length():
    """The other two factors in `-2 (rho/D) * remaining * cos(alpha)`, isolated.

    Doubling rho doubles the deposit; doubling the remaining path doubles it. Each is one
    line of arithmetic and neither needs a walker.
    """
    geom = Sphere(radius=R, permeability=0.0)
    reach = 0.02 * R
    start = jnp.asarray(np.array([[R - reach, 0.0, 0.0]], np.float32))
    step = jnp.asarray(np.array([[2 * reach, 0.0, 0.0]], np.float32))

    a = float(jax.vmap(lambda p, s: geom.interact(
        p, s, rho_over_D=jnp.float32(1.0e3)))(start, step).dlog_w[0])
    b = float(jax.vmap(lambda p, s: geom.interact(
        p, s, rho_over_D=jnp.float32(2.0e3)))(start, step).dlog_w[0])
    assert b == pytest.approx(2.0 * a, rel=1e-4), f"not linear in rho: {a} -> {b}"

    step2 = jnp.asarray(np.array([[3 * reach, 0.0, 0.0]], np.float32))
    c = float(jax.vmap(lambda p, s: geom.interact(
        p, s, rho_over_D=jnp.float32(1.0e3)))(start, step2).dlog_w[0])
    # remaining path after the hit doubles (reach -> 2*reach), so the deposit doubles
    assert c == pytest.approx(2.0 * a, rel=1e-3), f"not linear in path length: {a} -> {c}"


def test_a_granted_crossing_deposits_no_surface_weight():
    """Reflection deposits; transmission does not. One event, both branches.

    `dlog_w` is a REFLECTION weight -- a walker that crosses did not linger at the wall. An
    aggregate signal cannot separate "transmitted" from "reflected with small deposit"; a
    single designed event can, because `crossed` says which happened.
    """
    L = 5.0e-6
    reach = 0.02 * L
    start = jnp.asarray(np.array([[L / 2 - reach, 0.0, 0.0]] * 64, np.float32))
    step = jnp.asarray(np.array([[3 * reach, 0.0, 0.0]] * 64, np.float32))
    keys = jax.random.split(jax.random.PRNGKey(0), 64)

    # p_transmit = 2 (kappa/D) d_perp; with d_perp ~ 2e-7 m, kappa = 2.5e-3 puts it near
    # 0.5 so one sweep of 64 draws contains both branches.
    kappa = 2.5e-3
    geom = PermeableSlab1D(length=L, permeability=kappa)
    kod = jnp.float32(kappa / D)
    hits = jax.jit(jax.vmap(lambda p, s, k: geom.interact(
        p, s, kappa_over_D=kod, rho_over_D=jnp.float32(1.0e3), key=k)))(start, step, keys)

    crossed = np.asarray(hits.crossed)
    dlw = np.asarray(hits.dlog_w)
    assert crossed.any() and (~crossed).any(), (
        f"need both branches in one sweep; got crossed={crossed.sum()}/64")
    assert np.all(dlw[crossed] == 0.0), (
        f"{int((dlw[crossed] != 0).sum())} transmitted walkers deposited surface weight")
    assert np.all(dlw[~crossed] < 0.0), "reflected walkers deposited nothing"


def test_a_designed_crossing_lands_on_the_far_side_and_stays_there():
    """inside -> outside through one granted permeation, then confined on the far side.

    The event the compartment channel is built from, isolated: a walker that crosses must
    END on the other side, and a subsequent impermeable step must keep it there.
    """
    L = 5.0e-6
    geom_open = PermeableSlab1D(length=L, permeability=1.0e-3)   # kappa high: crossing likely
    geom_shut = PermeableSlab1D(length=L, permeability=0.0)
    reach = 0.02 * L
    n = 128
    start = jnp.asarray(np.array([[L / 2 - reach, 0.0, 0.0]] * n, np.float32))
    step = jnp.asarray(np.array([[3 * reach, 0.0, 0.0]] * n, np.float32))
    keys = jax.random.split(jax.random.PRNGKey(1), n)
    kod = jnp.float32(1.0e-3 / D)

    hits = jax.jit(jax.vmap(lambda p, s, k: geom_open.interact(
        p, s, kappa_over_D=kod, key=k)))(start, step, keys)
    crossed = np.asarray(hits.crossed)
    x = np.asarray(hits.r)[:, 0]
    assert crossed.any(), "no crossing granted at high kappa; test is vacuous"
    assert np.all(x[crossed] > L / 2), (
        f"{int((x[crossed] <= L / 2).sum())} walkers reported a crossing but stayed on the "
        f"near side -- `crossed` and the position disagree")

    # now step them again through an IMPERMEABLE membrane: they must stay where they are
    back = jnp.asarray(np.array([[-3 * reach, 0.0, 0.0]] * int(crossed.sum()), np.float32))
    after = jax.jit(jax.vmap(lambda p, s: geom_shut.interact(p, s)))(
        jnp.asarray(np.asarray(hits.r)[crossed]), back)
    assert np.all(np.asarray(after.r)[:, 0] > L / 2), (
        "a walker crossed back through an impermeable membrane")

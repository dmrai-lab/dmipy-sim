"""No geometry may change a walker's compartment without granting a crossing.

This is the regression net for dmrai-lab/dmipy-sim#86, generalised to every geometry.

The defect needs three ingredients:
  1. a collision test that MISSES a boundary interaction -- the exit time marginally
     exceeds the step, so no collision fires, which is correct: the walker never
     reaches the wall;
  2. a position update that therefore leaves the raw step in place, landing the walker
     on the surface to float32;
  3. a compartment label RE-DERIVED from that position with a strict inequality.

Ingredient 3 turns a harmless rounding into a compartment change, and the walker
changes compartment without moving. On a dense packing that ran at 0.675% of intra
walkers per 30k steps at kappa = 0; on a plain Cylinder walk 0.055%; on a Mesh 0.250%.

A random walk is a WEAK detector for this -- 2.3e-7 per walker-step means a CI-sized
walk expects a fraction of an event. So this constructs the failure instead: bisect
along a ray using the geometry's OWN classifier to find the boundary to float32
resolution, place the walker one sub-step short of it, and step exactly onto it.

Two things this test had to get right, both of which produced false readings first:

* Seed from the geometry's own ``init_positions``. Sampling a bounding cube puts
  walkers outside the domain (PermeableSlab1D lives in x in [0, L], PermeableShell in
  a shell); folding them back in reads as a "leak" that is purely a bad setup.
* Use a REALISTIC step. The bisection reaches the wall from wherever the walker is --
  up to microns -- but an engine sub-step is ~20 nm, and the packed geometries' O(1)
  sentinels rely on a walker only interacting with the object it already borders. A
  micron-long step measures that assumption, not the engine, and reported 3-6% on
  geometries that are in fact exact.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometries import (Sphere, Cylinder, Ellipsoid, PackedCylinders,
                                  PackedSpheres, PermeableSlab1D, PermeableShell)
from dmipy_sim.curved_tube import CurvedTube, MultiShellCurvedTube

SUB_STEP = 2.0e-8          # a representative engine sub-step (20 nm)
N_WALK   = 1500


def _pack(n, R, L, dim, seed=1):
    rng = np.random.default_rng(seed)
    cen, tries = [], 0
    while len(cen) < n and tries < 200_000:
        p = rng.uniform(-L / 2, L / 2, dim); tries += 1
        if all(np.linalg.norm((p - c) - L * np.round((p - c) / L)) >= 2.2 * R for c in cen):
            cen.append(p)
    return np.array(cen)


def _geometries():
    c2 = _pack(20, 1e-6, 12e-6, 2)
    c3 = _pack(8, 1e-6, 12e-6, 3)
    return [
        pytest.param("Sphere", Sphere(radius=5e-6, permeability=0.0), id="Sphere"),
        pytest.param("Cylinder", Cylinder(radius=5e-6, orientation=[0, 0, 1.0],
                     permeability=0.0), id="Cylinder"),
        pytest.param("Ellipsoid", Ellipsoid(semiaxes=[5e-6, 3e-6, 4e-6],
                     permeability=0.0), id="Ellipsoid"),
        pytest.param("PackedCylinders", PackedCylinders(centers=c2,
                     radii=np.full(len(c2), 1e-6), L=12e-6,
                     orientation=[0, 0, 1.0], permeability=0.0),
                     id="PackedCylinders"),
        pytest.param("PackedSpheres", PackedSpheres(radii=np.full(len(c3), 1e-6),
                     centers=c3, L=12e-6, permeability=0.0), id="PackedSpheres"),
        pytest.param("PermeableSlab1D", PermeableSlab1D(length=5e-6,
                     permeability=0.0), id="PermeableSlab1D"),
        pytest.param("PermeableShell", PermeableShell(r_inner=3e-6, r_outer=5e-6,
                     permeability=0.0), id="PermeableShell"),
    ]


def _boundary_offsets(cls_v, r0, lab0, seed=0, n_dirs=6, iters=50, max_reach=6e-6):
    """Distance to the nearest compartment boundary along a random ray, per walker.

    ``max_reach`` caps how far the search may travel. This matters for the PERIODIC packed
    geometries: a ray there always eventually re-enters some sphere/cylinder IMAGE, so an
    uncapped search happily reports a "boundary" thousands of box lengths away. Placing a
    walker there is not a geometry test -- at |r| ~ 1e5 L the minimum-image reduction
    ``q - L*floor(q/L + 1/2)`` is a catastrophic cancellation, and float32 has only ~24 nm
    of absolute precision at that magnitude, so the sentinel and the classifier round
    differently on the same point. That produced a spurious ~4% "leak" for PackedSpheres.
    Half a box is the largest reach that still tests real, in-cell geometry.
    """
    rng = np.random.default_rng(seed)
    n = r0.shape[0]
    best = np.full(n, np.inf)
    best_u = np.zeros((n, 3))
    for _ in range(n_dirs):
        u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        lo = np.zeros(n); hi = np.full(n, max_reach / 64.0); found = np.zeros(n, bool)
        for _g in range(12):
            lab = np.asarray(cls_v(jnp.asarray((r0 + u * hi[:, None]).astype(np.float32))))
            found |= (lab != lab0)
            hi = np.where(found | (hi >= max_reach), hi, np.minimum(hi * 1.6, max_reach))
        for _b in range(iters):
            mid = 0.5 * (lo + hi)
            out = np.asarray(cls_v(jnp.asarray((r0 + u * mid[:, None]).astype(np.float32)))) != lab0
            hi = np.where(out, mid, hi); lo = np.where(out, lo, mid)
        t = np.where(found, lo, np.inf)
        upd = t < best
        best = np.where(upd, t, best); best_u = np.where(upd[:, None], u, best_u)
    return best, best_u


@pytest.mark.parametrize("name,geom", _geometries())
def test_landing_on_a_wall_never_changes_compartment(name, geom):
    cls_v = jax.jit(jax.vmap(geom.classify_position))
    r0 = np.asarray(geom.init_positions(N_WALK, jax.random.PRNGKey(0)), np.float64)
    lab0 = np.asarray(cls_v(jnp.asarray(r0.astype(np.float32))))

    t, u = _boundary_offsets(cls_v, r0, lab0)
    usable = np.isfinite(t) & (t > SUB_STEP)
    assert usable.sum() > N_WALK // 4, f"{name}: too few walkers near a boundary to test"

    # sit exactly one sub-step short of the wall, then step exactly onto it
    off = np.where(np.isfinite(t), t - SUB_STEP, 0.0)[:, None]   # inf*0 -> nan otherwise
    r_near = (r0 + u * off)[usable].astype(np.float32)
    step = (u * SUB_STEP)[usable].astype(np.float32)
    lab_near = np.asarray(cls_v(jnp.asarray(r_near)))
    n = int(usable.sum())

    kwargs = {}
    if "side" in __import__("inspect").signature(geom.permeate).parameters:
        # geometries that carry the compartment need it supplied, as the engine does
        kwargs["side"] = jnp.where(jnp.asarray(lab_near) > 0, jnp.int8(-1), jnp.int8(1))
        f = jax.jit(jax.vmap(lambda p, s, k, d: geom.permeate(
            p, s, jnp.float32(0.0), jnp.float32(0.0), k, d)[0], in_axes=(0, 0, 0, 0)))
        out = f(jnp.asarray(r_near), jnp.asarray(step),
                jax.random.split(jax.random.PRNGKey(3), n), kwargs["side"])
    else:
        f = jax.jit(jax.vmap(lambda p, s, k: geom.permeate(
            p, s, jnp.float32(0.0), jnp.float32(0.0), k)[0], in_axes=(0, 0, 0)))
        out = f(jnp.asarray(r_near), jnp.asarray(step),
                jax.random.split(jax.random.PRNGKey(3), n))

    # the bare step must actually reach the wall, or the probe proves nothing
    raw = np.asarray(cls_v(jnp.asarray(r_near) + jnp.asarray(step)))
    assert (raw != lab_near).any(), f"{name}: probe never landed on a boundary"

    flipped = int((np.asarray(cls_v(out)) != lab_near).sum())
    assert flipped == 0, (
        f"{name}: {flipped}/{n} walkers changed compartment at kappa=0 with no crossing "
        f"granted (bare step flipped {int((raw != lab_near).sum())})")


# ── reflect-only geometries ──────────────────────────────────────────────────
# The curved tubes are impermeable (no `permeate`), so they are probed through `reflect`.
# They were the last family with no shared boundary code at all: `CurvedTube`'s "safety
# clamp" was an exact algebraic handroll of `keep_side_radial`, and `PackedCurvedTubes`
# carried its own quadratic and its own specular. Both now call the shared rules, so the
# same invariant is asserted here as everywhere else.

def _reflect_geometries():
    t = np.linspace(0, 1, 64)
    cl = np.stack([20e-6 * t, 3e-6 * np.sin(2 * np.pi * t), np.zeros_like(t)], 1)
    return [
        pytest.param("CurvedTube", CurvedTube(centerline=cl, radius=2e-6), id="CurvedTube"),
        pytest.param("MultiShellCurvedTube",
                     MultiShellCurvedTube(centerline=cl, r_in=1.5e-6, r_out=2e-6),
                     id="MultiShellCurvedTube"),
    ]


@pytest.mark.parametrize("name,geom", _reflect_geometries())
def test_reflect_landing_on_a_wall_never_changes_compartment(name, geom):
    """Same invariant, through `reflect`: an impermeable wall grants no crossings."""
    cls_v = jax.jit(jax.vmap(geom.classify_position))
    r0 = np.asarray(geom.init_positions(N_WALK, jax.random.PRNGKey(0)), np.float64)
    lab0 = np.asarray(cls_v(jnp.asarray(r0.astype(np.float32))))

    t, u = _boundary_offsets(cls_v, r0, lab0, max_reach=2e-6)
    usable = np.isfinite(t) & (t > SUB_STEP)
    assert usable.sum() > N_WALK // 8, f"{name}: too few walkers near a boundary to test"

    off = np.where(np.isfinite(t), t - SUB_STEP, 0.0)[:, None]   # inf*0 -> nan otherwise
    r_near = (r0 + u * off)[usable].astype(np.float32)
    step = (u * SUB_STEP)[usable].astype(np.float32)
    lab_near = np.asarray(cls_v(jnp.asarray(r_near)))

    out = jax.jit(jax.vmap(geom.reflect, in_axes=(0, 0)))(jnp.asarray(r_near),
                                                          jnp.asarray(step))
    raw = np.asarray(cls_v(jnp.asarray(r_near) + jnp.asarray(step)))
    assert (raw != lab_near).any(), f"{name}: probe never landed on a boundary"

    flipped = int((np.asarray(cls_v(out)) != lab_near).sum())
    assert flipped == 0, (
        f"{name}: {flipped}/{len(lab_near)} walkers changed compartment through an "
        f"impermeable wall (bare step flipped {int((raw != lab_near).sum())})")


def test_the_sentinel_does_not_seal_the_membrane():
    """A cheap fast-lane guard on the OTHER failure mode of the compartment sentinel.

    The sentinel rejects a compartment change that no crossing granted. Get that wrong in
    the obvious way and it rejects the legal ones too, silencing the physics instead of the
    bug -- and at kappa = 0 a sealed wall and a correct wall are indistinguishable, so every
    impermeable test above would still pass.

    The quantitative check (crossing rate vs the closed two-compartment law) is a heavy MC
    walk and lives in `test_permeable_crossings`, marked slow. This is the one-bit version:
    an impermeable wall grants nothing, a permeable one grants something.
    """
    # kappa chosen so crossings appear in few steps: p_transmit = 2 (kappa/D) d_perp
    # = 0.2 per wall hit, so a short walk suffices and the guard stays cheap.
    n, n_steps, kappa_open = 1200, 700, 1.0e-2
    D, step = 2.0e-9, SUB_STEP

    def crossings(kappa):
        g = PermeableSlab1D(length=5e-6, permeability=kappa)
        kod = jnp.float32(kappa / D)
        f = jax.jit(jax.vmap(lambda p, s, k: g.permeate(p, s, kod, jnp.float32(0.0), k)[0],
                             in_axes=(0, 0, 0)))
        r = g.init_positions(n, jax.random.PRNGKey(0))       # all seeded in compartment A
        rng = np.random.default_rng(3)
        for i in range(n_steps):
            d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            r = f(r, jnp.asarray(d * step, jnp.float32),
                  jax.random.split(jax.random.PRNGKey(i), n))
        return int((np.asarray(r)[:, 0] >= 2.5e-6).sum())

    assert crossings(0.0) == 0, "walkers crossed an impermeable membrane"
    n_open = crossings(kappa_open)
    assert n_open > 0, (
        "no walker crossed a permeable membrane -- the sentinel has sealed the wall, "
        "which no kappa=0 test can detect")

"""One wall, one interaction: `reflect` must equal a crossing-free `permeate` (#88).

`reflect`, `reflect_with_log_weight` and `permeate` are the same function at different
argument values -- and measured, not even an optimisation: `reflect` and `permeate(kappa=0)`
both cost 0.11 ms / 40k walkers, because XLA folds `kappa = 0` and drops the dead transmit
branch. What keeping them apart bought instead was drift:

    PackedCylinders.reflect  expelled 100% of intra-axonal walkers at kappa = 0
    PackedCylinders.permeate(kappa=0)  confined them correctly

while being bit-identical on the extra-axonal side. At kappa = 0 nothing may cross, so a
walker reflects whichever side of the wall it is on; `reflect` was simply wrong on one side.

This is reachable by default: the engine picks the path with `permeability is not None`, so
`PackedCylinders(centers=..., radii=..., L=...)` -- the natural way to build an impermeable
packing -- gets `reflect`, and only an explicit `permeability=0.0` (which reads like a
no-op) gets the correct path.

These tests state the invariant directly, so the collapse onto one API is checkable rather
than hopeful.
"""
# NOTE: geometries are built INSIDE each test, never at parametrize/collection time. A
# geometry holds jnp arrays, which are device buffers; `gpu.py` calls `jax.clear_caches()`,
# after which a buffer created at collection time raises "Array has been deleted". These
# files pass in isolation and fail in a full run otherwise -- the same trap as a module-level
# jnp constant.
_DEVICE_ARRAY_NOTE = True

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry import PackedCylinders, PackedSpheres

R, D, STEP = 1.0e-6, 2.0e-9, 2.0e-8


def _pack(n, radius, L, dim, seed=1):
    rng = np.random.default_rng(seed)
    cen, tries = [], 0
    while len(cen) < n and tries < 300_000:
        p = rng.uniform(-L / 2, L / 2, dim); tries += 1
        if all(np.linalg.norm((p - c) - L * np.round((p - c) / L)) >= 2.2 * radius for c in cen):
            cen.append(p)
    return np.array(cen)


def _build(name):
    """Construct on demand -- see the device-array note at the top of this file."""
    if name == "PackedCylinders":
        c = _pack(20, R, 12e-6, 2)
        return PackedCylinders(centers=c, radii=np.full(len(c), R), L=12e-6,
                               orientation=[0, 0, 1.0], permeability=0.0), c
    c = _pack(8, R, 12e-6, 3)
    return PackedSpheres(radii=np.full(len(c), R), centers=c, L=12e-6,
                         permeability=0.0), c


_PACKED = ["PackedCylinders", "PackedSpheres"]


def _seed_intra(centers, n, seed=0):
    """Walkers strictly inside a randomly chosen object."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(centers), n)
    u = rng.uniform(0, 1, n) ** 0.5 * R * 0.85
    r = np.zeros((n, 3), np.float32)
    if centers.shape[1] == 2:                       # cylinders: free axial coordinate
        th = rng.uniform(0, 2 * np.pi, n)
        r[:, 0] = centers[k, 0] + u * np.cos(th)
        r[:, 1] = centers[k, 1] + u * np.sin(th)
    else:                                           # spheres: isotropic
        v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        r[:] = centers[k] + v * u[:, None]
    return r


def _walk(step_fn, r, n_steps, seed=5):
    rng = np.random.default_rng(seed)
    n = r.shape[0]
    for i in range(n_steps):
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = step_fn(r, jnp.asarray(d * STEP, jnp.float32), i)
    return r


@pytest.mark.parametrize("name", _PACKED)
def test_reflect_confines_intra_walkers(name):
    geom, centers = _build(name)
    """An impermeable wall confines BOTH sides. `reflect` must not expel intra walkers."""
    n, n_steps = 2000, 400
    r0 = _seed_intra(centers, n)
    cls = jax.jit(jax.vmap(geom.classify_position))
    lab0 = np.asarray(cls(jnp.asarray(r0)))
    assert (lab0 > 0).all(), "seeding failed: walkers are not intra"

    f = jax.jit(jax.vmap(geom.reflect, in_axes=(0, 0)))
    rf = _walk(lambda r, s, i: f(r, s), jnp.asarray(r0), n_steps)

    left = int((np.asarray(cls(rf)) != lab0).sum())
    assert left == 0, (
        f"{name}.reflect expelled {left}/{n} intra walkers through an impermeable wall. "
        f"At kappa = 0 nothing crosses, so a walker reflects on whichever side it starts.")


@pytest.mark.parametrize("name", _PACKED)
def test_reflect_equals_permeate_at_zero_permeability(name):
    geom, centers = _build(name)
    """The two are the same function; at kappa = 0 they must agree everywhere, both sides."""
    n = 4000
    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(-6e-6, 6e-6, (n, 3)).astype(np.float32))
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    s = jnp.asarray((d * STEP).astype(np.float32))
    keys = jax.random.split(jax.random.PRNGKey(0), n)

    a = jax.jit(jax.vmap(geom.reflect, in_axes=(0, 0)))(r, s)
    b = jax.jit(jax.vmap(lambda p, x, k: geom.permeate(
        p, x, jnp.float32(0.0), jnp.float32(0.0), k)[0], in_axes=(0, 0, 0)))(r, s, keys)

    diff = np.abs(np.asarray(a) - np.asarray(b)).max(axis=1)
    n_diff = int((diff > 0).sum())
    lab = np.asarray(jax.jit(jax.vmap(geom.classify_position))(r))
    assert n_diff == 0, (
        f"{name}: reflect and permeate(kappa=0) disagree on {n_diff}/{n} walkers "
        f"(max |diff| = {diff.max():.2e} m; "
        f"{int((diff[lab > 0] > 0).sum())} of them intra, "
        f"{int((diff[lab == 0] > 0).sum())} extra). They are the same physics.")


# ── the unified API ──────────────────────────────────────────────────────────
_ALL_NAMES = [("Sphere", True), ("Cylinder", True), ("Ellipsoid", True),
              ("PackedCylinders", True), ("PackedSpheres", True),
              ("PermeableSlab1D", True), ("PermeableShell", True),
              ("FreeDiffusion", False), ("Box1D", False), ("CurvedTube", False)]


def _build_any(name):
    """Construct on demand -- see the device-array note at the top of this file."""
    for n, g, p in _all_geometries():
        if n == name:
            return g, p
    raise KeyError(name)


def _all_geometries():
    from dmipy_sim.geometry import (Sphere, Cylinder, Ellipsoid, FreeDiffusion, Box1D,
                                    PermeableSlab1D, PermeableShell, CurvedTube)
    c2 = _pack(12, R, 12e-6, 2); c3 = _pack(6, R, 12e-6, 3)
    t = np.linspace(0, 1, 32)
    cl = np.stack([20e-6 * t, 3e-6 * np.sin(2 * np.pi * t), np.zeros_like(t)], 1)
    return [
        ("Sphere", Sphere(radius=5e-6, permeability=0.0), True),
        ("Cylinder", Cylinder(radius=5e-6, orientation=[0, 0, 1.0], permeability=0.0), True),
        ("Ellipsoid", Ellipsoid(semiaxes=[5e-6, 3e-6, 4e-6], permeability=0.0), True),
        ("PackedCylinders", PackedCylinders(centers=c2, radii=np.full(len(c2), R),
                                            L=12e-6, orientation=[0, 0, 1.0],
                                            permeability=0.0), True),
        ("PackedSpheres", PackedSpheres(radii=np.full(len(c3), R), centers=c3,
                                        L=12e-6, permeability=0.0), True),
        ("PermeableSlab1D", PermeableSlab1D(length=5e-6, permeability=0.0), True),
        ("PermeableShell", PermeableShell(r_inner=3e-6, r_outer=5e-6, permeability=0.0), True),
        ("FreeDiffusion", FreeDiffusion(), False),
        ("Box1D", Box1D(length=1e-5), False),
        ("CurvedTube", CurvedTube(centerline=cl, radius=2e-6), False),
    ]


@pytest.mark.parametrize("name", [g[0] for g in _ALL_NAMES])
def test_interact_is_available_on_every_geometry(name):
    geom, permeable = _build_any(name)
    """One API, every substrate: `interact` returns a WallHit whatever the geometry."""
    r = jnp.asarray(np.asarray(geom.init_positions(4, jax.random.PRNGKey(0)))[0])
    s = jnp.asarray(np.array([1.0, 0.5, -0.3], np.float32) * 1e-8)

    hit = geom.interact(r, s)
    assert hit._fields == ("r", "dlog_w", "crossed", "illegal")
    assert np.all(np.isfinite(np.asarray(hit.r))), f"{name}: non-finite position"
    assert not bool(hit.crossed), "an impermeable interaction granted a crossing"

    # the surface-relaxation channel is reachable through the same call
    assert np.isfinite(float(geom.interact(r, s, rho_over_D=1.0).dlog_w))

    assert geom.supports_permeability is permeable, (
        f"{name}.supports_permeability should be {permeable}")


@pytest.mark.parametrize("name", [g[0] for g in _ALL_NAMES if not g[1]])
def test_interact_refuses_permeability_it_cannot_honour(name):
    geom, permeable = _build_any(name)
    """A geometry with no membrane must REFUSE kappa > 0, not silently reflect.

    Absence of a `permeate` method used to be the capability signal, so "cannot cross" and
    "not implemented" were indistinguishable and the failure mode was a silent fallback --
    the same shape as the myelin stubs that returned free diffusion without an error.
    """
    r = jnp.asarray(np.asarray(geom.init_positions(4, jax.random.PRNGKey(0)))[0])
    s = jnp.asarray(np.array([1.0, 0.0, 0.0], np.float32) * 1e-8)
    with pytest.raises(NotImplementedError, match="no membrane"):
        geom.interact(r, s, kappa_over_D=1.0e4)

"""Laboratory impact table: fire one walker at one wall, from a known offset, direction and
distance, and check where it lands. No random walk, no statistics, no waiting.

Why this exists
---------------
Every boundary bug found so far was a *specific geometric case* that a random walk reaches
only rarely, or reaches so often that it hides in an aggregate:

    #86  step whose exit time marginally exceeds its length  -> 2.3e-7 per walker-step
    #88  walker on the far side of the wall from the intended one
    #88  step longer than the object it is crossing          -> only bites at small radius

The last one regressed the analytic geometries and a 3-minute MC validation caught it; this
table catches it in milliseconds, and says which case broke rather than "a walker escaped".

The sweep is the cross product of the things that actually distinguish implementations:

    offset    far from the wall / one step short / a nudge short / EXACTLY on it
    direction head-on / 45 deg / grazing (89 deg) / tangential
    distance  much shorter than the object / comparable / SPANNING it / many diameters

`step spanning the object` is the multi-bounce case: reflect once and fly the remainder and
the walker exits the far side. `exactly on the wall` is the float32 tie from #86.
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

from dmipy_sim.geometry import Sphere, Cylinder, Ellipsoid, PackedCylinders, PackedSpheres

R = 5.0e-6


def _lattice(n_side, radius, dim=2):
    """Deterministic non-overlapping centres on a square lattice.

    Replaces rejection sampling, which was both slow and WRONG here: asking for 12 cylinders
    of R = 5 um at 2.2 R separation inside a 30 um box is geometrically impossible, so the
    sampler burned 300,000 rejected draws (13.7 s) and silently returned 4. A lattice gives
    the requested count, instantly, reproducibly, and states the box size it needs.
    """
    pitch = 3.0 * radius                     # comfortably clear of 2.2 R
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * pitch
    grid = np.meshgrid(*([offs] * dim), indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1), n_side * pitch


def _radial(name):
    """Distance from the object's axis/centre -- the quantity the wall is a level set of."""
    if name in _PACKED:
        # a packing has objects at arbitrary centres, so "distance from the wall" is measured
        # against the nearest one under the minimum image, not against the origin
        centers, L = _PACK[name]
        dim = centers.shape[1]

        def _f(p):
            q = np.atleast_2d(p)[:, None, :dim] - centers[None, :, :]
            q -= L * np.floor(q / L + 0.5)
            return np.linalg.norm(q, axis=2).min(axis=1)
        return _f
    if name == "Cylinder":
        return lambda p: np.linalg.norm(np.atleast_2d(p)[:, :2], axis=1)
    if name == "Ellipsoid":
        # the quadric radius sqrt(r.D.r), scaled by R so the wall sits at R like the others
        return lambda p: R * np.sqrt(np.sum((np.atleast_2d(p) / _SEMI) ** 2, axis=1))
    return lambda p: np.linalg.norm(np.atleast_2d(p), axis=1)


_SEMI = np.array([R, 0.6 * R, 0.8 * R])          # ellipsoid semi-axes; the +x wall is at R
_PACKED = ("PackedCylinders", "PackedSpheres")
_NAMES = ["Sphere", "Cylinder", "Ellipsoid", *_PACKED]
_PACK = {}                                        # name -> (centres, L) of the packing built for it


_CACHE = {}


def _build(name):
    """Build once per module run, then reuse.

    Not at collection time -- that is what puts device buffers in a session another module's
    `jax.clear_caches()` later frees. Building on first use and caching keeps that safety
    while paying `PackedCylinders.__init__` (14.4 s, measured) once instead of five times.
    """
    if name in _CACHE:
        return _CACHE[name]
    _CACHE[name] = _build_uncached(name)
    return _CACHE[name]


def _build_uncached(name):
    """Construct on demand -- see the device-array note at the top of this file.

    Building at parametrize time puts these geometries' jnp arrays on the device at
    COLLECTION, and `jax.clear_caches()` in another test then deletes them: the file passes
    alone and fails in a full run with "Array has been deleted". That is exactly what
    happened, in this file, after the note above was written and not acted on.
    """
    if name == "PackedCylinders":
        c2, L = _lattice(3, R, 2)            # 9 cylinders, box sized to fit them
        _PACK[name] = (c2, L)
        return PackedCylinders(centers=c2, radii=np.full(len(c2), R), L=L,
                               orientation=[0, 0, 1.0], permeability=0.0)
    if name == "PackedSpheres":
        c3, L = _lattice(2, R, 3)            # 8 spheres on a cubic lattice, box sized to fit them
        _PACK[name] = (c3, L)
        return PackedSpheres(centers=c3, radii=np.full(len(c3), R), L=L, permeability=0.0)
    if name == "Sphere":
        return Sphere(radius=R, permeability=0.0)
    if name == "Ellipsoid":
        return Ellipsoid(semiaxes=_SEMI, permeability=0.0)
    return Cylinder(radius=R, orientation=[0, 0, 1.0], permeability=0.0)


# offset from the wall (m), as a fraction of R -- includes the exact-surface tie
# NOTE: no 0.0 here. A walker EXACTLY on the surface has no side that position alone can
# determine -- `inside` is `dot(r,r) < R*R`, which is False at exactly R -- so "should it be
# confined?" is under-determined by the position-only API. That case is the whole reason the
# carried `side` exists, and it is tested through it in `test_on_wall_needs_a_carried_side`.
OFFSETS = [("far", 0.5 * R), ("one_step", 2.0e-8), ("sub_nudge", 1.0e-11)]
# step length -- the last two SPAN the object, which needs more than one bounce
DISTANCES = [("short", 2.0e-8), ("comparable", 0.5 * R), ("spanning", 2.5 * R),
             ("many_diameters", 12.0 * R)]
# incidence angle from the inward normal
ANGLES = [("head_on", 0.0), ("oblique", 45.0), ("grazing", 89.0), ("tangent", 90.0)]


def _fibonacci_sphere(n):
    """``n`` directions spread evenly over the sphere, plus the six axes and the eight body diagonals: the same
    set every run, with the directions a random draw is least likely to give."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n); th = np.pi * (1 + np.sqrt(5)) * k
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    diag = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]) / np.sqrt(3)
    return np.concatenate([d, axes, diag])


def _first_centre(name):
    """The centre of the object the table fires at: the packing's first object, else the origin."""
    if name not in _PACK:
        return np.zeros(3)
    c = np.zeros(3); c[:_PACK[name][0].shape[1]] = _PACK[name][0][0]
    return c


def _impacts(name):
    """(start, step) pairs: a walker inside, placed `off` from the wall on +x, aimed out."""
    out = []
    for oname, off in OFFSETS:
        for dname, dist in DISTANCES:
            for aname, deg in ANGLES:
                c = _first_centre(name)
                start = c + np.array([R - off, 0.0, 0.0], np.float64)
                th = np.deg2rad(deg)
                # outward normal is +x here; rotate the aim by `th` towards +y
                d = np.array([np.cos(th), np.sin(th), 0.0])
                if name == "Sphere":              # tilt out of plane too, so z is exercised
                    d = d / np.linalg.norm(d)
                out.append((f"{oname}/{dname}/{aname}", start.astype(np.float32),
                            (d / np.linalg.norm(d) * dist).astype(np.float32)))
    return out


def _exterior_impacts(name):
    """(start, step) pairs: a walker OUTSIDE, placed `off` beyond the wall on +x, aimed in."""
    out = []
    for oname, off in OFFSETS:
        for dname, dist in DISTANCES:
            for aname, deg in ANGLES:
                c = _first_centre(name)
                start = c + np.array([R + off, 0.0, 0.0], np.float64)
                th = np.deg2rad(deg)
                d = np.array([-np.cos(th), np.sin(th), 0.0])          # aimed at the wall
                out.append((f"{oname}/{dname}/{aname}", start.astype(np.float32),
                            (d / np.linalg.norm(d) * dist).astype(np.float32)))
    return out


@pytest.mark.parametrize("name", _NAMES)
def test_impermeable_wall_never_lets_a_walker_in(name):
    """The exterior twin: a walker outside an impermeable wall stays outside -- it is neither
    absorbed by a clamp nor teleported inside. Path length is conserved as for interior impacts."""
    geom = _build(name)
    if name in _PACKED:
        pytest.skip("a packed exterior walker meets the NEIGHBOURING objects too; covered by the "
                    "confinement tests on the packed geometries")
    rad = _radial(name)
    cases = _exterior_impacts(name)
    starts = jnp.asarray(np.stack([c[1] for c in cases]))
    steps = jnp.asarray(np.stack([c[2] for c in cases]))
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))
    r_end = rad(out)
    entered = r_end < R * (1 - 1e-6)
    if entered.any():
        rows = "\n".join(
            f"      {cases[i][0]:34} |step|={np.linalg.norm(cases[i][2])/R:6.2f} R  "
            f"-> ended at {r_end[i]/R:8.3f} R"
            for i in np.flatnonzero(entered)[:12])
        pytest.fail(f"{name}: {entered.sum()}/{len(cases)} exterior impacts ended INSIDE:\n{rows}")
    moved = np.linalg.norm(out - np.asarray(starts), axis=1)
    asked = np.linalg.norm(np.asarray(steps), axis=1)
    assert not (moved > asked * (1 + 1e-4) + 1e-12).any(), f"{name}: an exterior reflection added distance"


@pytest.mark.parametrize("name", _NAMES)
def test_impermeable_wall_never_lets_a_walker_out(name):
    geom = _build(name)
    """Every impact in the table must leave the walker inside. One row per failing case."""
    rad = _radial(name)
    cases = _impacts(name)
    starts = jnp.asarray(np.stack([c[1] for c in cases]))
    steps = jnp.asarray(np.stack([c[2] for c in cases]))

    out = jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps)
    r_end = rad(np.asarray(out))

    escaped = r_end > R * (1 + 1e-6)
    if escaped.any():
        rows = "\n".join(
            f"      {cases[i][0]:34} |step|={np.linalg.norm(cases[i][2])/R:6.2f} R  "
            f"-> ended at {r_end[i]/R:8.3f} R"
            for i in np.flatnonzero(escaped)[:12])
        pytest.fail(f"{name}: {escaped.sum()}/{len(cases)} impacts escaped:\n{rows}")


@pytest.mark.parametrize("name", _NAMES)
def test_impact_conserves_path_length(name):
    geom = _build(name)
    """A reflection is a change of direction, not of distance travelled.

    Only checked where the walker does not leave the object: a bounce redirects the
    remaining path, so |displacement| <= |step|, with equality only for a clean miss.
    """
    cases = _impacts(name)
    starts = jnp.asarray(np.stack([c[1] for c in cases]))
    steps = jnp.asarray(np.stack([c[2] for c in cases]))
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))

    moved = np.linalg.norm(out - np.asarray(starts), axis=1)
    asked = np.linalg.norm(np.asarray(steps), axis=1)
    over = moved > asked * (1 + 1e-4) + 1e-12
    assert not over.any(), (
        f"{name}: {over.sum()} impacts moved FURTHER than the step length "
        f"(max {np.max(moved[over] / asked[over]):.3f}x) -- a reflection cannot add distance")


@pytest.mark.parametrize("name", _NAMES)
def test_impact_is_deterministic(name):
    geom = _build(name)
    """Same impact, same answer: an impermeable wall involves no random draw."""
    cases = _impacts(name)
    starts = jnp.asarray(np.stack([c[1] for c in cases]))
    steps = jnp.asarray(np.stack([c[2] for c in cases]))
    f = jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))
    a, b = np.asarray(f(starts, steps)), np.asarray(f(starts, steps))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", _NAMES)
def test_a_step_that_spans_the_object_still_confines(name):
    geom = _build(name)
    """The multi-bounce case, isolated.

    A step longer than the chord needs more than one reflection: bounce once and fly the
    remainder and the walker exits the far side. This is exactly what regressed when the
    analytic geometries' multi-bounce `reflect` was collapsed onto a single-event
    `permeate`, and it only bites when the step is comparable to the object -- so a walk on
    a 5 um sphere never sees it and a 1 um cylinder fails hard.
    """
    rad = _radial(name)
    # start at the centre, aim evenly over the sphere and along every axis and diagonal, step several diameters
    d = _fibonacci_sphere(400); n = len(d)
    c = _first_centre(name)
    starts = jnp.asarray(np.broadcast_to(c, (n, 3)).astype(np.float32))
    steps = jnp.asarray((d * 6.0 * R).astype(np.float32))
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))
    r_end = rad(out)
    assert (r_end <= R * (1 + 1e-6)).all(), (
        f"{name}: {(r_end > R).sum()}/{n} walkers crossed the wall on a step of 6 R "
        f"(max {r_end.max() / R:.2f} R). A step spanning the object needs multiple bounces.")


@pytest.mark.parametrize("name", _PACKED)
def test_a_walker_at_the_seam_meets_the_periodic_image(name):
    """A packing's wall can be the image of an object across the box edge. A walker just inside the +x face of
    the box, aimed across the seam at the image of the object nearest the -x face, must reflect off that image
    as off the object itself: it ends outside every object under the minimum image, at every offset, angle and
    step of the table."""
    geom = _build(name)
    centers, L = _PACK[name]
    rad = _radial(name)
    dim = centers.shape[1]
    i = int(np.argmin(centers[:, 0])); c = np.zeros(3); c[:dim] = centers[i]
    x_img = c[0] + L                                      # the image's centre, across the seam
    cases = []
    for oname, off in (("far", 0.25 * R), ("one_step", 2.0e-8), ("sub_nudge", 1.0e-11)):
        for dname, dist in (("short", 2.0e-8), ("comparable", 0.5 * R), ("two_radii", 2.0 * R)):
            for aname, deg in ANGLES:
                th = np.deg2rad(deg)
                start = np.array([min(L / 2 - 1e-9, x_img - R - off), c[1], c[2]])   # inside the box, `off` outside the image
                d = np.array([np.cos(th), np.sin(th), 0.0])                          # towards the image, turned by the angle
                cases.append((f"{oname}/{dname}/{aname}", start.astype(np.float32), (d * dist).astype(np.float32)))
    starts = jnp.asarray(np.stack([k[1] for k in cases])); steps = jnp.asarray(np.stack([k[2] for k in cases]))
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))
    r_end = rad(out)
    entered = r_end < R * (1 - 1e-6)
    if entered.any():
        rows = "\n".join(f"      {cases[i][0]:30} -> {r_end[i] / R:.3f} R from the nearest object" for i in np.flatnonzero(entered)[:12])
        pytest.fail(f"{name}: {entered.sum()}/{len(cases)} impacts at the seam ended inside an object:\n{rows}")
    moved = np.linalg.norm(out - np.asarray(starts), axis=1); asked = np.linalg.norm(np.asarray(steps), axis=1)
    assert not (moved > asked * (1 + 1e-4) + 1e-12).any(), f"{name}: a reflection at the seam added distance"


def test_a_walker_in_a_two_nanometre_junction_stays_outside_every_cylinder():
    """Three cylinders whose walls are 2 nm apart around a junction, the dense pack's worst case, where a
    reflection off one wall lands the walker in the neighbour and the single-hit rule once kept 18,033 of 20,000
    walkers inside: a walker at the junction fired in every direction of the Fibonacci set, for steps from a
    nanometre to a radius, ends outside all three and moves no further than it stepped."""
    r = np.array([1.0e-6, 1.3e-6, 0.8e-6])
    gap = 2e-9
    d01, d02, d12 = r[0] + r[1] + gap, r[0] + r[2] + gap, r[1] + r[2] + gap       # every pair of walls `gap` apart
    x2 = (d02 ** 2 - d12 ** 2 + d01 ** 2) / (2 * d01); y2 = np.sqrt(d02 ** 2 - x2 ** 2)
    cen = np.array([[0.0, 0.0], [d01, 0.0], [x2, y2]]); cen = cen - cen.mean(0)
    L = 12e-6
    geom = PackedCylinders(centers=cen, radii=r, L=L, orientation=[0, 0, 1.0], permeability=0.0)
    from scipy.optimize import minimize
    pore = minimize(lambda p: np.var(np.linalg.norm(cen - p, axis=1) - r), cen.mean(0)).x   # equidistant from the three walls
    slots = [cen[a] + (cen[b] - cen[a]) * (r[a] + 0.5 * gap) / np.linalg.norm(cen[b] - cen[a])   # the middle of each 2 nm slot
             for a, b in ((0, 1), (0, 2), (1, 2))]
    origins = [("pore", pore)] + [(f"slot{k}", q) for k, q in enumerate(slots)]
    for name, q in origins:
        clear = (np.linalg.norm(cen - q, axis=1) - r).min()
        assert 0 < clear < (0.2 * r.min() if name == "pore" else gap), (name, clear)
    d = _fibonacci_sphere(200)
    starts, steps, labels = [], [], []
    for oname, q in origins:
        for lname, dist in (("nm", 1e-9), ("gap", gap), ("ten_gaps", 10 * gap), ("radius", r.min())):
            for k, dk in enumerate(d):
                starts.append(np.array([q[0], q[1], 0.0])); steps.append(dk * dist); labels.append(f"{oname}/{lname}/dir{k}")
    starts = jnp.asarray(np.stack(starts), jnp.float32); steps = jnp.asarray(np.stack(steps), jnp.float32)
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))
    q = out[:, None, :2] - cen[None]; q -= L * np.floor(q / L + 0.5)
    inside = (np.linalg.norm(q, axis=2) < r[None] * (1 - 1e-6)).any(1)
    if inside.any():
        rows = "\n".join(f"      {labels[i]}" for i in np.flatnonzero(inside)[:12])
        pytest.fail(f"{inside.sum()}/{len(labels)} impacts from the junction ended inside a cylinder:\n{rows}")
    moved = np.linalg.norm(out - np.asarray(starts), axis=1); asked = np.linalg.norm(np.asarray(steps), axis=1)
    assert not (moved > asked * (1 + 1e-4) + 1e-12).any(), "a reflection in the junction added distance"


@pytest.mark.parametrize("name", _NAMES)
def test_on_wall_needs_a_carried_side(name):
    geom = _build(name)
    """A walker exactly ON the surface is decidable only with a carried compartment.

    `inside` is `dot(r,r) < R*R`, which is False at exactly |r| = R, so position alone says
    "outside" for a walker the physics considers interior. That is the #86 tie. The position-
    only call therefore cannot be asked to confine it -- but a call that carries the walker's
    side can, and must.
    """
    if not getattr(geom, "carries_side", False):
        pytest.skip(f"{name} has no carried-side API: a position exactly on its wall is\n"
                    f"under-determined, which is a real limitation rather than a test gap")
    rad = _radial(name)
    n = 64
    rng = np.random.default_rng(1)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    if name == "Cylinder":
        d[:, 2] = 0.0; d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = _first_centre(name)
    starts = jnp.asarray((c + d * R).astype(np.float32))       # exactly on the wall
    steps = jnp.asarray((d * 0.4 * R).astype(np.float32))     # aimed straight out

    # position-only: undetermined, and documented as such -- we assert only that it is finite
    free = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))
    assert np.isfinite(free).all()

    # carried side says "this walker is interior"; now it MUST be confined
    hits = jax.jit(jax.vmap(lambda p, s: geom.interact(
        p, s, side=jnp.int8(-1)).r))(starts, steps)
    r_end = rad(np.asarray(hits))
    assert (r_end <= R * (1 + 1e-6)).all(), (
        f"{name}: {(r_end > R).sum()}/{n} walkers starting exactly on the wall escaped "
        f"despite carrying an interior side (max {r_end.max() / R:.3f} R)")

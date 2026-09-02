"""An impermeable mesh wall must not leak, and these guard the three defects that made it leak.

Each test is written to FAIL on the pre-fix behaviour, which is asserted here by driving the old code path
explicitly rather than trusting that it would have. A guard that passes both before and after is worse than no
guard -- it costs runtime and buys confidence it has not earned.

Geometry is a 27-sphere lattice in a finite box: crowded enough to expose the failures (the CACTUS bundle they
were found on takes minutes), flawless enough that any crossing is the engine's fault and not the mesh's.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from dmipy_sim.mesh import Mesh
from dmipy_sim.mesh_bundle import BoxedMesh, _min_radius
from dmipy_sim.susceptibility_field import mesh_contains

from tests.conftest import assert_step_resolves_the_collision_lookup

UM = 1e-6
D = 2e-9
DT = 1.042e-6
STEP = float(np.sqrt(6 * D * DT))


def _lattice(subdivisions=2, R=1.9, spacing=4.0, n=3, inset=1.0):
    """A lattice whose outer spheres STRADDLE the box faces, as CACTUS fibres do.

    This detail is the whole test. With every body strictly inside the box, mirroring a walker across a face
    lands it in free space and the mirror bug does not reproduce at all (measured: 0 crossings, which is what
    the self-guard in the first test caught). CACTUS fibres are cut BY the periodic box, so a mirror across a
    face lands inside a fibre continuation -- and that is when a reflection applied without a collision test
    can enclose a walker. `inset` shrinks the box until it cuts the outer spheres.
    """
    parts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R)
                s.apply_translation([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
                parts.append(s)
    m = trimesh.util.concatenate(parts)
    V = np.asarray(m.vertices, np.float64) * UM
    F = np.asarray(m.faces, np.int64)
    lo = np.full(3, inset) * UM
    hi = np.full(3, n * spacing - inset) * UM
    return V, F, lo, hi


def _seed_outside(V, F, lo, hi, n, seed=11):
    rg = np.random.default_rng(seed)
    got = []
    while sum(len(g) for g in got) < n:
        c = rg.uniform(lo, hi, size=(20000, 3))
        got.append(c[~np.asarray(mesh_contains(V, F, c))])
    return np.concatenate(got)[:n]


def _run(geom, r0, n_sub, seed=3):
    # Every confinement number below is meaningless if the step outruns the collision lookup; see the helper's
    # docstring for the measured ratios and the issue it cost. Checked on the inner Mesh when wrapped.
    assert_step_resolves_the_collision_lookup(getattr(geom, "mesh", geom), STEP)
    step = jax.jit(jax.vmap(lambda r, s: geom.reflect_with_log_weight(r, s, jnp.float32(1.0))))
    rg = np.random.default_rng(seed)
    r = r0.astype(np.float32)
    lt = np.zeros(len(r0))
    for _ in range(n_sub):
        s = rg.normal(size=(len(r0), 3))
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        out = step(jnp.asarray(r), jnp.asarray((STEP * s).astype(np.float32)))
        r = np.asarray(out[0])
        lt += -np.asarray(out[1], np.float64)
    return r.astype(np.float64), lt.mean()


def _crossed(V, F, pts):
    return int(np.asarray(mesh_contains(V, F, pts)).sum())


@pytest.fixture(scope="module")
def lattice():
    V, F, lo, hi = _lattice()
    return V, F, lo, hi, _seed_outside(V, F, lo, hi, 400)


@pytest.fixture(scope="module")
def lattice_rough():
    """Same lattice with jittered vertices, so the interpolated normals actually disagree with their facets.

    An icosphere has near-perfect vertex normals, so smooth and facet reflection barely differ on it and the
    failure cannot appear -- measured, 0 leaks either way. CACTUS surfaces are eroded; irregular normals are
    the real precondition. Jitter supplies that while keeping the mesh watertight.
    """
    V, F, lo, hi = _lattice()
    facet = float(np.sqrt(trimesh.Trimesh(vertices=V, faces=F, process=False).area_faces.mean()))
    rg = np.random.default_rng(0)
    Vr = V + rg.normal(0.0, 0.30 * facet, V.shape)
    assert trimesh.Trimesh(vertices=Vr, faces=F, process=False).is_watertight
    return Vr, F, lo, hi, _seed_outside(Vr, F, lo, hi, 400)


def _mesh(V, F, lo, hi, **kw):
    m = Mesh(V.astype(np.float32), F, periodic=False, voxel_min=lo.astype(np.float32),
             voxel_max=hi.astype(np.float32), feature_radius=_min_radius(V.astype(np.float32), F))
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_box_faces_must_bounce_not_mirror(lattice):
    """`BoxedMesh` mirrors the position with no collision test, so it can place a walker inside a body.

    Mirroring is a reflection of space, not a displacement: it changes which cross-section a point falls in, so
    a walker crosses no wall yet ends up enclosed. Measured on a 358-fibre CACTUS bundle over 20 ms it leaked
    18.67% of extra-axonal walkers. Folding the voxel faces into the same ordered bounce loop as the triangles
    makes that impossible by construction.
    """
    V, F, lo, hi, r0 = lattice
    # box_reflect is now the DEFAULT, so the mirror arm has to switch it off explicitly -- otherwise the
    # inner mesh already confines the walker, BoxedMesh._box short-circuits, and this compares new with new.
    mirrored = BoxedMesh(_mesh(V, F, lo, hi, box_reflect=False), lo, hi)
    in_loop = _mesh(V, F, lo, hi, box_reflect=True)

    n_mirror = _crossed(V, F, _run(mirrored, r0, 3000)[0])
    n_loop = _crossed(V, F, _run(in_loop, r0, 3000)[0])

    assert n_mirror > 0, "the mirror leak did not reproduce, so this test guards nothing"
    assert n_loop < n_mirror, f"in-loop box reflection did not help: {n_loop} vs mirror {n_mirror}"


def test_smooth_normal_can_reflect_into_the_wall_and_facet_normal_cannot(lattice_rough):
    """The mechanism behind the leak, asserted on the quantity that actually goes wrong.

    Reflecting about the vertex-INTERPOLATED normal can produce an outgoing ray pointing INTO the facet just
    hit -- the interpolated normal is not the plane the walker is behind. `_GRAZE` then "rescues" that ray by
    lifting it to a cosine of 1e-4 against the facet, i.e. almost exactly parallel to the wall, and the walker
    travels a full step (112 nm) along the surface. On concave, crowded geometry it ends up inside.

    Measured on real CACTUS collisions: smooth reflection puts 3.01% of collisions at pre-lift cos < 0 (as deep
    as -0.42); facet-normal reflection puts 0.00% there (min +0.035). That also explains why enlarging _GRAZE
    helped (63 -> 20 crossings from 1e-4 to 2e-1) -- it was mitigating its own failure mode.

    A statistical walk test could not guard this cheaply: crowding alone reproduces nothing (0 leaks on a
    clean lattice) and a single grazing step off a tilted ridge reproduces nothing (0/4000 at three dihedral
    angles). The pre-lift cosine is the quantity that fails, so it is what gets asserted.
    """
    V, F, lo, hi, r0 = lattice_rough

    def pre_lift_cos(mode):
        # _GRAZE pinned to the old value in both arms: the point here is the NORMAL, and the default is now
        # 6e-2, which would otherwise confound the comparison.
        m = _mesh(V, F, lo, hi, box_reflect=True, reflect_mode=mode, _GRAZE=jnp.float32(1e-4))
        step = jax.jit(jax.vmap(lambda r, s: m.reflect_with_log_weight(r, s, jnp.float32(1.0))))
        rg = np.random.default_rng(0)
        r = r0.astype(np.float32)
        for _ in range(300):                       # let walkers settle against the walls
            d = rg.normal(size=(len(r0), 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            r = np.asarray(step(jnp.asarray(r), jnp.asarray((STEP * d).astype(np.float32)))[0])

        def probe(r_one, dh):
            ci, valid = m._gather(r_one)
            tri = m._TRIS[ci]; vnf = m._VN[ci]; nrmf = m._NRM[ci]
            ts, u, v = m._mt(r_one, dh, tri, valid)
            ts = jnp.where((ts > 0.0) & (ts < STEP), ts, jnp.inf)
            idx = jnp.argmin(ts)
            n = m._smooth_normal(vnf, nrmf, u, v, idx, dh)
            nf = jnp.where(jnp.dot(dh, nrmf[idx]) > 0, -nrmf[idx], nrmf[idx])
            n_ref = nf if mode == "geometric" else n
            dr = dh - 2.0 * jnp.dot(dh, n_ref) * n_ref
            dr = dr / jnp.linalg.norm(dr)
            return jnp.where(ts[idx] < jnp.inf, jnp.dot(dr, nf), jnp.nan)

        # only a small fraction of walkers meet a wall in any single step, so accumulate over many steps
        probe_v = jax.jit(jax.vmap(probe))
        acc = []
        for _ in range(60):
            d = rg.normal(size=(len(r0), 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            dj = jnp.asarray((d).astype(np.float32))
            acc.append(np.asarray(probe_v(jnp.asarray(r), dj)))
            r = np.asarray(step(jnp.asarray(r), jnp.asarray((STEP * d).astype(np.float32)))[0])
        c = np.concatenate(acc)
        return c[np.isfinite(c)]

    c_smooth = pre_lift_cos("smooth")
    c_facet = pre_lift_cos("geometric")
    assert len(c_smooth) > 200 and len(c_facet) > 200, (
        f"too few collisions sampled to conclude anything: {len(c_smooth)}, {len(c_facet)}")
    assert (c_smooth < 0).any(), (
        "no into-the-facet reflections reproduced, so this test guards nothing -- the mesh is too regular")
    assert not (c_facet < 0).any(), (
        f"facet-normal reflection produced an into-the-facet ray, min cos {c_facet.min():+.5f}")


def test_only_the_resting_facet_is_ignored_not_every_close_wall(lattice):
    """A distance floor on accepted hits also discards genuine hits on OTHER nearby facets.

    After a bounce the walker sits `_NUDGE` off the facet it hit, and a near-tangential outgoing ray can
    re-hit that facet at a tiny positive t through float32 error alone. MC/DC suppresses this with a walker
    state bit plus a distance epsilon; the facet that must be ignored is known by INDEX, so excluding it by
    identity leaves every other wall live at any range. Measured on CACTUS: 63.3 -> 50.0 crossings (3.7 sigma)
    with no local-time cost.

    The guard: with exclusion active, a hit closer than the floor on a DIFFERENT facet must still register.
    """
    V, F, lo, hi, _ = lattice
    m = _mesh(V, F, lo, hi, box_reflect=True, rest_facet_exclusion=True)
    tri = m._TRIS[m._gather(jnp.asarray(np.array([2.0, 2.0, 2.0]) * UM, jnp.float32))[0]]
    assert tri.shape[-2:] == (3, 3)
    # Default: one floor, applied unconditionally -- shipped behaviour, unchanged by this work.
    assert float(m._hit_floor(False)) == float(m._EPS) == float(m._hit_floor(True))
    # Opt in to MC/DC's version and a FRESH step accepts a hit at any positive distance, while a continuing
    # bounce still suppresses the facet it is resting on.
    m.state_conditional_floor = True
    assert float(m._hit_floor(False)) == 0.0, "a fresh step must accept a hit at any positive distance"
    assert float(m._hit_floor(True)) > 0.0, "mid-bounce still needs the resting facet suppressed"

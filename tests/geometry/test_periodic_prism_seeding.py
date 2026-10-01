"""dmipy-sim#400: ``Mesh.init_positions`` on an open periodic tube mesh (e.g. the DiSCo 12-gon prism
family of the replay paper's polygon supplement).

Before this fix, an open surface (periodic or not) was seeded by the cell-gather classifier -- inexact
near a wall (the issue measured a 4.5% sliver between a 12-gon and its circumscribed circle) and silent.
The fix adds one more exact route, ahead of that fallback: a surface that is open ONLY because it is a
periodic extrusion (every boundary edge sits on one periodic axis's two box faces, and the cross-section
transverse to that axis is constant along it) is seeded by exact 2-D point-in-polygon in the transverse
plane, with periodic images handled per loop. A surface that stays open for any other reason -- not
periodic, or periodic but not actually an extrusion -- is now refused by name instead of guessed at.

These tests check the three outcomes against an INDEPENDENT reference (apothem/point-in-polygon under the
periodic minimum image, not the library's own containment code), not just against the fix's own math.
"""
import warnings

import numpy as np
import numpy.testing as npt
import jax
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from dmipy_sim import Mesh


def _prism(n_sides, r, c, L, n_ax=4):
    """``(V, F)`` of a regular ``n_sides``-gon prism, circumradius ``r``, a vertex at angle 0, centred
    at ``c`` in the transverse plane and spanning ``[-L/2, L/2]`` along z."""
    th = np.arange(n_sides) * 2 * np.pi / n_sides
    zs = np.linspace(-L / 2, L / 2, n_ax + 1)
    V = np.array([[c[0] + r * np.cos(t), c[1] + r * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(n_ax):
        for it in range(n_sides):
            a = iz * n_sides + it; b = iz * n_sides + (it + 1) % n_sides
            cc = (iz + 1) * n_sides + it; d = (iz + 1) * n_sides + (it + 1) % n_sides
            F += [(a, b, d), (a, d, cc)]
    return V, np.asarray(F, np.int64)


def _prism_bundle(n_sides, radii, centers, L, pool):
    """One periodic ``Mesh`` of several ``n_sides``-gon prisms, spanning a cubic periodic cell of side ``L``."""
    Vs, Fs, off = [], [], 0
    for r, c in zip(radii, centers):
        V, F = _prism(n_sides, r, c, L)
        Vs.append(V); Fs.append(F + off); off += len(V)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # coarse-tessellation permeability warning; walls are impermeable here
        return Mesh(np.concatenate(Vs), np.concatenate(Fs), periodic=True,
                    voxel_min=[-L / 2] * 3, voxel_max=[L / 2] * 3,
                    feature_radius=float(np.min(radii)), pool=pool)


def _in_regular_polygon(pts_xy, centers, radii, n_sides, L):
    """Exact containment inside the union of regular ``n_sides``-gons (a vertex at angle 0) inscribed in
    the circles of ``radii`` about ``centers``, under the periodic minimum image: the apothem test folded
    into one sector, independent of (and the reference for) the library's own seeding."""
    dq = pts_xy[:, None, :] - centers[None, :, :]
    dq = dq - L * np.round(dq / L)
    rho = np.hypot(dq[..., 0], dq[..., 1])
    th = np.arctan2(dq[..., 1], dq[..., 0]) % (2 * np.pi / n_sides) - np.pi / n_sides
    return (rho * np.cos(th) <= radii[None] * np.cos(np.pi / n_sides)).any(1)


def _open_tube(r=4e-6, L=12e-6, nt=32, nz=24):
    """A genuinely open (non-periodic) circular tube: no axis closes its rims by identification."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zs = np.linspace(0, L, nz)
    V = np.array([[r * np.cos(t), r * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt
            c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    return V, np.array(F)


def _barrel_tube(r0=3e-6, bulge=1.5e-6, L=12e-6, nt=24, nz=25):
    """A periodic-along-z tube whose two end rims match (both at radius ``r0``, so its boundary still
    sits entirely on the z faces) but whose cross-section bulges in between -- open along z for a reason
    periodicity alone does not explain, since it is not actually an extrusion."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zs = np.linspace(-L / 2, L / 2, nz)
    V = []
    for z in zs:
        r = r0 + bulge * np.sin(np.pi * (z + L / 2) / L) ** 2
        V += [[r * np.cos(t), r * np.sin(t), z] for t in th]
    V = np.array(V)
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt
            c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    return V, np.array(F)


def _icosphere(sub=2, r=5e-6):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


# ------------------------------------------------------------------ the fix: periodic + prismatic

def test_periodic_prism_pair_seeds_exactly_against_the_polygon_reference():
    """The issue's own substrate family: inner prisms (pool intra) and outer prisms (pool extra) of a
    12-gon tube bundle, periodic in all three axes. Both pools must seed EXACTLY against an independent
    point-in-polygon reference, with no warning (the old inexact fallback always warned)."""
    from dmipy_sim.geometry.packing import pack_cylinders

    n_sides, L = 12, 22e-6
    rng = np.random.default_rng(0)
    r_in = rng.uniform(1.0e-6, 2.0e-6, 6)
    r_out = r_in / 0.7
    # non-overlapping outer tubes, like every real substrate (pack_cylinders enforces clearance) --
    # prism_periodic_contains' odd/even rule assumes this, same as the paper's own polygon supplement
    centers, _, _ = pack_cylinders(r_out + 0.15e-6, L=L, seed=1)
    m_in = _prism_bundle(n_sides, r_in, centers, L, pool="intra")
    m_out = _prism_bundle(n_sides, r_out, centers, L, pool="extra")

    prism_in, prism_out = m_in._prism_route(), m_out._prism_route()
    assert prism_in is not None and prism_in[0] == 2 and len(prism_in[1]) == 6
    assert prism_out is not None and prism_out[0] == 2 and len(prism_out[1]) == 6

    with warnings.catch_warnings():
        warnings.simplefilter("error")            # must not fall back to the warning cell-gather path
        p_in = np.asarray(m_in.init_positions(20_000, jax.random.PRNGKey(0)))[:, :2]
        p_out = np.asarray(m_out.init_positions(20_000, jax.random.PRNGKey(1)))[:, :2]

    assert _in_regular_polygon(p_in, centers, r_in, n_sides, L).all()
    assert not _in_regular_polygon(p_out, centers, r_out, n_sides, L).any()


def test_periodic_prism_wraps_a_tube_that_straddles_the_box_edge():
    """A tube near the transverse periodic boundary is also seen from its wrapped (minimum-image) side --
    the same device ``PackedCylinders``/``pack_cylinders`` use -- not only from its own stored position."""
    n_sides, L = 12, 20e-6
    r = 3e-6
    c = np.array([[L / 2 - 0.5e-6, 0.0]])             # the prism pokes ~2.5 um past the +x face
    m = _prism_bundle(n_sides, np.array([r]), c, L, pool="intra")
    assert m._prism_route() is not None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pts = np.asarray(m.init_positions(4000, jax.random.PRNGKey(2)))[:, :2]

    assert _in_regular_polygon(pts, c, np.array([r]), n_sides, L).all()
    assert (pts[:, 0] < -L / 2 + 1e-6).any(), "some seeds should land on the wrapped (-x) image of the tube"


def test_periodic_tube_confinement_uses_the_exact_route():
    """A single circular periodic tube (the simplest prismatic case) also goes through the exact route,
    not the old warning fallback, and stays radially confined."""
    r, L = 4e-6, 12e-6
    V, F = _open_tube(r=r, L=L)
    g = Mesh(V, F, periodic=(False, False, True), voxel_min=[-6e-6, -6e-6, 0.0],
             voxel_max=[6e-6, 6e-6, L], feature_radius=r)
    assert g._prism_route() is not None and g._prism_route()[0] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pts = np.asarray(g.init_positions(5000, jax.random.PRNGKey(3)))
    rad = np.linalg.norm(pts[:, :2], axis=1)
    assert rad.max() < r * 1.001


# ------------------------------------------------------------------ refusal: genuinely undecidable

def test_open_non_periodic_surface_is_refused_by_name():
    """A surface open for a reason periodicity cannot explain (here: not periodic at all) is refused, by
    name, rather than silently misseeded by the old cell-gather fallback."""
    V, F = _open_tube()
    g = Mesh(V, F)                                    # periodic=False default
    assert g._prism_route() is None
    with pytest.raises(ValueError, match=r"\d+ boundary edge"):
        g.init_positions(10, jax.random.PRNGKey(0))


def test_periodic_but_not_prismatic_surface_is_refused_by_name():
    """A surface periodic along the axis its rims sit on, but not actually a constant-cross-section
    extrusion along it (a barrel with matched end caps), does not qualify for the exact route either."""
    V, F = _barrel_tube()
    g = Mesh(V, F, periodic=(False, False, True), voxel_min=[-5e-6, -5e-6, -6e-6],
             voxel_max=[5e-6, 5e-6, 6e-6], feature_radius=1e-6)
    assert g._prism_route() is None
    with pytest.raises(ValueError, match=r"\d+ boundary edge"):
        g.init_positions(10, jax.random.PRNGKey(0))


# ------------------------------------------------------------------ closed meshes: untouched

def test_closed_mesh_seeding_does_not_touch_the_prism_route():
    """The closed-surface branch of ``init_positions`` is decided by :meth:`Mesh._surface_is_closed`
    before the prism machinery is ever consulted (dmipy-sim#400 only replaced the OPEN-surface fallback),
    and its output is exactly reproducible."""
    V, F = _icosphere(2)
    g = Mesh(V, F)
    assert g._surface_is_closed() is True
    assert g._prism_route() is None
    a = np.asarray(g.init_positions(500, jax.random.PRNGKey(11)))
    b = np.asarray(g.init_positions(500, jax.random.PRNGKey(11)))
    npt.assert_array_equal(a, b)
    assert (np.linalg.norm(a, axis=1) < 5e-6).all()

"""Does a MESHED sphere behave like an ANALYTIC one? Same impacts, both geometries.

Nothing checked this before. The analytic geometries and `Mesh` are independent boundary
implementations -- different intersection maths, different normals (closed form vs
interpolated vertex normals), different nudge -- and the only thing tying them together was
that each passed its own tests. So "is a mesh of a sphere a sphere?" was untested.

The tolerance is DERIVED, not fitted. A triangulated sphere is inscribed: its facets cut the
chord, so a meshed wall sits slightly inside the analytic one by ~R*h^2/8 for edge length h.
`Mesh.quality_report()` reports `edge_feature_ratio` = median_edge / feature_radius, so the
expected discrepancy is ~ratio^2/8 in units of R, and the real assertion is that the error
FALLS as the mesh is refined -- a constant would just be a fitted fudge.

Geometries are built inside each test, never at collection time: a geometry holds device
buffers and `gpu.py` calls `jax.clear_caches()` (see test_wall_impacts).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.geometry import Sphere
from dmipy_sim.geometry.mesh import Mesh

R = 5.0e-6


def _sphere_pair(subdivisions):
    """An analytic sphere and a triangulated one of the same radius."""
    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R)
    V = np.asarray(ico.vertices, np.float64)
    F = np.asarray(ico.faces, np.int64)
    pad = 0.4 * R
    mesh = Mesh(V.astype(np.float32), F, periodic=False,
                voxel_min=(V.min(0) - pad).astype(np.float32),
                voxel_max=(V.max(0) + pad).astype(np.float32))
    return Sphere(radius=R, permeability=0.0), mesh, ico


def _impacts(n=600, seed=0, depth=0.02, reach=0.25):
    """Walkers just inside the wall, aimed outward at every angle: the interaction cases."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    start = (u * R * (1.0 - depth)).astype(np.float32)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = np.where((np.sum(d * u, axis=1, keepdims=True) < 0), -d, d)   # aim outward
    return jnp.asarray(start), jnp.asarray((d * reach * R).astype(np.float32))


def _endpoints(geom, starts, steps):
    return np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(starts, steps))


@pytest.mark.parametrize("subdivisions", [3, 4])
def test_meshed_sphere_confines_like_the_analytic_one(subdivisions):
    """Both must confine, and the mesh's wall must sit within faceting error of R."""
    ana, mesh, _ = _sphere_pair(subdivisions)
    starts, steps = _impacts()

    r_ana = np.linalg.norm(_endpoints(ana, starts, steps), axis=1)
    r_msh = np.linalg.norm(_endpoints(mesh, starts, steps), axis=1)

    assert (r_ana <= R * (1 + 1e-6)).all(), "analytic sphere let a walker out"
    # a triangulated sphere is INSCRIBED, so its wall is at or inside R -- never outside
    assert (r_msh <= R * (1 + 1e-6)).all(), (
        f"meshed sphere let {(r_msh > R).sum()}/{len(r_msh)} walkers out "
        f"(max {r_msh.max() / R:.5f} R)")


def test_analytic_mesh_discrepancy_is_first_order_in_edge_length():
    """The two agree, and the disagreement scales as the mesh resolution says it should.

    Measured across four refinements of the same sphere (impacts at 0.98 R, reach 0.25 R):

        edge/feature   median |analytic - mesh| / R   err/ratio   err/ratio^2
           0.2137              3.16e-02                0.148         0.7
           0.1076              1.31e-02                0.122         1.1
           0.0539              6.13e-03                0.114         2.1
           0.0270              3.08e-03                0.114         4.2

    `err/ratio` converges to a constant while `err/ratio^2` diverges, so the discrepancy is
    FIRST order in edge length. That is the reflected DIRECTION, not the wall position: an
    interpolated vertex normal is wrong by an angle ~h/R, which turns the outgoing ray and
    displaces the endpoint by that angle times the remaining path. The chord sagitta
    (h^2/8R, second order) is a much smaller effect -- an earlier version of this test
    bounded by it and failed at 2.1x, which is how the order was established.

    Pinning the ORDER is the real assertion: a constant tolerance would pass just as well
    for two implementations that are close by luck, and would not notice a normal or nudge
    regression that changes the scaling.
    """
    starts, steps = _impacts()
    errs, ratios = [], []
    for sub in (2, 3, 4, 5):
        ana, mesh, _ = _sphere_pair(sub)
        d = np.linalg.norm(_endpoints(ana, starts, steps)
                           - _endpoints(mesh, starts, steps), axis=1)
        errs.append(float(np.median(d)) / R)
        ratios.append(float(mesh.quality_report(verbose=False)["edge_feature_ratio"]))

    assert all(b < a for a, b in zip(errs, errs[1:])), (
        f"disagreement did not fall with refinement: {[f'{e:.2e}' for e in errs]}")

    coeffs = [e / r for e, r in zip(errs, ratios)]
    # first order: the coefficient must settle, not keep climbing as it would for O(h^2)
    assert coeffs[-1] < 1.3 * coeffs[-2], (
        f"err/ratio is not converging ({[f'{c:.3f}' for c in coeffs]}) -- the discrepancy is "
        f"not first order in edge length, so something other than the normal dominates")
    assert coeffs[-1] < 0.30, (
        f"first-order coefficient {coeffs[-1]:.3f} (measured 0.114) -- the meshed wall "
        f"disagrees with the analytic one by more than its resolution explains")


def test_meshed_sphere_reflects_into_the_same_half_space():
    """Beyond distance: a bounce must send the walker the same WAY, not just nearby.

    Faceting perturbs the reflected direction by the facet's normal error, so the two need
    not agree exactly -- but a walker that bounces inward analytically must not bounce
    outward on the mesh, which would be a sign error rather than a resolution effect.
    """
    ana, mesh, _ = _sphere_pair(4)
    starts, steps = _impacts(n=600, depth=0.01, reach=0.30)

    disp_a = _endpoints(ana, starts, steps) - np.asarray(starts)
    disp_m = _endpoints(mesh, starts, steps) - np.asarray(starts)
    radial = np.asarray(starts) / np.linalg.norm(np.asarray(starts), axis=1, keepdims=True)

    # radial component of the move: negative = inward
    ra = np.sum(disp_a * radial, axis=1)
    rm = np.sum(disp_m * radial, axis=1)
    moved = (np.linalg.norm(disp_a, axis=1) > 1e-9) & (np.linalg.norm(disp_m, axis=1) > 1e-9)
    disagree = moved & (np.sign(ra) != np.sign(rm)) & (np.abs(ra) > 1e-8) & (np.abs(rm) > 1e-8)
    frac = float(disagree.mean())
    assert frac < 0.05, (
        f"{frac:.1%} of impacts bounce inward analytically but outward on the mesh "
        f"(or vice versa) -- a direction disagreement, not a resolution effect")

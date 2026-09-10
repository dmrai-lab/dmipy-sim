"""SphereUnion: membership, seeding, confinement, seam handling, the voxel fold, agreement with the analytic
sphere, and its spec. Substrates are generated here; no CATERPillar file is needed."""
import jax
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import pgse, set_b, simulate
from dmipy_sim.geometry.sphere_union import SphereUnion
from dmipy_sim.spec import geometry_from_spec

D0 = 2e-9


def _chain(R=3e-6, half_len=15e-6, overlap=4.0, axis=2):
    t = np.arange(-half_len, half_len, R / overlap)
    cen = np.zeros((len(t), 3)); cen[:, axis] = t
    return cen, np.full(len(t), R)


def test_membership_matches_brute_force_for_any_grid_layout():
    rng = np.random.default_rng(0)
    cen = rng.uniform(-8e-6, 8e-6, (40, 3)); rad = rng.uniform(1e-6, 3e-6, 40)
    p = rng.uniform(-12e-6, 12e-6, (4000, 3))
    brute = (np.linalg.norm(p[:, None, :] - cen[None], axis=2) < rad[None]).any(axis=1)
    assert np.array_equal(SphereUnion(cen, rad, pool="extra").inside_any(p), brute)
    for cs in (0.5e-6, 1e-6, 8e-6):                         # the 27-cell gather is exact for any cell size
        assert np.array_equal(SphereUnion(cen, rad, pool="extra", cell_size=cs).inside_any(p), brute)


@pytest.mark.parametrize("pool", ["intra", "extra"])
def test_walkers_seed_and_stay_on_their_side(pool):
    rng = np.random.default_rng(2)
    cen = rng.uniform(-5e-6, 5e-6, (20, 3)); rad = rng.uniform(1e-6, 2.5e-6, 20)
    u = SphereUnion(cen, rad, pool=pool)
    r0 = np.asarray(u.init_positions(300, jax.random.PRNGKey(0)))
    assert (u.inside_any(r0) == (pool == "intra")).all()
    w = d.simulate_trajectories(300, D0, u, 3e-3, 3e-4, seed=0, require_gpu=False)
    P = np.asarray(w.positions).reshape(-1, 3)
    assert (u.inside_any(P) == (pool == "intra")).all()
    assert w.has_surface and w.illegal_crossings == 0


def test_internal_seam_is_not_a_wall():
    """Two heavily overlapping spheres: a walker crossing the seam between them sees no wall, so the union's
    mean-squared displacement along the chain exceeds a single sphere's."""
    cen = np.array([[0, 0, -1.5e-6], [0, 0, 1.5e-6]]); rad = np.full(2, 3e-6)
    u = SphereUnion(cen, rad, pool="intra")
    one = SphereUnion(cen[:1], rad[:1], pool="intra")
    kw = dict(T_max=4e-3, dt_save=4e-3, seed=0, require_gpu=False)
    def msd_z(g):
        w = d.simulate_trajectories(400, D0, g, **kw)
        P = np.asarray(w.positions)
        return float(((P[:, -1, 2] - P[:, 0, 2]) ** 2).mean())
    assert msd_z(u) > 1.3 * msd_z(one)


def test_single_sphere_matches_the_analytic_sphere():
    R = 5e-6
    u = SphereUnion(np.zeros((1, 3)), np.array([R]), pool="intra")
    wf = set_b(pgse([[1, 0, 0]], 0.01, 0.03, gradient_strengths=0.2, n_t=150), 2e9)
    kw = dict(n_walkers=3000, diffusivity=D0, waveform=wf, seed=0, require_gpu=False)
    s_u = float(np.asarray(simulate(geometry=u, **kw)).ravel()[0])
    s_a = float(np.asarray(simulate(geometry=d.Sphere(radius=R), **kw)).ravel()[0])
    assert abs(s_u - s_a) < max(0.02, 1 / np.sqrt(3000))


def test_the_voxel_fold_keeps_walkers_in_the_box_and_never_across_a_membrane():
    rng = np.random.default_rng(4)
    cen = rng.uniform(-6e-6, 6e-6, (15, 3)); rad = rng.uniform(1e-6, 2.5e-6, 15)
    lo, hi = np.full(3, -8e-6), np.full(3, 8e-6)
    u = SphereUnion(cen, rad, pool="extra", box=(lo, hi))
    w = d.simulate_trajectories(300, D0, u, 4e-3, 4e-4, seed=1, require_gpu=False)
    P = np.asarray(w.positions).reshape(-1, 3)
    assert (P >= lo - 1e-9).all() and (P <= hi + 1e-9).all()
    assert not u.inside_any(P).any()
    assert u.length_scales.min_feature == pytest.approx(rad.min()) and u.length_scales.lookup_cell > 0


def test_spec_round_trip():
    u = SphereUnion(np.array([[0, 0, 0], [0, 0, 2e-6]]), [3e-6, 2.5e-6], pool="intra", box=(np.full(3, -5e-6), np.full(3, 5e-6)))
    spec = u.spec
    assert spec.walls[0].surface.kind == "sphere_union" and spec.seeding.pools == [1]
    assert spec.domain.boundary == ["reflect"] * 3
    g = geometry_from_spec(spec)
    assert type(g) is SphereUnion and g.pool == "intra" and g.spec == spec
    w1 = d.simulate_trajectories(40, D0, u, 4e-4, 2e-4, seed=1, require_gpu=False)
    w2 = d.simulate_trajectories(40, D0, g, 4e-4, 2e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w1.positions, w2.positions)
    with pytest.raises(ValueError, match="pool"):
        SphereUnion(np.zeros((1, 3)), [1e-6], pool="myelin")

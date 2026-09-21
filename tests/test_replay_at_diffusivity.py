"""A pack replays at another diffusivity: the save grid divided by the ratio, nothing decoded (dmipy-sim#289).

A path walked at ``D`` and read on a grid divided by ``a`` is a path walked at ``a D``. The stored coefficients
are duration-agnostic, so the replay at ``a D`` is a view of the same arrays on the new grid, and every channel
follows in its own space. The checks are exact: a walk at ``2 D`` from the same seed on the halved grid is the
same walk to the bit, so the view of the pack walked at ``D`` must replay as the pack walked at ``2 D`` does.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

D0, T, DT = 2.0e-9, 40e-3, 5e-4


def _pack(tmp, geometry, D, T_, dt, name, seed=3, **kw):
    walk = d.simulate_trajectories(600, D, geometry, T_, dt, seed=seed, require_gpu=False)
    out = tmp / f"{name}.rpk"
    build_replay_pack(walk, id=f"test/{name}", license="x", citation="x", K=12, out_path=str(out), **kw)
    return read_rpk(str(out))


@pytest.fixture(scope="module")
def free(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("free")
    return _pack(tmp, d.FreeDiffusion(), D0, T, DT, "d"), _pack(tmp, d.FreeDiffusion(), 2 * D0, T / 2, DT / 2, "2d")


@pytest.fixture(scope="module")
def permeable(tmp_path_factory):
    """A permeable cylinder: the walk at (D, kappa) and, from the same seed on the halved grid, at (2D, 2kappa) --
    the same crossings, since the probability per encounter goes as kappa sqrt(dt / D)."""
    tmp = tmp_path_factory.mktemp("perm")
    K = 2e-5
    return (_pack(tmp, d.Cylinder(radius=4e-6, orientation=(0, 0, 1), permeability=K), D0, T, DT, "d"),
            _pack(tmp, d.Cylinder(radius=4e-6, orientation=(0, 0, 1), permeability=2 * K), 2 * D0, T / 2, DT / 2, "2d"), K)


@pytest.fixture(scope="module")
def pore(tmp_path_factory):
    """A cylinder with a surface relaxivity, so the surface tier is live and its 1/D divisor matters."""
    tmp = tmp_path_factory.mktemp("pore")
    g = d.Cylinder(radius=4e-6, orientation=(0, 0, 1), surface_relaxivity_t2=2e-5)
    return _pack(tmp, g, D0, T, DT, "d"), _pack(tmp, g, 2 * D0, T / 2, DT / 2, "2d")


def _seq(TE):
    return sequences.pgse([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 4e-3, 10e-3, bvalues=[1.5e9, 1.5e9], TE=TE, n_t=81)


def test_the_view_is_the_walk_at_the_other_diffusivity_to_the_bit(free):
    pk_d, pk_2d = free
    view = pk_d.at_diffusivity(2 * D0)
    assert view.diffusivity == 2 * D0 and view.dt == pytest.approx(DT / 2) and view.n_t == pk_d.n_t
    assert view.temporal_bandwidth_hz == pytest.approx(2 * pk_d.temporal_bandwidth_hz)
    for k in ("pos_x", "pos_y", "pos_z"):
        if k in pk_d.arrays:
            np.testing.assert_array_equal(pk_d.arrays[k], pk_2d.arrays[k])       # the same walk, coded
    seq = _seq(T / 2)
    np.testing.assert_allclose(view.replay(seq, complex_signal=True), pk_2d.replay(seq, complex_signal=True), rtol=1e-12)
    np.testing.assert_allclose(np.abs(view.replay(seq)), np.exp(-np.asarray(seq.b()) * 2 * D0), atol=4.0 / np.sqrt(600))
    assert view.meta["provenance"]["diffusivity_scaled"]["a"] == 2.0


def test_tissue_d_is_the_knob_on_every_route(free):
    pk_d, pk_2d = free
    seq = _seq(T / 2)
    t = Tissue(D=2 * D0)
    np.testing.assert_allclose(pk_d.replay(seq, tissue=t, complex_signal=True), pk_2d.replay(seq, complex_signal=True), rtol=1e-12)
    w1, e1, E1 = pk_d.walker_signals(seq, tissue=t)
    w2, e2, E2 = pk_2d.walker_signals(seq)
    np.testing.assert_allclose(E1, E2, rtol=1e-12)
    np.testing.assert_allclose(pk_d.walker_phases(seq, tissue=t)[2], pk_2d.walker_phases(seq)[2], rtol=1e-12)
    r1 = pk_d.pose_response(seq, tissue=t)
    r2 = pk_2d.pose_response(seq)
    np.testing.assert_allclose(r1.coeffs, r2.coeffs, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(pk_d.replay_bloch(seq, tissue=t, complex_signal=True),
                               pk_2d.replay_bloch(seq, complex_signal=True), rtol=1e-9)
    # and the walked diffusivity, or none, replays the pack as it is
    np.testing.assert_array_equal(pk_d.replay(seq, tissue=Tissue(D=D0)), pk_d.replay(seq))


def test_the_surface_term_changes_with_the_diffusivity_and_the_replay_is_still_exact(pore):
    """dmipy-sim#289's retraction, as a test: the surface term does NOT cancel under the rescaling -- only its
    mean exposure does, and the signal is an average of exponentials. The replay reproduces the change rather
    than cancelling it: per walker it is the walk at 2 D, and that walk's own pack agrees to the bit."""
    pk_d, pk_2d = pore
    seq = _seq(T / 2)
    rho = 2e-5
    at_d = pk_d.replay(seq, tissue=Tissue(rho=rho))
    at_2d = pk_d.replay(seq, tissue=Tissue(rho=rho, D=2 * D0))
    own = pk_2d.replay(seq, tissue=Tissue(rho=rho))
    np.testing.assert_allclose(at_2d, own, rtol=1e-12)
    w1, e1, _ = pk_d.walker_signals(seq, tissue=Tissue(rho=rho, D=2 * D0))
    w2, e2, _ = pk_2d.walker_signals(seq, tissue=Tissue(rho=rho))
    np.testing.assert_allclose(e1, e2, rtol=1e-12)                       # per walker, the same weights
    assert not np.allclose(at_2d, at_d, rtol=1e-3)                       # and the signal moved


def test_slower_than_walked_is_refused_and_the_walk_must_cover_the_acquisition(free):
    pk_d, _ = free
    with pytest.raises(ValueError, match="slower than the walk"):
        pk_d.at_diffusivity(0.5 * D0)
    with pytest.raises(ValueError, match="slower than the walk"):
        pk_d.replay(_seq(T / 2), tissue=Tissue(D=0.5 * D0))
    with pytest.raises(ValueError, match="positive"):
        pk_d.at_diffusivity(-1.0)
    assert pk_d.at_diffusivity(D0) is pk_d
    # the view's walk is half as long, and an acquisition that fit the walk need not fit the view
    with pytest.raises(ValueError, match="beyond the pack"):
        pk_d.replay(_seq(0.8 * T), tissue=Tissue(D=2 * D0))


def test_the_prefix_of_a_view_is_the_view_of_the_prefix(free, tmp_path):
    pk_d, pk_2d = free
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 1.5e-3, 4e-3, bvalues=[1.0e9], TE=T / 4, n_t=41, slew_rate=np.inf)
    a = pk_d.at_diffusivity(2 * D0).prefix(T / 4, out_path=str(tmp_path / "a.rpk"))
    b = pk_2d.prefix(T / 4, out_path=str(tmp_path / "b.rpk"))
    np.testing.assert_allclose(a.replay(seq), b.replay(seq), rtol=1e-6)
    assert a.dt == pytest.approx(b.dt)


def test_nothing_is_decoded_on_the_way(free, monkeypatch):
    """The no-decode lock of the coefficient-space replay, extended to the scaled view."""
    from dmipy_sim.replay import compression, replay as rp
    pk_d, _ = free
    seq = _seq(T / 2)

    def boom(*a, **k):
        raise AssertionError("a track was decoded")
    monkeypatch.setattr(compression, "decode_boundary_bridge", boom)
    monkeypatch.setattr(compression, "decode_occupancy", boom)
    monkeypatch.setattr(rp.ReplayPack, "positions", boom)
    pk_d.replay(seq, tissue=Tissue(D=2 * D0))
    pk_d.walker_signals(seq, tissue=Tissue(D=2 * D0))


def test_a_study_refuses_a_tissue_at_another_diffusivity_and_takes_the_view(free):
    """A study contracts the bands once on the walked grid and applies tissues afterwards, so a tissue at
    another diffusivity cannot be served from those primitives -- every term changes, not the surface divisor
    alone, which is the half-transformation a mismatched ``Tissue.D`` used to apply silently."""
    from dmipy_sim.replay.study import Acquisition, Protocol, Study
    pk_d, pk_2d = free
    seq = _seq(T / 2)
    with pytest.raises(ValueError, match="another grid"):
        pk_d.study(Study(Protocol([Acquisition(seq)]), tissues=[Tissue(D=2 * D0)]))
    S_view = pk_d.at_diffusivity(2 * D0).study(Study(Protocol([Acquisition(seq)]), tissues=[None]))
    S_own = pk_2d.study(Study(Protocol([Acquisition(seq)]), tissues=[None]))
    np.testing.assert_allclose(np.asarray(S_view), np.asarray(S_own), rtol=1e-12)


def test_the_walk_serves_its_line_of_diffusivity_and_permeability_pairs_and_no_other(permeable):
    """The compute optimisation. One walk at the slowest diffusivity over the longest time serves every faster
    setting on its line -- the same ratio kappa / D -- as a view; a pair off the line needs its own walk."""
    pk_d, pk_2d, K = permeable
    seq = _seq(T / 2)
    assert pk_d.permeability == K and pk_2d.permeability == 2 * K
    for k in ("pos_x", "pos_y", "pos_z"):
        if k in pk_d.arrays:
            np.testing.assert_array_equal(pk_d.arrays[k], pk_2d.arrays[k])       # the same crossings, coded
    own = pk_2d.replay(seq, complex_signal=True)
    np.testing.assert_allclose(pk_d.replay(seq, tissue=Tissue(kappa=2 * K), complex_signal=True), own, rtol=1e-12)
    np.testing.assert_allclose(pk_d.replay(seq, tissue=Tissue(D=2 * D0), complex_signal=True), own, rtol=1e-12)
    np.testing.assert_allclose(pk_d.replay(seq, tissue=Tissue(D=2 * D0, kappa=2 * K), complex_signal=True), own, rtol=1e-12)
    view = pk_d.at_permeability(2 * K)
    assert view.diffusivity == pytest.approx(2 * D0)
    assert view.meta["provenance"]["diffusivity_scaled"]["permeability"]["wall_0.in_to_out"]["replayed"] == pytest.approx(2 * K)
    with pytest.raises(ValueError, match="not on this walk's line"):
        pk_d.replay(seq, tissue=Tissue(D=2 * D0, kappa=K))
    with pytest.raises(ValueError, match="slower than the walk"):
        pk_d.replay(seq, tissue=Tissue(kappa=0.5 * K))


def test_a_permeability_needs_a_permeable_wall(free):
    pk_d, _ = free
    assert pk_d.permeability is None
    with pytest.raises(ValueError, match="no permeable wall"):
        pk_d.replay(_seq(T / 2), tissue=Tissue(kappa=1e-5))
    t = Tissue(D=2 * D0, kappa=3e-5)
    assert Tissue.from_meta(t.to_meta()) == t


@pytest.fixture(scope="module")
def sheathed(tmp_path_factory):
    """A sheathed axon with its field basis, walked at (D, T, dt) and at (2D, T/2, dt/2) from one seed: three
    pools, walls with contact, and the susceptibility path channel -- every tier a pack can carry."""
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    tmp = tmp_path_factory.mktemp("sheath")
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    g2 = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=2 * D0, D_extra=2 * D0)
    out = []
    for geom, D, T_, dt, name in ((g, D0, 6e-3, 3e-4, "d"), (g2, 2 * D0, 3e-3, 1.5e-4, "2d")):
        walk = d.simulate_trajectories(1500, D, geom, T_, dt, seed=0, require_gpu=False)
        path = tmp / f"{name}.rpk"
        build_replay_pack(walk, id=f"test/{name}", license="x", citation="x", K=8,
                          field=field_grid_of(geom, res=0.2e-6), susc_path_K=16, out_path=str(path))
        out.append(read_rpk(str(path)))
    return out


def test_the_field_and_the_relaxation_channels_follow_the_rescale_to_the_bit(sheathed):
    """The path channel is the field per save, a function of position only; the occupancy runs are per save.
    Both follow the new grid in their own space: the field's gate is taken on it and its phase carries the
    new dt, the relaxation gates read the new dt. The view of the pack walked at D must replay as the pack
    walked at 2D from the same seed does, under a field, under T2 per pool, and under both with a pose."""
    pk_d, pk_2d = sheathed
    assert pk_d.has_field and pk_2d.has_field and pk_d.has_relaxation
    for k in ("pos_x", "pos_y", "pos_z"):
        if k in pk_d.arrays:
            np.testing.assert_array_equal(pk_d.arrays[k], pk_2d.arrays[k])
    n_t, dt = pk_2d.n_t, pk_2d.dt
    seq = sequences.pgse([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 0.4e-3, 1.2e-3, bvalues=[2e8, 2e8],
                         TE=(n_t - 1) * dt, n_t=n_t, slew_rate=np.inf)
    chi = Tissue(chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
    t2 = Tissue(T2={"extra": 0.08, "intra": 0.03, "myelin": 0.01})
    both = Tissue(chi_iso=-0.1e-6, chi_aniso=-0.1e-6, T2={"extra": 0.08, "intra": 0.03, "myelin": 0.01})
    bare = pk_2d.replay(seq, complex_signal=True)
    # the tier is live: a spin echo refocuses the static part of the field, so what survives of it is the
    # diffusion through it, small at this TE; the relaxation moves the signal by percent
    for tissue, kw, live in ((chi, dict(scanner=3.0), 1e-7), (t2, {}, 1e-3), (both, dict(scanner=7.0), 1e-3)):
        own = pk_2d.replay(seq, tissue=tissue, complex_signal=True, **kw)
        view = pk_d.replay(seq, tissue=tissue.replace(D=2 * D0), complex_signal=True, **kw)
        np.testing.assert_allclose(view, own, rtol=1e-9, err_msg=repr(tissue))
        assert np.abs(view - bare).max() > live * np.abs(bare).max(), repr(tissue)
    # and through the pose expansion, which reads the field's grid on its own
    R = so3_rotation(0.3, 0.7, -0.4)
    own = pk_2d.replay(seq, tissue=both, scanner=3.0, orientation=R, complex_signal=True)
    view = pk_d.replay(seq, tissue=both.replace(D=2 * D0), scanner=3.0, orientation=R, complex_signal=True)
    np.testing.assert_allclose(view, own, rtol=1e-9)


def so3_rotation(a, b, c):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("zyx", [a, b, c]).as_matrix()

"""A pack replays at another size: every stored length times ``s``, the diffusivity times ``s^2``, nothing decoded
(dmipy-sim#542).

Scaling every length at fixed times leaves the dimensionless walk as it was, so the stored path times ``s`` IS a
walk of the substrate at ``s`` times the size at ``s^2 D``. The checks are exact where the physics is: the view at
``G`` is the pack at ``s G`` to the bit of float64, the surface term at ``rho2`` is the pack's at ``rho2 / s``; and a
walk of the scaled substrate from the same seed -- every float32 step and wall halved exactly at ``s = 1/2`` --
replays as the view does.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

D0, T, DT, R, KAPPA = 2.0e-9, 40e-3, 5e-4, 4e-6, 2e-5


def _pack(tmp, geometry, D, T_, dt, name, seed=3, sub_steps=None):
    walk = d.simulate_trajectories(600, D, geometry, T_, dt, seed=seed, require_gpu=False, K=12, sub_steps=sub_steps)
    out = tmp / f"{name}.rpk"
    _float_pack(walk, id=f"test/{name}", license="x", citation="x", K=12, out_path=str(out))
    return read_rpk(str(out))


def _cyl(r, kappa):
    return d.Cylinder(radius=r, orientation=(0, 0, 1), permeability=kappa, surface_relaxivity_t2=2e-5)


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    """The walked pack, its twin at half the size (``s = 1/2``: half the radius, a quarter of the diffusivity, half
    the permeability, same time), and the twin of that read back at ``D0`` (``a = 4``: a quarter of the time)."""
    tmp = tmp_path_factory.mktemp("size")
    ss = d.walk_cost(600, D0, _cyl(R, KAPPA), T, DT, K=12).sub_steps          # the twins take the walk's own steps
    return (_pack(tmp, _cyl(R, KAPPA), D0, T, DT, "d", sub_steps=ss),
            _pack(tmp, _cyl(R / 2, KAPPA / 2), D0 / 4, T, DT, "half", sub_steps=ss),
            _pack(tmp, _cyl(R / 2, 2 * KAPPA), D0, T / 4, DT / 4, "half_at_d0", sub_steps=ss))


def _seq(TE, b=1.5e9):
    return sequences.pgse([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 4e-3, 10e-3, bvalues=[b, b], TE=TE, n_t=81, slew_rate=np.inf)


@pytest.mark.parametrize("s, b", [(0.5, 1.5e9), (2.0, 0.3e9)])
def test_the_view_at_g_is_the_pack_at_s_g_on_every_route(packs, s, b):
    pk, _, _ = packs
    view = pk.at_length_scale(s)
    seq, seq_s = _seq(T / 2, b), _seq(T / 2, s * s * b)                     # b goes as G^2 at the same timing
    np.testing.assert_allclose(view.replay(seq, complex_signal=True), pk.replay(seq_s, complex_signal=True), rtol=1e-12)
    _, e1, E1 = view.walker_signals(seq)
    _, e2, E2 = pk.walker_signals(seq_s)
    np.testing.assert_allclose(E1, E2, rtol=1e-12)
    np.testing.assert_allclose(e1, e2, rtol=1e-12)
    np.testing.assert_allclose(view.walker_phases(seq)[2], pk.walker_phases(seq_s)[2], rtol=1e-12)
    np.testing.assert_allclose(view.pose_response(seq).coeffs, pk.pose_response(seq_s).coeffs, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(view.replay_bloch(seq, complex_signal=True), pk.replay_bloch(seq_s, complex_signal=True), rtol=1e-9)


def test_the_surface_term_at_rho2_is_the_packs_at_rho2_over_s(packs):
    pk, _, _ = packs
    s, rho2 = 0.5, 2e-5
    view = pk.at_length_scale(s)
    seq, seq_s = _seq(T / 2), _seq(T / 2, s * s * 1.5e9)
    np.testing.assert_allclose(view.replay(seq, tissue=Tissue(rho2=rho2)), pk.replay(seq_s, tissue=Tissue(rho2=rho2 / s)), rtol=1e-12)
    assert not np.allclose(view.replay(seq, tissue=Tissue(rho2=rho2)), pk.replay(seq_s, tissue=Tissue(rho2=rho2)), rtol=1e-3)


def test_the_view_says_what_it_realises(packs):
    pk, _, _ = packs
    s = 0.5
    view = pk.at_length_scale(s)
    assert view.diffusivity == pytest.approx(s * s * D0) and view.permeability == pytest.approx(s * KAPPA)
    assert view.dt == pk.dt and view.n_t == pk.n_t and view.temporal_bandwidth_hz == pk.temporal_bandwidth_hz
    spec = view.substrate
    assert spec.walls[0].surface.radius == pytest.approx(s * R)
    assert spec.validity.smallest_feature == pytest.approx(s * pk.substrate.validity.smallest_feature)
    assert spec.walls[0].surface_relaxivity.inside == pytest.approx(s * pk.substrate.walls[0].surface_relaxivity.inside)
    np.testing.assert_allclose(spec.domain.box_max, s * np.asarray(pk.substrate.domain.box_max))
    env, env0 = view.replay_envelope["acquisition"], pk.replay_envelope["acquisition"]
    assert env["b_max"] == pytest.approx(env0["b_max"] / s ** 2)
    ls = view.meta["provenance"]["length_scaled"]
    assert ls["s"] == s and ls["b_max"]["walked"] == env0["b_max"] and ls["diffusivity"]["walked"] == D0
    assert view.at_length_scale(2.0).meta["provenance"]["length_scaled"]["s"] == pytest.approx(1.0)   # cumulative
    for k in ("comp_events_offsets", "comp_events_t", "comp_events_label", "spin_weights"):
        assert view.arrays[k] is pk.arrays[k]                                 # no length: shared
    assert pk.at_length_scale(1.0) is pk
    seq = _seq(T / 2)
    np.testing.assert_array_equal(view.replay(seq, tissue=Tissue(D=s * s * D0)), view.replay(seq))   # the view's own D


def test_the_envelope_moves_with_the_size_and_is_held(packs):
    pk, _, _ = packs
    b_max = pk.replay_envelope["acquisition"]["b_max"]
    up = pk.at_length_scale(2.0)
    assert up.replay_envelope["acquisition"]["b_max"] == pytest.approx(b_max / 4)
    with pytest.raises(ValueError, match="beyond the .* this length-scaled view serves"):
        up.replay(_seq(T / 2, 0.5 * b_max))
    up.replay(_seq(T / 2, 0.2 * b_max))                                       # within it
    pk.at_length_scale(0.5).replay(_seq(T / 2, b_max))                        # s < 1 serves more than the pack states
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive"):
            pk.at_length_scale(bad)


def test_the_view_is_the_walk_of_the_scaled_substrate(packs):
    """The twin walked at half the size from the same seed: at ``s = 1/2`` every float32 step and every wall is
    halved exactly, so the two paths are one path up to the walk's own float32 rounding against its absolute
    tolerances (the signals agree to 1e-5; the surface term, an accumulated contact, to 1.1e-5)."""
    pk, half, half_at_d0 = packs
    seq = _seq(T / 2)
    view = pk.at_length_scale(0.5)
    np.testing.assert_allclose(view.replay(seq, complex_signal=True), half.replay(seq, complex_signal=True), rtol=1e-4)
    np.testing.assert_allclose(view.replay(seq, tissue=Tissue(rho2=2e-5)), half.replay(seq, tissue=Tissue(rho2=2e-5)), rtol=1e-4)
    # then faster: the size view followed by the time view is the walk at (R/2, D0, T/4)
    both = view.at_diffusivity(D0)
    assert both.diffusivity == D0 and both.permeability == pytest.approx(2 * KAPPA) and both.dt == pytest.approx(DT / 4)
    seq4 = sequences.pgse([[1.0, 0.0, 0.0]], 1.5e-3, 4e-3, bvalues=[1.0e9], TE=T / 4, n_t=41, slew_rate=np.inf)   # fits the quarter walk
    np.testing.assert_allclose(both.replay(seq4, complex_signal=True), half_at_d0.replay(seq4, complex_signal=True), rtol=1e-4)


def test_nothing_is_decoded_on_the_way(packs, monkeypatch):
    from dmipy_sim.replay import compression, replay as rp
    pk, _, _ = packs
    seq = _seq(T / 2)

    def boom(*a, **k):
        raise AssertionError("a track was decoded")
    monkeypatch.setattr(compression, "local_time_at", boom)
    monkeypatch.setattr(compression, "positions_at", boom)
    monkeypatch.setattr(compression, "decode_occupancy", boom)
    monkeypatch.setattr(rp.ReplayPack, "positions", boom)
    view = pk.at_length_scale(0.5)
    view.replay(seq, tissue=Tissue(rho2=2e-5))
    view.walker_signals(seq)


def test_the_prefix_of_a_view_is_the_view_of_the_prefix(packs, tmp_path):
    pk, half, _ = packs
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 1.5e-3, 4e-3, bvalues=[1.0e9], TE=T / 4, n_t=41, slew_rate=np.inf)
    a = pk.at_length_scale(0.5).prefix(T / 4, out_path=str(tmp_path / "a.rpk"))
    b = half.prefix(T / 4, out_path=str(tmp_path / "b.rpk"))
    np.testing.assert_allclose(a.replay(seq), b.replay(seq), rtol=1e-4)
    assert a.diffusivity == pytest.approx(D0 / 4)


def _float_pack(*a, **k):
    """``build_replay_pack`` in the float containers: this module compares packs and replays to the bit, which the
    derived integer default does not promise."""
    from dmipy_sim.replay.bank import build_replay_pack
    k.setdefault("position_container", None); k.setdefault("blt_container", None); k.setdefault("field_container", None)
    return build_replay_pack(*a, **k)

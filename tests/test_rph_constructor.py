"""The .rph constructor: a phantom is built from volumes, not hand-rolled arrays (RPH.md, issue #151)."""
import numpy as np
from dataclasses import replace

from dmipy_sim import Encoding, RFEvent, ScannerSequence
import pytest

from dmipy_sim.replay.fod import FOD
from dmipy_sim.replay.so3 import n_sh_coeffs
from dmipy_sim.replay.phantom import (BinghamField, FrameField, Grid, ODFField, PeakField, WatsonField,
                                      analytic_substrate, build_rph, inert_substrate, pack_substrate, read_rph)


def _pack(tmp_path, n_walkers=400):
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(n_walkers, 2e-9, g, 4e-3, 5e-4, seed=0, require_gpu=False)
    out = tmp_path / "tiny.rpk"
    build_replay_pack(walk, id="test/tiny", license="x", citation="x", K=8, out_path=str(out))
    return str(out)


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    return _pack(tmp_path_factory.mktemp("pk"))


def _annulus(n=8, r_in=2.0, r_out=3.5, sub=4):
    """Volume fractions of an annulus of tissue around a core, supersampled so the boundaries are genuine
    partial volume."""
    c = (n - 1) / 2.0
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off, off, indexing="ij")
    wm = np.zeros((n, n, 1)); csf = np.zeros((n, n, 1))
    for i in range(n):
        for j in range(n):
            rr = np.hypot(i + ox - c, j + oy - c)
            wm[i, j, 0] = np.mean((rr >= r_in) & (rr <= r_out))
            csf[i, j, 0] = np.mean(rr < r_in)
    return wm, csf


def _tangential(n=8):
    c = (n - 1) / 2.0
    mu = np.zeros((n, n, 1, 3))
    for i in range(n):
        for j in range(n):
            dx, dy = i - c, j - c
            r = np.hypot(dx, dy)
            mu[i, j, 0] = (-dy / r, dx / r, 0.0) if r > 1e-9 else (1.0, 0.0, 0.0)
    return mu


def _phantom(tmp_path, pack_path, **kw):
    wm, csf = _annulus()
    grid = Grid((8, 8, 1), (1e-3, 1e-3, 1e-3))
    subs = [pack_substrate("wm/tiny", pack_path, m0=0.7, tissue={"T2": [0.06, 0.06, 0.06]}),
            analytic_substrate("csf/free-water", "free_water", {"diffusivity": 3e-9}),
            inert_substrate()]
    args = dict(grid=grid, substrates=subs, occupancy={"wm/tiny": wm, "csf/free-water": csf},
                remainder="background/inert", orientation=WatsonField(12.0, _tangential(), lmax=8),
                id="test/annulus", license="CC-BY-4.0", citation="x", embed=False)
    args.update(kw)
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "p.rph"
    meta = build_rph(out, **args)
    return read_rph(out), meta


def test_the_grid_is_placed_in_the_scanner():
    g = Grid((4, 4, 2), (2e-3, 2e-3, 5e-3))
    np.testing.assert_allclose(g.origin_m, [-3e-3, -3e-3, -2.5e-3])          # centred by default
    np.testing.assert_allclose(g.isocenter_m, [0, 0, 0])                     # the grid centre
    np.testing.assert_allclose(g.positions_m([[0, 0, 0], [3, 3, 1]]), [[-3e-3, -3e-3, -2.5e-3], [3e-3, 3e-3, 2.5e-3]])
    assert g.radius_m([[0, 0, 0]])[0] == pytest.approx(np.linalg.norm([3e-3, 3e-3, 2.5e-3]))
    off = Grid((4, 4, 2), (2e-3,) * 3, origin_m=(0, 0, 0), isocenter_m=(1e-3, 0, 0))
    np.testing.assert_allclose(off.radius_m([[0, 0, 0]]), [1e-3])
    assert Grid.from_meta({"grid": off.to_meta()}) == off
    with pytest.raises(ValueError, match="three-dimensional"):
        Grid((4, 4), (1e-3, 1e-3))


def test_volumes_become_a_sparse_phantom_with_full_voxels(tmp_path, pack_path):
    ph, meta = _phantom(tmp_path, pack_path)
    assert meta["rph_schema_version"] == "0.4.0" and ph.mode == "odf_sh" and ph.lmax == 8
    assert ph.odf_sh.shape[-1] == n_sh_coeffs(8)
    wm, csf = _annulus()
    assert ph.n_voxels == int(((wm + csf) > 0).sum())                        # sparse: only occupied voxels
    np.testing.assert_allclose(ph.geometric_fraction.sum(axis=1), 1.0, atol=1e-4)
    # the pack's slot is the pack's fraction, the remainder went to inert, and partial volume is present
    f = ph.geometric_fraction
    sid = ph.substrate_id
    got = {i: f[v, p] for v in range(ph.n_voxels) for p, i in enumerate(sid[v]) if i >= 0}
    assert ph.index_of("wm/tiny") == 0 and ph.substrates[2]["kind"] == "inert"
    assert ((f[:, 0] > 0) & (f[:, 0] < 0.999)).any() and (got[2] if 2 in got else 0) >= 0
    assert ph.substrates[0]["tissue"] == {"T2": [0.06, 0.06, 0.06]}          # what this substrate replays at
    assert ph.substrates[0]["uri"] == pack_path and not ph.is_embedded(0)
    # the ODF is the Watson of that voxel, in the required basis
    c = (8 - 1) / 2.0
    v = int(np.argmax(np.hypot(ph.voxel_index[:, 0] - c, ph.voxel_index[:, 1] - c) * (f[:, 0] > 0.99)))
    i, j = ph.voxel_index[v, :2]
    mu = _tangential()[i, j, 0]
    np.testing.assert_allclose(ph.odf_sh[v, 0], FOD.watson(12.0, mu=mu, lmax=8).coeffs, atol=1e-5)


def test_a_short_row_is_refused_unless_a_remainder_takes_it(tmp_path, pack_path):
    with pytest.raises(ValueError, match="always full"):
        _phantom(tmp_path, pack_path, remainder=None)
    wm, csf = _annulus()
    with pytest.raises(ValueError, match="more than one"):
        _phantom(tmp_path, pack_path, occupancy={"wm/tiny": wm + 0.9, "csf/free-water": csf})


def test_a_label_volume_is_the_one_substrate_per_voxel_case(tmp_path, pack_path):
    lab = np.full((8, 8, 1), 2, np.int32)
    lab[2:6, 2:6, 0] = 0
    ph, _ = _phantom(tmp_path, pack_path, occupancy=lab, remainder=None)
    assert ph.n_voxels == 16 and ph.geometric_fraction[:, 0].min() == 1.0
    assert set(np.unique(ph.substrate_id)) == {-1, 0} or set(np.unique(ph.substrate_id)) == {0}


def test_peaks_mode_splits_a_crossing_into_slots_with_weights(tmp_path, pack_path):
    n = 4
    dirs = np.zeros((n, n, 1, 2, 3)); dirs[..., 0, :] = (0, 0, 1); dirs[..., 1, :] = (1, 0, 0)
    w = np.zeros((n, n, 1, 2)); w[..., 0] = 0.75; w[..., 1] = 0.25
    ph, _ = _phantom(tmp_path, pack_path, grid=Grid((n, n, 1), (1e-3,) * 3),
                     occupancy=np.zeros((n, n, 1), np.int32), remainder=None,
                     orientation=PeakField(dirs, weights=w))
    assert ph.mode == "peaks" and ph.n_voxels == n * n
    np.testing.assert_allclose(ph.geometric_fraction[:, :2], np.tile([0.75, 0.25], (n * n, 1)), atol=1e-6)
    np.testing.assert_allclose(ph.peak_dir[0, 0], [0, 0, 1]); np.testing.assert_allclose(ph.peak_dir[0, 1], [1, 0, 0])
    assert (ph.substrate_id[:, :2] == 0).all()


def test_a_macroscopic_layer_must_be_in_the_registry(tmp_path, pack_path):
    b1 = 1.0 + 0.1 * np.arange(8)[:, None, None] * np.ones((8, 8, 1))
    ph, _ = _phantom(tmp_path, pack_path, scalars={"kappa_B1": b1})
    assert ph.scalar_names == ("kappa_B1",)
    np.testing.assert_allclose(ph.scalar("kappa_B1"), b1[tuple(ph.voxel_index.T)], rtol=1e-6)
    with pytest.raises(ValueError, match="the phantom declares no"):
        ph.scalar("delta_B0_T")
    with pytest.raises(ValueError, match="registry"):
        _phantom(tmp_path, pack_path, scalars={"b1_map": b1})


def test_an_odf_volume_needs_its_basis_named(tmp_path, pack_path):
    mu = _tangential()
    sh = np.stack([FOD.watson(12.0, mu=mu[i, j, 0] if np.any(mu[i, j, 0]) else (0, 0, 1), lmax=8).coeffs
                   for i in range(8) for j in range(8)]).reshape(8, 8, 1, n_sh_coeffs(8))
    ph, _ = _phantom(tmp_path, pack_path, orientation=ODFField(sh, basis="tournier07", legacy=False, normalize=False))
    np.testing.assert_allclose(ph.odf_sh[0, 0], sh[tuple(ph.voxel_index[0])], atol=1e-5)
    with pytest.raises(TypeError):
        ODFField(sh)                                            # no default basis: guessing rescales m != 0
    legacy, _ = _phantom(tmp_path, pack_path, orientation=ODFField(sh, basis="tournier07", legacy=True, normalize=True))
    ph2 = read_rph(tmp_path / "p.rph")
    assert np.abs(ph2.odf_sh[0, 0] - ph.odf_sh[0, 0]).max() > 1e-3            # the conversion is not a no-op


def test_the_pack_can_be_embedded_so_the_phantom_is_standalone(tmp_path, pack_path):
    ph, meta = _phantom(tmp_path, pack_path, embed=True)
    assert ph.is_embedded(0) and meta["substrates"][0]["sha256"]
    pk = ph.pack(0)
    assert pk.meta["id"] == "test/tiny" and pk.n_walkers > 0
    with pytest.raises(ValueError, match="not a pack"):
        ph.pack(1)
    assert ph.tiers() >= {"gradient"}


# ------------------------------------------------------------------ replay
def _acq(pk, dirs, bvals, delta=6e-4, Delta=2e-3):
    """A PGSE on the pack's save grid, one measurement per (direction, b)."""
    from dmipy_sim.constants import GAMMA
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g
        G[i, ng:ng + nd] = -amp * g

    # a bare gradient echo on the pack grid: no pulses, so the effective gradient is the gradient itself
    return ScannerSequence(G=G, dt=dt, family="gre",
                           encoding=Encoding(bvalues=np.asarray(bvals, float), gradient_directions=np.asarray(dirs, float)))


def test_the_phantom_replays_every_voxel_from_one_pack_replay(tmp_path, pack_path):
    """Composition, substrate by substrate: the pack contributes its pose-composed response, free water its
    closed form, inert nothing -- and each is the same number the pack or the closed form gives on its own."""
    from dmipy_sim.replay import read_rpk
    ph, _ = _phantom(tmp_path, pack_path, embed=True)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1], [1, 0, 0]], [0.0, 1e9, 1e9])
    vi, S = ph.replay(seq)
    assert vi.shape[0] == ph.n_voxels and S.shape == (ph.n_voxels, 3)
    f_wm, f_csf = ph.fraction("wm/tiny"), ph.fraction("csf/free-water")
    # b = 0: the m0-weighted volume of each substrate times its own b = 0 response -- the pack's nominal T2
    # relaxation over the echo, free water's 1, and nothing at all from inert
    # the pack replays at the T2 the phantom declares for that substrate (RPH.md 3.2), not at its nominal value
    E0 = float(pk.replay(seq, T2=[0.06] * 3)[0])
    assert E0 < float(pk.replay(seq)[0])
    np.testing.assert_allclose(S[:, 0], f_wm * 0.7 * E0 + f_csf * 1.0, rtol=1e-6)
    np.testing.assert_allclose(f_wm + f_csf + ph.fraction("background/inert"), 1.0, atol=1e-4)
    # a pure free-water voxel is exp(-bD) times its m0
    csf = int(np.argmax(f_csf))
    if f_csf[csf] > 0.99:
        np.testing.assert_allclose(S[csf, 1], np.exp(-1e9 * 3e-9), rtol=1e-6)
    # a pure tissue voxel is the pack composed against that voxel's ODF, nothing else
    wm = int(np.argmax(f_wm))
    from dmipy_sim.replay.fod import FOD
    from dmipy_sim.replay.so3 import Distribution
    pr = pk.pose_response(seq, T2=[0.06] * 3)
    ref = pr.compose(Distribution.axis_density(FOD.native(ph.odf_sh[wm, 0].astype(float))))
    np.testing.assert_allclose(S[wm], np.abs(f_wm[wm] * 0.7 * ref + f_csf[wm] * np.exp(-seq.encoding.bvalues * 3e-9)), rtol=1e-6)
    vol = ph.to_volume(S[:, 1])
    assert vol.shape == (8, 8, 1) and np.isnan(vol).any() and np.nanmax(vol) > 0


def test_a_peak_and_a_concentrated_odf_give_the_same_signal(tmp_path, pack_path):
    """RPH.md 4: peaks are the zero-dispersion limit, an acceptance test rather than a remark."""
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.replay.fod import FOD
    n = 2
    d = np.zeros((n, n, 1, 3)); d[..., :] = (0.3, 0.0, 0.954)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    kw = dict(grid=Grid((n, n, 1), (1e-3,) * 3), occupancy=np.zeros((n, n, 1), np.int32), remainder=None, embed=True)
    peaks, _ = _phantom(tmp_path / "a", pack_path, orientation=PeakField(d), **kw)
    sharp, _ = _phantom(tmp_path / "b", pack_path, orientation=WatsonField(400.0, d, lmax=12), **kw)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    _, Sp = peaks.replay(seq)
    _, So = sharp.replay(seq)
    assert peaks.mode == "peaks" and sharp.mode == "odf_sh"
    np.testing.assert_allclose(Sp, So, atol=2e-2)


def test_a_layer_is_applied_or_refused_never_dropped(tmp_path, pack_path):
    from dmipy_sim.replay import read_rpk
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [0.0, 1e9])
    scale = np.full((8, 8, 1), 0.5)
    ph, _ = _phantom(tmp_path, pack_path, embed=True, scalars={"m0_scale": scale})
    base, _ = _phantom(tmp_path / "base", pack_path, embed=True)
    np.testing.assert_allclose(ph.replay(seq)[1], 0.5 * base.replay(seq)[1], rtol=1e-9)
    # kappa_B1 needs the RF-aware replay: refused here, not dropped
    b1 = np.ones((8, 8, 1))
    hard, _ = _phantom(tmp_path / "b1", pack_path, embed=True, scalars={"kappa_B1": b1})
    with pytest.raises(ValueError, match="vector-Bloch|RF-aware"):
        hard.replay(seq)
    # delta_B0_T: a uniform precession over the voxel, which a spin echo at TE/2 refocuses exactly
    dB = np.full((8, 8, 1), 1e-7)
    off, _ = _phantom(tmp_path / "b0", pack_path, embed=True, scalars={"delta_B0_T": dB})
    _, S = off.replay(seq, complex_signal=True)
    _, S0 = base.replay(seq, complex_signal=True)
    assert np.abs(np.angle(S / S0)).max() > 1e-3                     # a gradient echo carries the off-resonance
    seq = replace(seq, G=np.abs(np.asarray(seq.G)), family="pgse",             # the same encoding as a spin echo
                  rf=[RFEvent(0.0, 90), RFEvent((pk.n_t - 1) * pk.dt / 2, 180)])
    _, Se = off.replay(seq, complex_signal=True)
    _, Se0 = base.replay(seq, complex_signal=True)
    np.testing.assert_allclose(Se, Se0, rtol=1e-9)                   # refocused: the layer contributes nothing


def test_a_referenced_pack_must_be_supplied(tmp_path, pack_path):
    from dmipy_sim.replay import read_rpk
    ph, _ = _phantom(tmp_path, pack_path, embed=False)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    with pytest.raises(ValueError, match="uri|not embedded"):
        ph.replay(seq)
    _, S = ph.replay(seq, packs={"wm/tiny": pack_path})
    assert S.shape == (ph.n_voxels, 1) and np.isfinite(S).all()


def test_the_transmit_layer_goes_through_the_bloch_route(tmp_path, pack_path):
    """kappa_B1 scales every flip angle, which the pose expansion cannot carry, so it goes through a
    magnetisation propagation per pose. At kappa = 1 that route reproduces the ideal-pulse answer; at
    kappa != 1 it follows the spin-echo law; and an ODF phantom is refused rather than approximated."""
    from dmipy_sim.replay import read_rpk
    n = 2
    R = np.zeros((n, n, 1, 3, 3)); R[..., :, :] = np.eye(3)          # a stated pose: what a propagation needs
    kw = dict(grid=Grid((n, n, 1), (1e-3,) * 3), occupancy=np.zeros((n, n, 1), np.int32), remainder=None, embed=True)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    rf = [RFEvent(0.0, 90.0, axis_deg=0.0),
          RFEvent((pk.n_t - 1) * pk.dt / 2, 180.0, axis_deg=90.0)]
    phys = replace(seq, G=np.abs(np.asarray(seq.G)), rf=rf, family="pgse")   # the Bloch route takes the PHYSICAL waveform
    ideal, _ = _phantom(tmp_path / "one", pack_path, orientation=FrameField(R), **kw)
    assert ideal.mode == "frames"
    _, S_ideal = ideal.replay(seq)
    _, S_bloch = ideal.replay_bloch(phys, rf_events=rf)
    # the two routes differ by the pose expansion's truncation on a small pack, i.e. its Monte-Carlo floor
    np.testing.assert_allclose(S_bloch, S_ideal, atol=3.0 / np.sqrt(pk.n_walkers))
    # a transmit map is applied voxel by voxel, and each voxel is the pack propagated at that scale: the
    # nominal case is untouched, and a scaled one is exactly what the pack alone returns at that scale
    kap = np.where(np.arange(n)[:, None, None] * np.ones((n, n, 1)) > 0, 0.6, 1.0)
    ph, _ = _phantom(tmp_path / "b1", pack_path, orientation=FrameField(R), scalars={"kappa_B1": kap}, **kw)
    _, S_b1 = ph.replay_bloch(phys, rf_events=rf)
    i = ph.voxel_index[:, 0]
    np.testing.assert_allclose(S_b1[i == 0], S_bloch[i == 0], rtol=1e-9)          # kappa = 1: nothing changes
    alone = np.abs(pk.replay_bloch(phys, rf_events=rf, b1_scale=0.6, orientation=(0.0, 0.0, 1.0),
                                   T2=[0.06] * 3))                                # the T2 the phantom declares
    np.testing.assert_allclose(S_b1[i == 1], np.tile(0.7 * alone, (int((i == 1).sum()), 1)), rtol=1e-9)
    assert (S_b1[i == 1] < S_b1[i == 0] * 0.9).all()                              # a smaller flip, a smaller signal
    with pytest.raises(ValueError, match="vector-Bloch|ReplayPhantom.replay_bloch"):
        ph.replay(seq)                                      # the magnitude route refuses the layer
    mu = np.zeros((n, n, 1, 3)); mu[..., :] = (0.0, 0.0, 1.0)
    odf, _ = _phantom(tmp_path / "odf", pack_path, orientation=WatsonField(50.0, mu, lmax=8), **kw)
    with pytest.raises(ValueError, match="frames-mode"):
        odf.replay_bloch(phys, rf_events=rf)


def test_a_frame_is_one_pose_and_a_peak_is_that_pose_with_its_azimuth_unstated(tmp_path, pack_path):
    """RPH.md 4, the two modes side by side: a frames slot is a rotation, so its voxel is the pack replayed at
    that rotation; a peaks slot is the same direction with the azimuth left open, so its voxel is the average
    over that azimuth. They agree only where the substrate is axially symmetric, and the phantom says which it
    is asking for rather than leaving it to a convention."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    R0 = so3.rotation_of((0.3, 0.5, 0.81))
    R = np.zeros((n, n, 1, 3, 3)); R[..., :, :] = R0
    dirs = np.zeros((n, n, 1, 3)); dirs[..., :] = R0[:, 2]
    kw = dict(grid=Grid((n, n, 1), (1e-3,) * 3), occupancy=np.zeros((n, n, 1), np.int32), remainder=None, embed=True)
    frames, _ = _phantom(tmp_path / "fr", pack_path, orientation=FrameField(R), **kw)
    peaks, _ = _phantom(tmp_path / "pk", pack_path, orientation=PeakField(dirs), **kw)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    _, S_fr = frames.replay(seq)
    _, S_pk = peaks.replay(seq)
    direct = np.abs(0.7 * pk.replay(seq, orientation=R0, T2=[0.06] * 3, complex_signal=True))
    np.testing.assert_allclose(S_fr[0], direct, atol=3.0 / np.sqrt(pk.n_walkers))
    # this substrate is a single cylinder, so the azimuth it leaves open changes nothing beyond its own noise
    np.testing.assert_allclose(S_pk, S_fr, atol=3.0 / np.sqrt(pk.n_walkers))


def test_a_bingham_slot_is_a_fan_and_contains_the_watson(tmp_path, pack_path):
    """A frame plus two concentrations. Equal ones are the Watson cone of that width -- checked against the same
    analytic distribution composed directly, so nothing but the phantom's own bookkeeping is in the way -- and
    unequal ones are a fan the ODF mode cannot express."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    R0 = so3.rotation_of((0.0, 0.0, 1.0))
    R = np.zeros((n, n, 1, 3, 3)); R[..., :, :] = R0
    kw = dict(grid=Grid((n, n, 1), (1e-3,) * 3), occupancy=np.zeros((n, n, 1), np.int32), remainder=None, embed=True)
    cone, meta = _phantom(tmp_path / "cone", pack_path, orientation=BinghamField(R, (6.0, 6.0)), **kw)
    fan, _ = _phantom(tmp_path / "fan", pack_path, orientation=BinghamField(R, (0.2, 60.0)), **kw)
    assert cone.mode == "bingham" and meta["orientation"]["mode"] == "bingham"
    assert cone.bingham_kappa.shape == (cone.n_voxels, 1, 2)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    band = {}
    _, S_cone = cone.replay(seq, **band)
    _, S_fan = fan.replay(seq, **band)
    # the same distribution composed outside the phantom: a Watson of that concentration
    pr = pk.pose_response(seq, T2=[0.06] * 3, **band)
    # the phantom retains n = 0 (its azimuth is unstated) while this reference composes the whole band, so the
    # two agree to the projection's aliasing residual rather than to the bit
    ref = np.abs(0.7 * pr.compose(so3.Distribution.watson(6.0, mu=(0, 0, 1), lmax=pr.lmax, nmax=pr.nmax)))
    np.testing.assert_allclose(S_cone[0], ref, rtol=1e-3)
    assert np.abs(S_fan - S_cone).max() > 3e-3                       # the fan is not that cone


def test_peaks_agree_with_a_concentrated_odf(tmp_path, pack_path):
    """RPH.md 4 states this as a MUST: a peak is the zero-dispersion limit of an axis density, so a
    peaks-mode phantom must reproduce an ODF phantom concentrated on the same direction, to the
    spherical-harmonic truncation that separates them."""
    from dmipy_sim.replay import read_rpk
    n = 2
    d = np.zeros((n, n, 1, 3)); d[..., :] = (0.3, 0.5, 0.81)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    kw = dict(grid=Grid((n, n, 1), (1e-3,) * 3), occupancy=np.zeros((n, n, 1), np.int32),
              remainder=None, embed=True)
    peaks, _ = _phantom(tmp_path / "pk", pack_path, orientation=PeakField(d), **kw)
    sharp, _ = _phantom(tmp_path / "od", pack_path, orientation=WatsonField(400.0, d, lmax=12), **kw)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    band = {}
    np.testing.assert_allclose(peaks.replay(seq, **band)[1], sharp.replay(seq, **band)[1], atol=5e-3)


def test_the_same_substrate_cited_twice_is_a_crossing(tmp_path, pack_path):
    """Two populations of one solved microstructure at two poses, with two fractions: the voxel is
    their weighted sum, and its two slots cite the same substrate."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    dirs = np.zeros((n, n, 1, 2, 3))
    dirs[..., 0, :] = (0.0, 0.0, 1.0)
    dirs[..., 1, :] = (1.0, 0.0, 0.0)
    w = np.zeros((n, n, 1, 2)); w[..., 0], w[..., 1] = 0.7, 0.3
    ph, _ = _phantom(tmp_path / "x", pack_path, grid=Grid((n, n, 1), (1e-3,) * 3),
                     occupancy=np.zeros((n, n, 1), np.int32), remainder=None, embed=True,
                     orientation=PeakField(dirs, weights=w))
    assert (ph.substrate_id[:, :2] == 0).all()                       # one substrate, two slots
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    band = {}
    _, S = ph.replay(seq, **band)
    pr = pk.pose_response(seq, T2=[0.06] * 3, **band)
    ref = np.abs(0.7 * pr.compose(so3.Distribution.axis((0, 0, 1), pr.lmax, pr.nmax))
                 + 0.3 * pr.compose(so3.Distribution.axis((1, 0, 0), pr.lmax, pr.nmax))) * 0.7
    # a peak is the most alias-sensitive distribution there is -- its coefficients do not decay with l -- so the
    # phantom's retained (l, 0) projection and this full-band one agree to the pack's floor, which is the bound
    # the design certifies, rather than to the bit
    np.testing.assert_allclose(S[0], ref, atol=pr.floor)             # 0.7 is the substrate's m0


def test_an_analytic_slot_ignores_the_orientation_it_is_given(tmp_path, pack_path):
    """A closed form has no pose, so whatever a phantom writes in that slot's orientation entry must
    make no difference to the signal."""
    from dmipy_sim.replay import read_rpk
    wm, csf = _annulus()
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    band = {}
    out = []
    for axis in ((0.0, 0.0, 1.0), (0.3, 0.5, 0.81)):
        mu = np.zeros((8, 8, 1, 3)); mu[..., :] = axis
        ph, _ = _phantom(tmp_path / f"a{axis[0]}", pack_path, orientation=WatsonField(2.0, mu, lmax=8),
                         embed=True)
        f_csf = ph.fraction("csf/free-water")
        out.append(ph.replay(seq, **band)[1][f_csf > 0.999])
    assert out[0].size and np.abs(out[0] - out[1]).max() < 1e-12

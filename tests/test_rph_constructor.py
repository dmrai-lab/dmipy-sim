"""``Phantom.compose``: a phantom is built from volumes keyed by substrate objects, not hand-rolled arrays (RPH.md,
issues #151, #162, #186)."""
import numpy as np
from dataclasses import replace

import pytest

from dmipy_sim import Encoding, RFEvent, ScannerSequence
from dmipy_sim.phantom import Fan, Frames, FreeWater, Grid, Inert, ODF, PackSubstrate, Peaks, Phantom, Watson
from dmipy_sim.replay.fod import FOD
from dmipy_sim.replay.so3 import n_sh_coeffs


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


GRID8 = Grid(shape=(8, 8, 1), voxel_size_m=(1e-3, 1e-3, 1e-3))


def _subs(pack_path):
    wm = PackSubstrate(pack_path, m0=0.7, name="wm/tiny", T2_s=[0.06, 0.06, 0.06])
    return wm, FreeWater(D_m2_s=3e-9, m0=1.0), Inert()


def _phantom(pack_path, *, grid=GRID8, fractions=None, orientation=None, remainder="default", **kw):
    """The annulus phantom, or any variation of it: one pack, free water, inert background."""
    wm, csf, bg = _subs(pack_path)
    f_wm, f_csf = _annulus()
    if fractions is None:
        fractions = {wm: f_wm, csf: f_csf}
    if orientation is None:
        orientation = Watson(mu=_tangential(), kappa=12.0, lmax=8)
    if remainder == "default":
        remainder = bg
    ph = Phantom.compose(grid, fractions=fractions, orientation=orientation, remainder=remainder, **kw)
    return ph, wm, csf, bg


def _acq(pk, dirs, bvals, delta=6e-4, Delta=2e-3):
    """A PGSE on the pack's save grid, one measurement per (direction, b), read as a bare gradient echo."""
    from dmipy_sim.constants import GAMMA
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g
        G[i, ng:ng + nd] = -amp * g
    return ScannerSequence(G=G, dt=dt, family="gre",
                           encoding=Encoding(bvalues=np.asarray(bvals, float), gradient_directions=np.asarray(dirs, float)))


def _rows(ph, S):
    """The occupied rows of a replayed volume."""
    return ph.sparse(S)[1]


# ------------------------------------------------------------------ the pieces
def test_the_grid_is_placed_in_the_scanner():
    g = Grid(shape=(4, 4, 2), voxel_size_m=(2e-3, 2e-3, 5e-3))
    np.testing.assert_allclose(g.origin_m, [-3e-3, -3e-3, -2.5e-3])          # centred by default
    np.testing.assert_allclose(g.isocenter_m, [0, 0, 0])                     # the grid centre
    np.testing.assert_allclose(g.positions_m([[0, 0, 0], [3, 3, 1]]), [[-3e-3, -3e-3, -2.5e-3], [3e-3, 3e-3, 2.5e-3]])
    assert g.radius_m([[0, 0, 0]])[0] == pytest.approx(np.linalg.norm([3e-3, 3e-3, 2.5e-3]))
    off = Grid(shape=(4, 4, 2), voxel_size_m=(2e-3,) * 3, origin_m=(0, 0, 0), isocenter_m=(1e-3, 0, 0))
    np.testing.assert_allclose(off.radius_m([[0, 0, 0]]), [1e-3])
    assert Grid.from_meta({"grid": off.to_meta()}) == off
    with pytest.raises(ValueError, match="three-dimensional"):
        Grid(shape=(4, 4), voxel_size_m=(1e-3, 1e-3))
    with pytest.raises(TypeError):
        Grid((4, 4, 2), (2e-3,) * 3)                                         # every argument is a keyword


def test_a_grid_comes_from_a_nifti_affine_when_it_is_axis_aligned():
    A = np.diag([1.5, -1.5, 2.0, 1.0]); A[:3, 3] = (-30.0, 45.0, -10.0)      # millimetres, LPS-ish flip on y
    g = Grid.from_affine(A, (40, 60, 10))
    assert g.shape == (40, 60, 10) and g.axes == "RPS"
    np.testing.assert_allclose(g.voxel_size_m, [1.5e-3, 1.5e-3, 2e-3])
    np.testing.assert_allclose(g.origin_m, [-30e-3, 45e-3, -10e-3])
    A[0, 1] = 0.3                                                             # oblique
    with pytest.raises(ValueError, match="oblique"):
        Grid.from_affine(A, (40, 60, 10))


def test_substrates_are_objects_with_a_required_proton_density(pack_path):
    wm = PackSubstrate(pack_path, m0=0.7, T2_s={"intra": 0.06})
    assert wm.name == "tiny" and wm.uri == pack_path and wm.tissue == {"T2": {"intra": 0.06}}
    assert wm.pack.n_walkers == 400                                          # resolved from the path on first use
    with pytest.raises(TypeError):
        PackSubstrate(pack_path)                                              # m0 has no default
    with pytest.raises(TypeError):
        FreeWater(D_m2_s=3e-9)                                                # nor here: 1 is only a reference
    assert Inert().m0 == 0.0 and FreeWater(D_m2_s=3e-9, m0=1.0).to_meta()["params"] == {"diffusivity": 3e-9}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PackSubstrate("/nowhere/none.rpk", m0=1.0).pack


def test_volumes_become_a_sparse_phantom_with_full_voxels(pack_path):
    ph, wm, csf, bg = _phantom(pack_path)
    f = ph.file
    assert f.meta["rph_schema_version"] == "0.4.0" and ph.mode == "odf_sh" and f.lmax == 8
    assert f.odf_sh.shape[-1] == n_sh_coeffs(8)
    f_wm, f_csf = _annulus()
    assert ph.n_voxels == int(((f_wm + f_csf) > 0).sum())                     # sparse: only occupied voxels
    np.testing.assert_allclose(f.geometric_fraction.sum(axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(ph.fraction(wm) + ph.fraction(csf) + ph.fraction(bg), (f_wm + f_csf > 0) * 1.0, atol=1e-4)
    assert ph.index_of(wm) == 0 and ph.substrates[2].kind == "inert"
    assert ((f.geometric_fraction[:, 0] > 0) & (f.geometric_fraction[:, 0] < 0.999)).any()   # partial volume
    assert f.substrates[0]["tissue"] == {"T2": [0.06, 0.06, 0.06]}           # what this substrate replays at
    assert f.substrates[0]["uri"] == pack_path and not f.is_embedded(0)
    # the ODF is the Watson of that voxel, in the required basis
    c = (8 - 1) / 2.0
    fr = f.geometric_fraction
    v = int(np.argmax(np.hypot(f.voxel_index[:, 0] - c, f.voxel_index[:, 1] - c) * (fr[:, 0] > 0.99)))
    i, j = f.voxel_index[v, :2]
    mu = _tangential()[i, j, 0]
    np.testing.assert_allclose(f.odf_sh[v, 0], FOD.watson(12.0, mu=mu, lmax=8).coeffs, atol=1e-5)
    assert "Phantom(" in repr(ph) and "FreeWater" in repr(ph)


def test_a_short_row_is_refused_unless_a_remainder_takes_it(pack_path):
    with pytest.raises(ValueError, match="always full"):
        _phantom(pack_path, remainder=None)
    wm, csf, _ = _subs(pack_path)
    f_wm, f_csf = _annulus()
    with pytest.raises(ValueError, match="more than one"):
        _phantom(pack_path, fractions={wm: f_wm + 0.9, csf: f_csf})


def test_a_label_volume_is_the_one_substrate_per_voxel_case(pack_path):
    wm, csf, bg = _subs(pack_path)
    lab = np.full((8, 8, 1), 2, np.int32)
    lab[2:6, 2:6, 0] = 0
    ph = Phantom.compose(GRID8, fractions=lab, labels={0: wm, 2: bg}, orientation=Watson(mu=_tangential(), kappa=12.0))
    assert ph.n_voxels == 16 and ph.file.geometric_fraction[:, 0].min() == 1.0
    with pytest.raises(ValueError, match="no substrate in labels"):
        Phantom.compose(GRID8, fractions=lab, labels={0: wm}, orientation=Watson(mu=_tangential(), kappa=12.0))


def test_peaks_mode_splits_a_crossing_into_slots_with_weights(pack_path):
    n = 4
    wm, _, _ = _subs(pack_path)
    dirs = np.zeros((n, n, 1, 2, 3)); dirs[..., 0, :] = (0, 0, 1); dirs[..., 1, :] = (1, 0, 0)
    w = np.zeros((n, n, 1, 2)); w[..., 0] = 0.75; w[..., 1] = 0.25
    ph = Phantom.compose(Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3), fractions={wm: np.ones((n, n, 1))},
                         orientation=Peaks(dirs, weights=w))
    f = ph.file
    assert ph.mode == "peaks" and ph.n_voxels == n * n
    np.testing.assert_allclose(f.geometric_fraction[:, :2], np.tile([0.75, 0.25], (n * n, 1)), atol=1e-6)
    np.testing.assert_allclose(f.peak_dir[0, 0], [0, 0, 1]); np.testing.assert_allclose(f.peak_dir[0, 1], [1, 0, 0])
    assert (f.substrate_id[:, :2] == 0).all()


def test_a_macroscopic_layer_must_be_in_the_registry(pack_path):
    b1 = 1.0 + 0.1 * np.arange(8)[:, None, None] * np.ones((8, 8, 1))
    ph, *_ = _phantom(pack_path, layers={"kappa_B1": b1})
    assert ph.layers == ("kappa_B1",)
    np.testing.assert_allclose(ph.file.scalar("kappa_B1"), b1[tuple(ph.voxel_index.T)], rtol=1e-6)
    with pytest.raises(ValueError, match="the phantom declares no"):
        ph.file.scalar("delta_B0_T")
    with pytest.raises(ValueError, match="registry"):
        _phantom(pack_path, layers={"b1_map": b1})


def test_an_odf_volume_needs_its_basis_named(pack_path):
    mu = _tangential()
    sh = np.stack([FOD.watson(12.0, mu=mu[i, j, 0] if np.any(mu[i, j, 0]) else (0, 0, 1), lmax=8).coeffs
                   for i in range(8) for j in range(8)]).reshape(8, 8, 1, n_sh_coeffs(8))
    ph, *_ = _phantom(pack_path, orientation=ODF(sh, basis="mrtrix3"))
    np.testing.assert_allclose(ph.file.odf_sh[0, 0], sh[tuple(ph.voxel_index[0])], atol=1e-5)
    with pytest.raises(TypeError):
        ODF(sh)                                                 # no default basis: guessing rescales m != 0
    with pytest.raises(ValueError, match="name the source"):
        ODF(sh, basis="dipy")
    legacy, *_ = _phantom(pack_path, orientation=ODF(sh, basis="mrtrix-legacy"))
    assert np.abs(legacy.file.odf_sh[0, 0] - ph.file.odf_sh[0, 0]).max() > 1e-3     # the conversion is not a no-op
    # what a CSD integrates to is kept, not eaten: scaling the input leaves the phantom unchanged and the AFD known
    afd = ODF(2.5 * sh, basis="dmipy-fit")
    np.testing.assert_allclose(afd.integral[..., 0][np.any(sh, axis=-1)], 2.5, rtol=1e-6)
    scaled, *_ = _phantom(pack_path, orientation=afd)
    np.testing.assert_allclose(scaled.file.odf_sh, ph.file.odf_sh, atol=1e-6)


def test_the_pack_can_be_embedded_so_the_phantom_is_standalone(tmp_path, pack_path):
    ph, wm, *_ = _phantom(pack_path)
    meta = ph.write(tmp_path / "p.rph", id="test/annulus", license="CC-BY-4.0", citation="x", embed=True)
    back = Phantom.read(tmp_path / "p.rph")
    assert back.file.is_embedded(0) and meta["substrates"][0]["sha256"] and meta["id"] == "test/annulus"
    pk = back.file.pack(0)
    assert pk.meta["id"] == "test/tiny" and pk.n_walkers > 0
    assert back.substrates[0].pack is pk or back.substrates[0].pack.n_walkers == pk.n_walkers
    with pytest.raises(ValueError, match="not a pack"):
        back.file.pack(1)
    assert back.file.tiers() >= {"gradient"}
    # a referenced file resolves its recorded path by itself, and an in-memory pack cannot be referenced
    ph.write(tmp_path / "ref.rph", id="t", license="x", citation="x", embed=False)
    ref = Phantom.read(tmp_path / "ref.rph")
    assert ref.substrates[0].uri == pack_path and ref.substrates[0].pack.n_walkers == 400
    mem = PackSubstrate(pk, m0=0.7)
    ph2 = Phantom.compose(GRID8, fractions={mem: np.ones((8, 8, 1))}, orientation=Watson(mu=_tangential(), kappa=12.0))
    with pytest.raises(ValueError, match="embed=True"):
        ph2.write(tmp_path / "mem.rph", id="t", license="x", citation="x")


# ------------------------------------------------------------------ replay
def test_the_phantom_replays_every_voxel_from_one_pack_replay(pack_path):
    """Composition, substrate by substrate: the pack contributes its pose-composed response, free water its
    closed form, inert nothing -- and each is the same number the pack or the closed form gives on its own."""
    from dmipy_sim.replay import read_rpk
    ph, wm, csf, bg = _phantom(pack_path)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1], [1, 0, 0]], [0.0, 1e9, 1e9])
    vol = ph.replay(seq)
    assert vol.shape == (8, 8, 1, 3) and np.isnan(vol).any()
    vi, S = ph.sparse(vol)
    assert vi.shape[0] == ph.n_voxels and S.shape == (ph.n_voxels, 3)
    f_wm, f_csf = ph.sparse(ph.fraction(wm))[1], ph.sparse(ph.fraction(csf))[1]
    # b = 0: the m0-weighted volume of each substrate times its own b = 0 response -- at the T2 the phantom
    # declares for that substrate (RPH.md 3.2), not its nominal value; free water's 1; nothing from inert
    E0 = float(pk.replay(seq, T2=[0.06] * 3)[0])
    assert E0 < float(pk.replay(seq)[0])
    np.testing.assert_allclose(S[:, 0], f_wm * 0.7 * E0 + f_csf * 1.0, rtol=1e-6)
    csf_v = int(np.argmax(f_csf))
    if f_csf[csf_v] > 0.99:
        np.testing.assert_allclose(S[csf_v, 1], np.exp(-1e9 * 3e-9), rtol=1e-6)
    wm_v = int(np.argmax(f_wm))
    from dmipy_sim.replay.so3 import Distribution
    pr = pk.pose_response(seq, T2=[0.06] * 3)
    ref = pr.compose(Distribution.axis_density(FOD.native(ph.file.odf_sh[wm_v, 0].astype(float))))
    np.testing.assert_allclose(S[wm_v], np.abs(f_wm[wm_v] * 0.7 * ref + f_csf[wm_v] * np.exp(-seq.encoding.bvalues * 3e-9)),
                               rtol=1e-6)


def test_a_layer_is_applied_or_refused_never_dropped(pack_path):
    from dmipy_sim.replay import read_rpk
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [0.0, 1e9])
    base, *_ = _phantom(pack_path)
    ph, *_ = _phantom(pack_path, layers={"m0_scale": np.full((8, 8, 1), 0.5)})
    np.testing.assert_allclose(_rows(ph, ph.replay(seq)), 0.5 * _rows(base, base.replay(seq)), rtol=1e-9)
    hard, *_ = _phantom(pack_path, layers={"kappa_B1": np.ones((8, 8, 1))})
    with pytest.raises(ValueError, match="frames-mode"):
        hard.replay(seq)                  # kappa_B1 routes to the RF-aware replay, which this ODF phantom cannot take: refused
    off, *_ = _phantom(pack_path, layers={"delta_B0_T": np.full((8, 8, 1), 1e-7)})
    S, S0 = _rows(off, off.replay(seq, complex_signal=True)), _rows(base, base.replay(seq, complex_signal=True))
    assert np.abs(np.angle(S / S0)).max() > 1e-3                # a gradient echo carries the off-resonance
    from dmipy_sim.constants import GAMMA
    TE = (pk.n_t - 1) * pk.dt
    np.testing.assert_allclose(np.angle(S / S0), GAMMA * 1e-7 * TE, rtol=1e-6)   # gamma dB0 TE, over the acquisition
    # the same offset given at replay time, and the two adding
    np.testing.assert_allclose(_rows(base, base.replay(seq, off_resonance=1e-7, complex_signal=True)), S, rtol=1e-6)  # the file holds float32
    S2 = _rows(off, off.replay(seq, off_resonance=1e-7, complex_signal=True))
    np.testing.assert_allclose(np.angle(S2 / S0), GAMMA * 2e-7 * TE, rtol=1e-6)
    # the gate is the ACQUISITION's: a shorter gradient echo on its own raster dephases over its own TE
    short = _acq(pk, [[1, 0, 0]], [1e9], delta=4e-4, Delta=1.2e-3)
    short = ScannerSequence(G=np.asarray(short.G)[:, : pk.n_t - 2], dt=short.dt, family="gre",
                            encoding=short.encoding)                                    # two saves shorter than the pack
    Ss, Ss0 = _rows(base, base.replay(short, off_resonance=1e-7, complex_signal=True)), _rows(base, base.replay(short, complex_signal=True))
    np.testing.assert_allclose(np.angle(Ss / Ss0), GAMMA * 1e-7 * (short.n_t - 1) * short.dt, rtol=1e-6)
    se = replace(seq, G=np.abs(np.asarray(seq.G)), family="pgse",
                 rf=[RFEvent(0.0, 90), RFEvent((pk.n_t - 1) * pk.dt / 2, 180)])
    np.testing.assert_allclose(_rows(off, off.replay(se, complex_signal=True)),
                               _rows(base, base.replay(se, complex_signal=True)), rtol=1e-9)   # refocused: nothing
    # proton density at replay time is the m0_scale layer at call time, and they multiply
    np.testing.assert_allclose(_rows(base, base.replay(seq, proton_density=0.5)), _rows(ph, ph.replay(seq)), rtol=1e-9)
    np.testing.assert_allclose(_rows(ph, ph.replay(seq, proton_density=2.0)), _rows(base, base.replay(seq)), rtol=1e-9)


def test_a_referenced_pack_is_resolved_or_overridden(tmp_path, pack_path):
    from dmipy_sim.replay import read_rpk
    ph, wm, *_ = _phantom(pack_path)
    ph.write(tmp_path / "p.rph", id="t", license="x", citation="x", embed=False)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    back = Phantom.read(tmp_path / "p.rph")                     # the recorded uri resolves by itself
    S = _rows(back, back.replay(seq))
    assert S.shape == (ph.n_voxels, 1) and np.isfinite(S).all()
    S2 = _rows(back, back.replay(seq, packs={"wm/tiny": pk}))   # or an in-memory override, by name
    np.testing.assert_allclose(S2, S)
    with pytest.raises(ValueError, match="does not cite"):
        Phantom.read(tmp_path / "p.rph", packs={"nope": pk})


def test_the_transmit_layer_goes_through_the_bloch_route(pack_path):
    """kappa_B1 scales every flip angle, which the pose expansion cannot carry, so it goes through a
    magnetisation propagation per pose. At kappa = 1 that route reproduces the ideal-pulse answer; at
    kappa != 1 it follows the spin-echo law; and an ODF phantom is refused rather than approximated."""
    from dmipy_sim.replay import read_rpk
    n = 2
    wm, *_ = _subs(pack_path)
    R = np.zeros((n, n, 1, 3, 3)); R[..., :, :] = np.eye(3)          # a stated pose: what a propagation needs
    grid = Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3)
    full = {wm: np.ones((n, n, 1))}
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    rf = [RFEvent(0.0, 90.0, axis_deg=0.0), RFEvent((pk.n_t - 1) * pk.dt / 2, 180.0, axis_deg=90.0)]
    phys = replace(seq, G=np.abs(np.asarray(seq.G)), rf=rf, family="pgse")   # the Bloch route takes the PHYSICAL waveform
    ideal = Phantom.compose(grid, fractions=full, orientation=Frames(R))
    assert ideal.mode == "frames"
    S_ideal = _rows(ideal, ideal.replay(seq))
    S_bloch = _rows(ideal, ideal.replay(phys, transmit=1.0))                      # transmit= selects the Bloch route
    np.testing.assert_allclose(S_bloch, S_ideal, atol=3.0 / np.sqrt(pk.n_walkers))
    kap = np.where(np.arange(n)[:, None, None] * np.ones((n, n, 1)) > 0, 0.6, 1.0)
    ph = Phantom.compose(grid, fractions=full, orientation=Frames(R), layers={"kappa_B1": kap})
    S_b1 = _rows(ph, ph.replay(phys))                                            # the layer alone routes it too
    i = ph.voxel_index[:, 0]
    np.testing.assert_allclose(S_b1[i == 0], S_bloch[i == 0], rtol=1e-9)          # kappa = 1: nothing changes
    alone = np.abs(pk.replay_bloch(phys, b1_scale=0.6, orientation=(0.0, 0.0, 1.0), T2=[0.06] * 3))
    np.testing.assert_allclose(S_b1[i == 1], np.tile(0.7 * alone, (int((i == 1).sum()), 1)), rtol=1e-6)   # the file holds float32
    assert (S_b1[i == 1] < S_b1[i == 0] * 0.9).all()                              # a smaller flip, a smaller signal
    # the same map given at replay time, as a volume and as a function of position, is the same phantom
    np.testing.assert_allclose(_rows(ideal, ideal.replay(phys, transmit=kap)), S_b1, rtol=1e-6)
    x_split = ideal.grid.positions_m([[0, 0, 0], [1, 0, 0]])[:, 0].mean()
    np.testing.assert_allclose(_rows(ideal, ideal.replay(phys, transmit=lambda xyz: np.where(xyz[:, 0] > x_split, 0.6, 1.0))),
                               S_b1, rtol=1e-6)
    # file layer and replay-time map compose by multiplication
    np.testing.assert_allclose(_rows(ph, ph.replay(phys, transmit=1.0 / kap)), S_bloch, rtol=1e-6)
    with pytest.raises(ValueError, match="vector-Bloch|replay_bloch"):
        ph.file.replay(seq)                                     # the phase route itself refuses the layer
    mu = np.zeros((n, n, 1, 3)); mu[..., :] = (0.0, 0.0, 1.0)
    odf = Phantom.compose(grid, fractions=full, orientation=Watson(mu=mu, kappa=50.0))
    with pytest.raises(ValueError, match="frames-mode"):
        odf.replay(phys, transmit=0.8)
    with pytest.raises(ValueError, match="scalar, a volume"):
        ideal.replay(phys, transmit=np.ones(7))


def test_a_frame_is_one_pose_and_a_peak_is_that_pose_with_its_azimuth_unstated(pack_path):
    """RPH.md 4, the two modes side by side: a frames slot is a rotation, so its voxel is the pack replayed at
    that rotation; a peaks slot is the same direction with the azimuth left open, so its voxel is the average
    over that azimuth. They agree only where the substrate is axially symmetric."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    wm, *_ = _subs(pack_path)
    R0 = so3.rotation_of((0.3, 0.5, 0.81))
    R = np.zeros((n, n, 1, 3, 3)); R[..., :, :] = R0
    dirs = np.zeros((n, n, 1, 3)); dirs[..., :] = R0[:, 2]
    grid = Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3)
    full = {wm: np.ones((n, n, 1))}
    frames = Phantom.compose(grid, fractions=full, orientation=Frames(R))
    peaks = Phantom.compose(grid, fractions=full, orientation=Peaks(dirs))
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    S_fr, S_pk = _rows(frames, frames.replay(seq)), _rows(peaks, peaks.replay(seq))
    direct = np.abs(0.7 * pk.replay(seq, orientation=R0, T2=[0.06] * 3, complex_signal=True))
    np.testing.assert_allclose(S_fr[0], direct, atol=3.0 / np.sqrt(pk.n_walkers))
    np.testing.assert_allclose(S_pk, S_fr, atol=3.0 / np.sqrt(pk.n_walkers))     # a single cylinder: axially symmetric


def test_a_fan_is_a_frame_with_two_concentrations_and_contains_the_watson(pack_path):
    """Equal concentrations are the Watson cone of that width -- checked against the same analytic distribution
    composed directly -- and unequal ones are a fan the ODF mode cannot express. ``Fan.from_axis`` builds the
    frame from the two directions a user thinks in."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    wm, *_ = _subs(pack_path)
    grid = Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3)
    full = {wm: np.ones((n, n, 1))}
    axis = np.zeros((n, n, 1, 3)); axis[..., :] = (0.0, 0.0, 1.0)
    towards = np.zeros((n, n, 1, 3)); towards[..., :] = (1.0, 0.0, 0.3)          # not orthogonal: gets orthogonalised
    cone = Phantom.compose(grid, fractions=full, orientation=Fan.from_axis(axis=axis, fan_towards=towards, kappa_fan=6.0, kappa_perp=6.0))
    fan = Phantom.compose(grid, fractions=full, orientation=Fan.from_axis(axis=axis, fan_towards=towards, kappa_fan=0.2, kappa_perp=60.0))
    assert cone.mode == "bingham" and cone.file.meta["orientation"]["mode"] == "bingham"
    assert cone.file.bingham_kappa.shape == (cone.n_voxels, 1, 2)
    Rq = so3.rotations_from_quaternions(cone.file.pose_quat[0, 0])[0]
    np.testing.assert_allclose(Rq[:, 2], (0, 0, 1), atol=1e-6); np.testing.assert_allclose(Rq[:, 0], (1, 0, 0), atol=1e-6)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    S_cone, S_fan = _rows(cone, cone.replay(seq)), _rows(fan, fan.replay(seq))
    pr = pk.pose_response(seq, T2=[0.06] * 3)
    ref = np.abs(0.7 * pr.compose(so3.Distribution.watson(6.0, mu=(0, 0, 1), lmax=pr.lmax, nmax=pr.nmax)))
    np.testing.assert_allclose(S_cone[0], ref, rtol=1e-3)
    assert np.abs(S_fan - S_cone).max() > 3e-3                       # the fan is not that cone
    with pytest.raises(ValueError, match="parallel"):
        Fan.from_axis(axis=axis, fan_towards=axis, kappa_fan=1.0, kappa_perp=1.0)


def test_peaks_agree_with_a_concentrated_odf(pack_path):
    """RPH.md 4 states this as a MUST: a peak is the zero-dispersion limit of an axis density."""
    from dmipy_sim.replay import read_rpk
    n = 2
    wm, *_ = _subs(pack_path)
    d = np.zeros((n, n, 1, 3)); d[..., :] = (0.3, 0.5, 0.81)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    grid = Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3)
    full = {wm: np.ones((n, n, 1))}
    peaks = Phantom.compose(grid, fractions=full, orientation=Peaks(d))
    sharp = Phantom.compose(grid, fractions=full, orientation=Watson(mu=d, kappa=400.0, lmax=12))
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    np.testing.assert_allclose(_rows(peaks, peaks.replay(seq)), _rows(sharp, sharp.replay(seq)), atol=5e-3)


def test_the_same_substrate_cited_twice_is_a_crossing(pack_path):
    """Two populations of one solved microstructure at two poses, with two fractions: the voxel is
    their weighted sum, and its two slots cite the same substrate."""
    from dmipy_sim.replay import read_rpk, so3
    n = 2
    wm, *_ = _subs(pack_path)
    dirs = np.zeros((n, n, 1, 2, 3)); dirs[..., 0, :] = (0.0, 0.0, 1.0); dirs[..., 1, :] = (1.0, 0.0, 0.0)
    w = np.zeros((n, n, 1, 2)); w[..., 0], w[..., 1] = 0.7, 0.3
    ph = Phantom.compose(Grid(shape=(n, n, 1), voxel_size_m=(1e-3,) * 3), fractions={wm: np.ones((n, n, 1))},
                         orientation=Peaks(dirs, weights=w))
    assert (ph.file.substrate_id[:, :2] == 0).all()                  # one substrate, two slots
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    S = _rows(ph, ph.replay(seq))
    pr = pk.pose_response(seq, T2=[0.06] * 3)
    ref = np.abs(0.7 * pr.compose(so3.Distribution.axis((0, 0, 1), pr.lmax, pr.nmax))
                 + 0.3 * pr.compose(so3.Distribution.axis((1, 0, 0), pr.lmax, pr.nmax))) * 0.7
    np.testing.assert_allclose(S[0], ref, atol=pr.floor)             # 0.7 is the substrate's m0


def test_an_analytic_slot_ignores_the_orientation_it_is_given(pack_path):
    """A closed form has no pose, so whatever orientation field covers its voxels makes no difference."""
    from dmipy_sim.replay import read_rpk
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    out = []
    for axis in ((0.0, 0.0, 1.0), (0.3, 0.5, 0.81)):
        mu = np.zeros((8, 8, 1, 3)); mu[..., :] = axis
        ph, wm, csf, _ = _phantom(pack_path, orientation=Watson(mu=mu, kappa=2.0))
        f_csf = _rows(ph, ph.fraction(csf))
        out.append(_rows(ph, ph.replay(seq))[f_csf > 0.999])
    assert out[0].size and np.abs(out[0] - out[1]).max() < 1e-12
    wm, csf, bg = _subs(pack_path)
    with pytest.raises(ValueError, match="orientation-independent"):
        Phantom.compose(GRID8, fractions={wm: np.full((8, 8, 1), 0.5), csf: np.full((8, 8, 1), 0.5)},
                        orientation={wm: Peaks(np.zeros((8, 8, 1, 3)) + (0, 0, 1)), csf: Peaks(np.zeros((8, 8, 1, 3)) + (0, 0, 1))})


def test_a_prescribed_acquisition_must_share_the_grid_axes(pack_path):
    """The gradient and B0 directions are given in the scanner frame, so a sequence prescribed on other axes than
    the grid's is refused rather than rotated; a grid built from the prescription agrees by construction."""
    from dmipy_sim import Prescription
    from dmipy_sim.replay import read_rpk
    ph, *_ = _phantom(pack_path)
    pk = read_rpk(pack_path)
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    p_bad = Prescription(voxel_size_m=(1e-3,) * 3, matrix=(8, 8, 1), axes="LPS")
    with pytest.raises(ValueError, match="axes 'LPS'"):
        ph.replay(seq.with_prescription(p_bad))
    p_ok = Prescription(voxel_size_m=(1e-3,) * 3, matrix=(8, 8, 1))
    S = ph.replay(seq.with_prescription(p_ok))
    np.testing.assert_allclose(np.nan_to_num(S), np.nan_to_num(ph.replay(seq)))
    wm, csf, bg = _subs(pack_path)
    g = Grid.from_prescription(Prescription(voxel_size_m=(1e-3,) * 3, matrix=(8, 8, 1), axes="LPS"))
    ph2 = Phantom.compose(g, fractions={wm: np.ones((8, 8, 1))}, orientation=Watson(mu=_tangential(), kappa=12.0))
    assert np.isfinite(ph2.replay(seq.with_prescription(p_bad))).all()          # same axes: fine

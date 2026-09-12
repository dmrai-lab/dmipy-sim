"""``Phantom.partition``: one walk cut into voxels by where each walker started (RPH.md partition addressing,
dmipy-sim#76, #186). Membership is derived from the stored ``r0``, fractions are emergent from the weights, the
grid is free, and the anatomy rotates as a block."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import Encoding, Prescription, RFEvent, ScannerSequence
from dmipy_sim.phantom import FreeWater, Grid, Inert, PackSubstrate, PartitionPhantom, Phantom, Pose
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


@pytest.fixture(scope="module")
def walk():
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    return d.simulate_trajectories(2000, 2e-9, g, 4e-3, 5e-4, seed=1, require_gpu=False)


@pytest.fixture(scope="module")
def pack_paths(walk, tmp_path_factory):
    """The same walk at two K, plus the same walk split into its two pools by weight."""
    root = tmp_path_factory.mktemp("pk")
    out = {}
    for K in (2, 6):
        p = root / f"k{K}.rpk"
        build_replay_pack(walk, id=f"test/k{K}", license="x", citation="x", K=K, out_path=str(p))
        out[K] = str(p)
    comp = np.asarray(walk._bank_dict()["comp"]).reshape(2000, -1)[:, 0]
    for name, sel in (("intra", comp == 0), ("extra", comp != 0)):
        p = root / f"{name}.rpk"
        build_replay_pack(walk, id=f"test/{name}", license="x", citation="x", K=4, weights=sel.astype(float), out_path=str(p))
        out[name] = str(p)
    out["comp"] = comp
    return out


def _acq(pk, dirs, bvals, delta=6e-4, Delta=2e-3, rf=None):
    """A PGSE on the pack's save grid, as a bare gradient echo (rf None) or with a physical spin-echo schedule."""
    from dmipy_sim.constants import GAMMA
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g
        G[i, ng:ng + nd] = (amp if rf else -amp) * g
    enc = Encoding(bvalues=np.asarray(bvals, float), gradient_directions=np.asarray(dirs, float))
    if rf:
        return ScannerSequence(G=G, dt=dt, family="pgse", encoding=enc,
                               rf=[RFEvent(0.0, 90.0, axis_deg=0.0), RFEvent((n_t - 1) * dt / 2, 180.0, axis_deg=90.0)])
    return ScannerSequence(G=G, dt=dt, family="gre", encoding=enc)


def _grid(pk, size=2.5e-6, attach="substrate"):
    zext = float(np.ptp(pk.r0[:, 2])) + 1e-6
    return Grid.covering(pk.r0, voxel_size_m=(size, size, zext), attach=attach)


def _rows(ph, vol):
    return vol[tuple(ph.voxel_index.T)]


# ------------------------------------------------------------------ membership
def test_membership_is_derived_from_r0_and_independent_of_K(pack_paths):
    p2, p6 = read_rpk(pack_paths[2]), read_rpk(pack_paths[6])
    grid = _grid(p2)
    a = Phantom.partition(PackSubstrate(p2, m0=1.0), grid)
    b = Phantom.partition(PackSubstrate(p6, m0=1.0), grid)
    assert isinstance(a, PartitionPhantom) and a.mode == "rigid"
    ijk_a, in_a = a.membership(a.packs[0]); ijk_b, in_b = b.membership(b.packs[0])
    np.testing.assert_array_equal(ijk_a, ijk_b); assert in_a.all() and in_b.all()      # the bit, at any K
    np.testing.assert_array_equal(ijk_a, np.floor((p2.r0 - np.asarray(grid.corner_m)) / np.asarray(grid.voxel_size_m)).astype(int))
    counts = a.walkers_per_voxel(a.packs[0])
    assert counts.sum() == 2000 and a.n_voxels == int((counts > 0).sum())
    np.testing.assert_array_equal(a.voxel_index, np.argwhere(counts > 0))


def test_a_voxel_without_walkers_is_absent_or_is_the_outside_substrate(pack_paths):
    pk = read_rpk(pack_paths[2])
    big = Grid.covering(pk.r0, voxel_size_m=(2.5e-6, 2.5e-6, float(np.ptp(pk.r0[:, 2])) + 1e-6), margin_voxels=1)
    bare = Phantom.partition(PackSubstrate(pk, m0=0.7), big)
    assert bare.n_voxels < big.n_voxels                                              # the margin is not in the phantom
    assert np.isnan(bare.replay(_acq(pk, [[1, 0, 0]], [1e9]))[0, 0, 0, 0])
    with pytest.raises(ValueError, match="outside="):                                # a declared slot alone cannot fill a voxel
        Phantom.partition(PackSubstrate(pk, m0=0.7), big, declared={Inert(name="myelin"): np.full(big.shape, 0.3)}).fraction(0)
    csf = FreeWater(D_m2_s=3e-9, m0=1.0)
    ph = Phantom.partition(PackSubstrate(pk, m0=0.7), big, outside=csf)
    f_csf, f_pk = ph.fraction(csf), ph.fraction(ph.packs[0])
    assert ph.n_voxels == big.n_voxels and f_csf[0, 0, 0] == 1.0 and f_pk[0, 0, 0] == 0.0
    np.testing.assert_allclose(f_csf + f_pk, 1.0)
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    S = ph.replay(seq)
    np.testing.assert_allclose(S[0, 0, 0, 0], np.exp(-1e9 * 3e-9))                    # m0 = 1 free water, alone


# ------------------------------------------------------------------ fractions
def test_fractions_are_emergent_from_the_weights_and_declared_only_where_nothing_weighs_them(pack_paths):
    intra, extra = read_rpk(pack_paths["intra"]), read_rpk(pack_paths["extra"])
    comp = pack_paths["comp"]
    grid = _grid(intra)
    wm_in, wm_ex = PackSubstrate(intra, m0=1.0, name="intra"), PackSubstrate(extra, m0=1.0, name="extra")
    ph = Phantom.partition([wm_in, wm_ex], grid)
    f_in, f_ex = ph.fraction(wm_in), ph.fraction(wm_ex)
    np.testing.assert_allclose(_rows(ph, f_in + f_ex), 1.0)
    # weighted counts ARE the fractions: the intra weight is the intra mask, so its fraction is the intra count share
    ijk, _ = grid.bin(intra.r0)
    for v in ph.voxel_index[:5]:
        m = (ijk == v).all(axis=1)
        np.testing.assert_allclose(f_in[tuple(v)], (comp[m] == 0).mean(), atol=1e-12)
    # a declared walker-less slot takes its share first
    myelin = Inert(name="myelin")
    dec = Phantom.partition([wm_in, wm_ex], grid, declared={myelin: np.full(grid.shape, 0.3)})
    np.testing.assert_allclose(_rows(dec, dec.fraction(myelin)), 0.3)
    np.testing.assert_allclose(_rows(dec, dec.fraction(wm_in)), 0.7 * _rows(ph, f_in), atol=1e-12)
    with pytest.raises(ValueError, match="emergent"):
        Phantom.partition(wm_in, grid, declared={wm_ex: np.full(grid.shape, 0.3)})


# ------------------------------------------------------------------ the free grid
def test_regrid_is_an_exact_rebin(pack_paths):
    pk = read_rpk(pack_paths[6])
    fine = _grid(pk, 2.5e-6)
    ph = Phantom.partition(PackSubstrate(pk, m0=1.0), fine)
    coarse = ph.regrid(voxel_size_m=(5e-6, 5e-6, fine.voxel_size_m[2]))
    direct = Phantom.partition(PackSubstrate(pk, m0=1.0), coarse.grid)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 2e9])
    Sc, Sd = coarse.replay(seq, complex_signal=True), direct.replay(seq, complex_signal=True)
    np.testing.assert_array_equal(coarse.voxel_index, direct.voxel_index)
    np.testing.assert_allclose(np.nan_to_num(Sc), np.nan_to_num(Sd), atol=1e-15)     # built by regrid == built directly
    # and each coarse voxel is the weight-merged sum of its fine voxels
    Sf = ph.replay(seq, complex_signal=True)
    Wf = ph.walkers_per_voxel(ph.packs[0])
    for V in coarse.voxel_index[:4]:
        i, j = 2 * V[0], 2 * V[1]
        block = Sf[i:i + 2, j:j + 2, V[2]]; w = Wf[i:i + 2, j:j + 2, V[2]]
        merged = np.nansum(np.nan_to_num(block) * w[..., None], axis=(0, 1)) / w.sum()
        np.testing.assert_allclose(Sc[tuple(V)], merged, rtol=1e-12)
    assert coarse.grid.corner_m == pytest.approx(fine.corner_m)


def test_the_scanner_decides_the_voxels_when_the_partition_declares_no_grid(pack_paths):
    pk = read_rpk(pack_paths[2])
    free = Phantom.partition(PackSubstrate(pk, m0=1.0), outside=Inert())
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    with pytest.raises(ValueError, match="Prescription"):
        free.replay(seq)
    grid = _grid(pk)
    p = Prescription(isocenter_m=grid.isocenter_m, voxel_size_m=grid.voxel_size_m, matrix=grid.shape, origin_m=grid.origin_m)
    S = free.replay(seq.with_prescription(p))
    bound = Phantom.partition(PackSubstrate(pk, m0=1.0), Grid.from_prescription(p), outside=Inert())
    np.testing.assert_allclose(np.nan_to_num(S), np.nan_to_num(bound.replay(seq)))
    assert Grid.from_prescription(p).attach == "lab"


# ------------------------------------------------------------------ the pose
def test_a_pose_on_a_tissue_attached_grid_is_the_rotated_acquisition_with_membership_unchanged(pack_paths):
    from dmipy_sim.replay import so3
    pk = read_rpk(pack_paths[6])
    grid = _grid(pk, attach="substrate")
    R = so3.rotation_of((0.3, 0.5, 0.81))
    ph = Phantom.partition(PackSubstrate(pk, m0=1.0), grid)
    posed = ph.with_pose(Pose(R))
    np.testing.assert_array_equal(posed.voxel_index, ph.voxel_index)
    np.testing.assert_array_equal(posed.membership(posed.packs[0])[0], ph.membership(ph.packs[0])[0])
    seq = _acq(pk, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1e9, 1e9, 1e9])
    rotated = seq.with_gradient(np.asarray(seq.G) @ R)                # g . (R r) = (R^T g) . r
    np.testing.assert_allclose(np.nan_to_num(posed.replay(seq, complex_signal=True)),
                               np.nan_to_num(ph.replay(rotated, complex_signal=True)), rtol=1e-5)   # G is stored float32
    assert not np.allclose(np.nan_to_num(posed.replay(seq)), np.nan_to_num(ph.replay(seq)))    # the physics did change


def test_a_pose_on_a_bore_attached_grid_rebins_the_walkers(pack_paths):
    from dmipy_sim.replay import so3
    pk = read_rpk(pack_paths[6])
    R = so3.rotation_of((0.0, 1.0, 0.0))                                # tip the cylinder axis into the plane
    lab = Grid.covering(pk.r0 @ R.T, voxel_size_m=(2.5e-6, 2.5e-6, 2.5e-6), attach="lab", margin_voxels=1)
    ph = Phantom.partition(PackSubstrate(pk, m0=1.0), lab, outside=Inert()).with_pose(Pose(R))
    ijk, inside = ph.membership(ph.packs[0])
    ref, ref_in = lab.bin(pk.r0 @ R.T)
    np.testing.assert_array_equal(ijk, ref); assert inside.all()
    same_grid_unposed = Phantom.partition(PackSubstrate(pk, m0=1.0), lab, outside=Inert())
    assert not np.array_equal(ph.walkers_per_voxel(ph.packs[0]), same_grid_unposed.walkers_per_voxel(same_grid_unposed.packs[0]))
    np.testing.assert_allclose(_rows(ph, ph.fraction(ph.packs[0]) + ph.fraction(ph.outside)), 1.0)


# ------------------------------------------------------------------ maps per walker
def test_transmit_on_a_partition_is_the_per_walker_scale(pack_paths):
    pk = read_rpk(pack_paths[6])
    grid = _grid(pk)
    ph = Phantom.partition(PackSubstrate(pk, m0=0.8), grid)
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9], rf=True)
    kap = np.where(np.arange(grid.shape[0])[:, None, None] * np.ones(grid.shape) < grid.shape[0] // 2, 0.6, 1.0)
    S = ph.replay(seq, transmit=kap, complex_signal=True)
    # by hand: each walker at its own voxel's scale, summed by voxel
    ijk, _ = grid.bin(pk.r0)
    b1_w = kap[tuple(ijk.T)]
    w, ew, E = pk.walker_signals(seq, b1_scale=b1_w)
    for v in ph.voxel_index[:6]:
        m = (ijk == v).all(axis=1)
        ref = 0.8 * (ew[m, None] * E[m]).sum(0) / w[m].sum()
        np.testing.assert_allclose(S[tuple(v)], ref, rtol=1e-10)
    S1 = ph.replay(seq, transmit=1.0)
    S0 = ph.replay(seq)
    np.testing.assert_allclose(np.nan_to_num(S1), np.nan_to_num(S0), atol=3.0 / np.sqrt(60))   # ideal pulses vs propagated
    # a 0.6 transmit scale plays 54 / 108 degree pulses: the region's signal drops well below the ideal one. Summed
    # over the scaled voxels (a voxel holds ~60 walkers, so a per-voxel ratio is realisation-dependent: it sat at
    # 0.89 on aarch64 and crossed 0.95 on x86 for the same fixture)
    assert np.nansum(np.abs(S)[kap < 1]) < 0.8 * np.nansum(S1[kap < 1])


def test_off_resonance_and_proton_density_on_a_partition(pack_paths):
    from dmipy_sim.constants import GAMMA
    pk = read_rpk(pack_paths[2])
    grid = _grid(pk)
    ph = Phantom.partition(PackSubstrate(pk, m0=1.0), grid)
    seq = _acq(pk, [[1, 0, 0]], [1e9])
    S0 = ph.replay(seq, complex_signal=True)
    S = ph.replay(seq, off_resonance=1e-7, complex_signal=True)
    TE = (pk.n_t - 1) * pk.dt
    ang = np.angle(_rows(ph, S) / _rows(ph, S0))
    np.testing.assert_allclose(ang, GAMMA * 1e-7 * TE, rtol=1e-6)        # a gradient echo: the whole TE
    # with a transmit scale the offset goes through the propagation instead, walker by walker, to the same phase
    se = _acq(pk, [[1, 0, 0]], [1e9], rf=True)
    Sb = ph.replay(se, transmit=1.0, off_resonance=1e-7, complex_signal=True)
    Sb0 = ph.replay(se, transmit=1.0, complex_signal=True)
    np.testing.assert_allclose(np.abs(_rows(ph, Sb)), np.abs(_rows(ph, Sb0)), rtol=1e-6)    # a spin echo refocuses it
    np.testing.assert_allclose(np.nan_to_num(ph.replay(seq, proton_density=0.5)), 0.5 * np.nan_to_num(np.abs(S0)))


# ------------------------------------------------------------------ the file
def test_a_partition_writes_and_reads_back(pack_paths, tmp_path):
    pk = read_rpk(pack_paths[6])
    grid = _grid(pk, attach="lab")
    myelin = Inert(name="myelin")
    ph = Phantom.partition(PackSubstrate(pk, m0=0.7, name="wm", T2_s=[0.06, 0.06, 0.06]), grid,
                           declared={myelin: np.full(grid.shape, 0.2)}, outside=FreeWater(D_m2_s=3e-9, m0=1.0),
                           pose=Pose(np.eye(3)))
    meta = ph.write(tmp_path / "part.rph", id="t/part", license="x", citation="x", embed=True)
    assert meta["addressing"] == "partition" and meta["substrates"][0]["addressing"] == "partition"
    back = Phantom.read(tmp_path / "part.rph")
    assert isinstance(back, PartitionPhantom) and back.grid == grid and back.outside.name == "csf/free-water"
    seq = _acq(pk, [[1, 0, 0], [0, 0, 1]], [1e9, 1e9])
    np.testing.assert_allclose(np.nan_to_num(back.replay(seq)), np.nan_to_num(ph.replay(seq)), rtol=1e-6)
    np.testing.assert_allclose(back.fraction("myelin"), ph.fraction(myelin))
    assert back.substrates[0].tissue == {"T2": [0.06, 0.06, 0.06]}

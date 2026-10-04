"""Every pack declares its susceptibility field (the replay equation's C3) present or absent, and a tissue's chi meets
it and the scanner's B0 by one rule (dmipy-sim#593, ``replay.field_term``):

* absent -- nothing in the substrate is magnetic: a non-zero ``chi_iso`` / ``chi_aniso`` is refused by name, and any
  B0, of any strength along any axis, is accepted and changes nothing;
* present -- a non-zero chi with no B0 (``scanner=None``) is refused; a zero or absent chi at any B0 is the
  field-free signal, and chi with a B0 replays the field.

The spec, the pack and the layouts carry the declaration; one without it is refused."""
import copy

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import Encoding, ScannerSequence
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.constants import GAMMA
from dmipy_sim.fields.susceptibility_field import field_grid_of
from dmipy_sim.phantom import FreeWater, Grid, Inert, PackSubstrate, Peaks, Phantom
from dmipy_sim.replay import ReplayPack, so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.spec.substrate import SpecError, SubstrateSpec, load_spec
from dmipy_sim.spec.tissue import Tissue
from dmipy_sim.spec.walk import fill_field
from dmipy_sim.substrate.biophysical_constants import canonical_white_matter

D0 = 2.0e-9
ENV = dict(bvals=[0.0, 1e8], dirs=[[0, 0, 1], [1, 0, 0]], delta_frac=0.2, Delta_frac=0.5,
           ogse_periods=[1], shortd_b=1e8, shortd_deltas_frac=[0.2])
WM = canonical_white_matter(field_T=3.0)
MYELIN = Tissue(chi_iso=float(WM["chi_iso_myelin"]), chi_aniso=float(WM["delta_chi_a"]))   # the catalogue's chi


def _pgse(pk, dirs, bvals, delta=1e-3, Delta=3e-3):
    """A bipolar gradient echo on the pack's save grid."""
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g; G[i, ng:ng + nd] = -amp * g
    return ScannerSequence(G=G, dt=dt, family="gre",
                           encoding=Encoding(bvalues=np.asarray(bvals, float), gradient_directions=np.asarray(dirs, float)))


@pytest.fixture(scope="module")
def absent():
    """An impermeable cylinder: nothing in it is magnetic, its pack declares the field absent."""
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    walk = d.simulate_trajectories(400, D0, g, 6e-3, 3e-4, seed=1, require_gpu=False)
    pk = build_replay_pack(walk, id="test/absent", license="x", citation="x", K=8, envelope=ENV)
    return pk, _pgse(pk, [[1, 0, 0], [0, 0, 1]], [1e8, 1e8])


@pytest.fixture(scope="module")
def present():
    """One myelinated cylinder in a periodic cell with its field channel: the pack declares the field present."""
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    walk = d.simulate_trajectories(600, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    walk = fill_field(walk, field_grid_of(g, res=0.2e-6))
    pk = build_replay_pack(walk, id="test/present", license="x", citation="x", K=8, envelope=ENV, susc_path_K=16)
    return pk, _pgse(pk, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [0.0, 1e8, 1e8])


def test_the_spec_and_the_pack_carry_the_declaration(absent, present):
    pa, _ = absent
    pp, _ = present
    assert pa.susceptibility_field == "absent" and pa.substrate.susceptibility_field == "absent" and not pa.has_field
    assert pp.susceptibility_field == "present" and pp.substrate.susceptibility_field == "present" and pp.has_field
    assert pa.meta["susceptibility_field"] == "absent" and pp.meta["susceptibility_field"] == "present"


# ---- case 1: absent + chi is refused, by name, on every route
def test_absent_refuses_a_tissue_chi(absent):
    pk, seq = absent
    for t in (Tissue(chi_iso=1e-7), Tissue(chi_aniso=-1e-8), MYELIN):
        for scanner in (None, 3.0):
            with pytest.raises(ValueError, match="declares its susceptibility\\s+field absent"):
                pk.replay(seq, tissue=t, scanner=scanner)
    with pytest.raises(ValueError, match="chi_aniso = -1e-08"):
        pk.walker_signals(seq, tissue=Tissue(chi_aniso=-1e-8), scanner=3.0)
    with pytest.raises(ValueError, match="field absent"):
        pk.pose_response(seq, tissue=MYELIN, scanner=3.0, keep=(4, 0))
    with pytest.raises(ValueError, match="field absent"):
        pk.walker_primitives(Acquisition(seq)).signal(MYELIN, 3.0)          # the study / columnar / shape-moment route
    with pytest.raises(ValueError, match="field absent"):
        FreeWater(m0=1.0, tissue=Tissue(D=3e-9, chi_iso=1e-7))


# ---- case 2: absent + any B0 is accepted and changes nothing
def test_absent_accepts_any_B0_and_it_changes_nothing(absent):
    pk, seq = absent
    ref = pk.replay(seq, complex_signal=True)
    bi_planar = ScannerLimits(name="bi-planar", kind="scanner", regime="default", G_max=0.04, slew_max=150.0, field_T=0.5,
                              b0_axis=(0.0, 1.0, 0.0))
    for scanner in (3.0, 7.0, 0.064, bi_planar):
        np.testing.assert_array_equal(pk.replay(seq, scanner=scanner, complex_signal=True), ref)
        np.testing.assert_array_equal(pk.replay(seq, scanner=scanner, tissue=Tissue(chi_iso=0.0), complex_signal=True), ref)
    R = so3.haar_rotations(1, seed=3)[0]
    np.testing.assert_array_equal(pk.replay(seq, orientation=R, scanner=7.0, complex_signal=True),
                                  pk.replay(seq, orientation=R, complex_signal=True))
    np.testing.assert_array_equal(pk.pose_response(seq, scanner=7.0, keep=(4, 0)).coeffs, pk.pose_response(seq, keep=(4, 0)).coeffs)
    prim = pk.walker_primitives(Acquisition(seq))
    np.testing.assert_array_equal(prim.signal(None, 7.0), prim.signal(None, None))
    np.testing.assert_array_equal(pk.study(Study(Protocol([Acquisition(seq)]), scanners=[None, 3.0, 7.0]))[1:],
                                  np.repeat(pk.study(Study(Protocol([Acquisition(seq)])))[:1], 2, axis=0))


# ---- case 3: present + a non-zero chi + no B0 is refused
def test_present_refuses_chi_without_a_B0(present):
    pk, seq = present
    for t in (MYELIN, Tissue(chi_iso=-1e-7), Tissue(chi_aniso=-1e-8)):
        with pytest.raises(ValueError, match="no scanner field"):
            pk.replay(seq, tissue=t)
    with pytest.raises(ValueError, match="no scanner field"):
        pk.pose_response(seq, tissue=MYELIN, keep=(4, 0))
    with pytest.raises(ValueError, match="no scanner field"):
        pk.walker_primitives(Acquisition(seq)).signal(MYELIN, None)
    with pytest.raises(ValueError, match="no scanner field"):
        pk.study(Study(Protocol([Acquisition(seq)]), tissues=[MYELIN]))


def test_present_with_no_or_zero_chi_at_a_B0_is_the_field_free_signal(present):
    pk, seq = present
    ref = pk.replay(seq, complex_signal=True)
    for t in (None, Tissue(), Tissue(chi_iso=0.0, chi_aniso=0.0)):
        np.testing.assert_array_equal(pk.replay(seq, tissue=t, scanner=3.0, complex_signal=True), ref)
    np.testing.assert_array_equal(pk.replay(seq, tissue=MYELIN, scanner=0.0, complex_signal=True), ref)


# ---- case 4: present + chi + B0 replays the field, the same number on every route
def test_present_with_chi_and_a_B0_replays_the_field(present):
    pk, seq = present
    S = pk.replay(seq, tissue=MYELIN, scanner=3.0, complex_signal=True)
    assert np.abs(S - pk.replay(seq, complex_signal=True)).max() > 1e-4          # the field reaches the signal
    np.testing.assert_allclose(pk.walker_primitives(Acquisition(seq)).signal(MYELIN, 3.0, complex_signal=True), S, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(pk.study(Study(Protocol([Acquisition(seq)]), tissues=[MYELIN], scanners=[3.0]))[0],
                               np.abs(S), rtol=1e-9, atol=1e-12)


def test_the_brain_scripts_scanner_none_composition_is_refused(present):
    """The replayable_mc brain script replayed a composition whose white matter carried the catalogue's myelin chi
    with ``scanner=None``, and the myelin field silently left the signal (5.6e-4 of M0, disco-space#36)."""
    pk, seq = present
    wm = PackSubstrate(pk, m0=0.7, name="wm", tissue=MYELIN)
    csf = FreeWater(m0=1.0, tissue=Tissue(D=3e-9))
    axis = np.array([1.0, 0.0, 0.0])
    ph = Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3,) * 3),
                         fractions={wm: np.full((1, 1, 1), 0.8), csf: np.full((1, 1, 1), 0.2)}, remainder=Inert(),
                         orientation={wm: Peaks(np.broadcast_to(axis, (1, 1, 1, 1, 3)).copy())})
    with pytest.raises(ValueError, match="no scanner field"):
        ph.replay(seq, scanner=None)
    S = ph.replay(seq, scanner=3.0)
    assert np.all(np.isfinite(S))


# ---- no undeclared state: the spec, the pack and the builder refuse one without the declaration
def test_a_spec_without_the_declaration_or_against_its_pools_is_refused():
    from pathlib import Path
    fix = Path(__file__).parent / "fixtures" / "substrates"
    cyl = load_spec(fix / "isolated_cylinder.sub.json").to_dict()
    pm = load_spec(fix / "packed_myelinated_cylinders.sub.json").to_dict()
    with pytest.raises(SpecError, match="susceptibility_field"):
        SubstrateSpec.from_dict({k: v for k, v in cyl.items() if k != "susceptibility_field"})
    with pytest.raises(SpecError, match="'present' but no pool is a field source"):
        SubstrateSpec.from_dict(dict(cyl, susceptibility_field="present")).validate()
    with pytest.raises(SpecError, match="'absent' but the pool"):
        SubstrateSpec.from_dict(dict(pm, susceptibility_field="absent")).validate()
    with pytest.raises(SpecError, match="one of"):
        SubstrateSpec.from_dict(dict(cyl, susceptibility_field="zero")).validate()


def test_a_pack_without_the_declaration_or_against_its_content_is_refused(absent, present):
    pa, seq = absent
    pp, _ = present
    bare = ReplayPack(dict(pa.arrays), {k: v for k, v in copy.deepcopy(pa.meta).items() if k != "susceptibility_field"})
    with pytest.raises(ValueError, match="declares no susceptibility_field"):
        bare.replay(seq)
    lying = ReplayPack(dict(pp.arrays), dict(copy.deepcopy(pp.meta), susceptibility_field="absent"))
    with pytest.raises(ValueError, match="contradicts itself"):
        lying.replay(seq)
    meta = copy.deepcopy(pa.meta); meta["substrate"]["susceptibility_field"] = "present"
    with pytest.raises(ValueError, match="disagree"):
        ReplayPack(dict(pa.arrays), meta).replay(seq)


def test_a_stamp_declares_a_pack_in_its_metadata_only(absent, tmp_path):
    pa, seq = absent
    meta = copy.deepcopy(pa.meta)
    meta.pop("susceptibility_field"); meta["substrate"].pop("susceptibility_field")
    old = ReplayPack(dict(pa.arrays), meta)
    with pytest.raises(ValueError, match="stores no field channel"):
        old.stamp_susceptibility_field("present")
    old.stamp_susceptibility_field("absent", out_path=tmp_path / "stamped.rpk")
    back = ReplayPack.load(tmp_path / "stamped.rpk")
    assert back.susceptibility_field == "absent" and back.substrate.susceptibility_field == "absent"
    assert set(back.arrays) == set(pa.arrays) and all(np.array_equal(back.arrays[k], pa.arrays[k]) for k in pa.arrays)
    np.testing.assert_array_equal(back.replay(seq), pa.replay(seq))


def test_the_builder_needs_a_declaration_for_a_walk_without_a_spec():
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    walk = d.simulate_trajectories(100, D0, g, 2e-3, 2e-4, seed=2, require_gpu=False)
    m = walk._bank_dict()
    m.pop("substrate", None)
    with pytest.raises(ValueError, match="pass susceptibility_field="):
        build_replay_pack(m, id="t", license="x", citation="x", K=4)
    pk = build_replay_pack(m, id="t", license="x", citation="x", K=4, susceptibility_field="absent")
    assert pk.susceptibility_field == "absent"
    with pytest.raises(ValueError, match="no field basis"):
        build_replay_pack(m, id="t", license="x", citation="x", K=4, susceptibility_field="present")
    with pytest.raises(ValueError, match="declared differently"):
        build_replay_pack(walk, id="t", license="x", citation="x", K=4, susceptibility_field="present")

"""A substrate's susceptibility field (the replay equation's C3) is present exactly when one of its pools is magnetic
(carries a ``susceptibility`` block), and a tissue's chi meets it and the scanner's B0 by one rule (dmipy-sim#593,
``replay.field_term``):

* absent -- nothing in the substrate is magnetic: a non-zero ``chi_iso`` / ``chi_aniso`` is refused by name, and any
  B0, of any strength along any axis, is accepted and changes nothing;
* present -- a non-zero chi with no B0 (``scanner=None``) is refused; a zero or absent chi at any B0 is the
  field-free signal, and chi with a B0 replays the field.

Every spec, pack and layout holds one equivalence (dmipy-sim#597, ``spec.substrate.susceptibility_field_of``): some
pool magnetic <=> ``"field"`` in ``validity.tiers`` <=> the field channel (C3) is stored. Each mismatch is refused at
write and at read, whatever the surface kind; a pack without an embedded spec declares nothing and is refused; an
analytic or inert phantom substrate has no field by its kind."""
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
from dmipy_sim.spec.substrate import SpecError, SubstrateSpec, load_spec, susceptibility_field_of
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
def absent_walk():
    """An impermeable cylinder: nothing in it is magnetic."""
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    return d.simulate_trajectories(400, D0, g, 6e-3, 3e-4, seed=1, require_gpu=False)


@pytest.fixture(scope="module")
def present_walk():
    """One myelinated cylinder in a periodic cell, its field read along the walk: the myelin pool is magnetic."""
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    walk = d.simulate_trajectories(600, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    return fill_field(walk, field_grid_of(g, res=0.2e-6))


@pytest.fixture(scope="module")
def absent(absent_walk):
    """The cylinder's pack: no magnetic pool, no field channel, the field absent."""
    pk = build_replay_pack(absent_walk, id="test/absent", license="x", citation="x", K=8, envelope=ENV)
    return pk, _pgse(pk, [[1, 0, 0], [0, 0, 1]], [1e8, 1e8])


@pytest.fixture(scope="module")
def present(present_walk):
    """The myelinated cell's pack with its field channel: the field present."""
    pk = build_replay_pack(present_walk, id="test/present", license="x", citation="x", K=8, envelope=ENV, susc_path_K=16)
    return pk, _pgse(pk, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [0.0, 1e8, 1e8])


def test_the_field_is_read_from_the_pools(absent, present):
    pa, _ = absent
    pp, _ = present
    assert pa.susceptibility_field == "absent" and susceptibility_field_of(pa.substrate) == "absent" and not pa.has_field
    assert pp.susceptibility_field == "present" and susceptibility_field_of(pp.substrate) == "present" and pp.has_field
    assert [p.name for p in pp.substrate.field_source_pools] == ["myelin"] and "field" in pp.substrate.validity.tiers
    for pk in (pa, pp):                                       # the pools are the declaration: no key beside them
        assert "susceptibility_field" not in pk.meta and "susceptibility_field" not in pk.meta["substrate"]


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




# ---- the equivalence: some pool magnetic <=> "field" in validity.tiers <=> C3 stored, on every surface kind (#597)
CHI = {"chi_iso": None, "chi_aniso": None, "director": "none"}      # a magnetic pool whose chi is a replay knob


def _surface_specs(tmp):
    """One spec per surface kind a producer uses, each with no magnetic pool, and the pool a test makes magnetic."""
    from dmipy_sim.geometry.label_volume import LabelVolume
    from dmipy_sim.geometry.sphere_union import SphereUnion
    from dmipy_sim.spec import spec_of
    from tests.test_mesh_spec import _bundle_spec
    lab = np.ones((12, 6, 6), np.uint8); lab[3:9] = 0
    mesh = _bundle_spec(tmp).to_dict()
    mesh["pools"][2]["susceptibility"] = None
    mesh["validity"]["tiers"] = [t for t in mesh["validity"]["tiers"] if t != "field"]
    return {"cylinder": (spec_of(d.Cylinder(radius=3e-6, orientation=(0, 0, 1))).to_dict(), "intra"),
            "sphere": (spec_of(d.Sphere(radius=3e-6)).to_dict(), "intra"),
            "mesh": (mesh, "myelin"),
            "sphere_union": (spec_of(SphereUnion([[0.0, 0.0, 0.0], [2e-6, 0.0, 0.0]], 1.5e-6)).to_dict(), "intra"),
            "label_volume": (spec_of(LabelVolume(lab, 0.5e-6), surface_dir=str(tmp)).to_dict(), "free"),
            "swept_polyline": (spec_of(d.CurvedCylinder([[0, 0, -5e-6], [0, 1e-6, 0], [0, 0, 5e-6]], 1e-6)).to_dict(), "intra")}


@pytest.fixture(scope="module")
def surface_specs(tmp_path_factory):
    return _surface_specs(tmp_path_factory.mktemp("surfaces"))


def _variant(spec, pool, magnetic, tier):
    """``spec`` with ``pool`` magnetic or not and the ``"field"`` tier listed or not."""
    v = copy.deepcopy(spec)
    for p in v["pools"]:
        p["susceptibility"] = dict(CHI) if (magnetic and p["name"] == pool) else None
    v["validity"]["tiers"] = [t for t in v["validity"]["tiers"] if t != "field"] + (["field"] if tier else [])
    return v


#: (magnetic pool, field tier, C3 stored) -> the refusal, for every mismatch direction
MISMATCHES = [
    ((True, False, True), "lists no 'field'"),                       # a magnetic pool without the field tier
    ((True, False, False), "lists no 'field'"),
    ((False, True, True), "lists 'field' and no pool is magnetic"),  # the field tier without a magnetic pool
    ((False, True, False), "lists 'field' and no pool is magnetic"),
    ((True, True, False), "is not stored"),                          # magnetic pool and tier, no C3 stored
    ((False, False, True), "is stored and no pool is magnetic"),     # C3 stored, no magnetic pool
]
KINDS = ("cylinder", "sphere", "mesh", "sphere_union", "label_volume", "swept_polyline")


def _pack_with(base, spec):
    """``base``'s tensors (with or without C3) under ``spec``."""
    meta = copy.deepcopy(base.meta); meta["substrate"] = spec
    return ReplayPack(dict(base.arrays), meta)


@pytest.mark.parametrize("kind", KINDS)
def test_the_reader_refuses_every_mismatch_on_every_surface_kind(kind, surface_specs, absent, present, tmp_path):
    spec, pool = surface_specs[kind]
    for (mag, tier, stored), msg in MISMATCHES:
        pk = _pack_with((present if stored else absent)[0], _variant(spec, pool, mag, tier))
        with pytest.raises(SpecError, match=msg):
            pk.susceptibility_field
        with pytest.raises(SpecError, match=msg):
            pk.replay((present if stored else absent)[1])
        with pytest.raises(SpecError, match=msg):                        # at load
            pk.save(tmp_path / f"{kind}.rpk"); ReplayPack.load(tmp_path / f"{kind}.rpk")
    assert _pack_with(absent[0], _variant(spec, pool, False, False)).susceptibility_field == "absent"
    assert _pack_with(present[0], _variant(spec, pool, True, True)).susceptibility_field == "present"


@pytest.mark.parametrize("kind", KINDS)
def test_the_writers_refuse_every_mismatch_on_every_surface_kind(kind, surface_specs, absent_walk, present_walk, absent, present):
    from dmipy_sim.replay.bank import _walk_master, merge_packs
    spec, pool = surface_specs[kind]
    master = {False: _walk_master(absent_walk), True: _walk_master(present_walk)}
    for (mag, tier, stored), msg in MISMATCHES:
        v = _variant(spec, pool, mag, tier)
        with pytest.raises(SpecError, match=msg):                        # build_replay_pack, before anything is encoded
            build_replay_pack(dict(master[stored], substrate=v), id="t", license="x", citation="x", K=4)
        shard = _pack_with((present if stored else absent)[0], v)
        with pytest.raises(SpecError, match=msg):                        # merge_packs, on every shard
            merge_packs([shard, shard], id="t/merged")


def _columnar_dir(tmp, base, spec, stored):
    """A columnar layout's manifest and index around ``base``'s meta under ``spec``, with or without field modes."""
    import json
    from dmipy_sim.phantom import Grid
    meta = copy.deepcopy(base.meta); meta["substrate"] = spec
    meta["fidelity"]["per_voxel"] = dict(grid=Grid(shape=(1, 1, 1), voxel_size_m=(1e-3,) * 3).to_meta(), floor_median=0.01)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "manifest.json").write_text(json.dumps(dict(meta=meta, columns={})))
    (tmp / "index.json").write_text(json.dumps(dict(n_rows=0, band_groups=[8], path_groups=([16] if stored else []))))
    return str(tmp)


def _moments_dir(tmp, spec, stored):
    """A tier shape-moment layout's manifest under ``spec``, its one group storing field columns or not."""
    import json
    from dmipy_sim.phantom import Grid
    from dmipy_sim.replay.shape_moments import FORMAT, TILE
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "manifest.json").write_text(json.dumps(dict(
        format=FORMAT, n_rows=0, n_tiles=0, tile=TILE, columns=dict(w="w.npy", tiles="tiles.npy"),
        source=dict(grid=Grid(shape=(1, 1, 1), voxel_size_m=(1e-3,) * 3).to_meta(), pack="t"),
        tiers=dict(substrate=spec, groups=dict(g0=dict(field=stored))))))
    return str(tmp)


@pytest.mark.parametrize("kind", KINDS)
def test_the_layouts_refuse_every_mismatch_on_every_surface_kind(kind, surface_specs, absent, present, tmp_path):
    from dmipy_sim.replay.columnar import ColumnarPack
    from dmipy_sim.replay.shape_moments import ShapeMoments
    spec, pool = surface_specs[kind]
    for i, ((mag, tier, stored), msg) in enumerate(MISMATCHES):
        v = _variant(spec, pool, mag, tier)
        with pytest.raises(SpecError, match=msg):
            ColumnarPack(_columnar_dir(tmp_path / f"col{i}", (present if stored else absent)[0], v, stored))
        with pytest.raises(SpecError, match=msg):
            ShapeMoments(_moments_dir(tmp_path / f"sm{i}", v, stored))
    ok = ColumnarPack(_columnar_dir(tmp_path / "col_ok", present[0], _variant(spec, pool, True, True), True))
    assert ok.susceptibility_field == "present"
    assert ShapeMoments(_moments_dir(tmp_path / "sm_ok", _variant(spec, pool, False, False), False)).susceptibility_field == "absent"


def test_a_spec_on_its_own_holds_the_pool_and_tier_half():
    """A spec stores no channel: its pools and its tiers agree, or it is refused."""
    from pathlib import Path
    fix = Path(__file__).parent / "fixtures" / "substrates"
    cyl = load_spec(fix / "isolated_cylinder.sub.json").to_dict()
    pm = load_spec(fix / "packed_myelinated_cylinders.sub.json").to_dict()
    assert susceptibility_field_of(cyl) == "absent" and susceptibility_field_of(pm) == "present"
    with pytest.raises(SpecError, match="lists no 'field'"):
        SubstrateSpec.from_dict(_variant(cyl, "intra", True, False)).validate()
    with pytest.raises(SpecError, match="no pool is magnetic"):
        SubstrateSpec.from_dict(_variant(pm, "myelin", False, True)).validate()
    with pytest.raises(SpecError, match="unknown keys"):
        SubstrateSpec.from_dict(dict(cyl, susceptibility_field="absent"))      # the pools are the declaration


def test_a_pack_without_an_embedded_spec_is_refused(absent, tmp_path):
    pa, seq = absent
    meta = {k: v for k, v in copy.deepcopy(pa.meta).items() if k != "substrate"}
    bare = ReplayPack(dict(pa.arrays), meta)
    with pytest.raises(ValueError, match="embeds no substrate spec"):
        bare.replay(seq)
    bare.save(tmp_path / "bare.rpk")
    with pytest.raises(ValueError, match="embeds no substrate spec"):
        ReplayPack.load(tmp_path / "bare.rpk")
    m = d.simulate_trajectories(50, D0, d.Cylinder(radius=3e-6, orientation=(0, 0, 1)), 1e-3, 2e-4, seed=2, require_gpu=False)._bank_dict()
    m.pop("substrate", None)
    with pytest.raises(ValueError, match="embeds no substrate spec"):
        build_replay_pack(m, id="t", license="x", citation="x", K=4)


def test_a_closed_or_inert_substrate_has_no_field_by_its_kind():
    """``analytic`` and ``inert`` phantom substrates hold nothing magnetic: a chi on one is refused by ``field_term``,
    on the object and from the file's record."""
    from dmipy_sim.phantom.substrates import FIELD_BY_KIND, substrate_from_meta
    assert FIELD_BY_KIND == {"analytic": "absent", "inert": "absent"}
    for t in (Tissue(D=3e-9, chi_iso=1e-7), Tissue(D=3e-9, chi_aniso=-1e-8)):
        with pytest.raises(ValueError, match="field absent"):
            FreeWater(m0=1.0, tissue=t)
    FreeWater(m0=1.0, tissue=Tissue(D=3e-9, chi_iso=0.0))                              # a zero chi is no chi
    with pytest.raises(ValueError, match="field absent"):
        substrate_from_meta(dict(Inert().to_meta(), tissue={"chi_iso": 1e-7}))
    with pytest.raises(ValueError, match="field absent"):
        substrate_from_meta(dict(FreeWater(m0=1.0, tissue=Tissue(D=3e-9)).to_meta(), tissue={"chi_aniso": -1e-8}))
    assert isinstance(substrate_from_meta(Inert().to_meta()), Inert)


# ---- the forward engine: the walk computes and stores C3 exactly when a pool is magnetic, on every surface kind
def _shell_union_spec(magnetic):
    """Two unions of spheres, the outer at 1.5 x the inner's radii, the shell between them a dry pool: magnetic or
    not."""
    from dmipy_sim.spec.substrate import (Directional, Domain, Pool, Seeding, Sided, Surface, Susceptibility,
                                          Validity, Wall)
    c = [[-1.5e-6, 0.0, 0.0], [1.5e-6, 0.0, 0.0]]
    walls = [Wall("inner", Surface("sphere_union", instances={"centers": c, "radii": [0.8e-6, 0.8e-6]}), 1, 2,
                  Directional(), Sided()),
             Wall("outer", Surface("sphere_union", instances={"centers": c, "radii": [1.2e-6, 1.2e-6]}), 2, 0,
                  Directional(), Sided())]
    pools = [Pool(0, "extra", D0), Pool(1, "intra", D0),
             Pool(2, "shell", 0.0, water_fraction=0.0, susceptibility=(Susceptibility(None, None, "none") if magnetic else None))]
    return SubstrateSpec("test/shell-union", Domain([-4e-6] * 3, [4e-6] * 3, ["reflect"] * 3), pools, walls,
                         Seeding([0, 1]), Validity(0.4e-6, ["gradient", "relaxation", "surface"] + (["field"] if magnetic else []),
                                                   thinnest_shell=0.4e-6)).validate()


def _walk_and_pack(spec, **kw):
    from dmipy_sim.spec import walk_spec
    w = walk_spec(spec, 60, 6e-4, 2e-4, seed=0, n_probe=20_000, require_gpu=False, **kw)
    return w, build_replay_pack(w, id="t/walk", license="x", citation="x", K=4)


@pytest.mark.parametrize("kind", ["cylinder", "mesh", "sphere_union"])
def test_the_walk_stores_the_field_exactly_when_a_pool_is_magnetic(kind, tmp_path):
    from tests.conftest import spec_without_source
    from tests.test_mesh_spec import _bundle_spec
    if kind == "cylinder":
        magnetic = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 10e-6, N_max=2, D_intra=D0, D_extra=D0).spec
        plain = d.Cylinder(radius=3e-6, orientation=(0, 0, 1)).spec.replace(D={"intra": D0})
        res = dict(field_res=0.25e-6)
    elif kind == "mesh":
        magnetic = _bundle_spec(tmp_path); plain = spec_without_source(magnetic); res = dict(field_res=0.4e-6)
    else:
        magnetic, plain, res = _shell_union_spec(True), _shell_union_spec(False), dict(field_res=0.2e-6)
    wm, pm = _walk_and_pack(magnetic, **res)
    assert wm.field_samples is not None and pm.has_field and pm.susceptibility_field == "present"
    wp, pp = _walk_and_pack(plain)
    assert wp.field_basis is None and wp.field_samples is None and not pp.has_field and pp.susceptibility_field == "absent"


@pytest.mark.parametrize("kind", ["sphere", "label_volume", "cylinder"])
def test_a_magnetic_pool_no_field_route_computes_is_refused_before_the_walk(kind, surface_specs):
    """A sphere, a label volume or a lone cylinder wall around a magnetic pool: no route computes that field, so the
    walk is refused naming the surfaces, never walked without its field."""
    from dmipy_sim.spec import walk_spec
    spec, pool = surface_specs[kind]
    spec = SubstrateSpec.from_dict(_variant(spec, pool, True, True)).validate()
    with pytest.raises(SpecError, match=f"is magnetic.*'{kind}'.*no field route"):
        walk_spec(spec, 20, 4e-4, 2e-4, seed=0, require_gpu=False)
    plain = SubstrateSpec.from_dict(_variant(spec.to_dict(), pool, False, False))
    assert walk_spec(plain, 20, 4e-4, 2e-4, seed=0, diffusivity=D0, require_gpu=False).field_samples is None

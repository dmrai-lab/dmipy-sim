"""A study (dmipy-sim#297): the walker primitives of an acquisition, formed once, give every tissue and scanner's
signals exactly as the replay routes do; a study on a pack is the per-pair replay; and the columnar image of a study
is one pass over the rows with every pair's volume and its own floor."""
from __future__ import annotations
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as _seqmod
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.study import Acquisition, Protocol, Study, walker_primitives, study_signals
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along

D0 = 2e-9


def test_the_primitives_give_every_pairs_signals_as_the_replay_does(pack):
    """Relaxation, contact, both, none: the primitives formed once reproduce ``walker_signals`` for each tissue to
    rounding, the bands never contracted again."""
    seq = _seqmod.pgse([[1, 0, 0], [0, 1, 1]], 1e-3, 3e-3, bvalues=[1e9, 5e8], TE=6e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    prim = pack.walker_primitives(seq)
    assert prim.phi.shape == (pack.n_walkers, 2) and prim.exposure_t2.shape[0] == pack.n_walkers and prim.contact.shape == (pack.n_walkers,)
    for t in (None, Tissue(T2=[0.08, 0.03]), Tissue(T1=[1.0, 1.2]), Tissue(rho=1e-5, D=D0), Tissue(T2=[0.08, 0.03], T1=[1.0, 1.2], rho=1e-5, D=D0)):
        w, ew, E = prim.signals(t, None); w2, ew2, E2 = pack.walker_signals(seq, tissue=t)
        np.testing.assert_array_equal(w, w2); np.testing.assert_allclose(ew, ew2, rtol=1e-12, atol=1e-300); np.testing.assert_allclose(E, E2, rtol=0, atol=1e-12)


def test_the_field_term_is_linear_in_the_scanner_and_the_susceptibilities(field_pack):
    """On the path route the field phase is ``B0 (chi_iso A + chi_aniso B)`` from two per-walker scalars formed once
    per acquisition (under its pose): 3 T and 7 T, and two susceptibilities, from the same primitives."""
    pk = field_pack
    seq = _seqmod.gre(6e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    seq_lab, R = field_along(seq, (0.6, 0.0, 0.8))
    prim = pk.walker_primitives(Acquisition(seq_lab, orientation=R))
    assert prim.field_iso.shape == (pk.n_walkers,) and prim.field_aniso.shape == (pk.n_walkers,)
    for B0 in (3.0, 7.0):
        for t in (Tissue(chi_iso=-1e-7, chi_aniso=-5e-8), Tissue(chi_iso=-2e-7)):
            w, ew, E = prim.signals(t, B0); w2, ew2, E2 = pk.walker_signals(seq_lab, orientation=R, scanner=B0, tissue=t)
            np.testing.assert_allclose(E, E2, rtol=0, atol=1e-10); np.testing.assert_allclose(ew, ew2, rtol=1e-12)
    with pytest.raises(ValueError, match="chi_iso"):
        prim.signals(None, 3.0)


def test_a_study_is_the_per_pair_replay(pack):
    """Two acquisitions, two tissues (one resolved on the scanner), the pairs picked: the study's rows are the
    per-pair replays side by side, and its record carries the resolved values."""
    seq1 = _seqmod.pgse([[1, 0, 0], [0, 1, 1]], 1e-3, 3e-3, bvalues=[1e9, 5e8], TE=6e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    seq2 = _seqmod.pgse([[0, 0, 1]], 1e-3, 2e-3, bvalues=[2e9], TE=5e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    catalogue = lambda scanner: Tissue(T2=[0.08, 0.03] if scanner is None else [0.05, 0.02], rho=1e-5, D=D0)   # a tissue resolved on the scanner
    study = Study(Protocol([seq1, Acquisition(seq2, name="axial")]), tissues=[None, catalogue], scanners=[None, 0.0], pairs=[(0, 0), (1, 0), (1, 1)], name="t")
    assert len(study) == 3 and study.protocol.n_meas == 3 and study.needs_contact and study.needs_relaxation and not study.needs_field
    S = pack.study(study)
    assert S.shape == (3, 3)
    for k in range(3):
        t, s = study.resolved(k)
        expect = np.concatenate([pack.replay(seq1, tissue=t, scanner=s), pack.replay(seq2, tissue=t, scanner=s)])
        np.testing.assert_allclose(S[k], expect, rtol=1e-9)
    meta = study.to_meta()
    assert meta["pairs"][1]["tissue"]["T2"] == [0.08, 0.03] and meta["pairs"][2]["tissue"]["T2"] == [0.05, 0.02] and meta["protocol"]["acquisitions"][1]["name"] == "axial"
    with pytest.raises(IndexError):
        Study(Protocol([seq1]), tissues=[None], scanners=[None], pairs=[(1, 0)])
    with pytest.raises(ValueError, match="chi_iso"):                      # a field on a tissue without a susceptibility is refused, as replay refuses it
        pack.study(Study(Protocol([seq1]), tissues=[catalogue], scanners=[3.0]))


def test_the_columnar_image_of_a_study_is_one_pass_with_a_floor_per_volume(tmp_path):
    """Two shards consolidated; a study of two acquisitions on two tissue/scanner pairs imaged in one pass over the
    rows: every pair's volume equals the per-voxel replay of the merged pack, and each carries its own split-half
    floor, finite where the voxel has rows."""
    import os
    from dmipy_sim.io.strands import write_tck
    from dmipy_sim.phantom import Grid
    from dmipy_sim.replay.bank import merge_packs
    from dmipy_sim.replay.replay import ReplayPack
    from dmipy_sim.fill.consolidate import consolidate
    from dmipy_sim.spec import disco_spec, walk_spec, StratifiedByVoxel
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp_path / "t.tck"), str(tmp_path / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6); grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    os.makedirs(str(tmp_path / "shards")); packs = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=7 + b, require_gpu=False, field=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        packs.append(build_replay_pack(w, id=f"t/{b}", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp_path / "shards" / f"block-000{b}.p1.rpk")))
    consolidate(str(tmp_path / "shards"), str(tmp_path / "layout"), blocks=[0, 1], id="t/columns")
    merged = merge_packs(packs, id="t/merged"); col = ReplayPack.open(str(tmp_path / "layout"))
    seq1 = d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=merged.n_t, slew_rate=np.inf), [1e9, 1e9])
    seq2 = d.set_b(d.pgse([[0, 1, 0]], 0.2e-3, 0.4e-3, gradient_strengths=0.1, n_t=merged.n_t, slew_rate=np.inf), [5e8])
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho=1e-5)
    study = Study(Protocol([seq1, seq2]), tissues=[None, t], scanners=[None])
    S, floor, plan = col.image(study, tol=1e-9, chunk_rows=5)
    assert S.shape == (2,) + tuple(grid.shape) + (3,) and floor.shape == (2,) + tuple(grid.shape) and plan["settings"] == 2
    ijk, _ = grid.bin(merged.r0); v = np.ravel_multi_index(ijk.T, grid.shape)
    for k in range(2):
        tk, sk = study.resolved(k)
        for s, sl in zip((seq1, seq2), study.protocol.slices):
            w, ew, E = merged.walker_signals(s, tissue=tk, scanner=sk)
            keys, inv = np.unique(v, return_inverse=True); num = np.zeros((len(keys), E.shape[1]), complex); den = np.zeros(len(keys))
            np.add.at(num, inv, ew[:, None] * E); np.add.at(den, inv, w)
            np.testing.assert_allclose(S[k].reshape(-1, 3)[keys][:, sl], np.abs(num / den[:, None]), rtol=1e-9, atol=1e-12)
        f = floor[k].reshape(-1)[np.unique(v)]
        assert np.isfinite(f).all() and (f >= 0).all() and (f < 1).all()

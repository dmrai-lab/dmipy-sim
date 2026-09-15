"""A walk as a file (`PersistentWalk.save` / `load`), the spool of a walk's batches in its run record, and the
resume that reads them back: a killed walk costs the batch in progress, not the walk."""
import os
import numpy as np
import pytest

from dmipy_sim import Cylinder, simulate_trajectories
from dmipy_sim.io.strands import write_tck
from dmipy_sim.persistent_walk import PersistentWalk
from dmipy_sim.spec import disco_spec, walk_spec
from dmipy_sim.replay.bank import build_replay_pack


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("spool")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def _walk(spec, run_dir=None, spool=False):
    return walk_spec(spec, 60, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, field=True, adaptive_steps=True,
                     field_cutoff_max_m=25e-6, field_sample_every=4, walker_batch_size=20, run_dir=run_dir, spool=spool)


def test_a_walk_round_trips_through_its_file_and_packs(spec, tmp_path):
    w = _walk(spec)
    w.save(str(tmp_path / "w.walk"))
    back = PersistentWalk.load(str(tmp_path / "w.walk"))
    np.testing.assert_array_equal(back.positions, w.positions)
    np.testing.assert_array_equal(back.boundary_local_time, w.boundary_local_time)
    np.testing.assert_array_equal(back.field_samples, w.field_samples)
    assert back.dt == w.dt and back.field_sample_every == 4 and back.spec.to_dict() == w.spec.to_dict() and back.geometry is None
    assert back.field_basis.meta == w.field_basis.meta and back.run["producer"] == "walk_spec" and back.stepping == w.stepping
    with pytest.raises(ValueError, match="evaluates nothing"):
        back.field_basis.channels(np.zeros((2, 3)))
    pk = build_replay_pack(back, id="t/loaded", license="x", citation="x", K=8, susc_path_K=4, device="numpy")
    pk0 = build_replay_pack(w, id="t/loaded", license="x", citation="x", K=8, susc_path_K=4, device="numpy")
    for k in pk.arrays:
        np.testing.assert_array_equal(pk.arrays[k], pk0.arrays[k])
    assert pk.meta["compression"]["channels"]["susceptibility_grid"]["source"] == pk0.meta["compression"]["channels"]["susceptibility_grid"]["source"]
    plain = simulate_trajectories(50, 2e-9, Cylinder(radius=3e-6, orientation=(0, 0, 1)), T_max=2e-3, dt_save=1e-4, seed=0, require_gpu=False)
    plain.save(str(tmp_path / "p.walk")); back = PersistentWalk.load(str(tmp_path / "p.walk"))
    np.testing.assert_array_equal(back.positions, plain.positions); assert back.field_basis is None


def test_a_spooled_walk_resumes_from_its_batches(spec, tmp_path):
    """Three batches per pool spooled; the run resumed with one batch missing walks only that one and is the same
    walk to the bit; resumed complete, it walks nothing; a different walk in the same record is refused."""
    d = str(tmp_path / "rec")
    w0 = _walk(spec, run_dir=d, spool=True)
    files = sorted(os.listdir(os.path.join(d, "spool")))
    assert len(files) == 6 and files[0].startswith("extra-batch-0000") or len(files) >= 2, files
    victim = os.path.join(d, "spool", files[-1]); os.remove(victim)
    w1 = _walk(spec, run_dir=d, spool=True)                            # the missing batch walked again, the rest read
    np.testing.assert_array_equal(w1.positions, w0.positions)
    np.testing.assert_array_equal(w1.boundary_local_time, w0.boundary_local_time)
    np.testing.assert_array_equal(w1.field_samples, w0.field_samples)
    assert os.path.isfile(victim)
    from dmipy_sim import run as R
    rows = R.read_events(d)
    assert sum(1 for r in rows if r["kind"] == "start") == 2 and rows[-1]["kind"] == "end"
    assert [r["resumed"] for r in rows if r["kind"] == "start"] == [False, True]
    w2 = _walk(spec, run_dir=d, spool=True)                            # everything read back
    np.testing.assert_array_equal(w2.positions, w0.positions)
    with pytest.raises(ValueError, match="another run"):
        walk_spec(spec, 61, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, field=True, adaptive_steps=True,
                  field_cutoff_max_m=25e-6, field_sample_every=4, walker_batch_size=20, run_dir=d, spool=True)

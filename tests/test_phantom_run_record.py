"""A phantom replay is a run producer: phases, progress in encoding classes, and a report of what each response cost
(dmrai-lab/dmipy-sim#449). A pose expansion inside it joins the same record with its own phases."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, Phantom, PackSubstrate, Watson
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(500, 2e-9, g, 10e-3, 5e-4, seed=2, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


def _phantom(pack):
    grid = Grid(shape=(2, 2, 1), voxel_size_m=(1e-3,) * 3)
    mu = np.zeros((2, 2, 1, 3)); mu[..., 2] = 1.0
    return Phantom.compose(grid, fractions={PackSubstrate(pack, m0=1.0): np.ones((2, 2, 1))}, orientation=Watson(mu=mu, kappa=8.0))


def test_a_replay_reports_its_phases_and_each_response(pack):
    ph = _phantom(pack)
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3)
    rep = {}
    ph.replay(seq, report=rep)
    assert rep["n_encoding_classes"] == 1
    assert {"classes", "gather", "layers"} <= set(rep["seconds"])           # the phantom's own phases
    assert {"moments", "bessel", "harmonics"} <= set(rep["seconds"])        # the pose expansion joined the record
    (row,) = rep["responses"]
    assert row["route"] == "closed" and row["lmax"] >= 0 and row["n_bodies"] >= 1 and row["seconds"] >= 0.0


def test_a_pose_expansion_on_its_own_is_a_run(pack):
    from dmipy_sim.run import current
    seq = sequences.pgse([[1, 0, 0]], 2e-3, 5e-3, gradient_strengths=[0.3], TE=10e-3)
    assert current() is None
    resp = pack.pose_response(seq)
    assert resp.route == "closed" and current() is None                    # opened and closed its own record

"""A pack built from a walk stored at its saves (the DST-I basis) is byte for byte what main built before the
continuous basis existed: the basis dispatch adds a route beside the save-grid pack's transform, container derivation
and quantisation, and changes none of them. The fixture ``tests/fixtures/raw_pack_dst1_main.npz`` is every tensor of
the pack of the synthetic walk below, built by dmipy-sim main at aab5763 (the numpy route)."""
from pathlib import Path
import warnings

import numpy as np

import dmipy_sim as d
from dmipy_sim.persistent_walk import PersistentWalk
from dmipy_sim.replay.bank import build_replay_pack

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "raw_pack_dst1_main.npz"


def _walk():
    rng = np.random.default_rng(741)
    n, n_t, dt = 400, 81, 1.25e-4
    pos = (np.cumsum(rng.normal(size=(n, n_t, 3)) * 2e-7, axis=1) + rng.uniform(-1e-6, 1e-6, (n, 1, 3))).astype(np.float32)
    pos[:, :, 2] *= 3
    dlog = -(rng.exponential(1e-3, (n, n_t)) * (rng.uniform(size=(n, n_t)) < 0.3)).astype(np.float32)
    dlog[:, 0] = 0
    comp = np.ones((n, n_t), np.int8)
    return PersistentWalk(pos, dt, 4, dt / 4, boundary_local_time=dlog, compartment=comp, seed=0, diffusivity=2e-9,
                          geometry=d.Cylinder(radius=3e-6, orientation=(0, 0, 1)))


def test_the_save_grid_pack_is_mains_byte_for_byte():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk = build_replay_pack(_walk(), id="fixture/raw", license="x", citation="x", K=24, device="numpy")
    ref = np.load(FIXTURE)
    assert sorted(pk.arrays) == sorted(ref.files)
    differ = [k for k in ref.files if not np.array_equal(np.asarray(pk.arrays[k]), ref[k])]
    assert not differ, f"the save-grid pack's tensors differ from main's: {differ}"
    assert "basis" not in pk.meta["compression"]                       # a save-grid pack declares nothing new

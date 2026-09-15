"""A fill's recipe on a fake hub: the 20 um three-strand DiSCo-like spec of `tests/test_pack_merge.py`, a 2 x 2 x 2
grid of 10 um voxels, two blocks (the lower and upper k layer), six planned walkers per pool per voxel, a plan in
two passes of one half each."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.fill import FakeHub
from dmipy_sim.io.strands import write_tck


def make_recipe(root, *, passes=True, field=False):
    os.makedirs(os.path.join(root, "substrate"), exist_ok=True); os.makedirs(os.path.join(root, "plan"), exist_ok=True)
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    write_tck(os.path.join(root, "substrate/t.tck"), cls_, coordinate_unit_m=25e-6)
    np.savetxt(os.path.join(root, "substrate/d.txt"), np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    shape = (2, 2, 2)
    np.savez(os.path.join(root, "plan/counts.npz"), count_extra=np.full(shape, 6, np.int64), count_intra=np.full(shape, 6, np.int64))
    blocks = [dict(block=b, i=[0, 2], j=[0, 2], k=[b, b + 1], seed=7 + b, walkers=dict(extra=24, intra=24), voxels=4) for b in (0, 1)]
    json.dump(dict(grid=dict(shape=list(shape)), target_walkers_per_block=48, blocks=blocks), open(os.path.join(root, "plan/blocks.json"), "w"))
    man = dict(id="test/fill", license="x", citation="x", code=dict(repo="dmrai-lab/dmipy-sim", commit="test"),
               substrate=dict(kind="disco", tracks="substrate/t.tck", diameters="substrate/d.txt", coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=20e-6),
               variants=dict(t=dict(field=bool(field))), default_variant="t",
               grid=dict(shape=list(shape), voxel_size_m=[1e-5] * 3, origin_m=[5e-6] * 3, attach="substrate"),
               walk=dict(T_max_s=8e-4, dt_save_s=2e-4, scanner="connectom", floor_fraction=0.1, adaptive_steps=True, walker_batch_size=4096, census_draws=50),
               pack=dict(K=3, K_path=4, position_container="bands", blt_container="bands", blt_K=2),
               plan=dict(file="plan/counts.npz", blocks="plan/blocks.json", walkers=96, n_blocks=2,
                         **(dict(passes=[dict(**{"pass": 1}, scale=0.5), dict(**{"pass": 2}, scale=0.5)]) if passes else {})))
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"), indent=1)
    return root


@pytest.fixture
def fake(tmp_path):
    """``(hub, workdir)``: a fake hub holding the recipe, and an empty work directory."""
    hub = FakeHub(make_recipe(str(tmp_path / "hub")))
    return hub, str(tmp_path / "work")


@pytest.fixture
def certified(fake):
    """The fake hub with the fill's certificate: block 0 walked once with ``certify`` at a budget."""
    from dmipy_sim.fill import Fill, Options, Recipe
    hub, work = fake
    rc = Recipe(hub)
    Fill(hub, rc, Options(workdir=work, host="cert", block=0, certify=True, budget=40, require_gpu=False)).run()
    assert hub.exists("certificate/t.json") and hub.exists("certificate/t-block-0000.rpk")
    return hub, work

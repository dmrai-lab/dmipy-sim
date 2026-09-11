"""The README's phantom snippets run (#162, #186): every ```python block under "Replay phantoms" is executed
against a small pack standing in for the CACTUS / DiSCo ones, with the names the prose defines around it. A
snippet that stops running fails here, by block."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _blocks():
    text = (ROOT / "README.md").read_text()
    section = text[text.index("## Replay phantoms"):text.index("## Substrates")]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


@pytest.fixture(scope="module")
def stand_in_pack(tmp_path_factory):
    """A 30 ms walk of one cylinder: long enough for the README's PGSE (delta 6, Delta 15, TE 30 ms)."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(300, 2e-9, g, 30e-3, 5e-4, seed=0, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "stand_in.rpk"
    build_replay_pack(walk, id="test/stand-in", license="x", citation="x", K=8, out_path=str(out))
    return str(out)


def test_the_readme_phantom_snippets_run(stand_in_pack, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("ring", ROOT / "examples" / "rph" / "circular_wm_phantom.py")
    ring = importlib.util.module_from_spec(spec); spec.loader.exec_module(ring)
    from dmipy_sim.replay import so3
    ns = {"annulus": ring.annulus, "frames": ring.frames,                       # the volumes the prose points at
          "dirs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          "rotation": so3.rotation_of((0.3, 0.5, 0.81)),
          "kappa_map": 1.0, "f_myelin": np.full((40, 40, 40), 0.2), "np": np}
    blocks = _blocks()
    assert len(blocks) == 2, "the section has a composition block and a partition block"
    for src in blocks:
        src = src.replace('"cactus.rpk"', repr(stand_in_pack)).replace('"disco.rpk"', repr(stand_in_pack))
        src = src.replace(", B0_T=7.0, b0_dir=(0, 1, 0), chi_iso=-1e-7", "")   # the stand-in carries no field tier
        exec(compile(src, "README.md", "exec"), ns)
    S, S_part = ns["S"], ns["S"]
    ph, scanner = ns["ph"], ns["scanner"]
    assert ns["ph50"].grid.voxel_size_m[0] == pytest.approx(50e-6) and ns["ph_r"].pose is not None
    assert S.shape[:3] == (20, 20, 20) and np.isfinite(S).all()                 # the last S: the scanner-prescribed replay
    assert (tmp_path / "wm.rph").exists()
    assert ph.mode == "rigid" and ns["ph50"].n_voxels >= 1

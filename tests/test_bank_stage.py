"""Tests for the dmipy_sim substrate-BANK layer (Phase 4): local staging mirror
(catalog / croissant / substrate cards / dataset README), a no-network round-trip via
``pull(local_path=)``, the C4 MT convenience oracle, and the HuggingFace publish paths
exercised WITHOUT any upload (monkeypatched ``huggingface_hub``).

CPU-only. Auto-marked ``slow`` in conftest because the fixture rebuilds a JAX master walk.
"""
import hashlib
import json
import sys
import types

import numpy as np
import pytest

from dmipy_sim import simulate_trajectories
from dmipy_sim.geometries import FreeDiffusion
from dmipy_sim import bank
from dmipy_sim.bank_card import substrate_card
from dmipy_sim import compression as cx
from dmipy_sim.constants import GAMMA

AID = "demo/free-c1"


@pytest.fixture(scope="module")
def env():
    e = cx.default_envelope(); e["ogse_periods"] = [1, 2]; e["B0_list"] = []
    return e


@pytest.fixture(scope="module")
def rpk_path(tmp_path_factory, env):
    """Build one small C1 (bulk-relaxation) pack + an MT pool descriptor -> .rpk file."""
    res = simulate_trajectories(
        n_walkers=1000, diffusivity=2e-9, geometry=FreeDiffusion(),
        T_max=16e-3, dt_save=16e-3 / 32, seed=0, require_gpu=False,
        save_relaxation_data=True)
    mt_params = dict(f_bound=0.1, k_forward=4.0, T1_free=1.0, T2_free=0.05,
                     T1_bound=1.0, T2_bound=1.0e-5, S_over_V=5.0e5)
    m = bank.master_from_walk(res, D=2e-9, T2_per_comp=[0.05], T1_per_comp=[1.0],
                              w=np.ones(np.asarray(res[0]).shape[0]), mt_params=mt_params)
    out = tmp_path_factory.mktemp("build") / "pack.rpk"
    bank.build_replay_pack(m, id=AID, method="lowrank", K=24, tol=3.0,
                           license="CC-BY-4.0", citation="dmipy-sim test",
                           provenance={"substrate": "free water", "real_or_synthetic": "synthetic"},
                           envelope=env, out_path=str(out))
    return out


def _pgse(Nt, dt, b, d):
    t = np.arange(Nt) * dt; T = Nt * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    q = GAMMA * dt * np.cumsum(prof); amp = np.sqrt(b / (dt * np.sum(q ** 2))) if b > 0 else 0.0
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    return (amp * prof[:, None] * d[None, :]).astype(np.float32)


# --------------------------------------------------------------- staged layout
def test_stage_pack_layout(rpk_path, tmp_path):
    staging = tmp_path / "bank"
    dst = bank.stage_pack(str(rpk_path), str(staging), AID)

    # .rpk copied under the id path (id contains a "/": nested)
    assert dst == staging / f"{AID}.rpk"
    assert dst.exists() and dst.parent == staging / "demo"

    # croissant sidecar + per-substrate card + dataset README all beside it
    croissant = staging / f"{AID}.croissant.jsonld"
    card = staging / f"{AID}.md"
    assert croissant.exists() and card.exists()
    assert (staging / "README.md").exists()
    cr = json.loads(croissant.read_text())
    assert cr["@type"] == "Dataset" and cr["name"] == AID and cr["license"] == "CC-BY-4.0"

    # substrate card is generated from meta and names its id + tiers
    md = card.read_text()
    assert f"`{AID}`" in md and "C1" in md and "Bulk relaxation" in md
    # public: field tier absent -> shows "—", never crashes / claims a stored field channel
    assert "| C3 |" in md and "✅" not in md.split("| C3 |")[1].split("\n")[0]
    # C4 MT pool descriptor present -> MT section rendered
    assert "Magnetization transfer (two-pool qMT)" in md

    # manifest.json: schema string + one entry with the id and correct sha256
    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["schema"] == "dmipy-sim substrate-bank/1.2"
    assert manifest["n_entries"] == 1
    entry = manifest["entries"][0]
    assert entry["id"] == AID and entry["file"] == f"{AID}.rpk"
    good = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert entry["sha256"] == good

    # SHA256SUMS carries the same (correct) hash for the staged file
    sums = (staging / "SHA256SUMS").read_text().strip().splitlines()
    assert f"{good}  {AID}.rpk" in sums


def test_refresh_catalog_idempotent(rpk_path, tmp_path):
    staging = tmp_path / "bank"
    bank.stage_pack(str(rpk_path), str(staging), AID)
    m1 = (staging / "manifest.json").read_text()
    s1 = (staging / "SHA256SUMS").read_text()
    bank._refresh_catalog(staging)               # re-run over the same tree
    bank._refresh_catalog(staging)
    assert (staging / "manifest.json").read_text() == m1
    assert (staging / "SHA256SUMS").read_text() == s1


# --------------------------------------------------------- no-network round-trip
def test_pull_local_path_roundtrips_and_replays(rpk_path, tmp_path, env):
    staging = tmp_path / "bank"
    dst = bank.stage_pack(str(rpk_path), str(staging), AID)

    # local_path bypasses the network entirely (no huggingface_hub import path)
    pack = bank.pull(AID, local_path=str(dst))
    assert isinstance(pack, bank.ReplayPack)
    assert pack.meta["id"] == AID and pack.method == "lowrank"

    # a hash mismatch is caught (integrity gate) without touching the network
    with pytest.raises(ValueError):
        bank.pull(AID, local_path=str(dst), expected_sha256="0" * 64)
    # correct hash passes
    good = hashlib.sha256(dst.read_bytes()).hexdigest()
    bank.pull(AID, local_path=str(dst), expected_sha256=good)

    # the pulled pack replays (C0 gradient + C1 relaxation), no network
    Nt = pack.n_t; dt = pack.dt

    class WF:
        G = np.stack([_pgse(Nt, dt, b, [1, 0, 0]) for b in (0.0, 1e9)])
    S = np.asarray(pack.replay(WF, T2=pack.meta["per_comp"]["T2"], T1=pack.meta["per_comp"]["T1"]))
    assert S.shape == (2,)
    assert 0.5 < abs(S[0]) < 0.85          # b=0 is T2-weighted (TE=16ms, T2=50ms)
    assert abs(S[1]) < abs(S[0])           # b>0 attenuates further


# ------------------------------------------------------------------- C4 MT
def test_mt_zspectrum_and_mtr(rpk_path, tmp_path):
    pack = bank.read_rpk(str(rpk_path))
    assert pack.replay_envelope["magnetization_transfer"] is True
    assert pack.meta.get("mt") is not None
    offsets = np.array([-2e4, -1e4, 0.0, 1e4, 2e4])
    Z = np.asarray(pack.mt_zspectrum(offsets, w1_hz=200.0, t_sat=0.3))
    assert Z.shape == offsets.shape
    assert np.all(Z <= 1.0 + 1e-6) and np.all(Z >= 0.0)
    # on-resonance is the deepest saturation (smallest Mz)
    assert Z[2] == pytest.approx(Z.min())
    mtr = pack.mtr(offset_hz=1e4, w1_hz=200.0, t_sat=0.3)
    assert 0.0 <= mtr <= 1.0

    # a pack with no MT tier raises cleanly
    no_mt = bank.ReplayPack(pack.arrays, {k: v for k, v in pack.meta.items() if k != "mt"})
    with pytest.raises(ValueError):
        no_mt.mt_zspectrum([0.0])


def test_substrate_card_field_absent_degrades():
    """C3 (field) is provider-driven in public -> the card must render it as absent, not crash."""
    meta = dict(id="x/y", license="CC-BY-4.0", citation="c",
                compression=dict(method="lowrank", K=8),
                walk_params=dict(n_walkers=10, n_t=8, T_max=0.01),
                replay_envelope=dict(gradient=True, bulk_relaxation=True, field=False),
                fidelity=dict(err_max=1e-3), provenance={})
    md = substrate_card(meta)
    assert "| C3 |" in md and "not a stored channel" in md
    assert "# Substrate card — `x/y`" in md


# -------------------------------------------- publish paths WITHOUT any upload
def test_publish_paths_no_upload(rpk_path, tmp_path, monkeypatch):
    """Exercise publish_dir/publish with a fake huggingface_hub so NOTHING is uploaded and
    no token is required — assert they call create_repo + upload_* with the right targets."""
    staging = tmp_path / "bank"
    bank.stage_pack(str(rpk_path), str(staging), AID)

    calls = []
    fake = types.ModuleType("huggingface_hub")

    def create_repo(repo_id, repo_type=None, private=None, exist_ok=None):
        calls.append(("create_repo", repo_id, repo_type, private))

    class HfApi:
        def upload_folder(self, folder_path=None, repo_id=None, repo_type=None):
            calls.append(("upload_folder", folder_path, repo_id, repo_type))

        def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None):
            calls.append(("upload_file", path_in_repo, repo_id, repo_type))

    fake.create_repo = create_repo
    fake.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    url = bank.publish_dir(str(staging))          # default repo_id = DEFAULT_REPO
    assert url == "https://huggingface.co/datasets/dmrai-lab/substrate-bank"
    assert ("create_repo", "dmrai-lab/substrate-bank", "dataset", True) in calls
    assert any(c[0] == "upload_folder" and c[2] == "dmrai-lab/substrate-bank" for c in calls)

    ref = bank.publish(str(staging / f"{AID}.rpk"), AID)
    assert ref == f"dmrai-lab/substrate-bank:{AID}"
    assert any(c[0] == "upload_file" and c[1] == f"{AID}.rpk" for c in calls)


def test_default_repo_is_public_org():
    assert bank.DEFAULT_REPO == "dmrai-lab/substrate-bank"

"""The shape vector's stored width (dmipy-sim#731): a vector enters a measurement as the phase ``q . m``, so its step is
``sqrt(12) (eps / sqrt 2) / q_max`` with ``eps`` the band container's at the pack's floor target and ``q_max`` the
largest amplitude the pack's envelope reaches on the shape; three components to one 32-bit word (10-bit fields) or one
64-bit word (16-bit fields), decoded in the image kernel."""
from __future__ import annotations
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.columnar import ColumnarPack
from dmipy_sim.replay.compression import BUDGET, container_eps
from dmipy_sim.replay.time_segments import (TimeSegmentLayout, _components, _signed_words, vector_code, vector_fields,
                                             vector_width, pack_vectors, unpack_vectors, write_time_segment_layout)
from dmipy_sim.fill.consolidate import consolidate
from dmipy_sim.spec import walk_spec, StratifiedByVoxel
from tests.conftest import spec_without_source

D_WATER = 3e-9                                                          # m^2/s


def test_the_width_is_the_issues_worked_example():
    """Free water at the floor target 1e-2, b 3000 s/mm^2 at an effective diffusion time of 30 ms: ``eps = 0.032``,
    ``q_max = 3.2e5 rad/m``; at the full ``eps`` the step is 0.35 um and the largest of 1e9 displacements (about six
    rms, 80 um) takes 9 bits; at the vector's share ``eps / sqrt 2`` the step is 0.24 um and 10 bits, three
    components to one 32-bit word. At twice that q_max a component takes 11 bits: 11 / 11 / 10 is still one 32-bit
    word, 11 / 11 / 11 one 64-bit word."""
    eps = container_eps(1e-2)
    assert eps == pytest.approx(np.sqrt(0.1 * 0.01))
    q_max = np.sqrt(3000e6 / 0.030)
    assert q_max == pytest.approx(3.16e5, rel=1e-3)
    step = vector_width(1e-2, q_max)                                    # metres: the vector in displacement units
    assert step * np.sqrt(2) == pytest.approx(0.346e-6, rel=1e-2) and step == pytest.approx(0.245e-6, rel=1e-2)
    a = 6 * np.sqrt(2 * D_WATER * 0.030) * np.ones(3)
    bits_full, _ = vector_code(a, step * np.sqrt(2))
    bits, scale = vector_code(a, step)
    assert list(bits_full) == [9, 9, 9] and list(bits) == [10, 10, 10] and np.all(scale <= step)
    assert vector_fields(bits) == (((0, 10), (10, 10), (20, 10)), 32)
    assert pack_vectors(np.zeros((1, 3)), scale, *vector_fields(bits)).itemsize == 4
    step2 = vector_width(1e-2, 2 * q_max)
    bits2, _ = vector_code(a * [1.0, 1.0, 0.45], step2)
    assert list(bits2) == [11, 11, 10] and vector_fields(bits2) == (((0, 11), (11, 11), (22, 10)), 32)
    bits3, _ = vector_code(a, step2)
    assert list(bits3) == [11, 11, 11] and vector_fields(bits3) == (((0, 11), (11, 11), (32, 11)), 64)
    assert vector_code(a, 1e-30)[0].max() == 16 and vector_code(a, 1.0)[0].min() == 3
    assert vector_fields([16, 16, 16]) == (((0, 16), (16, 16), (32, 16)), 64)


@pytest.mark.parametrize("bits", [[3, 7, 10], [10, 10, 10], [11, 11, 10], [11, 11, 11], [16, 3, 16], [16, 16, 16]])
def test_the_packing_round_trips(bits):
    """Every code of every field layout comes back exactly, a value within half a step, the extremes and zero
    included, through the host decoder and the device one alike; the word is 32 bits exactly when the widths sum to
    at most 32."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    step = 1e-8
    a = step * 0.9 * 2.0 ** (np.array(bits) - 1)                       # the rule gives exactly these bits
    got, scale = vector_code(a, step)
    assert list(got) == bits
    fields, word = vector_fields(got)
    assert word == (32 if sum(bits) <= 32 else 64)
    m = rng.uniform(-1, 1, (64, 128, 3)) * a
    m[0, 0] = a; m[0, 1] = -a; m[0, 2] = 0.0
    words = pack_vectors(m, scale, fields, word)
    assert words.dtype == (np.uint32 if word == 32 else np.uint64) and words.shape == m.shape[:2]
    back = unpack_vectors(words, scale, fields, word)
    assert back.dtype == np.float32
    assert np.all(np.abs(back - m) <= 0.5 * scale + 2.0 ** -23 * np.abs(m))     # half a step, plus float32 rounding
    np.testing.assert_allclose(back[0, :3], m[0, :3], rtol=1e-6, atol=0)
    np.testing.assert_array_equal(np.rint(back / scale.astype(np.float32)), np.rint(m / scale))
    dev = _components(jnp.asarray(_signed_words(words, word)), jnp.asarray(scale, jnp.float32), fields)
    np.testing.assert_array_equal(np.stack([np.asarray(c) for c in dev], -1), back)


@pytest.fixture(scope="module")
def layout(spec_grid):
    """Two blocks of the three-strand spec at 400 walkers per voxel and pool, consolidated into columns, and its n_t."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    os.makedirs(str(tmp / "mw_shards"), exist_ok=True)
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 400
        w = walk_spec(spec, T_max=8e-4, seed=31 + b, require_gpu=False, K=8,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        pk = build_replay_pack(w, id=f"t/mw{b}", license="x", citation="x", voxel_grid=grid,
                               out_path=str(tmp / "mw_shards" / f"block-000{b}.p1.rpk"))
        n_t = pk.n_t
    consolidate(str(tmp / "mw_shards"), str(tmp / "mw_columns"), blocks=[0, 1], id="t/mw-columns")
    col = ColumnarPack(str(tmp / "mw_columns"))
    out = str(tmp / "mw_moments")
    shape = d.pgse([[0.0, 0.0, 1.0]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    manifest = write_time_segment_layout(col, {"pgse": shape}, out, tol=1e-9, chunk_rows=500)
    return col, TimeSegmentLayout(out), manifest, n_t


def test_the_header_records_what_the_rule_used(layout):
    """The manifest's vector record is the rule at the pack's declared floor target and envelope: ``q_max`` the
    shape's amplitude at the envelope's b_max, the step :func:`vector_width`, the bits and scales :func:`vector_code`
    of the decoded vectors' largest components and the fields' offsets, one 32-bit word per walker on disk."""
    col, sm, manifest, n_t = layout
    rec = manifest["shapes"]["pgse"]["vector"]
    assert rec["b_max"] == col.meta["replay_envelope"]["acquisition"]["b_max"]
    assert rec["floor_target"] == pytest.approx(col.floor)             # no target_floor declared: the per-voxel floor
    assert rec["q_max"] == pytest.approx(np.sqrt(rec["b_max"] / manifest["shapes"]["pgse"]["b_unit"]))
    assert rec["step"] == pytest.approx(vector_width(rec["floor_target"], rec["q_max"]))
    m = sm.vectors("pgse")
    a = np.abs(m).max(axis=(0, 1)).astype(np.float64)
    bits, scale = vector_code(a, rec["step"])
    fields, word = vector_fields(bits)
    assert rec["bits"] == list(bits) and rec["offsets"] == [o for o, _ in fields] and rec["word"] == word == 32
    np.testing.assert_allclose(rec["scale"], scale, rtol=1e-6)
    words = np.load(os.path.join(sm.path, manifest["columns"]["pgse"]), mmap_mode="r")
    assert words.dtype == np.uint32 and words.shape == (sm.n_tiles, sm.tile)
    with pytest.raises(ValueError, match="derived for at most"):
        sm.image("pgse", [1.01 * rec["b_max"]], [[0.0, 0.0, 1.0]])


def test_the_kernel_decodes_the_stored_vectors_exactly(layout):
    """The image's in-kernel decode is the host decoder's: the image equals the per-voxel sum over the decoded float
    vectors to float32 arithmetic."""
    col, sm, manifest, n_t = layout
    b = np.array([1e9, 3e9]); dirs = np.array([[0.6, 0.0, 0.8], [0.0, 1.0, 0.0]])
    S, _ = sm.image("pgse", b, dirs)
    m = sm.vectors("pgse").reshape(-1, 3).astype(np.float64); w = np.asarray(sm._column("w"), np.float64).reshape(-1)
    seg = np.repeat(np.asarray(sm._column("tiles")) // 2, sm.tile)
    g = sm.amplitude("pgse", b)
    E = w[:, None] * np.exp(1j * g[None, :] * (m @ dirs.T))
    num = np.stack([np.bincount(seg, E[:, k].real, minlength=sm.n_vox) + 1j * np.bincount(seg, E[:, k].imag, minlength=sm.n_vox)
                    for k in range(len(b))], 1)
    den = np.bincount(seg, w, minlength=sm.n_vox)
    ref = np.abs(num / den[:, None])
    assert np.nanmax(np.abs(S.reshape(-1, len(b)) - ref)) < 2e-6


def test_a_pgse_battery_replays_within_the_budget_of_the_floor_target(layout):
    """Thirty directions at five b-values up to the envelope's b_max: the packed layout's image against the float
    replay of the same bands (:meth:`ColumnarPack.image`) stays within ``BUDGET`` x the floor target in every voxel and
    measurement."""
    col, sm, manifest, n_t = layout
    rec = manifest["shapes"]["pgse"]["vector"]
    rng = np.random.default_rng(11)
    u = rng.normal(size=(30, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.repeat([0.5e9, 1e9, 1.5e9, 2e9, rec["b_max"]], 30); dirs = np.tile(u, (5, 1))
    S, _ = sm.image("pgse", b, dirs)
    seq = d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)
    S_float, _, _ = col.image(seq, tol=1e-9, chunk_rows=500)
    err = np.nanmax(np.abs(S - S_float))
    assert err < BUDGET * rec["floor_target"], (err, rec["floor_target"])


def _layout_at(col, tmp, name, floor_target, n_t):
    """The layout of ``col``'s rows written as if the pack declared ``floor_target`` (a copy of its columns with
    ``fidelity.target_floor`` set)."""
    import json
    import shutil
    src = str(tmp / f"{name}_columns")
    shutil.copytree(col.uri, src)
    path = os.path.join(src, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    man["meta"]["fidelity"]["target_floor"] = float(floor_target)
    with open(path, "w") as f:
        json.dump(man, f)
    shape = d.pgse([[0.0, 0.0, 1.0]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    out = str(tmp / f"{name}_moments")
    manifest = write_time_segment_layout(ColumnarPack(src), {"pgse": shape}, out, tol=1e-9, chunk_rows=500)
    return TimeSegmentLayout(out), manifest["shapes"]["pgse"]["vector"]


@pytest.mark.parametrize("k, word", [(9, 32), (10, 64)])
def test_mixed_widths_replay_within_the_budget(layout, spec_grid, k, word):
    """A floor target that puts the step between the components' largest vectors gives them different widths (the
    largest one bit more): 10 / 10 / 11 bits in one 32-bit word, 11 / 11 / 12 in one 64-bit word. Either replays the
    battery against the float replay within ``BUDGET`` x the walkers' own floor."""
    col, sm, manifest, n_t = layout
    _, _, tmp = spec_grid
    rec = manifest["shapes"]["pgse"]["vector"]
    a = np.sort(np.asarray(rec["scale"]) * (2.0 ** (np.asarray(rec["bits"]) - 1) - 1))
    assert a[2] / a[1] < 2                                              # one bit separates the largest from the others
    step = np.sqrt(a[1] * a[2]) / 2.0 ** k
    eps = step * rec["q_max"] * np.sqrt(2.0) / np.sqrt(12.0)           # vector_width inverted, in the sqrt branch
    sm_k, rec_k = _layout_at(col, tmp, f"mw_k{k}", eps ** 2 / BUDGET, n_t)
    assert sorted(rec_k["bits"]) == [k + 1, k + 1, k + 2] and rec_k["word"] == word
    words = np.load(os.path.join(sm_k.path, sm_k.manifest["columns"]["pgse"]), mmap_mode="r")
    assert words.itemsize == word // 8
    rng = np.random.default_rng(11)
    u = rng.normal(size=(30, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.repeat([0.5e9, 1.5e9, rec["b_max"]], 30); dirs = np.tile(u, (3, 1))
    S, _ = sm_k.image("pgse", b, dirs)
    seq = d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)
    S_float, _, _ = col.image(seq, tol=1e-9, chunk_rows=500)
    err = np.nanmax(np.abs(S - S_float))
    assert err < BUDGET * col.floor, (err, col.floor)

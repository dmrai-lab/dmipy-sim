"""dmipy-sim#689: the substrate kit (``Recipe.kit``) and its per-batch assembly (``assemble_request``), checked
against the shared request-directory writer (``dmipy_sim.engine.backends.write_request_directory``,
dmrai-lab/dmipy-sim#691) applied from scratch to a LIVE ``WalkRequest`` -- ``dmipy_sim_cuda`` is not installed on
this box, so this exercises the same serialisation the real package's ``write_request`` calls, without it (the
PR also runs the real package's ``write_request`` on the L40S against the same kit, ``test_kit_cuda.py``)."""
import os

import numpy as np
import pytest

from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
from dmipy_sim.engine.backends import Backend, write_request_directory
from dmipy_sim.fill import Recipe
from dmipy_sim.replay.bank import storage_band_hz
from dmipy_sim.fill.kit import _pool_geometry, _sha256, assemble_request


class _Captured(Exception):
    """Raised by ``_CaptureBackend.walk_batch`` to abort a walk the instant its ``WalkRequest`` is built -- no
    stepping, no GPU, just the request a real backend would have walked."""


class _CaptureBackend(Backend):
    name = "capture"

    def __init__(self):
        self.request = None

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys, out=None):
        self.request = request
        raise _Captured()


def _capture_request(geom, D, T_max, dt_save, *, seed, r0, field_basis, field_sample_every, field_reuse_intervals, f_band=None):
    """The live ``WalkRequest`` ``simulate_trajectories_adaptive`` would hand a backend for this pool -- aborted
    before any stepping, by ``_CaptureBackend``."""
    be = _CaptureBackend()
    try:
        simulate_trajectories_adaptive(len(r0), D, geom, T_max, dt_save, seed=seed, r0=r0, require_gpu=False,
                                       walker_batch_size=4096, field_basis=field_basis, field_sample_every=field_sample_every,
                                       field_reuse_intervals=field_reuse_intervals, backend=be, f_band=f_band)
    except _Captured:
        pass
    assert be.request is not None
    return be.request


def _starts(ctx, pid, n, seed):
    """``n`` valid start positions of pool ``pid`` -- the spec's own sampler (``_walk_bundle``'s seeding route)."""
    sampler = ctx.tests.sampler(pid)
    rng = np.random.default_rng(seed)
    pts, ok = sampler(8 * n, rng)
    pts = pts[ok]
    assert len(pts) >= n, f"only {len(pts)} of {n} wanted starts landed in pool {pid}"
    return np.ascontiguousarray(pts[:n], np.float32)


def _assert_dirs_match(oracle_dir, batch_dir):
    names_a, names_b = set(os.listdir(oracle_dir)), set(os.listdir(batch_dir))
    assert names_a == names_b, (names_a, names_b)
    for name in names_a:
        if name == "request.txt":
            a = dict(line.split("=", 1) for line in (oracle_dir / name).read_text().splitlines())
            b = dict(line.split("=", 1) for line in (batch_dir / name).read_text().splitlines())
            assert a == b, name
        else:
            assert _sha256(oracle_dir / name) == _sha256(batch_dir / name), name


def test_kit_layout_sha256_and_sections(field_fake, tmp_path):
    """The kit's layout: one request per diffusing pool with its strand tables, the field's tables and uncropped
    far grid once, the plan and the manifest, and ``kit.json`` naming the recipe and every file's sha256."""
    import json
    hub, _ = field_fake
    rc = Recipe(hub)
    out = rc.kit(tmp_path / "kit")
    kit = json.loads((out / "kit.json").read_text())
    assert kit["recipe_id"] == "test/fill" and kit["commit"] == "test" and kit["variant"] == "t"
    assert kit["dmipy_sim_version"]
    assert kit["field"]["present"] is True
    assert set(kit["pools"]) == {"intra", "extra"}
    for pool in ("intra", "extra"):
        pdir = out / "pools" / pool
        for name in ("request.txt", "request.json", "A.f32", "AB.f32", "AB2.f32", "rr.f32", "tube.i32", "cell_off.i32", "cell_ids.i32"):
            assert (pdir / name).is_file(), (pool, name)
    for name in ("fA.f32", "fAB.f32", "fAB2.f32", "fa.f32", "fb.f32", "fsid.i32", "fcell_off.i32", "fcell_ids.i32",
                "fbox.i16", "far_full.u16", "far_full.json", "field_mean.f32"):
        assert (out / name).is_file(), name
    mean = np.fromfile(out / "field_mean.f32", np.float32)
    assert mean.shape == (13,) and np.array_equal(mean, np.asarray(Recipe(hub).context().field_basis().mean, np.float32))
    assert kit["files"]["field_mean.f32"] == _sha256(out / "field_mean.f32")
    assert (out / "plan" / "blocks.json").is_file() and (out / "plan" / "counts.npz").is_file()
    assert (out / "manifest.json").is_file()
    assert "kit.json" not in kit["files"]
    for rel, sha in kit["files"].items():
        assert _sha256(out / rel) == sha, rel
    # no certificate was built on this hub: the kit carries none, and does not pretend to
    assert not (out / "certificate").exists()


def test_kit_without_a_field_variant(fake, tmp_path):
    """The field-off recipe's kit has no field tables and no far grid, but still one request per pool."""
    import json
    hub, _ = fake
    rc = Recipe(hub)
    out = rc.kit(tmp_path / "kit")
    kit = json.loads((out / "kit.json").read_text())
    assert kit["field"]["present"] is False
    assert not (out / "far_full.u16").exists()
    assert not (out / "field_mean.f32").exists() and "field_mean.f32" not in kit["files"]
    assert "field" not in (out / "pools" / "intra" / "request.json").read_text()


@pytest.mark.parametrize("fixture_name", ["fake", "field_fake"])
def test_kit_template_holds_the_scanner_class_band_not_the_save_grids_nyquist(fixture_name, request, tmp_path):
    """The kit's ``sub_steps``, kernel steps per class and step lengths are the plan's at the class's stored band,
    as the recipe's own walk resolves them, and never more than the plan at the save grid's Nyquist frequency.
    (This fake recipe's micron pores set the step through the pore term, so the two plans coincide here; on a
    substrate whose band term sets the step the band's is the smaller.)"""
    import json
    from dmipy_sim.engine.adaptive import adaptive_plan
    hub, _ = request.getfixturevalue(fixture_name)
    rc = Recipe(hub)
    ctx = rc.context(); g = ctx.tests; W = rc.man["walk"]
    out = rc.kit(tmp_path / "kit")
    n_t = int(round(float(W["T_max_s"]) / rc.dt_save())) + 1
    dt = float(W["T_max_s"]) / (n_t - 1)
    band = storage_band_hz(W["scanner"])
    for pid in g.seeded:
        pool = g.pools[pid]
        if pool.D in (None, 0.0) or (bool(g.inside_w[pid]) and bool(g.outside_w[pid])):
            continue
        geom = _pool_geometry(ctx, pid)
        banded = adaptive_plan(geom, float(pool.D), dt, f_band=band)
        nyq = adaptive_plan(geom, float(pool.D), dt)
        req = json.loads((out / "pools" / pool.name / "request.json").read_text())
        msg = f"{pool.name}: band {band:.0f} Hz sub_steps {banded.n_min}, Nyquist {nyq.f_band_hz:.0f} Hz sub_steps {nyq.n_min}"
        assert req["sub_steps"] == banded.n_min <= nyq.n_min, msg
        assert req["adaptive_steps_c"] == [int(m) for m in banded.steps_c] + [0] * (8 - banded.n_classes), msg
        assert sum(req["adaptive_steps_c"]) <= sum(int(m) for m in nyq.steps_c) + 0, msg


@pytest.mark.parametrize("fixture_name,pool_name", [("fake", "intra"), ("fake", "extra"),
                                                     ("field_fake", "intra"), ("field_fake", "extra")])
def test_a_batch_request_from_the_kit_matches_a_live_walk_requests_dump(fixture_name, pool_name, request, tmp_path):
    """The kit's template + a batch's own ``r0``/``keys``/``seed`` (+ the far-grid crop, with a field) is
    file-identical to what a live ``WalkRequest`` of this exact batch would be serialised as."""
    hub, _ = request.getfixturevalue(fixture_name)
    rc = Recipe(hub)
    ctx = rc.context(); g = ctx.tests; W = rc.man["walk"]
    pid = next(p.id for p in g.pools.values() if p.name == pool_name)
    basis = ctx.field_basis()

    kit_dir = rc.kit(tmp_path / "kit")

    geom = _pool_geometry(ctx, pid)
    D = float(g.pools[pid].D)
    T_max, dt_save = float(W["T_max_s"]), rc.dt_save()
    r0 = _starts(ctx, pid, 40, seed=1000 + pid)
    seed = 555 + pid
    keys = np.random.default_rng(2000 + pid).integers(0, 2 ** 31 - 1, size=(len(r0), 2)).astype(np.uint32)

    req = _capture_request(geom, D, T_max, dt_save, seed=seed, r0=r0, field_basis=basis,
                           field_sample_every=int(W.get("field_sample_every", 1)), field_reuse_intervals=int(W.get("field_gather_every", 4)),
                           f_band=storage_band_hz(W["scanner"]))
    assert (req.field is not None) == (basis is not None)

    oracle_dir = write_request_directory(req, r0, keys, tmp_path / f"oracle-{pool_name}")
    batch_dir = assemble_request(kit_dir, pool_name, r0, keys, seed, tmp_path / f"batch-{pool_name}")
    _assert_dirs_match(oracle_dir, batch_dir)

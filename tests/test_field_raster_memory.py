"""The memory a mesh substrate's field raster takes: the build one channel at a time (the same channels to the bit), the
refusal of a raster the host cannot build or the device cannot hold (before anything is allocated, in the walk and in the
family plan), the raster cache, and the workers of the mesh queries (spawned, never forked)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from dmipy_sim.fields import raster_cache
from dmipy_sim.fields import susceptibility_field as sf
from dmipy_sim.spec import SpecError, check_field_raster, field_grid_of_spec, field_raster_plan
from dmipy_sim.spec.walk import WalkContext

ROOT = Path(__file__).resolve().parents[1]
UM = 1e-6


def _monolithic_field_basis(myelin_mask, radial_dir, voxel_size, include_aniso=True, kspace_lowpass=None):
    """The oracle: the build with every k-space grid held at once, as it was written before the channels were made one
    at a time."""
    m = np.asarray(myelin_mask, float)
    shape = m.shape
    khat, dc = sf._khat(shape, voxel_size)
    W = 1.0
    if kspace_lowpass is not None:
        axf = [np.fft.fftfreq(n) * 2.0 for n in shape]
        r = np.sqrt(sum(g ** 2 for g in np.meshgrid(*axf, indexing="ij")) / len(shape))
        W = np.ones_like(r); f = float(kspace_lowpass)
        ramp = (r > f) & (r < 1.0)
        W[ramp] = 0.5 * (1.0 + np.cos(np.pi * (r[ramp] - f) / (1.0 - f)))
        W[r >= 1.0] = 0.0
    mk = np.fft.fftn(m)
    iso_local = (m - m.mean()) / 3.0
    iso_P = np.empty((6,) + shape, float)
    for c, (i, j) in enumerate(sf._SYM6):
        iso_P[c] = np.real(np.fft.ifftn(mk * khat[i] * khat[j] * W))
    aniso_G = None
    if include_aniso:
        n = sf._unit(np.asarray(radial_dir, float), axis=-1)
        Tk = {}
        for (i, j) in sf._SYM6:
            Td = m * (n[..., i] * n[..., j] - (1.0 / 3.0 if i == j else 0.0))
            Tk[(i, j)] = np.fft.fftn(Td)
            Tk[(j, i)] = Tk[(i, j)]
        aniso_G = np.empty((6,) + shape, complex)
        for c, (i, j) in enumerate(sf._SYM6):
            Kij = (1.0 / 3.0) * Tk[(i, j)]
            Kji = (1.0 / 3.0) * Tk[(j, i)]
            sec_ij = sum(khat[i] * khat[cc] * Tk[(cc, j)] for cc in range(3))
            sec_ji = sum(khat[j] * khat[cc] * Tk[(cc, i)] for cc in range(3))
            Ksym = 0.5 * ((Kij - sec_ij) + (Kji - sec_ji))
            Ksym[dc] = 0.0
            aniso_G[c] = Ksym
        aniso_G = np.stack([np.real(np.fft.ifftn(aniso_G[c] * W)) for c in range(6)], axis=0)
    return dict(iso_local=iso_local, iso_P=iso_P, aniso_G=aniso_G)


def _equal(a, b):
    assert np.array_equal(a["iso_local"], b["iso_local"]) and np.array_equal(a["iso_P"], b["iso_P"])
    assert (a["aniso_G"] is None) == (b["aniso_G"] is None)
    if a["aniso_G"] is not None:
        assert np.array_equal(a["aniso_G"], b["aniso_G"])


@pytest.mark.parametrize("shape,lowpass,aniso", [((12, 10, 9), None, True), ((12, 10, 9), 0.5, True), ((8, 8, 8), 0.5, False)])
def test_the_per_channel_build_is_the_monolithic_build_to_the_bit(shape, lowpass, aniso):
    """The channels made one at a time are the channels made together: no FFT is batched, every channel is the same
    elementwise arithmetic on the same operands, so the arrays are equal, not close."""
    rng = np.random.default_rng(0)
    m = rng.random(shape); n = rng.normal(size=shape + (3,)); vs = [1.0, 1.3, 0.9]
    _equal(sf.field_basis(m, n, vs, include_aniso=aniso, kspace_lowpass=lowpass),
           _monolithic_field_basis(m, n, vs, include_aniso=aniso, kspace_lowpass=lowpass))
    got = list(sf.field_basis_channels(m, n, vs, include_aniso=aniso, kspace_lowpass=lowpass))
    assert [(g, c) for g, c, _ in got] == [("iso_local", None)] + [("iso_P", c) for c in range(6)] + \
        ([("aniso_G", c) for c in range(6)] if aniso else [])
    assert all(a.shape == shape and a.dtype == np.float64 for _, _, a in got)


def test_a_mesh_and_a_predicate_raster_are_unchanged_by_the_per_channel_build(monkeypatch):
    """The whole raster of a meshed hollow cylinder (partial-volume edge, nearest-face director) and of a predicate shell
    is the monolithic build's, to the bit."""
    import trimesh
    def cyl(r):
        m = trimesh.creation.cylinder(radius=r, height=6.0, sections=32)
        return np.asarray(m.vertices) * UM, np.asarray(m.faces)
    box = (np.full(3, -2.0 * UM), np.full(3, 2.0 * UM))
    mesh = lambda: sf.mesh_field_basis(sf.MeshBodies([cyl(0.8)], workers=1), sf.MeshBodies([cyl(1.2)], workers=1), *box, res=0.25 * UM)[0]
    ball = lambda p, r: np.linalg.norm(p[:, :2], axis=1) < r * UM
    pred = lambda: sf.predicate_field_basis(lambda p: ball(p, 0.8), lambda p: ball(p, 1.2), *box, res=0.25 * UM)[0]
    new = [mesh(), pred()]
    monkeypatch.setattr(sf, "field_basis", lambda *a, **k: dict(_monolithic_field_basis(*a, **k), shape=a[0].shape, voxel_size=sf._as_voxel_size(a[2])))
    old = [mesh(), pred()]
    for a, b in zip(new, old):
        _equal(a, b)
        assert a["shell_fraction"] == b["shell_fraction"] and 0 < a["shell_fraction"] < 1


def test_one_channels_work_is_held_at_a_time():
    """The k-space work above the finished channels is one channel's: the generator, drained, peaks well under the
    monolithic build (measured 273 against 562 B/node at 64^3, the 13 float64 channels it leaves 104 B/node)."""
    n = 32
    rng = np.random.default_rng(1)
    m = rng.random((n,) * 3); d = rng.normal(size=(n,) * 3 + (3,)); N = n ** 3
    def peak(f):
        tracemalloc.start(); base = tracemalloc.get_traced_memory()[0]
        f(); p = tracemalloc.get_traced_memory()[1] - base; tracemalloc.stop()
        return p / N
    def drain():
        for _ in sf.field_basis_channels(m, d, 1.0, True, 0.5):
            pass
    streamed = peak(drain)
    whole = peak(lambda: _monolithic_field_basis(m, d, 1.0, True, 0.5))
    assert streamed < 0.6 * whole, (streamed, whole)


# ---------------------------------------------------------------- the refusals, the plan and the cache
@pytest.fixture(scope="module")
def fr():
    spec = importlib.util.spec_from_file_location("family_rewalk", ROOT / "tools" / "family_rewalk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cactus_spec_small(fr, tmp_path_factory):
    """A two-strand CACTUS-shaped spec of a 6 um cell (the family driver's smoke substrate)."""
    from dmipy_sim.spec import cactus_spec
    out = tmp_path_factory.mktemp("smoke")
    run = fr._smoke_cactus_run(fr.argparse.Namespace(out=str(out), smoke_side_um=6.0))
    return cactus_spec(run, side_um=6.0)


def test_the_raster_bytes_are_the_nodes_times_the_channels(cactus_spec_small):
    plan = field_raster_plan(cactus_spec_small)
    assert plan.n_nodes == int(np.prod(plan.shape)) and plan.channels == 12
    assert plan.device_bytes == sf.FIELD_DEVICE_COPIES * 4 * 12 * plan.n_nodes
    assert plan.host_bytes == sf.FIELD_BYTES_PER_NODE * plan.n_nodes
    grid_bytes = 4 * 12 * plan.n_nodes
    fg = field_grid_of_spec(cactus_spec_small, device_limit=10 * 2 ** 30)
    assert fg.device_arrays()[0].nbytes == grid_bytes                     # the walk's argument is one copy of it


def test_a_raster_beyond_the_device_is_refused_by_name_before_it_is_built(cactus_spec_small, monkeypatch):
    plan = field_raster_plan(cactus_spec_small)
    built = []
    monkeypatch.setattr(sf, "mesh_field_basis", lambda *a, **k: built.append(1))
    with pytest.raises(SpecError) as e:
        field_grid_of_spec(cactus_spec_small, device_limit=(plan.device_bytes - 1, "the injected limit"))
    msg = str(e.value)
    for part in (cactus_spec_small.id, f"{plan.n_nodes:.2e} nodes", "12 channels", "GiB on the device", "the injected limit"):
        assert part in msg, part
    assert not built                                                      # refused before any allocation
    with pytest.raises(SpecError, match="voxel budget"):                   # the host side as before
        field_grid_of_spec(cactus_spec_small, field_budget=10, device_limit=10 * 2 ** 40)
    check_field_raster(cactus_spec_small, plan, device_limit=plan.device_bytes)       # exactly enough passes
    check_field_raster(cactus_spec_small, plan, device_limit=(None, "cpu"))           # a CPU device: the host budget governs


def test_the_device_limit_is_read_from_the_device_that_walks(monkeypatch):
    import jax
    class Dev:
        platform = "gpu"
        def memory_stats(self):
            return {"bytes_limit": 123}
        def __str__(self):
            return "FakeGPU"
    monkeypatch.setattr(jax, "devices", lambda *a, **k: [Dev()])
    lim, src = sf.field_device_limit()
    assert lim == 123 and "bytes_limit" in src and "FakeGPU" in src
    Dev.memory_stats = lambda self: None
    with pytest.raises(RuntimeError, match="no bytes_limit"):
        sf.field_device_limit()
    assert sf.field_device_limit.__doc__ and sf.field_device_bytes(10) == sf.FIELD_DEVICE_COPIES * 4 * 12 * 10


def test_the_family_plan_refuses_a_raster_the_device_cannot_hold(fr, tmp_path, capsys):
    args = ["plan", "cactus", "--smoke", "--smoke-n", "60", "--T", "0.004", "--backend", "jax", "--out", str(tmp_path)]
    fr.main(args + ["--device-memory-gib", "10"])
    rows = fr_rows(tmp_path)
    assert rows[0]["accepted"] and rows[0]["field_raster"]["device_checked"] and rows[0]["field_raster"]["device_limit_source"] == "--device-memory-gib"
    fr.main(args + ["--device-memory-gib", "0.0001"])
    row = fr_rows(tmp_path)[0]
    assert not row["accepted"] and "on the device in the walk" in row["refusal"] and "--device-memory-gib" in row["refusal"]
    fr.main(args)                                                          # a CPU host states that the device is not checked
    assert "not checked" in capsys.readouterr().out
    assert fr_rows(tmp_path)[0]["field_raster"]["device_checked"] is False


def fr_rows(out):
    import json
    return json.load(open(Path(out) / "plan_cactus.json"))["rows"]


def test_the_cache_round_trip_is_the_same_grid(cactus_spec_small, tmp_path):
    ctx = WalkContext(cactus_spec_small)
    built = field_grid_of_spec(cactus_spec_small, context=ctx, cache_dir=tmp_path)
    key = raster_cache.field_raster_key(cactus_spec_small, field_raster_plan(cactus_spec_small, context=ctx).res)
    path = tmp_path / f"{key}.safetensors"
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    loaded = field_grid_of_spec(cactus_spec_small, context=ctx, cache_dir=tmp_path)      # a hit: not rebuilt, not rewritten
    assert path.stat().st_mtime_ns == mtime
    for k in ("iso_local", "iso_P", "aniso_G"):
        assert np.array_equal(built.basis[k], loaded.basis[k]) and built.basis[k].dtype == loaded.basis[k].dtype
    assert tuple(loaded.basis["shape"]) == tuple(built.basis["shape"])
    assert np.array_equal(loaded.basis["voxel_size"], built.basis["voxel_size"]) and loaded.basis["shell_fraction"] == built.basis["shell_fraction"]
    assert np.array_equal(loaded.origin, built.origin) and loaded.periodic == built.periodic
    assert loaded.certificate == built.certificate and loaded.meta == built.meta
    rec = raster_cache.field_raster_cache_record(cactus_spec_small, field_raster_plan(cactus_spec_small, context=ctx).res, tmp_path)
    assert rec["exists"] and rec["key"] == key and rec["content_sha256"] == raster_cache.read_field_raster_header(path)["content_sha256"]


def test_a_cache_file_that_is_not_the_raster_asked_for_is_refused(cactus_spec_small, tmp_path):
    field_grid_of_spec(cactus_spec_small, cache_dir=tmp_path)
    (path,) = list(tmp_path.glob("*.safetensors"))
    key = path.stem
    other = tmp_path / ("0" * 64 + ".safetensors")
    other.write_bytes(path.read_bytes())                                     # right content under another key
    with pytest.raises(ValueError, match="another raster"):
        raster_cache.load_field_grid(str(other), "0" * 64)
    with pytest.raises(ValueError, match="another raster"):
        raster_cache.load_field_grid(str(path), "1" * 64)
    blob = bytearray(path.read_bytes()); blob[-9] ^= 0xFF                    # one corrupt byte of the channels
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="corrupt"):
        field_grid_of_spec(cactus_spec_small, cache_dir=tmp_path)
    assert path.stem == key


def test_the_key_names_the_spec_the_spacing_and_the_format(cactus_spec_small, monkeypatch):
    res = field_raster_plan(cactus_spec_small).res
    key = raster_cache.field_raster_key(cactus_spec_small, res)
    assert key == raster_cache.field_raster_key(cactus_spec_small, res) and len(key) == 64
    assert raster_cache.field_raster_key(cactus_spec_small, res * 1.01) != key
    monkeypatch.setattr(raster_cache, "FIELD_RASTER_FORMAT", "field-raster-next")
    assert raster_cache.field_raster_key(cactus_spec_small, res) != key


# ---------------------------------------------------------------- no fork after JAX
def test_the_mesh_workers_are_spawned_never_forked(monkeypatch):
    """A membership query over a million points and the signed distances of several chunks run on spawned workers (a
    fork of a process that has initialised JAX is incompatible with its threads) and give the serial answer."""
    import trimesh
    bodies = []
    for c in [(0, 0, 0), (3, 0, 0), (0, 3, 0)]:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        bodies.append((np.asarray(m.vertices) + np.array(c, float), np.asarray(m.faces)))
    pts = np.random.default_rng(0).uniform(-1.5, 4.5, (1_000_000, 3))
    serial, par = sf.MeshBodies(bodies, workers=1), sf.MeshBodies(bodies, workers=2)
    want = serial.contains(pts); s_want = serial.signed_distance(pts[:400_000], resolve_m=0.4)
    def no_fork():
        raise AssertionError("os.fork was called")
    monkeypatch.setattr(os, "fork", no_fork)
    try:
        assert np.array_equal(par.contains(pts), want)
        s, n = par.signed_distance(pts[:400_000], resolve_m=0.4)
    finally:
        par.close()
    assert np.array_equal(s, s_want[0]) and np.array_equal(n, s_want[1])
    assert not par._dpools                                                   # the pool and its shared memory released


def test_a_failed_walk_of_a_family_run_returns_promptly(tmp_path):
    """A smoke walk of the bundle family whose walk fails after the mesh workers have run: the run reports the failure
    and exits (status 1) in seconds, with no worker left to wait on."""
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, importlib.util\n"
        "import numpy as np, jax\n"
        "from dmipy_sim.fields.susceptibility_field import MeshBodies\n"
        "import trimesh\n"
        "if __name__ == '__main__':\n"
        "    jax.devices()                                    # JAX is up, as in a walk\n"
        "    spec = importlib.util.spec_from_file_location('family_rewalk', sys.argv[1])\n"
        "    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)\n"
        "    def walk(self, name, seed, backend):\n"
        "        ms = [trimesh.creation.icosphere(subdivisions=1) for _ in range(2)]\n"
        "        b = MeshBodies([(np.asarray(m.vertices) + 3 * i, np.asarray(m.faces)) for i, m in enumerate(ms)], workers=2)\n"
        "        b.contains(np.zeros((1_000_000, 3)))\n"
        "        raise RuntimeError('injected walk failure')\n"
        "    fr.Cactus.walk = walk\n"
        "    fr.main(['walk', 'cactus', '--smoke', '--smoke-n', '60', '--T', '0.004', '--backend', 'jax', '--out', sys.argv[2]])\n")
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(driver), str(ROOT / "tools" / "family_rewalk.py"), str(tmp_path / "out")],
                       env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 1 and "FAILED RuntimeError: injected walk failure" in r.stdout, (r.stdout[-800:], r.stderr[-800:])
    assert time.time() - t0 < 120

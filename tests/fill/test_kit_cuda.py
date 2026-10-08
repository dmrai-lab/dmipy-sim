"""dmipy-sim#689: the kit's per-batch assembly checked against the REAL ``dmipy_sim_cuda.cli.write_request`` --
the cross-check ``tests/fill/test_kit.py`` could only run against a from-scratch reproduction of it (that package
is not installed on the CPU box the suite otherwise runs on). Skipped unless ``dmipy_sim_cuda`` is importable and
built (the L40S bench venv, dmrai-lab/dmipy-sim-cuda)."""
import math
import os

import numpy as np
import pytest

pytest.importorskip("dmipy_sim_cuda")
from dmipy_sim_cuda.cli import write_request  # noqa: E402

from dmipy_sim.engine.adaptive import adaptive_plan  # noqa: E402
from dmipy_sim.engine.backends import FieldSampling, WalkRequest  # noqa: E402
from dmipy_sim.fill import Recipe  # noqa: E402
from dmipy_sim.fill.kit import _pool_geometry, _sha256, assemble_request  # noqa: E402


def _starts(ctx, pid, n, seed):
    sampler = ctx.tests.sampler(pid)
    rng = np.random.default_rng(seed)
    pts, ok = sampler(8 * n, rng)
    pts = pts[ok]
    assert len(pts) >= n, f"only {len(pts)} of {n} wanted starts landed in pool {pid}"
    return np.ascontiguousarray(pts[:n], np.float32)


def _assert_dirs_match(a_dir, b_dir):
    names_a, names_b = set(os.listdir(a_dir)), set(os.listdir(b_dir))
    assert names_a == names_b, (names_a, names_b)
    for name in names_a:
        if name == "request.txt":
            la = dict(line.split("=", 1) for line in (a_dir / name).read_text().splitlines())
            lb = dict(line.split("=", 1) for line in (b_dir / name).read_text().splitlines())
            assert la == lb, name
        else:
            assert _sha256(a_dir / name) == _sha256(b_dir / name), name


@pytest.mark.parametrize("fixture_name,pool_name", [("fake", "intra"), ("fake", "extra"),
                                                     ("field_fake", "intra"), ("field_fake", "extra")])
def test_a_kit_batch_matches_dmipy_sim_cudas_own_write_request(fixture_name, pool_name, request, tmp_path):
    """A batch request assembled from the kit (no dmipy-sim, no spec, disk and arrays alone) is file-identical to
    what ``dmipy_sim_cuda.cli.write_request`` itself writes for the SAME live ``WalkRequest``."""
    hub, _ = request.getfixturevalue(fixture_name)
    rc = Recipe(hub)
    ctx = rc.context(); g = ctx.tests; W = rc.man["walk"]
    pid = next(p.id for p in g.pools.values() if p.name == pool_name)
    basis = ctx.field_basis()

    kit_dir = rc.kit(tmp_path / "kit")

    geom = _pool_geometry(ctx, pid)
    D = float(g.pools[pid].D)
    T_max, dt_save = float(W["T_max_s"]), rc.dt_save()
    n_t = int(round(T_max / dt_save)) + 1
    dt_actual = T_max / (n_t - 1)
    plan = adaptive_plan(geom, D, dt_actual)

    r0 = _starts(ctx, pid, 40, seed=3000 + pid)
    seed = 909 + pid
    keys = np.random.default_rng(4000 + pid).integers(0, 2 ** 31 - 1, size=(len(r0), 2)).astype(np.uint32)

    field = None
    if basis is not None:
        f_every = max(1, int(W.get("field_sample_every", 1)))
        f_reuse = max(1, int(W.get("field_gather_every", 4)))
        f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)
        f_reach = float(basis.gather_radius_m)
        f_radius = min(f_reach + f_margin, 2.0 * f_reach)
        field = FieldSampling(basis, f_every, f_reuse, float(f_radius), 256)

    req = WalkRequest(geometry=geom, n_t=n_t, dt_save=dt_actual, sub_steps=int(plan.n_min), dt_sim=float(plan.dt_min),
                      diffusivity=D, record=True, kappa_over_D=0.0, count_walls=bool(geom.count_walls),
                      bounce_budget=geom.bounce_budget, seed=seed, stepping=plan, field=field)

    real_dir = write_request(req, r0, keys, tmp_path / f"real-{pool_name}")
    batch_dir = assemble_request(kit_dir, pool_name, r0, keys, seed, tmp_path / f"batch-{pool_name}")
    _assert_dirs_match(real_dir, batch_dir)

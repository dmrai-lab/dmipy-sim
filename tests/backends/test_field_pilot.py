"""The field pilot walked on the walk's backend (dmrai-lab/dmipy-sim#822): the backend walks the pilot's walkers one step
a block with all of them in the raw subset, reads its own field at their positions, and the host computes the pilot's
statistics (accumulate.field_read_statistics). A stand-in backend that walks that request by the JAX kernels -- the
same keys, the same physics, so the same trajectories as the JAX pilot -- and reads by the JAX reader gives the JAX
pilot's read record and band: the definitions agree, the plumbing holds (the request, the batches, the labels)."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.engine.physics import seed_walkers
from dmipy_sim.fields.susceptibility_field import FieldGrid

from .test_backend_hook import coefficient_batch

D = 2e-9
R = 2e-6


class _JaxProbe(backends.Backend):
    """Walks the field pilot's request (every walker in the raw subset, one step a block) by the JAX kernels from the
    request's own keys, and reads a FieldGrid by the JAX reader; any other batch is walkers that never move."""
    name = "jaxprobe"

    def __init__(self):
        self.probes = []

    def walk_batch(self, request, r0, keys, *, out=None):
        n = int(r0.shape[0])
        b = coefficient_batch(request, r0)
        if not (request.raw_subset == n and request.sub_steps == 1):
            return b
        self.probes.append(request)
        assert request.field is None and request.K_c3 == 0
        _, _, k = seed_walkers(request.geometry, n, int(request.seed), r0)
        assert np.array_equal(np.asarray(k, np.uint32), keys), "the stand-in reproduces the JAX keys of a whole-walk pilot only"
        w = _walk_geometry(n, float(request.diffusivity), request.geometry, float(request.T), float(request.block_h),
                                  seed=int(request.seed), sub_steps=1, r0=r0, raw_subset=n, K=int(request.K), window_T=float(request.T),
                                  tiers=("all" if request.record else ()), walker_batch_size=n, require_gpu=False)
        return b._replace(saves=b.saves._replace(positions=np.asarray(w.raw_positions, np.float32)))

    def sample_raster(self, grid, points):
        return np.asarray(grid.channels_at_device()(np.asarray(points, np.float32)), np.float32)


def _grid():
    """A field varying on the step's scale across the sphere: iso_local a sine of x, iso_P[0] of y (60 nodes per axis)."""
    nx = 24; vs = np.full(3, 2 * R / nx); x = -R + (np.arange(nx) + 0.5) * vs[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P = np.zeros((6, nx, nx, nx), np.float32)
    P[0] = np.cos(Y / (0.3 * R)); P[3] = np.sin((X + Z) / (0.4 * R))
    return FieldGrid(dict(iso_local=np.sin(X / (0.25 * R)).astype(np.float32), iso_P=P, shape=(nx,) * 3, voxel_size=vs),
                     np.full(3, -R))


def test_the_backends_pilot_is_the_jax_pilot():
    """The same walkers, keys and physics: the backend's pilot reads every step the JAX pilot reads, so the read record
    (the rule's interval and read, the per-pool and per-window variation, the trapezoid's measured constants at every
    check) and the band are the JAX pilot's -- the record's numbers to float32 accumulation (the JAX pilot sums a window
    in float32, the host in float64: rtol 1e-4 covers (L + 4) 2^-24 for the window's 1,600 steps), the chosen read and
    band identical. The record names the engine that walked the pilot."""
    g = d.Sphere(R)
    fg = _grid()
    kw = dict(field_basis=fg, walker_batch_size=200)
    run = lambda **k: _walk_geometry(200, D, g, 4e-3, 0.5e-3, seed=7, tiers="all", require_gpu=False, **kw, **k)
    be = _JaxProbe()
    a = run(backend=be, field_pilot="jax")
    assert not be.probes
    b = run(backend=be)
    assert len(be.probes) == 1 and be.probes[0].raw_subset == 200
    fa, fb = a.block["field_read"], b.block["field_read"]
    assert fa["pilot_backend"] == "jax" and fb["pilot_backend"] == "jaxprobe"
    assert (fa["n_read"], fa["dt_f"]) == (fb["n_read"], fb["dt_f"])
    assert a.K_c3 == b.K_c3 and a.block["field_band"]["K"] == b.block["field_band"]["K"]
    np.testing.assert_allclose(fa["term_pilot"], fb["term_pilot"], rtol=1e-4)
    for name in fa["pools"]:
        np.testing.assert_allclose(fa["pools"][name]["qv_increments"], fb["pools"][name]["qv_increments"], rtol=1e-4)
    assert fa["measured_constants"].keys() == fb["measured_constants"].keys()
    for dd in fa["measured_constants"]:
        np.testing.assert_allclose(np.asarray(fa["measured_constants"][dd], float), np.asarray(fb["measured_constants"][dd], float), rtol=1e-4)
    print(f"\nK_c3 {a.K_c3}, read every {fa['n_read']} steps; constants at the read {fb['constants']}")

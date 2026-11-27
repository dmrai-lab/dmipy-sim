"""Fixed-step backend pilot dispatch and parity with the JAX pilot."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends, core
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.fields.susceptibility_field import FieldGrid

from .test_backend_hook import _Still
from .test_backend_capabilities import FIELD, _CellField, _cell, _walk
from dmipy_sim.fields.hollow_cylinder import HollowCylinderField

D = 2e-9
R = 2e-6


class _JaxProbe(_Still):
    """The JAX pilot kernels receive the backend request, starts and keys."""
    name = "jaxprobe"
    capabilities = frozenset({backends.FIELD_PILOT, backends.STORAGE_WINDOWS})

    def __init__(self):
        super().__init__()
        self.probes = []

    def field_pilot(self, request, r0, keys, checks):
        self.probes.append(request)
        return core._jax_field_pilot_for_request(request, checks, r0, keys)


def _grid():
    """A field varying on the step's scale across the sphere: iso_local a sine of x, iso_P[0] of y (24 nodes per axis)."""
    nx = 24; vs = np.full(3, 2 * R / nx); x = -R + (np.arange(nx) + 0.5) * vs[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P = np.zeros((6, nx, nx, nx), np.float32)
    P[0] = np.cos(Y / (0.3 * R)); P[3] = np.sin((X + Z) / (0.4 * R))
    return FieldGrid(dict(iso_local=np.sin(X / (0.25 * R)).astype(np.float32), iso_P=P, shape=(nx,) * 3, voxel_size=vs),
                     np.full(3, -R))


def test_the_backends_pilot_is_the_jax_pilot():
    """Wrong production blocks, reads or pool reductions would change the JAX pilot's read or band record."""
    g = d.Sphere(R)
    fg = _grid()
    kw = dict(field_basis=fg, walker_batch_size=200, first_walker=13)
    run = lambda **k: _walk_geometry(200, D, g, 4e-3, 0.5e-3, seed=7, tiers="all", require_gpu=False, **kw, **k)
    be = _JaxProbe()
    a = run(backend=be, field_pilot="jax")
    assert not be.probes
    b = run(backend=be)
    probe, = be.probes
    assert probe.field is fg and probe.raw_subset == 0
    assert (probe.n_blocks, probe.sub_steps, probe.n_windows) == (8, a.sub_steps, 1)
    assert all(req.n_blocks == probe.n_blocks and req.sub_steps == probe.sub_steps for req in be.requests)
    fa, fb = a.block["field_read"], b.block["field_read"]
    assert fa["pilot_backend"] == "jax" and fb["pilot_backend"] == "jaxprobe"
    assert (fa["n_read"], fa["dt_f"]) == (fb["n_read"], fb["dt_f"])
    assert a.K_c3 == b.K_c3 and a.block["field_band"]["K"] == b.block["field_band"]["K"]
    np.testing.assert_allclose(fa["term_pilot"], fb["term_pilot"], rtol=1e-4)
    assert fa["pools"].keys() == fb["pools"].keys()
    for name in fa["pools"]:
        assert fa["pools"][name]["walkers"] == fb["pools"][name]["walkers"]
        np.testing.assert_allclose(fa["pools"][name]["qv_increments"], fb["pools"][name]["qv_increments"], rtol=1e-4)
    assert fa["measured_constants"].keys() == fb["measured_constants"].keys()
    for dd in fa["measured_constants"]:
        np.testing.assert_allclose(np.asarray(fa["measured_constants"][dd], float), np.asarray(fb["measured_constants"][dd], float), rtol=1e-4)


@pytest.mark.parametrize("permeability", [None, 1e-5])
def test_the_field_pilot_uses_the_production_pool_change_setting(permeability):
    """The pilot must preserve the production pool-change policy for sealed and permeable membranes."""
    class Probe(_JaxProbe):
        def walk_batch(self, request, r0, keys, *, out=None):
            batch = super().walk_batch(request, r0, keys, out=out)
            return batch._replace(pool_changed=np.zeros(len(r0), bool))

    be = Probe()
    drops = permeability is None
    _walk_geometry(16, D, d.Sphere(R, permeability=permeability), 4e-3, 0.5e-3, seed=7, tiers="all", require_gpu=False,
                   field_basis=_grid(), walker_batch_size=16, backend=be, drop_pool_changes=drops)
    probe, = be.probes
    assert probe.drop_pool_changes == drops
    assert be.requests and all(request.drop_pool_changes == drops for request in be.requests)


@pytest.mark.parametrize("permeable", [False, True])
def test_a_cells_field_pilot_does_not_request_extra_pool_change_tracking(permeable):
    """Even on a sealed cell the pilot must preserve the default production policy on pool changes."""
    g = _cell(kappa_outer=[0.0, 4e-5, 0.0, 4e-5]) if permeable else _cell()
    be = _CellField(events=lambda request, r0: (np.zeros(len(r0) + 1, np.int64), np.zeros(0), np.zeros(0, np.int8)))
    _walk(g, n=16, walker_batch_size=16, backend=be, field_basis=HollowCylinderField.of(g), **FIELD)
    probe, = be.pilot_requests
    assert not probe.drop_pool_changes
    assert be.requests and all(not request.drop_pool_changes for request in be.requests)


def test_backend_without_field_pilot_is_refused_before_seeding(monkeypatch):
    """A stated band and read interval still need a pilot, and the JAX override bypasses the backend requirement."""
    def forbidden(*args, **kwargs):
        pytest.fail("seeded walkers before refusing the backend")

    g, be = d.Sphere(R), _Still()
    kw = dict(field_basis=_grid(), K_c3=3, field_read_interval=4e-5, backend=be, require_gpu=False)
    with monkeypatch.context() as patch:
        patch.setattr(core, "seed_walkers", forbidden)
        with pytest.raises(ValueError, match=r"backend 'still'.*capability 'field_pilot'.*field_pilot='jax'.*field_calibration"):
            _walk_geometry(16, D, g, 4e-3, 0.5e-3, **kw)
    assert not be.requests
    got = _walk_geometry(16, D, g, 4e-3, 0.5e-3, field_pilot="jax", **kw)
    assert got.block["field_read"]["pilot_backend"] == "jax"


def test_the_field_pilot_receives_the_production_mt_start_dwell():
    """Reseeding or omitting the initial MT dwell would change when pilot walkers start moving."""
    from .test_mt_backend import _StillMT, _walk as walk_mt

    class Probe(_StillMT):
        capabilities = _StillMT.capabilities | {backends.FIELD_PILOT}

        def field_pilot(self, request, r0, keys, checks):
            self.pilot_dwell = request.dwell0
            return core._jax_field_pilot_for_request(request, checks, r0, keys)

    be = Probe()
    walk_mt(be, field_basis=_grid(), equilibrate_binding="fast")
    production_dwell = np.concatenate([dwell for request, dwell in be.calls if request.K_c3])
    assert be.pilot_dwell.dtype == np.float32 and np.any(production_dwell > 0)
    np.testing.assert_array_equal(be.pilot_dwell, production_dwell)


@pytest.mark.parametrize("key", ["fs", "qv_d", "chk"])
@pytest.mark.parametrize("defect", ["shape", "dtype"])
def test_malformed_backend_pilot_arrays_are_refused_by_name(key, defect):
    """Every pilot array is checked before reduction, even means that a stated band leaves unused."""
    class Malformed(_JaxProbe):
        name = "malformed"

        def field_pilot(self, request, r0, keys, checks):
            n, w, c = len(r0), request.n_windows, len(request.field.channel_names)
            arrays = dict(fs=np.zeros((n, request.n_blocks + 1, c), np.float32),
                          qv_d=np.zeros((n, w, c), np.float64),
                          chk=np.zeros((n, w, 2, len(checks), c), np.float64))
            arr = arrays[key]
            arrays[key] = arr[..., :-1] if defect == "shape" else arr.astype(np.int32)
            return arrays

    be = Malformed()
    with pytest.raises(ValueError, match=rf"backend 'malformed' field_pilot returned '{key}'.*expected shape .* dtype"):
        _walk_geometry(16, D, d.Sphere(R), 4e-3, 0.5e-3, backend=be, require_gpu=False,
                       field_basis=_grid(), K_c3=3, field_read_interval=4e-5)
    assert not be.requests

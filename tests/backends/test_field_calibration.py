"""Native field pilots and scoped reuse on an actual CUDA device, with CPU JAX host utilities."""
import time
import numpy as np
import pytest

pytest.importorskip("dmipy_sim_cuda")
from dmipy_sim_cuda.backend import CudaBackend
from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
from dmipy_sim.engine.backends import parity_cases
from dmipy_sim.fields.strand_field import StrandFieldBasis
from dmipy_sim.engine.calibration import FieldCalibration, REQUIRED_TERMS, SIGNAL_METRIC
from dmipy_sim.math.accuracy import ErrorBound, assess_accuracy
from dmipy_sim.math.step_law import assess_step_certificate


class TrackedBackend(CudaBackend):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.pilot_reads = None

    def walk_batch(self, request, r0, keys, **kw):
        self.requests.append(request)
        result = super().walk_batch(request, r0, keys, **kw)
        if request.K_c3 == 0 and request.raw_subset:
            self.pilot_reads = np.array(result.saves.field_samples, copy=True)
        return result


@pytest.fixture
def backend():
    be = TrackedBackend()
    try:
        be.device_info()
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"native CUDA device unavailable: {exc}")
    return be


@pytest.mark.parametrize("case", ["strands_extra", "strands_intra"])
def test_native_pilot_uses_native_reads_and_reuse_skips_all_pilot_work(backend, monkeypatch, case):
    geometry = parity_cases()[case]()
    geometry.count_walls = True
    basis = StrandFieldBasis(geometry.centerlines, .7*geometry.radii, geometry.radii,
                             cutoff_m=12e-6, domain=geometry.box)
    def forbidden(*args, **kwargs):
        raise AssertionError("native calibration invoked the JAX field reader")
    monkeypatch.setattr(basis, "channels_at_device", lambda: forbidden)
    query = dict(identity="synthetic-workflow-only", metric=SIGNAL_METRIC, target=.1, confidence=.99)
    kwargs = dict(T_max=.002, block_h=.0005, K=8, K_c3=3, field_basis=basis, query_scope=query,
                  backend=backend, require_gpu=True, walker_batch_size=256, field_list_k=16, calibration_floor=.001,
                  envelope=dict(B0_list=[.1]))
    start = time.monotonic()
    fresh = simulate_trajectories_adaptive(128, 2e-9, geometry, seed=3, **kwargs)
    cold = time.monotonic()-start
    assert fresh.stepping["field_pilot_backend"] == "cuda"
    assert len(backend.requests) == 2
    probe, production = backend.requests
    assert probe.block_h*probe.field.reuse_intervals == pytest.approx(
        production.block_h*production.field.reuse_intervals)
    reads = backend.pilot_reads - np.asarray(basis.mean, np.float32)[None,None,:]
    expected = np.diff(reads.astype(np.float64),axis=1)
    expected = np.sum(expected**2,axis=1).mean(axis=0)
    pool = next(iter(fresh.block["field_read"]["pools"].values()))
    np.testing.assert_allclose(pool["qv_increments"], expected, rtol=2e-6, atol=1e-22)
    # Synthetic evidence tests reuse mechanics only; it does not admit this fixture as a physical calibration.
    term = lambda:ErrorBound(.001, SIGNAL_METRIC, 1.0, "synthetic workflow fixture only")
    accuracy = assess_accuracy(target=.1, metric=SIGNAL_METRIC, confidence=.99, statistical=term(),
                               numerical={name:term() for name in REQUIRED_TERMS}, required=REQUIRED_TERMS)
    step = assess_step_certificate(dict(eps_measured=.0001, eps_se=.00001, floor=.0001,
                                       eps_declared=.001, resolves=True))
    calibration = FieldCalibration.from_walk(fresh, accuracy=accuracy, step=step)
    print(f"measured read charge={fresh.block['field_read']['term_pilot']:.6g}")
    with pytest.raises(ValueError, match="refine the walk before production"):
        simulate_trajectories_adaptive(128, 2e-9, geometry, seed=3,
            **dict(kwargs, calibration_floor=1e-30))
    backend.requests.clear()
    start = time.monotonic()
    reused = simulate_trajectories_adaptive(160, 2e-9, geometry, seed=19, field_calibration=calibration, **kwargs)
    warm = time.monotonic()-start
    assert len(backend.requests) == 1 and backend.requests[0].raw_subset == 0
    assert reused.stepping["field_pilot_backend"] is None
    assert reused.block["field_read"]["reused"] and reused.block["field_read"]["pilot_seconds"] == 0
    print(f"\n{case}: fresh={cold:.4f}s, reused={warm:.4f}s, native pilot={fresh.block['field_read']['pilot_seconds']:.4f}s")

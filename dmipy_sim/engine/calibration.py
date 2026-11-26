"""Inspectable field calibration whose reuse is conditional on scoped accuracy evidence."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from ..math.accuracy import ErrorBound, assess_accuracy
from ..math.step_law import assess_step_certificate, STEP_METRIC

SIGNAL_METRIC = "max_abs_complex_signal_error"
REQUIRED_TERMS = ("trajectory", "spatial_field", "field_read", "field_band", "representation")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    """Identity of JSON-ready physical/query metadata, independent of walker count and seed."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_fingerprint(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical(list(array.shape)).encode())
    # Slabs avoid a second full copy of memory-mapped input fields.
    if array.ndim:
        for slab in array:
            digest.update(np.ascontiguousarray(slab).tobytes())
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()



@lru_cache(maxsize=1)
def code_fingerprint():
    """Public simulator source identity for an immutable worker process."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace(chr(92), "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _ScopedCalibration:
    document: str

    def _checked_record(self, kind):
        data = self.record
        digest = data.pop("digest", None)
        if digest != fingerprint(data):
            raise ValueError("field calibration checksum does not match its evidence")
        if data.get("kind") != kind:
            raise ValueError(f"not a {kind.replace('-', ' ')} record")
        return data

    @property
    def record(self):
        return json.loads(self.document)

    @classmethod
    def _create(cls, **data):
        data["digest"] = fingerprint(data)
        return cls(canonical(data))

    def _record_for_scope(self, scope):
        record = self.record
        expected = record["scope"]
        if canonical(expected) != canonical(scope):
            changed = sorted(key for key in set(scope) | set(expected)
                             if key not in expected or key not in scope or canonical(scope[key]) != canonical(expected[key]))
            raise ValueError(f"incompatible field calibration scope: {', '.join(changed)}")
        return record

    def save(self, path):
        Path(path).write_text(self.document + "\n")

    @classmethod
    def load(cls, path):
        return cls(Path(path).read_text())


@dataclass(frozen=True)
class FieldCalibration(_ScopedCalibration):
    """A JSON snapshot of physics, numerics, query, backend and evidence; no mutable shared dictionaries."""

    def __post_init__(self):
        data = self._checked_record("field-calibration")
        scope = data.get("scope", {})
        if set(scope) != {"physics", "numerics", "query", "backend", "code"} or not scope["query"]:
            raise ValueError("field calibration needs physical, numerical, query, backend and code scope")
        accuracy = data["accuracy"]
        request = scope["query"].get("request", {}) if isinstance(scope["query"], dict) else {}
        if (request.get("metric") != SIGNAL_METRIC or
                request.get("target") != accuracy.get("target") or
                request.get("confidence") != accuracy.get("confidence")):
            raise ValueError("calibration evidence must match its requested answer accuracy")
        if accuracy.get("metric") != SIGNAL_METRIC:
            raise ValueError("field calibration requires an absolute complex-signal accuracy budget")
        terms = dict(accuracy["terms"])
        statistical = ErrorBound(metric=SIGNAL_METRIC, **terms.pop("statistical"))
        checked = assess_accuracy(target=accuracy["target"], metric=SIGNAL_METRIC,
                                  confidence=accuracy["confidence"], statistical=statistical,
                                  numerical={name:ErrorBound(metric=SIGNAL_METRIC, **term) for name,term in terms.items()},
                                  required=REQUIRED_TERMS)
        if checked["status"] != "passed":
            raise ValueError(f"field calibration accuracy is {checked['status']}; required evidence cannot be skipped")
        step = data["step"]
        if "z" not in step:
            raise ValueError("field calibration's step assessment states no z (its confidence multiplier); assess the step "
                             "with math.step_law.assess_step_certificate, which records it")
        reassessed = assess_step_certificate(step, z=step["z"])
        if (step.get("status") != "passed" or reassessed["status"] != "passed" or
                step.get("metric") != STEP_METRIC or
                reassessed["confidence"] < accuracy["confidence"]):
            raise ValueError("field calibration needs a resolved, passing step-size assessment at its confidence")
        read, band = data["read"], data["band"]
        if read.get("budget_met") is False or read.get("calibration_resolved") is False:
            raise ValueError("field-read calibration is unresolved or over budget")
        if (read.get("term_pilot") is not None and
                float(read["term_pilot"]) > float(terms["field_read"]["upper"])):
            raise ValueError("field-read evidence is below the measured calibration charge")
        constants = np.asarray(read.get("constants"), np.float64)
        if (not np.isfinite(float(read["dt_f"])) or read["dt_f"] <= 0 or int(read["n_read"]) < 1 or
                constants.ndim != 1 or not constants.size or not np.isfinite(constants).all() or (constants < 0).any()):
            raise ValueError("field calibration needs finite measured or bounded field-read settings")
        if not 1 <= int(band["K"]) <= int(scope["numerics"]["blocks_per_window"]):
            raise ValueError("field calibration band is outside its sampled window")

    @classmethod
    def create(cls, *, scope, read, band, accuracy, step):
        return cls._create(kind="field-calibration", scope=scope, read=read, band=band, accuracy=accuracy, step=step)

    @classmethod
    def from_walk(cls, walk, *, accuracy, step, pool=None):
        block = walk.block or {}
        if "calibration_pools" in block:
            if pool is None:
                raise ValueError("a joined walk needs the pool name for calibration")
            try:
                block = block["calibration_pools"][pool]
            except KeyError as exc:
                raise ValueError(f"walk has no calibration provenance for pool {pool!r}") from exc
        elif pool is not None:
            raise ValueError("pool selection is for a joined walk")
        if "calibration_scope" not in block:
            raise ValueError("walk has no scoped calibration provenance; generate it with a query scope")
        return cls.create(scope=block["calibration_scope"], read=block["field_read"],
                          band=block["field_band"], accuracy=accuracy, step=step)

    def for_scope(self, scope):
        record = self._record_for_scope(scope)
        return record["read"], record["band"]


@dataclass(frozen=True)
class FieldReadCalibration(_ScopedCalibration):
    """Fixed-step field-read constants and standard errors by channel at one check rung.

    Time entries are in seconds, field strength in tesla, and susceptibility and constants are dimensionless.
    A channel with zero pilot quadratic variation uses the trapezoid constant and has no standard error.
    A one-step block has no coarser check rung and uses the trapezoid constant for all channels.
    """

    def __post_init__(self):
        from .accumulate import READ_CONSTANT
        data = self._checked_record("field-read-calibration")
        scope, evidence = data["scope"], data["evidence"]
        n_channels = len(scope["channels"])
        constants = data["constants"]
        errors = evidence["standard_error"]
        if not n_channels or len(constants) != n_channels or len(errors) != n_channels:
            raise ValueError("field-read calibration needs a finite nonnegative constant and a standard error entry per channel")
        for constant, error in zip(constants, errors):
            if type(constant) not in (int, float) or not np.isfinite(constant) or constant < 0:
                raise ValueError("field-read calibration constants must be finite nonnegative numbers")
            if ((error is None and constant != READ_CONSTANT) or
                    (error is not None and (type(error) not in (int, float) or not np.isfinite(error) or error < 0))):
                raise ValueError("field-read calibration needs a finite nonnegative standard error or the trapezoid constant")
        checks = scope["checks"]
        rung = next((d for d in checks if d >= scope["n_read"]), None)
        if data["rung"] != rung or (rung is None and any(error is not None for error in errors)):
            raise ValueError("field-read calibration rung must be the check at or above n_read")
        if type(evidence["pilot_walkers"]) is not int or evidence["pilot_walkers"] < 2 or type(evidence["seed"]) is not int:
            raise ValueError("field-read calibration requires at least two pilot walkers and an integer seed")

    @classmethod
    def create(cls, *, scope, rung, constants, standard_error, pilot_walkers, seed):
        return cls._create(kind="field-read-calibration", scope=scope, rung=rung, constants=constants,
                           evidence=dict(standard_error=standard_error, pilot_walkers=pilot_walkers, seed=seed))

    def for_scope(self, scope):
        """The stated record when every entry matches the resolved walk scope."""
        return self._record_for_scope(scope)


def fixed_step_engine(backend, precision):
    """Engine and declared precision used to measure or apply a fixed-step calibration."""
    if backend is not None:
        precision = getattr(backend, "precision", None)
        if precision not in ("float32", "float64"):
            raise ValueError(f"field_calibration: backend {backend.name!r} must declare its precision "
                             "as 'float32' or 'float64'")
        build = getattr(backend, "build", None)
        return dict(name=str(backend.name), precision=precision, build=build() if callable(build) else None)
    import jax
    import jaxlib
    device = jax.config.jax_default_device
    return dict(name="jax", precision=precision, version=jax.__version__, jaxlib=jaxlib.__version__,
                platform=jax.default_backend() if device is None else device.platform, x64=bool(jax.config.jax_enable_x64))


def fixed_step_constants(check, step_s, rung):
    """Per-channel ratio of mean squared trapezoid error to mean interval-scaled QV.

    The input has axes (walker, window, error or QV, channel). Average within each walker before
    estimating standard errors by the delta method across independent walkers, retaining the
    covariance of error squared and QV. Zero mean QV has no ratio estimate.
    """
    from .accumulate import READ_CONSTANT, measured_read_constants
    check = np.asarray(check, np.float64)
    errors_sq, variations = check[:, :, 0] ** 2, check[:, :, 1]
    if not np.isfinite(check).all() or (variations < 0).any():
        raise ValueError("field-read calibration needs finite pilot errors and finite nonnegative quadratic variation")
    error_sq = errors_sq.mean(axis=1)
    scaled_qv = (rung * step_s) ** 2 * variations.mean(axis=1)
    qv = scaled_qv.mean(axis=0)
    constants = measured_read_constants(errors_sq.mean(axis=(0, 1))[None],
                                        variations.mean(axis=(0, 1))[None], step_s, (rung,))[rung]
    resolved = qv > 0.0
    residual = error_sq[:, resolved] - scaled_qv[:, resolved] * constants[resolved]
    se = residual.std(axis=0, ddof=1) / (np.sqrt(len(check)) * qv[resolved])
    errors = [None] * check.shape[-1]
    for channel, value in zip(np.flatnonzero(resolved), se):
        errors[channel] = float(value)
    return np.where(resolved, constants, READ_CONSTANT).tolist(), errors


def adaptive_scope(*, geometry, diffusivity, field_basis, spec, plan, T, K, m, h, envelope,
                   query, backend, reuse, floor, windows=1):
    """Exact immutable-input scope for the adaptive strand calibration; counts and seeds are absent."""
    if (not isinstance(query, dict) or not query.get("identity") or query.get("metric") != SIGNAL_METRIC or
            not np.isfinite(query.get("target", np.nan)) or query["target"] <= 0 or
            "confidence" not in query or not 0 < query["confidence"] < 1):
        raise ValueError("query scope states protocol/tissue identity, absolute signal target and confidence")
    import jax
    physics = dict(geometry=type(geometry).__qualname__,
                   centerlines=[array_fingerprint(c) for c in geometry.centerlines],
                   radii=array_fingerprint(geometry.radii), cell_size=float(geometry.cell_size),
                   box=None if geometry.box is None else [array_fingerprint(c) for c in geometry.box],
                   box_reflect=bool(geometry.box_reflect), interior=bool(geometry.interior),
                   diffusivity=float(diffusivity),
                   field_centerlines=[array_fingerprint(c) for c in field_basis.centerlines],
                   inner=array_fingerprint(field_basis.inner_radii), outer=array_fingerprint(field_basis.outer_radii),
                   field=field_basis.meta, spec=None if spec is None else spec.to_dict())
    far = field_basis.far
    if far is not None and np.asarray(far.values).flags.writeable:
        raise ValueError("scoped field calibration requires an immutable precomputed far field")
    backend_id = (dict(name="jax", version=jax.__version__, platform=jax.devices()[0].platform,
                       x64=bool(jax.config.jax_enable_x64)) if backend is None else
                  dict(name=backend.name, build=backend.build()))
    if backend is not None:
        library = getattr(getattr(backend, "lib", None), "_name", None)
        if library is not None:
            digest = hashlib.sha256()
            with open(library, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            backend_id["library_sha256"] = digest.hexdigest()
    return dict(physics=fingerprint(physics),
                numerics=dict(plan=plan._asdict(), T=float(T), K=int(K), m=int(m), h=float(h),
                              blocks_per_window=int(round(T / h)), windows=int(windows), reuse=int(reuse), floor=float(floor)),
                query=dict(request=query, envelope=envelope), backend=backend_id, code=code_fingerprint())

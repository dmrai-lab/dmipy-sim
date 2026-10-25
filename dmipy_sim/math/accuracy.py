"""An answer's error budget: numerical bounds and statistical uncertainty in one declared metric."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ErrorBound:
    """A bound in a named metric at a stated confidence; None records missing or unresolved evidence."""
    upper: float | None
    metric: str
    confidence: float
    method: str

    def __post_init__(self):
        if not self.metric or not self.method:
            raise ValueError("an error bound states its metric and method")
        if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("confidence must be finite and in (0, 1]")
        if self.upper is not None and (not math.isfinite(self.upper) or self.upper < 0):
            raise ValueError("an error bound is finite and nonnegative, or None when unresolved")


def assess_accuracy(*, target, metric, confidence, statistical, numerical, required):
    """Sum bounds in the requested answer's metric; combine confidence by the union bound.

    Numerical bias is never scaled by walker count. Statistical uncertainty is supplied separately by its
    estimator. Evidence at another metric (for example relative diffusion-spectrum error against an absolute
    complex-signal target) cannot certify this answer. Missing or unresolved terms cannot yield a pass.
    The caller declares the physical terms required for the enabled model and the validity domain of each
    bound remains the responsibility of its producer; this arithmetic is not a new physics certificate.
    """
    target, confidence = float(target), float(confidence)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("the answer's target must be finite and positive")
    if not metric or not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("state the answer's metric and a confidence in (0, 1]")
    required = tuple(required)
    if not required or len(set(required)) != len(required):
        raise ValueError("declare distinct required numerical error terms")
    terms = dict(numerical)
    if any(not isinstance(name, str) or not name for name in terms):
        raise ValueError("numerical error terms have nonempty names")
    if "statistical" in terms:
        raise ValueError("statistical uncertainty is supplied separately")
    all_bounds = dict(terms, statistical=statistical)
    for name, bound in all_bounds.items():
        if not isinstance(bound, ErrorBound) or bound.metric != metric:
            raise ValueError(f"{name}: error evidence must use the requested metric {metric!r}")
    unresolved = sorted(set(required) - set(terms) |
                        {name for name, bound in all_bounds.items() if bound.upper is None})
    known_numerical = math.fsum(bound.upper for bound in terms.values() if bound.upper is not None)
    known_total = known_numerical + (statistical.upper or 0.0)
    joint_confidence = max(0.0, 1.0 - sum(1.0 - bound.confidence for bound in all_bounds.values()))
    status = ("over_budget" if known_total > target else
              "unresolved" if unresolved or joint_confidence < confidence else "passed")
    return dict(status=status, metric=metric, target=target, confidence=confidence,
                joint_confidence=joint_confidence, unresolved=unresolved,
                numerical_upper=None if unresolved else known_numerical,
                statistical_upper=statistical.upper,
                total_upper=None if unresolved else known_total,
                terms={name:dict(upper=bound.upper, confidence=bound.confidence, method=bound.method)
                       for name,bound in all_bounds.items()})

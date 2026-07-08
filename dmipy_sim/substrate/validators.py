"""Lightweight physics-parameter validators (uniform, early, clear messages).

dmipy-sim is the forward truth, so a physically-impossible parameter should fail
*at construction* with a clear message -- not silently produce a wrong signal or
raise cryptically deep inside a Monte-Carlo run.  These helpers generalise the
patterns already used by ``resolve_slew`` and ``check_acquisition_scheme`` so the
:class:`~dmipy_sim.substrate.Substrate` (and its dmipy-fit subclass) can audit
every field in one ``__post_init__``.

Each raises ``ValueError`` naming the parameter and the value.  ``None`` always
fails (it is never a valid physical value here).
"""
from __future__ import annotations

import math


def _num(name, v):
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a real number, got {v!r}.")
    if math.isnan(v):
        raise ValueError(f"{name} must not be NaN.")
    return float(v)


def positive(name, v):
    """Require v > 0 (strictly positive: diffusivities, T1/T2, B0, radii scale)."""
    v = _num(name, v)
    if not v > 0.0:
        raise ValueError(f"{name} must be > 0 (got {v!r}).")
    return v


def non_negative(name, v):
    """Require v >= 0 (surface relaxivity, permeability, fractions, M0)."""
    v = _num(name, v)
    if not v >= 0.0:
        raise ValueError(f"{name} must be >= 0 (got {v!r}).")
    return v


def non_positive(name, v):
    """Require v <= 0 (myelin susceptibility is diamagnetic vs water)."""
    v = _num(name, v)
    if not v <= 0.0:
        raise ValueError(
            f"{name} must be <= 0: myelin is diamagnetic relative to water, so "
            f"the susceptibility difference is non-positive (got {v!r}). The "
            f"field formula uses its magnitude; a positive value is a sign error.")
    return v


def in_open_interval(name, v, lo, hi):
    """Require lo < v < hi (e.g. g_ratio in (0, 1))."""
    v = _num(name, v)
    if not (lo < v < hi):
        raise ValueError(f"{name} must be in ({lo}, {hi}) (got {v!r}).")
    return v


def in_closed_interval(name, v, lo, hi):
    """Require lo <= v <= hi (e.g. volume fractions in [0, 1])."""
    v = _num(name, v)
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}] (got {v!r}).")
    return v

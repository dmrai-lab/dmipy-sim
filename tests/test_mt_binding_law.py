"""The MT walk's binding law (dmrai-lab/dmipy-sim#921): the binding clock on the boundary local time, the dwell that
carries an encounter's geminate rebinds, and the step law's binding term. Pure numpy and a few JAX calls: fast.

* Binding is a Poisson clock on the local time, so an encounter split in two binds as the whole does.
* A walker released after a bind resumes the unspent half of its encounter on average and retries there with ``1 -
  exp(-x / 2)``; the dwells of a bind and its rebinds sum to one exponential of mean ``tau exp(x / 2)``, the walk's
  dwell (``dwell_scale``).
* The step law's binding term holds the binding within its tolerance: ``binding_step`` inverts ``binding_error``, and it
  asks for sub-steps only where the binding itself is resolved at the step.
"""
import numpy as np
import pytest


def test_binding_is_a_clock_on_the_local_time():
    import jax.numpy as jnp
    from dmipy_sim.geometry._boundary import bind_probability
    kod = jnp.float32(3.0e5)                                 # kappa_MT / D, 1/m
    l1, l2 = jnp.float32(1.2e-7), jnp.float32(0.7e-7)
    p1, p2, p12 = (float(bind_probability(kod, l)) for l in (l1, l2, l1 + l2))
    assert 1.0 - (1.0 - p1) * (1.0 - p2) == pytest.approx(p12, rel=1e-5)
    assert p1 == pytest.approx(float(kod * l1), rel=0.03)     # the linear law to first order


def test_the_dwell_carries_the_geminate_rebinds():
    """Explicit retries at each release, each with ``1 - exp(-x / 2)``, a fresh exponential dwell when it binds: the
    total bound time is exponential with the walk's mean ``tau dwell_scale(x)``."""
    import jax.numpy as jnp
    from dmipy_sim.geometry._boundary import dwell_scale
    rng = np.random.default_rng(3)
    tau, x, n = 1.0, 0.8, 200_000
    p_retry = -np.expm1(-x / 2.0)
    total = rng.exponential(tau, n)
    going = np.ones(n, bool)
    while going.any():
        going &= rng.random(n) < p_retry
        total[going] += rng.exponential(tau, going.sum())
    mean = tau * float(dwell_scale(jnp.float32(1.0), jnp.float32(x)))
    assert mean == pytest.approx(tau * np.exp(x / 2.0), rel=1e-6)
    assert abs(total.mean() - mean) < 4.0 * mean / np.sqrt(n)
    # exponential: the survival at the mean and at twice it
    for k in (1.0, 2.0):
        s = (total > k * mean).mean()
        assert abs(s - np.exp(-k)) < 4.0 * np.sqrt(np.exp(-k) / n)


def test_the_step_law_holds_the_binding_within_its_tolerance():
    from dmipy_sim.engine.physics import (MT_COEFFICIENT, STEP_TOLERANCE, binding_error, binding_length, binding_step,
                                          mt_sub_steps, resolve_sub_steps)
    from dmipy_sim.geometry import PermeableSlab1D, Sphere
    D, R = 2e-9, 5e-6
    slab = PermeableSlab1D(length=2 * R, permeability=1e-4)
    assert binding_length(slab) == pytest.approx(R)          # the layer's width: pi / sqrt(pore_rate)
    for kappa_MT in (1.5e-5, 2e-4, 2e-3):
        dt = binding_step(D, kappa_MT, STEP_TOLERANCE, R)
        assert binding_error(D, kappa_MT, dt, R) == pytest.approx(STEP_TOLERANCE, rel=1e-9)
    assert binding_error(D, 0.0, 1e-5, R) == pytest.approx(MT_COEFFICIENT * 6 * D * 1e-5 / R ** 2)   # the wall's term
    # at a qMT reactivity the binding asks no more than the walk's other rules; at a reactivity resolved within the step
    # (kappa_MT h / D = 0.2 at h = 0.09 R) it asks for sub-steps
    sph = Sphere(radius=2e-6)
    assert mt_sub_steps(sph, D, 1e-4, 5e-3, kappa_MT=1.5e-5) == mt_sub_steps(sph, D, 1e-4, 5e-3)
    h = 0.09 * R
    dt = h * h / (6 * D)
    k_stress = 0.2 * D / h
    assert mt_sub_steps(slab, D, dt, 33 * dt, kappa_MT=k_stress) > mt_sub_steps(slab, D, dt, 33 * dt)
    n = resolve_sub_steps(slab, D, dt, mt_dwell_time=33 * dt, mt_kappa=k_stress)
    assert binding_error(D, k_stress, dt / n, R) <= STEP_TOLERANCE

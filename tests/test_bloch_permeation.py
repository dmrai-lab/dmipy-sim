"""Membrane permeability in the vector-Bloch forward (`simulate_bloch`).

The scalar `core.simulate` walk has always supported the Powles crossing; these check that
`simulate_bloch` now does too — a diffusion-weighted spin echo on a restricted cylinder attenuates
MORE as the wall becomes permeable (walkers escape the restriction), and approaches the free-
diffusion signal in the high-κ limit. This is what lets the Bloch engine model exchange during a
longitudinal-storage mixing time (FEXI).
"""
import numpy as np
import pytest

from dmipy_sim import Cylinder, FreeDiffusion
from dmipy_sim.pulse_sequence import spin_echo, run_bloch_sequence

GAMMA = 2.675e8


def _dw_spin_echo(g=0.20, TE=30e-3, dt=1e-4, delta=5e-3):
    """PGSE (physical same-sign lobes; the 180 folds the sign) perpendicular to the cylinder."""
    seq = spin_echo(TE, dt)
    n_t = seq.n_t
    nd = int(round(delta / dt))
    i1 = int(round(4e-3 / dt))
    i2 = int(round((TE / 2 + 1e-3) / dt))
    G = np.zeros((1, n_t, 3))
    G[0, i1:i1 + nd, 0] = g
    G[0, i2:i2 + nd, 0] = g
    seq.G[:] = G
    sgn = np.where(np.arange(n_t) < int(round(TE / 2 / dt)), 1.0, -1.0)
    q = GAMMA * np.cumsum(sgn * G[0, :, 0]) * dt
    return seq, float(np.sum(q ** 2) * dt)


def _S(kappa, seq, D=2e-9, n=4000, seed=0):
    geom = Cylinder(radius=3e-6, orientation=(0, 0, 1),
                    permeability=(None if kappa == 0 else kappa))
    return abs(complex(run_bloch_sequence(seq, n, D, geom, seed=seed, require_gpu=False)[0]))


def test_permeability_increases_attenuation_monotonically():
    seq, b = _dw_spin_echo()
    assert b > 3e8                                   # a real diffusion weighting
    S = [_S(k, seq) for k in (0.0, 3e-5, 3e-3)]
    # more permeable wall -> walkers leave the restriction -> more signal loss
    assert S[0] > S[1] > S[2]
    assert S[0] - S[2] > 0.1                          # a clear effect, not noise


def test_high_permeability_approaches_free_diffusion():
    seq, b = _dw_spin_echo()
    S_free = abs(complex(run_bloch_sequence(seq, 4000, 2e-9, FreeDiffusion(), seed=0,
                                            require_gpu=False)[0]))
    S_perm = _S(1e-2, seq)                            # nearly transparent wall
    assert abs(S_perm - S_free) < 0.12               # high-κ limit ≈ free diffusion


def test_impermeable_bloch_walk_is_unchanged():
    """permeability=None must reproduce the plain reflecting walk (no regression)."""
    seq, _ = _dw_spin_echo()
    a = _S(0.0, seq, seed=7)
    b = _S(0.0, seq, seed=7)
    assert a == pytest.approx(b, abs=1e-9)           # deterministic, reflecting path

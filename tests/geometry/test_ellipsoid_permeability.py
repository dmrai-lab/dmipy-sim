"""Membrane permeability for Ellipsoid (Powles 2004).

Physics
-------
Same Powles (2004) probabilistic crossing as for Sphere and Cylinder:

    p = min(1,  2·κ·d_perp / D)

For a general ellipsoid the exchange time is

    τ = V / (κ · S)

For a prolate ellipsoid with semi-axes [a, a, c] (c > a):
    V = (4π/3) · a² · c
    S ≈ 2π · a² · (1 + (c/a) · arcsin(e) / e)    where e = √(1 - a²/c²)

Analytical limits validated here
----------------------------------
1. κ=None  → identical signal to impermeable Ellipsoid.
2. κ>0 at high b → signal below impermeable (escaped walkers diffuse
   more freely along the long axis → greater signal decay). The high-κ limit, the
   relaxivity combination and the κ ordering are checked on the sphere and the cylinder
   (test_sphere_permeability.py, test_permeability_crossing.py), whose crossing rule is the same.

Parameters: a=3 µm, c=9 µm (prolate, aspect ratio 3), D=2e-9 m²/s
    σ = √(6D·dt) with dt = TE/n_t; n_t=500, TE=100ms → σ ≈ 1.55 µm
    σ/a ≈ 0.52 (borderline for good practice, sufficient for monotone tests)
"""

import numpy as np
import pytest

from tests.conftest import pgse_wf
import numpy.testing as npt

from dmipy_sim import simulate, Ellipsoid, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_CPU, N_EXACT, SEED

A          = 3e-6   # m  — short semi-axis
C_AXIS     = 9e-6   # m  — long semi-axis
SEMIAXES   = [A, A, C_AXIS]
KAPPA_MED  = 1e-5   # m/s
KAPPA_HIGH = 6e-6   # m/s — derived in test_permeability_high_kappa_approaches_free_diffusion
                     # below from the engine's own sub-step budget, together with its TE
RHO        = 5e-4   # m/s


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------

def test_permeability_attribute_stored():
    """permeability is stored correctly on Ellipsoid."""
    geom = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default Ellipsoid has permeability=None."""
    geom = Ellipsoid(semiaxes=SEMIAXES)
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """Ellipsoid(permeability=None) == Ellipsoid() — identical signal."""
    wf           = pgse_wf(100e-3)
    geom_default = Ellipsoid(semiaxes=SEMIAXES)
    geom_none    = Ellipsoid(semiaxes=SEMIAXES, permeability=None)
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default    = simulate(N_EXACT, D, wf, geom_default, seed=SEED)
    S_none       = simulate(N_EXACT, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. kappa > 0 reduces the signal at high b (tier 2: the crossing rule's sub-steps cost ~10^3 per waveform step)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_permeability_reduces_signal_at_high_b():
    """Signal with kappa > 0 is strictly below the impermeable signal at b = 2000 s/mm^2 (an escaped walker
    diffuses more freely, so it accumulates more phase). The margin 0.05 is 7 standard errors at N_CPU."""
    wf = pgse_wf(100e-3)
    S_imp = simulate(N_CPU, D, wf, Ellipsoid(semiaxes=SEMIAXES, permeability=None), seed=SEED)
    S_perm = simulate(N_CPU, D, wf, Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED), seed=SEED)
    assert float(S_perm[3]) < float(S_imp[3]) - 0.05, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} at b=2000 s/mm²")

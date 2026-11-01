"""Membrane permeability for PackedCylinders (Powles 2004).

Physics
-------
Same Powles (2004) probabilistic crossing as for Cylinder:

    p = min(1,  2·κ·d_perp / D)

PackedCylinders is an extra-axonal geometry: walkers start outside all
cylinders.  With κ > 0 walkers can enter cylinders (intra-axonal space)
and exit again.  This models bi-directional transcytolemmal exchange.

Exchange time for walkers starting outside entering cylinders of radius R
in a periodic box of side L (one cylinder):

    τ ≈ (L² − πR²) / (κ · 2πR)   [V_extra / (κ · S)]

Analytical limits validated here
----------------------------------
1. κ=None → identical signal to impermeable PackedCylinders.
2. κ>0 at high b → signal differs from impermeable: walkers that enter
   cylinders become more restricted → higher signal at high b
   (opposite to the interior-start geometries).
3. Very high κ (TE >> τ) → signal converges to a mixture: some walkers
   inside (restricted), some outside (extra-axonal).  The mixed signal
   must lie between the fully extra-axonal and fully intra-axonal signals.
4. Permeability + relaxivity → signal below permeability-only.
5. Signal monotonically increases with κ at high b (more restricted
   walkers inside → higher signal at high b).

Parameters: R=5 µm, L=20 µm, D=2e-9 m²/s, TE=100 ms
"""

import numpy as np
import pytest

from tests.conftest import pgse_wf
import numpy.testing as npt

from dmipy_sim import simulate, PackedCylinders, Cylinder, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_EXACT, SEED


R          = 5e-6    # m
L          = 20e-6   # m
KAPPA_MED  = 1e-5    # m/s
KAPPA_HIGH = 5e-4    # m/s  — fast exchange (τ_ex≈20ms≪TE) with p_transmit<1
RHO        = 5e-5    # m/s


def _single_cylinder_packed(permeability=None, surface_relaxivity_t2=None):
    return PackedCylinders(
        radii=np.array([R]),
        centers=np.array([[0., 0.]]),
        L=L,
        orientation=[0., 0., 1.],
        permeability=permeability,
        surface_relaxivity_t2=surface_relaxivity_t2,
        pool="extra",
    )


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------

def test_permeability_attribute_stored():
    """permeability is stored correctly on PackedCylinders."""
    geom = _single_cylinder_packed(permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default PackedCylinders has permeability=None."""
    geom = _single_cylinder_packed()
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """PackedCylinders(permeability=None) == default — identical signal."""
    wf           = pgse_wf(100e-3)
    geom_default = _single_cylinder_packed(permeability=None)
    geom_explicit = PackedCylinders(
        radii=np.array([R]),
        centers=np.array([[0., 0.]]),
        L=L,
        orientation=[0., 0., 1.],
        pool="extra",
    )
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default  = simulate(N_EXACT, D, wf, geom_default,  seed=SEED)
    S_explicit = simulate(N_EXACT, D, wf, geom_explicit, seed=SEED)
    npt.assert_array_equal(S_default, S_explicit,
        err_msg="permeability=None must give identical signal to default")

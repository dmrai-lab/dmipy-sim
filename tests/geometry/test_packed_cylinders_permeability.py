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

from tests.conftest import pgse_wf
import numpy.testing as npt

from dmipy_sim import simulate, PackedCylinders, Cylinder, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_CPU, N_EXACT, SEED, mc_atol

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
    )
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default  = simulate(N_EXACT, D, wf, geom_default,  seed=SEED)
    S_explicit = simulate(N_EXACT, D, wf, geom_explicit, seed=SEED)
    npt.assert_array_equal(S_default, S_explicit,
        err_msg="permeability=None must give identical signal to default")


_WALKS = {}


def _signal(kappa, rho=None):
    """The signal of one permeable walk, computed once per (kappa, rho) and shared by the tests below (the increases,
    relaxivity and monotone tests all read the same kappa = None and kappa = KAPPA_MED walks)."""
    if (kappa, rho) not in _WALKS:
        geom = _single_cylinder_packed(permeability=kappa, surface_relaxivity_t2=rho)
        _WALKS[(kappa, rho)] = np.asarray(simulate(N_CPU, D, pgse_wf(100e-3), geom, seed=SEED))
    return _WALKS[(kappa, rho)]


# ---------------------------------------------------------------------------
# 3. κ>0 changes signal at high b: walkers enter cylinders → more restriction
# ---------------------------------------------------------------------------

def test_permeability_increases_signal_at_high_b():
    """Signal with kappa > 0 is above the impermeable signal at b = 2000 s/mm^2: extra-axonal walkers that enter the
    cylinders are more restricted. Both walks share the seed, so they share their trajectories until the first
    transmission and the difference carries far less noise than two independent walks."""
    S_imp, S_perm = _signal(None), _signal(KAPPA_MED)
    assert float(S_perm[3]) > float(S_imp[3]) + 0.005, (
        f"S_perm={S_perm[3]:.4f} must be above S_imp={S_imp[3]:.4f} "
        f"at b=2000 s/mm² (permeable walkers are more restricted)")


# ---------------------------------------------------------------------------
# 4. High κ: signal lies between fully extra-axonal and fully intra-axonal
# ---------------------------------------------------------------------------

def test_permeability_high_kappa_between_compartments():
    """High κ: permeable signal lies between free diffusion and intra-axonal.

    Walkers start OUTSIDE cylinders.  Impermeable walls act as obstacles
    (hindered diffusion, S_extra > S_free).  With high κ, walkers pass
    through walls more easily (reduced obstacle effect), so their signal
    drops toward free diffusion S_free = exp(-b·D).  However, walkers that
    spend time inside the cylinder experience restriction and contribute
    higher signal, so S_mixed > S_free.

    The Karger "between S_extra and S_intra" result applies to equilibrium-
    weighted initialisation; for all-outside starts the lower bound is S_free.
    """
    b_idx = 2   # b = 1000 s/mm²
    b_val = 1000e6   # s/m²
    wf = pgse_wf(100e-3, n_t=1000)

    geom_mixed = _single_cylinder_packed(permeability=KAPPA_HIGH)
    geom_intra = Cylinder(radius=R, orientation=[0., 0., 1.])

    S_mixed = simulate(N_CPU, D, wf, geom_mixed, seed=SEED)
    S_intra = simulate(N_CPU, D, wf, geom_intra, seed=SEED)

    S_free = float(np.exp(-b_val * D))   # analytical free-diffusion limit
    hi = float(S_intra[b_idx])
    sm = float(S_mixed[b_idx])

    assert S_free < sm < hi, (
        f"High-κ mixed signal {sm:.4f} should lie between free diffusion "
        f"{S_free:.4f} and intra-axonal {hi:.4f} at b=1000 s/mm²")


# ---------------------------------------------------------------------------
# 5. Permeability + relaxivity: signal below permeability-only
# ---------------------------------------------------------------------------

def test_permeability_with_relaxivity_reduces_signal():
    """Surface relaxivity on a permeable PackedCylinders reduces the signal at b = 0 (reflected walkers take the
    Brownstein-Tarr weight, transmitted ones do not)."""
    S_perm, S_perm_rho = _signal(KAPPA_MED), _signal(KAPPA_MED, RHO)
    assert float(S_perm_rho[0]) < float(S_perm[0]) - 0.005, (
        f"Relaxivity+permeability {S_perm_rho[0]:.4f} must be below "
        f"permeability-only {S_perm[0]:.4f} at b=0")


# ---------------------------------------------------------------------------
# 6. Signal monotonically increases with κ at high b
# ---------------------------------------------------------------------------

def test_permeability_signal_monotone_in_kappa():
    """Signal at b = 2000 s/mm^2 increases with kappa (more walkers inside the cylinders, more restriction)."""
    kappas = [0.0, 1e-6, 1e-5]
    signals = [float(_signal(k if k > 0 else None)[3]) for k in kappas]
    for i in range(len(signals) - 1):
        assert signals[i] <= signals[i + 1] + 1e-3, (
            f"Signal not monotone: κ={kappas[i]:.0e} → {signals[i]:.4f}, "
            f"κ={kappas[i+1]:.0e} → {signals[i+1]:.4f}")



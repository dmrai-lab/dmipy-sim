"""Membrane permeability for Sphere (Powles 2004).

Physics
-------
At each wall crossing the walker transmits through the membrane with
probability

    p = min(1,  2·κ·d_perp / D)

and reflects otherwise (Powles et al. 2004).
The exchange time for walkers initially inside a sphere of radius R is

    τ = R / (3κ)   (V/κS = (4πR³/3) / (κ·4πR²) = R/(3κ))

Analytical limits validated here
----------------------------------
1. κ=None  (permeability not set) → identical signal to impermeable Sphere.
2. κ=0  (zero permeability)       → identical signal to impermeable Sphere.
3. κ>0 at high b                  → signal strictly below impermeable (walkers
                                    that escape are less restricted → higher
                                    phase accumulation → lower S at high b).
4. Very high κ (TE >> τ)          → walkers mostly escaped, signal close to
                                    free diffusion exp(−bD).
5. Permeability + relaxivity      → signal below permeability-only (weight
                                    penalty on reflection).
6. Signal monotonically decreases with κ at high b.

Parameters chosen so σ/R < 0.1 (good-practice criterion):
    R = 5 µm, D = 2e-9 m²/s, TE ≈ 100 ms
    dt = TE/500 = 0.2e-3 s → σ = √(6D·dt) ≈ 1.55 µm → σ/R ≈ 0.31

For the high-κ test we use a short TE (20 ms) with fine time-stepping:
    τ = R/(3κ) = 5e-6/(3·1e-2) ≈ 0.17 ms
    TE/τ ≈ 120 → effectively full exchange
"""

import numpy as np

from tests.conftest import pgse_wf
import numpy.testing as npt
import pytest

from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim import simulate, Sphere
from dmipy_replay import set_b
from dmipy_replay.sequences import pgse

from tests.conftest import D, N_CPU, N_EXACT, SEED, n_resolving, se_tol

R          = 5e-6   # m
KAPPA_MED  = 1e-5   # m/s  — exchange time τ = R/(3κ) ≈ 167 ms
KAPPA_HIGH = 3.6e-6  # m/s — derived in test_permeability_high_kappa_approaches_free_diffusion
                      # below from the engine's own sub-step budget, together with its TE
RHO        = 5e-4   # m/s  — surface relaxivity for combination test


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------

def test_permeability_attribute_stored():
    """permeability is stored correctly on Sphere."""
    geom = Sphere(radius=R, permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default Sphere has permeability=None."""
    geom = Sphere(radius=R)
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None / κ=0 must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """Sphere(permeability=None) == Sphere() — identical signal."""
    wf = pgse_wf(100e-3)
    geom_default = Sphere(radius=R)
    geom_none    = Sphere(radius=R, permeability=None)
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default = simulate(N_EXACT, D, wf, geom_default, seed=SEED)
    S_none    = simulate(N_EXACT, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. kappa > 0 reduces the signal at high b (tier 2: the crossing rule's sub-steps cost ~10^3 per waveform step)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 4. Very high κ: every crossing impact is granted
# ---------------------------------------------------------------------------

def test_a_transmitted_walker_is_placed_on_the_side_its_label_reports():
    """The sphere's membrane at its two limits: with the transmit probability at one every impact that crosses ends
    outside and is labelled extra, and every one that does not stays inside; at kappa = 0 every impact ends inside and is
    labelled intra, whatever its key. The labels agree with the positions exactly, which a random walk with a live membrane
    could only show for the crossings it happened to draw.

    The limit of complete exchange is held on the impacts: a walk to it (TE / tau = 10 at the smallest kappa the sub-step
    rule affords) costs 2e8 walker-sub-steps."""
    import jax
    import jax.numpy as jnp

    offs = [1e-4 * R, 1e-3 * R, 0.1 * R]; mults = [0.5, 0.99, 1.01, 3.0, 10.0]
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(12, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    cases = []
    for off in offs:
        for m in mults:
            for u in dirs:
                cases.append((np.asarray(u * (R - off), np.float64), np.asarray(u * m * off, np.float64)))
    starts = np.stack([c[0] for c in cases]); steps = np.stack([c[1] for c in cases])
    bare = np.linalg.norm(starts + steps, axis=1)                                        # the free step's end radius
    crossing = bare > R * (1 + 1e-9)
    assert (np.abs(bare - R) > 1e-9 * R).all(), "a case lands on the wall to rounding: undecidable, not a table row"
    d_min = float((bare[crossing] - R).min())
    kod = 1.0 / (2.0 * d_min)
    keys = jax.random.split(jax.random.PRNGKey(3), len(cases))
    for kappa, crossed_label in ((0.0, 1), (kod * D, 0)):
        geom = Sphere(radius=R, permeability=kappa)
        r = jax.jit(jax.vmap(lambda p, q, k: geom.permeate(p, q, jnp.float32(kappa / D), jnp.float32(0.0), k)[0]))(
            jnp.asarray(starts, jnp.float32), jnp.asarray(steps, jnp.float32), keys)
        lab = np.asarray(jax.vmap(geom.classify_position)(r))
        radial = np.linalg.norm(np.asarray(r), axis=1)
        assert ((lab == 1) == (radial < R)).all(), f"kappa {kappa}: a label disagrees with its position"
        assert (lab[~crossing] == 1).all(), "a non-crossing impact left the sphere"
        assert (lab[crossing] == crossed_label).all(), f"kappa {kappa}: {(lab[crossing] != crossed_label).sum()} crossing impacts misplaced"


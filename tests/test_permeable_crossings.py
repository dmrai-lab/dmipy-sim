"""The membrane must still let walkers through, at the rate it implies.

The compartment sentinel (dmrai-lab/dmipy-sim#86) rejects a compartment change that no
crossing granted. The obvious way to get that wrong is to seal the wall instead: a walker
that legitimately transmits also "changes compartment", and a sentinel that cannot tell the
difference silences the physics rather than the bug. That failure mode is invisible to the
impermeable tests -- at kappa = 0 a sealed wall and a correct wall look identical.

So these check the other side: crossings HAPPEN, at the rate the membrane implies, and the
compartment labels that come back describe where the walkers actually are.

The reference is the closed two-compartment exchange law. For `PermeableSlab1D` -- a closed
slab of length L with a permeable membrane at L/2 and reflecting outer walls -- it is exact:

    f_A(t) = 1/2 + 1/2 exp(-t / tau),   tau = L / (4 kappa)

with all walkers starting in compartment A. No curvature, no exterior re-entry, so this
isolates the membrane rule from everything else.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometries import PermeableSlab1D, Cylinder

D = 2.0e-9
STEP = 2.0e-8
L = 5.0e-6


def _walk(geom, r, n_steps, seed=5):
    kod = jnp.float32(float(geom.permeability) / D)
    f = jax.jit(jax.vmap(lambda p, s, k: geom.permeate(p, s, kod, jnp.float32(0.0), k)[0],
                         in_axes=(0, 0, 0)))
    rng = np.random.default_rng(seed)
    m = r.shape[0]
    for i in range(n_steps):
        d = rng.normal(size=(m, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = f(r, jnp.asarray(d * STEP, jnp.float32),
              jax.random.split(jax.random.PRNGKey(i), m))
    return r


# Calibrated, not assumed. The full exponential f_A = 1/2 + 1/2 exp(-4 kappa t / L) cannot
# be reached in a CI-sized walk: the well-mixed law needs kappa L / D << 1 (kappa << 4e-4),
# and tau = L/(4 kappa) is then tens of ms while 10k sub-steps span 0.33 ms. So the check is
# the SHORT-TIME limit of that same law, f_B -> 2 kappa T / L, which is where the walk is.
#
# Measured (6000 walkers, 10000 steps, L = 5 um):
#     kappa 0        f_B 0.00000   crossings   0
#     kappa 2.5e-5   f_B 0.00367   theory 0.00333   ratio 1.10   crossings 22
#     kappa 5.0e-5   f_B 0.00633   theory 0.00667   ratio 0.95   crossings 38
# Ratio scatter is Poisson on those counts (~20%), so the band below is set from the data.

_T_STEP = STEP ** 2 / (6.0 * D)          # <r^2> = 6 D t for a fixed-length 3-D step


@pytest.mark.parametrize("kappa", [2.5e-5, 5.0e-5])
def test_slab_crossing_rate_matches_short_time_exchange(kappa):
    """Crossing RATE against theory -- catches a sealed wall and a leaky one alike."""
    n, n_steps = 6000, 10000
    geom = PermeableSlab1D(length=L, permeability=kappa)
    r0 = geom.init_positions(n, jax.random.PRNGKey(0))      # all start in compartment A
    assert float((np.asarray(r0)[:, 0] < L / 2).mean()) == 1.0

    rf = np.asarray(_walk(geom, r0, n_steps))
    f_B = float((rf[:, 0] >= L / 2).mean())

    T = n_steps * _T_STEP
    expected = 2.0 * kappa * T / L
    assert kappa * L / D < 0.2, "outside the barrier-limited regime the law does not apply"

    n_cross = int(round(f_B * n))
    assert n_cross > 0, "no walker crossed at all -- the membrane is sealed"
    ratio = f_B / expected
    assert 0.6 < ratio < 1.5, (
        f"kappa={kappa:.1e}: f_B={f_B:.5f} vs theory {expected:.5f} (ratio {ratio:.2f}, "
        f"{n_cross} crossings). Low means the sentinel is eating legal crossings; "
        f"high means the membrane leaks.")


def test_crossing_rate_is_linear_in_permeability():
    """Doubling kappa doubles the short-time crossing rate; a sealed wall flattens this."""
    n, n_steps = 6000, 10000
    f = {}
    for kappa in (2.5e-5, 5.0e-5):
        geom = PermeableSlab1D(length=L, permeability=kappa)
        r0 = geom.init_positions(n, jax.random.PRNGKey(0))
        rf = np.asarray(_walk(geom, r0, n_steps))
        f[kappa] = float((rf[:, 0] >= L / 2).mean())
    assert f[2.5e-5] > 0
    slope = f[5.0e-5] / f[2.5e-5]
    assert 1.4 < slope < 2.7, f"expected ~2x, got {slope:.2f} from {f}"


def test_impermeable_slab_is_exactly_conserved():
    """The same geometry at kappa = 0 must grant nothing -- the other end of the same rule."""
    n, n_steps = 4000, 6000
    geom = PermeableSlab1D(length=L, permeability=0.0)
    r0 = geom.init_positions(n, jax.random.PRNGKey(0))
    rf = np.asarray(_walk(geom, r0, n_steps))
    assert float((rf[:, 0] < L / 2).mean()) == 1.0, "walkers crossed an impermeable membrane"


def test_crossing_count_rises_monotonically_with_permeability():
    """More permeable wall, more crossings -- the sentinel must not flatten this."""
    n, n_steps = 3000, 4000
    fracs = []
    for kappa in (0.0, 1e-5, 5e-5, 2e-4):
        geom = PermeableSlab1D(length=L, permeability=kappa)
        r0 = geom.init_positions(n, jax.random.PRNGKey(0))
        rf = np.asarray(_walk(geom, r0, n_steps))
        fracs.append(float((rf[:, 0] >= L / 2).mean()))     # fraction that reached B
    assert fracs[0] == 0.0, f"kappa=0 leaked: {fracs[0]}"
    assert all(b > a for a, b in zip(fracs, fracs[1:])), f"not monotonic in kappa: {fracs}"


def test_permeable_cylinder_labels_match_positions():
    """`classify_position` must describe where the walkers ACTUALLY are, after crossings.

    The sentinel corrects positions; if it and the classifier ever disagreed, the compartment
    channel would be fiction. This asserts they agree exactly on a geometry that is really
    exchanging -- i.e. with the permeable path live, not just at kappa = 0.
    """
    R, n, n_steps = 5e-6, 3000, 4000
    geom = Cylinder(radius=R, orientation=[0, 0, 1.0], permeability=1e-4)
    r0 = geom.init_positions(n, jax.random.PRNGKey(0))
    rf = np.asarray(_walk(geom, r0, n_steps))

    labels = np.asarray(jax.jit(jax.vmap(geom.classify_position))(jnp.asarray(rf)))
    radial = np.linalg.norm(rf[:, :2], axis=1)
    # convention: 0 = intra (|r_xy| < R), 1 = extra
    assert ((labels == 0) == (radial < R)).all(), (
        f"{int(((labels == 0) != (radial < R)).sum())} labels disagree with the positions")
    # and the walk must genuinely have exchanged, or this proves nothing
    frac_extra = float((labels == 1).mean())
    assert frac_extra > 0.002, f"only {frac_extra:.4f} exchanged; test is vacuous"

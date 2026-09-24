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

from dmipy_sim.geometry import PermeableSlab1D, Cylinder

D = 2.0e-9
STEP = 2.0e-8
L = 5.0e-6


def _walk(geom, r, n_steps, seed=5):
    """One isotropic fixed-length walk through `geom.permeate`, run entirely on device.

    This was a Python loop: `n_steps` host-side numpy draws, each followed by a host->device
    transfer and a separate dispatch -- 10,000 round trips for the longest walk here, which cost
    far more than the physics. It is now one `lax.scan`, so the whole walk is a single dispatch.

    The step law is unchanged (isotropic direction, fixed length STEP, same `permeate` call with
    the same kappa/D and zero rho). What changes is the SOURCE of the random numbers -- numpy's
    generator becomes JAX's -- so the specific realisation differs and the measured values in the
    calibration note above were re-measured against it. That is a different sample of the same
    process, not a different process.
    """
    kod = jnp.float32(float(geom.permeability) / D)
    m = r.shape[0]

    def body(pos, key):
        k_dir, k_perm = jax.random.split(key)
        d = jax.random.normal(k_dir, (m, 3), dtype=jnp.float32)
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
        step = (d * jnp.float32(STEP)).astype(jnp.float32)
        pos = jax.vmap(lambda p, s, k: geom.permeate(p, s, kod, jnp.float32(0.0), k)[0],
                       in_axes=(0, 0, 0))(pos, step, jax.random.split(k_perm, m))
        return pos, None

    keys = jax.random.split(jax.random.PRNGKey(seed), n_steps)
    r_final, _ = jax.lax.scan(body, r, keys)
    return r_final


# Calibrated, not assumed. The full exponential f_A = 1/2 + 1/2 exp(-4 kappa t / L) cannot
# be reached in a CI-sized walk: the well-mixed law needs kappa L / D << 1 (kappa << 4e-4),
# and tau = L/(4 kappa) is then tens of ms while 10k sub-steps span 0.33 ms. So the check is
# the SHORT-TIME limit of that same law, f_B -> 2 kappa T / L, which is where the walk is.
#
# Re-measured after the walk moved to lax.scan with JAX's RNG (6000 walkers, 10000 steps,
# L = 5 um) -- same process, different realisation:
#     kappa 0        f_B 0.00000   crossings   0
#     kappa 2.5e-5   f_B 0.00383   theory 0.00333   ratio 1.15   crossings 23
#     kappa 5.0e-5   f_B 0.00717   theory 0.00667   ratio 1.08   crossings 43
# Ratio scatter is Poisson on those counts (~20%), so the band below is set from the data. A seed
# sweep at kappa=2.5e-5 (seeds 5/11/23) gives 1.15 / 0.95 / 1.25 -- the band is comfortable, not
# fitted to one lucky draw, and it is the SAME band the numpy-RNG version used.

_T_STEP = STEP ** 2 / (6.0 * D)          # <r^2> = 6 D t for a fixed-length 3-D step


# One walk per (kappa, n, n_steps), shared by every test that wants it. The rate test and the
# linearity test below asked for the SAME two walks -- same kappas, walker count, step count and
# seed -- and each ran them itself, so the module walked 6000 walkers x 10000 steps four times to
# look at two results. Caching is not a statistical compromise here: the two tests were already
# consuming identical samples, they just each paid to generate them. Assertions are untouched.
#
# Populated on first USE, never at import/collection (#91).
_WALKS = {}


def _slab_walk(kappa, n, n_steps, seed=5):
    """(r0, final positions) for a PermeableSlab1D walk, computed once per distinct request."""
    key = (kappa, n, n_steps, seed)
    if key not in _WALKS:
        geom = PermeableSlab1D(length=L, permeability=kappa)
        r0 = geom.init_positions(n, jax.random.PRNGKey(0))
        _WALKS[key] = (np.asarray(r0), np.asarray(_walk(geom, r0, n_steps, seed)))
    return _WALKS[key]


@pytest.mark.parametrize("kappa", [2.5e-5, 5.0e-5])
def test_slab_crossing_rate_matches_short_time_exchange(kappa):
    """Crossing RATE against theory -- catches a sealed wall and a leaky one alike."""
    n, n_steps = 6000, 10000
    r0, rf = _slab_walk(kappa, n, n_steps)                  # all start in compartment A
    assert float((r0[:, 0] < L / 2).mean()) == 1.0

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
        _r0, rf = _slab_walk(kappa, n, n_steps)             # the rate test's walks, reused
        f[kappa] = float((rf[:, 0] >= L / 2).mean())
    assert f[2.5e-5] > 0
    slope = f[5.0e-5] / f[2.5e-5]
    assert 1.4 < slope < 2.7, f"expected ~2x, got {slope:.2f} from {f}"


def _slab_impacts():
    """(label, start, step) of a walker in compartment A fired at the membrane at ``x = L/2``: from four offsets
    inside it (the sentinel's nudge, a thousandth and a tenth of the width, near the outer wall), for steps of a
    half, one, three and ten times the offset (the last three cross), and steps that fold at the outer wall."""
    xm, w = L / 2, L / 2
    out = []
    for oname, off in (("nudge", 1e-4 * xm), ("1e-3w", 1e-3 * w), ("0.1w", 0.1 * w), ("0.9w", 0.9 * w)):
        for dname, mult in (("half", 0.5), ("one", 1.0), ("three", 3.0), ("ten", 10.0)):
            out.append((f"{oname}/{dname}", np.array([xm - off, 0.0, 0.0]), np.array([mult * off, 0.3 * off, 0.0])))
        out.append((f"{oname}/fold", np.array([xm - off, 0.0, 0.0]), np.array([1.5 * L, 0.0, 0.0])))       # past both outer walls
    return out


def _fire(geom, cases, kappa_over_D):
    starts = jnp.asarray(np.stack([c[1] for c in cases]), jnp.float32)
    steps = jnp.asarray(np.stack([c[2] for c in cases]), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(11), len(cases))
    r = jax.jit(jax.vmap(lambda p, s, k: geom.permeate(p, s, jnp.float32(kappa_over_D), jnp.float32(0.0), k)[0]))(starts, steps, keys)
    return np.asarray(r), np.asarray(jax.vmap(geom.classify_position)(r))


def test_the_impermeable_membrane_reflects_every_impact():
    """kappa = 0 grants nothing: every impact of the table ends in compartment A, inside the slab, and the ones that
    would have crossed are mirrored at the membrane."""
    geom = PermeableSlab1D(length=L, permeability=0.0)
    cases = _slab_impacts()
    r, lab = _fire(geom, cases, 0.0)
    wrong = lab != 1
    if wrong.any():
        rows = "\n".join(f"      {cases[i][0]:14} -> x = {r[i, 0] / L:.4f} L" for i in np.flatnonzero(wrong))
        pytest.fail(f"{wrong.sum()}/{len(cases)} impacts crossed an impermeable membrane:\n{rows}")
    assert (r[:, 0] >= 0.0).all() and (r[:, 0] <= L).all()
    for i, (name, start, step) in enumerate(cases):
        if name.endswith(("three", "ten")):                     # a crossing step: mirrored at the membrane, then folded at the outer walls
            x = L - (start[0] + step[0]); x = np.mod(x, 2 * L); x = 2 * L - x if x > L else x
            assert abs(r[i, 0] - x) <= 1e-4 * L / 2 + 1e-12, name


def test_a_membrane_that_must_transmit_transmits_every_impact():
    """The other end of the same rule: with ``2 kappa/D d_perp >= 1`` for every crossing step of the table the transmit
    probability is one, so every crossing impact ends in B whatever its key, and every non-crossing one stays in A.
    The probability itself rises linearly in kappa up to that cap."""
    from dmipy_sim.geometry._boundary import transmit_probability
    cases = _slab_impacts()
    d_min = min(abs(st[0] + sp[0] - L / 2) for n, st, sp in cases if n.endswith(("three", "ten")))
    kod = 1.0 / (2.0 * d_min)
    geom = PermeableSlab1D(length=L, permeability=kod * D)
    r, lab = _fire(geom, cases, kod)
    crossing = np.array([n.endswith(("three", "ten")) for n, _, _ in cases])
    assert (lab[crossing] == 0).all(), f"{(lab[crossing] != 0).sum()} crossing impacts were refused at p = 1"
    assert (lab[~crossing] == 1).all(), "a non-crossing impact changed compartment"
    p = np.asarray(transmit_probability(jnp.asarray([0.0, 0.25, 0.5, 1.0, 2.0]) * kod, jnp.float32(d_min)))
    assert np.allclose(p, [0.0, 0.25, 0.5, 1.0, 1.0], atol=1e-6)                 # linear in kappa, capped at one


def test_a_transmitted_walker_is_placed_on_the_side_its_label_reports():
    """After a granted crossing of the cylinder's membrane the walker sits where its label says: fired at the wall
    with the transmit probability at one, every impact ends outside and is labelled extra; at kappa = 0 every one
    ends inside and is labelled intra. The labels of the membrane's two verdicts agree with the positions exactly,
    which a random walk with a live membrane could only show for the crossings it happened to draw."""
    R = 5e-6
    offs = [1e-4 * R, 1e-3 * R, 0.1 * R]; mults = [0.5, 0.99, 1.01, 3.0, 10.0]; angles = [0.0, 45.0, 89.0]
    cases = []
    for off in offs:
        for m in mults:
            for deg in angles:
                th = np.deg2rad(deg)
                cases.append((f"{off / R:.0e}R/{m}/{deg}", np.array([R - off, 0.0, 0.0]), np.array([np.cos(th), np.sin(th), 0.0]) * m * off))
    bare = np.array([np.linalg.norm((c[1] + c[2])[:2]) for c in cases])   # the free step's end radius
    crossing = bare > R * (1 + 1e-9)
    assert (np.abs(bare - R) > 1e-9 * R).all(), "a case lands on the wall to rounding: undecidable, not a table row"
    d_min = float((bare[crossing] - R).min())
    kod = 1.0 / (2.0 * d_min)
    for kappa, want in ((0.0, 1), (kod * D, 0)):
        geom = Cylinder(radius=R, orientation=[0, 0, 1.0], permeability=kappa)
        r, lab = _fire(geom, cases, kappa / D)
        radial = np.linalg.norm(r[:, :2], axis=1)
        assert ((lab == 1) == (radial < R)).all(), f"kappa {kappa}: a label disagrees with its position"
        if kappa == 0.0:
            assert (lab == 1).all(), "an impact crossed an impermeable wall"
        else:
            assert (lab[crossing] == want).all(), f"{(lab[crossing] != want).sum()} crossing impacts refused at p = 1"
            assert (lab[~crossing] == 1).all()

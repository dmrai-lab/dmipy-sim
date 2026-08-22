"""Every walk driver must start where its `r0` says, not where its geometry would have put it.

`Mesh.init_positions` takes an `intra` flag that defaults to **True** -- inside the surface -- so a
driver that seeds itself picks a compartment on the caller's behalf. On a sphere or cylinder that guess
is right, which is why it went unnoticed for so long: every MT test used a sphere. On a fibre bundle's
extra-axonal pool, whose geometry is the OUTER surface and whose walkers belong outside it, the guess
re-simulated the intra pool and labelled the result "extra" -- ensemble bound fraction 0.0924 against a
target 0.1390, fixed to 0.1355 by passing the seeds through (#67). `simulate_bloch`,
`_simulate_bloch_mt` and `simulate_cpmg` had the same omission (#68).

**Why an analytic Sphere and not a mesh here.** The natural test is pool contrast -- seed outside a mesh
sphere and watch the signal fall to the free limit -- and it does not work, for a reason worth writing
down: plain `simulate_bloch` ignores its own `sub_steps` argument and takes one displacement per
waveform step, so it cannot confine a mesh at all (0.0505 on a 2 um mesh sphere where `core.simulate`
gives 0.9631 and the analytic `Sphere` 0.9644). A pool test on a mesh would measure that bug rather than
this one. An analytic `Sphere` reflects exactly at any step length, so the driver is sound there, and
`r0` shows up instead as the START DISTRIBUTION: at a diffusion time short against R^2/D, centre-start
walkers have explored less than uniformly-seeded ones and retain more signal.

The mesh pool contrast IS tested, on the driver that handles meshes correctly --
`test_mt_mesh_binding.py::test_the_mt_driver_walks_the_pool_its_seeds_name`.

Both directions are asserted every time. Explicit uniform seeds must reproduce the default (so a driver
that ignored `r0` outright cannot pass by luck), and centre seeds must not (so `r0` demonstrably reaches
the walk). Measured separation is 0.130 against a 0.011 spread between two independent uniform draws.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax

from dmipy_sim import Sphere, cpmg, pgse, set_b, simulate_cpmg
from dmipy_sim.bloch import simulate_bloch

R = 2e-6
D = 2e-9
B = 2.0e9
N = 2000
EXC = [{"t_s": 0.0, "flip_deg": 90.0, "axis_deg": 90.0}]      # 90_y -> Mx = cos(phi)

# tau_D = R^2/D = 2.0 ms, so DELTA = 2 ms leaves the ensemble short of full exploration and the start
# distribution still visible. slew_rate=inf: reaching b=2e9 in a 0.2 ms pulse needs ~19 T/m, which no
# real gradient slews to, and the square limit is what keeps the comparison clean.
DELTA_PULSE, DELTA_SEP = 0.2e-3, 2.0e-3


@pytest.fixture(scope="module")
def sphere():
    return Sphere(radius=R)


@pytest.fixture(scope="module")
def seeds(sphere):
    """(uniform, centre). Uniform uses the geometry's own sampler, so it is the same DISTRIBUTION as
    the default with a different draw -- which is exactly what makes assertion 1 a distribution test
    and not an identity test."""
    uniform = np.asarray(sphere.init_positions(N, jax.random.PRNGKey(9)), np.float64)
    assert np.linalg.norm(uniform, axis=1).max() <= R, "uniform seeds are not inside the sphere"
    return uniform, np.zeros((N, 3), np.float64)


@pytest.fixture(scope="module")
def waveform():
    return set_b(pgse(delta=DELTA_PULSE, DELTA=DELTA_SEP, G_magnitude=0.05,
                      bvecs=[[1.0, 0.0, 0.0]], n_t=300, slew_rate=np.inf), B)


def _assert_r0_reaches_the_walk(run, seeds, label, min_sep=0.08):
    """`run(r0) -> real signal`, held to both directions so neither can pass vacuously.

    `min_sep` is per-sequence because the separation is a physics quantity, not a tolerance: it is how
    much of the start distribution survives to the echo, which depends on the pulse width against
    R^2/D. Each caller's value is set from a measured signal-to-noise, where "noise" is the spread
    between two independent uniform DRAWS of the same distribution.
    """
    uniform, centre = seeds
    e_default, e_uniform, e_centre = run(None), run(uniform), run(centre)

    # 1. supplying the geometry's OWN distribution reproduces the default: r0 replaces the internal
    #    seeding rather than perturbing it, and a driver that ignored r0 would also land here -- which
    #    is why (2) exists.
    assert e_default == pytest.approx(e_uniform, abs=0.05), (
        f"{label}: default {e_default:.4f} vs explicit uniform seeds {e_uniform:.4f} -- same "
        f"distribution should give the same signal to within the MC spread")
    # 2. a genuinely different start gives a different signal: r0 reached the walk.
    assert abs(e_centre - e_uniform) > min_sep, (
        f"{label}: centre-start {e_centre:.4f} vs uniform {e_uniform:.4f} differ by "
        f"{abs(e_centre - e_uniform):.4f}, below the {min_sep:.3f} this sequence resolves -- "
        f"r0 is being ignored or overwritten")


@pytest.mark.slow
def test_simulate_bloch_starts_where_r0_says(sphere, seeds, waveform):
    def run(r0):
        return float(np.real(simulate_bloch(N, D, waveform, sphere, EXC, seed=3, r0=r0,
                                            require_gpu=False)[0]))

    _assert_r0_reaches_the_walk(run, seeds, "simulate_bloch")


@pytest.mark.slow
def test_the_bloch_mt_path_starts_where_r0_says(sphere, seeds, waveform):
    """`kappa_MT > 0` dispatches to `_simulate_bloch_mt`, which seeds at its own separate site."""
    def run(r0):
        return float(np.real(simulate_bloch(N, D, waveform, sphere, EXC, seed=3, r0=r0,
                                            kappa_MT=1e-6, dwell_time=2e-3,
                                            equilibrate_binding="off", require_gpu=False)[0]))

    _assert_r0_reaches_the_walk(run, seeds, "_simulate_bloch_mt")


@pytest.mark.slow
def test_simulate_cpmg_starts_where_r0_says(sphere, seeds):
    """Same short-time argument, on the echo train: TE 2 ms against tau_D 2 ms, first echo read.

    G set directly rather than through set_b, because b accumulates ALONG a train and a single
    b_target is not the quantity that governs any one echo. The first echo behaves as delta ~ TE/2,
    DELTA ~ TE. At 19 T/m both arms decayed into the noise (0.024 and -0.004); TE=1 ms at 12 T/m keeps
    the signal near 0.5 where the start distribution is still visible.
    """
    wf = cpmg(n_echoes=2, TE=1e-3, G_magnitude=12.0, bvecs=[[1.0, 0.0, 0.0]], n_t_per_echo=150)

    def run(r0):
        s = np.asarray(simulate_cpmg(N, D, wf, sphere, seed=3, r0=r0, require_gpu=False))
        return float(np.real(s.ravel()[0]))       # first echo: least explored, most r0-sensitive

    # 0.04 rather than the PGSE 0.08: measured separation 0.0648 against a 0.0021 spread between two
    # independent uniform draws (31:1). A train cannot do better -- its gradient is continuous with
    # sign flips, so its effective pulse width is ~TE/2 and much of the start distribution is already
    # washed out by the first echo. At TE=2 ms it separated by only 0.026 on a ~0.022 MC floor.
    _assert_r0_reaches_the_walk(run, seeds, "simulate_cpmg", min_sep=0.04)


def test_r0_with_the_wrong_shape_is_rejected(sphere):
    from dmipy_sim.geometries import initial_positions

    assert initial_positions(sphere, 8, jax.random.PRNGKey(0), None).shape == (8, 3)
    assert initial_positions(sphere, 8, jax.random.PRNGKey(0), np.zeros((8, 3))).shape == (8, 3)
    for bad in (np.zeros((7, 3)), np.zeros((8, 2)), np.zeros(8)):
        with pytest.raises(ValueError, match=r"r0 must have shape"):
            initial_positions(sphere, 8, jax.random.PRNGKey(0), bad)

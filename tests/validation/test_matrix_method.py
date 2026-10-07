"""``examples/validation/matrix_method.py`` -- the eigenmode (matrix) closed form of restricted diffusion in a
planar, cylindrical or spherical pore (Codd & Callaghan 1999) that grades the canonical-pores family's
``analytic`` reference. It had no test (dmipy-sim#497): the only exercise it got was
``build_canonical_pores.measured_quantities`` refining its own mode count against itself, which measures
self-convergence, not correctness -- a wrong normalisation would converge beautifully to the wrong number.

This module holds it to analytical limits that need no Monte Carlo and no other package:

- the **free-diffusion limit**: a pore far larger than the diffusion length reproduces ``exp(-bD)``.
- the **long-time limit**: at a mixing time long enough that every excited mode has decayed, a narrow-pulse
  pair's signal is the pore's own form factor ``|F(q)|^2`` (Callaghan's "diffraction" result).
- the **short-time (Mitra) limit**: the measured ADC approaches ``D0 (1 - 4/(9 sqrt(pi)) (S/V) sqrt(D0 t))``
  as the diffusion time shrinks, a universal result independent of pore shape.
- the **narrow-pulse closed forms**: the ground state's coupling to each excited mode, ``B[0,k]^2``, against
  van Gelderen's (cylinder), Murday-Cotts's (sphere) and Neuman's (plane) published expressions, and their
  sum against the pore's own second moment.

and it pins the two latent defects dmipy-sim#497 names:

- the run-product (exact) path was gated on the number of *distinct gradient amplitudes* rather than the
  number of *runs* ``_runs`` already computes, so a slew-limited waveform (many amplitudes from its ramps,
  a handful of runs) silently fell back to the ``O(dt^2)`` Strang split with nothing in the reference record
  to say the reference was approximated.
- the plane's position operator was built on ``x in [0, L]`` rather than centred on the slab's midplane, so
  ``B[0, 0] = L/2`` instead of 0 (against 0 for the cylinder and the sphere) -- harmless under the ``abs()``
  every current caller takes, but a spurious global phase for any caller that wants the complex signal.

Quadrature resolution (``RADIAL_NODES``/``ANGULAR_NODES``) is a module constant, not a constructor argument
(dmipy-sim#497's "also worth folding in"), so the slower tests here lower it with ``monkeypatch`` rather than
waiting on the default (4000, 2048) grid; each test states the measured residual that justifies its tolerance
at the resolution and mode count it runs at.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy import linalg, special
from scipy.optimize import brentq

import examples.validation.matrix_method as mm
from examples.validation.matrix_method import GAMMA, MatrixPore, _runs
from dmipy_sim.sequences import pgse

D0 = 2e-9    # m^2/s, a representative free diffusivity
A = 5e-6     # m, the pore radius (cylinder/sphere) / half-thickness (plane) used throughout


# --------------------------------------------------------------------------------------- defect 1: the gate
def test_signal_takes_exact_path_for_slew_limited_pgse():
    """A slew-limited PGSE (``pgse`` default ``slew_rate=200`` T/m/s) has 11 distinct amplitudes from its
    ramps but only 12 runs of constant value -- cheap for the exact run-product path and exactly what
    dmipy-sim#497 names: gating on the amplitude count sends it through the ``O(dt^2)`` Strang split instead
    with nothing recorded to say so. ``_runs`` is a tested primitive, so re-driving the same run-product
    formula with it here (independent of whichever branch ``signal`` itself takes) is a legitimate oracle for
    *which path was taken*, and the two disagree by 2.9e-5 when the gate misfires (``_signal_strang`` on this
    waveform) against 0 when it does not -- so 1e-8 cleanly separates them with no floating-point slack spent.
    """
    mp = MatrixPore("cylinder", 2 * A, D0, n_modes=(8, 6))
    wf = pgse([[1.0, 0.0, 0.0]], 10e-3, 40e-3, bvalues=[1e9], n_t=300)
    g = np.asarray(wf.G_eff, np.float64)[0][:, 0]
    dt = float(wf.dt)

    n_unique = len(np.unique(np.round(g, 15)))
    n_runs = len(_runs(g))
    assert n_unique > 6, "the waveform must exercise the old >6-amplitude gate"
    assert n_runs <= 64, "and still be cheap in runs, or this is not the regime #497 describes"

    def exact_run_product(pore, g_axis, dt):
        """The run-product formula, driven independently of ``MatrixPore.signal``'s own gate."""
        M = np.eye(len(pore.lam), dtype=complex)
        for amp, ns in _runs(np.asarray(g_axis, np.float64)):
            tau = ns * float(dt)
            if amp == 0.0:
                M = np.exp(-tau * pore.D * pore.lam)[:, None] * M
            else:
                M = linalg.expm(-tau * (pore.D * np.diag(pore.lam) + 1j * GAMMA * amp * pore.B)) @ M
        return complex(M[0, 0])

    oracle = exact_run_product(mp, g, dt)
    measured = mp.signal(g, dt)
    assert abs(measured - oracle) < 1e-8, (
        f"signal() disagreed with the exact run-product formula by {abs(measured - oracle):.3e}: "
        "it took the approximate (Strang) path on a waveform cheap enough for the exact one")


# ------------------------------------------------------------------------------- defect 2: the plane's B[0,0]
def test_position_operator_is_centred_on_every_shape():
    """``B`` is the eigenbasis's representation of the position operator along the measured axis; its (0, 0)
    entry is the ground state's own mean position, which is the pore's centre by construction for all three
    shapes (the cylinder's and sphere's radial coordinate already runs outward from the centre, and the
    plane's grid is centred by the #497 fix). Exact to quadrature float noise, so 1e-9 m -- nine orders below
    the micron pores this module is built for -- is generous rather than tight.
    """
    for shape, size in (("cylinder", 2 * A), ("sphere", 2 * A), ("plane", 2 * A)):
        mp = MatrixPore(shape, size, D0, n_modes=(8, 6))
        assert abs(mp.B[0, 0]) < 1e-9, f"{shape}: B[0,0] = {mp.B[0, 0]:.3e}, not centred on the pore"


def test_sum_of_squared_couplings_is_the_pores_second_moment():
    """``sum_k B[0,k]^2 = <0|x (sum_k |k><k|) x|0> = <0|x^2|0>`` by completeness of the eigenbasis -- the
    pore's own second moment about its centre, independent of any gradient or waveform: ``a^2/4`` for a disk
    of radius ``a`` (``<r^2>=a^2/2``, ``<cos^2(theta)>=1/2``), ``a^2/5`` for a ball (``<r^2>=3a^2/5``,
    ``<cos^2(theta)>=1/3``), ``L^2/12`` for a slab of thickness ``L`` centred at 0. The plane only gets the
    right answer once its position operator is centred (#497's second defect): an uncentred ``x in [0,L]``
    gives ``<x^2> = L^2/3`` instead, 4x too big. Matches to 2e-4 relative at ``n_modes=(8,6)``, the quadrature
    residual the issue measured (4.5e-6 at full resolution, 4000 radial nodes; this runs at the default too,
    since the three shapes build in under five seconds combined).
    """
    mp_c = MatrixPore("cylinder", 2 * A, D0, n_modes=(8, 6))
    mp_s = MatrixPore("sphere", 2 * A, D0, n_modes=(8, 6))
    L = 2 * A
    mp_p = MatrixPore("plane", L, D0, n_modes=(8, 6))

    npt.assert_allclose(np.sum(mp_c.B[0, :] ** 2), A ** 2 / 4, rtol=2e-4, err_msg="cylinder <x^2>")
    npt.assert_allclose(np.sum(mp_s.B[0, :] ** 2), A ** 2 / 5, rtol=2e-4, err_msg="sphere <x^2>")
    npt.assert_allclose(np.sum(mp_p.B[0, :] ** 2), L ** 2 / 12, rtol=2e-4, err_msg="plane <x^2>")


# ----------------------------------------------------------------------- narrow-pulse closed-form coefficients
def test_coupling_coefficients_match_published_closed_forms():
    """The ground state (m=0 / l=0 / n=0, even) couples only to the first excited angular/parity family
    (m=1 for the cylinder, l=1 for the sphere, odd n for the plane) because ``x`` is a parity-1 operator; the
    weight of that coupling, ``B[0,k]^2``, is published in closed form -- van Gelderen (1994) for the
    cylinder, Murday & Cotts (1968) for the sphere, Neuman (1974) for the plane -- in terms of the same roots
    this module's own root finders locate. Re-deriving the roots here with ``scipy.special`` (cylinder) and
    an independent ``brentq`` scan (sphere) rather than importing the module's ``_cyl_neumann_roots`` /
    ``_sph_neumann_roots`` keeps this an external check of both the roots and the quadrature. Matches to 5e-4
    relative (measured; the issue's own hand-check gets 5.8e-5 at the default 4000-node grid, so this has
    headroom), and the even-parity plane couplings are asserted zero outright (the selection rule, not a
    fitted number).
    """
    m_max, n_max = 8, 6

    # -- cylinder vs van Gelderen: a_k = 2 a^2 / (beta_k^2 (beta_k^2 - 1)), beta_k the roots of J1'(x) = 0.
    mp_c = MatrixPore("cylinder", 2 * A, D0, n_modes=(m_max, n_max))
    states_c = [(m, n) for m in range(m_max + 1) for n in range(n_max)]
    idx1 = [i for i, (m, _) in enumerate(states_c) if m == 1]
    beta = special.jnp_zeros(1, n_max)
    van_gelderen = 2 * A ** 2 / (beta ** 2 * (beta ** 2 - 1))
    npt.assert_allclose(mp_c.B[0, idx1] ** 2, van_gelderen, rtol=5e-4, err_msg="cylinder vs van Gelderen")

    # -- sphere vs Murday-Cotts: a_k = 2 a^2 / (beta_k^2 (beta_k^2 - 2)), beta_k the roots of j1'(x) = 0.
    mp_s = MatrixPore("sphere", 2 * A, D0, n_modes=(m_max, n_max))
    states_s = [(l, n) for l in range(m_max + 1) for n in range(n_max)]
    idx_l1 = [i for i, (l, _) in enumerate(states_s) if l == 1]

    def j1_prime(x):
        return special.spherical_jn(1, x, derivative=True)

    xs = np.linspace(1e-6, 4 * (n_max + 3) + 20, 40_000)
    sign = np.sign(j1_prime(xs))
    roots = []
    for i in np.where(np.diff(sign) != 0)[0]:
        r = brentq(j1_prime, xs[i], xs[i + 1])
        if not roots or r - roots[-1] > 1e-3:
            roots.append(r)
        if len(roots) >= n_max:
            break
    beta_sph = np.asarray(roots)
    murday_cotts = 2 * A ** 2 / (beta_sph ** 2 * (beta_sph ** 2 - 2))
    npt.assert_allclose(mp_s.B[0, idx_l1] ** 2, murday_cotts, rtol=5e-4, err_msg="sphere vs Murday-Cotts")

    # -- plane vs Neuman: a_n = 8 L^2 / (n^4 pi^4) for odd n, 0 for even n.
    L = 2 * A
    mp_p = MatrixPore("plane", L, D0, n_modes=(m_max, n_max))
    n = np.arange(1, mp_p.B.shape[0])
    odd = n % 2 == 1
    neuman = 8 * L ** 2 / (n[odd] ** 4 * np.pi ** 4)
    npt.assert_allclose(mp_p.B[0, 1:][odd] ** 2, neuman, rtol=5e-4, err_msg="plane vs Neuman")
    npt.assert_allclose(mp_p.B[0, 1:][~odd], 0.0, atol=1e-18, err_msg="plane even-parity couplings must vanish")


# --------------------------------------------------------------------------------------- free-diffusion limit
def test_free_diffusion_limit_huge_pore():
    """A pore 45x the rms displacement (``sqrt(2 D0 delta) = 4.47 um`` at ``delta=5 ms``; ``A=200 um``) sees
    essentially no restriction, so the signal must tend to the Stejskal-Tanner free-diffusion value
    ``exp(-bD0)``. The gradient needed for a *given* b grows with 1/delta while ``B``'s matrix elements grow
    with the pore size, so a huge pore at a strong gradient needs impractically many modes (verified: at
    b=3e8 the (6,6)- and (8,8)-mode answers for a 100 um pore already disagree by 1-3%, i.e. not converged);
    keeping b modest (1e7) keeps the eigenmode truncation converged (measured: (6,6) and (8,8) modes agree
    to 1e-5 at this b) so what remains is the pore's own residual confinement correction, measured here at
    6.9-9.4e-4 across the three shapes and shrinking as the pore grows (9.4e-4 at 200 um against 2.6e-3 at
    50 um) -- 2e-3 tolerance is a 2x margin on the converged residual, not slack for truncation error.
    """
    delta = Delta = 5e-3
    b = 1e7
    wf = pgse([[1.0, 0.0, 0.0]], delta, Delta, bvalues=[b], n_t=20, slew_rate=np.inf)
    g = np.asarray(wf.G_eff, np.float64)[0][:, 0]
    dt = float(wf.dt)
    free = np.exp(-b * D0)

    huge = 200e-6
    for shape in ("cylinder", "sphere", "plane"):
        mp = MatrixPore(shape, huge, D0, n_modes=(6, 6))
        s = abs(mp.signal(g, dt))
        npt.assert_allclose(s, free, rtol=2e-3, err_msg=f"{shape}: huge-pore signal vs free diffusion")


# ----------------------------------------------------------------------------------------- long-time limit
def test_long_time_limit_tends_to_form_factor(monkeypatch):
    """At a mixing time long enough that every excited mode has decayed (``D0 lambda_1 Delta >> 1``: for a
    5 um cylinder, ``lambda_1 ~ 1.4e11``, so ``Delta=100 ms`` gives ``D0 lambda_1 Delta ~ 27``), a
    narrow-pulse pair's echo is ``<0|e^{-iqB} diag(e^{-D lambda Delta}) e^{iqB}|0> -> |<0|e^{iqB}|0>|^2 =
    |F(q)|^2`` -- Callaghan's long-time "diffraction" result, with ``F(q)`` the pore's own Fourier-transformed
    density: ``2 J1(qa)/(qa)`` (cylinder), ``3(sin(qa)-qa cos(qa))/(qa)^3`` (sphere), ``sin(qa)/(qa)``
    (plane, half-thickness ``a``). ``RADIAL_NODES``/``ANGULAR_NODES`` are lowered 600/384 (from 4000/2048)
    for this test's signal() calls, measured to change the result by <1e-4 relative at these mode counts, well
    under the 1% tolerance set from the 0.53% residual measured at ``n_modes=(8,6)`` (stable across the three
    shapes, so it is the mode truncation plus the finite 50 us pulse, not noise).
    """
    monkeypatch.setattr(mm, "RADIAL_NODES", 600)
    monkeypatch.setattr(mm, "ANGULAR_NODES", 384)

    delta = 50e-6     # narrow: D0 * lambda_1 * delta ~ 0.014, negligible diffusion during the pulse
    Delta = 100e-3    # long: D0 * lambda_1 * Delta ~ 27, every excited mode fully decayed
    qa = 2.0          # within the form factor's first lobe for all three shapes
    q = qa / A
    g_amp = q / (GAMMA * delta)
    n_pulse = 2
    dt = delta / n_pulse
    n_gap = int(round(Delta / dt))
    g = np.concatenate([np.full(n_pulse, g_amp), np.zeros(n_gap), np.full(n_pulse, -g_amp)])

    form_factors = {
        "cylinder": 2 * special.j1(q * A) / (q * A),
        "sphere": 3 * (np.sin(q * A) - q * A * np.cos(q * A)) / (q * A) ** 3,
        "plane": np.sin(q * A) / (q * A),
    }
    for shape, Fq in form_factors.items():
        mp = MatrixPore(shape, 2 * A, D0, n_modes=(8, 6))
        s = abs(mp.signal(g, dt))
        npt.assert_allclose(s, Fq ** 2, rtol=1e-2, err_msg=f"{shape}: long-time signal vs |F(q)|^2")


# -------------------------------------------------------------------------------------- short-time Mitra limit
def test_short_time_limit_trends_toward_mitra_surface_to_volume(monkeypatch):
    """Mitra, Sen & Schwartz's universal short-time result, ``D(t)/D0 = 1 - 4/(9 sqrt(pi)) (S/V) sqrt(D0 t)``,
    is the leading term of an expansion in ``sqrt(t)``: it is not exact at any finite t, so this checks the
    one thing it predicts unconditionally -- that the gap between the measured ADC and the formula SHRINKS as
    t shrinks -- rather than a point tolerance a leading-order formula cannot promise. ``S/V`` for perpendicular
    diffusion in a cylinder/sphere/slab of radius or half-thickness ``a`` is ``2/a``, ``3/a``, ``1/a``
    (the plane's two walls over its full thickness ``2a``). Modes (8,6) and the same lowered quadrature as the
    long-time test keep each (build + signal) pair under 3 s; at these settings the gap is NOT yet converged
    in mode count (it is still moving at (14,10)), so this intentionally does not assert a tolerance on the
    gap itself, only its sign and its monotone shrinkage -- a method that did not expose the short-time limit
    at all would show no such trend.
    """
    monkeypatch.setattr(mm, "RADIAL_NODES", 600)
    monkeypatch.setattr(mm, "ANGULAR_NODES", 384)

    def mitra_D_over_D0(t, s_over_v):
        return 1.0 - (4.0 / (9.0 * np.sqrt(np.pi))) * s_over_v * np.sqrt(D0 * t)

    def measured_adc_over_d0(mp, t):
        delta = t / 20.0       # narrow relative to the diffusion time
        Delta = t
        b = 2e6                # modest: perturbative, away from kurtosis-regime curvature
        g_amp = np.sqrt(b / (GAMMA ** 2 * delta ** 2 * (Delta - delta / 3)))
        n_pulse = 2
        dt = delta / n_pulse
        n_gap = int(round((Delta - delta) / dt))
        g = np.concatenate([np.full(n_pulse, g_amp), np.zeros(n_gap), np.full(n_pulse, -g_amp)])
        s = abs(mp.signal(g, dt))
        return -np.log(s) / b / D0

    shapes = {"cylinder": 2.0 / A, "sphere": 3.0 / A, "plane": 1.0 / A}
    for shape, s_over_v in shapes.items():
        mp = MatrixPore(shape, 2 * A, D0, n_modes=(8, 6))
        t_late, t_early = 0.20e-3, 0.05e-3
        gap_late = measured_adc_over_d0(mp, t_late) - mitra_D_over_D0(t_late, s_over_v)
        gap_early = measured_adc_over_d0(mp, t_early) - mitra_D_over_D0(t_early, s_over_v)

        assert gap_late < 0, f"{shape}: expected the restricted ADC below Mitra's at t={t_late}"
        assert gap_early < 0, f"{shape}: expected the restricted ADC below Mitra's at t={t_early}"
        assert abs(gap_early) < abs(gap_late), (
            f"{shape}: the gap to Mitra's short-time prediction did not shrink from t={t_late} "
            f"({gap_late:+.4f}) to t={t_early} ({gap_early:+.4f})")

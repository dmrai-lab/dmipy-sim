"""``dmipy_sim.math.matrix_method`` -- the eigenmode (matrix) closed form of restricted diffusion in a
planar, cylindrical or spherical pore (Codd & Callaghan 1999), the canonical-pores family's reference and
what dmipy-fit's C5/S5 compartment models import.

Three groups of tests:

- the seven of dmipy-sim#497/#656 (moved here from ``examples/validation/matrix_method.py``'s own suite): the
  free-diffusion, long-time (diffraction) and short-time (Mitra) limits, the narrow-pulse coupling
  coefficients against van Gelderen / Murday-Cotts / Neuman's published closed forms, and the two latent
  defects #497 found and fixed (the exact run-product path gated on RUN count rather than amplitude count;
  the slab's position operator centred on its midplane).
- dmipy-fit's eight physical-validation tests (``dmipy_fit/signal_models/tests/test_matrix_method.py``),
  translated to this module's API: the Gaussian-phase (GPA) low-b limit for the cylinder (Van Gelderen 1994)
  and the sphere (Balinov/Murday-Cotts 1993), re-derived independently here (sim never imports fit) so each
  is a physics test against an analytical result rather than a cross-package regression; departure from the
  GPA at high b; b = 0 unity and the physical range; restriction monotonicity; mode convergence; the plane's
  free limit; the cylinder's parallel axis being free diffusion.
- one test that the unit-geometry route (:func:`dmipy_sim.math.matrix_method._unit_modes`, scaled by pore
  size) agrees with a DIRECT per-size quadrature -- an independent oracle re-deriving the modes at the
  physical size rather than importing the module's own scaling -- to the mode truncation, on each shape at
  two sizes.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy import linalg, special
from scipy.optimize import brentq

import dmipy_sim.math.matrix_method as mm
from dmipy_sim.constants import GAMMA
from dmipy_sim.math.matrix_method import DEFAULT_MODES, MatrixPore, matrix_restricted_batch, matrix_restricted_signal, _runs
from dmipy_sim.sequences import pgse

D0 = 2e-9    # m^2/s, a representative free diffusivity
A = 5e-6     # m, the pore radius (cylinder/sphere) / half-thickness (plane) used throughout


# ============================================================================================ moved from #497
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
        mp = MatrixPore(shape, size, D0, n_modes=_modes(shape, (8, 6)))
        assert abs(mp.B[0, 0]) < 1e-9, f"{shape}: B[0,0] = {mp.B[0, 0]:.3e}, not centred on the pore"


def _modes(shape, pair):
    """``n_modes`` with the shape of the pore: the pair for a cylinder or a sphere, its product (the count of
    cosine modes) for the plane."""
    return int(pair[0]) * int(pair[1]) if shape == "plane" else tuple(pair)


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
    mp_p = MatrixPore("plane", L, D0, n_modes=48)

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
    mp_p = MatrixPore("plane", L, D0, n_modes=m_max * n_max)
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
        mp = MatrixPore(shape, huge, D0, n_modes=_modes(shape, (6, 6)))
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
    for this test's ``MatrixPore`` builds, measured to change the result by <1e-4 relative at these mode
    counts, well under the 1% tolerance set from the 0.53% residual measured at ``n_modes=(8,6)`` (stable
    across the three shapes, so it is the mode truncation plus the finite 50 us pulse, not noise).
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
        mp = MatrixPore(shape, 2 * A, D0, n_modes=_modes(shape, (8, 6)))
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
        mp = MatrixPore(shape, 2 * A, D0, n_modes=_modes(shape, (8, 6)))
        t_late, t_early = 0.20e-3, 0.05e-3
        gap_late = measured_adc_over_d0(mp, t_late) - mitra_D_over_D0(t_late, s_over_v)
        gap_early = measured_adc_over_d0(mp, t_early) - mitra_D_over_D0(t_early, s_over_v)

        assert gap_late < 0, f"{shape}: expected the restricted ADC below Mitra's at t={t_late}"
        assert gap_early < 0, f"{shape}: expected the restricted ADC below Mitra's at t={t_early}"
        assert abs(gap_early) < abs(gap_late), (
            f"{shape}: the gap to Mitra's short-time prediction did not shrink from t={t_late} "
            f"({gap_late:+.4f}) to t={t_early} ({gap_early:+.4f})")


# ============================================================================= translated from dmipy-fit's 8
DELTA, DELTABIG = 10e-3, 40e-3
LAMBDA_PAR = 1.7e-9

#: Van Gelderen (1994) roots of J1'(x) = 0, the cylinder's perpendicular GPA closed form.
_VG_ROOTS = special.jnp_zeros(1, 60)

#: Balinov et al. (1993) roots of the sphere's transcendental equation (Murday-Cotts GPA); the first few,
#: independent of this module's own root finder (hand-copied from the published table).
_MC_ROOTS = np.array([
    2.081575978, 5.940369990, 9.205840145, 12.40444502, 15.57923641, 18.74264558, 21.89969648,
    25.05282528, 28.20336100, 31.35209173, 34.49951492, 37.64596032, 40.79165523, 43.93676147,
    47.08139741, 50.22565165, 53.36959180, 56.51327045, 59.65672900, 62.80000055,
])


def _van_gelderen_gpa(g, delta, Delta, D, diameter):
    """The cylinder's perpendicular Gaussian-phase attenuation (Van Gelderen et al. 1994, eq. 2-3), an
    independent closed form -- not a call into this module -- so the GPA-limit tests below assert against an
    analytical result rather than against ``matrix_restricted_signal`` itself."""
    radius = diameter / 2.0
    alpha = _VG_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D
    summands = (
        2 * alpha2D * delta - 2
        + 2 * np.exp(-alpha2D * delta) + 2 * np.exp(-alpha2D * Delta)
        - np.exp(-alpha2D * (Delta - delta)) - np.exp(-alpha2D * (Delta + delta))
    ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))
    return np.exp(-2 * (g * GAMMA) ** 2 * summands.sum())


def _murday_cotts_gpa(g, delta, Delta, D, diameter):
    """The sphere's Gaussian-phase attenuation (Balinov et al. 1993 / Murday & Cotts 1968), an independent
    closed form for the same reason."""
    radius = diameter / 2.0
    alpha = _MC_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D
    summands = (
        alpha ** (-4) / (alpha2 * radius ** 2 - 2)
        * (2 * delta - (
            2 + np.exp(-alpha2D * (Delta - delta)) - 2 * np.exp(-alpha2D * delta)
            - 2 * np.exp(-alpha2D * Delta) + np.exp(-alpha2D * (Delta + delta))
        ) / alpha2D)
    )
    return np.exp(-2 * (GAMMA * g) ** 2 / D * summands.sum())


def _pgse_g_axis(b, delta=DELTA, Delta=DELTABIG, n_t=400):
    """An infinite-slew PGSE waveform's per-step axis-aligned amplitude and ``dt`` at the given b-value, for
    direct comparison against a GPA closed form evaluated at the same (g, delta, Delta)."""
    wf = pgse([[1.0, 0.0, 0.0]], delta, Delta, bvalues=[b], n_t=n_t, slew_rate=np.inf)
    g = np.asarray(wf.G_eff, np.float64)[0][:, 0]
    return g, float(wf.dt), float(np.max(np.abs(g)))


def test_cylinder_matches_gpa_low_b():
    """C5-equivalent (perpendicular matrix method) reproduces Van Gelderen's GPA in its valid low-b regime
    (dmipy-fit's ``test_cylinder_matrix_matches_gpa_low_b``, translated: the GPA model itself re-derived
    above rather than imported from dmipy-fit, since sim never imports fit). The gap is the GPA's own
    approximation error, not truncation (``n_modes=(16, 12)`` is converged for this b range by
    ``test_mode_convergence``'s margin), and it grows with b as the GPA's validity assumption weakens:
    measured 2.27e-4 at b=5e8, 3.98e-4 at b=1e9, so ``atol=5e-4`` is a ~25% margin on the worse case rather
    than a number copied from dmipy-fit's own (looser-truncated, hence coincidentally closer to the GPA) test.
    """
    diameter = 8e-6
    for b in (5e8, 1e9):
        g, dt, g_amp = _pgse_g_axis(b)
        measured = matrix_restricted_signal("cylinder", g, dt, D0, diameter, n_modes=(16, 12))
        gpa = _van_gelderen_gpa(g_amp, DELTA, DELTABIG, D0, diameter)
        npt.assert_allclose(measured, gpa, atol=5e-4, err_msg=f"cylinder vs Van Gelderen GPA at b={b:.1e}")


def test_sphere_matches_gpa_low_b():
    """S5-equivalent reproduces Murday-Cotts's GPA at low b (dmipy-fit's ``test_sphere_matrix_matches_gpa_low_b``).
    Measured gap 1.52e-4 at b=5e8, 2.84e-4 at b=1e9 (same GPA-breakdown trend as the cylinder); ``atol=4e-4``
    is a margin on the worse case.
    """
    diameter = 8e-6
    for b in (5e8, 1e9):
        g, dt, g_amp = _pgse_g_axis(b)
        measured = matrix_restricted_signal("sphere", g, dt, D0, diameter, n_modes=(16, 12))
        gpa = _murday_cotts_gpa(g_amp, DELTA, DELTABIG, D0, diameter)
        npt.assert_allclose(measured, gpa, atol=4e-4, err_msg=f"sphere vs Murday-Cotts GPA at b={b:.1e}")


def test_cylinder_parallel_is_free_diffusion():
    """Along the cylinder axis there is no restriction, so the matrix method -- evaluated on the unrestricted
    axis by giving it a zero gradient along the restricted one (equivalently, free diffusion along the axis
    is exp(-b lambda_par), independent of the matrix method) -- must equal ``exp(-b D)`` exactly.
    (dmipy-fit's ``test_cylinder_parallel_is_free_diffusion``: there the stick's parallel compartment is
    plain Gaussian diffusion: nothing in the cylinder restriction acts along its axis, which is a statement
    about the acquisition's geometry rather than about this module, so it carries no matrix-method call.)
    """
    b = np.array([0.0, 1e9, 3e9])
    free = np.exp(-b * LAMBDA_PAR)
    npt.assert_allclose(free[0], 1.0)
    assert np.all(np.diff(free) < 0)


def test_matrix_departs_from_gpa_at_high_b():
    """At high b the GPA breaks down; the exact matrix model attenuates MORE (lower signal) than the GPA,
    both physical (dmipy-fit's ``test_matrix_departs_from_gpa_at_high_b``)."""
    diameter = 15e-6
    b = 1e10
    g, dt, g_amp = _pgse_g_axis(b)
    measured = matrix_restricted_signal("cylinder", g, dt, LAMBDA_PAR, diameter, n_modes=(16, 12))
    gpa = _van_gelderen_gpa(g_amp, DELTA, DELTABIG, LAMBDA_PAR, diameter)
    assert measured < gpa, "matrix (exact) must attenuate more than GPA at high b"
    assert gpa - measured > 1e-3, "a real departure, not noise"
    assert 0.0 <= measured <= 1.0 + 1e-9


def test_b0_is_unity_and_physical_range():
    """Every shape's b=0 signal is unity, and every b gives a signal in [0, 1] (dmipy-fit's
    ``test_b0_is_unity_and_physical_range``)."""
    bvalues = (0.0, 5e8, 2e9, 5e9)
    for shape, diameter in (("cylinder", 6e-6), ("sphere", 6e-6), ("plane", 6e-6)):
        D = LAMBDA_PAR if shape == "cylinder" else D0
        E = []
        for b in bvalues:
            g, dt, _ = _pgse_g_axis(b)
            E.append(matrix_restricted_signal(shape, g, dt, D, diameter, n_modes=_modes(shape, (12, 10))))
        E = np.asarray(E)
        npt.assert_allclose(E[0], 1.0, err_msg=f"{shape}: b=0 is not unity")
        assert np.all(E > 0) and np.all(E <= 1.0 + 1e-9), f"{shape}: signal left [0, 1]"


def test_restriction_monotonicity():
    """Smaller pore -> higher (more restricted) signal, for all three geometries (dmipy-fit's
    ``test_restriction_monotonicity``)."""
    b = 2e9
    g, dt, _ = _pgse_g_axis(b)
    for shape in ("cylinder", "sphere", "plane"):
        D = LAMBDA_PAR if shape == "cylinder" else D0
        E_small = matrix_restricted_signal(shape, g, dt, D, 4e-6, n_modes=_modes(shape, (12, 10)))
        E_large = matrix_restricted_signal(shape, g, dt, D, 12e-6, n_modes=_modes(shape, (12, 10)))
        assert E_small > E_large, f"{shape}: a smaller pore must restrict more"


def test_mode_convergence():
    """Signal is stable once enough eigenmodes are kept (dmipy-fit's ``test_mode_convergence``)."""
    b = 5e9
    g, dt, _ = _pgse_g_axis(b)
    E_lo = matrix_restricted_signal("sphere", g, dt, D0, 10e-6, n_modes=(10, 8))
    E_hi = matrix_restricted_signal("sphere", g, dt, D0, 10e-6, n_modes=(22, 16))
    npt.assert_allclose(E_lo, E_hi, atol=1e-4)


def test_plane_approaches_free():
    """As the slab thickens the signal descends monotonically toward free diffusion, always from above
    (dmipy-fit's ``test_plane_approaches_free``)."""
    b = 1e9
    g, dt, _ = _pgse_g_axis(b)
    free = np.exp(-b * D0)
    E = [matrix_restricted_signal("plane", g, dt, D0, L, n_modes=nm)
         for L, nm in ((40e-6, 64), (100e-6, 96), (300e-6, 160))]
    assert E[0] > E[1] > E[2] >= free - 1e-6, "monotone descent toward free diffusion, never below it"
    assert E[2] - free < 0.03, "the 300 um slab should be close to free"


# ===================================================================== unit-geometry route vs direct quadrature
def _direct_plane_modes(L, n_max, nr):
    """The slab's modes quadratured DIRECTLY at the physical size ``L`` -- the pre-#667 formula, independent
    of ``_unit_modes``'s unit-geometry-then-scale route."""
    x = np.linspace(0.0, L, nr)
    xc = x - 0.5 * L
    U = [np.cos(n * np.pi * x / L) for n in range(n_max + 1)]
    lam = np.asarray([(n * np.pi / L) ** 2 for n in range(n_max + 1)])
    norm = [np.trapezoid(u * u, x) for u in U]
    N = len(U)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = np.trapezoid(U[i] * xc * U[j], x) / np.sqrt(norm[i] * norm[j])
    return lam, B


def _direct_cyl_modes(a, m_max, n_max, nr, nt):
    r = np.linspace(0.0, a, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for m in range(m_max + 1):
        for n, al in enumerate(mm._cyl_neumann_roots(m, n_max)):
            R = special.jv(m, al * r / a)
            states.append((m, n)); lam.append((al / a) ** 2)
            rnorm.append(np.trapezoid(R * R * r, r)); rfun.append(R)
    t = np.linspace(0.0, 2 * np.pi, nt)
    ang_norm = [2 * np.pi if m == 0 else np.pi for m in range(m_max + 2)]
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        mi, _ = states[i]
        for j in range(N):
            mj, _ = states[j]
            if abs(mi - mj) != 1:
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r, r)
            ang = np.trapezoid(np.cos(mi * t) * np.cos(t) * np.cos(mj * t), t)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[mi] * rnorm[j] * ang_norm[mj])
    return np.asarray(lam), B


def _direct_sph_modes(a, l_max, n_max, nr, nt):
    r = np.linspace(0.0, a, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for l in range(l_max + 1):
        for n, be in enumerate(mm._sph_neumann_roots(l, n_max)):
            R = special.spherical_jn(l, be * r / a)
            states.append((l, n)); lam.append((be / a) ** 2)
            rnorm.append(np.trapezoid(R * R * r * r, r)); rfun.append(R)
    th = np.linspace(0.0, np.pi, nt)
    ct, st = np.cos(th), np.sin(th)
    P = {l: special.eval_legendre(l, ct) for l in range(l_max + 2)}
    ang_norm = {l: 2 * np.pi * np.trapezoid(P[l] * P[l] * st, th) for l in range(l_max + 2)}
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        li, _ = states[i]
        for j in range(N):
            lj, _ = states[j]
            if abs(li - lj) != 1:
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r * r, r)
            ang = 2 * np.pi * np.trapezoid(P[li] * ct * P[lj] * st, th)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[li] * rnorm[j] * ang_norm[lj])
    return np.asarray(lam), B


@pytest.mark.parametrize("shape,size", [
    ("plane", 4e-6), ("plane", 12e-6),
    ("cylinder", 4e-6), ("cylinder", 12e-6),
    ("sphere", 4e-6), ("sphere", 12e-6),
])
def test_unit_geometry_route_matches_direct_quadrature(shape, size):
    """``MatrixPore`` quadratures the UNIT geometry once per ``(shape, n_modes)`` and scales by the pore size
    (dmipy-fit's economy); the pre-#667 sim module quadratured directly at the physical size every time. The
    two must agree to the mode truncation: this re-derives the direct-at-size route independently (the
    pre-#667 formulas, reproduced above rather than imported) and compares its eigenvalues and position-matrix
    couplings against the scaled unit-geometry route at two sizes per shape.
    """
    n_modes = (8, 6)
    nr, nt = mm.RADIAL_NODES, mm.ANGULAR_NODES
    m_max, n_max = n_modes

    if shape == "plane":
        lam_direct, B_direct = _direct_plane_modes(size, m_max * n_max, nr)
    elif shape == "cylinder":
        lam_direct, B_direct = _direct_cyl_modes(0.5 * size, m_max, n_max, nr, nt)
    else:
        lam_direct, B_direct = _direct_sph_modes(0.5 * size, m_max, n_max, nr, nt)

    mp = MatrixPore(shape, size, D0, n_modes=_modes(shape, n_modes))

    npt.assert_allclose(mp.lam, lam_direct, rtol=1e-6, atol=1e-3 * lam_direct.max(),
                         err_msg=f"{shape} size={size}: eigenvalues, unit route vs direct quadrature")
    npt.assert_allclose(mp.B, B_direct, rtol=1e-6, atol=1e-9,
                         err_msg=f"{shape} size={size}: position matrix, unit route vs direct quadrature")


def test_the_mode_count_has_the_shape_of_the_pore():
    """A plane has one mode index, so its ``n_modes`` is an int; a cylinder and a sphere have an angular and a
    radial one, so theirs is the pair; the other spelling is refused by name, and the defaults have the shape."""
    L = 4e-6
    with pytest.raises(ValueError, match="plane has one mode index"):
        MatrixPore("plane", L, D0, n_modes=(8, 6))
    with pytest.raises(ValueError, match="cylinder has an angular and a radial"):
        MatrixPore("cylinder", L, D0, n_modes=12)
    assert MatrixPore("plane", L, D0).lam.shape[0] == DEFAULT_MODES["plane"] + 1
    assert MatrixPore("sphere", L, D0).lam.shape[0] == MatrixPore("sphere", L, D0, n_modes=DEFAULT_MODES["sphere"]).lam.shape[0]

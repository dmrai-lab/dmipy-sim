"""Susceptibility-aware FOD composition (paper Sec. 2.6): Gaunt table + two-axis contraction.

The tests that matter here are the ones a cursory suite would miss.  A non-orthonormal
spherical-harmonic basis breaks the addition theorem the contraction rests on, but the
resulting error *vanishes exactly when g is parallel to B0* -- so every geometry test below
is run off-axis and skew as well as aligned.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import eval_legendre

from dmipy_sim.gaunt import (real_sh, sphere_quadrature, assert_orthonormal,
                             gaunt_table, n_sh_coeffs, sh_block)
from dmipy_sim.sh_convolution import (apply_odf, apply_odf_coupled, coupled_spectrum,
                                      invariant_grid, rank1_residual, isotropic_odf_sh)

LMAX, LG, LB = 8, 8, 6
GEOMS = [("aligned", [0, 0, 1.], [0, 0, 1.]),
         ("oblique", [np.sin(0.96), 0, np.cos(0.96)], [0, 0, 1.]),
         ("skew", [.6, .5, .62], [-.3, .8, .52])]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_basis_is_orthonormal():
    dirs, w = sphere_quadrature(32, 64)
    assert assert_orthonormal(real_sh(LMAX, dirs), w) < 1e-10


def test_assert_orthonormal_rejects_a_bad_basis():
    """The guardrail must actually fire -- rescale m!=0 like the legacy MRtrix basis."""
    dirs, w = sphere_quadrature(32, 64)
    Y = real_sh(LMAX, dirs).copy()
    for l in range(2, LMAX + 1, 2):
        blk = sh_block(l)
        cols = np.arange(blk.start, blk.stop)
        Y[:, cols[cols != blk.start + l]] /= np.sqrt(2.0)      # m != 0
    with pytest.raises(ValueError, match="not orthonormal"):
        assert_orthonormal(Y, w)


def test_gaunt_matches_closed_form_monopole():
    G = gaunt_table(4, 4, 4)
    npt.assert_allclose(G[0, 0, 0], 1.0 / (2 * np.sqrt(np.pi)), rtol=1e-12)


def test_gaunt_is_sparse_by_selection_rules():
    G = gaunt_table(LMAX, LG, LB)
    assert np.count_nonzero(G) / G.size < 0.15
    # parity: l1 + l2 + L odd must vanish (all even here, so check a triangle violation)
    L, l1, l2 = 8, 2, 2                       # |l1-l2|=0 <= L=8 > l1+l2=4 -> forbidden
    assert np.abs(G[sh_block(L), sh_block(l1), sh_block(l2)]).max() < 1e-12


@pytest.mark.parametrize("name,g,b", GEOMS)
def test_contraction_equals_brute_force_sphere_integral(name, g, b):
    """A deliberately NON-separable Lambda, against direct quadrature on the sphere."""
    rng = np.random.default_rng(0)
    n1, n2 = LG // 2 + 1, LB // 2 + 1
    lam = (rng.standard_normal((n1, n2))
           * np.exp(-0.8 * np.arange(n1))[:, None]
           * np.exp(-0.8 * np.arange(n2))[None, :])
    lam[0, 0] = 1.0
    assert rank1_residual(lam) > 0.05, "test fixture must not be separable"

    f = rng.standard_normal(n_sh_coeffs(LMAX)) * 0.3
    f[0] = 1.0 / (2 * np.sqrt(np.pi))
    g, b = _unit(g), _unit(b)

    dirs, w = sphere_quadrature(48, 96)
    F = real_sh(LMAX, dirs) @ f
    E = sum(lam[i, j] * eval_legendre(2 * i, dirs @ g) * eval_legendre(2 * j, dirs @ b)
            for i in range(n1) for j in range(n2))
    brute = np.sum(w * F * E)
    got = apply_odf_coupled(lam, f, g, b, l_fod=LMAX, l_g=LG, l_b=LB)
    npt.assert_allclose(got, brute, rtol=1e-10)


@pytest.mark.parametrize("name,g,b", GEOMS)
def test_no_susceptibility_reduces_to_ordinary_convolution(name, g, b):
    """Lambda supported on l2=0 must reproduce the single-axis Funk-Hecke path."""
    rng = np.random.default_rng(1)
    n1 = LG // 2 + 1
    a = rng.standard_normal(n1) * np.exp(-0.7 * np.arange(n1))
    a[0] = 1.0
    lam = np.zeros((n1, LB // 2 + 1))
    lam[:, 0] = a                                    # P_0(v) = 1  ->  no B0 dependence

    f = rng.standard_normal(n_sh_coeffs(LMAX)) * 0.3
    f[0] = 1.0 / (2 * np.sqrt(np.pi))
    g, b = _unit(g), _unit(b)

    dirs, w = sphere_quadrature(48, 96)
    F = real_sh(LMAX, dirs) @ f
    E = sum(a[i] * eval_legendre(2 * i, dirs @ g) for i in range(n1))
    brute = np.sum(w * F * E)
    got = apply_odf_coupled(lam, f, g, b, l_fod=LMAX, l_g=LG, l_b=LB)
    npt.assert_allclose(got, brute, rtol=1e-10)


def test_matches_apply_odf_on_the_z_axis():
    """The l2=0 contraction must equal the existing apply_odf for a z-aligned gradient."""
    rng = np.random.default_rng(2)
    n1 = LG // 2 + 1
    a = rng.standard_normal(n1) * np.exp(-0.7 * np.arange(n1)); a[0] = 1.0
    lam = np.zeros((n1, LB // 2 + 1)); lam[:, 0] = a
    f = rng.standard_normal(n_sh_coeffs(LMAX)) * 0.3
    f[0] = 1.0 / (2 * np.sqrt(np.pi))
    got = apply_odf_coupled(lam, f, [0, 0, 1.], [0, 0, 1.], l_fod=LMAX, l_g=LG, l_b=LB)
    ref = apply_odf(a, f, lmax=LMAX)
    npt.assert_allclose(got, ref, rtol=1e-10)


@pytest.mark.parametrize("name,g,b", GEOMS)
def test_uniform_fod_leaves_the_diagonal_coupling(name, g, b):
    """A uniform F does NOT reduce the response to Lambda_00.

    By the addition theorem ``int P_l1(n.g) P_l2(n.B0) dn = delta_{l1l2} 4pi/(2l+1)
    P_l(g.B0)``, so a uniform orientation distribution leaves

        S = sum_l Lambda_{ll} P_l(g.B0) / (2l+1),

    i.e. the diagonal of the coupled spectrum, weighted by the angle between the gradient
    and the field.  Only the separate spherical mean over *gradient directions* collapses
    the susceptibility weight to a scalar.
    """
    rng = np.random.default_rng(3)
    n1, n2 = LG // 2 + 1, LB // 2 + 1
    lam = rng.standard_normal((n1, n2)) * 0.2
    lam[0, 0] = 1.0
    g, b = _unit(g), _unit(b)
    got = apply_odf_coupled(lam, isotropic_odf_sh(LMAX), g, b,
                            l_fod=LMAX, l_g=LG, l_b=LB)
    expect = sum(lam[i, i] * eval_legendre(2 * i, g @ b) / (4 * i + 1)
                 for i in range(min(n1, n2)))
    npt.assert_allclose(got, expect, rtol=1e-10)


def test_coupled_spectrum_round_trips():
    """Projecting a known Lambda off its own (u, v) samples must recover it."""
    rng = np.random.default_rng(4)
    n1, n2 = LG // 2 + 1, LB // 2 + 1
    lam = rng.standard_normal((n1, n2)) * np.exp(-0.6 * np.arange(n1))[:, None]
    u, v, _, _ = invariant_grid(24, 24)
    E = sum(lam[i, j] * eval_legendre(2 * i, u)[:, None] * eval_legendre(2 * j, v)[None, :]
            for i in range(n1) for j in range(n2))
    npt.assert_allclose(coupled_spectrum(E, u, v, l_g=LG, l_b=LB), lam, atol=1e-10)


def test_rank1_residual_detects_separability():
    a = np.array([1.0, 0.4, 0.1, 0.02, 0.005])
    b = np.array([1.0, 0.3, 0.05, 0.01])
    assert rank1_residual(np.outer(a, b)) < 1e-12
    mixed = np.outer(a, b); mixed[2, 3] += 0.5
    assert rank1_residual(mixed) > 0.05


# --- Lambda built at the exact requested geometry (scanner-side, no interpolation) -------

def _phys_response(g, b):
    """A response that is a genuine function of (n.g, n.B0) and is NOT separable, and is not
    built in the product-Legendre basis -- so projecting it is not a circular test."""
    def f(dirs):
        u, v = dirs @ g, dirs @ b
        return np.exp(-1.3 * u ** 2) * np.exp(-0.65 * (1 - v ** 2) ** 2) \
            * (1 + 0.35 * u ** 2 * v ** 2)
    return f


@pytest.mark.parametrize("ang", [0.0, 0.35, 0.79, 1.20, np.pi / 2, 2.36, np.pi])
def test_lambda_built_at_requested_geometry_reproduces_the_integral(ang):
    from dmipy_sim.sh_convolution import coupled_spectrum_at
    g = np.array([0, 0, 1.0])
    b = np.array([np.sin(ang), 0, np.cos(ang)])
    lam, resid, rank = coupled_spectrum_at(_phys_response(g, b), g, b, l_g=LG, l_b=LB)

    dirs, w = sphere_quadrature(64, 128)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(n_sh_coeffs(LMAX)) * 0.3
    f[0] = 1.0 / (2 * np.sqrt(np.pi))
    brute = np.sum(w * (real_sh(LMAX, dirs) @ f) * _phys_response(g, b)(dirs))
    got = apply_odf_coupled(lam, f, g, b, l_fod=LMAX, l_g=LG, l_b=LB)
    # the composition error tracks the projection residual, which is the truncation
    npt.assert_allclose(got, brute, rtol=max(20 * resid, 1e-9))


def test_product_basis_is_rank_deficient_when_g_parallel_b0():
    """u == v collapses the family -- a common acquisition geometry, so lstsq not solve."""
    from dmipy_sim.sh_convolution import coupled_spectrum_at
    g = b = np.array([0, 0, 1.0])
    lam, resid, rank = coupled_spectrum_at(_phys_response(g, b), g, b, l_g=LG, l_b=LB,
                                           chiral=False)
    assert rank < len(lam["terms"]), "expected degeneracy at g || B0"
    assert resid < 1e-6, "minimum-norm solution must still reconstruct the response"


# --- wiring to a real pack ---------------------------------------------------------------

_WINTHER = "/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk"


@pytest.mark.skipif(not __import__("os").path.exists(_WINTHER),
                    reason="needs the Winther susceptibility pack")
def test_pack_response_runs_in_coefficient_space_and_is_orientation_dependent():
    """The pack bridge must never reconstruct a trajectory, and must show real anisotropy."""
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.sh_convolution import pack_response
    pk = read_rpk(_WINTHER)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_t, dt = int(pm["n_t"]), pk.dt
    t = np.arange(n_t) * dt; T = n_t * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    r = pack_response(pk, prof, [1, 0, 0], [0, 0, 1], amplitude=0.05, B0=3.0,
                      chi_iso=-9.4e-6, chi_aniso=-1.0e-7, refocus_time=0.5 * T)
    par = abs(r(np.array([[1., 0, 0]]))[0])      # fibre along the gradient -> free -> low
    perp = abs(r(np.array([[0, 0, 1.]]))[0])     # fibre across it -> restricted -> high
    assert perp > par, f"expected restriction perpendicular to the fibre ({perp} vs {par})"


@pytest.mark.skipif(not __import__("os").path.exists(_WINTHER),
                    reason="needs the Winther susceptibility pack")
def test_real_axon_is_chiral_so_two_invariants_do_not_span_it():
    """A realistic axon is NOT mirror-symmetric, which bounds the (u, v) model.

    E can depend on ``n`` through ``n.g`` and ``n.B0`` alone only if the substrate is achiral;
    otherwise it also sees the triple product ``n.(g x B0)``, which flips sign under mirroring
    and which no function of the two invariants can represent.  This pins the effect rather
    than the fit, so a later change to the projection cannot quietly hide it.
    """
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.sh_convolution import pack_response
    pk = read_rpk(_WINTHER)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_t, dt = int(pm["n_t"]), pk.dt
    t = np.arange(n_t) * dt; T = n_t * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    n = np.array([[0, 0, 1.0]])
    thg, thb, dphi = 1.3, 0.7, 2.0
    g = np.array([np.sin(thg), 0, np.cos(thg)])
    kw = dict(amplitude=0.05, B0=3.0, chi_iso=-9.4e-6, chi_aniso=-1.0e-7,
              refocus_time=0.5 * T, chunk=8)
    out = []
    for sgn in (+1, -1):
        b = np.array([np.sin(thb) * np.cos(dphi), sgn * np.sin(thb) * np.sin(dphi),
                      np.cos(thb)])
        out.append(abs(pack_response(pk, prof, g, b, **kw)(n)[0]))
    # same (u, v, c), opposite triple product -- a mirror pair
    assert abs(out[0] - out[1]) / np.mean(out) > 1e-2


@pytest.mark.skipif(not __import__("os").path.exists(_WINTHER),
                    reason="needs the Winther susceptibility pack")
def test_chiral_sector_closes_the_gap_on_a_real_axon():
    """The (u, v) family plateaus on a handed substrate; adding n.(g x B0) closes it.

    Both the projection residual and the composed signal are checked, because a better fit
    that did not improve the composition would mean the extra sector was absorbing noise.
    """
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.sh_convolution import pack_response, coupled_spectrum_at, watson_odf_sh
    pk = read_rpk(_WINTHER)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_t, dt = int(pm["n_t"]), pk.dt
    t = np.arange(n_t) * dt; T = n_t * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    g = np.array([np.sin(0.96), 0, np.cos(0.96)])
    b = np.array([0, 0, 1.0])
    resp = pack_response(pk, prof, g, b, amplitude=0.05, B0=3.0, chi_iso=-9.4e-6,
                        chi_aniso=-1.0e-7, refocus_time=0.5 * T, chunk=256)
    # the FOD must be TILTED off B0: an orientation distribution axisymmetric about the field
    # annihilates the w-odd part, so an on-axis Watson would pass this test without testing it
    f = watson_odf_sh(6.0, mu=[0.6, 0.5, 0.62], lmax=LMAX)
    dirs, w = sphere_quadrature(32, 64)
    brute = np.sum(w * (real_sh(LMAX, dirs) @ f) * np.real(resp(dirs)))

    out = {}
    for chiral in (False, True):
        lam, resid, _ = coupled_spectrum_at(resp, g, b, l_g=8, l_b=6, n_theta=32, n_phi=64,
                                            chiral=chiral)
        got = np.real(apply_odf_coupled(lam, f, g, b, l_fod=LMAX, l_g=8, l_b=6))
        out[chiral] = (resid, abs(got - brute) / abs(brute))
    assert out[True][0] < 0.6 * out[False][0], f"chiral sector must cut the residual: {out}"
    assert out[True][1] < 0.05 * out[False][1], f"and the composition error with it: {out}"


@pytest.mark.skipif(not __import__("os").path.exists(_WINTHER),
                    reason="needs the Winther susceptibility pack")
def test_chiral_sector_is_inert_for_a_field_symmetric_fod():
    """An FOD axisymmetric about B0 kills the w-odd part -- the sector must cost nothing there."""
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.sh_convolution import pack_response, coupled_spectrum_at, watson_odf_sh
    pk = read_rpk(_WINTHER)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_t, dt = int(pm["n_t"]), pk.dt
    t = np.arange(n_t) * dt; T = n_t * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    g = np.array([np.sin(0.96), 0, np.cos(0.96)]); b = np.array([0, 0, 1.0])
    resp = pack_response(pk, prof, g, b, amplitude=0.05, B0=3.0, chi_iso=-9.4e-6,
                        chi_aniso=-1.0e-7, refocus_time=0.5 * T, chunk=256)
    f = watson_odf_sh(6.0, mu=[0, 0, 1.], lmax=LMAX)          # symmetric about B0
    vals = []
    for chiral in (False, True):
        lam, _, _ = coupled_spectrum_at(resp, g, b, l_g=8, l_b=6, n_theta=32, n_phi=64,
                                        chiral=chiral)
        vals.append(np.real(apply_odf_coupled(lam, f, g, b, l_fod=LMAX, l_g=8, l_b=6)))
    npt.assert_allclose(vals[1], vals[0], rtol=1e-6)

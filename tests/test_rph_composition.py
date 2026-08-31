"""Replay phantom composition: multi-substrate voxels, peaks mode, and the SH basis.

These pin the parts of RPH.md that a phantom built elsewhere would get wrong: several
substrates in one voxel with independent orientations and proton densities, up to three
crossing peaks, and the spherical-harmonic convention the composition depends on.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import eval_legendre

from dmipy_sim.gaunt import sphere_quadrature, real_sh, sh_block, n_sh_coeffs
from dmipy_sim.sh_convolution import apply_odf_coupled, coupled_spectrum_at, watson_odf_sh

LMAX, LG, LB = 8, 8, 6
G = np.array([np.sin(0.96), 0.0, np.cos(0.96)])
B0 = np.array([0.0, 0.0, 1.0])


def _resp(g, b, seed=0):
    """A non-separable, chiral two-axis response -- the general case, not a zonal product."""
    k = np.cross(g, b)
    nk = np.linalg.norm(k)
    k = k / nk if nk > 1e-12 else None          # no chiral axis when g is parallel to B0

    def f(dirs):
        u, v = dirs @ g, dirs @ b
        out = (np.exp(-1.1 * u ** 2) * np.exp(-0.5 * (1 - v ** 2) ** 2)
               * (1 + 0.3 * u ** 2 * v ** 2))
        return out if k is None else out + 0.15 * (dirs @ k) * u
    return f


@pytest.fixture(scope="module")
def lam():
    lm, resid, _ = coupled_spectrum_at(_resp(G, B0), G, B0, l_g=LG, l_b=LB)
    # the fixture is smooth but not band-limited, so a small residual is expected; the
    # composition tests below compare against brute force and inherit it
    assert resid < 1e-3, f"fixture response should be near-representable, got {resid}"
    return lm


def _brute(fod_sh, resp):
    dirs, w = sphere_quadrature(48, 96)
    return float(np.sum(w * (real_sh(LMAX, dirs) @ fod_sh) * resp(dirs)))


# --- multi-substrate voxels ---------------------------------------------------------------

def test_multi_substrate_voxel_is_the_weighted_sum_of_its_slots(lam):
    """RPH.md Sec. 6: S_v = sum_p geometric_fraction * m0 * <F_p, E>."""
    mus = [[0, 0, 1.], [1, 0, 0.], [0.6, 0.5, 0.62]]
    fracs = np.array([0.45, 0.30, 0.15])          # sums to 0.90; remainder unmodelled
    m0 = np.array([0.70, 0.85, 1.00])             # WM / GM / CSF-like proton densities
    fods = [watson_odf_sh(6.0, mu=m, lmax=LMAX) for m in mus]

    voxel = sum(f * m * apply_odf_coupled(lam, sh, G, B0, l_fod=LMAX, l_g=LG, l_b=LB)
                for f, m, sh in zip(fracs, m0, fods))
    expect = sum(f * m * _brute(sh, _resp(G, B0)) for f, m, sh in zip(fracs, m0, fods))
    npt.assert_allclose(np.real(voxel), expect, rtol=5e-3)   # tracks the fixture residual


def test_fractions_are_not_renormalised(lam):
    """Partial volume is only expressible if a short row stays short.

    A voxel that is 60% WM and 30% GM must return 90% of what the same voxel returns when the
    two fill it entirely.  Renormalising would turn every tissue boundary into pure tissue.
    """
    from dmipy_sim.sh_convolution import compose_voxel
    sh = np.stack([watson_odf_sh(6.0, mu=m, lmax=LMAX) for m in ([0, 0, 1.], [1, 0, 0.])])
    m0 = np.array([0.70, 0.85])
    sid = np.array([0, 1])
    kw = dict(l_fod=LMAX, l_g=LG, l_b=LB)

    short = compose_voxel([lam, lam], sid, [0.6, 0.3], sh, m0, G, B0, **kw)
    full = compose_voxel([lam, lam], sid, [2 / 3, 1 / 3], sh, m0, G, B0, **kw)
    npt.assert_allclose(np.real(short), 0.9 * np.real(full), rtol=1e-12)
    assert abs(np.real(short) - np.real(full)) > 1e-3 * abs(np.real(full))


def test_fraction_row_over_one_is_rejected_not_rescaled(lam):
    from dmipy_sim.sh_convolution import compose_voxel
    sh = np.stack([watson_odf_sh(6.0, mu=m, lmax=LMAX) for m in ([0, 0, 1.], [1, 0, 0.])])
    with pytest.raises(ValueError, match="not renormalised"):
        compose_voxel([lam, lam], np.array([0, 1]), [0.7, 0.5], sh,
                      np.array([0.7, 0.85]), G, B0, l_fod=LMAX, l_g=LG, l_b=LB)


def test_empty_slots_are_skipped(lam):
    """substrate_id = -1 marks an unused slot and must contribute nothing."""
    from dmipy_sim.sh_convolution import compose_voxel
    sh = np.stack([watson_odf_sh(6.0, mu=m, lmax=LMAX)
                   for m in ([0, 0, 1.], [1, 0, 0.], [0, 1, 0.])])
    m0 = np.array([0.70, 0.85])
    kw = dict(l_fod=LMAX, l_g=LG, l_b=LB)
    two = compose_voxel([lam, lam], np.array([0, 1]), [0.6, 0.3], sh[:2], m0, G, B0, **kw)
    pad = compose_voxel([lam, lam], np.array([0, 1, -1]), [0.6, 0.3, 0.0], sh, m0, G, B0, **kw)
    npt.assert_allclose(np.real(pad), np.real(two), rtol=1e-12)


def test_same_substrate_cited_twice_is_a_crossing(lam):
    """Two slots may cite one substrate at different poses -- a pure crossing."""
    a = watson_odf_sh(20.0, mu=[0, 0, 1.], lmax=LMAX)
    b = watson_odf_sh(20.0, mu=[1, 0, 0.], lmax=LMAX)
    crossing = 0.5 * apply_odf_coupled(lam, a, G, B0, l_fod=LMAX, l_g=LG, l_b=LB) \
        + 0.5 * apply_odf_coupled(lam, b, G, B0, l_fod=LMAX, l_g=LG, l_b=LB)
    both = apply_odf_coupled(lam, 0.5 * (a + b), G, B0, l_fod=LMAX, l_g=LG, l_b=LB)
    npt.assert_allclose(crossing, both, rtol=1e-10)     # composition is linear in the ODF


# --- peaks mode ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_peaks", [1, 2, 3])
def test_peaks_agree_with_concentrated_odfs(lam, n_peaks):
    """RPH.md Sec. 4: a peak is the zero-dispersion limit; both routes must agree."""
    dirs = np.array([[0, 0, 1.], [1, 0, 0.], [0.6, 0.5, 0.62]])[:n_peaks]
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    wts = np.full(n_peaks, 1.0 / n_peaks)
    resp = _resp(G, B0)
    peaks = sum(w * resp(d[None, :])[0] for w, d in zip(wts, dirs))
    odf = sum(w * watson_odf_sh(400.0, mu=d, lmax=LMAX) for w, d in zip(wts, dirs))
    viaodf = np.real(apply_odf_coupled(lam, odf, G, B0, l_fod=LMAX, l_g=LG, l_b=LB))
    # agreement is limited by the SH truncation of a near-delta, not by the composition
    npt.assert_allclose(viaodf, peaks, rtol=0.05)


def test_three_peaks_are_distinguishable(lam):
    """Three crossings must not collapse to the same number -- the test above would pass on a
    degenerate response that ignored orientation."""
    resp = _resp(G, B0)
    vals = [resp(d[None, :])[0] for d in np.eye(3)]
    assert max(vals) - min(vals) > 1e-3 * abs(np.mean(vals))


# --- the SH basis (RPH.md Sec. 4.1) --------------------------------------------------------

def _dipy(name, lmax, dirs):
    from dipy.reconst.shm import real_sh_tournier, real_sh_descoteaux
    th = np.arccos(np.clip(dirs[:, 2], -1, 1)); ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    if name == "mrtrix":
        return real_sh_tournier(lmax, th, ph, legacy=True)[0]
    if name == "tournier":
        return real_sh_tournier(lmax, th, ph, legacy=False)[0]
    return real_sh_descoteaux(lmax, th, ph, legacy=False)[0]


def test_required_basis_is_dipy_tournier_non_legacy():
    dirs, w = sphere_quadrature(48, 96)
    npt.assert_allclose(real_sh(LMAX, dirs), _dipy("tournier", LMAX, dirs), atol=1e-12)


def test_mrtrix_basis_is_not_orthonormal_and_converts_by_a_scale():
    """c_required = c_mrtrix / sqrt(2) for m != 0 (RPH.md table)."""
    dirs, w = sphere_quadrature(48, 96)
    Ym = _dipy("mrtrix", LMAX, dirs)
    gram = np.einsum("qa,qb,q->ab", Ym, Ym, w)
    assert np.abs(gram - np.eye(gram.shape[0])).max() > 0.4, "MRtrix basis is not orthonormal"

    Yo = real_sh(LMAX, dirs)
    rng = np.random.default_rng(0)
    f = Yo @ (rng.standard_normal(Yo.shape[1]) * 0.3)
    c_req = np.einsum("qa,q,q->a", Yo, f, w)
    c_mrt = np.linalg.lstsq(Ym * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
    scale = np.ones(Yo.shape[1]) / np.sqrt(2.0)
    for l in range(0, LMAX + 1, 2):
        scale[sh_block(l).start + l] = 1.0                      # m = 0 unchanged
    npt.assert_allclose(c_req, scale * c_mrt, atol=1e-10)


def test_descoteaux_is_orthonormal_but_a_different_basis():
    """Converts by a signed m -> -m permutation, s_m = (-1)^m for m > 0 (RPH.md table)."""
    dirs, w = sphere_quadrature(48, 96)
    Yd = _dipy("descoteaux", LMAX, dirs)
    gram = np.einsum("qa,qb,q->ab", Yd, Yd, w)
    npt.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)   # orthonormal ...

    Yo = real_sh(LMAX, dirs)
    assert np.abs(Yo - Yd).max() > 0.1, "... but not the same basis"

    rng = np.random.default_rng(1)
    f = Yo @ (rng.standard_normal(Yo.shape[1]) * 0.3)
    c_req = np.einsum("qa,q,q->a", Yo, f, w)
    c_des = np.einsum("qa,q,q->a", Yd, f, w)
    perm = np.arange(Yo.shape[1]); sgn = np.ones(Yo.shape[1])
    for l in range(0, LMAX + 1, 2):
        blk = sh_block(l)
        for i, m in enumerate(range(-l, l + 1)):
            perm[blk.start + i] = blk.start + (l - m)
            sgn[blk.start + i] = -1.0 if (m > 0 and m % 2 == 1) else 1.0
    npt.assert_allclose(c_req, sgn * c_des[perm], atol=1e-12)


def test_unconverted_mrtrix_error_hides_when_g_is_parallel_to_b0(lam):
    """Why the convention is normative: the MRtrix scale error VANISHES at g || B0.

    A phantom imported without conversion is wrong by an amount a cursory check would miss,
    because the aligned geometry is the natural thing to test first.
    """
    sh = watson_odf_sh(6.0, mu=[0.6, 0.5, 0.62], lmax=LMAX)
    bad = sh.copy()                                    # as if read straight from MRtrix
    for l in range(0, LMAX + 1, 2):
        blk = sh_block(l)
        cols = np.arange(blk.start, blk.stop)
        bad[cols[cols != blk.start + l]] *= np.sqrt(2.0)

    def rel(g):
        lm, _, _ = coupled_spectrum_at(_resp(g, B0), g, B0, l_g=LG, l_b=LB)
        ok = np.real(apply_odf_coupled(lm, sh, g, B0, l_fod=LMAX, l_g=LG, l_b=LB))
        no = np.real(apply_odf_coupled(lm, bad, g, B0, l_fod=LMAX, l_g=LG, l_b=LB))
        return abs(no - ok) / abs(ok)

    assert rel(np.array([0, 0, 1.0])) < 1e-12, "aligned geometry hides the error"
    assert rel(G) > 1e-3, "off-axis geometry must expose it"

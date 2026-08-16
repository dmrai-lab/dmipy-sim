"""Analytic oracles for the mesh/grid susceptibility field (dmipy_sim.susceptibility_field).

The k-space dipole field is validated against closed-form infinite-cylinder solutions, which is what
pins the ABSOLUTE field amplitude. The lumen null test is the sharp one: for a coaxial (hollow)
cylinder the field inside the axon is EXACTLY zero at every orientation, so any nonzero lumen field is
pure discretization error -- with a hard binary myelin source the non-decaying dipole kernel rings into
the lumen at ~2.6% of chi*B0, which silently inflates intra-axonal dephasing. Partial-volume
occupancy suppresses it to ~0.1%. These tests lock that in.

Geometry here is analytic (no meshes), so the tests are fast and CPU-only.
"""
import numpy as np
import pytest

from dmipy_sim.susceptibility_field import field_basis, assemble_field

CHI, B0 = 1.06e-6, 7.0                      # myelin-water susceptibility contrast, 7T
R_I, R_O = 1.3773e-6, 1.9676e-6             # g-ratio 0.7 coaxial axon
RES, NXY, NZ = 0.131e-6, 256, 6             # the calibrated field resolution


def _grid(nxy=NXY, nz=NZ, res=RES):
    ax = (np.arange(nxy) - (nxy - 1) / 2) * res
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    return X, Y, np.sqrt(X ** 2 + Y ** 2), np.arctan2(Y, X)


def _source(kind, ss, nxy=NXY, nz=NZ, res=RES):
    """Analytic annulus/disc occupancy, partial-volume at supersampling `ss` (exact, analytic)."""
    n = nxy * ss
    f = (np.arange(n) - (n - 1) / 2) * (res / ss)
    Xf, Yf = np.meshgrid(f, f, indexing="ij")
    rf = np.sqrt(Xf ** 2 + Yf ** 2)
    occ = ((rf >= R_I) & (rf < R_O)) if kind == "annulus" else (rf < R_O)
    occ = occ.astype(np.float32).reshape(nxy, ss, nxy, ss).mean(axis=(1, 3))
    return np.repeat(occ[:, :, None], nz, axis=2)


def _field(m, b0_dir, res=RES):
    basis = field_basis(m, np.zeros(m.shape + (3,), np.float32), np.array([res, res, res]),
                        include_aniso=False, kspace_lowpass=None)
    return assemble_field(basis, b0_dir, B0=B0, chi_iso=CHI)[:, :, m.shape[2] // 2]


@pytest.mark.parametrize("theta_deg,expect_factor", [(0.0, 2.0), (90.0, -1.0)])
def test_solid_cylinder_interior_is_uniform_and_analytic(theta_deg, expect_factor):
    """Interior of a SOLID cylinder: dB/B0 = (chi/6)(3cos^2(theta)-1), uniform."""
    m = _source("solid", ss=4)
    t = np.deg2rad(theta_deg)
    dB = _field(m, [np.sin(t), 0.0, np.cos(t)])
    _, _, r, _ = _grid()
    deep = r < 0.6 * R_O
    got = dB[deep].mean() / B0
    analytic = CHI / 6.0 * expect_factor                     # (3cos^2-1) = 2 at 0deg, -1 at 90deg
    assert abs(got - analytic) / abs(analytic) < 0.10, (got, analytic)
    assert dB[deep].std() / abs(dB[deep].mean()) < 0.05      # uniform


def test_coaxial_lumen_null_requires_partial_volume():
    """A coaxial cylinder has EXACTLY zero lumen field. A binary source rings badly; partial-volume
    occupancy must suppress the residual by more than an order of magnitude."""
    _, _, r, _ = _grid()
    lumen = r < 0.9 * R_I
    res = {}
    for ss in (1, 4):
        dB = _field(_source("annulus", ss=ss), [1.0, 0.0, 0.0])
        v = dB[lumen] - dB[lumen].mean()
        res[ss] = v.std() / (CHI * B0)                        # fraction of chi*B0
    assert res[4] < 0.005, f"PV lumen residual too large: {res[4]:.4f}"
    assert res[4] < res[1] / 5.0, f"PV did not suppress ringing: {res}"


def test_coaxial_annulus_matches_analytic_amplitude_and_structure():
    """Inside the sheath at theta=90: dB/B0 = chi[-1/6 - (1/2)(R_i^2/r^2)cos(2phi)].

    The sign on the cos(2phi) term is MINUS, and the comparison below is signed. An annulus is a solid
    cylinder of radius R_o minus one of radius R_i, and inside the sheath we are OUTSIDE the removed core,
    so the subtracted term is that core's EXTERNAL field, +chi/2 (R_i^2/r^2)cos(2phi) -- subtracting it
    leaves a minus. Verified numerically link by link: the solid interior is -0.1649 chi*B0 against the
    textbook -1/6; the core's external field correlates +0.9997 with the 2-D dipole form; and
    annulus == solid - core holds to 2.3e-16.

    This assertion used to compare abs(corr), which cannot tell a field from its negative -- and the
    reference carried a + here, so a perfectly inverted sheath field passed at |corr| 0.9995. The lumen
    null is sign-agnostic (zero has no sign) and the solid-cylinder test only probes a UNIFORM interior,
    so this is the only oracle covering the spatial STRUCTURE of the sheath field, which is what dephases
    intra-axonal spins.
    """
    _, _, r, phi = _grid()
    dB = _field(_source("annulus", ss=4), [1.0, 0.0, 0.0])
    ana = CHI * B0 * (-1 / 6 - 0.5 * (R_I ** 2 / np.maximum(r, 1e-12) ** 2) * np.cos(2 * phi))
    ann = (r >= 1.05 * R_I) & (r < 0.95 * R_O)                # inset off the staircased edge
    a = ana[ann] - ana[ann].mean(); g = dB[ann] - dB[ann].mean()
    assert abs(g.std() / a.std() - 1.0) < 0.05, g.std() / a.std()
    # SIGNED: an inverted field must fail, so no abs() here and the slope must be +1
    assert np.corrcoef(a, g)[0, 1] > 0.99, np.corrcoef(a, g)[0, 1]
    assert abs(np.polyfit(a, g, 1)[0] - 1.0) < 0.05, np.polyfit(a, g, 1)[0]

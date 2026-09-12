"""The closed-form hollow-cylinder basis against the k-space dipole route it was derived from, component by
component and region by region, and the forward-Bloch myelin provider against the same form (its sheath and
outside fields were the docstring's remembered ones before: anticorrelated in the sheath, 6x too large outside)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import dmipy_sim as d
from dmipy_sim.fields.hollow_cylinder import hollow_cylinder_basis, contract, CHANNEL_NAMES, q_of_H
from dmipy_sim.fields.susceptibility_field import field_grid_of, assemble_field
from dmipy_sim.fields.susceptibility import MyelinSusceptibility

A, B = 1.0e-6, 1.4e-6


@pytest.fixture(scope="module")
def kspace():
    """One hollow cylinder along z on a 20 um box at 0.1 um (images ~0.2 %), the 13 k-space channels per voxel."""
    lo, hi = np.array([-10e-6, -10e-6, -0.3e-6]), np.array([10e-6, 10e-6, 0.3e-6])
    fg = field_grid_of(d.MyelinatedCylinder(A, B, (0, 0, 1), 1.7e-9, 1.7e-9), res=0.1e-6, box=(lo, hi), mask_supersample=4)
    bs = fg.basis; shape = tuple(bs["shape"]); vs = np.asarray(bs["voxel_size"]); org = np.asarray(fg.origin)
    ax = [org[k] + (np.arange(shape[k]) + 0.5) * vs[k] for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    K = np.concatenate([bs["iso_local"].reshape(-1, 1), bs["iso_P"].reshape(6, -1).T, bs["aniso_G"].reshape(6, -1).T], 1)
    return P, K


def test_every_channel_matches_the_kspace_route_in_every_region(kspace):
    P, K = kspace
    rv = P.copy(); rv[:, 2] = 0.0
    C = np.asarray(hollow_cylinder_basis(rv, np.tile([0.0, 0.0, 1.0], (len(P), 1)), np.full(len(P), A), np.full(len(P), B)))
    rho = np.hypot(P[:, 0], P[:, 1])
    # a voxel straddling an edge is partial volume on the grid and a step in the closed form: inset by one voxel
    regions = {"lumen": rho < A - 0.12e-6, "sheath": (rho > A + 0.12e-6) & (rho < B - 0.12e-6),
               "near": (rho > B + 0.12e-6) & (rho < 3 * B), "far": (rho > 3 * B) & (rho < 6e-6)}
    tol = {"lumen": 0.03, "sheath": 0.06, "near": 0.02, "far": 0.01}       # of the component's max |value|
    for j, name in enumerate(CHANNEL_NAMES):
        scale = np.abs(K[:, j]).max()
        if scale < 1e-9:                                                   # xz, yz, zz(iso) vanish for an axis along z
            assert np.abs(C[:, j] - C[:, j].mean()).max() < 1e-9, name
            continue
        Cj = C[:, j] - C[:, j].mean()                                      # the k-space grids are zero-mean
        for r, sel in regions.items():
            err = np.sqrt(np.mean((Cj[sel] - K[sel, j]) ** 2)) / scale
            assert err < tol[r], (name, r, err)


def test_the_contraction_is_the_grid_assembly(kspace):
    """`contract` on channels equals `assemble_field` on the grids the channels came from."""
    P, K = kspace
    basis = {"iso_local": K[:, 0], "iso_P": K[:, 1:7].T, "aniso_G": K[:, 7:13].T, "shape": (len(P),), "voxel_size": np.ones(1)}
    for H in ([0.3, -0.4, 0.86], [1.0, 0.0, 0.0]):
        got = contract(K, H, B0=3.0, chi_iso=-0.05e-6, chi_aniso=-0.1e-6)
        want = assemble_field(basis, H, B0=3.0, chi_iso=-0.05e-6, chi_aniso=-0.1e-6)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("theta_deg,alpha_deg", [(90.0, 0.0), (60.0, 30.0), (30.0, 120.0)])
def test_the_bloch_myelin_provider_is_the_closed_form_everywhere(theta_deg, alpha_deg):
    """`MyelinSusceptibility.delta_bz_fn` (the forward Bloch engine's analytic myelin field) equals
    chi_A B0 H.M_A.H in the lumen, the sheath and outside, at any B0 angle."""
    th, al = np.deg2rad(theta_deg), np.deg2rad(alpha_deg)
    dchi, B0 = -1e-7, 3.0
    prov = MyelinSusceptibility(centers=[[0.0, 0.0]], inner_radii=[A], outer_radii=[B], L=1.0, delta_chi_a=dchi,
                                B0=B0, theta=th, alpha=al, n_images=0, periodic=False)
    f = jax.jit(jax.vmap(prov.delta_bz_fn()))
    H = np.array([np.sin(th) * np.cos(al), np.sin(th) * np.sin(al), np.cos(th)])
    r = np.array([0.3, 0.7, 1.05, 1.2, 1.35, 1.6, 2.5, 5.0]) * 1e-6
    ph = np.array([0.0, 0.4, 1.1, 2.0, 3.0, 4.2, 5.0, 6.0])
    p = np.stack([r * np.cos(ph), r * np.sin(ph), np.zeros(8)], 1)
    C = np.asarray(hollow_cylinder_basis(p, np.tile([0.0, 0.0, 1.0], (8, 1)), np.full(8, A), np.full(8, B)))
    want = dchi * B0 * (C[:, 7:13] @ q_of_H(H))
    got = np.asarray(f(jnp.asarray(p, jnp.float32)))
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=1e-13)


def test_the_lumen_is_wharton_bowtell_and_zero_along_the_fibre():
    p = np.array([[0.4e-6, 0.2e-6, 0.0]]); u = np.array([[0.0, 0.0, 1.0]])
    C = np.asarray(hollow_cylinder_basis(p, u, np.array([A]), np.array([B])))[0]
    for th in (0.0, np.pi / 3, np.pi / 2):
        H = [np.sin(th), 0.0, np.cos(th)]
        np.testing.assert_allclose(contract(C, H, B0=3.0, chi_aniso=-1e-7), 0.5 * -1e-7 * 3.0 * np.sin(th) ** 2 * np.log(B / A), rtol=1e-6, atol=1e-30)      # float32 arithmetic
        assert contract(C, H, B0=3.0, chi_iso=-1e-7) == 0.0                     # an isotropic sheath makes no lumen field

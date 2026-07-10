"""B-tensor encoding: calc_btensor, btensor_invariants, ste(), pte().

Tests are ordered from cheapest (analytical only) to most expensive (MC).
All MC tests are skipped when no GPU is present so the suite stays fast on CI.
"""

import numpy as np
import pytest

import dmipy_sim as ds
from dmipy_sim import (
    pgse, ogse, trapezoidal_ogse,
    ste, pte,
    calc_b, calc_btensor, btensor_invariants,
    set_b, simulate, FreeDiffusion,
)

try:
    import jax
    _HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:
    _HAS_GPU = False

gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="GPU required for MC tests")

# ── Shared waveform parameters ──────────────────────────────────────────────
DELTA   = 10e-3   # s  gradient block duration
BIGDEL  = 40e-3   # s  separation
G       = 0.06    # T/m
N_T     = 1000


# ── trace(B) == calc_b ─────────────────────────────────────────────

@pytest.mark.parametrize("waveform_fn, kwargs", [
    ("pgse",  {}),
    ("ogse",  {"frequency": 50.0, "T_total": DELTA + BIGDEL}),
    ("trap",  {"N": 2}),
])
def test_trace_equals_calc_b(waveform_fn, kwargs):
    """trace(calc_btensor(wf)) must equal calc_b(wf) to float64 precision."""
    bvecs = np.array([[1., 0., 0.]])

    if waveform_fn == "pgse":
        wf = pgse(DELTA, BIGDEL, G, bvecs, N_T)
    elif waveform_fn == "ogse":
        wf = ogse(kwargs["frequency"], kwargs["T_total"], G, bvecs, N_T)
    elif waveform_fn == "trap":
        wf = trapezoidal_ogse(kwargs["N"], DELTA, BIGDEL, G, bvecs, N_T)

    B    = calc_btensor(wf)       # (1, 3, 3)
    b_tr = np.trace(B[0])
    b_cb = calc_b(wf)[0]

    # Same trapezoid integration over float32 G, but different reduction order
    # (sum-then-integrate vs integrate-then-sum) introduces ~1e-6 relative error.
    np.testing.assert_allclose(b_tr, b_cb, rtol=1e-5,
        err_msg=f"trace(B)={b_tr:.6e} != calc_b={b_cb:.6e} for {waveform_fn}")


# ── LTE b_delta = 1 ────────────────────────────────────────────────

@pytest.mark.parametrize("bvec", [
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 0.],   # will be normalised by pgse caller
])
def test_lte_b_delta(bvec):
    """PGSE (LTE) must give b_delta=1.0 regardless of gradient direction."""
    bv = np.array([bvec], dtype=np.float32)
    bv /= np.linalg.norm(bv)
    wf = pgse(DELTA, BIGDEL, G, bv, N_T)
    B  = calc_btensor(wf)
    b, b_delta, b_eta = btensor_invariants(B)
    assert abs(b_delta[0] - 1.0) < 1e-3, f"LTE b_delta={b_delta[0]:.6f}, expected 1.0"
    assert abs(b_eta[0])         < 1e-3, f"LTE b_eta={b_eta[0]:.6f}, expected 0.0"


# ── STE: b_delta = 0 and B isotropic ──────────────────────────────

def test_ste_b_delta():
    """STE waveform must have b_delta=0 (isotropic B-tensor)."""
    wf = ste(DELTA, BIGDEL, G, N_T)
    B  = calc_btensor(wf)
    b, b_delta, b_eta = btensor_invariants(B)
    assert abs(b_delta[0]) < 1e-3, f"STE b_delta={b_delta[0]:.6f}, expected 0.0"
    assert abs(b_eta[0])   < 1e-3, f"STE b_eta={b_eta[0]:.6f}, expected 0.0"


def test_ste_btensor_diagonal():
    """B-tensor for STE must have equal diagonal and near-zero off-diagonal."""
    wf = ste(DELTA, BIGDEL, G, N_T)
    B  = calc_btensor(wf)[0]          # (3, 3)
    b  = np.trace(B)
    # Each diagonal element should be b/3 (float32 G: ~1e-5 relative error)
    np.testing.assert_allclose(np.diag(B), b / 3, rtol=1e-5,
        err_msg="STE B diagonal entries not equal to b/3")
    # Off-diagonal elements should be zero; float32 rounding gives ~1e-7 relative residuals
    off = B - np.diag(np.diag(B))
    np.testing.assert_allclose(off, 0.0, atol=1e-5 * b,
        err_msg="STE B off-diagonal should be near zero")


def test_ste_total_b():
    """STE total b-value must equal 3 × b per axis (by symmetry)."""
    wf_ste = ste(DELTA, BIGDEL, G, N_T)
    B      = calc_btensor(wf_ste)[0]
    b_total = np.trace(B)
    # Each diagonal entry should be b_total/3 (float32: ~1e-5 relative)
    np.testing.assert_allclose(np.diag(B), b_total / 3, rtol=1e-5)


# ── PTE: b_delta = -0.5 ───────────────────────────────────────────

@pytest.mark.parametrize("normal", [
    [0., 0., 1.],
    [1., 0., 0.],
    [1., 1., 1.],
])
def test_pte_b_delta(normal):
    """PTE waveform must have b_delta=-0.5 for any plane normal."""
    n = np.array(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    wf = pte(DELTA, BIGDEL, G, n, N_T)
    B  = calc_btensor(wf)
    b, b_delta, b_eta = btensor_invariants(B)
    assert abs(b_delta[0] - (-0.5)) < 1e-3, \
        f"PTE b_delta={b_delta[0]:.6f}, expected -0.5 (normal={normal})"


def test_pte_zero_eigenvalue():
    """PTE B-tensor must have one zero eigenvalue along the normal axis."""
    n = np.array([0., 0., 1.])
    wf = pte(DELTA, BIGDEL, G, n, N_T)
    B  = calc_btensor(wf)[0]
    eigvals = np.sort(np.linalg.eigvalsh(B))[::-1]  # descending
    # λ_1 ≈ λ_2 ≈ b/2 (float32: ~1e-5 relative off-diagonal residuals)
    # λ_3 = 0 exactly (no z-gradient ever played)
    b = np.trace(B)
    np.testing.assert_allclose(eigvals[0], b / 2, rtol=1e-5)
    np.testing.assert_allclose(eigvals[1], b / 2, rtol=1e-5)
    np.testing.assert_allclose(eigvals[2], 0.0,   atol=1e-30)  # exact: B_zz=0


def test_pte_total_b():
    """PTE total b-value must equal 2 × b per in-plane axis (by symmetry)."""
    wf_pte = pte(DELTA, BIGDEL, G, np.array([0., 0., 1.]), N_T)
    B      = calc_btensor(wf_pte)[0]
    eigs   = np.sort(np.linalg.eigvalsh(B))[::-1]  # descending
    # Two equal non-zero eigenvalues (float32: ~1e-5 relative)
    np.testing.assert_allclose(eigs[0], eigs[1], rtol=1e-5)
    b_total = np.trace(B)
    np.testing.assert_allclose(eigs[0], b_total / 2, rtol=1e-5)


# ── set_b compatibility: STE and PTE scale correctly ────────────────────────

def test_set_b_ste():
    """set_b should scale STE to the requested b while preserving b_delta=0."""
    wf   = ste(DELTA, BIGDEL, G, N_T)
    b_target = 1e9
    wf2  = set_b(wf, b_target)
    b_out, b_delta, _ = btensor_invariants(calc_btensor(wf2))
    np.testing.assert_allclose(b_out[0], b_target, rtol=1e-6)
    assert abs(b_delta[0]) < 1e-3, f"b_delta after set_b={b_delta[0]:.6f}"


def test_set_b_pte():
    """set_b should scale PTE to the requested b while preserving b_delta=-0.5."""
    wf   = pte(DELTA, BIGDEL, G, np.array([0., 0., 1.]), N_T)
    b_target = 1e9
    wf2  = set_b(wf, b_target)
    b_out, b_delta, _ = btensor_invariants(calc_btensor(wf2))
    np.testing.assert_allclose(b_out[0], b_target, rtol=1e-4)
    assert abs(b_delta[0] - (-0.5)) < 1e-3, f"b_delta after set_b={b_delta[0]:.6f}"


# ── MC physics: STE isotropic on FreeDiffusion ───────────────────────────────

@gpu_only
def test_ste_free_diffusion_isotropic():
    """STE signal on free diffusion must equal exp(-b*D).

    For isotropic diffusion, E = exp(-trace(B·D_tensor)) = exp(-b·D_iso)
    regardless of the B-tensor shape.  Verify this matches LTE.
    """
    D = 2e-9   # m²/s
    b_val = 1e9  # s/m²

    wf_lte = set_b(pgse(DELTA, BIGDEL, G, np.array([[1., 0., 0.]]), N_T), b_val)
    wf_ste = set_b(ste(DELTA, BIGDEL, G, N_T), b_val)

    geom  = FreeDiffusion()
    E_lte = simulate(100_000, D, wf_lte, geom, seed=1)
    E_ste = simulate(100_000, D, wf_ste, geom, seed=2)

    expected = np.exp(-b_val * D)
    mc_tol = 3.0 / np.sqrt(100_000)   # 3-sigma MC noise floor

    assert abs(float(E_lte[0]) - expected) < mc_tol, \
        f"LTE E={E_lte[0]:.4f}, expected {expected:.4f}"
    assert abs(float(E_ste[0]) - expected) < mc_tol, \
        f"STE E={E_ste[0]:.4f}, expected {expected:.4f}, LTE E={E_lte[0]:.4f}"

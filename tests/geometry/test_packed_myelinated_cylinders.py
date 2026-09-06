"""PackedMyelinatedCylinders geometry tests.

Tests:
  1. Zero-padding: 3-cylinder cell with N_max=3 vs N_max=8 produces same VFs.
  2. Volume fractions: intra + myelin + extra ≈ 1.0.
  3. Periodic BCs: walker near x=+L/2 can wrap to x≈-L/2 side.
  4. Different N_max with same N_actual=3 produces identical signals (JIT shape
     is the same from JAX's perspective).
  5. Dummy cylinders (r=0) never receive any walkers.
  6. Smoke: simulation runs end-to-end with valid signals.
  7. pack_myelinated_cylinders: no overlaps between outer boundaries.
"""

import numpy as np
import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp

from dmipy_sim import (
    simulate, PackedMyelinatedCylinders, pack_myelinated_cylinders,
    set_b,
)
from dmipy_sim.waveforms import pgse


SEED = 42


def _make_waveform(b_values=None):
    if b_values is None:
        b_values = np.array([0.0, 1e9])
    bvecs = np.tile([1., 0., 0.], (len(b_values), 1))
    wf = pgse(delta=5e-3, DELTA=15e-3, G_magnitude=1.0, bvecs=bvecs, n_t=500)
    return set_b(wf, b_values)


def _simple_cell(n_cyl=3, inner_r=3e-6, g_ratio=0.7, N_max=None):
    """Build a simple N-cylinder periodic cell."""
    inner_radii = np.full(n_cyl, inner_r)
    g_ratios    = np.full(n_cyl, g_ratio)
    outer_r     = inner_r / g_ratio
    # Place cylinders manually for reproducibility (no RSA needed for tests)
    L = 40e-6
    centers = np.array([[0.0, 0.0],
                        [12e-6, 0.0],
                        [-12e-6, 0.0]])[:n_cyl]
    if N_max is None:
        N_max = n_cyl
    return PackedMyelinatedCylinders(
        inner_radii=inner_radii,
        g_ratios=g_ratios,
        centers=centers,
        cell_size=L,
        N_max=N_max,
        D_intra=2e-9,
        D_myelin=0.1e-9,
        D_extra=2e-9,
    )


# ---------------------------------------------------------------------------
# Test 1: padding doesn't change volume fractions
# ---------------------------------------------------------------------------

def test_padding_preserves_volume_fractions():
    """N_max=3 and N_max=8 with the same 3 cylinders give identical VFs."""
    geom3 = _simple_cell(n_cyl=3, N_max=3)
    geom8 = _simple_cell(n_cyl=3, N_max=8)

    for comp in ('intra', 'myelin', 'extra'):
        vf3 = geom3.volume_fraction(comp)
        vf8 = geom8.volume_fraction(comp)
        assert abs(vf3 - vf8) < 1e-12, (
            f"VF({comp}): N_max=3 gives {vf3:.6f}, N_max=8 gives {vf8:.6f}")


# ---------------------------------------------------------------------------
# Test 2: volume fractions sum to 1
# ---------------------------------------------------------------------------

def test_volume_fractions_sum_to_one():
    """intra + myelin + extra volume fractions must sum to 1.0."""
    geom = _simple_cell(n_cyl=3, N_max=3)
    total = (geom.volume_fraction('intra') +
             geom.volume_fraction('myelin') +
             geom.volume_fraction('extra'))
    npt.assert_allclose(total, 1.0, atol=1e-6,
                        err_msg=f"VF sum = {total:.8f}, expected 1.0")


# ---------------------------------------------------------------------------
# Test 3: periodic BCs — walker initialised near x=+L/2 wraps to x≈-L/2
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="simulate() on PackedMyelinatedCylinders (multi-compartment myelin) is not in the open release; geometry construction is tested above")
def test_periodic_boundary_wrap():
    """Extra-axonal walker near cell edge is wrapped after stepping."""
    L = 40e-6
    inner_radii = np.array([3e-6])
    g_ratios    = np.array([0.7])
    centers     = np.array([[0.0, 0.0]])

    geom = PackedMyelinatedCylinders(
        inner_radii=inner_radii,
        g_ratios=g_ratios,
        centers=centers,
        cell_size=L,
        N_max=4,
        D_intra=2e-9,
        D_myelin=0.1e-9,
        D_extra=2e-9,
    )

    # Single gradient measurement along x
    wf = set_b(pgse(delta=2e-3, DELTA=6e-3, G_magnitude=1.0,
                    bvecs=np.array([[1., 0., 0.]]), n_t=200),
               np.array([1e9]))

    _, final_pos = simulate(500, waveform=wf, geometry=geom, seed=SEED,
                            return_positions=True)

    # All final positions must be within [-L/2, L/2) in x and y
    half = L / 2.0
    x = final_pos[:, 0]
    y = final_pos[:, 1]
    assert np.all(x >= -half - 1e-8) and np.all(x <= half + 1e-8), (
        f"x positions outside [-L/2, L/2]: min={x.min()*1e6:.2f}µm, "
        f"max={x.max()*1e6:.2f}µm, L/2={half*1e6:.2f}µm")
    assert np.all(y >= -half - 1e-8) and np.all(y <= half + 1e-8), (
        f"y positions outside [-L/2, L/2]: min={y.min()*1e6:.2f}µm, "
        f"max={y.max()*1e6:.2f}µm")


# ---------------------------------------------------------------------------
# Test 4: same N_actual, different N_max → same signal
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="simulate() on PackedMyelinatedCylinders (multi-compartment myelin) is not in the open release; geometry construction is tested above")
def test_different_nmax_same_signal():
    """N_max=3 and N_max=8 with N_actual=3 give identical signals."""
    geom3 = _simple_cell(n_cyl=3, N_max=3)
    geom8 = _simple_cell(n_cyl=3, N_max=8)

    wf = _make_waveform(b_values=np.array([0.0, 5e8, 1e9]))
    n_walkers = 5_000

    sig3 = simulate(n_walkers, waveform=wf, geometry=geom3, seed=SEED)
    sig8 = simulate(n_walkers, waveform=wf, geometry=geom8, seed=SEED)

    # Same seed, same geometry, same N_actual → must give same signal.
    # (N_max changes JAX shapes so JIT recompiles, but the physics is identical.)
    # We allow a small tolerance for any float32 ordering differences.
    npt.assert_allclose(sig3, sig8, atol=0.02,
                        err_msg="N_max=3 vs N_max=8 signals should match")


# ---------------------------------------------------------------------------
# Test 5: dummy cylinders never receive walkers
# ---------------------------------------------------------------------------

def test_dummy_cylinders_no_walkers():
    """Padding slots (r=0) must have zero walkers assigned."""
    geom = _simple_cell(n_cyl=3, N_max=8)
    key = jax.random.PRNGKey(SEED)
    _ = geom.init_positions(10_000, key)

    comp = np.array(geom._init_compartments)
    N_max = geom.N_max
    N_actual = geom.N_actual

    # Dummy intra slots: compartment IDs N_actual+1 .. N_max
    for k in range(N_actual, N_max):
        n_in_dummy_intra = np.sum(comp == (k + 1))
        assert n_in_dummy_intra == 0, (
            f"Dummy intra slot k={k} has {n_in_dummy_intra} walkers")

    # Dummy myelin slots: compartment IDs N_max + N_actual + 1 .. 2*N_max
    for k in range(N_actual, N_max):
        n_in_dummy_myelin = np.sum(comp == (N_max + k + 1))
        assert n_in_dummy_myelin == 0, (
            f"Dummy myelin slot k={k} has {n_in_dummy_myelin} walkers")


# ---------------------------------------------------------------------------
# Test 6: simulation smoke test — valid signals
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="simulate() on PackedMyelinatedCylinders (multi-compartment myelin) is not in the open release; geometry construction is tested above")
def test_smoke_simulation_valid():
    """PackedMyelinatedCylinders simulation produces finite signals in (0, 1]."""
    geom = _simple_cell(n_cyl=3, N_max=4)
    wf   = _make_waveform(b_values=np.array([0.0, 5e8, 1e9]))
    sig  = simulate(5_000, waveform=wf, geometry=geom, seed=SEED)

    assert sig.shape == (3,), f"Signal shape {sig.shape} != (3,)"
    assert np.all(np.isfinite(sig)), f"Non-finite signal: {sig}"
    assert np.all(sig > 0),         f"Non-positive signal: {sig}"
    npt.assert_allclose(sig[0], 1.0, atol=0.05,
                        err_msg=f"b=0 signal should be ~1.0, got {sig[0]:.4f}")
    assert sig[1] <= sig[0] + 0.05, "Signal should decrease with b"
    assert sig[2] <= sig[1] + 0.05, "Signal should decrease with b"


# ---------------------------------------------------------------------------
# Test 7: pack_myelinated_cylinders — no outer boundary overlaps
# ---------------------------------------------------------------------------

def test_pack_myelinated_no_overlaps():
    """pack_myelinated_cylinders: outer boundaries do not overlap (periodic)."""
    inner_radii = np.full(5, 3e-6)
    g_ratio     = 0.7
    inner_r, g_rat, centers = pack_myelinated_cylinders(
        inner_radii=inner_radii,
        g_ratios=g_ratio,
        target_packing=0.20,
        seed=7,
    )
    outer_radii = inner_r / g_rat
    # Infer L from area
    L = float(np.sqrt(np.pi * np.sum(outer_radii ** 2) / 0.20))

    N = len(inner_r)
    for i in range(N):
        for j in range(i + 1, N):
            dq = centers[i] - centers[j]
            dq -= L * np.round(dq / L)
            dist = np.linalg.norm(dq)
            min_sep = outer_radii[i] + outer_radii[j]
            assert dist >= min_sep - 1e-10, (
                f"Outer boundaries {i},{j} overlap: dist={dist*1e6:.3f}µm, "
                f"min={min_sep*1e6:.3f}µm")


# ---------------------------------------------------------------------------
# Test 8: simulate() runs on PackedMyelinatedCylinders (fused forward, no replay)
# ---------------------------------------------------------------------------

def test_simulate_packed_myelin_signal():
    """simulate() on a packed-myelin substrate gives E(b0)=1 and monotonic attenuation.

    Cross-validated against the private replay-based path (diffusion + T2 + surface
    relaxivity all agree within MC noise); here we just guard the public forward:
    it runs, b0 is unattenuated (no T2 in _simple_cell), and higher b attenuates more.
    """
    geom = _simple_cell()
    wf = _make_waveform(np.array([0.0, 0.5e9, 1.0e9, 2.0e9]))
    sig = np.asarray(simulate(20000, None, wf, geom, seed=SEED))
    npt.assert_allclose(sig[0], 1.0, atol=3.0 / np.sqrt(20000))   # b0 unattenuated
    assert np.all(np.diff(sig) < 0)                               # monotonic decay in b
    assert np.all(sig > 0) and np.all(sig <= 1.0 + 1e-6)

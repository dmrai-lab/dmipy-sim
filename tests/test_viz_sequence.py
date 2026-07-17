"""Sequence-diagram visualisation: physical-gradient display + storage shading.

The simulation gradient ``wf.G`` is bipolar (second lobe negated so the phase
integral refocuses at the echo without explicitly modelling the 180°). For a
*pulse-sequence diagram* we must instead show the physical scanner gradient
``wf.G_display`` (same-sign lobes, since the 180° performs the flip). These tests
lock in that the viz layer uses the physical gradient and shades the PGSTE
longitudinal-storage (T_M) window.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmipy_sim import pgse, pgste, set_b
from dmipy_sim import viz


def test_display_gradient_is_same_sign_while_sim_gradient_is_bipolar():
    """PGSE: the physical display gradient has two same-sign lobes (net area > 0),
    while the simulation gradient is bipolar and refocuses (net area ~ 0)."""
    wf = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05,
                    bvecs=[[1, 0, 0]], n_t=200), 1.0e9)

    disp = viz._display_G(wf)[0, :, 0]      # physical scanner gradient, x-axis
    sim = np.array(wf.G)[0, :, 0]           # bipolar simulation gradient, x-axis

    disp_area = np.abs(disp).sum() * wf.dt
    assert disp_area > 0
    # physical gradient: net (signed) area is the sum of two same-sign lobes
    assert abs(disp.sum() * wf.dt) > 0.5 * disp_area
    # simulation gradient: bipolar, so the signed area refocuses to ~0
    assert abs(sim.sum() * wf.dt) < 0.05 * disp_area


def test_pgse_display_lobes_share_polarity():
    """The two PGSE display lobes have the same sign; the two simulation lobes
    have opposite signs."""
    wf = pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05,
              bvecs=[[1, 0, 0]], n_t=200)
    disp = viz._display_G(wf)[0, :, 0]
    sim = np.array(wf.G)[0, :, 0]
    half = len(disp) // 2
    assert np.sign(disp[:half].sum()) == np.sign(disp[half:].sum())
    assert np.sign(sim[:half].sum()) == -np.sign(sim[half:].sum())


def test_pgste_storage_window_is_shaded():
    """PGSTE carries a chi_perp storage mask; _shade_storage adds a shaded span
    over the mixing time, and none is drawn for an all-transverse PGSE."""
    wf_ste = pgste(delta=4e-3, TM=20e-3, G_magnitude=0.05,
                   bvecs=[[1, 0, 0]], n_t=200)
    assert wf_ste.chi_perp is not None and (~np.asarray(wf_ste.chi_perp).astype(bool)).any()

    t_plot = np.arange(wf_ste.G.shape[1]) * wf_ste.dt * 1e3
    fig, ax = plt.subplots()
    n_before = len(ax.patches)
    viz._shade_storage(ax, wf_ste, t_plot)
    assert len(ax.patches) > n_before          # a storage span was added
    plt.close(fig)

    wf_se = pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05,
                 bvecs=[[1, 0, 0]], n_t=200)
    fig, ax = plt.subplots()
    viz._shade_storage(ax, wf_se, t_plot)      # chi_perp is None -> no-op
    assert len(ax.patches) == 0
    plt.close(fig)


def test_plot_helpers_run_on_pgse_and_pgste():
    """The public plotters build without error for both sequences."""
    wf_se = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05,
                       bvecs=[[1, 0, 0]], n_t=200), 1.0e9)
    wf_ste = set_b(pgste(delta=4e-3, TM=20e-3, G_magnitude=0.05,
                         bvecs=[[1, 0, 0]], n_t=200), 1.0e9)
    fig, _ = viz.plot_waveform(wf_ste, title="PGSTE")
    plt.close(fig)
    fig = viz.plot_sequence_comparison([wf_se, wf_ste],
                                       titles=["PGSE", "PGSTE"])
    plt.close(fig)

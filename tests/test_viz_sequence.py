"""Sequence-diagram visualisation: physical-gradient display + storage shading.

``wf.G`` is the physical scanner gradient (same-sign lobes, since the 180° performs the
flip) and ``wf.G_eff`` the effective one the phase integral walks (bipolar). A *pulse-sequence
diagram* shows ``G``; the q-trace integrates ``G_eff``. These tests lock in that the viz layer
draws the physical gradient and shades the PGSTE longitudinal-storage (T_M) window.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmipy_sim import pgse, pgste, set_b
from dmipy_sim import sequences as S
from dmipy_sim.viz import viz


def test_display_gradient_is_same_sign_while_sim_gradient_is_bipolar():
    """PGSE: the physical display gradient has two same-sign lobes (net area > 0),
    while the simulation gradient is bipolar and refocuses (net area ~ 0)."""
    wf = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05,
                    bvecs=[[1, 0, 0]], n_t=200), 1.0e9)

    disp = np.array(wf.G)[0, :, 0]      # physical scanner gradient, x-axis
    sim = np.array(wf.G_eff)[0, :, 0]       # bipolar effective gradient, x-axis

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
    disp = np.array(wf.G)[0, :, 0]
    sim = np.array(wf.G_eff)[0, :, 0]
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


def test_a_sequence_is_drawn_with_the_physical_gradient_too():
    """A ``Sequence`` carries the same RF panel as a ``Waveform``, so it must carry the same
    physical gradient: drawing the bipolar simulation gradient NEXT TO a 180 asserts the flip
    twice and shows a sequence that would not refocus."""
    seq = S.pgse([1.0e9], [[1, 0, 0]], 4e-3, 20e-3, n_t=200)
    assert any(e.flip_deg == 180 for e in seq.rf_events)      # the panel does draw a 180

    disp = np.array(seq.G)[0, :, 0]
    sim = np.asarray(seq.G_eff)[0, :, 0]
    assert not np.allclose(disp, sim), "the physical and effective gradients coincide: no 180 was folded"

    half = len(disp) // 2
    assert np.sign(disp[:half].sum()) == np.sign(disp[half:].sum())   # scanner: same polarity
    assert np.sign(sim[:half].sum()) == -np.sign(sim[half:].sum())    # phase integral: bipolar
    assert abs(disp.sum()) > 0.5 * np.abs(disp).sum()                 # net area, not refocused
    plt.close(viz.plot_waveform(seq, title="PGSE Sequence")[0])

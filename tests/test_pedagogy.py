"""Pedagogy viz on the public engine: idealised-pulse magnetisation history.

Validates that the Bloch-free (idealised instantaneous-pulse) magnetisation history reproduces
the gradient-driven echo physics, and that the figure/movie renderers run.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dmipy_sim import Cylinder, FreeDiffusion
from dmipy_replay import pgse, cpmg, set_b, calc_b
from dmipy_sim.viz import pedagogy as ped

NW = 5000
#: The two-sided normal quantile at a false-failure probability of 1e-3 for ONE comparison: a Monte-Carlo assertion
#: holds its statistic within Z standard errors of the closed form.
Z = 3.29


def test_free_diffusion_echo_matches_analytic():
    """PGSE spin echo on free diffusion: the echo is ``exp(-b D)`` attenuated by the T2 decay the history applies over
    its ``echo_idx + 1`` steps, within Z standard errors of the NW walkers' mean.

    The phase of a free walk is Gaussian of variance ``2 b D``, so one walker's ``cos(phi)`` has the variance
    ``(1 + exp(-4 b D)) / 2 - exp(-2 b D)`` and the mean of NW of them the standard error ``sqrt(that / NW)``: 2.4 % of
    the echo here (bD = 1). Measured: the mean of 50,000 walkers lands within 1.7 of its standard errors on three
    seeds, so the walk's discretisation adds nothing the tolerance has to carry."""
    D, T2 = 1e-9, 80e-3
    wf = set_b(pgse([[1, 0, 0]], 4e-3, 20e-3, gradient_strengths=0.05, n_t=160), 1.0e9)
    b = float(np.asarray(calc_b(wf)).ravel()[0])
    h = ped.replay_with_history(FreeDiffusion(), wf, diffusivity=D, T2=T2, n_walkers=NW, seed=0)
    Mc = h["M_comp"]
    net = np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])
    ei = wf.echo_idx
    relax = np.exp(-(ei + 1) * float(wf.dt) / T2)                     # the decay of every step up to the echo
    analytic = np.exp(-b * D) * relax
    se = relax * np.sqrt(((1 + np.exp(-4 * b * D)) / 2 - np.exp(-2 * b * D)) / NW)
    assert abs(net[ei] - analytic) < Z * se, (net[ei], analytic, se)
    assert net[1] > 0.99                                   # full transverse just after the 90


def test_restricted_echo_above_free():
    """A restricted cylinder attenuates less than free diffusion at the same b: the two echoes differ by more than Z
    standard errors of their difference, each at most ``relax / sqrt(NW)`` (a walker's ``cos(phi)`` is within 1)."""
    D, T2 = 1e-9, 80e-3
    wf = set_b(pgse([[1, 0, 0]], 4e-3, 20e-3, gradient_strengths=0.05, n_t=160), 1.0e9)
    ei = wf.echo_idx
    def echo(geom):
        Mc = ped.replay_with_history(geom, wf, diffusivity=D, T2=T2, n_walkers=NW, seed=0)["M_comp"]
        return np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])[ei]
    se_diff = np.exp(-(ei + 1) * float(wf.dt) / T2) * np.sqrt(2.0 / NW)
    assert echo(Cylinder(radius=5e-6, orientation=(0, 0, 1))) - echo(FreeDiffusion()) > Z * se_diff


def test_cpmg_echoes_decay_exponentially():
    """Instant-pulse CPMG (180 train) with no gradient: no walker's phase moves, so every echo peak IS
    ``exp(-k TE / T2)`` -- the decay of the ``k TE / dt`` steps before it -- to float64 rounding: one rounding per step
    of the train and per walker of the mean."""
    TE, T2, n_echoes = 12e-3, 60e-3, 4
    wf = cpmg(n_echoes, TE, gradient_strengths=0.0, gradient_directions=[[1, 0, 0]], n_t_per_echo=50)
    h = ped.replay_with_history(Cylinder(radius=5e-6, orientation=(0, 0, 1)), wf,
                                diffusivity=1e-9, T2=T2, n_walkers=NW, seed=0)
    Mc = h["M_comp"]; dt = float(wf.dt)
    net = np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])
    peaks = np.array([net[int(round((k + 1) * TE / dt)) - 1] for k in range(n_echoes)])
    np.testing.assert_allclose(peaks, np.exp(-(np.arange(1, n_echoes + 1)) * TE / T2),
                               rtol=(wf.n_t + NW) * np.finfo(np.float64).eps)


def test_sequence_story_renders(tmp_path):
    wf = set_b(pgse([[1, 0, 0]], 4e-3, 20e-3, gradient_strengths=0.05, n_t=120), 1.0e9)
    h = ped.replay_with_history(FreeDiffusion(), wf, diffusivity=1e-9, T2=80e-3, n_walkers=NW, seed=0)
    fig = ped.sequence_story(h, save=str(tmp_path / "story.png"))
    assert len(fig.axes) == 3
    assert (tmp_path / "story.png").exists()

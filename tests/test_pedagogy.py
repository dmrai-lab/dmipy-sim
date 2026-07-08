"""Pedagogy viz on the public engine: idealised-pulse magnetisation history.

Validates that the Bloch-free (idealised instantaneous-pulse) magnetisation history reproduces
the gradient-driven echo physics, and that the figure/movie renderers run.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dmipy_sim import Cylinder, FreeDiffusion, pgse, cpmg, set_b, calc_b
from dmipy_sim import pedagogy as ped

NW = 5000


def test_free_diffusion_echo_matches_analytic():
    """PGSE spin echo on free diffusion: net(echo) ~ exp(-b D - TE/T2)."""
    D, T2 = 1e-9, 80e-3
    wf = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=160), 1.0e9)
    b = float(np.asarray(calc_b(wf)).ravel()[0])
    h = ped.replay_with_history(FreeDiffusion(), wf, diffusivity=D, T2=T2, n_walkers=NW, seed=0)
    Mc = h["M_comp"]
    net = np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])
    ei = wf.echo_idx
    TE = h["t"][ei]
    analytic = np.exp(-b * D - TE / T2)
    assert abs(net[ei] - analytic) / analytic < 0.06, (net[ei], analytic)
    assert net[1] > 0.99                                   # full transverse just after the 90


def test_restricted_echo_above_free():
    """A restricted cylinder attenuates less than free diffusion at the same b."""
    D = 1e-9
    wf = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=160), 1.0e9)
    ei = wf.echo_idx
    def echo(geom):
        Mc = ped.replay_with_history(geom, wf, diffusivity=D, T2=80e-3, n_walkers=NW, seed=0)["M_comp"]
        return np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])[ei]
    assert echo(Cylinder(radius=5e-6, orientation=(0, 0, 1))) > echo(FreeDiffusion()) + 0.05


def test_cpmg_echoes_decay_exponentially():
    """Instant-pulse CPMG (180 train): echo peaks decay by ~exp(-TE/T2) per echo."""
    TE, T2, n_echoes = 12e-3, 60e-3, 4
    wf = cpmg(n_echoes=n_echoes, TE=TE, G_magnitude=0.0, bvecs=[[1, 0, 0]], n_t_per_echo=50)
    h = ped.replay_with_history(Cylinder(radius=5e-6, orientation=(0, 0, 1)), wf,
                                diffusivity=1e-9, T2=T2, n_walkers=NW, seed=0)
    Mc = h["M_comp"]; dt = float(wf.dt)
    net = np.abs((Mc[:, 0, :].sum(1) + 1j * Mc[:, 1, :].sum(1)) / Mc.shape[2])
    peaks = np.array([net[int(round((k + 1) * TE / dt)) - 1] for k in range(n_echoes)])
    assert np.allclose(peaks, np.exp(-(np.arange(1, n_echoes + 1)) * TE / T2), rtol=0.05)


def test_sequence_story_renders(tmp_path):
    wf = set_b(pgse(delta=4e-3, DELTA=20e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=120), 1.0e9)
    h = ped.replay_with_history(FreeDiffusion(), wf, diffusivity=1e-9, T2=80e-3, n_walkers=NW, seed=0)
    fig = ped.sequence_story(h, save=str(tmp_path / "story.png"))
    assert len(fig.axes) == 3
    assert (tmp_path / "story.png").exists()

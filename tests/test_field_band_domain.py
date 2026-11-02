"""The field band is held to the query domain its envelope promises: a gradient echo and a spin echo at every echo time
the band resolves, each within twice the production walk's standard error there, besides the CPMG train and the
stimulated echo of the window.

The pilot here is a field that jumps once per walker, at a time spread over four windows (a membrane crossing into a
magnetised layer): before its jump every walker reads the same field, so the early echoes of the window are nearly
certain and their standard errors small, and a band that smears a late jump over the window shows there. The window's
own gradient echo and spin echo do not see it."""
import numpy as np
from scipy.fft import dct, idct

from dmipy_replay.replay.field import ISO_CHANNEL_NAMES, contract
from dmipy_sim.replay.bank import (_battery_gates, _echo_errors, _echo_times, _gate_battery, derive_field_band)

N_W, N_T, H = 2000, 200, 0.25e-3                 # 50 ms of 0.25 ms blocks
B0, CHI = 7.0, 1e-7
ENV = dict(B0_list=[B0], theta_deg=[0], max_refocus_pulses=1)


def _pilot(seed=11):
    """Block means of the six isotropic channels: ``iso_P_xx`` is ``-1/6`` before the walker's jump and ``+1/6``
    after it (the field of a layer of normal x at a main field along z), the jump uniform over four windows."""
    rng = np.random.default_rng(seed)
    jump = rng.uniform(0.0, 4 * N_T, N_W)
    s = np.zeros((N_W, N_T, len(ISO_CHANNEL_NAMES)))
    s[:, :, 1] = np.where(np.arange(N_T)[None, :] + 0.5 < jump[:, None], -1.0, 1.0) / 6.0
    return s


def _field(series):
    return np.asarray(contract(series, [0.0, 0.0, 1.0], B0=B0, chi_iso=CHI), np.float64)


def test_the_echo_times_span_the_window_at_the_band_resolution():
    gre, se = _echo_times(785, 200)
    assert np.all(np.diff(gre[:-1]) == 4) and gre[0] == 4 and gre[-1] == 785       # T / K is 3.9 blocks: every 4
    assert np.all(se % 2 == 0) and se[-1] == 784 and np.all(np.isin(gre[:-1], se))
    gre, se = _echo_times(32, 16)
    assert gre.tolist() == list(range(2, 33, 2)) and se.tolist() == gre.tolist()


def test_a_spin_echo_refocuses_a_static_field_and_a_gradient_echo_winds_it():
    """A field constant along each path: the spin echo's phase is zero at every echo time, so its signal is one with no
    spread; the gradient echo's phase is ``gamma f t``, so its standard error is the spread of ``exp(i gamma f t)``. Two
    identical series have no error."""
    from dmipy_analytic.constants import GAMMA
    n = 50
    f = np.repeat(np.linspace(-1e-8, 1e-8, n)[:, None], N_T, axis=1)
    gre, se = _echo_times(N_T, 100)
    err, sd = _echo_errors(f, f, H, gre, se, np.ones(n))
    assert np.all(err == 0.0) and np.all(sd[len(gre):] < 1e-12)
    z = np.exp(1j * GAMMA * f[:, :1] * gre[None, :] * H)
    want = np.sqrt(np.mean(np.abs(z - z.mean(0)) ** 2, axis=0) / n)
    assert np.allclose(sd[:len(gre)], want, rtol=1e-9)


def test_a_band_the_window_echoes_accept_fails_the_early_echoes_and_the_derived_band_holds_them():
    series = _pilot()
    f = _field(series)
    X = dct(f, type=2, norm="ortho", axis=1)
    f2 = idct(np.pad(X[:, :2], ((0, 0), (0, N_T - 2))), type=2, norm="ortho", axis=1)     # the two-mode band
    w = np.ones(N_W)
    half = np.random.RandomState(0).permutation(N_W)
    err_window, floor_window = _gate_battery(f, f2, _battery_gates(N_T, 1), w, half[:N_W // 2], half[N_W // 2:], H)
    assert err_window <= floor_window                     # the window's four gates take two modes
    gre, se = _echo_times(N_T, N_T // 2)
    err, sd = _echo_errors(f, f2, H, gre, se, w)
    assert np.any(err > 2.0 * sd)                         # an echo of the domain does not
    K, record = derive_field_band(series, ISO_CHANNEL_NAMES, H, ENV, K_max=N_T // 2, B0=B0, chi=CHI)
    rungs = {r["K"]: r for r in record["ladder"]}
    assert rungs[2]["worst"]["ratio"] > 1.0 and K > 2
    assert record["within_floor"] and record["worst"]["ratio"] <= 1.0
    assert record["domain"]["gates_per_setting"] == len(gre) + len(se) + 2
    assert record["err_truncation"] >= record["worst"]["err_truncation"]

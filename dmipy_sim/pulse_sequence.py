"""Lean forward-native pulse-sequence construction for the vector-Bloch engine.

Builds the engine-ready representation ``dmipy_sim.bloch.simulate_bloch`` consumes:
a PHYSICAL gradient ``G(t)`` on a ``dt`` grid, a finite-pulse RF event list, and an
optional emergent voxel-scale crusher.  Scoped to what the magnetization-transfer /
three-observables work needs -- a spin-echo or gradient-echo readout, optionally
preceded by an off-resonance MT-prep saturation block.  (No replay, no
susceptibility; the private repo's pgste / b-tensor / ogse families are out of scope
here -- the readout does the refocusing emergently via its 180, no sign folding.)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .bloch import simulate_bloch
from .constants import GAMMA

__all__ = ["BlochSequence", "gradient_echo", "spin_echo", "fexi", "prepend_mt_prep",
           "run_bloch_sequence", "emergent_z_spectrum"]


@dataclass
class BlochSequence:
    """Engine-ready emergent representation of one acquisition (forward, no replay)."""
    G: np.ndarray                       # (n_meas, n_t, 3) PHYSICAL gradient, dt grid
    dt: float                           # time step (s)
    rf_events: list                     # [{t_s, flip_deg, axis_deg, duration_s, offset_hz}]
    complex_signal: bool = True         # True -> read (Mx+iMy); False -> Re only
    echo_steps: list = None             # multi-echo sample steps; None -> read the last step
    crusher: dict = None                # {'windows_s':[(t0,t1)], 'n_cycles':...}
    family: str = "gre"
    notes: str = ""

    @property
    def n_t(self) -> int:
        return self.G.shape[1]

    @property
    def n_meas(self) -> int:
        return self.G.shape[0]


# ── readout builders ────────────────────────────────────────────────────────────
def gradient_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """90 excitation then free evolution to ``TE``; read the transverse (complex).

    No 180, so a static off-resonance is NOT refocused -- the readout reports the
    (relaxed) transverse of whatever longitudinal magnetisation the excitation tips,
    which is exactly the MT-reduced ``Mz`` after an MT-prep block.
    """
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': exc_axis_deg,
           'duration_s': 0.0, 'offset_hz': 0.0}]
    return BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=True,
                         family="gre", notes="90 + free evolution to TE (complex)")


def spin_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """90 at 0, 180 at TE/2, echo at TE (real).  Gradient is zero unless set on ``G``.

    The 180 refocuses static dephasing EMERGENTLY (it conjugates the accumulated
    phase) -- no ``eps_P`` sign, no folded/effective gradient convention.
    """
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': exc_axis_deg,
           'duration_s': 0.0, 'offset_hz': 0.0},
          {'t_s': TE / 2.0, 'flip_deg': 180.0, 'axis_deg': 0.0,
           'duration_s': 0.0, 'offset_hz': 0.0}]
    return BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=False,
                         family="se", notes="90 - 180@TE/2 - echo (emergent refocusing)")


def fexi(delta, t_mix, dt, *, g_filter, g_detect, delta_detect=None,
         direction=(1., 0., 0.), crush_cycles=32.0, exc_axis_deg=0.0):
    """FEXI (filter-exchange) stimulated-echo diffusion sequence.

    A double-diffusion-encoding stimulated echo for measuring water exchange (Lasič et al.
    2011): a self-refocused **diffusion filter** dephases fast-diffusing water, a 90° stores
    the survivor longitudinally over a **mixing time** ``t_mix`` (during which nothing encodes
    but walkers keep diffusing and *exchanging* across membranes), a 90° recalls it, and a
    second self-refocused **detection** block measures the apparent diffusivity. As ``t_mix``
    grows, the filtered ADC recovers toward equilibrium at the exchange rate (AXR).

    Structure (each block a self-refocused bipolar pair — the "PGSTE with 2 lobes each side"):

        90 ─ [+g_f −g_f] ─ 90(store) ─·crusher·─ t_mix ─ 90(recall) ─ [+g_d −g_d] ─ echo
             └─ filter ─┘             └── longitudinal storage; EXCHANGE ──┘  └ detection ┘

    Unlike PGSTE, each block refocuses ``q→0`` before the store, so the mixing time carries no
    diffusion encoding — it is pure exchange weighting. Runs through :func:`simulate_bloch`
    (the crusher + stimulated-echo storage select the filtered pathway — a scalar ``chi_perp``
    walk cannot); exchange needs a **permeable** substrate. Returns a :class:`BlochSequence`
    with the per-measurement detection b-value on ``.b_detect`` (s/m²).

    Parameters
    ----------
    delta : float
        Duration of each filter (and, unless ``delta_detect``, detection) gradient lobe (s).
    t_mix : float
        Mixing time (s) — the longitudinal-storage / exchange period.
    dt : float
        Time step (s).
    g_filter : float
        Filter gradient amplitude (T/m) — the fixed diffusion filter (suppresses fast water).
    g_detect : float or array
        Detection gradient amplitude(s) (T/m); an array gives one measurement per value (e.g.
        ``[0, g]`` to fit an ADC).
    delta_detect : float, optional
        Detection lobe duration (s); defaults to ``delta``.
    direction : (3,) array
        Gradient direction (filter and detection share it).
    crush_cycles : float
        Voxel-scale crusher strength over the mixing window (dephases the non-stored pathway).
    exc_axis_deg : float
        B1 phase of the three 90° pulses (they share an axis so the store keeps ``cos φ``).
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    g_detect = np.atleast_1d(np.asarray(g_detect, dtype=np.float64))
    n_meas = g_detect.shape[0]
    ndf = int(round(delta / dt))
    ndd = int(round((delta if delta_detect is None else delta_detect) / dt))
    nmix = int(round(t_mix / dt))
    i_store = 2 * ndf
    i_recall = i_store + nmix
    n_t = i_recall + 2 * ndd + 1

    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    for m in range(n_meas):
        G[m, 0:ndf] = g_filter * d
        G[m, ndf:2 * ndf] = -g_filter * d
        G[m, i_recall:i_recall + ndd] = g_detect[m] * d
        G[m, i_recall + ndd:i_recall + 2 * ndd] = -g_detect[m] * d

    rf = [{'t_s': i * dt, 'flip_deg': 90.0, 'axis_deg': exc_axis_deg,
           'duration_s': 0.0, 'offset_hz': 0.0} for i in (0, i_store, i_recall)]
    crusher = {'windows_s': [((i_store + 1) * dt, (i_recall - 1) * dt)],
               'n_cycles': float(crush_cycles)}

    # detection b per measurement: q = γ·∫G dt over the (self-refocused) detection block
    b_detect = np.empty(n_meas)
    for m in range(n_meas):
        qd = GAMMA * np.cumsum(G[m, i_recall:, :], axis=0) * dt
        b_detect[m] = float(np.sum(qd ** 2) * dt)

    seq = BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=True, crusher=crusher,
                        family="fexi", notes="filter - store - t_mix (exchange) - recall - detect")
    seq.b_detect = b_detect
    return seq


# ── MT-prep saturation block ────────────────────────────────────────────────────
def prepend_mt_prep(seq, mt_prep):
    """Prepend an off-resonance MT-prep saturation block to a readout ``seq``.

    The saturation pulse is *just another RF event* -- long, off-resonance
    (``offset_hz``) -- placed before the readout, FOLLOWED BY a SEPARATE voxel-scale
    crusher window (RF off) that dephases the residual transverse so only the
    (MT-reduced) longitudinal magnetisation is excited by the readout.  A crusher
    concurrent with the pulse would be continuously refilled by the RF, so it must be
    its own window afterwards.  The crusher is modelled at the mm voxel scale it
    physically acts on (a um cell-scale gradient cannot wind >> 2 pi across the cell).

    ``mt_prep`` keys: ``offset_hz``, ``duration_s`` (sat pulse), ``b1_hz``
    (= gamma*B1/2pi -> ``flip_deg = 360*b1_hz*duration``) or ``flip_deg``; optional
    ``axis_deg`` (0), ``spoiler_s`` (crusher window, default 0.5 ms), ``n_cycles``
    (crusher strength, default 32).
    """
    dt = seq.dt
    off = float(mt_prep['offset_hz'])
    dur = float(mt_prep['duration_s'])
    axis = float(mt_prep.get('axis_deg', 0.0))
    flip = (float(mt_prep['flip_deg']) if mt_prep.get('flip_deg') is not None
            else 360.0 * float(mt_prep['b1_hz']) * dur)
    n_sat = max(1, int(round(dur / dt)))
    n_spoil = int(round(float(mt_prep.get('spoiler_s', 0.5e-3)) / dt))
    shift = n_sat + n_spoil                              # sat pulse THEN a crusher

    Gpre = np.zeros((seq.n_meas, shift, 3), dtype=seq.G.dtype)   # no gradient in the prep
    G_new = np.concatenate([Gpre, seq.G], axis=1)
    t_shift = shift * dt
    sat = {'t_s': dur / 2.0, 'flip_deg': flip, 'axis_deg': axis,
           'duration_s': dur, 'offset_hz': off}          # RF only over [0, dur]
    rf_new = [sat] + [{**e, 't_s': float(e['t_s']) + t_shift} for e in seq.rf_events]
    echo_new = (None if seq.echo_steps is None
                else [int(s) + shift for s in seq.echo_steps])
    spoil_win = (n_sat * dt, (n_sat + n_spoil) * dt)
    crush = (dict(windows_s=[spoil_win], n_cycles=float(mt_prep.get('n_cycles', 32.0)))
             if n_spoil > 0 else seq.crusher)
    return replace(seq, G=G_new, rf_events=rf_new, echo_steps=echo_new, crusher=crush,
                   notes=seq.notes + f"; MT-prep {off:.0f} Hz / {dur*1e3:.0f} ms")


# ── run a BlochSequence through the forward engine ──────────────────────────────
def run_bloch_sequence(seq, n_walkers, diffusivity, geometry, *, seed=0, **kw):
    """Run a :class:`BlochSequence` through ``simulate_bloch`` and return the signal.

    Extra keywords (``T2``, ``T1``, ``M0``, ``off_resonance_hz``, ``kappa_MT``,
    ``dwell_time``, ``T2_bound``, ``T1_bound``, ``off_resonance_bound``,
    ``return_mz``, ``require_gpu``) pass straight to ``simulate_bloch``.  With
    ``seq.echo_steps`` None the last step (the readout echo) is returned.
    """
    wf = SimpleNamespace(G=np.asarray(seq.G), dt=float(seq.dt))
    return simulate_bloch(n_walkers, diffusivity, wf, geometry, seq.rf_events,
                          seed=seed, echo_steps=seq.echo_steps, crusher=seq.crusher, **kw)


# ── turnkey emergent Z-spectrum sweep ─────────────────────────────────────────────
def emergent_z_spectrum(offsets_hz, geometry, *, n_walkers, diffusivity, w1_hz, t_sat, dt,
                        T2, kappa_MT, dwell_time, T1=1.0, T2_bound=1e-5, T1_bound=1.0,
                        equilibrate_binding="auto", seed=0):
    """Emergent CW-saturation Z-spectrum from the forward vector-Bloch engine.

    For each off-resonance ``offset`` (Hz), apply a continuous-wave saturation pulse of
    nutation ``w1_hz`` (= gamma*B1/2pi) and duration ``t_sat`` (s) to the MT-binding
    substrate, then read the walker-mean longitudinal magnetization ``Mz`` (normalised to
    ``M0=1``).  The broad, short-``T2_bound`` bound pool saturates over a wide offset range
    while the narrow free-water line is spared -> the emergent MT dip.  No super-Lorentzian
    lineshape is imposed; the dip is produced by real short-``T2_bound`` spins.

    This is the *emergent* (Monte-Carlo) counterpart of the analytic two-pool oracle
    :func:`dmipy_sim.mt.mt_z_spectrum`; the two agree to the MC noise floor once the bound
    pool is burned in (``equilibrate_binding`` other than ``'off'``).  Fine ``dt`` is
    required so the carrier ``2*pi*offset*dt`` does not alias.

    Parameters
    ----------
    offsets_hz : array-like
        Saturation offsets from the free-water resonance (Hz).
    geometry : Sphere | Cylinder | ...
        Any MT-capable geometry (the wall the spins bind to).
    n_walkers, diffusivity : int, float
        Walker count and free-water diffusivity (m^2/s).
    w1_hz, t_sat, dt : float
        CW saturation nutation rate (Hz), duration (s), and timestep (s).
    T2, T1 : float
        Free-pool relaxation times (s).
    kappa_MT, dwell_time, T2_bound, T1_bound : float
        MT wall reactivity (m/s), bound dwell time (s, = 1/k_r), and bound-pool T2/T1 (s).
    equilibrate_binding : {'auto', 'burnin', 'fast', 'off'}
        Bound-pool initialisation; see :func:`dmipy_sim.simulate_bloch`.

    Returns
    -------
    numpy.ndarray
        ``Mz`` of the free pool at each offset, shape ``(len(offsets_hz),)``.
    """
    offsets = np.atleast_1d(np.asarray(offsets_hz, dtype=float))
    n_t = int(round(float(t_sat) / float(dt))) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=float(dt))
    flip = 360.0 * float(w1_hz) * float(t_sat)              # CW total flip over the window
    mz = np.empty(offsets.shape, dtype=float)
    for i, off in enumerate(offsets):
        rf = [{'t_s': float(t_sat) / 2, 'flip_deg': flip, 'axis_deg': 0.0,
               'duration_s': float(t_sat), 'offset_hz': float(off)}]
        _, m = simulate_bloch(n_walkers, diffusivity, wf, geometry, rf,
                              T2=T2, T1=T1, kappa_MT=kappa_MT, dwell_time=dwell_time,
                              T2_bound=T2_bound, T1_bound=T1_bound, return_mz=True,
                              equilibrate_binding=equilibrate_binding, seed=seed)
        mz[i] = float(m[0])
    return mz

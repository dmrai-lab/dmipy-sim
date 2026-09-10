"""Sequence -- the physical acquisition (forward signal definition), sim-owned.

A ``Sequence`` carries the *physical* per-measurement encoding -- gradient
directions, b-values, q-values, gradient strengths, timing (delta/Delta/TM/TE),
the family-specific physical flags, and the REAL gradient waveform ``G(t)`` --
plus the directly-derived ``btensor`` and the square/infinite-slew ``instantaneous``
limit.  The ``from_X`` constructors build the waveform from first principles.

This is the half the user assigned to dmipy-sim: "defining the sequence in terms
of gradient directions, b-values, q-values ... that defines the forward signal
generation."  The analytical-inverse organisation of these measurements into
shells / SH orders / rotational harmonics stays in dmipy-fit, which consumes a
``Sequence`` (fit eats sim's real constructors).
"""
from __future__ import annotations

import numpy as np

from ..acquisition.waveforms import apply_rf_schedule
from ..acquisition.rf import RFEvent, RFSchedule

from ..math.gradient_conversions import g_from_b, q_from_b
from ..constants import GAMMA, DEFAULT_SLEW_RATE, resolve_slew as _resolve_slew
from ._helpers import (
    _trap_profile, _trap_cosine_profile, _calc_b_from_waveform,
    _btensor_from_waveform, _refocusing_residual, _resolve_te, _scale_to_b,
    unify_length_reference_delta_Delta, check_acquisition_scheme,
    _REFOCUS_ATOL,
)

# Physical flag attributes a Sequence may carry (copied verbatim onto a
# consuming analytical scheme).  Core fields are explicit __init__ args.
_FLAG_ATTRS = (
    'sequence_type', '_minimum_te', '_te_auto', '_refocus_gap',
    '_ogse_two_train', '_refocus_duration',
    'TM', 'tau_perp_SE', 'ste_flip_angles', '_ramp_time',
    'oscillation_frequency', 'gradient_rise_time', 'n_oscillation_cycles',
    'gradient_duration', 'cpmg_n_echoes', 'cpmg_TE', 'cpmg_beta_deg',
    'n_t_per_echo', 'refocused',
)


class Sequence:
    """Physical acquisition: the gradient the scanner plays, its RF schedule, and the per-measurement encoding.

    Build with the ``from_X`` classmethods (or the module-level
    ``dmipy_sim.sequences.pgse(...)`` etc.).  Carries:
      * ``G`` -- the PHYSICAL gradient ``(n_meas, n_t, 3)`` on ``dt``; :attr:`G_eff` is the effective
        gradient the phase integral walks, ``G`` with the schedule's sign folded in, derived on read.
      * core encoding -- ``bvalues, gradient_directions, qvalues, gradient_strengths, delta, Delta, TE``
      * family flags -- ``sequence_type`` and the physical markers each family sets (``TM``,
        ``ste_flip_angles``, ``oscillation_frequency``, ``cpmg_*`` ...).
      * the :class:`dmipy_sim.acquisition.waveforms.Waveform` readout attributes -- ``echo_idx``,
        ``echo_indices``, ``rf_events``, ``chi_perp``, ``TM``, ``stimulated_echo`` -- so a ``Sequence``
        is accepted wherever a ``Waveform`` is (``simulate`` in either engine, ``simulate_cpmg``,
        ``to_pulseq``, ``viz``).

    Every constructor declares the pulses it plays, places each measurement's gradient relative to them
    (lobes symmetric about a 180, whatever the row's own timing), finishes with a numeric scaling so the
    declared ``bvalues`` equal :func:`dmipy_sim.acquisition.waveforms.b_from_gradient` of ``G_eff``, and
    :meth:`validate`\ s: the gradient is off across every finite pulse and refocused at the echo.
    """

    def __init__(self, G, dt, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta, TE):
        self.G = np.asarray(G, dtype=np.float32)
        self.dt = float(dt)
        # Waveform readout protocol (see dmipy_sim.acquisition.waveforms.Waveform)
        self.echo_idx = int(self.G.shape[1] - 1)
        self.echo_indices = None
        self.rf_events = RFSchedule()
        self.chi_perp = None
        self.TM = None
        self.stimulated_echo = False
        self.bvalues = np.asarray(bvalues, dtype=np.float64)
        self.gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        self.qvalues = None if qvalues is None else np.asarray(qvalues, dtype=np.float64)
        self.gradient_strengths = (None if gradient_strengths is None
                                   else np.asarray(gradient_strengths, dtype=np.float64))
        self.delta = None if delta is None else np.asarray(delta, dtype=np.float64)
        self.Delta = None if Delta is None else np.asarray(Delta, dtype=np.float64)
        self.TE = None if TE is None else np.asarray(TE, dtype=np.float64)
        self._build_spec = None     # (constructor_name, kwargs) for instantaneous()

    # ── physical accessors ────────────────────────────────────────────────────
    @property
    def number_of_measurements(self):
        return self.gradient_directions.shape[0]

    @property
    def G_eff(self):
        """The EFFECTIVE gradient the phase integral walks: ``G * rf_events.sign(t)`` (RPK.md 6.6). Derived,
        never stored. ``(n_meas, n_t, 3)`` float32."""
        s = self.rf_events.sign(np.arange(self.G.shape[1]) * self.dt)
        return self.G * s[None, :, None]

    def btensor(self):
        """b-tensor B_ij = gamma^2 ∫ q_i q_j dt per measurement, (n_m, 3, 3), of the effective gradient."""
        return _btensor_from_waveform(self.G_eff, self.dt)

    @property
    def refocusing_residual(self):
        """max over measurements of the relative net gradient moment |q(TE)|/max|q| of the effective gradient."""
        G_eff = self.G_eff
        return max(_refocusing_residual(G_eff[m], self.dt)
                   for m in range(self.number_of_measurements))

    def validate(self):
        """What every constructor guarantees: the gradient is OFF across every finite pulse (a hard pulse
        occupies an instant and constrains nothing -- a constant gradient through an ideal 180 train is
        Carr-Purcell), and the effective gradient refocuses at the echo. Raises naming the failure."""
        t = np.arange(self.G.shape[1]) * self.dt
        for e in self.rf_events:
            if e.duration_s > 0.0:
                t0, t1 = e.window
                inside = (t >= t0 - 1e-9 * self.dt) & (t <= t1 + 1e-9 * self.dt)
                if np.any(np.abs(self.G[:, inside, :]) > 0.0):
                    raise ValueError(f"the gradient is on during the {e.flip_deg:g} pulse at {e.t_s*1e3:.3f} ms "
                                     f"(window {t0*1e3:.3f}-{t1*1e3:.3f} ms): a finite pulse needs zero gradient")
        res = self.refocusing_residual
        if res > _REFOCUS_ATOL and float(np.abs(self.G).max()) > 0.0:
            raise ValueError(f"the effective gradient is not refocused at the echo (|q(TE)|/max|q| = {res:.2e} > "
                             f"{_REFOCUS_ATOL:.0e})")
        return self

    def to_gradient_array(self, n_t=1000):
        """Return ``(G_eff, dt)`` -- the EFFECTIVE gradient -- of the square (instantaneous) PGSE with
        this scheme's b-values, directions, delta and Delta on an ``n_t`` grid: :meth:`from_pgse` at
        infinite slew, as the analytical layer integrates it."""
        if self.delta is None or self.Delta is None or self.gradient_strengths is None:
            raise ValueError(
                "to_gradient_array() requires delta, Delta, and gradient_strengths.")
        delta_tol = np.float32(1e-6)
        if (np.max(self.delta) - np.min(self.delta)) > delta_tol:
            raise ValueError("to_gradient_array() requires uniform delta.")
        if (np.max(self.Delta) - np.min(self.Delta)) > delta_tol:
            raise ValueError("to_gradient_array() requires uniform Delta.")
        sq = Sequence.from_pgse(self.bvalues, self.gradient_directions, float(self.delta[0]),
                                float(self.Delta[0]), n_t=n_t, slew_rate=np.inf)
        return sq.G_eff, sq.dt

    def instantaneous(self):
        """Idealised instantaneous (infinite-slew / square) limit of this sequence.

        Rebuilds with ``slew_rate=np.inf`` for the families that slew-limit (pgse/
        ogse); families that are already square (cpmg/ste/pte) return an
        equivalent rebuild.  This is the idealised view the analytical model reads.
        """
        if self._build_spec is None:
            return self
        name, kwargs = self._build_spec
        kwargs = dict(kwargs)
        if 'slew_rate' in kwargs:
            kwargs['slew_rate'] = np.inf
        return getattr(type(self), name)(**kwargs)

    def _carry(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)
        return self

    # ── constructors (physical waveform generation) ───────────────────────────
    @classmethod
    def from_pgse(cls, bvalues, gradient_directions, delta, Delta, TE=None,
                  n_t=1000, slew_rate=DEFAULT_SLEW_RATE):
        """PGSE: two same-sign lobes separated by Delta, the 180 midway between them.

        Every measurement's lobe pair is centred on the one 180 at ``T_total / 2`` (``T_total`` is the
        longest row's ``Delta + delta + ramp``), so a row with a shorter ``Delta`` sits inside the same
        echo with its 180 in its own gap rather than starting at t = 0 and having the pulse land inside
        its second lobe. Slew-limited (realizable) by default (``slew_rate`` in T/m/s); pass
        ``slew_rate=np.inf`` for the idealized instantaneous (square) limit -- vertical ramps, same structure.
        """
        bvalues = np.asarray(bvalues, dtype=np.float64)
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        delta_, Delta_, TE_in = unify_length_reference_delta_Delta(
            bvalues, delta, Delta, TE)
        check_acquisition_scheme(bvalues, gradient_directions, delta_, Delta_, TE_in)

        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
        qvalues = q_from_b(bvalues, delta_, Delta_)

        n_m = len(bvalues)
        slew_rate, square = _resolve_slew(slew_rate)
        if square:
            eps_ = np.zeros(n_m)
        else:
            eps_ = np.minimum(gradient_strengths / float(slew_rate), delta_)
        T_total = float(np.max(Delta_ + delta_ + eps_))
        dt = T_total / (n_t - 1)
        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        # ideal instantaneous 90/180: the echo forms at the END of the grid (T_total, the longest row), so the
        # 180 sits at T_total / 2 and every row's lobe pair is centred on it -- a row with a shorter Delta is
        # shifted right by half its slack instead of starting at t = 0 with the pulse inside its second lobe
        t_180 = T_total / 2.0
        schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(t_180, 180, 'refocus')])
        t_grid = np.arange(n_t) * dt
        shift = (T_total - (Delta_ + delta_ + eps_)) / 2.0                 # per row; 0 for the longest
        G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)                  # the PHYSICAL gradient
        if square:
            for m in range(n_m):
                n_pulse = max(1, round(float(delta_[m]) / dt))
                n_Delta = round(float(Delta_[m]) / dt)
                n0 = round(float(shift[m]) / dt)
                g_vec = gradient_strengths[m] * gradient_directions[m]
                G_arr[m, n0:n0 + n_pulse, :] = g_vec
                G_arr[m, n0 + n_Delta:n0 + n_Delta + n_pulse, :] = g_vec
        else:
            for m in range(n_m):
                tm = t_grid - shift[m]
                prof = (_trap_profile(tm, 0.0, delta_[m], eps_[m]) +
                        _trap_profile(tm, Delta_[m], delta_[m], eps_[m]))
                G_arr[m] = (gradient_strengths[m] * prof)[:, None] * gradient_directions[m]
        # the declared b IS the numeric b of the effective gradient the walk integrates
        sign = schedule.sign(t_grid)[None, :, None]
        G_arr = _scale_to_b(G_arr * sign, dt, bvalues) * sign

        seq = cls(G_arr, dt, bvalues, gradient_directions, qvalues,
                  gradient_strengths, delta_, Delta_, TE_)
        seq.rf_events = schedule
        apply_rf_schedule(seq)
        seq._carry(sequence_type='pgse', _minimum_te=T_total, _te_auto=te_auto,
                   _refocus_gap=float(np.min(Delta_ - delta_ - eps_)))
        seq.validate()
        seq._build_spec = ('from_pgse', dict(
            bvalues=bvalues, gradient_directions=gradient_directions,
            delta=delta, Delta=Delta, TE=TE, n_t=n_t, slew_rate=slew_rate))
        return seq


    @classmethod
    def from_cpmg(cls, n_echoes, TE, bvalues=None, gradient_directions=None,
                  beta_deg=180.0, n_t_per_echo=100):
        """CPMG multi-echo spin echo; optional per-echo bipolar diffusion lobe."""
        if n_t_per_echo % 2 != 0:
            raise ValueError(f"n_t_per_echo must be even, got {n_t_per_echo}.")
        n_echoes = int(n_echoes)
        TE_echo = float(TE)
        TE_ = (np.arange(n_echoes, dtype=np.float64) + 1.0) * TE_echo
        if bvalues is None:
            bvalues = np.zeros(n_echoes, dtype=np.float64)
        bvalues = np.broadcast_to(np.asarray(bvalues, float), (n_echoes,)).copy()
        if gradient_directions is None:
            gradient_directions = np.tile([0.0, 0.0, 1.0], (n_echoes, 1))
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)

        n_half = n_t_per_echo // 2
        n_t_total = n_echoes * n_t_per_echo
        dt = TE_echo / n_t_per_echo
        Delta_lobe = np.full(n_echoes, n_half * dt)
        delta_lobe = np.full(n_echoes, n_half * dt)
        gstr = np.where(bvalues > 0,
                        g_from_b(np.maximum(bvalues, 1.0), delta_lobe, Delta_lobe), 0.0)
        qvals = np.where(bvalues > 0,
                         q_from_b(np.maximum(bvalues, 1.0), delta_lobe, Delta_lobe), 0.0)
        # echo k forms at the end of its interval; the 180s sit at (k + 1/2) TE
        schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy')] +
                              [RFEvent((k + 0.5) * TE_echo, 180, 'refocus') for k in range(n_echoes)])
        # This family is defined by its EFFECTIVE gradient: a bipolar lobe pair per echo interval, so each
        # echo self-refocuses. Its physical gradient is that un-folded through the declared 180 train.
        G_eff = np.zeros((n_echoes, n_t_total, 3), dtype=np.float64)
        for m in range(n_echoes):
            lobe = gstr[m] * gradient_directions[m]
            for k in range(n_echoes):
                base = k * n_t_per_echo
                G_eff[m, base:base + n_half, :] = lobe
                G_eff[m, base + n_half:base + n_t_per_echo, :] = -lobe
        sign = schedule.sign(np.arange(n_t_total) * dt)[None, :, None]
        G_arr = _scale_to_b(G_eff, dt, bvalues) * sign

        seq = cls(G_arr, dt, bvalues, gradient_directions, qvals,
                  gstr, delta_lobe, Delta_lobe, TE_)
        seq.rf_events = schedule
        apply_rf_schedule(seq)
        seq._carry(sequence_type='cpmg', refocused=True, cpmg_n_echoes=n_echoes,
                   cpmg_TE=TE_echo, cpmg_beta_deg=float(beta_deg),
                   n_t_per_echo=int(n_t_per_echo))
        seq.validate()
        seq._build_spec = ('from_cpmg', dict(
            n_echoes=n_echoes, TE=TE, bvalues=bvalues,
            gradient_directions=gradient_directions, beta_deg=beta_deg,
            n_t_per_echo=n_t_per_echo))
        return seq

    @classmethod
    def from_waveform(cls, G, dt, gradient_directions, delta=None, Delta=None,
                      TE=None, allow_unrefocused=False):
        """Build from an arbitrary gradient waveform; b numerically from G."""
        from warnings import warn
        G = np.asarray(G, dtype=np.float32)
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        bvalues = _calc_b_from_waveform(G, dt)
        delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
            bvalues, delta, Delta, TE)
        qvalues = gradient_strengths = None
        if delta_ is not None and Delta_ is not None:
            gradient_strengths = g_from_b(bvalues, delta_, Delta_)
            qvalues = q_from_b(bvalues, delta_, Delta_)
        seq = cls(G, dt, bvalues, gradient_directions, qvalues,
                  gradient_strengths, delta_, Delta_, TE_)
        seq._carry(sequence_type='waveform')
        try:
            seq.validate()
        except ValueError as e:
            msg = (f"from_waveform: {e}; stationary spins will not rephase at TE. Ensure the waveform "
                   f"refocuses, or pass allow_unrefocused=True if intentional.")
            if allow_unrefocused:
                warn(msg)
            else:
                raise ValueError(msg) from None
        return seq

    @classmethod
    def from_ogse(cls, bvalues, gradient_directions, oscillation_frequency,
                  gradient_duration, n_cycles=1, gradient_rise_time=0.,
                  TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE, refocus_duration=0.0):
        """Cosine OGSE: two slew-limited trains straddling a declared 180 at ``T_total / 2`` by default;
        ``slew_rate=np.inf`` gives the idealized single continuous cosine (no 180 modelled). Each
        measurement's train pair is centred on the 180, whatever its own ``gradient_duration``."""
        bvalues = np.asarray(bvalues, dtype=np.float64)
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        n_m = len(bvalues)
        gamma = GAMMA

        osc_freq = np.broadcast_to(np.asarray(oscillation_frequency, float), (n_m,)).copy()
        sigma = np.broadcast_to(np.asarray(gradient_duration, float), (n_m,)).copy()
        n_cyc = np.broadcast_to(np.asarray(n_cycles, float), (n_m,)).copy()
        t_r = np.broadcast_to(np.asarray(gradient_rise_time, float), (n_m,)).copy()

        safe_sigma = np.where(sigma > 0, sigma, np.ones_like(sigma))
        safe_freq = np.where(osc_freq > 0, osc_freq, np.ones_like(osc_freq))
        G_mag = np.sqrt(bvalues * (8.0 * np.pi ** 2 * safe_freq ** 2) /
                        (gamma ** 2 * safe_sigma))
        G_mag = np.where(bvalues > 0, G_mag, 0.0)

        gap = float(refocus_duration)
        slew_rate, square = _resolve_slew(slew_rate)
        if square:
            T_total = float(np.max(sigma))
        else:
            T_total = float(np.max(2.0 * sigma + gap))
        dt = T_total / (n_t - 1)
        t_full = np.arange(n_t) * dt
        schedule = (RFSchedule() if square else
                    RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(T_total / 2.0, 180, 'refocus')]))
        sign = schedule.sign(t_full)[None, :, None]
        G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)                  # the PHYSICAL gradient
        for m in range(n_m):
            if bvalues[m] <= 0 or G_mag[m] == 0:
                continue
            if square:
                n_sig = max(1, round(sigma[m] / dt))
                t = np.arange(n_sig) * dt
                g_t = G_mag[m] * np.cos(2.0 * np.pi * osc_freq[m] * t)
                G_arr[m, :n_sig, :] = g_t[:, None] * gradient_directions[m]
            else:
                sg, fm, sr = float(sigma[m]), osc_freq[m], float(slew_rate)
                shift = (T_total - (2.0 * sg + gap)) / 2.0                 # centre this row's pair on the 180
                tm = t_full - shift
                # the slew-limited trapezoidal ramps depend on the amplitude, so iterate the
                # profile to the requested b before the exact scaling below
                g_amp = G_mag[m]
                for _ in range(6):
                    pre = _trap_cosine_profile(tm, sg, fm, sr, g_amp)
                    post = _trap_cosine_profile(tm - (sg + gap), sg, fm, sr, g_amp)
                    Gm = (pre + post)[:, None] * gradient_directions[m]
                    b_m = _calc_b_from_waveform((Gm * sign[0])[None], dt)[0]
                    if b_m <= 0 or abs(b_m - bvalues[m]) <= 1e-6 * bvalues[m]:
                        break
                    g_amp *= np.sqrt(bvalues[m] / b_m)
                G_arr[m] = Gm
        # the declared b IS the numeric b of the effective gradient the walk integrates
        G_arr = _scale_to_b(G_arr * sign, dt, bvalues) * sign

        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        qvalues = G_mag * gamma * sigma / (2.0 * np.pi)
        gradient_strengths = G_mag

        seq = cls(G_arr, dt, bvalues, gradient_directions, qvalues,
                  gradient_strengths, None, None, TE_)
        seq.rf_events = schedule
        if schedule:
            apply_rf_schedule(seq)
        seq._carry(oscillation_frequency=osc_freq, gradient_rise_time=t_r,
                   n_oscillation_cycles=n_cyc, gradient_duration=sigma,
                   _minimum_te=T_total, _te_auto=te_auto, sequence_type='ogse')
        if square:
            seq._carry(_refocus_gap=0.0)
        else:
            seq._carry(_refocus_gap=float(gap), _ogse_two_train=True, _refocus_duration=float(gap))
        seq.validate()
        seq._build_spec = ('from_ogse', dict(
            bvalues=bvalues, gradient_directions=gradient_directions,
            oscillation_frequency=oscillation_frequency,
            gradient_duration=gradient_duration, n_cycles=n_cycles,
            gradient_rise_time=gradient_rise_time, TE=TE, n_t=n_t,
            slew_rate=slew_rate, refocus_duration=refocus_duration))
        return seq

    @classmethod
    def from_btensor_ste(cls, bvalues, delta, Delta, TE=None, n_t=1000):
        """Spherical tensor encoding (b_delta=0): three orthogonal bipolar pairs."""
        bvalues = np.atleast_1d(np.asarray(bvalues, dtype=np.float64))
        n_m = len(bvalues)
        T_total = float(delta) + float(Delta)
        dt = T_total / (n_t - 1)
        n_seg = max(1, n_t // 6)                 # whole pairs on the grid: each refocuses exactly
        G_template = np.zeros((n_t, 3), dtype=np.float32)
        for i in range(3):
            enc_start = 2 * i * n_seg
            enc_end = min((2 * i + 1) * n_seg, n_t)
            dec_end = min((2 * i + 2) * n_seg, n_t)
            G_template[enc_start:enc_end, i] = 1.0
            G_template[enc_end:dec_end, i] = -1.0
        b_unit = _calc_b_from_waveform(G_template[None], dt)[0]
        G_arr = np.zeros((n_m, n_t, 3), dtype=np.float32)
        for m in range(n_m):
            scale = float(np.sqrt(bvalues[m] / b_unit)) if b_unit > 0 else 0.0
            G_arr[m] = G_template * scale
        gradient_directions = np.tile([0., 0., 1.], (n_m, 1))
        bvalues_num = _calc_b_from_waveform(G_arr, dt)
        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        seq = cls(G_arr, dt, bvalues_num, gradient_directions,
                  None, None, None, None, TE_)
        seq._carry(_minimum_te=T_total, _te_auto=te_auto, sequence_type='btensor_ste')
        seq.validate()
        seq._build_spec = ('from_btensor_ste', dict(
            bvalues=bvalues, delta=delta, Delta=Delta, TE=TE, n_t=n_t))
        return seq

    @classmethod
    def from_btensor_pte(cls, bvalues, plane_normal, delta, Delta, TE=None, n_t=1000):
        """Planar tensor encoding (b_delta=-0.5): two in-plane bipolar pairs."""
        bvalues = np.atleast_1d(np.asarray(bvalues, dtype=np.float64))
        n_m = len(bvalues)
        n = np.asarray(plane_normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        u = ref - np.dot(ref, n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)
        T_total = float(delta) + float(Delta)
        dt = T_total / (n_t - 1)
        n_seg = max(1, n_t // 4)                 # whole pairs on the grid: each refocuses exactly
        G_template = np.zeros((n_t, 3), dtype=np.float32)
        for i, axis in enumerate([u, v]):
            enc_start = 2 * i * n_seg
            enc_end = min((2 * i + 1) * n_seg, n_t)
            dec_end = min((2 * i + 2) * n_seg, n_t)
            G_template[enc_start:enc_end] = axis.astype(np.float32)
            G_template[enc_end:dec_end] = -axis.astype(np.float32)
        b_unit = _calc_b_from_waveform(G_template[None], dt)[0]
        G_arr = np.zeros((n_m, n_t, 3), dtype=np.float32)
        for m in range(n_m):
            scale = float(np.sqrt(bvalues[m] / b_unit)) if b_unit > 0 else 0.0
            G_arr[m] = G_template * scale
        gradient_directions = np.tile(u, (n_m, 1))
        bvalues_num = _calc_b_from_waveform(G_arr, dt)
        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        seq = cls(G_arr, dt, bvalues_num, gradient_directions,
                  None, None, None, None, TE_)
        seq._carry(_minimum_te=T_total, _te_auto=te_auto, sequence_type='btensor_pte')
        seq.validate()
        seq._build_spec = ('from_btensor_pte', dict(
            bvalues=bvalues, plane_normal=plane_normal, delta=delta,
            Delta=Delta, TE=TE, n_t=n_t))
        return seq

    @classmethod
    def from_btensor_waveform(cls, G, dt, *, echo_idx=None, TE=None):
        """Wrap a precomputed b-tensor gradient waveform -- the PHYSICAL gradient, as played -- as a spin echo.

        For an externally designed b-tensor encoding (e.g. a dmipy-design ``design_waveform`` output)
        rather than the canonical square bipolar pairs of ``from_btensor_ste`` / ``from_btensor_pte``. The
        b-tensor SHAPE (spherical / planar / linear, i.e. b_delta) is whatever the numbers produce: it is
        computed from ``G_eff``, not declared, so there is no shape argument; ``scheme.btensor()`` reads
        the realised shape.

        The 180 is declared at ``echo_idx`` (default TE/2 -- the only position at which the static field
        refocuses at the echo, so a non-TE/2 ``echo_idx`` raises) and folds into ``G_eff`` through the
        schedule's sign, as for every family.

        Parameters
        ----------
        G : (n_t, 3) or (1, n_t, 3) array, T/m -- the physical gradient the scanner plays.
        dt : float -- time step (s).
        echo_idx : int or None -- sample index of the 180. Default TE/2.
        TE : float or None -- echo time; defaults to the waveform span.
        """
        G = np.asarray(G, dtype=np.float32)
        if G.ndim == 2:
            G = G[None]
        n_m, n_t, _ = G.shape
        dt = float(dt)
        T_total = (n_t - 1) * dt
        gradient_directions = np.tile([0., 0., 1.], (n_m, 1))
        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        te2 = int(round((float(TE_[0]) / 2.0) / dt))      # the spin-echo 180 position
        if echo_idx is None:
            echo_idx = te2
        echo_idx = int(np.clip(echo_idx, 0, n_t - 1))
        if abs(echo_idx - te2) > 1:                       # beyond rounding
            raise ValueError(
                f"from_btensor_waveform: echo_idx={echo_idx} places the 180 at {echo_idx * dt * 1e3:.2f} ms, not "
                f"TE/2={te2 * dt * 1e3:.2f} ms. A spin echo refocuses the static field at 2*t_180="
                f"{2 * echo_idx * dt * 1e3:.2f} ms, not at the echo TE={float(TE_[0]) * 1e3:.2f} ms.")
        schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(echo_idx * dt, 180, 'refocus')])
        sign = schedule.sign(np.arange(n_t) * dt)[None, :, None]
        bvalues_num = _calc_b_from_waveform(G * sign, dt)
        seq = cls(G, dt, bvalues_num, gradient_directions,
                  None, None, None, None, TE_)
        seq.rf_events = schedule
        apply_rf_schedule(seq)
        seq._carry(_minimum_te=T_total, _te_auto=te_auto, sequence_type='btensor')
        seq.validate()
        seq._build_spec = ('from_btensor_waveform', dict(echo_idx=echo_idx))
        return seq

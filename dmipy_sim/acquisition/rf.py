"""The RF side of an acquisition: the schedule of pulses (:class:`RFEvent`) and the played envelope
(:class:`B1Pulse`).

:class:`RFEvent` is the one spelling of a pulse -- the instant, the flip, the coherence label, the B1
axis, the duration, the carrier offset, and optionally the :class:`B1Pulse` that is its shape. Every
builder emits it, every reader reads it. :class:`RFSchedule` is the schedule -- the events in time
order, valid by construction, and the one place the coherence mask, the spin-echo sign, the refocus and
mixing times are derived from them. The only dict an event is ever read from is its own
:meth:`RFEvent.to_dict` record. :class:`B1Pulse` is the RF analogue of the gradient array ``G``:
the base representation is the *actual* transmit field

    B1(t) = B1x(t) + i B1y(t)   in Tesla, on a uniform raster dt,

NOT an idealised flip angle, and NOT the transverse-coherence mask a ``ScannerSequence``
derives (``chi_perp``, which describes a pulse's *effect* — which intervals are
transverse vs longitudinal). ``B1Pulse`` describes the pulse you actually play on the
coil, with the same status the ``G(t)`` array has for gradients: it is the ground truth,
and the constructors are conveniences on top of it — simple shapes (hard / windowed_sinc /
from_samples) and B1-robust ones (adiabatic_hs, adiabatic_half_passage, bir4, composite) that
also serve as warm starts for dmipy-design's RF optimiser.

Deliverability is the RF mirror of the gradient slew/amplitude limits: the peak |B1|,
the B1+rms / SAR proxy, and the transmit raster are read from
``acquisition.scanner_constants`` (per-vendor ``rf`` + ``safety`` catalogue), never
hard-coded.

The forward is a single-spin Bloch integrator (``bloch_simulate`` / ``slice_profile``)
that maps B1(t) -> magnetisation across an ensemble of off-resonance, B1+ transmit
scaling, and (for slice-selective pulses) spatial position. Crucially it does NOT
renormalise to a prescribed flip: the flip that emerges is whatever ``gamma * integral
B1 dt`` and the off-resonance tilt produce. This is exactly the object DeepRF (Shin et
al., Nat. Mach. Intell. 2021) designs: a complex envelope on a ~10 us raster,
peak-B1/SAR constrained, scored by a Bloch-simulated magnetisation profile.
"""

from dataclasses import dataclass
import numpy as np

from ..constants import GAMMA                       # rad / (s * T)
from . import scanner_constants as scc


@dataclass
class B1Pulse:
    """A complex transmit waveform B1(t) in Tesla on a uniform raster dt.

    Attributes
    ----------
    b1 : np.ndarray, complex, shape (n,)
        B1x + i*B1y in Tesla at each raster sample.  The rotation axis at sample k
        is along the phase angle(b1[k]); the instantaneous nutation rate is
        gamma*|b1[k]|.
    dt : float
        Raster (dwell) time in seconds.  Typical transmit raster is ~1-10 us.
    label : str
        Free-form tag ('exc', 'refocus', 'inv', ...).  Not used by the physics.
    """
    b1: np.ndarray
    dt: float
    label: str = ""
    flip_deg: float = None          # design/intended net flip (deg); None if unknown.
                                    # the on-resonance Bloch flip is used if this is None.

    def __post_init__(self):
        self.b1 = np.asarray(self.b1, dtype=np.complex128)
        if self.b1.ndim != 1:
            raise ValueError("b1 must be 1-D (n_samples,)")
        self.dt = float(self.dt)

    # ── geometry / basic quantities ──────────────────────────────────────────
    @property
    def n(self):
        return self.b1.shape[0]

    @property
    def duration(self):
        """Pulse duration in seconds."""
        return self.n * self.dt

    @property
    def times(self):
        """Sample times (s), left edge of each raster step."""
        return np.arange(self.n) * self.dt

    @property
    def magnitude(self):
        """|B1(t)| in Tesla."""
        return np.abs(self.b1)

    @property
    def phase(self):
        """angle(B1(t)) in radians."""
        return np.angle(self.b1)

    @property
    def peak_b1(self):
        """Peak |B1| in Tesla -- the amplitude-limited quantity."""
        return float(np.max(np.abs(self.b1))) if self.n else 0.0

    @property
    def b1_rms(self):
        """Root-mean-square B1 over the pulse, in Tesla (drives B1+rms / SAR)."""
        return float(np.sqrt(np.mean(np.abs(self.b1) ** 2))) if self.n else 0.0

    @property
    def sar_proxy(self):
        """Integral of |B1|^2 dt (T^2 s) -- proportional to deposited RF energy / SAR."""
        return float(np.sum(np.abs(self.b1) ** 2) * self.dt)

    @property
    def nominal_flip_rad(self):
        """Integrated nutation gamma * integral |B1| dt (rad).

        The on-resonance, phase-coherent flip "area" of the pulse.  For a real
        envelope (constant phase) this is the actual on-resonance flip; for a
        genuinely complex pulse it is the total nutation magnitude (a label, not
        the off-resonance flip -- use bloch_simulate for that)."""
        return float(GAMMA * np.sum(np.abs(self.b1)) * self.dt)

    @property
    def nominal_flip_deg(self):
        return np.degrees(self.nominal_flip_rad)

    @property
    def max_slew_b1(self):
        """Max |dB1/dt| in T/s -- the RF-amplifier envelope rate (modulation bandwidth)."""
        if self.n < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self.b1))) / self.dt)

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def from_samples(cls, b1, dt, label=""):
        """Wrap an explicit complex (or real) B1(t) array in Tesla."""
        return cls(b1=np.asarray(b1, dtype=np.complex128), dt=dt, label=label)

    @classmethod
    def from_magnitude_phase(cls, mag_T, phase_rad, dt, label=""):
        """Build B1(t) from magnitude (Tesla) and phase (rad) arrays."""
        mag = np.asarray(mag_T, float)
        ph = np.zeros_like(mag) if phase_rad is None else np.asarray(phase_rad, float)
        return cls(b1=mag * np.exp(1j * ph), dt=dt, label=label)

    @classmethod
    def hard(cls, flip_deg, duration, dt, phase_deg=0.0, label="hard"):
        """Constant-amplitude rectangular pulse of the given flip and duration.

        Amplitude solved from gamma*B1*duration = flip, i.e. B1 = flip/(gamma*duration).
        This is the deliverable analogue of an instantaneous hard pulse."""
        n = max(1, int(round(float(duration) / float(dt))))
        flip = np.deg2rad(float(flip_deg))
        amp = flip / (GAMMA * n * float(dt))                  # T, exact for the discretised area
        b1 = np.full(n, amp, dtype=np.complex128) * np.exp(1j * np.deg2rad(phase_deg))
        return cls(b1=b1, dt=dt, label=label, flip_deg=float(flip_deg))

    @classmethod
    def windowed_sinc(cls, flip_deg, duration, dt, time_bw=4.0, n_zeros=None,
                      window="hamming", phase_deg=0.0, label="sinc"):
        """Apodised sinc envelope (the standard slice-selective excitation/refocus shape).

        ``time_bw`` is the time-bandwidth product (number of side-lobe zero crossings
        sets the slice sharpness).  The amplitude is scaled so the integrated nutation
        equals ``flip_deg`` (small-tip / linear-phase convention)."""
        n = max(3, int(round(float(duration) / float(dt))))
        nz = float(n_zeros) if n_zeros is not None else float(time_bw) / 2.0
        x = np.linspace(-nz, nz, n)
        env = np.sinc(x)
        if window == "hamming":
            env = env * (0.54 + 0.46 * np.cos(np.pi * x / nz))
        elif window in (None, "none", "rect"):
            pass
        else:
            raise ValueError(f"unknown window {window!r}")
        flip = np.deg2rad(float(flip_deg))
        area = GAMMA * np.sum(np.abs(env)) * float(dt)        # nutation per unit amplitude
        amp = flip / area if area > 0 else 0.0
        b1 = (env * amp).astype(np.complex128) * np.exp(1j * np.deg2rad(phase_deg))
        return cls(b1=b1, dt=dt, label=label, flip_deg=float(flip_deg))

    # ── adiabatic / composite shapes (B1-robust) ──────────────────────────────
    @classmethod
    def adiabatic_hs(cls, duration, dt, *, peak_b1, mu=4.0, beta=5.3, label="adiabatic_hs"):
        """Hyperbolic-secant adiabatic full passage (Silver, Joseph & Hoult 1985).

        ``B1(t) = peak_b1 · sech(βτ)^(1+iμ)`` over ``τ = 2t/T − 1 ∈ [−1, 1]``: a sech amplitude
        with a tanh frequency sweep of bandwidth ``≈ 2μβ/(πT)``.  A B1-robust inversion /
        refocusing pulse — the magnetisation follows the swept effective field to −z regardless
        of ``B1⁺`` (above the adiabatic threshold).  ``peak_b1`` is the peak amplitude (Tesla);
        raise it or lengthen ``duration`` for stronger adiabaticity."""
        n = max(3, int(round(float(duration) / float(dt))))
        tau = np.linspace(-1.0, 1.0, n)
        sech = 1.0 / np.cosh(beta * tau)
        b1 = peak_b1 * sech * np.exp(1j * mu * np.log(sech + 1e-300))
        return cls(b1=b1.astype(np.complex128), dt=dt, label=label, flip_deg=180.0)

    @classmethod
    def adiabatic_half_passage(cls, duration, dt, *, peak_b1, beta=6.0, kappa=np.arctan(20.0),
                               sweep_hz=1.0e4, label="ahp"):
        """Adiabatic half-passage: a B1-insensitive 90° excitation (+z → transverse).

        Amplitude ramps ``0 → peak_b1`` (tanh) while the frequency sweeps ``+sweep_hz → 0``
        (tan), so the effective field rotates from +z into the transverse plane and the
        magnetisation follows it — B1-insensitively.  Half of an adiabatic full passage; the
        building block of BIR-4."""
        n = max(3, int(round(float(duration) / float(dt))))
        tau = np.linspace(0.0, 1.0, n)
        amp = peak_b1 * np.tanh(beta * tau)
        freq = sweep_hz * np.tan(kappa * (1.0 - tau)) / np.tan(kappa)
        phase = np.cumsum(2.0 * np.pi * freq * float(dt))
        return cls(b1=(amp * np.exp(1j * phase)).astype(np.complex128), dt=dt, label=label,
                   flip_deg=90.0)

    @classmethod
    def bir4(cls, flip_deg, duration, dt, *, peak_b1, beta=6.0, kappa=np.arctan(20.0),
             sweep_hz=1.0e4, label="bir4"):
        """BIR-4: a B1-insensitive rotation by an ARBITRARY angle (Garwood & Ke 1991).

        Four adiabatic half-passages (``adiabatic_half_passage`` shape) with the frequency sweep
        sign-alternated across segments and two phase jumps of ``φ = flip_deg/2`` on the middle
        two segments; the net flip is ``≈ flip_deg`` and it is held across a wide ``B1⁺`` range
        (unlike a full passage, which only inverts).  ``peak_b1`` is the peak amplitude (T)."""
        n = max(8, int(round(float(duration) / float(dt))))
        q = n // 4
        tau = np.linspace(0.0, 1.0, q)
        amp0 = np.tanh(beta * tau)
        fr0 = sweep_hz * np.tan(kappa * (1.0 - tau)) / np.tan(kappa)
        fm = (1.0, -1.0, 1.0, -1.0)
        phi = np.deg2rad(float(flip_deg) / 2.0)
        pj = (0.0, phi, phi, 0.0)
        amp = np.concatenate([amp0[::-1] if k % 2 else amp0 for k in range(4)])
        fr = np.concatenate([(fr0[::-1] if k % 2 else fr0) * fm[k] for k in range(4)])
        jump = np.concatenate([np.full(q, pj[k]) for k in range(4)])
        gph = np.cumsum(2.0 * np.pi * fr * float(dt))
        b1 = peak_b1 * amp * np.exp(1j * (gph + jump))
        return cls(b1=b1.astype(np.complex128), dt=dt, label=label, flip_deg=float(flip_deg))

    @classmethod
    def composite(cls, segments, dt, *, peak_b1, label="composite"):
        """Composite pulse: a sequence of constant-amplitude hard sub-pulses.

        ``segments`` is a list of ``(flip_deg, phase_deg)``; each is a hard pulse at ``peak_b1``
        with duration ``|flip| / (γ·peak_b1)``, concatenated.  Robustness comes from the phase
        pattern (e.g. Levitt's ``[(90, 0), (180, 90), (90, 0)]`` inversion)."""
        parts = []
        for flip_deg, phase_deg in segments:
            dur = abs(np.deg2rad(float(flip_deg))) / (GAMMA * float(peak_b1))
            parts.append(cls.hard(flip_deg, dur, dt, phase_deg=phase_deg).b1)
        return cls(b1=np.concatenate(parts).astype(np.complex128), dt=dt, label=label)

    # ── deliverability (read limits from scanner_constants) ───────────────────
    def deliverability(self, model, *, coil="body", tol=1.02):
        """Return a dict of deliverability checks against a vendor scanner.

        Reads peak-B1 (rf catalogue) and, when available, the IEC B1+rms / raster.
        ``tol`` allows a small (2%) numerical margin, matching the gradient checks.
        """
        peak_name = f"peak_B1_{coil}_coil"
        peak_lim = scc.get_limit(model, "rf", peak_name, si=True)     # Tesla (raises if absent)
        try:                                                          # raster not catalogued for every vendor
            raster = scc.get_limit(model, "rf", "rf_raster_time", si=True)
        except KeyError:
            raster = None
        rep = {
            "model": model,
            "peak_b1_T": self.peak_b1,
            "peak_b1_limit_T": peak_lim,
            "peak_b1_ok": self.peak_b1 <= peak_lim * tol,
            "raster_s": self.dt,
            "raster_limit_s": raster,
            "raster_ok": (raster is None) or (self.dt >= raster / tol),
            "b1_rms_T": self.b1_rms,
            "sar_proxy_T2s": self.sar_proxy,
            "duration_s": self.duration,
            "nominal_flip_deg": self.nominal_flip_deg,
        }
        rep["deliverable"] = bool(rep["peak_b1_ok"] and rep["raster_ok"])
        return rep

    def is_deliverable(self, model, *, coil="body", tol=1.02):
        return self.deliverability(model, coil=coil, tol=tol)["deliverable"]


# ── Bloch forward: B1(t) -> magnetisation over an ensemble ────────────────────
def bloch_simulate(pulse, df_hz=0.0, b1_scale=1.0, M0=None, T1=np.inf, T2=np.inf,
                   return_history=False):
    """Integrate the Bloch equation for ``pulse`` over an ensemble of spins.

    Each spin is defined by an off-resonance ``df_hz`` (Hz) and a B1+ transmit
    multiplier ``b1_scale``; both broadcast to a common ensemble shape (E,).  The
    per-step rotation is the standard hard-step (Rodrigues) about the effective field

        n = ( gamma*B1x*b1_scale*dt , gamma*B1y*b1_scale*dt , 2*pi*df*dt )   [rad]

    (an RF nutation increment plus the free-precession about z over the step).  No flip
    renormalisation: the flip is whatever the field produces.

    Parameters
    ----------
    pulse : B1Pulse
    df_hz : float or array        off-resonance per spin (Hz)
    b1_scale : float or array     B1+ transmit multiplier per spin (1.0 = ideal)
    M0 : array (3,) or (3, E)     initial magnetisation (default +z)
    T1, T2 : float                relaxation times (s); default inf (design limit)
    return_history : bool         also return M at every step, shape (n+1, 3, E)

    Returns
    -------
    Mxy : complex array (E,)       transverse magnetisation at end (Mx + i My)
    Mz  : array (E,)               longitudinal magnetisation at end
    (history) : (n+1, 3, E) if return_history
    """
    df = np.asarray(df_hz, float)
    bs = np.asarray(b1_scale, float)
    E = int(np.broadcast(df, bs).size)
    df = np.broadcast_to(df, (E,)).astype(float)
    bs = np.broadcast_to(bs, (E,)).astype(float)

    M = np.zeros((3, E), float)
    if M0 is None:
        M[2] = 1.0
    else:
        M0 = np.asarray(M0, float)
        M[:] = M0[:, None] if M0.ndim == 1 else M0

    dt = pulse.dt
    nz0 = 2.0 * np.pi * df * dt                               # off-resonance z-rotation / step
    e1 = np.exp(-dt / T1) if np.isfinite(T1) else 1.0
    e2 = np.exp(-dt / T2) if np.isfinite(T2) else 1.0

    hist = None
    if return_history:
        hist = np.empty((pulse.n + 1, 3, E), float)
        hist[0] = M

    for k in range(pulse.n):
        b1k = pulse.b1[k]
        nx = GAMMA * (b1k.real * bs) * dt
        ny = GAMMA * (b1k.imag * bs) * dt
        nzk = nz0                                            # (E,)
        theta = np.sqrt(nx * nx + ny * ny + nzk * nzk)
        # Rodrigues; guard theta=0
        small = theta < 1e-30
        th = np.where(small, 1.0, theta)
        kx, ky, kz = nx / th, ny / th, nzk / th
        c = np.cos(th)
        s = np.sin(th)
        omc = 1.0 - c
        Mx, My, Mz = M[0], M[1], M[2]
        kdotM = kx * Mx + ky * My + kz * Mz
        # cross product k x M
        cx = ky * Mz - kz * My
        cy = kz * Mx - kx * Mz
        cz = kx * My - ky * Mx
        Mx2 = Mx * c + cx * s + kx * kdotM * omc
        My2 = My * c + cy * s + ky * kdotM * omc
        Mz2 = Mz * c + cz * s + kz * kdotM * omc
        # where theta ~ 0, rotation is identity
        Mx2 = np.where(small, Mx, Mx2)
        My2 = np.where(small, My, My2)
        Mz2 = np.where(small, Mz, Mz2)
        # relaxation over the step (decay toward +z equilibrium = 1)
        M = np.stack([Mx2 * e2, My2 * e2, Mz2 * e1 + (1.0 - e1)])
        if return_history:
            hist[k + 1] = M

    Mxy = M[0] + 1j * M[1]
    if return_history:
        return Mxy, M[2], hist
    return Mxy, M[2]


def slice_profile(pulse, slice_gradient, positions_m, df_hz=0.0, b1_scale=1.0,
                  **kw):
    """Slice profile: magnetisation vs spatial position along the slice axis.

    A slice-select gradient ``slice_gradient`` (T/m) makes a spin at position z see
    an extra off-resonance gamma*Gss*z, i.e. (gamma/2pi)*Gss*z Hz; this is added to
    any intrinsic ``df_hz``.  Returns (positions_m, Mxy, Mz)."""
    z = np.asarray(positions_m, float)
    df_slice = (GAMMA / (2.0 * np.pi)) * float(slice_gradient) * z   # Hz
    Mxy, Mz = bloch_simulate(pulse, df_hz=df_slice + np.asarray(df_hz, float),
                             b1_scale=b1_scale, **kw)
    return z, Mxy, Mz


# ── the RF schedule: one event dialect ────────────────────────────────────────────────────────────
@dataclass(frozen=True, eq=False)
class RFEvent:
    """One pulse of a sequence's RF schedule: the one dialect every builder emits and every reader reads.

    ``t_s`` is the pulse's instant (the centre of a finite pulse), ``flip_deg`` its nominal flip, ``label``
    the coherence role the schedule readers key on (``'Mz→Mxy'`` excitation, ``'store'`` / ``'recall'`` of
    a stimulated echo, ``'refocus'``), ``axis_deg`` the B1 phase (0 = x, 90 = y), ``duration_s`` the pulse
    length (0 is the instantaneous hard pulse), ``offset_hz`` the carrier offset. ``envelope`` is the played
    ``B1(t)`` as a :class:`B1Pulse`; when given it IS the shape: ``duration_s`` is its length, ``flip_deg``
    is ``gamma * int |B1| dt``, and a declared value that disagrees with either raises -- the rule
    ``chi_perp`` follows on the ``ScannerSequence`` that carries the schedule.

    :meth:`flip_split` is the one place a pulse becomes what a grid does with it; both rasterisers
    (:func:`dmipy_sim.engine.bloch._build_rf_schedule`, :func:`dmipy_sim.replay.trajectories._bloch_timeline`)
    read it, so a hard pulse, a finite hard pulse and a shaped pulse are the same object at three settings.
    A sequence's pulses live in an :class:`RFSchedule`.
    """
    t_s: float
    flip_deg: float = None
    label: str = ""
    axis_deg: float = 0.0
    duration_s: float = 0.0
    offset_hz: float = 0.0
    envelope: B1Pulse = None

    def __post_init__(self):
        _set = lambda k, v: object.__setattr__(self, k, v)
        _set("t_s", float(self.t_s)); _set("label", str(self.label or ""))
        _set("axis_deg", float(self.axis_deg or 0.0)); _set("offset_hz", float(self.offset_hz or 0.0))
        dur = float(self.duration_s or 0.0)
        if dur < 0.0:
            raise ValueError("duration_s must be >= 0")
        if self.envelope is not None:
            env = self.envelope
            if not isinstance(env, B1Pulse):
                raise TypeError(f"envelope must be a B1Pulse, got {type(env).__name__}")
            if dur > 0.0 and abs(dur - env.duration) > 1e-9 * max(dur, env.duration):
                raise ValueError(f"duration_s = {dur:.6g} s disagrees with the envelope's {env.duration:.6g} s")
            dur = env.duration
            nominal = env.nominal_flip_deg
            if self.flip_deg is None:
                _set("flip_deg", float(nominal))
            elif abs(float(self.flip_deg) - nominal) > 1e-6 * max(abs(nominal), 1.0):
                raise ValueError(f"flip_deg = {float(self.flip_deg):.6g} disagrees with the envelope's "
                                 f"gamma * int |B1| dt = {nominal:.6g} deg")
        if self.flip_deg is None:
            raise ValueError("an RFEvent needs flip_deg, or an envelope to derive it from")
        _set("flip_deg", float(self.flip_deg)); _set("duration_s", dur)

    @property
    def is_hard(self):
        return self.duration_s == 0.0

    @property
    def window(self):
        """``(t0, t1)`` the pulse occupies; a hard pulse occupies its instant."""
        return self.t_s - self.duration_s / 2.0, self.t_s + self.duration_s / 2.0

    def flip_split(self, nsub):
        """``(dflips_rad, axes_rad)`` of the ``nsub`` sub-rotations that make up this pulse, in time order.

        Without an envelope the flip splits evenly about the one axis. With one it splits as ``int |B1| dt``
        over ``nsub`` equal sub-windows -- the total is exactly ``flip_deg`` at any ``nsub`` -- and each
        sub-rotation's axis is ``axis_deg`` plus the phase of ``int B1 dt`` over its window (exact for a
        constant-phase pulse, the mean phase otherwise).
        """
        nsub = max(1, int(nsub))
        total, ax0 = np.deg2rad(self.flip_deg), np.deg2rad(self.axis_deg)
        if self.envelope is None or nsub == 1:
            return np.full(nsub, total / nsub), np.full(nsub, ax0)
        b1 = self.envelope.b1
        grid = np.arange(b1.shape[0] + 1, dtype=np.float64)
        edges = np.linspace(0.0, b1.shape[0], nsub + 1)
        cum = lambda x: np.diff(np.interp(edges, grid, np.concatenate([[0.0], np.cumsum(x)])))
        mag = cum(np.abs(b1)); re_, im_ = cum(b1.real), cum(b1.imag)
        dflips = total * mag / (mag.sum() + 1e-300)
        axes = ax0 + np.where(mag > 0.0, np.angle(re_ + 1j * im_), 0.0)
        return dflips, axes

    def to_dict(self):
        """The event as plain data (the envelope as its samples): what a ``.seq`` definition or a JSON holds."""
        d = {"t_s": self.t_s, "flip_deg": self.flip_deg, "label": self.label}
        for k in ("axis_deg", "duration_s", "offset_hz"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.envelope is not None:
            d["envelope"] = {"b1_re": self.envelope.b1.real.tolist(), "b1_im": self.envelope.b1.imag.tolist(),
                             "dt": self.envelope.dt}
        return d

    @classmethod
    def from_dict(cls, d):
        """The inverse of :meth:`to_dict`: exactly its keys, an ``envelope`` as ``{b1_re, b1_im, dt}``. This is
        deserialisation of what dmipy-sim wrote (a ``.seq`` definition, a JSON), not an alternative spelling:
        an unknown key raises. :meth:`RFSchedule.from_dicts` reads a whole schedule."""
        unknown = set(d) - {"t_s", "flip_deg", "label", "axis_deg", "duration_s", "offset_hz", "envelope"}
        if unknown:
            raise ValueError(f"not an RFEvent.to_dict record: unknown keys {sorted(unknown)}")
        env = d.get("envelope")
        if env is not None:
            env = B1Pulse(np.asarray(env["b1_re"], float) + 1j * np.asarray(env["b1_im"], float), float(env["dt"]))
        return cls(t_s=d["t_s"], flip_deg=d.get("flip_deg"), label=d.get("label", ""), axis_deg=d.get("axis_deg", 0.0),
                   duration_s=d.get("duration_s", 0.0), offset_hz=d.get("offset_hz", 0.0), envelope=env)

    def _key(self):
        return (self.t_s, self.flip_deg, self.label, self.axis_deg, self.duration_s, self.offset_hz)

    def __eq__(self, other):
        if not isinstance(other, RFEvent) or self._key() != other._key():
            return False
        a, b = self.envelope, other.envelope
        return (a is None and b is None) or (a is not None and b is not None and a.dt == b.dt
                                              and np.array_equal(a.b1, b.b1))

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        extra = "".join(f", {k}={getattr(self, k):g}" for k in ("axis_deg", "duration_s", "offset_hz") if getattr(self, k))
        env = f", envelope=<B1Pulse {self.envelope.n}x{self.envelope.dt:g}s>" if self.envelope is not None else ""
        return f"RFEvent({self.t_s:g}, {self.flip_deg:g}, {self.label!r}{extra}{env})"


#: a labelled pulse is what its label says, whatever its flip -- a stimulated echo's store may be 60 degrees
_ROLE_OF_LABEL = {"Mz→Mxy": "excite", "excitation": "excite", "store": "store", "recall": "recall",
                  "refocus": "refocus", "refocusing": "refocus"}


class RFSchedule(tuple):
    """The RF schedule of an acquisition: its :class:`RFEvent`\ s in time order, valid by construction, and
    the one place anything is derived from them. Empty is a gradient echo (no pulse).

    Build it once and hold it -- ``ScannerSequence.rf`` is one -- and read
    what it derives: :meth:`coherence` (the transverse mask, mixing time, stimulated-echo state and echo
    times an ideal schedule implies), :meth:`sign` (the spin-echo gate ``s(t)`` of RPK.md 6.6, the un-fold
    between the effective and the physical gradient), :attr:`refocus_time`, :attr:`mixing_time`,
    Nothing else re-derives these from a list of events. Anything that is not an
    ``RFEvent`` is refused; :meth:`to_dicts` / :meth:`from_dicts` are its serialised record.
    """
    __slots__ = ()

    def __new__(cls, events=()):
        if events is None:
            events = ()
        if isinstance(events, RFSchedule):
            return events
        evs = tuple(events)
        for e in evs:
            if not isinstance(e, RFEvent):
                raise TypeError(f"an RF event is an RFEvent, got {type(e).__name__}: build it as "
                                f"RFEvent(t_s, flip_deg, label, ...) -- or RFSchedule.from_dicts for serialised data")
        return super().__new__(cls, sorted(evs, key=lambda e: e.t_s))

    def __repr__(self):
        return "RFSchedule(" + ", ".join(repr(e) for e in self) + ")"

    def __getnewargs__(self):
        return (tuple(self),)

    @property
    def refocus_time(self):
        """The instant of the first refocusing pulse (the one a spin-echo gate flips at), or ``None``: a pulse
        labelled ``refocus`` whatever its flip (an adiabatic passage's integrated nutation is far from 180), or
        an unlabelled 180."""
        for e in self:
            if _ROLE_OF_LABEL.get(e.label) == "refocus" or (not e.label and int(round(e.flip_deg)) == 180):
                return e.t_s
        return None

    def sign(self, t_grid):
        """The sign of the EFFECTIVE gradient over ``t_grid`` -- ``(len(t_grid),)`` of +-1: the spin-echo gate
        ``s(t)`` (RPK.md 6.6), and the one un-fold between the two gradients a sequence has (the PHYSICAL one a
        scanner plays, the EFFECTIVE one the phase integral walks). ``s`` is +-1 and its own inverse.

        A refocusing pulse inverts the accumulated phase, so ``s`` flips after it; a stimulated echo does the same
        across its storage/recall pair, so ``s`` flips at RECALL. It is a sign, not the quadrature weight
        :func:`dmipy_sim.replay._replay_kernel.se_gate` builds for integrating against a path: that one half-weights
        the grid endpoints, right under an integral and wrong for a waveform.
        """
        t = np.asarray(t_grid, dtype=np.float64)
        s = np.ones_like(t, dtype=np.float32)
        for e in self:
            if abs(e.flip_deg - 180.0) < 20.0 or e.label in ('refocus', 'recall'):
                s[t >= e.t_s - 1e-12] *= -1.0                    # a sample AT the pulse (to rounding) is after it
        return s

    def coherence(self, n_t, dt):
        """Coherence bookkeeping of the schedule on an ``n_t``-sample grid of ``dt``.

        Returns ``(chi_perp, TM, stimulated_echo, echo_times)``: the transverse-coherence mask, the total
        longitudinal-storage time (``None`` when there is none), whether the readout is a stimulated echo (a
        store / recall pair), and the echo times of the refocusing pulses. Magnetisation starts along z; a 90
        excites it, a 90 while transverse stores it along z, the next 90 recalls it; a 180 while transverse
        refocuses, forming an echo at ``2 t_180 - t_ref`` where ``t_ref`` is the previous echo or excitation.
        A labelled pulse plays the role its label says whatever its flip (a stimulated echo's store may be 60
        degrees); an unlabelled one is inferred from its flip, and other flips are not tracked. Each transition
        happens at the pulse's instant ``t_s``.

        For a hard pulse the mask is binary. Over a FINITE pulse's window it is the transverse fraction of the
        pathway, averaged over the ensemble's azimuth: an excitation or a recall tips z into the plane as
        ``sin^2(theta)``, a store tips the plane onto z as ``cos^2(theta)``, with ``theta`` running 0 to pi/2
        across the pulse; a 180 keeps the component along B1 transverse and swings the perpendicular one through
        z, ``1/2 + cos^2(theta)/2`` with ``theta`` 0 to pi -- a quarter of the pulse spent longitudinal in all,
        the ensemble mean of :func:`dmipy_sim.replay.trajectories.finite_180_longitudinal_dwell`. The mask is
        then float; a schedule of hard pulses keeps the binary one.
        """
        n_t = int(n_t); dt = float(dt)
        if len(self) == 0:                                  # no pulses declared: the gradient is read as it stands
            return np.ones(n_t, dtype=bool), None, False, []
        chi = np.zeros(n_t, dtype=bool)
        transverse = False
        t_ref = None
        stored_from = None
        TM = 0.0
        stores = 0
        echoes = []
        i_prev = 0
        roles = []                                          # (event, role) for the finite-pulse profiles
        for e in self:
            t = e.t_s
            i = int(np.clip(int(round(t / dt)), 0, n_t))
            chi[i_prev:i] = transverse
            i_prev = i
            role = _ROLE_OF_LABEL.get(e.label)
            if role is None:                                    # unlabelled: infer from the flip and the state
                flip = int(round(e.flip_deg))
                if flip == 90:
                    role = "store" if transverse else ("recall" if stored_from is not None else "excite")
                elif flip == 180 and transverse and t_ref is not None:
                    role = "refocus"
            if role in ("excite", "recall") and not transverse:
                transverse = True
                if role == "recall" and stored_from is not None:
                    TM += t - stored_from
                    stored_from = None
                t_ref = t
                roles.append((e, role))
            elif role == "store" and transverse:
                transverse = False
                stored_from = t
                stores += 1
                roles.append((e, "store"))
            elif role == "refocus" and transverse and t_ref is not None:
                echoes.append(2.0 * t - t_ref)
                t_ref = echoes[-1]
                roles.append((e, "refocus"))
        chi[i_prev:] = transverse
        finite = [(e, role) for e, role in roles if e.duration_s > 0.0]
        if finite:
            chi = chi.astype(np.float64)
            tg = np.arange(n_t) * dt
            for e, role in finite:
                t0, t1 = e.window
                inside = (tg >= t0 - 1e-12) & (tg <= t1 + 1e-12)
                frac = np.clip((tg[inside] - t0) / e.duration_s, 0.0, 1.0)
                if role in ("excite", "recall"):
                    chi[inside] = np.sin(0.5 * np.pi * frac) ** 2
                elif role == "store":
                    chi[inside] = np.cos(0.5 * np.pi * frac) ** 2
                else:                                           # refocus
                    chi[inside] = 0.5 + 0.5 * np.cos(np.pi * frac) ** 2
        return chi, (TM if TM > 0.0 else None), stores > 0, echoes

    @property
    def mixing_time(self):
        """``(TM, stimulated_echo)`` from the pulses alone: a stimulated echo is defined by a store / recall pair,
        so the mixing time is their gap -- by label where the schedule is labelled, else by the flip pattern
        (three 90s: TM between the second and third). A 90/180 spin echo is ``(None, False)``."""
        by_label = {e.label.lower(): e for e in self}
        if 'store' in by_label and 'recall' in by_label:
            return by_label['recall'].t_s - by_label['store'].t_s, True
        nineties = [e for e in self if abs(e.flip_deg - 90.0) < 1e-3]
        if len(nineties) >= 3:
            return nineties[2].t_s - nineties[1].t_s, True
        return None, False

    def shifted(self, dt_s):
        """The same pulses ``dt_s`` later."""
        from dataclasses import replace
        return RFSchedule(replace(e, t_s=e.t_s + float(dt_s)) for e in self)

    def to_dicts(self):
        """The schedule as plain records (:meth:`RFEvent.to_dict` each): a ``.seq`` definition, a JSON."""
        return [e.to_dict() for e in self]

    @classmethod
    def from_dicts(cls, records):
        """The inverse of :meth:`to_dicts`; each record through :meth:`RFEvent.from_dict`, strictly."""
        return cls(RFEvent.from_dict(d) for d in (records or ()))

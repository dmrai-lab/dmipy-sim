"""Continuous RF pulse representation: the complex B1(t) envelope.

This is the RF analogue of the gradient ``Waveform`` (``waveforms.py``): the base
representation is the *actual* transmit field

    B1(t) = B1x(t) + i B1y(t)   in Tesla, on a uniform raster dt,

NOT an idealised flip angle, and NOT the transverse-coherence mask a ``Waveform``
carries (``Waveform.chi_perp``, which describes a pulse's *effect* — which intervals are
transverse vs longitudinal). ``B1Pulse`` describes the pulse you actually play on the
coil, with the same status the ``G(t)`` array has for gradients: it is the ground truth,
and the constructors (hard / windowed-sinc / from_samples) are conveniences on top of it.

Deliverability is the RF mirror of the gradient slew/amplitude limits: the peak |B1|,
the B1+rms / SAR proxy, and the transmit raster are read from
``sequences.scanner_constants`` (per-vendor ``rf`` + ``safety`` catalogue), never
hard-coded.

The forward is a single-spin Bloch integrator (``bloch_simulate`` / ``slice_profile``)
that maps B1(t) -> magnetisation across an ensemble of off-resonance, B1+ transmit
scaling, and (for slice-selective pulses) spatial position. Crucially it does NOT
renormalise to a prescribed flip: the flip that emerges is whatever ``gamma * integral
B1 dt`` and the off-resonance tilt produce. This is exactly the object DeepRF (Shin et
al., Nat. Mach. Intell. 2021) designs: a complex envelope on a ~10 us raster,
peak-B1/SAR constrained, scored by a Bloch-simulated magnetisation profile.
"""

from dataclasses import dataclass, field
import numpy as np

from .constants import GAMMA                       # rad / (s * T)
from .sequences import scanner_constants as scc


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

"""The one acquisition object: what the scanner does from t = 0 to the echo.

A :class:`ScannerSequence` is the gradient the scanner plays, the pulses it fires, the budget it was built to
and the samples it reads -- and nothing about tissue, walkers or packs. Everything a consumer once asked of
three containers is one field or one derivation here:

* ``G`` -- the PHYSICAL gradient ``(n_meas, n_t, 3)`` in T/m on ``dt``; a spin echo's lobes share a sign.
  :attr:`G_eff` is the effective gradient the phase integral walks, ``G * rf.sign(t)`` (RPK.md 6.6), derived.
* ``rf`` -- the :class:`~dmipy_sim.acquisition.rf.RFSchedule`. :attr:`chi_perp`, :attr:`TM`,
  :attr:`stimulated_echo`, :attr:`echoes` and :attr:`refocus_gap` are derived from it and ``G``.
* ``readout`` -- the sample indices read; :attr:`echo_idx` is the last. Defaults to the grid's last sample (the
  grid ends at the readout), or to every echo of a multi-echo train; it must agree with where the schedule forms its echo.
* ``timing`` -- the :class:`~dmipy_sim.acquisition.timing.SequenceTiming` budget, when built to one.
* ``encoding`` -- the per-measurement :class:`Encoding` an analytical layer reads (b, directions, delta, ...).
* ``crusher`` -- the emergent voxel-scale crusher the vector-Bloch engine models as windings over windows.

The scalar engine, the b integrals and the pack's replay read ``G_eff``; the vector-Bloch routes read ``G`` and
apply ``rf`` themselves. A :class:`Protocol` is a tuple of these -- one echo time each -- for a multi-TE scheme.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from .rf import RFSchedule
from .timing import SequenceTiming

__all__ = ["Encoding", "ScannerSequence", "Protocol", "ECHO_TOL"]

#: samples: the rounding freedom of placing a pulse at a lobe midpoint
ECHO_TOL = 2

#: relative net gradient moment |q(TE)| / max|q| above which an echo is not refocused
REFOCUS_ATOL = 1e-3


@dataclass(frozen=True)
class Encoding:
    """The per-measurement encoding an analytical layer reads: what the gradient means, not what it is.

    ``bvalues`` (s/m^2), ``gradient_directions`` (unit vectors), ``TE`` (s) per measurement; ``qvalues``,
    ``gradient_strengths``, ``delta``, ``Delta`` where the family defines them; the family's own parameters
    (``oscillation_frequency`` ... ``cpmg_n_echoes``); and ``minimum_te`` / ``te_auto``, the echo time floor
    the gradient schedule sets and whether ``TE`` was left to it.
    """
    bvalues: np.ndarray
    gradient_directions: np.ndarray
    TE: np.ndarray = None
    qvalues: np.ndarray = None
    gradient_strengths: np.ndarray = None
    delta: np.ndarray = None
    Delta: np.ndarray = None
    minimum_te: float = None
    te_auto: bool = False
    tau_perp_SE: np.ndarray = None
    ste_flip_angles: tuple = None
    ramp_time: np.ndarray = None
    oscillation_frequency: np.ndarray = None
    gradient_rise_time: np.ndarray = None
    n_oscillation_cycles: np.ndarray = None
    gradient_duration: np.ndarray = None
    cpmg_n_echoes: int = None
    cpmg_TE: float = None
    cpmg_beta_deg: float = None
    n_t_per_echo: int = None
    refocused: bool = None

    def __post_init__(self):
        object.__setattr__(self, "bvalues", np.asarray(self.bvalues, dtype=np.float64))
        object.__setattr__(self, "gradient_directions", np.asarray(self.gradient_directions, dtype=np.float64))
        for k in ("TE", "qvalues", "gradient_strengths", "delta", "Delta", "tau_perp_SE", "ramp_time",
                  "oscillation_frequency", "gradient_rise_time", "n_oscillation_cycles", "gradient_duration"):
            v = getattr(self, k)
            if v is not None:
                object.__setattr__(self, k, np.asarray(v, dtype=np.float64))

    @property
    def number_of_measurements(self):
        return int(self.gradient_directions.shape[0])


@dataclass(frozen=True, eq=False)
class ScannerSequence:
    """What the scanner does from t = 0 to the echo. See the module docstring for the fields."""
    G: np.ndarray
    dt: float
    rf: RFSchedule = None
    readout: tuple = None
    timing: SequenceTiming = None
    encoding: Encoding = None
    crusher: dict = None
    family: str = "waveform"
    notes: str = ""
    build_spec: tuple = None

    def __post_init__(self):
        _set = lambda k, v: object.__setattr__(self, k, v)
        G = np.asarray(self.G, dtype=np.float32)
        if G.ndim == 2:
            G = G[None]
        if G.ndim != 3 or G.shape[2] != 3:
            raise ValueError(f"G must be (n_meas, n_t, 3), got {G.shape}")
        _set("G", G); _set("dt", float(self.dt)); _set("rf", RFSchedule(self.rf))
        n_t = G.shape[1]
        chi, TM, ste, echo_times = self.rf.coherence(n_t, self.dt)
        idx = tuple(int(i) for i in np.clip(np.rint(np.asarray(echo_times) / self.dt).astype(int), 0, n_t - 1)) if echo_times else ()
        if self.readout is None:                       # the grid ends at the readout; a train reads every echo
            ro = idx if len(idx) >= 2 else (n_t - 1,)
        else:
            ro = tuple(int(i) for i in (self.readout if np.ndim(self.readout) else (self.readout,)))
        if not ro or any(i < 0 or i >= n_t for i in ro):
            raise ValueError(f"readout samples {ro} must lie in [0, {n_t})")
        if len(idx) >= 2 and (len(ro) != len(idx) or max(abs(a - b) for a, b in zip(ro, idx)) > ECHO_TOL):
            raise ValueError(f"readout {list(ro)} disagrees with the schedule's echoes {list(idx)}")
        if len(idx) == 1 and abs(ro[-1] - idx[0]) > ECHO_TOL:
            raise ValueError(f"the readout is at sample {ro[-1]} but the RF schedule forms its echo at sample {idx[0]}")
        _set("readout", ro)
        _set("_chi", None if np.all(chi == 1) else chi)
        _set("_TM", TM); _set("_ste", bool(ste)); _set("_echoes", tuple(echo_times))
        if self.encoding is not None and self.encoding.number_of_measurements != G.shape[0]:
            raise ValueError(f"encoding has {self.encoding.number_of_measurements} measurements, G has {G.shape[0]}")

    # ── shape ──────────────────────────────────────────────────────────────────────────────────────
    @property
    def n_meas(self):
        return int(self.G.shape[0])

    @property
    def n_t(self):
        return int(self.G.shape[1])

    @property
    def number_of_measurements(self):
        return self.n_meas

    @property
    def T(self):
        """The span of the grid (s): the echo time of a sequence read at its last sample."""
        return (self.n_t - 1) * self.dt

    @property
    def echo_idx(self):
        """The sample the signal is read at: the last of :attr:`readout`."""
        return int(self.readout[-1])

    @property
    def echoes(self):
        """The echo times the schedule forms (s), in order."""
        return self._echoes

    # ── derived from G and rf ──────────────────────────────────────────────────────────────────────
    @property
    def G_eff(self):
        """The EFFECTIVE gradient the phase integral walks: ``G * rf.sign(t)`` (RPK.md 6.6). Derived, never
        stored. ``(n_meas, n_t, 3)`` float32."""
        s = self.rf.sign(np.arange(self.n_t) * self.dt)
        return self.G * s[None, :, None]

    @property
    def chi_perp(self):
        """The transverse-coherence mask over the grid, or ``None`` when transverse throughout (a spin echo);
        a float profile across any finite pulse (:meth:`RFSchedule.coherence`)."""
        return self._chi

    @property
    def TM(self):
        """The longitudinal-storage (mixing) time of a stimulated echo (s), or ``None``."""
        return self._TM

    @property
    def stimulated_echo(self):
        """Whether the readout is a stimulated echo (the schedule stores and recalls)."""
        return self._ste

    @property
    def refocus_gap(self):
        """The gradient-free span around the first 180, the shortest over measurements (s): what a finite
        refocusing pulse (and its crushers) can occupy without lengthening the echo. ``None`` without a 180;
        0 when the gradient is on at the pulse (Carr-Purcell)."""
        t180 = self.rf.refocus_time
        if t180 is None:
            return None
        k = int(np.clip(round(t180 / self.dt), 0, self.n_t - 1))
        gaps = []
        for m in range(self.n_meas):
            on = np.abs(self.G[m]).sum(1) > 0.0
            if on[k]:
                return 0.0
            lo = k
            while lo > 0 and not on[lo - 1]:
                lo -= 1
            hi = k
            while hi < on.shape[0] - 1 and not on[hi + 1]:
                hi += 1
            gaps.append((hi - lo + 1) * self.dt)
        return float(min(gaps))

    def b(self):
        """The b-value of each measurement (s/m^2), of the effective gradient."""
        from .waveforms import b_from_gradient                  # the integrals live beside the builders
        return b_from_gradient(self.G_eff, self.dt)

    def btensor(self):
        """b-tensor ``B_ij = gamma^2 int q_i q_j dt`` per measurement, ``(n_meas, 3, 3)``, of the effective gradient."""
        from .waveforms import btensor_from_gradient
        return btensor_from_gradient(self.G_eff, self.dt)

    @property
    def refocusing_residual(self):
        """max over measurements of the relative net gradient moment |q(TE)| / max|q| of the effective gradient.
        ``q`` at sample ``i`` is what the walk has accumulated by then: ``G[k]`` acts over ``[k dt, (k + 1) dt)``,
        so the samples before ``i`` count and the one at the readout does not."""
        G_eff = np.asarray(self.G_eff, dtype=np.float64)
        q = np.cumsum(G_eff * self.dt, axis=1)
        qmax = np.max(np.abs(q), axis=(1, 2))
        q_echo = q[:, self.echo_idx - 1, :] if self.echo_idx > 0 else np.zeros_like(q[:, 0, :])
        res = np.where(qmax > 0, np.max(np.abs(q_echo), axis=1) / np.where(qmax > 0, qmax, 1.0), 0.0)
        return float(np.max(res))

    # ── validity ───────────────────────────────────────────────────────────────────────────────────
    def validate(self):
        """What every builder guarantees: the gradient is OFF across every finite pulse (a hard pulse occupies an
        instant and constrains nothing -- a constant gradient through an ideal 180 train is Carr-Purcell), off
        in the lead-in and the readout tail before every readout sample of the budget it was built to, and the
        effective gradient refocuses at the echo. Raises naming the failure; returns ``self``."""
        t = np.arange(self.n_t) * self.dt
        for e in self.rf:
            if e.duration_s > 0.0:
                t0, t1 = e.window
                inside = (t >= t0 - 1e-9 * self.dt) & (t <= t1 + 1e-9 * self.dt)
                if np.any(np.abs(self.G[:, inside, :]) > 0.0):
                    raise ValueError(f"the gradient is on during the {e.flip_deg:g} pulse at {e.t_s*1e3:.3f} ms "
                                     f"(window {t0*1e3:.3f}-{t1*1e3:.3f} ms): a finite pulse needs zero gradient")
        if self.timing is not None:                    # the budget's dead times: the lead-in and the readout tails
            if self.T < self.timing.min_TE() - 1e-9:
                raise ValueError(f"TE = {self.T*1e3:.3f} ms is below min_TE = {self.timing.min_TE()*1e3:.3f} ms of the "
                                 f"timing budget: an encoding window would vanish")
            windows = [(0.0, self.timing.t_lead, "lead-in")]
            windows += [(i * self.dt - self.timing.t_readout_pre_echo, i * self.dt, "readout") for i in self.readout]
            for t0, t1, what in windows:                # a step wholly inside a dead time; a straddling step is rounding
                inside = (t >= t0 - 1e-9 * self.dt) & (t + self.dt <= t1 + 1e-9 * self.dt)
                if np.any(np.abs(self.G[:, inside, :]) > 0.0):
                    raise ValueError(f"the gradient is on in the {what} window {t0*1e3:.3f}-{t1*1e3:.3f} ms of the "
                                     f"timing budget")
        res = self.refocusing_residual
        if res > REFOCUS_ATOL and float(np.abs(self.G).max()) > 0.0:
            raise ValueError(f"the effective gradient is not refocused at the echo (|q(TE)|/max|q| = {res:.2e} > "
                             f"{REFOCUS_ATOL:.0e})")
        return self

    # ── the measurement axis ───────────────────────────────────────────────────────────────────────
    def with_gradient(self, G):
        """The same acquisition with another physical gradient of the same shape (a rescale, a rotation)."""
        G = np.asarray(G, dtype=np.float32)
        if G.shape != self.G.shape:
            raise ValueError(f"G must keep the shape {self.G.shape}, got {G.shape}")
        return replace(self, G=G)

    def __repr__(self):
        enc = f", b={np.array2string(self.encoding.bvalues, precision=3)}" if self.encoding is not None else ""
        return (f"ScannerSequence({self.family!r}, n_meas={self.n_meas}, n_t={self.n_t}, dt={self.dt:g}, "
                f"rf={self.rf!r}, readout={self.readout}{enc})")


class Protocol(tuple):
    """A multi-TE scheme: a tuple of :class:`ScannerSequence`\\ s, one echo time each. It is the shell list an
    analytical layer groups measurements into; a single ``ScannerSequence`` has one schedule and one TE, so a
    scheme with several TEs is several of them."""
    __slots__ = ()

    def __new__(cls, sequences):
        seqs = tuple(sequences)
        for s in seqs:
            if not isinstance(s, ScannerSequence):
                raise TypeError(f"a Protocol holds ScannerSequences, got {type(s).__name__}")
        return super().__new__(cls, seqs)

    @property
    def echo_times(self):
        return tuple(s.T for s in self)

    @property
    def n_meas(self):
        return sum(s.n_meas for s in self)

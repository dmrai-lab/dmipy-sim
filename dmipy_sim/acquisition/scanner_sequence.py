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
* ``prescription`` -- optionally, where in the bore and on what voxels (:class:`~dmipy_sim.acquisition.prescription.Prescription`):
  the acquisition in space, as the rest of the object is the acquisition in time. Nothing is derived from it.
* ``background_gradient`` -- optionally, how much of ``G`` is the MAGNET's rather than the builder's: the
  constant a non-uniform static field contributes at this position (:meth:`ScannerSequence.with_background_gradient`).
  It is already folded into ``G``, so every derived quantity carries it; it is recorded separately because a
  builder's guarantees are about what the builder laid out, which is :attr:`ScannerSequence.designed_gradient`.
* ``concomitant`` -- optionally, the position and static field at which the gradient coils' own Maxwell term
  was evaluated (:meth:`ScannerSequence.with_concomitant`); like the background, already folded into ``G``.
* ``imposed_gradient`` -- what those two transforms added, ``(n_meas, n_t, 3)``: the part of ``G`` no builder
  designed. ``designed_gradient`` is ``G`` minus this.

The scalar engine, the b integrals and the pack's replay read ``G_eff``; the vector-Bloch routes read ``G`` and
apply ``rf`` themselves. A :class:`Protocol` is a tuple of these -- one echo time each -- for a multi-TE scheme.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from .prescription import Prescription
from .rf import RFSchedule
from .timing import SequenceTiming

__all__ = ["Encoding", "ScannerSequence", "Protocol", "Prescription", "ECHO_TOL"]

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
    cpmg_refocus_axis_deg: float = None
    splice_n_echoes: int = None
    splice_TE_echo: float = None
    splice_TE_prep: float = None
    splice_beta_deg: float = None
    splice_refocus_axis_deg: float = None
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
    prescription: Prescription = None
    split_echo: bool = False
    background_gradient: tuple = None
    concomitant: dict = None
    gradient_nonlinearity: tuple = None
    imposed_gradient: np.ndarray = None

    def __post_init__(self):
        _set = lambda k, v: object.__setattr__(self, k, v)
        if self.prescription is not None and not isinstance(self.prescription, Prescription):
            raise TypeError(f"prescription is a Prescription; got {type(self.prescription).__name__}")
        G = np.asarray(self.G, dtype=np.float32)
        if G.ndim == 2:
            G = G[None]
        if G.ndim != 3 or G.shape[2] != 3:
            raise ValueError(f"G must be (n_meas, n_t, 3), got {G.shape}")
        _set("G", G); _set("dt", float(self.dt)); _set("rf", RFSchedule(self.rf))
        if self.imposed_gradient is not None:
            imp = np.asarray(self.imposed_gradient, dtype=np.float32)
            if imp.shape != G.shape:
                raise ValueError(f"imposed_gradient must match G {G.shape}; got {imp.shape}")
            _set("imposed_gradient", imp)
        if self.background_gradient is not None:
            bg = np.asarray(self.background_gradient, dtype=np.float64).reshape(-1, 3)
            if bg.shape[0] not in (1, G.shape[0]):
                raise ValueError(f"background_gradient is one vector or one per measurement "
                                 f"({G.shape[0]}); got {bg.shape}")
            _set("background_gradient", tuple(tuple(float(v) for v in row) for row in bg))
        n_t = G.shape[1]
        chi, TM, ste, echo_times = self.rf.coherence(n_t, self.dt)
        idx = tuple(int(i) for i in np.clip(np.rint(np.asarray(echo_times) / self.dt).astype(int), 0, n_t - 1)) if echo_times else ()
        if self.readout is None:                       # the grid ends at the readout; a train reads every echo
            ro = idx if len(idx) >= 2 else (n_t - 1,)
        else:
            ro = tuple(int(i) for i in (self.readout if np.ndim(self.readout) else (self.readout,)))
        if not ro or any(i < 0 or i >= n_t for i in ro):
            raise ValueError(f"readout samples {ro} must lie in [0, {n_t})")
        if self.split_echo:
            # A split-echo family reads TWO echoes per refocusing interval, a quarter of an interval either
            # side of where the schedule refocuses -- so each pair straddles its own echo.
            if not idx or len(ro) != 2 * (len(idx) - 1):
                raise ValueError(f"a split-echo family reads two echoes per refocusing interval: expected "
                                 f"{2 * (len(idx) - 1)} readout samples, got {len(ro)}")
            mids = [0.5 * (ro[2 * k] + ro[2 * k + 1]) for k in range(len(idx) - 1)]
            if max(abs(m - i) for m, i in zip(mids, idx[1:])) > ECHO_TOL:
                raise ValueError(f"the split pairs are centred at {mids} but the schedule forms its echoes "
                                 f"at {list(idx[1:])}: each pair must straddle its own echo")
        elif len(idx) >= 2 and (len(ro) != len(idx) or max(abs(a - b) for a, b in zip(ro, idx)) > ECHO_TOL):
            raise ValueError(f"readout {list(ro)} disagrees with the schedule's echoes {list(idx)}")
        if not self.split_echo and len(idx) == 1 and abs(ro[-1] - idx[0]) > ECHO_TOL:
            raise ValueError(f"the readout is at sample {ro[-1]} but the RF schedule forms its echo at sample {idx[0]}")
        _set("readout", ro)
        _set("_schedule_echo_idx", idx)
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
                if np.any(np.abs(self.designed_gradient[:, inside, :]) > 0.0):
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
                if np.any(np.abs(self.designed_gradient[:, inside, :]) > 0.0):
                    raise ValueError(f"the gradient is on in the {what} window {t0*1e3:.3f}-{t1*1e3:.3f} ms of the "
                                     f"timing budget")
        res = self.refocusing_residual
        if res > REFOCUS_ATOL and float(np.abs(self.G).max()) > 0.0:
            raise ValueError(f"the effective gradient is not refocused at the echo (|q(TE)|/max|q| = {res:.2e} > "
                             f"{REFOCUS_ATOL:.0e})")
        return self

    # ── the measurement axis ───────────────────────────────────────────────────────────────────────
    def with_prescription(self, prescription):
        """The same acquisition, prescribed in the bore: isocenter, axes, voxel size and matrix
        (:class:`~dmipy_sim.acquisition.prescription.Prescription`). Nothing about the waveform changes."""
        if not isinstance(prescription, Prescription):
            raise TypeError(f"prescription is a Prescription; got {type(prescription).__name__}")
        return replace(self, prescription=prescription)

    @property
    def designed_gradient(self):
        """The gradient the BUILDER laid out, with the magnet's own contribution taken back out: ``G`` itself
        unless :meth:`with_background_gradient` has been applied. This is what the builder's guarantees are
        about -- off through a finite pulse, off in a dead time -- because a background gradient is imposed by
        the magnet and obeys none of them."""
        if self.imposed_gradient is None:
            return self.G
        return self.G - self.imposed_gradient

    def with_background_gradient(self, g):
        """The same acquisition in a magnet whose own field is not uniform: a constant ``g`` (T/m, the field's
        spatial gradient at this position, in the gradient's frame) added to the PHYSICAL gradient over the
        whole grid -- through the pulses and the dead times alike, because a magnet does not switch off.

        One vector, or one per measurement. The effective gradient carries it through the RF sign like
        anything else, so a symmetric spin echo refocuses the background's zeroth MOMENT -- and that is as
        far as the refocusing goes. Its contribution to b does NOT vanish. Stejskal and Tanner say so in one
        line of their 1965 paper: with the pulsed gradient off, "only the term in g0^2 remains", and that
        term is ``gamma^2 g0^2 (2/3) tau^3`` with ``tau = TE/2``. For this magnet at TE = 84 ms it is about
        7 s/mm^2, so even the b = 0 image is diffusion-weighted -- roughly 0.7 % of S0 at a typical brain
        ADC -- and it grows as TE^3.

        TWO TERMS, AND THEY BEHAVE DIFFERENTLY. The CROSS term with the pulsed gradient is linear in the
        background and so flips sign with the diffusion direction; it is the large one, up to 16 % of ADC at
        the edge of this magnet's DSV, and it is what a directional mean hides. The SELF term is quadratic,
        unsigned, present in every measurement including the unweighted one, and much smaller. Reporting
        only the first is the usual simplification and it is not quite true.

        A REFOCUSING TRAIN IS NOT ONE LONG SPIN ECHO. The TE^3 above is a single echo. A train re-refocuses
        the background at every pulse, so its self-term accrues per ECHO -- ``gamma^2 g0^2 esp^3 / 12`` each,
        or ``gamma^2 g0^2 T esp^2 / 12`` over a readout of duration ``T`` -- which is ``(esp/T)^2`` of the
        single-echo value. Treating a seventy-echo train as one spin echo of the same duration overestimates
        it by a factor of about five thousand.

        WHAT THIS DOES NOT MODEL. The background is taken as constant over a voxel, which is the standard
        treatment, and two channels are neglected with it: intravoxel dephasing, which is a signal loss
        rather than a b change, and intravoxel b dispersion, since a voxel's signal is the average of
        ``exp(-b(r) D)`` and not ``exp(-b_mean D)`` -- by Jensen's inequality an ADC fitted from the average
        is biased low. Both are small at 3 mm over this magnet's law and neither is included.

        The cure, worth knowing because it says the effect belongs to the scheme rather than to low field:
        a bipolar sensitising pair nulls the cross term exactly (Neeman 1991). The Swoop's monopolar
        preparation does not.

        ``encoding`` is left alone: it records what was PRESCRIBED at isocentre, and :meth:`b` reports what is
        played here. The difference between them is the effect.
        """
        g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
        if g.shape[0] not in (1, self.n_meas):
            raise ValueError(f"a background gradient is one vector or one per measurement ({self.n_meas}); "
                             f"got {g.shape}")
        if self.background_gradient is not None:
            raise ValueError("this acquisition already carries a background gradient; apply it to the "
                             "acquisition the builder returned, not on top of one that has it")
        add = np.broadcast_to(g.astype(np.float32).reshape(-1, 1, 3), self.G.shape)
        imposed = add if self.imposed_gradient is None else self.imposed_gradient + add
        return replace(self, G=(self.G + add), imposed_gradient=np.ascontiguousarray(imposed),
                       background_gradient=tuple(tuple(float(v) for v in row) for row in g))

    def with_split_readout(self, cycles_per_quarter=16.0):
        """The same refocusing train played as a SPLIT acquisition: two echoes read in every interval rather
        than one, which is what SPLICE does (Schick 1997; Rahbek et al. 2023 for the flip-angle schemes).

        What splits them is an **unbalanced** readout. An ordinary fast spin echo pre-phases by half the area
        its readout then plays, so every pathway returns to coherence order zero at the same instant and
        there is one echo. SPLICE pre-phases by a QUARTER, so the interval winds three orders instead of two
        and two echoes form -- a quarter-interval either side of where the balanced train would put its one.

        The two are not spin echoes against stimulated echoes; each carries both. They are the two
        conjugation parities, and that is the point: a diffusion preparation leaves every spin an arbitrary
        phase, that phase enters the families as ``+phi`` and ``-phi``, constant within each, so each
        family's MAGNITUDE survives it. The two are reconstructed separately and their magnitude images
        summed, which is how this sequence tolerates violating the CPMG condition.

        The winding is voxel-scale -- a micron cell cannot wind an order geometrically -- so it is declared
        on ``crusher`` rather than played into ``G``, one block per quarter-interval running continuously
        from the preparation's echo. ``cycles_per_quarter`` must be a WHOLE number of turns: a fractional
        winding leaves the family that should be empty partly in phase with itself, and the split blurs.

        At ``beta = 180`` the split degenerates, one pathway landing alternately in one family and the other,
        so a real train runs below it.
        """
        C = float(cycles_per_quarter)
        if abs(C - round(C)) > 1e-9 or C < 1:
            raise ValueError(f"cycles_per_quarter is a whole number of turns (a fraction leaves the empty "
                             f"family partly in phase with itself); got {cycles_per_quarter}")
        idx = self._schedule_echo_idx
        if len(idx) < 2 or self.split_echo:
            raise ValueError("a split readout needs a refocusing train that is not already split")
        esp = int(round(np.mean(np.diff(idx))))
        q = esp // 4
        if q < 1:
            raise ValueError(f"{esp} samples an interval is too few to place a split readout")
        win, cyc, k = [], [], idx[0] + q
        while k + q <= idx[-1] + q:
            win.append((k * self.dt, (k + q) * self.dt)); cyc.append(C); k += q
        ro = tuple(int(v) for e in idx[1:] for v in (e - q, e + q))
        n_t = max(self.n_t, ro[-1] + 1)          # the last family is read past the builder's last echo
        G = np.zeros((self.n_meas, n_t, 3), np.float32)
        G[:, :self.n_t, :] = np.asarray(self.G, np.float32)
        return replace(self, G=G, readout=ro, split_echo=True,
                       crusher={"windows_s": win, "n_cycles": cyc},
                       family=(self.family if self.family.endswith("split") else self.family + "-split"))

    def with_concomitant(self, position_m, B0_T):
        """The same acquisition as it is actually played at ``position_m``, with the gradient coils' own
        concomitant (Maxwell) field included -- the term that makes a gradient system produce a field whose
        magnitude, not just whose z component, varies.

        For ``B0`` along the bore's z the field is
        ``B_c = (Gx^2 + Gy^2) z^2 / 2B0 + Gz^2 (x^2 + y^2) / 8B0 - (Gx Gz x z + Gy Gz y z) / 2B0``
        and what a spin at ``position_m`` sees as an extra encoding gradient is its spatial derivative there.
        That derivative is **quadratic in G(t)**, so unlike a background gradient it varies through the
        sequence, and unlike the pulsed gradient it does not change sign when the coils reverse: a symmetric
        pair therefore leaves it almost intact where the pulsed gradient refocuses, and an unbalanced train
        does not refocus it at all (dmipy-sim#285).

        It scales as ``1 / B0``, which is why it is a low-field problem: at 64 mT it is some 47 times what the
        same gradient produces at 3 T. Zero at isocentre, by construction.

        ``position_m`` is ``(3,)`` or one per measurement, in the gradient's frame; ``B0_T`` is the static
        field in tesla.
        """
        r = np.asarray(position_m, dtype=np.float64).reshape(-1, 3)
        if r.shape[0] not in (1, self.n_meas):
            raise ValueError(f"position_m is one point or one per measurement ({self.n_meas}); got {r.shape}")
        B0 = float(B0_T)
        if B0 <= 0.0:
            raise ValueError(f"B0_T must be positive; got {B0}")
        if self.concomitant is not None:
            raise ValueError("this acquisition already carries a concomitant term; apply it once, at the "
                             "position the measurement is made")
        G = np.asarray(self.designed_gradient, dtype=np.float64)          # the coils' own, not the magnet's
        Gx, Gy, Gz = G[..., 0], G[..., 1], G[..., 2]
        x, y, z = (r[:, i][:, None] for i in range(3))
        gc = np.stack([Gz * (Gz * x - 2.0 * Gx * z) / (4.0 * B0),
                       Gz * (Gz * y - 2.0 * Gy * z) / (4.0 * B0),
                       (2.0 * z * (Gx ** 2 + Gy ** 2) - Gz * (Gx * x + Gy * y)) / (2.0 * B0)], axis=-1)
        gc = np.broadcast_to(gc.astype(np.float32), self.G.shape)
        imposed = gc if self.imposed_gradient is None else self.imposed_gradient + gc
        return replace(self, G=(self.G + gc), imposed_gradient=np.ascontiguousarray(imposed),
                       concomitant={"position_m": tuple(tuple(float(v) for v in p) for p in r), "B0_T": B0})

    def with_gradient_nonlinearity(self, L):
        """The same acquisition as the COILS actually deliver it at one position: every commanded gradient
        vector replaced by ``L @ g``, with ``L`` the gradient-nonlinearity tensor there
        (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.gradient_tensor`).

        Unlike a background gradient this ADDS nothing -- it rescales and tilts what was already asked for,
        so it vanishes wherever the commanded gradient does and cannot encode during a dead time. That is the
        signature separating the two: a magnet's own gradient is on when nothing is played, a coil's
        nonlinearity is not.

        The consequence is an encoding error rather than a shading. The delivered b along a commanded unit
        direction ``u`` is ``|L u|^2`` times the one asked for, along the rotated direction ``L u / |L u|``,
        so a diffusivity fitted against the nominal b is wrong by that factor -- doubled, because b enters
        the exponent through the square of the gradient.

        ``L`` is one ``(3, 3)`` tensor or one per measurement.
        """
        L = np.asarray(L, dtype=np.float64)
        if L.shape == (3, 3):
            L = L[None]
        if L.shape[1:] != (3, 3) or L.shape[0] not in (1, self.n_meas):
            raise ValueError(f"L is one 3x3 tensor or one per measurement ({self.n_meas}); got {L.shape}")
        if self.gradient_nonlinearity is not None:
            raise ValueError("this acquisition already carries a gradient-nonlinearity tensor; apply it "
                             "once, at the position the measurement is made")
        G = np.asarray(self.G, dtype=np.float64)
        new = np.einsum("mij,mtj->mti", np.broadcast_to(L, (self.n_meas, 3, 3)), G)
        delta = (new - G).astype(np.float32)
        imposed = delta if self.imposed_gradient is None else self.imposed_gradient + delta
        return replace(self, G=new.astype(np.float32), imposed_gradient=np.ascontiguousarray(imposed),
                       gradient_nonlinearity=tuple(tuple(float(v) for v in row.ravel()) for row in L))

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
    """A multi-TE scheme: a tuple of :class:`ScannerSequence`\\ s, one echo time each, and how their measurements
    interleave in the acquisition. A single ``ScannerSequence`` has one schedule and one TE, so a scheme with
    several TEs is several of them; an analytical layer groups its measurements into these.

    ``rows`` says which measurement of the whole acquisition each sequence's rows are (``rows[i]`` the indices
    of sequence ``i``'s rows in acquisition order); by default the sequences follow one another. A consumer that
    returns one value per measurement (``simulate``, a pack's ``replay``) places each sequence's results at its
    rows, so the output is in acquisition order whatever the grouping.
    """

    def __new__(cls, sequences, rows=None):
        seqs = tuple(sequences)
        for s in seqs:
            if not isinstance(s, ScannerSequence):
                raise TypeError(f"a Protocol holds ScannerSequences, got {type(s).__name__}")
        self = super().__new__(cls, seqs)
        n = sum(s.n_meas for s in seqs)
        if rows is None:
            offs = np.cumsum([0] + [s.n_meas for s in seqs])
            rows = tuple(np.arange(offs[i], offs[i + 1]) for i in range(len(seqs)))
        else:
            rows = tuple(np.asarray(r, dtype=int).reshape(-1) for r in rows)
            if len(rows) != len(seqs) or any(len(r) != s.n_meas for r, s in zip(rows, seqs)):
                raise ValueError("rows must give one index array per sequence, of that sequence's n_meas")
            if sorted(np.concatenate(rows).tolist()) != list(range(n)):
                raise ValueError(f"rows must be a partition of range({n})")
        self._rows = rows
        return self

    @property
    def rows(self):
        """One index array per sequence: its rows' positions in the acquisition."""
        return self._rows

    @property
    def echo_times(self):
        return tuple(s.T for s in self)

    @property
    def n_meas(self):
        return sum(s.n_meas for s in self)

    def scatter(self, parts, axis=-1):
        """Place per-sequence results (each with its sequence's ``n_meas`` along ``axis``) at their rows: the
        acquisition-ordered result."""
        parts = [np.asarray(p) for p in parts]
        out = np.empty(parts[0].shape[:axis % parts[0].ndim] + (self.n_meas,) + parts[0].shape[axis % parts[0].ndim + 1:],
                       dtype=np.result_type(*parts))
        for p, r in zip(parts, self.rows):
            idx = [slice(None)] * out.ndim
            idx[axis % out.ndim] = r
            out[tuple(idx)] = p
        return out

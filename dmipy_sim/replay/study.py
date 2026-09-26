"""A study: a protocol replayed on some tissues on some scanners, the rows read once and the bands contracted once per
acquisition, every (tissue, scanner) pair elementwise (dmipy-sim#297).

The knobs of a replay touch the contraction at different depths. For a walker with bridge coefficients ``C`` an
acquisition contributes a gradient phase ``phi = C . W(sequence, orientation)``, the matmul over the bands, a
function of the sequence and the pose only. The field tier adds ``B0 (chi_iso A + chi_aniso B)``, where ``A`` and
``B`` are the walker's gated path integrals contracted with the field direction, then linear in the scanner's
field and the tissue's susceptibilities. Relaxation is a weight from the walker's transverse and longitudinal
exposure per pool, contact a weight from its boundary local time under the gate. So per acquisition there is one
set of walker primitives, :class:`Primitives`, and every tissue and scanner is arithmetic on them.

    acq = Acquisition(sequences.pgse(...))                        # a sequence in a pose (orientation=None: the pack's frame)
    protocol = Protocol([acq, ...])                               # ordered; its measurements side by side
    study = Study(protocol, tissues=[wm], scanners=[3.0, 7.0])     # the cross product by default, pairs= to pick
    S = pack.study(study)                                          # (pairs, measurements) for one pack
    S, floor, plan = ReplayPack.open(uri).image(study)             # (pairs, *grid, measurements): one pass over the rows

The tissue is the sample and the scanner the machine; they are not merged. A tissue may be given as a callable
``scanner -> Tissue`` (the catalogue's values at the scanner's field), resolved once per pair and recorded.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Acquisition:
    """A sequence in a pose: ``sequence`` a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` (or an
    object carrying one as ``.waveform``), ``orientation`` the substrate's pose as :meth:`ReplayPack.replay` takes
    it (``None``: the pack's own frame), ``name`` for the record."""
    sequence: object
    orientation: object = None
    name: Optional[str] = None

    @property
    def waveform(self):
        s = self.sequence
        return s.waveform if hasattr(s, "waveform") else s

    @property
    def n_meas(self):
        return int(np.asarray(self.waveform.G_eff).shape[0])


class Protocol:
    """An ordered list of acquisitions; ``n_meas`` their measurements side by side, ``slices`` where each sits."""

    def __init__(self, acquisitions, name=None):
        acqs = []
        for a in acquisitions:
            acqs.append(a if isinstance(a, Acquisition) else Acquisition(a))
        self.acquisitions = tuple(acqs); self.name = name
        counts = [a.n_meas for a in self.acquisitions]; ends = np.cumsum([0] + counts)
        self.slices = [slice(int(ends[i]), int(ends[i + 1])) for i in range(len(counts))]
        self.n_meas = int(ends[-1])

    def __iter__(self):
        return iter(self.acquisitions)

    def __len__(self):
        return len(self.acquisitions)

    def to_meta(self):
        return dict(name=self.name, acquisitions=[dict(name=a.name, n_meas=a.n_meas, oriented=a.orientation is not None) for a in self.acquisitions])


class Study:
    """``protocol`` on ``tissues`` on ``scanners``: every pair of the two lists unless ``pairs`` (a list of
    ``(i_tissue, i_scanner)``) picks. A tissue is a :class:`~dmipy_sim.spec.tissue.Tissue`, ``None`` (bare diffusion)
    or a callable ``scanner -> Tissue``; ``resolved(k)`` is the ``k``-th pair as ``(tissue, scanner)`` with the
    callable applied. ``to_meta`` is the record of what was replayed."""

    def __init__(self, protocol, tissues=(None,), scanners=(None,), pairs=None, name=None):
        self.protocol = protocol if isinstance(protocol, Protocol) else Protocol(protocol)
        self.tissues = list(tissues); self.scanners = list(scanners); self.name = name
        self.pairs = list(pairs) if pairs is not None else [(i, j) for i in range(len(self.tissues)) for j in range(len(self.scanners))]
        for i, j in self.pairs:
            if not (0 <= i < len(self.tissues) and 0 <= j < len(self.scanners)):
                raise IndexError(f"pair ({i}, {j}) names a tissue or scanner the study does not have")

    def __len__(self):
        return len(self.pairs)

    def resolved(self, k):
        i, j = self.pairs[k]; t, s = self.tissues[i], self.scanners[j]
        return (t(s) if callable(t) else t), s

    @property
    def needs_field(self):
        from .replay import scanner_field
        return any(scanner_field(s).B0 not in (None, 0.0) and t is not None and t.chi_iso is not None for t, s in (self.resolved(k) for k in range(len(self))))

    @property
    def needs_contact(self):
        return any(t is not None and t.rho is not None and float(t.rho) != 0.0 for t, _ in (self.resolved(k) for k in range(len(self))))

    @property
    def needs_relaxation(self):
        return any(t is not None and t.relaxes for t, _ in (self.resolved(k) for k in range(len(self))))

    def to_meta(self):
        from .replay import scanner_field
        out = []
        for k in range(len(self)):
            t, s = self.resolved(k)
            f = scanner_field(s)
            out.append(dict(tissue=(None if t is None else t.to_meta()), scanner=(None if s is None else dict(field_T=f.B0, name=f.name))))
        return dict(name=self.name, protocol=self.protocol.to_meta(), pairs=out)


@dataclass
class Primitives:
    """What one acquisition leaves of every walker before any tissue or scanner is applied: ``w`` the weights,
    ``phi`` the gradient phase ``(n, m)``, ``field_iso`` / ``field_aniso`` the gated path integrals under the field
    direction ``(n,)`` (None without the path channel), ``exposure_t2`` / ``exposure_t1`` the transverse and
    longitudinal time per pool ``(n, n_pools)`` (None without the compartment channel), ``contact`` the gated
    boundary local time ``(n,)`` (None without the contact channel), ``D_walk`` the walk's diffusivity, and
    ``pathway`` the amplitude of the coherence pathway the acquisition's readout is
    (:func:`~dmipy_sim.acquisition.epg.pathway_weight`; 1 for a refocused echo, a stimulated echo's
    ``0.5 sin a1 sin a2 sin a3`` for a store-and-recall schedule). A tissue and a scanner turn these into the
    per-walker weights and phases of :meth:`signals`."""
    w: np.ndarray
    phi: np.ndarray
    field_iso: Optional[np.ndarray]
    field_aniso: Optional[np.ndarray]
    exposure_t2: Optional[np.ndarray]
    exposure_t1: Optional[np.ndarray]
    contact: Optional[np.ndarray]
    D_walk: Optional[float]
    pathway: float = 1.0
    by_pool: object = field(repr=False, default=None)          # the pack's resolver of a per-pool value

    @property
    def n_pools(self):
        return None if self.exposure_t2 is None else int(self.exposure_t2.shape[1])

    def rates(self, tissue):
        """``(invT2, invT1, rho_over_D)`` of a tissue on this pack's pools: per-pool rates (0 where the tissue
        declares no time, or ``inf``) and the contact rate (0 without a relaxivity). A per-pool value is judged
        complete on the pack's spec, not on the pools this chunk of walkers labels."""
        t = tissue
        n = self.n_pools or 1
        invT2 = np.zeros(n); invT1 = np.zeros(n); rho_D = 0.0
        if t is not None and (t.T2 is not None or t.T1 is not None):
            if self.exposure_t2 is None:
                raise ValueError("T2 / T1 were given but the pack carries no compartment channel (C1)")
            for name, vals in (("T2", self.by_pool(t.T2, "T2")), ("T1", self.by_pool(t.T1, "T1"))):
                if vals is None:
                    continue
                if len(vals) < n:
                    raise ValueError(f"the compartment channel addresses pool id {n - 1} and the embedded spec "
                                     f"declares {len(vals)} pools: the pack is inconsistent")
                rate = 1.0 / np.asarray(vals, float)[:n]           # the spec may name more pools than the channel labels (a dry sheath)
                if name == "T2":
                    invT2 = rate
                else:
                    invT1 = rate
        if t is not None and t.D is not None and self.D_walk is not None and float(t.D) != float(self.D_walk):
            raise ValueError(
                f"the primitives were contracted on the walked grid, at D = {float(self.D_walk):.3g} m^2/s, and a "
                f"tissue at D = {float(t.D):.3g} reads the pack on another grid (dmipy-sim#289): every term "
                f"changes, not the surface divisor alone. Contract the view instead -- "
                f"pack.at_diffusivity(D).study(...) or .walker_primitives(...)")
        if t is not None and t.rho is not None and float(t.rho) != 0.0:
            if self.contact is None:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            D = self.D_walk if t.D is None else t.D
            if D is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            rho_D = float(t.rho) / float(D)
        return invT2, invT1, rho_D

    def field_scalars(self, tissue, scanner):
        """``(B0 chi_iso, B0 chi_aniso)`` for a pair, ``(0, 0)`` without a field."""
        from .replay import scanner_field
        B0 = scanner_field(scanner).B0
        if B0 is None or float(B0) == 0.0:
            return 0.0, 0.0
        if tissue is None or tissue.chi_iso is None:
            raise ValueError("a scanner field was given without a chi_iso in the tissue; give a tissue with chi_iso (and chi_aniso)")
        if self.field_iso is None:
            raise ValueError("a field was asked for but this pack carries no path channel (C3)")
        return float(B0) * float(tissue.chi_iso), float(B0) * float(tissue.chi_aniso or 0.0)

    def signals(self, tissue=None, scanner=None):
        """``(w, ew, E)`` as :meth:`ReplayPack.walker_signals` gives them for this acquisition under the pair."""
        invT2, invT1, rho_D = self.rates(tissue); a_i, a_a = self.field_scalars(tissue, scanner)
        logw = np.zeros(len(self.w))
        if self.exposure_t2 is not None:
            logw = logw - self.exposure_t2 @ invT2 - self.exposure_t1 @ invT1
        if rho_D:
            logw = logw + rho_D * self.contact
        phi = self.phi
        if a_i or a_a:
            phi = phi + (a_i * self.field_iso + a_a * self.field_aniso)[:, None]
        return self.w, self.pathway * self.w * np.exp(logw), np.exp(1j * phi)


def walker_primitives(pack, acquisition):
    """The :class:`Primitives` of ``pack`` under ``acquisition``: the bands contracted once, the path channel
    contracted once under the pose's field direction, the exposures and the contact read once -- each a sum over
    the windows the walk is stored in (RPK.md 4.3)."""
    from .compression import read_position_coeffs, relaxation_logweight_runs
    from .replay import _compile_effective, surface_logweight
    from ._replay_kernel import effective_gradient
    acq = acquisition if isinstance(acquisition, Acquisition) else Acquisition(acquisition)
    P = pack._prepare(acq.waveform, tissue=None, scanner=None, orientation=acq.orientation, compartment=None)
    n_w, dt, ch = P["n_w"], P["dt"], P["ch"]
    col = n_ids = None
    if pack.has_relaxation:
        from .compression import is_current_c1, decode_occupancy
        if not is_current_c1(ch["compartment"]):
            decode_occupancy(pack.arrays, ch["compartment"])
        col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
        n_ids = pack._n_pool_ids(col)
        # exposure per pool: a unit rate in one pool and none (an infinite time) in the others
        unit = np.where(np.eye(n_ids) > 0, 1.0, np.inf); zero = [np.inf] * n_ids
    from .compression import has_c2
    pm = ch.get("susceptibility_path")
    phi = exposure_t2 = exposure_t1 = contact = Psi = None
    for (seg, t0, n_s), (chi_s, act_s) in zip(P["windows"], P["window_gates"]):
        Geff_s = effective_gradient(P["G_eff_wf"], P["dt_wf"], n_s, dt, t0=t0)
        C = read_position_coeffs(seg.arrays, dtype=np.float64)
        phi_s = C.reshape(n_w, seg.n_coeffs * 3) @ _compile_effective(Geff_s, dt, seg.K, n_s)
        phi = phi_s if phi is None else phi + phi_s
        if col is not None:
            e2 = np.stack([-relaxation_logweight_runs(seg.arrays, col, unit[p].tolist(), zero, dt, chi_s, act_s) for p in range(n_ids)], axis=1)
            e1 = np.stack([-relaxation_logweight_runs(seg.arrays, col, zero, unit[p].tolist(), dt, chi_s, act_s) for p in range(n_ids)], axis=1)
            exposure_t2 = e2 if exposure_t2 is None else exposure_t2 + e2
            exposure_t1 = e1 if exposure_t1 is None else exposure_t1 + e1
        if has_c2(seg.arrays):
            c_s = np.asarray(surface_logweight(seg.arrays, 1.0, ch.get("boundary_local_time"), chi_s), np.float64)
            contact = c_s if contact is None else contact + c_s
        if pm is not None:
            from .bank import path_field_integral
            Psi_s, names = path_field_integral(seg.arrays, pm, acq.waveform, n_s, dt, t0=t0, n_w=n_w)
            Psi = Psi_s if Psi is None else Psi + Psi_s
    field_iso = field_aniso = None
    if pm is not None:
        from ..fields.hollow_cylinder import field_terms
        field_iso, aniso = field_terms(Psi, P["b0_dir"])
        gm = ch.get("susceptibility_grid") or {}
        field_aniso = aniso if (gm.get("has_aniso") and aniso is not None) else np.zeros(n_w)
    from ..acquisition.epg import pathway_weight
    return Primitives(w=P["w"], phi=phi, field_iso=field_iso, field_aniso=field_aniso, exposure_t2=exposure_t2, exposure_t1=exposure_t1,
                      contact=contact, D_walk=pack.diffusivity, pathway=pathway_weight(acq.waveform),
                      by_pool=pack._by_pool)


def study_signals(pack, study):
    """``(pairs, n_meas)``: the ensemble signal of ``pack`` for every pair of ``study``, the primitives of each
    acquisition formed once."""
    study = study if isinstance(study, Study) else Study(study)
    S = np.zeros((len(study), study.protocol.n_meas))
    for a, sl in zip(study.protocol, study.protocol.slices):
        prim = walker_primitives(pack, a)
        for k in range(len(study)):
            t, s = study.resolved(k)
            w, ew, E = prim.signals(t, s)
            S[k, sl] = np.abs((ew[:, None] * E).sum(0)) / w.sum()
    return S

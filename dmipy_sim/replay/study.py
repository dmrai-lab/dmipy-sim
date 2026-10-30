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

from dmipy_replay.replay.planning import PLAN_TOL, planned
from dmipy_replay.replay.primitives import Primitives


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
        from dmipy_replay.replay.replay import scanner_field
        def magnetised(t):
            return t is not None and any(v is not None and float(v) != 0.0 for v in (t.chi_iso, t.chi_aniso))
        return any(scanner_field(s).B0 not in (None, 0.0) and magnetised(t) for t, s in (self.resolved(k) for k in range(len(self))))

    @property
    def needs_contact(self):
        def wants(t):
            return t is not None and ((t.rho2 is not None and float(t.rho2) != 0.0)
                                       or (t.rho1 is not None and float(t.rho1) != 0.0))
        return any(wants(t) for t, _ in (self.resolved(k) for k in range(len(self))))

    @property
    def needs_relaxation(self):
        return any(t is not None and t.relaxes for t, _ in (self.resolved(k) for k in range(len(self))))

    def to_meta(self):
        from dmipy_replay.replay.replay import scanner_field
        out = []
        for k in range(len(self)):
            t, s = self.resolved(k)
            f = scanner_field(s)
            out.append(dict(tissue=(None if t is None else t.to_meta()), scanner=(None if s is None else dict(field_T=f.B0, name=f.name))))
        return dict(name=self.name, protocol=self.protocol.to_meta(), pairs=out)




def walker_primitives(pack, acquisition, *, tol=PLAN_TOL, K=None, modes=None):
    """The :class:`Primitives` of ``pack`` under ``acquisition``: the bands contracted once, the path channel
    contracted once under the pose's field direction, the exposures and the contact read once -- each a sum over
    the windows the walk is stored in (RPK.md 4.3). The bands and modes read are the plan's (``tol=``, ``K=``, ``modes=`` as in
    :meth:`~dmipy_sim.replay.replay.ReplayPack.replay`); the tissue and the scanner are not known here, and the field modes are planned at the pack's certified
    field scale (``field_envelope``: the largest ``B0`` and ``chi``) for every replay depth, so they are the modes
    :meth:`~dmipy_sim.replay.replay.ReplayPack.replay` reads; every mode when the pack states no envelope."""
    from dmipy_replay.replay.compression import relaxation_logweight_runs
    from dmipy_replay.replay.replay import _band_phase, _path_field_channels, _gated_contact, _has_aniso
    acq = acquisition if isinstance(acquisition, Acquisition) else Acquisition(acquisition)
    P = pack._prepare(acq.waveform, tissue=None, scanner=None, orientation=acq.orientation, compartment=None,
                      read=dict(tol=tol, K=K, modes=modes), any_field=True)
    n_w, ch = P["n_w"], P["ch"]
    col = n_ids = None
    if pack.has_relaxation:
        from dmipy_replay.replay.compression import is_current_c1, decode_occupancy
        if not is_current_c1(ch["compartment"]):
            decode_occupancy(pack.arrays, ch["compartment"])
        col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
        n_ids = pack._n_pool_ids(col)
        # exposure per pool: a unit rate in one pool and none (an infinite time) in the others
        unit = np.where(np.eye(n_ids) > 0, 1.0, np.inf); zero = [np.inf] * n_ids
    from dmipy_replay.replay.compression import has_c2
    pm = ch.get("susc_field_basis")
    phi = _band_phase(P)                                        # the one band contraction, windows summed
    exposure_t2 = exposure_t1 = contact = contact_t1 = None
    chi_c, act_c = P["gate_cum"]
    for j, (seg, t0, T_s) in enumerate(P["windows"]):
        t1 = t0 + T_s
        if col is not None:
            e2 = np.stack([-relaxation_logweight_runs(seg.arrays, col, unit[p].tolist(), zero, chi_c, act_c, t0=t0, t1=t1) for p in range(n_ids)], axis=1)
            e1 = np.stack([-relaxation_logweight_runs(seg.arrays, col, zero, unit[p].tolist(), chi_c, act_c, t0=t0, t1=t1) for p in range(n_ids)], axis=1)
            exposure_t2 = e2 if exposure_t2 is None else exposure_t2 + e2
            exposure_t1 = e1 if exposure_t1 is None else exposure_t1 + e1
        if has_c2(seg.arrays):
            c_s = np.asarray(_gated_contact(P["windows"], P["gate_wf"], j, 1.0, ch.get("boundary_local_time")),
                             np.float64)
            contact = c_s if contact is None else contact + c_s
            # the SAME channel, gated by the complement (on while stored along B0): the longitudinal contact
            # rho1 reads at reduction time, built once here exactly as `contact` is (dmipy-sim#574)
            c1_s = np.asarray(_gated_contact(P["windows"], P["gate_wf"], j, 1.0, ch.get("boundary_local_time"),
                                             parallel=True), np.float64)
            contact_t1 = c1_s if contact_t1 is None else contact_t1 + c1_s
    field_iso = field_aniso = None
    if pm is not None and P["M_use"] == 0:                      # a plan that reads no mode: the field scalars exist and are zero
        field_iso, field_aniso = np.zeros(n_w), np.zeros(n_w)
    elif pm is not None:
        from dmipy_replay.replay.field import field_terms
        Psi, names = _path_field_channels(P, acq.waveform)       # the one read of C3, windows summed
        field_iso, aniso = field_terms(Psi, P["b0_dir"])
        field_aniso = aniso if _has_aniso(ch.get("susceptibility_grid"), names) else np.zeros(n_w)
    from dmipy_replay.acquisition.epg import pathway_weight
    return Primitives(w=P["w"], phi=phi, field_iso=field_iso, field_aniso=field_aniso, exposure_t2=exposure_t2, exposure_t1=exposure_t1,
                      contact=contact, contact_t1=contact_t1, D_walk=pack.diffusivity, pathway=pathway_weight(acq.waveform),
                      rho2_over_D_max=pack.rho2_over_D_max, field_envelope=pack.field_envelope, voxel=P["voxel"], by_pool=pack._by_pool,
                      susceptibility_field=pack.susceptibility_field, plan=P["plan"])


def study_signals(pack, study):
    """``(pairs, n_meas)``: the ensemble signal of ``pack`` for every pair of ``study``, the primitives of each
    acquisition formed once."""
    study = study if isinstance(study, Study) else Study(study)
    S = np.zeros((len(study), study.protocol.n_meas))
    for a, sl in zip(study.protocol, study.protocol.slices):
        prim = walker_primitives(pack, a)
        for k in range(len(study)):
            S[k, sl] = prim.signal(*study.resolved(k))
    return S

"""The plan of a read: how many position bands and field modes a replay of one acquisition needs, and what the rest costs.

A pack stores ``K`` position bands and ``K_f`` field modes per walker; an acquisition's gradient projects on the bands
(:func:`~dmipy_sim.replay.compression.band_projection`) and its gate on the modes
(:func:`~dmipy_sim.replay.compression.cosine_projection`), and a band or mode the projection leaves almost empty
contributes nothing. The rule, stated once here and read by every layout (:meth:`~dmipy_sim.replay.replay.ReplayPack.plan`
on a pack, :meth:`~dmipy_sim.replay.remote.RemotePack.plan` on a pack on the Hub,
:meth:`~dmipy_sim.replay.columnar.ColumnarPack.plan` on the columnar layout):

* the bands beyond ``K'`` carry the phase variance ``gamma^2 sum_{k>K'} sum_d var_{k,d} W_{m,k,d}^2`` under the
  waveform's projection ``W``; a dropped independent phase under-attenuates the signal by half of it, and the plan's
  band error is the largest of that half over the measurements;
* the field modes beyond ``M`` carry ``(gamma B0 (|chi_iso| + |chi_aniso|))^2 sum_{k>M} gate_k^2 sum_ch var_{ch,k}``,
  halved, the channel sum being the worst case over field directions;
* the plan is the fewest bands and the fewest modes whose error is at most ``tol`` times the pack's Monte-Carlo
  floor.

The variances come from a stated source, named in the plan. The bands': the header's measured per-band record where the
pack carries one (a view of the columnar layout does), else the free bridge's ``2 D T^2 / (pi k)^2`` at the pack's
recorded diffusivity (:func:`~dmipy_sim.replay.compression.band_variances`), the walk-free closed form the pack's
certificate truncates with (``fidelity.terms.truncation_from``) and a restricted walk's variance never exceeds. The modes':
the header's record, else measured on a pilot of the pack's walkers (:func:`pilot_size`). A pack with neither is refused by
name.
"""
from dataclasses import dataclass, asdict

import numpy as np

from ..constants import GAMMA

#: The plan's tolerance: the dropped bands and modes may cost this fraction of the pack's Monte-Carlo floor.
PLAN_TOL = 0.25

#: Walkers a measured variance reads: the first this many of a pack whose walkers are declared shuffled
#: (``compression.precision_tiers.walkers_shuffled``, so a prefix is an unbiased sample), else all of them.
PILOT_WALKERS = 4096


def floor_of(meta, n_walkers):
    """The Monte-Carlo floor a plan is measured against: the certificate's ``fidelity.floor_max``; a pack whose certificate
    states none (walkers that never moved) is held to the largest floor its walkers could have, ``1 / sqrt(n_walkers)``."""
    return float((meta.get("fidelity") or {}).get("floor_max") or 0.0) or 1.0 / np.sqrt(float(n_walkers))


def pilot_size(meta, n_walkers):
    """``(n, why)``: the walkers a measured variance reads and the sentence the plan states for it."""
    shuffled = bool(((meta.get("compression") or {}).get("precision_tiers") or {}).get("walkers_shuffled", False))
    if shuffled and n_walkers > PILOT_WALKERS:
        return PILOT_WALKERS, f"measured on the first {PILOT_WALKERS} of {n_walkers} walkers (declared shuffled)"
    return int(n_walkers), f"measured on all {n_walkers} walkers"


def stored_band_variance(meta, K):
    """``((K, 3) variance, source)`` from the header's per-band record (the columnar layout's ``columnar.band_variance_max``,
    the worst populated voxel's, else ``band_variance``), or ``None`` when the header carries none."""
    rec = meta.get("columnar") or {}
    for key, what in (("band_variance_max", "worst populated voxel"), ("band_variance", "walker mean")):
        if rec.get(key):
            return np.asarray(rec[key], np.float64).T[:K], f"stored: header columnar.{key} ({what})"
    return None


def stored_mode_variance(meta):
    """``((K_f,) variance summed over channels, source)`` from the header's ``columnar.path_mode_variance``, or ``None``."""
    rec = (meta.get("columnar") or {}).get("path_mode_variance")
    if not rec:
        return None
    return np.asarray(rec, np.float64).sum(axis=0), "stored: header columnar.path_mode_variance (worst case over channels)"


def band_variance(meta, K, T, D):
    """``((K, 3) variance, source)`` of the position bands a plan weighs: the header's record, else the free bridge at the
    pack's diffusivity ``D`` on the window ``T``; refused by name when the pack has neither."""
    stored = stored_band_variance(meta, K)
    if stored is not None:
        return stored
    if D is None:
        raise ValueError("this pack stores no per-band variance and records no diffusivity, so the bands an acquisition "
                         "needs cannot be planned: pass K= to read a stated number of bands")
    from .compression import band_variances
    var = band_variances(K, T, D)
    return np.repeat(var[:, None], 3, axis=1), f"free bridge 2 D T^2 / (pi k)^2 at D = {float(D):.4g} m^2/s, T = {float(T):.4g} s"


def band_tail(G_eff, dt_wf, windows, var):
    """``(n_meas, K + 1)``: column ``j`` the band error of keeping the lowest ``j`` bands -- half the phase variance of
    bands ``j .. K - 1`` under ``G_eff`` ``(n_meas, n_wf, n_c)`` (on its grid ``dt_wf``), summed over the walk's windows
    ``[(T, t0), ...]`` (the windows' phases are independent). ``var`` is ``(K, n_c)`` and column ``K`` is 0."""
    from .compression import band_projection
    G = np.asarray(G_eff, np.float64)
    K, n_c = var.shape[0], G.shape[2]
    out = np.zeros((G.shape[0], K + 1))
    for T, t0 in windows:
        W = band_projection(G, float(dt_wf), float(T), K, t0=t0)[:, 2:, :]
        per_band = GAMMA ** 2 * (W ** 2 * var[None, :, :n_c]).sum(axis=2)
        out[:, :K] += 0.5 * np.cumsum(per_band[:, ::-1], axis=1)[:, ::-1]
    return out


def mode_tail(gate, dt_wf, windows, K_f, var, B0, chi):
    """``(K_f + 1,)``: entry ``j`` the field error of keeping the lowest ``j`` modes -- half the phase variance of modes
    ``j .. K_f - 1`` for the gate ``gate`` (on its grid ``dt_wf``) at the field ``B0`` and the susceptibility scale
    ``chi = |chi_iso| + |chi_aniso|``. ``var`` is the ``(K_f,)`` variance summed over the channels; windows as in
    :func:`band_tail`."""
    from .compression import cosine_projection
    g = np.asarray(gate, np.float64).reshape(1, -1, 1)
    out = np.zeros(K_f + 1)
    for T, t0 in windows:
        proj = cosine_projection(g, float(dt_wf), float(T), K_f, t0=t0)[0, :, 0]
        per_mode = (GAMMA * float(B0) * float(chi)) ** 2 * proj ** 2 * var[:K_f]
        out[:K_f] += 0.5 * np.cumsum(per_mode[::-1])[::-1]
    return out


def fewest(err, candidates, limit):
    """``(n, err[n])`` for the first of ``candidates`` (ascending counts) whose error is at most ``limit``; the last
    candidate, the whole stored set, when none is."""
    for n in candidates:
        if err[n] <= limit:
            return int(n), float(err[n])
    n = candidates[-1]
    return int(n), float(err[n])


@dataclass(frozen=True)
class ReadPlan:
    """What a replay of one acquisition reads of a pack, stated before it reads: ``K`` of the ``K_stored`` position bands
    and ``modes`` of the ``modes_stored`` field modes, the truncation error each costs (signal units, the largest over
    the measurements) against ``limit = tol * floor``, where the variances came from, and the stored bytes of the
    position and field tensors read against all of them.

    ``bands_chosen`` / ``modes_chosen`` say whether the count is the plan's (``"planned"``) or the caller's
    (``"explicit"``, ``K=`` / ``modes=``); an explicit count carries the same error. ``modes_stored`` is 0 and ``modes``
    is 0 when the replay reads no field. Dropped independent phases add, so :attr:`error` is the sum."""
    K: int
    K_stored: int
    band_error: float
    band_variance: str
    bands_chosen: str
    modes: int
    modes_stored: int
    mode_error: float
    mode_variance: str
    modes_chosen: str
    tol: float
    floor: float
    bytes_read: int
    bytes_stored: int

    @property
    def limit(self):
        """The error the plan allows: ``tol`` times the floor."""
        return self.tol * self.floor

    @property
    def error(self):
        """The planned truncation error of the read: the band and the mode errors together."""
        return self.band_error + self.mode_error

    def as_dict(self):
        return dict(asdict(self), error=self.error, limit=self.limit)

    def __str__(self):
        s = (f"K = {self.K}/{self.K_stored} bands ({self.bands_chosen}, error {self.band_error:.2e}; variance {self.band_variance})")
        if self.modes_stored:
            s += f"; {self.modes}/{self.modes_stored} modes ({self.modes_chosen}, error {self.mode_error:.2e}; variance {self.mode_variance})"
        return s + f"; limit {self.limit:.2e} = {self.tol:g} x floor {self.floor:.2e}; reads {self.bytes_read} of {self.bytes_stored} bytes"


class Planned(np.ndarray):
    """An array that carries the :class:`ReadPlan` of the replay that made it as :attr:`plan`: a replay's signal, and the
    per-walker factors of :meth:`~dmipy_sim.replay.replay.ReplayPack.walker_signals`. It is an ``ndarray`` in every other
    respect; slicing, arithmetic and reductions keep the plan, and the array a plan-free reader makes has ``plan = None``."""
    plan = None

    def __array_finalize__(self, obj):
        self.plan = getattr(obj, "plan", None)

    def __reduce__(self):
        fn, args, state = super().__reduce__()
        return fn, args, (state, self.plan)

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.plan = state[1]


def planned(array, plan):
    """``array`` as a :class:`Planned` carrying ``plan`` (no copy)."""
    out = np.asarray(array).view(Planned)
    out.plan = plan
    return out


# ------------------------------------------------------------------ the bytes a plan reads
_RANGE = r"_b(\d+)(?:u\d+)?$"


def planned_bytes(table, meta, K, modes):
    """``(read, stored)``: the stored bytes of the position and field tensors a read of ``K`` bands and ``modes`` modes
    touches, and of all of them. ``table`` maps a tensor name to ``(shape, itemsize)``. A band range or mode range is
    read whole when it holds a band or mode of the prefix; a tensor of one piece (the float container's ``pos_x``, an
    unbanded field tensor) is read whole."""
    import re
    cx = meta.get("compression") or {}
    band_starts = [int(r["bands"][0]) for r in cx.get("container") or []]
    mode_starts = [int(r["modes"][0]) for r in ((cx.get("channels") or {}).get("susc_field_basis") or {}).get("container") or []]
    read = stored = 0
    for name, (shape, itemsize) in table.items():
        base = name.split("/")[-1]
        n = int(np.prod(shape, dtype=np.int64)) * int(itemsize)
        m = re.search(_RANGE, base)
        if base.startswith("pos_") and base != "pos_band_scale":
            keep = True if not m else int(m.group(1)) < len(band_starts) and band_starts[int(m.group(1))] < K
        elif base.startswith("susc_path_dct"):
            keep = True if not m else int(m.group(1)) < len(mode_starts) and mode_starts[int(m.group(1))] < modes
            if modes == 0:
                keep = False
        elif base in ("pos_band_scale", "susc_path_scale"):
            keep = base == "pos_band_scale" or modes > 0
        else:
            continue
        stored += n
        read += n if keep else 0
    return read, stored

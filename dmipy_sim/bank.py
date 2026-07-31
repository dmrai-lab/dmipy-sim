"""dmipy-sim replay packs — build, read and replay compressed master walks.

A **replay pack** (``.rpk``) is a single self-describing ``safetensors`` file: the
*compressed master walk* of a substrate (see :mod:`dmipy_sim.compression`) plus every
per-walker channel needed to replay its physics — the run-length-coded compartment map
(per-compartment relaxation), optionally the boundary local time (surface relaxivity)
and the MT bound-pool occupancy — and a JSON metadata block (provenance, source licence,
the **measured** fidelity report, walk parameters, and the certified replay envelope).
safetensors carries no code, so a pack is safe to download and open. The container is the
open **Replay Pack Specification** (https://github.com/dmrai-lab/replay-pack-spec).

Public vs private: the public engine stores POSITIONS and applies susceptibility /
off-resonance at replay via a *provider* (``ReplayPack.replay(susceptibility=...)`` — a
:mod:`dmipy_sim.susceptibility` ``delta_bz_fn``), so packs carry NO packed field-map
channels (``susc_field_{0,C,S}``) and declare no stored Field tier. There is also no
``UnifiedWhiteMatterModel`` in public dmipy-sim; build a pack from a raw master dict, e.g.
:func:`master_from_walk` on a :func:`dmipy_sim.simulate_trajectories` result.

Generate::

    from dmipy_sim import simulate_trajectories, bank
    from dmipy_sim.geometries import Sphere
    res  = simulate_trajectories(n_walkers=20000, diffusivity=2e-9, geometry=Sphere(5e-6),
                                 T_max=40e-3, dt_save=40e-3/64, seed=0,
                                 save_relaxation_data=True)
    m    = bank.master_from_walk(res, D=2e-9, T2_per_comp=[0.05])
    pack = bank.build_replay_pack(m, id="demo/sphere", license="CC-BY-4.0",
                                  citation="...", out_path="sphere.rpk")

Consume (CPU-only)::

    pack   = bank.read_rpk("sphere.rpk")
    signal = pack.replay(waveform, T2=[0.05])                # any acquisition; add
                                                             # surface_relaxivity=... for C2

``safetensors`` is a hard dependency of this module (the ``[bank]`` extra); ``scipy`` is
a core dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import compression as _cx

try:
    from safetensors.numpy import save_file as _st_save, load_file as _st_load
    from safetensors import safe_open as _st_open
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "dmipy_sim.bank needs `safetensors`. Install the bank extra:\n"
        "    pip install 'dmipy-sim[bank]'"
    ) from _e

# Container schema version. The open spec is at "1.2" (the private bank still writes the
# stale "1.1"); public writes 1.2. Readers accept any 1.x (minor bumps are additive).
RPK_SCHEMA_VERSION = "1.2"

# The PUBLIC substrate bank (a HuggingFace Hub *dataset* repo whose file tree is the
# catalogue). The private bank uses a personal repo; public packs live under the org.
DEFAULT_REPO = "dmrai-lab/substrate-bank"


# ----------------------------------------------------------------------------- io
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_rpk(path, arrays: dict, metadata: dict):
    """Write a replay pack. `arrays` maps names→ndarray; `metadata` is a JSON-able dict
    stored (serialised) in the safetensors ``__metadata__`` header under the canonical
    key ``"rpk"`` (Replay Pack Specification §12)."""
    meta = dict(metadata)
    meta.setdefault("rpk_schema_version", RPK_SCHEMA_VERSION)
    tens = {k: np.ascontiguousarray(v) for k, v in arrays.items() if v is not None}
    _st_save(tens, str(path), metadata={"rpk": json.dumps(meta)})
    return path


def read_rpk(path) -> "ReplayPack":
    arrays = _st_load(str(path))
    with _st_open(str(path), framework="numpy") as f:
        hdr = f.metadata() or {}
        # canonical key is "rpk"; "json" is the accepted 1.x legacy alias (spec §12) so
        # packs written before the header-key migration still load.
        meta = json.loads(hdr.get("rpk") or hdr.get("json") or "{}")
    return ReplayPack(arrays, meta, source=str(path))


# --------------------------------------------------------- master normalisation
def master_from_walk(result, *, D, T2_per_comp=None, T1_per_comp=None, w=None,
                     cell_size=None, R=None, mt_params=None, seed=0):
    """Assemble a master dict for :func:`build_replay_pack` from a
    :func:`dmipy_sim.simulate_trajectories` return tuple (walk → dict → pack).

    Handles the 4-tuple ``(traj, dt, sub_steps, dt_sim)``, the 6-tuple with
    ``(..., dlog_boundary_unit, comp_traj)`` (``save_relaxation_data=True``), and the
    7-tuple with a trailing ``bound_frac`` (``kappa_MT>0``). ``D`` is the walk diffusivity
    (m²/s); ``T2_per_comp``/``T1_per_comp`` index the compartment channel; ``w`` are optional
    spin weights; ``cell_size``/``R`` are optional geometry provenance."""
    traj = np.asarray(result[0])
    dt = float(result[1])
    n_t = traj.shape[1]
    m = dict(traj=traj, dt_traj=dt, T_max=dt * (n_t - 1), D_intra=float(D),
             n_walkers=int(traj.shape[0]), seed=int(seed))
    if len(result) >= 6:                       # save_relaxation_data=True
        m["dlog_b"] = np.asarray(result[4])
        m["comp"] = np.asarray(result[5])
    if len(result) >= 7:                       # kappa_MT>0 -> emergent MT occupancy
        m["bfrac"] = np.asarray(result[6])
    if T2_per_comp is not None:
        m["T2_per_comp"] = np.asarray(T2_per_comp)
    if T1_per_comp is not None:
        m["T1_per_comp"] = np.asarray(T1_per_comp)
    if w is not None:
        m["w"] = np.asarray(w)
    if cell_size is not None:
        m["cell_size"] = float(cell_size)
    if R is not None:
        m["R"] = np.asarray(R)
    if mt_params is not None:                  # C4 parametric two-pool qMT pool descriptor
        m["mt_params"] = dict(mt_params)
    return m


def _master_arrays(src) -> dict:
    """Normalise a raw master dict / npz to a common master dict.

    Public deviation: there is no ``UnifiedWhiteMatterModel`` branch (that model does not
    exist in public dmipy-sim), and no packed-myelin Φ-map / per-walker susc-basis keys
    (susceptibility is provider-driven, applied at replay). Required keys: ``traj``,
    ``dt_traj``, ``T_max``; optional ``comp``, ``comp0``, ``w``, ``dlog_b``, ``bfrac``,
    ``seed``, ``T2_per_comp``, ``T1_per_comp``, ``D_intra``, ``R``, ``cell_size``."""
    if hasattr(src, "files"):                  # npz
        m = {k: src[k] for k in src.files}
    elif isinstance(src, dict):
        m = dict(src)
    else:
        raise TypeError(
            "build_replay_pack expects a raw master dict or an .npz (public dmipy-sim has "
            "no UnifiedWhiteMatterModel). Build one from a simulate_trajectories(...) call "
            "with bank.master_from_walk(...).")
    def g(k, d=None):
        return np.asarray(m[k]) if (k in m and m[k] is not None) else d
    def f(k, d=None):
        return float(np.asarray(m[k])) if (k in m and m[k] is not None) else d
    traj = np.asarray(m["traj"])
    return dict(traj=traj, dt_traj=f("dt_traj"), T_max=f("T_max"),
                comp=g("comp"), comp0=g("comp0"), w=g("w"),
                dlog_b=g("dlog_b"), bfrac=g("bfrac"),
                cell_size=f("cell_size"), R=g("R"), D_intra=f("D_intra"),
                T2_per_comp=g("T2_per_comp"), T1_per_comp=g("T1_per_comp"),
                mt_params=(m.get("mt_params") if isinstance(m.get("mt_params"), dict) else None),
                n_walkers=int(traj.shape[0]),
                seed=int(np.asarray(m.get("seed", 0))))


# ------------------------------------------------------------- pack generation
def build_replay_pack(src, *, id, method="lowrank", envelope=None, tol=2.0, K=None,
                      err_target=None, sigma_star=None,
                      license, citation, provenance=None, surface_relaxivity=False,
                      mt=False, out_path=None, verbose=False):
    """Compress a master walk and assemble a self-certifying replay pack.

    `src` is a raw master dict/npz (see :func:`master_from_walk`). The position ensemble is
    compressed by `method` (default `lowrank`); `K` is chosen automatically to keep the
    *measured* replay error within `tol`× the Monte-Carlo floor over `envelope` (default
    :func:`compression.default_envelope`). All per-walker channels required by the certified
    envelope are carried; the fidelity report is stored.

    Walker-preserving methods (`lowrank`, `temporal_dct`) carry the full multi-physics button
    space; distributional methods (`gaussian`, `marginal`) are gradient-only (resampling
    breaks per-walker channel alignment) and set the envelope accordingly.

    Susceptibility is NOT a stored tier in public: it is replayed provider-driven off the
    stored positions (``ReplayPack.replay(susceptibility=...)``), so ``replay_envelope.field``
    is always False here.
    """
    m = _master_arrays(src)
    env = envelope or _cx.default_envelope()
    X = np.asarray(m["traj"], np.float64)
    wp_method = _cx.is_walker_preserving(method)
    if K is None:
        K, fid = _cx.auto_select_modes(X, m, method=method, env=env, tol=tol,
                                       err_target=err_target, verbose=verbose)
    else:
        arrays0, meta0, _ = _cx.encode(X, method, K)
        pos = _cx.decode(arrays0, meta0, n_walkers=(X.shape[0] if wp_method else None))
        fid = _cx.measure_fidelity(m, pos, env)
    if sigma_star is not None:                       # adaptive floor-target policy (build_to_floor)
        fid = dict(fid, target_floor=float(sigma_star),
                   meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))
    pos_arrays, pos_meta, pos_bytes = _cx.encode(X, method, K)

    arrays = dict(pos_arrays)
    chan_meta = {}                                   # per-channel codec params
    channels = {"gradient": True, "T1T2": False, "rho": False, "mt": False}
    if wp_method:
        # compartment map (RLE) -> per-compartment T1/T2; boundary local time / MT opt-in
        if m.get("comp") is not None and m.get("T2_per_comp") is not None:
            # integer labels -> lossless RLE; fractional occupancy (permeable) -> quantized RLE
            _a, _cm = _cx.encode_compartment(np.asarray(m["comp"]))
            arrays.update(_a); chan_meta["compartment"] = _cm
            arrays["spin_weights"] = np.asarray(m["w"], np.float32) if m.get("w") is not None else None
            channels["T1T2"] = True
        # dense per-walker physics channels get their own codecs (compression.py):
        # boundary local time -> sparse/dense; bound_frac -> RLE.
        if surface_relaxivity and m.get("dlog_b") is not None:
            _a, _mm = _cx.encode_boundary_local_time(np.asarray(m["dlog_b"]))
            arrays.update(_a); chan_meta["boundary_local_time"] = _mm; channels["rho"] = True
        if mt and m.get("bfrac") is not None:
            _a, _mm = _cx.encode_bound_fraction(np.asarray(m["bfrac"]), Q=256)
            arrays.update(_a); chan_meta["bound_fraction"] = _mm; channels["mt"] = True

    n_t = X.shape[1]
    comp_meta = dict(method=method, K=int(K), walker_preserving=bool(wp_method))
    if chan_meta:
        comp_meta["channels"] = chan_meta      # per-channel codec params (Q, scale, ...)
    meta = dict(
        rpk_schema_version=RPK_SCHEMA_VERSION, id=id,
        compression=comp_meta,
        walk_params=dict(n_walkers=int(m["n_walkers"]), n_t=int(n_t), dt_traj=float(m["dt_traj"]),
                         T_max=float(m["T_max"]), diffusivity=m.get("D_intra"), seed=int(m["seed"]),
                         cell_size=m.get("cell_size")),
        per_comp=dict(T2=(None if m.get("T2_per_comp") is None else np.asarray(m["T2_per_comp"]).tolist()),
                      T1=(None if m.get("T1_per_comp") is None else np.asarray(m["T1_per_comp"]).tolist()),
                      R=(None if m.get("R") is None else np.asarray(m["R"]).tolist())),
        # explicit, self-describing tier flags (Replay Pack Specification §7/§10).
        # field=False: public susceptibility is provider-driven, not a stored channel.
        replay_envelope=dict(gradient=True,
                             bulk_relaxation=channels["T1T2"],
                             surface_relaxivity=channels["rho"],
                             field=False,
                             magnetization_transfer=channels["mt"],
                             diffusivity_fixed=True, acquisition=_envelope_summary(env)),
        fidelity=fid, provenance=provenance or {}, license=license, citation=citation)
    if m.get("mt_params") is not None:              # C4 parametric two-pool qMT (pool-level knob)
        meta["mt"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in m["mt_params"].items()}
        # SPEC §7/§10: C4 is satisfied by the `mt` pool descriptor (no bound_fraction channel).
        meta["replay_envelope"]["magnetization_transfer"] = True
    pack = ReplayPack(arrays, meta, source=out_path)
    if out_path is not None:
        write_rpk(out_path, arrays, meta)
    if verbose:
        print(f"[pack] {id} method={method} K={K} err={fid['err_max']:.4f} "
              f"floor={fid['floor_max']:.4f} within2x={fid['within_2x_floor']}", flush=True)
    return pack


def build_to_floor(make_model, *, id, envelope=None, sigma_star=1e-3, pilot_n=8000,
                   safety=1.4, max_n=400000, walk=None, method="lowrank", verbose=True, **bp):
    """Adaptive floor-targeting generation policy.

    Size the walker count so the split-half Monte-Carlo floor <= ``sigma_star`` (default
    0.1% of M0), then build a pack whose codec error is <= ``sigma_star`` too. This
    converges to a defined precision — sufficient for any question one would ask, and no
    finer — instead of a wasteful ultra-high N, minimising pack size for that precision.

    ``make_model(n_walkers)`` MUST return a fresh master dict on the SAME fixed geometry
    (only the walker count changes). ``walk(model)`` normalises it (default
    :func:`_master_arrays`). Records ``sigma_star`` + the achieved floor in the pack.
    """
    env = envelope or _cx.default_envelope()
    _walk = walk or _master_arrays
    m0 = make_model(pilot_n); f0 = _cx.measure_floor(_walk(m0), env)
    n_star = int(min(max_n, max(pilot_n, round(pilot_n * (f0 / sigma_star) ** 2 * safety))))
    if verbose: print(f"[floor-target] pilot N={pilot_n}: floor={f0:.4g}; sigma*={sigma_star:.4g} -> N*~{n_star}", flush=True)
    m = make_model(n_star); f = _cx.measure_floor(_walk(m), env)
    if f > sigma_star and n_star < max_n:            # undershoot -> one re-estimate/top-up
        n_star = int(min(max_n, round(n_star * (f / sigma_star) ** 2 * safety)))
        if verbose: print(f"[floor-target] floor={f:.4g} > sigma*; topping up to N*={n_star}", flush=True)
        m = make_model(n_star); f = _cx.measure_floor(_walk(m), env)
    if verbose: print(f"[floor-target] N={n_star}: achieved floor={f:.4g} ({'<=' if f <= sigma_star else '>'} sigma*)", flush=True)
    return build_replay_pack(m, id=id, envelope=env, method=method,
                             err_target=sigma_star, sigma_star=sigma_star, verbose=verbose, **bp)


def _envelope_summary(env):
    return dict(b_max=float(max(env["bvals"])), ogse_periods=list(env["ogse_periods"]),
                B0_list=list(env.get("B0_list", [])), note="temporal band set by max OGSE period / min delta")


# ------------------------------------------------------------------------- pack
class ReplayPack:
    """A prewalked, compressed substrate ready for CPU replay of any acquisition and
    relaxation setting inside its certified envelope. Susceptibility is replayed by
    passing a :mod:`dmipy_sim.susceptibility` provider to :meth:`replay`."""

    def __init__(self, arrays: dict, metadata: dict, source: str | None = None):
        self.arrays = arrays
        self.meta = metadata
        self.source = source
        self.cx = metadata.get("compression", {})
        self.method = self.cx.get("method", "lowrank")
        wp = metadata.get("walk_params", {})
        self.n_t = int(wp.get("n_t") or metadata.get("compression", {}).get("n_t", 0))
        self.dt = float(wp.get("dt_traj", 0.0))
        self._decode_meta = {"method": self.method, "K": int(self.cx.get("K", 0)),
                             "n_t": self.n_t, "n_walkers": int(wp.get("n_walkers", 0)),
                             "Q": int(self.arrays["coeff_quantiles"].shape[0]) if "coeff_quantiles" in arrays else 0}

    # ---- surfaced metadata ----
    license = property(lambda self: self.meta.get("license"))
    citation = property(lambda self: self.meta.get("citation"))
    fidelity = property(lambda self: self.meta.get("fidelity"))
    replay_envelope = property(lambda self: self.meta.get("replay_envelope"))

    def reconstruct_walkers(self, n_walkers=None, seed=0):
        return _cx.decode(self.arrays, self._decode_meta, n_walkers=n_walkers, seed=seed)

    def _comp(self):
        if "comp_rle_vals" not in self.arrays:
            return None
        cm = self._chan_meta("compartment")
        if cm:                                    # codec-aware (integer or fractional-occupancy)
            return _cx.decode_compartment(self.arrays, cm)
        # legacy packs (no channel meta): integer RLE
        return _cx.rle_decode_rows(self.arrays["comp_rle_vals"], self.arrays["comp_rle_lens"],
                                   self.arrays["comp_rle_counts"], self.n_t).astype(np.int8)

    def _chan_meta(self, name):
        return (self.cx.get("channels", {}) or {}).get(name, {})

    def boundary_local_time(self):
        """Decode the surface-relaxivity channel (sparse/dense codec, or legacy raw float16)."""
        if "blt_counts" in self.arrays or "blt_dense_q" in self.arrays:
            return _cx.decode_boundary_local_time(self.arrays, self._chan_meta("boundary_local_time"))
        if "dlog_boundary_unit" in self.arrays:            # legacy raw
            return np.asarray(self.arrays["dlog_boundary_unit"], np.float32)
        return None

    def bound_fraction(self):
        """Decode the MT occupancy channel (RLE codec, or legacy raw float16)."""
        if "bfrac_rle_vals" in self.arrays:
            return _cx.decode_bound_fraction(self.arrays, self._chan_meta("bound_fraction"))
        if "bound_frac" in self.arrays:                    # legacy raw
            return np.asarray(self.arrays["bound_frac"], np.float32)
        return None

    def replay_gradient(self, waveform, weights=None, complex_signal=False,
                        backend="numpy", precision="float64", n_walkers=None, seed=0):
        """Fast **mode-space** gradient replay: evaluate the gradient signal without
        reconstructing the trajectory (compression.mode_space_signal) — exact, ~N_t·3/K
        fewer phase FLOPs. `backend='jax'` runs the matmul + exp-reduce on the device
        (GPU → near-instant); `precision='float32'` is ~1.6x faster on CPU (opt-in).
        `waveform.G` must be on the pack's save grid (n_t)."""
        G = np.asarray(getattr(waveform, "G", waveform), np.float64)
        w = weights if weights is not None else self.arrays.get("spin_weights")
        S = _cx.mode_space_signal(self.arrays, self._decode_meta, G, self.dt, weights=w,
                                  backend=backend, precision=precision, n_walkers=n_walkers, seed=seed)
        return S if complex_signal else S.real

    def replay(self, sequence, *, T2=None, T1=None, surface_relaxivity=None,
               susceptibility=None, eps_P=None, weights=None, complex_signal=None,
               n_walkers=None, seed=0):
        """Replay an acquisition on the pre-walked substrate (CPU).

        ``sequence`` is either a **gradient** :class:`~dmipy_sim.waveforms.Waveform`
        (``.G`` ``(n_meas, n_t, 3)``, optional ``.dt`` / ``.chi_perp`` /
        ``.stimulated_echo``) — or a raw ``G`` array — dispatched to
        :func:`dmipy_sim.trajectories.replay`; **or** a
        :class:`~dmipy_sim.pulse_sequence.BlochSequence` (carrying ``rf_events`` /
        ``echo_steps``), dispatched to the vector-Bloch
        :func:`dmipy_sim.trajectories.replay_bloch`. Dispatch is by *type*
        (``isinstance(sequence, BlochSequence)``), because a ``Waveform`` also exposes an
        ``rf_events`` attribute.

        Relaxation and surface relaxivity are applied **when the caller passes them**:

        * ``T2`` / ``T1`` — a scalar is uniform relaxation; an array is per-compartment
          (indexed by the pack's stored compartment map, :meth:`_comp`).
        * ``surface_relaxivity`` (m/s) — replayed against the pack's boundary channel
          (:meth:`boundary_local_time`), scaled by the walk diffusivity.

        Susceptibility off-resonance is applied when a ``susceptibility=`` provider (a
        :mod:`dmipy_sim.susceptibility` provider or a bare ``r -> ΔBz`` callable) is passed;
        for the gradient path ``eps_P`` is the spin-echo refocusing pathway sign (see
        :func:`dmipy_sim.trajectories.pathway_sign_se`) — the Bloch path refocuses
        emergently and ignores it.

        Returns the ``(n_meas,)`` signal (or ``(n_meas, n_echo)`` for a BlochSequence with
        ``echo_steps``). The gradient path is real (magnitude) by default and complex when
        ``complex_signal=True``; the Bloch path is complex (``Mx + i·My``) by default and
        real when ``complex_signal=False``.
        """
        from .pulse_sequence import BlochSequence
        from .trajectories import replay as _replay, replay_bloch as _replay_bloch
        traj = self.reconstruct_walkers(n_walkers, seed)
        D = float(self.meta["walk_params"]["diffusivity"])
        per = lambda v: v if np.ndim(v) else None       # array -> per-compartment; scalar -> None
        T2pc, T1pc = per(T2), per(T1)
        T2s = None if T2pc is not None else T2
        T1s = None if T1pc is not None else T1
        comp = self._comp() if (T2pc is not None or T1pc is not None) else None
        surf = surface_relaxivity
        dlog = self.boundary_local_time() if surf else None
        Dsurf = D if surf else None
        w = weights if weights is not None else self.arrays.get("spin_weights")
        G = np.asarray(getattr(sequence, "G", sequence), np.float64)
        dt_wf = getattr(sequence, "dt", self.dt)

        if isinstance(sequence, BlochSequence):
            S = _replay_bloch(traj, self.dt, G, dt_wf, sequence.rf_events,
                              T2=T2s, T1=T1s, comp_traj=comp,
                              T2_per_comp=T2pc, T1_per_comp=T1pc,
                              susceptibility=susceptibility, surface_relaxivity=surf, D=Dsurf,
                              dlog_boundary_unit=dlog, echo_steps=sequence.echo_steps,
                              weights=w)
            S = np.asarray(S)
            return S.real if complex_signal is False else S

        # ---- gradient (Waveform) path ----
        chi = getattr(sequence, "chi_perp", None)
        ste = bool(getattr(sequence, "stimulated_echo", False))
        kw = dict(chi_perp=chi, T2=T2s, T1=T1s, comp_traj=comp,
                  T2_per_comp=T2pc, T1_per_comp=T1pc,
                  surface_relaxivity=surf, D=Dsurf, dlog_boundary_unit=dlog,
                  susceptibility=susceptibility, eps_P=eps_P, stimulated_echo=ste)
        if complex_signal:
            phi, logw, _ = _replay(traj, self.dt, G, dt_wf, return_walker_signals=True, **kw)
            phi = np.asarray(phi); logw = np.asarray(logw)
            if logw.shape[-1] == 1:
                logw = np.broadcast_to(logw, phi.shape)
            nw = np.asarray(traj).shape[0]
            ww = np.ones(nw) if w is None else np.asarray(w, float)
            return (np.exp(logw) * np.exp(1j * phi) * (ww / ww.sum())[None, :]).sum(1)
        return np.asarray(_replay(traj, self.dt, G, dt_wf, **kw))

    def replay_dispersed(self, *args, **kwargs):
        """Analytical orientation-dispersion overlay (ODF-weighted superposition of the
        single stored bundle over orientations — SPEC §6.7).

        DEFERRED in public dmipy-sim: the private overlay depends on
        ``sh_convolution.rotate_waveform_by_theta`` / ``fiber_response_from_signal``, which
        are not part of the public ``sh_convolution`` API yet. Rotate the acquisition with
        :func:`dmipy_sim.rotate_waveform` and combine with :func:`dmipy_sim.apply_odf`
        manually until this is ported.
        """
        raise NotImplementedError(
            "replay_dispersed is not yet available in public dmipy-sim (needs the "
            "rotate-waveform / RH-response helpers from sh_convolution). Rotate the "
            "waveform with dmipy_sim.rotate_waveform and use dmipy_sim.apply_odf directly.")

    # ------------------------------------------------------------- C4 MT (two-pool qMT)
    def _mt(self):
        mtp = self.meta.get("mt")
        if mtp is None:
            raise ValueError(f"pack {self.meta.get('id')} carries no MT tier")
        f_b = float(mtp["f_bound"]); k_f = float(mtp["k_forward"])
        k_r = k_f * (1.0 - f_b) / f_b if f_b > 0 else 0.0
        return mtp, k_f, k_r

    def mt_zspectrum(self, offsets_hz, *, w1_hz=1.5, t_sat=1.0):
        """Steady-state MT Z-spectrum (free-pool Mz vs saturation offset, Hz) from the
        geometry-derived two-pool qMT parameters. `w1_hz`=γB1/2π, `t_sat`=sat duration.
        Reuses the :mod:`dmipy_sim.mt` Bloch–McConnell oracle."""
        from .mt import mt_z_spectrum
        mtp, k_f, k_r = self._mt()
        return mt_z_spectrum(np.asarray(offsets_hz, float), w1_hz=w1_hz, t_sat=t_sat,
                             T1a=mtp["T1_free"], T2a=mtp["T2_free"],
                             T1b=mtp["T1_bound"], T2b=mtp["T2_bound"], k_f=k_f, k_r=k_r)

    def mtr(self, *, offset_hz=1.0e4, w1_hz=1.5, t_sat=1.0):
        """MT ratio (1 − M_sat/M0): free-pool signal loss from an off-resonance saturation
        pulse — the standard MT contrast, from the geometry-derived pool. The Z-spectrum is
        normalised to M0, so MTR = 1 − Mz(offset)."""
        s_off = float(np.atleast_1d(self.mt_zspectrum([offset_hz], w1_hz=w1_hz, t_sat=t_sat))[0])
        return 1.0 - s_off

    def __repr__(self):
        f = self.fidelity or {}
        return (f"<ReplayPack {self.meta.get('id','?')} {self.method} K={self.cx.get('K','?')} "
                f"n_t={self.n_t} err={f.get('err_max','?')} licence={self.license}>")


# --------------------------------------------------------- catalogue / croissant
def write_croissant(meta: dict, path, repo_url=None):
    """Write a Croissant (schema.org/Dataset + MLCommons) sidecar for a pack, carrying
    its licence, citation and provenance for dataset-search discoverability (SPEC §12)."""
    cr = {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
        "@type": "Dataset", "name": meta.get("id"),
        "description": f"dmipy-sim replay pack: compressed Monte-Carlo master walk "
                       f"({meta.get('compression',{}).get('method')} K="
                       f"{meta.get('compression',{}).get('K')}) with replay envelope "
                       f"{meta.get('replay_envelope',{})}.",
        "license": meta.get("license"), "citeAs": meta.get("citation"),
        "cr:provenance": meta.get("provenance", {}),
        "cr:replayEnvelope": meta.get("replay_envelope", {}),
        "cr:fidelity": meta.get("fidelity", {}),
        "distribution": [{"@type": "cr:FileObject", "name": f"{meta.get('id')}.rpk",
                          "encodingFormat": "application/octet-stream",
                          "contentUrl": (f"{repo_url}/{meta.get('id')}.rpk" if repo_url else None)}],
    }
    Path(path).write_text(json.dumps(cr, indent=2))
    return path


def stage_pack(rpk_path, staging_dir, artifact_id):
    """Assemble the local mirror of the HF dataset repo: copy the .rpk under its id
    path, write its croissant sidecar + human-readable substrate card, and refresh
    manifest.json + SHA256SUMS + README.md. This is the whole publish step short of the
    network upload — feed the result to :func:`publish_dir`."""
    staging = Path(staging_dir)
    dst = staging / f"{artifact_id}.rpk"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rpk_path, dst)
    pack = read_rpk(dst)
    write_croissant(pack.meta, staging / f"{artifact_id}.croissant.jsonld")
    from .bank_card import write_card                     # human-readable substrate card
    write_card(pack.meta, str(staging / f"{artifact_id}.md"))
    _refresh_catalog(staging)
    return dst


def _refresh_catalog(staging: Path):
    """(Re)build manifest.json (the catalogue index) and SHA256SUMS from the .rpk tree.
    Idempotent — deterministic in the current tree, independent of call count."""
    staging = Path(staging)
    entries, sums = [], []
    for rpk in sorted(staging.rglob("*.rpk")):
        rel = rpk.relative_to(staging).as_posix()
        h = sha256(rpk)
        try:
            pk = read_rpk(rpk); meta = pk.meta
        except Exception:
            meta = {}
        entries.append(dict(id=meta.get("id", rel[:-4]), file=rel, sha256=h,
                            bytes=rpk.stat().st_size, license=meta.get("license"),
                            replay_envelope=meta.get("replay_envelope"),
                            fidelity=meta.get("fidelity", {}).get("err_max"),
                            compression=meta.get("compression")))
        sums.append(f"{h}  {rel}")
    (staging / "manifest.json").write_text(json.dumps(
        dict(schema=f"dmipy-sim substrate-bank/{RPK_SCHEMA_VERSION}",
             n_entries=len(entries), entries=entries), indent=2))
    (staging / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    _write_bank_readme(staging, entries)
    return staging / "manifest.json"


def _write_bank_readme(staging: Path, entries: list):
    """Repo-level dataset card (README.md): what the bank is + an index of substrates,
    each linking its own substrate card. HF renders this as the dataset landing page."""
    _tier = [("gradient", "C0"), ("bulk_relaxation", "C1"), ("surface_relaxivity", "C2"),
             ("field", "C3"), ("magnetization_transfer", "C4")]
    rows = []
    for e in entries:
        env = e.get("replay_envelope") or {}
        tiers = " ".join(cx for k, cx in _tier if env.get(k))
        aid = e.get("id", "")
        rows.append(f"| [`{aid}`]({aid}.md) | {tiers or '—'} | "
                    f"{e.get('bytes', 0)/1e6:.2f} MB | {e.get('license', '')} |")
    L = ["---", "license: cc-by-4.0",
         "tags:", "  - diffusion-mri", "  - monte-carlo", "  - replay-pack", "  - microstructure",
         "pretty_name: dmrai-lab substrate bank", "---", "",
         "# dmrai-lab substrate bank", "",
         "Compressed Monte-Carlo **replay packs** (`.rpk`) for diffusion-MRI microstructure. "
         "Each pack is one master walk on a substrate plus the physics channels needed to "
         "**replay** any acquisition inside its certified envelope — *one walk, every "
         "acquisition* — in the open [Replay Pack Specification]"
         "(https://github.com/dmrai-lab/replay-pack-spec) format.", "",
         "```python", "from dmipy_sim import bank",
         'pack = bank.pull("<id>", repo_id="dmrai-lab/substrate-bank")',
         "S = pack.replay(waveform, T2=[0.05])", "```", "",
         "## Substrates", "",
         "| Substrate | Replay tiers | Size | License |", "|---|---|---|---|"]
    L += sorted(rows)
    L += ["", "Each substrate links to its own **substrate card** with geometry, provenance, "
          "fidelity and load/replay instructions. Tier legend: C0 gradient · C1 bulk relaxation "
          "· C2 surface relaxivity · C3 field · C4 magnetization transfer.", ""]
    (staging / "README.md").write_text("\n".join(L))


# --------------------------------------------------------------------- remote
def _cache_dir() -> Path:
    d = Path(os.environ.get("DMRAI_BANK_CACHE", Path.home() / ".cache" / "dmrai-bank"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def pull(artifact_id, repo_id=DEFAULT_REPO, revision="main", expected_sha256=None, local_path=None):
    """Resolve `artifact_id` → download (cached) → hash-verify → ReplayPack.
    `local_path` bypasses the network (open a staged/local `.rpk` directly)."""
    if local_path is not None:
        path = Path(local_path)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover
            raise ImportError("remote pull needs `huggingface_hub`, or pass local_path=") from e
        path = Path(hf_hub_download(repo_id=repo_id, filename=f"{artifact_id}.rpk",
                                    repo_type="dataset", revision=revision, cache_dir=str(_cache_dir())))
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"hash mismatch for {artifact_id}")
    return read_rpk(path)


def publish_dir(staging_dir, repo_id=DEFAULT_REPO, create=True, private=True):
    """Upload a staged bank mirror (rpk + croissant + manifest + SHA256SUMS + cards + README)
    in one commit. Requires an ambient HuggingFace write token (`hf auth login`). Repos are
    created PRIVATE by default (flip to public on the Hub when ready). Returns the repo URL."""
    from huggingface_hub import HfApi, create_repo
    if create:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    HfApi().upload_folder(folder_path=str(staging_dir), repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"


def publish(local_file, artifact_id, repo_id=DEFAULT_REPO, private=True):
    """Upload a single .rpk (one commit). Requires an ambient HuggingFace write token."""
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    HfApi().upload_file(path_or_fileobj=local_file, path_in_repo=f"{artifact_id}.rpk",
                        repo_id=repo_id, repo_type="dataset")
    return f"{repo_id}:{artifact_id}"

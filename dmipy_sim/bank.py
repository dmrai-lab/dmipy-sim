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
    signal = pack.replay(waveform, relaxation=True)          # any acquisition

``safetensors`` is a hard dependency of this module (the ``[bank]`` extra); ``scipy`` is
a core dependency.
"""
from __future__ import annotations

import hashlib
import json

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
                     cell_size=None, R=None, seed=0):
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

    def replay(self, waveform, *, susceptibility=None, eps_P=None,
               relaxation=True, rho=0.0, complex_signal=False, n_walkers=None, seed=0,
               backend="numpy", precision="float64"):
        """Replay a waveform on the CPU. `waveform` has `.G` (n_meas,n_t,3) and `.dt`
        (or pass a raw G array). Per-compartment relaxation is applied when `relaxation`
        and the pack carries the compartment map; surface relaxivity when `rho>0` and the
        pack carries the boundary channel. Susceptibility off-resonance is applied when a
        `susceptibility=` provider (a :mod:`dmipy_sim.susceptibility` provider or a bare
        ``r -> ΔBz`` callable) is passed — `eps_P` is the SE refocusing pathway sign (see
        :func:`dmipy_sim.trajectories.pathway_sign_se`). Returns the (complex or magnitude)
        signal (n_meas,).

        Pure-gradient / relaxation replay (no susceptibility) on a walker-preserving pack
        takes the fast mode-space path (no trajectory reconstruction) when the waveform is
        on the save grid; susceptibility (nonlinear-in-position field lookup) or an off-grid
        waveform fall back to reconstruct-then-contract via the Phase-1/2 replay ops.
        """
        G = np.asarray(getattr(waveform, "G", waveform), np.float64)
        dt_wf = getattr(waveform, "dt", self.dt)
        Nt = G.shape[1]
        wp = self.meta.get("walk_params", {}); pc = self.meta.get("per_comp", {})
        w = self.arrays.get("spin_weights")
        # ---- fast path: mode-space phase + separable relaxation/surface log-weights ----
        # Everything except susceptibility (nonlinear field lookup) replays WITHOUT
        # reconstructing the trajectory: φ in the K-mode space (compression.mode_space_phi)
        # times a per-walker log-weight summed from the compartment map (relaxation) and the
        # boundary channel (surface). Needs a walker-preserving pack and a waveform on the
        # save grid; a susceptibility provider or an off-grid waveform take the general path.
        _wp = bool(self.cx.get("walker_preserving"))
        _fast_method = self.method in ("lowrank", "temporal_dct", "gaussian", "marginal")
        if (susceptibility is None) and _fast_method and Nt == self.n_t and (n_walkers is None or not _wp):
            if _wp:                                    # lowrank / dct: per-walker + log-weights
                logw = None
                if relaxation and "comp_rle_vals" in self.arrays and pc.get("T2"):
                    logw = _cx.relaxation_logweight(self._comp(), pc["T2"], pc.get("T1"), self.dt)
                if rho:
                    blt = self.boundary_local_time()
                    if blt is not None:
                        sl = _cx.surface_logweight(blt, float(rho) / float(wp.get("diffusivity")))
                        logw = sl if logw is None else logw + sl
                S = _cx.mode_space_signal(self.arrays, self._decode_meta, G, self.dt,
                                          logw=logw, weights=w, backend=backend, precision=precision)
            else:                                      # distributional: gradient-only, resample n_walkers
                S = _cx.mode_space_signal(self.arrays, self._decode_meta, G, self.dt,
                                          backend=backend, precision=precision,
                                          n_walkers=n_walkers, seed=seed)
            return S if complex_signal else S.real
        # general path: reconstruct the trajectory + contract (relaxation / surface / susc)
        from .trajectories import (apply_waveform_with_relaxation,
                                   apply_waveform_to_trajectories)
        traj = np.asarray(self.reconstruct_walkers(n_walkers, seed), np.float32)
        nw = traj.shape[0]
        G = G.astype(np.float32)
        kw = dict(chi_perp=np.ones(Nt, np.float32))
        walker_preserving = nw == wp.get("n_walkers") and self.cx.get("walker_preserving")
        comp = self._comp() if (relaxation and walker_preserving) else None
        if comp is not None and pc.get("T2"):
            kw.update(comp_traj=comp, T2_per_comp=np.asarray(pc["T2"]))
            if pc.get("T1") is not None:   # T2-only packs carry T1=None
                kw["T1_per_comp"] = np.asarray(pc["T1"])
        blt = self.boundary_local_time() if walker_preserving else None
        if rho and blt is not None:
            kw.update(dlog_boundary_unit=np.asarray(blt),
                      rho=float(rho), D=float(wp.get("diffusivity")))
        if susceptibility is not None:
            kw.update(susceptibility=susceptibility, eps_P=eps_P)
        if not (kw.keys() - {"chi_perp"}):            # pure gradient, no extra physics
            return np.asarray(apply_waveform_to_trajectories(traj, self.dt, G, dt_wf))
        if complex_signal:
            phi, logw, _ = apply_waveform_with_relaxation(
                traj, self.dt, G, dt_wf, return_walker_signals=True, **kw)
            phi = np.asarray(phi); logw = np.asarray(logw)
            if logw.shape[-1] == 1:
                logw = np.broadcast_to(logw, phi.shape)
            ww = np.ones(nw) if w is None else np.asarray(w, float)
            sig = (np.exp(logw) * np.exp(1j * phi) * (ww / ww.sum())[None, :]).sum(1)
            return sig
        return np.asarray(apply_waveform_with_relaxation(traj, self.dt, G, dt_wf, **kw))

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

    def __repr__(self):
        f = self.fidelity or {}
        return (f"<ReplayPack {self.meta.get('id','?')} {self.method} K={self.cx.get('K','?')} "
                f"n_t={self.n_t} err={f.get('err_max','?')} licence={self.license}>")

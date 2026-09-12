"""Replay-pack assembler — walk once, compress, self-certify, freeze into a ``.rpk``.

:func:`build_replay_pack` turns a **master walk** (from
:func:`dmipy_sim.engine.core.simulate_trajectories`) into a compressed,
self-describing replay pack: the position ensemble is compressed by a :mod:`dmipy_sim.replay.compression`
codec (``bridge_dst``), the tier channels the certified envelope needs are carried
(bulk relaxation via the compartment map, surface relaxivity via the boundary-local-time channel,
magnetization transfer via the bound-fraction channel), and the pack MEASURES its own replay
fidelity against the split-half Monte-Carlo floor before it is written. The result is a single
``safetensors`` file readable by :func:`dmipy_sim.replay.replay.read_rpk` and replayable by the
:mod:`dmipy_sim.replay.replay` / :mod:`dmipy_sim.replay.trajectories` operators.

:func:`build_to_floor` is the bank's default generation policy: size the walker count so the
Monte-Carlo floor meets a target ``sigma_star``, then build a pack whose codec error meets it too.

This is the *producer* side of the substrate bank. Publishing/pulling packs to a HuggingFace
dataset (the federation layer) lives separately. The susceptibility (field) replay tier needs the
per-walker susceptibility-basis channel and its ``Q(H)`` contraction (a separate module) and is not
assembled here yet.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

import numpy as np

from ..persistent_walk import PersistentWalk

from . import compression as _cx
from ._replay_kernel import se_gate, gradient_phase
from ..acquisition.rf import RFEvent
from .replay import ReplayPack, read_rpk, write_rpk

__all__ = ["build_replay_pack", "build_to_floor", "replay_susc", "frame_from_axis", "frame_from_bundles",
           "read_rpk", "write_rpk", "RPK_SCHEMA_VERSION"]

RPK_SCHEMA_VERSION = "0.4"


# --------------------------------------------------------------- master-walk normalisation
def _master_arrays(src) -> dict:
    """Normalise a master walk to the dict consumed by :func:`build_replay_pack`.

    ``src`` is a :class:`~dmipy_sim.persistent_walk.PersistentWalk` (what ``simulate_trajectories`` and
    ``simulate_mt_trajectories`` return), or its bank dict / ``.npz`` (``PersistentWalk._bank_dict``) exposing
    at least ``traj`` (n_walkers, n_t, 3), ``dt_traj`` and ``T_max``;
    the tier channels (``comp`` for bulk relaxation, ``dlog_b`` for
    surface relaxivity, ``bfrac`` for MT) are optional."""
    if isinstance(src, PersistentWalk):
        src = src._bank_dict()
    if not (isinstance(src, dict) or hasattr(src, "files")):
        raise TypeError(
            "build_replay_pack expects a PersistentWalk (the output of simulate_trajectories(..., "
            "tiers=\"all\")) or a master-walk dict / .npz with keys "
            "traj/dt_traj/T_max[/comp/dlog_b/bfrac]; "
            f"got {type(src).__name__}.")
    keys = src.files if hasattr(src, "files") else src.keys()
    m = {k: src[k] for k in keys}
    g = lambda k, d=None: (np.asarray(m[k]) if k in m and m[k] is not None else d)
    traj = np.asarray(m["traj"])
    scal = lambda k: (float(np.asarray(m[k])) if m.get(k) is not None else None)   # idempotent re-normalise
    return dict(traj=traj, dt_traj=float(np.asarray(m["dt_traj"])),
                T_max=float(np.asarray(m["T_max"])), comp=g("comp"), comp0=g("comp0"),
                w=g("w"), dlog_b=g("dlog_b"), bfrac=g("bfrac"),
                # static field-grid susceptibility channel (dict of grids + world origin + chi)
                susc_field_basis=(m.get("susc_field_basis") if isinstance(m, dict) else None),
                susc_field_sampler=(m.get("susc_field_sampler") if isinstance(m, dict) else None),
                susc_grid_origin=(np.asarray(m["susc_grid_origin"]) if "susc_grid_origin" in m else None),
                susc_chi_iso=scal("susc_chi_iso"), delta_chi_a=scal("delta_chi_a"),
                cell_size=scal("cell_size"), R=g("R"), D_intra=scal("D_intra"),
                substrate_frame=g("substrate_frame"),
                walkers_shuffled=bool(m.get("walkers_shuffled", False)),
                substrate=(m.get("substrate") if isinstance(m, dict) else None),
                n_walkers=int(traj.shape[0]), seed=int(np.asarray(m.get("seed", 0))))


# --------------------------------------------------------------- substrate frames
def frame_from_axis(axis):
    """Deterministic orthonormal substrate frame R (3x3, columns [x, y, z]) with z = `axis`
    (the primary fibre direction) and a FIXED perpendicular x/y basis (Gram-Schmidt seeded
    from the global axis least aligned with z). A single fibre vector leaves a free rotation
    about itself; this pins x/y so directions are reproducible run-to-run and gradient schemes
    are oriented unambiguously. For an isotropic substrate any axis works — the frame is still
    fixed, giving uniform behaviour across packs."""
    z = np.asarray(axis, float); z = z / np.linalg.norm(z)
    seed = np.eye(3)[int(np.argmin(np.abs(z)))]        # global axis least aligned with z
    x = seed - z * float(seed @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def frame_from_bundles(axes, *, primary=0, weights=None, tol=1e-3):
    """Substrate frame for a MULTI-bundle (e.g. crossing) substrate, anchored to a PRIMARY bundle.

    Do NOT use a PCA over all positions here: for two crossing bundles PCA returns the *bisector*,
    which is no real bundle's axis and rotates as the crossing angle opens. Instead pin the primary
    bundle (z) and let the most-transverse secondary define the plane (-> +y); x = y x z. With one
    bundle / all parallel it degrades to :func:`frame_from_axis`. `axes` is (n_bundles, 3) of
    per-bundle mean tangents (NOT a vertex PCA)."""
    A = np.asarray(axes, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    if primary is None:
        primary = int(np.argmax(weights)) if weights is not None else 0
    z = A[primary]
    perp = A - (A @ z)[:, None] * z[None, :]
    mag = np.linalg.norm(perp, axis=1); mag[primary] = 0.0
    j = int(np.argmax(mag))
    if mag[j] < tol:
        return frame_from_axis(z)
    y = perp[j] / mag[j]
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


# --------------------------------------------------------------- fidelity helpers
def _envelope_summary(env):
    return dict(b_max=float(max(env["bvals"])), ogse_periods=list(env.get("ogse_periods", [])),
                B0_list=list(env.get("B0_list", [])),
                note="temporal band set by max OGSE period / min delta")


def _measure_floor(m, env):
    """Split-half Monte-Carlo floor of the RAW walk over the envelope (decoded == raw ->
    fidelity err is 0, so ``floor_max`` is the substrate's own finite-N statistical noise)."""
    traj = np.asarray(m["traj"], np.float64)
    return float(_cx.measure_fidelity(traj, float(m["dt_traj"]), traj, env)["floor_max"])


def _surface_fidelity(m, arrays, chan_meta, env):
    """Certify the surface tier (C2): the surface-relaxivity attenuation reconstructed from the
    STORED boundary channel vs the RAW per-step boundary local time, over a rho battery, against
    the split-half MC floor of the raw surface signal. Returns ``dict(err, floor)`` or None."""
    raw = m.get("dlog_b")
    has_stored = any(k in arrays for k in ("blt_bridge_dst", "blt_dense_q", "blt_counts"))
    if raw is None or not has_stored:
        return None
    raw = np.asarray(raw, np.float64); n_w = raw.shape[0]
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    D = float(m.get("D_intra") or 0.0) or 1.0
    if "blt_bridge_dst" in arrays:
        decoded = _cx.decode_boundary_bridge(arrays, chan_meta)
    else:
        decoded = _cx.decode_boundary_local_time(arrays, chan_meta)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    fac = lambda sl, idx: float(np.sum(w[idx] * np.exp(sl[idx])) / np.sum(w[idx]))
    err = floor = 0.0
    for rho in (env.get("rho_list") or [1e-5, 3e-5, 1e-4]):
        rd = float(rho) / D
        sl_raw = _cx.surface_logweight_series(raw, rd)
        sl_dec = _cx.surface_logweight_series(decoded, rd)
        err = max(err, abs(fac(sl_raw, slice(None)) - fac(sl_dec, slice(None))))
        floor = max(floor, abs(fac(sl_raw, A) - fac(sl_raw, B)))
    return dict(err=float(err), floor=float(floor))


def voxel_fidelity(traj, dt, decoded_pos, grid, comp, env, *, w=None, logw=None, chunk=20_000):
    """The pack's fidelity PER VOXEL AND POOL, for a walk meant to be partitioned: every walker is binned by where
    it started (``grid.bin``, the partition's own rule) and by its pool; per (voxel, pool) the ensemble signal
    over the envelope's acquisition battery is formed from the raw and the decoded paths, and a split-half
    Monte-Carlo floor from the voxel's own walkers. Returns ``(ijk, pools, n, floor, err)``: the occupied voxels
    ``(n_v, 3)``, the pool ids, and ``(n_v, n_pools)`` arrays of the walker count, the floor (max over the
    battery of ``|S_a - S_b| / 2``) and the codec error (max over the battery of ``|S_dec - S_raw|``); a voxel
    with fewer than two walkers of a pool has ``nan`` for its floor. Walker chunks keep the memory at
    ``chunk x n_meas`` complex."""
    traj = np.asarray(traj, np.float64); dec = np.asarray(decoded_pos, np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    G, _ = _cx.acquisition_battery(n_t, dt, env); n_m = G.shape[0]
    ijk_all, inside = grid.bin(traj[:, 0])
    pools = sorted(set(np.unique(np.asarray(comp)[:, 0]).tolist())) if comp is not None else [0]
    pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(n_w, np.int64)
    ijk = np.unique(ijk_all[inside], axis=0)
    key = {tuple(v): i for i, v in enumerate(map(tuple, ijk))}
    row = np.full(n_w, -1, np.int64)
    for i in np.flatnonzero(inside):
        row[i] = key[tuple(ijk_all[i])]
    col = np.searchsorted(pools, pid)
    ww = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    lw = np.zeros(n_w) if logw is None else np.asarray(logw, np.float64)
    half = (np.random.default_rng(0).permutation(n_w) % 2).astype(bool)      # a fixed split of every voxel's walkers
    n_v, n_p = ijk.shape[0], len(pools)
    S_raw = np.zeros((n_v, n_p, n_m), complex); S_dec = np.zeros_like(S_raw)
    S_a = np.zeros_like(S_raw); S_b = np.zeros_like(S_raw)
    W = np.zeros((n_v, n_p)); W_a = np.zeros((n_v, n_p)); W_b = np.zeros((n_v, n_p)); N = np.zeros((n_v, n_p), np.int64)
    for i in range(0, n_w, chunk):
        sl = slice(i, i + chunk); m = row[sl] >= 0
        if not m.any():
            continue
        idx = np.arange(i, min(i + chunk, n_w))[m]
        e_raw = np.exp(lw[idx, None] + 1j * _cx._walker_phases(traj[idx], dt, G)) * ww[idx, None]
        e_dec = np.exp(lw[idx, None] + 1j * _cx._walker_phases(dec[idx], dt, G)) * ww[idx, None]
        r, c, h = row[idx], col[idx], half[idx]
        np.add.at(S_raw, (r, c), e_raw); np.add.at(S_dec, (r, c), e_dec)
        np.add.at(S_a, (r[h], c[h]), e_raw[h]); np.add.at(S_b, (r[~h], c[~h]), e_raw[~h])
        np.add.at(W, (r, c), ww[idx]); np.add.at(W_a, (r[h], c[h]), ww[idx][h]); np.add.at(W_b, (r[~h], c[~h]), ww[idx][~h])
        np.add.at(N, (r, c), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.abs(S_dec / W[..., None] - S_raw / W[..., None]).max(-1)
        floor = 0.5 * np.abs(S_a / W_a[..., None] - S_b / W_b[..., None]).max(-1)
    err[W == 0] = np.nan; floor[(W_a == 0) | (W_b == 0)] = np.nan
    return ijk, pools, N, floor, err


def voxel_fidelity_volumes(pack):
    """The per-voxel certificate of a pack built with ``voxel_grid=`` as dense volumes on that grid:
    ``(grid, {pool name: floor volume}, {pool name: walker-count volume})``, the pools named by the pack's
    embedded spec -- the input of :func:`dmipy_sim.spec.seeding.plan_seeding`."""
    from ..phantom.grid import Grid
    pv = (pack.meta.get("fidelity") or {}).get("per_voxel")
    if pv is None or "voxel_certificate" not in pack.arrays:
        raise ValueError("this pack carries no per-voxel certificate; build it with voxel_grid=")
    if pack.substrate is None:
        raise ValueError("this pack embeds no spec, so its pools have no names")
    names = {p.id: p.name for p in pack.substrate.pools}
    grid = Grid.from_meta(pv["grid"]); ijk = np.asarray(pack.arrays["voxel_ijk"], np.int64)
    cert = np.asarray(pack.arrays["voxel_certificate"], np.float64)              # (n_v, n_pools, 3): n, floor, err
    floors, counts = {}, {}
    for j, pid in enumerate(pv["pools"]):
        fl = np.zeros(grid.shape); n = np.zeros(grid.shape)
        fl[tuple(ijk.T)] = np.nan_to_num(cert[:, j, 1], nan=0.0); n[tuple(ijk.T)] = cert[:, j, 0]
        floors[names[int(pid)]] = fl; counts[names[int(pid)]] = n
    return grid, floors, counts


def _field_of(m):
    """The master's field source as a sampler (``channels(points)`` / ``field(points, ...)``): the grid as a
    :class:`FieldGrid`, or the strand substrate's :class:`StrandFieldBasis`; None without a field tier."""
    fb = m.get("susc_field_basis")
    if fb is not None:
        from ..fields.susceptibility_field import FieldGrid
        return FieldGrid(fb, np.asarray(m["susc_grid_origin"], float))
    return m.get("susc_field_sampler")


def _field_along(field, traj, b0_dir, *, B0, chi_iso, chi_aniso, walkers_per_chunk=None):
    """``(n_w, n_t)`` field (Tesla) along every walker's path, in walker chunks."""
    traj = np.asarray(traj, np.float64); n_w, n_t = traj.shape[0], traj.shape[1]
    step = walkers_per_chunk or max(1, int(2e8 // max(n_t * 13, 1)))
    out = np.empty((n_w, n_t), np.float64)
    for i in range(0, n_w, step):
        out[i:i + step] = field.field(traj[i:i + step].reshape(-1, 3), b0_dir, B0=B0, chi_iso=chi_iso,
                                      chi_aniso=chi_aniso).reshape(-1, n_t)
    return out


def _susc_grid_fidelity(m, arrays, gm, decoded_pos, dt, env):
    """Certify the static field-grid tier (SE + GRE): the STORED f16 grid sampled at the DECODED
    trajectory vs the RAW f64 grid at the FULL-resolution trajectory (folds in f16 quantisation AND
    the position-codec error), against the split-half MC floor of the raw signal. Returns dict or None."""
    from ..constants import GAMMA
    from ..fields.susceptibility_field import assemble_field, sample_grid
    fb = m.get("susc_field_basis")
    if fb is None or "susc_grid_iso_local" not in arrays:
        return None
    origin = np.asarray(gm["origin"], float); vs = np.asarray(gm["voxel_size"], float)
    raw_traj = np.asarray(m["traj"], np.float64); n_w, n_t = raw_traj.shape[0], raw_traj.shape[1]
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    braw = {"iso_local": np.asarray(fb["iso_local"], np.float64), "iso_P": np.asarray(fb["iso_P"], np.float64),
            "aniso_G": (np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None),
            "shape": tuple(fb["shape"]), "voxel_size": vs}
    bsto = {"iso_local": np.asarray(arrays["susc_grid_iso_local"], np.float64),
            "iso_P": np.asarray(arrays["susc_grid_iso_P"], np.float64),
            "aniso_G": (np.asarray(arrays["susc_grid_aniso_G"], np.float64) if "susc_grid_aniso_G" in arrays else None),
            "shape": tuple(gm["shape"]), "voxel_size": vs}
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    se = np.sign(0.5 * (n_t - 1) - np.arange(n_t)).astype(float); gre = np.ones(n_t)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            s_raw = sample_grid(assemble_field(braw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), raw_traj, origin, vs)
            s_dec = sample_grid(assemble_field(bsto, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), decoded_pos, origin, vs)
            for gate in (se, gre):
                cr = np.cos(GAMMA * dt * (s_raw * gate[None, :]).sum(1))
                cd = np.cos(GAMMA * dt * (s_dec * gate[None, :]).sum(1))
                err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
                floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return dict(err=float(err), floor=float(floor))


_ISO_P_NAMES = ("xx", "yy", "zz", "xy", "xz", "yz")


def _cpmg_gate(n_t, n_pulses):
    """Transverse-phase gate for a CPMG train of ``n_pulses`` 180s at the usual (k+1/2)*TE/n times."""
    t = np.arange(n_t) / max(n_t - 1, 1)
    s = np.ones(n_t)
    for k in range(n_pulses):                       # each 180 flips the accumulated-phase sign
        s[t >= (k + 0.5) / n_pulses] *= -1.0
    return s


def _susc_path_bloch_fidelity(m, arrays, pm, gm, env, n_sub=8000):
    """Certify the susc_path tier ON THE VECTOR-BLOCH PATH -- the primary consumer.

    The scalar gate in :func:`_susc_path_fidelity` integrates the right quantity, but it does not
    exercise the Bloch operator, where the RF rotations do NOT commute with the field's z-precession
    and relaxation interleaves between pulses. A pack that claims CPMG capability should be certified
    by the operator that actually replays CPMG, so this drives ``replay_bloch`` with the STORED
    (K-truncated, f16) channels against the RAW full-resolution field, at spin echo and at the CPMG
    train the pack advertises.

    Walker-subsampled (``n_sub``) because the quantity measured is codec error, which is per-walker;
    the comparison floor is the split-half floor OF THAT SUBSAMPLE, so the verdict stays self-consistent.
    """
    from ..constants import GAMMA
    from .trajectories import replay_bloch
    field = _field_of(m)
    if field is None or "susc_path_dct" not in arrays:
        return None
    traj = np.asarray(m["traj"], np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    dt = float(m["dt_traj"]); TE = (n_t - 1) * dt
    k = int(min(n_sub, n_w))
    pos = traj[:k]
    w = (np.asarray(m["w"], np.float64)[:k] if m.get("w") is not None else np.ones(k))
    b_dec, _ = susc_path_decode(arrays, pm, n_w=k)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)

    def rf_train(npul):
        ev = [RFEvent(0.0, 90.0, 'Mz→Mxy')]
        for j in range(npul):
            ev.append(RFEvent((j + 0.5) * TE / npul, 180.0, 'refocus', axis_deg=90.0))
        return ev

    G0 = np.zeros((1, n_t, 3))                       # b=0 isolates the susceptibility physics
    def sig(field, rf, idx=None):
        f = field if idx is None else field[idx]
        p = pos if idx is None else pos[idx]
        ww = w if idx is None else w[idx]
        out = replay_bloch(p, dt, G0, dt, rf, T2=None, T1=None,
                           extra_phase_per_step=GAMMA * dt * f, weights=ww)
        a = np.asarray(out[0] if isinstance(out, tuple) else out)
        return float(np.abs(np.atleast_1d(a.ravel())[-1]))

    perm = np.random.RandomState(0).permutation(k); A, B = perm[:k // 2], perm[k // 2:]
    err = floor = 0.0
    B0s = (env.get("B0_list") or [7.0])[:1]          # one field strength: the tier scales linearly in B0
    for B0 in B0s:
        for th in (env.get("theta_deg") or [90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = _field_along(field, pos, d, B0=B0, chi_iso=chi_i, chi_aniso=ca)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            for npul in (1, n_p):
                rf = rf_train(npul)
                sr = sig(f_raw, rf)
                err = max(err, abs(sig(f_dec, rf) - sr))
                floor = max(floor, abs(sig(f_raw, rf, A) - sig(f_raw, rf, B)))
    return dict(err=float(err), floor=float(floor), n_pulses=n_p, n_walkers=k)


def _susc_path_fidelity(m, arrays, pm, gm, env):
    """Certify the susc_path_dct tier AT ITS DECLARED CAPABILITY.

    Compares the stored (K-truncated, f16) channels against the raw full-resolution field sampled on
    the true trajectory, under the gates the pack claims to support -- GRE, spin echo, AND the CPMG
    train at ``max_refocus_pulses``. The CPMG gate is the binding one: truncation error grows with
    gate bandwidth, so certifying on SE alone would pass a pack that fails the trains it advertises.
    """
    from ..constants import GAMMA
    field = _field_of(m)
    if field is None or "susc_path_dct" not in arrays:
        return None
    traj = np.asarray(m["traj"], np.float64); n_w, n_t = traj.shape[0], traj.shape[1]
    dt = float(m["dt_traj"])
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    b_dec, _ = susc_path_decode(arrays, pm)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)
    gates = [np.ones(n_t), _cpmg_gate(n_t, 1), _cpmg_gate(n_t, n_p)]
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = _field_along(field, traj, d, B0=B0, chi_iso=chi_i, chi_aniso=ca)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            for g in gates:
                cr = np.cos(GAMMA * dt * (f_raw * g[None, :]).sum(1))
                cd = np.cos(GAMMA * dt * (f_dec * g[None, :]).sum(1))
                err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
                floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return dict(err=float(err), floor=float(floor), n_pulses_certified=n_p)


def susc_path_encode(field, traj, *, K=32, bits=8, dtype=np.float16, atol_trace=1e-6):
    """Encode the off-resonance field ALONG each walker's path as K temporal DCT-II coefficients.

    ``field`` is the substrate's field source -- a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` or a
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` -- read through ``channels(points)``; ``traj`` the
    full-resolution walk ``(n_w, n_t, 3)``. The ``iso_P_zz`` channel is implied by the trace identity
    ``iso_P_xx + iso_P_yy + iso_P_zz = 3 iso_local`` and left out when the source satisfies it to ``atol_trace``.
    """
    from scipy.fft import dct
    traj = np.asarray(traj, np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    names_all = list(field.channel_names)
    step = max(1, int(2e8 // max(n_t * 13, 1)))                 # walkers per chunk: ~1.6 GB of channels
    first = field.channels(traj[:min(step, n_w)].reshape(-1, 3))
    tr = first[:, 1] + first[:, 2] + first[:, 3]
    scale = float(np.max(np.abs(first[:, 0]))) or 1.0
    trace_res = float(np.max(np.abs(tr - 3.0 * first[:, 0]))) / (3.0 * scale)
    drop_zz = bool(trace_res <= atol_trace)
    keep = [i for i, n in enumerate(names_all) if not (drop_zz and n == "iso_P_zz")]
    names = [names_all[i] for i in keep]
    grids = keep                                                  # the channel columns stored
    K = int(min(K, n_t))
    coeffs = np.empty((n_w, len(grids), K), np.float64)
    for i in range(0, n_w, step):                      # walker chunks: the channels of a chunk, then their DCT
        ch = first if i == 0 else field.channels(traj[i:i + step].reshape(-1, 3))
        ch = ch.reshape(-1, n_t, ch.shape[-1])
        for c, col in enumerate(grids):
            coeffs[i:i + step, c, :] = dct(ch[:, :, col], type=2, norm="ortho", axis=1)[:, :K]
    meta = dict(channel="susc_path_dct", K=K, n_t=int(n_t), n_ch=len(grids), channels=names,
                iso_P_zz=("implied" if drop_zz else "stored"), trace_residual=trace_res,
                max_refocus_pulses=K // 2)
    if bits is None:
        meta["bits"] = None; meta["dtype"] = np.dtype(dtype).name
        return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta
    if bits not in (8, 16):
        raise ValueError("susc_path bits must be 8, 16, or None (got %r); sub-byte depths need "
                         "bit-packing to save bytes and 6-bit measured above the MC floor" % (bits,))
    itype = np.int8 if bits == 8 else np.int16
    lim = 2 ** (bits - 1) - 1
    scale = np.abs(coeffs).max(axis=0) / lim                     # (n_ch, K), per channel AND band
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(coeffs / scale), -lim, lim).astype(itype)
    meta["bits"] = int(bits); meta["dtype"] = np.dtype(itype).name
    return {"susc_path_dct": q, "susc_path_scale": np.asarray(scale, np.float32)}, meta


def susc_path_coeffs(arrays, meta):
    """Dequantised C3 coefficients with ``iso_P_zz`` re-inserted -- ``(n_w, n_ch_full, K)``.

    The ONE place that undoes the storage container. ``susc_path_dct`` may be an integer array whose
    scale lives in ``susc_path_scale`` (see :func:`susc_path_encode`), so reading it raw yields
    integers ~1e3 off with no error -- every consumer goes through here. The zz identity is applied
    in coefficient space; the DCT is linear, so this is identical to applying it after the transform.
    """
    C = np.asarray(arrays["susc_path_dct"], np.float64)
    if "susc_path_scale" in arrays:
        C = C * np.asarray(arrays["susc_path_scale"], np.float64)[None]
    names = list(meta["channels"])
    if meta.get("iso_P_zz") == "implied":
        i_loc, i_xx, i_yy = (names.index(n) for n in ("iso_local", "iso_P_xx", "iso_P_yy"))
        at = names.index("iso_P_xy")
        C = np.insert(C, at, 3.0 * C[:, i_loc] - C[:, i_xx] - C[:, i_yy], axis=1)
        names = names[:at] + ["iso_P_zz"] + names[at:]
    return C, names


def susc_path_decode(arrays, meta, *, n_w=None):
    """Reconstruct b_c(t) per walker from the stored coefficients; re-inserts iso_P_zz if implied.

    Returns ``(field, names)`` with field ``(n_w, n_ch_full, n_t)`` in the canonical channel order
    (iso_local, iso_P_xx..yz, [aniso_G_xx..yz]) so the Q(H) contraction indexes it directly.
    """
    from scipy.fft import idct
    C, names = susc_path_coeffs(arrays, meta)
    if n_w is not None:
        C = C[:int(n_w)]
    n_t = int(meta["n_t"])
    b = idct(C, type=2, norm="ortho", axis=2, n=n_t) if C.shape[2] == n_t else \
        idct(np.pad(C, ((0, 0), (0, 0), (0, n_t - C.shape[2]))), type=2, norm="ortho", axis=2)
    return b, names


def susc_path_field(b, b0_dir, *, B0, chi_iso, chi_aniso=0.0, has_aniso=False):
    """Contract decoded path channels into dB(t) for one (B0, direction, chi) -- mirrors assemble_field."""
    from ..fields.susceptibility_field import _q_of_H
    q = _q_of_H(b0_dir)
    dB = chi_iso * B0 * (b[:, 0] - np.einsum("c,cwt->wt", q, np.swapaxes(b[:, 1:7], 0, 1)))
    if has_aniso and chi_aniso and b.shape[1] >= 13:
        dB = dB + chi_aniso * B0 * np.einsum("c,cwt->wt", q, np.swapaxes(b[:, 7:13], 0, 1))
    return dB


# --------------------------------------------------------------- susceptibility replay (consume)
def _pack_positions(pack):
    """The pack's decoded trajectory (see :meth:`ReplayPack.positions`)."""
    return pack.positions()


def replay_susc(pack, waveform, *, b0_dir=(0.0, 0.0, 1.0), B0=0.0, chi_iso=0.0, chi_aniso=0.0,
                T2=None, T1=None, complex_signal=False, compartment=None):
    """``pack.replay(waveform, B0=..., ...)``: the field-tier replay, kept under its old name; per-pool
    ``T2`` / ``T1`` are given here, the pack carries none; the 180 is the waveform's own."""
    return pack.replay(waveform, tissue=False, T2=T2, T1=T1, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso,
                       compartment=compartment, complex_signal=complex_signal)


# --------------------------------------------------------------- pack generation
def _precision_tiers(arrays, n_walkers, floor_max, walkers_shuffled):
    """How many leading walkers a consumer must read to reach a given statistical floor.

    Every walker-indexed array is stored walkers-LEADING, so the first ``n`` walkers are one
    contiguous byte range per tensor -- a range GET / ``safetensors.get_slice(name)[0:n]``, no need to
    pull the whole pack. The MC floor scales as 1/sqrt(n), so n(eps) = n_walkers*(floor_max/eps)^2.

    This is only meaningful if walker ORDER is exchangeable. Producers that seed pool-by-pool (e.g.
    intra-axonal then myelin) must shuffle, or a prefix silently returns a one-pool substrate; the
    producer asserts that by setting ``walkers_shuffled`` on the master. Without it the tiers are
    still reported but flagged unusable, rather than quietly offering a biased read.
    """
    per_walker, fixed, whole = 0, 0, []
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == n_walkers:
            per_walker += int(a.nbytes // max(n_walkers, 1))
        else:
            # Not walker-leading -- e.g. a run-length channel whose runs vary per walker. Row n of such
            # an array is not walker n, so it cannot be prefix-sliced and must be read whole. Counting
            # it as fixed overhead keeps a small tier's quoted size honest instead of understating it.
            fixed += int(a.nbytes); whole.append(k)
    tiers = []
    if floor_max > 0:
        for eps in (1e-2, 3e-3, 1e-3):
            n = int(min(n_walkers, max(1, np.ceil(n_walkers * (floor_max / eps) ** 2))))
            tiers.append(dict(eps=float(eps), n_walkers=n, bytes=int(n * per_walker + fixed),
                              reachable=bool(n < n_walkers or floor_max <= eps)))
    return dict(bytes_per_walker=per_walker, fixed_bytes=fixed,
                not_prefix_sliceable=sorted(whole), floor_at_full_n=float(floor_max),
                walkers_shuffled=bool(walkers_shuffled),
                usable=bool(walkers_shuffled), tiers=tiers,
                note=("prefix reads are unbiased" if walkers_shuffled else
                      "WALKER ORDER NOT DECLARED SHUFFLED -- a prefix may be a biased sub-ensemble"))


def _select_boundary_codec(m, dlog, env, tol, dtype, verbose=False):
    """Choose the C2 (boundary-local-time) codec by COST subject to the surface-fidelity gate.

    The historical default was sparse CSR, which is exact but costs ~one entry per wall contact, so it
    scales with walk length (574 B/walker at n_t=1601 for an axon, and worse the longer you record).
    The detrended-cumulative DCT is flat in n_t and, at K=32 in f16, lands thousands of times below the
    surface split-half floor -- but it was opt-in, so the *default* pack got the expensive channel.
    Cost each candidate, keep the cheapest that passes, and fall back to exact sparse if none do.
    """
    cands = []
    for K in (8, 16, 32, 64):
        a, mm = _cx.encode_boundary_bridge(dlog, K=int(K), dtype=dtype)
        cf = _surface_fidelity(m, a, mm, env)
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        cands.append((nb, K, a, mm, cf))
        if cf is not None and cf["err"] <= tol * cf["floor"]:
            if verbose:
                log.info(f"[bank] C2 codec: boundary_dct K={K} {np.dtype(dtype).name} "
                      f"({nb:.0f} B/walker, err={cf['err']:.2e} vs floor {cf['floor']:.2e})")
            return a, mm
    a, mm = _cx.encode_boundary_local_time(dlog)
    if verbose:
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        best = min(cands, key=lambda c: (c[4] or {}).get("err", np.inf))
        log.info(f"[bank] C2 codec: no DCT K passed the surface gate (best K={best[1]} "
              f"err={(best[4] or {}).get('err', float('nan')):.2e}); using exact sparse "
              f"({nb:.0f} B/walker)")
    return a, mm


def preflight_master(m, *, susc_path_K=None, sigma_star=None, K=None):
    """Check a master walk against the tiers its content will assemble. Returns a list of
    problems, empty if it will work.

    Every one of these is knowable from metadata, yet each cost a full walk to discover:

    * the CACTUS master emits the per-walker ``susc_basis`` unless the producer asks for
      ``field_store='grid'``; ``build_replay_pack`` then refuses the master outright -- AFTER the
      walk, because the walk happens inside ``_master_arrays``.
    * a master carrying ``susc_field_basis=None`` silently yields a pack with no field tier, so
      the pack looks fine and is missing a capability. Presence of the KEY is not enough; the
      value is what matters.
    * an auto-selected ``K`` is sized against this walk's own floor. For a pack that will later be
      merged with others, the floor falls as 1/sqrt(N) while codec error does not, so a K that
      passes here can make compression the dominant error in the union.

    Call it before walking where possible, or at least before encoding.
    """
    bad = []
    if m.get("PhiC") is not None or m.get("susc_basis") is not None:
        bad.append("master carries the private susceptibility forms (susc_basis/PhiC), which are "
                   "refused here; rebuild the master with field_store='grid', or drop them and "
                   "rely on susc_field_basis")
    if susc_path_K and m.get("susc_field_basis") is None and m.get("susc_field_sampler") is None:
        bad.append("susc_path_K was requested but the master carries no field basis (grid or sampler), so the field tier (C3) "
                   "cannot be assembled and the pack would be silently missing it "
                   "(field_store='grid' is what produces it)")
    if sigma_star is not None and K is None:
        bad.append(f"sigma_star={sigma_star:g} with K unpinned: K is auto-selected against THIS "
                   f"walk's floor, which is the wrong reference for a pack that will be merged "
                   f"or compared to an absolute target. Pin K.")
    return bad



def _walk_master(walk, *, weights=None, field=None, diffusivity=None, substrate_frame=None):
    """The bank's master dict from a PersistentWalk plus the substrate metadata; a bank dict / .npz passes
    through."""
    from ..persistent_walk import PersistentWalk
    from ..compartments import Compartments
    from ..fields.susceptibility_field import FieldGrid, field_grid_of
    if not isinstance(walk, PersistentWalk):
        if any(v is not None for v in (weights, diffusivity, substrate_frame)) or field not in ("auto", None, False):
            raise TypeError("weights=, field=, diffusivity= and substrate_frame= go with a PersistentWalk; a master "
                            "dict carries them as its own keys")
        return walk
    geometry = walk.geometry
    spec = walk.spec if walk.spec is not None else (getattr(geometry, "spec", None) if geometry is not None else None)
    if weights is None and walk.weights is None and spec is not None and walk.compartment is not None:
        wf = [p.water_fraction for p in sorted(spec.pools, key=lambda p: p.id)]
        if any(f != 1.0 for f in wf):                     # the seeding rule's weights, from the spec
            weights = np.asarray(wf, float)[np.asarray(walk.compartment)[:, 0].astype(int)]
    if field == "auto":
        field = None
        if walk.field_basis is not None:
            field = walk.field_basis
        elif geometry is not None and type(geometry).__name__ in ("MyelinatedCylinder", "PackedMyelinatedCylinders"):
            field = field_grid_of(geometry)                          # geometry only; chi is a replay knob
    elif field is False:
        field = None
    if weights is None and walk.weights is not None:
        weights = walk.weights
    extra = {}
    if spec is not None:
        extra["substrate"] = spec.to_dict()
    if weights is not None:
        w = np.asarray(weights, float).reshape(-1)
        if w.shape[0] != walk.n_walkers:
            raise ValueError(f"weights has {w.shape[0]} entries for {walk.n_walkers} walkers")
        extra["w"] = w
    if field is not None:
        from ..fields.strand_field import StrandFieldBasis
        if isinstance(field, FieldGrid):
            extra.update(susc_field_basis=field.basis, susc_grid_origin=np.asarray(field.origin, float))
        elif isinstance(field, StrandFieldBasis):
            extra["susc_field_sampler"] = field
        else:
            raise TypeError("field must be a fields.susceptibility_field.FieldGrid (basis, origin) or a "
                            f"fields.strand_field.StrandFieldBasis, got {type(field).__name__}")
    if diffusivity is not None:
        extra["D_intra"] = float(diffusivity)
    if substrate_frame is not None:
        extra["substrate_frame"] = np.asarray(substrate_frame, float)
    extra["walkers_shuffled"] = True        # the producer draws walkers i.i.d.: any prefix is a fair subsample
    return walk._bank_dict(**extra)

def build_replay_pack(walk, *, id, license, citation, weights=None, field="auto",
                      method=_cx.POSITION_METHOD, envelope=None, tol=2.0, K=None,
                      err_target=None, sigma_star=None, provenance=None,
                      blt_temporal_K=None, blt_dtype=np.float16, susc_path_K=None, susc_path_bits=8, voxel_grid=None,
                      diffusivity=None, substrate_frame=None, out_path=None, verbose=False):
    """Compress a persistent walk and assemble a self-certifying replay pack.

    ``walk`` is the :class:`~dmipy_sim.persistent_walk.PersistentWalk` a producer returned (the
    walk's bank dict / ``.npz`` is also accepted). The tiers assembled are the ones the
    walk CARRIES, with what they need read from the geometry the walk was run on
    (``walk.geometry`` / ``walk.spec``): **gradient** (C0, always); **bulk relaxation** (C1) when the walk
    has a compartment channel (the pools' T2 / T1 are replay knobs; the pack carries none); **surface relaxivity**
    (C2) when the walk has the boundary local time; **magnetization transfer** (C4) when it has the
    bound fraction; **field** (C3) when a static field basis exists for the substrate --
    ``field="auto"`` derives it from a myelinated geometry (:func:`fields.susceptibility_field.field_grid_of`),
    a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` supplies one (a mesh substrate), a
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` the per-segment closed form of a strand substrate
    (path channel only: it has no grid),
    ``field=False`` leaves the tier out; the basis is geometry only, and B0, its direction and the
    susceptibilities are replay knobs. ``weights`` are per-walker proton-density weights (default:
    the pools' water fractions by compartment, else uniform).

    The position ensemble is compressed by ``method`` (default ``bridge_dst``: endpoints plus a
    Brownian bridge on the sine basis, which holds both endpoints exactly and pairs its first two
    coefficients with the gradient moments -- see :func:`compression.encode_bridge_dst`); ``K``
    (mode count) is chosen automatically to keep the *measured* replay error within ``tol`` x the
    Monte-Carlo floor over ``envelope`` (default :func:`compression.default_envelope`) unless given.
    ``voxel_grid`` (a :class:`~dmipy_sim.phantom.Grid`, substrate-attached) adds the PER-VOXEL certificate a
    partition needs: walkers binned by where they started and by pool, and per (voxel, pool) the split-half
    floor and the codec error over the acquisition battery, stored as ``voxel_ijk`` / ``voxel_certificate``
    with a summary in ``fidelity["per_voxel"]`` (:func:`voxel_fidelity`; :func:`voxel_fidelity_volumes` reads
    it back, :func:`dmipy_sim.spec.seeding.plan_seeding` turns a pilot's into the next walk's counts).

    Returns a :class:`dmipy_sim.replay.replay.ReplayPack`; writes it to ``out_path`` if given.
    """
    src = _walk_master(walk, weights=weights, field=field, diffusivity=diffusivity, substrate_frame=substrate_frame)
    _cx.require_position_method(method)
    m = _master_arrays(src)
    env = envelope or _cx.default_envelope()
    X = np.asarray(m["traj"], np.float64)
    dt = float(m["dt_traj"])
    wp_method = _cx.is_walker_preserving(method)
    if K is None:
        K, fid = _cx.auto_select_modes(X, X, dt, method=method, env=env, tol=tol,
                                       err_target=err_target, verbose=verbose)
    else:
        arrays0, meta0, _ = _cx.encode(X, method, K)
        pos = _cx.decode(arrays0, meta0, n_walkers=(X.shape[0] if wp_method else None))
        fid = _cx.measure_fidelity(X, dt, pos, env)
    if sigma_star is not None:                       # adaptive floor-target policy (build_to_floor)
        fid = dict(fid, target_floor=float(sigma_star),
                   meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))

    pos_arrays, pos_meta, _ = _cx.encode(X, method, K)
    arrays = dict(pos_arrays)
    chan_meta = {}                                   # per-channel codec params
    channels = {"gradient": True, "susceptibility": False, "T1T2": False, "rho": False,
                "mt": (m.get("bfrac") is not None)}
    # STATIC field-grid susceptibility channel: store the geometry-only field-basis grids ONCE
    # (a substrate property); replay assembles the field for any (B0,dir,chi) and samples it along
    # the pos-codec-decoded trajectory (replay_susc). O(N_vox) not O(N_w*N_t) and SE-exact (a static
    # field at a frozen point cancels under the SE gate to machine precision). f16 grids: O(1) geometry.
    _field = _field_of(m)
    if _field is not None and m.get("susc_field_basis") is None:
        # a strand substrate's per-segment field: no grid to store, the path channel is the tier
        if not susc_path_K:
            raise ValueError("a StrandFieldBasis has no grid to store: the field tier (C3) needs susc_path_K")
        chan_meta["susceptibility_grid"] = dict(has_aniso=True, arrays_in_pack=False, replay_route="path", source=_field.meta)
        channels["susceptibility"] = True
        _a, _pm = susc_path_encode(_field, np.asarray(m["traj"], np.float64), K=int(susc_path_K), bits=susc_path_bits)
        arrays.update(_a); chan_meta["susceptibility_path"] = _pm
    if m.get("susc_field_basis") is not None:
        fb = m["susc_field_basis"]
        # The GRID route samples the field at codec-DECODED positions, so it is only sound when the
        # position codec is lossless. The PATH route samples the FULL-RESOLUTION trajectory at build
        # time, which is precisely what frees the positions to be lossy -- so the two cannot both be
        # advertised: shipping grid arrays next to lossy positions would offer a replay route whose
        # accuracy silently depends on a property the pack no longer has. Path wins when present;
        # the grid is then published as a separate per-substrate companion artefact, not per walker.
        # A full-rank walker-preserving codec is an exact rewrite; the rank is n_t for
        # n_t-2 for bridge_dst, which stores two endpoints outside the bands.
        _pos_lossless = _cx.is_lossless_at(method, int(K), int(X.shape[1]))
        _grid_in_pack = (not susc_path_K) or _pos_lossless
        if _grid_in_pack:
            arrays["susc_grid_iso_local"] = np.asarray(fb["iso_local"], np.float16)
            arrays["susc_grid_iso_P"] = np.asarray(fb["iso_P"], np.float16)
            if fb.get("aniso_G") is not None:
                arrays["susc_grid_aniso_G"] = np.asarray(fb["aniso_G"], np.float16)
        chan_meta["susceptibility_grid"] = dict(
            origin=np.asarray(m["susc_grid_origin"], float).tolist(),
            voxel_size=np.asarray(fb["voxel_size"], float).tolist(),
            shape=[int(s) for s in fb["shape"]], has_aniso=(fb.get("aniso_G") is not None),
            arrays_in_pack=bool(_grid_in_pack),
            replay_route=("grid+path" if (_grid_in_pack and susc_path_K)
                          else ("path" if susc_path_K else "grid")))
        channels["susceptibility"] = True
        # PATH form (C3, preferred): the field sampled along each walker's FULL-RESOLUTION path and
        # compressed in time. Decouples the susceptibility tier from the position codec -- which is
        # what lets the positions go back to K << n_t, since the only reason they had to be stored
        # losslessly was that grid-sampling needed exact r(t). See susc_path_encode for why K is a
        # gate-bandwidth capability rather than a fidelity knob.
        if susc_path_K:
            _a, _pm = susc_path_encode(_field, np.asarray(m["traj"], np.float64),
                                       K=int(susc_path_K), bits=susc_path_bits)
            arrays.update(_a); chan_meta["susceptibility_path"] = _pm
    if wp_method:
        # C1 (occupancy): the geometric compartment plus, when the walk bound spins, the MT bound
        # pool as a SECOND COLUMN on an independent axis -- not a channel of its own. Replay weights
        # the per-pool rates by occupancy either way; what makes MT a distinct tier is the replay
        # side (vector-Bloch RF, bound-pool knobs, equilibrium start), not the storage.
        if m.get("comp") is not None:
            _cols = {"comp": np.asarray(m["comp"])}
            if m.get("bfrac") is not None:
                _cols["bound"] = np.asarray(m["bfrac"]); channels["mt"] = True
            _a, _cm = _cx.encode_occupancy(_cols)
            arrays.update(_a); chan_meta["compartment"] = _cm
            if m.get("w") is not None:
                arrays["spin_weights"] = np.asarray(m["w"], np.float32)
            channels["T1T2"] = True
        elif m.get("bfrac") is not None:
            raise ValueError("an MT (C4) pack carries its bound pool as a C1 occupancy column, so "
                             "it needs the compartment channel too.")
        # dense per-walker physics channels get their own codecs (compression.py):
        # boundary local time -> sparse/dense or the cumulative bridge.
        if m.get("dlog_b") is not None:
            if blt_temporal_K:
                _a, _mm = _cx.encode_boundary_bridge(np.asarray(m["dlog_b"]), K=int(blt_temporal_K),
                                                 dtype=blt_dtype)
            else:
                _a, _mm = _select_boundary_codec(m, np.asarray(m["dlog_b"]), env, tol,
                                                 blt_dtype, verbose)
            arrays.update(_a); chan_meta["boundary_local_time"] = _mm; channels["rho"] = True

    # Surface tier (C2) fidelity: certify the boundary channel reproduces the surface-relaxivity
    # signal from its stored coeffs, vs the raw boundary local time.
    if channels["rho"] and chan_meta.get("boundary_local_time") is not None:
        _cf = _surface_fidelity(m, arrays, chan_meta["boundary_local_time"], env)
        if _cf is not None:
            fid = dict(fid, err_surface=_cf["err"], floor_surface=_cf["floor"],
                       err_max=max(float(fid.get("err_max", 0.0)), _cf["err"]),
                       floor_max=max(float(fid.get("floor_max", 0.0)), _cf["floor"]))
            fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
            if sigma_star is not None:
                fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

    # Field tier (C3) fidelity: certify the stored f16 grid sampled at the decoded trajectory
    # reproduces the raw-grid/true-trajectory susceptibility signal (SE + GRE, split-half floor).
    if channels["susceptibility"] and chan_meta.get("susceptibility_path") is not None:
        _pf = _susc_path_fidelity(m, arrays, chan_meta["susceptibility_path"],
                                  chan_meta["susceptibility_grid"], env)
        if _pf is not None:
            _bf = _susc_path_bloch_fidelity(m, arrays, chan_meta["susceptibility_path"],
                                            chan_meta["susceptibility_grid"], env)
            fid = dict(fid, err_susc_path=_pf["err"], floor_susc_path=_pf["floor"],
                       susc_path_pulses_certified=_pf["n_pulses_certified"],
                       err_susc_bloch=(None if _bf is None else _bf["err"]),
                       floor_susc_bloch=(None if _bf is None else _bf["floor"]),
                       susc_bloch_walkers=(None if _bf is None else _bf["n_walkers"]),
                       err_max=max(float(fid.get("err_max", 0.0)), _pf["err"],
                                   (0.0 if _bf is None else _bf["err"])),
                       floor_max=max(float(fid.get("floor_max", 0.0)), _pf["floor"],
                                     (0.0 if _bf is None else _bf["floor"])))
            fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
            if sigma_star is not None:
                fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)
    if channels["susceptibility"] and m.get("susc_field_basis") is not None:
        _dpos = _cx.decode(pos_arrays, pos_meta, n_walkers=(X.shape[0] if wp_method else None))
        _gf = _susc_grid_fidelity(m, arrays, chan_meta["susceptibility_grid"], _dpos, dt, env)
        if _gf is not None:
            fid = dict(fid, err_susc_se=_gf["err"], floor_susc_se=_gf["floor"],
                       err_max=max(float(fid.get("err_max", 0.0)), _gf["err"]),
                       floor_max=max(float(fid.get("floor_max", 0.0)), _gf["floor"]))
            fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
            if sigma_star is not None:
                fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

    n_t = X.shape[1]
    if voxel_grid is not None:
        from ..phantom.grid import Grid
        if not isinstance(voxel_grid, Grid) or voxel_grid.attach != "substrate":
            raise TypeError("voxel_grid must be a dmipy_sim.phantom.Grid attached to the substrate")
        _dpos = _cx.decode(pos_arrays, pos_meta, n_walkers=(X.shape[0] if wp_method else None))
        _w = np.asarray(m["w"], np.float64) if m.get("w") is not None else None
        _ijk, _pools, _n, _floor, _err = voxel_fidelity(X, dt, _dpos, voxel_grid, m.get("comp"), env, w=_w)
        arrays["voxel_ijk"] = _ijk.astype(np.int32)
        arrays["voxel_certificate"] = np.stack([_n.astype(np.float32), _floor.astype(np.float32), _err.astype(np.float32)], axis=-1)
        _ok = np.isfinite(_floor)
        fid = dict(fid, per_voxel=dict(grid=voxel_grid.to_meta(), pools=[int(p) for p in _pools], n_voxels=int(_ijk.shape[0]),
                                       walkers_min=int(_n[_n > 0].min()) if (_n > 0).any() else 0,
                                       floor_max=float(np.nanmax(_floor)) if _ok.any() else None,
                                       floor_median=float(np.nanmedian(_floor)) if _ok.any() else None,
                                       err_max=float(np.nanmax(_err)) if np.isfinite(_err).any() else None,
                                       within_2x_floor_fraction=(float(np.mean(_err[_ok] <= 2.0 * _floor[_ok])) if _ok.any() else None),
                                       thin_voxels=int(((_n > 0) & (_n < 2)).sum())))
        if sigma_star is not None and _ok.any():
            fid["per_voxel"]["meets_target"] = bool(np.nanmax(_floor) <= sigma_star and np.nanmax(_err) <= sigma_star)
    comp_meta = dict(method=method, K=int(pos_meta.get("K", K)),        # the K stored: the codec clamps a short walk
                     walker_preserving=bool(wp_method), n_t=int(n_t))
    if wp_method:
        comp_meta["precision_tiers"] = _precision_tiers(arrays, int(m["n_walkers"]),
                                                        float(fid.get("floor_max") or 0.0),
                                                        bool(m.get("walkers_shuffled")))
    if chan_meta:
        comp_meta["channels"] = chan_meta      # per-channel codec params (Q, scale, ...)
    meta = dict(
        rpk_schema_version=RPK_SCHEMA_VERSION, id=id,
        compression=comp_meta,
        walk_params=dict(n_walkers=int(m["n_walkers"]), n_t=int(n_t), dt_traj=dt,
                         T_max=float(m["T_max"]), diffusivity=m.get("D_intra"), seed=int(m["seed"]),
                         cell_size=m.get("cell_size"),
                         substrate_frame=(None if m.get("substrate_frame") is None
                                          else np.asarray(m["substrate_frame"], float).tolist())),
        replay_envelope=dict(gradient=True,
                             bulk_relaxation=channels["T1T2"],
                             surface_relaxivity=channels["rho"],
                             field=channels["susceptibility"],
                             magnetization_transfer=channels["mt"],
                             diffusivity_fixed=True, acquisition=_envelope_summary(env)),
        fidelity=fid, provenance=provenance or {}, license=license, citation=citation)
    if m.get("substrate") is not None:
        meta["substrate"] = m["substrate"]           # the spec the walk was driven by (#130)
    pack = ReplayPack(arrays, meta, source=out_path)
    if out_path is not None:
        write_rpk(out_path, {k: v for k, v in arrays.items() if v is not None}, meta)
    if verbose:
        log.info(f"[pack] {id} method={method} K={K} err={fid['err_max']:.4f} "
              f"floor={fid['floor_max']:.4f} within2x={fid['within_2x_floor']}")
    return pack


def build_to_floor(make_model, *, id, envelope=None, sigma_star=1e-3, pilot_n=8000,
                   safety=1.4, max_n=400000, walk=None, method="bridge_dst", verbose=True, **bp):
    """Adaptive floor-targeting generation policy (the bank default).

    Size the walker count so the split-half Monte-Carlo floor <= ``sigma_star``, then build a pack
    whose codec error is <= ``sigma_star`` too — converging to a defined precision instead of a
    wasteful ultra-high N. ``make_model(n_walkers)`` MUST return a fresh master walk (dict/.npz) on
    the SAME fixed geometry (only the walker count changes). ``walk(model)`` returns its master dict
    (default: the model already IS one). Records ``sigma_star`` + the achieved floor in the pack.
    """
    env = envelope or _cx.default_envelope()
    _walk = walk or _master_arrays
    f0 = _measure_floor(_walk(make_model(pilot_n)), env)
    n_star = int(min(max_n, max(pilot_n, round(pilot_n * (f0 / sigma_star) ** 2 * safety))))
    if verbose:
        log.info(f"[floor-target] pilot N={pilot_n}: floor={f0:.4g}; sigma*={sigma_star:.4g} -> N*~{n_star}")
    model = make_model(n_star); f = _measure_floor(_walk(model), env)
    if f > sigma_star and n_star < max_n:            # undershoot -> one re-estimate/top-up
        n_star = int(min(max_n, round(n_star * (f / sigma_star) ** 2 * safety)))
        if verbose:
            log.info(f"[floor-target] floor={f:.4g} > sigma*; topping up to N*={n_star}")
        model = make_model(n_star); f = _measure_floor(_walk(model), env)
    if verbose:
        log.info(f"[floor-target] N={n_star}: achieved floor={f:.4g} "
              f"({'<=' if f <= sigma_star else '>'} sigma*)")
    # build_replay_pack normalises the raw model itself (idempotent if already a master dict)
    return build_replay_pack(model, id=id, envelope=env, method=method,
                             err_target=sigma_star, sigma_star=sigma_star, verbose=verbose, **bp)

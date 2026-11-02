"""The operations that re-encode a replay pack (RPK.md 8): each takes a pack and returns a new one (or restates it),
certified as the walk's own pack is.

* :func:`truncate_pack` -- the first windows of a windowed pack, a range of its tensors.
* :func:`rewindow_pack` -- a one-window walk stored in windows of a shorter duration.
* :func:`prefix_pack` -- the walk cut at a shorter echo time and re-projected on the prefix's own basis.
* :func:`stated_envelope` -- the certificate settings a re-encoding is certified under.
* :func:`restate_surface_envelope` -- the contact tier stated at the envelope it serves.

Writing a pack is :func:`~dmipy_sim.replay.container.save_pack`; continuing its walk is
:func:`~dmipy_sim.replay.continuation.extend_pack`; publishing it is :func:`~dmipy_sim.replay.publish.publish`.
"""
import numpy as np
from dmipy_replay.replay.compression import c2_bands_K as _cx_bands_K
from dmipy_replay.replay.hub import sha256_of

from .container import save_pack, write_rpk

__all__ = ["truncate_pack", "rewindow_pack", "prefix_pack", "stated_envelope", "restate_surface_envelope"]


def truncate_pack(pack, n_keep, *, id=None, out_path=None):
    """The first ``n_keep`` windows as a pack: a prefix of whole windows is the range of their tensors and
    nothing is re-encoded (RPK.md 8.3). The windows keep their certificates and the whole's is the bound
    over them (:func:`~dmipy_sim.replay.bank.combine_window_fidelity`)."""
    import copy
    from .bank import combine_window_fidelity
    seg = pack.windows; n = int(seg["n"]); n_keep = int(n_keep)
    if n_keep < 1 or n_keep > n:
        raise ValueError(f"keep between 1 and {n} windows; got {n_keep}")
    if n_keep == n:
        return pack
    keep = {k: v for k, v in pack.arrays.items()
            if "/" not in k or int(k[1:k.index("/")]) < n_keep}
    meta = copy.deepcopy(pack.meta)
    wp = meta["walk_params"]; T_seg = float(seg["T"])
    walks = [dict(w, last=min(int(w["last"]), n_keep - 1)) for w in seg.get("walks") or [] if int(w["first"]) < n_keep]
    wp.update(T_max=n_keep * T_seg, windows=dict(n=n_keep, T=T_seg, walks=walks))
    fid = dict(meta.get("fidelity") or {})
    per = fid.get("windows")
    if per:
        meta["fidelity"] = combine_window_fidelity(per[:n_keep])
    meta.setdefault("provenance", {})["truncated"] = dict(parent_id=pack.meta.get("id"), windows_kept=n_keep, of=n)
    if id is not None:
        meta["id"] = id
    out = type(pack)(keep, meta, source=None)
    if out_path is not None:
        save_pack(out, out_path)
    return out


def rewindow_pack(pack, T_seg, *, out_path, id=None, max_dropped_fraction=1e-3, envelope=None, tol=2.0):
    """This one-window walk stored in windows of ``T_seg`` seconds (RPK.md 4.3), so that a consumer holds only
    the windows its acquisitions reach (#525).

    Every window is the parent's stored functions cut to it and re-projected on the window's own basis
    (:func:`~dmipy_sim.replay.bank.window_of_pack`: the bridge and the field modes by their closed-form window
    maps, the events sliced), at the parent's bands per second (``K_seg = ceil(K T_seg / T)``, doubled until
    window 0's certificate passes), joined by :func:`~dmipy_sim.replay.bank.join_windows`. A whole number of
    windows cover the walk up to a trailing remainder of at most ``max_dropped_fraction`` of its duration,
    which is dropped (``time_dropped``); a walk that would leave more is refused. Each window's certificate
    (``fidelity.windows[i]``) carries the walk-free terms and the re-projection's measured truncation against
    the parent's exact reading of the window, and the whole's is the bound over them
    (:func:`~dmipy_sim.replay.bank.combine_window_fidelity`); a window outside ``tol`` times its floor is
    refused. ``provenance.rewindowed`` records the parent (id, sha256 of its file, certificate), ``T_seg``, the
    bands and ``time_dropped``. ``envelope`` is the certificate settings every window is certified under: by default
    :func:`stated_envelope`, the parent's envelope on the default battery."""
    import copy
    from .bank import window_of_pack, combine_window_fidelity, join_windows
    if pack.n_windows != 1:
        raise ValueError(f"this pack already stores its walk in {pack.n_windows} windows of "
                         f"{float(pack.windows['T']) * 1e3:.3f} ms; rewindow reads a one-window walk")
    T = float(pack.T_window)
    T_seg = float(T_seg)
    n = int(np.floor(T / T_seg * (1.0 + 1e-9)))
    if n < 2:
        raise ValueError(f"a {T * 1e3:.3f} ms walk holds fewer than two windows of {T_seg * 1e3:.3f} ms")
    dropped = T - n * T_seg
    if dropped > float(max_dropped_fraction) * T + 1e-12:
        raise ValueError(f"windows of {T_seg * 1e3:.3f} ms leave {dropped * 1e3:.3f} ms of the {T * 1e3:.3f} ms walk uncovered, "
                         f"more than {max_dropped_fraction:g} of it; give a T_seg that divides it, or a larger max_dropped_fraction")
    env = dict(envelope) if envelope is not None else stated_envelope(pack)
    K = max(2, int(np.ceil(pack.K * T_seg / T)))
    stamp = id or f"{pack.meta.get('id')}/seg{T_seg * 1e3:.0f}ms"
    packs, tried = [], []
    blt_K = path_K = None
    for i in range(n):
        while True:
            pk = window_of_pack(pack, i * T_seg, T_seg, K=K, blt_K=blt_K, path_K=path_K, id=stamp, envelope=env)
            fid = pk.meta["fidelity"]
            if i > 0 or fid["err_reprojection"] <= tol * fid["floor_max"] or K >= int(pack.K):
                break
            tried.append(K)
            K = min(2 * K, int(pack.K))
        if fid["err_reprojection"] > tol * fid["floor_max"]:                 # what the window ADDS to the parent's own terms
            raise ValueError(f"window {i} of {n} re-projected at K = {K} does not reproduce the parent's reading of it within "
                             f"{tol:g} x its Monte-Carlo floor (re-projection error {fid['err_reprojection']:.3g}, floor {fid['floor_max']:.3g})")
        if i == 0:
            c2 = (pk.meta["compression"].get("channels") or {}).get("boundary_local_time")
            blt_K = None if c2 is None else int(c2["K"])
            pw = (pk.meta["compression"].get("channels") or {}).get("susc_field_basis")
            path_K = None if pw is None else int(pw["K"])
        packs.append(pk)
    fid = combine_window_fidelity([pk.meta["fidelity"] for pk in packs])
    wp = pack.meta.get("walk_params", {}) or {}
    arrays, meta = join_windows(packs, fidelity=fid, walks=[dict(first=0, last=n - 1, seed=wp.get("seed"))],
                                 walkers_shuffled=bool(pack.meta.get("compression", {}).get("precision_tiers", {}).get("walkers_shuffled", False)))
    src = pack.source
    parent_sha = sha256_of(src) if src is not None and not str(src).startswith("hf://") else None
    pf = pack.meta.get("fidelity") or {}
    meta["provenance"] = dict(copy.deepcopy(pack.meta.get("provenance") or {}), rewindowed=dict(
        parent_id=pack.meta.get("id"), parent_sha256=parent_sha, parent_digest=(pack.digest if parent_sha is None else None),
        parent_fidelity=dict(err_max=pf.get("err_max"), floor_max=pf.get("floor_max"), certified=pf.get("certified")),
        T_seg=float(T_seg), n_seg=int(n), time_dropped=float(dropped),
        parent_K=int(pack.K), K=int(K), K_tried=tried + [int(K)], blt_K=blt_K, path_K=path_K,
        note="every window the parent's stored functions cut to it and re-projected on its basis; its certificate carries the "
             "re-projection's truncation measured against the parent's exact reading of the window, beside the walk-free terms"))
    meta["id"] = id if id is not None else stamp
    meta["license"], meta["citation"] = pack.license, pack.citation
    out = type(pack)(arrays, meta, source=None)
    if out.has_surface:
        restate_surface_envelope(out, reference=pack, tol=tol)
    save_pack(out, out_path)
    return out


def stated_envelope(pack):
    """The certificate settings a re-encoding of this pack (a window, a prefix) is certified under: the pack's
    :attr:`envelope` on :func:`~dmipy_sim.replay.compression.default_battery` -- its name, band, ``G_max`` and field
    strengths (``acquisition.B0_list``) -- with the refocusing depth its field channel serves
    (``max_refocus_pulses``), its contact envelope as ``rho2_list`` (over the walk's ``D``) and its ``chi_max``."""
    from .compression import default_battery
    e = pack.envelope
    env = dict(default_battery(), name=e.name, band_hz=e.band_hz, G_max=e.G_max)
    if e.B0_list:
        env["B0_list"] = [float(b) for b in e.B0_list]
    pm = ((pack.meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis")
    if pm is not None and pm.get("max_refocus_pulses"):
        env["max_refocus_pulses"] = int(pm["max_refocus_pulses"])
    if pack.rho2_over_D_max is not None and pack.diffusivity:
        env["rho2_list"] = [float(pack.rho2_over_D_max) * float(pack.diffusivity)]
    if pack._envelope_tissue().get("chi_max") is not None:
        env["chi_max"] = float(pack._envelope_tissue()["chi_max"])
    return env


def restate_surface_envelope(pack, *, reference=None, tol=2.0, out_path=None):
    """State the contact tier at the envelope it serves (RPK.md 8.7): per window the largest ``rho2 / D`` at which
    the decoded contact gives a physical attenuation within ``tol`` floors
    (:func:`~dmipy_sim.replay.bank.surface_envelope`), the pack's ``replay_envelope.tissue.rho2_over_D_max`` the
    smallest over the windows, and every window's surface rows (``err_surface``, ``floor_surface``) restated up
    to that edge rather than at the stated envelope's top (the ``rho2_over_D_max`` the pack was built under), its
    maxima re-read and the whole's certificate rebuilt from them. ``reference`` is the one-window pack the windows were encoded from, whose decoded contact
    is then the error's reference; without it the error is the unphysical gain alone. Only metadata changes;
    ``out_path`` writes the pack with its tensors as they are. ``provenance.surface_envelope_restated`` records
    the edges, the battery's top and the reference."""
    import copy
    from .bank import surface_envelope, restate_maxima, combine_window_fidelity
    from dmipy_replay.replay.compression import local_time_at
    if not pack.has_surface:
        raise ValueError("this pack carries no contact channel (C2): there is no surface envelope to state")
    D = pack.diffusivity
    if D is None:
        raise ValueError("the pack records no diffusivity, so rho2 / D cannot be stated")
    if reference is not None and reference.n_windows != 1:
        raise ValueError("the reference is the one-window pack the windows were encoded from")
    hi = pack.rho2_over_D_max
    if hi is None:
        raise ValueError("this pack states no contact envelope (replay_envelope.tissue.rho2_over_D_max), the range its "
                         "certificate was read over: build it under an envelope that states rho2_list")
    hi = float(hi)
    w = np.asarray(pack.spin_weights, np.float64)
    reads = []                                                   # (contact per block, reference sums) per window
    if reference is not None:
        rm = dict((reference.meta["compression"].get("channels") or {})["boundary_local_time"])
        rm.setdefault("K", _cx_bands_K(reference.arrays, rm))
    for win, t0, T_w in pack._windows():
        cm = dict((win.meta["compression"].get("channels") or {})["boundary_local_time"])
        cm.setdefault("K", _cx_bands_K(win.arrays, cm))
        ref = None
        if reference is not None:                                # the parent's contact over the same window, exact
            B = local_time_at(reference.arrays, rm, [t0, t0 + T_w])
            ref = B[:, 1] - B[:, 0]
        B = local_time_at(win.arrays, cm, pack._block_ends(T_w))
        reads.append((np.diff(B, axis=1, prepend=B[:, :1]), ref))
    rows = [surface_envelope(ell, w, rho2_over_D_hi=hi, reference=ref, tol=tol) for ell, ref in reads]
    edge = float(min(r[0] for r in rows))
    restated = [surface_envelope(ell, w, rho2_over_D_hi=edge, reference=ref, tol=np.inf)[1:] if edge > 0 else (0.0, 0.0)
                for ell, ref in reads]
    del reads
    fid = copy.deepcopy(pack.meta.get("fidelity") or {})
    segs = fid.get("windows")
    if segs:
        for f_i, (e, f) in zip(segs, restated):
            f_i.update(err_surface=e, floor_surface=f, surface_rho2_over_D_max=edge)
            restate_maxima(f_i)
        if fid.get("combined") == "bound_over_windows":
            fid = combine_window_fidelity(segs)
        else:
            fid.update(err_surface=float(sum(e for e, _ in restated)), floor_surface=float(max(f for _, f in restated)), windows=segs)
            restate_maxima(fid)
    else:
        fid.update(err_surface=restated[0][0], floor_surface=restated[0][1])
        restate_maxima(fid)
    fid["surface_rho2_over_D_max"] = edge
    pack.meta["fidelity"] = fid
    env = pack.meta.setdefault("replay_envelope", {})
    env.setdefault("tissue", {})["rho2_over_D_max"] = edge
    pack.meta.setdefault("provenance", {})["surface_envelope_restated"] = dict(
        rho2_over_D_max=edge, rho2_max_m_per_s=edge * float(D), D=float(D), per_window=[r[0] for r in rows],
        battery_rho2_over_D_max=hi, tol=float(tol), grid_points=len(np.geomspace(1.0, hi, 256)),
        reference=(None if reference is None else reference.meta.get("id")))
    pack._digest = None
    if out_path is not None:
        save_pack(pack, out_path)
    return edge


def prefix_pack(pack, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
    """This walk cut at the shorter echo time ``TE`` (s) and re-projected on the prefix's own basis, at the same
    bands per second (#199).

    The TE-prefix property (RPK.md 3) makes the walk to ``TE`` a valid walk. Every channel's stored function is
    cut there in closed form (:func:`~dmipy_sim.replay.bank.window_of_pack`: the bridge and the field modes by
    their window maps, the events sliced), so the result is a pack like any other with its own certificate --
    and that certificate is what licenses the prefix: re-pinning the bridge at ``TE`` is a new truncation,
    measured against the parent's exact reading of the window on the fidelity battery
    (``fidelity.err_reprojection``), and a prefix whose re-projection error exceeds ``tol`` times its
    Monte-Carlo floor is refused rather than written. The band starts at the same bands per second,
    ``K' = ceil(K TE / T)``, and is doubled (up to the parent's ``K``) until the certificate passes: a short
    prefix needs more than its share, since the pinned residual's truncation error does not scale with the
    duration. ``K=`` fixes the band instead and is refused as it stands. On a pack of several windows a prefix
    on whole windows is a range of them (:func:`truncate_pack`: the stored bands of those windows, unchanged); a cut inside
    a later window is refused by name, since the window table holds windows of one duration.
    ``provenance.prefix`` records the parent (digest, id, T, K), the cut and every band tried.
    """
    h = float(pack.dt)
    T = float(pack.meta["walk_params"]["T_max"])
    TE = float(TE)
    if TE < 2.0 * h * (1.0 - 1e-9) or TE > T * (1.0 + 1e-9):
        raise ValueError(f"TE = {TE * 1e3:.3f} ms is not a prefix of this {T * 1e3:.3f} ms walk (block {h * 1e6:.1f} us): "
                         f"it needs at least two blocks and at most the walk")
    if pack.n_windows > 1:
        T_w = pack.T_window
        whole = TE / T_w
        if abs(whole - round(whole)) <= 1e-9 * max(whole, 1.0):
            return truncate_pack(pack, int(round(whole)), id=id, out_path=out_path)   # a prefix of whole windows is a range read
        raise ValueError(f"TE = {TE * 1e3:.3f} ms cuts inside window {int(np.floor(whole))} of this pack's {pack.n_windows}: a "
                         "prefix of a window table is a whole number of its windows (truncate_pack), or the prefix of one window")
    return _prefix_within(pack, TE, K=K, out_path=out_path, tol=tol, id=id, provenance=provenance)


def _prefix_within(pack, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
    """:func:`prefix_pack` by re-projection: the stored functions cut at ``TE`` and re-projected on the prefix's basis
    (:func:`~dmipy_sim.replay.bank.window_of_pack` at ``t0 = 0``), as one window of its own duration."""
    from .bank import window_of_pack
    T = float(pack.T_window)
    T_cut = float(TE)
    K_new = int(K) if K is not None else max(2, int(np.ceil(pack.K * T_cut / T)))
    tried = []
    while True:
        prov = dict(provenance or {})
        prov["prefix"] = dict(parent_digest=pack.digest, parent_id=pack.meta.get("id"), parent_T_s=T, parent_K=int(pack.K),
                              TE_s=T_cut, K=K_new, K_tried=tried + [K_new],
                              note="the parent's stored functions cut at TE and re-projected on the prefix's basis; the band "
                                   "starts at the parent's bands per second and doubles until the certificate passes")
        pk = window_of_pack(pack, 0.0, T_cut, K=K_new, id=(id or f"{pack.meta.get('id')}/prefix-{T_cut * 1e3:.0f}ms"), provenance=prov)
        fid = pk.meta.get("fidelity", {})
        # what the prefix ADDS is judged: its re-projection against the parent's exact reading of the window; the
        # walk-free terms the parent carries (its band truncation at the free variances) are inherited, scaled
        if K is not None or fid["err_reprojection"] <= tol * fid["floor_max"] or K_new >= int(pack.K):
            break
        tried.append(K_new)
        K_new = min(2 * K_new, int(pack.K))
    if fid["err_reprojection"] > tol * fid["floor_max"]:
        raise ValueError(f"the prefix at TE = {T_cut * 1e3:.1f} ms with K' = {K_new} does not reproduce the parent's reading of "
                         f"it within {tol:g} x its Monte-Carlo floor (re-projection error {fid.get('err_reprojection'):.3g}, floor "
                         f"{fid.get('floor_max'):.3g}); give a larger K= or keep the parent")
    if out_path is not None:
        write_rpk(out_path, pk.arrays, pk.meta)
        pk.source = out_path
    return pk

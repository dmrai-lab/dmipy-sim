"""The measurements of the walk's band accumulation (``simulate_trajectories``, dmrai-lab/dmipy-sim#741), each a
function the tests read and a table this script prints:

* ``exactness``  -- at ``block_order=0`` the pack of the accumulated walk against the pack of the same walk's block
  ends (its raw subset, every walker; same seed, same steps), on a sphere, a cylinder, a packed cell, a mesh and a
  myelinated cylinder, and the bands against ``sqrt(h) sinc(pi k / 2N)`` times the DST-I of those block ends;
* ``quadrature`` -- the phase variance the block accumulation adds per measurement of the envelope's battery, against
  the same walk accumulated per step (``h = dt_sim``), beside the table ``rel(m, h)`` it is held to, absolutely;
* ``memory``     -- the walk's peak host bytes (VmHWM, read from ``/proc``) over three blocks spanning 16x at a fixed
  ``K``, accumulated alone and keeping every walker's block ends, each in a process of its own; on a GPU also the
  device's peak;
* ``oracle``     -- a sphere and a cylinder replayed from the accumulated pack against the matrix method;
* ``surface``    -- the contact tier (C2) at the signal: surface-relaxation rows at the envelope's largest ``rho2``
  replayed from the accumulated pack, from the pack of the same walk's block ends and by the fused engine.

* ``multiphysics`` -- ONE walk on a myelinated cylinder with C0-C3 (#742), packed walk-free, replayed against the
  matrix method (C0), the fused engine's surface relaxation (C2) and the forward vector-Bloch field (C3);
* ``strands``    -- the field read on the strands fixture (the adaptive producer): the read each pool's pilot derives
  from its measured read series, and the walk at it and at a read once per block against the same walk read every
  round, per channel and at the signal, against the term the certificate charges;
* ``read_check`` -- the same on the myelinated cylinder of ``multiphysics``, against the same walk read every step;
* ``permeable``  -- the event form of C1: a permeable cell's per-pool T2 replayed from the events against the fused engine;
* ``mt``         -- the MT bound column as events: the bound fraction over time against the fused MT engine;
* ``timing``     -- seconds per tier on this device (C0, C0 + C2, C0 + C1 + C2, + C3 at the derived read and at a
  read once per block), the accumulation's and the field read's shares, the device bytes per walker at two walk
  lengths.

    python tools/coefficient_walk_measure.py --what exactness quadrature memory oracle
    python tools/coefficient_walk_measure.py --what multiphysics timing --n 20000 --gpu --json out.json

CPU by default (``JAX_PLATFORMS=cpu`` is set unless the environment says otherwise).
"""
from __future__ import annotations

import argparse
from dmipy_sim.engine.bloch import _simulate_bloch
import json
import os
import subprocess
import sys
import warnings

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def bands(cw):
    """``(n_w, K+2, 3)``: a one-window coefficient walk's coefficients, float64."""
    return np.asarray(cw.position_coeffs, np.float64)[:, 0]


# ----------------------------------------------------------------------------------------------- exactness at m = 0
def _geometries():
    import dmipy_sim as d
    from dmipy_sim.geometry import pack_cylinders
    out = {"sphere": lambda: d.Sphere(radius=4e-6), "cylinder": lambda: d.Cylinder(radius=3e-6, orientation=(0, 0, 1))}

    def packed():
        radii = np.full(4, 3e-6)
        centers, L, _ = pack_cylinders(radii, target_vf=0.30, seed=0)
        return d.PackedCylinders(radii=radii, centers=centers, L=L, orientation=(0.0, 0.0, 1.0))
    out["packed cell"] = packed

    def mesh():
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=2, radius=4e-6)
        return d.Mesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32), feature_radius=1e-6)
    out["mesh"] = mesh
    out["myelinated cylinder"] = lambda: d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), 2e-9, 2e-9)
    return out


def exactness(names=None, *, n=400, T=0.01, block_h=2e-4, K=24, sub_steps=4, seed=11):
    """Per geometry the accumulation at ``block_order=0`` against the same walk's block ends (its raw subset, every
    walker): the start (m) and the displacement relative to its largest value, the bands against the exact relation
    ``beta = sqrt(h) sinc(pi k / 2N) beta_DST-I`` with the orthonormal DST-I of the block ends' bridge, the contact's
    endpoint and bands likewise, whether the occupancy read off the events at the block ends is the raw one, and the
    measured certificate of the pack (the battery replayed on the block ends against the bands)."""
    import dmipy_sim as d
    from scipy.fft import dst
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.replay.compression import read_position_coeffs
    seq = d.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, bvalues=[2e9, 2e9], TE=0.009, slew_rate=np.inf)
    out = {}
    import dataclasses
    for name, make in _geometries().items():
        if names and name not in names:
            continue
        kw = dict(T_max=T, block_h=block_h, seed=seed, require_gpu=False, sub_steps=sub_steps, window_T=T)
        pk = dict(license="x", citation="x", K=K, position_container=None, blt_container=None, blt_temporal_K=K,
                  blt_dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cw = d.simulate_trajectories(n, 2e-9, make(), K=K, block_order=0, raw_subset=n, **kw)
            spec = cw.spec
            if spec.field_source_pools:            # the sheath's field is the C3 tier, not this measurement's: declared inert
                spec = spec.replace(susceptibility={p_.name: None for p_ in spec.field_source_pools},
                                    validity=dataclasses.replace(spec.validity, tiers=[t for t in spec.validity.tiers if t != "field"]))
                cw = dataclasses.replace(cw, spec=spec)
            acc = build_replay_pack(cw, id="exact/acc", fidelity="measured", **pk)
        B = read_position_coeffs(acc.arrays)
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
        X = np.asarray(cw.raw_positions, np.float64)
        N = X.shape[1] - 1
        tau = np.arange(N + 1) / N
        u = X - (X[:, :1] + (X[:, -1:] - X[:, :1]) * tau[None, :, None])
        k = np.arange(1, K + 1)
        exact = np.sqrt(cw.h) * np.sinc(k / (2.0 * N))[None, :, None] * dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
        row = dict(n_blocks=int(cw.n_blocks), start_m=float(np.abs(X[:, 0] - B[:, 0]).max()),
                   displacement_rel=rel(X[:, -1] - X[:, 0], B[:, 1]), bands_vs_block_ends_rel=rel(exact, B[:, 2:]),
                   replay_err=float(acc.fidelity["err_max"]), replay_floor=float(acc.fidelity["floor_max"]))
        if cw.raw_boundary is not None:
            ell = np.asarray(cw.raw_boundary, np.float64)
            Bc = np.cumsum(ell, axis=1)
            uc = Bc - (Bc[:, :1] + (Bc[:, -1:] - Bc[:, :1]) * tau[None, :])
            exact_c = np.sqrt(cw.h) * np.sinc(k / (2.0 * N))[None, :] * dst(uc[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
            row.update(contact_endpoint_rel=rel(Bc[:, -1], np.asarray(cw.boundary_endpoint)[:, 0]),
                       contact_bands_vs_block_ends_rel=rel(exact_c, np.asarray(cw.boundary_bands)[:, 0]),
                       contact_err=float(acc.fidelity.get("err_surface", 0.0)), contact_floor=float(acc.fidelity.get("floor_surface", 0.0)))
        if cw.raw_compartment is not None:
            from dmipy_sim.coefficient_walk import events_at_instants
            lab = events_at_instants(cw.comp_events, cw.compartment, np.arange(N + 1) * cw.h)
            row["occupancy_equal"] = bool(np.array_equal(lab, np.asarray(cw.raw_compartment)))
        out[name] = row
    return out


# ------------------------------------------------------------------------------------------------- the quadrature
def quadrature(geometry, *, D=2.95e-9, T=0.02, dt_sim=1e-6, K=64, cases=((0, 2e-5), (1, 2e-5), (2, 5e-5), (3, 1e-4)),
               n=2000, seed=5, env=None):
    """The phase variance the block accumulation at each ``(m, h)`` of ``cases`` adds per measurement of the battery,
    against the same walk accumulated per step (``h = dt_sim``; the same seed, starts and steps), all read on the
    continuous basis through one projection of the battery. Per case: the largest measured variance over the
    battery, the table's prediction for that measurement (``sum_k gamma^2 W_k^2 rel(m, h, omega_k) var_k`` at the free
    ``D``), the ratio of the two at the measurement the table predicts largest, the absolute bound ``rel(m, h,
    omega_K) 2 b D`` and whether every measurement is within it, the range of measured over predicted across the
    measurements, and the relative error of the top quarter of the modes against the table on the axis where it is
    largest (what must NOT be the criterion on a restricted walk)."""
    import dmipy_sim as d
    from dmipy_sim.constants import GAMMA
    from dmipy_sim.replay import compression as cx
    h_max = max(h for _, h in cases)
    n_t = int(round(T / h_max)) + 1
    G, meta = cx.acquisition_battery(n_t, h_max, env or cx.default_envelope())
    W = cx.band_projection(G, h_max, T, K)                                           # (n_meas, K+2, 3)
    b = np.array([x["b"] for x in meta])
    var = cx.band_variances(K, T, D)
    om = np.pi * np.arange(1, K + 1) / T
    per_band = GAMMA ** 2 * (W[:, 2:, :] ** 2).sum(axis=2) * var[None, :]

    def walk(h, m):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return d.simulate_trajectories(n, D, geometry, T_max=T, block_h=h, seed=seed, require_gpu=False, tiers=(),
                                           sub_steps=int(round(h / dt_sim)), K=K, block_order=m, window_T=T)
    ref_walk = walk(dt_sim, 1)
    ref = bands(ref_walk)
    phi_ref = GAMMA * np.einsum("wkc,mkc->wm", ref, W)
    qv = np.asarray(ref_walk.qv_positions, np.float64)[:, 0].mean(axis=0)                # the measured QV per axis
    per_band_qv = GAMMA ** 2 * np.einsum("mkc,k,c->mk", W[:, 2:, :] ** 2, T / (np.pi * np.arange(1, K + 1)) ** 2, qv)
    top = slice(3 * K // 4, K)
    rows = []
    for m, h in cases:
        C = bands(walk(h, m))
        if not np.array_equal(C[:, :2], ref[:, :2]):
            raise AssertionError("the case and the reference did not walk the same paths")
        meas = np.var(GAMMA * np.einsum("wkc,mkc->wm", C, W) - phi_ref, axis=0)
        pred = (per_band * cx.quadrature_rel(m, h, om)[None, :]).sum(axis=1)
        pred_qv = (per_band_qv * cx.quadrature_rel(m, h, om)[None, :]).sum(axis=1)
        bound = cx.quadrature_rel(m, h, np.pi * K / T) * 2.0 * b * D
        i = int(np.argmax(pred))
        rel_meas = np.var(C[:, 2:, :] - ref[:, 2:, :], axis=0) / np.var(ref[:, 2:, :], axis=0)   # (K, 3)
        rel_tab = cx.quadrature_rel(m, h, om)[:, None]
        live = pred > 1e-6 * pred.max()
        rows.append(dict(m=int(m), h_us=float(h * 1e6), measured_max=float(meas.max()), predicted_at_max=float(pred[int(np.argmax(meas))]),
                         ratio_at_largest_prediction=float(meas[i] / pred[i]), measured_at_largest_prediction=float(meas[i]),
                         predicted_largest=float(pred[i]), bound_max=float(bound.max()), within_bound=bool(np.all(meas <= bound)),
                         worst_fraction_of_bound=float(np.max(meas / np.where(bound > 0, bound, np.inf))),
                         ratio_range=[float((meas[live] / pred[live]).min()), float((meas[live] / pred[live]).max())],
                         qv_over_2DT=[float(x) for x in qv / (2.0 * D * T)],
                         ratio_range_qv=[float((meas[live] / pred_qv[live]).min()), float((meas[live] / pred_qv[live]).max())],
                         relative_vs_table_top_modes=float((rel_meas[top] / rel_tab[top]).mean(axis=0).max())))
    return rows


# -------------------------------------------------------------------------------------------------------- memory
def _probe(mode, block_h, n, T, dt_sim, K):
    """One walk in this process (``mode`` ``"accumulated"``: the bands alone; ``"raw_subset"``: every walker's block ends
    kept beside them, ``raw_subset=n``): VmHWM (from /proc/self/status) over the walk's second call, the high-water
    mark reset after the first (which compiles), and the device's peak bytes when it reports them."""
    import dmipy_sim as d

    def vm(key):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    kw = dict(T_max=T, block_h=block_h, seed=1, require_gpu=False, sub_steps=int(round(block_h / dt_sim)),
              walker_batch_size=n, K=K, raw_subset=(0 if mode == "accumulated" else n))
    geom = d.Cylinder(radius=4e-6, orientation=(0, 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = d.simulate_trajectories(n, 2e-9, geom, **kw)
        del w
        base = vm("VmRSS")
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")                                                          # the high-water mark := now
        w = d.simulate_trajectories(n, 2e-9, geom, **kw)
    out = dict(mode=mode, block_h=block_h, n_blocks=int(w.n_blocks), host_peak_over_base=vm("VmHWM") - base,
               walk_host_bytes=int(w.nbytes))
    try:
        import jax
        st = jax.devices()[0].memory_stats()
        if st and "peak_bytes_in_use" in st:
            out["device_peak_bytes"] = int(st["peak_bytes_in_use"])
    except Exception:
        pass
    print(json.dumps(out))


def memory(*, n=20000, T=0.01, dt_sim=3.125e-6, K=32, block_hs=(5e-4, 1.25e-4, 3.125e-5)):
    """``memory_probe`` in a fresh process per (mode, block): the accumulated walk's peak host bytes against the same
    walk keeping every walker's block ends, over blocks spanning 16x at a fixed ``K`` and a fixed physics step (the
    walkers take the same steps at every block), a cylinder recording every tier."""
    rows = []
    for mode in ("accumulated", "raw_subset"):
        for h in block_hs:
            # one malloc arena: glibc's per-thread arenas hold freed blocks of XLA's worker threads and add tens of MB
            # of noise to the high-water mark, the same for both modes
            r = subprocess.run([sys.executable, __file__, "--memory-probe", mode, repr(h), str(n), repr(T), repr(dt_sim), str(K)],
                               capture_output=True, text=True, env=dict(os.environ, MALLOC_ARENA_MAX="1"), check=True)
            rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return rows


# -------------------------------------------------------------------------------------------------------- the oracle
def oracle(*, n=20000, D=2e-9, T=0.04, K=80, f=400.0, block_h=None, seed=1):
    """A sphere (radius 4 um) and a cylinder (radius 3 um) walked (the band's own block and order unless ``block_h``
    gives the block), packed, and replayed under a cosine OGSE at ``f`` -- a band the pack's ``K`` only just holds --
    against the matrix method
    (``dmipy_sim.math.matrix_method``) on the same waveform. Per pore and measurement: both signals, the difference and
    the standard error of the replay's ensemble mean."""
    import dmipy_sim as d
    from dmipy_sim.math.matrix_method import MatrixPore
    from dmipy_sim.replay.bank import build_replay_pack
    seq = d.ogse([[1, 0, 0], [1, 0, 0]], f, 0.015, shape="cosine", bvalues=[1e9, 2e9], TE=T, n_t=4001, slew_rate=np.inf)
    out = {}
    for name, geom, shape, size in (("sphere", d.Sphere(radius=4e-6), "sphere", 8e-6),
                                    ("cylinder", d.Cylinder(radius=3e-6, orientation=(0, 0, 1)), "cylinder", 6e-6)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cw = d.simulate_trajectories(n, D, geom, T_max=T, block_h=block_h, seed=seed, require_gpu=False, K=K, tiers=(),
                                         window_T=T)
        pk = build_replay_pack(cw, id=f"oracle/{name}", license="x", citation="x")
        w, ew, E = pk.walker_signals(seq)
        S = np.abs((ew[:, None] * E).sum(0) / w.sum())
        se = np.std(np.real(E * np.conj(E.mean(0) / np.abs(E.mean(0)))), axis=0) / np.sqrt(E.shape[0])
        pore = MatrixPore(shape, size, D)
        Sm = np.array([abs(pore.signal(np.asarray(seq.G_eff)[i, :, 0], seq.dt)) for i in range(seq.G_eff.shape[0])])
        out[name] = dict(replay=S.tolist(), matrix_method=Sm.tolist(), diff=(S - Sm).tolist(), se=se.tolist(),
                         band_needed=pk.waveform_band(seq)[1], K=pk.K,
                         block=dict(m=cw.block["m"], h=cw.block["h"], rule=cw.block["rule"]), sub_steps=cw.sub_steps,
                         certificate=dict(err_max=pk.fidelity["err_max"], floor_max=pk.fidelity["floor_max"]))
    return out


def surface(*, n=20000, D=2e-9, T=0.02, K=64, R=4e-6, rho=1e-4, seed=3):
    """The contact tier at the signal (C2): a reflecting sphere of radius ``R`` walked once (the band's own block),
    packed from its accumulation with the measured certificate, read densely on its block ends (every walker kept),
    and walked again by the fused engine with ``surface_relaxivity_t2 = rho`` (the envelope's largest, rho R / D =
    0.2 at the defaults). PGSE rows at b = 0, 1 and 3 ms/um^2 with TE = T (the whole contact, the exact endpoint)
    and TE = T / 2 (the contact gated at the acquisition's end, read from the bands) replayed with ``rho2 = rho``.
    Per row the three signals, the replay's standard error, and the differences accumulated - raw (the codec) and
    accumulated - fused (two walks)."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.spec.tissue import Tissue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = d.Sphere(radius=R)
        cw = d.simulate_trajectories(n, D, g, T_max=T, seed=seed, require_gpu=False, K=K, raw_subset=n, window_T=T)
    pa = build_replay_pack(cw, id="surface/acc", license="x", citation="x", fidelity="measured")
    tissue = Tissue(rho2=rho)
    rows = []
    for TE in (T, T / 2):
        n_t = 401
        seq = d.pgse([[1, 0, 0]] * 3, 0.2 * TE, 0.5 * TE, bvalues=[0.0, 1e9, 3e9], TE=TE, n_t=n_t, slew_rate=np.inf)
        w, ew, E = pa.walker_signals(seq, tissue=tissue)
        S_acc = np.abs((ew[:, None] * E).sum(0) / w.sum())
        se = np.std(np.real((ew[:, None] / w.mean()) * E * np.conj(E.mean(0) / np.abs(E.mean(0)))), axis=0) / np.sqrt(E.shape[0])
        # the dense oracle: the raw block ends' contact gated at the acquisition's end, the raw path's phase
        from dmipy_sim.replay import _replay_kernel as rk
        ell = np.asarray(cw.raw_boundary, np.float64)
        n_s = int(round(TE / cw.h)) + 1
        lw = (rho / D) * ell[:, 1:n_s].sum(1)
        G = rk.effective_gradient(np.asarray(seq.G_eff, np.float64), float(seq.dt), n_s, float(cw.h))
        phi = np.asarray(rk.gradient_phase(G, np.asarray(cw.raw_positions, np.float32)[:, :n_s], float(cw.h)), np.float64)
        S_raw = np.abs((np.exp(lw)[None, :] * np.exp(1j * phi)).mean(1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_fused = np.asarray(d.simulate(n, D, seq, d.Sphere(radius=R, surface_relaxivity_t2=rho), seed=seed + 100,
                                            require_gpu=False), np.float64).reshape(-1)
        rows.append(dict(TE_ms=TE * 1e3, b=[0.0, 1e9, 3e9], accumulated=S_acc.tolist(), raw=np.asarray(S_raw).tolist(),
                         fused=S_fused.tolist(), se=se.tolist(), acc_minus_raw=(S_acc - S_raw).tolist(),
                         acc_minus_fused=(S_acc - S_fused).tolist()))
    return dict(block=dict(m=cw.m, h=cw.h, n_blocks=cw.n_blocks, sub_steps=cw.sub_steps),
                qv_contact_mean=float(np.mean(cw.qv_boundary)), rho_over_D=rho / D,
                quadrature_local_time=pa.fidelity["terms"]["quadrature"].get("local_time"),
                err_surface=pa.fidelity.get("err_surface"), floor_surface=pa.fidelity.get("floor_surface"), rows=rows)



# ------------------------------------------------------------------------------- the multiphysics admission (#742)
def _pose_with_field_along(d_sub):
    """The rotation ``R`` (substrate -> lab) whose pose puts the bore's field along ``d_sub`` in the substrate frame."""
    from dmipy_sim.replay import so3
    d_sub = np.asarray(d_sub, np.float64); d_sub = d_sub / np.linalg.norm(d_sub)
    return np.asarray(so3.rotation_of(d_sub), np.float64).T


def _ensemble(pk, seq, **kw):
    """``(S, se)`` of a replay: the modulus of the weighted ensemble mean (``ReplayPack.replay``) and its standard error
    per measurement over the walkers the knobs select (``compartment=``: the others weigh nothing)."""
    S = np.asarray(pk.replay(seq, **kw), np.float64).reshape(-1)
    w, ew, E = pk.walker_signals(seq, **kw)
    sel = (np.asarray(ew) != 0) & (np.asarray(w) > 0)
    E, ew, w = E[sel], ew[sel], w[sel]
    u = (ew[:, None] * E).sum(0); u = u / np.abs(u)
    se = np.std(np.real((ew[:, None] / w.mean()) * E * np.conj(u)[None, :]), axis=0) / np.sqrt(E.shape[0])
    return S, se


def _myelin(D, r_in, r_out):
    import dmipy_sim as d
    return d.MyelinatedCylinder(r_in, r_out, (0, 0, 1), D, D)              # D_myelin = 0: the sheath's water stuck


def _myelin_field(g, field):
    """The field a myelinated substrate's walk reads: ``"closed"`` the hollow cylinder's closed form
    (:class:`~dmipy_sim.fields.hollow_cylinder.HollowCylinderField`, exact at the walker's position; a cell's with its
    periodic images, the forward Bloch provider's), ``"raster"`` the sheath rasterised by ``field_grid_of``."""
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    if field == "closed":
        return HollowCylinderField.of(g)
    if field == "raster":
        return field_grid_of(g)
    raise ValueError(f"field is 'closed' or 'raster', got {field!r}")


def _shuffled_starts(g, n, seed):
    """``n`` starts of ``g``'s own seeding in a random order: the geometry seeds pool by pool, and a raw subset or a
    pilot of the first walkers would be one pool."""
    import jax
    r0 = np.asarray(g.init_positions(int(n), jax.random.PRNGKey(int(seed))), np.float32)
    return r0[np.random.default_rng(int(seed) + 77).permutation(r0.shape[0])]


def multiphysics(*, n=4000, D=2e-9, T=0.1, K=256, seed=1, B0=7.0, dchi_a=-0.1e-6, rho=None, theta_deg=90.0, n_raw=None,
                 r_in=2e-6, r_out=3e-6, require_gpu=None, field_bits=None, field="closed"):
    """ONE multiphysics walk on a myelinated cylinder (a restricted lumen of radius ``r_in``, a stuck sheath to
    ``r_out`` whose susceptibility is the field source, ``fields.hollow_cylinder``'s closed form as the basis
    (``field="raster"``: the rasterised sheath), the extra space outside, the starts in a random order so that the
    raw subset holds every pool): C0 + C1 (static) + C2 (the contact) + C3 (the field read in the walk at its derived interval),
    ``K`` bands over ``T`` at block order 3 with the derived block, packed walk-free and with the measured certificate
    on its raw subset, and replayed against three oracles:

    * C0 -- the lumen pool against the matrix method (``math.matrix_method``) on a PGSE (0.1 T / 0.4 T, b = 1, 2 ms/um^2)
      and a cosine OGSE at half the band (``0.5 K / (2 T)``: the highest a cosine block's spread spectrum fits in it);
    * C2 -- the lumen pool with ``rho2 = rho`` (the envelope's largest) against the fused engine on the reflecting
      cylinder of the lumen's radius with ``surface_relaxivity_t2 = rho`` (the lumen's physics behind an impermeable
      inner wall), seeded at the SAME lumen starts (``r0``);
    * C3 -- the whole ensemble under a gradient echo and a spin echo at ``B0`` with the field at ``theta_deg`` to the
      axis, ``chi_aniso = dchi_a``: the scalar route against the pack's vector-Bloch route beside the pack's own
      ``_susc_path_bloch_fidelity`` (``err_susc_bloch``); and, since the forward vector-Bloch engine does not step
      the myelinated substrates, the cross-engine oracle on the SAME hollow-cylinder field: the one-axon periodic
      cell's extra-axonal pool walked with the closed form and its periodic images (``n_images=6``), against
      ``_simulate_bloch`` with ``MyelinSusceptibility`` (the same closed form, the same images) on that pool (and,
      with ``field="raster"``, the cell's periodic grid against the forward engine on that grid).

    Every row carries both signals, their difference and the replay's standard error."""
    import dmipy_sim as d
    from dmipy_sim.acquisition.waveforms import rotate_waveform
    from dmipy_sim.engine.pulse_sequence import bare_gradient_echo, bare_spin_echo
    from dmipy_sim.fields.susceptibility import MyelinSusceptibility
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    from dmipy_sim.math.matrix_method import MatrixPore
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.spec.tissue import Tissue
    import time
    n_raw = n if n_raw is None else min(int(n_raw), n)
    g = _myelin(D, r_in, r_out)
    fg = _myelin_field(g, field)
    env = dict(cx.default_envelope(), B0_list=[B0], theta_deg=[float(theta_deg)], max_refocus_pulses=8, rho2_list=[1e-5, 1e-4])
    rho = max(env["rho2_list"]) if rho is None else float(rho)
    env["rho2_list"] = sorted(set(env["rho2_list"]) | {rho})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        w = d.simulate_trajectories(n, D, g, T, seed=seed, require_gpu=require_gpu, K=K, window_T=T, field_basis=fg,
                                    raw_subset=n_raw, envelope=env, r0=_shuffled_starts(g, n, seed))
        t_walk = time.time() - t0
        t0 = time.time()
        bits_kw = {} if field_bits is None else dict(field_container=int(field_bits))   # derived for the floor unless a flat depth is given
        pk = build_replay_pack(w, id="admission/myelin", license="x", citation="x", envelope=env, **bits_kw)
        t_pack = time.time() - t0
        pm = build_replay_pack(w, id="admission/myelin-measured", license="x", citation="x", envelope=env, fidelity="measured", **bits_kw)
    bytes_pack = sum(np.asarray(v).nbytes for v in pk.arrays.values())
    fr = dict(w.block["field_read"]); fb = {k: v for k, v in w.block["field_band"].items() if k != "ladder"}
    out = dict(walk=dict(n=n, n_raw=n_raw, T=T, K=int(w.K), m=int(w.m), h=float(w.h), n_blocks=int(w.n_blocks), sub_steps=int(w.sub_steps),
                         field_container=(int(field_bits) if field_bits is not None else "derived"),
                         dt_sim=float(w.dt_sim), K_c3=int(w.K_c3), field_read=fr, field_band=fb, seconds=t_walk, pack_seconds=t_pack,
                         pack_bytes_per_walker=bytes_pack / n, pools=dict(zip(*[x.tolist() for x in np.unique(w.compartment, return_counts=True)]))),
               certificate=dict(walk_free=dict(terms=pk.fidelity["terms"], err_max=pk.fidelity["err_max"], floor_max=pk.fidelity["floor_max"]),
                                measured={k: pm.fidelity[k] for k in ("err_max", "floor_max", "err_surface", "floor_surface", "err_susc_path",
                                                                       "floor_susc_path", "err_susc_bloch", "floor_susc_bloch") if k in pm.fidelity}))
    # ── C0: the lumen against the matrix method
    f_edge = 0.5 * K / (2.0 * T)                 # a cosine block's spectrum spreads about f: half the band is where it still fits
    sigma = max(1, int(round(f_edge * 0.2 * T))) / f_edge                           # a whole number of periods in a fifth of the walk
    seqs = {f"PGSE {0.1 * T * 1e3:g}/{0.4 * T * 1e3:g} ms, b = 1, 2": d.pgse([[1, 0, 0]] * 2, 0.1 * T, 0.4 * T, bvalues=[1e9, 2e9], TE=T, n_t=4001, slew_rate=np.inf),
            f"OGSE cosine {f_edge:.0f} Hz x {sigma * 1e3:.2f} ms, b = 0.5, 1": d.ogse([[1, 0, 0]] * 2, f_edge, sigma, shape="cosine",
                                                                                     bvalues=[0.5e9, 1e9], TE=T, n_t=8001, slew_rate=np.inf)}
    pore = MatrixPore("cylinder", 2.0 * r_in, D)
    c0 = {}
    for name, seq in seqs.items():
        S, se = _ensemble(pk, seq, compartment=1)
        Sm = np.array([abs(pore.signal(np.asarray(seq.G_eff)[i, :, 0], seq.dt)) for i in range(seq.G_eff.shape[0])])
        c0[name] = dict(replay=S.tolist(), matrix_method=Sm.tolist(), diff=(S - Sm).tolist(), se=se.tolist(),
                        band_needed=int(pk.waveform_band(seq)[1]))
    out["C0_lumen_vs_matrix_method"] = c0
    # ── C2: the lumen's contact at rho against the fused engine seeded at the same lumen starts (every lumen walker's
    # exact start, the coefficients' first entry)
    mask = np.asarray(w.compartment) == 1
    r0 = np.asarray(w.position_coeffs, np.float32)[mask, 0, 0, :]
    seq2 = d.pgse([[1, 0, 0]] * 3, 0.1 * T, 0.4 * T, bvalues=[0.0, 1e9, 2e9], TE=T, n_t=4001, slew_rate=np.inf)
    c2 = {}
    for rho_ in sorted(env["rho2_list"]):
        cell = d.Cylinder(radius=r_in, orientation=(0, 0, 1), surface_relaxivity_t2=rho_)
        S, se = _ensemble(pk, seq2, tissue=Tissue(rho2=rho_), compartment=mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_f = np.asarray(d.simulate(int(r0.shape[0]), D, seq2, cell, seed=seed + 100, r0=r0, require_gpu=require_gpu), np.float64).reshape(-1)
        c2[f"rho2 = {rho_:g} m/s (rho R / D = {rho_ * r_in / D:.2g})"] = dict(rho=rho_, rho_over_D=rho_ / D, b=[0.0, 1e9, 2e9], replay=S.tolist(),
                                                                            fused=S_f.tolist(), diff=(S - S_f).tolist(), se=se.tolist())
    out["C2_lumen_vs_fused"] = dict(n_lumen=int(r0.shape[0]), rows=c2)
    out["field_read_envelope"] = pk.meta["replay_envelope"]["field_read"]
    # ── C3: GRE and SE at B0 with the field at theta to the axis. The forward vector-Bloch engine does not step the
    # myelinated substrates (their kernel is physics.make_myelin_step_fn), so the myelinated cylinder's field tier is
    # held to the pack's own Bloch-route certificate on its raw subset (err_susc_bloch above) and to the scalar route
    # against the pack's vector-Bloch route; the cross-engine oracle is the SAME hollow-cylinder field on the one-axon
    # periodic cell's extra-axonal pool (PackedCylinders, which the forward engine steps), walked with the exact periodic
    # field grid and replayed against _simulate_bloch with MyelinSusceptibility (its images: n_images=2).
    th = np.deg2rad(float(theta_deg))
    d_sub = (np.sin(th), 0.0, np.cos(th))
    R = _pose_with_field_along(d_sub)
    TE3, dt3 = 0.4 * T, 1e-4
    seqs3 = (("GRE", bare_gradient_echo(TE3, dt3)), ("SE", bare_spin_echo(TE3, dt3)))
    kw3 = dict(scanner=B0, tissue=Tissue(chi_aniso=dchi_a), orientation=R)
    c3 = {}
    for name, seq in seqs3:
        seq_lab = rotate_waveform(seq, R)
        S, se = _ensemble(pk, seq_lab, **kw3)
        S_b = np.abs(pk.replay_bloch(seq_lab, complex_signal=True, jax=True, **kw3)).reshape(-1)
        c3[name] = dict(TE=TE3, replay_scalar=S.tolist(), replay_bloch=S_b.tolist(), diff=(S - S_b).tolist(), se=se.tolist())
    out["C3_myelinated_scalar_vs_bloch_route"] = dict(B0=B0, chi_aniso=dchi_a, theta_deg=theta_deg, rows=c3)
    L = 6.0 * r_out
    import dataclasses
    cellp = d.PackedCylinders([r_out], [[0.0, 0.0]], L, orientation=(0.0, 0.0, 1.0), pool="extra")
    cellm = d.PackedMyelinatedCylinders([r_in], [r_in / r_out], [[0.0, 0.0]], L, D_intra=D, D_extra=D, D_myelin=0.0)
    fg2 = _myelin_field(cellm, field)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w2 = d.simulate_trajectories(n, D, cellp, T, seed=seed + 7, require_gpu=require_gpu, K=K, window_T=T, field_basis=fg2,
                                     raw_subset=n_raw, envelope=env)
        w2 = dataclasses.replace(w2, spec=cellm.spec, weights=None)             # the cell's spec names the sheath as the source
        pk2 = build_replay_pack(w2, id="admission/cell-extra", license="x", citation="x", envelope=env, fidelity="measured")
    prov = MyelinSusceptibility(centers=[[0.0, 0.0]], inner_radii=[r_in], outer_radii=[r_out], L=L, delta_chi_a=dchi_a, B0=B0,
                                theta=th, alpha=0.0, periodic=True, n_images=6)
    # the forward engine on the SAME grid the walk read (the 13 channels contracted at this (B0, chi, H)), so the two
    # engines read one field and the comparison is the accumulation and the replay alone; the closed form's periodic
    # images (n_images=6) beside it are the independent physics
    from dmipy_sim.fields.hollow_cylinder import contract
    from dmipy_sim.fields.susceptibility import GridSusceptibility
    prov_grid = prov                                                     # the closed form: the walk read the provider's field
    if field == "raster":
        dB = contract(np.moveaxis(fg2.grids(), 0, -1), d_sub, B0=B0, chi_iso=0.0, chi_aniso=dchi_a)
        prov_grid = GridSusceptibility(dB=dB, origin=np.asarray(fg2.origin, float), voxel_size=np.asarray(fg2.basis["voxel_size"], float), periodic=True)
    c3x = {}
    for name, seq in seqs3:
        seq_lab = rotate_waveform(seq, R)
        S, se = _ensemble(pk2, seq_lab, **kw3)
        S_b = np.abs(pk2.replay_bloch(seq_lab, complex_signal=True, jax=True, **kw3)).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_f = np.abs(np.asarray(_simulate_bloch(n, D, seq, cellp, seed=seed + 200, susceptibility=prov, require_gpu=require_gpu))).reshape(-1)
            S_g = (S_f if prov_grid is prov else
                   np.abs(np.asarray(_simulate_bloch(n, D, seq, cellp, seed=seed + 200, susceptibility=prov_grid, require_gpu=require_gpu))).reshape(-1))
        c3x[name] = dict(TE=TE3, replay_scalar=S.tolist(), replay_bloch=S_b.tolist(), fused_bloch_images=S_f.tolist(), fused_bloch_grid=S_g.tolist(),
                         diff_scalar_grid=(S - S_g).tolist(), diff_bloch_grid=(S_b - S_g).tolist(), diff_scalar_images=(S - S_f).tolist(),
                         se=se.tolist(), z_grid=(np.abs(S - S_g) / (np.sqrt(2.0) * se)).tolist())
    out["field"] = field
    out["raw_subset_pools"] = dict(zip(*[x.tolist() for x in np.unique(np.asarray(w.compartment)[:n_raw], return_counts=True)]))
    out["C3_cell_extra_vs_forward_bloch"] = dict(B0=B0, chi_aniso=dchi_a, theta_deg=theta_deg, cell_m=L, n=n, rows=c3x, field=field,
                                                 walk=dict(K_c3=int(w2.K_c3), field_read=w2.block["field_read"]),
                                                 certificate={k: pk2.fidelity[k] for k in ("err_susc_path", "floor_susc_path", "err_susc_bloch", "floor_susc_bloch") if k in pk2.fidelity})
    return out


def _read_gates(T, n=4001):
    """The gates a read is checked under over a window of ``T``: a gradient echo and a spin echo, ``(2, n, 1)``."""
    t = np.arange(n) * (T / (n - 1))
    return np.stack([np.ones(n), np.where(t < 0.5 * T, 1.0, -1.0)])[:, :, None], T / (n - 1)


def _read_compare(fine, other, *, B0, chi, names, pools=None):
    """The walk ``other`` (read at its interval) against ``fine`` (the same walk read every step or round), per channel
    and pool: the gated integral of every channel ``I_c = int s b_c dt`` from the modes (a gradient echo and a spin
    echo over the window), its difference per walker, the measured constant ``kappa_c = Var(dI_c) / (dt_f^2 <QV_c>)``
    against the constant the certificate charges, and the channel's signal ``|<exp(i gamma B0 chi I_c)>|`` against the
    fine walk's with the channel's term ``(gamma B0 chi dt_f)^2 kappa_c <QV_c> / 2`` (a bound on the bias) and the
    standard error of the difference (the walkers' own spread of it: its first-order, zero-mean part), ``z`` the
    excess of the difference over the term in those errors (within the term when ``z <= 2``).
    ``pools`` is ``{name: (label, dt_f)}`` (a bundle's pools read at their own intervals)."""
    from dmipy_sim.constants import GAMMA
    from dmipy_sim.engine.accumulate import read_constants
    from dmipy_sim.replay.compression import cosine_projection
    if not np.array_equal(np.asarray(other.position_coeffs), np.asarray(fine.position_coeffs)):
        raise AssertionError("the read moved the walk")
    T = float(fine.T_window)
    gates, dtg = _read_gates(T)
    K = int(fine.K_c3)
    g = cosine_projection(gates, dtg, T, K)[:, :, 0]                                   # (2, K)
    I_f = np.einsum("gk,wkc->gwc", g, np.asarray(fine.field_coeffs, np.float64)[:, 0])
    I_o = np.einsum("gk,wkc->gwc", g, np.asarray(other.field_coeffs, np.float64)[:, 0])
    dI = I_o - I_f
    qv = np.asarray(other.qv_field_increments, np.float64)[:, 0]                         # (n_w, n_ch): the charged QV
    fr = other.block["field_read"]
    kappa = read_constants(fr.get("constants"), qv.shape[1])
    lab = np.asarray(other.compartment) if other.compartment is not None else np.zeros(qv.shape[0], int)
    pools = pools or {"all": (None, float(other.field_read_interval))}
    phase = GAMMA * float(B0) * float(chi)
    out = {}
    for pname, (label, dt_f) in pools.items():
        sel = np.ones(qv.shape[0], bool) if label is None else (lab == label)
        if not sel.any():
            continue
        rows = {}
        for c, name in enumerate(names):
            q = float(qv[sel, c].mean())
            if q <= 0.0:
                continue
            k_meas = [float(np.var(dI[gi, sel, c]) / (dt_f ** 2 * q)) for gi in range(2)]
            term = 0.5 * (phase * dt_f) ** 2 * kappa[c] * q
            S_f = [np.exp(1j * phase * I_f[gi, sel, c]) for gi in range(2)]
            S_o = [np.exp(1j * phase * I_o[gi, sel, c]) for gi in range(2)]
            diff = [float(abs(abs(S_o[gi].mean()) - abs(S_f[gi].mean()))) for gi in range(2)]
            u = [S_f[gi].mean() / abs(S_f[gi].mean()) for gi in range(2)]
            se = [float(np.std(np.real((S_o[gi] - S_f[gi]) * np.conj(u[gi]))) / np.sqrt(sel.sum())) for gi in range(2)]
            z = [float((diff[gi] - term) / se[gi]) if se[gi] > 0 else (0.0 if diff[gi] <= term else float("inf")) for gi in range(2)]
            rows[name] = dict(qv_increments=q, kappa_measured_GRE_SE=k_meas, kappa_charged=float(kappa[c]),
                              term_charged=float(term), signal_diff_GRE_SE=diff, se_diff=se, z=z,
                              within_term=bool(max(z) <= 2.0), kappa_within=bool(max(k_meas) <= kappa[c]))
        out[pname] = dict(walkers=int(sel.sum()), dt_f=float(dt_f), channels=rows,
                          all_within_term=bool(all(r["within_term"] for r in rows.values())),
                          all_kappa_within=bool(all(r["kappa_within"] for r in rows.values())),
                          kappa_max=(max(max(r["kappa_measured_GRE_SE"]) for r in rows.values()) if rows else None))
    return out


def _signal_compare(fine, other, seq, kw, env):
    """The whole ensemble's replay of ``seq`` from the packs of ``fine`` and ``other``: ``|S|`` of each, the
    difference, the split-half standard error and the term ``other``'s certificate charges."""
    from dmipy_sim.replay.bank import build_replay_pack
    pf = build_replay_pack(fine, id="read/fine", license="x", citation="x", envelope=env)
    po = build_replay_pack(other, id="read/other", license="x", citation="x", envelope=env)
    S0, se0 = _ensemble(pf, seq, **kw)
    S1, _ = _ensemble(po, seq, **kw)
    _, ew0, E0 = pf.walker_signals(seq, **kw)
    _, ew1, E1 = po.walker_signals(seq, **kw)
    u = (ew0[:, None] * E0).sum(0); u = u / np.abs(u)
    se_d = np.std(np.real((ew1[:, None] * E1 - ew0[:, None] * E0) * np.conj(u)[None, :]), axis=0) / np.sqrt(E0.shape[0])
    term = po.fidelity["terms"]["field_read"]
    z = (np.abs(S1 - S0) - term) / np.where(se_d > 0, se_d, np.inf)
    return dict(S_fine=S0.tolist(), S=S1.tolist(), diff=(S1 - S0).tolist(), se=se0.tolist(), se_diff=se_d.tolist(), z=z.tolist(),
                term_charged=term, term_fine=pf.fidelity["terms"]["field_read"], within_term=bool(np.max(z) <= 2.0),
                env_B0_chi_max_T=po.meta["replay_envelope"]["field_read"].get("B0_chi_max_T"))


def read_check(*, n=20000, D=2e-9, T=0.1, K=256, seed=1, B0=7.0, theta_deg=90.0, r_in=2e-6, r_out=3e-6, require_gpu=None,
               field="closed"):
    """The field read on the myelinated cylinder of :func:`multiphysics` (``K`` bands over ``T`` at block order 3,
    the derived block): the read the pilot derives per pool (the record), the walk at it, and the SAME walk (same
    seed and keys: the read does not touch the path) read every step and once per block (``field_read_interval=``,
    the field band fixed at the derived walk's), each compared with the per-step read per channel and pool
    (:func:`_read_compare`) and at the signal (a gradient echo at ``0.8 T``, ``B0``, the rule's ``chi``, the field at
    ``theta_deg`` to the axis); the seconds of each walk."""
    import time
    import dmipy_sim as d
    from dmipy_sim.acquisition.waveforms import rotate_waveform
    from dmipy_sim.engine.pulse_sequence import bare_gradient_echo
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.spec.tissue import Tissue
    g = _myelin(D, r_in, r_out)
    fg = _myelin_field(g, field)
    env = dict(cx.default_envelope(), B0_list=[B0], theta_deg=[float(theta_deg)])
    secs = {}

    def walk(name, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.time()
            w = d.simulate_trajectories(n, D, g, T, seed=seed, require_gpu=require_gpu, K=K, window_T=T, field_basis=fg,
                                        envelope=env, **kw)
            secs[name] = time.time() - t0
        return w
    derived = walk("derived")
    fr = dict(derived.block["field_read"])
    K_c3 = int(derived.K_c3)
    fine = walk("every step", field_read_interval=float(derived.dt_sim), K_c3=K_c3)
    block = walk("once per block", field_read_interval=float(derived.h), K_c3=K_c3)
    chi = float(fr["chi"])
    names = list(fg.channel_names)
    pools = {name: (lab, None) for lab, name in ((0, "extra"), (1, "intra"), (2, "myelin"))}
    th = np.deg2rad(float(theta_deg)); R = _pose_with_field_along((np.sin(th), 0.0, np.cos(th)))
    seq = rotate_waveform(bare_gradient_echo(0.8 * T, 1e-5), R)
    kw = dict(scanner=B0, tissue=Tissue(chi_iso=chi), orientation=R)
    out = dict(field=field, n=n, T=T, K=int(derived.K), sub_steps=int(derived.sub_steps), dt_sim=float(derived.dt_sim), h=float(derived.h),
               K_c3=K_c3, seconds=secs, field_read={k: v for k, v in fr.items()}, compare={})
    for name, w in (("derived", derived), ("once per block", block)):
        dt_f = float(w.field_read_interval)
        out["compare"][name] = dict(interval_s=dt_f, n_read=int(w.block["field_read"]["n_read"]),
                                    per_channel=_read_compare(fine, w, B0=B0, chi=chi, names=names,
                                                              pools={k: (lab, dt_f) for k, (lab, _) in pools.items()}),
                                    signal=_signal_compare(fine, w, seq, kw, env))
    return out


def _strands_spec(tmp):
    """The three-strand DiSCo fixture of the test suite (a 20 um cell), as a spec."""
    from dmipy_sim.io.strands import write_tck, concat_centerlines
    from dmipy_sim.spec import disco_spec
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = os.path.join(tmp, "t.tck"), os.path.join(tmp, "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def strands(*, n=2000, T=2e-3, block_h=2e-4, K=8, seed=0, B0=7.0, theta_deg=90.0, require_gpu=False):
    """The field read on the strands fixture (the adaptive producer, ``walk_spec(adaptive_steps=True)``): the read
    each pool's pilot derives from its measured read series (the bundle's record), the walk at it, and the SAME walk
    read every round of each pool and once per block (``field_read_interval=``, the band fixed at the derived walk's),
    each compared with the per-round read per channel and pool (:func:`_read_compare`, each pool at its own interval)
    and at the signal (a gradient echo at ``0.8 T``, ``B0``, the rule's ``chi``, the field across the strands)."""
    import tempfile
    from dmipy_sim.engine.pulse_sequence import bare_gradient_echo
    from dmipy_sim.acquisition.waveforms import rotate_waveform
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.spec import walk_spec
    from dmipy_sim.spec.tissue import Tissue
    import time
    spec = _strands_spec(tempfile.mkdtemp())
    env = dict(cx.default_envelope(), B0_list=[B0], theta_deg=[float(theta_deg)])
    secs = {}

    def walk(name, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.time()
            w = walk_spec(spec, n, T, block_h, seed=seed, n_probe=20_000, require_gpu=require_gpu, adaptive_steps=True, K=K,
                          envelope=env, **kw)
            secs[name] = time.time() - t0
        return w
    derived = walk("derived")
    fr = dict(derived.block["field_read"])
    K_c3 = int(derived.K_c3)
    rounds = {name: float(st["round_s"]) for name, st in (derived.stepping.get("pools") or {}).items()}
    fine = walk("every round", field_read_interval=min(rounds.values()), K_c3=K_c3)
    block = walk("once per block", field_read_interval=float(derived.h), K_c3=K_c3)
    chi = float(fr["chi"])
    names = list(derived.field_basis.channel_names)
    ids = {p.name: p.id for p in spec.pools}
    R = _pose_with_field_along((1.0, 0.0, 0.0))
    seq = rotate_waveform(bare_gradient_echo(0.8 * T, 1e-5), R)
    kw = dict(scanner=B0, tissue=Tissue(chi_iso=chi), orientation=R)
    out = dict(n=n, T=T, K=int(derived.K), h=float(derived.h), K_c3=K_c3, rounds_s=rounds, seconds=secs,
               field_read={k: v for k, v in fr.items()}, compare={})
    for name, w in (("derived", derived), ("once per block", block)):
        pr = w.block["field_read"]["pools"]
        out["compare"][name] = dict(pools={p: dict(dt_f=v["dt_f"], n_read=v["n_read"], rule=v["rule"]) for p, v in pr.items()},
                                    per_channel=_read_compare(fine, w, B0=B0, chi=chi, names=names,
                                                              pools={p: (ids[p], float(v["dt_f"])) for p, v in pr.items()}),
                                    signal=_signal_compare(fine, w, seq, kw, env))
    return out


def _permeable_mesh(kappa, T2=None, R=3e-6, L=10e-6, subdivisions=3):
    """A permeable icosphere in a periodic cell: two pools that exchange (intra inside, extra outside)."""
    import trimesh
    import dmipy_sim as d
    from dmipy_sim import Compartments, Pool
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R)
    comps = None if T2 is None else Compartments(intra=Pool(T2=T2[0]), extra=Pool(T2=T2[1]))
    return d.Mesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32), permeability=kappa, compartments=comps,
                  periodic=True, voxel_min=[-L / 2] * 3, voxel_max=[L / 2] * 3, feature_radius=R)


def permeable(*, n=4000, D=2e-9, T=0.025, seed=2, kappa=2e-5, T2=(0.03, 0.08), require_gpu=None):
    """The event form of C1 at the signal: a permeable icosphere (radius 3 um) in a periodic 10 um cell, both pools
    walked once (every walker's block ends kept), packed (events), replayed with per-pool T2 against the fused engine's
    per-pool relaxation on the same mesh (``Mesh(compartments=)``, another walk of the same count; the only fused
    route with per-pool T2 across a permeable wall), and against the dense reading of the walk's own block-end labels;
    first the impermeable cell against the static closed form ``sum_p f_p exp(-TE / T2_p)``, then the permeable one.
    PGSE rows at b = 0, 1 ms/um^2 and TE = T / 2 and T. Per row the signals, the differences and the replay's
    standard error."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.spec.tissue import Tissue
    from dmipy_sim.coefficient_walk import events_at_instants
    T2i, T2e = T2
    out = dict(kappa=kappa, T2=dict(intra=T2i, extra=T2e), cases={})
    tissue = Tissue(T2={"extra": T2e, "intra": T2i})
    for name, k in (("impermeable", 0.0), ("permeable", kappa)):
        plain, cell = _permeable_mesh(k), _permeable_mesh(k, T2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = d.simulate_trajectories(n, D, plain, T, seed=seed, require_gpu=require_gpu, window_T=T, raw_subset=n)
        pk = build_replay_pack(w, id=f"permeable/{name}", license="x", citation="x")
        lab = np.asarray(w.raw_compartment); T2v = np.array([T2e, T2i])
        lab_ev = events_at_instants(w.comp_events, w.compartment, np.arange(w.n_blocks + 1) * w.h) if w.comp_events is not None else np.repeat(np.asarray(w.compartment)[:, None], w.n_blocks + 1, axis=1)
        f_intra = float((np.asarray(w.compartment) == 1).mean())
        rows = []
        for TE in (T / 2, T):
            seq = d.pgse([[1, 0, 0]] * 2, 0.2 * TE, 0.5 * TE, bvalues=[0.0, 1e9], TE=TE, n_t=2001, slew_rate=np.inf)
            S, se = _ensemble(pk, seq, tissue=tissue)
            n_s = int(round(TE / w.h)) + 1
            dense = float(np.exp(-(w.h / T2v[lab[:, 1:n_s]]).sum(1)).mean())                    # the raw block-end labels, b = 0
            static = f_intra * np.exp(-TE / T2i) + (1 - f_intra) * np.exp(-TE / T2e)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S_f = np.asarray(d.simulate(n, D, seq, cell, seed=seed + 100, require_gpu=require_gpu), np.float64).reshape(-1)
            rows.append(dict(TE=TE, b=[0.0, 1e9], replay=S.tolist(), fused=S_f.tolist(), diff=(S - S_f).tolist(), se=se.tolist(),
                             dense_raw_labels_b0=dense, static_b0=float(static)))
        out["cases"][name] = dict(rows=rows, f_intra=f_intra, events=(None if w.comp_events is None else dict(
            n=int(w.comp_events.n_events), per_walker=float(w.comp_events.n_events / n), capacity=w.block.get("events"),
            label_mismatch_at_block_ends=int((lab_ev != lab).sum()))),
            certificate=dict(err_max=pk.fidelity["err_max"], floor_max=pk.fidelity["floor_max"]))
    return out


def mt(*, n=4000, D=2e-9, T=0.05, R=3e-6, seed=3, kappa_MT=2e-5, dwell=2e-3, require_gpu=None):
    """The MT bound column as events: a sphere walked with binding (``simulate_mt_trajectories``), the bound fraction
    over time read off the events at 51 instants against the fused vector-Bloch MT engine's (``_simulate_bloch(...,
    return_bound_frac=True)``, another walk of the same count under a bare gradient echo), and both against the
    two-pool equilibrium ``f_b`` of ``engine.mt``; the largest difference against the binomial standard error."""
    import dmipy_sim as d
    from dmipy_sim.coefficient_walk import events_at_instants
    from dmipy_sim.engine import mt as mtmod
    from dmipy_sim.engine.pulse_sequence import bare_gradient_echo
    g = d.Sphere(R)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = d.simulate_mt_trajectories(n, D, g, T, kappa_MT=kappa_MT, dwell_time=dwell, seed=seed, require_gpu=require_gpu, window_T=T)
    times = np.linspace(0.0, T, 51)
    bf_ev = events_at_instants(w.bound_events, w.bound_start, times).mean(0)
    dt = 1e-4
    seq = bare_gradient_echo(T, dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, bf_f = _simulate_bloch(n, D, seq, g, seed=seed + 100, kappa_MT=kappa_MT, dwell_time=dwell, return_bound_frac=True,
                                   require_gpu=require_gpu)
    bf_f = np.asarray(bf_f, np.float64)
    idx = np.rint(times / dt).astype(int)
    bf_fused = bf_f[np.minimum(idx, bf_f.shape[0] - 1)]
    S_over_V = mtmod.surface_to_volume(g)
    f_b = mtmod.bound_fraction(kappa_MT, dwell, S_over_V)
    se = np.sqrt(np.maximum(bf_ev * (1 - bf_ev), 1e-12) / n)
    return dict(kappa_MT=kappa_MT, dwell=dwell, S_over_V=float(S_over_V), f_b_equilibrium=(None if f_b is None else float(f_b)),
                events=dict(n=int(w.bound_events.n_events), capacity=w.block.get("events")),
                times=times.tolist(), bound_events=bf_ev.tolist(), bound_fused=bf_fused.tolist(),
                diff_max=float(np.abs(bf_ev - bf_fused).max()), se_max=float(se.max()),
                mean_events=float(bf_ev[5:].mean()), mean_fused=float(bf_fused[5:].mean()))


def timing(*, n=20000, D=2e-9, T=0.1, K=256, seed=1, reps=2, require_gpu=None, lengths=(0.1, 1.0)):
    """The walk's seconds per tier on this device at ``n`` walkers over ``T`` at ``K``, each the best of ``reps``
    after one compiling call: C0 (positions alone) on the lumen's own cylinder (the myelinated substrates are
    stepped by their kernel with every channel recorded, so a positions-only walk of one does not exist), C0 + C2
    on that cylinder (the contact and the static occupancy), C0 + C2 on the myelinated cylinder of
    :func:`multiphysics` (the three-pool kernel), and C0 + C2 + C3 on it (the field read in the walk at its derived
    interval). The shares: the accumulation's (C0 at ``K`` against the same walk at ``K = 1``, block order 0: the
    cheapest accumulation), the contact's (C0 + C2 against C0, the cylinder) and the field read's (C3 against C2, the
    myelinated cylinder). The device bytes per walker the resolver states at two walk lengths, both substrates."""
    import dmipy_sim as d
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    import time
    r_in = 2e-6
    cyl = d.Cylinder(radius=r_in, orientation=(0, 0, 1))
    my = _myelin(D, r_in, 3e-6)
    fg = _myelin_field(my, "closed")

    def best(g, **kw):
        ts = []
        for i in range(reps + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.time(); w = d.simulate_trajectories(n, D, g, T, seed=seed, require_gpu=require_gpu, window_T=T, **kw); ts.append(time.time() - t0)
        return min(ts[1:]), w

    rows, walks = {}, {}
    rows["cylinder C0 minimal (K=1, m=0)"], _ = best(cyl, K=1, block_order=0, tiers=())
    rows["cylinder C0"], walks["cylinder C0"] = best(cyl, K=K, tiers=())
    rows["cylinder C0+C2"], walks["cylinder C0+C2"] = best(cyl, K=K)
    rows["myelinated C0+C2"], walks["myelinated C0+C2"] = best(my, K=K)
    rows["myelinated C0+C2+C3"], walks["myelinated C0+C2+C3"] = best(my, K=K, field_basis=fg)
    w3 = walks["myelinated C0+C2+C3"]
    # one raster row: the sheath rasterised, its trilinear read smearing the jump over a voxel
    rows["myelinated C0+C2+C3, raster"], walks["myelinated C0+C2+C3, raster"] = best(my, K=K, field_basis=_myelin_field(my, "raster"))
    pilot_s = {k: float(w.block["field_read"].get("pilot_seconds", 0.0)) for k, w in walks.items() if w.block.get("field_read")}
    shares = dict(accumulation=1.0 - rows["cylinder C0 minimal (K=1, m=0)"] / rows["cylinder C0"],
                  contact=1.0 - rows["cylinder C0"] / rows["cylinder C0+C2"],
                  field_read=1.0 - rows["myelinated C0+C2"] / rows["myelinated C0+C2+C3"],
                  field_read_without_pilot=1.0 - rows["myelinated C0+C2"] / (rows["myelinated C0+C2+C3"] - pilot_s["myelinated C0+C2+C3"]),
                  pilot=pilot_s["myelinated C0+C2+C3"] / rows["myelinated C0+C2+C3"],
                  field_read_raster=1.0 - rows["myelinated C0+C2"] / rows["myelinated C0+C2+C3, raster"],
                  field_read_raster_without_pilot=1.0 - rows["myelinated C0+C2"] / (
                      rows["myelinated C0+C2+C3, raster"] - pilot_s["myelinated C0+C2+C3, raster"]))
    cost = {}
    for L in lengths:
        for name, g, kw in (("cylinder C0", cyl, dict(tiers=())), ("cylinder C0+C2", cyl, {}), ("myelinated C0+C2+C3", my, dict(field_basis=fg))):
            # the same band per storage window (window_T = T): a longer walk is more windows, never a wider band
            c = d.walk_cost(n, D, g, L, K=K, window_T=T, **kw)
            cost[f"{name} @ {L:g} s"] = dict(bytes_per_walker=int(c.bytes_per_walker), accumulator_bytes_per_walker=int(c.accumulator_bytes_per_walker),
                                             K=int(c.K), n_windows=int(round(L / T)), n_blocks=int(c.n_blocks), block_s=float(c.block_s), sub_steps=int(c.sub_steps))
    try:
        import jax
        dev = str(jax.devices()[0])
    except Exception:
        dev = "?"
    return dict(device=dev, n=n, T=T, K=K, seconds=rows, pilot_seconds=pilot_s, walker_steps_per_second={k: n * int(walks.get(k, w3).sub_steps) * int(walks.get(k, w3).n_blocks) / v
                                                                                   for k, v in rows.items() if k in walks},
                shares=shares, steps=dict(cylinder=dict(sub_steps=int(walks["cylinder C0"].sub_steps), dt_sim=float(walks["cylinder C0"].dt_sim)),
                                          myelinated=dict(sub_steps=int(w3.sub_steps), dt_sim=float(w3.dt_sim), K_c3=int(w3.K_c3),
                                                          field_read={k: v for k, v in w3.block["field_read"].items()
                                                                      if k not in ("measured_constants",)},
                                                          raster_field_read={k: v for k, v in walks["myelinated C0+C2+C3, raster"].block["field_read"].items()
                                                                             if k not in ("measured_constants",)})), bytes=cost)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--what", nargs="+", default=["exactness", "quadrature", "memory", "oracle", "surface"],
                    choices=["exactness", "quadrature", "memory", "oracle", "surface", "multiphysics", "strands", "permeable", "mt", "timing",
                             "read_check"])
    ap.add_argument("--memory-probe", nargs=6, metavar=("MODE", "BLOCK_H", "N", "T", "DT_SIM", "K"))
    ap.add_argument("--n", type=int, default=None, help="walkers for the admissions and the timing")
    ap.add_argument("--T", type=float, default=None, help="walk duration (s) for the admissions and the timing")
    ap.add_argument("--K", type=int, default=None, help="bands for the admissions and the timing")
    ap.add_argument("--n-raw", type=int, default=None, help="the raw subset of the multiphysics walk (its measured certificate)")
    ap.add_argument("--bits", type=int, default=None, help="the multiphysics pack's field-mode depth (8 | 16), derived for the floor unless given")
    ap.add_argument("--gpu", action="store_true", help="require the JAX device to be a GPU (the timing table)")
    ap.add_argument("--field", default=None, choices=["closed", "raster"], help="the myelin field multiphysics / read_check read")
    ap.add_argument("--json", default=None, help="write every result as JSON here")
    a = ap.parse_args(argv)
    if a.memory_probe:
        mode, h, n, T, dt_sim, K = a.memory_probe
        _probe(mode, float(h), int(n), float(T), float(dt_sim), int(K))
        return
    kw = {k: v for k, v in dict(n=a.n, T=a.T, K=a.K).items() if v is not None}
    out = {}
    for what in a.what:
        fn = globals()[what]
        extra = dict(kw)
        if what in ("multiphysics", "permeable", "mt", "timing", "read_check", "strands") and a.gpu:
            extra["require_gpu"] = True
        if what == "multiphysics" and a.n_raw is not None:
            extra["n_raw"] = a.n_raw
        if what in ("multiphysics", "read_check") and a.field is not None:
            extra["field"] = a.field
        if what == "multiphysics" and a.bits is not None:
            extra["field_bits"] = a.bits
        if what in ("exactness", "quadrature", "memory", "oracle", "surface"):
            extra = {k: v for k, v in kw.items() if k in fn.__code__.co_varnames}
        if what == "quadrature":
            import dmipy_sim as d
            out[what] = fn(d.FreeDiffusion(), **extra)
        elif what == "strands":
            out[what] = fn(**{k: v for k, v in extra.items() if k not in ("T", "K")})        # the fixture's own window and band
        else:
            out[what] = fn(**extra)
        print(f"== {what}"); print(json.dumps(out[what], indent=1, default=float))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()

"""The parity fixtures as published replay packs: one certified pack per fixture, to SubstrateCommons.

The fixtures, the walks, the references and the floors are
``examples/validation/cross_engine_parity.py`` in dmipy-sim; this script is the pack builder and the hub
path, and nothing else. Two published Monte-Carlo references:

* **MC/DC** (Rafael-Patino et al. 2020, Front. Neuroinform. 14:8, LGPL-2.1,
  https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations)
* **Disimpy / MISST** (Kerkelae et al. 2020, JOSS 5(52):2527, MIT, https://github.com/kerkelae/disimpy)

Stages, in order (dmrai-lab/dmipy-sim#482); each writes ``records/<stage>.json`` and reads only the record
before it, and the gate reads recorded numbers and never walks:

    python examples/substrate_bank/build_parity_fixtures.py --stage source    --mcdc-data DIR --disimpy-data DIR
    python examples/substrate_bank/build_parity_fixtures.py --stage reference --mcdc-data DIR --disimpy-data DIR
    python examples/substrate_bank/build_parity_fixtures.py --stage design    --mcdc-data DIR --require-gpu
    python examples/substrate_bank/build_parity_fixtures.py --mcdc-data DIR --n-walkers 100000 --K 64  # the walks
    python examples/substrate_bank/build_parity_fixtures.py --stage build     --mcdc-data DIR
    python examples/substrate_bank/build_parity_fixtures.py --stage gate      --mcdc-data DIR
    python examples/substrate_bank/build_parity_fixtures.py --publish-only
"""
import os
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/hf-session"))
import argparse
import json
import logging
import subprocess
import sys
import time

import numpy as np

#: Where the packs and the protocol's records live. Outside the repository by default -- an 82 MB pack is not
#: a source file -- overridden with ``--work``; the records ARE the family's report and are meant to be kept.
HERE = os.environ.get("DMIPY_SIM_PARITY_WORK") or os.path.join(os.path.expanduser("~"), "dmrai-ws",
                                                               "parity-fixtures")
REPO = "SubstrateCommons/parity-fixtures"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def code_commit(*, require_clean=True):
    """The commit of the code that produces the pack, or a refusal.

    ``git rev-parse HEAD`` alone is a lie on a dirty tree: the three packs already published carry
    `30ad9b9a` (main), which contains none of `io/mcdc.py`, the producer or the mesh fix, because the run that
    built them read the HEAD of a tree whose changes were not committed. A manifest row that names a commit
    the code was not at is worse than no row.

    So: refused unless the working tree is clean AND its HEAD exists on a remote. ``require_clean=False`` is
    for the pilot and the design stage, which record a commit for information and publish nothing.
    """
    import dmipy_sim
    root = os.path.dirname(os.path.dirname(os.path.abspath(dmipy_sim.__file__)))

    def git(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()

    rc, head = git("rev-parse", "HEAD")
    if rc:
        if require_clean:
            raise SystemExit(f"{root} is not a git checkout, so the code that built these packs cannot be "
                             f"named; publishing would put an unverifiable commit in the manifest")
        return None
    rc, dirty = git("status", "--porcelain", "--untracked-files=no")
    rc2, contains = git("branch", "--remotes", "--contains", head)
    if not require_clean:
        return head
    if dirty:
        raise SystemExit(f"{root} has uncommitted changes to tracked files:\n{dirty}\n"
                         f"HEAD {head[:12]} does not describe the code in this tree. Commit them, push, and "
                         f"publish again -- a manifest row must name the code that produced the pack.")
    if rc2 or not contains:
        raise SystemExit(f"{root} is at {head[:12]}, which is on no remote branch. Push it before publishing: "
                         f"a manifest row that names a commit nobody can fetch is not provenance.")
    return head


def fixtures(sim_root=None):
    """``examples.validation.cross_engine_parity``: the one place the fixtures, the walks, the references, the
    floors and the gate's threshold derivation are defined. This script lives beside it in the same checkout,
    so ``sim_root`` defaults to that checkout and only needs giving when the module is elsewhere."""
    root = os.path.abspath(sim_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    sys.path.insert(0, root)
    os.environ.setdefault("DMIPY_SIM_ROOT", root)
    import importlib
    return importlib.import_module("examples.validation.cross_engine_parity")


# ----------------------------------------------------------------------------------------------- the pilot
def pilot(a, X):
    """Measure what the family's parameters must be, on a short walk of each fixture: the envelope floor at the pilot
    count, the walker count that floor implies, K against the fixture's own scheme, the timing and the memory.
    Publishes nothing."""
    from dmipy_sim.replay.bank import _measure_floor, _master_arrays, build_replay_pack
    from dmipy_sim.io import mcdc
    import tempfile
    if a.mcdc_data:
        amp, wL = X.MCDC_FIXTURES[0]
        seq = mcdc.read_scheme(os.path.join(a.mcdc_data, "Simulator-Conf-files", X.MCDC_SCHEME), n_t=X.MCDC_N_T)
        log(f"MC/DC pilot amp {amp} wL {wL}: N={a.pilot_n}, {X.MCDC_N_T} saves of "
            f"{X.MCDC_TE / (X.MCDC_N_T - 1) * 1e6:.1f} us over TE {X.MCDC_TE * 1e3:.2f} ms")
        ref, n_theirs = X.mcdc_reference(a.mcdc_data, amp, wL, seq)
        log(f"  their released signal: {len(ref)} measurements, N = {n_theirs:,}")
        t0 = time.time()
        walk = X.mcdc_walk(a.mcdc_data, amp, wL, a.pilot_n, require_gpu=a.require_gpu)
        log(f"  walked in {time.time() - t0:.0f} s; sub_steps {walk.sub_steps} (dt_sim "
            f"{walk.dt_sim * 1e9:.0f} ns, step {np.sqrt(6 * X.MCDC_D * walk.dt_sim) * 1e9:.0f} nm); "
            f"refused {walk.illegal_crossings}; RSS {X.peak_rss_gb():.1f} GB")
        log(f"  MC/DC's own step: dt {X.MCDC_TE / 5000 * 1e9:.0f} ns (T = 5000), "
            f"step {np.sqrt(6 * X.MCDC_D * X.MCDC_TE / 5000) * 1e9:.0f} nm")
        cos = X.walk_cos_phi(walk, seq)
        p = X.parity(cos, ref, n_theirs)
        log(f"  parity: max|dS| {p['max_abs_diff']:.5f} at m={p['at_measurement']} "
            f"(ours {p['ours_at']:.5f} theirs {p['theirs_at']:.5f}); rms {p['rms_diff']:.5f}; "
            f"our floor {p['floor_ours_max']:.5f} theirs {p['floor_theirs_max']:.5f} "
            f"-> {p['worst_in_tolerance_units']:.2f} x tolerance")
        # MC/DC walked this substrate in T = 5000 steps of 10.7 us, a 196 nm step against a 500 nm lumen
        # radius (step / R = 0.39). Our sub-step rule takes a far finer one, so a difference between the two
        # signals is either the engines or that step. The same walk at THEIR step separates the two.
        sub_their = max(1, int(round((X.MCDC_TE / (X.MCDC_N_T - 1)) / (X.MCDC_TE / 5000))))
        t1 = time.time()
        walk_their = X.mcdc_walk(a.mcdc_data, amp, wL, a.pilot_n, require_gpu=a.require_gpu, sub_steps=sub_their)
        pt = X.parity(X.walk_cos_phi(walk_their, seq), ref, n_theirs)
        log(f"  at MC/DC's OWN step (sub_steps {sub_their}, dt_sim {walk_their.dt_sim * 1e9:.0f} ns, step "
            f"{np.sqrt(6 * X.MCDC_D * walk_their.dt_sim) * 1e9:.0f} nm, {time.time() - t1:.0f} s): "
            f"max|dS| {pt['max_abs_diff']:.5f} rms {pt['rms_diff']:.5f} "
            f"-> {pt['worst_in_tolerance_units']:.2f} x tolerance; refused {walk_their.illegal_crossings}")
        del walk_their
        f0 = _measure_floor(_master_arrays(walk), X.mcdc_envelope())
        log(f"  envelope floor at N={a.pilot_n}: {f0:.5f} -> N* ~ "
            f"{int(round(a.pilot_n * (f0 / X.SIGMA) ** 2 * 1.4)):,} for sigma* {X.SIGMA}")
        tmp = tempfile.mkdtemp()
        for K in (32, 64, 128, 256):
            pk = build_replay_pack(walk, id="pilot/mcdc", license=X.MCDC_LICENSE, citation="pilot", K=K,
                                   envelope=X.mcdc_envelope(), segment_T=X.MCDC_TE, verbose=False, device=a.device)
            fl = pk.meta["fidelity"]
            pc = X.parity(X.pack_cos_phi(pk, seq), ref, n_theirs)
            f = os.path.join(tmp, f"mcdc_K{K}.rpk"); pk.save(f); mb = os.path.getsize(f) / 1e6; os.remove(f)
            log(f"    K={K:4d} band {pk.temporal_bandwidth_hz:8.0f} Hz  codec err {fl['err_max']:.5f} "
                f"floor {fl['floor_max']:.5f}  pack parity {pc['max_abs_diff']:.5f}  {mb:6.1f} MB "
                f"band-ok {pk.waveform_band(seq)[0] <= pk.temporal_bandwidth_hz}")
    if a.disimpy_data:
        log(f"Disimpy pilot: N={a.pilot_n}, {X.DISIMPY_N_T} saves of "
            f"{X.DISIMPY_TE / (X.DISIMPY_N_T - 1) * 1e6:.0f} us over TE {X.DISIMPY_TE * 1e3:.0f} ms")
        ref = X.disimpy_reference(a.disimpy_data)
        t0 = time.time()
        walk = X.disimpy_walk(a.disimpy_data, a.pilot_n, require_gpu=a.require_gpu)
        log(f"  walked in {time.time() - t0:.0f} s; sub_steps {walk.sub_steps}; "
            f"refused {walk.illegal_crossings}; RSS {X.peak_rss_gb():.1f} GB")
        seq = X.disimpy_sequence()
        p = X.parity(X.walk_cos_phi(walk, seq), ref)
        log(f"  parity vs MISST (delta 30 / Delta 40 ms): max|dS| {p['max_abs_diff']:.5f} at "
            f"m={p['at_measurement']} (ours {p['ours_at']:.5f} MISST {p['theirs_at']:.5f}); "
            f"our floor {p['floor_ours_max']:.5f} -> {p['worst_in_tolerance_units']:.2f} x tolerance")
        seq_fix = X.disimpy_sequence(delta=299 * X.DISIMPY_TE / 699, Delta=399 * X.DISIMPY_TE / 699, n_t=X.DISIMPY_N_T)
        p2 = X.parity(X.walk_cos_phi(walk, seq_fix), ref)
        log(f"  the fixture's OWN array timing (29.943 / 39.957 ms): max|dS| {p2['max_abs_diff']:.5f} "
            f"-- the shift a 700-sample 70 ms grid puts on MISST's 30 / 40 ms")
        f0 = _measure_floor(_master_arrays(walk), X.disimpy_envelope())
        log(f"  envelope floor at N={a.pilot_n}: {f0:.5f} -> N* ~ "
            f"{int(round(a.pilot_n * (f0 / X.SIGMA) ** 2 * 1.4)):,} for sigma* {X.SIGMA}")
        tmp = tempfile.mkdtemp()
        for K in (32, 64, 128):
            pk = build_replay_pack(walk, id="pilot/disimpy", license=X.DISIMPY_LICENSE, citation="pilot", K=K,
                                   envelope=X.disimpy_envelope(), segment_T=X.DISIMPY_TE, verbose=False, device=a.device)
            fl = pk.meta["fidelity"]
            pc = X.parity(X.pack_cos_phi(pk, seq), ref)
            f = os.path.join(tmp, f"dis_K{K}.rpk"); pk.save(f); mb = os.path.getsize(f) / 1e6; os.remove(f)
            log(f"    K={K:4d} band {pk.temporal_bandwidth_hz:8.0f} Hz  codec err {fl['err_max']:.5f} "
                f"floor {fl['floor_max']:.5f}  pack parity {pc['max_abs_diff']:.5f}  {mb:6.1f} MB "
                f"band-ok {pk.waveform_band(seq)[0] <= pk.temporal_bandwidth_hz}")
    log(f"pilot done; peak RSS {X.peak_rss_gb():.1f} GB")


# ------------------------------------------------------------------------------------ the family and the hub
def family(a, X):
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.replay.publish import publish
    from dmipy_sim.io import mcdc
    out = os.path.join(HERE, "packs"); os.makedirs(out, exist_ok=True)
    status = os.path.join(HERE, "status.jsonl")
    commit = code_commit()
    jobs = []
    if a.mcdc_data:
        jobs += [("mcdc", amp, wL) for amp, wL in X.MCDC_FIXTURES]
    if a.disimpy_data:
        jobs += [("disimpy", None, None)]
    rows, sources, published = [], [], []
    for kind, amp, wL in jobs:
        t0 = time.time()
        if kind == "mcdc":
            name = f"mcdc-{amp}-{wL}"
            seq = mcdc.read_scheme(os.path.join(a.mcdc_data, "Simulator-Conf-files", X.MCDC_SCHEME), n_t=X.MCDC_N_T)
            ref, n_theirs = X.mcdc_reference(a.mcdc_data, amp, wL, seq)
            walk = X.mcdc_walk(a.mcdc_data, amp, wL, a.n_walkers, require_gpu=a.require_gpu)
            env, lic, cit, seg = X.mcdc_envelope(), X.MCDC_LICENSE, X.MCDC_CITATION, X.MCDC_TE
            pid = f"parity-fixtures/mcdc-uAxon_d_1.0_amp_{amp}_wL_{wL}"
            prov = {"family": "parity-fixtures", "engine_compared": "MC/DC (Rafael-Patino et al. 2020)",
                    "reference_signal": os.path.relpath(X.mcdc_paths(a.mcdc_data, amp, wL)["dwi"], a.mcdc_data),
                    "reference_walkers": n_theirs, "reference_steps": 5000,
                    "reference_repo": "https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations",
                    "reference_commit": a.mcdc_commit, "scheme": X.MCDC_SCHEME,
                    "acquisition": f"ActiveAx {seq.G.shape[0]} measurements, TE {X.MCDC_TE * 1e3:.2f} ms, 4 shells",
                    "diffusivity": X.MCDC_D, "window_s": X.MCDC_TE, "save_interval_s": X.MCDC_TE / (X.MCDC_N_T - 1),
                    "relaxation": "not in the walk: MC/DC's walls are reflecting; rho and T2 are replay knobs",
                    "code_commit": commit}
        else:
            name = "disimpy-cylinder"
            seq = X.disimpy_sequence()
            ref, n_theirs = X.disimpy_reference(a.disimpy_data), None
            walk = X.disimpy_walk(a.disimpy_data, (a.disimpy_n_walkers or a.n_walkers),
                                  require_gpu=a.require_gpu, sub_steps=a.disimpy_sub_steps)
            env, lic, cit, seg = X.disimpy_envelope(), X.DISIMPY_LICENSE, X.DISIMPY_CITATION, X.DISIMPY_TE
            pid = "parity-fixtures/disimpy-cylinder_mesh_closed"
            prov = {"family": "parity-fixtures", "engine_compared": "MISST (exact), via Disimpy's fixture",
                    "reference_signal": "disimpy/tests/misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
                    "reference_walkers": None, "reference_repo": "https://github.com/kerkelae/disimpy",
                    "reference_commit": a.disimpy_commit,
                    "acquisition": "PGSE along x, delta 30 ms, Delta 40 ms, TE 70 ms, 100 b to 3e9 s/m^2",
                    "diffusivity": X.DISIMPY_D, "window_s": X.DISIMPY_TE,
                    "save_interval_s": X.DISIMPY_TE / (X.DISIMPY_N_T - 1),
                    "relaxation": "not in the walk: rho and T2 are replay knobs", "code_commit": commit}
        log(f"{name}: walked N={a.n_walkers:,} in {time.time() - t0:.0f} s; sub_steps {walk.sub_steps}; "
            f"refused {walk.illegal_crossings}; RSS {X.peak_rss_gb():.1f} GB")
        cos = X.walk_cos_phi(walk, seq)
        p_walk = X.parity(cos, ref, n_theirs)
        pk = build_replay_pack(walk, id=pid, license=lic, citation=cit, K=a.K, envelope=env, segment_T=seg,
                               sigma_star=X.SIGMA, err_target=X.SIGMA, provenance=prov, verbose=False, device=a.device)
        p_pack = X.parity(X.pack_cos_phi(pk, seq), ref, n_theirs)
        band_hz, _ = pk.waveform_band(seq)[0], None
        fl = pk.meta["fidelity"]
        pk.meta.setdefault("provenance", {})["parity"] = dict(
            walk=p_walk, pack=p_pack, sigma_star=X.SIGMA, n_walkers=int(pk.n_walkers),
            envelope_band_hz=float(band_hz), pack_band_hz=float(pk.temporal_bandwidth_hz),
            in_band=bool(band_hz <= pk.temporal_bandwidth_hz))
        local = os.path.join(out, f"{name}.rpk")
        pk.save(local)
        rec = dict(name=name, id=pid, K=int(pk.meta["compression"].get("K") or a.K or 0),
                   n_walkers=int(pk.n_walkers), n_t=int(pk.n_t), dt=float(pk.dt),
                   sub_steps=int(walk.sub_steps), illegal_crossings=int(walk.illegal_crossings),
                   floor=float(fl["floor_max"]), err=float(fl["err_max"]),
                   band_hz=float(pk.temporal_bandwidth_hz), needs_hz=float(band_hz),
                   parity_walk=p_walk, parity_pack=p_pack, bytes=os.path.getsize(local),
                   seconds=round(time.time() - t0, 1), peak_rss_gb=round(X.peak_rss_gb(), 1),
                   code_commit=commit)
        log(f"{name}: K {rec['K']} floor {rec['floor']:.5f} err {rec['err']:.5f} band {rec['band_hz']:.0f} Hz "
            f"(needs {rec['needs_hz']:.0f}) parity walk {p_walk['max_abs_diff']:.5f} pack "
            f"{p_pack['max_abs_diff']:.5f} tol {p_walk['tol_max']:.5f} {rec['bytes'] / 1e6:.1f} MB "
            f"{rec['seconds']:.0f} s")
        if band_hz > pk.temporal_bandwidth_hz:
            log(f"{name}: the fixture's own scheme needs {band_hz:.0f} Hz and the pack stores "
                f"{pk.temporal_bandwidth_hz:.0f} Hz -- NOT published"); continue
        if not a.dry:
            rec["uri"] = publish(local, a.repo, path=f"packs/{name}.rpk",
                                 message=f"parity fixtures: {name} ({rec['n_walkers']:,} walkers, K={rec['K']})")
            log(f"  {rec['uri']}")
        with open(status, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


FAMILY_README = """
## What these packs are

**Cross-engine parity fixtures.** Each pack is one Monte-Carlo walk of dmipy-sim through a substrate that
another published simulator already walked, at that simulator's own acquisition, and it carries the parity
number -- our signal against theirs, measurement by measurement, against the Monte-Carlo floors of BOTH runs.

| fixture | geometry | reference signal | licence | citation |
|---|---|---|---|---|
{rows}

**{n_packs} pack{plural} here.** Each row of the table above is one axon and carries its own substrate spec
(`pack.substrate`): the mesh it was walked on, that mesh's own bounding box, and the digest of the PLY it came
from. They are different axons, not one substrate in three versions.

**Relaxation is not in any of these walks.** The walls are purely reflecting, as the reference engines' were,
and each pack stores the boundary local time (C2) and the compartment occupancy (C1) as channels, so `rho`,
`T2` and `T1` are replay knobs.

**What each pack is certified for** is its own fixture's acquisition and nothing wider: the four ActiveAx
shells of `ActiveAxG140_PM.scheme` (b = 1925 / 1932 / 3094 / 13190 s/mm^2, delta 7.62-17.74 ms,
Delta 16.70-45.90 ms, TE 53.52 ms). A gradient beyond the stored band is refused by `waveform_band` rather
than served a wrong number.

## Use me

Reproduce the parity number of a pack in one call:

```python
from dmipy_sim.replay import ReplayPack
from dmipy_sim.io import mcdc
import numpy as np

pack = ReplayPack.load("hf://{repo}/packs/mcdc-1.0-12.0.rpk")
seq  = mcdc.read_scheme("ActiveAxG140_PM.scheme", n_t=2677)          # their scheme, from their repository
theirs = mcdc.read_bfloat("uAxon_d_1.0_amp_1.0_wL_12.0_DWI.bfloat", n_measurements=372)
theirs = theirs / mcdc.walker_count(theirs, seq.b())                  # their b = 0 entry IS their walker count

ours = np.cos(np.asarray(pack.walker_primitives(seq).phi)).mean(axis=0)
print("max |dS| =", np.abs(ours - theirs).max())                      # the number in this pack's provenance
```

## Where the geometry and the reference signals come from

The meshes and the released signals are NOT redistributed here -- each pack cites its source repository and
the commit it was read at, in `pack.meta["provenance"]`, with the sha256 of every file the spec references.

{sources}

*This section is written by `parity-fixtures/build.py`; a bare republish regenerates the rendered part and
drops it.*
"""


def render_card(a, X, rows, sources):
    """The rendered card plus this family's section."""
    from dmipy_sim.fill.hub import Hub
    from dmipy_sim.replay.publish import _load_manifest, render_readme
    manifest = _load_manifest(Hub(a.repo))
    return render_readme(manifest, a.repo) + FAMILY_README.format(
        repo=a.repo, rows="\n".join(rows), sources="\n".join(sources),
        n_packs=len(rows), plural="" if len(rows) == 1 else "s") + _held_section()


def _held_section():
    """What the family deliberately does NOT publish, and why -- so a reader does not look for it in the table.

    Two kinds of hold, and the card must show both: a fixture whose pack was BUILT and whose gate HELD it
    (read from the gate's per-fixture verdict, with the reference record's reason), and a reference this family
    reproduces but has built no pack for at all. Reading only the second is how a card comes to promise three
    packs and describe four -- the one thing a card must not do.
    """
    per = (_read_record("gate").get("per_fixture") or {})
    built = set((_read_record("build").get("fixtures") or {}))
    out = []
    for name in sorted(n for n in built if not per.get(n, {}).get("passed")):
        key = "disimpy" if name.startswith("disimpy") else "mcdc"
        ref = REFERENCES.get(key, {}).get("held") or {}
        fails = ", ".join(f"`{f}`" for f in per.get(name, {}).get("failures", []))
        out.append(f"* **{name}** -- its pack is built, recorded and gated, and the gate HELD it on {fails}. "
                   + (f"{ref.get('issue')}: {ref.get('why')}" if ref else ""))
    for key, v in sorted(REFERENCES.items()):
        if v.get("held") and not any(n.startswith(key) for n in built):
            out.append(f"* **{key}** -- no pack built. {v['held'].get('issue')}: {v['held'].get('why')}")
    if not out:
        return ""
    return "\n## Not published\n\n" + "\n".join(out) + "\n"


def publish_only(a):
    """Publish the ``.rpk`` files already built into ``packs/``, and write the family's card.

    The packs may have been built on another box: everything the card and the manifest need is in the pack
    header, so nothing is walked again. The bytes uploaded are the bytes hashed.
    """
    from dmipy_sim.replay.publish import publish, header_of
    gate = _gate_allows_publish()
    log(f"gate: {gate['n_checks']} checks passed, written {gate['written']}")
    commit = code_commit()                      # refuses a dirty tree or an unpushed HEAD
    log(f"code commit {commit}")
    out = os.path.join(HERE, "packs")
    # The gate's own record names the fixtures it passed. Globbing `packs/*.rpk` made the hold on a fixture a
    # convention -- drop a file in the directory and it ships -- so the list comes from the gate.
    bld = _read_record("build")
    per = gate["per_fixture"]
    passed = sorted(n for n in bld["fixtures"] if per.get(n, {}).get("passed"))
    withheld = sorted(n for n in bld["fixtures"] if not per.get(n, {}).get("passed"))
    for n in withheld:
        log(f"HELD by the gate, not published: {n} -- {per[n]['failures']}")
    if not passed:
        raise SystemExit("the gate cleared no pack")
    files = [f"{n}.rpk" for n in passed]
    missing = [f for f in files if not os.path.exists(os.path.join(out, f))]
    if missing:
        raise SystemExit(f"the gate passed {missing} but they are not in {out}")
    stray = sorted(set(f for f in os.listdir(out) if f.endswith(".rpk"))
                   - set(files) - {f"{n}.rpk" for n in withheld})
    if stray:
        raise SystemExit(f"{out} holds {stray}, which the gate did not see. Gate them or move them out; "
                         f"publishing what the gate has not seen is exactly what the gate is for.")
    held = bld.get("held") or {}
    if held:
        log(f"held, not published: {', '.join(held)} -- " + "; ".join(v.get("issue", "?") for v in held.values()))
    from dmipy_sim.fill.hub import sha256_of
    already = {}
    if not a.dry:
        from dmipy_sim.fill.hub import Hub
        from dmipy_sim.replay.publish import _load_manifest
        already = {r["path"]: r.get("sha256") for r in (_load_manifest(Hub(a.repo)).get("packs") or [])}
    rows, sources, uris = [], [], []
    for f in files:
        local = os.path.join(out, f)
        meta = header_of(local)
        prov = meta.get("provenance") or {}
        par = (prov.get("parity") or {}).get("pack") or {}
        log(f"{f}: id {meta['id']} N {meta.get('n_walkers')} parity {par.get('max_abs_diff')} "
            f"{os.path.getsize(local) / 1e6:.1f} MB")
        rows.append(f"| `packs/{f}` | {prov.get('acquisition', '').split(',')[0]} | "
                    f"`{os.path.basename(prov.get('reference_signal', '?'))}` | {meta.get('license')} | "
                    f"{str(meta.get('citation', '')).split(';')[0].strip()} |")
        sources.append(f"* **{f[:-4]}**: {prov.get('reference_repo')} at `{prov.get('reference_commit')}` "
                       f"({meta.get('license')}) -- {prov.get('engine_compared')}; parity max|dS| "
                       f"{par.get('max_abs_diff', float('nan')):.5f} over {par.get('n_meas')} measurements, "
                       f"tolerance {par.get('tol_max', float('nan')):.5f}")
        if not a.dry:
            if already.get(f"packs/{f}") == sha256_of(local):
                log(f"  already published with these bytes, not re-uploaded")
                uris.append(f"hf://{a.repo}/packs/{f}")
                continue
            uris.append(publish(local, a.repo, path=f"packs/{f}",
                                message=f"parity fixtures: {f[:-4]} ({meta.get('n_walkers')} walkers)"))
            log(f"  {uris[-1]}")
    if uris:
        from dmipy_sim.fill.hub import Hub
        hub = Hub(a.repo)
        card = render_card(a, None, rows, sources)
        hub.commit({"README.md": card.encode()}, [], "parity fixtures: the family's card", parent=hub.head())
        log("card written; " + ", ".join(uris))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
# The reference-pack protocol (dmrai-lab/dmipy-sim#482), as records on disk and a gate that reads them.
#
# Each stage writes ONE record file into `records/`; the next stage reads the previous record and nothing
# else; the gate reads recorded numbers only and never walks, measures or fits. `--publish-only` refuses to
# upload unless `records/gate.json` says the family passed and its inputs still hash to what it gated.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

RECORDS = os.path.join(HERE, "records")

#: Stage 1 input, filled BY HAND from the primary sources: what the family reads, where it came from, and
#: the licence the host states -- copied verbatim into `records/licences/`, not paraphrased.
SOURCES = {
    "mcdc-robust": dict(
        url="https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations",
        commit="e7bbe41c12b473cfc027adb8d12a6919c0c63a61",
        commit_date="2020-07-03",
        license_id="LGPL-2.1",
        license_file="LICENSE",
        license_heading="GNU LESSER GENERAL PUBLIC LICENSE",
        host_record="the repository's own LICENSE file at that commit; no separate data licence is stated",
        role="the undulating-axon PLY meshes, their initial-walker lists, the ActiveAx scheme, and the raw "
             "*_DWI.bfloat signals MC/DC produced",
        files=["Simulator-Conf-files/ActiveAxG140_PM.scheme",
               "Simulator-Conf-files/ActiveAxG300_PM.scheme",
               "Simulator-Conf-files/uAxon_d_1.0_amp_0.0_wL_4.0.conf"]),
    "disimpy": dict(
        url="https://github.com/kerkelae/disimpy",
        commit="f1d97421ee56fa0624e37e3aa4bb0ff3437e456b",
        commit_date="2024-11-02",
        license_id="MIT",
        license_file="license.txt",
        license_heading="MIT License",
        host_record="the repository's own license.txt at that commit",
        role="the closed cylinder mesh fixture and the MISST reference signal it is asserted against",
        files=["disimpy/tests/cylinder_mesh_closed.pkl",
               "disimpy/tests/misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt"]),
}

#: Stage 2 input, filled BY HAND: the published quantity each fixture reproduces, and -- the thing that sets
#: the gate's tolerance -- WHICH SIDE of the comparison is a Monte-Carlo estimate.
REFERENCES = {
    "mcdc": dict(
        doi="10.3389/fninf.2020.00008",
        title="Robust Monte-Carlo Simulations in Diffusion-MRI: Effect of the Substrate Complexity and "
              "Parameter Choice on the Reproducibility of Results",
        crossref_checked=True,
        quantity="the direction-resolved PGSE signal of one undulating axon over the ActiveAx protocol "
                 "(372 measurements, TE 53.52 ms, four shells at b = 1925 / 1932 / 3094 / 13190 s/mm^2)",
        released_as="Experiments-raw-signals./Undulated_fibers/d_1/<fixture>/<fixture>_DWI.bfloat, an "
                    "unnormalised sum_walkers cos(phi) as little-endian float32, one per measurement",
        sample="the same object: their released PLY, walked here at their diffusivity and seeded from their "
               "released initial-walker list",
        their_walker_count=50_000,
        their_walker_count_source="the b = 0 entries of the released .bfloat, which ARE the count because "
                                  "the file is unnormalised; the one released .conf states N 1000 and is a "
                                  "reduced example, so it is NOT used",
        their_steps=5_000,
        their_step_source="the released .conf's `T`; at TE 53.52 ms that is a 10.7 us step, 196 nm",
        their_uncertainty="none stated; it is one realisation, so its floor is derived as the standard error "
                          "at 50,000 walkers with the variance of cos(phi) measured on OUR walk",
        value_quantisation=None,     # filled by stage_reference from the module's measured constant
        monte_carlo_sides="BOTH: theirs (50,000 walkers) and ours (the pack's walkers). The gate tolerance is "
                          "3 sqrt(ours^2 + theirs^2).",
        free_parameters={"diffusivity": "theirs (the .conf's 0.6e-9 m^2/s)",
                         "TE / delta / Delta / G": "theirs (the scheme file)",
                         "seed positions": "theirs (the released initial-walker list)",
                         "surface relaxivity": "none in the walk: their walls are reflecting",
                         "sub-step rule": "ours (the engine's own; their step is recorded for comparison)"},
        grade_459="A"),
    "disimpy": dict(
        doi="10.21105/joss.02527",
        title="Disimpy: A massively parallel Monte Carlo simulator for generating diffusion-weighted MRI "
              "data in Python",
        crossref_checked=True,
        quantity="the normalised PGSE signal of a radius-5 um cylinder, gradient perpendicular, delta 30 ms, "
                 "Delta 40 ms, TE 70 ms, 100 b-values linearly spaced 1 .. 3e9 s/m^2, D = 2e-9 m^2/s",
        released_as="disimpy/tests/misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt, "
                    "100 normalised values",
        sample="the same material, not the same object: MISST solves the SMOOTH cylinder analytically while "
               "the fixture's mesh is a 49-gon prism inscribed in it",
        their_walker_count=None,
        their_walker_count_source="not a Monte-Carlo estimate: MISST is an exact eigenfunction solution "
                                  "(Drobnjak, Zhang, Hall, Alexander)",
        their_steps=None,
        their_step_source=None,
        their_uncertainty="exact to the solver's truncation; no Monte-Carlo floor -- but the FILE is "
                          "distributed to seven decimal places, which is the resolution any comparison against "
                          "it is limited to",
        value_quantisation=None,     # filled by stage_reference from the module's measured constant
        monte_carlo_sides="OURS ONLY. The gate tolerance is 3 x our floor alone.",
        free_parameters={"diffusivity": "theirs (2e-9 m^2/s, test_mesh_diffusion)",
                         "radius / delta / Delta / TE / b": "theirs (the fixture and the file's name)",
                         "surface relaxivity": "none in the walk",
                         "sub-step rule": "ours"},
        grade_459="A",
        held=dict(issue="dmrai-lab/dmipy-sim#488",
                  why="#479 (the pickle's face winding) is fixed by #483 and the mesh now walks correctly, but "
                      "at 100,000 walkers neither the mesh NOR the analytic cylinder of the same radius "
                      "reproduces this MISST reference to the Monte-Carlo floor: 1.97e-3 and 1.31e-3 "
                      "respectively, 7.27 and 5.66 sigma, 31 and 26 of 100 measurements outside their own "
                      "3-sigma band. So the residual is not the mesh and not the faceting (a 49-gon's second "
                      "moment is 0.27 % low, worth 4e-6 here). It is invisible below ~30,000 walkers and the "
                      "repo's own cylinder-vs-MISST test carries atol=0.02, forty times the floor at this "
                      "count, so nothing has probed it before. Held until #488 says whether the 1.3e-3 is "
                      "MISST's own truncation -- the reference's uncertainty -- or ours.")),
}



def _multiplicity(n_live, dof, **kw):
    """The gate's parity thresholds, from the one definition in the repo
    (:func:`examples.validation.cross_engine_parity.multiplicity_thresholds`), so the slow test and this
    builder cannot drift apart."""
    import importlib, sys
    sys.path.insert(0, os.path.abspath(os.environ.get("DMIPY_SIM_ROOT", ".")))
    X = importlib.import_module("examples.validation.cross_engine_parity")
    return X.multiplicity_thresholds(n_live, dof, **kw)


def _write_record(name, payload):
    """Write one record and return ``(path, sha256)``. A record is written once, by the stage that measured
    it: a second write with different content is refused, so no later stage can quietly restate a number."""
    from dmipy_sim.fill.hub import sha256_of
    os.makedirs(RECORDS, exist_ok=True)
    path = os.path.join(RECORDS, f"{name}.json")
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    if os.path.exists(path):
        old = open(path).read()
        if old != body:
            raise SystemExit(f"{path} already exists with different content; a record is written once by the "
                             f"stage that measured it. Delete it deliberately to re-measure.")
        return path, sha256_of(path)
    with open(path, "w") as fh:
        fh.write(body)
    log(f"record {name}.json written ({len(body)} bytes)")
    return path, sha256_of(path)


def _read_record(name):
    path = os.path.join(RECORDS, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run the stage that writes it before this one")
    return json.load(open(path))


def _digest(path):
    from dmipy_sim.fill.hub import sha256_of
    return dict(path=os.path.basename(path), sha256=sha256_of(path), bytes=os.path.getsize(path))


def stage_source(a, X):
    """Stage 1. Every input file with its URL, commit, verbatim licence and sha256.

    The licence is COPIED, not named: `records/licences/<source>-<file>` holds the host's own bytes, and the
    record carries that copy's digest, so an audit can compare it with the host without trusting this record.
    """
    os.makedirs(os.path.join(RECORDS, "licences"), exist_ok=True)
    roots = {"mcdc-robust": a.mcdc_data, "disimpy": a.disimpy_data}
    out = {}
    for key, meta in SOURCES.items():
        root = roots.get(key)
        if not root:
            log(f"source {key}: no --{key.split('-')[0]}-data given, skipped")
            continue
        lic_src = os.path.join(root, meta["license_file"])
        if not os.path.exists(lic_src):
            raise SystemExit(f"{key}: no licence file at {lic_src}; a source without a licence is refused")
        text = open(lic_src, encoding="utf-8", errors="replace").read()
        if meta["license_heading"].lower() not in text.lower():
            raise SystemExit(f"{key}: {meta['license_file']} does not contain "
                             f"{meta['license_heading']!r}; the recorded licence is not the host's")
        lic_copy = os.path.join(RECORDS, "licences", f"{key}-{os.path.basename(meta['license_file'])}")
        with open(lic_copy, "w") as fh:
            fh.write(text)
        files = []
        for rel in meta["files"]:
            fp = os.path.join(root, rel)
            if not os.path.exists(fp):
                raise SystemExit(f"{key}: {rel} is not in {root}")
            files.append(dict(_digest(fp), relpath=rel))
        if key == "mcdc-robust":
            for amp, wL in X.MCDC_FIXTURES:
                pth = X.mcdc_paths(root, amp, wL)
                for rel_key in ("ply", "ini", "dwi"):
                    files.append(dict(_digest(pth[rel_key]),
                                      relpath=os.path.relpath(pth[rel_key], root), fixture=f"{amp}-{wL}"))
        out[key] = dict(meta, files=files,
                        license_copy=dict(_digest(lic_copy), first_line=text.splitlines()[0].strip()),
                        read_from=os.path.abspath(root))
        out[key].pop("files", None) or None
        out[key]["files"] = files
    if not out:
        raise SystemExit("no sources recorded")
    return _write_record("source", dict(stage="source", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        sources=out))


def stage_reference(a, X):
    """Stage 2. The published quantity each fixture reproduces, and which side of the comparison is a
    Monte-Carlo estimate -- the fact that sets the gate's tolerance. Refused without a DOI or a digest."""
    src = _read_record("source")
    out = {}
    quant = dict(mcdc=float(X.MCDC_QUANTISATION), disimpy=float(X.DISIMPY_QUANTISATION))
    for key, ref in REFERENCES.items():
        if not ref.get("doi"):
            raise SystemExit(f"reference {key}: no DOI; a reference record without one is refused")
        if not ref.get("crossref_checked"):
            raise SystemExit(f"reference {key}: the DOI was not resolved and its title compared")
        out[key] = dict(ref, value_quantisation=quant[key],
                        value_quantisation_why=(
                            "a float32 sum over 50,000 walkers: half an ulp of the stored value over the count"
                            if key == "mcdc" else
                            "seven decimal places as distributed (1.0000000, 0.8763042): a half-ulp of 5e-8"),
                        value_quantisation_enters="the per-measurement band, in quadrature with the floors -- "
                                                  "at b = 1 s/m^2 the walk's own standard error is 1.6e-11 and "
                                                  "the file states the value to 5e-8, so without it the "
                                                  "comparison asks the file for digits it does not have (68 "
                                                  "sigma, measured)")
    # the reference QUANTITY as data: the digest of the file that holds it, per fixture
    files = {}
    if a.mcdc_data:
        for amp, wL in X.MCDC_FIXTURES:
            p = X.mcdc_paths(a.mcdc_data, amp, wL)
            files[f"mcdc-{amp}-{wL}"] = dict(_digest(p["dwi"]), kind="their DWI",
                                             walkers=REFERENCES["mcdc"]["their_walker_count"],
                                             scheme=dict(_digest(p["scheme"]), name=X.MCDC_SCHEME))
    if a.disimpy_data:
        mp = os.path.join(a.disimpy_data, "disimpy", "tests",
                          "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt")
        files["disimpy-cylinder"] = dict(_digest(mp), kind="MISST, exact", walkers=None,
                                        scheme="built by cross_engine_parity.disimpy_sequence (PGSE, "
                                               "delta 30 ms, Delta 40 ms, TE 70 ms, 100 b to 3e9 s/m^2)")
    if not files:
        raise SystemExit("no reference quantities recorded")
    return _write_record("reference", dict(stage="reference", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           source_sha256=_digest(os.path.join(RECORDS, "source.json"))["sha256"],
                                           references=out, quantities=files))


def stage_spec(a, X):
    """Stage 3. Each fixture's substrate spec, round-tripped, citing its source files by digest.

    ``spec_of(geometry_from_spec(spec))`` must return the spec: anything the geometry cannot carry is a field
    that a re-walk would silently get wrong. This is where that is checked rather than asserted -- the first
    version of this family lost ``extra.water_fraction`` (1.0 for a pool MC/DC never seeded or read) and
    ``seeding.rule`` (``explicit`` became ``uniform_by_volume``, so a re-walk would have seeded the whole 250 um
    tube instead of the central 40 um their list covers).
    """
    _read_record("reference")
    from dmipy_sim.spec import geometry_from_spec, spec_of
    specs = {}
    for amp, wL in X.MCDC_FIXTURES:
        name = f"mcdc-{amp}-{wL}"
        spec = X.mcdc_spec(a.mcdc_data, amp, wL)
        back = spec_of(geometry_from_spec(spec), id=spec.id)
        fields = {}
        for f, got, want in (("seeding.rule", back.seeding.rule, spec.seeding.rule),
                             ("seeding.positions", back.seeding.positions, spec.seeding.positions),
                             ("pools[extra].water_fraction", back.pools[0].water_fraction, spec.pools[0].water_fraction),
                             ("pools[intra].water_fraction", back.pools[1].water_fraction, spec.pools[1].water_fraction),
                             ("domain.box_min", list(back.domain.box_min), list(spec.domain.box_min)),
                             ("domain.boundary", list(back.domain.boundary), list(spec.domain.boundary))):
            fields[f] = dict(round_trips=bool(got == want), value=want)
        lost = sorted(k for k, v in fields.items() if not v["round_trips"])
        if lost:
            raise SystemExit(f"{name}: {lost} do not survive spec -> geometry -> spec, so a re-walk from this "
                             f"spec would not be the walk that was done")
        specs[name] = dict(spec=spec.to_dict(), round_trip=fields,
                           seeded_positions=dict(spec.seeding.positions or {}))
        log(f"  {name}: spec round-trips ({len(fields)} fields checked), seeds from "
            f"{spec.seeding.positions['count']} cited positions read {spec.seeding.positions['read']}")
    if a.disimpy_data:
        spec = X.disimpy_spec(a.disimpy_data)
        back = spec_of(geometry_from_spec(spec), id=spec.id)
        fields = {f: dict(round_trips=bool(got == want), value=want) for f, got, want in (
            ("seeding.rule", back.seeding.rule, spec.seeding.rule),
            ("pools[extra].water_fraction", back.pools[0].water_fraction, spec.pools[0].water_fraction),
            ("pools[intra].water_fraction", back.pools[1].water_fraction, spec.pools[1].water_fraction),
            ("domain.box_max", list(back.domain.box_max), list(spec.domain.box_max)),
            ("domain.boundary", list(back.domain.boundary), list(spec.domain.boundary)))}
        lost = sorted(k for k, v in fields.items() if not v["round_trips"])
        if lost:
            raise SystemExit(f"disimpy-cylinder: {lost} do not survive spec -> geometry -> spec")
        # The surface is DERIVED from the pickle -- merged, then oriented by `Mesh.__init__` (#483) -- so the
        # record states the repair as well as the digest, since neither is the pickle's own bytes.
        V, F, rep = X.disimpy_mesh(a.disimpy_data)
        specs["disimpy-cylinder"] = dict(spec=spec.to_dict(), round_trip=fields, seeded_positions={},
                                         derived_surface=dict(rep, winding=getattr(
                                             geometry_from_spec(spec), "winding", None)))
        log(f"  disimpy-cylinder: spec round-trips ({len(fields)} fields checked), surface derived from the "
            f"pickle ({rep['vertices_before']} -> {rep['vertices_after']} vertices, then oriented)")
    return _write_record("spec", dict(stage="spec", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      reference_sha256=_digest(os.path.join(RECORDS, "reference.json"))["sha256"],
                                      round_trip_rule="spec_of(geometry_from_spec(spec)) == spec on every field "
                                                      "listed; a field that does not survive is refused, not noted",
                                      fixtures=specs))


def stage_design(a, X):
    """Stage 4. The envelope, the target floor, the memory budget, and the PILOT that sets the walker count.

    The pilot walks the REAL window (``MCDC_N_T`` saves over the fixture's own TE) at ``--pilot-n`` and
    measures the envelope floor and the bytes per walker of the walk and of the pack stage. The walker count
    is then derived from those measured numbers; both the pilot's and the derived are in the record, and so is
    the TRADE when the target floor and the memory budget cannot both hold.
    """
    _read_record("spec")
    from dmipy_sim.replay.bank import _measure_floor, _master_arrays, build_replay_pack
    amp, wL = X.MCDC_FIXTURES[0]
    n0 = int(a.pilot_n)
    log(f"design pilot: MC/DC amp {amp} wL {wL}, N={n0} on the REAL window "
        f"({X.MCDC_N_T} saves of {X.MCDC_TE / (X.MCDC_N_T - 1) * 1e6:.1f} us over TE {X.MCDC_TE * 1e3:.2f} ms)")
    t0 = time.time()
    rss0 = X.peak_rss_gb()
    walk = X.mcdc_walk(a.mcdc_data, amp, wL, n0, require_gpu=a.require_gpu)
    walk_s, walk_rss = time.time() - t0, X.peak_rss_gb()
    floor0 = float(_measure_floor(_master_arrays(walk), X.mcdc_envelope()))
    t1 = time.time()
    pk = build_replay_pack(walk, id="pilot/design", license="pilot", citation="pilot", K=a.K,
                           envelope=X.mcdc_envelope(), segment_T=X.MCDC_TE, verbose=False, device=a.device)
    import tempfile
    tmp = tempfile.mkdtemp(); f = os.path.join(tmp, "p.rpk"); pk.save(f)
    pack_bytes, pack_s, pack_rss = os.path.getsize(f), time.time() - t1, X.peak_rss_gb()
    os.remove(f)
    per_walker_pack_bytes = pack_bytes / n0
    per_walker_rss = (pack_rss - rss0) * 1024 ** 3 / n0
    sigma_target = float(a.sigma)
    n_for_floor = int(round(n0 * (floor0 / sigma_target) ** 2 * 1.4))
    budget = float(a.memory_budget_gb)
    n_for_budget = int((budget - rss0) * 1024 ** 3 / max(per_walker_rss, 1.0))
    # n_live and the dof are properties of the scheme and of the count this stage chooses, so the gate's
    # thresholds are derived HERE and the gate only applies them to a parity record that matches them.
    from dmipy_sim.io import mcdc as _mcdc
    _seq = _mcdc.read_scheme(os.path.join(a.mcdc_data, "Simulator-Conf-files", X.MCDC_SCHEME), n_t=X.MCDC_N_T)
    _b = np.asarray(_seq.b())
    n_meas_mcdc, n_live_mcdc = int(_b.size), int((_b > 0).sum())
    their_floor = 1.0 / (2.0 ** 0.5 * REFERENCES["mcdc"]["their_walker_count"] ** 0.5)
    n_match_theirs = int(round(n0 * (floor0 / their_floor) ** 2))
    n_chosen = int(a.n_walkers) if a.n_walkers else min(n_for_floor, n_for_budget)
    trade = None
    if n_for_floor > n_for_budget:
        trade = (f"the target floor {sigma_target:g} needs ~{n_for_floor:,} walkers and the "
                 f"{budget:g} GB budget holds ~{n_for_budget:,} at the measured "
                 f"{per_walker_rss / 1024:.1f} kB of peak RSS per walker. The GRADIENT tier therefore holds "
                 f"at the budget's floor, not at the target; the family's purpose does not need the target, "
                 f"because the parity tolerance is 3 sqrt(ours^2 + theirs^2) and theirs is fixed at "
                 f"{their_floor:.5f} by their 50,000 walkers -- driving ours far below that buys nothing. "
                 f"~{n_match_theirs:,} walkers would make ours EQUAL theirs; this family walks "
                 f"{n_chosen:,}, at which ours is ~{floor0 * (n0 / n_chosen) ** 0.5 / their_floor:.1f}x "
                 f"theirs and the combined band is within {((1 + (floor0 ** 2 * n0 / n_chosen) / their_floor ** 2) ** 0.5 / 2 ** 0.5 - 1) * 100:+.0f}% "
                 f"of what matching would give, so the extra walkers are not worth their memory.")
    # The Disimpy fixture has its own window (TE 70 ms, 1401 saves), its own diffusivity and an EXACT
    # reference, so its walker count cannot be inherited from the MC/DC pilot: with no Monte-Carlo floor on
    # their side the band is ours alone, and the count is set by the floor the comparison needs rather than by
    # matching anybody. Piloted on its own real window.
    dis, n_live_dis = None, len(X.DISIMPY_BVALUES)
    if a.disimpy_data:
        t2 = time.time(); rss2 = X.peak_rss_gb()
        dw = X.disimpy_walk(a.disimpy_data, a.pilot_n, require_gpu=a.require_gpu,
                            sub_steps=a.disimpy_sub_steps)
        dfloor = float(_measure_floor(_master_arrays(dw), X.disimpy_envelope()))
        dpar = X.parity(X.walk_cos_phi(dw, X.disimpy_sequence()), X.disimpy_reference(a.disimpy_data),
                        reference_quantisation=X.DISIMPY_QUANTISATION)
        n_live_dis = int(dpar["n_live"])
        d_rss = (X.peak_rss_gb() - rss2) * 1024 ** 3 / a.pilot_n
        n_dis = int(a.disimpy_n_walkers or a.pilot_n)
        dis = dict(n_walkers=a.pilot_n, on_real_window=True, window_s=X.DISIMPY_TE, n_t=X.DISIMPY_N_T,
                   envelope_floor=dfloor, sub_steps=int(dw.sub_steps),
                   step_m=float(np.sqrt(6 * X.DISIMPY_D * dw.dt_sim)),
                   illegal_crossings=int(dw.illegal_crossings), seconds=round(time.time() - t2, 1),
                   rss_bytes_per_walker=round(d_rss, 1), winding=getattr(dw, "winding", None),
                   parity_at_pilot={k: dpar[k] for k in ("max_abs_diff", "floor_ours_max", "worst_sigma",
                                                         "n_over_k_sigma", "n_live", "n_exact")},
                   n_walkers_chosen=n_dis,
                   reference="MISST, exact: the band is OUR standard error alone, so the count is set by the "
                             "precision the comparison needs, not by matching a reference's own floor")
        log(f"design disimpy pilot: N={a.pilot_n}, sub_steps {dw.sub_steps}, envelope floor {dfloor:.5f}; "
            f"parity max|dS| {dpar['max_abs_diff']:.5f} worst {dpar['worst_sigma']:.3f} sigma; chose {n_dis:,}")
        del dw
    rec = dict(stage="design", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
               spec_sha256=_digest(os.path.join(RECORDS, "spec.json"))["sha256"],
               envelope=dict(mcdc=X.mcdc_envelope(), disimpy=X.disimpy_envelope()),
               envelope_note="the fixture's own scheme plus two OGSE periods, which is stricter than PGSE alone",
               save_grid=dict(n_t=X.MCDC_N_T, dt_s=X.MCDC_TE / (X.MCDC_N_T - 1), window_s=X.MCDC_TE,
                              why="10704 / 2676 = 4, so every lobe edge of the scheme is still on a sample"),
               target_floor=sigma_target, memory_budget_gb=budget, K=int(a.K),
               pilot=dict(n_walkers=n0, on_real_window=True, envelope_floor=floor0,
                          walk_seconds=round(walk_s, 1), pack_seconds=round(pack_s, 1),
                          sub_steps=int(walk.sub_steps), illegal_crossings=int(walk.illegal_crossings),
                          pack_bytes=int(pack_bytes), bytes_per_walker_pack=round(per_walker_pack_bytes, 1),
                          peak_rss_gb_walk=round(walk_rss, 2), peak_rss_gb_pack=round(pack_rss, 2),
                          rss_bytes_per_walker=round(per_walker_rss, 1), codec_err=float(pk.meta["fidelity"]["err_max"]),
                          band_hz=float(pk.temporal_bandwidth_hz)),
               derived=dict(n_for_target_floor=n_for_floor, n_for_budget=n_for_budget,
                            their_floor=their_floor, n_to_match_their_floor=n_match_theirs,
                            n_walkers=n_chosen,
                            rule="measured pilot floor scaled as 1/sqrt(N) with a 1.4 safety factor, capped by "
                                 "the measured bytes per walker against the budget; never an estimate"),
               disimpy_pilot=dis,
               gate_tolerance=dict(
                   policy=dict(k_per=3.0, expected_family_wise=0.5, poisson_sigma=3.0,
                               derived_by="examples/validation/cross_engine_parity.multiplicity_thresholds"),
                   mcdc=_multiplicity(n_live_mcdc, n_chosen - 1),
                   disimpy=_multiplicity(n_live_dis, (dis or {}).get("n_walkers_chosen", n_chosen) - 1),
                   note="the per-measurement band is 3 sqrt(ours^2 + theirs^2) where the reference is itself "
                        "Monte-Carlo and 3 x ours alone where it is exact (reference.json's "
                        "monte_carlo_sides), with the standard error taken analytically from the walkers; "
                        "these are the MULTIPLICITY corrections on top of it. n_live counts the measurements "
                        f"with a non-zero gradient: {n_live_mcdc} of {n_meas_mcdc} rows of "
                        f"{X.MCDC_SCHEME}, the other {n_meas_mcdc - n_live_mcdc} being b = 0 with no spread, "
                        "no band and no degrees of freedom"),
               trade=trade)
    log(f"design: pilot floor {floor0:.5f} at N={n0}; target needs {n_for_floor:,}; budget holds "
        f"{n_for_budget:,}; their floor {their_floor:.5f} is matched at {n_match_theirs:,}; chose {n_chosen:,}")
    if trade:
        log(f"design trade: {trade}")
    return _write_record("design", rec)


def stage_records_from_build(a, X):
    """Stages 5 and 6 as records. It walks NOTHING.

    The engine's counters and the certificate come from ``status.jsonl`` and each pack's header, written by the
    run that measured them. The **parity is recomputed here from the packs themselves** -- a replay of the
    stored bands through ``walker_primitives``, which is what a consumer gets and is not a walk -- because the
    number embedded in the packs was computed with the superseded split-half floor. The packs are not
    re-encoded: the record carries the recomputed number and the superseded one side by side.
    """
    des = _read_record("design")
    from dmipy_sim.replay.publish import header_of
    from dmipy_sim.replay import ReplayPack
    from dmipy_sim.io import mcdc
    seq_mcdc = (mcdc.read_scheme(os.path.join(a.mcdc_data, "Simulator-Conf-files", X.MCDC_SCHEME),
                                 n_t=X.MCDC_N_T) if a.mcdc_data else None)
    status = os.path.join(HERE, "status.jsonl")
    if not os.path.exists(status):
        raise SystemExit(f"{status} is missing: the family stage has not run")
    rows = {}
    for line in open(status):
        r = json.loads(line)
        rows[r["name"]] = r                                    # a re-run replaces its own row
    packs = {}
    for name, r in sorted(rows.items()):
        local = os.path.join(HERE, "packs", f"{name}.rpk")
        if not os.path.exists(local):
            log(f"{name}: no pack at {local}, skipped")
            continue
        meta = header_of(local)
        prov = meta.get("provenance") or {}
        par = prov.get("parity") or {}
        fid = meta["fidelity"]
        if name.startswith("disimpy"):
            seq, ref, n_theirs, quant = (X.disimpy_sequence(), X.disimpy_reference(a.disimpy_data), None,
                                         X.DISIMPY_QUANTISATION)
        else:
            amp, wL = name.split("-")[1], name.split("-")[2]
            seq = seq_mcdc
            ref, n_theirs = X.mcdc_reference(a.mcdc_data, float(amp), float(wL), seq_mcdc)
            quant = X.MCDC_QUANTISATION
        pk = ReplayPack.load(local)
        phi = X.pack_walker_phase(pk, seq)                     # the bands a consumer contracts, not a walk
        recomputed = X.parity(np.cos(phi), ref, n_theirs, reference_quantisation=quant)
        # Served vs decoded, PER MEASUREMENT: the pack's two routes to the same signal -- the per-walker one
        # (`walker_primitives`, the bands contracted once) against the scalar one (`replay`) -- reduced THE SAME
        # WAY. `replay` returns the modulus of the complex ensemble mean; `mean(cos phi)` is its real part, and
        # the two differ by the quadrature term s^2/(2c), which grows with b. Comparing them across reductions
        # is what made this check read 3.10e-03 at b = 13190 s/mm^2 and put #484 on the engine; the phases are
        # bit-identical across the routes. Both read this pack only, so no walk is involved.
        served_gap = float(np.abs(X.ensemble_modulus(phi, np.asarray(pk.spin_weights, float))
                                  - np.asarray(pk.replay(seq), float).ravel()).max())
        del pk, phi
        log(f"  {name}: recomputed parity max|dS| {recomputed['max_abs_diff']:.6f}, worst "
            f"{recomputed['worst_sigma']:.3f} sigma over {recomputed['n_live']} live measurements, "
            f"{recomputed['n_over_k_sigma']} over {recomputed['k']:g} sigma "
            f"(the pack embeds {par.get('pack', {}).get('max_abs_diff', float('nan')):.6f} from the "
            f"superseded split-half floor)")
        packs[name] = dict(
            pack=dict(_digest(local), id=meta["id"], license=meta["license"], K=r["K"], n_t=r["n_t"],
                      dt_s=r["dt"], band_hz=r["band_hz"],
                      certificate={k: float(v) for k, v in fid.items() if isinstance(v, (int, float))},
                      meets_target=dict(gradient=bool(fid["floor_max"] <= _read_record("design")["target_floor"]),
                                        codec=bool(fid["err_max"] <= fid["floor_max"]))),
            walk=dict(n_walkers=r["n_walkers"], sub_steps=r["sub_steps"],
                      illegal_crossings=r["illegal_crossings"], seconds=r["seconds"],
                      peak_rss_gb=r["peak_rss_gb"], code_commit=r.get("code_commit")),
            parity=dict(served=recomputed, served_vs_walk_max_per_measurement=served_gap,
                        monte_carlo_sides=(REFERENCES["disimpy"] if name.startswith("disimpy")
                                           else REFERENCES["mcdc"])["monte_carlo_sides"],
                        served_vs_decoded="max over measurements of |walker_primitives route - replay route|, "
                                          "both read from this pack",
                        superseded_in_pack=par.get("pack"),
                        superseded_note="the pack's embedded parity used the one-permutation split-half floor; "
                                        "`served` is the same replay against the analytic standard error, and "
                                        "is what the gate reads. The pack is not re-encoded.",
                        envelope_band_hz=par.get("envelope_band_hz"), pack_band_hz=par.get("pack_band_hz"),
                        in_band=par.get("in_band")),
            reference=dict(signal=prov.get("reference_signal"), walkers=prov.get("reference_walkers"),
                           commit=prov.get("reference_commit"), repo=prov.get("reference_repo")))
    if not packs:
        raise SystemExit("no packs to record")
    return _write_record("build", dict(stage="build", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       design_sha256=_digest(os.path.join(RECORDS, "design.json"))["sha256"],
                                       fixtures=packs,
                                       held={k: v["held"] for k, v in REFERENCES.items()
                                             if v.get("held") and not any(n.startswith(k) for n in packs)}))


def stage_gate(a, X):
    """Stage 7. Deterministic, reads recorded numbers, and CAN FAIL. It never walks, measures or fits.

    Four checks per fixture, each against a tolerance derived in the records rather than typed here:

    1. **the served replay equals the decoded channel, PER MEASUREMENT, in the same reduction.** The largest
       difference over all measurements between the pack's per-walker route (``walker_primitives``, the bands
       contracted once) and its scalar route (``replay``). Two mistakes are possible here and both were made:
       comparing one MAXIMUM with another passes a pack whose maxima merely coincide at different
       measurements; and comparing ``mean(cos phi)`` (the real part) with ``replay`` (the MODULUS of the
       complex ensemble mean) reads the quadrature term ``s^2/(2c)`` as a disagreement that grows with b, which
       is what produced dmrai-lab/dmipy-sim#484 against an engine whose routes are bit-identical. Reduced the
       same way the difference is exactly 0.
    2. **our signal against theirs, within the combined floors.** ``3 sqrt(ours^2 + theirs^2)`` for a
       reference that is itself Monte-Carlo, and ``3 x ours`` alone for MISST, as the reference record says,
       with the standard error taken analytically from the walkers. Per measurement at the design record's
       derived thresholds: at most ``max_exceedances`` over the band, and the worst inside ``k_family_wise``.
    3. **every waveform of the declared envelope is inside the stored band**, from the recorded
       ``waveform_band`` requirement against the recorded ``temporal_bandwidth_hz``.
    4. **the records are complete**: a licence copy, a digest for every input, a resolved DOI, and -- where
       a certificate tier is below its target -- the design record's trade note.
    """
    src, ref, spc, des, bld = (_read_record(n) for n in ("source", "reference", "spec", "design", "build"))
    fails, checks = [], []

    def check(ok, what, detail):
        checks.append(dict(check=what, passed=bool(ok), detail=detail))
        if not ok:
            fails.append(f"{what}: {detail}")

    # 4 -- the records themselves
    for key, s in src["sources"].items():
        check(bool(s.get("license_copy", {}).get("sha256")), f"source/{key}/licence",
              f"licence copied verbatim to records/licences ({s.get('license_id')}), "
              f"sha256 {s.get('license_copy', {}).get('sha256', '')[:16]}")
        check(all(f.get("sha256") for f in s["files"]), f"source/{key}/digests",
              f"{len(s['files'])} input files, every one with a sha256")
        check(bool(s.get("commit")), f"source/{key}/commit", f"commit {s.get('commit')}")
    for name, f in sorted(spc["fixtures"].items()):
        lost = sorted(k for k, v in f["round_trip"].items() if not v["round_trips"])
        check(not lost, f"{name}/spec-round-trips",
              f"{len(f['round_trip'])} fields survive spec -> geometry -> spec"
              + (f"; LOST {lost}" if lost else ""))
        rule = f["spec"]["seeding"]["rule"]
        if rule == "explicit":
            check(bool(f["seeded_positions"].get("sha256")), f"{name}/seeding-cites-its-positions",
                  f"rule explicit, {f['seeded_positions'].get('count')} positions read "
                  f"{f['seeded_positions'].get('read')} from {f['seeded_positions'].get('file')}")
        else:
            check(not f["seeded_positions"], f"{name}/seeding-rule-needs-no-positions",
                  f"rule {rule}: the walk draws its own seeds, so there are no positions to cite")
    for key, r in ref["references"].items():
        check(bool(r.get("doi")) and bool(r.get("crossref_checked")), f"reference/{key}/doi",
              f"{r.get('doi')} resolved and its title compared")
        check(bool(r.get("monte_carlo_sides")), f"reference/{key}/floors", r.get("monte_carlo_sides", ""))

    for name, f in sorted(bld["fixtures"].items()):
        pp = f["parity"]["served"]
        cert = f["pack"]["certificate"]
        codec = float(cert["err_max"])
        # 1 -- served == decoded, PER MEASUREMENT
        served_gap = float(f["parity"]["served_vs_walk_max_per_measurement"])
        check(served_gap <= max(codec, 1e-9), f"{name}/served-equals-decoded",
              f"the largest PER-MEASUREMENT difference between the pack's two serving routes "
              f"(walker_primitives and replay), both reduced to the modulus of the complex ensemble mean, is "
              f"{served_gap:.2e}, against the certified codec error {codec:.2e}")
        check(codec <= float(cert["floor_max"]), f"{name}/codec-below-floor",
              f"codec error {codec:.2e} <= the walk's own split-half floor {cert['floor_max']:.5f}")
        # 2 -- ours against theirs, within the recorded floors, at the design record's derived thresholds.
        #      NOT max|dS| against tol_max: that compares a maximum with a maximum and lets a measurement
        #      whose OWN floor is small sit outside its OWN band unnoticed. The per-measurement count and the
        #      worst measurement in units of its own band are what the recorded numbers say.
        mult = des["gate_tolerance"]["disimpy" if "disimpy" in name else "mcdc"]
        if (int(pp["n_live"]), int(pp["dof"])) != (int(mult["n_live"]), int(mult["dof"])):
            raise SystemExit(f"{name}: {pp['n_live']} live measurements on {pp['dof']} dof against the design "
                             f"record's {mult['n_live']} on {mult['dof']}; its thresholds do not apply")
        check(int(pp["n_over_k_sigma"]) <= int(mult["max_exceedances"]),
              f"{name}/parity-exceedance-count",
              f"{pp['n_over_k_sigma']} of {pp['n_live']} live measurements outside their own "
              f"{mult['k_per_measurement']:g}-sigma band, against the {mult['max_exceedances']} that chance "
              f"allows (expected {mult['expected_exceedances']:.3f} at {mult['dof']} dof)")
        check(float(pp["worst_sigma"]) <= float(mult["k_family_wise"]),
              f"{name}/parity-worst-measurement",
              f"the worst measurement is at {pp['worst_sigma']:.3f} sigma of its own standard error "
              f"(measurement {pp['worst_sigma_at']}; max|dS| {pp['max_abs_diff']:.5f}; our floor "
              f"{pp['floor_ours_max']:.5f}, theirs {pp['floor_theirs_max']:.5f}), against the family-wise "
              f"band {mult['k_family_wise']:.3f} sigma")
        q_rec = float(ref["references"]["disimpy" if "disimpy" in name else "mcdc"]["value_quantisation"])
        check(float(pp["reference_quantisation"]) == q_rec, f"{name}/band-uses-the-recorded-quantisation",
              f"the band was widened by the reference's own {q_rec:.2e} resolution, as the reference record "
              f"states; a band finer than the reference's resolution asks it for digits it does not have")
        check(float(pp["max_abs_diff_exact"]) == 0.0, f"{name}/zero-variance-measurements-are-exact",
              f"the {pp['n_exact']} measurements with no per-walker spread (b = 0) differ by "
              f"{pp['max_abs_diff_exact']:.3e}, which must be 0: there is no band for them to be inside")
        # 3 -- the envelope inside the stored band
        check(bool(f["parity"]["in_band"]) and
              float(f["parity"]["envelope_band_hz"]) <= float(f["parity"]["pack_band_hz"]),
              f"{name}/envelope-in-band",
              f"the fixture's scheme needs {f['parity']['envelope_band_hz']:.0f} Hz and the pack stores "
              f"{f['parity']['pack_band_hz']:.0f} Hz")
        # 4 -- a tier below target needs the design's trade note
        if not f["pack"]["meets_target"]["gradient"]:
            check(bool(des.get("trade")), f"{name}/tier-below-target-has-a-trade",
                  f"the gradient tier's floor {cert['floor_max']:.5f} is above the target "
                  f"{des['target_floor']:g}; the design record states the trade"
                  + ("" if des.get("trade") else " -- IT DOES NOT"))

    # The verdict is PER FIXTURE as well as whole. A family's fourth pack failing must not stop the three that
    # passed from being published, and a pack must not be published because the others passed: a pack ships iff
    # every check naming it passed AND every family-level check (the source and reference records) passed.
    family_ok = [c for c in checks if c["check"].startswith(("source/", "reference/"))]
    names = sorted({c["check"].split("/")[0] for c in checks if not c["check"].startswith(("source/", "reference/"))})
    per_fixture = {}
    for n in names:
        mine = [c for c in checks if c["check"].startswith(n + "/")]
        bad = [c["check"] for c in mine + family_ok if not c["passed"]]
        per_fixture[n] = dict(passed=not bad, n_checks=len(mine), failures=bad)
    verdict = dict(stage="gate", written=time.strftime("%Y-%m-%dT%H:%M:%SZ"), per_fixture=per_fixture,
                   inputs={n: _digest(os.path.join(RECORDS, f"{n}.json"))["sha256"]
                           for n in ("source", "reference", "spec", "design", "build")},
                   passed=not fails, n_checks=len(checks), failures=fails, checks=checks,
                   held=bld.get("held") or {})
    for c in checks:
        log(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']}: {c['detail']}")
    path, _ = _write_record("gate", verdict)
    ok = sorted(n for n, v in per_fixture.items() if v["passed"])
    bad = sorted(n for n, v in per_fixture.items() if not v["passed"])
    log(f"GATE: {len(checks) - len(fails)} of {len(checks)} checks passed; "
        f"{len(ok)} of {len(per_fixture)} fixtures clear ({', '.join(ok) or 'none'})"
        + (f"; HELD: {', '.join(bad)}" if bad else ""))
    if fails:
        log("GATE FAILED on: " + "; ".join(fails))
    log(f"  {path}")
    return path, None


def _gate_allows_publish():
    """The gate's verdict, and that its inputs still hash to what it gated. Publication reads this and nothing
    else, and it reads the PER-FIXTURE verdict: a pack ships iff every check naming it passed and every
    family-level check passed, so one failing fixture holds itself and not the family."""
    g = _read_record("gate")
    per = g.get("per_fixture") or {}
    if not per:
        raise SystemExit("records/gate.json has no per-fixture verdict; re-run the gate")
    if not any(v["passed"] for v in per.values()):
        raise SystemExit(f"records/gate.json clears no fixture: "
                         f"{ {k: v['failures'] for k, v in per.items()} }")
    for n, sha in g["inputs"].items():
        now = _digest(os.path.join(RECORDS, f"{n}.json"))["sha256"]
        if now != sha:
            raise SystemExit(f"records/{n}.json changed after the gate ran ({sha[:12]} -> {now[:12]}); "
                             f"re-run the gate")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=None,
                    help="the dmipy-sim checkout holding examples/validation/cross_engine_parity.py "
                         "(default: the one this script is in)")
    ap.add_argument("--work", default=None,
                    help="where packs/ and records/ live (default: $DMIPY_SIM_PARITY_WORK, else "
                         "~/dmrai-ws/parity-fixtures)")
    ap.add_argument("--mcdc-data", help="the Robust-Monte-Carlo-Simulations checkout")
    ap.add_argument("--disimpy-data", help="the disimpy checkout")
    ap.add_argument("--mcdc-commit", default=None)
    ap.add_argument("--disimpy-commit", default=None)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n-walkers", type=int, default=100_000)
    ap.add_argument("--disimpy-n-walkers", type=int, default=None,
                    help="the Disimpy pack's walker count (its reference is exact, so its band is ours alone)")
    ap.add_argument("--disimpy-sub-steps", type=int, default=None,
                    help="pin the Disimpy walk's sub-steps (the MISST gap is first order in the step)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--require-gpu", action="store_true", default=None)
    ap.add_argument("--publish-only", action="store_true",
                    help="publish the .rpk already in packs/ and write the card; walk nothing")
    ap.add_argument("--stage", choices=["source", "reference", "spec", "design", "build", "gate"],
                    help="run one stage of the reference-pack protocol (dmrai-lab/dmipy-sim#482); each "
                         "writes records/<stage>.json and reads only the record before it")
    ap.add_argument("--sigma", type=float, default=3e-3, help="the design record's target floor")
    ap.add_argument("--memory-budget-gb", type=float, default=60.0,
                    help="the hard cap the design's pilot checks the pack stage against")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.work:
        global HERE, RECORDS
        HERE = os.path.abspath(a.work); RECORDS = os.path.join(HERE, "records")
    logging.getLogger("dmipy_sim").setLevel(logging.WARNING)
    if a.stage:
        X = fixtures(a.sim)
        if a.mcdc_data:
            os.environ["DMIPY_SIM_SURFACE_DIR"] = os.path.abspath(
                os.path.join(a.mcdc_data, "Experiments-mesh -files", "Undulated-fibres", "d_1"))
        return dict(source=stage_source, reference=stage_reference, spec=stage_spec, design=stage_design,
                    build=stage_records_from_build, gate=stage_gate)[a.stage](a, X)
    if a.publish_only:
        return publish_only(a)
    if not (a.mcdc_data or a.disimpy_data):
        ap.error("give --mcdc-data and/or --disimpy-data")
    if a.mcdc_data:
        os.environ["DMIPY_SIM_SURFACE_DIR"] = os.path.abspath(
            os.path.join(a.mcdc_data, "Experiments-mesh -files", "Undulated-fibres", "d_1"))
    X = fixtures(a.sim)
    return pilot(a, X) if a.pilot else family(a, X)


if __name__ == "__main__":
    main()

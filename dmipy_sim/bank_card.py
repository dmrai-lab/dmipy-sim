"""Generate a HuggingFace-renderable **substrate card** from a replay-pack's metadata.

A substrate card is the per-substrate analogue of a HF model/dataset card: a Markdown file
(``<id>.md``) that sits beside ``<id>.rpk`` in the bank repo and documents, from the pack's
own self-describing metadata, what the substrate is, how it was generated (provenance), which
replay tiers it carries, its measured fidelity, and how to load and replay it.  Because it is
generated from ``pack.meta`` there is no way for the card to drift from the artifact.

``substrate_card(meta) -> str`` returns the Markdown (with YAML front-matter so the Hub shows
tags / license).  ``write_card(meta, path)`` writes it.

Public note: public replay packs are provider-driven for susceptibility, so the C3 (field)
tier is absent (``replay_envelope.field`` is False) and its row degrades to "not present"
rather than crashing — the card reads only what is actually in ``pack.meta``.
"""
from __future__ import annotations

_TIER_ORDER = [
    ("gradient", "C0", "Gradient", "positions — any q/b, direction, waveform (PGSE/PGSTE/OGSE/…)"),
    ("bulk_relaxation", "C1", "Bulk relaxation", "per-compartment T2/T1 via the compartment map"),
    ("surface_relaxivity", "C2", "Surface relaxivity", "boundary-local-time channel (any ρ)"),
    ("field", "C3", "Field / susceptibility", "provider-driven off-resonance at replay (public: not a stored channel)"),
    ("magnetization_transfer", "C4", "Magnetization transfer", "MT saturation (Z-spectrum / MTR)"),
]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def substrate_card(meta: dict) -> str:
    aid = meta.get("id", "unknown")
    prov = meta.get("provenance", {}) or {}
    env = meta.get("replay_envelope", {}) or {}
    fid = meta.get("fidelity", {}) or {}
    wp = meta.get("walk_params", {}) or {}
    comp = meta.get("compression", {}) or {}
    lic = meta.get("license", "unspecified")
    cite = meta.get("citation", "")
    name = aid.split("/")[-1]

    tags = ["diffusion-mri", "monte-carlo", "replay-pack", "microstructure", "dmipy"]
    gen = prov.get("generator")
    gen_name = gen.get("name") if isinstance(gen, dict) else gen
    if gen_name:
        tags.append(str(gen_name).lower())
    if "white" in str(prov.get("substrate", "")).lower() or "axon" in aid or "wm" in aid:
        tags.append("white-matter")

    fm = ["---", f"license: {str(lic).lower()}", "tags:"]
    fm += [f"  - {t}" for t in tags]
    fm += [f"pretty_name: {name}", "---", ""]

    L = fm
    L += [f"# Substrate card — `{aid}`", ""]
    real = prov.get("real_or_synthetic", "synthetic")
    L += [f"A **replay pack** (`.rpk`) — a compressed Monte-Carlo master walk on a *{real}* "
          f"diffusion-MRI substrate, plus every physics channel needed to **replay** any "
          f"acquisition inside its certified envelope without re-simulating. One walk, every "
          f"acquisition. Format: the open [Replay Pack Specification]"
          f"(https://github.com/dmrai-lab/replay-pack-spec).", ""]

    # ---- geometry & provenance ----
    L += ["## Geometry & provenance", ""]
    rows = []
    if gen_name:
        url = gen.get("url") if isinstance(gen, dict) else None
        rows.append(("Generator", f"[{gen_name}]({url})" if url else gen_name))
        if isinstance(gen, dict) and gen.get("authors"):
            rows.append(("Generator authors", gen["authors"]))
    for k, label in [("substrate", "Substrate"), ("n_fibres", "Fibres"), ("g_ratio", "g-ratio"),
                     ("icvf", "ICVF (intra+myelin)"), ("f_intra", "f_intra"),
                     ("f_myelin", "f_myelin"), ("f_extra", "f_extra"),
                     ("box_um", "Voxel box (µm)"), ("fibre_axis", "Fibre axis"),
                     ("boundary", "Boundary condition")]:
        if k in prov:
            rows.append((label, _fmt(prov[k])))
    L += ["| Property | Value |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in rows]
    L += [""]

    # ---- replay tiers ----
    L += ["## Replay capability", "",
          "| Tier | Channel | Present | Meaning |", "|---|---|:--:|---|"]
    for key, cx, label, meaning in _TIER_ORDER:
        on = "✅" if env.get(key) else "—"
        L.append(f"| {cx} | {label} | {on} | {meaning} |")
    L += [""]
    deferred = prov.get("tiers_deferred")
    if deferred:
        L += ["**Deferred tiers (with reason):**", ""]
        for k, why in deferred.items():
            L.append(f"- **{k}** — {why}")
        L += [""]
    if env.get("diffusivity_fixed"):
        L += ["> Diffusivity is fixed at walk time (a geometry property, not a replay knob).", ""]

    # ---- fidelity ----
    L += ["## Fidelity", ""]
    L += ["| Metric | Value |", "|---|---|"]
    for k, label in [("err_max", "Max replay error"), ("floor_max", "MC noise floor"),
                     ("within_2x_floor", "Within 2× floor"), ("target_floor", "Target floor σ*"),
                     ("meets_target", "Meets σ*")]:
        if k in fid:
            L.append(f"| {label} | {_fmt(fid[k])} |")
    L += [f"| Walkers | {wp.get('n_walkers')} |",
          f"| Saved time points | {wp.get('n_t')} |",
          f"| T_max (s) | {_fmt(wp.get('T_max'))} |",
          f"| Compression | {comp.get('method')} (K={comp.get('K')}) |", ""]

    # ---- MT pool (C4) ----
    mt = meta.get("mt")
    if mt:
        L += ["## Magnetization transfer (two-pool qMT)", "",
              "Geometry-derived pool (from the myelin surface-to-free-volume ratio); replay the "
              "Z-spectrum / MTR as a pool-level knob.", "",
              "| Parameter | Value |", "|---|---|",
              f"| Bound fraction f_b | {_fmt(mt.get('f_bound'))} |",
              f"| Forward rate k_f (1/s) | {_fmt(mt.get('k_forward'))} |",
              f"| Surface-to-volume (1/m) | {_fmt(mt.get('S_over_V'))} |",
              f"| T2 bound (s) | {_fmt(mt.get('T2_bound'))} |", ""]

    # ---- usage ----
    L += ["## Load & replay", "", "```python", "import numpy as np",
          "from dmipy_sim import bank", "",
          f'pack = bank.pull("{aid}", repo_id="dmrai-lab/substrate-bank")  # cached + hash-verified',
          "",
          "# C0/C1/C2 — build any waveform G on the pack's time grid (n_meas, pack.n_t, 3),",
          "# e.g. PGSE perpendicular vs parallel to the fibre axis (recovers the anisotropy):",
          "S = pack.replay(waveform, T2=[0.05])                # C1; add surface_relaxivity=... for C2"]
    if env.get("field"):
        L += ["# C3 field — provider-driven off-resonance at replay (any B0 / direction / χ):",
              "S = pack.replay(waveform, susceptibility=provider, eps_P=eps_P)"]
    if mt:
        L += ["# C4 MT — geometry-derived two-pool Z-spectrum / MTR:",
              "Z = pack.mt_zspectrum(np.linspace(-3e4, 3e4, 41), w1_hz=200.0, t_sat=0.5)",
              "mtr = pack.mtr(offset_hz=1e4, w1_hz=200.0, t_sat=0.5)"]
    L += ["```", ""]

    # ---- config appendix ----
    cfg = prov.get("cactus_config")
    if cfg:
        L += ["## Generator configuration", "",
              "The exact generator config that produced this geometry (also embedded in the "
              "`.rpk` provenance):", "", "```", cfg.strip(), "```", ""]
    if prov.get("optimized_final_header"):
        h = prov["optimized_final_header"]
        L += [f"Geometry header (`optimized_final.txt`): voxel_side={h[0]}, "
              f"n_fibres={h[1]}, n_control_points={h[2]}.", ""]
    patch = gen.get("patch") if isinstance(gen, dict) else None
    if patch:
        L += [f"> Reproducibility note: {patch}", ""]

    # ---- footer ----
    L += ["## Citation & license", "", f"**License:** {lic}  ", f"**Cite:** {cite}", ""]
    return "\n".join(L)


def write_card(meta: dict, path: str) -> str:
    md = substrate_card(meta)
    with open(path, "w") as f:
        f.write(md)
    return path

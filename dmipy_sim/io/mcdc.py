"""Readers for what the MC/DC simulator (Rafael-Patino et al. 2020) writes and reads: its Camino-style
``.scheme`` file, its configuration file, its initial-walker list and its raw ``.bfloat`` signal.

The unit system is MC/DC's own (``src/constants.h``, ``src/parameters.cpp``): lengths are **millimetres** and
times **milliseconds** inside the engine, and ``scale_from_stu 1`` in a configuration file means the *scheme*
and the *diffusivity and duration* are written in SI and converted on load -- the voxel corners, ``ply_scale``
and the initial-walker file are **not** converted and are already in millimetres. Every reader here returns
SI: metres, seconds, T/m.

These readers construct nothing. The substrate producer for an MC/DC mesh is
:func:`dmipy_sim.spec.mcdc_axon_spec`.
"""
import os
import re

import numpy as np

__all__ = ["read_scheme", "read_bfloat", "read_conf", "read_ini_walkers", "McdcConf"]

_MM = 1e-3


# ------------------------------------------------------------------------------------------ the scheme file
def read_scheme(path, *, n_t=None, gradient_directions=None, min_samples_per_delta=500):
    """A Camino ``VERSION: STEJSKALTANNER`` scheme as ONE :class:`~dmipy_sim.acquisition.ScannerSequence`.

    Seven whitespace-separated numbers per measurement -- ``gx gy gz |G| Delta delta TE``, the direction a unit
    vector, ``|G|`` in T/m and the three times in seconds -- which MC/DC plays as an **effective** gradient
    centred on the echo: ``pad = (TE - Delta - delta) / 2``, ``+G`` over ``[pad, pad + delta]`` and ``-G`` over
    ``[pad + Delta, pad + Delta + delta]`` (``PGSESequence::getGradImpulse``). That is a PGSE with square lobes
    and no ramp, so the sequence is built through :func:`dmipy_sim.pgse` at ``slew_rate = inf`` with per-row
    ``delta`` and ``Delta`` about one 180 at ``TE/2``.

    ``n_t`` defaults to the coarsest grid on which every ``delta``, ``Delta``, ``pad`` and ``TE`` of the file is
    a whole number of samples -- so no lobe edge is quantised -- refined until the shortest lobe holds
    ``min_samples_per_delta`` samples (:func:`_exact_n_t`); an ``n_t`` that is not a coarsening of that lattice
    is refused. Measured on ``ActiveAxG140_PM.scheme``: the default is 10,705 samples over TE and the realised
    ``b`` is the Stejskal-Tanner value of the file's own columns to **3.1e-5** relative; at the coarsening the
    parity family uses, ``n_t = 2677``, it is **1.7e-3**, which changes no signal (the two grids give parity
    numbers identical to 1e-12, since what matters is that the lobe EDGES stay on samples and both grids keep
    them there).

    Refused by name, because the format states no answer: a header that is not ``STEJSKALTANNER`` (``APGSE``
    and ``WAVEFORM`` are other layouts, not this one); a token count that is not a multiple of seven; a
    direction that is neither unit nor zero; a zero direction with a non-zero amplitude, or the reverse; more
    than one ``TE`` (one sequence is one echo time); ``delta > Delta``; ``Delta + delta > TE``; and a row whose
    ``delta`` is zero, which states no diffusion time at all (three rows of ``ActiveAxG300_PM.scheme`` state
    ``Delta = delta = 0``, and what MC/DC plays for them -- nothing -- is not a PGSE this builder can express,
    so that file is refused entire rather than read with a guessed timing).
    """
    from ..sequences import pgse
    with open(path) as fh:
        text = fh.read()
    head, _, body = text.partition("\n")
    kind = head.strip().upper()
    if "STEJSKALTANNER" not in kind:
        raise ValueError(f"{path}: header {head.strip()!r} is not a STEJSKALTANNER scheme; MC/DC also writes "
                         f"APGSE (nine columns) and WAVEFORM (a sampled gradient), which are other layouts")
    tok = body.split()
    if len(tok) % 7:
        raise ValueError(f"{path}: {len(tok)} numbers is not a whole number of 7-column rows "
                         f"(gx gy gz |G| Delta delta TE)")
    A = np.asarray(tok, dtype=np.float64).reshape(-1, 7)
    if not len(A):
        raise ValueError(f"{path}: no measurements")
    dirs, G, Delta, delta, TE = A[:, :3], A[:, 3], A[:, 4], A[:, 5], A[:, 6]

    norm = np.linalg.norm(dirs, axis=1)
    bad = ~(np.isclose(norm, 1.0, atol=1e-3) | np.isclose(norm, 0.0, atol=1e-12))
    if bad.any():
        raise ValueError(f"{path}: row(s) {np.flatnonzero(bad).tolist()[:8]} have a direction of length "
                         f"{norm[bad][:8].tolist()}, neither unit nor zero")
    mismatch = np.isclose(norm, 0.0, atol=1e-12) & (G != 0)
    if mismatch.any():
        raise ValueError(f"{path}: row(s) {np.flatnonzero(mismatch).tolist()[:8]} carry an amplitude "
                         f"{G[mismatch][:8].tolist()} T/m along a zero direction")
    if len(np.unique(np.round(TE, 12))) != 1:
        raise ValueError(f"{path}: {len(np.unique(np.round(TE, 12)))} distinct TE "
                         f"{np.unique(np.round(TE, 12)).tolist()[:8]} s; one ScannerSequence is one echo time, "
                         f"so a multi-TE scheme is a Protocol and must be split by TE first")
    if (delta <= 0).any():
        i = np.flatnonzero(delta <= 0)
        raise ValueError(f"{path}: row(s) {i.tolist()[:8]} state delta = {delta[i][:8].tolist()} s. A zero "
                         f"pulse width states no diffusion time; MC/DC plays no gradient at all for such a row "
                         f"and the scheme does not say what timing the measurement had")
    if (delta > Delta).any():
        i = np.flatnonzero(delta > Delta)
        raise ValueError(f"{path}: row(s) {i.tolist()[:8]} have delta > Delta "
                         f"({delta[i][:4].tolist()} > {Delta[i][:4].tolist()} s): the lobes overlap")
    te = float(TE[0])
    if (Delta + delta > te + 1e-12).any():
        i = np.flatnonzero(Delta + delta > te + 1e-12)
        raise ValueError(f"{path}: row(s) {i.tolist()[:8]} need Delta + delta = "
                         f"{(Delta + delta)[i][:4].tolist()} s, longer than TE = {te} s")

    if n_t is None:
        n_t = _exact_n_t(te, Delta, delta, min_samples_per_delta=min_samples_per_delta)
    else:
        n_t = int(n_t)
        exact = _exact_n_t(te, Delta, delta)
        if (exact - 1) % (n_t - 1):
            raise ValueError(f"{path}: n_t = {n_t} does not put every delta, Delta and TE on a sample; the "
                             f"coarsest grid that does has n_t = {exact} (or a refinement of it)")
    if gradient_directions is None:
        gradient_directions = np.where(np.linalg.norm(dirs, axis=1, keepdims=True) > 0.5, dirs, [[0.0, 0.0, 1.0]])
    seq = pgse(gradient_directions, delta, Delta, gradient_strengths=G, TE=te, n_t=int(n_t), slew_rate=np.inf)
    import dataclasses
    shells = sorted({(float(g), float(dd), float(d)) for g, dd, d in zip(G, Delta, delta) if g != 0})
    return dataclasses.replace(seq, notes=f"{os.path.basename(path)}: {len(A)} measurements, "
                                          f"{int((G == 0).sum())} at b = 0, TE {te * 1e3:.2f} ms, "
                                          f"{len(shells)} shells (G, Delta, delta) {shells}")


def _exact_n_t(TE, Delta, delta, *, max_n_t=200_001, min_samples_per_delta=500):
    """The coarsest ``n_t`` that puts TE, every Delta, every delta AND every ``pad = (TE - Delta - delta)/2``
    on a sample, refined until the SHORTEST lobe holds ``min_samples_per_delta`` samples.

    Two conditions, and both are needed. Every time is written as a fraction of TE
    (``Fraction.limit_denominator``, so a file's six decimals are read as the exact ratio they state) and
    ``n_t - 1`` is the lowest common multiple of the denominators: that puts the lobe edges MC/DC integrates
    exactly onto grid points rather than inside a step. But a lattice can be edge-exact and still far too
    coarse -- 24 intervals over TE serves ``delta = TE/12`` with two samples -- and the builder samples a
    lobe mid-step, so the realised ``b`` is then percent-wrong. The lattice is therefore multiplied up until
    the shortest lobe is resolved. Measured on ``ActiveAxG140_PM.scheme``, which needs no refinement: b is the
    Stejskal-Tanner value of its own columns to 3.1e-5 relative.

    A scheme whose times share no coarse-enough lattice is refused rather than silently sampled off its edges.
    """
    from fractions import Fraction
    from math import lcm
    TE = float(TE)
    times = np.concatenate([np.asarray(Delta, float), np.asarray(delta, float),
                            (TE - np.asarray(Delta, float) - np.asarray(delta, float)) / 2.0])
    den = 1
    for t in np.unique(np.round(times / TE, 12)):
        if t <= 0:
            continue
        den = lcm(den, Fraction(float(t)).limit_denominator(100_000).denominator)
        if den >= max_n_t:
            raise ValueError(f"the scheme's times share no grid coarser than {den + 1} samples over TE = {TE} s; "
                             f"pass n_t= to sample it knowingly")
    shortest = float(np.min(np.asarray(delta, float)))
    k = max(1, int(np.ceil(min_samples_per_delta * TE / (shortest * den))))
    if den * k >= max_n_t:
        raise ValueError(f"resolving the shortest lobe ({shortest} s) with {min_samples_per_delta} samples on "
                         f"the edge-exact lattice of {den} intervals needs {den * k + 1} samples over "
                         f"TE = {TE} s; pass n_t= to sample it knowingly")
    return int(den * k) + 1


# ------------------------------------------------------------------------------------------- the raw signal
def read_bfloat(path, *, n_measurements=None):
    """MC/DC's raw ``*_DWI.bfloat``: the **unnormalised** signal, one ``float32`` per measurement.

    ``ParallelMCSimulation::writeDWSignal`` writes ``reinterpret_cast<char*>(&float)``, i.e. the host's own
    byte order, and every released file was written on x86: **little-endian**, which the values themselves
    confirm (read big-endian they are 1e-39 and 1e+27 denormals). Each entry is ``sum_walkers cos(phi)``, so
    a ``b = 0`` entry is exactly the walker count -- divide by it to normalise.

    ``n_measurements`` is checked against the file, never assumed.
    """
    nbytes = os.path.getsize(path)
    if nbytes % 4:
        raise ValueError(f"{path}: {nbytes} bytes is not a whole number of float32")
    S = np.fromfile(path, dtype="<f4").astype(np.float64)
    if n_measurements is not None and len(S) != int(n_measurements):
        raise ValueError(f"{path}: holds {len(S)} measurements, the scheme has {int(n_measurements)}")
    return S


def walker_count(S, b):
    """The walker count a raw signal was summed over: its ``b = 0`` entries, which must agree and be whole.

    MC/DC accumulates ``sum cos(phi)`` and a ``b = 0`` measurement has ``phi = 0`` for every walker, so the
    entry IS ``N``. Refused when the file has no ``b = 0`` entry or its ``b = 0`` entries disagree -- the count
    is then not in the file and must come from the configuration.
    """
    S, b = np.asarray(S, float), np.asarray(b, float)
    z = np.isclose(b, 0.0, atol=1e-6 * max(1.0, float(np.max(b))))
    if not z.any():
        raise ValueError("the signal has no b = 0 measurement, so the walker count is not in it")
    vals = np.unique(np.round(S[z], 6))
    if len(vals) != 1 or not np.isclose(vals[0], round(float(vals[0]))):
        raise ValueError(f"the b = 0 entries are {S[z].tolist()[:8]}, which is not one whole walker count")
    return int(round(float(vals[0])))


# ------------------------------------------------------------------------- the configuration and its walkers
class McdcConf(dict):
    """An MC/DC configuration file in SI, as a dict with attribute access."""
    __getattr__ = dict.__getitem__


_SCALARS = {"N": int, "T": int, "duration": float, "diffusivity": float, "scale_from_stu": int,
            "deportation": int, "sphere_size": float, "write_txt": int, "write_bin": int,
            "write_traj_file": int, "num_process": int, "permeability": float}
_PATHS = {"scheme_file", "ini_walker_file", "ini_walkers_file", "out_traj_file_index", "subdivision_file",
          "exp_prefix"}


def read_conf(path):
    """An MC/DC configuration file -> :class:`McdcConf` in SI.

    ``N``, ``T``, ``duration`` (s), ``diffusivity`` (m^2/s), ``scheme_file``, ``ini_walkers_file``, the
    ``<obstacle>`` block's ``ply`` and ``ply_scale``, and the ``<voxels>`` corners as ``voxel_min`` /
    ``voxel_max`` **in metres** -- the corners and ``ply_scale`` are written in MC/DC's millimetres and are not
    touched by ``scale_from_stu`` (``Parameters::readSchemeFile`` scales only the diffusivity and the
    duration), so the corners are multiplied by 1e-3 here and ``ply_scale`` is reported as the metres per PLY
    unit it realises, ``ply_scale * 1e-3``.

    A configuration without ``scale_from_stu 1`` is refused: its ``duration`` and ``diffusivity`` are then in
    MC/DC's own ms and mm^2/ms and this reader does not guess which.
    """
    out, plys, voxels = McdcConf(path=os.fspath(path)), [], []
    block, ply_scale = None, None
    for raw in open(path):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<"):
            tag = line.strip("<>/").lower()
            block = None if (line.startswith("</") or tag == "end") else tag
            continue
        f = line.split()
        key = f[0]
        if block == "voxels" or block == "voxel":
            voxels.append([float(x) for x in f])
            continue
        if len(f) < 2:
            continue
        if block == "obstacle" and key == "ply":
            plys.append(f[1])
        elif block == "obstacle" and key == "ply_scale":
            ply_scale = float(f[1])
        elif key in _SCALARS:
            out[key] = _SCALARS[key](float(f[1]))
        elif key in _PATHS:
            out[key] = f[1]
    if int(out.get("scale_from_stu", 0)) != 1:
        raise ValueError(f"{path}: no `scale_from_stu 1`, so its duration and diffusivity are in MC/DC's own "
                         f"ms and mm^2/ms; this reader reads SI configurations only")
    for k in ("N", "T", "duration", "diffusivity"):
        if k not in out:
            raise ValueError(f"{path}: no `{k}`")
    if len(voxels) == 2:
        out["voxel_min"] = np.asarray(voxels[0], float) * _MM
        out["voxel_max"] = np.asarray(voxels[1], float) * _MM
    elif voxels:
        raise ValueError(f"{path}: a <voxels> block with {len(voxels)} corner line(s); it states two")
    if plys:
        out["ply"] = plys if len(plys) > 1 else plys[0]
    if ply_scale is not None:
        out["ply_scale"] = float(ply_scale)
        out["mesh_scale"] = float(ply_scale) * _MM          # metres per PLY unit
    if "ini_walker_file" in out:                            # the spelling the released configurations use
        out["ini_walkers_file"] = out.pop("ini_walker_file")
    return out


def read_ini_walkers(path, *, scale=_MM):
    """MC/DC's initial-walker list -> ``(n, 3)`` metres. Three numbers per line in MC/DC's millimetres.

    ``DynamicsSimulation::initWalkerPosition`` reads this file **cyclically**: walker ``i`` starts at line
    ``i mod n``, so a run of more walkers than the file has lines repeats it. :func:`seed_positions` is that
    rule.
    """
    A = np.loadtxt(path, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"{path}: {A.shape} is not (n, 3) positions")
    return A * float(scale)


def seed_positions(ini, n_walkers):
    """``n_walkers`` start positions from an initial-walker list, MC/DC's own cyclic rule (``i mod n``)."""
    ini = np.asarray(ini, float)
    if not len(ini):
        raise ValueError("no initial-walker positions")
    return ini[np.arange(int(n_walkers)) % len(ini)]

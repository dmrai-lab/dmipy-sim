"""The quantity the Winther axon families reproduce, and the reader that gets the published one out of the
study's own released output.

Winther et al. 2024 (Sci Rep 14:29636) release, per axon and per configuration, the intra-axonal signal their
MC/DC walk produced on the SAME released surface meshes: ``in_silico/MCDC/G<n>-z_aligned/axon<NN>-inner-...
.signals``, a python pickle of torch tensors holding one number per measurement of a scheme their code also
releases. That array IS their measurement, so this family's reference is read from it and needs no printed line.

Three things live here, and nowhere else:

* :func:`read_signals` -- their ``.signals`` pickle, read with **numpy alone**. The file is a plain (non-zip)
  torch pickle whose tensors are views into shared storages; ``torch.load`` refuses it ("invalid magic number")
  and honouring the pickle by hand is what lets the published number be read on any box, including the one that
  runs the gate. Verified bit-identical to ``torch.load`` over all 116 released files.
* :func:`their_scheme` -- the released Stejskal-Tanner scheme as a
  :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` per shell: PGSE, delta 7.2 ms, Delta 20.2 ms,
  TE 36 ms, 21 directions on each of seven shells. Their lobes are ideal squares, so ``slew_rate`` is infinite
  and the leading-edge Delta of a Stejskal-Tanner scheme is the centre-to-centre Delta of ``pgse``.
* :func:`powder` and its two feeds -- the quantity: the **direction average over one shell's 21 directions of
  the intra-axonal signal**, reduced as ``mean(cos phi)`` over the intra walkers.

  ``mean(cos phi)`` is not ``|mean(exp(i phi))|``, and the difference is not small at these b: MC/DC
  accumulates ``DWI[s] += cos(phase_shift)`` (``PGSESequence::update_DWI_signal``), so their released numbers
  are real parts. Both sides of every comparison are reduced through
  :func:`~examples.validation.cross_engine_parity.ensemble_modulus`'s sibling path -- the per-walker phases of
  :func:`~examples.validation.cross_engine_parity.walk_cos_phi` for a walk and
  :func:`~examples.validation.cross_engine_parity.pack_walker_phase` for a pack -- so the estimator is one
  function and the codec is the only thing the ``reproduces-`` check can see.

The susceptibility ratio their Fig. 1b reports is read too (:func:`their_susceptibility_ratio`), because it is
what the G6 family reproduces and because its magnitude is the reason it is NOT this family's quantity: on
G1/G3/G4/G5 the published intra-axonal attenuation at b = 0 is 0.1 % to 0.5 % of the signal, an order of
magnitude under the floor a pack of this size certifies, so a gate on it would compare two numbers that are
both 1 and say nothing. The family's card states the published values and the reason.

::

    PYTHONPATH=<dmipy-sim> python examples/validation/winther_axons.py --signals <their_signals_ns7> \\
      --scheme <ns7.scheme> --out <work>/published.json        # their numbers, per axon and configuration
"""
import argparse
import json
import os
import pickle
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

#: The angles, in degrees, their seven B-field solves put between B0 and the axon's axis (the ``B_z_angle`` in
#: radians of the field file each ``only_susceptibility_gradients`` key names).
THEIR_ANGLES_DEG = (0, 15, 30, 45, 60, 75, 90)

#: The storage classes of their pickles, as numpy dtypes. Their signals are doubles; the rest are here because
#: a storage class this reader does not know is a refusal rather than a silent misread.
_STORAGE_DTYPES = {"DoubleStorage": np.dtype("<f8"), "FloatStorage": np.dtype("<f4"),
                   "HalfStorage": np.dtype("<f2"), "LongStorage": np.dtype("<i8"),
                   "IntStorage": np.dtype("<i4"), "ShortStorage": np.dtype("<i2"),
                   "CharStorage": np.dtype("i1"), "ByteStorage": np.dtype("u1"),
                   "BoolStorage": np.dtype("?")}


class _Storage:
    """One torch storage: its element type, and the bytes the payload section fills in."""

    def __init__(self, dtype):
        self.dtype, self.data, self.numel = dtype, None, None


def _legacy_tensor_payload(blob):
    """The object a torch **legacy** (pre-zip) serialisation holds, with every storage's bytes read.

    The layout is five pickles followed by the raw data: a magic number, a protocol version, the host's type
    sizes, the object itself (whose storages are persistent ids), the list of storage keys in payload order,
    and then, per key, a little-endian int64 element count and that many elements. Nothing here interprets a
    device: a storage saved from ``cuda:0`` is the same bytes.
    """
    fh = io_of(blob)
    storages = {}

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if name in _STORAGE_DTYPES:
                return lambda dt=_STORAGE_DTYPES[name]: _Storage(dt)
            raise pickle.UnpicklingError(
                f"{module}.{name} is not a storage class this reader accepts; a Winther .signals pickle holds "
                f"torch storages and nothing else")

        def persistent_load(self, pid):
            kind, cls, key, _device, numel = pid[0], pid[1], pid[2], pid[3], pid[4]
            if kind != "storage":
                raise pickle.UnpicklingError(f"a persistent id of kind {kind!r} is not a storage")
            s = storages.setdefault(key, cls())
            s.numel = int(numel)
            return s

    for _ in range(4):                            # magic number, protocol version, type sizes, the object
        obj = _U(fh).load()
    keys = _U(fh).load()
    for key in keys:
        (n,) = struct.unpack("<q", fh.read(8))
        s = storages[key]
        raw = fh.read(n * s.dtype.itemsize)
        if len(raw) != n * s.dtype.itemsize:
            raise pickle.UnpicklingError(f"storage {key!r} declares {n} elements and the payload holds "
                                         f"{len(raw)} bytes; the file is truncated")
        s.data = np.frombuffer(raw, dtype=s.dtype).copy()
    return obj


def io_of(blob):
    import io
    return io.BytesIO(blob)


def read_signals(path):
    """Their ``.signals`` pickle as plain numpy: ``{key: array or {field file: array}}``.

    Tensors come back as ``float64`` arrays. Several of them are views at different offsets into ONE storage --
    their seven field orientations share a 7 x n_meas buffer -- and the offset and stride of each view are
    honoured, which is what makes the seven orientations distinct rather than seven copies of the first.
    """
    def rebuild(storage, offset, size, stride, *_rest):
        if storage.data is None:
            raise pickle.UnpicklingError("a tensor references a storage the payload never filled")
        strides = tuple(int(x) * storage.dtype.itemsize for x in stride)
        return np.lib.stride_tricks.as_strided(storage.data[int(offset):], tuple(int(x) for x in size),
                                               strides).astype(np.float64)

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            if (module, name) == ("torch.storage", "_load_from_bytes"):
                return _legacy_tensor_payload
            if (module, name) == ("collections", "OrderedDict"):
                return dict
            if module.startswith("numpy"):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"{module}.{name} is not something a Winther .signals pickle is made of; this reader accepts "
                f"torch tensors, numpy scalars and dicts, so that reading a published number executes nothing")

    with open(path, "rb") as fh:
        return _U(fh).load()


def their_S0(signals):
    """The number their normalisation divides by, which is also their walker count: ``.signals`` holds SUMS of
    ``cos(phase)`` over their intra particles, so the b = 0 entry is exactly how many they walked."""
    return float(np.asarray(signals["only_diffusion_gradients"], np.float64)[0])


def their_diffusion_signal(signals):
    """Their normalised intra-axonal signal per measurement of the scheme, ``(n_meas,)``."""
    s = np.asarray(signals["only_diffusion_gradients"], np.float64)
    return s / s[0]


def their_susceptibility_ratio(signals):
    """``{angle in degrees: S_susc / S_0}`` at b = 0: their Fig. 1b, read off the released arrays.

    The key of each orientation is the path of the B-field file it used, whose ``B_z_angle`` in radians is the
    angle between B0 and the axon's axis; the entries of one orientation are all the same number, because with
    no diffusion gradient every measurement of the scheme is the same b = 0 spin echo.
    """
    import re
    s0 = np.asarray(signals["only_diffusion_gradients"], np.float64)[0]
    out = {}
    for key, value in signals["only_susceptibility_gradients"].items():
        m = re.search(r"B_z_angle=(\d+\.\d+)", str(key))
        if m is None:
            raise ValueError(f"the orientation key {str(key)[:120]!r} states no B_z_angle")
        out[int(round(np.degrees(float(m.group(1)))))] = float(np.asarray(value, np.float64)[0] / s0)
    if sorted(out) != sorted(THEIR_ANGLES_DEG):
        raise ValueError(f"the released orientations are {sorted(out)}, not {sorted(THEIR_ANGLES_DEG)}")
    return out


# ------------------------------------------------------------------------------------- their acquisition
def read_scheme(path):
    """Their Stejskal-Tanner scheme as ``dict(dirs, G, Delta, delta, TE, b)``, one row per measurement.

    The columns are the unit gradient direction, ``|G|`` in T/m, ``Delta`` and ``delta`` in seconds and ``TE``
    in seconds; ``b = (gamma |G| delta)^2 (Delta - delta/3)`` is the Stejskal-Tanner b the rows encode, in
    s/m^2. A b = 0 row carries zeros for everything but ``TE``.
    """
    from dmipy_sim.constants import GAMMA
    rows = [line.split() for line in open(path) if line.strip() and not line.startswith("VERSION")]
    a = np.asarray(rows, np.float64)
    if a.shape[1] != 7:
        raise ValueError(f"{path}: {a.shape[1]} columns; a STEJSKALTANNER scheme has seven")
    G, Delta, delta = a[:, 3], a[:, 4], a[:, 5]
    return dict(dirs=a[:, :3], G=G, Delta=Delta, delta=delta, TE=a[:, 6],
                b=(GAMMA * G * delta) ** 2 * (Delta - delta / 3.0))


def shells_of(scheme, *, tol=1e6):
    """The distinct b of ``scheme`` (s/m^2), rounded to ``tol``, with the row mask of each: ``{b: mask}``."""
    b = np.round(np.asarray(scheme["b"], np.float64) / tol) * tol
    return {float(v): (b == v) for v in np.unique(b)}


def their_sequence(scheme, b, *, n_t, tol=1e6):
    """The one shell of ``scheme`` at ``b`` as a ``ScannerSequence`` on ``n_t`` samples: their 21 directions at
    their delta, Delta and TE, with ideal square lobes.

    Their gradients are switched instantaneously (MC/DC plays a Stejskal-Tanner pair with no ramp), so the slew
    rate is infinite; a finite one would move the b of every row and the comparison would be with another
    acquisition. ``b`` selects the shell, and the sequence's own ``b()`` is what the design and pack stages
    check against the envelope and the band.
    """
    from dmipy_sim import pgse
    mask = shells_of(scheme, tol=tol)[float(b)]
    if not mask.any():
        raise ValueError(f"the scheme has no shell at b = {b:g} s/m^2; it has {sorted(shells_of(scheme, tol=tol))}")
    delta = float(np.unique(scheme["delta"][mask]).item())
    Delta = float(np.unique(scheme["Delta"][mask]).item())
    TE = float(np.unique(scheme["TE"][mask]).item())
    return pgse(scheme["dirs"][mask], delta, Delta, bvalues=[float(b)] * int(mask.sum()), TE=TE, n_t=int(n_t),
                slew_rate=np.inf)


def their_powder(signals, scheme, b, *, tol=1e6):
    """Their direction-averaged intra-axonal signal on the shell at ``b``: the quantity, from their own array."""
    return float(their_diffusion_signal(signals)[shells_of(scheme, tol=tol)[float(b)]].mean())


# ------------------------------------------------------------------------------------- our side of it
def powder(cos_phi, weights):
    """``(value, relative standard error)`` of the direction average, from per-walker cosines.

    ``cos_phi`` is ``(n_meas, n_w)`` -- one shell's measurements -- and the powder is the weighted mean over
    walkers of each walker's OWN direction average. That makes it an ensemble mean of a per-walker statistic,
    so its standard error is analytic and is the weighted one: no resampling, and the 21 directions' complete
    correlation across walkers is carried exactly instead of being assumed away.
    """
    p = np.asarray(cos_phi, np.float64).mean(axis=0)
    w = np.asarray(weights, np.float64)
    if p.shape != w.shape:
        raise ValueError(f"{p.shape[0]} walkers in the cosines and {w.shape[0]} weights")
    W = w.sum()
    value = float((w * p).sum() / W)
    var = float((w ** 2 * (p - value) ** 2).sum()) / W ** 2      # the weighted mean's own variance
    return value, float(np.sqrt(var) / abs(value))


SE_DERIVATION = ("the analytic standard error of a weighted ensemble mean: the quantity is the weighted mean "
                 "over intra walkers of each walker's direction average of cos(phi) over the shell's 21 "
                 "directions, so sqrt(sum w^2 (p_w - P)^2) / sum w, divided by P. Nothing is resampled and the "
                 "directions' correlation across walkers is carried rather than assumed away")


INTRA = 1                     #: the pool id ``winther_spec`` gives the axonal water, the one their walk seeds


def walk_intra(walk):
    """``(mask, weights)`` of the intra pool of a walk, from its compartment channel at the first save.

    Their walk seeds intra only (``ini_walker_pos intra``); ours seeds intra AND a frozen myelin pool, so the
    quantity is taken over the intra walkers alone. The membranes are impermeable and the myelin pool is frozen,
    so the label at the first save is the label throughout and reading it there costs one column.
    """
    if walk.compartment is None:
        raise ValueError("this walk recorded no compartment channel, so its intra pool cannot be selected")
    m = np.asarray(walk.compartment)[:, 0].round().astype(int) == INTRA
    w = np.ones(walk.n_walkers) if walk.weights is None else np.asarray(walk.weights, np.float64)
    return m, w[m]


def walk_powder(walk, scheme, b, *, tol=1e6):
    """The quantity from the walk itself, at the walk's full resolution: ``(value, relative se, n_intra)``.

    This is the DIRECT measurement the gate holds the replay to. It reads the walk's own positions through the
    engine's own gradient phase, so the only thing between it and the replayed number is the position codec.
    """
    from examples.validation.cross_engine_parity import walk_cos_phi       # the phase and its reduction live once
    seq = their_sequence(scheme, b, n_t=walk.positions.shape[1], tol=tol)
    m, w = walk_intra(walk)
    value, se = powder(walk_cos_phi(walk, seq)[:, m], w)
    return value, se, int(m.sum())


def pack_powder(pack, scheme, b, *, tol=1e6):
    """The quantity from the pack: ``(value, relative se, n_intra)``, the same estimator on the decoded bands.

    ``walker_phases(..., compartment=INTRA)`` is the pack's own selection: with no tissue the effective weights
    it returns ARE the statistical weights, zero outside the pool, so the powder is the same weighted mean the
    walk's is and the difference between the two numbers is the position codec and nothing else.
    """
    seq = their_sequence(scheme, b, n_t=pack.n_t, tol=tol)
    _w, ew, phi = pack.walker_phases(seq, compartment=INTRA)
    value, se = powder(np.cos(np.asarray(phi, np.float64)).T, ew)
    return value, se, int((np.asarray(ew) > 0).sum())


def walk_field_along(walk, b0_dir, *, B0, chi_iso, chi_aniso=0.0, chunk=2000):
    """``(n_w, n_t)`` field in tesla along every walker's path of ``walk``, in walker chunks.

    The walk's OWN samples when it took them (the mean of the field basis over each save interval, so the
    sub-step motion inside a save is integrated rather than aliased), else the basis evaluated at the saved
    positions -- which is the rule the pack's own path channel is encoded and certified by
    (``replay.bank._raw_field``), so the direct number and the replayed one differ by the codec and not by which
    field either of them read.
    """
    from dmipy_sim.fields.hollow_cylinder import contract
    if walk.field_samples is not None:
        return contract(np.asarray(walk.field_samples, np.float64), b0_dir, B0=float(B0), chi_iso=float(chi_iso),
                        chi_aniso=float(chi_aniso))
    if walk.field_basis is None:
        raise ValueError("this walk carries neither field samples nor a field basis, so it has no field tier")
    pos = walk.positions
    n_w, n_t = pos.shape[0], pos.shape[1]
    out = np.empty((n_w, n_t), np.float64)
    for lo in range(0, n_w, chunk):
        p = np.asarray(pos[lo:lo + chunk], np.float64)
        out[lo:lo + p.shape[0]] = walk.field_basis.field(p.reshape(-1, 3), b0_dir, B0=float(B0),
                                                        chi_iso=float(chi_iso),
                                                        chi_aniso=float(chi_aniso)).reshape(-1, n_t)
    return out


def walk_susceptibility_ratio(walk, *, angles_deg=THEIR_ANGLES_DEG, B0=7.0, chi_iso=1.06e-6, chi_aniso=0.0):
    """``{angle: S_susc / S_0}`` at b = 0 under a spin echo, from the walk's own field.

    Their Fig. 1b quantity, measured on the direct walk: the field along each intra walker's path contracted at
    each orientation, gated by the sequence's own sign, and reduced as ``mean(cos phi)``. At b = 0 with no
    relaxation the denominator is exactly one, so the ratio is the numerator.
    """
    from dmipy_sim import pgse
    from dmipy_sim.constants import GAMMA
    from dmipy_sim.replay._replay_kernel import field_gate
    n_t, dt = walk.positions.shape[1], float(walk.dt)
    seq = pgse([[0.0, 0.0, 1.0]], 5e-3, 20e-3, bvalues=[0.0], TE=float((n_t - 1) * dt), n_t=n_t, slew_rate=np.inf)
    gate = field_gate(seq, n_t, dt)
    m, w = walk_intra(walk)
    out = {}
    for theta in angles_deg:
        t = np.radians(float(theta))
        d = [float(np.sin(t)), 0.0, float(np.cos(t))]
        dB = walk_field_along(walk, d, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)[m]
        phi = GAMMA * dt * (dB * gate[None, :]).sum(1)
        out[int(theta)] = float((w * np.cos(phi)).sum() / w.sum())
    return out


# ------------------------------------------------------------------------------------- their numbers, as a file
def published(signals_dir, scheme_path, configs=("G1", "G3", "G4", "G5"), axons=None, b=3e9):
    """Every released number this family's reference rests on, read from the released files.

    ``{f"{config}/axon{NN}": dict(powder, ratio, S0, n_meas, sha256, file)}``: the direction-averaged intra
    signal on the shell at ``b``, their Fig. 1b ratio per orientation, their walker count, and the digest of
    the file each came from -- which is what the reference record cites, since a released array is the
    measurement and its digest is its identity.
    """
    from dmipy_sim.fill.hub import sha256_of
    scheme = read_scheme(scheme_path)
    axons = list(axons or AXONS)
    out = {}
    for config in configs:
        for axon in axons:
            path = os.path.join(signals_dir, config, f"axon{axon}.signals")
            s = read_signals(path)
            out[f"{config}/axon{axon}"] = dict(
                powder=their_powder(s, scheme, b), b=float(b), ratio=their_susceptibility_ratio(s),
                S0=their_S0(s), n_meas=int(np.asarray(s["only_diffusion_gradients"]).size),
                scheme_file=str(s["path_scheme_file"]).split("/")[-1],
                file=os.path.relpath(path, signals_dir), sha256=sha256_of(path))
    return out


#: The 29 axons of the paper's own white list (``production/1b``, ``white_list``), which is what "29 axons"
#: means in the study and therefore in this family.
AXONS = ("06", "08", "12", "13", "14", "15", "18", "22", "24", "25", "26", "27", "31", "32", "34", "38", "40",
         "41", "43", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--signals", required=True, help="the directory holding <config>/axon<NN>.signals")
    p.add_argument("--scheme", required=True, help="their released .scheme file")
    p.add_argument("--out", required=True)
    p.add_argument("--b", type=float, default=3e9, help="the shell the quantity is averaged over, s/m^2")
    a = p.parse_args(argv)
    rows = published(os.path.abspath(a.signals), os.path.abspath(a.scheme), b=a.b)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)
    v = [r["powder"] for r in rows.values()]
    print(f"{len(rows)} axons; their powder at b = {a.b:g}: min {min(v):.5f} max {max(v):.5f}")
    print(f"their walker counts: min {min(r['S0'] for r in rows.values()):.0f} "
          f"max {max(r['S0'] for r in rows.values()):.0f}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()

"""The far grid's crop as a truncated, shell-quantized spectrum (dmrai-lab/tessera#49 step 6): ``far_spec.npz``.

A block kit carries the far-field crop ``V`` (float16, shape ``(nx, ny, nz, 12)``, the 12 channels of
:data:`FIELD_CHANNELS`) so the walker can sample the susceptibility field away from the strands. The field is
smooth at the scale of the walk, so the crop is stored as the Fourier coefficients of its wavenumbers up to
``k_cut = 2 pi / wavelength_min_m``, quantized per shell the way the replay pack's container quantizes bands:

``shape``     int64 ``(3,)``      ``(nx, ny, nz)``;
``spacing_m`` float64             the node spacing;
``k_cut``     float64             the wavenumber kept;
``n_shells``  int64               equal-width shells of ``|k|`` in ``[0, k_cut]``;
``k16``       float64             shells whose lower edge is below ``k16`` are stored at 16 bit, the rest at 8;
``mean``      float32 ``(11,)``   per stored channel, the mean of ``V`` over the crop;
``scale``     float32 ``(n_shells, 11)``  per shell and channel the max of ``|re|`` and ``|im|`` over the shell's
                                  coefficients (0 for an empty shell);
``q16``       int16 ``(n16, 11, 2)``  (re, im) of the 16-bit shells' coefficients scaled by ``scale`` to ``[-32767, 32767]``;
``q8``        int8 ``(n8, 11, 2)``    the same for the 8-bit shells, to ``[-127, 127]``.

The stored channels are :data:`STORED` (11 of the 12): ``iso_local`` is zero in the far part (:func:`pack_far`
refuses a crop where it is not) and is rebuilt as zero (``iso_P_zz`` is not a channel: ``3 iso_local - xx - yy``, formed by
the reader). There is no index:
the kept set is the wavenumbers ``|k| <= k_cut`` of the ``rfftn`` grid ``(nx, ny, nz // 2 + 1)`` (``k = 2 pi
fftfreq / spacing`` on axes 0 and 1, ``rfftfreq`` on axis 2); the coefficients run by increasing shell
``floor(|k| / k_cut * n_shells)`` (clipped to ``n_shells - 1``) and within a shell in C order of that grid.

:func:`unpack_far` rebuilds the mask and shells, dequantizes (``q * scale / 32767`` or ``/ 127``), scatters, inverts
with ``irfftn(s=shape)``, adds the mean, rebuilds ``iso_local`` and casts to float16. It uses numpy only, so a
node without dmipy-sim ports it verbatim."""
from __future__ import annotations

import io
import zipfile

import numpy as np

FIELD_CHANNELS = ("iso_local", "iso_P_xx", "iso_P_yy", "iso_P_xy", "iso_P_xz", "iso_P_yz",
                  "aniso_G_xx", "aniso_G_yy", "aniso_G_zz", "aniso_G_xy", "aniso_G_xz", "aniso_G_yz")
STORED = tuple(c for c in FIELD_CHANNELS if c != "iso_local")
_STORED_IDX = [FIELD_CHANNELS.index(c) for c in STORED]
_LOC = FIELD_CHANNELS.index("iso_local")
N_SHELLS = 16
K16 = 2.0 * np.pi / 20e-6


def _layout(shape, spacing_m, k_cut, n_shells, k16):
    """``(index, shell, n16)``: the flat indices of the kept coefficients in storage order, their shells, and how
    many belong to the 16-bit shells."""
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=spacing_m)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=spacing_m)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=spacing_m)
    k = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).ravel()
    index = np.flatnonzero(k <= float(k_cut))
    shell = np.minimum(np.floor(k[index] / float(k_cut) * n_shells).astype(np.int64), n_shells - 1)
    order = np.argsort(shell, kind="stable")                  # by shell, C order within it
    index, shell = index[order].astype(np.int64), shell[order]
    wide = np.arange(n_shells) * float(k_cut) / n_shells < float(k16)
    return index, shell, int(np.count_nonzero(wide[shell]))


def pack_far(values, spacing_m, wavelength_min_m=10e-6, *, n_shells=N_SHELLS, k16=K16):
    """The quantized truncated-spectrum form (module docstring) of the far crop ``values`` on a grid of node spacing
    ``spacing_m``, keeping wavelengths down to ``wavelength_min_m``. Refuses a crop whose ``iso_local`` is not zero."""
    v = np.asarray(values)
    nx, ny, nz, c = v.shape
    if c != len(FIELD_CHANNELS):
        raise ValueError(f"the far crop has {c} channels, expected {len(FIELD_CHANNELS)}")
    peak = float(np.abs(v[..., _LOC].astype(np.float32)).max())
    if peak > 0:
        raise ValueError(f"iso_local is not zero on the crop (max |iso_local| = {peak:g}): far_spec does not store it")
    k_cut = 2.0 * np.pi / float(wavelength_min_m)
    index, shell, n16 = _layout((nx, ny, nz), spacing_m, k_cut, n_shells, k16)
    mean = np.empty(len(STORED), np.float32); coef = np.empty((index.size, len(STORED)), np.complex64)
    for j, ch in enumerate(_STORED_IDX):                      # one channel at a time: the spectrum of all never coexists
        x = v[..., ch].astype(np.float32)
        mean[j] = x.mean(dtype=np.float64)
        coef[:, j] = np.fft.rfftn(x - mean[j], axes=(0, 1, 2)).reshape(-1)[index]
    mag = np.maximum(np.abs(coef.real), np.abs(coef.imag)).astype(np.float32)
    scale = np.zeros((n_shells, len(STORED)), np.float32)
    np.maximum.at(scale, shell, mag)
    sc = scale[shell]
    ri = np.stack([coef.real, coef.imag], axis=-1)            # (n, 11, 2)
    unit = np.divide(ri, sc[..., None], out=np.zeros_like(ri), where=sc[..., None] > 0)
    q16 = np.rint(unit[:n16] * 32767).astype(np.int16)
    q8 = np.rint(unit[n16:] * 127).astype(np.int8)
    return dict(shape=np.array([nx, ny, nz], np.int64), spacing_m=np.float64(spacing_m), k_cut=np.float64(k_cut),
                n_shells=np.int64(n_shells), k16=np.float64(k16), mean=mean, scale=scale, q16=q16, q8=q8)


def unpack_far(npz):
    """The float16 crop ``(nx, ny, nz, 12)`` of a packed far field (:func:`pack_far`'s dict or the loaded ``.npz``)."""
    shape = tuple(int(s) for s in npz["shape"]); nx, ny, nz = shape
    n_shells = int(npz["n_shells"])
    index, shell, n16 = _layout(shape, float(npz["spacing_m"]), float(npz["k_cut"]), n_shells, float(npz["k16"]))
    scale = np.asarray(npz["scale"], np.float32)
    sc = scale[shell][..., None]
    ri = np.concatenate([np.asarray(npz["q16"], np.float32) / np.float32(32767), np.asarray(npz["q8"], np.float32) / np.float32(127)]) * sc
    mean = np.asarray(npz["mean"], np.float32)
    out = np.empty(shape + (len(FIELD_CHANNELS),), np.float32)
    F = np.zeros((nx, ny, nz // 2 + 1), np.complex64)
    for j, ch in enumerate(_STORED_IDX):
        F.reshape(-1)[index] = (ri[:, j, 0] + 1j * ri[:, j, 1]).astype(np.complex64)
        out[..., ch] = np.fft.irfftn(F, s=shape, axes=(0, 1, 2)).astype(np.float32) + mean[j]
    out[..., _LOC] = 0
    return out.astype(np.float16)


def save_far(path, packed):
    """``packed`` written to ``path`` as an uncompressed ``.npz`` with fixed member timestamps, so the same field
    always gives the same bytes (and the same sha256)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in ("shape", "spacing_m", "k_cut", "n_shells", "k16", "mean", "scale", "q16", "q8"):
            buf = io.BytesIO(); np.lib.format.write_array(buf, np.asarray(packed[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())
    return path

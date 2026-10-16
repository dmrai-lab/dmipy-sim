"""The far grid's crop as a truncated spectrum (dmrai-lab/tessera#49 step 6): ``far_spec.npz``.

A block kit carries the far-field crop ``V`` (float16, shape ``(nx, ny, nz, 13)``) so the walker can sample the
susceptibility field away from the strands. The field is smooth at the scale of the walk, so the crop is stored as
the Fourier coefficients of its wavenumbers up to ``k_cut = 2 pi / wavelength_min_m``:

``mean``  float32 ``(13,)``       the per-channel mean of ``V`` over the crop;
``shape`` int64 ``(3,)``          ``(nx, ny, nz)``;
``k_cut`` float64                 the wavenumber kept;
``index`` int64 ``(n_kept,)``     flat C-order indices into the ``rfftn`` array of shape ``(nx, ny, nz // 2 + 1)`` of
                                  the coefficients with ``|k| <= k_cut``, ``k = 2 pi fftfreq / spacing`` per axis
                                  (``rfftfreq`` on the last);
``coef``  complex64 ``(n_kept, 13)``  ``rfftn(V - mean, axes=(0, 1, 2))[index]`` per channel.

:func:`unpack_far` scatters the coefficients into zeros, inverts with ``irfftn(s=shape)``, adds the mean and casts to
float16. It uses numpy only, so a node without dmipy-sim ports it verbatim."""
from __future__ import annotations

import io
import zipfile

import numpy as np

CHANNELS = 13


def _kept(shape, spacing_m, k_cut):
    """Flat indices (C order, over the ``rfftn`` grid ``(nx, ny, nz // 2 + 1)``) of the wavenumbers ``|k| <= k_cut``."""
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=spacing_m)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=spacing_m)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=spacing_m)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    return np.flatnonzero((k2 <= float(k_cut) ** 2).ravel()).astype(np.int64)


def pack_far(values, spacing_m, wavelength_min_m=10e-6):
    """The truncated-spectrum form (module docstring) of the far crop ``values`` on a grid of node spacing
    ``spacing_m``, keeping wavelengths down to ``wavelength_min_m``."""
    v = np.asarray(values)
    nx, ny, nz, c = v.shape
    mean = v.astype(np.float32).mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    k_cut = 2.0 * np.pi / float(wavelength_min_m)
    index = _kept((nx, ny, nz), spacing_m, k_cut)
    coef = np.empty((index.size, c), np.complex64)
    for ch in range(c):                                       # one channel at a time: the spectrum of all 13 never coexists
        F = np.fft.rfftn(v[..., ch].astype(np.float32) - mean[ch], axes=(0, 1, 2))
        coef[:, ch] = F.reshape(-1)[index]
    return dict(mean=mean, shape=np.array([nx, ny, nz], np.int64), k_cut=np.float64(k_cut), index=index, coef=coef)


def unpack_far(npz):
    """The float16 crop ``(nx, ny, nz, 13)`` of a packed far field (:func:`pack_far`'s dict or the loaded ``.npz``)."""
    mean, shape, index, coef = npz["mean"], npz["shape"], npz["index"], npz["coef"]
    nx, ny, nz = (int(s) for s in shape)
    F = np.zeros((nx, ny, nz // 2 + 1, coef.shape[1]), np.complex64)
    F.reshape(-1, coef.shape[1])[index] = coef
    v = np.fft.irfftn(F, s=(nx, ny, nz), axes=(0, 1, 2)).astype(np.float32) + np.asarray(mean, np.float32)
    return v.astype(np.float16)


def save_far(path, packed):
    """``packed`` written to ``path`` as an uncompressed ``.npz`` with fixed member timestamps, so the same field
    always gives the same bytes (and the same sha256)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in ("mean", "shape", "k_cut", "index", "coef"):
            buf = io.BytesIO(); np.lib.format.write_array(buf, np.asarray(packed[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())
    return path

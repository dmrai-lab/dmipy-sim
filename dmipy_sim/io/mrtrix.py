"""Reader for the MRtrix image format (``.mif`` / ``.mih``): what a CSD result and a tissue segmentation come as.

Only what a replay phantom needs: the image data in its **canonical** axis order (the order the header's
``dim`` and ``transform`` refer to, whatever the storage ``layout``), the voxel-to-scanner affine in
millimetres (the NIfTI convention, so :meth:`dmipy_sim.phantom.Grid.from_affine` reads it), and the header.
MRtrix's ``transform`` is the rotation-and-translation of the **canonical** axes; the affine is that with the
voxel sizes folded in, ``affine[:3, :3] = transform[:3, :3] @ diag(vox)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["MifImage", "read_mif"]

_DTYPES = {"Float32LE": "<f4", "Float32BE": ">f4", "Float64LE": "<f8", "Float64BE": ">f8",
           "Int8": "i1", "UInt8": "u1", "Int16LE": "<i2", "UInt16LE": "<u2", "Int16BE": ">i2", "UInt16BE": ">u2",
           "Int32LE": "<i4", "UInt32LE": "<u4", "Int32BE": ">i4", "UInt32BE": ">u4",
           "Int64LE": "<i8", "UInt64LE": "<u8"}


@dataclass(frozen=True)
class MifImage:
    data: np.ndarray          # canonical axis order, sign of every axis positive
    affine: np.ndarray        # (4, 4): voxel index (canonical) -> scanner coordinate, millimetres
    vox: tuple                # voxel size per canonical axis, millimetres (NaN for a non-spatial axis)
    header: dict              # every header key, values as strings (lists when repeated)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dw_scheme(self):
        """``(n, 4)`` gradient table ``(x, y, z, b)`` in scanner coordinates, or None."""
        rows = self.header.get("dw_scheme")
        if rows is None:
            return None
        rows = rows if isinstance(rows, list) else [rows]
        return np.array([[float(v) for v in r.split(",")] for r in rows])


def read_mif(path):
    """Read a ``.mif`` (data in the same file) or ``.mih`` (data in the file it names)."""
    path = Path(path)
    raw = path.read_bytes() if path.suffix == ".mif" else path.read_bytes()
    end = raw.find(b"\nEND\n")
    if end < 0 or not raw.startswith(b"mrtrix image"):
        raise ValueError(f"{path} is not an MRtrix image (no 'mrtrix image' magic / END marker)")
    header = {}
    for line in raw[:end].decode("utf-8", "replace").splitlines()[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k in header:
            header[k] = (header[k] if isinstance(header[k], list) else [header[k]]) + [v]
        else:
            header[k] = v
    dim = [int(v) for v in header["dim"].split(",")]
    n = len(dim)
    vox = tuple(float(v) for v in header["vox"].split(","))
    layout = [tok.strip() for tok in header["layout"].split(",")]
    if len(layout) != n:
        raise ValueError(f"layout {header['layout']!r} does not match dim {dim}")
    sign = [1 if tok[0] == "+" else -1 for tok in layout]
    rank = [int(tok[1:]) for tok in layout]                       # 0 = fastest varying
    if sorted(rank) != list(range(n)):
        raise ValueError(f"layout {header['layout']!r} is not a permutation")
    fspec = header["file"].split()
    fname, offset = fspec[0], int(fspec[1]) if len(fspec) > 1 else 0
    blob = raw if fname == "." else (path.parent / fname).read_bytes()
    count = int(np.prod(dim))
    if header.get("datatype") == "Bit":                          # a mask: eight voxels per byte, least significant first
        packed = np.frombuffer(blob, dtype=np.uint8, count=(count + 7) // 8, offset=offset)
        flat = np.unpackbits(packed, bitorder="little")[:count].astype(bool)
        dtype = "?"
    else:
        dtype = _DTYPES.get(header.get("datatype", ""))
        if dtype is None:
            raise ValueError(f"datatype {header.get('datatype')!r} is not supported here; convert with mrconvert -datatype float32")
        flat = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
    # storage is C-order with the rank-0 axis last (fastest): shape by descending rank
    order = sorted(range(n), key=lambda ax: rank[ax], reverse=True)     # canonical axes, slowest first
    arr = flat.reshape([dim[ax] for ax in order])
    pos = {ax: i for i, ax in enumerate(order)}
    arr = arr.transpose([pos[ax] for ax in range(n)])
    for ax in range(n):
        if sign[ax] < 0:
            arr = np.flip(arr, axis=ax)
    data = np.ascontiguousarray(arr)
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    if "scaling" in header:
        off, scale = (float(v) for v in header["scaling"].split(","))
        data = data * scale + off
    rows = header.get("transform")
    rows = rows if isinstance(rows, list) else [rows]
    T = np.array([[float(v) for v in r.split(",")] for r in rows], np.float64)
    affine = np.eye(4)
    affine[:3, :3] = T[:3, :3] @ np.diag(vox[:3])
    affine[:3, 3] = T[:3, 3]
    return MifImage(data, affine, vox, header)

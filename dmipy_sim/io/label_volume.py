"""Readers for segmented 3-D images: a label grid, its voxel size in metres and its origin.

A label volume is the native form of half the substrates a diffusion simulation is asked for --
micro-CT rocks, segmented electron microscopy, vessel and placenta masks -- and each community
distributes it in its own container. The five here cover them: NRRD (detached ``.nhdr`` + ``.raw``
or single-file ``.nrrd``), MetaImage (``.mhd`` + ``.raw``), NIfTI, a TIFF stack (a multi-page file
or a directory of slices) and HDF5.

Every reader returns the same :class:`LabelVolumeFile`: ``labels`` as ``uint8`` in ``(nx, ny, nz)``
index order, ``voxel_size`` as three lengths in **metres**, ``origin`` as the position of the
``(0, 0, 0)`` voxel's lower corner in metres. A geometry cannot be built without the voxel size, so
a header that does not state one is **refused** by name; ``voxel_size=`` supplies it for the formats
whose containers carry no spacing (a bare TIFF stack), and then overrides nothing, because a header
that states a size and a caller who states a different one is also refused.

Nothing here constructs a geometry: :class:`~dmipy_sim.geometry.label_volume.LabelVolume` takes
these three arrays, and :func:`~dmipy_sim.spec.label_volume_spec` writes the spec that cites the file.
"""
from __future__ import annotations

import os
import re
from typing import NamedTuple

import numpy as np

#: Length units a header may name, in metres.
UNITS_M = {"m": 1.0, "meter": 1.0, "metre": 1.0, "meters": 1.0, "metres": 1.0,
           "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
           "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "micron": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6,
           "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9}

#: Reader per file suffix.
FORMATS = {".nhdr": "nrrd", ".nrrd": "nrrd", ".mhd": "mhd", ".mha": "mhd",
           ".nii": "nifti", ".gz": "nifti", ".tif": "tiff", ".tiff": "tiff",
           ".h5": "hdf5", ".hdf5": "hdf5"}


class LabelVolumeFile(NamedTuple):
    """A segmented image as read from disk.

    Attributes
    ----------
    labels : (nx, ny, nz) uint8
        The label of each voxel.
    voxel_size : (3,) float64
        The voxel's extent along each index axis, in metres.
    origin : (3,) float64
        The lower corner of voxel ``(0, 0, 0)``, in metres.
    """
    labels: np.ndarray
    voxel_size: np.ndarray
    origin: np.ndarray


class LabelVolumeError(ValueError):
    """A segmented image that cannot be walked: no voxel size, a payload that does not match the
    header, or a container this module does not read. The message names the file and the field."""


def _unit(name, path):
    key = str(name).strip().strip('"').strip("'").lower()
    if key not in UNITS_M:
        raise LabelVolumeError(f"{path}: the length unit {name!r} is not one of {sorted(set(UNITS_M))}")
    return UNITS_M[key]


def _voxel_size(header_size, given, path):
    """The voxel size in metres: the header's, or the caller's when the header states none.

    Both is a contradiction unless they agree to a part in 1e-9, and a contradiction is refused
    rather than resolved by precedence -- a substrate walked at the wrong scale is wrong everywhere
    and nowhere visibly.
    """
    g = None if given is None else np.broadcast_to(np.asarray(given, np.float64).ravel(), (3,)).copy()
    if g is not None and np.any(g <= 0):
        raise LabelVolumeError(f"{path}: voxel_size must be positive on every axis, got {list(g)}")
    if header_size is None:
        if g is None:
            raise LabelVolumeError(
                f"{path}: the header states no voxel size, and none was given. A walk needs the "
                f"physical extent of a voxel -- pass voxel_size= in metres.")
        return g
    h = np.asarray(header_size, np.float64)
    if np.any(h <= 0):
        raise LabelVolumeError(f"{path}: the header's voxel size must be positive on every axis, got {list(h)}")
    if g is not None and not np.allclose(h, g, rtol=1e-9, atol=0.0):
        raise LabelVolumeError(f"{path}: the header states a voxel size of {list(h)} m and the caller "
                               f"{list(g)} m; one of the two is wrong.")
    return h


def _as_labels(a, path):
    """The payload as ``uint8`` labels: a segmentation has at most 256 classes, and a float or wide
    integer array that does not hold small non-negative integers is not one."""
    a = np.asarray(a)
    if a.ndim != 3:
        raise LabelVolumeError(f"{path}: a label volume is 3-D, got shape {a.shape}")
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    lo, hi = float(a.min()), float(a.max())
    if not (lo >= 0 and hi <= 255) or (a.dtype.kind == "f" and not np.all(a == np.floor(a))):
        raise LabelVolumeError(f"{path}: the payload is {a.dtype} in [{lo}, {hi}], not a label field "
                               f"(0..255 integers). Segment it first.")
    return np.ascontiguousarray(a.astype(np.uint8))


def _check_payload(n_bytes, shape, itemsize, path):
    want = int(np.prod(shape)) * int(itemsize)
    if n_bytes != want:
        raise LabelVolumeError(
            f"{path}: the header declares {tuple(shape)} of {itemsize}-byte samples ({want} bytes) but the "
            f"payload holds {n_bytes}. The two do not describe the same image.")


# ------------------------------------------------------------------------------------ NRRD
_NRRD_TYPES = {"uchar": np.uint8, "unsigned char": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
               "signed char": np.int8, "int8": np.int8, "int8_t": np.int8,
               "short": np.int16, "int16": np.int16, "ushort": np.uint16, "uint16": np.uint16,
               "int": np.int32, "int32": np.int32, "uint": np.uint32, "uint32": np.uint32,
               "float": np.float32, "double": np.float64}


def read_nrrd(path, *, voxel_size=None):
    """A NRRD image: a detached header (``.nhdr`` naming its ``data file``) or a single ``.nrrd``.

    The voxel size is the norm of each ``space directions`` vector, or ``spacings``, in the header's
    ``space units`` (metres when it names none); the origin is ``space origin``. ``raw``, ``gzip``
    and ``gz`` encodings are read.
    """
    path = os.fspath(path)
    with open(path, "rb") as fh:
        blob = fh.read()
    if not blob.startswith(b"NRRD"):
        raise LabelVolumeError(f"{path}: not a NRRD file (no NRRD magic)")
    sep = blob.find(b"\n\n") if b"\n\n" in blob else blob.find(b"\r\n\r\n")
    head_text = (blob if sep < 0 else blob[:sep]).decode("utf-8", "replace")
    inline = b"" if sep < 0 else blob[sep + (2 if blob[sep:sep + 2] == b"\n\n" else 4):]

    hdr = {}
    for line in head_text.splitlines():
        if not line or line.startswith("#") or line.startswith("NRRD"):
            continue
        if ":=" in line:
            k, v = line.split(":=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        hdr[k.strip().lower()] = v.strip()

    for key in ("type", "sizes", "encoding"):
        if key not in hdr:
            raise LabelVolumeError(f"{path}: the NRRD header is missing {key!r}")
    dtype = _NRRD_TYPES.get(hdr["type"].strip().lower())
    if dtype is None:
        raise LabelVolumeError(f"{path}: NRRD type {hdr['type']!r} is not one of {sorted(_NRRD_TYPES)}")
    shape = tuple(int(x) for x in hdr["sizes"].split())
    if len(shape) != 3:
        raise LabelVolumeError(f"{path}: a label volume is 3-D, the header declares sizes {shape}")

    unit = 1.0
    if "space units" in hdr:
        units = re.findall(r'"([^"]*)"', hdr["space units"]) or hdr["space units"].split()
        us = {_unit(u, path) for u in units}
        if len(us) != 1:
            raise LabelVolumeError(f"{path}: 'space units' mixes units ({hdr['space units']!r}); "
                                   f"one length unit per image")
        unit = us.pop()
    vox = None
    if "space directions" in hdr:
        vecs = [[float(x) for x in g.split(",")] for g in re.findall(r"\(([^)]*)\)", hdr["space directions"])]
        if len(vecs) != 3:
            raise LabelVolumeError(f"{path}: 'space directions' must give three vectors, got {len(vecs)}")
        vox = np.array([np.linalg.norm(v) for v in vecs], np.float64) * unit
    elif "spacings" in hdr:
        vox = np.array([float(x) for x in hdr["spacings"].split()], np.float64) * unit
    origin = (np.array([float(x) for x in re.findall(r"[-+0-9.eE]+", hdr["space origin"])], np.float64) * unit
              if "space origin" in hdr else np.zeros(3))

    src = hdr.get("data file") or hdr.get("datafile")
    if src:
        data_path = src if os.path.isabs(src) else os.path.join(os.path.dirname(path), src)
        with open(data_path, "rb") as fh:
            payload = fh.read()
    else:
        data_path, payload = path, inline
    enc = hdr["encoding"].strip().lower()
    if enc in ("gzip", "gz"):
        import gzip
        payload = gzip.decompress(payload)
    elif enc != "raw":
        raise LabelVolumeError(f"{path}: NRRD encoding {enc!r} is not read here (raw, gzip)")
    _check_payload(len(payload), shape, np.dtype(dtype).itemsize, data_path)
    # NRRD is fastest-axis-first, which is the x axis of `sizes`.
    a = np.frombuffer(payload, dtype=dtype).reshape(shape[::-1]).transpose(2, 1, 0)
    if hdr.get("endian", "little").strip().lower() == "big" and np.dtype(dtype).itemsize > 1:
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    return LabelVolumeFile(_as_labels(a, path), _voxel_size(vox, voxel_size, path), origin)


# ------------------------------------------------------------------------------------ MetaImage
_MHD_TYPES = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
              "MET_UINT": np.uint32, "MET_INT": np.int32, "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64}


def read_mhd(path, *, voxel_size=None):
    """A MetaImage: a ``.mhd`` header with its ``ElementDataFile``, or a self-contained ``.mha``.

    ``ElementSpacing`` and ``Offset`` are in millimetres, the format's unit.
    """
    path = os.fspath(path)
    with open(path, "rb") as fh:
        blob = fh.read()
    key = b"ElementDataFile"
    cut = blob.find(key)
    if cut < 0:
        raise LabelVolumeError(f"{path}: the MetaImage header is missing 'ElementDataFile'")
    end = blob.find(b"\n", cut)
    head_text = blob[:end if end > 0 else len(blob)].decode("utf-8", "replace")
    hdr = {}
    for line in head_text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        hdr[k.strip()] = v.strip()

    if hdr.get("CompressedData", "False").strip().lower() == "true":
        raise LabelVolumeError(f"{path}: compressed MetaImage data is not read here; decompress it first")
    dtype = _MHD_TYPES.get(hdr.get("ElementType", ""))
    if dtype is None:
        raise LabelVolumeError(f"{path}: ElementType {hdr.get('ElementType')!r} is not one of {sorted(_MHD_TYPES)}")
    if "DimSize" not in hdr:
        raise LabelVolumeError(f"{path}: the MetaImage header is missing 'DimSize'")
    shape = tuple(int(x) for x in hdr["DimSize"].split())
    if len(shape) != 3:
        raise LabelVolumeError(f"{path}: a label volume is 3-D, the header declares DimSize {shape}")
    spacing = hdr.get("ElementSpacing") or hdr.get("ElementSize")
    vox = (np.array([float(x) for x in spacing.split()], np.float64) * 1e-3) if spacing else None
    origin = (np.array([float(x) for x in hdr["Offset"].split()], np.float64) * 1e-3
              if "Offset" in hdr else np.zeros(3))

    src = hdr["ElementDataFile"]
    if src.upper() == "LOCAL":
        data_path, payload = path, blob[end + 1:]
    else:
        if "%" in src or " " in src.strip():
            raise LabelVolumeError(f"{path}: a per-slice ElementDataFile pattern ({src!r}) is not read here")
        data_path = src if os.path.isabs(src) else os.path.join(os.path.dirname(path), src)
        with open(data_path, "rb") as fh:
            payload = fh.read()
    _check_payload(len(payload), shape, np.dtype(dtype).itemsize, data_path)
    a = np.frombuffer(payload, dtype=dtype).reshape(shape[::-1]).transpose(2, 1, 0)
    if hdr.get("BinaryDataByteOrderMSB", "False").strip().lower() == "true" and np.dtype(dtype).itemsize > 1:
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    return LabelVolumeFile(_as_labels(a, path), _voxel_size(vox, voxel_size, path), origin)


# ------------------------------------------------------------------------------------ NIfTI
def read_nifti(path, *, voxel_size=None):
    """A NIfTI image through ``nibabel``: the voxel size is the header's zooms in its own
    ``xyzt_units`` (millimetres when it declares none), the origin the affine's translation."""
    path = os.fspath(path)
    try:
        import nibabel as nib
    except ImportError as e:
        raise LabelVolumeError(f"{path}: reading NIfTI needs nibabel (pip install nibabel)") from e
    img = nib.load(path)
    hdr = img.header
    zooms = np.asarray(hdr.get_zooms()[:3], np.float64)
    unit = 1e-3
    try:
        name = hdr.get_xyzt_units()[0]
        if name and name != "unknown":
            unit = _unit(name, path)
    except Exception:                                  # a header with no unit field: the format's mm
        pass
    vox = zooms * unit if np.all(zooms > 0) else None
    origin = np.asarray(img.affine, np.float64)[:3, 3] * unit
    return LabelVolumeFile(_as_labels(np.asanyarray(img.dataobj), path),
                           _voxel_size(vox, voxel_size, path), origin)


# ------------------------------------------------------------------------------------ TIFF stack
def read_tiff(path, *, voxel_size=None):
    """A TIFF stack through ``tifffile``: a multi-page file, or a directory whose ``.tif`` files are
    the slices in name order (the slowest index).

    A TIFF carries in-plane resolution tags at best and no slice spacing at all, so the voxel size
    comes from ``voxel_size=``; the ImageJ ``spacing`` / ``unit`` tags are read when the file has them.
    """
    path = os.fspath(path)
    try:
        import tifffile
    except ImportError as e:
        raise LabelVolumeError(f"{path}: reading a TIFF stack needs tifffile (pip install tifffile)") from e
    vox = None
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".tif", ".tiff")))
        if not files:
            raise LabelVolumeError(f"{path}: the directory holds no .tif slices")
        planes = [tifffile.imread(os.path.join(path, f)) for f in files]
        if any(p.ndim != 2 or p.shape != planes[0].shape for p in planes):
            raise LabelVolumeError(f"{path}: the slices are not 2-D images of one shape")
        a = np.stack(planes, axis=0)                    # (z, y, x): the slice index is slowest
    else:
        with tifffile.TiffFile(path) as tf:
            a = tf.asarray()                            # (z, y, x)
            ij = tf.imagej_metadata or {}
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            if "spacing" in ij and xres is not None and xres.value[0]:
                unit = _unit(ij.get("unit", "um"), path)
                inplane = float(xres.value[1]) / float(xres.value[0]) * unit
                vox = np.array([inplane, inplane, float(ij["spacing"]) * unit], np.float64)
    if a.ndim != 3:
        raise LabelVolumeError(f"{path}: a label volume is 3-D, the stack holds shape {a.shape}")
    # ONE transpose for both containers. A directory of slices and the multi-page file of the same
    # slices are the same image, so they are read into the same (i, j, k) order: each plane is
    # (y, x) and the slice index is z, which is (z, y, x) either way.
    a = np.transpose(a, (2, 1, 0))
    return LabelVolumeFile(_as_labels(a, path), _voxel_size(vox, voxel_size, path), np.zeros(3))


# ------------------------------------------------------------------------------------ HDF5
def read_hdf5(path, *, dataset=None, voxel_size=None):
    """One dataset of an HDF5 file through ``h5py``: ``dataset`` names it, or the file's single 3-D
    dataset. The voxel size is the dataset's ``element_size_um`` attribute (the ilastik / ImageJ
    convention, in micrometres, slowest axis first) or ``voxel_size=``."""
    path = os.fspath(path)
    try:
        import h5py
    except ImportError as e:
        raise LabelVolumeError(f"{path}: reading HDF5 needs h5py (pip install h5py)") from e
    with h5py.File(path, "r") as fh:
        if dataset is None:
            found = []
            fh.visititems(lambda n, o: found.append(n) if getattr(o, "ndim", 0) == 3 else None)
            if len(found) != 1:
                raise LabelVolumeError(f"{path}: holds {len(found)} 3-D datasets ({found}); name one with dataset=")
            dataset = found[0]
        if dataset not in fh:
            raise LabelVolumeError(f"{path}: no dataset {dataset!r}")
        d = fh[dataset]
        es = d.attrs.get("element_size_um")
        vox = np.asarray(es, np.float64).ravel()[::-1] * 1e-6 if es is not None else None
        a = np.transpose(np.asarray(d), (2, 1, 0))      # h5py datasets are written (z, y, x)
    return LabelVolumeFile(_as_labels(a, path), _voxel_size(vox, voxel_size, path), np.zeros(3))


# ------------------------------------------------------------------------------------ dispatch
_READERS = {"nrrd": read_nrrd, "mhd": read_mhd, "nifti": read_nifti, "tiff": read_tiff, "hdf5": read_hdf5}


def format_of(path):
    """The reader name of a path, from its suffix (a directory of slices is a ``tiff`` stack)."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return "tiff"
    name = os.path.basename(path).lower()
    if name.endswith(".nii.gz"):
        return "nifti"
    fmt = FORMATS.get(os.path.splitext(name)[1])
    if fmt is None:
        raise LabelVolumeError(f"{path}: no reader for this suffix; formats are {sorted(set(FORMATS.values()))}")
    return fmt


def read_label_volume(path, *, format=None, voxel_size=None, dataset=None):
    """A segmented image in any of the formats here: ``(labels, voxel_size, origin)``.

    ``format`` names the reader (``nrrd``, ``mhd``, ``nifti``, ``tiff``, ``hdf5``); by default it is
    taken from the suffix. ``voxel_size`` (metres, a scalar or three numbers) supplies a spacing the
    container does not carry. ``dataset`` names the HDF5 dataset.
    """
    fmt = format_of(path) if format is None else str(format).lower()
    reader = _READERS.get(fmt)
    if reader is None:
        raise LabelVolumeError(f"{path}: format {fmt!r} is not one of {sorted(_READERS)}")
    kw = {"voxel_size": voxel_size}
    if fmt == "hdf5":
        kw["dataset"] = dataset
    elif dataset is not None:
        raise LabelVolumeError(f"{path}: dataset= names an HDF5 dataset; format {fmt!r} has none")
    return reader(path, **kw)


def payload_files(path, *, format=None):
    """Every file the container at ``path`` reads, the header first.

    A detached container is two files: ``LV60A.nhdr`` names ``LV60A.raw`` and the image is in the
    second one. A spec's ``surface.sha256`` covers the file it cites, so a producer that wants the
    IMAGE covered records each of these with its own digest (``provenance.files``).
    """
    path = os.fspath(path)
    fmt = format_of(path) if format is None else str(format).lower()
    out = [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith((".tif", ".tiff")))
    if fmt in ("nrrd", "mhd"):
        with open(path, "rb") as fh:
            head = fh.read(8192).decode("utf-8", "replace")
        for line in head.splitlines():
            key, _, value = (line.partition(":=") if ":=" in line
                             else (line.partition(":") if fmt == "nrrd" else line.partition("=")))[0:3]
            k = key.strip().lower()
            if k in ("data file", "datafile", "elementdatafile"):
                src = value.strip()
                if src and src.upper() != "LOCAL" and "%" not in src:
                    out.append(src if os.path.isabs(src) else os.path.join(os.path.dirname(path), src))
                break
    return out


def crop_labels(volume, crop):
    """The sub-volume ``crop = (i0, j0, k0, i1, j1, k1)`` (half-open, in voxels) of a
    :class:`LabelVolumeFile`, its origin moved to the crop's own lower corner.

    A published reference measurement is made on a stated sub-volume, so the crop is part of the
    substrate and travels in its spec; a crop outside the image is refused."""
    lo = np.asarray(crop, np.int64)[:3]
    hi = np.asarray(crop, np.int64)[3:]
    if len(np.asarray(crop).ravel()) != 6:
        raise LabelVolumeError(f"a crop is (i0, j0, k0, i1, j1, k1), got {crop!r}")
    n = np.asarray(volume.labels.shape, np.int64)
    if np.any(lo < 0) or np.any(hi > n) or np.any(hi <= lo):
        raise LabelVolumeError(f"the crop {list(lo)}..{list(hi)} is not a non-empty sub-volume of "
                               f"{tuple(int(x) for x in n)}")
    sub = np.ascontiguousarray(volume.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    return LabelVolumeFile(sub, volume.voxel_size, volume.origin + lo * volume.voxel_size)


def write_nrrd(path, labels, voxel_size, origin=(0.0, 0.0, 0.0)):
    """Write a label grid as a single-file NRRD with a raw payload, in metres.

    The counterpart of :func:`read_nrrd` for a volume that exists only in memory: a spec cites a
    segmented image as a file, so a geometry built from an array needs somewhere to be cited from.
    """
    lab = _as_labels(np.asarray(labels), str(path))
    vox = np.broadcast_to(np.asarray(voxel_size, np.float64).ravel(), (3,))
    org = np.broadcast_to(np.asarray(origin, np.float64).ravel(), (3,))
    if np.any(vox <= 0):
        raise LabelVolumeError(f"{path}: voxel_size must be positive on every axis, got {list(vox)}")
    head = ["NRRD0004", "type: uchar", "endian: little", "encoding: raw", "dimension: 3",
            "space: 3D-right-handed", "kinds: domain domain domain",
            "sizes: %d %d %d" % lab.shape,
            "space directions: (%.17g,0,0) (0,%.17g,0) (0,0,%.17g)" % tuple(vox),
            "space origin: (%.17g,%.17g,%.17g)" % tuple(org),
            'space units: "m" "m" "m"', ""]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(np.ascontiguousarray(lab.transpose(2, 1, 0)).tobytes())
    return path

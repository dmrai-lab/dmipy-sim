"""A streaming writer of the MRtrix ``.tck`` track format :func:`dmipy_sim.io.strands.write_tck` also writes,
byte-for-byte compatible with :func:`dmipy_sim.io.strands.read_tck`, but fed one centerline at a time rather
than one ``(N, 3)`` array of every point: what a tiled generator needs so a 5 mm pack's ~80 M centerline points
are never concatenated into one array before being written (dmipy-sim#697's memory rule)."""
from __future__ import annotations

import os

import numpy as np

__all__ = ["StreamingTckWriter"]


class StreamingTckWriter:
    """Open on ``path``; call :meth:`add` once per strand, in any order, holding nothing between calls but the
    open file handles; :meth:`close` (or the context manager) finalises the header and the end-of-file marker.
    The body is written to ``path + '.body'`` as it comes in and copied into the final file at :meth:`close`
    (a header of fixed content but data-dependent width -- the strand count -- can only be written once the
    count is known), copied in fixed-size chunks rather than read whole."""

    _COPY_CHUNK = 1 << 20

    def __init__(self, path, *, coordinate_unit_m):
        self.path = str(path)
        self.coordinate_unit_m = float(coordinate_unit_m)
        self._body_path = self.path + ".body"
        self._body = open(self._body_path, "wb")
        self._n = 0
        self._closed = False

    def add(self, centerline_m):
        """Append one strand's centerline (``(k, 3)`` array, metres, ``k >= 2``)."""
        pts = np.asarray(centerline_m, np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError(f"a centerline is (k >= 2, 3); got {pts.shape}")
        body = (pts / self.coordinate_unit_m).astype("<f4")
        self._body.write(body.tobytes())
        self._body.write(np.full((1, 3), np.nan, dtype="<f4").tobytes())
        self._n += 1

    @property
    def n_strands(self):
        return self._n

    def close(self):
        if self._closed:
            return self._n
        self._body.close()
        head_lines = ["mrtrix tracks", f"count: {self._n}", "datatype: Float32LE"]
        head = "\n".join(head_lines) + "\n"
        offset = len(head) + len("file: . ") + 8 + 1 + len("END\n")
        head += f"file: . {offset:8d}\nEND\n"
        with open(self.path, "wb") as out, open(self._body_path, "rb") as body:
            out.write(head.encode("ascii"))
            while True:
                chunk = body.read(self._COPY_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
            out.write(np.full((1, 3), np.inf, dtype="<f4").tobytes())
        os.remove(self._body_path)
        self._closed = True
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

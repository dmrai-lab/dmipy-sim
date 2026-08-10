"""``CactusBundle`` — a loaded axon bundle: inner (axonal) + outer (myelin) surface meshes in
metres, a bounding box, and volume fractions. The common substrate container consumed by the mesh
axon master-walk builder (:func:`dmipy_sim.mesh_axon.mesh_axon_master`) and the field-basis builder
(:func:`dmipy_sim.susceptibility_field.mesh_field_basis`).

(The full CACTUS run-directory loader lives with the private substrate-bank tooling; the public
package ships this container plus the Winther-axon loader :mod:`dmipy_sim.io.winther`.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_UM = 1e-6


@dataclass
class CactusBundle:
    """Two multi-surface meshes (metres) + box + volume fractions."""
    inner: tuple           # (V, F) metres — axonal (inner) surface(s)
    outer: tuple           # (V, F) metres — myelin (outer) surface(s)
    box_min: np.ndarray    # (3,) metres
    box_max: np.ndarray    # (3,) metres
    g_ratio: float
    f_intra: float
    f_myelin: float
    f_extra: float
    n_fibres: int
    fibre_axis: int        # 0/1/2 — the elongated (fibre) axis in mesh coords
    run_dir: str = ""
    fibre_tangents: np.ndarray = None   # (n_fibres, 3) unit per-fibre mean tangents

    @property
    def box_side(self):
        return self.box_max - self.box_min

    def summary(self):
        return (f"CactusBundle({self.n_fibres} fibre(s), g={self.g_ratio:.2f}, "
                f"box={np.round(self.box_side / _UM, 1)} um, axis={'xyz'[self.fibre_axis]}, "
                f"f=[intra {self.f_intra:.3f}, myelin {self.f_myelin:.3f}, extra {self.f_extra:.3f}])")

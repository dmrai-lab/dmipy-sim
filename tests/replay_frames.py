"""The field along a chosen direction of the SUBSTRATE frame, with the acquisition as given in that frame -- the
thought experiment the field tests run. The bore's field is z, so this is the specimen posed with ``R``
(``R^T z = d``) under the acquisition turned into the lab (``G R^T``), which ``replay(seq_lab, orientation=R)``
reads back as ``(G, d)``."""
import numpy as np

from dmipy_sim.acquisition.waveforms import rotate_waveform
from dmipy_sim.replay import so3


def pose_with_field_along(d):
    """The rotation ``R`` (substrate -> lab) whose pose puts the bore's field along ``d`` in the substrate frame."""
    d = np.asarray(d, np.float64); d = d / np.linalg.norm(d)
    return np.asarray(so3.rotation_of(d), np.float64).T              # R^T z = d  <=>  R d = z


def field_along(seq, d):
    """``(seq_lab, R)`` such that ``pack.replay(seq_lab, orientation=R, scanner=B0)`` is the substrate under the
    acquisition ``seq`` with the field along ``d`` in its own frame."""
    R = pose_with_field_along(d)
    return rotate_waveform(seq, R), R                                  # G @ R^T, which the pose turns back into G

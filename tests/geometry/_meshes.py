"""Small triangle meshes the geometry tests generate on the fly: an icosphere and an open circular tube."""
import numpy as np


def icosphere(sub, r=5e-6):
    """``(V, F)`` of trimesh's icosphere of ``sub`` subdivisions and radius ``r`` (m): closed and outward."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def open_tube(r=4e-6, L=12e-6, nt=32, nz=24):
    """``(V, F)`` of a genuinely open (non-periodic) circular tube of radius ``r`` along z from 0 to ``L``,
    ``nt`` vertices round and ``nz`` rings: no axis closes its rims by identification."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zs = np.linspace(0, L, nz)
    V = np.array([[r * np.cos(t), r * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt
            c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    return V, np.array(F)

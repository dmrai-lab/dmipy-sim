"""Triangular-mesh geometry with spatial acceleration and optional 3D periodicity.

``Mesh`` lets the Monte-Carlo engine run arbitrary closed *or* periodic triangular
surface meshes (e.g. dense multi-cell microstructure phantoms exported as PLY) with
the same physics as the analytic geometries: restricted diffusion, surface
relaxivity (Brownstein–Tarr) and membrane permeability (Powles).

Design
------
* **Spatial acceleration.** A static uniform grid buckets every triangle by the
  cells its axis-aligned bounding box overlaps.  Each step only tests the triangles
  in the walker's 27-cell neighbourhood, turning the per-step cost from
  ``O(n_triangles)`` into ``O(candidates)`` — the difference between "intractable"
  and "seconds" on a 10^6-triangle mesh.  This is exact provided the grid cell size
  is at least the maximum single step (guaranteed by the ``cell_size`` default,
  which is tied to the diffusion step).
* **3D periodicity** (``periodic=True``).  Triangles within one cell of a periodic
  face are replicated ("ghosts") across to the opposite side, so a walker crossing
  the box sees the continuing structure.  Geometry queries use the *wrapped*
  position while the returned position stays *continuous* (unfolded), which keeps
  the gradient phase correct — the same convention the packed geometries use.  The
  box faces are wrap planes, **not** reflecting walls, so open (clipped) cells on
  the boundary are stitched to their periodic partners.
* **Smooth reflection.** Reflections and surface-relaxivity path lengths use a
  barycentrically-interpolated vertex normal, reducing the flat-facet error from
  ``O(h/R)`` to ``O(h^2/R^2)`` in the triangle edge length ``h``.
* **Leak-proof permeation.** The Powles crossing decision is taken once, at the
  first membrane hit; if the walker reflects, the remainder of the step is resolved
  by a multi-bounce reflection scan (no further crossing draws).  This neither
  leaks walkers through convex corners nor double-counts crossings there.

Accuracy note
-------------
Restricted diffusion and surface relaxivity reach the Monte-Carlo noise floor even
for coarse meshes; membrane *permeability* is more sensitive to the surface
tessellation (its bias falls ~``O(h^2)``), so accurate permeability needs a fairly
fine mesh (edge length ``<~ 0.04`` of the local feature radius).
"""

import itertools
import warnings
from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometries import Geometry

# Above this median-edge / feature-radius ratio the surface is too coarsely
# tessellated for membrane permeability to reach the MC noise floor (its faceting
# bias falls ~O(h^2); measured: ratio 0.075 -> ~8x noise, 0.038 -> at noise floor).
# Restricted diffusion and surface relaxivity are unaffected at these ratios.
_PERM_EDGE_RATIO_MAX = 0.05


def _rotation_from_z(axis):
    """Rotation matrix R (mesh->lab) with R @ [0,0,1] = axis / |axis|.

    The in-plane (azimuthal) choice is arbitrary; pass an explicit ``R`` instead
    for meshes whose in-plane orientation matters.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    c = float(np.dot(z, a))
    if np.linalg.norm(v) < 1e-12:                 # parallel or anti-parallel
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))   # Rodrigues


def _smooth_vertex_normals(V, F):
    """Area-weighted vertex normals, shape (n_vertices, 3)."""
    tris = V[F]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # ||=2*area
    vn = np.zeros((len(V), 3))
    np.add.at(vn, F[:, 0], cross)
    np.add.at(vn, F[:, 1], cross)
    np.add.at(vn, F[:, 2], cross)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    return vn



def _parse_ascii_ply(path):
    """Vertices/faces from an ASCII PLY, ignoring the declared index type.

    Returns ``(None, None)`` for anything that is not a plain ascii PLY with float x/y/z vertices and a
    triangle face list -- the caller then keeps whatever the general loader produced.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        head_end = raw.find(b"end_header")
        if head_end < 0:
            return None, None
        header = raw[:head_end].decode("ascii", "replace").splitlines()
        if not any(l.strip() == "format ascii 1.0" for l in header):
            return None, None
        n_v = n_f = None
        for line in header:
            t = line.split()
            if len(t) == 3 and t[0] == "element" and t[1] == "vertex":
                n_v = int(t[2])
            elif len(t) == 3 and t[0] == "element" and t[1] == "face":
                n_f = int(t[2])
        if not n_v or not n_f:
            return None, None
        body = raw[raw.find(b"\n", head_end) + 1:].decode("ascii", "replace").split()
        # vertices: 3 floats each (properties beyond x/y/z are not handled -> bail out via length check)
        need_v = 3 * n_v
        vals = body[:need_v]
        if len(vals) < need_v:
            return None, None
        V = np.asarray(vals, dtype=np.float64).reshape(n_v, 3)
        rest = body[need_v:]
        faces, i = [], 0
        for _ in range(n_f):
            if i >= len(rest):
                return None, None
            k = int(rest[i]); i += 1
            if k != 3:
                return None, None                      # only triangles
            faces.append(rest[i:i + 3]); i += 3
        F = np.asarray(faces, dtype=np.int64)
        if F.max() >= n_v:
            return None, None
        return V, F
    except Exception:
        return None, None


def load_ply(path, scale=1.0, recenter=False):
    """Load vertices and faces from a mesh file (PLY/STL/OBJ/...).

    Uses :mod:`trimesh` (install the ``mesh`` extra: ``pip install dmipy-sim[mesh]``).

    Parameters
    ----------
    path : str
        Path to the mesh file.
    scale : float
        Multiply all coordinates by this factor — use it to convert a mesh stored
        in arbitrary/normalised units into **metres** (the simulator's unit).
    recenter : bool
        If True, translate the mesh so its bounding box is centred on the origin.

    Returns
    -------
    vertices : (n_vertices, 3) float64 ndarray, metres
    faces    : (n_faces, 3) int64 ndarray
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(
            "load_ply requires trimesh — install with `pip install dmipy-sim[mesh]` "
            "or `pip install trimesh`."
        ) from exc
    m = trimesh.load(path, process=False)
    V = np.asarray(m.vertices, np.float64)
    F = np.asarray(m.faces, np.int64)
    # An ASCII PLY may declare a face-index type too small for its own vertex count -- MATLAB's plywrite
    # emits `property list uchar ushort vertex_indices` even past 65535 vertices. The decimal text holds
    # the true indices, but a reader that honours the declared type masks them to 16 bits, silently
    # rewiring every face above the limit: the mesh keeps a plausible bounding box while its volume and
    # surface area become meaningless. Detect it (no face reaches the upper vertices) and re-parse.
    if len(V) > 65536 and F.size and int(F.max()) == 65535:
        Vf, Ff = _parse_ascii_ply(path)
        if Vf is not None and int(Ff.max()) > 65535:
            warnings.warn(f"{path}: face indices were truncated to 16 bits by the declared PLY type "
                 f"(max index {int(F.max())} for {len(V)} vertices); re-parsed the ASCII payload "
                 f"(max index now {int(Ff.max())}).")
            V, F = Vf.astype(np.float64), Ff.astype(np.int64)
    if recenter:
        V = V - 0.5 * (V.min(0) + V.max(0))
    return V * scale, F


class _MeshArrays(NamedTuple):
    """The mesh's JAX geometry arrays, bundled as a pytree.

    Passing this bundle as an *argument* to a jitted function makes jax treat the arrays as runtime device
    buffers rather than baking them into the compiled executable as constants. For a large-bounding-box mesh
    the grid array (CELL) alone can be gigabytes, which as a captured constant blows up the executable and
    exhausts memory on load; as an argument it is just data living in device memory.
    """
    NRM: jnp.ndarray
    CENT: jnp.ndarray
    CELL: jnp.ndarray
    dims_arr: jnp.ndarray
    GMIN: jnp.ndarray
    CS: jnp.ndarray
    VMIN: jnp.ndarray
    L: jnp.ndarray
    PER: jnp.ndarray
    OFF: jnp.ndarray


def _wrap_arr(A, r):
    w = A.VMIN + jnp.mod(r - A.VMIN, A.L)
    return jnp.where(A.PER > 0, w, r)


def _gather_arr(A, r_w):
    c = jnp.clip(jnp.floor((r_w - A.GMIN) / A.CS).astype(jnp.int32), 0, A.dims_arr - 1)
    nb = jnp.clip(c[None, :] + A.OFF, 0, A.dims_arr - 1)
    # dims taken from the traced array, not as static Python ints, so nothing is baked in
    cids = (nb[:, 0] * A.dims_arr[1] + nb[:, 1]) * A.dims_arr[2] + nb[:, 2]
    cand = A.CELL[cids].reshape(-1)
    valid = cand >= 0
    return jnp.where(valid, cand, 0), valid


def _gather_is_populated(A, r):
    """Does the 27-cell gather around ``r`` contain any triangle at all?

    The classifier's answer is only meaningful where this is true. Exposed because "no wall nearby" and
    "outside" are different statements, and conflating them is what makes the failure silent.
    """
    return _gather_arr(A, _wrap_arr(A, r))[1].any()


def _classify_arr(A, r):
    """Interior (0) / exterior (1) from the array bundle.

    Single source of truth for both ``Mesh.classify_position`` and the ``init_positions`` seeding jit.
    Sidedness comes from the nearest surface triangle in the local 27-cell gather. A point with **no**
    triangle anywhere in that gather cannot be enclosed by a wall, so it is exterior. Without that guard
    the ``argmin`` over an all-``inf`` distance array silently falls back to a far-away triangle whose
    outward normal gives a *random* sign, and empty space near a thin feature is misclassified as interior
    -- which poisons ``init_positions``, seeding "intra" walkers into open space where they diffuse freely.
    THE GUARD'S VALIDITY CONDITION IS ON THE OBJECT, NOT THE MESH. It holds only while every genuine
    interior point has a wall inside its gather -- i.e. while the enclosed feature is no wider than the
    gather neighbourhood. But the cell size scales with the TRIANGLE size, not the object size, so a
    finely-meshed thick fibre breaks it: measured on a 366-strand axon bundle (median edge 0.371 um -> cell
    0.124 um -> gather reach ~0.19 um, lumen radii ~1-2 um), 19.6% of genuinely interior points are
    reported EXTERIOR, and 99.4% of those have an empty gather. Deep interior reads as outside.

    Use :func:`_gather_is_populated` to tell "outside" from "cannot tell", and resolve the latter with an
    exact test (:func:`dmipy_sim.susceptibility_field.mesh_contains`). :meth:`Mesh.init_positions` does
    exactly that. Per-step compartment tracking in the walk does NOT yet -- see dmrai-lab/dmipy-sim#33.
    """
    r_w = _wrap_arr(A, r)
    ci, valid = _gather_arr(A, r_w)
    cent = A.CENT[ci]; nrm = A.NRM[ci]
    dist = jnp.where(valid, jnp.linalg.norm(r_w[None] - cent, axis=1), jnp.inf)
    idx = jnp.argmin(dist)
    side = jnp.dot(r_w - cent[idx], nrm[idx])
    inside = (side < 0) & valid.any()          # no nearby wall => not enclosed => exterior
    return jnp.where(inside, jnp.int32(0), jnp.int32(1))


class Mesh(Geometry):
    """Reflecting/permeable triangular-mesh geometry (see module docstring).

    Parameters
    ----------
    vertices : (n_vertices, 3) array-like, metres
    faces : (n_faces, 3) array-like of int
        Triangle vertex-index triples.
    periodic : bool or (bool, bool, bool)
        Wrap walkers periodically along the given axes (default False = closed mesh).
        When any axis is periodic you must pass ``voxel_min``/``voxel_max`` defining
        the periodic box (the mesh bbox is usually slightly larger than the true
        period, so it is not a safe default).
    voxel_min, voxel_max : (3,) array-like, optional
        The simulation box.  Defaults to the mesh bounding box (closed meshes only).
    feature_radius : float, optional
        Characteristic feature size (e.g. a cell/pore radius), used to size the
        diffusion sub-step (``step ~ feature_radius/6``, or ``/25`` when permeable)
        and the grid.  Defaults to half the smallest box side; **pass the real cell
        radius for packed substrates**, otherwise the step may be too coarse.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ (m/s), symmetric (same on both sides of the wall).
        Applies a Brownstein–Tarr weight at the wall. For a side-dependent ρ, use
        ``intra=``/``extra=`` instead.
    permeability : float or dict, optional
        Membrane permeability κ (m/s).  A float is symmetric (same both directions,
        the default). A dict ``{"intra_to_extra": κ_out, "extra_to_intra": κ_in}``
        makes it direction-dependent — note asymmetric κ is a *pump* (net flux, not
        passive equilibrium). None → impermeable.
    intra, extra : dict, optional
        Per-compartment properties for the intra (inside a cell) and extra sides.
        Supported keys:
        ``surface_relaxivity_t2`` — a wall effect; a spin hitting the wall from
        inside vs outside takes a different relaxivity weight (may be set on one
        side only).
        ``D`` (m²/s), ``T2`` (s), ``T1`` (s) — per-compartment bulk properties (a
        per-step effect: the walker's step uses its compartment's D, and its
        log-weight that compartment's 1/T2 · χ + 1/T1 · (1−χ)); if given, a value
        is required for BOTH sides. T1 only acts during longitudinal storage
        (χ=0, e.g. a PGSTE mixing time). Unequal ``D`` across a permeable wall is
        rejected (diffusivity-discontinuity interface).
    orientation : (3,) array-like, optional
        Direction, in the scanner frame (B0 = +z), along which the mesh's native
        +z axis (e.g. a periodic / fibre axis) is placed in the bore.  Applied as
        an acquisition rotation (the gradient is rotated into the mesh frame in
        ``simulate``), so the walk itself is unchanged.  The in-plane rotation is
        arbitrary — pass ``R`` for meshes whose in-plane orientation matters.
    R : (3, 3) array-like, optional
        Explicit mesh→lab rotation matrix (mutually exclusive with ``orientation``).
    cell_size : float, optional
        Acceleration-grid cell size.  Defaults to ``4 * step`` (safe for the 27-cell
        neighbourhood).  Larger = fewer/denser cells; must be ≥ the maximum step.
    """

    def __init__(self, vertices, faces, *, periodic=False, voxel_min=None,
                 voxel_max=None, feature_radius=None, surface_relaxivity_t2=None,
                 permeability=None, intra=None, extra=None, orientation=None, R=None,
                 cell_size=None, cap=None, max_bounces=10):
        V = np.asarray(vertices, np.float64)
        F = np.asarray(faces, np.int64)
        self.vertices = V
        self.faces = F

        if isinstance(periodic, bool):
            periodic = (periodic, periodic, periodic)
        self.periodic = tuple(bool(p) for p in periodic)

        bbmin, bbmax = V.min(0), V.max(0)
        self.vmin = np.asarray(voxel_min, np.float64) if voxel_min is not None else bbmin.copy()
        self.vmax = np.asarray(voxel_max, np.float64) if voxel_max is not None else bbmax.copy()
        self.L = self.vmax - self.vmin

        if feature_radius is None:
            sides = self.vmax - self.vmin
            feature_radius = 0.5 * float(np.min(sides[sides > 0]))
        self.radius = float(feature_radius)              # read by core sub-step auto-tune
        self.reject_escape = True                        # impermeable-leak safety net
        # Reflect at the voxel faces INSIDE the bounce loop (dmipy-sim#61). The alternative --
        # `mesh_bundle.BoxedMesh`, which mirrors the position after the step -- applies a reflection of space
        # with no collision test, so it can teleport a walker across a fibre wall. Measured on the 358-fibre
        # CACTUS bundle over 20 ms: BoxedMesh leaks 19.17% of extra-axonal walkers into fibres (47.4% before
        # its mirror veto). Treating the faces as ordinary specular walls in the same loop makes teleportation
        # impossible by construction, which is what that class's own docstring asks for.
        self.box_reflect = False
        # Barycentric slack in the ray-triangle inside test. 0.0 reproduces the exact (non-watertight)
        # bounds; see `_mt` for the measurement that motivates a non-zero value.
        self.bary_tol = 0.0
        # MC/DC layer 3: a hit within `edge_margin` (barycentric) of a facet EDGE or VERTEX has no
        # well-defined normal, so reflecting off the interpolated one is arbitrary. MC/DC refuses to:
        # `if (col_location == on_edge || on_vertex) bounced_direction = -step`. Back-scatter is retro-
        # reflection rather than specular, but at a seam any choice is arbitrary and this one cannot send the
        # walker through the wall. On 0.25 um CACTUS facets, margin 1e-4 in barycentric units is ~2.5e-11 m,
        # comparable to _NUDGE, so it fires on a narrow band of hits and costs little in physics.
        self.edge_backscatter = False
        self.edge_margin = 1e-4
        # How to bounce when the hit is within `edge_margin` of a facet seam.
        #   'backscatter' -- MC/DC's choice, d_ref = -dh. Cannot escape, but retro-reflection is unphysical
        #                    and costs displacement preferentially at walls, i.e. it biases local time.
        #   'geometric'   -- ours: specular off the facet's GEOMETRIC normal, which is exact for a plane even
        #                    where the vertex-interpolated normal is ill-conditioned. Preserves step length
        #                    and the outward sense, so it should bias the boundary channel less. If it sends
        #                    the walker at the neighbouring facet, the same bounce loop catches that next.
        self.edge_mode = "backscatter"
        # REST-FACET EXCLUSION (ours; an improvement on MC/DC's `Walker::bouncing` + distance epsilon).
        # After a bounce the walker sits `_NUDGE` off the facet it just hit, and a nearly-tangential outgoing
        # ray can re-hit that same facet at a tiny positive t purely through float32 error. MC/DC suppresses
        # this with a distance floor, which necessarily also suppresses genuine hits on OTHER facets at short
        # range -- the measured tunnelling mechanism. The thing that actually must be ignored is one specific
        # facet, and it is known by INDEX, so exclude it by identity and leave every other wall live at any
        # distance. Then a walker stepping into a neighbouring wall can never have that hit discarded.
        self.rest_facet_exclusion = False
        # Which normal the OUTGOING direction is mirrored about.
        #   'smooth'    -- vertex-interpolated, so a coarse mesh diffuses like the curved surface it samples.
        #                  Physically motivated, but it is not the plane the walker is actually behind, so the
        #                  reflected ray can point below the facet -- which is what `_GRAZE` exists to patch,
        #                  and the _GRAZE sweep improving monotonically with size is evidence for this cause.
        #   'geometric' -- mirror about the FACET normal. The facet is the wall the walker cannot pass, so this
        #                  is the confinement truth and needs no _GRAZE rescue. Cost: faceted rather than
        #                  smooth curvature, i.e. slightly wrong tortuosity on a coarse mesh.
        # The relaxation weight keeps using the SMOOTH normal either way -- it is the best local estimate of
        # surface orientation, and it is what the surface channel (C2 relaxivity, C4 MT) is built from. So
        # confinement and physics are taken from the object each describes correctly.
        self.reflect_mode = "smooth"
        self._REST_THR = jnp.float32(8.0 * 4.8e-11)
        self._BLO = jnp.asarray(self.vmin, jnp.float32)
        self._BHI = jnp.asarray(self.vmax, jnp.float32)

        # ---- compartment (intra / extra) surface properties ----
        # The membrane between the intra (inside a cell) and extra compartments can
        # relax spins and let them cross with DIFFERENT strength depending on the
        # side/direction of the collision — the side is already known at the wall
        # (sign of the outward-normal · step).  This first layer supports a
        # side-dependent surface relaxivity ρ and a direction-dependent
        # permeability κ; internally each is stored as a nominal value (read by the
        # engine's step builder) times a per-side/-direction multiplier applied in
        # reflect_with_log_weight / permeate.  Bulk diffusivity and T2 remain single
        # (set on simulate()); per-compartment D/T2 is a later layer.
        intra = dict(intra or {}); extra = dict(extra or {})
        _allowed = {"surface_relaxivity_t2", "D", "T2", "T1"}
        for _side, _d in (("intra", intra), ("extra", extra)):
            _bad = set(_d) - _allowed
            if _bad:
                raise NotImplementedError(
                    f"Mesh {_side}={sorted(_bad)}: supported per-compartment properties are "
                    f"{sorted(_allowed)}.")
        rho_i = intra.get("surface_relaxivity_t2", surface_relaxivity_t2)
        rho_e = extra.get("surface_relaxivity_t2", surface_relaxivity_t2)
        rho_i = float(rho_i) if rho_i is not None else 0.0
        rho_e = float(rho_e) if rho_e is not None else 0.0
        rho_nom = max(rho_i, rho_e)
        self.surface_relaxivity_t2 = rho_nom if rho_nom > 0 else None   # nominal (engine reads this)
        self._rho_mult_intra = jnp.float32(rho_i / rho_nom if rho_nom > 0 else 1.0)
        self._rho_mult_extra = jnp.float32(rho_e / rho_nom if rho_nom > 0 else 1.0)

        # permeability: scalar κ (symmetric, default) OR
        # dict(intra_to_extra=…, extra_to_intra=…) for a direction-dependent (rectifying) wall.
        if isinstance(permeability, dict):
            k_out = float(permeability.get("intra_to_extra", 0.0))   # leaving a cell
            k_in = float(permeability.get("extra_to_intra", 0.0))    # entering a cell
        elif permeability is not None:
            k_out = k_in = float(permeability)
        else:
            k_out = k_in = 0.0
        k_nom = max(k_out, k_in)
        self.permeability = k_nom if k_nom > 0 else None
        self._kappa_mult_out = jnp.float32(k_out / k_nom if k_nom > 0 else 1.0)
        self._kappa_mult_in = jnp.float32(k_in / k_nom if k_nom > 0 else 1.0)

        # ---- per-compartment BULK diffusivity / T2 / T1 (per-step effects) ----
        # index convention: 0 = intra, 1 = extra (matches classify_position).
        # T2 acts while transverse; T1 acts only during longitudinal storage — both
        # are gated by the waveform's coherence flag chi_t in make_step_fn.
        def _pair(key):
            vi, ve = intra.get(key), extra.get(key)
            if vi is None and ve is None:
                return None
            if vi is None or ve is None:
                raise ValueError(f"per-compartment '{key}' needs a value for BOTH intra and "
                                 f"extra (got intra={vi}, extra={ve}).")
            return (float(vi), float(ve))
        D_pair = _pair("D")
        T2_pair = _pair("T2")
        T1_pair = _pair("T1")
        if D_pair is not None and self.permeability is not None and D_pair[0] != D_pair[1]:
            raise NotImplementedError(
                "per-compartment D across a PERMEABLE wall is unequal-D at an interface "
                "(a diffusivity-discontinuity problem not yet handled). Use equal D with "
                "permeability, or an impermeable wall for distinct intra/extra D.")
        self._D_comp = D_pair
        self._T2_comp = T2_pair
        self._T1_comp = T1_pair
        self._has_bulk_comp = any(p is not None for p in (D_pair, T2_pair, T1_pair))
        self._D_comp_jax = jnp.asarray(D_pair, jnp.float32) if D_pair is not None else None
        self._inv_T2_comp_jax = (jnp.asarray([1.0 / T2_pair[0], 1.0 / T2_pair[1]], jnp.float32)
                                 if T2_pair is not None else None)
        self._inv_T1_comp_jax = (jnp.asarray([1.0 / T1_pair[0], 1.0 / T1_pair[1]], jnp.float32)
                                 if T1_pair is not None else None)
        self._D_comp_max = max(D_pair) if D_pair is not None else None

        # ---- placement in the scanner frame (B0 = +z convention) ----
        # The mesh, its grid and periodic box live in the mesh's NATIVE frame and
        # the walk runs entirely there.  `orientation`/`R` declare how that frame
        # is placed in the lab/scanner frame (B0 = +z); simulate() then rotates the
        # ACQUISITION (gradient vectors -- and later B0) into the mesh frame,
        # exactly the "rotate the waveform, not the geometry" convention used by the
        # mesoscopic orchestration.  This keeps the (validated) walk untouched and
        # makes the placement a pure acquisition rotation.  Default: mesh frame IS
        # the scanner frame (native +z, e.g. a periodic/fibre axis, along B0 = +z).
        self.orientation = orientation
        if R is not None:
            Rm = np.asarray(R, np.float64).reshape(3, 3)
        elif orientation is not None:
            Rm = _rotation_from_z(orientation)
        else:
            Rm = None
        # mesh->lab rotation; None when unoriented (simulate skips the hook).
        self._orient_R = None if Rm is None else np.ascontiguousarray(Rm, np.float32)

        # ---- surface-resolution diagnostics + permeability coarseness warning ----
        _e = V[F]
        edge = np.concatenate([
            np.linalg.norm(_e[:, 1] - _e[:, 0], axis=1),
            np.linalg.norm(_e[:, 2] - _e[:, 1], axis=1),
            np.linalg.norm(_e[:, 0] - _e[:, 2], axis=1)])
        self.edge_median = float(np.median(edge))
        self.edge_p90 = float(np.percentile(edge, 90))
        self.edge_feature_ratio = self.edge_median / self.radius
        if self.permeability is not None and self.edge_feature_ratio > _PERM_EDGE_RATIO_MAX:
            warnings.warn(
                f"Mesh is likely too coarse for MC-noise-accurate PERMEABILITY: "
                f"median edge / feature_radius = {self.edge_feature_ratio:.3f} "
                f"(need <~ {_PERM_EDGE_RATIO_MAX}). The permeability faceting bias "
                f"falls ~O(h^2); restricted diffusion and surface relaxivity are "
                f"unaffected. Use a finer / less-decimated mesh for permeability, or "
                f"call .quality_report() for details.",
                stacklevel=2)

        step_l = self.radius / (25.0 if self.permeability is not None else 6.0)
        self.cell_size = float(cell_size) if cell_size is not None else 4.0 * step_l
        self.margin = self.cell_size

        vn = _smooth_vertex_normals(V, F)
        tri_v = V[F]
        tri_vn = vn[F]
        tmin, tmax = tri_v.min(1), tri_v.max(1)
        base = np.arange(len(F))
        all_tri, all_vn = [tri_v], [tri_vn]

        # ghost replication across periodic faces/edges/corners
        for combo in itertools.product((-1, 0, 1), repeat=3):
            if all(c == 0 for c in combo):
                continue
            if any(combo[a] != 0 and not self.periodic[a] for a in range(3)):
                continue
            keep = np.ones(len(F), bool)
            shift = np.zeros(3)
            for a in range(3):
                if combo[a] == -1:
                    keep &= tmax[:, a] > self.vmax[a] - self.margin
                    shift[a] = -self.L[a]
                elif combo[a] == +1:
                    keep &= tmin[:, a] < self.vmin[a] + self.margin
                    shift[a] = +self.L[a]
            if keep.any():
                all_tri.append(tri_v[keep] + shift)
                all_vn.append(tri_vn[keep])
        tri_all = np.concatenate(all_tri, 0)
        vn_all = np.concatenate(all_vn, 0)
        self.n_ghost = len(tri_all) - len(F)

        nrm_all = np.cross(tri_all[:, 1] - tri_all[:, 0], tri_all[:, 2] - tri_all[:, 0])
        nrm_all /= np.maximum(np.linalg.norm(nrm_all, axis=1, keepdims=True), 1e-30)

        # uniform grid over [vmin - margin, vmax + margin]
        self.grid_min = self.vmin - self.margin
        grid_max = self.vmax + self.margin
        self.dims = np.maximum(1, np.ceil((grid_max - self.grid_min) / self.cell_size).astype(int))
        cs = self.cell_size
        lo = np.clip(np.floor((tri_all.min(1) - self.grid_min) / cs).astype(int), 0, self.dims - 1)
        hi = np.clip(np.floor((tri_all.max(1) - self.grid_min) / cs).astype(int), 0, self.dims - 1)
        buckets = defaultdict(list)
        for t in range(len(tri_all)):
            for ix in range(lo[t, 0], hi[t, 0] + 1):
                for iy in range(lo[t, 1], hi[t, 1] + 1):
                    for iz in range(lo[t, 2], hi[t, 2] + 1):
                        buckets[(ix * self.dims[1] + iy) * self.dims[2] + iz].append(t)
        occ = np.array([len(v) for v in buckets.values()]) if buckets else np.array([0])
        C = int(occ.max()) if cap is None else int(cap)
        self.C = C
        self.max_occ = int(occ.max())
        cell_tri = np.full((int(np.prod(self.dims)), C), -1, np.int32)
        self.overflow = 0
        for cid, lst in buckets.items():
            if len(lst) > C:
                self.overflow += len(lst) - C
                lst = lst[:C]
            cell_tri[cid, :len(lst)] = lst

        self._TRIS = jnp.asarray(tri_all, jnp.float32)
        self._VN = jnp.asarray(vn_all, jnp.float32)
        self._NRM = jnp.asarray(nrm_all, jnp.float32)
        self._CENT = jnp.asarray(tri_all.mean(1), jnp.float32)
        self._CELL = jnp.asarray(cell_tri, jnp.int32)
        self._DIMS = tuple(int(x) for x in self.dims)
        self._dims_arr = jnp.asarray(self._DIMS, jnp.int32)
        self._GMIN = jnp.asarray(self.grid_min, jnp.float32)
        self._CS = jnp.float32(self.cell_size)
        self._VMIN = jnp.asarray(self.vmin, jnp.float32)
        self._L = jnp.asarray(self.L, jnp.float32)
        self._PER = jnp.asarray([1.0 if p else 0.0 for p in self.periodic], jnp.float32)
        _scale = float(np.min(self.L))
        # Smallest length the float32 coordinates can still resolve. A candidate collision closer than
        # this is numerical noise -- in practice the walker re-detecting the triangle it just left -- so
        # the segment test discards it. Domain-scaled because float32 precision is relative.
        self._EPS = jnp.float32(1e-7 * _scale)
        # ...which means the post-collision nudge has to CLEAR that noise floor by a wide margin. It is
        # what stands between the walker and the wall it just bounced off; if it is comparable to _EPS,
        # the guard that rejects the re-hit also rejects a genuine wall lying just ahead, and the walker
        # passes through. The two were scaled independently -- _EPS to the box, the nudge to the step --
        # so refining a mesh shrank the nudge while _EPS stayed put and the leak grew: _EPS/nudge ran
        # 0.075 at 384 triangles and 0.160 at 6144, and escape tracked it. Measured at 20,000 walkers,
        # scaling _EPS by 100 (the same squeeze, from the other side) escapes 24% of the ensemble.
        # The floor keeps that ratio bounded whatever the mesh does. 16x, not more: the nudge is an
        # unphysical outward displacement applied at EVERY collision, so buying margin with it has a
        # price. 16 puts the ratio at 1/16 = 0.0625 -- an order of magnitude under the 1.33 that leaks,
        # and tighter than the 0.16 that already measures 0 escapes in 4,000 -- while keeping the
        # displacement at 2.6e-3 of a step on the tightest real geometry (CACTUS: 30 um box, 0.135 um
        # features). A 64x floor would have bounded the ratio just as well and cost 1% of a step there,
        # which is trading one bias for another.
        self._NUDGE = jnp.float32(max(1e-4 * step_l, 16.0 * float(self._EPS)))
        # Floor for the ADAPTIVE nudge (see the bounce body): even in the tightest crevice the walker must be
        # displaced by something float32 can represent at these coordinates, or it stays on the surface and
        # the next collision is discarded as ts <= _EPS.
        self._MIN_NUDGE = jnp.float32(8.0 * float(self._EPS))
        # Opt-in near-surface confinement guards. Both OFF by default: together they cut crossings 2.8x on a
        # CACTUS bundle (39 -> 14 per 3000 walkers over 20 ms, 3.5 sigma) at no cost in boundary local time
        # (0.519x vs 0.522x of (S/V)D, i.e. unchanged), but they do NOT reach zero, and an impermeable wall
        # that leaks 0.47% is still wrong. Enabling them is a strict improvement; relying on them for
        # airtightness is not. See dmipy-sim#61.
        self.adaptive_nudge = False
        self.net_cross_check = False
        # minimum sine of the angle the outgoing ray must make with the triangle it left
        self._GRAZE = jnp.float32(1e-4)
        self._MAX_BOUNCES = int(max_bounces)
        self._OFF = jnp.asarray([[dx, dy, dz] for dx in (-1, 0, 1)
                                 for dy in (-1, 0, 1) for dz in (-1, 0, 1)], jnp.int32)
        self._A = _MeshArrays(NRM=self._NRM, CENT=self._CENT, CELL=self._CELL,
                              dims_arr=self._dims_arr, GMIN=self._GMIN, CS=self._CS,
                              VMIN=self._VMIN, L=self._L, PER=self._PER, OFF=self._OFF)

    # ------------------------------------------------------------------
    def _wrap(self, r):
        w = self._VMIN + jnp.mod(r - self._VMIN, self._L)
        return jnp.where(self._PER > 0, w, r)

    def _gather(self, r_w):
        c = jnp.clip(jnp.floor((r_w - self._GMIN) / self._CS).astype(jnp.int32),
                     0, self._dims_arr - 1)
        nb = jnp.clip(c[None, :] + self._OFF, 0, self._dims_arr - 1)
        cids = (nb[:, 0] * self._DIMS[1] + nb[:, 1]) * self._DIMS[2] + nb[:, 2]
        cand = self._CELL[cids].reshape(-1)
        valid = cand >= 0
        return jnp.where(valid, cand, 0), valid

    def _hit_floor(self, bouncing):
        """Lower bound on an accepted hit distance, conditioned on whether the walker is MID-BOUNCE.

        A fresh step and a continuing bounce need different answers, and one constant cannot serve both. On a
        continuing bounce the walker sits `_NUDGE` off the facet it just hit, so a hit at t~0 is that same
        facet and must be discarded. On a FRESH step the walker also sits `_NUDGE` off a facet -- from the
        previous step -- but now a hit at t~0 is a genuine collision with a wall it is moving INTO, and
        discarding it walks the walker straight through.

        Applying `_EPS` unconditionally (as this engine did) drops exactly that case. Measured on the CACTUS
        bundle: leaked walkers sat at d = -4.8e-11 m (one nudge) before the step, with the crossing 1.1e-10 m
        in -- `_NUDGE`/cos(theta) -- and every crossed triangle already in the gather.

        This follows MC/DC (Rafael-Patino et al.), the reference implementation for mesh diffusion and the
        source of the CACTUS substrates, which keys the same rejection on a `Walker::bouncing` status:
        "a spin that's bouncing ignores collision at 0 (is in a wall)" ... "if we are not bouncing, all
        collisions counts."
        """
        return jnp.where(bouncing, self._EPS, jnp.float32(0.0))

    def _mt(self, r0, d_hat, tri, valid):
        """Moller-Trumbore, with the inside test widened by ``self.bary_tol`` and a guard on ``det``.

        Exact barycentric bounds in float32 are NOT watertight. A ray crossing an edge shared by two
        triangles can be rejected by BOTH -- rounding places it marginally outside each -- so the wall has a
        seam along every edge, and a near-parallel ray drives ``det`` to zero so ``t, u, v`` become garbage
        (previously an unguarded division). Either way the hit is silently dropped and the walker flies
        straight through a wall that was in the gather all along.

        Measured on the 358-fibre CACTUS bundle over 20 ms: of 63 leaked walkers, 58 genuinely crossed a
        triangle, every one of those triangles WAS in the walker's gather (0 absent, so the grid is sound),
        and 43 of 58 crossings sat within 1% of the step start -- at 1.1e-10 m, which is `_NUDGE`/cos(theta).
        The walker is parked at nudge distance off a facet, steps back toward it, and the intersection is
        missed near the facet edge. This is also why the parity guard changed nothing: parity is built on this
        same routine and inherits the same blind spot.

        Widening the bounds makes adjacent triangles overlap slightly, so an edge-crossing ray hits at least
        one of them. That is conservative for confinement -- a spurious hit just reflects the walker -- which
        is the right way to be wrong about an impermeable wall.
        """
        A = tri[:, 0]; E1 = tri[:, 1] - A; E2 = tri[:, 2] - A; T = r0[None] - A
        P = jnp.cross(jnp.broadcast_to(d_hat, E2.shape), E2); det = (P * E1).sum(1)
        Q = jnp.cross(T, E1)
        tiny = jnp.float32(1e-30)
        safe = jnp.where(jnp.abs(det) < tiny, tiny, det)
        t = (Q * E2).sum(1) / safe
        u = (P * T).sum(1) / safe
        v = (Q * jnp.broadcast_to(d_hat, E2.shape)).sum(1) / safe
        b = jnp.float32(self.bary_tol)
        ok = ((u >= -b) & (u <= 1 + b) & (v >= -b) & (u + v <= 1 + b)
              & (jnp.abs(det) > tiny) & valid)
        return jnp.where(ok, t, jnp.inf), u, v

    def classify_position(self, r):
        """Compartment tag: 0 = interior (inside a cell), 1 = exterior.

        Delegates to :func:`_classify_arr` so the seeding jit and this method cannot diverge.
        """
        return _classify_arr(self._A, r)

    def _escaped(self, r, r_out):
        """Did this step cross a wall that the collision search failed to catch?

        The leak safety net, and it may only fire where the classifier can actually tell. `_classify_arr`
        calls a point with an empty 27-cell gather *exterior* -- correct for open space, but inside a pore
        wider than the gather the deep interior reads exterior too (see `_classify_arr`: 19.6% of genuinely
        interior points on a real axon bundle). Comparing the raw labels then flags the *gather frontier*
        as a crossing and rejects the step in BOTH directions, sealing the near-wall shell off from the
        bulk: measured 18.3% of steps discarded at r/R=0.8 on an R=2um mesh sphere with no wall within
        reach. That starves the boundary-local-time channel, which is what surface relaxation and MT
        binding are built from -- the mesh sphere reached only 67% of the analytic local time and an
        equilibrium bound fraction of 0.2296 against 0.3333.

        Gating on `_gather_is_populated` is not merely conservative, it is the physical statement: a walker
        with no triangle anywhere in its gather has no wall within reach, so it cannot have crossed one.
        Genuine escapes stay caught, because a walker that just passed through a wall is by construction
        within one step of it and so still has a populated gather.
        """
        decidable = _gather_is_populated(self._A, r) & _gather_is_populated(self._A, r_out)
        return decidable & (self.classify_position(r) != self.classify_position(r_out))

    def _smooth_normal(self, vnf, nrmf, u, v, idx, d_hat):
        bu, bv = u[idx], v[idx]
        ns = (1 - bu - bv) * vnf[idx, 0] + bu * vnf[idx, 1] + bv * vnf[idx, 2]
        ns = ns / jnp.linalg.norm(ns)
        n = jnp.where(jnp.dot(d_hat, nrmf[idx]) > 0, -ns, ns)  # side by face, dir by smooth
        return n

    # ------------------------------------------------------------------
    def _bounce(self, vnf, nrmf, u, v, idx, dh):
        """Outgoing direction, smooth normal and geometric normal for one collision.

        The smooth (vertex-interpolated) normal is what lets a coarse mesh reflect like the curved
        surface it stands for, so it sets the reflection. But it is an interpolation and can sit far from
        the plane actually hit: 85 degrees at the rim of a capped cylinder, where one vertex is shared by
        the wall and the flat cap. Real data is mostly better behaved and still not clean -- axon06-inner
        of the Winther set runs a median of 3.1 degrees but reaches 179.9, with 0.01% of its 159k
        triangle corners pointing the wrong side of their own face.

        Reflecting off such a normal can send the ray BELOW the triangle it just bounced off; the walker
        then travels into the wall and out the far side. It is a per-COLLISION escape, so it does not
        shrink with the timestep, and sub-stepping makes it worse by resolving more collisions. Grazing
        hits make it the common case rather than a corner case: there dot(dh, n) is small, the outgoing
        ray lies almost in the surface, and a few degrees of normal error is enough to tip it through.

        The geometric normal therefore gets the final say on which side the walker ends up -- the
        outgoing ray is lifted back above the triangle's own plane. On a capped cylinder with the escape
        guard off this takes retention from 64.7% to 94.7% (coarse) and 72.2% to 99.2% (fine); reflecting
        off the geometric normal alone scores the same, i.e. the correction removes the whole effect.
        """
        n = self._smooth_normal(vnf, nrmf, u, v, idx, dh)
        nf = jnp.where(jnp.dot(dh, nrmf[idx]) > 0, -nrmf[idx], nrmf[idx])
        n_ref = nf if self.reflect_mode == "geometric" else n
        d_ref = dh - 2.0 * jnp.dot(dh, n_ref) * n_ref
        d_ref /= jnp.linalg.norm(d_ref)
        c = jnp.dot(d_ref, nf)
        d_ref = jnp.where(c < self._GRAZE, d_ref + (self._GRAZE - c) * nf, d_ref)
        d_ref /= jnp.linalg.norm(d_ref)
        if self.edge_backscatter:
            # barycentric distance to the nearest edge; w = 1-u-v is the third coordinate
            bu, bv = u[idx], v[idx]
            edge = jnp.minimum(jnp.minimum(bu, bv), 1.0 - bu - bv) < jnp.float32(self.edge_margin)
            if self.edge_mode == "geometric":
                d_geo = dh - 2.0 * jnp.dot(dh, nf) * nf
                d_geo = d_geo / jnp.linalg.norm(d_geo)
                alt = d_geo
            else:
                alt = -dh
            d_ref = jnp.where(edge, alt, d_ref)
        return d_ref, n, nf

    def reflect(self, r, step):
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l

        def one(carry, i):
            r0, dh, rem = carry
            # Candidates are gathered once, around the step's START, and reused for every bounce. That is
            # sound only because `physics.collision_sub_steps` caps a sub-step at a fraction of a cell:
            # the whole bounce polyline is then contained in the 27-cell neighbourhood already gathered.
            # Measured: re-gathering per bounce moves retention by 0.0 points, and costs a gather per
            # bounce instead of one per step. Without the sub-step cap it would NOT be sound.
            ts, u, v = self._mt(r0, dh, tri, valid)
            vm = (ts > self._hit_floor(i > 0)) & (ts < rem); ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            d_ref, n, nf = self._bounce(vnf, nrmf, u, v, idx, dh)
            r_hit = r0 + d * dh
            # Nudge along the GEOMETRIC normal: it is perpendicular to the triangle just hit, so it
            # provably clears it, which the interpolated normal does not.
            return (jnp.where(hit, r_hit + self._NUDGE * nf, r0),
                    jnp.where(hit, d_ref, dh),
                    jnp.where(hit, rem - d - self._NUDGE, rem)), hit
        (rf, df, remf), hits = jax.lax.scan(one, (r_w, d_hat, step_l),
                                            jnp.arange(self._MAX_BOUNCES))
        # Flying the leftover path is only safe if the final iteration found NO hit -- that is precisely
        # the statement that nothing lies within `rem` of here, so the free flight was already tested. If
        # it DID hit, the bounce budget ran out mid-step and the leftover is untested: flying it walks the
        # walker straight through whatever it was about to bounce off. Stop at the last verified position
        # instead. That loses a sliver of path length, which shrinks with dt; an escape does not.
        exhausted = hits[-1]
        r_out = r + (rf + df * jnp.where(exhausted, 0.0, jnp.maximum(remf, 0.0)) - r_w)
        if self.reject_escape:
            r_out = jnp.where(self._escaped(r, r_out), r, r_out)
        return r_out

    def _box_face_hit(self, r0, dh, rem):
        """Distance along ``dh`` to the nearest voxel face within ``rem``, and that face's inward normal.

        The six faces are axis-aligned planes, so this is one division per axis -- no triangle test. Only the
        face being APPROACHED on each axis can be hit, which is the one selected by the sign of ``dh``.
        """
        safe = jnp.where(jnp.abs(dh) < jnp.float32(1e-30), jnp.float32(1e-30), dh)
        t = jnp.where(dh > 0, (self._BHI - r0) / safe, (self._BLO - r0) / safe)
        t = jnp.where(jnp.abs(dh) < jnp.float32(1e-30), jnp.inf, t)
        # `t >= 0`, NOT `t > _EPS`. A plane has no on-surface ambiguity to protect against: only the face being
        # approached is selected, so a walker sitting exactly ON a face and moving outward must reflect at
        # t=0 rather than have the hit discarded. With `> _EPS` it instead flies out, and once outside
        # `(hi - r0)/dh` is negative and excluded, so it can never come back -- a one-way trap that stranded
        # 0.4% of walkers outside the box.
        t = jnp.where((t >= 0.0) & (t < rem), t, jnp.inf)
        ax = jnp.argmin(t)
        n = jnp.zeros(3, jnp.float32).at[ax].set(-jnp.sign(dh[ax]))
        return t[ax], n, ax

    def _net_side_changed(self, r_w, r_out, tri, valid):
        """Did the NET displacement change which side of the surface the walker is on? By PARITY.

        The confinement invariant has to hold for the net displacement, not just per bounce: the
        post-collision nudge can land a walker inside a NEIGHBOURING body in a tight crevice, and a hit at
        ``ts <= _EPS`` is discarded for a walker sitting on a surface. Both leave the walker across a wall it
        never legitimately crossed.

        Parity rather than presence. Counting crossings strictly inside the segment and asking whether the
        count is ODD is immune to the endpoint ambiguity that defeats a presence test: a walker that starts ON
        a surface and reflects back crosses it 0 or 2 times, while one that tunnels crosses exactly once. A
        presence test (``any``) cannot tell those apart, so it both misses tunnelling whose crossing sits within
        _EPS of an endpoint and rejects legitimate reflections -- measured at only a 2.8x reduction, never zero.

        Sound because the segment is at most one step long and the gather reaches 1.5 cells, so every triangle
        it can cross is in ``tri``.
        """
        seg = r_out - r_w
        n = jnp.linalg.norm(seg)
        safe = jnp.maximum(n, jnp.float32(1e-30))
        ts, _u, _v = self._mt(r_w, seg / safe, tri, valid)
        crossings = jnp.sum(jnp.where((ts > 0.0) & (ts < n), 1, 0))
        return (crossings % 2 == 1) & (n > 0.0)

    def reflect_with_log_weight(self, r, step, rho_over_D):
        r_w = self._wrap(r)
        if self.box_reflect:
            # Recovery for a walker that is already outside (float32 excursion, or state inherited from a
            # previous configuration). Clamping to the face moves it by at most that excursion, so unlike a
            # mirror it cannot carry the walker across a fibre wall.
            r_w = jnp.clip(r_w, self._BLO, self._BHI)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l

        def one(carry, i):
            r0, dh, rem, last = carry
            ts, u, v = self._mt(r0, dh, tri, valid)
            # i > 0 means the walker is mid-bounce; see `_hit_floor`
            vm = (ts > self._hit_floor(i > 0)) & (ts < rem)
            if self.rest_facet_exclusion:
                # ignore ONLY the facet just bounced off, and only at grazing range; see __init__
                same = jnp.arange(ts.shape[0]) == last
                vm = (ts > 0.0) & (ts < rem) & jnp.logical_not(same & (ts < self._REST_THR))
            ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            d_ref, n, nf = self._bounce(vnf, nrmf, u, v, idx, dh)
            # side of the collision: dh·(outward face normal) > 0 -> spin leaving a
            # cell (intra side), < 0 -> entering (extra side).
            rho_mult = jnp.where(jnp.dot(dh, nrmf[idx]) > 0,
                                 self._rho_mult_intra, self._rho_mult_extra)
            # A voxel face competes with the triangles in the SAME ordered loop, so whichever the walker
            # reaches first acts first and no reflection is ever applied without a collision test. Box faces
            # carry no surface log-weight -- only the axon walls do -- so the surface channel stays
            # myelin-only, exactly as when the mirror lived outside the loop.
            if self.box_reflect:
                d_bx, n_bx, ax_bx = self._box_face_hit(r0, dh, rem)
                use_box = d_bx < d
                d = jnp.where(use_box, d_bx, d)
                hit = hit | (d_bx < jnp.inf)
                d_ref = jnp.where(use_box, dh.at[ax_bx].multiply(-1.0), d_ref)
                nf = jnp.where(use_box, n_bx, nf)
                n = jnp.where(use_box, n_bx, n)
            else:
                use_box = False
            r_hit = r0 + d * dh
            # relaxation still weights by the SMOOTH normal: it is the surface the walker physically
            # met. Only the side it ends up on is decided geometrically.
            cos_a = -jnp.dot(dh, n)
            d_perp = jnp.where(hit & jnp.logical_not(use_box),
                               rho_mult * (rem - d) * cos_a, jnp.float32(0.0))
            # ADAPTIVE nudge. A fixed nudge is a compromise with no good value: too small and it falls below
            # float32 resolution so the walker stays ON the surface and the next hit is discarded by the
            # `ts > _EPS` guard; too large and in a tight crevice it lands the walker inside a NEIGHBOURING
            # body. Measured on a CACTUS bundle (clean seeds, 5 ms, crossings per 3000 walkers): 4.8e-13 m
            # -> 4.70%, 4.8e-12 -> 0.37%, 4.8e-11 (shipped) -> 0.27%, 4.8e-9 -> 1.33%. A minimum, not a
            # plateau, so no constant is safe.
            # Instead cap it at a fraction of the clearance actually available along the outgoing normal,
            # which is what the crevice case violates, while keeping the float32 floor that the on-surface
            # case needs.
            if self.adaptive_nudge:
                ts_n, _un, _vn = self._mt(r_hit, nf, tri, valid)
                clear = jnp.min(jnp.where(ts_n > self._EPS, ts_n, jnp.inf))
                nudge = jnp.minimum(self._NUDGE, jnp.maximum(0.25 * clear, self._MIN_NUDGE))
            else:
                nudge = self._NUDGE
            return (jnp.where(hit, r_hit + nudge * nf, r0),
                    jnp.where(hit, d_ref, dh),
                    jnp.where(hit, rem - d - nudge, rem),
                    jnp.where(hit, idx, last)), (d_perp, hit)
        (rf, df, remf, _lastf), (dperps, hits) = jax.lax.scan(
            one, (r_w, d_hat, step_l, jnp.int32(-1)), jnp.arange(self._MAX_BOUNCES))
        dlog_w = -2.0 * jnp.float32(rho_over_D) * jnp.sum(dperps)
        # see `reflect`: the leftover path may only be flown if the last bounce found nothing
        r_out = r + (rf + df * jnp.where(hits[-1], 0.0, jnp.maximum(remf, 0.0)) - r_w)
        if self.reject_escape:
            escaped = self._escaped(r, r_out)
            if self.net_cross_check:
                escaped = escaped | self._net_side_changed(r_w, r_w + (r_out - r), tri, valid)
            return jnp.where(escaped, r, r_out), jnp.where(escaped, jnp.float32(0.0), dlog_w)
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l
        u_rand = jax.random.uniform(perm_key, dtype=jnp.float32)

        def one(carry, i):
            r0, dh, rem, decided, dlogw = carry
            ts, u, v = self._mt(r0, dh, tri, valid)
            vm = (ts > self._hit_floor(i > 0)) & (ts < rem); ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            d_ref, n, nf = self._bounce(vnf, nrmf, u, v, idx, dh)
            # crossing direction: dh·(outward normal) > 0 -> leaving a cell
            # (intra->extra), < 0 -> entering (extra->intra).
            outward = jnp.dot(dh, nrmf[idx]) > 0
            kappa_mult = jnp.where(outward, self._kappa_mult_out, self._kappa_mult_in)
            rho_mult = jnp.where(outward, self._rho_mult_intra, self._rho_mult_extra)
            cos_a = -jnp.dot(dh, n)
            d_perp = (rem - d) * cos_a
            first_hit = hit & (~decided)
            p_t = jnp.minimum(1.0, 2.0 * jnp.float32(kappa_over_D) * kappa_mult * d_perp)
            transmit = first_hit & (u_rand < p_t)
            r_hit = r0 + d * dh
            do_reflect = hit & ~transmit
            r_new = jnp.where(do_reflect, r_hit + self._NUDGE * nf, r0 + rem * dh)
            d_new = jnp.where(do_reflect, d_ref, dh)
            rem_new = jnp.where(do_reflect, rem - d - self._NUDGE, jnp.float32(0.0))
            dperp_refl = jnp.where(first_hit & ~transmit, rho_mult * d_perp, jnp.float32(0.0))
            return (r_new, d_new, rem_new, decided | first_hit,
                    dlogw - 2.0 * jnp.float32(rho_over_D) * dperp_refl), do_reflect
        (rf, df, remf, _, dlogw), refls = jax.lax.scan(
            one, (r_w, d_hat, step_l, False, jnp.float32(0.0)), jnp.arange(self._MAX_BOUNCES))
        # see `reflect`. A transmitted walker already zeroed `rem`, so only a still-bouncing final
        # iteration means the budget ran out with untested path left.
        r_out = r + (rf + df * jnp.where(refls[-1], 0.0, jnp.maximum(remf, 0.0)) - r_w)
        return r_out, dlogw

    # ------------------------------------------------------------------
    def init_positions(self, n_walkers, key, intra=True):
        """Seed walkers inside (intra=True) or outside the surface, by exact rejection sampling.

        Every candidate is decided by :func:`mesh_contains` -- ray-crossing parity, a global test.

        It used to be decided by the cell-gather classifier, with the exact test reserved for points
        whose gather was empty. That is the wrong way round. The classifier is nearest-CENTROID
        sidedness, so the points it is least able to judge are the ones NEAR a wall, which are exactly
        the points with a populated gather; the branch that got the exact treatment was the one that
        needed it least. Measured on a closed cylinder, seeding "intra" put 6.3% of the pool OUTSIDE the
        surface at the coarsest grid setting, 1.1% and 0.7% as it was refined -- coarser mesh, bigger
        triangles, worse. Walkers that were never inside then read as walkers that escaped, which is how
        this masqueraded as a leak while #40/#41 were being chased.

        The cost is bearable because ``mesh_contains`` is itself a cascade: ``mesh_inside`` proposes and
        only the proposals are ray cast, and it has no false-OUTSIDE, so nothing genuinely inside is
        discarded before the exact stage sees it. Seeding happens once per simulation, at setup.

        Parity needs a CLOSED surface. A deliberately open one -- a periodic tube, whose rims are open
        because the geometry continues through them -- has no parity, so those keep the old cell-gather
        path and its known inaccuracy. That is a preserved behaviour, not an endorsement; the accurate
        treatment for an open surface is tracked with this issue.
        """
        from .susceptibility_field import mesh_contains
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
        V = np.asarray(self.vertices, float)
        F = np.asarray(self.faces, np.int64)
        out, need = [], n_walkers
        if self._surface_is_closed():
            while need > 0:
                pts = rng.uniform(self.vmin, self.vmax, (max(need * 4, 1024), 3)).astype(np.float32)
                inside = mesh_contains(V, F, pts.astype(float))
                out.append(pts[inside if intra else ~inside])
                need = n_walkers - sum(len(a) for a in out)
            return jnp.asarray(np.concatenate(out)[:n_walkers], jnp.float32)

        warnings.warn(
            "seeding an OPEN surface: ray parity is undefined through its rims, so this falls back to "
            "the cell-gather classifier, which decides sidedness from the nearest triangle centroid and "
            "misplaces walkers near a wall (measured 6.3% on a coarse closed cylinder). Cap the surface "
            "to get exact seeding.", stacklevel=2)
        want = 0 if intra else 1
        classify = jax.jit(jax.vmap(_classify_arr, in_axes=(None, 0)))
        populated = jax.jit(jax.vmap(_gather_is_populated, in_axes=(None, 0)))
        n_resolved = 0
        while need > 0:
            pts = rng.uniform(self.vmin, self.vmax, (max(need * 4, 1024), 3)).astype(np.float32)
            jpts = jnp.asarray(pts)
            lab = np.array(classify(self._A, jpts))     # copy: a jax-backed view is read-only
            undecided = ~np.asarray(populated(self._A, jpts))
            if undecided.any():
                # an empty gather is not a verdict, it is a default -- resolve those exactly, exactly
                # as this path did before (#39); an open surface that reaches here kept working because
                # it never produced an undecided point, and that is preserved rather than reasoned about
                inside = mesh_contains(V, F, pts[undecided].astype(float))
                lab[undecided] = np.where(inside, 0, 1)
                n_resolved += int(undecided.sum())
            out.append(pts[lab == want])
            need = n_walkers - sum(len(a) for a in out)
        if n_resolved:
            self._n_gather_undecided = n_resolved
        return jnp.asarray(np.concatenate(out)[:n_walkers], jnp.float32)

    def _surface_is_closed(self):
        """Does every edge have two faces? Cached -- it is a property of the mesh, not of a call."""
        if getattr(self, "_closed", None) is None:
            try:
                import trimesh
                m = trimesh.Trimesh(vertices=np.asarray(self.vertices, float),
                                    faces=np.asarray(self.faces, np.int64), process=False)
                n_open = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
                self._closed = bool(n_open == 0)
            except Exception:
                self._closed = False
        return self._closed

    def classify_position_carry(self, r, comp_prev):
        """Compartment label, keeping ``comp_prev`` wherever the gather cannot decide.

        A walker whose 27-cell gather is empty has no wall within reach, so it cannot have crossed one
        since the previous step -- its compartment is whatever it already was. Re-deriving the label from
        local geometry instead makes the walker's compartment depend on how finely its own fibre happens
        to be meshed: deep inside a thick fibre the gather is empty, the raw classifier defaults to
        exterior, and an intra-axonal walker reads as extra-axonal for as long as it stays away from the
        wall (measured: 19.6% of interior points on a real axon bundle, 66% on a subdivided tube).

        Compartment is a state that changes at crossings, not a property to be re-measured every step, so
        carrying it is both cheaper and more correct. Only the initial labels need an exact test.
        """
        return jnp.where(_gather_is_populated(self._A, r),
                         _classify_arr(self._A, r), comp_prev).astype(jnp.int32)

    def classify_positions_exact(self, pts):
        """Initial labels, resolving undecidable points with exact parity (setup-time, host-side)."""
        pts = np.asarray(pts, float)
        lab = np.array(jax.jit(jax.vmap(_classify_arr, in_axes=(None, 0)))(
            self._A, jnp.asarray(pts, jnp.float32)))
        undecided = ~np.asarray(jax.jit(jax.vmap(_gather_is_populated, in_axes=(None, 0)))(
            self._A, jnp.asarray(pts, jnp.float32)))
        if undecided.any():
            from .susceptibility_field import mesh_contains
            inside = mesh_contains(np.asarray(self.vertices, float),
                                   np.asarray(self.faces, np.int64), pts[undecided])
            lab[undecided] = np.where(inside, 0, 1)
        return jnp.asarray(lab, jnp.int32)

    def quality_report(self, verbose=True):
        """Surface-resolution diagnostics + per-effect accuracy verdict.

        Returns a dict; also prints a table when ``verbose``.  Uses trimesh (if
        installed) to add watertight / component info.  The key number is
        ``edge_feature_ratio`` = median edge / feature_radius: permeability needs
        it ``<~ 0.05`` to reach the MC noise floor; diffusion and surface
        relaxivity are fine at much coarser ratios.
        """
        ratio = self.edge_feature_ratio
        perm_ok = ratio <= _PERM_EDGE_RATIO_MAX
        rep = {
            "n_vertices": int(len(self.vertices)),
            "n_faces": int(len(self.faces)),
            "n_ghost_faces": int(self.n_ghost),
            "feature_radius": self.radius,
            "edge_median": self.edge_median,
            "edge_p90": self.edge_p90,
            "edge_feature_ratio": ratio,
            "grid_dims": tuple(int(x) for x in self.dims),
            "grid_max_occupancy": int(self.max_occ),
            "grid_overflow": int(self.overflow),
            "periodic": self.periodic,
            "diffusion_noise_floor": True,
            "relaxivity_noise_floor": True,
            "permeability_noise_floor": bool(perm_ok),
        }
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
            rep["watertight"] = bool(tm.is_watertight)
            rep["n_components"] = int(tm.body_count)
        except Exception:
            pass
        if verbose:
            print(f"Mesh quality report")
            print(f"  vertices/faces        : {rep['n_vertices']:,} / {rep['n_faces']:,}"
                  + (f"  (+{rep['n_ghost_faces']:,} periodic ghosts)" if self.n_ghost else ""))
            if "watertight" in rep:
                print(f"  watertight/components : {rep['watertight']} / {rep['n_components']}")
            print(f"  feature_radius        : {self.radius*1e6:.3f} um")
            print(f"  edge median / p90     : {self.edge_median*1e6:.3f} / {self.edge_p90*1e6:.3f} um")
            print(f"  edge/feature ratio    : {ratio:.3f}  (permeability needs <~ {_PERM_EDGE_RATIO_MAX})")
            print(f"  grid dims / max-occ   : {rep['grid_dims']} / {rep['grid_max_occupancy']}"
                  + (f"  OVERFLOW={rep['grid_overflow']}" if self.overflow else ""))
            print(f"  MC-noise-floor accuracy:")
            print(f"    restricted diffusion : YES")
            print(f"    surface relaxivity   : YES")
            print(f"    permeability         : {'YES' if perm_ok else 'NO -- mesh too coarse; use a finer mesh'}")
        return rep

    @classmethod
    def from_ply(cls, path, scale=1.0, recenter=False, **kwargs):
        """Construct a Mesh directly from a mesh file (see :func:`load_ply`)."""
        V, F = load_ply(path, scale=scale, recenter=recenter)
        return cls(V, F, **kwargs)

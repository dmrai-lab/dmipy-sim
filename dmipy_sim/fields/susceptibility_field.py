"""Mesh/grid susceptibility off-resonance field (isotropic + anisotropic myelin).

The analytic ``PackedMyelinatedCylinders.compute_susceptibility_field_maps`` gives the
off-resonance field of *straight, parallel* hollow cylinders in closed form (the
2-D ``Phi_C/Phi_S/Phi_0`` maps).  That specialisation cannot represent a myelin
sheath whose geometry varies along the axon (undulating thickness, a bare sector,
a branch, or any PLY mesh), because there is no global cylinder axis to define the
perpendicular plane or the radial (lipid) director.

This module computes the field of an **arbitrary 3-D susceptibility distribution**
on a regular grid via the Lorentz-sphere-corrected k-space dipole model
(Salomir 2003; Marques & Bowtell 2005 for the isotropic case; Wharton & Bowtell
2012 for the anisotropic myelin tensor).  It handles

* an **isotropic** scalar susceptibility ``chi_I`` (special case), and
* an **anisotropic** uniaxial myelin tensor with the symmetry axis along the local
  radial (membrane-normal) direction ``n``:  ``chi = chi_I I + chi_A (n n^T - I/3)``.

The local director ``n`` comes from the geometry itself (the gradient of a signed
distance field, or the mesh's smooth surface normal), so the model generalises to
undulating / half-bare / meshed sheaths where an analytic normal does not exist.

Design mirroring the existing susceptibility phasor
---------------------------------------------------
Because the along-field component ``dB(r)`` is a **quadratic form in the unit main
field direction ``H``** and **linear in ``(chi_I, chi_A)``**, the geometry-only part
factorises into a small set of basis grids (:func:`field_basis`) computed once by
FFT.  Any main-field direction, field strength ``B0`` and susceptibility values are
then applied by a cheap real contraction (:func:`assemble_field`) — the 3-D
generalisation of applying ``(theta, alpha, dchi_a, B0)`` to ``Phi_C/Phi_S/Phi_0``.
Sampling the basis along a stored trajectory (:func:`sample_grid`) therefore lets a
single mesh walk replay an entire ``(B0-direction, B0, chi_I, chi_A, g)`` sweep.

Units: lengths m, ``voxel_size`` m, ``B0`` Tesla, susceptibilities dimensionless
(SI volume susceptibility; e.g. ``chi_A = -0.1e-6``).  ``dB`` is returned in Tesla.
The phase accrued by a spin is ``phi = GAMMA * integral eps(t) dB(r(t)) dt`` with
``GAMMA`` from :mod:`dmipy_sim.constants`, exactly as the analytic path.

Reference / zero level
----------------------
The k=0 (mean-field) term is set to zero — the Lorentz-sphere convention shared by
QSM forward models and by the analytic maps here.  The spatial *structure* (and
hence every intra/extra difference and the diffusion dephasing) is independent of
this global offset, and a spatially uniform offset is exactly refocused by a spin
echo, so this choice is physically inert for SE/STE signals.
"""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple, Optional

import numpy as np

# Symmetric 3x3 tensor stored as 6 components in this fixed order.
_SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

# The node spacing of a rasterised field basis follows the field source's thinnest shell (dmipy-sim#304). MEASURED
# against the Wharton-Bowtell hollow cylinder with a 0.120 um sheath (the CACTUS bundle's thinnest), the boundary
# voxels supersampled 4 x 4 x 4 (tests/test_field_resolution.py): the lumen's anisotropic term is off by 1.4 % at one
# node across the sheath, 0.6 % at one and a half, 0.27 % at two and 0.38 % at three, the isotropic lumen null 1.1 %,
# 0.3 %, 0.3 % and 0.2 % of chi -- two nodes across reach the reference's own floor. The dipole kernel does not decay
# with |k|, so a hard binary edge rings into the lumen (2.6 % of chi B0 where the exact answer is 0); the supersample
# of the boundary voxels brings that to ~0.1 %, and four is where it settles.
FIELD_NODES_ACROSS = 2.0
FIELD_MASK_SUPERSAMPLE = 4
# The peak host bytes a basis build takes per node, MEASURED (a meshed hollow cylinder at 4.1e6 and 8.0e6 nodes: the
# slope of the process's peak RSS is 464 B/node -- the 13 float64 channels, one channel's k-space work at a time
# (field_basis_channels) and the membership tests of the supersampled edge in chunks of _EDGE_CHUNK_POINTS), rounded up.
FIELD_BYTES_PER_NODE = 500
_EDGE_CHUNK_POINTS = 20_000_000                 # sub-samples per membership call on the boundary voxels (~0.5 GB)


def field_resolution(thinnest_shell):
    """The node spacing (metres) that resolves a field source whose thinnest shell is ``thinnest_shell`` (metres):
    :data:`FIELD_NODES_ACROSS` nodes across it."""
    t = float(thinnest_shell)
    if not t > 0:
        raise ValueError(f"the thinnest shell must be a positive length, got {thinnest_shell!r}")
    return t / FIELD_NODES_ACROSS


def field_node_budget(ceiling=None):
    """The nodes a field basis may have on this host: half of the memory ceiling (``ceiling`` bytes, else
    :func:`dmipy_sim.run.memory_ceiling`) over the build's peak bytes per node (:data:`FIELD_BYTES_PER_NODE`)."""
    if ceiling is None:
        from ..run import memory_ceiling
        ceiling = memory_ceiling()
    if ceiling is None:
        raise RuntimeError("no memory ceiling is readable on this host: pass field_budget= (nodes) explicitly")
    return float(0.5 * float(ceiling) / FIELD_BYTES_PER_NODE)



def field_raster_shape(box_min, box_max, res):
    """The nodes per axis ``(3,)`` of the raster :func:`predicate_field_basis` builds over the box ``box_min .. box_max``
    at node spacing ``res`` (metres): the box's side over ``res``, rounded, at least one."""
    side = np.asarray(box_max, float) - np.asarray(box_min, float)
    return np.maximum(1, np.round(side / float(res)).astype(int))


#: The device copies of a raster a walk's program holds: the raster is a program argument and XLA's allocation for the
#: argument-and-result buffers of the compiled walk is twice its bytes (measured on the CUDA walk of a 612^3 x 12
#: float32 raster, 11 002 604 544 B: 22 005 209 088 B of "input/output arguments").
FIELD_DEVICE_COPIES = 2
#: The channels the walk reads on the device: ``iso_local``, five ``iso_P`` (``iso_P_zz`` is formed from the trace) and
#: six ``aniso_G`` -- the six of an isotropic-only source.
FIELD_DEVICE_CHANNELS = 12


def field_device_bytes(n_nodes, n_channels=FIELD_DEVICE_CHANNELS):
    """The device bytes the walk of a raster of ``n_nodes`` nodes needs for it: :data:`FIELD_DEVICE_COPIES` copies of
    ``n_channels`` float32 channels (:meth:`FieldGrid.device_arrays`)."""
    return int(FIELD_DEVICE_COPIES) * 4 * int(n_channels) * int(n_nodes)


def field_device_limit():
    """``(bytes, source)``: the memory JAX may use on the device the walk runs on -- ``bytes_limit`` of the first
    device's allocator (the card's share after ``XLA_PYTHON_CLIENT_MEM_FRACTION`` / ``XLA_PYTHON_CLIENT_PREALLOCATE``)
    -- and where it was read from; ``(None, "cpu")`` on a CPU device, where the raster lives in host memory and the host
    budget (:func:`field_node_budget`) governs. An accelerator that reports no limit is an error, not an unlimited
    device."""
    import jax
    dev = jax.devices()[0]
    if dev.platform == "cpu":
        return None, "cpu"
    stats = dev.memory_stats() or {}
    lim = stats.get("bytes_limit")
    if not lim:
        raise RuntimeError(f"the {dev.platform} device {dev} reports no bytes_limit in memory_stats(); pass the device "
                           "memory limit (bytes) explicitly")
    return int(lim), f"jax.devices()[0].memory_stats()['bytes_limit'] of {dev}"


class FieldGridRecord:
    """The record of a :class:`FieldGrid` a walk read along its path (a walk file, a loaded walk): the grid's
    placement, shape and raster certificate and its channel names -- no values, so it evaluates nothing. What a
    pack of a walk that carries its field modes needs (``susceptibility_grid`` in the pack's header)."""

    def __init__(self, meta):
        self.meta = dict(meta)
        from dmipy_replay.replay.field import CHANNEL_NAMES, ISO_CHANNEL_NAMES
        self.channel_names = CHANNEL_NAMES if self.meta.get("has_aniso") else ISO_CHANNEL_NAMES
        self.origin = np.asarray(self.meta["origin"], float)
        self.periodic = tuple(bool(p) for p in self.meta.get("periodic") or (False, False, False))
        self.certificate = self.meta.get("raster")
        self.basis = dict(voxel_size=np.asarray(self.meta["voxel_size"], float), shape=tuple(int(s) for s in self.meta["shape"]),
                          aniso_G=(True if self.meta.get("has_aniso") else None))


class FieldGrid(NamedTuple):
    """A substrate's static susceptibility field on a voxel grid, ready for a replay pack's field tier.

    ``basis`` is the geometry-only field basis (:func:`field_basis` / :func:`mesh_field_basis`:
    ``iso_local``, ``iso_P``, optional ``aniso_G``, ``shape``, ``voxel_size``) from which
    :func:`assemble_field` builds the field for any (B0 direction, B0, chi_iso, chi_aniso) at
    replay; ``origin`` is the world position of voxel (0, 0, 0). No susceptibility value lives here:
    the grid is the substrate's shape, the field is a replay knob. ``certificate`` is the raster's record when
    a spec built it (:func:`dmipy_sim.spec.field_grid_of_spec`): the node spacing, the nodes across the thinnest
    shell, and the shell fraction the raster holds against a Monte-Carlo membership estimate. ``periodic`` is per
    axis whether the grid is one period of the field (a packed substrate's cell): a periodic axis is sampled
    wrapped, so a continuous position outside the grid reads the field of its image in the cell; any other axis is
    clamped at the edge.
    """
    basis: dict
    origin: np.ndarray
    certificate: Optional[dict] = None
    periodic: tuple = (False, False, False)

    @property
    def meta(self):
        """The grid's record (:class:`FieldGridRecord`): placement, shape, periodicity, raster certificate."""
        return dict(origin=np.asarray(self.origin, float).tolist(), voxel_size=np.asarray(self.basis["voxel_size"], float).tolist(),
                    shape=[int(s) for s in self.basis["shape"]], has_aniso=(self.basis.get("aniso_G") is not None),
                    periodic=[bool(p) for p in self.periodic], raster=self.certificate)

    @property
    def channel_names(self):
        """The channels :meth:`channels` returns, in order: ``iso_local``, the five stored ``iso_P`` (``iso_P_zz`` is
        formed from the trace), and the six ``aniso_G`` when the basis carries them."""
        from dmipy_replay.replay.field import CHANNEL_NAMES, ISO_CHANNEL_NAMES
        return CHANNEL_NAMES if self.basis.get("aniso_G") is not None else ISO_CHANNEL_NAMES

    @property
    def trace_residual(self):
        """The trace identity's residual of this basis, ``max |iso_P_xx + iso_P_yy + iso_P_zz - 3 iso_local| / (3 max
        |iso_local|)``: 0 for an un-apodised dipole solve, the diagnostic a producer states for a grid whose numerics
        break it (a k-space window applied to ``iso_P`` but not to ``iso_local``). None when there is no local term."""
        loc = np.asarray(self.basis["iso_local"], float); P = np.asarray(self.basis["iso_P"], float)
        scale = float(np.abs(loc).max())
        return None if scale == 0.0 else float(np.abs(P[0] + P[1] + P[2] - 3.0 * loc).max() / (3.0 * scale))

    def grids(self, dtype=np.float64):
        """The basis grids stacked in :attr:`channel_names` order, ``(6 | 12, nx, ny, nz)``."""
        g = [np.asarray(self.basis["iso_local"], dtype)] + [np.asarray(self.basis["iso_P"][c], dtype) for c in (0, 1, 3, 4, 5)]
        if self.basis.get("aniso_G") is not None:
            g += [np.asarray(self.basis["aniso_G"][c], dtype) for c in range(6)]
        return np.stack(g)

    def channels(self, points):
        """``(n, 6 | 12)`` basis channels at ``points`` ``(n, 3)`` (metres), trilinear on the grid: the same
        protocol as :meth:`dmipy_sim.fields.strand_field.StrandFieldBasis.channels`."""
        P = np.asarray(points, float).reshape(-1, 3)
        vs = np.asarray(self.basis["voxel_size"], float); org = np.asarray(self.origin, float)
        return np.stack([sample_grid(g, P, org, vs, periodic=self.periodic) for g in self.grids()], axis=1)

    def device_arrays(self):
        """``(grids, origin, voxel_size)`` as float32 device arrays: what :func:`grid_channels_at` reads, passed to
        a jitted walk as arguments (so one compiled walk serves every grid of a shape)."""
        import jax.numpy as jnp
        return (jnp.asarray(self.grids(np.float32)), jnp.asarray(np.asarray(self.origin, float).ravel(), jnp.float32),
                jnp.asarray(_as_voxel_size(self.basis["voxel_size"], 3), jnp.float32))

    def device_read(self):
        """``(arrays, read, key)``: the device arrays a walk passes as arguments (:meth:`device_arrays`), ``read(arrays,
        r) -> (6 | 12,)`` at one position (:func:`grid_channels_at`, the periodic axes wrapped), and the key a compiled
        walk is cached on."""
        periodic = tuple(bool(p) for p in self.periodic)
        arrays = self.device_arrays()
        return arrays, (lambda a, r: grid_channels_at(*a, periodic, r)), ("grid", tuple(arrays[0].shape), periodic)

    def channels_at_device(self):
        """The jitted ``(n, 3) -> (n, 6 | 12)`` read of :func:`grid_channels_at` on the device, float32: the read a
        walk takes at every sub-step (:func:`dmipy_sim.engine.core.simulate_trajectories` with ``field_basis=``)."""
        import jax
        arrays, periodic = self.device_arrays(), tuple(bool(p) for p in self.periodic)
        f = jax.jit(jax.vmap(lambda r, a: grid_channels_at(*a, periodic, r), in_axes=(0, None)))
        return lambda points: f(points, arrays)


def grid_channels_at(grids, origin, voxel_size, periodic, r):
    """The channels ``(C,)`` of ``grids`` ``(C, nx, ny, nz)`` at one position ``r`` ``(3,)`` (metres), trilinear
    between voxel centres (voxel ``i`` centred at ``origin + (i + 1/2) voxel_size``): an axis ``periodic`` names
    (a static tuple of three) wraps, any other clamps at its edge -- :func:`sample_grid`'s rule at ``order=1``, on
    the device. Elementwise arithmetic and gathers only: a position is never contracted by a matmul, which a GPU
    may run at TF32."""
    import jax.numpy as jnp
    shape = grids.shape[1:]
    idx = (r - origin) / voxel_size - 0.5
    lo, hi, fr = [], [], []
    for a in range(3):
        n = shape[a]
        x = jnp.mod(idx[a], n) if periodic[a] else idx[a]
        i0 = jnp.floor(x)
        fr.append(x - i0)
        i0 = i0.astype(jnp.int32)
        if periodic[a]:
            lo.append(jnp.mod(i0, n)); hi.append(jnp.mod(i0 + 1, n))
        else:
            lo.append(jnp.clip(i0, 0, n - 1)); hi.append(jnp.clip(i0 + 1, 0, n - 1))
    out = jnp.zeros(grids.shape[0], grids.dtype)
    for cx in (0, 1):
        wx = fr[0] if cx else 1.0 - fr[0]
        ix = hi[0] if cx else lo[0]
        for cy in (0, 1):
            wy = fr[1] if cy else 1.0 - fr[1]
            iy = hi[1] if cy else lo[1]
            for cz in (0, 1):
                wz = fr[2] if cz else 1.0 - fr[2]
                iz = hi[2] if cz else lo[2]
                out = out + (wx * wy * wz) * grids[:, ix, iy, iz]
    return out

    def field(self, points, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0):
        """``dB`` (Tesla) at ``points`` for one configuration."""
        from dmipy_replay.replay.field import contract
        return contract(self.channels(points), b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)


def field_grid_of(geometry, *, res=None, include_aniso=True, box=None, margin=None, n_z=4,
                  mask_supersample=FIELD_MASK_SUPERSAMPLE, kspace_lowpass=0.5):
    """The :class:`FieldGrid` of an analytic myelinated substrate, for a pack's field tier.

    Rasterises the sheath of a :class:`~dmipy_sim.geometry.MyelinatedCylinder` or
    :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` (every voxel's myelin occupancy, partial
    volume at the two walls from ``mask_supersample``^2 sub-samples, and the radial director toward the
    nearest axon) onto a voxel grid of pitch ``res`` (default :func:`field_resolution` of the thinnest sheath:
    :data:`FIELD_NODES_ACROSS` nodes across it) and calls :func:`field_basis`, exactly as
    :func:`mesh_field_basis` does for a meshed axon -- so a myelinated cylinder's pack carries the
    same C3 channel as a mesh pack and is replayed by the same code.

    The grid is the geometry's periodic cell for a packed substrate (the FFT solve is periodic, so
    the images are exact, and the grid is sampled periodically), or ``box = (lo, hi)`` (2-vectors, metres)
    for a single cylinder, default ``margin`` (default ``4 * outer_radius``) beyond the sheath; the field is
    invariant along the axis, so ``n_z`` voxels carry it (periodic in z). The basis is in the geometry frame (axis = z):
    give ``b0_dir`` at replay in that frame. ``include_aniso`` (default True) adds the six anisotropic
    grids so ``chi_aniso`` can be applied at replay; False halves the work for an isotropic-only pack.
    """
    from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
    if isinstance(geometry, MyelinatedCylinder):
        centers = np.zeros((1, 2)); inner = np.array([geometry.inner_radius]); outer = np.array([geometry.outer_radius])
        periodic = False
        if box is None:
            m = 4.0 * float(outer[0]) if margin is None else float(margin)
            half = float(outer[0]) + m
            box = (np.array([-half, -half]), np.array([half, half]))
        lo, hi = (np.asarray(box[0], float), np.asarray(box[1], float))
    elif isinstance(geometry, PackedMyelinatedCylinders):
        N = geometry.N_actual
        centers = np.asarray(geometry._centers_np[:N], float)
        inner = np.asarray(geometry._inner_radii_np[:N], float); outer = np.asarray(geometry._outer_radii_np[:N], float)
        periodic = True
        L = float(geometry._L_float)
        if box is not None:
            raise ValueError("a packed substrate's field grid is its periodic cell; box= does not apply")
        lo, hi = np.array([-L / 2, -L / 2]), np.array([L / 2, L / 2])
    else:
        raise TypeError(f"field_grid_of takes a MyelinatedCylinder or PackedMyelinatedCylinders, got "
                        f"{type(geometry).__name__}; a mesh substrate uses mesh_field_basis")
    if res is None:
        res = field_resolution(float(np.min(outer - inner)))
    side = hi - lo
    n_xy = np.maximum(2, np.round(side / res).astype(int))
    vs_xy = side / n_xy
    vs = np.array([vs_xy[0], vs_xy[1], float(vs_xy.mean())])
    # voxel centres; the sheath occupancy by SS x SS in-plane sub-samples of every voxel
    SS = max(1, int(mask_supersample))
    off = (np.arange(SS) + 0.5) / SS - 0.5
    xs = lo[0] + (np.arange(n_xy[0]) + 0.5) * vs[0]
    ys = lo[1] + (np.arange(n_xy[1]) + 0.5) * vs[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros(X.shape); rad = np.zeros(X.shape + (2,)); dmin = np.full(X.shape, np.inf)
    for c, a, b in zip(centers, inner, outer):
        dx, dy = X - c[0], Y - c[1]
        if periodic:
            dx -= side[0] * np.round(dx / side[0]); dy -= side[1] * np.round(dy / side[1])
        r = np.hypot(dx, dy)
        near = r < dmin                                               # nearest axon owns the director
        rad[near] = np.stack([dx, dy], -1)[near] / np.maximum(r[near], 1e-30)[:, None]
        dmin = np.minimum(dmin, r)
        for ox in off:
            for oy in off:
                rr = np.hypot(dx + ox * vs[0], dy + oy * vs[1])
                occ += ((rr > a) & (rr < b)) / (SS * SS)
    occ = np.clip(occ, 0.0, 1.0)
    mask = np.repeat(occ[:, :, None], int(n_z), axis=2)
    radial = np.zeros(mask.shape + (3,)); radial[..., 0] = rad[:, :, None, 0]; radial[..., 1] = rad[:, :, None, 1]
    basis = field_basis(mask, radial, vs, include_aniso=bool(include_aniso),
                        kspace_lowpass=(None if periodic else kspace_lowpass))
    basis["shell_fraction"] = float(mask.mean())
    origin = np.array([lo[0], lo[1], -0.5 * int(n_z) * vs[2]])
    return FieldGrid(basis, origin, periodic=(periodic,) * 3)

def _unit(v, axis=-1, eps=1e-30):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _as_voxel_size(voxel_size, ndim=3):
    vs = np.atleast_1d(np.asarray(voxel_size, float))
    if vs.size == 1:
        vs = np.repeat(vs, ndim)
    if vs.size != ndim:
        raise ValueError(f"voxel_size must be a scalar or length-{ndim}, got {vs}")
    return vs


def _khat(shape, voxel_size):
    """Unit wave-vector component grids ``khat_i`` and a DC mask.

    Returns ``(khat, dc)`` where ``khat`` has shape ``(3, *shape)`` (the components
    of ``k / |k|``, zero at k=0) and ``dc`` is the boolean k=0 location.
    """
    vs = _as_voxel_size(voxel_size, len(shape))
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(shape, vs)]
    K = np.meshgrid(*ks, indexing="ij")
    k2 = sum(k ** 2 for k in K)
    dc = k2 == 0.0
    inv = np.where(dc, 0.0, 1.0 / np.sqrt(np.where(dc, 1.0, k2)))
    khat = np.stack([k * inv for k in K], axis=0)     # (3, *shape); zero at DC
    return khat, dc


def myelin_susceptibility_tensor(myelin_mask, radial_dir, chi_iso=0.0, chi_aniso=0.0):
    """Per-voxel symmetric susceptibility tensor of a myelin distribution.

    ``chi(r) = chi_iso * I + chi_aniso * (n n^T - I/3)`` inside the myelin, zero
    elsewhere.  With ``chi_aniso = 0`` this is a purely isotropic source (the
    general special case); with ``chi_iso = 0`` a purely anisotropic one.

    Parameters
    ----------
    myelin_mask : (Nx,Ny,Nz) array
        Myelin volume fraction per voxel in ``[0, 1]`` (a 0/1 indicator, or a
        partial-volume fraction for a sub-voxel-accurate source).
    radial_dir : (Nx,Ny,Nz,3) array
        Local radial (membrane-normal / lipid-director) unit vectors.  Only used
        where ``myelin_mask > 0``; need not be normalised (it is re-normalised).
    chi_iso, chi_aniso : float
        Isotropic and anisotropic susceptibility (SI, dimensionless).

    Returns
    -------
    chi6 : (Nx,Ny,Nz,6) float64
        The 6 independent components (xx,yy,zz,xy,xz,yz) of ``chi(r)``.
    """
    m = np.asarray(myelin_mask, float)
    n = _unit(np.asarray(radial_dir, float), axis=-1)
    chi6 = np.zeros(m.shape + (6,), float)
    for c, (i, j) in enumerate(_SYM6):
        term = 0.0
        if chi_aniso != 0.0:
            term = term + chi_aniso * (n[..., i] * n[..., j] - (1.0 / 3.0 if i == j else 0.0))
        if chi_iso != 0.0 and i == j:
            term = term + chi_iso
        chi6[..., c] = m * term
    return chi6


def dipole_field(chi6, voxel_size, b0_dir, B0=1.0):
    """Off-resonance field ``dB(r)`` (Tesla) of a susceptibility-tensor grid.

    Direct single-configuration solver (one forward + one inverse FFT).  Implements
    the along-field component

        dB(k)/B0 = (1/3) H . chi(k) . H  -  (H . khat)(khat . chi(k) . H) ,

    with the k=0 term zeroed (Lorentz-sphere / zero-mean reference).  ``H`` is the
    unit main-field direction ``b0_dir``.

    Parameters
    ----------
    chi6 : (Nx,Ny,Nz,6) array
        Symmetric susceptibility tensor per voxel (see
        :func:`myelin_susceptibility_tensor`), SI dimensionless.
    voxel_size : float or (3,) array
        Voxel edge length(s) in metres.
    b0_dir : (3,) array
        Main-field direction in the grid frame (need not be unit).
    B0 : float
        Field strength in Tesla.

    Returns
    -------
    dB : (Nx,Ny,Nz) float64
        Field offset along B0, in Tesla.
    """
    chi6 = np.asarray(chi6, float)
    shape = chi6.shape[:3]
    H = _unit(np.asarray(b0_dir, float).ravel())
    khat, dc = _khat(shape, voxel_size)

    # chi(k), 6 complex components
    chik = np.stack([np.fft.fftn(chi6[..., c]) for c in range(6)], axis=0)

    def full(t6):
        """Expand a 6-vector field into the symmetric 3x3 (indexed [i][j])."""
        M = [[None] * 3 for _ in range(3)]
        for c, (i, j) in enumerate(_SYM6):
            M[i][j] = t6[c]
            M[j][i] = t6[c]
        return M

    C = full(chik)
    # (chi . H)_i = sum_j chi_ij H_j ; H . chi . H = sum_i H_i (chi.H)_i
    chiH = [sum(C[i][j] * H[j] for j in range(3)) for i in range(3)]
    HchiH = sum(H[i] * chiH[i] for i in range(3))
    khat_chiH = sum(khat[i] * chiH[i] for i in range(3))          # khat . chi . H
    Hkhat = sum(H[i] * khat[i] for i in range(3))                 # H . khat
    kernel = (1.0 / 3.0) * HchiH - Hkhat * khat_chiH
    kernel[dc] = 0.0
    dB = np.real(np.fft.ifftn(kernel)) * float(B0)
    return dB


def _khat_axes(shape, voxel_size):
    """``(khat, dc)`` of :func:`_khat` with ``khat`` a list of three arrays: the unit wave-vector components, zero at
    k = 0, each computed by broadcasting the three one-dimensional wave-number axes (the same elementwise arithmetic
    as :func:`_khat`, so the same values to the bit, without its three full-size ``k`` grids)."""
    vs = _as_voxel_size(voxel_size, len(shape))
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(shape, vs)]
    nd = len(shape)
    kb = [k.reshape([-1 if a == i else 1 for a in range(nd)]) for i, k in enumerate(ks)]
    k2 = 0
    for k in kb:
        k2 = k2 + k ** 2
    k2 = np.broadcast_to(k2, shape)
    dc = k2 == 0.0
    inv = np.where(dc, 0.0, 1.0 / np.sqrt(np.where(dc, 1.0, k2)))
    return [k * inv for k in kb], dc


def _kspace_window(shape, kspace_lowpass):
    """The raised-cosine k-space window of :func:`field_basis` (1.0 when ``kspace_lowpass`` is ``None``)."""
    if kspace_lowpass is None:
        return 1.0
    nd = len(shape)
    axf = [(np.fft.fftfreq(n) * 2.0).reshape([-1 if a == i else 1 for a in range(nd)]) for i, n in enumerate(shape)]
    r2 = 0
    for g in axf:
        r2 = r2 + g ** 2
    r = np.sqrt(np.broadcast_to(r2, shape) / nd)
    W = np.ones_like(r); f = float(kspace_lowpass)
    ramp = (r > f) & (r < 1.0)
    W[ramp] = 0.5 * (1.0 + np.cos(np.pi * (r[ramp] - f) / (1.0 - f)))
    W[r >= 1.0] = 0.0
    return W


def field_basis_channels(myelin_mask, radial_dir, voxel_size, include_aniso=True, kspace_lowpass=None):
    """The channels of :func:`field_basis`, one at a time: a generator of ``(group, index, array)`` --
    ``("iso_local", None, a)``, then ``("iso_P", c, a)`` for ``c`` in 0..5 and, with ``include_aniso``,
    ``("aniso_G", c, a)`` for ``c`` in 0..5 -- each ``a`` a float64 array of the mask's shape.

    The k-space work of a channel (its forward and inverse FFT and the complex temporaries between) is released
    before the next is made, so the peak above the finished channels is one channel's: the mask, the three wave-vector
    components, the window, and ``mk`` (the mask's transform; released after the isotropic channels) or the six
    transforms of the anisotropic tensor (released with the last channel). Every channel is the value :func:`field_basis`
    has always returned, to the bit."""
    m = np.asarray(myelin_mask, float)
    shape = m.shape
    khat, dc = _khat_axes(shape, voxel_size)

    # Optional k-space low-pass (raised-cosine roll-off from `kspace_lowpass`·Nyquist to Nyquist).
    # The dipole kernel k̂ᵢk̂ⱼ does not decay with |k|, so grid-scale source roughness (a hard/thin
    # mask, a gradient-derived director) rings at the grid Nyquist — content the grid cannot
    # faithfully represent anyway. Damping it removes that ringing (standard in QSM forward models).
    W = _kspace_window(shape, kspace_lowpass)

    # isotropic local term (1/3) m(r), zero-meaned (DC removed for consistency).
    yield "iso_local", None, (m - m.mean()) / 3.0

    # iso_P_ij = IFFT[ mk * khat_i khat_j ]  (khat=0 at DC -> already zero-mean).
    mk = np.fft.fftn(m)
    for c, (i, j) in enumerate(_SYM6):
        yield "iso_P", c, np.real(np.fft.ifftn(mk * khat[i] * khat[j] * W))
    del mk

    if include_aniso:
        n = _unit(np.asarray(radial_dir, float), axis=-1)
        # anisotropic tensor field T_cd(r) = m (n_c n_d - delta_cd/3), FFT -> Tk.
        Tk = {}
        for (i, j) in _SYM6:
            Td = m * (n[..., i] * n[..., j] - (1.0 / 3.0 if i == j else 0.0))
            Tk[(i, j)] = np.fft.fftn(Td)
            Tk[(j, i)] = Tk[(i, j)]
        del n, Td
        # K_ij(k) = (1/3) Tk_ij - sum_c khat_i khat_c Tk_cj ; symmetrise in (i,j).
        for c, (i, j) in enumerate(_SYM6):
            Kij = (1.0 / 3.0) * Tk[(i, j)]
            Kji = (1.0 / 3.0) * Tk[(j, i)]
            sec_ij = sum(khat[i] * khat[cc] * Tk[(cc, j)] for cc in range(3))
            sec_ji = sum(khat[j] * khat[cc] * Tk[(cc, i)] for cc in range(3))
            Ksym = 0.5 * ((Kij - sec_ij) + (Kji - sec_ji))
            Ksym[dc] = 0.0
            yield "aniso_G", c, np.real(np.fft.ifftn(Ksym * W))


def field_basis(myelin_mask, radial_dir, voxel_size, include_aniso=True, kspace_lowpass=None):
    """Geometry-only field-basis grids for cheap ``(H, B0, chi_I, chi_A)`` replay.

    The along-field response factorises as

        dB(r; H, B0, chi_I, chi_A)/B0
            = chi_I * [ iso_local(r) - Q(H) . iso_P(r) ]
            + chi_A *   Q(H) . aniso_G(r)

    where ``Q(H) = (Hx^2, Hy^2, Hz^2, 2 HxHy, 2 HxHz, 2 HyHz)`` contracts the 6
    symmetric components (see :func:`assemble_field`).  Every returned grid depends
    only on the geometry (mask + director), so a single set serves any main-field
    direction, strength and susceptibility — the 3-D analogue of the analytic
    ``Phi_C/Phi_S/Phi_0`` phasor maps.  All grids are zero-mean (Lorentz reference).
    The channels are made one at a time (:func:`field_basis_channels`).

    Parameters
    ----------
    myelin_mask : (Nx,Ny,Nz) array
        Myelin fraction per voxel in ``[0, 1]``.
    radial_dir : (Nx,Ny,Nz,3) array
        Local radial unit vectors (used only where ``include_aniso``).
    voxel_size : float or (3,) array
        Voxel edge length(s), metres.
    include_aniso : bool
        Compute the anisotropic basis (6 extra IFFTs).  Set False for a purely
        isotropic source to save work.

    Returns
    -------
    basis : dict
        ``iso_local`` (Nx,Ny,Nz), ``iso_P`` (6,Nx,Ny,Nz),
        ``aniso_G`` (6,Nx,Ny,Nz) or None, plus ``shape`` and ``voxel_size``.
    """
    shape = np.shape(myelin_mask)
    iso_P = np.empty((6,) + shape, float)
    aniso_G = np.empty((6,) + shape, float) if include_aniso else None
    iso_local = None
    for group, c, a in field_basis_channels(myelin_mask, radial_dir, voxel_size, include_aniso=include_aniso,
                                            kspace_lowpass=kspace_lowpass):
        if group == "iso_local":
            iso_local = a
        elif group == "iso_P":
            iso_P[c] = a
        else:
            aniso_G[c] = a
    return {
        "iso_local": iso_local,
        "iso_P": iso_P,
        "aniso_G": aniso_G,
        "shape": shape,
        "voxel_size": _as_voxel_size(voxel_size, 3),
    }


def assemble_field(basis, b0_dir, B0=1.0, chi_iso=0.0, chi_aniso=0.0):
    """Assemble ``dB(r)`` (Tesla) from a :func:`field_basis` for one configuration.

    Cheap real contraction — no FFT.  Sweep ``b0_dir``/``B0``/``chi_*`` for free.
    """
    from dmipy_replay.replay.field import q_of_H
    q = q_of_H(b0_dir)
    dB = np.zeros(basis["shape"], float)
    if chi_iso != 0.0:
        contr = np.tensordot(q, basis["iso_P"], axes=(0, 0))     # Q . iso_P
        dB = dB + chi_iso * (basis["iso_local"] - contr)
    if chi_aniso != 0.0:
        if basis["aniso_G"] is None:
            raise ValueError("basis was built with include_aniso=False; cannot apply chi_aniso.")
        dB = dB + chi_aniso * np.tensordot(q, basis["aniso_G"], axes=(0, 0))
    return dB * float(B0)


def sample_grid(grid, positions, origin, voxel_size, periodic=False, order=1):
    """Sample a scalar grid at continuous positions (trilinear by default).

    Parameters
    ----------
    grid : (Nx,Ny,Nz) array
        Field grid (e.g. ``dB`` from :func:`assemble_field`).
    positions : (..., 3) array
        Query positions in metres (any leading shape, e.g. ``(n_walkers, n_t, 3)``).
    origin : (3,) array
        World coordinate of the voxel **corner** of index ``(0,0,0)`` (metres); the
        voxel spans ``[origin, origin+voxel_size)`` with its centre at
        ``origin + 0.5*voxel_size`` — the convention returned by
        :func:`dmipy_sim.geometry.mesh_shapes.grid_axes`.
    voxel_size : float or (3,) array
        Voxel edge length(s), metres.
    periodic : bool or (bool,bool,bool)
        Wrap sampling per axis (use for a periodic cell); else clamp at the edge.
    order : int
        Interpolation order for :func:`scipy.ndimage.map_coordinates` (1 = linear).

    Returns
    -------
    values : (...) array
        Interpolated values, same leading shape as ``positions``.
    """
    from scipy.ndimage import map_coordinates

    pos = np.asarray(positions, float)
    lead = pos.shape[:-1]
    vs = _as_voxel_size(voxel_size, 3)
    org = np.asarray(origin, float).ravel()
    # index of a point at voxel-centre i is i (map_coordinates samples at integer
    # = voxel centre); centre i sits at org + (i+0.5)*vs, hence the -0.5 shift.
    idx = (pos.reshape(-1, 3) - org) / vs - 0.5       # fractional voxel coordinates
    if isinstance(periodic, bool):
        periodic = (periodic, periodic, periodic)
    coords = np.empty((3, idx.shape[0]), float)
    for a in range(3):
        coords[a] = np.mod(idx[:, a], grid.shape[a]) if periodic[a] else idx[:, a]
    mode = "grid-wrap" if any(periodic) else "nearest"
    vals = map_coordinates(grid, coords, order=order, mode=mode)
    return vals.reshape(lead)


def radial_from_sdf(sdf, voxel_size):
    """Local radial (membrane-normal) unit field from a signed distance grid.

    ``n = grad(sdf) / |grad(sdf)|`` — robust for undulating / asymmetric sheaths
    where an analytic cylinder normal is unavailable.  ``sdf`` should increase
    outward across the membrane (sign convention only affects ``n``'s sign, which
    is immaterial to ``n n^T``).
    """
    vs = _as_voxel_size(np.asarray(voxel_size, float), 3)
    g = np.gradient(np.asarray(sdf, float), *vs, edge_order=2)
    return _unit(np.stack(g, axis=-1), axis=-1)


# ---------------------------------------------------------------------------
# Mesh -> field-grid: voxelise a myelin sheath (inner/outer surface meshes) and
# build its geometry-only susceptibility field basis. This is the producer side of
# the static field-grid replay channel (dmipy_sim.replay.bank): store the basis once, then
# replay any (B0, direction, chi) by contracting + sampling along the walk.
# ---------------------------------------------------------------------------
def mesh_inside(V, F, pts, *, clip_axis=None, chunk=2_000_000):
    """Boolean "inside this surface" for arbitrary points, by **global nearest-triangle sidedness**.

    ``sign((p - centroid_nearest) . outward_normal_nearest) < 0`` — the same sidedness test the walk's
    reflection uses, but with an EXACT global nearest neighbour (scipy KD-tree) instead of a local
    cell-neighbourhood gather. That matters for a thin structure inside a large box: a gather-based
    test has no triangle in its neighbourhood for most of the box and silently returns an arbitrary
    side (which would seed walkers in free space and corrupt a voxel mask).

    ``clip_axis`` (e.g. 2) additionally requires the point to lie within the mesh's extent along that
    axis — needed for a surface with OPEN ends (a cropped axon tube), where points beyond the rim have
    no meaningful inside/outside. Degenerate (zero-area) faces are dropped.

    Accurate to about one triangle size at the boundary (nearest-*centroid* rather than nearest-point);
    for the axon meshes here the triangles (~0.15 um) are far finer than the field grid, and the
    resulting volumes match the closed-surface mesh volumes to within a few percent.

    NEAR-FIELD ONLY, because the triangle is selected by nearest CENTROID. Sidedness against the TRUE
    nearest triangle is correct at any distance, but the nearest centroid need not belong to the nearest
    triangle, and the further away the point is the more triangles sit at comparable centroid distance,
    so the wrong one is picked and its normal reports the wrong side. Measured on the Winther axon06
    inner surface over the full padded box: 13.6% of accepted points are false, at a median 10.2 um from
    a wall bounding a lumen of radius <= 1.24 um. Substituting a proper closest-point query (trimesh's
    BVH) resolves all 16 disputed points in a 6k sample, confirming the cause is triangle SELECTION and
    not the sidedness principle; taking more k-nearest centroids does not fix it (k=8 changed nothing).

    Close to the surface the selection is reliable and so is the result (99.8% agreement with ray casting,
    no false-outside), which is why this remains the right per-sample test for classifying field-grid
    voxels — they all sit on or near the sheath. It is NOT a containment test over a large box: use
    :func:`mesh_contains`, which calls this as a candidate filter and then verifies each candidate.
    """
    from scipy.spatial import cKDTree
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0                                            # drop degenerate faces
    tri = tri[good]; n = n[good] / ln[good][:, None]
    cent = tri.mean(1)
    tree = cKDTree(cent)
    pts = np.asarray(pts, float)
    out = np.zeros(len(pts), bool)
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        _, idx = tree.query(p, workers=-1)
        out[i:i + chunk] = np.einsum("ij,ij->i", p - cent[idx], n[idx]) < 0
    if clip_axis is not None:
        a = int(clip_axis)
        out &= (pts[:, a] > V[:, a].min()) & (pts[:, a] < V[:, a].max())
    return out


# Separating "a defect" from "an open surface" has to scale with the mesh, because defects accumulate with
# component count while openness does not. Measured: a 366-strand axon bundle carries 22 boundary edges of
# 2.08M (1.1e-05) -- cleaner per strand than a single Winther axon that is accepted -- while half a tube,
# genuinely open, carries 64 of 192 (3.3e-01). An absolute cap cannot separate those as meshes grow.
#
# The threshold is set by what it must REJECT, with margin: an open surface exposes a rim, which is a
# fraction of order 1e-1 of its edges, so 1e-2 rejects that by ~30x while accepting defect rates several
# orders of magnitude below it. The absolute floor keeps small meshes (a one-triangle hole in a 144-edge
# cylinder) on the defect side, where a fraction alone would be too strict.
_MAX_BOUNDARY_EDGE_FRACTION = 1e-2
_MIN_REPAIRABLE_BOUNDARY_EDGES = 16


def _warn_if_bruteforce_ray_engine(mesh):
    """trimesh chooses its ray backend at import and says nothing either way.

    With the native ``embreex`` it uses a BVH; without it, a pure-NumPy engine that tests every
    ray against every triangle. The two differ by ~3 orders of magnitude, and the fallback is
    silent -- which is how a 650 ms/point containment test hid inside a walk for a day. It is not
    always fixable by installing the extension: ``embreex`` publishes no aarch64 wheels, so on ARM
    the slow engine is permanent. Say so, once, rather than let the caller discover it as an OOM.
    """
    if type(mesh.ray).__module__.endswith("ray_triangle"):
        warnings.warn(
            "trimesh has no native ray backend (embreex/pyembree), so `contains` is running its "
            "pure-NumPy engine: every ray is tested against EVERY triangle, costing ~O(points x "
            "triangles) in both time and memory (measured on a 1.6M-triangle bundle: ~650 ms and "
            "~84 MB per point). Prefer `mesh_contains(..., method='grid')`, which is exact and "
            "bins triangles by their xy footprint instead.", RuntimeWarning, stacklevel=3)


def mesh_contains(V, F, pts, *, method="grid", prefilter=False, chunk=2_000_000):
    """Exact "inside this CLOSED surface" for arbitrary points, by ray-crossing parity.

    The containment test to use when the points can be anywhere in a large box — seeding a compartment
    and measuring its volume fraction — where :func:`mesh_inside` is unreliable in the far field (see its
    docstring). Parity is a global test and has no such failure mode, but it costs far more per point,
    so this runs in two stages: :func:`mesh_inside` proposes candidates, and only those are ray cast.
    The prefilter is OFF by default, because that soundness argument does not hold in general. It rests on
    ``mesh_inside`` producing no false-OUTSIDE -- measured on single axons (0 of 51-142 interior points missed,
    at 3k/4k/6k sample sizes) and FALSE on a dense multi-body bundle: on the 366-fibre CACTUS outer surface it
    misses 1.67% of genuine interior points (33 of 1977). Because the proposal gate sits UPSTREAM of the ray
    cast, every one of those is inherited -- the cascade returned exactly the same 33 false-OUTSIDEs as the
    fast test alone, so the "exact" stage never saw them. The consequence is not subtle: seeding a compartment
    on that answer put 3.8% of a nominally extra-axonal pool inside a fibre, which after the fact is
    indistinguishable from walkers leaking through an impermeable wall.

    ``prefilter=True`` restores the cascade for the case it was written for -- a thin structure in a big box,
    where it is a large speed-up and its assumption holds. Spot-check it against a rescaled-parity reference
    before trusting it on a new substrate.

    Requires a closed surface: with open ends a ray can exit through the rim and parity is meaningless.
    A mesh whose boundary edges are a negligible FRACTION of its edges (``_MAX_BOUNDARY_EDGE_FRACTION``,
    with a small absolute floor for tiny meshes) is treated as defective rather than open: ``trimesh.repair.fill_holes`` is tried, and if the defect is not a fillable hole (one axon of
    the 29-axon Winther set has a two-edge slit) it proceeds with a ``RuntimeWarning``, since an opening
    that small perturbs parity only for rays threading it. More than that raises: a genuinely open surface
    has no inside, and silently returning the parity of a leaky mesh would be worse than failing. For a
    deliberately open surface use ``mesh_inside(..., clip_axis=...)`` and accept its near-field contract.

    Parity comes from :mod:`trimesh` (imported lazily, as elsewhere in the package) rather than being
    reimplemented here.
    """
    if method not in ("grid", "trimesh"):
        raise ValueError(f"method must be 'grid' or 'trimesh', got {method!r}")
    pts = np.asarray(pts, float)
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    import trimesh
    # Rescale so triangle edges are O(1) before handing the mesh to trimesh. Its geometry predicates
    # unitize against an absolute tolerance, so at SI scale (edges ~1e-7 m) face normals collapse to
    # zero and `contains` reports almost everything outside -- silently, and only in the direction that
    # looks like a clean, restrictive answer. mesh_inside is scale-free (it normalises its own normals),
    # so only this stage needs it. The factor cancels: points are scaled identically.
    s = 1.0 / max(float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))), 1e-300)
    m = trimesh.Trimesh(vertices=V * s, faces=F, process=False)
    n_open = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
    if n_open:
        # A handful of boundary edges is a defect (a missing triangle), not an open surface: repair and
        # continue. The cap is what separates the two -- a torn or genuinely open surface must still fail,
        # because fill_holes would happily span a wide rim and hand back confident nonsense.
        allowed = max(_MIN_REPAIRABLE_BOUNDARY_EDGES,
                      int(_MAX_BOUNDARY_EDGE_FRACTION * len(m.edges_sorted)))
        if n_open <= allowed:
            rep = m.copy()
            trimesh.repair.fill_holes(rep)
            if not len(trimesh.grouping.group_rows(rep.edges_sorted, require_count=1)):
                m = rep
                n_open = 0
            else:
                # Not every tiny defect is a fillable hole: one axon of the 29-axon Winther set has two
                # boundary edges meeting at a single vertex, which is a slit rather than a hole, and no
                # repair closes it without making that edge non-manifold. An opening this small is
                # negligible for parity -- only a ray threading the slit itself is affected -- so proceed,
                # but say so, and check the result against the mesh volume if the fraction matters.
                import warnings as _w
                _w.warn(f"mesh_contains: {n_open} boundary edges remain after trimesh.repair.fill_holes "
                        f"(a slit, not a hole). Proceeding: an opening this small changes parity only for "
                        f"rays passing through it. Cross-check the accepted fraction against the mesh "
                        f"volume if it matters.", RuntimeWarning, stacklevel=2)
                n_open = 0
        if n_open:
            raise ValueError(
                f"mesh_contains requires a closed surface; this one has {n_open} boundary edges "
                f"({n_open/max(len(m.edges_sorted),1):.2e} of all edges), too many to be a defect. Ray parity is undefined when a ray can leave through an open rim. Use "
                f"mesh_inside(..., clip_axis=...) for a deliberately open surface, noting it is a "
                f"near-field test only.")
    if method == "grid":
        # `m` is the surface the gate accepted (holes filled), and scaled by `s`; the grid path is
        # scale-free so either is fine, but it must see the REPAIRED faces, not the caller's.
        return mesh_contains_fast(np.asarray(m.vertices, float) / s,
                                  np.asarray(m.faces, np.int64), pts)

    _warn_if_bruteforce_ray_engine(m)
    out = np.zeros(len(pts), bool)
    cand = mesh_inside(V, F, pts) if prefilter else np.ones(len(pts), bool)
    if cand.any():
        sub = pts[cand]
        got = np.zeros(len(sub), bool)
        try:
            for i in range(0, len(sub), chunk):
                got[i:i + chunk] = m.contains(sub[i:i + chunk] * s)
        except ModuleNotFoundError as exc:      # trimesh's ray engine is an EXTRA, not a hard dependency
            raise ModuleNotFoundError(
                f"mesh_contains needs trimesh's ray-casting extras ({exc.name} is missing). A bare "
                f"'pip install trimesh' imports fine and only fails here, inside the containment call. "
                f"Install pip install 'dmipy-sim[mesh]' (or 'trimesh[easy]')."
            ) from exc
        out[cand] = got
    return out


_WORKER = {}                                     #: the state of a spawned worker (set by its initializer)


class _SharedArrays:
    """Arrays in shared memory, for the spawned workers of a pool: ``specs`` (``name -> (segment, shape, dtype)``) is
    what a worker attaches by (:func:`_attach`); :meth:`close` releases the segments."""

    def __init__(self, arrays):
        from multiprocessing import shared_memory
        self._shms, self.specs = [], {}
        try:
            for k, a in arrays.items():
                a = np.ascontiguousarray(a)
                shm = shared_memory.SharedMemory(create=True, size=max(int(a.nbytes), 1))
                self._shms.append(shm)
                np.ndarray(a.shape, a.dtype, buffer=shm.buf)[...] = a
                self.specs[k] = (shm.name, a.shape, a.dtype.str)
        except BaseException:
            self.close()
            raise

    def close(self):
        for shm in self._shms:
            try:
                shm.close(); shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = []


def _attach(spec):
    """The array of a :class:`_SharedArrays` spec, in a worker (its segment kept alive in ``_WORKER``)."""
    from multiprocessing import shared_memory
    name, shape, dtype = spec
    shm = shared_memory.SharedMemory(name=name)          # the creating process owns the segment (and its tracker entry)
    _WORKER.setdefault("segments", []).append(shm)
    return np.ndarray(shape, np.dtype(dtype), buffer=shm.buf)


def _spawn_pool(workers, initializer, initargs):
    """A pool of ``workers`` processes started by ``spawn``: fresh interpreters, so a process that has initialised JAX (a
    multithreaded runtime) is never forked."""
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(int(workers), initializer=initializer, initargs=initargs)


def _body_contains(V, F, lo, hi, pts):
    """The indices of ``pts`` inside the closed surface ``(V, F)`` whose bounding box is ``lo .. hi``: those in the box,
    by ray parity across the surface's smallest extent, the bins of the parity index sized to its faces (256 bins
    over 7 000 faces spent 90 % of the call in empty bins)."""
    sel = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
    if sel.size == 0:
        return sel
    a = int(np.argmin(hi - lo))                                  # the ray axis: across a tube, never along it
    perm = [j for j in range(3) if j != a] + [a]
    n_bins = int(np.clip(np.sqrt(len(F) / 2.0), 8, 256))
    return sel[mesh_contains_fast(V[:, perm], F, pts[sel][:, perm], n_bins=n_bins)]


def _contains_init(bodies, bounds, pts_spec):
    _WORKER.update(bodies=bodies, bounds=bounds, pts=_attach(pts_spec))


def _contains_task(i):
    (V, F), (lo, hi) = _WORKER["bodies"][i], _WORKER["bounds"][i]
    return _body_contains(V, F, lo, hi, _WORKER["pts"])


def _closest_on(tri, nrm, unit, tree, k, pts):
    """``(d, n)`` per point of ``pts``: the distance to the closest point on the ``k`` faces nearest by centroid (in
    ``tree``) and the outward normal of the face that carries it; ``tri`` and ``pts`` are in units of ``unit``."""
    from trimesh.triangles import closest_point
    k = min(k, len(tri))
    p = pts / unit
    _, idx = tree.query(p, k=k, workers=1)
    idx = np.asarray(idx).reshape(len(p), k)
    c = closest_point(tri[idx.ravel()], np.repeat(p, k, axis=0)).reshape(len(p), k, 3)
    d = np.linalg.norm(c - p[:, None, :], axis=2)
    j = d.argmin(1); rows = np.arange(len(p))
    return d[rows, j] * unit, nrm[idx[rows, j]]


def _distance_init(specs, unit, k):
    from scipy.spatial import cKDTree
    arrays = {name: _attach(spec) for name, spec in specs.items()}
    _WORKER.update(tri=arrays["tri"], nrm=arrays["nrm"], unit=unit, k=k, tree=cKDTree(arrays["cent"]))


def _distance_task(pts):
    return _closest_on(_WORKER["tri"], _WORKER["nrm"], _WORKER["unit"], _WORKER["tree"], _WORKER["k"], pts)


def _close_distance_pools(pools):
    for pool, shared in pools.values():
        pool.terminate(); pool.join(); shared.close()
    pools.clear()


class MeshBodies:
    """Closed triangle surfaces ``[(V, F), ...]`` (metres) as ONE boundary: the membership, the signed distance and
    the radial director a field raster and a walk's seeding read.

    Membership is ray parity per BODY (:func:`mesh_contains_fast`), a point inside the boundary when it is inside
    any body. Two things make that the test rather than the union's parity or a nearest-face sidedness: bodies
    overlap (the sheath tubes of a packed bundle), and inside an overlap the union's parity is even; and a
    nearest-face test reads a neighbour's wall from inside a tube -- on the CACTUS bundle it misses 0.9 % of the
    cell, up to a micron inside a strand, spread over most strands. Each body's rays run along its axis of
    smallest extent, so a tube open at its ends is cast across, never along, and a closed body pays the least.
    ``signed_distance`` is the distance to the closest point on the boundary, exact on the ``k`` faces nearest by
    centroid, positive inside, with the outward normal of that face: what a node's partial volume is taken from.
    A centroid is a proxy for a face only when the face is small against the distance asked, so the faces are
    first split until no edge exceeds ``resolve_m`` (:func:`trimesh.remesh.subdivide_to_size`; a 30 um sliver of
    a coarse tube mesh puts its centroid 15 um from the wall it bounds). ``directors`` is :func:`mesh_directors`
    on the union. The union's ``V``, ``F`` are exposed for the walk.
    """

    def __init__(self, bodies, *, k=16, workers=None):
        self.bodies = [(np.asarray(V, float), np.asarray(F, np.int64)) for V, F in bodies]
        if not self.bodies:
            raise ValueError("MeshBodies needs at least one (V, F) surface")
        offs = np.cumsum([0] + [len(V) for V, _ in self.bodies])
        self.V = np.concatenate([V for V, _ in self.bodies])
        self.F = np.concatenate([F + o for (_, F), o in zip(self.bodies, offs[:-1])])
        self.bounds = [(V.min(0), V.max(0)) for V, _ in self.bodies]
        self.k = int(k)
        # the processes a large query is spread over (spawned, the arrays in shared memory): a share of the host, never all of it
        self.workers = int(workers) if workers is not None else max(1, min(16, (os.cpu_count() or 4) // 4))
        self._near = {}
        self._dpools = {}                                            # resolve_m -> (spawned pool, its shared arrays)
        import weakref
        weakref.finalize(self, _close_distance_pools, self._dpools)

    def _contains_body(self, i, pts):
        """The indices of ``pts`` inside body ``i`` (:func:`_body_contains`)."""
        (V, F), (lo, hi) = self.bodies[i], self.bounds[i]
        return _body_contains(V, F, lo, hi, pts)

    @classmethod
    def of(cls, surfaces):
        """``surfaces`` as a :class:`MeshBodies`: one already, one ``(V, F)`` pair, or a list of pairs."""
        if isinstance(surfaces, cls):
            return surfaces
        if len(surfaces) == 2 and not isinstance(surfaces[0], (tuple, list)) and np.asarray(surfaces[0]).ndim == 2:
            return cls([surfaces])                                   # one (V, F) pair
        return cls(list(surfaces))

    def contains(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        out = np.zeros(len(pts), bool)
        if self.workers > 1 and len(self.bodies) > 1 and len(pts) >= 1_000_000:
            shared = _SharedArrays({"pts": pts})
            try:
                with _spawn_pool(min(self.workers, len(self.bodies)), _contains_init,
                                 (self.bodies, self.bounds, shared.specs["pts"])) as pool:
                    parts = pool.map(_contains_task, range(len(self.bodies)), chunksize=1)
            finally:
                shared.close()
            for idx in parts:
                out[idx] = True
        else:
            for i in range(len(self.bodies)):
                out[self._contains_body(i, pts)] = True
        return out

    def _nearest(self, resolve_m):
        key = None if resolve_m is None else float(resolve_m)
        if key not in self._near:
            from scipy.spatial import cKDTree
            V, F = self.V, self.F
            if key is not None:
                from trimesh.remesh import subdivide_to_size
                unit = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
                V, F = subdivide_to_size(V / unit, F, max_edge=key / unit, max_iter=12)
                V = np.asarray(V, float) * unit; F = np.asarray(F, np.int64)
            tri = V[F]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(n, axis=1); good = ln > 0
            tri = tri[good]; n = n[good] / ln[good][:, None]
            unit = float(np.median(np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)))    # trimesh's triangle geometry
            cent = tri.mean(1) / unit
            self._near[key] = (tri / unit, n, unit, cKDTree(cent), cent)              # carries absolute tolerances
        return self._near[key]

    def _distance_chunk(self, pts, resolve_m):
        tri, nrm, unit, tree, _ = self._nearest(resolve_m)
        return _closest_on(tri, nrm, unit, tree, self.k, pts)

    def _distance_pool(self, resolve_m):
        """The spawned pool that answers :meth:`signed_distance` for ``resolve_m``: its workers hold the faces and the
        k-d tree (built once in each), kept until :meth:`close`."""
        key = None if resolve_m is None else float(resolve_m)
        if key not in self._dpools:
            tri, nrm, unit, _, cent = self._nearest(resolve_m)
            shared = _SharedArrays({"tri": tri, "nrm": nrm, "cent": cent})
            try:
                pool = _spawn_pool(self.workers, _distance_init, (shared.specs, unit, self.k))
            except BaseException:
                shared.close()
                raise
            self._dpools[key] = (pool, shared)
        return self._dpools[key][0]

    def close(self):
        """Release the worker processes and shared memory of :meth:`signed_distance` (they are made again on the next call)."""
        _close_distance_pools(self._dpools)

    def signed_distance(self, pts, *, resolve_m=None, inside=None, chunk=200_000):
        """``(s, n)`` at ``pts`` ``(m, 3)``: the distance to the closest point on the boundary, positive inside, and
        the outward unit normal of the face that carries it; ``resolve_m`` the edge length the faces are split to
        first (the scale of the distances asked, a voxel), ``None`` for the faces as they are; ``inside`` the
        points' membership when the caller has it, else :meth:`contains` decides the sign."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        self._nearest(resolve_m)                                     # built once, before the workers are
        chunks = [pts[i:i + chunk] for i in range(0, len(pts), chunk)]
        if self.workers > 1 and len(chunks) > 1:
            parts = self._distance_pool(resolve_m).map(_distance_task, chunks, chunksize=1)
        else:
            parts = [self._distance_chunk(c, resolve_m) for c in chunks]
        dist = np.concatenate([d for d, _ in parts]) if parts else np.zeros(0)
        normal = np.concatenate([n for _, n in parts]) if parts else np.zeros((0, 3))
        if inside is None:
            inside = self.contains(pts)
        return np.where(np.asarray(inside, bool), dist, -dist), normal

    def directors(self, pts):
        return mesh_directors(self.V, self.F, pts)


def mesh_field_basis(inner, outer, box_min, box_max, *, res, include_aniso=True,
                     mask_supersample=FIELD_MASK_SUPERSAMPLE, kspace_lowpass=0.5, director=None):
    """Geometry-only myelin susceptibility field basis on a voxel grid, from inner (axonal) and outer (myelin)
    surface meshes (metres; each a :class:`MeshBodies`, one ``(V, F)`` pair or a list of pairs):
    :func:`predicate_field_basis` with the bodies' membership as the two tests, their signed distances for the
    partial volume at the boundary, and, by default, the nearest face of the axon surface as the radial director
    (:func:`mesh_directors`). ``director=`` overrides it; ``director=False`` falls back to the gradient of the
    voxelised mask (dmipy-sim#213: ~28 % off on the anisotropic term across the axis)."""
    bi = MeshBodies.of(inner) if inner is not None else None
    bo = MeshBodies.of(outer)
    if director is False:
        director = None
    elif director is None:
        director = (bi if bi is not None else bo).directors
    try:
        return predicate_field_basis(bi.contains if bi is not None else None, bo.contains,
                                     box_min, box_max, res=res, include_aniso=include_aniso,
                                     mask_supersample=mask_supersample, kspace_lowpass=kspace_lowpass, director=director,
                                     signed_distance=(bi.signed_distance if bi is not None else None, bo.signed_distance))
    finally:
        for b in (bi, bo):
            if b is not None:
                b.close()                                          # the workers of the signed distances


def mesh_directors(V, F, pts, *, chunk=2_000_000):
    """``(n, 3)`` unit vectors: the radial (lipid) director of a sheath at each point, from the surface ``(V, F)``
    -- the normal of the nearest face (by centroid, a k-d tree). The director is a line field (the tensor is
    ``n n^T``), so its sign is immaterial and none is imposed. On a sheath mesh the nearest face's normal is the
    local surface normal to the faceting angle, which is the radial direction; the gradient of a voxelised mask was
    ~28 % off on the anisotropic term (dmipy-sim#213), and trimesh's closest-point query, exact, costs ~360 us and
    up to 280 kB per point (36 s and 28 GB per 100k points), which no field grid can afford. Degenerate faces are
    dropped."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0
    tri = tri[good]; n = n[good] / ln[good][:, None]
    cent = tri.mean(1)
    tree = cKDTree(cent)
    P = np.asarray(pts, float).reshape(-1, 3)
    out = np.zeros_like(P)
    for i in range(0, P.shape[0], chunk):
        q = P[i:i + chunk]
        _, idx = tree.query(q, workers=-1)
        out[i:i + chunk] = n[idx]
    return out


def predicate_field_basis(inside_inner, inside_outer, box_min, box_max, *, res, include_aniso=True,
                          mask_supersample=FIELD_MASK_SUPERSAMPLE, kspace_lowpass=0.5, director=None,
                          signed_distance=None):
    """Geometry-only myelin susceptibility field basis on a voxel grid from two membership tests on
    points (metres): ``inside_inner`` (the axon) and ``inside_outer`` (the sheath's outer surface); any
    surface family that can answer "is this point inside" (a mesh, a sphere union, a strand pack).

    Voxelises the shell (inside outer, outside inner). The per-voxel radial director comes from ``director``
    -- a callable ``(n, 3) points -> (n, 3)`` unit vectors, the geometry's own radial vector (a strand pack's
    nearest-segment normal, :meth:`PackedCurvedCylinders.radial_directors`) -- when given, else from the
    signed distance to the inner surface (or to the shell itself when ``inside_inner`` is ``None``: a
    solid source), then calls :func:`field_basis`. ``res`` is the node spacing (metres): a caller derives it from
    the source's thinnest shell with :func:`field_resolution`. ``mask_supersample`` (default
    :data:`FIELD_MASK_SUPERSAMPLE`) gives a partial-volume occupancy in ``[0,1]`` at the boundary — essential
    because the dipole kernel ``k̂ᵢk̂ⱼ`` does not decay with ``|k|`` so a hard binary edge rings into the interior.
    The sub-samples are put to the membership tests, or, with ``signed_distance=(inner, outer)`` (callables
    ``points -> (s, n)``, the signed distance positive inside and the outward normal at the closest point; the inner
    one ``None`` for a solid source), to the local plane of each surface: a sub-sample at offset ``o`` from the node
    is inside where ``s - o . n > 0``, which is exact for a surface flat across a voxel and costs arithmetic rather
    than a membership test per sub-sample. Returns ``(basis, origin, voxel_size)`` with ``origin`` = box corner
    (the :func:`sample_grid` convention); ``basis["shell_fraction"]`` is the mask's mean, the shell's share of the box."""
    from scipy import ndimage

    box_min = np.asarray(box_min, float); box_max = np.asarray(box_max, float)
    side = box_max - box_min
    N = field_raster_shape(box_min, box_max, res)
    vs = side / N
    axes = [box_min[a] + (np.arange(N[a]) + 0.5) * vs[a] for a in range(3)]
    XX, YY, ZZ = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    del XX, YY, ZZ

    in_in = (inside_inner(pts) if inside_inner is not None else np.zeros(len(pts), bool)).reshape(N)   # SDF/director
    in_out = np.asarray(inside_outer(pts), bool).reshape(N)
    binary = (in_out & ~in_in)
    SS = max(1, int(mask_supersample))
    if SS == 1:
        myelin_mask = binary.astype(float)
    else:
        # Partial-volume (anti-aliased) occupancy. Essential: with a hard binary source the
        # non-decaying dipole kernel rings into the lumen (measured against the analytic hollow
        # cylinder: 2.6% of chi*B0 where the exact answer is 0; PV-4 reduces that to ~0.1% while
        # keeping the annulus amplitude at 1.00x analytic). Only voxels ON the boundary can have
        # fractional occupancy, so supersample just those -- exact, and ~100x cheaper than the
        # whole grid.
        myelin_mask = binary.astype(float)
        edge = ndimage.binary_dilation(binary) ^ ndimage.binary_erosion(binary)
        ei = np.flatnonzero(edge.ravel())
        if ei.size:
            sub = (np.arange(SS) + 0.5) / SS - 0.5                  # sub-voxel offsets in [-0.5,0.5)
            ox, oy, oz = np.meshgrid(sub * vs[0], sub * vs[1], sub * vs[2], indexing="ij")
            off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)      # (SS^3, 3)
            flat = myelin_mask.ravel()
            chunk = max(1, _EDGE_CHUNK_POINTS // len(off))                    # boundary voxels per membership call
            for i in range(0, ei.size, chunk):
                base = pts[ei[i:i + chunk]]                                   # (n, 3) voxel centres
                if signed_distance is None:
                    q = (base[:, None, :] + off[None, :, :]).reshape(-1, 3)
                    occ = np.asarray(inside_outer(q), bool) & ~(inside_inner(q) if inside_inner is not None else np.zeros(len(q), bool))
                    occ = occ.reshape(len(base), -1)
                else:
                    sd_in, sd_out = signed_distance
                    sel = ei[i:i + chunk]
                    s_o, n_o = sd_out(base, resolve_m=float(vs.max()), inside=in_out.ravel()[sel])
                    occ = (s_o[:, None] - n_o @ off.T) > 0.0                  # (n, SS^3): inside the outer's local plane
                    if sd_in is not None:
                        s_i, n_i = sd_in(base, resolve_m=float(vs.max()), inside=in_in.ravel()[sel])
                        occ &= (s_i[:, None] - n_i @ off.T) <= 0.0             # and outside the inner's
                flat[ei[i:i + chunk]] = occ.mean(axis=1)
    if director is not None:                                       # the geometry's own radial vector at every voxel
        radial_dir = _unit(np.asarray(director(pts), float), axis=-1).reshape(tuple(N) + (3,))
    else:                                                          # the gradient of the voxelised mask's signed distance
        ref = in_in if inside_inner is not None else in_out        # (dmipy-sim#213: ~28 % off on the anisotropic term)
        sdf = (ndimage.distance_transform_edt(~ref, sampling=tuple(vs))
               - ndimage.distance_transform_edt(ref, sampling=tuple(vs)))
        radial_dir = radial_from_sdf(sdf, vs)
    del pts, binary, in_in, in_out
    basis = field_basis(myelin_mask, radial_dir, vs, include_aniso=include_aniso,
                        kspace_lowpass=kspace_lowpass)
    basis["shell_fraction"] = float(myelin_mask.mean())
    return basis, box_min, vs


# ─────────────────────────────────────────────────────────────────────────────
def _xy_bins(tri, n_bins):
    """CSR (offsets, tri_ids, lo, inv) binning triangles by their xy footprint.

    A vertical (+z) ray from ``p`` can only hit a triangle whose xy AABB contains ``p.xy``, so
    binning in xy alone is both complete and conservative -- no intersection can be missed, which
    is what makes the acceleration exact rather than heuristic.
    """
    lo = tri[:, :, :2].min(axis=(0, 1))
    hi = tri[:, :, :2].max(axis=(0, 1))
    span = np.maximum(hi - lo, 1e-30)
    inv = n_bins / (span * (1.0 + 1e-9))
    t_lo = np.floor((tri[:, :, :2].min(axis=1) - lo) * inv).astype(np.int64)
    t_hi = np.floor((tri[:, :, :2].max(axis=1) - lo) * inv).astype(np.int64)
    np.clip(t_lo, 0, n_bins - 1, out=t_lo)
    np.clip(t_hi, 0, n_bins - 1, out=t_hi)

    wi = t_hi[:, 0] - t_lo[:, 0] + 1
    wj = t_hi[:, 1] - t_lo[:, 1] + 1
    counts = wi * wj
    total = int(counts.sum())
    tri_ids = np.repeat(np.arange(len(tri), dtype=np.int64), counts)
    start = np.cumsum(counts) - counts
    off = np.arange(total, dtype=np.int64) - np.repeat(start, counts)
    wj_r = np.repeat(wj, counts)
    bi = np.repeat(t_lo[:, 0], counts) + off // wj_r
    bj = np.repeat(t_lo[:, 1], counts) + off % wj_r
    key = bi * n_bins + bj

    order = np.argsort(key, kind="stable")
    key = key[order]; tri_ids = tri_ids[order]
    offsets = np.searchsorted(key, np.arange(n_bins * n_bins + 1))
    return offsets, tri_ids, lo, inv


def _parity_vertical(tri, pts, offsets, tri_ids, lo, inv, n_bins, tol):
    """Crossing parity of a +z ray, testing only the triangles in each point's xy bin.

    Returns ``(inside, ambiguous)``. A point is ambiguous when the ray passes within ``tol`` of a
    triangle edge in projection: the shared edge of two coplanar-adjacent faces is then counted
    twice or not at all, and parity is unreliable. Those points are re-cast from a jittered origin
    rather than silently trusted -- the failure is a flipped bit, not a small error.
    """
    n = len(pts)
    inside = np.zeros(n, bool)
    ambig = np.zeros(n, bool)
    b = np.floor((pts[:, :2] - lo) * inv).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)
    key = b[:, 0] * n_bins + b[:, 1]

    order = np.argsort(key, kind="stable")
    ks = key[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    for e0, e1 in zip(edges, np.r_[edges[1:], len(ks)]):
        k = ks[e0]
        s, t = offsets[k], offsets[k + 1]
        if s == t:
            continue                                   # no triangle overhead: outside
        idx = order[e0:e1]
        T = tri[tri_ids[s:t]]                          # (m, 3, 3)
        P = pts[idx]                                   # (q, 3)
        ax, ay, az = T[:, 0, 0], T[:, 0, 1], T[:, 0, 2]
        bx, by, bz = T[:, 1, 0], T[:, 1, 1], T[:, 1, 2]
        cx, cy, cz = T[:, 2, 0], T[:, 2, 1], T[:, 2, 2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)      # 2 x signed xy area
        # relative to the typical face: an absolute floor cannot distinguish a vertical
        # face (never crossed by a vertical ray) from a merely small one.
        ok = np.abs(d) > 1e-12 * np.median(np.abs(d)) if len(d) else np.zeros(0, bool)
        dd = np.where(ok, d, 1.0)
        px = P[:, 0][:, None]; py = P[:, 1][:, None]; pz = P[:, 2][:, None]
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / dd
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / dd
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0) & ok[None, :]
        z = l1 * az + l2 * bz + l3 * cz
        cross = hit & (z > pz)
        inside[idx] = (cross.sum(axis=1) % 2) == 1
        near = (np.minimum(np.minimum(np.abs(l1), np.abs(l2)), np.abs(l3)) < tol) & ok[None, :]
        ambig[idx] = (near & (z > pz - tol)).any(axis=1)
    return inside, ambig


def mesh_contains_fast(V, F, pts, *, n_bins=256, tol=1e-9, max_retries=6, seed=0):
    """Exact "inside this CLOSED surface", accelerated by an xy bin index instead of a ray engine.

    Same ray-parity semantics as :func:`mesh_contains`, and the same answers -- what changes is
    only WHICH triangles are tested. trimesh picks its ray backend at import: the native
    ``embreex`` when present, else a pure-NumPy engine that tests every ray against every
    triangle, with no warning either way. On this 1.57M-triangle bundle that engine measured
    ~650 ms and ~84 MB PER POINT, so seeding 6,000 walkers would need ~4 h and ~2 TB. And it is
    not installable everywhere -- ``embreex`` publishes no aarch64 wheels -- so the cliff is
    silent, platform-dependent, and permanent on ARM.

    A +z ray from ``p`` can only meet a triangle whose xy footprint contains ``p.xy``, so binning
    triangles by that footprint is complete: no intersection is missed, and the acceleration is
    exact rather than approximate. Each point then tests the ~100 triangles over its own bin
    instead of all 1.57M.

    Points whose ray grazes an edge in projection are re-cast from a jittered origin. That case is
    not a rounding error -- a shared edge counted twice or zero times flips the parity bit outright
    -- so it is detected and retried rather than trusted.
    """
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return np.zeros(0, bool)
    # Work in units of the median edge, exactly as `mesh_contains` does before handing trimesh a
    # mesh. Substrate coordinates are in METRES, so a triangle's doubled xy area is ~1e-14 there;
    # the barycentric divide then amplifies rounding, and an absolute degeneracy threshold cannot
    # tell a vertical face from a small one. Rescaling makes both O(1) and the predicates honest.
    scale = 1.0 / max(float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))), 1e-300)
    tri = V[F] * scale
    pts = pts * scale
    offsets, tri_ids, lo, inv = _xy_bins(tri, n_bins)

    inside, ambig = _parity_vertical(tri, pts, offsets, tri_ids, lo, inv, n_bins, tol)
    if ambig.any():
        rng = np.random.default_rng(seed)
        jit = float(np.median(np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)))
        for _ in range(max_retries):
            j = np.flatnonzero(ambig)
            if j.size == 0:
                break
            q = pts[j].copy()
            q[:, :2] += rng.normal(0.0, 1e-3 * jit, (len(j), 2))
            ins_j, amb_j = _parity_vertical(tri, q, offsets, tri_ids, lo, inv, n_bins, tol)
            inside[j] = ins_j
            ambig[j] = amb_j
        if ambig.any():
            warnings.warn(
                f"{int(ambig.sum())} of {len(pts)} points still graze a triangle edge after "
                f"{max_retries} jittered re-casts; their inside/outside is unreliable. That "
                f"usually means degenerate or duplicated faces rather than bad luck.",
                RuntimeWarning, stacklevel=2)
    return inside

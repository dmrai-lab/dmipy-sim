"""Myelinated geometries: concentric intra / myelin / extra compartments.

These are NOT stepped by `reflect`: their compartment is carried state, so the engine
routes them to the fused kernels in `physics` (see each class's `reflect`, which raises).
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import keep_side_radial, specular, off_wall, step_off_wall
from .base import Geometry, _rotation_to_z


class MyelinatedCylinder(Geometry):
    """Three-compartment myelinated cylinder: intra-axonal, myelin sheath, extra-axonal.

    The geometry consists of two concentric cylinders (inner radius R_inner,
    outer radius R_outer) along a given orientation axis:
      - Compartment 0 (intra-axonal): r_xy < R_inner
      - Compartment 1 (myelin sheath): R_inner <= r_xy <= R_outer
      - Compartment 2 (extra-axonal):  r_xy > R_outer

    Each compartment has a single **isotropic** diffusivity. Myelin water is trapped
    between the lipid bilayers with a short T2 and barely moves on any realistic
    diffusion time, so ``D_myelin`` defaults to 0 (a stuck pool that only carries
    ``T2_myelin``; the analytical counterpart is a stationary ``S1Dot``). Set it to any
    value > 0 to let myelin water diffuse. Two boundaries have independent permeability.

    Parameters
    ----------
    inner_radius : float
        Inner cylinder radius in metres (axon radius).
    outer_radius : float
        Outer cylinder radius in metres (outer myelin boundary).
    orientation : array-like of shape (3,)
        Cylinder axis direction (normalised internally).
    D_intra : float
        Intra-axonal diffusivity in m^2/s (isotropic).
    D_extra : float
        Extra-axonal diffusivity in m^2/s (isotropic).
    D_myelin : float, optional
        Myelin-water diffusivity in m^2/s (isotropic). Default 0 (stuck pool).
    kappa_inner : float, optional
        Permeability at inner boundary (m/s). Default None (impermeable).
    kappa_outer : float, optional
        Permeability at outer boundary (m/s). Default None (impermeable).
    T2_intra : float, optional
        T2 relaxation time for intra-axonal compartment (s).
    T2_myelin : float, optional
        T2 relaxation time for myelin compartment (s).
    T2_extra : float, optional
        T2 relaxation time for extra-axonal compartment (s).
    water_fractions : tuple of 3 floats, optional
        Relative water content (proton density) per compartment (intra, myelin,
        extra). ``None`` (default) uses the biophysical table: myelin =
        ``myelin_water_proton_density`` (~0.40, the water per unit myelin VOLUME), intra
        and extra = 1.0. Do NOT pass the MWF signal value (~0.15) here -- that is a
        measured signal fraction, not a per-volume weight.
    """

    # Marker: this geometry provides its own step function (not make_step_fn)
    _is_myelinated = True

    def __init__(self, inner_radius, outer_radius, orientation,
                 D_intra, D_extra, D_myelin=0.0,
                 kappa_inner=None, kappa_outer=None,
                 T2_intra=None, T2_myelin=None, T2_extra=None,
                 water_fractions=None):
        if outer_radius <= inner_radius:
            raise ValueError("outer_radius must be > inner_radius")

        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.D_intra = float(D_intra)
        self.D_myelin = float(D_myelin)
        self.D_extra = float(D_extra)
        self.kappa_inner = float(kappa_inner) if kappa_inner is not None else None
        self.kappa_outer = float(kappa_outer) if kappa_outer is not None else None
        self.T2_intra = float(T2_intra) if T2_intra is not None else None
        self.T2_myelin = float(T2_myelin) if T2_myelin is not None else None
        self.T2_extra = float(T2_extra) if T2_extra is not None else None
        if water_fractions is None:
            from ..substrate.biophysical_constants import get_default_value
            water_fractions = (1.0, float(get_default_value('myelin_water_proton_density')), 1.0)
        self.water_fractions = tuple(float(w) for w in water_fractions)

        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        _R_np = _rotation_to_z(self.orientation)
        self._R = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))

    def init_positions(self, n_walkers, key):
        """Distribute walkers proportional to volume * water_fraction per compartment.

        Extra-axonal volume is set to the annulus between R_outer and 2*R_outer
        for allocation purposes (actual walkers are placed uniformly).
        """
        R_in = self.inner_radius
        R_out = self.outer_radius
        wf = list(self.water_fractions)

        # Volumes (per unit length)
        vol_intra = np.pi * R_in**2
        vol_myelin = np.pi * (R_out**2 - R_in**2)
        # For extra-axonal, use an annulus up to 2*R_out for allocation
        R_extra = 2.0 * R_out
        vol_extra = np.pi * (R_extra**2 - R_out**2)

        # Weighted volumes
        # Homogeneous placement (by volume). The per-compartment water proton density
        # (water_fractions) is applied as a SIGNAL weight in simulate(), not here.
        w_intra = vol_intra
        w_myelin = vol_myelin
        w_extra = vol_extra
        w_total = w_intra + w_myelin + w_extra

        n_intra = int(round(n_walkers * w_intra / w_total))
        n_myelin = int(round(n_walkers * w_myelin / w_total))
        n_extra = n_walkers - n_intra - n_myelin

        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))

        positions = np.zeros((n_walkers, 3), dtype=np.float32)
        compartments = np.zeros(n_walkers, dtype=np.int32)

        idx = 0

        # Intra-axonal: uniform in circle of radius R_in
        if n_intra > 0:
            pts = []
            while sum(len(p) for p in pts) < n_intra:
                xy = rng.uniform(-R_in, R_in, (n_intra * 3, 2))
                pts.append(xy[np.linalg.norm(xy, axis=1) < R_in])
            xy_intra = np.concatenate(pts, axis=0)[:n_intra].astype(np.float32)
            positions[idx:idx + n_intra, 0] = xy_intra[:, 0]
            positions[idx:idx + n_intra, 1] = xy_intra[:, 1]
            compartments[idx:idx + n_intra] = 0
            idx += n_intra

        # Myelin: uniform in annulus R_in <= r < R_out
        if n_myelin > 0:
            pts = []
            while sum(len(p) for p in pts) < n_myelin:
                xy = rng.uniform(-R_out, R_out, (n_myelin * 3, 2))
                r_xy = np.linalg.norm(xy, axis=1)
                pts.append(xy[(r_xy >= R_in) & (r_xy < R_out)])
            xy_myelin = np.concatenate(pts, axis=0)[:n_myelin].astype(np.float32)
            positions[idx:idx + n_myelin, 0] = xy_myelin[:, 0]
            positions[idx:idx + n_myelin, 1] = xy_myelin[:, 1]
            compartments[idx:idx + n_myelin] = 1
            idx += n_myelin

        # Extra-axonal: uniform in annulus R_out <= r < R_extra
        if n_extra > 0:
            pts = []
            while sum(len(p) for p in pts) < n_extra:
                xy = rng.uniform(-R_extra, R_extra, (n_extra * 3, 2))
                r_xy = np.linalg.norm(xy, axis=1)
                pts.append(xy[(r_xy >= R_out) & (r_xy < R_extra)])
            xy_extra = np.concatenate(pts, axis=0)[:n_extra].astype(np.float32)
            positions[idx:idx + n_extra, 0] = xy_extra[:, 0]
            positions[idx:idx + n_extra, 1] = xy_extra[:, 1]
            compartments[idx:idx + n_extra] = 2
            idx += n_extra

        # Positions are in cylinder frame (xy = cross-section, z = axis).
        # Rotate to lab frame.
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ positions.T).T

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Not a usable boundary rule for this geometry -- raises.

        A three-compartment geometry cannot be stepped by a single-surface reflect: the
        walker's compartment is STATE that changes only at a granted crossing, which is why
        the real kernel (`physics.make_myelin_step_fn`) carries `compartment_id` and clamps
        the position to match it.  This method used to clamp every walker to the inner
        cylinder instead, so a `simulate_trajectories` call on a MyelinatedCylinder returned
        a silently single-compartment walk -- myelin and extra-axonal water simply absent,
        with no error.  Failing loudly is the point: a boundary interaction that does not
        happen is exactly the class of defect that produced dmrai-lab/dmipy-sim#86.
        """
        raise NotImplementedError(
            "MyelinatedCylinder has no single-surface reflect: its three compartments are "
            "stepped by the fused kernel physics.make_myelin_step_fn, which carries the "
            "compartment id. Use simulate(...) (which dispatches to that kernel) rather "
            "than a generic reflect/trajectory walk.")
    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID from position: 0=intra, 1=myelin, 2=extra.

        Classification is based on the radial distance in the cylinder
        cross-section (r_xy):
          - |r_xy| < R_inner  → 0 (intra-axonal)
          - R_inner <= |r_xy| < R_outer → 1 (myelin)
          - |r_xy| >= R_outer → 2 (extra-axonal)
        """
        R_in  = jnp.float32(self.inner_radius)
        R_out = jnp.float32(self.outer_radius)
        r_c   = r if self._is_identity_rotation else self._R @ r
        r_xy_sq = jnp.dot(r_c[:2], r_c[:2])
        in_intra  = r_xy_sq < R_in  * R_in
        in_myelin = (r_xy_sq >= R_in * R_in) & (r_xy_sq < R_out * R_out)
        comp = jnp.where(in_intra, jnp.int32(0),
               jnp.where(in_myelin, jnp.int32(1), jnp.int32(2)))
        return comp

    def volume(self, compartment: str, L: float = 1.0) -> float:
        """Volume of a compartment per unit length L (m³).

        Compartment 'extra' uses the annulus between R_outer and 2·R_outer as
        its bounding region (matching the convention used in init_positions).

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        L = float(L)
        R_in  = self.inner_radius
        R_out = self.outer_radius
        if compartment == 'intra':
            return np.pi * R_in ** 2 * L
        elif compartment == 'myelin':
            return np.pi * (R_out ** 2 - R_in ** 2) * L
        elif compartment == 'extra':
            R_extra = 2.0 * R_out
            return np.pi * (R_extra ** 2 - R_out ** 2) * L
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def surface_area(self, compartment: str, L: float = 1.0) -> float:
        """Lateral surface area bounding a compartment per unit length L (m²).

        Returns the area of the cylindrical wall(s) that bound the compartment:
          - 'intra':  inner wall only, area = 2π·R_inner·L
          - 'myelin': both walls, area = 2π·(R_inner + R_outer)·L
          - 'extra':  outer wall only (inner boundary of extra-axonal space),
                      area = 2π·R_outer·L

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        L = float(L)
        R_in  = self.inner_radius
        R_out = self.outer_radius
        if compartment == 'intra':
            return 2.0 * np.pi * R_in * L
        elif compartment == 'myelin':
            return 2.0 * np.pi * (R_in + R_out) * L
        elif compartment == 'extra':
            return 2.0 * np.pi * R_out * L
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def volume_fraction(self, compartment: str) -> float:
        """Volume fraction of a compartment within the bounding cylinder.

        The bounding cylinder has radius 2·R_outer (matching init_positions).
        Volume fractions sum to 1 over {'intra', 'myelin', 'extra'}.

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        """
        R_in   = self.inner_radius
        R_out  = self.outer_radius
        R_extra = 2.0 * R_out
        total = np.pi * R_extra ** 2  # bounding cylinder cross-section area
        if compartment == 'intra':
            return np.pi * R_in ** 2 / total
        elif compartment == 'myelin':
            return np.pi * (R_out ** 2 - R_in ** 2) / total
        elif compartment == 'extra':
            return np.pi * (R_extra ** 2 - R_out ** 2) / total
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")


class PackedMyelinatedCylinders:
    """Periodic RVE with N_actual myelinated cylinders — three-compartment.

    Combines ``PackedCylinders`` (periodic, multi-cylinder substrate) with
    ``MyelinatedCylinder`` (myelin sheath, per-compartment D/T2/permeability)
    into a single JIT-stable geometry.

    Each axon k (k=0..N_actual-1) has:
      - Inner radius  R_inner_k  (axon boundary)
      - Outer radius  R_outer_k = R_inner_k / g_ratio_k  (myelin/extra boundary)
      - Center        (cx_k, cy_k) in the periodic cell [-L/2, L/2)

    Compartment numbering
    ~~~~~~~~~~~~~~~~~~~~~
    - 0              : extra-axonal
    - 1 .. N_max     : intra_k  (axon k, k+1-th slot)
    - N_max+1 .. 2*N_max : myelin_k  (myelin sheath of axon k, k+1-th slot)

    Zero-radius padding
    ~~~~~~~~~~~~~~~~~~~
    Arrays are padded to length ``N_max`` with zeros.  Dummy cylinders (r=0)
    receive zero walkers (area = 0) and have SDF = inf in the step function
    (they never win ``argmin``), so different N_actual with the same N_max
    compile to the same JAX program.

    Parameters
    ----------
    inner_radii : array-like, shape (N_actual,)
        Axon radii in metres.
    g_ratios : array-like, shape (N_actual,) or scalar
        g-ratio per cylinder.
    centers : np.ndarray, shape (N_actual, 2)
        Cylinder centre positions in metres, [-L/2, L/2) convention.
    cell_size : float
        Side-length of the periodic square cell in metres.
    N_max : int, optional
        Fixed JIT padding length (>= N_actual).  Default 128.
    orientation : array-like, shape (3,), optional
        Shared cylinder axis direction. Default [0, 0, 1].
    D_intra, D_myelin, D_extra : float or array-like (N_actual,)
        Diffusivities in m^2/s.  Scalar is broadcast to all cylinders.  ``D_myelin``
        defaults to 0 (stuck myelin water; set > 0 to let it diffuse).
    T2_intra, T2_myelin, T2_extra : float or array-like (N_actual,) or None
        T2 relaxation times in seconds.  None = no T2.
    kappa_inner, kappa_outer : float or array-like (N_actual,) or None
        Inner/outer wall permeabilities in m/s.  None / 0.0 = impermeable.
    rho_inner, rho_outer : float or array-like (N_actual,) or None
        Surface relaxivity at inner/outer myelin walls in m/s.  0.0 = no
        relaxivity.
    """

    _is_packed_myelinated = True

    def __init__(
        self,
        inner_radii,
        g_ratios,
        centers,
        cell_size,
        N_max=128,
        orientation=(0., 0., 1.),
        D_intra=2e-9,
        D_myelin=0.0,
        D_extra=2e-9,
        T2_intra=None,
        T2_myelin=None,
        T2_extra=None,
        kappa_inner=0.0,
        kappa_outer=0.0,
        rho_inner=0.0,
        rho_outer=0.0,
    ):
        inner_radii = np.asarray(inner_radii, dtype=np.float64).ravel()
        N_actual = len(inner_radii)
        g_ratios = np.broadcast_to(
            np.asarray(g_ratios, dtype=np.float64).ravel(), (N_actual,)).copy()
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape != (N_actual, 2):
            raise ValueError(
                f"centers shape {centers.shape} must be ({N_actual}, 2)")
        if N_max < N_actual:
            raise ValueError(f"N_max={N_max} < N_actual={N_actual}")

        outer_radii = inner_radii / g_ratios
        self.N_actual = N_actual
        self.N_max = N_max
        self._cell_size = float(cell_size)
        from ..substrate.biophysical_constants import get_default_value as _gdv
        self._myelin_proton_density = float(_gdv('myelin_water_proton_density'))

        # Broadcast per-cylinder physics parameters to (N_actual,)
        def _bcast(x, name):
            x = np.asarray(x, dtype=np.float64).ravel()
            if x.size == 1:
                return np.broadcast_to(x, (N_actual,)).copy()
            if x.size != N_actual:
                raise ValueError(
                    f"{name} must be scalar or length {N_actual}, got {x.size}")
            return x.copy()

        def _bcast_kappa(x, name):
            if x is None:
                x = 0.0
            return _bcast(x, name)

        def _bcast_opt(x, name):
            if x is None:
                return None
            return _bcast(x, name)

        D_intra_arr   = _bcast(D_intra,  'D_intra')
        D_myelin_arr  = _bcast(D_myelin, 'D_myelin')
        D_extra_arr   = _bcast(D_extra,  'D_extra')
        kappa_inner_arr = _bcast_kappa(kappa_inner, 'kappa_inner')
        kappa_outer_arr = _bcast_kappa(kappa_outer, 'kappa_outer')
        rho_inner_arr   = _bcast_kappa(rho_inner,   'rho_inner')
        rho_outer_arr   = _bcast_kappa(rho_outer,   'rho_outer')
        T2_intra_arr  = _bcast_opt(T2_intra,  'T2_intra')
        T2_myelin_arr = _bcast_opt(T2_myelin, 'T2_myelin')
        T2_extra_arr  = _bcast_opt(T2_extra,  'T2_extra')

        # Pad to N_max with zeros
        def _pad(arr):
            out = np.zeros(N_max, dtype=np.float64)
            out[:N_actual] = arr
            return out

        def _pad_opt(arr):
            return None if arr is None else _pad(arr)

        inner_p   = _pad(inner_radii)
        outer_p   = _pad(outer_radii)
        cx_np     = np.zeros(N_max, dtype=np.float64)
        cy_np     = np.zeros(N_max, dtype=np.float64)
        cx_np[:N_actual] = centers[:, 0]
        cy_np[:N_actual] = centers[:, 1]
        centers_padded = np.stack([cx_np, cy_np], axis=1)  # (N_max, 2)

        D_intra_p  = _pad(D_intra_arr)
        D_myelin_p = _pad(D_myelin_arr)
        D_extra_p  = _pad(D_extra_arr)
        T2_intra_p  = _pad_opt(T2_intra_arr)
        T2_myelin_p = _pad_opt(T2_myelin_arr)
        T2_extra_p  = _pad_opt(T2_extra_arr)
        kappa_inner_p = _pad(kappa_inner_arr)
        kappa_outer_p = _pad(kappa_outer_arr)
        rho_inner_p   = _pad(rho_inner_arr)
        rho_outer_p   = _pad(rho_outer_arr)

        # Store numpy arrays for init_positions
        self._inner_radii_np = inner_p
        self._outer_radii_np = outer_p
        self._centers_np     = centers_padded  # (N_max, 2)

        # JAX constants (baked at construction, one JIT per N_max)
        self._L_jax            = jnp.float32(cell_size)
        self._L_float          = float(cell_size)
        self._inner_radii_jax  = jnp.array(inner_p,        dtype=jnp.float32)
        self._outer_radii_jax  = jnp.array(outer_p,        dtype=jnp.float32)
        self._centers_jax      = jnp.array(centers_padded, dtype=jnp.float32)
        self._D_intra_jax      = jnp.array(D_intra_p,      dtype=jnp.float32)
        self._D_myelin_jax     = jnp.array(D_myelin_p,     dtype=jnp.float32)
        self._D_extra_jax      = jnp.array(D_extra_p,      dtype=jnp.float32)

        has_t2 = (T2_intra is not None or T2_myelin is not None or
                  T2_extra is not None)
        self._has_t2 = has_t2
        _BIG = np.float32(1e6)
        if has_t2:
            self._T2_intra_jax = jnp.array(
                T2_intra_p  if T2_intra_p  is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)
            self._T2_myelin_jax = jnp.array(
                T2_myelin_p if T2_myelin_p is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)
            self._T2_extra_jax = jnp.array(
                T2_extra_p  if T2_extra_p  is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)

        self._kappa_inner_jax = jnp.array(kappa_inner_p, dtype=jnp.float32)
        self._kappa_outer_jax = jnp.array(kappa_outer_p, dtype=jnp.float32)
        self._rho_inner_jax   = jnp.array(rho_inner_p,   dtype=jnp.float32)
        self._rho_outer_jax   = jnp.array(rho_outer_p,   dtype=jnp.float32)

        # Rotation matrix (shared cylinder axis)
        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        _R_np = _rotation_to_z(self.orientation)
        self._R     = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))

        # EPS/NUDGE — scale by smallest non-zero inner radius
        nonzero = inner_p[inner_p > 0]
        ref_r = float(np.min(nonzero)) if len(nonzero) > 0 else 1e-6
        self._eps   = jnp.float32(1e-7 * ref_r)
        self._nudge = jnp.float32(1e-4 * ref_r)

        # Minimum gap (diagnostic, uses outer radii)
        self.min_gap = self._compute_min_gap()

    def _compute_min_gap(self):
        """Minimum clear gap between outer boundaries (actual cylinders only)."""
        N = self.N_actual
        L = self._L_float
        centers = self._centers_np[:N]
        outer   = self._outer_radii_np[:N]
        min_gap = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                dq = centers[i] - centers[j]
                dq -= L * np.round(dq / L)
                gap = np.linalg.norm(dq) - outer[i] - outer[j]
                min_gap = min(min_gap, gap)
            min_gap = min(min_gap, L - 2.0 * outer[i])
        return float(min_gap) if np.isfinite(min_gap) else float('inf')

    def volume_fraction(self, compartment: str) -> float:
        """Volume fraction of a named compartment within the periodic cell.

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        """
        L = self._L_float
        N = self.N_actual
        inner = self._inner_radii_np[:N]
        outer = self._outer_radii_np[:N]
        cell_area = L * L
        if compartment == 'intra':
            return float(np.pi * np.sum(inner ** 2) / cell_area)
        elif compartment == 'myelin':
            return float(np.pi * np.sum(outer ** 2 - inner ** 2) / cell_area)
        elif compartment == 'extra':
            total_cyl = np.pi * np.sum(outer ** 2)
            return float((cell_area - total_cyl) / cell_area)
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def init_positions(self, n_walkers, key):
        """Distribute walkers proportional to compartment area in the periodic cell.

        Walker allocation (area-weighted):
          - Extra-axonal : area = L^2 - sum(pi*R_outer_k^2)
          - Intra_k      : area = pi*R_inner_k^2  (zero for dummy cylinders)
          - Myelin_k     : area = pi*(R_outer_k^2 - R_inner_k^2)

        Dummy cylinders (R=0) automatically receive zero walkers.
        """
        L = self._L_float
        N = self.N_actual
        N_max = self.N_max
        inner = self._inner_radii_np[:N]
        outer = self._outer_radii_np[:N]
        centers = self._centers_np[:N]

        cell_area = L * L
        area_intra  = np.pi * inner ** 2                  # (N,)
        area_myelin = np.pi * (outer ** 2 - inner ** 2)   # (N,)
        area_extra  = cell_area - np.pi * np.sum(outer ** 2)

        total_area = area_extra + np.sum(area_intra) + np.sum(area_myelin)

        n_intra  = np.array([int(round(n_walkers * a / total_area))
                             for a in area_intra], dtype=int)
        n_myelin = np.array([int(round(n_walkers * a / total_area))
                             for a in area_myelin], dtype=int)

        # Extra fills remainder (handles rounding)
        n_extra = n_walkers - int(np.sum(n_intra)) - int(np.sum(n_myelin))
        if n_extra < 0:
            excess = -n_extra
            for k in range(N):
                trim = min(excess, n_intra[k])
                n_intra[k] -= trim
                excess -= trim
                if excess == 0:
                    break
            n_extra = 0

        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2 ** 30)))

        positions    = np.zeros((n_walkers, 3), dtype=np.float32)
        compartments = np.zeros(n_walkers, dtype=np.int32)
        idx = 0

        # Intra-axonal: compartment_id = k+1  (1-based, slot 1..N_max)
        for k in range(N):
            nk = int(n_intra[k])
            if nk == 0:
                continue
            r_k = float(inner[k])
            cx_k, cy_k = float(centers[k, 0]), float(centers[k, 1])
            pts = []
            while sum(len(p) for p in pts) < nk:
                batch = max(nk * 4, 64)
                xy = rng.uniform(-r_k, r_k, (batch, 2))
                pts.append(xy[np.linalg.norm(xy, axis=1) < r_k])
            xy_k = np.concatenate(pts, axis=0)[:nk].astype(np.float32)
            # Shift to cylinder centre
            positions[idx:idx + nk, 0] = xy_k[:, 0] + cx_k
            positions[idx:idx + nk, 1] = xy_k[:, 1] + cy_k
            compartments[idx:idx + nk] = k + 1
            idx += nk

        # Myelin: compartment_id = N_max + k + 1  (slots N_max+1..2*N_max)
        for k in range(N):
            nk = int(n_myelin[k])
            if nk == 0:
                continue
            r_in  = float(inner[k])
            r_out = float(outer[k])
            cx_k, cy_k = float(centers[k, 0]), float(centers[k, 1])
            pts = []
            while sum(len(p) for p in pts) < nk:
                batch = max(nk * 4, 64)
                xy = rng.uniform(-r_out, r_out, (batch, 2))
                d = np.linalg.norm(xy, axis=1)
                pts.append(xy[(d >= r_in) & (d < r_out)])
            xy_k = np.concatenate(pts, axis=0)[:nk].astype(np.float32)
            positions[idx:idx + nk, 0] = xy_k[:, 0] + cx_k
            positions[idx:idx + nk, 1] = xy_k[:, 1] + cy_k
            compartments[idx:idx + nk] = N_max + k + 1
            idx += nk

        # Extra-axonal: compartment_id = 0
        n_extra_actual = n_walkers - idx
        if n_extra_actual > 0:
            accepted = []
            n_have = 0
            while n_have < n_extra_actual:
                batch = max(n_extra_actual * 4, 1024)
                xy = rng.uniform(-L / 2.0, L / 2.0, (batch, 2))
                outside = np.ones(batch, dtype=bool)
                for k in range(N):
                    dxy = xy - centers[k]
                    dxy -= L * np.round(dxy / L)   # min-image
                    outside &= np.sum(dxy ** 2, axis=1) > outer[k] ** 2
                accepted.append(xy[outside])
                n_have = sum(len(a) for a in accepted)
            xy_ex = np.concatenate(accepted)[:n_extra_actual].astype(np.float32)
            positions[idx:idx + n_extra_actual, 0] = xy_ex[:, 0]
            positions[idx:idx + n_extra_actual, 1] = xy_ex[:, 1]
            compartments[idx:idx + n_extra_actual] = 0

        # Rotate to lab frame (positions in cylinder-frame xy-plane, z=0)
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ positions.T).T

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Not a usable boundary rule for this geometry -- raises.

        This returned `r + step`: free diffusion, no boundaries at all.  Reached through
        `simulate_trajectories` it produced an unrestricted walk on a packed axon substrate
        (measured: RMS displacement 1.02x free through 1 um axons) with no error and no
        warning.  The real kernel is `physics.make_packed_myelin_traj_step_fn`, which
        carries the compartment.
        """
        raise NotImplementedError(
            "PackedMyelinatedCylinders has no single-surface reflect: it is stepped by the "
            "fused kernel physics.make_packed_myelin_traj_step_fn, which carries the "
            "compartment id. Use simulate(...) or simulate_trajectories(..., "
            "save_relaxation_data=True), which dispatch to that kernel.")

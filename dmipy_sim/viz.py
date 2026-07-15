"""Waveform visualization utilities.

Main entry points
-----------------
plot_waveform(wf, ...)               — single waveform, 3-panel figure
plot_sequence_comparison(wfs, ...)   — N sequences side-by-side, 3-row × N-col grid
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from .constants import GAMMA

# x / y / z colour palette
_COLORS = ['#2196F3', '#4CAF50', '#F44336']   # blue, green, red
_LABELS = ['x', 'y', 'z']

# RF event colours: 90° pulse vs 180° pulse
_RF_COLOR_90  = '#7B1FA2'   # purple
_RF_COLOR_180 = '#E65100'   # deep orange


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for waveform visualisation.")


def _q_from_waveform(wf):
    """Return q(t) array of shape (n_meas, n_t, 3) in m⁻¹."""
    G = np.array(wf.G)                              # (n_meas, n_t, 3)
    return np.cumsum(G * wf.dt, axis=1) * GAMMA     # (n_meas, n_t, 3)


def _active_axes(arr_nt3):
    """Return list of axis indices (0–2) where the signal is non-trivially non-zero."""
    return [i for i in range(3) if np.any(np.abs(arr_nt3[:, i]) > 1e-12 * np.max(np.abs(arr_nt3) + 1e-30))]


def _draw_rf_panel(ax, wf, t_plot):
    """Populate the RF-event timeline axis."""
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(0, 1)
    ax.axis('off')

    rf_events = wf.rf_events or []
    t_total_s = wf.dt * (len(t_plot) - 1)
    disp_per_s = (t_plot[-1] - t_plot[0]) / t_total_s if t_total_s > 0 else 1.0

    # Two height tiers (tall / short) — alternate for closely-spaced events
    # Tier 0 (tall):  arrow tip at y=0.30, flip label at 0.92, event label at 0.72
    # Tier 1 (short): arrow tip at y=0.30, flip label at 0.65, event label at 0.45
    TIERS = [
        {'flip_y': 0.92, 'label_y': 0.70, 'line_ymax': 0.60},
        {'flip_y': 0.60, 'label_y': 0.38, 'line_ymax': 0.28},
    ]
    LINE_YMIN = 0.05

    t_evs = [ev['t_s'] * disp_per_s for ev in rf_events]
    min_gap = (t_plot[-1] - t_plot[0]) * 0.18

    # Greedy tier assignment: events within min_gap of same tier get bumped to tier 1
    tiers = [0] * len(rf_events)
    for k in range(1, len(rf_events)):
        for j in range(k):
            if abs(t_evs[k] - t_evs[j]) < min_gap and tiers[j] == tiers[k]:
                tiers[k] = (tiers[k] + 1) % 2

    for idx, ev in enumerate(rf_events):
        t_ev  = t_evs[idx]
        flip  = ev['flip_deg']
        label = ev['label']
        color = _RF_COLOR_90 if flip == 90 else _RF_COLOR_180
        lw    = 2.0 if flip == 90 else 3.0
        tier  = TIERS[min(tiers[idx], 1)]

        # Vertical line from baseline up to just below the label
        ax.axvline(t_ev, ymin=LINE_YMIN, ymax=tier['line_ymax'],
                   color=color, lw=lw, alpha=0.9, solid_capstyle='round')

        # Flip angle (bold, larger)
        ax.text(t_ev, tier['flip_y'], f'{flip}°',
                ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold',
                transform=ax.transData)

        # Event label (italic, slightly smaller)
        ax.text(t_ev, tier['label_y'], label,
                ha='center', va='bottom',
                fontsize=8, color=color, style='italic',
                transform=ax.transData)

    # Echo marker — dashed line + label
    echo_t = t_plot[wf.echo_idx]
    ax.axvline(echo_t, ymin=LINE_YMIN, ymax=0.55,
               color='#555555', lw=1.4, ls='--', alpha=0.55)
    ax.text(echo_t, 0.60, 'echo',
            ha='center', va='bottom',
            fontsize=8, color='#555555', alpha=0.85)


def _draw_G_panel(ax, wf, t_plot, meas_idx, show_ylabel=True):
    """Populate the G(t) axis."""
    G_all = np.array(wf.G)          # (n_meas, n_t, 3)
    n_meas = G_all.shape[0]
    G_hi = G_all[meas_idx]          # (n_t, 3)

    # All measurements at low alpha (fan)
    if n_meas > 1:
        for m in range(n_meas):
            for i in range(3):
                ax.plot(t_plot, G_all[m, :, i], color=_COLORS[i],
                        alpha=0.06, lw=0.6)

    # Highlighted measurement
    active = _active_axes(G_hi)
    for i in active:
        ax.plot(t_plot, G_hi[:, i], color=_COLORS[i], lw=1.6,
                label=f'G$_{_LABELS[i]}$')

    ax.axhline(0, color='k', lw=0.5, alpha=0.25)
    echo_t = t_plot[wf.echo_idx]
    ax.axvline(echo_t, color='#333333', lw=1.0, ls='--', alpha=0.35)

    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    plt.setp(ax.get_xticklabels(), visible=False)
    if show_ylabel:
        ax.set_ylabel('G (T/m)', fontsize=8)
    if active:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6,
                  handlelength=1.2, borderpad=0.4)


def _draw_q_panel(ax, wf, t_plot, meas_idx, show_ylabel=True, t_unit='ms'):
    """Populate the q(t) axis."""
    q_all = _q_from_waveform(wf)    # (n_meas, n_t, 3)
    n_meas = q_all.shape[0]
    q_hi = q_all[meas_idx]          # (n_t, 3)
    G_hi = np.array(wf.G)[meas_idx]

    # Fan
    if n_meas > 1:
        for m in range(n_meas):
            for i in range(3):
                ax.plot(t_plot, q_all[m, :, i], color=_COLORS[i],
                        alpha=0.06, lw=0.6)

    # Highlighted measurement
    active = _active_axes(G_hi)
    for i in active:
        ax.plot(t_plot, q_hi[:, i], color=_COLORS[i], lw=1.6,
                label=f'q$_{_LABELS[i]}$')

    # |q| magnitude
    q_mag = np.linalg.norm(q_hi, axis=1)
    ax.plot(t_plot, q_mag, color='k', lw=1.4, ls='--', alpha=0.55, label='|q|')

    ax.axhline(0, color='k', lw=0.5, alpha=0.25)
    echo_t = t_plot[wf.echo_idx]
    ax.axvline(echo_t, color='#333333', lw=1.0, ls='--', alpha=0.35)

    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    ax.set_xlabel(f't ({t_unit})', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('q (m⁻¹)', fontsize=8)
    if active:
        handles, labs = ax.get_legend_handles_labels()
        ax.legend(handles, labs, fontsize=7, loc='upper right',
                  framealpha=0.6, handlelength=1.2, borderpad=0.4)


def plot_waveform(wf, meas_idx=0, title=None, t_unit='ms', figsize=(9, 5)):
    """Plot a single waveform: RF timeline, G(t), q(t).

    Parameters
    ----------
    wf : Waveform
    meas_idx : int
        Measurement index to highlight (default 0).
    title : str or None
        Figure suptitle.
    t_unit : {'ms', 's'}
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of (ax_rf, ax_G, ax_q)
    """
    _require_mpl()

    n_t = wf.G.shape[1]
    dt  = wf.dt
    t_scale = 1e3 if t_unit == 'ms' else 1.0
    t_plot = np.arange(n_t) * dt * t_scale

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[1.4, 2, 2],
                            hspace=0.12)
    ax_rf = fig.add_subplot(gs[0])
    ax_G  = fig.add_subplot(gs[1])
    ax_q  = fig.add_subplot(gs[2])

    _draw_rf_panel(ax_rf, wf, t_plot)
    _draw_G_panel(ax_G, wf, t_plot, meas_idx)
    _draw_q_panel(ax_q, wf, t_plot, meas_idx, t_unit=t_unit)

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)

    fig.tight_layout()
    return fig, (ax_rf, ax_G, ax_q)


def plot_sequence_comparison(waveforms, titles=None, meas_idx=0,
                              t_unit='ms', figsize=(14, 6)):
    """Plot N sequences side-by-side: 3 rows × N columns.

    Row 0 — RF event timeline
    Row 1 — G(t) with x/y/z components
    Row 2 — q(t) with x/y/z components + |q|

    Parameters
    ----------
    waveforms : list of Waveform
    titles : list of str or None
    meas_idx : int
        Measurement index to highlight in each waveform.
    t_unit : {'ms', 's'}
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_mpl()

    n = len(waveforms)
    titles = titles or [''] * n
    t_scale = 1e3 if t_unit == 'ms' else 1.0

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, n, figure=fig,
                            height_ratios=[1.4, 2, 2],
                            hspace=0.10, wspace=0.38)

    for col, (wf, title) in enumerate(zip(waveforms, titles)):
        n_t    = wf.G.shape[1]
        t_plot = np.arange(n_t) * wf.dt * t_scale

        ax_rf = fig.add_subplot(gs[0, col])
        ax_G  = fig.add_subplot(gs[1, col])
        ax_q  = fig.add_subplot(gs[2, col])

        show_ylabel = (col == 0)

        _draw_rf_panel(ax_rf, wf, t_plot)
        _draw_G_panel(ax_G,  wf, t_plot, meas_idx, show_ylabel=show_ylabel)
        _draw_q_panel(ax_q,  wf, t_plot, meas_idx, show_ylabel=show_ylabel,
                      t_unit=t_unit)

        if title:
            ax_rf.set_title(title, fontsize=9.5, fontweight='bold', pad=3)

        # Hide y tick labels on non-leftmost columns for G and q
        if col > 0:
            plt.setp(ax_G.get_yticklabels(), visible=False)
            plt.setp(ax_q.get_yticklabels(), visible=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Mesh substrate observability
# ---------------------------------------------------------------------------

_AXES = {'x': 0, 'y': 1, 'z': 2}
_MESH_WALL = '#1f3a5f'
_MESH_INTRA = '#e6550d'
_MESH_EXTRA = '#3182bd'


def _mesh_slice_segments(vertices, faces, axis_idx, offset):
    """In-plane line segments where each triangle crosses the plane.

    Robust triangle-plane slice (captures the open cross-sections that a
    closed-loop contour tracer drops on clipped / periodic meshes).  Returns
    ``(segments, plane_axes)`` where segments is a list of (2, 2) arrays in the
    two in-plane coordinates and plane_axes are their global axis indices.
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int)
    plane_axes = [i for i in range(3) if i != axis_idx]
    tz = V[F][:, :, axis_idx] - offset                       # (nF, 3)
    straddle = (tz.min(1) < 0) & (tz.max(1) > 0)
    tri = V[F][straddle]
    tzz = tz[straddle]
    segs = []
    for tv, zz in zip(tri, tzz):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if zz[i] * zz[j] < 0:
                t = zz[i] / (zz[i] - zz[j])
                pts.append(tv[i, plane_axes] + t * (tv[j, plane_axes] - tv[i, plane_axes]))
        if len(pts) == 2:
            segs.append(np.array(pts))
    return segs, plane_axes


def _walker_compartments(mesh, walkers):
    import jax
    return np.asarray(jax.vmap(mesh.classify_position)(np.asarray(walkers, np.float32)))


def plot_mesh_section(mesh, axis='z', offset=0.0, walkers=None, compartments=None,
                      slab=None, units=1e6, unit_label='µm', ax=None, save=None):
    """Cross-section of a :class:`~dmipy_sim.Mesh` through a plane, drawing every
    cell wall, with optional walker positions overlaid.

    Parameters
    ----------
    mesh : Mesh
        The mesh geometry (uses its ``vertices`` / ``faces``, in the mesh frame).
    axis : {'x', 'y', 'z'}
        Plane normal.  The section is drawn in the other two coordinates.
    offset : float
        Plane position along ``axis`` (metres).
    walkers : (n, 3) array, optional
        Walker positions to overlay (those within ``slab`` of the plane).
    compartments : (n,) int array, optional
        0 = intra, 1 = extra, colouring the overlaid walkers.  Computed from the
        mesh if omitted.
    slab : float, optional
        Half-thickness (metres) of the walker slab around the plane.  Defaults to
        the mesh feature radius / 4.
    units, unit_label : float, str
        Axis scaling for display (default metres -> µm).
    ax : matplotlib Axes, optional
    save : str, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib Axes
    """
    _require_mpl()
    from matplotlib.collections import LineCollection
    ai = _AXES[axis]
    segs, pax = _mesh_slice_segments(mesh.vertices, mesh.faces, ai, offset)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.add_collection(LineCollection([s * units for s in segs],
                                     colors=_MESH_WALL, linewidths=0.5))
    if walkers is not None and len(walkers):
        w = np.asarray(walkers, float)
        if slab is None:
            slab = getattr(mesh, 'radius', 1.0) / 4.0
        keep = np.abs(w[:, ai] - offset) < slab
        wk = w[keep]
        if compartments is None:
            comp = _walker_compartments(mesh, wk)
        else:
            comp = np.asarray(compartments)[keep]
        for c, col, lab in ((0, _MESH_INTRA, 'intra'), (1, _MESH_EXTRA, 'extra')):
            m = comp == c
            if m.any():
                ax.scatter(wk[m, pax[0]] * units, wk[m, pax[1]] * units,
                           s=5, c=col, alpha=0.7, label=lab)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    lo = np.asarray(mesh.vmin)[pax] * units
    hi = np.asarray(mesh.vmax)[pax] * units
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    names = 'xyz'
    ax.set_xlabel(f'{names[pax[0]]} ({unit_label})')
    ax.set_ylabel(f'{names[pax[1]]} ({unit_label})')
    ax.set_title(f'{axis}={offset*units:.2f} {unit_label} section — {len(segs)} cell-wall segments')
    if save:
        ax.figure.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def plot_walkers_3d(mesh, walkers, compartments=None, sub_box=None, units=1e6,
                    unit_label='µm', ax=None, save=None):
    """3D scatter of walker positions coloured by compartment (0 intra / 1 extra).

    ``sub_box`` (metres) restricts to ``|coord| < sub_box`` about the origin so a
    dense pack stays legible.  Returns the matplotlib Axes.
    """
    _require_mpl()
    w = np.asarray(walkers, float)
    comp = _walker_compartments(mesh, w) if compartments is None else np.asarray(compartments)
    if sub_box is not None:
        keep = np.all(np.abs(w) < sub_box, axis=1)
        w, comp = w[keep], comp[keep]
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(111, projection='3d')
    for c, col, lab, al in ((0, _MESH_INTRA, 'intra', 0.5), (1, _MESH_EXTRA, 'extra', 0.3)):
        m = comp == c
        if m.any():
            ax.scatter(w[m, 0] * units, w[m, 1] * units, w[m, 2] * units,
                       s=4, c=col, alpha=al, label=lab)
    ax.set_xlabel(f'x ({unit_label})'); ax.set_ylabel(f'y ({unit_label})')
    ax.set_zlabel(f'z ({unit_label})')
    ax.legend(loc='upper left', fontsize=8)
    if save:
        ax.figure.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def plot_cell_surface(mesh, index=0, units=1e6, unit_label='µm', ax=None, save=None,
                      color=_MESH_EXTRA):
    """Render a single connected cell of the mesh as a 3D surface.

    ``index`` selects the cell by descending size (0 = largest).  Requires
    trimesh (the ``mesh`` extra) to split the mesh into connected components.
    Returns the matplotlib Axes.
    """
    _require_mpl()
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("plot_cell_surface needs trimesh (pip install dmipy-sim[mesh]).") from exc
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                         process=False)
    comps = tm.split(only_watertight=False)
    comps = sorted(comps, key=lambda c: len(c.vertices), reverse=True)
    c = comps[min(index, len(comps) - 1)]
    V = np.asarray(c.vertices) * units
    Fc = np.asarray(c.faces)
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(V[:, 0], V[:, 1], Fc, V[:, 2],
                    color=color, alpha=0.85, edgecolor='none', linewidth=0)
    ax.set_xlabel(f'x ({unit_label})'); ax.set_ylabel(f'y ({unit_label})')
    ax.set_zlabel(f'z ({unit_label})')
    ax.set_title(f'cell {index} — {len(c.faces):,} faces')
    if save:
        ax.figure.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def _split_cells(mesh):
    """Connected components of the mesh, largest first (needs trimesh)."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("cell extraction needs trimesh (pip install dmipy-sim[mesh]).") from exc
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                         process=False)
    return sorted(tm.split(only_watertight=False), key=lambda c: len(c.vertices), reverse=True)


def seed_in_cell(cell, n_walkers, seed=0):
    """Sample ``n_walkers`` positions strictly inside a single cell (a trimesh
    component), by rejection with a ray-cast parity test (robust on the wavy,
    concave cell walls where a nearest-normal side test misfires).  Returns
    (n, 3) metres."""
    from .geometries import _is_inside_batch
    V = np.asarray(cell.vertices, float)
    F = np.asarray(cell.faces, np.int64)
    lo, hi = V.min(0), V.max(0)
    rng = np.random.default_rng(seed)
    out, need = [], n_walkers
    while need > 0:
        p = rng.uniform(lo, hi, (max(need * 8, 2048), 3))
        inside = np.asarray(_is_inside_batch(p, V, F))       # +X ray parity
        out.append(p[inside])
        need = n_walkers - sum(len(a) for a in out)
    return np.concatenate(out)[:n_walkers]


def walk_paths(mesh, n_walkers, n_steps, diffusivity, dt, seed=0, intra=True, r0=None):
    """Record plain reflecting random-walk paths for visualisation.

    A diffusion-only walk (no gradient, no phase) that just steps the geometry's
    ``reflect`` and stores every position — a lightweight path recorder for
    plotting, independent of the signal-simulation path.

    Returns an array of shape ``(n_walkers, n_steps + 1, 3)`` (metres).
    """
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(seed)
    k0, kw = jax.random.split(key)
    if r0 is None:
        r0 = mesh.init_positions(n_walkers, k0, intra=intra)
    else:
        r0 = jnp.asarray(r0, jnp.float32)
    step_l = jnp.float32(np.sqrt(6.0 * diffusivity * dt))

    def one(r, k):
        noise = jax.random.normal(k, (n_walkers, 3), dtype=jnp.float32)
        step = noise / jnp.linalg.norm(noise, axis=1, keepdims=True) * step_l
        r_new = jax.vmap(mesh.reflect)(r, step)
        return r_new, r_new
    keys = jax.random.split(kw, n_steps)
    _, traj = jax.lax.scan(one, r0, keys)                    # (n_steps, n_walkers, 3)
    traj = jnp.concatenate([r0[None], traj], axis=0)         # (n_steps+1, n_walkers, 3)
    return np.asarray(jnp.transpose(traj, (1, 0, 2)))        # (n_walkers, n_steps+1, 3)


def plot_trajectories(mesh, paths, axis='z', offset=0.0, slab=None, wrap=True,
                      units=1e6, unit_label='µm', ax=None, save=None, max_paths=40):
    """Overlay walker paths on a mesh cross-section.

    ``paths`` is ``(n_walkers, n_t, 3)`` from :func:`walk_paths` (continuous /
    unfolded positions).

    Important for a fair picture of confinement: a path is 3D, but the drawn cell
    walls are a single ``offset`` slice.  On a periodic and/or elongated substrate
    a walker roams along the free axis and across periodic images, so flattening
    the whole path onto one slice makes an in-cell walker look like it crosses
    walls.  Therefore, by default:

    * ``wrap=True`` folds continuous positions back into the periodic box (per the
      mesh's periodic axes), and
    * ``slab`` (default: feature_radius) keeps only the path points whose
      out-of-plane coordinate is within ``slab`` of the plane, drawn as points —
      so what you see genuinely sits in this slice and inside the cells.

    Pass ``slab=None`` and ``wrap=False`` to draw the raw projected polylines.
    Returns the matplotlib Axes.
    """
    _require_mpl()
    ai = _AXES[axis]
    pax = [i for i in range(3) if i != ai]
    ax = plot_mesh_section(mesh, axis=axis, offset=offset, units=units,
                           unit_label=unit_label, ax=ax)
    P = np.array(paths, float)[:max_paths]
    if wrap:
        vmin = np.asarray(mesh.vmin); L = np.asarray(mesh.vmax) - vmin
        for a in range(3):
            if getattr(mesh, 'periodic', (False, False, False))[a]:
                P[:, :, a] = vmin[a] + np.mod(P[:, :, a] - vmin[a], L[a])
    if slab is None:
        slab = getattr(mesh, 'radius', np.inf)
    for p in P:
        if np.isfinite(slab):
            m = np.abs(p[:, ai] - offset) < slab
            ax.scatter(p[m, pax[0]] * units, p[m, pax[1]] * units, s=4, alpha=0.8)
        else:
            ax.plot(p[:, pax[0]] * units, p[:, pax[1]] * units, lw=0.8, alpha=0.8)
    kind = f'|{axis}-{offset*units:.0f}|<{slab*units:.1f}{unit_label} slab' if np.isfinite(slab) else 'projected'
    ax.set_title(ax.get_title() + f'  ·  {len(P)} walker paths ({kind})')
    if save:
        ax.figure.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def plot_mesh_3d(mesh, cells=(0,), paths=None, wrap=True, alpha=0.13,
                 cell_color=_MESH_WALL, units=1e6, unit_label='µm', ax=None, save=None):
    """True 3D view: selected cells drawn as transparent surfaces with walker
    paths scribbling inside them — the honest confinement view for a 3D substrate
    (no plane slice, no projection).

    Parameters
    ----------
    mesh : Mesh
    cells : iterable of int
        Indices (largest-first) of connected cells to render transparently.
    paths : (n_walkers, n_t, 3) array, optional
        Walker paths from :func:`walk_paths` (ideally seeded inside a rendered
        cell via :func:`seed_in_cell`).  On a periodic mesh pass ``wrap=True`` so
        the continuous path folds back into the box and stays within the cell.
    alpha : float
        Cell-surface transparency.
    Returns the matplotlib Axes.
    """
    _require_mpl()
    comps = _split_cells(mesh)
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    for ci in cells:
        c = comps[min(ci, len(comps) - 1)]
        V = np.asarray(c.vertices) * units
        ax.plot_trisurf(V[:, 0], V[:, 1], np.asarray(c.faces), V[:, 2],
                        color=cell_color, alpha=alpha, edgecolor='none', linewidth=0,
                        shade=True)
    if paths is not None:
        P = np.array(paths, float)
        if wrap:
            vmin = np.asarray(mesh.vmin); L = np.asarray(mesh.vmax) - vmin
            per = getattr(mesh, 'periodic', (False, False, False))
            for a in range(3):
                if per[a]:
                    P[:, :, a] = vmin[a] + np.mod(P[:, :, a] - vmin[a], L[a])
            # break each polyline at periodic wrap seams so it doesn't streak
            L_arr = L
            for p in P:
                seg = p.copy()
                jump = np.any(np.abs(np.diff(seg, axis=0)) > 0.5 * L_arr, axis=1)
                seg[1:][jump] = np.nan
                ax.plot(seg[:, 0] * units, seg[:, 1] * units, seg[:, 2] * units,
                        lw=0.9, alpha=0.9)
        else:
            for p in P:
                ax.plot(p[:, 0] * units, p[:, 1] * units, p[:, 2] * units, lw=0.9, alpha=0.9)
    ax.set_xlabel(f'x ({unit_label})'); ax.set_ylabel(f'y ({unit_label})')
    ax.set_zlabel(f'z ({unit_label})')
    if save:
        ax.figure.savefig(save, dpi=130, bbox_inches='tight')
    return ax


def save_rotation(ax3d, path, n_frames=48, elev=22, fps=16):
    """Save a spinning view of a 3D Axes as an animated GIF (azimuth sweep).

    Uses the Pillow writer (no ffmpeg needed).  Returns the output path.
    """
    _require_mpl()
    from matplotlib import animation
    fig = ax3d.figure

    def update(i):
        ax3d.view_init(elev=elev, azim=i * 360.0 / n_frames)
        return ()
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    return path

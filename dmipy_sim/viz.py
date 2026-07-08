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

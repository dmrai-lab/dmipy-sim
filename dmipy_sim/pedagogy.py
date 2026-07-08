"""Pedagogic visualisation of spin-population dynamics on the forward MC engine.

Turns one diffusion walk under a standard pulse sequence into pictures an MRI physicist can
read: what the transverse magnetisation of each compartment does, instant by instant, as the
sequence plays out on the substrate -- the 90 tipping it transverse, the gradient fanning it
out (dephasing), a 180 refocusing it into an echo.

**Idealised pulses.** The educational content is the gradient-driven PHASE: spins dephase as
``d phi = gamma * G(t).r(t) * dt`` and refocus when a 180 conjugates the accumulated phase.
Whether the refocusing pulse is a real shaped B1 or an idealised hard pulse of constant
nutation is immaterial to that story, so this module uses idealised rotations (Rodrigues about
the in-plane B1 axis) and draws a sinc-like placeholder glyph for the RF. No Bloch field solve,
no susceptibility, no pulse design -- it runs on the standard ``pgse`` / ``ogse`` waveforms that
dmipy-sim and dmipy-fit share.

    replay_with_history(geometry, waveform, D)  -> history          # walk once, keep M(t)
    sequence_story(history, save=...)           -> a multi-panel figure (sequence + M(t))
    spin_movie(history, save=...)               -> an animation of the spin populations

Colours: compartment 0 green, 1 blue, 2 orange (e.g. extra / intra / myelin).
"""
from __future__ import annotations

import numpy as np

from .constants import GAMMA

_COMP_COLOR = {0: '#2ca02c', 1: '#1f77b4', 2: '#ff7f0e'}
_COMP_NAME = {0: 'intra-axonal', 1: 'myelin', 2: 'extra-axonal'}


# ── idealised RF rotation (Rodrigues about the in-plane B1 axis) ─────────────
def _rf_increment(M, flip, ax):
    """Rotate M (3, N) by ``flip`` rad about the in-plane axis ``ax`` rad = (cos,sin,0)."""
    ux, uy = np.cos(ax), np.sin(ax)
    c, s = np.cos(flip), np.sin(flip)
    omc = 1.0 - c
    Mx, My, Mz = M[0], M[1], M[2]
    return np.stack([
        (c + ux*ux*omc)*Mx + (ux*uy*omc)*My + (uy*s)*Mz,
        (ux*uy*omc)*Mx + (c + uy*uy*omc)*My + (-ux*s)*Mz,
        (-uy*s)*Mx + (ux*s)*My + c*Mz,
    ])


# ── viz-scoped walk recorder (positions + compartment id over time) ──────────
def _walk_record(geometry, diffusivity, n_t, dt, n_walkers, seed):
    """Walk ``n_walkers`` spins on ``geometry`` and return their positions r(t) and
    compartment id at each of ``n_t`` steps.  Scoped to visualisation (a small ensemble,
    one sequence) -- it is NOT a reusable trajectory-replay primitive."""
    import jax
    import jax.numpy as jnp

    # Packed myelinated cylinders carry their own per-compartment (intra/myelin/extra) walk
    # with per-compartment diffusivity + permeability, so use the dedicated trajectory step
    # function rather than the single-diffusivity generic reflect below.
    if getattr(geometry, '_is_packed_myelinated', False):
        from .physics import make_packed_myelin_traj_step_fn
        step_fn = make_packed_myelin_traj_step_fn(geometry, dt)
        N_max = geometry.N_max
        pk, wk = jax.random.split(jax.random.PRNGKey(seed))
        r0 = geometry.init_positions(n_walkers, pk)
        comp0 = geometry._init_compartments        # encoded: 0=extra, 1..N=intra, >N=myelin

        def one(r0_w, key_w, c0):
            def emit(carry, _):
                nc, _ = step_fn(carry, None)
                return nc, (nc[0], nc[3])          # (position, compartment_id)
            _, (pos, cid) = jax.lax.scan(
                emit, (r0_w, key_w, jnp.float32(0.0), c0), None, length=n_t)
            return pos, cid

        pos, cid = jax.vmap(one)(r0, jax.random.split(wk, n_walkers), comp0)
        traj = np.asarray(pos)                     # (n_w, n_t, 3)
        cid = np.asarray(cid)
        # map the encoded id to the canonical 3-class label: 0=intra, 1=myelin, 2=extra
        comp = np.where(cid == 0, 2, np.where(cid > N_max, 1, 0)).astype(int)
        return traj, comp

    step_l = jnp.float32(np.sqrt(6.0 * diffusivity * dt))
    reflect = geometry.reflect
    has_perm = getattr(geometry, 'permeability', None) is not None

    def step(carry, _):
        r, key = carry
        key, sk = jax.random.split(key)
        if has_perm:
            key, pk = jax.random.split(key)
            n = jax.random.normal(sk, (3,), jnp.float32)
            r2, _dw = geometry.permeate(r, n / jnp.linalg.norm(n) * step_l,
                                        jnp.float32(geometry.permeability / diffusivity),
                                        jnp.float32(0.0), pk)
        else:
            n = jax.random.normal(sk, (3,), jnp.float32)
            r2 = reflect(r, n / jnp.linalg.norm(n) * step_l)
        return (r2, key), r2

    def one(r0, key):
        (_, _), pos = jax.lax.scan(step, (r0, key), None, length=n_t)
        return pos

    mk = jax.random.PRNGKey(seed)
    pk, wk = jax.random.split(mk)
    r0 = geometry.init_positions(n_walkers, pk)
    traj = np.asarray(jax.vmap(one)(r0, jax.random.split(wk, n_walkers)))  # (n_w, n_t, 3)
    if hasattr(geometry, 'classify_position'):
        comp = np.asarray(jax.vmap(jax.vmap(geometry.classify_position))(jnp.asarray(traj)))
        comp = comp.astype(int)
    else:
        comp = np.zeros(traj.shape[:2], dtype=int)
    return traj, comp


def _rf_events_for(waveform):
    """Return the ``[{t_s, flip_deg, axis_deg, duration_s}]`` RF list for a waveform.

    Standard waveforms carry ideal (instantaneous) ``rf_events`` directly; a bare
    excitation at t=0 is the fallback."""
    rf = getattr(waveform, 'rf_events', None)
    if rf:
        return [{'t_s': float(e['t_s']), 'flip_deg': float(e.get('flip_deg', 180.0)),
                 'axis_deg': float(e.get('axis_deg') or 0.0),
                 'duration_s': float(e.get('duration_s', 0.0) or 0.0)} for e in rf]
    return [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 0.0, 'duration_s': 0.0}]


def _idealised_history(traj, comp, dt, G, rf_events, T2=None, T1=None,
                       T2_per_comp=None, T1_per_comp=None, sub_idx=None):
    """Evolve each walker's magnetisation M=(Mx,My,Mz) through gradient precession +
    idealised hard-pulse rotations + T2/T1 relaxation, recording per-walker (sub-sample) and
    per-compartment mean M(t).  G is (n_t, 3) on the trajectory grid."""
    n_w, n_t, _ = traj.shape
    G = np.asarray(G, dtype=np.float64)
    if T2_per_comp is not None:
        invT2 = (1.0 / np.asarray(T2_per_comp, float))[comp]
    else:
        invT2 = np.full((n_w, n_t), 0.0 if T2 is None else 1.0 / T2)
    if T1_per_comp is not None:
        invT1 = (1.0 / np.asarray(T1_per_comp, float))[comp]
    else:
        invT1 = np.full((n_w, n_t), 0.0 if T1 is None else 1.0 / T1)
    E2, E1 = np.exp(-dt * invT2), np.exp(-dt * invT1)

    # map RF events onto trajectory steps (finite duration -> equal sub-rotations)
    rf_at = {}
    for e in rf_events:
        i0 = int(round(e['t_s'] / dt))
        nsub = max(1, int(round(e['duration_s'] / dt))) if e['duration_s'] > 0 else 1
        i_start = max(0, i0 - nsub // 2)
        dflip = np.deg2rad(e['flip_deg']) / nsub
        ax = np.deg2rad(e['axis_deg'])
        for j in range(nsub):
            rf_at.setdefault(min(i_start + j, n_t - 1), []).append((dflip, ax))

    comps = np.unique(comp[:, 0])
    sub = np.arange(min(300, n_w)) if sub_idx is None else np.asarray(sub_idx, int)
    hist_sub = np.zeros((n_t, 3, sub.size))
    hist_comp = np.zeros((n_t, 3, comps.size))
    comp0 = comp[:, 0]

    M = np.zeros((3, n_w)); M[2] = 1.0
    for t in range(n_t):
        for dflip, ax in rf_at.get(t, ()):
            M = _rf_increment(M, dflip, ax)
        dphi = GAMMA * dt * (traj[:, t, :] @ G[t])
        c, s = np.cos(dphi), np.sin(dphi)
        M = np.stack([c*M[0] - s*M[1], s*M[0] + c*M[1], M[2]])
        M = np.stack([M[0]*E2[:, t], M[1]*E2[:, t], M[2]*E1[:, t]])
        hist_sub[t] = M[:, sub]
        for ci, cv in enumerate(comps):
            sel = comp0 == cv
            hist_comp[t, :, ci] = M[:, sel].mean(axis=1) if sel.any() else 0.0
    return dict(t=np.arange(n_t) * dt, M_sub=hist_sub, M_comp=hist_comp,
                comps=comps, sub_idx=sub, comp0_sub=comp0[sub])


def replay_with_history(geometry, waveform, diffusivity, *, T2=None, T1=None,
                        T2_per_comp=None, T1_per_comp=None, n_walkers=20_000,
                        n_sub=240, seed=0):
    """Walk once on ``geometry`` under ``waveform`` and record the idealised magnetisation
    history M(t) (gradient phase + hard-pulse rotations + T2/T1).  Returns a ``history`` dict
    for :func:`sequence_story` / :func:`spin_movie`.  ``waveform`` is any standard
    ``pgse``/``ogse`` Waveform."""
    # Use the PHYSICAL same-sign gradient (G_display) if present: the idealised 180
    # rotation does the refocusing, so the effective/bipolar convention must NOT be used
    # (it would double-count the sign flip and cancel the echo).
    _gd = getattr(waveform, 'G_display', None)
    if _gd is not None:
        G = np.asarray(_gd)
        G = G[0] if G.ndim == 3 else G         # (n_t, 3)
    else:
        G = np.asarray(waveform.G)[0]          # (n_t, 3), first measurement
    dt = float(waveform.dt)
    n_t = G.shape[0]
    rf_events = _rf_events_for(waveform)
    traj, comp = _walk_record(geometry, diffusivity, n_t, dt, n_walkers, seed)

    # stratified sub-sample so every compartment shows a real cloud
    comp0 = comp[:, 0]
    uc = np.unique(comp0)
    per = max(1, int(n_sub) // len(uc))
    rng = np.random.default_rng(seed + 7)
    sub_idx = np.concatenate([rng.choice(np.where(comp0 == cv)[0],
                                         min(per, int((comp0 == cv).sum())), replace=False)
                              for cv in uc])
    history = _idealised_history(traj, comp, dt, G, rf_events, T2=T2, T1=T1,
                                 T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
                                 sub_idx=sub_idx)
    history['rf_events'] = rf_events
    history['G'] = G
    history['dt_wf'] = dt
    history['family'] = getattr(waveform, 'family', type(waveform).__name__.lower())
    return history


# ── renderers ────────────────────────────────────────────────────────────────
def _rf_glyph(ax, t_ms, flip_deg, ymax):
    """Draw a sinc-like placeholder for an (idealised) RF pulse at time t_ms."""
    w = 0.6                                     # ms, cosmetic width
    tt = np.linspace(t_ms - 1.6 * w, t_ms + 1.6 * w, 100)
    x = (tt - t_ms) / (0.5 * w)
    env = np.sinc(x) * (0.35 + 0.4 * (flip_deg / 180.0)) * ymax
    ax.plot(tt, env, color='#111', lw=1.3)
    ax.annotate(f"{int(round(flip_deg))}°", (t_ms, env.max()),
                textcoords='offset points', xytext=(0, 2), ha='center', fontsize=8)


def sequence_story(history, title=None, save=None, figsize=(10, 7)):
    """Multi-panel figure: (top) the RF (sinc glyphs) + gradient G(t); (middle) per-compartment
    transverse magnitude |Mxy|(t); (bottom) the net signal |<Mxy>|(t) with the echo(es)."""
    import matplotlib.pyplot as plt
    t = history['t'] * 1e3
    Mc = history['M_comp']                       # (n_t, 3, n_comp)
    comps = list(history['comps'])
    G = np.asarray(history['G']); tg = np.arange(G.shape[0]) * history['dt_wf'] * 1e3
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1.2, 1.0]))

    gmax = float(np.abs(G).max()) or 1.0
    for a in range(3):
        lbl = ['Gx', 'Gy', 'Gz'][a]
        if np.any(G[:, a] != 0):
            ax[0].plot(tg, G[:, a] * 1e3, lw=1.1, label=lbl)
    for e in history['rf_events']:
        _rf_glyph(ax[0], e['t_s'] * 1e3, e['flip_deg'], gmax * 1e3)
    ax[0].set_ylabel('G (mT/m) + RF'); ax[0].legend(loc='upper right', fontsize=8, ncol=3)
    ax[0].set_title(title or f"{history['family']}: spin-population dynamics (idealised pulses)")

    for ci, cv in enumerate(comps):
        mxy = np.hypot(Mc[:, 0, ci], Mc[:, 1, ci])
        ax[1].plot(t, mxy, color=_COMP_COLOR.get(int(cv), None),
                   label=_COMP_NAME.get(int(cv), f'comp {cv}'))
    ax[1].set_ylabel('|Mxy| per pool'); ax[1].legend(loc='upper right', fontsize=8)
    ax[1].set_ylim(-0.02, 1.02)

    # net signal = magnitude of the population-weighted mean transverse magnetisation
    net = np.zeros(len(t), dtype=complex)
    for ci in range(len(comps)):
        net += Mc[:, 0, ci] + 1j * Mc[:, 1, ci]
    net /= max(1, len(comps))
    ax[2].plot(t, np.abs(net), color='k', label='|net signal|')
    ax[2].set_ylabel('|net signal|'); ax[2].set_xlabel('time (ms)')
    ax[2].set_ylim(-0.02, 1.02); ax[2].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
    return fig


def _draw_player(ax, history):
    """Scoped pulse-program 'player' for the transverse / instant-pulse public engine.

    An RF track marking each instantaneous 90/180 flip (a stem line + flip-angle label) above
    ONE gradient track (Gx/Gy/Gz on a shared amplitude scale), the echo marked at the end, and
    a vertical playhead that sweeps in time. Returns the playhead ``Line2D`` (moved per frame).

    This is the transverse-only counterpart of the full Bloch pulse-program player: no finite
    B1(t) envelope (pulses are instantaneous), no longitudinal-storage / crusher windows.
    """
    G = np.asarray(history['G'])
    tg = np.arange(G.shape[0]) * history['dt_wf'] * 1e3          # ms
    tmax = float(tg[-1]) if len(tg) else 1.0
    Y_RF, Y_G, H = 2.0, 0.5, 1.0
    gmax = float(np.max(np.abs(G))) or 1e-12                     # shared scale across x,y,z

    for y, label in ((Y_RF, 'RF'), (Y_G, 'gradient')):
        ax.plot([0, tmax], [y, y], color='0.75', lw=0.7, zorder=1)
        ax.text(-0.015 * tmax, y, label, ha='right', va='center', fontsize=9, color='0.25')

    # instantaneous RF flips: a stem line + a downward tick + the flip-angle label
    for e in history['rf_events']:
        ts = float(e['t_s']) * 1e3
        flip = float(e['flip_deg'])
        h = 0.9 * H * min(flip / 180.0, 1.0)
        ax.plot([ts, ts], [Y_RF, Y_RF + h], color='crimson', lw=2.0, zorder=3)
        ax.plot([ts], [Y_RF + h], marker='v', color='crimson', ms=5, zorder=3)
        ax.annotate(f'{flip:.0f}°', (ts, Y_RF + h + 0.06), color='crimson', ha='center',
                    va='bottom', fontsize=8, annotation_clip=False)

    # all three physical gradient axes on the SAME scale
    for ax_i, (lab, col) in enumerate([('Gx', '#1f77b4'), ('Gy', '#2ca02c'), ('Gz', '#ff7f0e')]):
        if np.max(np.abs(G[:, ax_i])) > 1e-12:
            ax.fill_between(tg, Y_G, Y_G + 0.8 * H * G[:, ax_i] / gmax, color=col,
                            alpha=0.45, lw=0.8, edgecolor=col, zorder=2, label=lab)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', fontsize=6.5, ncol=3, frameon=False,
                  handlelength=1.0, columnspacing=1.0)

    ax.annotate('echo', (tmax, Y_RF), ha='left', va='center', fontsize=8, color='crimson')
    ax.axvline(tmax, color='crimson', ls=':', lw=1.0, alpha=0.7, zorder=1)
    ax.set_xlim(-0.02 * tmax, tmax * 1.02); ax.set_ylim(Y_G - 1.0, Y_RF + 1.3 * H)
    ax.set_xlabel('time [ms]'); ax.set_yticks([])
    for s in ('left', 'right', 'top'):
        ax.spines[s].set_visible(False)
    return ax.axvline(tg[0], color='k', lw=2.2, zorder=5)


def spin_movie(history, save, stride=2, n_cloud=200, fps=20, dpi=110, title=None):
    """Animate the transverse spin populations with a synced pulse-program player.

    Top row: one panel per compartment -- the transverse ``(Mx, My)`` cloud of individual
    walker magnetisations plus the bold net-signal vector (their mean). Watch the cloud fan out
    (dephasing) and snap back (refocusing / echo). Bottom (spanning all panels): the pulse
    program -- the instantaneous 90/180 RF flips (each a line + flip-angle label) and the
    diffusion gradient ``G(t)`` -- with a vertical playhead sweeping in time, so you see which
    event drives each motion.

    Magnetisation is transverse throughout (ideal instantaneous pulses). ``save`` is a
    .gif (pillow) or .mp4 (ffmpeg) path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    Ms = history['M_sub']; comp0_sub = np.asarray(history['comp0_sub'])
    Mc = history['M_comp']; comps = list(history['comps'])
    t_ms = history['t'] * 1e3
    frames = range(0, Ms.shape[0], max(1, int(stride)))

    n = len(comps)
    fig = plt.figure(figsize=(3.4 * n + 1.4, 6.6))
    gs = GridSpec(2, n, height_ratios=[2.2, 1.5], hspace=0.32,
                  top=0.86, bottom=0.12, figure=fig)
    ax_play = fig.add_subplot(gs[1, :])
    playhead = _draw_player(ax_play, history)
    ax_play.set_title('pulse-program player — 90°/180° flips + gradient G(t)',
                      fontsize=9, loc='left', color='0.3')
    if title:
        fig.suptitle(title, fontsize=12, color='0.15', weight='bold')

    clouds = []
    for ci, cv in enumerate(comps):
        a = fig.add_subplot(gs[0, ci]); col = _COMP_COLOR.get(int(cv), '#777')
        a.set_aspect('equal'); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15)
        a.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='#ccc'))
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel('Mx'); a.set_ylabel('My')
        a.set_title(_COMP_NAME.get(int(cv), f'compartment {cv}'), fontsize=10, color=col)
        sub = np.where(comp0_sub == cv)[0][:n_cloud]
        sc = a.scatter(Ms[0, 0, sub], Ms[0, 1, sub], s=8, c=col, alpha=0.5)
        arr = a.annotate('', xy=(Mc[0, 0, ci], Mc[0, 1, ci]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6))
        clouds.append((sc, sub, arr, ci))

    def draw(frame):
        for sc, sub, arr, ci in clouds:
            sc.set_offsets(np.c_[Ms[frame, 0, sub], Ms[frame, 1, sub]])
            arr.xy = (Mc[frame, 0, ci], Mc[frame, 1, ci])
        playhead.set_xdata([t_ms[frame], t_ms[frame]])
        return [c[0] for c in clouds] + [playhead]

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    writer = 'pillow' if str(save).endswith('.gif') else 'ffmpeg'
    anim.save(save, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return save


def _magnitude_walk(geometry, waveform, rho, T2_per_comp, n_walkers, seed, want_pos=False):
    """Shared walk for the magnitude renderers: walk the packed-myelin substrate and return
    each walker's transverse weight ``|M_i|(t) = exp(-∫dt/T2[comp] - (rho/D)·ℓ_i(t))``, its
    origin compartment (0=intra, 1=myelin, 2=extra), and (optionally) its position track."""
    import jax
    import jax.numpy as jnp
    from .physics import make_packed_myelin_traj_step_fn

    G = np.asarray(waveform.G)
    G = G[0] if G.ndim == 3 else G
    dt = float(waveform.dt)
    n_t = G.shape[0]
    step_fn = make_packed_myelin_traj_step_fn(geometry, dt)
    N_max = geometry.N_max
    pk, wk = jax.random.split(jax.random.PRNGKey(seed))
    r0 = geometry.init_positions(n_walkers, pk)
    comp0 = geometry._init_compartments

    def one(r, k, c):
        def emit(carry, _):
            nc, _ = step_fn(carry, None)
            return nc, ((nc[0], nc[3], nc[2]) if want_pos else (nc[3], nc[2]))
        _, out = jax.lax.scan(emit, (r, k, jnp.float32(0.0), c), None, length=n_t)
        return out

    out = jax.vmap(one)(r0, jax.random.split(wk, n_walkers), comp0)
    if want_pos:
        pos, cid, dlog = (np.asarray(o) for o in out)
    else:
        cid, dlog = (np.asarray(o) for o in out); pos = None
    lab = np.where(cid == 0, 2, np.where(cid > N_max, 1, 0))
    T2 = np.asarray(T2_per_comp, float)
    logw_t2 = -dt * np.cumsum((1.0 / T2)[lab], axis=1)
    D_by_lab = np.array([float(np.max(geometry._D_intra_jax)), 1.0,
                         float(np.max(geometry._D_extra_jax))])
    D_w = D_by_lab[lab[:, 0]][:, None]
    weight = np.exp(logw_t2 + (rho / D_w) * dlog)
    return dict(weight=weight, origin=lab[:, 0], pos=pos, G=G, dt=dt, n_t=n_t,
                rf_events=_rf_events_for(waveform), t_s=np.arange(n_t) * dt)


def magnitude_zoom_movie(geometry, waveform, save, *, rho, T2_per_comp, n_walkers=8000,
                         n_coarse=26, n_fine=600, stride=16, fps=20, dpi=95, title=None, seed=0):
    """Magnitude distribution at *real* relaxivity, revealed by a moving zoom.

    Three top panels + the player. **Left**: both pools' |M| histograms on the full 0-1 scale --
    at physiological ``rho`` this is a spike hugging the bulk-T2 ceiling with a moving crop box
    marking the tiny region below it. **Middle / right**: that crop box for intra / extra with
    thin bins, so the small real-``rho`` effect -- each pool sitting a little below its dashed
    bulk-T2 ceiling, intra (higher S/V) further than extra -- is visible *without exaggeration*.
    Packed-myelin substrate only.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation
    from scipy.stats import gaussian_kde

    w = _magnitude_walk(geometry, waveform, rho, T2_per_comp, n_walkers, seed)
    weight = w['weight']; origin = w['origin']; t_s = w['t_s']; t_ms = t_s * 1e3
    T2 = np.asarray(T2_per_comp, float)
    i_idx = np.where(origin == 0)[0]; e_idx = np.where(origin == 2)[0]
    frames = range(0, w['n_t'], max(1, int(stride)))

    cbins = np.linspace(0, 1, n_coarse + 1); cc = 0.5 * (cbins[:-1] + cbins[1:]); cw = cbins[1] - cbins[0]
    fbins = np.linspace(0, 1, n_fine + 1); fc = 0.5 * (fbins[:-1] + fbins[1:]); fw = fbins[1] - fbins[0]
    GREEN = _COMP_COLOR[0]; ORANGE = _COMP_COLOR[2]

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = GridSpec(2, 3, height_ratios=[2.4, 1.2], hspace=0.42, wspace=0.28,
                  top=0.86, bottom=0.13, figure=fig)
    ax_ov = fig.add_subplot(gs[0, 0]); ax_zi = fig.add_subplot(gs[0, 1]); ax_ze = fig.add_subplot(gs[0, 2])
    ax_play = fig.add_subplot(gs[1, :])
    playhead = _draw_player(ax_play, {'G': w['G'], 'dt_wf': w['dt'], 'rf_events': w['rf_events']})
    ax_play.set_title('pulse-program player', fontsize=9, loc='left', color='0.3')
    if title:
        fig.suptitle(title, fontsize=12, color='0.15', weight='bold')

    ax_ov.set_xlim(0, 1); ax_ov.set_ylim(0, 1)
    ax_ov.set_title('both pools — full |M| scale', fontsize=10, color='0.3')
    ax_ov.set_xlabel('|M|'); ax_ov.set_ylabel('fraction')
    b_ov_i = ax_ov.bar(cc, np.zeros_like(cc), width=cw, color=GREEN, alpha=0.55)
    b_ov_e = ax_ov.bar(cc, np.zeros_like(cc), width=cw, color=ORANGE, alpha=0.45)
    ln_ov_i = ax_ov.axvline(1, color=GREEN, ls='--', lw=1.2)
    ln_ov_e = ax_ov.axvline(1, color=ORANGE, ls='--', lw=1.2)
    crop_i = Rectangle((0.9, 0), 0.1, 1.0, fill=False, ec=GREEN, lw=1.4, zorder=6)
    crop_e = Rectangle((0.9, 0), 0.1, 1.0, fill=False, ec=ORANGE, lw=1.4, zorder=6)
    ax_ov.add_patch(crop_i); ax_ov.add_patch(crop_e)

    def _zoom_panel(ax, col, name):
        ax.set_title(f'{name} — zoom', fontsize=10, color=col)
        ax.set_xlabel('|M|'); ax.set_ylabel('density')
        bars = ax.bar(fc, np.zeros_like(fc), width=fw, color=col, alpha=0.35)
        kde_line, = ax.plot([], [], color=col, lw=2.2)          # smooth density (spread)
        cap = ax.axvline(1, color=col, ls='--', lw=1.5, label=r'bulk-$T_2$ ceiling')
        ax.legend(loc='upper left', fontsize=7, frameon=False)
        return bars, kde_line, cap

    bzi, kzi, czi = _zoom_panel(ax_zi, GREEN, 'intra-axonal')
    bze, kze, cze = _zoom_panel(ax_ze, ORANGE, 'extra-axonal')

    def draw(frame):
        wi = weight[i_idx, frame]; we = weight[e_idx, frame]
        ci = float(np.exp(-t_s[frame] / T2[0])); ce = float(np.exp(-t_s[frame] / T2[2]))
        for r, h in zip(b_ov_i, np.histogram(wi, bins=cbins)[0] / max(1, len(i_idx))):
            r.set_height(h)
        for r, h in zip(b_ov_e, np.histogram(we, bins=cbins)[0] / max(1, len(e_idx))):
            r.set_height(h)
        ln_ov_i.set_xdata([ci, ci]); ln_ov_e.set_xdata([ce, ce])

        # each zoom panel's window is anchored to ITS OWN bulk-T2 ceiling at a fixed on-screen
        # position (below / (below+above) = 0.9), so the ceiling stays put and only the
        # distribution drifts left as surface relaxivity pulls it below the ceiling.
        def _win(ceiling, below=0.12, ppos=0.9):
            return ceiling * (1 - below), ceiling * (1 + below * (1.0 / ppos - 1.0))
        xi0, xi1 = _win(ci); xe0, xe1 = _win(ce)
        crop_i.set_x(xi0); crop_i.set_width(xi1 - xi0)
        crop_e.set_x(xe0); crop_e.set_width(xe1 - xe0)

        for bars, kde_line, cap, ww, cl, (x0, x1) in (
                (bzi, kzi, czi, wi, ci, (xi0, xi1)), (bze, kze, cze, we, ce, (xe0, xe1))):
            hd = np.histogram(ww, bins=fbins, density=True)[0]           # density (bin backdrop)
            for r, hh in zip(bars, hd):
                r.set_height(hh)
            grid = np.linspace(x0, x1, 256)
            dens = gaussian_kde(ww)(grid) if ww.std() > 1e-6 else np.zeros_like(grid)
            kde_line.set_data(grid, dens)
            ax = bars[0].axes; ax.set_xlim(x0, x1)
            in_win = hd[(fc >= x0) & (fc <= x1)]
            ymax = max(float(in_win.max()) if in_win.size else 0.0, float(dens.max()))
            ax.set_ylim(0, max(1e-6, 1.2 * ymax))
            cap.set_xdata([cl, cl])
        playhead.set_xdata([t_ms[frame], t_ms[frame]])
        return [playhead]

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    writer = 'pillow' if str(save).endswith('.gif') else 'ffmpeg'
    anim.save(save, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return save


def magnitude_movie(geometry, waveform, save, *, rho, T2_per_comp, n_walkers=4000,
                    n_bins=28, stride=3, fps=20, dpi=90, title=None,
                    panels=('intra', 'extra'), seed=0):
    """Animate the per-compartment distribution of transverse-magnetisation MAGNITUDE ``|M|``.

    The magnitude counterpart of :func:`spin_movie` (which shows phase). Each walker's weight is
    ``|M_i|(t) = exp(-∫dt/T2[comp] - (rho/D)·ℓ_i(t))``, where ``ℓ_i`` is that walker's accumulated
    surface local time (wall contact) recorded by the packed-myelin walk. Without relaxivity
    every walker in a compartment shares the bulk-T2 magnitude (a spike); surface relaxivity
    gives each its own wall-contact, fanning the spike into a distribution capped at the bulk-T2
    value. Intra and extra fan differently (different ``rho·a/D``). Packed-myelin substrate only.

    ``T2_per_comp`` is ordered ``[T2_intra, T2_myelin, T2_extra]``; ``rho`` is the surface
    relaxivity (m/s). ``save`` is a .gif/.mp4 path.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    from .physics import make_packed_myelin_traj_step_fn

    G = np.asarray(waveform.G)
    G = G[0] if G.ndim == 3 else G
    dt = float(waveform.dt)
    n_t = G.shape[0]
    rf_events = _rf_events_for(waveform)

    # --- walk: per-walker (compartment_id, cumulative surface local time) over time ---
    step_fn = make_packed_myelin_traj_step_fn(geometry, dt)
    N_max = geometry.N_max
    pk, wk = jax.random.split(jax.random.PRNGKey(seed))
    r0 = geometry.init_positions(n_walkers, pk)
    comp0 = geometry._init_compartments

    def one(r, k, c):
        def emit(carry, _):
            nc, _ = step_fn(carry, None)
            return nc, (nc[3], nc[2])              # (compartment_id, cumulative dlog)
        _, (cid, dlog) = jax.lax.scan(emit, (r, k, jnp.float32(0.0), c), None, length=n_t)
        return cid, dlog

    cid, dlog = jax.vmap(one)(r0, jax.random.split(wk, n_walkers), comp0)
    cid = np.asarray(cid); dlog = np.asarray(dlog)           # (n_w, n_t)
    lab = np.where(cid == 0, 2, np.where(cid > N_max, 1, 0))  # 0=intra, 1=myelin, 2=extra

    # --- per-walker magnitude weight |M|(t) = exp(-T2 decay - (rho/D)·surface local time) ---
    T2 = np.asarray(T2_per_comp, float)
    logw_t2 = -dt * np.cumsum((1.0 / T2)[lab], axis=1)        # (n_w, n_t)
    D_by_lab = np.array([float(np.max(geometry._D_intra_jax)), 1.0,
                         float(np.max(geometry._D_extra_jax))])
    origin = lab[:, 0]
    D_w = D_by_lab[origin][:, None]
    weight = np.exp(logw_t2 + (rho / D_w) * dlog)             # dlog <= 0 -> weight in (0, 1]

    name2lab = {'intra': 0, 'myelin': 1, 'extra': 2}
    panel_labs = [name2lab[p] for p in panels]
    idx_by_panel = [np.where(origin == pl)[0] for pl in panel_labs]

    t_ms = np.arange(n_t) * dt * 1e3
    frames = range(0, n_t, max(1, int(stride)))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.9

    fig = plt.figure(figsize=(4.2 * len(panels) + 0.6, 6.2))
    gs = GridSpec(2, len(panels), height_ratios=[2.2, 1.4], hspace=0.38,
                  top=0.86, bottom=0.13, figure=fig)
    ax_play = fig.add_subplot(gs[1, :])
    playhead = _draw_player(ax_play, {'G': G, 'dt_wf': dt, 'rf_events': rf_events})
    ax_play.set_title('pulse-program player', fontsize=9, loc='left', color='0.3')
    if title:
        fig.suptitle(title, fontsize=12, color='0.15', weight='bold')

    t_s = np.arange(n_t) * dt
    bars = []
    for pi, (pl, idx) in enumerate(zip(panel_labs, idx_by_panel)):
        a = fig.add_subplot(gs[0, pi]); col = _COMP_COLOR.get(pl, '#777')
        a.set_xlim(0.0, 1.0); a.set_ylim(0.0, 1.0)
        a.set_xlabel('|M|  (transverse magnitude)'); a.set_ylabel('fraction of spins')
        a.set_title(_COMP_NAME.get(pl, panels[pi]), fontsize=10, color=col)
        h0 = np.histogram(weight[idx, 0], bins=bins)[0] / max(1, len(idx))
        b = a.bar(centers, h0, width=width, color=col, alpha=0.7)
        # bulk-T2 ceiling: |M| a zero-wall-contact spin would keep. Surface relaxivity only
        # subtracts, so the whole distribution stays at or below this line; the gap IS the
        # surface-relaxivity effect. Slides left at the compartment's bulk-T2 rate.
        cap = a.axvline(1.0, color='0.2', ls='--', lw=1.6, label=r'bulk-$T_2$ ceiling')
        a.legend(loc='upper left', fontsize=7, frameon=False)
        bars.append((b, idx, float(T2[pl]), cap))

    def draw(frame):
        for b, idx, t2c, cap in bars:
            h = np.histogram(weight[idx, frame], bins=bins)[0] / max(1, len(idx))
            for rect, hh in zip(b, h):
                rect.set_height(hh)
            xcap = float(np.exp(-t_s[frame] / t2c))
            cap.set_xdata([xcap, xcap])
        playhead.set_xdata([t_ms[frame], t_ms[frame]])
        return [playhead]

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    writer = 'pillow' if str(save).endswith('.gif') else 'ffmpeg'
    anim.save(save, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return save


def magnitude_spatial_movie(geometry, waveform, save, *, rho, T2_per_comp, n_walkers=4000,
                            n_show=3000, stride=16, fps=20, dpi=100, title=None, seed=0,
                            cmap='viridis'):
    """Spatial cross-section of the packed substrate with walkers coloured by ``|M|``.

    The companion to :func:`magnitude_movie`: same walk (use the same ``seed`` / substrate /
    waveform / ``rho``), but instead of the |M| histogram this shows *where* the low-magnitude
    spins are. Walkers hugging the axolemma accumulate surface local time and go dark first,
    while spins in the interior of the extra-axonal "holes" between fibres stay bright -- the
    spatial origin of the histogram fan. Cylinder inner/outer walls are drawn; a colourbar maps
    ``|M|``; the pulse-program player sweeps below. Packed-myelin substrate only.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    from .physics import make_packed_myelin_traj_step_fn

    G = np.asarray(waveform.G)
    G = G[0] if G.ndim == 3 else G
    dt = float(waveform.dt)
    n_t = G.shape[0]
    rf_events = _rf_events_for(waveform)

    step_fn = make_packed_myelin_traj_step_fn(geometry, dt)
    N_max = geometry.N_max
    pk, wk = jax.random.split(jax.random.PRNGKey(seed))
    r0 = geometry.init_positions(n_walkers, pk)
    comp0 = geometry._init_compartments

    def one(r, k, c):
        def emit(carry, _):
            nc, _ = step_fn(carry, None)
            return nc, (nc[0], nc[3], nc[2])       # (position, compartment_id, cumulative dlog)
        _, (pos, cid, dlog) = jax.lax.scan(emit, (r, k, jnp.float32(0.0), c), None, length=n_t)
        return pos, cid, dlog

    pos, cid, dlog = jax.vmap(one)(r0, jax.random.split(wk, n_walkers), comp0)
    pos = np.asarray(pos); cid = np.asarray(cid); dlog = np.asarray(dlog)
    lab = np.where(cid == 0, 2, np.where(cid > N_max, 1, 0))

    T2 = np.asarray(T2_per_comp, float)
    logw_t2 = -dt * np.cumsum((1.0 / T2)[lab], axis=1)
    D_by_lab = np.array([float(np.max(geometry._D_intra_jax)), 1.0,
                         float(np.max(geometry._D_extra_jax))])
    D_w = D_by_lab[lab[:, 0]][:, None]
    weight = np.exp(logw_t2 + (rho / D_w) * dlog)          # (n_w, n_t)

    sh = slice(0, min(n_show, n_walkers))
    xy = pos[sh, :, :2]                                     # (n_show, n_t, 2)
    w = weight[sh]
    t_ms = np.arange(n_t) * dt * 1e3
    frames = range(0, n_t, max(1, int(stride)))

    L = float(geometry._cell_size)
    centers = np.asarray(geometry._centers_jax)
    r_in = np.asarray(geometry._inner_radii_jax)
    r_out = np.asarray(geometry._outer_radii_jax)

    fig = plt.figure(figsize=(6.4, 7.4))
    gs = GridSpec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[24, 1], hspace=0.28,
                  wspace=0.06, top=0.9, bottom=0.11, figure=fig)
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    ax_play = fig.add_subplot(gs[1, :])
    playhead = _draw_player(ax_play, {'G': G, 'dt_wf': dt, 'rf_events': rf_events})
    ax_play.set_title('pulse-program player', fontsize=9, loc='left', color='0.3')

    ax.set_aspect('equal'); ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('walkers coloured by |M| — dark = surface-relaxed', fontsize=10, color='0.3')
    # cylinder walls (inner + outer), tiled over the 3x3 periodic neighbourhood
    for k in range(len(r_out)):
        if r_out[k] <= 0:
            continue
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                cx, cy = centers[k, 0] + ix * L, centers[k, 1] + iy * L
                ax.add_patch(plt.Circle((cx, cy), r_out[k], fill=False, color='0.6', lw=0.8))
                ax.add_patch(plt.Circle((cx, cy), r_in[k], fill=False, color='0.8', lw=0.7))
    scat = ax.scatter(xy[:, 0, 0], xy[:, 0, 1], c=w[:, 0], s=7, cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(scat, cax=cax, label='|M|  (transverse magnitude)')
    if title:
        fig.suptitle(title, fontsize=12, color='0.15', weight='bold')

    def draw(frame):
        scat.set_offsets(xy[:, frame, :])
        scat.set_array(w[:, frame])
        playhead.set_xdata([t_ms[frame], t_ms[frame]])
        return [scat, playhead]

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    writer = 'pillow' if str(save).endswith('.gif') else 'ffmpeg'
    anim.save(save, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return save

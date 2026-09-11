"""A circular white-matter phantom: one CACTUS replay pack, arranged in space (RPH.md).

The phantom is an annulus of tangentially oriented fibres around a free-water core, with inert background
outside. Every tissue voxel cites the *same* solved pack at its own pose, so the expensive object -- the walk --
is paid for once and read from every pose and every acquisition. Most of the ring is a tight cone of poses; one
sector fans out anisotropically (a Bingham with two different concentrations), which is what the SO(3)
composition is for and what an axially symmetric representation cannot hold.

Run it to build the ``.rph``, replay two sweeps and write the animation used in the README:

    python examples/rph/circular_wm_phantom.py <pack.rpk> [out_dir]

The sweeps are cached next to the phantom, so a rerun redraws the picture without recomputing them; delete
``circular_wm_sweeps.npz`` to force the replays.
"""
import sys
import numpy as np

from dmipy_sim import Encoding, RFEvent, ScannerSequence
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay import read_rpk, so3
from dmipy_sim.replay.phantom import (BinghamField, Grid, analytic_substrate, build_rph, inert_substrate,
                                      pack_substrate, read_rph)

N, R_IN, R_OUT = 40, 11.0, 17.0
KAPPA, KAPPA_FAN = (16.0, 16.0), (1.0, 40.0)     # a cone around the ring, and a fan in one sector
FAN_SECTOR = (100.0, 170.0)                      # degrees of the ring that fan out in the plane
LMAX, NMAX = 8, 4                                # the pose band, and how much azimuthal structure is kept
B_VALUE, B0_T = 1.5e9, 7.0                       # 1500 s/mm^2 at 7 T
CHI_ISO, CHI_ANISO = -1.0e-7, -1.0e-7            # myelin, Wharton & Bowtell 2012


# ------------------------------------------------------------------ the phantom
def annulus(n=N, r_in=R_IN, r_out=R_OUT, sub=6):
    """Volume fractions of tissue and core, supersampled so the two boundaries are genuine partial volume."""
    c = (n - 1) / 2.0
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off, off, indexing="ij")
    wm, csf = np.zeros((n, n, 1)), np.zeros((n, n, 1))
    for i in range(n):
        for j in range(n):
            rr = np.hypot(i + ox - c, j + oy - c)
            wm[i, j, 0] = np.mean((rr >= r_in) & (rr <= r_out))
            csf[i, j, 0] = np.mean(rr < r_in)
    return wm, csf


def frames(n=N):
    """A rotation per voxel, and a concentration pair per voxel (RPH.md 4, bingham mode).

    The third column of each frame is the fibre direction -- tangent to the circle, so the phantom spans every
    pose in the plane -- and the first is the direction the fan opens along, which is the tangent circle's own
    plane. Most of the ring is a cone of one width; one sector fans out anisotropically, which is a distribution
    no single dispersion parameter can express and no axially symmetric representation can compose.
    """
    c = (n - 1) / 2.0
    R = np.zeros((n, n, 1, 3, 3))
    kappa = np.zeros((n, n, 1, 2))
    for i in range(n):
        for j in range(n):
            dx, dy = i - c, j - c
            r = np.hypot(dx, dy)
            t = np.array([-dy / r, dx / r, 0.0]) if r > 1e-9 else np.array([1.0, 0.0, 0.0])
            e1 = np.array([dx / r, dy / r, 0.0]) if r > 1e-9 else np.array([0.0, 1.0, 0.0])  # radial: the fan plane
            R[i, j, 0] = np.stack([e1, np.cross(t, e1), t], axis=1)
            ang = np.degrees(np.arctan2(dy, dx)) % 360.0
            kappa[i, j, 0] = KAPPA_FAN if FAN_SECTOR[0] <= ang <= FAN_SECTOR[1] else KAPPA
    return R, kappa


def fan_density(kappa, dirs):
    """The axis density a bingham slot declares, over in-plane directions: what the picture draws."""
    k1, k2 = kappa
    return np.exp(-k1 * dirs[:, 0] ** 2 - k2 * dirs[:, 1] ** 2)


def build(pack_path, out, n=N):
    """Three substrates and two volumes -- the constructor derives the sparse file (RPH.md 3)."""
    wm, csf = annulus(n)
    return build_rph(
        out,
        grid=Grid((n, n, 1), (1.5e-3, 1.5e-3, 1.5e-3)),
        substrates=[pack_substrate("cactus/bundle_00000_capped", pack_path, m0=0.75),
                    analytic_substrate("csf/free-water", "free_water", {"diffusivity": 3.0e-9}),
                    inert_substrate()],
        occupancy={"cactus/bundle_00000_capped": wm, "csf/free-water": csf},
        remainder="background/inert",
        orientation=BinghamField(*frames(n)),
        id="phantoms/circular-wm/cactus-annulus", license="CC-BY-4.0",
        citation="Villarreal-Haro et al. 2023 (CACTUS substrate); dmipy-sim replay phantom",
        embed=False)


# ------------------------------------------------------------------ the acquisition
def pgse(pack, dirs, b=B_VALUE, delta=6.0e-3, Delta=1.5e-2, refocus=True):
    """A PGSE on the pack's save grid, one measurement per direction, refocused at the middle of the echo --
    or, with ``refocus=False``, the same diffusion weighting read as a gradient echo, which keeps the static
    field dephasing a spin echo would have refocused."""
    n_t, dt = pack.n_t, pack.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt)))
    # the PHYSICAL gradient: a spin echo plays two same-sign lobes and the 180 flips the second one; a gradient
    # echo has no 180 and plays the bipolar pair itself
    for i, g in enumerate(dirs):
        g = np.asarray(g, float) / np.linalg.norm(g)
        G[i, :nd] = amp * g
        G[i, ng:ng + nd] = (amp if refocus else -amp) * g
    rf = [RFEvent(0.0, 90, "Mz→Mxy", axis_deg=0.0)]
    if refocus:
        rf.append(RFEvent((n_t - 1) * dt / 2.0, 180, "refocus", axis_deg=90.0))
    seq = ScannerSequence(G=G, dt=dt, rf=rf, family="pgse" if refocus else "gre",
                          encoding=Encoding(bvalues=np.full(len(dirs), float(b)),
                                            gradient_directions=np.asarray(dirs, float)))
    return seq, amp


def sweeps(ph, pack_path, n_frames=36):
    """The two sweeps of the animation.

    * **g turns, B0 fixed north.** The classic diffusion weighting: a voxel is dark where the gradient runs
      along its fibres and bright where it runs across them, so the contrast is a two-lobed pattern that
      rotates with ``g``. One replay, ``n_frames`` measurements.
    * **B0 turns, g fixed.** The same diffusion gradient, held **through the plane** so it is perpendicular to
      every fibre in it: the diffusion weighting is then the same in every voxel and every frame, and the only
      thing left moving is the susceptibility. The refocusing pulse is dropped for this sweep, which is what
      makes the effect a large one -- a spin echo refocuses the static myelin field and leaves only the part a
      walker diffuses through, worth a few tenths of a percent here, while a gradient echo keeps the static
      dephasing, which at 7 T over this echo is tens of percent and depends strongly on the angle between the
      field and the fibre. One replay per field direction, because ``B0`` is one direction per replay.

      Through-plane also keeps ``g`` away from ``B0``: where the two are parallel the two-axis expansion's
      chiral sector is degenerate (its ``g x B0`` axis is undefined) and a frame there is fit in a basis one
      sector smaller than its neighbours, which is worth more than the effect being measured (issue #155).
    """
    ang = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    pack = read_rpk(pack_path)
    kw = dict(packs={"cactus/bundle_00000_capped": pack}, B0=B0_T, chi_iso=CHI_ISO, chi_aniso=CHI_ANISO,
              lmax=LMAX, nmax=NMAX)
    seq, _ = pgse(pack, dirs)
    _, S_g = ph.replay(seq, b0_dir=(0.0, 1.0, 0.0), **kw)                   # B0 north, g turning
    seq1, _ = pgse(pack, [[0.0, 0.0, 1.0]], refocus=False)                  # through the plane, across every fibre
    S_b = np.stack([ph.replay(seq1, **kw, b0_dir=d)[1][:, 0] for d in dirs], axis=1)
    return ang, dirs, S_g, S_b


# ------------------------------------------------------------------ the picture
def _pools(pack):
    """The pool each walker started in: what colours the substrate picture."""
    from dmipy_sim.replay.compression import decode_occupancy
    ch = pack.meta["compression"]["channels"]
    return np.asarray(decode_occupancy(pack.arrays, ch["compartment"])["comp"])[:, 0]


def figure(ph, pack_path, ang, dirs, S_g, S_b, out_gif, fps=10, slab_um=1.2):
    """Substrate, FODs, the signal map, and the signal of two voxels as each sweep advances.

    The gradient sweep is a factor-of-twenty contrast and is drawn as ``|S|``. The field sweep is a two percent
    one -- integrating over the azimuth each slot leaves unstated washes out most of the frame-specific field,
    which is the honest answer for tissue with no preferred azimuth -- so those frames are drawn as each voxel's
    departure from its own mean over the sweep: the pattern is the point, and the colour scale says how small it
    is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pack = read_rpk(pack_path)
    n = ph.grid.shape[0]
    fig = plt.figure(figsize=(11.0, 4.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1.25, 1.15], wspace=0.45, hspace=0.45,
                          left=0.055, right=0.95, top=0.84, bottom=0.13)
    ax_sub, ax_fod = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_map, ax_cur = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])

    # the substrate: where the walkers start, in a thin slab so the strands read as cross-sections
    p0 = pack.positions()[:, 0, :] * 1e6
    pool = _pools(pack)
    keep = np.abs(p0[:, 2] - np.median(p0[:, 2])) < slab_um
    for pid in (2, 1, 0):                                                 # extra first, so the strands sit on top
        col, lab = [("#2d6cdf", "intra"), ("#d9534f", "myelin"), ("#e8e8e8", "extra")][pid]
        m = keep & (pool == pid)
        if m.any():
            ax_sub.scatter(p0[m, 0], p0[m, 1], s=2.2, c=col, lw=0, label=f"{lab} {(pool == pid).mean():.0%}")
    ax_sub.set_aspect("equal"); ax_sub.set_xticks([]); ax_sub.set_yticks([])
    ax_sub.set_title("CACTUS substrate, one walk", fontsize=8)
    ax_sub.legend(fontsize=5.5, loc="upper right", framealpha=0.9, handletextpad=0.1, borderpad=0.2,
                  labelspacing=0.2, markerscale=2.5)

    # the orientation distributions the phantom replays, drawn where they sit
    circ = np.linspace(0, 2 * np.pi, 73)
    d_sh = np.stack([np.cos(circ), np.sin(circ), np.zeros_like(circ)], axis=1)
    step = max(1, n // 13)
    f_wm = ph.fraction("cactus/bundle_00000_capped")
    for v in range(ph.n_voxels):
        i, j = ph.voxel_index[v, :2]
        if i % step or j % step or f_wm[v] < 0.5:
            continue
        R = so3.rotations_from_quaternions(ph.pose_quat[v, 0])[0]
        r = fan_density(ph.bingham_kappa[v, 0], d_sh @ R)                 # the density, in the frame's own axes
        r = r / max(r.max(), 1e-12)
        r = 0.62 * step * (0.25 + 0.75 * r)                               # a floor, so a tight cone is visible
        fan = tuple(ph.bingham_kappa[v, 0]) != KAPPA
        ax_fod.fill(i + r * d_sh[:, 0], j + r * d_sh[:, 1], color="#b3452c" if fan else "#3b3b6d", lw=0)
    ax_fod.set_xlim(-1, n); ax_fod.set_ylim(-1, n); ax_fod.set_aspect("equal")
    ax_fod.set_xticks([]); ax_fod.set_yticks([])
    ax_fod.set_title(f"replayed poses: a {KAPPA[0]:.0f}/{KAPPA[1]:.0f} cone, and a "
                     f"{KAPPA_FAN[0]:.0f}/{KAPPA_FAN[1]:.0f} fan", fontsize=8)

    def vol(s):
        return ph.to_volume(np.asarray(s))[:, :, 0].T                     # (j, i) for imshow

    S_g, S_b = np.abs(S_g), np.abs(S_b)
    mean_b = S_b.mean(axis=1, keepdims=True)
    pct_b = 100.0 * (S_b - mean_b) / np.where(mean_b > 0, mean_b, 1.0)
    lim_b = float(np.percentile(np.abs(pct_b[f_wm > 0.5]), 98))
    im = ax_map.imshow(vol(S_g[:, 0]), origin="lower", cmap="magma", vmin=0.0, vmax=float(S_g.max()),
                       interpolation="nearest")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    cb = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("|S|", fontsize=7)
    c = (n - 1) / 2.0
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=2.2, foreground="#222222")]
    q_g = ax_map.annotate("", xy=(c, c), xytext=(c, c),
                          arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8, shrinkA=0, shrinkB=0))
    q_b = ax_map.annotate("", xy=(c, c), xytext=(c, c),
                          arrowprops=dict(arrowstyle="-|>", color="#33d17a", lw=1.8, shrinkA=0, shrinkB=0))
    ax_map.text(0.03, 0.97, "g", color="white", fontsize=9, transform=ax_map.transAxes, va="top",
                path_effects=stroke)
    (g_dot,) = ax_map.plot([c], [c], marker="$\\odot$", ms=11, color="white", visible=False)
    ax_map.text(0.10, 0.97, "B$_0$", color="#33d17a", fontsize=9, transform=ax_map.transAxes, va="top",
                path_effects=stroke)

    # two voxels on the ring, a quarter turn apart: their signal as the sweep advances
    ring = np.flatnonzero(f_wm > 0.98)
    xy = ph.voxel_index[ring, :2] - c
    a_ring = np.arctan2(xy[:, 1], xy[:, 0])
    v1 = int(ring[np.argmin(np.abs(a_ring))])                             # 3 o'clock: fibres run vertically
    v2 = int(ring[np.argmin(np.abs(a_ring - np.pi / 2))])                 # 12 o'clock: fibres run horizontally
    deg = np.degrees(ang)
    lines = [ax_cur.plot([], [], lw=1.7, color=col, label=lab)[0]
             for col, lab in [("#2d6cdf", "3 o'clock voxel"), ("#d9534f", "12 o'clock voxel")]]
    ax_cur.set_xlim(0, 360); ax_cur.set_xlabel("sweep angle [deg]", fontsize=8)
    ax_cur.tick_params(labelsize=7)
    ax_cur.legend(fontsize=7, loc="lower right")
    ttl = fig.suptitle("", fontsize=10)
    nf = len(ang)

    def draw(k):
        phase, k = divmod(k, nf)
        turning_g = phase == 0
        g = dirs[k] if turning_g else np.array([0.0, 0.0, 1.0])
        b = np.array([0.0, 1.0, 0.0]) if turning_g else dirs[k]
        if turning_g:
            im.set_data(vol(S_g[:, k])); im.set_cmap("magma"); im.set_clim(0.0, float(S_g.max()))
            cb.set_label("|S|", fontsize=7)
            ttl.set_text("g turns 360$^\\circ$, B$_0$ north: the diffusion contrast, dark along the fibres")
            curve, ylab = S_g, "|S|"
            mod, unit = 100.0 * np.ptp(S_g[[v1, v2]], axis=1) / S_g[[v1, v2]].mean(axis=1), "%"
        else:
            im.set_data(vol(pct_b[:, k])); im.set_cmap("coolwarm"); im.set_clim(-lim_b, lim_b)
            cb.set_label("|S| departure from its sweep mean [%]", fontsize=7)
            ttl.set_text("B$_0$ turns 360$^\\circ$, g fixed through the plane: the susceptibility contrast")
            curve, ylab = pct_b, "|S| departure [%]"
            mod, unit = np.ptp(pct_b[[v1, v2]], axis=1), " points"
        L_g, L_b = 0.31 * n, 0.22 * n
        g_dot.set_visible(abs(g[2]) > 0.9)                                 # g through the plane: a dot, not an arrow
        q_g.set_position((c - L_g * g[0], c - L_g * g[1])); q_g.xy = (c + L_g * g[0], c + L_g * g[1])
        q_b.set_position((c - L_b * b[0], c - L_b * b[1])); q_b.xy = (c + L_b * b[0], c + L_b * b[1])
        for line, v in zip(lines, (v1, v2)):
            line.set_data(deg[:k + 1], curve[v, :k + 1])
        y = curve[[v1, v2]]
        pad = max(0.05 * (y.max() - y.min()), 1e-5)
        ax_cur.set_ylim(y.min() - pad, y.max() + pad)
        ax_cur.set_ylabel(ylab, fontsize=8)
        # what the two curves trace: the modulation each of those voxels goes through over the sweep
        ax_cur.set_title(f"sweep modulation {mod.min():.1f}-{mod.max():.1f}{unit}", fontsize=8)
        return [im, q_g, q_b, g_dot, *lines, ttl]

    anim = FuncAnimation(fig, draw, frames=2 * nf, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=92)
    plt.close(fig)
    return out_gif


if __name__ == "__main__":
    import os, time
    pack_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    rph = os.path.join(out_dir, "circular_wm.rph")
    t0 = time.time()
    meta = build(pack_path, rph)
    ph = read_rph(rph)
    print(f"{ph!r}\n  {os.path.getsize(rph) / 1e3:.0f} kB citing a "
          f"{os.path.getsize(pack_path) / 1e6:.0f} MB pack  [{time.time() - t0:.0f}s]", flush=True)
    cache = os.path.join(out_dir, "circular_wm_sweeps.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        ang, dirs, S_g, S_b = z["ang"], z["dirs"], z["S_g"], z["S_b"]
        print(f"  sweeps read from {os.path.basename(cache)}", flush=True)
    else:
        ang, dirs, S_g, S_b = sweeps(ph, pack_path)
        np.savez_compressed(cache, ang=ang, dirs=dirs, S_g=S_g, S_b=S_b)
    live = ph.fraction("cactus/bundle_00000_capped") > 0.5
    def modulation(S):                                                  # peak-to-peak over the sweep, per voxel
        a = np.abs(S)[live]
        return 100.0 * np.ptp(a, axis=1) / a.mean(axis=1)
    print(f"  g sweep: |S| in [{np.abs(S_g).min():.3f}, {np.abs(S_g).max():.3f}], modulation "
          f"{np.median(modulation(S_g)):.0f}% median; B0 sweep modulation {np.median(modulation(S_b)):.1f}% "
          f"median, {modulation(S_b).max():.1f}% max  [{time.time() - t0:.0f}s]", flush=True)
    gif = figure(ph, pack_path, ang, dirs, S_g, S_b, os.path.join(out_dir, "circular_wm.gif"))
    print(f"  wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)  [{time.time() - t0:.0f}s]", flush=True)

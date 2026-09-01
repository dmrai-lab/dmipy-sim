"""Render the circular white-matter phantom: what fills each voxel, and what it replays to.

The physics checks this is for: the bright lobes must sit where the tangential fibre is
PERPENDICULAR to the gradient and rotate with it, the free-water core must be flat at exp(-bD)
and identical between gradient directions, and the ragged edge must be dim because those voxels
are mostly inert.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from circular_wm_phantom import (circular_wm, write_rph, replay_rph, amplitude_for_b,
                                 PACK, N, R_IN, R_OUT)
from dmipy_sim.replay import read_rpk
from dmipy_sim.sh_convolution import PackResponder

vi, sid, frac, sh = circular_wm()
pk = read_rpk(PACK)
pm = pk.meta['compression']['channels']['susceptibility_path']
n_t, dt = int(pm['n_t']), pk.dt; t = np.arange(n_t)*dt; T = n_t*dt
prof = ((t < 0.2*T).astype(float) - ((t>=0.5*T)&(t<0.7*T)).astype(float))
b0 = np.array([0,0,1.0])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_tmp.rph')
write_rph(out, PACK, "winther/g6/axon06", (vi,sid,frac,sh), m0=0.70, embed=False)
BV = 1e9
amp = amplitude_for_b(prof, dt, BV)
R = PackResponder(pk, prof, b0, amplitude=amp, B0=3.0, chi_iso=-9.4e-6,
                  chi_aniso=-1.0e-7, refocus_time=0.5*T, n_theta=32, n_phi=64)

def grid(vals, fill=np.nan):
    g = np.full((N,N), fill, float)
    g[vi[:,1], vi[:,0]] = vals            # (row=j, col=i)
    return g

INK, MUT, RULE = "#0b0b0b", "#52514e", "#dcdbd7"
fig, ax = plt.subplots(1, 4, figsize=(11.6, 3.3),
                       gridspec_kw=dict(left=.03, right=.97, top=.86, bottom=.06, wspace=.28))
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal')
    for sp in a.spines.values(): sp.set_visible(False)

# A: what occupies each voxel -- categorical, WM / free water / inert
comp = np.full((N,N,3), 1.0)
fw, fc, fi = grid(frac[:,0],0), grid(frac[:,1],0), grid(frac[:,2],0)
comp[...,0] = 1 - 0.10*fc - 0.75*fw
comp[...,1] = 1 - 0.55*fc - 0.35*fw
comp[...,2] = 1 - 0.80*fc - 0.05*fw
ax[0].imshow(comp, origin='lower')
ax[0].set_title("substrates", fontsize=9, color=INK, pad=6)
ax[0].text(.5,-.06,"white matter (blue) · free water (orange) · inert (white)",
           transform=ax[0].transAxes, ha='center', fontsize=6.6, color=MUT)

# B: fibre orientation, tangential
c=(N-1)/2; keep = frac[:,0] > 0.5
xs, ys = vi[keep,0], vi[keep,1]
dx, dy = -(ys-c), (xs-c); nrm = np.hypot(dx,dy); nrm[nrm==0]=1
ax[1].quiver(xs, ys, dx/nrm, dy/nrm, color="#2a78d6", pivot='mid',
             headwidth=0, headlength=0, headaxislength=0, width=.006, scale=34)
ax[1].set_xlim(-1,N); ax[1].set_ylim(-1,N)
ax[1].set_title("fibre orientation", fontsize=9, color=INK, pad=6)
ax[1].text(.5,-.06,"tangential, as in the studio phantom", transform=ax[1].transAxes,
           ha='center', fontsize=6.6, color=MUT)

# C,D: replayed signal, sequential ramp
for k,(nm,g) in enumerate((("g ∥ x",[1,0,0.]), ("g ∥ y",[0,1,0.]))):
    _, S, _, _ = replay_rph(out, R, np.asarray(g,float), b0, BV)
    im = ax[2+k].imshow(grid(np.abs(S)), origin='lower', cmap='magma')
    ax[2+k].set_title(f"|S|,  {nm},  b=1000", fontsize=9, color=INK, pad=6)
    cb = fig.colorbar(im, ax=ax[2+k], fraction=.045, pad=.02)
    cb.ax.tick_params(labelsize=6.5, length=2, colors=MUT); cb.outline.set_visible(False)
    ax[2+k].text(.5,-.06,"dark = fibre along the gradient", transform=ax[2+k].transAxes,
                 ha='center', fontsize=6.6, color=MUT)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'circular_wm.png')
fig.savefig(p, dpi=155, facecolor='white'); print("wrote", p)

"""Prototype: what the Lambda build costs once the shared work is hoisted out.

The 8.2 s per measurement is NOT arithmetic-bound -- the complex exponential is only 17% of
it. The rest is memory traffic from (n_dirs x n_walkers) temporaries, ~73 MB each in float64,
several per chunk.

Three things are direction-independent and are currently rebuilt for every measurement:

  q_w      depends on the gradient PROFILE, not its direction
  Psi_w    depends on the refocusing gate, not the direction
  Q(R^T B0) and hence the whole susceptibility phase depend only on B0 and the quadrature,
           so at fixed B0 they are identical for every gradient direction in a shell

Hoisting those and carrying the phase in float32 gives 8.20 s -> 1.93 s per measurement
(4.3x), with a 1.87 s one-off per shell. This is a prototype, not wired into
pack_response, which still rebuilds everything per call.

What remains is a pure elementwise map + reduction over (directions x walkers) -- the shape
that XLA fuses and a GPU vmaps. dmipy_sim.replay.replay_signal_jax is the existing precedent
in this repo. Not demonstrated here: jaxlib on this machine is CPU-only.
"""
import time
import numpy as np
from scipy.fft import dct
from dmipy_sim.replay import read_rpk
from dmipy_sim.gaunt import sphere_quadrature
from dmipy_sim.sh_convolution import _rotations_from_axis, _se_gate_local
from dmipy_sim.compression import read_position_coeffs
from dmipy_sim.constants import GAMMA
pk = read_rpk('/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk')
pm = pk.meta['compression']['channels']['susceptibility_path']
n_t, dt = int(pm['n_t']), pk.dt; t = np.arange(n_t)*dt; T = n_t*dt
prof = ((t < 0.2*T).astype(float) - ((t >= 0.5*T) & (t < 0.7*T)).astype(float))
b = np.array([0,0,1.0]); dirs, w = sphere_quadrature(32,64)

# --- shared, direction-independent (currently rebuilt per measurement) -------------
C = read_position_coeffs(pk.arrays, dtype=np.float64); K = C.shape[1]
q = (GAMMA*dt*0.05)*np.einsum('k,wkd->wd', dct(prof,type=2,norm='ortho')[:K], C)
Cs, _ = __import__('dmipy_sim.bank', fromlist=['x']).susc_path_coeffs(pk.arrays, pm)
names = list(pm['channels']); i_loc,i_xx,i_yy = (names.index(s) for s in ('iso_local','iso_P_xx','iso_P_yy'))
zz = 3*Cs[:,i_loc]-Cs[:,i_xx]-Cs[:,i_yy]; at = names.index('iso_P_xy')
Cs = np.insert(Cs, at, zz, axis=1); names = names[:at]+['iso_P_zz']+names[at:]
gate = dct(_se_gate_local(n_t,dt,0.5*T), type=2, norm='ortho')[:Cs.shape[2]]
Psi = (GAMMA*dt)*np.einsum('k,wck->wc', gate, Cs)
ip = names.index('iso_P_xx'); ia = names.index('aniso_G_xx')
R = _rotations_from_axis([0,0,1.], dirs); bp = np.einsum('nji,j->ni', R, b)
Q = np.stack([bp[:,0]**2,bp[:,1]**2,bp[:,2]**2,2*bp[:,0]*bp[:,1],2*bp[:,0]*bp[:,2],2*bp[:,1]*bp[:,2]],1)
t0=time.time()
phi_chi = (-9.4e-6*3.0)*(Psi[:,i_loc][None,:] - Q@Psi[:,ip:ip+6].T) \
        + (-1.0e-7*3.0)*(Q@Psi[:,ia:ia+6].T)                       # (n_dirs, n_w) ONCE
Echi = np.exp(1j*phi_chi.astype(np.float32))                        # shared factor
t_shared = time.time()-t0
ew = np.asarray(pk.arrays.get('spin_weights', np.ones(q.shape[0])), np.float64); norm = ew.sum()

def one_measurement(gv, chunk=512):
    out = np.empty(dirs.shape[0], np.complex128)
    gp = np.einsum('nji,j->ni', R, gv)
    for s in range(0, dirs.shape[0], chunk):
        pg = (gp[s:s+chunk] @ q.T).astype(np.float32)
        out[s:s+chunk] = ((ew[None,:]*Echi[s:s+chunk])*np.exp(1j*pg)).sum(1)/norm
    return out

angs = np.linspace(0.2, 1.4, 12)
t0=time.time(); _=[one_measurement(np.array([np.sin(a),0,np.cos(a)])) for a in angs]
t_batch = (time.time()-t0)/len(angs)
print(f"shared susceptibility factor (once per shell) : {t_shared:6.2f} s")
print(f"per-measurement after hoisting                : {t_batch:6.2f} s")
print(f"current per-measurement (measured earlier)    :   8.20 s")
print(f"\nspeedup per measurement: {8.20/t_batch:.1f}x   (60-dir shell: "
      f"{(t_shared+60*t_batch)/60:.2f} s/meas amortised vs 8.20)")

import time
import numpy as np
from dmipy_sim.replay import read_rpk
from dmipy_sim.gaunt import sphere_quadrature, real_sh
from dmipy_sim.sh_convolution import (pack_response, coupled_spectrum_at,
                                      apply_odf_coupled, watson_odf_sh)
PACK = '/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk'
pk = read_rpk(PACK)
pm = pk.meta['compression']['channels']['susceptibility_path']
n_t, dt = int(pm['n_t']), pk.dt; t = np.arange(n_t)*dt; T = n_t*dt
prof = ((t < 0.2*T).astype(float) - ((t >= 0.5*T) & (t < 0.7*T)).astype(float))
g = np.array([np.sin(0.96),0,np.cos(0.96)]); b = np.array([0,0,1.0])

t0=time.time(); resp = pack_response(pk, prof, g, b, amplitude=0.05, B0=3.0, chi_iso=-9.4e-6,
                                     chi_aniso=-1.0e-7, refocus_time=0.5*T, chunk=256)
t_setup = time.time()-t0
t0=time.time(); lam,resid,_ = coupled_spectrum_at(resp, g, b, l_g=8, l_b=6, n_theta=32, n_phi=64)
t_lam = time.time()-t0

# 200 voxels, each a differently-oriented Watson FOD
mus = np.random.default_rng(0).standard_normal((200,3)); mus/=np.linalg.norm(mus,axis=1,keepdims=True)
fods = [watson_odf_sh(6.0, mu=m, lmax=8) for m in mus]
t0=time.time(); vals=[apply_odf_coupled(lam,f,g,b,l_fod=8,l_g=8,l_b=6) for f in fods]
t_comp = (time.time()-t0)/len(fods)

# what a per-voxel sphere integral would cost instead
dirs,w = sphere_quadrature(32,64)
t0=time.time(); E = np.real(resp(dirs)); t_eval = time.time()-t0
t0=time.time(); _=[np.sum(w*(real_sh(8,dirs)@f)*E) for f in fods[:20]]; t_dir=(time.time()-t0)/20

print(f"pack setup (once, coefficient contraction) : {t_setup*1e3:8.1f} ms")
print(f"Lambda build (once per measurement)        : {t_lam*1e3:8.1f} ms   resid {resid:.1e}")
print(f"compose one voxel (Gaunt contraction)      : {t_comp*1e3:8.2f} ms")
print(f"  -- per-voxel sphere integral instead     : {(t_eval+t_dir)*1e3:8.1f} ms  (response eval + quad)")
print(f"\nspeedup per voxel: {(t_eval+t_dir)/t_comp:.0f}x")
for n in (1_000, 100_000):
    a = t_lam + n*t_comp; c = n*(t_eval+t_dir)
    print(f"  {n:>7} voxels: RPH {a:8.1f} s   vs per-voxel integral {c:9.1f} s")

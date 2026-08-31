"""Lambda build: hoisted vs un-hoisted, numpy vs accelerator.

Measured on an NVIDIA GH200 (winther_g6/axon06, 35909 walkers, n_t=1601, 2048 quadrature
directions, so 73.5M complex exponentials per measurement):

    un-hoisted (pack_response)          7264 ms/measurement
    PackResponder, numpy                1894 ms      3.8x
    PackResponder, jax on GH200            0.94 ms   7769x

  60-direction shell:  7.3 min  ->  114 s  ->  0.06 s

The numpy step comes from hoisting the direction-independent work (q_w, Psi_w, and the whole
susceptibility phase at fixed B0). The rest is the elementwise map and reduction over
(directions x walkers), which XLA fuses -- the un-hoisted version was ~83% memory traffic from
temporaries rather than arithmetic.

float32 on the accelerator costs 3.1e-5 max relative error in the response, which does not
reach the composed signal: it agrees with the numpy path to the printed precision, and sits
far below the 1.5e-2 projection residual.

Run with the GPU venv and its library path:
    export LD_LIBRARY_PATH="$(ls -d /home/rutger/dmipy-venv/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        python examples/rph/bench_lambda_backends.py numpy jax
"""
import time, sys
import numpy as np
from dmipy_sim.replay import read_rpk
from dmipy_sim.sh_convolution import PackResponder, pack_response, apply_odf_coupled, watson_odf_sh
from dmipy_sim.gaunt import sphere_quadrature, real_sh
pk = read_rpk('/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk')
pm = pk.meta['compression']['channels']['susceptibility_path']
n_t, dt = int(pm['n_t']), pk.dt; t = np.arange(n_t)*dt; T = n_t*dt
prof = ((t < 0.2*T).astype(float) - ((t >= 0.5*T) & (t < 0.7*T)).astype(float))
b = np.array([0,0,1.0]); g = np.array([np.sin(0.96),0,np.cos(0.96)])
kw = dict(amplitude=0.05, B0=3.0, chi_iso=-9.4e-6, chi_aniso=-1.0e-7, refocus_time=0.5*T)

# reference: the un-hoisted path
t0=time.time(); ref = pack_response(pk, prof, g, b, **kw, chunk=256)
dirs, w = sphere_quadrature(32,64); Eref = ref(dirs); t_old = time.time()-t0

res = {}
for backend in sys.argv[1:]:
    t0=time.time(); R = PackResponder(pk, prof, b, **kw, n_theta=32, n_phi=64, backend=backend)
    t_setup = time.time()-t0
    E = R.evaluate(g)                                   # warm up / compile
    angs = np.linspace(0.2, 1.4, 12)
    t0=time.time()
    for a in angs: R.evaluate(np.array([np.sin(a),0,np.cos(a)]))
    t_meas = (time.time()-t0)/len(angs)
    err = np.abs(E-Eref).max()/np.abs(Eref).max()
    print(f"{backend:>6}: setup {t_setup:6.2f} s   per-measurement {t_meas*1e3:9.2f} ms"
          f"   max rel err vs reference {err:.2e}")
    res[backend]=t_meas
print(f"\nun-hoisted reference per-measurement: {t_old*1e3:.0f} ms")
for k,v in res.items(): print(f"  {k:>6} speedup {t_old/v:8.1f}x   60-dir shell: {60*v:.2f} s")

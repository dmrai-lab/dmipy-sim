# Native permeable shell walk against the confirmed reference

Native permeable shell walk against the confirmed Bloch--Torrey reference (#48).

REGISTERED BEFORE ANY WALK (2026-11-11).

Prediction: both geometries, all three permeabilities and all final-pool signal
contributions agree with the intra-start reference within the following budget.
The pool signals are E[cos(phi) 1(final pool)], NOT conditional means or magnitudes;
the total is E[cos(phi)]. Imaginary means should be zero by radial symmetry.
The populations from the same paths are a zero-gradient exchange test: gradients
change only phase, never motion. All starts are the geometry's native placement.

R=5 um, R_out=8 um, D_in=D_out=2e-9 m^2/s; kappa=1e-5,1e-4,1e-3 m/s.
PGSE delta=5 ms, Delta=20 ms, gradient perpendicular to the cylinder. Three G
values are chosen BEFORE walking by the reference to give total E~0.8,0.4,0.1.
Physics dt=50,25,12.5 us (explicit sub_steps=1); N=262144 at EACH dt, with fresh,
nonzero, unrelated master seeds for every batch/run, distinct upper 16-bit seed
families. The three G values share paths, not independent walks; Bonferroni does
not assume their independence. The fast check uses N=65536 and only E~0.8.

Tolerance: |walk-ref| <= z_F SE + reference_error + B(dt), with empirical
SE=sd(per-walker REAL contribution, ddof=1)/sqrt(N). N=262144 guarantees
SE<=1/sqrt(N)=0.001954 (up to ddof); the fast low-G case must itself meet 0.002.
There are M=234 primary residuals (162 signal and 72 population), and also 162
imaginary-signal diagnostics, so z_F=norm.isf(0.01/(2*396)) (~4.21), a 1% family
error budget. The typical maximum is sqrt(2 ln 396)=3.46 SE, not 2 SE.

The CURRENT physics.py docstrings give a relative modal rate error c*lambda*dt/5,
with wall c=0.85 and membrane c=0.33, plus a one-sided-start transient bound
0.3*h^2/(D*t)=1.8*dt/t. Use the declared envelope lambda=(2.0815759778181)^2 D/R^2
for both geometries (the shell declares only min_feature). A rate perturbation
changes exp(-lambda*t) by at most its relative perturbation/e. Register the
absolute PGSE bias envelope B=(0.85+0.33)*lambda*dt/(5e)+1.8*dt/(Delta+delta).
For a population use the same modal envelope plus the transient relative error
on the exchanged fraction: B_p=(0.85+0.33)*lambda*dt/(5e)
+1.8*dt/t*(1-P_in). These are predictions to TEST, not a theorem extending the
slab calibration to curved membranes. They are not increased after observation.
Reference signals request <=2e-5 absolute error (reported separately), far below
MC SE. Exchange uses 160/320 intra cells, 96/192 extra cells, Richardson error.
Weighted fits E_walk-E_ref = intercept + slope*dt use all three independent dt
runs; slopes are per second, with standard errors, and no sign is assumed.

Prediction for placement: r^d/R^d is U(0,1), two-sided KS p>0.005 per geometry.
Also test six radius/direction features per geometry by Pearson correlation,
with 1% Bonferroni over 12. A passing finite sample cannot prove PRNG stream
independence. The inspected code reuses the parent key after splitting it;
this is a key-reuse hazard even if marginal and correlation tests pass.

Public API route: dmipy_sim.simulate(N,D,ScannerSequence,PermeableShell,...)
returns only the total real signal. dmipy_sim.walk(geometry,...,envelope=...)
returns CoefficientWalk.raw_positions/raw_compartment when raw_subset is given;
these permit per-pool phase sums, but retaining every step is expensive. Here
use the documented PUBLIC engine.physics.seed_walkers and make_step_fn, the
native scan body used by simulate, without reading any private attributes or
reimplementing motion/phase. Its carry contains phase and current compartment.
A separate fresh-seed simulate check verifies the ensemble entry point too.
No replay compression or geometry changes are involved.

RESOURCE AMENDMENT before the completed grid: the first 262144-walker coarse
run took 66.75 s. Extrapolation (six cases, cost ratio 1+2+4, ~1.8 CPU/wall)
slightly exceeds 90 CPU minutes. Reduce N to 163840 for EVERY completed-grid
resolution; cut no cases. The pilot's largest SE scales to 0.00174 at this N;
the test still REQUIRES measured SE<=0.002. The distribution-free guarantee
becomes 1/sqrt(N)=0.002471, while a cosine with sd~1/sqrt(2) has SE~0.001747.
No reference, target, tolerance, family count, or bias prediction changes.
The original timing data and the first KS outlier are retained in diagnostic
artifacts, not used to choose a physics expectation.
N=163,840 per full resolution; dt=50/25/12.5 us. z_F=4.21253; B_signal(dt)=102.094840 dt (dt in seconds).
Signals are real final-pool contributions, normalized by all starting walkers. Raw z=(walk-reference)/SE; the pass budget additionally includes reference error and registered step bias.
All three G share paths at each resolution; different resolutions and batches have fresh master seeds. Full seeds, imaginary diagnostics and fit intercepts are in the companion JSON.

## Signal comparison (all resolutions)

|scope / case|kappa m/s|G T/m|dt us|pool|reference ± error|walk ± SE|difference|z|budget|pass|
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
|full sphere|1e-05|0.186028|50|intra|0.717152 ± 6.9e-06|0.715881 ± 0.000840|-0.001271|-1.51|0.008651|True|
|full sphere|1e-05|0.186028|50|extra|0.083106 ± 2.6e-06|0.084067 ± 0.000635|+0.000961|+1.51|0.007784|True|
|full sphere|1e-05|0.186028|50|total|0.800258 ± 9.5e-06|0.799948 ± 0.000612|-0.000310|-0.51|0.007694|True|
|full sphere|1e-05|0.374167|50|intra|0.390932 ± 1.6e-05|0.387820 ± 0.001321|-0.003112|-2.36|0.010685|True|
|full sphere|1e-05|0.374167|50|extra|0.009599 ± 1.4e-06|0.010653 ± 0.000641|+0.001054|+1.64|0.007807|True|
|full sphere|1e-05|0.374167|50|total|0.400531 ± 1.7e-05|0.398473 ± 0.001451|-0.002058|-1.42|0.011234|True|
|full sphere|1e-05|0.576344|50|intra|0.106262 ± 8.1e-06|0.102231 ± 0.001611|-0.004032|-2.50|0.011901|True|
|full sphere|1e-05|0.576344|50|extra|-0.005922 ± 7.9e-07|-0.005510 ± 0.000642|+0.000411|+0.64|0.007809|True|
|full sphere|1e-05|0.576344|50|total|0.100340 ± 7.3e-06|0.096720 ± 0.001737|-0.003620|-2.08|0.012427|True|
|full sphere|1e-05|0.186028|25|intra|0.717152 ± 6.9e-06|0.716866 ± 0.000841|-0.000286|-0.34|0.006103|True|
|full sphere|1e-05|0.186028|25|extra|0.083106 ± 2.6e-06|0.084168 ± 0.000637|+0.001062|+1.67|0.005236|True|
|full sphere|1e-05|0.186028|25|total|0.800258 ± 9.5e-06|0.801034 ± 0.000613|+0.000776|+1.27|0.005145|True|
|full sphere|1e-05|0.374167|25|intra|0.390932 ± 1.6e-05|0.391449 ± 0.001319|+0.000517|+0.39|0.008126|True|
|full sphere|1e-05|0.374167|25|extra|0.009599 ± 1.4e-06|0.010847 ± 0.000640|+0.001248|+1.95|0.005248|True|
|full sphere|1e-05|0.374167|25|total|0.400531 ± 1.7e-05|0.402295 ± 0.001448|+0.001765|+1.22|0.008671|True|
|full sphere|1e-05|0.576344|25|intra|0.106262 ± 8.1e-06|0.107445 ± 0.001612|+0.001183|+0.73|0.009351|True|
|full sphere|1e-05|0.576344|25|extra|-0.005922 ± 7.9e-07|-0.005862 ± 0.000645|+0.000060|+0.09|0.005268|True|
|full sphere|1e-05|0.576344|25|total|0.100340 ± 7.3e-06|0.101584 ± 0.001738|+0.001243|+0.72|0.009882|True|
|full sphere|1e-05|0.186028|12.5|intra|0.717152 ± 6.9e-06|0.717560 ± 0.000838|+0.000407|+0.49|0.004814|True|
|full sphere|1e-05|0.186028|12.5|extra|0.083106 ± 2.6e-06|0.082262 ± 0.000630|-0.000843|-1.34|0.003932|True|
|full sphere|1e-05|0.186028|12.5|total|0.800258 ± 9.5e-06|0.799822 ± 0.000615|-0.000436|-0.71|0.003878|True|
|full sphere|1e-05|0.374167|12.5|intra|0.390932 ± 1.6e-05|0.389973 ± 0.001321|-0.000958|-0.73|0.006855|True|
|full sphere|1e-05|0.374167|12.5|extra|0.009599 ± 1.4e-06|0.009238 ± 0.000636|-0.000361|-0.57|0.003958|True|
|full sphere|1e-05|0.374167|12.5|total|0.400531 ± 1.7e-05|0.399212 ± 0.001451|-0.001319|-0.91|0.007405|True|
|full sphere|1e-05|0.576344|12.5|intra|0.106262 ± 8.1e-06|0.104625 ± 0.001612|-0.001638|-1.02|0.008074|True|
|full sphere|1e-05|0.576344|12.5|extra|-0.005922 ± 7.9e-07|-0.006234 ± 0.000641|-0.000312|-0.49|0.003975|True|
|full sphere|1e-05|0.576344|12.5|total|0.100340 ± 7.3e-06|0.098391 ± 0.001737|-0.001950|-1.12|0.008599|True|
|full cylinder|1e-05|0.162857|50|intra|0.746182 ± 5.1e-06|0.746403 ± 0.000760|+0.000221|+0.29|0.008310|True|
|full cylinder|1e-05|0.162857|50|extra|0.054020 ± 1.3e-06|0.053411 ± 0.000513|-0.000610|-1.19|0.007268|True|
|full cylinder|1e-05|0.162857|50|total|0.800202 ± 6.4e-06|0.799814 ± 0.000595|-0.000389|-0.65|0.007617|True|
|full cylinder|1e-05|0.323855|50|intra|0.398421 ± 1.0e-05|0.398589 ± 0.001349|+0.000168|+0.12|0.010798|True|
|full cylinder|1e-05|0.323855|50|extra|0.001993 ± 6.0e-07|0.001249 ± 0.000524|-0.000745|-1.42|0.007312|True|
|full cylinder|1e-05|0.323855|50|total|0.400414 ± 1.1e-05|0.399837 ± 0.001445|-0.000577|-0.40|0.011203|True|
|full cylinder|1e-05|0.488707|50|intra|0.107871 ± 4.1e-06|0.107742 ± 0.001668|-0.000129|-0.08|0.012135|True|
|full cylinder|1e-05|0.488707|50|extra|-0.007609 ± 6.1e-07|-0.007758 ± 0.000524|-0.000149|-0.29|0.007313|True|
|full cylinder|1e-05|0.488707|50|total|0.100262 ± 3.5e-06|0.099984 ± 0.001751|-0.000278|-0.16|0.012485|True|
|full cylinder|1e-05|0.162857|25|intra|0.746182 ± 5.1e-06|0.746786 ± 0.000760|+0.000604|+0.80|0.005758|True|
|full cylinder|1e-05|0.162857|25|extra|0.054020 ± 1.3e-06|0.054131 ± 0.000517|+0.000110|+0.21|0.004730|True|
|full cylinder|1e-05|0.162857|25|total|0.800202 ± 6.4e-06|0.800917 ± 0.000592|+0.000715|+1.21|0.005053|True|
|full cylinder|1e-05|0.323855|25|intra|0.398421 ± 1.0e-05|0.399743 ± 0.001349|+0.001322|+0.98|0.008245|True|
|full cylinder|1e-05|0.323855|25|extra|0.001993 ± 6.0e-07|0.002530 ± 0.000525|+0.000536|+1.02|0.004765|True|
|full cylinder|1e-05|0.323855|25|total|0.400414 ± 1.1e-05|0.402273 ± 0.001443|+0.001859|+1.29|0.008643|True|
|full cylinder|1e-05|0.488707|25|intra|0.107871 ± 4.1e-06|0.109414 ± 0.001667|+0.001543|+0.93|0.009577|True|
|full cylinder|1e-05|0.488707|25|extra|-0.007609 ± 6.1e-07|-0.006934 ± 0.000525|+0.000675|+1.28|0.004766|True|
|full cylinder|1e-05|0.488707|25|total|0.100262 ± 3.5e-06|0.102480 ± 0.001750|+0.002217|+1.27|0.009928|True|
|full cylinder|1e-05|0.162857|12.5|intra|0.746182 ± 5.1e-06|0.745460 ± 0.000762|-0.000722|-0.95|0.004491|True|
|full cylinder|1e-05|0.162857|12.5|extra|0.054020 ± 1.3e-06|0.054144 ± 0.000516|+0.000123|+0.24|0.003451|True|
|full cylinder|1e-05|0.162857|12.5|total|0.800202 ± 6.4e-06|0.799603 ± 0.000595|-0.000599|-1.01|0.003790|True|
|full cylinder|1e-05|0.323855|12.5|intra|0.398421 ± 1.0e-05|0.397938 ± 0.001347|-0.000483|-0.36|0.006961|True|
|full cylinder|1e-05|0.323855|12.5|extra|0.001993 ± 6.0e-07|0.001293 ± 0.000526|-0.000700|-1.33|0.003491|True|
|full cylinder|1e-05|0.323855|12.5|total|0.400414 ± 1.1e-05|0.399231 ± 0.001444|-0.001184|-0.82|0.007369|True|
|full cylinder|1e-05|0.488707|12.5|intra|0.107871 ± 4.1e-06|0.106979 ± 0.001663|-0.000892|-0.54|0.008285|True|
|full cylinder|1e-05|0.488707|12.5|extra|-0.007609 ± 6.1e-07|-0.008455 ± 0.000527|-0.000846|-1.61|0.003497|True|
|full cylinder|1e-05|0.488707|12.5|total|0.100262 ± 3.5e-06|0.098524 ± 0.001747|-0.001739|-0.99|0.008641|True|
|full sphere|1e-04|0.142677|50|intra|0.325466 ± 3.7e-06|0.324494 ± 0.001076|-0.000971|-0.90|0.009641|True|
|full sphere|1e-04|0.142677|50|extra|0.474700 ± 5.4e-06|0.475636 ± 0.001042|+0.000936|+0.90|0.009501|True|
|full sphere|1e-04|0.142677|50|total|0.800166 ± 1.7e-06|0.800131 ± 0.000600|-0.000035|-0.06|0.007633|True|
|full sphere|1e-04|0.284593|50|intra|0.223927 ± 3.0e-06|0.223451 ± 0.000974|-0.000476|-0.49|0.009210|True|
|full sphere|1e-04|0.284593|50|extra|0.176403 ± 1.6e-05|0.176881 ± 0.001287|+0.000478|+0.37|0.010540|True|
|full sphere|1e-04|0.284593|50|total|0.400331 ± 1.3e-05|0.400332 ± 0.001456|+0.000001|+0.00|0.011252|True|
|full sphere|1e-04|0.436969|50|intra|0.109836 ± 5.2e-06|0.110240 ± 0.001025|+0.000404|+0.39|0.009427|True|
|full sphere|1e-04|0.436969|50|extra|-0.009625 ± 2.4e-07|-0.009711 ± 0.001389|-0.000086|-0.06|0.010955|True|
|full sphere|1e-04|0.436969|50|total|0.100211 ± 5.5e-06|0.100529 ± 0.001730|+0.000318|+0.18|0.012396|True|
|full sphere|1e-04|0.142677|25|intra|0.325466 ± 3.7e-06|0.323663 ± 0.001075|-0.001803|-1.68|0.007083|True|
|full sphere|1e-04|0.142677|25|extra|0.474700 ± 5.4e-06|0.476151 ± 0.001042|+0.001450|+1.39|0.006947|True|
|full sphere|1e-04|0.142677|25|total|0.800166 ± 1.7e-06|0.799813 ± 0.000599|-0.000353|-0.59|0.005077|True|
|full sphere|1e-04|0.284593|25|intra|0.223927 ± 3.0e-06|0.222154 ± 0.000971|-0.001774|-1.83|0.006644|True|
|full sphere|1e-04|0.284593|25|extra|0.176403 ± 1.6e-05|0.176868 ± 0.001289|+0.000465|+0.36|0.007996|True|
|full sphere|1e-04|0.284593|25|total|0.400331 ± 1.3e-05|0.399022 ± 0.001457|-0.001309|-0.90|0.008702|True|
|full sphere|1e-04|0.436969|25|intra|0.109836 ± 5.2e-06|0.108170 ± 0.001023|-0.001666|-1.63|0.006868|True|
|full sphere|1e-04|0.436969|25|extra|-0.009625 ± 2.4e-07|-0.009759 ± 0.001389|-0.000134|-0.10|0.008402|True|
|full sphere|1e-04|0.436969|25|total|0.100211 ± 5.5e-06|0.098411 ± 0.001729|-0.001800|-1.04|0.009839|True|
|full sphere|1e-04|0.142677|12.5|intra|0.325466 ± 3.7e-06|0.324897 ± 0.001077|-0.000568|-0.53|0.005815|True|
|full sphere|1e-04|0.142677|12.5|extra|0.474700 ± 5.4e-06|0.474966 ± 0.001042|+0.000266|+0.25|0.005670|True|
|full sphere|1e-04|0.142677|12.5|total|0.800166 ± 1.7e-06|0.799863 ± 0.000600|-0.000303|-0.50|0.003807|True|
|full sphere|1e-04|0.284593|12.5|intra|0.223927 ± 3.0e-06|0.224227 ± 0.000973|+0.000299|+0.31|0.005377|True|
|full sphere|1e-04|0.284593|12.5|extra|0.176403 ± 1.6e-05|0.175507 ± 0.001286|-0.000896|-0.70|0.006709|True|
|full sphere|1e-04|0.284593|12.5|total|0.400331 ± 1.3e-05|0.399734 ± 0.001456|-0.000597|-0.41|0.007422|True|
|full sphere|1e-04|0.436969|12.5|intra|0.109836 ± 5.2e-06|0.110937 ± 0.001022|+0.001101|+1.08|0.005588|True|
|full sphere|1e-04|0.436969|12.5|extra|-0.009625 ± 2.4e-07|-0.011275 ± 0.001386|-0.001650|-1.19|0.007114|True|
|full sphere|1e-04|0.436969|12.5|total|0.100211 ± 5.5e-06|0.099662 ± 0.001726|-0.000549|-0.32|0.008554|True|
|full cylinder|1e-04|0.131050|50|intra|0.462868 ± 9.0e-06|0.461423 ± 0.001112|-0.001445|-1.30|0.009798|True|
|full cylinder|1e-04|0.131050|50|extra|0.337268 ± 1.4e-05|0.339230 ± 0.001013|+0.001962|+1.94|0.009387|True|
|full cylinder|1e-04|0.131050|50|total|0.800136 ± 5.3e-06|0.800653 ± 0.000593|+0.000517|+0.87|0.007609|True|
|full cylinder|1e-04|0.260761|50|intra|0.305781 ± 3.2e-06|0.304092 ± 0.001082|-0.001689|-1.56|0.009667|True|
|full cylinder|1e-04|0.260761|50|extra|0.094500 ± 6.9e-06|0.097296 ± 0.001135|+0.002796|+2.46|0.009892|True|
|full cylinder|1e-04|0.260761|50|total|0.400282 ± 1.0e-05|0.401388 ± 0.001448|+0.001106|+0.76|0.011216|True|
|full cylinder|1e-04|0.397590|50|intra|0.137258 ± 9.4e-06|0.135669 ± 0.001230|-0.001589|-1.29|0.010295|True|
|full cylinder|1e-04|0.397590|50|extra|-0.037074 ± 4.2e-06|-0.034931 ± 0.001188|+0.002143|+1.80|0.010112|True|
|full cylinder|1e-04|0.397590|50|total|0.100184 ± 5.2e-06|0.100739 ± 0.001727|+0.000554|+0.32|0.012383|True|
|full cylinder|1e-04|0.131050|25|intra|0.462868 ± 9.0e-06|0.464649 ± 0.001112|+0.001781|+1.60|0.007245|True|
|full cylinder|1e-04|0.131050|25|extra|0.337268 ± 1.4e-05|0.334985 ± 0.001009|-0.002282|-2.26|0.006819|True|
|full cylinder|1e-04|0.131050|25|total|0.800136 ± 5.3e-06|0.799635 ± 0.000596|-0.000501|-0.84|0.005067|True|
|full cylinder|1e-04|0.260761|25|intra|0.305781 ± 3.2e-06|0.306226 ± 0.001084|+0.000444|+0.41|0.007122|True|
|full cylinder|1e-04|0.260761|25|extra|0.094500 ± 6.9e-06|0.092844 ± 0.001133|-0.001656|-1.46|0.007332|True|
|full cylinder|1e-04|0.260761|25|total|0.400282 ± 1.0e-05|0.399070 ± 0.001453|-0.001212|-0.83|0.008683|True|
|full cylinder|1e-04|0.397590|25|intra|0.137258 ± 9.4e-06|0.136629 ± 0.001233|-0.000629|-0.51|0.007756|True|
|full cylinder|1e-04|0.397590|25|extra|-0.037074 ± 4.2e-06|-0.037209 ± 0.001182|-0.000135|-0.11|0.007537|True|
|full cylinder|1e-04|0.397590|25|total|0.100184 ± 5.2e-06|0.099421 ± 0.001726|-0.000764|-0.44|0.009830|True|
|full cylinder|1e-04|0.131050|12.5|intra|0.462868 ± 9.0e-06|0.465081 ± 0.001112|+0.002213|+1.99|0.005969|True|
|full cylinder|1e-04|0.131050|12.5|extra|0.337268 ± 1.4e-05|0.335711 ± 0.001010|-0.001557|-1.54|0.005544|True|
|full cylinder|1e-04|0.131050|12.5|total|0.800136 ± 5.3e-06|0.800791 ± 0.000591|+0.000656|+1.11|0.003773|True|
|full cylinder|1e-04|0.260761|12.5|intra|0.305781 ± 3.2e-06|0.307034 ± 0.001082|+0.001253|+1.16|0.005836|True|
|full cylinder|1e-04|0.260761|12.5|extra|0.094500 ± 6.9e-06|0.094075 ± 0.001131|-0.000426|-0.38|0.006048|True|
|full cylinder|1e-04|0.260761|12.5|total|0.400282 ± 1.0e-05|0.401109 ± 0.001448|+0.000827|+0.57|0.007387|True|
|full cylinder|1e-04|0.397590|12.5|intra|0.137258 ± 9.4e-06|0.136953 ± 0.001232|-0.000305|-0.25|0.006475|True|
|full cylinder|1e-04|0.397590|12.5|extra|-0.037074 ± 4.2e-06|-0.037143 ± 0.001182|-0.000069|-0.06|0.006262|True|
|full cylinder|1e-04|0.397590|12.5|total|0.100184 ± 5.2e-06|0.099810 ± 0.001726|-0.000374|-0.22|0.008551|True|
|full sphere|1e-03|0.125970|50|intra|0.205452 ± 1.2e-06|0.203640 ± 0.000923|-0.001812|-1.96|0.008993|True|
|full sphere|1e-03|0.125970|50|extra|0.594682 ± 2.8e-06|0.596370 ± 0.000988|+0.001688|+1.71|0.009268|True|
|full sphere|1e-03|0.125970|50|total|0.800134 ± 1.6e-06|0.800010 ± 0.000587|-0.000124|-0.21|0.007577|True|
|full sphere|1e-03|0.249610|50|intra|0.122189 ± 2.7e-06|0.121935 ± 0.000828|-0.000254|-0.31|0.008594|True|
|full sphere|1e-03|0.249610|50|extra|0.278080 ± 6.9e-06|0.278123 ± 0.001345|+0.000043|+0.03|0.010779|True|
|full sphere|1e-03|0.249610|50|total|0.400269 ± 4.2e-06|0.400059 ± 0.001443|-0.000211|-0.15|0.011186|True|
|full sphere|1e-03|0.375978|50|intra|0.047782 ± 2.5e-06|0.048622 ± 0.000849|+0.000840|+0.99|0.008682|True|
|full sphere|1e-03|0.375978|50|extra|0.052394 ± 7.7e-06|0.051531 ± 0.001516|-0.000863|-0.57|0.011497|True|
|full sphere|1e-03|0.375978|50|total|0.100176 ± 5.2e-06|0.100153 ± 0.001728|-0.000023|-0.01|0.012390|True|
|full sphere|1e-03|0.125970|25|intra|0.205452 ± 1.2e-06|0.206259 ± 0.000926|+0.000806|+0.87|0.006455|True|
|full sphere|1e-03|0.125970|25|extra|0.594682 ± 2.8e-06|0.592813 ± 0.000991|-0.001869|-1.89|0.006728|True|
|full sphere|1e-03|0.125970|25|total|0.800134 ± 1.6e-06|0.799072 ± 0.000589|-0.001063|-1.80|0.005034|True|
|full sphere|1e-03|0.249610|25|intra|0.122189 ± 2.7e-06|0.122804 ± 0.000831|+0.000615|+0.74|0.006055|True|
|full sphere|1e-03|0.249610|25|extra|0.278080 ± 6.9e-06|0.275107 ± 0.001347|-0.002973|-2.21|0.008232|True|
|full sphere|1e-03|0.249610|25|total|0.400269 ± 4.2e-06|0.397912 ± 0.001446|-0.002358|-1.63|0.008649|True|
|full sphere|1e-03|0.375978|25|intra|0.047782 ± 2.5e-06|0.047905 ± 0.000857|+0.000124|+0.14|0.006166|True|
|full sphere|1e-03|0.375978|25|extra|0.052394 ± 7.7e-06|0.050584 ± 0.001515|-0.001810|-1.19|0.008940|True|
|full sphere|1e-03|0.375978|25|total|0.100176 ± 5.2e-06|0.098490 ± 0.001732|-0.001686|-0.97|0.009853|True|
|full sphere|1e-03|0.125970|12.5|intra|0.205452 ± 1.2e-06|0.205514 ± 0.000925|+0.000062|+0.07|0.005173|True|
|full sphere|1e-03|0.125970|12.5|extra|0.594682 ± 2.8e-06|0.594751 ± 0.000989|+0.000068|+0.07|0.005447|True|
|full sphere|1e-03|0.125970|12.5|total|0.800134 ± 1.6e-06|0.800265 ± 0.000585|+0.000131|+0.22|0.003742|True|
|full sphere|1e-03|0.249610|12.5|intra|0.122189 ± 2.7e-06|0.122089 ± 0.000829|-0.000100|-0.12|0.004772|True|
|full sphere|1e-03|0.249610|12.5|extra|0.278080 ± 6.9e-06|0.278134 ± 0.001342|+0.000054|+0.04|0.006935|True|
|full sphere|1e-03|0.249610|12.5|total|0.400269 ± 4.2e-06|0.400223 ± 0.001440|-0.000046|-0.03|0.007346|True|
|full sphere|1e-03|0.375978|12.5|intra|0.047782 ± 2.5e-06|0.047333 ± 0.000854|-0.000449|-0.53|0.004878|True|
|full sphere|1e-03|0.375978|12.5|extra|0.052394 ± 7.7e-06|0.051787 ± 0.001511|-0.000607|-0.40|0.007649|True|
|full sphere|1e-03|0.375978|12.5|total|0.100176 ± 5.2e-06|0.099120 ± 0.001727|-0.001056|-0.61|0.008557|True|
|full cylinder|1e-03|0.115290|50|intra|0.330995 ± 1.9e-06|0.329679 ± 0.001061|-0.001316|-1.24|0.009577|True|
|full cylinder|1e-03|0.115290|50|extra|0.469115 ± 2.5e-06|0.469519 ± 0.001051|+0.000403|+0.38|0.009534|True|
|full cylinder|1e-03|0.115290|50|total|0.800110 ± 6.3e-07|0.799198 ± 0.000584|-0.000912|-1.56|0.007566|True|
|full cylinder|1e-03|0.227999|50|intra|0.200986 ± 3.0e-06|0.199981 ± 0.001004|-0.001004|-1.00|0.009338|True|
|full cylinder|1e-03|0.227999|50|extra|0.199241 ± 5.6e-06|0.198143 ± 0.001249|-0.001098|-0.88|0.010372|True|
|full cylinder|1e-03|0.227999|50|total|0.400227 ± 2.6e-06|0.398124 ± 0.001444|-0.002103|-1.46|0.011190|True|
|full cylinder|1e-03|0.341859|50|intra|0.082772 ± 1.9e-06|0.082119 ± 0.001074|-0.000653|-0.61|0.009632|True|
|full cylinder|1e-03|0.341859|50|extra|0.017378 ± 6.2e-06|0.016599 ± 0.001363|-0.000779|-0.57|0.010852|True|
|full cylinder|1e-03|0.341859|50|total|0.100150 ± 4.3e-06|0.098717 ± 0.001730|-0.001432|-0.83|0.012399|True|
|full cylinder|1e-03|0.115290|25|intra|0.330995 ± 1.9e-06|0.330875 ± 0.001062|-0.000120|-0.11|0.007028|True|
|full cylinder|1e-03|0.115290|25|extra|0.469115 ± 2.5e-06|0.468909 ± 0.001051|-0.000206|-0.20|0.006982|True|
|full cylinder|1e-03|0.115290|25|total|0.800110 ± 6.3e-07|0.799784 ± 0.000582|-0.000327|-0.56|0.005005|True|
|full cylinder|1e-03|0.227999|25|intra|0.200986 ± 3.0e-06|0.200820 ± 0.001005|-0.000166|-0.16|0.006790|True|
|full cylinder|1e-03|0.227999|25|extra|0.199241 ± 5.6e-06|0.198372 ± 0.001245|-0.000869|-0.70|0.007803|True|
|full cylinder|1e-03|0.227999|25|total|0.400227 ± 2.6e-06|0.399192 ± 0.001440|-0.001035|-0.72|0.008622|True|
|full cylinder|1e-03|0.341859|25|intra|0.082772 ± 1.9e-06|0.082497 ± 0.001074|-0.000275|-0.26|0.007078|True|
|full cylinder|1e-03|0.341859|25|extra|0.017378 ± 6.2e-06|0.016145 ± 0.001362|-0.001233|-0.91|0.008296|True|
|full cylinder|1e-03|0.341859|25|total|0.100150 ± 4.3e-06|0.098642 ± 0.001730|-0.001508|-0.87|0.009843|True|
|full cylinder|1e-03|0.115290|12.5|intra|0.330995 ± 1.9e-06|0.331029 ± 0.001063|+0.000034|+0.03|0.005756|True|
|full cylinder|1e-03|0.115290|12.5|extra|0.469115 ± 2.5e-06|0.469719 ± 0.001051|+0.000604|+0.57|0.005706|True|
|full cylinder|1e-03|0.115290|12.5|total|0.800110 ± 6.3e-07|0.800748 ± 0.000580|+0.000638|+1.10|0.003720|True|
|full cylinder|1e-03|0.227999|12.5|intra|0.200986 ± 3.0e-06|0.202079 ± 0.001005|+0.001093|+1.09|0.005514|True|
|full cylinder|1e-03|0.227999|12.5|extra|0.199241 ± 5.6e-06|0.199375 ± 0.001246|+0.000134|+0.11|0.006532|True|
|full cylinder|1e-03|0.227999|12.5|total|0.400227 ± 2.6e-06|0.401454 ± 0.001439|+0.001227|+0.85|0.007343|True|
|full cylinder|1e-03|0.341859|12.5|intra|0.082772 ± 1.9e-06|0.084175 ± 0.001075|+0.001403|+1.31|0.005805|True|
|full cylinder|1e-03|0.341859|12.5|extra|0.017378 ± 6.2e-06|0.017119 ± 0.001363|-0.000259|-0.19|0.007024|True|
|full cylinder|1e-03|0.341859|12.5|total|0.100150 ± 4.3e-06|0.101294 ± 0.001730|+0.001144|+0.66|0.008570|True|

## Fitted step-bias slopes

Weighted residual = intercept + slope dt; one fit degree of freedom.

|case|kappa|G T/m|pool|slope ± SE (1/s)|intercept ± SE|chi² (1 dof)|
|---|---:|---:|---|---:|---:|---:|
|sphere|1e-05|0.186028|intra|-44.01 ± 31.09|+0.000900 ± 0.001028|0.02|
|sphere|1e-05|0.186028|extra|+40.92 ± 23.45|-0.000804 ± 0.000774|2.71|
|sphere|1e-05|0.186028|total|-3.42 ± 22.72|+0.000110 ± 0.000753|2.34|
|sphere|1e-05|0.374167|intra|-69.97 ± 48.91|+0.000857 ± 0.001617|1.78|
|sphere|1e-05|0.374167|extra|+31.38 ± 23.67|-0.000270 ± 0.000781|2.04|
|sphere|1e-05|0.374167|total|-38.77 ± 53.73|+0.000596 ± 0.001776|3.39|
|sphere|1e-05|0.576344|intra|-84.53 ± 59.68|+0.000970 ± 0.001974|3.24|
|sphere|1e-05|0.576344|extra|+18.56 ± 23.75|-0.000488 ± 0.000786|0.03|
|sphere|1e-05|0.576344|total|-65.95 ± 64.31|+0.000480 ± 0.002127|2.99|
|cylinder|1e-05|0.162857|intra|+19.31 ± 28.17|-0.000528 ± 0.000932|1.14|
|cylinder|1e-05|0.162857|extra|-20.88 ± 19.05|+0.000483 ± 0.000632|0.13|
|cylinder|1e-05|0.162857|total|-1.54 ± 22.03|-0.000043 ± 0.000728|2.82|
|cylinder|1e-05|0.323855|intra|+8.33 ± 49.93|+0.000092 ± 0.001650|0.89|
|cylinder|1e-05|0.323855|extra|-8.39 ± 19.42|-0.000058 ± 0.000643|3.65|
|cylinder|1e-05|0.323855|total|-0.02 ± 53.49|+0.000034 ± 0.001768|2.49|
|cylinder|1e-05|0.488707|intra|+8.01 ± 61.69|-0.000061 ± 0.002038|1.10|
|cylinder|1e-05|0.488707|extra|+11.10 ± 19.45|-0.000430 ± 0.000645|3.86|
|cylinder|1e-05|0.488707|total|+19.25 ± 64.80|-0.000496 ± 0.002141|2.53|
|sphere|1e-04|0.142677|intra|-4.43 ± 39.85|-0.000986 ± 0.001318|0.67|
|sphere|1e-04|0.142677|extra|+12.39 ± 38.59|+0.000523 ± 0.001276|0.55|
|sphere|1e-04|0.142677|total|+7.94 ± 22.22|-0.000462 ± 0.000735|0.03|
|sphere|1e-04|0.284593|intra|-10.31 ± 36.05|-0.000352 ± 0.001191|2.24|
|sphere|1e-04|0.284593|extra|+31.49 ± 47.64|-0.000904 ± 0.001576|0.32|
|sphere|1e-04|0.284593|total|+21.15 ± 53.93|-0.001252 ± 0.001783|0.25|
|sphere|1e-04|0.436969|intra|-4.20 ± 37.91|+0.000069 ± 0.001253|3.95|
|sphere|1e-04|0.436969|extra|+36.06 ± 51.38|-0.001676 ± 0.001698|0.33|
|sphere|1e-04|0.436969|total|+31.86 ± 64.01|-0.001606 ± 0.002116|0.51|
|cylinder|1e-04|0.131050|intra|-102.05 ± 41.18|+0.003826 ± 0.001362|0.32|
|cylinder|1e-04|0.131050|extra|+104.62 ± 37.47|-0.003678 ± 0.001237|2.27|
|cylinder|1e-04|0.131050|total|+2.55 ± 21.95|+0.000153 ± 0.000726|2.24|
|cylinder|1e-04|0.260761|intra|-79.44 ± 40.07|+0.002319 ± 0.001325|0.02|
|cylinder|1e-04|0.260761|extra|+98.96 ± 41.97|-0.002648 ± 0.001387|2.66|
|cylinder|1e-04|0.260761|total|+19.58 ± 53.64|-0.000328 ± 0.001775|1.39|
|cylinder|1e-04|0.397590|intra|-34.83 ± 45.58|+0.000175 ± 0.001509|0.00|
|cylinder|1e-04|0.397590|extra|+63.55 ± 43.90|-0.001208 ± 0.001449|0.30|
|cylinder|1e-04|0.397590|total|+28.76 ± 63.93|-0.001033 ± 0.002114|0.11|
|sphere|1e-03|0.125970|intra|-57.84 ± 34.20|+0.001371 ± 0.001133|1.41|
|sphere|1e-03|0.125970|extra|+57.38 ± 36.60|-0.001709 ± 0.001212|4.02|
|sphere|1e-03|0.125970|total|-0.56 ± 21.70|-0.000333 ± 0.000717|2.29|
|sphere|1e-03|0.249610|intra|-8.49 ± 30.68|+0.000334 ± 0.001016|0.55|
|sphere|1e-03|0.249610|extra|+16.81 ± 49.77|-0.001445 ± 0.001645|3.25|
|sphere|1e-03|0.249610|total|+8.38 ± 53.39|-0.001112 ± 0.001765|1.57|
|sphere|1e-03|0.375978|intra|+33.57 ± 31.52|-0.000808 ± 0.001046|0.02|
|sphere|1e-03|0.375978|extra|-0.52 ± 56.06|-0.001077 ± 0.001852|0.35|
|sphere|1e-03|0.375978|total|+33.08 ± 63.99|-0.001886 ± 0.002117|0.20|
|cylinder|1e-03|0.115290|intra|-37.69 ± 39.33|+0.000632 ± 0.001301|0.05|
|cylinder|1e-03|0.115290|extra|-1.11 ± 38.92|+0.000299 ± 0.001287|0.32|
|cylinder|1e-03|0.115290|total|-38.84 ± 21.57|+0.000933 ± 0.000712|0.38|
|cylinder|1e-03|0.227999|intra|-52.73 ± 37.21|+0.001512 ± 0.001231|0.20|
|cylinder|1e-03|0.227999|extra|-29.49 ± 46.21|+0.000248 ± 0.001527|0.15|
|cylinder|1e-03|0.227999|total|-82.26 ± 53.41|+0.001762 ± 0.001764|0.41|
|cylinder|1e-03|0.341859|intra|-49.16 ± 39.79|+0.001592 ± 0.001316|0.55|
|cylinder|1e-03|0.341859|extra|-9.29 ± 50.47|-0.000486 ± 0.001669|0.22|
|cylinder|1e-03|0.341859|total|-58.45 ± 64.08|+0.001106 ± 0.002119|0.69|

## Exchange check (start intra, no force from gradient)

|scope / case|kappa|dt us|time ms|reference ± error|walk ± SE|difference|z|budget|pass|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|full sphere|1e-05|50|1|0.994065 ± 1.6e-09|0.993835 ± 0.000193|-0.000230|-1.19|0.002853|True|
|full sphere|1e-05|50|5|0.970847 ± 5.1e-09|0.970898 ± 0.000415|+0.000051|+0.12|0.003779|True|
|full sphere|1e-05|50|10|0.942836 ± 9.3e-09|0.943481 ± 0.000570|+0.000645|+1.13|0.004422|True|
|full sphere|1e-05|50|25|0.865116 ± 2.0e-08|0.864722 ± 0.000845|-0.000394|-0.47|0.005550|True|
|full sphere|1e-05|25|1|0.994065 ± 1.6e-09|0.993701 ± 0.000195|-0.000364|-1.86|0.001843|True|
|full sphere|1e-05|25|5|0.970847 ± 5.1e-09|0.970618 ± 0.000417|-0.000230|-0.55|0.002772|True|
|full sphere|1e-05|25|10|0.942836 ± 9.3e-09|0.943060 ± 0.000572|+0.000224|+0.39|0.003421|True|
|full sphere|1e-05|25|25|0.865116 ± 2.0e-08|0.864441 ± 0.000846|-0.000675|-0.80|0.004558|True|
|full sphere|1e-05|12.5|1|0.994065 ± 1.6e-09|0.994000 ± 0.000191|-0.000065|-0.34|0.001313|True|
|full sphere|1e-05|12.5|5|0.970847 ± 5.1e-09|0.970612 ± 0.000417|-0.000236|-0.57|0.002265|True|
|full sphere|1e-05|12.5|10|0.942836 ± 9.3e-09|0.942670 ± 0.000574|-0.000166|-0.29|0.002924|True|
|full sphere|1e-05|12.5|25|0.865116 ± 2.0e-08|0.866248 ± 0.000841|+0.001132|+1.35|0.004040|True|
|full cylinder|1e-05|50|1|0.996043 ± 9.9e-10|0.996277 ± 0.000150|+0.000234|+1.56|0.002495|True|
|full cylinder|1e-05|50|5|0.980520 ± 2.9e-09|0.980579 ± 0.000341|+0.000058|+0.17|0.003292|True|
|full cylinder|1e-05|50|10|0.961681 ± 5.3e-09|0.961658 ± 0.000474|-0.000023|-0.05|0.003848|True|
|full cylinder|1e-05|50|25|0.908696 ± 1.1e-08|0.908875 ± 0.000711|+0.000179|+0.25|0.004829|True|
|full cylinder|1e-05|25|1|0.996043 ± 9.9e-10|0.996161 ± 0.000153|+0.000118|+0.77|0.001574|True|
|full cylinder|1e-05|25|5|0.980520 ± 2.9e-09|0.980365 ± 0.000343|-0.000155|-0.45|0.002372|True|
|full cylinder|1e-05|25|10|0.961681 ± 5.3e-09|0.961243 ± 0.000477|-0.000438|-0.92|0.002934|True|
|full cylinder|1e-05|25|25|0.908696 ± 1.1e-08|0.908826 ± 0.000711|+0.000130|+0.18|0.003913|True|
|full cylinder|1e-05|12.5|1|0.996043 ± 9.9e-10|0.995978 ± 0.000156|-0.000065|-0.42|0.001124|True|
|full cylinder|1e-05|12.5|5|0.980520 ± 2.9e-09|0.980566 ± 0.000341|+0.000046|+0.13|0.001900|True|
|full cylinder|1e-05|12.5|10|0.961681 ± 5.3e-09|0.961322 ± 0.000476|-0.000359|-0.75|0.002469|True|
|full cylinder|1e-05|12.5|25|0.908696 ± 1.1e-08|0.907788 ± 0.000715|-0.000908|-1.27|0.003469|True|
|full sphere|1e-04|50|1|0.946011 ± 1.3e-07|0.946771 ± 0.000555|+0.000760|+1.37|0.008700|True|
|full sphere|1e-04|50|5|0.769772 ± 3.0e-07|0.769586 ± 0.001040|-0.000186|-0.18|0.010032|True|
|full sphere|1e-04|50|10|0.610470 ± 4.0e-07|0.610675 ± 0.001205|+0.000205|+0.17|0.010085|True|
|full sphere|1e-04|50|25|0.368147 ± 3.2e-07|0.367023 ± 0.001191|-0.001124|-0.94|0.008796|True|
|full sphere|1e-04|25|1|0.946011 ± 1.3e-07|0.945673 ± 0.000560|-0.000338|-0.60|0.005541|True|
|full sphere|1e-04|25|5|0.769772 ± 3.0e-07|0.770105 ± 0.001040|+0.000333|+0.32|0.007204|True|
|full sphere|1e-04|25|10|0.610470 ± 4.0e-07|0.611597 ± 0.001204|+0.001126|+0.94|0.007578|True|
|full sphere|1e-04|25|25|0.368147 ± 3.2e-07|0.366315 ± 0.001190|-0.001832|-1.54|0.006904|True|
|full sphere|1e-04|12.5|1|0.946011 ± 1.3e-07|0.946014 ± 0.000558|+0.000003|+0.01|0.003943|True|
|full sphere|1e-04|12.5|5|0.769772 ± 3.0e-07|0.768512 ± 0.001042|-0.001260|-1.21|0.005802|True|
|full sphere|1e-04|12.5|10|0.610470 ± 4.0e-07|0.610376 ± 0.001205|-0.000094|-0.08|0.006328|True|
|full sphere|1e-04|12.5|25|0.368147 ± 3.2e-07|0.367224 ± 0.001191|-0.000923|-0.77|0.005962|True|
|full cylinder|1e-04|50|1|0.963933 ± 7.9e-08|0.964539 ± 0.000457|+0.000606|+1.33|0.006676|True|
|full cylinder|1e-04|50|5|0.843304 ± 1.8e-07|0.843372 ± 0.000898|+0.000067|+0.08|0.008108|True|
|full cylinder|1e-04|50|10|0.727801 ± 2.5e-07|0.727960 ± 0.001099|+0.000159|+0.14|0.008586|True|
|full cylinder|1e-04|50|25|0.529961 ± 2.4e-07|0.528680 ± 0.001233|-0.001280|-1.04|0.008392|True|
|full cylinder|1e-04|25|1|0.963933 ± 7.9e-08|0.964594 ± 0.000457|+0.000660|+1.45|0.004299|True|
|full cylinder|1e-04|25|5|0.843304 ± 1.8e-07|0.845294 ± 0.000893|+0.001990|+2.23|0.005926|True|
|full cylinder|1e-04|25|10|0.727801 ± 2.5e-07|0.730090 ± 0.001097|+0.002289|+2.09|0.006597|True|
|full cylinder|1e-04|25|25|0.529961 ± 2.4e-07|0.532397 ± 0.001233|+0.002437|+1.98|0.006791|True|
|full cylinder|1e-04|12.5|1|0.963933 ± 7.9e-08|0.964178 ± 0.000459|+0.000245|+0.53|0.003122|True|
|full cylinder|1e-04|12.5|5|0.843304 ± 1.8e-07|0.843835 ± 0.000897|+0.000531|+0.59|0.004859|True|
|full cylinder|1e-04|12.5|10|0.727801 ± 2.5e-07|0.729828 ± 0.001097|+0.002027|+1.85|0.005610|True|
|full cylinder|1e-04|12.5|25|0.529961 ± 2.4e-07|0.532495 ± 0.001233|+0.002534|+2.06|0.005992|True|
|full sphere|1e-03|50|1|0.726684 ± 2.4e-06|0.733014 ± 0.001093|+0.006330|+5.79|0.030710|True|
|full sphere|1e-03|50|5|0.350679 ± 1.4e-06|0.352161 ± 0.001180|+0.001481|+1.26|0.018165|True|
|full sphere|1e-03|50|10|0.260421 ± 3.8e-07|0.263025 ± 0.001088|+0.002604|+2.39|0.012743|True|
|full sphere|1e-03|50|25|0.244199 ± 3.1e-09|0.241571 ± 0.001057|-0.002628|-2.48|0.008680|True|
|full sphere|1e-03|25|1|0.726684 ± 2.4e-06|0.728296 ± 0.001099|+0.001612|+1.47|0.017684|True|
|full sphere|1e-03|25|5|0.350679 ± 1.4e-06|0.351605 ± 0.001180|+0.000926|+0.79|0.011567|True|
|full sphere|1e-03|25|10|0.260421 ± 3.8e-07|0.260779 ± 0.001085|+0.000358|+0.33|0.008650|True|
|full sphere|1e-03|25|25|0.244199 ± 3.1e-09|0.245001 ± 0.001063|+0.000803|+0.76|0.006589|True|
|full sphere|1e-03|12.5|1|0.726684 ± 2.4e-06|0.728333 ± 0.001099|+0.001649|+1.50|0.011158|True|
|full sphere|1e-03|12.5|5|0.350679 ± 1.4e-06|0.351343 ± 0.001179|+0.000664|+0.56|0.008268|True|
|full sphere|1e-03|12.5|10|0.260421 ± 3.8e-07|0.259808 ± 0.001083|-0.000613|-0.57|0.006604|True|
|full sphere|1e-03|12.5|25|0.244199 ± 3.1e-09|0.244287 ± 0.001062|+0.000088|+0.08|0.005528|True|
|full cylinder|1e-03|50|1|0.815225 ± 1.6e-06|0.818854 ± 0.000952|+0.003628|+3.81|0.022144|True|
|full cylinder|1e-03|50|5|0.522947 ± 1.0e-06|0.525391 ± 0.001234|+0.002443|+1.98|0.015290|True|
|full cylinder|1e-03|50|10|0.422009 ± 4.3e-07|0.422760 ± 0.001220|+0.000751|+0.61|0.011848|True|
|full cylinder|1e-03|50|25|0.391044 ± 1.3e-08|0.389581 ± 0.001205|-0.001463|-1.21|0.008772|True|
|full cylinder|1e-03|25|1|0.815225 ± 1.6e-06|0.817944 ± 0.000953|+0.002719|+2.85|0.013085|True|
|full cylinder|1e-03|25|5|0.522947 ± 1.0e-06|0.524518 ± 0.001234|+0.001571|+1.27|0.010244|True|
|full cylinder|1e-03|25|10|0.422009 ± 4.3e-07|0.423895 ± 0.001221|+0.001886|+1.54|0.008497|True|
|full cylinder|1e-03|25|25|0.391044 ± 1.3e-08|0.390906 ± 0.001206|-0.000138|-0.11|0.006927|True|
|full cylinder|1e-03|12.5|1|0.815225 ± 1.6e-06|0.816223 ± 0.000957|+0.000998|+1.04|0.008566|True|
|full cylinder|1e-03|12.5|5|0.522947 ± 1.0e-06|0.523389 ± 0.001234|+0.000441|+0.36|0.007722|True|
|full cylinder|1e-03|12.5|10|0.422009 ± 4.3e-07|0.421826 ± 0.001220|-0.000183|-0.15|0.006817|True|
|full cylinder|1e-03|12.5|25|0.391044 ± 1.3e-08|0.390363 ± 0.001205|-0.000681|-0.57|0.006001|True|

## Early-time transient

In the archived run, at kappa=1e-3 m/s, dt=50 µs and t=1 ms, the sphere intra population was 0.733014 versus 0.726684 (difference +0.006330, +5.79 SE); the cylinder was 0.818854 versus 0.815225 (+0.003628, +3.81 SE). These exceed a sampling-only interpretation but remain inside the registered transient budgets (0.030710 and 0.022144). At 25/12.5 µs the sphere differences were +0.001612/+0.001649; the cylinder differences were +0.002719/+0.000998. The envelope was not enlarged. All 162 signal and 72 population comparisons passed their fixed budgets; all 162 imaginary diagnostics passed. The largest measured full-grid signal SE was below 0.002. Slopes above include their uncertainty; the grid does not establish a universal slope sign.

## Reproduction

This report and its JSON were migrated from `.codex-out/permeable-shell-full-grid.json` without rerunning the grid. The JSON retains the full-grid rows and batch seeds verbatim; the earlier fast/placement checks are outside the diagnostic scope. Use the same N and ordered seeds to reproduce the numerical tables, up to backend floating-point differences; timings will differ. Batch size is 32768; loop order is kappa, geometry (sphere, cylinder), then dt. No seeds are generated, and all CLI inputs are required.

```sh
JAX_PLATFORMS=cpu python -m diagnostics.permeable_shell_walk --seeds docs/permeable-shell-walk.json --n 163840 --output /tmp/permeable-shell-walk.json --note /tmp/permeable-shell-walk.md
```

The CPU suite retains the low-G sphere and native placement checks. Archived placement (N=131072) had KS p=0.27922/0.21043 and minimum feature-correlation p=0.11071/0.05974 for sphere/cylinder. These checks cannot certify stream independence: init_positions splits the key then reuses the parent for radial U. A proposed fix is separate child keys for direction, radius and axial U; no geometry change is made here.

Measured native runs: 1716.2 s. Archived study including fast and placement checks: 1740.1 s.

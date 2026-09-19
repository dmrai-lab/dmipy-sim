"""Adding measurement noise, and the two floors that decide whether an effect is measurable at all.

A replayed signal is noiseless. Real data is not, and the question a simulation is usually asked is whether
some effect survives the noise a scanner actually delivers. So the last step of most studies is to add
magnitude noise at a stated SNR and see what the analysis does to it.

Magnitude noise is Rician, not Gaussian, and that matters more than it looks. Taking the magnitude of a
complex measurement rectifies the noise, so at low signal the expected magnitude is BIASED UPWARDS: a
measurement that should be near zero reads near sigma instead. Every diffusion metric estimated from
high-b data inherits that bias, and it always points the same way, toward less attenuation and so toward a
lower apparent diffusivity.

There are then two floors underneath any result. The scanner's, set by SNR, and the simulation's, set by how
many walkers were walked (rung 20). An effect smaller than either is not measurable, and which one binds is
worth knowing before spending compute on the wrong one.

Assumes rung 20.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.acquisition.noise import add_rician_noise
from dmipy_sim.replay.bank import build_replay_pack

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(8_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
mc_floor = float(pack.fidelity["floor_max"])

bvals = np.array([0.0, 0.5e9, 1.0e9, 1.5e9, 2.0e9])
seq = pgse([[0.0, 0.0, 1.0]] * len(bvals), 0.008, 0.030, bvalues=list(bvals), TE=0.05, n_t=600)
clean = np.abs(np.asarray(pack.replay(seq))).reshape(-1)


def adc(S):
    """The log-linear slope over the whole shell set, in units of 1e-9 m^2/s."""
    ok = S > 0
    return -np.polyfit(bvals[ok], np.log(S[ok]), 1)[0] * 1e9


truth = adc(clean)
print(f"the walk: {pack.n_walkers:,} walkers, Monte-Carlo floor {mc_floor:.4f}")
print(f"noiseless signal at b = 0 to 2000 s/mm^2 along the axis: "
      f"{' '.join(f'{v:.3f}' for v in clean)}")
print(f"apparent diffusivity from it: {truth:.4f} e-9 m^2/s\n")

print(f"{'SNR':>5} {'sigma':>8} {'sigma vs MC floor':>18} {'ADC, 200 draws':>18} {'bias':>9}")
for snr in (100, 50, 20, 10, 5):
    sigma = 1.0 / snr
    draws = np.array([adc(np.asarray(add_rician_noise(clean, sigma, seed=s))) for s in range(200)])
    print(f"{snr:5d} {sigma:8.4f} {sigma/mc_floor:18.1f} {draws.mean():11.4f} +- {draws.std():4.4f} "
          f"{draws.mean()-truth:9.4f}")

print("\nThe bias is one-sided and grows as the SNR falls, because the rectified noise lifts the attenuated")
print("shells more than the unattenuated ones and flattens the slope. It is not removed by averaging more")
print("draws: the mean of the estimates moves, not just their spread.")
print("\nThe third column is the comparison worth making before spending compute. Where it is above one the")
print("scanner's noise already exceeds this walk's own, and more walkers buy nothing; where it is below one")
print(f"the simulation is the binding floor. The crossing is at an SNR of about {1.0/mc_floor:.0f} for this walk,")
print("so a study at an SNR of 20 is scanner-limited and one at an SNR of 100 is limited by the simulation.")

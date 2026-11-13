"""The pose is a knob as well, and one walk covers every pose at once.

Rotating the substrate relative to the gradient does not change the walk. It changes which direction the
gradient has in the substrate's own frame, and nothing else. So a pose is applied to the QUESTION, not to
the record: the gradient is rotated into the substrate frame and the same stored path answers it. That is
pose covariance, and it is exact rather than interpolated.

One pose at a time is the obvious use. The useful one is all of them: a real voxel holds a distribution of
poses, not a single one, and integrating a distribution by re-walking a rotated substrate per sample is the
expensive way to get a number that a single expansion already contains. `pose_response` computes the pack's
response over the whole rotation group for one acquisition, and any orientation statement -- one pose, a
Watson cone, a Bingham fan, a measured ODF -- composes against it as an inner product.

The cost that buys: a dispersed bundle is the same walk, contracted differently.

Assumes rung 13.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.so3 import Distribution

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)


def about_y(deg):
    """The substrate's axis tipped ``deg`` from z, in the plane of the gradient."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# One acquisition; the substrate's axis swept from along the gradient to across it.
response = pack.pose_response(seq)
print(f"the response over every pose: lmax {response.lmax}, {response.n_meas} measurement, route via "
      f"{getattr(response, 'route', 'closed')}")
print(f"\n{'axis tipped from z':20s} {'replayed at the pose':>21s} {'from the expansion':>19s}")
for deg in (0.0, 30.0, 60.0, 90.0):
    R = about_y(deg)
    direct = float(np.abs(np.asarray(pack.replay(seq, orientation=R))[0]))
    expanded = float(np.abs(np.asarray(response.at(R))[0]))
    print(f"{f'{deg:.0f} deg':20s} {direct:21.4f} {expanded:19.4f}")

print("\nThe two columns are the same number by two routes: the left rotates the gradient and replays, the")
print("right evaluates an expansion that was computed once and covers every pose there is.")

# A voxel of dispersed axes, without walking a single rotated substrate.
print(f"\n{'voxel':34s} {'S':>8}")
print(f"{'one axis along z':34s} {float(np.abs(response.at(about_y(0.0))[0])):8.4f}")
for kappa in (16.0, 4.0, 1.0):
    d = Distribution.watson(kappa, mu=(0.0, 0.0, 1.0))
    print(f"{f'Watson cone about z, kappa {kappa:.0f}':34s} {float(np.abs(response.compose(d)[0])):8.4f}")
print(f"{'uniform over all poses':34s} {float(np.abs(response.compose(Distribution.uniform())[0])):8.4f}")

print("\nEvery line came from the one walk at the top. Dispersion is a statement about the voxel, made at")
print("replay, and a fibre-orientation distribution measured in a brain enters the same way.")

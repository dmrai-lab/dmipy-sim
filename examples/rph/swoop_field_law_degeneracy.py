"""What two published figures leave undetermined about the Swoop's field, measured rather than assumed.

    python examples/rph/swoop_field_law_degeneracy.py

A magnet's field in the imaging volume solves Laplace's equation, so it is a sum of real solid harmonics in
the standard shim basis and nothing else. The Swoop publishes two scalars about its own -- 1100 ppm
peak-to-peak over a 16 cm DSV, and a steepest 1.4 mT/m inside it -- against fifteen coefficients of order 2
to 4. This samples the laws that are admissible under BOTH figures and reports what they still disagree
about, so that a number derived from the field comes with the width of what the data left open.

The result, in one line: the two figures fix the SCALE and leave the SHAPE open. The steepest gradient is
identical in every admissible law by construction, and the shell-average magnitude varies 18 %, but the
gradient DIRECTION at a point differs by a median 71 degrees.

The basis is LINEAR in the coefficients, so the terms and their gradients are evaluated once into design
matrices and every candidate law is then two matrix products. And both published constraints scale with
the coefficients, so their RATIO is scale-invariant: a law is admissible iff `max|grad| / peak-to-peak`
matches, and the size follows by one division. That makes the search a one-dimensional bisection on a
smooth function rather than a root find on a non-smooth one.
"""
import numpy as np
from dmipy_sim.acquisition import solid_harmonics as sh

B0, R = 0.064, 0.08
TARGET = (1.4e-3 / B0) / 1100e-6 * R
NAMES = [n for n in sh.TERMS if sh.TERMS[n][0] >= 2]          # l=1 was shimmed away

rng = np.random.default_rng(0)
u = rng.normal(size=(9000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
BALL = u * (0.999 * R * rng.random(9000) ** (1 / 3))[:, None]
s = rng.normal(size=(300, 3)); s /= np.linalg.norm(s, axis=1, keepdims=True)
SHELL = 0.0755 * s


def design(P):
    """``(F, G)``: each term's value and gradient at every point, built once."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    F = np.stack([sh.TERMS[n][1](x, y, z) for n in NAMES], axis=1)
    G = np.stack([np.stack(sh.TERMS[n][2](x, y, z), axis=1) for n in NAMES], axis=2)
    return F, np.ascontiguousarray(G)


FB, GB = design(BALL)
FS, GS = design(SHELL)


def ratio(c):
    f = FB @ c
    pp = f.max() - f.min()
    if pp <= 1e-30:
        return np.inf
    return np.linalg.norm(GB @ c, axis=1).max() / pp * R


def admissible(n_want=300, tries=4000):
    out = []
    for _ in range(tries):
        a, b = rng.normal(size=len(NAMES)), rng.normal(size=len(NAMES))
        ts = np.linspace(-6, 6, 25)
        rs = np.array([ratio(a + t * b) for t in ts])
        ok = np.isfinite(rs)
        for i in np.where(np.diff(np.sign(rs - TARGET))[: len(ts) - 1] != 0)[0]:
            if not (ok[i] and ok[i + 1]):
                continue
            lo, hi = ts[i], ts[i + 1]
            flo = ratio(a + lo * b) - TARGET
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                fm = ratio(a + mid * b) - TARGET
                if flo * fm <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            c = a + 0.5 * (lo + hi) * b
            if abs(ratio(c) - TARGET) > 2e-3:
                continue
            f = FB @ c
            out.append(c * (1100e-6 / (f.max() - f.min())))
            break
        if len(out) >= n_want:
            break
    return np.array(out)


if __name__ == "__main__":
    C = admissible()
    pp = np.ptp(FB @ C.T, axis=0) * 1e6
    gm = np.linalg.norm(np.einsum("pkt,nt->npk", GB, C), axis=2).max(axis=1) * B0 * 1e3
    print(f"{len(C)} admissible laws -- each harmonic, each reproducing BOTH published figures")
    print(f"   peak-to-peak {pp.mean():.0f} +- {pp.std():.2f} ppm    steepest {gm.mean():.3f} "
          f"+- {gm.std():.4f} mT/m\n")
    S = np.einsum("pkt,nt->npk", GS, C) * B0 * 1e3           # (n_law, n_shell, 3)
    mag = np.linalg.norm(S, axis=2)
    print("what they still disagree about, on the 7.55 cm shell where the ADC bias lives:")
    print(f"   |grad| family mean {mag.mean():.3f} mT/m")
    print(f"   at a given point the family spans a median of {np.median(np.ptp(mag, axis=0)):.3f} mT/m")
    print(f"   relative spread at a point (sd/mean), median over the shell: "
          f"{np.median(mag.std(axis=0) / mag.mean(axis=0)):.0%}")
    d = S / np.linalg.norm(S, axis=2, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.einsum("npk,pk->np", d, d[0]), -1, 1)))
    print(f"   gradient DIRECTION vs one arbitrary member: median {np.median(ang):.0f} deg, "
          f"90th pct {np.percentile(ang, 90):.0f} deg")
    print(f"\n   shell-average |grad| varies {mag.mean(axis=1).std() / mag.mean():.0%} across the family")

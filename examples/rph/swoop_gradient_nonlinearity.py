"""Derive the Swoop's gradient-nonlinearity coefficients from the open NIST dual-field DWI database.

Data: NIST "Database of Diffusion MRI Brain Scans at 64 mT and 3 T", doi 10.18434/mds2-3896 (public
domain), 20 participants on a Hyperfine Swoop (hw 1.8, sw rc8.6.0). Each participant has the diffusivity
fitted separately along x, y and z, on a grid whose affine is already FOV-centred.

THE MODEL. A real gradient coil does not produce a perfectly linear field, so a commanded gradient vector
G is delivered as L(r) G with L(0) = I -- the standard gradient-nonlinearity tensor (Bammer 2003, Janke
2004). Encoding along a unit axis u and fitting the ADC against the NOMINAL b therefore measures

    ADC_u(r) = (L(r) u)^T D (L(r) u)  ~=  D_uu (1 + 2 dL_uu)      (diagonal, first order)

so a fractional error dL in the delivered gradient shows up DOUBLED in the fitted diffusivity.

THE ESTIMATOR, and what it can and cannot see. Normalising each axis by the trace removes the unknown true
diffusivity but also imposes sum_u (ADC_u / trace) = 3, so the three slopes sum to zero by construction and
only the TRACELESS part of dL is observable. What is measured here is therefore dL_uu - mean_u(dL_uu), per
unit displacement. The common mode -- all three axes mis-scaled together -- is invisible to this estimator
and would need an absolute ADC reference.

WHY THE x AXIS IS THE ONE TO REGRESS ON. Every per-axis diffusivity is invariant under a left-right mirror,
so mirror-symmetric anatomy cannot produce a LINEAR dependence on x. Any such term is instrumental. The
same is not true of y or z, where anatomy varies systematically and would be indistinguishable.
"""
import glob, os, re
import h5py
import numpy as np

# Where the downloaded files are. Nothing is vendored: the database is 6.5 GB and public domain, and the
# per-axis files this needs are a few megabytes each:
#   BASE=https://data.nist.gov/od/ds/ark:/88434/mds2-3896/dualFieldDatabase-20250107
#   for p in 00 01 02 03 04 05 06 07 08 09; do for f in dxx dyy dzz; do
#     curl -O --output-dir "$NIST" "$BASE/participant$p/${f}_64mT.hdf5"; done; done
NIST = os.environ.get("SWOOP_NIST_DIR", os.path.expanduser("~/data/nist-dualfield"))


def _load(p, name):
    f = os.path.join(NIST, f"p{p}_{name}.hdf5")
    if not os.path.exists(f):
        return None
    with h5py.File(f, "r") as h:
        return dict(D=np.asarray(h["diffusion"]), A=np.asarray(h["A"]),
                    P=np.asarray(h["position_matrix"]))


def participants():
    ps = sorted({re.match(r"p(\d+)_dxx", os.path.basename(f)).group(1)
                 for f in glob.glob(os.path.join(NIST, "p*_dxx_64mT.hdf5"))})
    return [p for p in ps if all(_load(p, f"d{a}{a}_64mT") is not None for a in "xyz")]


def per_participant_slopes():
    """``(names, slopes)`` -- d(ADC_u / trace)/dx per cm, one row per participant, columns x/y/z."""
    out = []
    for p in participants():
        dx, dy, dz = (_load(p, f"d{a}{a}_64mT") for a in "xyz")
        P = dx["P"]
        ijk = np.argwhere(np.ones(dx["D"].shape, bool))
        xyz = (ijk @ P[:3, :3].T + P[:3, 3]) * 1e-3                 # metres, FOV-centred
        D = np.stack([dx["D"], dy["D"], dz["D"]], -1).reshape(-1, 3)   # um^2/s
        A = dx["A"].reshape(-1)
        tr = D.mean(1)
        ok = ((A > 0.25 * A.max()) & (tr > 200) & (tr < 2200)
              & np.all(D > 50, axis=1) & np.all(D < 4000, axis=1))
        X = xyz[ok].copy()
        X[:, 0] -= np.median(X[:, 0])          # about the HEAD's midline, not the FOV's
        AN = D[ok] / tr[ok, None]
        out.append([np.polyfit(X[:, 0] * 100, AN[:, a], 1)[0] for a in range(3)])
    return participants(), np.array(out)


def coefficients():
    """``{axis: (value, stderr)}`` -- the traceless part of dL_uu per metre of x displacement."""
    _names, S = per_participant_slopes()
    n = len(S)
    out = {}
    for a, axis in enumerate("xyz"):
        v = S[:, a] / 2.0 * 100.0              # /2 for the doubling, *100 for per-cm -> per-metre
        out[axis] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)))
    return out, n


if __name__ == "__main__":
    names, S = per_participant_slopes()
    coef, n = coefficients()
    print(f"{n} participants: {names}")
    print("\nd(ADC_u/trace)/dx, per cm, per participant:")
    for nm, row in zip(names, S):
        print(f"   p{nm}  " + "".join(f"{v:+10.5f}" for v in row))
    print("\ntraceless gradient-nonlinearity coefficient dL_uu - <dL>, per METRE of x:")
    for axis, (v, e) in coef.items():
        t = v / e
        print(f"   dL_{axis}{axis} = {v:+.4f} +- {e:.4f} /m   (t = {t:+.1f}, "
              f"{'significant' if abs(t) > 3 else 'NOT significant'})")
    print(f"\n   sum = {sum(v for v, _ in coef.values()):+.2e} /m  (zero by construction)")
    print(f"\nAt 7 cm from the midline that is "
          f"{max(abs(v) for v, _ in coef.values()) * 0.07 * 100:.1f} % of gradient amplitude.")

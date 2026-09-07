"""`FOD`: an orientation distribution with its provenance, the object `ReplayPack.replay(fod=)` takes.

The Gaunt composition (:mod:`dmipy_sim.replay.sh_convolution`) rests on the spherical-harmonic addition theorem,
which holds only for an **orthonormal** basis. Several conventions in circulation are not (DIPY's
``tournier07`` with ``legacy=True``, older MRtrix, whose ``m != 0`` functions have norm ``1/sqrt(2)``), one
orders the coefficients differently (``descoteaux07``), and a CSD output is not a probability density (its
integral is the apparent fibre density). Any of those, fed in as a bare array, composes to a wrong number
that is exactly right when the gradient is parallel to B0 -- so a bare array is refused. An FOD enters with
its basis named (:meth:`FOD.from_sh`, converted through DIPY as the oracle for that basis), already in the
required basis (:meth:`FOD.native`, checked), or from a distribution this package knows (:meth:`FOD.watson`,
:meth:`FOD.isotropic`), and is normalised to unit integral.

The required basis is :func:`dmipy_sim.replay.gaunt.real_sh`: orthonormal real spherical harmonics, even
orders, compact ``m = -l..l`` blocks -- DIPY's ``real_sh_tournier(..., legacy=False)`` (replay-pack-spec
RPH.md section 4.1).
"""
from dataclasses import dataclass

import numpy as np

_C00 = 1.0 / (2.0 * np.sqrt(np.pi))          # the l = 0 coefficient of a unit-integral density


def _block(l):
    """First index of the order-``l`` block in the compact even-order layout."""
    M = l // 2
    return M * (2 * M - 1) if M > 0 else 0


def _lmax_of(n):
    l = int(round((-3 + np.sqrt(1 + 8 * n)) / 2))
    if (l + 1) * (l + 2) // 2 != n or l % 2:
        raise ValueError(f"{n} coefficients is not an even-order compact SH array ((l+1)(l+2)/2 for even l)")
    return l


@dataclass(frozen=True)
class FOD:
    coeffs: np.ndarray        # in the required basis, unit integral
    lmax: int
    source: str               # where the coefficients came from and how they were converted

    # ---- constructors ------------------------------------------------------------------------------------
    @classmethod
    def native(cls, coeffs, *, normalize=False, source="native (required basis)"):
        """Coefficients already in the required basis. Refuses one that is not a unit-integral density unless
        ``normalize`` says to scale it (a CSD output's integral is the apparent fibre density)."""
        c = np.asarray(coeffs, np.float64).reshape(-1)
        lmax = _lmax_of(c.size)
        if c[0] <= 0:
            raise ValueError(f"the l = 0 coefficient is {c[0]:.3g}: not a density (negative or zero integral)")
        if normalize:
            c = c * (_C00 / c[0])
        elif abs(c[0] - _C00) > 1e-6 * _C00:
            raise ValueError(f"integral {c[0] / _C00:.4g} != 1: not a unit-integral density; pass normalize=True to scale "
                             f"it (a CSD FOD integrates to the apparent fibre density)")
        return cls(c, lmax, source)

    @classmethod
    def from_sh(cls, coeffs, *, basis, legacy=False, normalize=True):
        """Coefficients in a named DIPY / MRtrix basis, converted to the required one by the exact per-coefficient
        relations of RPH.md section 4.1 (no resampling):

        * ``"tournier07"``, ``legacy=False`` (MRtrix3): the required basis itself, identity;
        * ``"tournier07"``, ``legacy=True`` (DIPY's default flag, pre-3.0 MRtrix ``.mif`` FODs): not orthonormal,
          ``c <- c / sqrt(2)`` for ``m != 0``;
        * ``"descoteaux07"`` (DIPY): orthonormal but a different basis, ``c_{l,m} <- s_m c_{l,-m}`` with
          ``s_m = (-1)^m`` for ``m > 0`` and ``+1`` otherwise.
        """
        c_in = np.asarray(coeffs, np.float64).reshape(-1)
        lmax = _lmax_of(c_in.size)
        c = c_in.copy()
        if basis == "tournier07":
            if legacy:
                for l in range(0, lmax + 1, 2):
                    blk = _block(l)
                    for m in range(-l, l + 1):
                        if m != 0:
                            c[blk + l + m] /= np.sqrt(2.0)
        elif basis == "descoteaux07":
            for l in range(0, lmax + 1, 2):
                blk = _block(l)
                for m in range(-l, l + 1):
                    c[blk + l + m] = ((-1) ** m if m > 0 else 1.0) * c_in[blk + l - m]
        else:
            raise ValueError(f"unknown basis {basis!r}; known: 'tournier07' (legacy False | True), 'descoteaux07'")
        return cls.native(c, normalize=normalize, source=f"{basis} legacy={legacy}, converted per RPH 4.1")

    @classmethod
    def watson(cls, kappa, mu=(0.0, 0.0, 1.0), lmax=8):
        """A Watson distribution about ``mu`` (exact coefficients, :func:`math.sh_analytical.watson_sh`)."""
        from ..math.sh_analytical import watson_sh
        mu = np.asarray(mu, np.float64); mu = mu / np.linalg.norm(mu)
        return cls.native(watson_sh(mu, float(kappa), l_max=lmax), source=f"Watson kappa={kappa:g}")

    @classmethod
    def isotropic(cls, lmax=8):
        c = np.zeros((lmax + 1) * (lmax + 2) // 2); c[0] = _C00
        return cls(c, lmax, "isotropic")

    # ---- queries -----------------------------------------------------------------------------------------
    def evaluate(self, dirs):
        """The density at unit directions ``dirs`` (n, 3)."""
        from .gaunt import real_sh
        return real_sh(self.lmax, np.asarray(dirs, np.float64).reshape(-1, 3)) @ self.coeffs

    @property
    def integral(self):
        return float(self.coeffs[0] / _C00)

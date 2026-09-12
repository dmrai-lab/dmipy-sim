"""A closed form with an axis composes like a pack (RPH.md 3.1 / 6): expanded over SO(3) from its response at a pose
and contracted with the voxel's orientation distribution; a namespaced model is read by its package or refused."""
import os
import sys
import textwrap
import warnings

import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, Inert, Peaks, Phantom, Watson
from dmipy_sim.phantom.substrates import substrate_from_meta
from dmipy_sim.replay import analytic_pose_response, so3


class Stick:
    """``exp(-b D (g . axis)^2)`` at a pose: the one-parameter closed form with an axis."""
    kind = "analytic"
    model = "stickpkg:Stick"
    oriented = True

    def __init__(self, *, D_m2_s, m0, name="wm/stick"):
        self.D_m2_s, self.m0, self.name = float(D_m2_s), float(m0), name

    def response(self, seq, pose=None):
        axis = (np.eye(3) if pose is None else np.asarray(pose, float))[:, 2]
        b = np.asarray(seq.encoding.bvalues, float); g = np.asarray(seq.encoding.gradient_directions, float)
        return np.exp(-b * self.D_m2_s * (g @ axis) ** 2).astype(np.complex128)

    def to_meta(self):
        return {"id": self.name, "kind": "analytic", "m0": self.m0, "model": self.model, "params": {"D": self.D_m2_s}}

    @classmethod
    def from_meta(cls, meta):
        return cls(D_m2_s=meta["params"]["D"], m0=meta["m0"], name=meta["id"])

    def __repr__(self):
        return f"Stick(D_m2_s={self.D_m2_s:g}, m0={self.m0:g})"


def _seq():
    dirs = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return sequences.pgse(dirs, 5e-3, 15e-3, bvalues=[1e9, 1e9, 2e9, 2e9, 3e9], TE=25e-3)


def test_a_peak_evaluates_the_form_at_that_pose_exactly():
    seq = _seq(); stick = Stick(D_m2_s=1.7e-9, m0=1.0)
    axis = np.array([1.0, 2.0, 0.5]); axis /= np.linalg.norm(axis)
    ph = Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Peaks(np.broadcast_to(axis, (1, 1, 1, 1, 3)).copy()))
    S = ph.replay(seq, complex_signal=True)[0, 0, 0]
    np.testing.assert_allclose(S, stick.response(seq, pose=so3.rotation_of(axis)), atol=5e-4)   # the band the form needed
    R = analytic_pose_response(stick, seq, (None, 0))
    assert R.route == "analytic" and R.floor == 0.0 and R.misfit.max() < 1e-5 and R.nmax == 0


def test_a_watson_field_on_the_form_is_the_dispersed_integral():
    """The composition equals the Watson-weighted average of the form over the sphere, computed by brute force."""
    seq = _seq(); stick = Stick(D_m2_s=1.7e-9, m0=1.0)
    mu = np.array([0.3, 0.2, 1.0]); mu /= np.linalg.norm(mu); kappa = 6.0
    ph = Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Watson(mu=np.broadcast_to(mu, (1, 1, 1, 3)).copy(), kappa=np.full((1, 1, 1), kappa)))
    S = ph.replay(seq)[0, 0, 0]
    rng = np.random.default_rng(0)                           # Watson by rejection on a large uniform sample
    n = rng.normal(size=(400000, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    w = np.exp(kappa * (n @ mu) ** 2); w /= w.sum()
    b = np.asarray(seq.encoding.bvalues); g = np.asarray(seq.encoding.gradient_directions)
    E = np.exp(-b[None, :] * stick.D_m2_s * (n @ g.T) ** 2)
    ref = w @ E
    np.testing.assert_allclose(S, ref, atol=3e-3)          # the Watson is stated at the phantom's own band


def test_a_form_with_an_axis_needs_an_orientation_and_free_water_refuses_one():
    from dmipy_sim.phantom import FreeWater
    stick = Stick(D_m2_s=1.7e-9, m0=1.0)
    with pytest.raises(ValueError, match="no orientation given"):
        Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={stick: np.ones((1, 1, 1))}, orientation={})
    water = FreeWater(D_m2_s=3e-9, m0=1.0)
    with pytest.raises(ValueError, match="orientation-independent"):
        Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={water: np.ones((1, 1, 1))},
                        orientation={water: Peaks(np.zeros((1, 1, 1, 1, 3)) + [0, 0, 1.0])})


def test_a_namespaced_model_is_read_by_its_package_or_refused_naming_it(tmp_path, monkeypatch):
    meta = Stick(D_m2_s=1.7e-9, m0=0.7).to_meta()
    with pytest.raises(ValueError, match="'stickpkg'"):
        substrate_from_meta(meta)                              # the package is not installed: refused, named
    pkg = tmp_path / "stickpkg"; pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "phantom.py").write_text(textwrap.dedent('''
        import numpy as np
        class Stick:
            kind = "analytic"; model = "stickpkg:Stick"; oriented = True
            def __init__(self, *, D_m2_s, m0, name):
                self.D_m2_s, self.m0, self.name = float(D_m2_s), float(m0), name
            def response(self, seq, pose=None):
                axis = (np.eye(3) if pose is None else np.asarray(pose, float))[:, 2]
                b = np.asarray(seq.encoding.bvalues, float); g = np.asarray(seq.encoding.gradient_directions, float)
                return np.exp(-b * self.D_m2_s * (g @ axis) ** 2).astype(np.complex128)
            def to_meta(self):
                return {"id": self.name, "kind": "analytic", "m0": self.m0, "model": self.model, "params": {"D": self.D_m2_s}}
        def analytic_substrate(meta):
            return Stick(D_m2_s=meta["params"]["D"], m0=meta["m0"], name=meta["id"])
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    form = substrate_from_meta(meta)
    assert form.oriented and form.m0 == 0.7 and form.D_m2_s == 1.7e-9
    # a phantom written with it reads back and replays through the package's reader
    seq = _seq(); stick = Stick(D_m2_s=1.7e-9, m0=1.0)
    axis = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    ph = Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Peaks(np.broadcast_to(axis, (1, 1, 1, 1, 3)).copy()))
    ph.write(tmp_path / "stick.rph", id="test/stick", license="CC0", citation="test")
    back = Phantom.read(tmp_path / "stick.rph")
    np.testing.assert_allclose(back.replay(seq), ph.replay(seq), rtol=1e-6)


def test_a_closed_form_is_full_tier_with_zeros():
    """A form with no susceptibility source has a field of zero at any B0: the signal is unchanged and nothing is
    said; free water carries its bulk relaxation when declared, and none when not."""
    from dmipy_sim.phantom import FreeWater
    seq = _seq(); stick = Stick(D_m2_s=1.7e-9, m0=1.0)
    ph = Phantom.compose(Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3)), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Peaks(np.zeros((1, 1, 1, 1, 3)) + [0, 0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S1 = ph.replay(seq, B0_T=3.0, chi_iso=1e-7)
    np.testing.assert_allclose(S1, ph.replay(seq))
    water = FreeWater(D_m2_s=3e-9, m0=1.0, T2_s=2.0)
    b = np.asarray(seq.encoding.bvalues); TE = float(np.max(seq.encoding.TE))
    np.testing.assert_allclose(water.response(seq), np.exp(-b * 3e-9) * np.exp(-TE / 2.0))
    np.testing.assert_allclose(FreeWater(D_m2_s=3e-9, m0=1.0).response(seq), np.exp(-b * 3e-9))
    assert FreeWater.from_meta(water.to_meta()).T2_s == 2.0 and "T2_s" not in FreeWater(D_m2_s=3e-9, m0=1.0).to_meta()

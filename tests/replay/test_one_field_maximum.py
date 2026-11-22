"""A pack states ONE maximum field strength, and the field tier is certified at it.

``certificate_settings`` refuses an envelope that states several field strengths, by name, saying what to pass; the
settings it gives carry ``B0_max`` and no list; ``certified_envelope`` refuses settings that carry the old list key;
the field band is derived at the maximum alone; and a built field pack's header holds the one entry under
``replay_envelope.acquisition.B0_list``.
"""
import warnings

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay import Envelope
from dmipy_sim.engine.accumulate import spec_chi_scale
from dmipy_sim.fields.strand_field import CHANNEL_NAMES
from dmipy_sim.io.strands import write_tck, concat_centerlines
from dmipy_sim.replay.bank import build_replay_pack, derive_field_band
from dmipy_sim.replay.compression import certificate_settings, certified_envelope
from dmipy_sim.spec import disco_spec
from tests._envelope import stated, walk_envelope

BATTERY = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.1],
               delta_frac=0.2, Delta_frac=0.5)


def _envelope(*B0):
    return Envelope(name="test", band_hz=1e9, G_max=2.0, B0_list=tuple(B0), chi_max=1e-6, rho2_max=None,
                    max_refocusing_pulses=None)


def _series():
    t = (np.arange(32) + .5) / 32
    values = np.zeros((12, 32, len(CHANNEL_NAMES)))
    values[:, :, 1] = np.linspace(.8, 1.2, 12)[:, None] * 1e-4 * np.sin(2 * np.pi * t)
    values[:, :, 2] = -values[:, :, 1]
    return values


def test_an_envelope_with_several_field_strengths_is_refused_by_name():
    """A1: the refusal says that a pack states one maximum field strength and names the largest as what to pass."""
    with pytest.raises(ValueError, match=r"one maximum field strength.*B0_list=\(7\.0,\)"):
        certificate_settings(_envelope(3.0, 7.0))


def test_the_settings_state_the_maximum_and_no_list():
    """A2: one entry gives ``B0_max`` as a float; none gives ``B0_max`` None; the settings carry no ``B0_list``."""
    one = certificate_settings(_envelope(7.0))
    assert one["B0_max"] == 7.0 and isinstance(one["B0_max"], float) and "B0_list" not in one
    none = certificate_settings(_envelope())
    assert none["B0_max"] is None and "B0_list" not in none


def test_settings_that_carry_the_list_key_are_refused_by_name():
    """A3: the key is refused, saying ``B0_max``, not ignored (which would certify at the library default instead)."""
    with pytest.raises(ValueError, match="B0_max"):
        certified_envelope(dict(B0_list=[3.0]))
    assert certified_envelope(dict(B0_max=3.0))["B0_max"] == 3.0


def test_the_field_band_is_derived_at_the_stated_maximum_alone():
    """A4: ``derive_field_band`` under settings stating ``B0_max = 3`` records one field strength, 3 T, and the band of
    an explicit ``B0 = 3`` call."""
    kw = dict(K_max=16, chi=1e-6, target_floor=1e-4)
    k_stated, rec = derive_field_band(_series(), CHANNEL_NAMES, .001,
                                      env=dict(B0_max=3.0, theta_deg=[90], max_refocus_pulses=1), **kw)
    k_explicit, _ = derive_field_band(_series(), CHANNEL_NAMES, .001, B0=3.0,
                                      env=dict(theta_deg=[90], max_refocus_pulses=1), **kw)
    assert rec["B0_list"] == [3.0] and k_stated == rec["K"] == k_explicit


@pytest.fixture(scope="module")
def field_pack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("one_field_maximum")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        walk = d.walk(spec, 60, 8e-4, 1e-4, seed=3, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                      field_cutoff_max_m=25e-6, envelope=walk_envelope(chi_max=spec_chi_scale(spec)))
        return build_replay_pack(walk, id="t/one-maximum", license="x", citation="x", K=4, susc_path_K=4,
                                 device="numpy", **stated(dict(BATTERY, rho2_list=[3e-6], B0_max=7.0),
                                                          chi_max=spec_chi_scale(spec)), blt_temporal_K=2)


def test_a_built_field_pack_states_one_entry(field_pack):
    """A5: the header keeps its key ``B0_list`` with one entry, and the field envelope reads the maximum."""
    assert field_pack.replay_envelope["acquisition"]["B0_list"] == [7.0]
    assert field_pack.field_envelope["B0_max"] == 7.0

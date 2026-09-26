"""The MC/DC readers: the Camino scheme, the raw ``.bfloat``, the configuration and the walker list.

The hand-written cases fix the format; the cases on the released files fix what the format ACTUALLY holds,
and are skipped by name when ``DMIPY_SIM_MCDC_ROBUST_DIR`` does not point at a checkout of
https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations.
"""
import os

import numpy as np
import pytest

from dmipy_sim.io import mcdc

DATA = os.environ.get("DMIPY_SIM_MCDC_ROBUST_DIR")
needs_data = pytest.mark.skipif(not DATA, reason="set DMIPY_SIM_MCDC_ROBUST_DIR to the MC/DC data checkout")

GAMMA_MCDC = 267.51525e3 * 1e3      # rad/(ms T) -> rad/(s T), MC/DC's `giro` (src/constants.h)

SCHEME = """VERSION: STEJSKALTANNER
0.000000 0.000000 0.000000 0.000000 0.040000 0.010000 0.060000
0.000000 0.000000 1.000000 0.200000 0.040000 0.010000 0.060000
1.000000 0.000000 0.000000 0.100000 0.020000 0.005000 0.060000
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ------------------------------------------------------------------------------------------- the scheme
def test_scheme_reads_the_seven_columns_as_one_sequence(tmp_path):
    seq = mcdc.read_scheme(write(tmp_path, "s.scheme", SCHEME))
    assert seq.G.shape[0] == 3
    assert seq.family == "pgse"
    np.testing.assert_allclose(seq.dt * (seq.G.shape[1] - 1), 0.060, rtol=1e-12)
    np.testing.assert_allclose(seq.encoding.delta, [0.010, 0.010, 0.005], rtol=1e-12)
    np.testing.assert_allclose(seq.encoding.Delta, [0.040, 0.040, 0.020], rtol=1e-12)


def test_scheme_b_is_the_stejskal_tanner_value_of_its_own_columns(tmp_path):
    """``b = (gamma G delta)^2 (Delta - delta/3)``, the square-lobe value the columns state, to 1e-4 relative.

    The residual is the sequence grid: the builder samples a lobe mid-step, and ``n_t`` is the coarsest
    edge-exact lattice refined until the shortest lobe holds 500 samples. Measured here: 9.9e-5 relative on
    this three-row file, and 3.1e-5 on the released ``ActiveAxG140_PM.scheme``.
    """
    seq = mcdc.read_scheme(write(tmp_path, "s.scheme", SCHEME))
    G, Delta, delta = np.array([0.0, 0.2, 0.1]), np.array([0.04, 0.04, 0.02]), np.array([0.01, 0.01, 0.005])
    want = (GAMMA_MCDC * G * delta) ** 2 * (Delta - delta / 3)
    got = np.asarray(seq.b())
    np.testing.assert_allclose(got[1:], want[1:], rtol=2e-4)
    assert got[0] == 0.0


def test_the_lobes_are_square_and_centred_on_the_echo(tmp_path):
    """MC/DC's ``getGradImpulse``: ``pad = (TE - Delta - delta)/2``, ``+G`` then ``-G``, no ramp."""
    seq = mcdc.read_scheme(write(tmp_path, "s.scheme", SCHEME))
    g = np.asarray(seq.G_eff)[1, :, 2]
    t = np.arange(len(g)) * seq.dt
    pad = (0.060 - 0.040 - 0.010) / 2
    on1 = (t > pad + seq.dt) & (t < pad + 0.010 - seq.dt)
    on2 = (t > pad + 0.040 + seq.dt) & (t < pad + 0.050 - seq.dt)
    off = (t < pad - seq.dt) | ((t > pad + 0.010 + seq.dt) & (t < pad + 0.040 - seq.dt))
    np.testing.assert_allclose(g[on1], 0.2, atol=1e-6)
    np.testing.assert_allclose(g[on2], -0.2, atol=1e-6)
    np.testing.assert_allclose(g[off], 0.0, atol=1e-9)


@pytest.mark.parametrize("bad,says", [
    ("VERSION: APGSE\n0 0 1 0.1 0.1 0.04 0.01 0.005 0.06\n", "APGSE"),
    ("VERSION: STEJSKALTANNER\n0 0 1 0.2 0.04 0.01\n", "7-column"),
    ("VERSION: STEJSKALTANNER\n0 0 2 0.2 0.04 0.01 0.06\n", "neither unit nor zero"),
    ("VERSION: STEJSKALTANNER\n0 0 1 0.2 0.04 0.01 0.06\n0 0 1 0.2 0.04 0.01 0.07\n", "one echo time"),
    ("VERSION: STEJSKALTANNER\n0 0 1 0.0 0.0 0.0 0.06\n", "no diffusion time"),
    ("VERSION: STEJSKALTANNER\n0 0 1 0.2 0.01 0.04 0.06\n", "lobes overlap"),
    ("VERSION: STEJSKALTANNER\n0 0 1 0.2 0.05 0.02 0.06\n", "longer than TE"),
])
def test_the_scheme_refuses_by_name_what_the_format_leaves_ambiguous(tmp_path, bad, says):
    with pytest.raises(ValueError, match=says):
        mcdc.read_scheme(write(tmp_path, "bad.scheme", bad))


def test_a_grid_that_cannot_hold_the_lobe_edges_is_refused(tmp_path):
    p = write(tmp_path, "s.scheme", SCHEME)
    exact = mcdc.read_scheme(p).G.shape[1]
    mcdc.read_scheme(p, n_t=(exact - 1) // 2 + 1)                    # a coarsening of the exact grid is fine
    with pytest.raises(ValueError, match="does not put every delta"):
        mcdc.read_scheme(p, n_t=exact - 2)


# ------------------------------------------------------------------------------------------- the signal
def test_bfloat_is_little_endian_float32(tmp_path):
    want = np.array([50000.0, 123.5, -7.25], np.float32)
    p = tmp_path / "x.bfloat"
    p.write_bytes(want.astype("<f4").tobytes())
    np.testing.assert_array_equal(mcdc.read_bfloat(str(p)), want.astype(np.float64))


def test_bfloat_refuses_a_ragged_file_and_a_count_that_disagrees(tmp_path):
    p = tmp_path / "x.bfloat"
    p.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="not a whole number of float32"):
        mcdc.read_bfloat(str(p))
    p.write_bytes(np.array([1.0, 2.0], np.float32).astype("<f4").tobytes())
    with pytest.raises(ValueError, match="holds 2 measurements"):
        mcdc.read_bfloat(str(p), n_measurements=3)


def test_the_walker_count_is_the_b0_entry():
    S = np.array([1000.0, 1000.0, 500.0, 12.0])
    b = np.array([0.0, 0.0, 1e9, 3e9])
    assert mcdc.walker_count(S, b) == 1000
    with pytest.raises(ValueError, match="no b = 0 measurement"):
        mcdc.walker_count(S[2:], b[2:])
    with pytest.raises(ValueError, match="not one whole walker count"):
        mcdc.walker_count(np.array([1000.0, 999.5, 500.0]), np.array([0.0, 0.0, 1e9]))


# ------------------------------------------------------------ the configuration and the walker list
CONF = """N 1000
T 5000
duration 0.0535200000
diffusivity 0.0000000006
scheme_file /somewhere/ActiveAxG140_PM.scheme
scale_from_stu 1
<obstacle>
ply /somewhere/axon.ply
ply_scale 0.00100
</obstacle>
<voxels>
-0.0015000000 -0.0015000000 -0.1250000000
0.0015000000 0.0015000000 0.1250000000
</voxels>
ini_walker_file /somewhere/axon_ini_points.txt
<END>
"""


def test_conf_returns_si_and_leaves_the_voxel_in_millimetres_alone(tmp_path):
    """``scale_from_stu`` scales the diffusivity and the duration only (``Parameters::readSchemeFile``); the
    voxel corners and ``ply_scale`` are already in MC/DC's millimetres."""
    c = mcdc.read_conf(write(tmp_path, "a.conf", CONF))
    assert (c["N"], c["T"]) == (1000, 5000)
    assert c["duration"] == pytest.approx(0.05352)
    assert c["diffusivity"] == pytest.approx(6e-10)
    np.testing.assert_allclose(c["voxel_min"], [-1.5e-6, -1.5e-6, -1.25e-4])
    np.testing.assert_allclose(c["voxel_max"], [1.5e-6, 1.5e-6, 1.25e-4])
    assert c["mesh_scale"] == pytest.approx(1e-6)          # a PLY unit is a micrometre
    assert c["ini_walkers_file"].endswith("axon_ini_points.txt")


def test_conf_refuses_a_configuration_in_mcdc_units(tmp_path):
    with pytest.raises(ValueError, match="scale_from_stu"):
        mcdc.read_conf(write(tmp_path, "b.conf", CONF.replace("scale_from_stu 1", "scale_from_stu 0")))


def test_walker_list_is_millimetres_and_is_read_cyclically(tmp_path):
    p = tmp_path / "ini.txt"
    p.write_text("0.0001 0.0 0.0\n-0.0001 0.0 0.001\n")
    ini = mcdc.read_ini_walkers(str(p))
    np.testing.assert_allclose(ini, [[1e-7, 0, 0], [-1e-7, 0, 1e-6]])
    np.testing.assert_allclose(mcdc.seed_positions(ini, 5), ini[[0, 1, 0, 1, 0]])


# ------------------------------------------------------------------------------ the released files
@needs_data
def test_the_released_G140_scheme_is_the_activeax_protocol():
    seq = mcdc.read_scheme(os.path.join(DATA, "Simulator-Conf-files", "ActiveAxG140_PM.scheme"))
    assert seq.G.shape[0] == 372                       # 4 shells x (90 directions + 3 b = 0)
    assert seq.dt * (seq.G.shape[1] - 1) == pytest.approx(0.05352)
    b = np.asarray(seq.b())
    assert int((b == 0).sum()) == 12
    shells = np.unique(np.round(b[b > 0] / 1e7).astype(int) * 10)   # s/mm^2, to the nearest 10
    np.testing.assert_array_equal(shells, [1930, 3090, 13190])      # the ActiveAx shells, 1930 twice over


@needs_data
def test_the_released_G300_scheme_is_refused_for_its_zero_timing_b0_rows():
    """Its three b = 0 rows state ``Delta = delta = 0``: MC/DC plays no gradient for them and the file says
    nothing about what timing the measurement had. Read as a sequence, that is a guess, so it is refused."""
    with pytest.raises(ValueError, match="no diffusion time"):
        mcdc.read_scheme(os.path.join(DATA, "Simulator-Conf-files", "ActiveAxG300_PM.scheme"))


@needs_data
def test_the_released_signals_are_little_endian_sums_over_fifty_thousand_walkers():
    """Read big-endian the same bytes are 1e-39 and 1e+27 denormals; read little-endian every b = 0 entry is
    exactly 50000, which is the walker count MC/DC summed ``cos(phi)`` over."""
    seq = mcdc.read_scheme(os.path.join(DATA, "Simulator-Conf-files", "ActiveAxG140_PM.scheme"))
    b = np.asarray(seq.b())
    root = os.path.join(DATA, "Experiments-raw-signals.", "Undulated_fibers", "d_1")
    for amp, wL in [(0.2, 32.0), (1.0, 12.0), (2.6, 4.0)]:
        f = os.path.join(root, f"uAxon_amp_{amp}_wL_{wL}", f"uAxon_d_1.0_amp_{amp}_wL_{wL}_DWI.bfloat")
        S = mcdc.read_bfloat(f, n_measurements=372)
        assert mcdc.walker_count(S, b) == 50_000
        assert np.all(np.abs(S / 50_000) <= 1.0 + 1e-6)

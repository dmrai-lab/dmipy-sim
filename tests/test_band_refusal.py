"""A replay refuses a waveform whose gradient reaches beyond the pack's temporal band (dmipy-sim#277).

The pack resolves motion up to ``K / (2 T)`` Hz. A waveform projected on the bridge basis beyond that band would
contract against bands the pack never stored, and the signal would come back smooth, plausible and wrong. The
predicate is walk-free: the free bridge's band variances at the walk's diffusivity give the phase the dropped
bands carry, so a pack of walkers that never move (every band zero) judges a waveform by its band alone.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.coefficient_walk import CoefficientWalk
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.geometry import FreeDiffusion
from dmipy_sim.spec import spec_of
from tests.test_bank import _lean_env

N_W, N_T, DT, D0 = 40, 201, 1e-4, 2e-9                       # a 20 ms grid


def _static_pack(K, seed=0):
    """Walkers that never move: the positions' bands are all zero, so only K and the grid say what the pack resolves."""
    rng = np.random.default_rng(seed)
    C = np.zeros((N_W, 1, K + 2, 3), np.float32)
    C[:, 0, 0, :] = rng.uniform(-2e-6, 2e-6, (N_W, 3))                                # the start; no displacement, no band
    walk = CoefficientWalk(
        position_coeffs=C, T=(N_T - 1) * DT, K=K, m=0,
        block=dict(h=DT, m=0, n_blocks=N_T - 1, rule="given", precision="float64", k0_residual_relative=0.0),
        n_windows=1, dt_sim=DT, sub_steps=1, qv_positions=np.zeros((N_W, 1, 3), np.float32), seed=seed,
        diffusivity=D0, spec=spec_of(FreeDiffusion()))
    return build_replay_pack(walk, id=f"t/static-K{K}", envelope=_lean_env(), K=K, license="x", citation="x")


def _ogse(f_hz, sigma, n_t=801):
    return sequences.ogse([[1.0, 0.0, 0.0]], f_hz, sigma, shape="cosine", Delta=sigma + 2e-3, bvalues=[1e9],
                          TE=2 * sigma + 4e-3, n_t=n_t, slew_rate=np.inf)


def test_an_ogse_above_the_band_is_refused_and_one_below_is_not():
    pk = _static_pack(K=20)                                                          # 500 Hz
    assert pk.temporal_bandwidth_hz == pytest.approx(500.0)
    low, high = _ogse(125.0, 8e-3), _ogse(750.0, 4e-3)                              # one period, and three
    hz_low, k_low, err_low = pk.waveform_band(low)
    hz_high, k_high, err_high = pk.waveform_band(high)
    assert k_low <= 20 < k_high and hz_low <= 500.0 < hz_high, (hz_low, k_low, hz_high, k_high)
    floor = pk.meta["fidelity"]["floor_max"]
    assert err_low <= floor < err_high and err_high > 1.0, (err_low, floor, err_high)   # above the band the signal is not merely off, it is unrelated
    assert np.isfinite(pk.replay(low)).all()
    with pytest.raises(ValueError, match="temporal band") as e:
        pk.replay(high)
    assert "500 Hz" in str(e.value) and f"{hz_high:.4g}" in str(e.value)               # both frequencies stated
    for route in (lambda p, s: p.walker_signals(s), lambda p, s: p.walker_phases(s), lambda p, s: p.replay_bloch(s),
                  lambda p, s: p.walker_primitives(s)):
        with pytest.raises(ValueError, match="temporal band"):
            route(pk, high)
    wide = _static_pack(K=80)                                                        # 2000 Hz: the same waveform replays
    assert wide.waveform_band(high)[1] <= 80 and np.isfinite(wide.replay(high)).all()


def test_the_battery_waveforms_of_a_walked_pack_are_within_its_band(pack):
    """The certified packs of the suite replay their own kind of waveform: a PGSE's trapezoid tail beyond the band
    is below the pack's floor, so nothing that replayed before is refused now."""
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    hz, bands, err = pack.waveform_band(seq)
    assert bands <= pack.K and hz <= pack.temporal_bandwidth_hz and 0.0 < err <= pack.meta["fidelity"]["floor_max"]
    assert np.isfinite(pack.replay(seq)).all()


# ---- the refusal is tied to the pack's stated numerical cap (fidelity.target_floor)

def _with_cap(pk, cap, **terms):
    """``pk`` with its certificate stating ``cap`` as its numerical cap and ``terms`` as its other certified rows."""
    import copy
    from dmipy_sim.replay.replay import ReplayPack
    meta = copy.deepcopy(pk.meta)
    meta["fidelity"].update(target_floor=cap, **terms)
    return ReplayPack(pk.arrays, meta)


def test_a_waveform_is_refused_when_its_truncation_exceeds_the_cap_less_the_other_terms_and_not_the_floor():
    """The threshold is ``target_floor`` less the certified non-truncation rows, to the digit: a cap just above the
    waveform's truncation plus those rows admits it, just below refuses it; the Monte-Carlo floor (far above both)
    decides nothing once the pack states a cap."""
    pk = _static_pack(K=20)
    full = _ogse(500.0, 4e-3)
    wf = full.with_gradient(np.asarray(full.G, np.float64) * 0.05)                    # a weak one: a small tail, under the floor
    err = pk.waveform_band(wf, tol=1e9)[2]                                           # its truncation at K
    other = dict(err_quantisation=3e-4, err_quadrature=1e-5, err_field_read=2e-5)
    rest = sum(other.values())
    assert 0.0 < err < pk.meta["fidelity"]["floor_max"]                              # the floor would admit it
    ok = _with_cap(pk, (err + rest) * 1.001, **other)
    ok._check_band(wf)
    assert ok.truncation_budget["other_terms"] == pytest.approx(rest) and ok.truncation_budget["cap_source"] == "fidelity.target_floor"
    tight = _with_cap(pk, (err + rest) * 0.999, **other)
    with pytest.raises(ValueError, match="temporal band") as e:
        tight._check_band(wf)
    assert "fidelity.target_floor" in str(e.value) and f"{tight.truncation_budget['budget']:.2e}" in str(e.value)
    with pytest.raises(ValueError, match="temporal band"):
        tight.replay(wf)


def test_a_cap_already_spent_by_the_other_terms_admits_only_what_costs_nothing():
    pk = _with_cap(_static_pack(K=20), 1e-3, err_quantisation=2e-3)
    assert pk.truncation_budget["budget"] == 0.0
    with pytest.raises(ValueError, match="temporal band"):
        pk._check_band(_ogse(750.0, 4e-3))


def test_a_pack_without_a_cap_is_held_to_its_monte_carlo_floor_by_name():
    pk = _static_pack(K=20)
    b = pk.truncation_budget
    assert b["cap_source"] in ("fidelity.floor_max", "1/sqrt(n_walkers)") and b["budget"] == b["cap"] and b["other_terms"] == 0.0


def test_the_estimate_is_the_free_tail_beyond_k_to_infinity_not_to_a_grid():
    """err at K is half the bridge's closed-form phase variance less the retained bands' (an upper bound for any
    restricted walk), whatever the number of bands sought."""
    from dmipy_sim.replay.compression import band_phase_variances
    pk = _static_pack(K=20)
    wf = _ogse(750.0, 4e-3)
    per_band, total = band_phase_variances(np.asarray(wf.G_eff, np.float64), float(wf.dt), pk.T_window, pk.K, pk.diffusivity)
    assert pk.waveform_band(wf, tol=1e9)[2] == pytest.approx(0.5 * float((total - per_band.sum(axis=1)).max()), rel=1e-12)


def _layout(K, *, cap, floor=0.02, other=None):
    """A columnar layout's header: ``K`` bands of a free walk at ``D0`` over 20 ms, every stored band at the free variance."""
    from dmipy_sim.replay.columnar import ColumnarPack
    from dmipy_sim.replay.compression import band_variances
    T = (N_T - 1) * DT
    lay = ColumnarPack.__new__(ColumnarPack)
    var = band_variances(K, T, D0)[:, None] * np.ones((1, 3))
    fid = dict(target_floor=cap, **(other or {})) if cap is not None else {}
    lay.meta = dict(walk_params=dict(diffusivity=D0, windows=dict(n=1, T=T), T_max=T), fidelity=fid,
                    columnar=dict(band_variance=var.T.tolist(), band_variance_max=var.T.tolist()))
    lay.K, lay.floor, lay.band_groups = K, floor, [K // 4, K // 2, K]
    return lay


def test_a_columnar_layout_refuses_a_waveform_whose_tail_beyond_its_stored_bands_exceeds_its_budget():
    """``bands_for`` states the bands beyond ``K`` (never stored) at the free bridge's variance and refuses at the budget,
    where it used to return ``(K, 0)`` for any waveform."""
    wf = _ogse(750.0, 4e-3)
    lay = _layout(20, cap=1e-3)
    with pytest.raises(ValueError, match="temporal band.*fidelity.target_floor"):
        lay.bands_for(wf)
    tail = _static_pack(K=20).waveform_band(wf, tol=1e9)[2]
    admits = _layout(20, cap=tail * 1.001)
    Kp, err = admits.bands_for(wf)
    assert Kp == 20 and err == pytest.approx(tail, rel=1e-6)
    with pytest.raises(ValueError, match="temporal band"):
        _layout(20, cap=tail * 0.999).bands_for(wf)
    assert _layout(20, cap=None, floor=tail * 1.5).bands_for(wf)[0] == 20          # no cap stated: held to the median floor
    with pytest.raises(ValueError, match="per_voxel.floor_median"):
        _layout(20, cap=None, floor=tail * 0.5).bands_for(wf)

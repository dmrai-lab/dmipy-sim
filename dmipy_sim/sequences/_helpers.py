"""The few helpers the builders and the assembler share: the b integral of a gradient array, the echo-time
floor, the per-row broadcast of delta / Delta / TE."""
import numpy as np

_TE_FLOOR_ATOL = 1e-9


def _calc_b_from_waveform(G, dt):
    """b per measurement, ``(n_m, n_t, 3) -> (n_m,)``: :func:`dmipy_sim.acquisition.waveforms.b_from_gradient`."""
    from ..acquisition.waveforms import b_from_gradient
    return b_from_gradient(G, dt)


def _resolve_te(TE, t_total_min, n_m):
    """Resolve echo time against the minimum echo time; (TE_arr, was_auto)."""
    if TE is None:
        return np.full(n_m, float(t_total_min)), True
    TE_arr = np.broadcast_to(np.asarray(TE, dtype=np.float64), (n_m,)).copy()
    if np.any(TE_arr < t_total_min - _TE_FLOOR_ATOL):
        raise ValueError(
            "Echo time TE = {:.3f} ms is below the minimum echo time "
            "{:.3f} ms set by the gradient schedule; the echo cannot form "
            "before the encoding completes.".format(
                float(np.min(TE_arr)) * 1e3, float(t_total_min) * 1e3))
    return TE_arr, False


def unify_length_reference_delta_Delta(reference_array, delta, Delta, TE):
    """Broadcast scalar delta/Delta/TE to arrays the length of reference_array."""
    if delta is None:
        delta_ = delta
    elif isinstance(delta, (float, int)):
        delta_ = np.tile(delta, len(reference_array))
    else:
        delta_ = delta.copy()
    if Delta is None:
        Delta_ = Delta
    elif isinstance(Delta, (float, int)):
        Delta_ = np.tile(Delta, len(reference_array))
    else:
        Delta_ = Delta.copy()
    if TE is None:
        TE_ = TE
    elif isinstance(TE, (float, int)):
        TE_ = np.tile(TE, len(reference_array))
    else:
        TE_ = TE.copy()
    return delta_, Delta_, TE_



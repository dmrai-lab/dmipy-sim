"""The GPU tier's stall guards (tools/gpu_tier.py) run without Modal, a card or the network."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_tier", ROOT / "tools" / "gpu_tier.py")
gpu_tier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_tier)


def test_a_silent_command_is_killed_after_the_stall_window(capsys):
    started = time.monotonic()
    rc, stalled = gpu_tier.run_streamed([sys.executable, "-u", "-c", "import time; print('up'); time.sleep(60)"], ROOT, 1.0)
    assert stalled and rc == 1
    assert time.monotonic() - started < 15
    assert "up" in capsys.readouterr().out


def test_a_command_that_keeps_printing_runs_to_its_end(capsys):
    code = "import time\nfor i in range(6):\n    print(i); time.sleep(.4)"
    rc, stalled = gpu_tier.run_streamed([sys.executable, "-u", "-c", code], ROOT, 1.0)
    assert (rc, stalled) == (0, False)
    assert "5" in capsys.readouterr().out


class _Call:
    """A call that has its result ``delay`` seconds after creation (never, when ``result`` is None)."""

    def __init__(self, result=None, delay=0.0):
        self.result, self.ready, self.cancelled = result, time.monotonic() + delay, False

    def get(self, timeout):
        time.sleep(timeout)
        if self.result is None or time.monotonic() < self.ready:
            raise TimeoutError
        return self.result

    def cancel(self):
        self.cancelled = True


def test_a_call_whose_container_never_starts_is_cancelled_at_the_start_window():
    call = _Call()
    res, card_s, basis = gpu_tier.wait_for_result(call, {}, start_s=0.3, total_s=60, poll_s=0.1, timeout_error=TimeoutError)
    assert call.cancelled and res["rc"] == 1 and (card_s, basis) == (0.0, "none")


def test_a_started_call_is_waited_for_and_its_card_seconds_are_the_functions_own():
    call = _Call(dict(rc=0, tail="ok", seconds=312.0), delay=0.2)
    res, card_s, basis = gpu_tier.wait_for_result(call, {"started": time.time()}, start_s=0.1, total_s=60, poll_s=0.1,
                                                  timeout_error=TimeoutError)
    assert not call.cancelled and res["rc"] == 0 and (card_s, basis) == (312.0, "card")


def test_a_started_call_that_never_returns_is_cancelled_at_the_total_with_an_upper_bound():
    call = _Call()
    res, card_s, basis = gpu_tier.wait_for_result(call, {"started": time.time() - 5}, start_s=0.1, total_s=0.3, poll_s=0.1,
                                                  timeout_error=TimeoutError)
    assert call.cancelled and basis == "upper-bound" and card_s >= 5

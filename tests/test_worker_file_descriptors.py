"""Workers must not run the process out of file descriptors.

The first full run with `num_workers: 6` trained a whole epoch and then died at
the first validation pass:

    File ".../torch/multiprocessing/reductions.py", rebuild_storage_fd
    RuntimeError: received 0 items of ancdata

Nothing was wrong with the model or the data. Under the Linux default sharing
strategy every tensor a worker sends travels as an open file descriptor, and a
study item here is a *list* of per-series tensors, so one batch is a dozen or
more of them. Several batches in flight across several workers overrun the
process limit, and the failure lands hours in rather than at startup -- it needs
enough tensors alive at once, which the first validation pass provided.

`seed_worker` now switches each worker to the `file_system` strategy, which
names the shared memory instead of passing a descriptor. These tests run a real
loader under a deliberately small limit: one arm with the package's worker
setup, one arm with the seeding it had before, so the failure this prevents is
demonstrated rather than asserted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from rsna_knee.runtime import relax_file_descriptor_limit, resolve_runtime, share_tensors_by_file  # noqa: E402

PROBE = Path(__file__).parent / "_fd_probe.py"
SMALL_LIMIT = "256"


def _run_probe(arm: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = (
        source + os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else source
    )
    return subprocess.run(
        [sys.executable, str(PROBE), arm, SMALL_LIMIT],
        capture_output=True, text=True, timeout=600, env=environment,
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="descriptor passing is a Linux limit")
def test_a_loader_with_workers_survives_a_small_file_descriptor_limit():
    """The property the fix buys: many tensors in flight, few descriptors."""
    result = _run_probe("fixed")
    assert result.returncode == 0, (
        "a loader using this package's worker setup ran out of file descriptors, "
        f"which is the ancdata crash:\n{result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="descriptor passing is a Linux limit")
def test_the_same_loop_without_the_fix_exhausts_them():
    """The control arm. Without it the test above proves only that 200 items load.

    If this ever starts passing, the loop stopped holding enough tensors at once
    to reach the limit, and the test above is no longer measuring anything.
    """
    result = _run_probe("unfixed")
    assert result.returncode == 1, (
        "the pre-fix worker setup completed under a 256-descriptor limit, so this "
        f"pair of tests no longer demonstrates anything:\n{result.stdout}"
    )
    assert "FAIL" in result.stdout


def test_the_worker_init_function_switches_the_sharing_strategy():
    """Set in the worker, because the worker is the process that sends tensors.

    PyTorch does not carry the parent's choice into a `spawn`-ed worker, so
    setting it only in `resolve_runtime` would leave the senders on the default.
    """
    before = torch.multiprocessing.get_sharing_strategy()
    try:
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
        from rsna_knee.runtime import seed_worker

        seed_worker(0)
        assert torch.multiprocessing.get_sharing_strategy() == "file_system"
    finally:
        torch.multiprocessing.set_sharing_strategy(before)


def test_a_run_with_workers_reports_what_it_changed():
    """Both settings are global side effects, so the run states them in its log."""
    before = torch.multiprocessing.get_sharing_strategy()
    try:
        runtime = resolve_runtime({"num_workers": 4, "device": "cpu"})
        assert runtime.sharing_strategy == "file_system"
        assert runtime.open_file_limit > 0
        assert "sharing=file_system" in runtime.describe()
        assert "open_files=" in runtime.describe()
    finally:
        torch.multiprocessing.set_sharing_strategy(before)


def test_a_single_process_run_is_left_exactly_as_it_was():
    """`num_workers: 0` shares nothing, so it keeps the stock settings.

    The 0.834998 baseline was measured that way, and a run that changes nothing
    else is the point of the comparison.
    """
    before = torch.multiprocessing.get_sharing_strategy()
    try:
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
        runtime = resolve_runtime({"num_workers": 0, "device": "cpu"})
        assert runtime.sharing_strategy is None
        assert runtime.open_file_limit == 0
        assert torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
        assert "sharing=" not in runtime.describe()
    finally:
        torch.multiprocessing.set_sharing_strategy(before)


def test_raising_the_open_file_limit_never_lowers_it():
    """Belt and braces beside the strategy, and it must not make things worse."""
    import resource

    soft_before, hard_before = resource.getrlimit(resource.RLIMIT_NOFILE)
    before, after = relax_file_descriptor_limit()
    soft_after, hard_after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert before == soft_before
    assert after >= before
    assert soft_after == after
    assert hard_after == hard_before, "the hard limit is not ours to change"


def test_switching_the_strategy_reports_the_one_in_force():
    before = torch.multiprocessing.get_sharing_strategy()
    try:
        assert share_tensors_by_file() == "file_system"
    finally:
        torch.multiprocessing.set_sharing_strategy(before)

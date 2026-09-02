"""The supervised-cell count is a guard, and a deliberate teacher change is not.

`build_report_only_surface` pins how many cells the report labels supervise, so
an export edited or replaced between runs cannot silently change what a model
trains on. That guard earned its place: the export is five files on disk that
nothing else fingerprints at load time.

It also blocked the one thing worth doing with it. The teacher audit found that
filling only `negated` cells beats every other export measured -- balanced
accuracy 0.8043 against 0.7378 -- and that export supervises 25,524 cells rather
than 34,010, so the guard refused it with a message naming neither number.

The pin is now declarable. Changing the teacher means stating the count, which
an accident cannot do.
"""

from __future__ import annotations

import inspect

import pytest

from rsna_knee.checkpoints import EXPECTED_BASE_CELLS
from rsna_knee.training.loop import train
from rsna_knee.training.surface import build_report_only_surface


def test_the_default_is_still_the_frozen_count():
    """Every completed run used this. A permissive default would waive the guard."""
    default = inspect.signature(build_report_only_surface).parameters["expected_cells"].default
    assert default == EXPECTED_BASE_CELLS == 34010


def test_train_defaults_to_declaring_nothing_and_gets_the_frozen_count():
    default = inspect.signature(train).parameters["expected_cells"].default
    assert default is None, "train must not carry a second copy of the constant"


def test_the_message_names_both_counts_and_how_to_proceed():
    """The old message was four words and no numbers, on a guard whose whole job
    is to tell you what changed."""
    source = inspect.getsource(build_report_only_surface)
    assert "B48 weak supervision surface changed" not in source
    for fragment in ("observed_cells:,", "--expected-cells", "on purpose"):
        assert fragment in source, f"the error message no longer says {fragment!r}"


def test_the_declared_count_reaches_the_supervision_record():
    """Which surface trained a model must be recoverable from the model."""
    source = inspect.getsource(build_report_only_surface)
    assert '"expected_cells": int(expected_cells)' in source


@pytest.mark.parametrize("declared", [25524, 34010])
def test_the_guard_compares_against_what_was_declared(declared):
    """Not against the constant, or declaring a number would achieve nothing."""
    source = inspect.getsource(build_report_only_surface)
    assert "observed_cells != int(expected_cells)" in source
    assert "!= EXPECTED_BASE_CELLS" not in source
    assert isinstance(declared, int)


def test_the_base_checkpoints_own_pin_is_untouched():
    """Two different questions shared one constant. Only one of them moved.

    The base checkpoint's recorded training-cell count says what the frozen base
    model was trained on and must not become negotiable; the surface count says
    what this run trains on.
    """
    from rsna_knee.checkpoints import require_base_checkpoint  # noqa: PLC0415

    source = inspect.getsource(require_base_checkpoint)
    assert "EXPECTED_BASE_CELLS" in source
    assert "expected_cells" not in inspect.signature(require_base_checkpoint).parameters

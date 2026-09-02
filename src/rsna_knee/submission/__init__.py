"""Turning a trained checkpoint into a competition submission.

A model trained by this package could not be submitted from it. Every inference
launcher lived in the archive this package was extracted from, so a run here had
to be carried back there to be scored -- through a path that failed three hidden
runs before it worked, each time on something the three visible example studies
could not reveal.

This is that path, rebuilt where the model is. It keeps the two properties the
archive learned the hard way, and adds one it never had:

* **one TTA view at a time**, built from a series normalised once, so a
  fourteen-series study costs roughly 0.6 GiB rather than 3.2;
* **the runtime projection is telemetry**, never an exception, because a single
  slow early study in a 650-study shard can otherwise forecast past the budget
  and kill a run that would have finished;
* **it runs on one GPU**. The archive's launcher refuses to start below two, so
  the submission path could only ever be exercised on Kaggle itself. Here a
  smoke test on the local card is possible, which is the only way to find a
  defect before it costs a submission slot.
"""
from __future__ import annotations

from .launcher import generate_submission
from .runner import ON_UNREADABLE_FALLBACK, ON_UNREADABLE_RAISE, infer_shard
from .test_surface import build_test_surface, load_test_csv
from .views import build_study_view, load_normalised_study

__all__ = [
    "ON_UNREADABLE_FALLBACK",
    "ON_UNREADABLE_RAISE",
    "build_study_view",
    "build_test_surface",
    "generate_submission",
    "infer_shard",
    "load_normalised_study",
    "load_test_csv",
]

"""The package's shape, which is the thing this repository exists to fix.

The archive this code came from had 270 modules named after the experiment that
first needed them: `b42_constant_area_aspect_sparse_mil`,
`b50_adapted_hierarchy_training`. Reading it meant knowing the history. Half of
what a run imported was ancestry nothing called.

These tests hold the new shape in place: every module imports, nothing runs on
import, no public name carries an experiment number, and the one thing a rename
must never touch -- checkpoint keys -- is pinned by a separate suite.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "rsna_knee"
NUMBERED = re.compile(r"(?:^|_)[Bb]\d{1,2}(?:_|$)|^[Bb]\d{1,2}[A-Z]")


def module_names() -> list[str]:
    import rsna_knee

    return sorted(m.name for m in pkgutil.walk_packages(rsna_knee.__path__, "rsna_knee."))


def source_files() -> list[Path]:
    return sorted(p for p in SOURCE.rglob("*.py") if p.name != "__init__.py")


def test_every_module_imports():
    failures = []
    for name in module_names():
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{name}: {type(error).__name__}: {error}")
    assert not failures, "\n".join(failures)


def test_no_module_name_carries_an_experiment_number():
    offenders = [p.name for p in source_files() if NUMBERED.search(p.stem)]
    assert not offenders, f"modules still named after an experiment: {offenders}"


def test_no_public_name_carries_an_experiment_number():
    """A caller should never have to know which experiment first needed a thing."""
    offenders = []
    for path in source_files():
        for node in ast.parse(path.read_text()).body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if not name.startswith("_") and NUMBERED.search(name):
                    offenders.append(f"{path.relative_to(SOURCE)}::{name}")
    assert not offenders, f"public names still numbered: {offenders}"


def test_importing_a_module_runs_nothing():
    """A module that does work on import makes the package unusable as a library.

    Definitions, imports and plain constants are fine. A call at the top level
    is not, unless it is building a constant this package needs at import time.
    """
    # Building a constant from another constant is fine: `N_TARGETS = len(TARGETS)`
    # is a definition, not work.
    allowed_calls = {"len", "AugmentationPolicy", "RuntimeConfig", "dict", "tuple", "frozenset"}
    offenders = []
    for path in source_files():
        for node in ast.parse(path.read_text()).body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
            ):
                continue
            if isinstance(node, ast.Assign):
                for call in ast.walk(node.value):
                    if isinstance(call, ast.Call):
                        called = getattr(call.func, "id", getattr(call.func, "attr", ""))
                        if called not in allowed_calls:
                            offenders.append(f"{path.relative_to(SOURCE)}: {called}() at import")
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # a docstring
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
                continue
            offenders.append(f"{path.relative_to(SOURCE)}: {type(node).__name__} at import")
    assert not offenders, "\n".join(offenders)


def test_the_layout_says_what_each_part_does():
    """Directory names carry the map, so a reader does not need the history."""
    expected = {
        "constants.py", "runtime.py", "evaluation.py", "checkpoints.py",
        "imaging/dicom_io.py", "imaging/dicom_metadata.py", "imaging/physical_scale.py",
        "imaging/crop_policy.py", "imaging/triplets.py",
        "data/tables.py", "data/series_policy.py", "data/coverage.py", "data/labels.py",
        "data/splits.py", "data/slice_selection.py", "data/dataset.py", "data/augmentation.py",
        "model/encoder.py", "model/study_hierarchy.py", "model/sparse_head.py",
        "model/study_model.py",
        "training/supervision.py", "training/losses.py", "training/memory.py",
        "training/surface.py", "training/loop.py",
    }
    found = {str(p.relative_to(SOURCE)) for p in source_files()}
    assert found == expected, f"unexpected: {found - expected}, missing: {expected - found}"


def test_every_module_has_a_docstring_that_says_what_it_is_for():
    thin = []
    for path in source_files():
        doc = ast.get_docstring(ast.parse(path.read_text()))
        if not doc or len(doc) < 40:
            thin.append(str(path.relative_to(SOURCE)))
    assert not thin, f"modules without a real docstring: {thin}"


def test_the_public_entry_points_are_importable_by_their_new_names():
    """The names a caller actually uses, pinned so a later tidy cannot break them."""
    from rsna_knee.data.augmentation import AugmentationPolicy, AugmentedStudyDataset
    from rsna_knee.data.coverage import require_dicom_coverage
    from rsna_knee.data.dataset import StudyDataset, collate_studies
    from rsna_knee.evaluation import macro_auc
    from rsna_knee.model.study_model import KneeStudyModel
    from rsna_knee.training.loop import preflight, train

    for thing in (
        AugmentationPolicy, AugmentedStudyDataset, require_dicom_coverage,
        StudyDataset, collate_studies, macro_auc, KneeStudyModel, preflight, train,
    ):
        assert thing is not None


def test_the_config_keys_the_code_reads_still_exist():
    """The config file is the frozen contract; the code reads it by key name.

    Those keys keep their original spelling on purpose -- they name values in a
    file that predates this repository, and renaming one would silently fall
    back to a default.
    """
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "training.yaml").read_text()
    )
    required = [
        "b7_rotation_deg", "b7_translate_frac", "b7_scale_jitter", "b7_gamma_jitter",
        "b7_bias_field_strength", "b7_noise_std", "b7_slice_dropout",
        "b37_grid_size", "b37_top_k", "b37_temperature", "b37_encoder_chunk_size",
        "b37_local_aux_weight",
    ]
    missing = [key for key in required if key not in config]
    assert not missing, f"the config no longer carries: {missing}"


@pytest.mark.parametrize("entry", ["rsna_knee.training.loop", "rsna_knee.data.coverage"])
def test_the_command_line_entry_points_have_a_main(entry):
    module = importlib.import_module(entry)
    assert callable(getattr(module, "main", None)), f"{entry} has no main()"

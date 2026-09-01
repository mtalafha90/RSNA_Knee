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


def test_every_relative_import_resolves_to_a_module_that_exists():
    """Including the lazy ones, which `import the module` cannot reach.

    A `from .x import y` inside a function only executes when that function is
    called, so a module can import perfectly while carrying an import that will
    fail an hour into a run. The migration copied function bodies verbatim, and
    three such imports still named modules that no longer exist -- two of them
    in a function the trainer calls on every run.
    """
    offenders = []
    for path in source_files():
        package = path.relative_to(SOURCE).parent.parts
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            # level 1 is this package, level 2 is its parent, and so on.
            base = list(package)[: len(package) - (node.level - 1)] if node.level > 1 else list(package)
            target = base + (node.module.split(".") if node.module else [])
            module = SOURCE.joinpath(*target).with_suffix(".py")
            package_init = SOURCE.joinpath(*target, "__init__.py")
            if not module.is_file() and not package_init.is_file():
                offenders.append(
                    f"{path.relative_to(SOURCE)}:{node.lineno} -> "
                    f"{'.' * node.level}{node.module or ''} (no such module)"
                )
    assert not offenders, "\n".join(offenders)


def test_the_lazy_imports_actually_execute():
    """The behavioural half: call the functions that carry a deferred import.

    The structural test above would miss an import that resolves to a real
    module but asks it for a name it does not have.
    """
    import pandas as pd

    from rsna_knee.data.splits import load_selection_gate
    from rsna_knee.data.tables import backfill_series_metadata

    # Each must reach its own error or return, not an ImportError.
    with pytest.raises(FileNotFoundError):
        load_selection_gate("/nonexistent-gate")

    frame = pd.DataFrame([{
        "StudyInstanceUID": "s", "SeriesInstanceUID": "x",
        "Anatomical_Plane": "", "Fluid_Sensitive": None, "Fat_Suppression": None,
    }])
    _repaired, stats = backfill_series_metadata(frame, "/nonexistent-data", split="train")
    assert stats["rows_needing_metadata"] == 1


def test_the_artefact_defaults_are_consistent():
    """A default that names a file the setup script never creates is a trap."""
    from rsna_knee.training import loop

    assert loop.DEFAULT_CONFIG == "config/training.yaml"
    assert Path(loop.DEFAULT_CONFIG).parts[0] == "config"
    for default in (
        loop.DEFAULT_DATA_ROOT, loop.DEFAULT_LABELS_ROOT, loop.DEFAULT_SERIES_POLICY,
        loop.DEFAULT_BASE_CHECKPOINT, loop.DEFAULT_DOMAIN_SPLIT,
    ):
        assert default.startswith(loop.ARTEFACTS + "/"), f"{default} is outside {loop.ARTEFACTS}/"

    setup = (Path(__file__).resolve().parents[1] / "scripts" / "setup_artefacts.sh").read_text()
    for default in (
        loop.DEFAULT_LABELS_ROOT, loop.DEFAULT_SERIES_POLICY,
        loop.DEFAULT_BASE_CHECKPOINT, loop.DEFAULT_DOMAIN_SPLIT, loop.DEFAULT_DATA_ROOT,
    ):
        assert default in setup, f"setup_artefacts.sh never creates {default}"


def test_a_checkout_without_artefacts_says_what_is_missing():
    """argparse's required=True would force five paths onto every command line.

    Defaulting them means a set-up checkout runs bare -- and a checkout that is
    not set up must get one message naming everything it needs, not fail on
    them one at a time.
    """
    from rsna_knee.training import loop

    arguments = loop.build_argument_parser().parse_args([]) if hasattr(
        loop, "build_argument_parser"
    ) else None
    if arguments is None:  # the parser is built inside main()
        import argparse

        arguments = argparse.Namespace(
            config="nope.yaml", data_root="nope", labels_root="nope",
            series_policy="nope", base_checkpoint="nope", domain_split="nope",
        )
    with pytest.raises(SystemExit) as caught:
        loop.require_artefacts(arguments)

    message = str(caught.value)
    for flag in ("--data-root", "--labels-root", "--series-policy",
                 "--base-checkpoint", "--domain-split"):
        assert flag in message, f"{flag} is not named in the message"


def test_no_function_shadows_a_name_it_also_calls():
    """The rename hazard that a passing import cannot see.

    `b48_fill_artifacts` became `fill_artifacts`, which is also what the caller
    named the variable it assigned the result to:

        fill_artifacts = fill_artifacts(labels_root)

    Python binds the local for the whole function body, so the call on the right
    raises UnboundLocalError instead of reaching the import. The module still
    imports cleanly; the failure waits until the function runs, which for the
    trainer is after the gate, the checkpoint and the split have all loaded.

    Python's own symbol table decides what counts as a local here, rather than a
    guess about assignment.
    """
    import symtable

    offenders = []
    for path in source_files():
        source = path.read_text()
        imported = {
            alias.asname or alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        top = symtable.symtable(source, str(path), "exec")

        def visit(table):
            for child in table.get_children():
                if child.get_type() == "function":
                    called = {
                        node.func.id
                        for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    }
                    for symbol in child.get_symbols():
                        name = symbol.get_name()
                        if (
                            symbol.is_local()
                            and symbol.is_assigned()
                            and name in imported
                            and name in called
                        ):
                            offenders.append(
                                f"{path.relative_to(SOURCE)}: {child.get_name()}() "
                                f"assigns to {name!r}, which it also imports and calls"
                            )
                visit(child)

        visit(top)
    assert not offenders, "\n".join(offenders)


def test_the_trainer_reaches_its_first_artefact_check():
    """Walk `train()` far enough to prove its early body executes.

    It cannot be run to completion here: the label export must have exactly
    4,349 rows and match a pinned Phase-8 fingerprint, and the base checkpoint
    must carry a recorded arm. Those cannot be fabricated, and fabricating them
    would defeat their purpose.

    What this does establish is that `train()` is entered, its arguments are
    validated, and it fails where a missing gate should make it fail -- rather
    than on a NameError or an UnboundLocalError in code that was never executed.
    """
    from rsna_knee.training.loop import train

    with pytest.raises(FileNotFoundError, match="selection gate"):
        train(
            {"b37_grid_size": 6, "b37_top_k": 8},
            data_root="/nonexistent-data",
            labels_root="/nonexistent-labels",
            series_policy_path="/nonexistent-policy.json",
            base_checkpoint="/nonexistent-base.pt",
            domain_split="/nonexistent-gate",
            epochs=1,
        )


def test_the_trainer_validates_its_arguments_before_touching_disk():
    """A bad epoch count or stage count should not cost a checkpoint load."""
    from rsna_knee.training.loop import train

    common = dict(
        data_root="/nonexistent", labels_root="/nonexistent",
        series_policy_path="/nonexistent", base_checkpoint="/nonexistent",
        domain_split="/nonexistent",
    )
    with pytest.raises(ValueError, match="at least one epoch"):
        train({}, epochs=0, **common)
    with pytest.raises(ValueError, match="stages must be"):
        train({}, epochs=1, encoder_trainable_stages=99, **common)

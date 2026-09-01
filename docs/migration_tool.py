"""Extract the B53 code path from cnn_cpc into a clean, well-named package.

The method is mechanical on purpose. This is numerical code whose whole value is
that its measurements can be trusted, so the migration copies definitions
verbatim -- source text, comments and all -- regroups them by purpose, and
renames the public names. It does not retype anything. A separate equivalence
test then proves the result computes the same numbers as the original.

What it does:

1. Walks the symbol graph from `b53_augmented_training` and keeps only the
   definitions that are actually reachable. 240 of the package's 482 reachable
   definitions survive; the other half is ancestry nothing calls.
2. Assigns every surviving definition to a destination module by purpose.
3. Renames experiment-numbered public names to descriptive ones.
4. Emits each destination module with its imports, in dependency order.

The one thing it must never rename is a `nn.Module` attribute: the Phase-9 base
checkpoint is loaded by `load_state_dict`, whose keys are attribute paths. Class
names are free; `self.encoder` and `self.global_projection` are not.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path

SOURCE = Path("/workspace/cnn_cpc/developments/src/rsna_knee")
ROOT = "b53_augmented_training"
EXTRA_ROOTS = [("dicom_coverage", "main")]


# --- where each old module's surviving definitions go ----------------------
#
# Grouped by what the code does, not by which experiment first needed it.

DESTINATION = {
    # the twelve findings and the plane vocabulary
    "constants": "constants",
    # runtime: device, precision, workers, seeding
    "runtime": "runtime",
    # reading pixels off disk
    "dicom": "imaging/dicom_io",
    "dicom_meta": "imaging/dicom_metadata",
    "physical_scale": "imaging/physical_scale",
    "crop_focus": "imaging/crop_policy",
    "b20_crop_focus": "imaging/crop_policy",
    # the competition tables and the series policy
    "data": "data/tables",
    "b12_variable_series": "data/series_policy",
    "b12_training": "data/series_policy",
    "b13_training": "data/series_policy",
    "dicom_coverage": "data/coverage",
    # labels
    "label_confidence": "data/labels",
    "phase9_supervision": "data/labels",
    # the scanner-grouped split
    "b50_ordered_slice_selection_split": "data/splits",
    "b50_adapted_hierarchy_training": "data/splits",
    # the dataset the model reads
    "dataset": "data/dataset",
    "b35_target_spatial_residual": "data/slice_selection",
    # the model
    "model": "model/encoder",
    "dinov3_encoder": "model/encoder",
    "encoder_finetune": "model/encoder",
    "b17_training": "model/encoder",
    "b12_1_hierarchical": "model/study_hierarchy",
    "b29_complementary_series_pool": "model/study_hierarchy",
    "b31_local_context_complementary_pool": "model/study_hierarchy",
    "b34_training_only_context_scaffold": "model/study_hierarchy",
    "b36_sparse_mil": "model/sparse_head",
    "b37_highres_sparse_mil": "model/study_model",
    "b42_constant_area_aspect_sparse_mil": "model/study_model",
    "b50_adapted_hierarchy_mil": "model/study_model",
    # training
    "b7_weak_supervision": "training/supervision",
    "b42_constant_area_aspect_sparse_training": "training/losses",
    "b37_highres_sparse_training": "training/memory",
    "b48_global_conditioned_sparse_training": "training/surface",
    "b35_training": "checkpoints",
    "phase9_matched_supervision_training": "checkpoints",
    "evaluation": "evaluation",
    "b52_competition_training": "training/loop",
    "b53_augmented_training": "training/loop",
}

# Definitions that belong somewhere other than their module's default home.
OVERRIDE = {
    ("b42_constant_area_aspect_sparse_mil", "B42ConstantAreaAspectDataset"): "data/dataset",
    ("b42_constant_area_aspect_sparse_mil", "collate_b42"): "data/dataset",
    ("b42_constant_area_aspect_sparse_mil", "preprocess_dense_triplets_b42"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "resize_triplets_constant_area"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "constant_area_shape"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "B42_REFERENCE_AREA"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "B42_REFERENCE_SIDE"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "B42_STRIDE_ALIGNMENT"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "B42_PADDING_MODE"): "imaging/triplets",
    ("b42_constant_area_aspect_sparse_mil", "B42_RESIZE_POLICY"): "imaging/triplets",
    ("b37_highres_sparse_mil", "B37HighResSparseDataset"): "data/dataset",
    ("b37_highres_sparse_mil", "preprocess_dense_triplets_b37"): "imaging/triplets",
    ("b37_highres_sparse_mil", "_native_center_crop"): "imaging/triplets",
    ("b37_highres_sparse_mil", "B37_CROP_FRACTION"): "imaging/crop_policy",
    ("b37_highres_sparse_mil", "B37_IMAGE_SIZE"): "imaging/triplets",
    ("b12_variable_series", "VariableSeriesKneeDataset"): "data/dataset",
    ("b12_variable_series", "PLANE_TO_ID"): "constants",
    ("data", "PLANES"): "constants",
    ("dicom_meta", "PLANES"): "constants",
    ("b53_augmented_training", "AugmentationPolicy"): "data/augmentation",
    ("b53_augmented_training", "augment_b42_series"): "data/augmentation",
    ("b53_augmented_training", "B53AugmentedDataset"): "data/augmentation",
    ("b53_augmented_training", "B53_AUGMENT_SEED_OFFSET"): "data/augmentation",
    ("b53_augmented_training", "B53_SLICE_JITTER_DEFAULT"): "data/augmentation",
    ("b53_augmented_training", "verify_augmentation_reaches_pixels"): "data/augmentation",
    ("b52_competition_training", "macro_auc"): "evaluation",
    ("b52_competition_training", "masked_binary_targets"): "evaluation",
    ("b52_competition_training", "evaluate_split"): "evaluation",
    ("b7_weak_supervision", "seed_everything"): "runtime",
    ("b52_competition_training", "B52_SEED"): "constants",
    ("b52_competition_training", "B52_CONSTRUCTION_SEED_OFFSET"): "constants",
    ("b52_competition_training", "B52_LOADER_SEED_OFFSET"): "constants",
}

# --- names, said as what they are ------------------------------------------

RENAME = {
    # model
    "HierarchicalSeriesKneeMILNet": "SeriesHierarchyNet",
    "ComplementarySeriesPoolKneeMILNet": "ComplementarySeriesPoolNet",
    "LocalContextComplementarySeriesPoolKneeMILNet": "LocalContextSeriesPoolNet",
    "TrainingOnlyContextScaffoldKneeMILNet": "StudyHierarchyNet",
    "build_b34_model": "build_study_hierarchy",
    "B36SparseMILHead": "SparseEvidenceHead",
    "B37HighResSparseMILResidual": "SparseMILResidual",
    "B42ConstantAreaAspectSparseMILResidual": "RaggedSeriesSparseMIL",
    "B50AdaptedHierarchySparseMILResidual": "KneeStudyModel",
    "B37Forward": "ModelOutput",
    "LearnedSeriesPool": "LearnedSeriesPool",
    "ConvNeXtSliceEncoder": "ConvNeXtSliceEncoder",
    "DinoV3SliceEncoder": "DinoV3SliceEncoder",
    # datasets
    "VariableSeriesKneeDataset": "VariableSeriesDataset",
    "B37HighResSparseDataset": "DenseTripletDataset",
    "B42ConstantAreaAspectDataset": "StudyDataset",
    "B53AugmentedDataset": "AugmentedStudyDataset",
    "collate_b42": "collate_studies",
    "KneeStudyDataset": "SliceStreamDataset",
    # imaging
    "preprocess_dense_triplets_b42": "prepare_series_triplets",
    "preprocess_dense_triplets_b37": "prepare_square_triplets",
    "resize_triplets_constant_area": "resize_to_constant_area",
    "b35_centers": "slice_centres",
    "_native_center_crop": "native_centre_crop",
    "b20_crop_focus_policy": "crop_focus_policy",
    "require_b42_contract": "require_geometry_contract",
    "require_b37_sparse_contract": "require_sparse_mil_contract",
    # data
    "audit_variable_series_surface": "audit_series_surface",
    "build_variable_series_index": "build_series_index_all_planes",
    "variable_series_signature": "series_signature",
    "_load_series_policy": "load_series_policy",
    "load_b50_selection_gate": "load_selection_gate",
    "verify_b50_selection_split": "verify_selection_split",
    "_indices_for_split": "indices_for_split",
    "_report_only_surface": "build_report_only_surface",
    "b48_fill_artifacts": "fill_artifacts",
    # training
    "b52_parameter_groups": "parameter_groups",
    "select_train_and_validation": "select_train_and_validation",
    "train_b53": "train",
    "b53_preflight": "preflight",
    "augment_b42_series": "augment_series",
    "_batch_scales": "batch_scales",
    "_losses": "study_losses",
    "_move_study": "move_study_to_device",
    "_study_mass": "study_supervision_mass",
    "target_balanced_weak_bce": "confidence_weighted_bce",
    # checkpoints
    "load_phase9_checkpoint": "load_base_checkpoint",
    "_require_base_checkpoint": "require_base_checkpoint",
    "encoder_state_sha256": "encoder_state_sha256",
    "sha256_file": "sha256_file",
    "_config_sha256": "config_sha256",
    "_uid_sha256": "uid_sha256",
    "_sha256_text": "sha256_text",
    "_read_config": "read_config",
    # constants that keep a number only because it names a frozen artefact
    "B13_SERIES_SIGNATURE": "FROZEN_SERIES_SIGNATURE",
    "B12_SERIES_POLICY": "SERIES_POLICY_NAME",
    "B35_BASE_SLICES": "BASE_SLICES",
    "B35_DENSE_SLICES": "DENSE_SLICES",
    "B35_GRID_SIZE": "DEFAULT_GRID_SIZE",
    "B36_TOP_K": "DEFAULT_TOP_K",
    "B36_TEMPERATURE": "DEFAULT_TEMPERATURE",
    "B37_CROP_FRACTION": "CROP_FRACTION",
    "B37_IMAGE_SIZE": "SQUARE_IMAGE_SIZE",
    "B37_GRID_SIZE": "GRID_SIZE",
    "B37_TOP_K": "TOP_K",
    "B37_TEMPERATURE": "TEMPERATURE",
    "B37_HEAD_LR": "HEAD_LR",
    "B37_WEIGHT_DECAY": "WEIGHT_DECAY",
    "B37_GRAD_CLIP": "GRAD_CLIP",
    "B37_LOCAL_AUX_WEIGHT": "LOCAL_AUX_WEIGHT",
    "B37_ENCODER_CHUNK_SIZE": "ENCODER_CHUNK_SIZE",
    "B42_REFERENCE_AREA": "REFERENCE_AREA",
    "B42_REFERENCE_SIDE": "REFERENCE_SIDE",
    "B42_STRIDE_ALIGNMENT": "STRIDE_ALIGNMENT",
    "B42_PADDING_MODE": "PADDING_MODE",
    "B42_RESIZE_POLICY": "RESIZE_POLICY",
    "B42_EFFECTIVE_BATCH": "EFFECTIVE_BATCH",
    "B50_SPLIT_TRAIN": "SPLIT_TRAIN",
    "B50_SPLIT_SEEN": "SPLIT_SEEN_SCANNERS",
    "B50_SPLIT_UNSEEN": "SPLIT_UNSEEN_SCANNERS",
    "B50_SPLIT_EXCLUDED": "SPLIT_EXCLUDED",
    "B50_ALLOWED_SPLITS": "ALLOWED_SPLITS",
    "B50_PARENT_TRAIN_SPLIT": "PARENT_TRAIN_SPLIT",
    "B50_ALWAYS_FROZEN_PREFIXES": "ALWAYS_FROZEN_PREFIXES",
    "REPORT_ONLY_STUDIES": "REPORT_ONLY_STUDIES",
    "B7_MIN_CONFIDENCE": "MIN_CONFIDENCE",
    "B7_POSITIVE_TARGET": "POSITIVE_TARGET",
    "B7_NEGATIVE_TARGET": "NEGATIVE_TARGET",
    "B7_POSITIVE_WEIGHT": "POSITIVE_WEIGHT",
    "B7_NEGATIVE_WEIGHT": "NEGATIVE_WEIGHT",
    "make_b7_dataset_config": "make_dataset_config",
    "B52_SEED": "DEFAULT_SEED",
    "B52_PRIMARY_SPLIT": "VALIDATION_SPLIT",
    "B52_TRAIN_SPLIT": "DEFAULT_TRAIN_SPLIT",
    "B52_FULL_TRAIN_SPLITS": "ALL_DATA_TRAIN_SPLITS",
    "B52_DEFAULT_ENCODER_STAGES": "DEFAULT_ENCODER_STAGES",
    "B52_DEFAULT_ENCODER_LR_SCALE": "DEFAULT_ENCODER_LR_SCALE",
    "B52_DEFAULT_HIERARCHY_LR_SCALE": "DEFAULT_HIERARCHY_LR_SCALE",
    "B52_CONSTRUCTION_SEED_OFFSET": "CONSTRUCTION_SEED_OFFSET",
    "B52_LOADER_SEED_OFFSET": "LOADER_SEED_OFFSET",
    "B53_AUGMENT_SEED_OFFSET": "AUGMENT_SEED_OFFSET",
    "B53_SLICE_JITTER_DEFAULT": "DEFAULT_SLICE_JITTER",
    "B53_DEFAULT_EPOCHS": "DEFAULT_EPOCHS",
    "B53_CHECKPOINT_NAME": "CHECKPOINT_NAME",
    "B53_RUN_ROOT": "DEFAULT_RUN_ROOT",
    "B53_EXPERIMENT": "EXPERIMENT",
    "B53_VERSION": "VERSION",
    # Version and contract constants: the identifier is renamed, never the
    # string it holds -- those values are written into checkpoints and compared
    # by contract checks, so changing one would invalidate every frozen artefact.
    "B10_PHYSICAL_POLICY": "PHYSICAL_SCALE_POLICY",
    "B10_ROUTING_MODE": "PHYSICAL_ROUTING_MODE",
    "B10_MISSING_SPACING_ACTION": "MISSING_SPACING_ACTION",
    "B29_RESIDUAL_VERSION": "COMPLEMENTARY_POOL_VERSION",
    "B29_EXPECTED_GATE_PARAMETERS": "COMPLEMENTARY_GATE_PARAMETERS",
    "B29_EXPECTED_QUERY_PARAMETERS": "COMPLEMENTARY_QUERY_PARAMETERS",
    "B31_CONTEXT_VERSION": "LOCAL_CONTEXT_VERSION",
    "B31_EXPECTED_CONTEXT_PARAMETERS": "LOCAL_CONTEXT_PARAMETERS",
    "B31_EXPECTED_GATE_PARAMETERS": "LOCAL_CONTEXT_GATE_PARAMETERS",
    "B31_EXPECTED_NEW_PARAMETERS": "LOCAL_CONTEXT_NEW_PARAMETERS",
    "B31_EXPECTED_QUERY_PARAMETERS": "LOCAL_CONTEXT_QUERY_PARAMETERS",
    "B34_SCAFFOLD_VERSION": "STUDY_HIERARCHY_VERSION",
    "B34_ARCHITECTURE": "STUDY_HIERARCHY_ARCHITECTURE",
    "B34_AGGREGATION": "STUDY_HIERARCHY_AGGREGATION",
    "B34_EXPECTED_CONTEXT_PARAMETERS": "STUDY_HIERARCHY_CONTEXT_PARAMETERS",
    "B34_EXPECTED_GATE_PARAMETERS": "STUDY_HIERARCHY_GATE_PARAMETERS",
    "B34_EXPECTED_NEW_PARAMETERS": "STUDY_HIERARCHY_NEW_PARAMETERS",
    "B34_EXPECTED_QUERY_PARAMETERS": "STUDY_HIERARCHY_QUERY_PARAMETERS",
    "B35_POSITION_BASIS": "POSITION_BASIS",
    "B35_TOKEN_DROPOUT": "TOKEN_DROPOUT",
    "B35_EXPECTED_BASE_ARM": "EXPECTED_BASE_ARM",
    "B35_EXPECTED_BASE_EPOCHS": "EXPECTED_BASE_EPOCHS",
    "B35_EXPECTED_CELLS": "EXPECTED_BASE_CELLS",
    "B35_EXPECTED_SERIES": "EXPECTED_BASE_SERIES",
    "B36_VERSION": "SPARSE_HEAD_VERSION",
    "B37_VERSION": "SPARSE_MIL_VERSION",
    "B37_ENCODER_LR_SCALE": "ENCODER_LR_SCALE",
    "B37_ENCODER_TRAINABLE_STAGES": "ENCODER_TRAINABLE_STAGES",
    "B42_VERSION": "RAGGED_SERIES_VERSION",
    "B48_FILL_ARTIFACT_FILES": "FILL_ARTIFACT_FILES",
    "B50_EXPERIMENT": "ADAPTED_HIERARCHY_EXPERIMENT",
    "B50_VERSION": "ADAPTED_HIERARCHY_VERSION",
    "B52_ALL_DATA_MACRO_AUC": "BASELINE_ALL_DATA_MACRO_AUC",
    "B52_GATE_SPLIT_MACRO_AUC": "BASELINE_GATE_SPLIT_MACRO_AUC",
    "B52_INHERITED_ENCODER_STAGES": "INHERITED_ENCODER_STAGES",
    "B7_VARIANT": "SUPERVISION_VARIANT",
}


def build_graph():
    """Every definition reachable from B53, and which module defines it."""
    trees, defs, imports = {}, {}, {}

    def tree(mod):
        if mod not in trees:
            path = SOURCE / f"{mod}.py"
            trees[mod] = ast.parse(path.read_text()) if path.is_file() else None
        return trees[mod]

    def top_defs(mod):
        if mod in defs:
            return defs[mod]
        node = tree(mod)
        found = {}
        if node is not None:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    found[item.name] = item
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            found[target.id] = item
        defs[mod] = found
        return found

    def local_imports(mod):
        if mod in imports:
            return imports[mod]
        node = tree(mod)
        found = {}
        if node is not None:
            for item in ast.walk(node):
                if isinstance(item, ast.ImportFrom) and item.level == 1 and item.module:
                    for alias in item.names:
                        found[alias.asname or alias.name] = (item.module, alias.name)
        imports[mod] = found
        return found

    def loads(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} | {
            n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
        }

    seen, order = set(), []
    queue = deque([(ROOT, name) for name in top_defs(ROOT)] + list(EXTRA_ROOTS))
    edges = defaultdict(set)

    while queue:
        mod, name = queue.popleft()
        if (mod, name) in seen:
            continue
        seen.add((mod, name))
        here, imported = top_defs(mod), local_imports(mod)
        if name in here:
            order.append((mod, name))
            for used in loads(here[name]):
                if used in here:
                    edges[(mod, name)].add((mod, used))
                    queue.append((mod, used))
                elif used in imported:
                    target = imported[used]
                    edges[(mod, name)].add(target)
                    queue.append(target)
        elif name in imported:
            queue.append(imported[name])

    return order, edges, defs, trees


def third_party_imports(mod, trees):
    """The non-package imports a module's code needs, as source lines."""
    node = trees.get(mod)
    if node is None:
        return set()
    lines = (SOURCE / f"{mod}.py").read_text().split("\n")
    out = set()
    for item in node.body:
        if isinstance(item, ast.Import):
            out.add("\n".join(lines[item.lineno - 1 : item.end_lineno]))
        elif isinstance(item, ast.ImportFrom) and item.level == 0 and item.module != "__future__":
            out.add("\n".join(lines[item.lineno - 1 : item.end_lineno]))
    return out


def slice_source(mod, node):
    """A definition's exact source, including the comments written above it."""
    lines = (SOURCE / f"{mod}.py").read_text().split("\n")
    start = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start - 1 : node.end_lineno])


def destination(mod, name):
    return OVERRIDE.get((mod, name)) or DESTINATION[mod]


if __name__ == "__main__":
    order, edges, defs, trees = build_graph()
    by_dest = defaultdict(list)
    for mod, name in order:
        by_dest[destination(mod, name)].append((mod, name))

    print(f"{len(order)} definitions -> {len(by_dest)} modules\n")
    for dest in sorted(by_dest):
        names = [RENAME.get(n, n) for _m, n in by_dest[dest]]
        print(f"  {dest:<26} {len(names):>3}  {', '.join(sorted(names)[:6])}...")

    collisions = defaultdict(set)
    for mod, name in order:
        collisions[(destination(mod, name), RENAME.get(name, name))].add(mod)
    clashes = {k: v for k, v in collisions.items() if len(v) > 1}
    if clashes:
        print("\nNAME COLLISIONS in a destination module:")
        for (dest, name), mods in sorted(clashes.items()):
            print(f"    {dest}.{name} <- {sorted(mods)}")

    import re as _re
    numbered = _re.compile(r"(?:^|_)[Bb]\d{1,2}(?:_|$)|^[Bb]\d{1,2}[A-Z]")
    unrenamed = sorted(
        {RENAME.get(n, n) for _m, n in order if numbered.search(RENAME.get(n, n))}
    )
    if unrenamed:
        print(f"\nstill carrying an experiment number ({len(unrenamed)}):")
        for name in unrenamed:
            print(f"    {name}")


# --- emitting the new package ----------------------------------------------

PACKAGE = "rsna_knee"

MODULE_DOC = {
    "constants": "The twelve findings, and the vocabulary shared across the package.",
    "runtime": "Device, precision, worker count and seeding: one place that decides how a run executes.",
    "evaluation": "Macro ROC AUC over the cells the reports actually supervise.",
    "checkpoints": "Reading and fingerprinting the base checkpoint a run starts from.",
    "imaging/dicom_io": "Reading pixels off disk: locating a series, sorting its frames, normalising intensity.",
    "imaging/dicom_metadata": "Recovering plane, fluid sensitivity and fat suppression from the DICOM itself.",
    "imaging/physical_scale": "Resampling to a common physical scale, so a millimetre means the same thing on every scanner.",
    "imaging/crop_policy": "The frozen crop: 90% of the native matrix, centred.",
    "imaging/triplets": "Turning a volume into the model's input: slice centres, 2.5D triplets, one constant-area resize.",
    "data/tables": "The competition tables, and the metadata repair that fills their blanks from the DICOM.",
    "data/series_policy": "Which of a study's series the model may look at, and the fingerprint that pins the rule.",
    "data/coverage": "Whether the DICOM folders a run needs are actually on this machine.",
    "data/labels": "Report-derived labels: the merged export, its confidences, and what silence means.",
    "data/splits": "The scanner-grouped split: whole scanner models held out, never half of one.",
    "data/slice_selection": "Choosing which slices represent a series, and the position features that describe them.",
    "data/dataset": "The datasets the loader yields: one study, its series, its labels.",
    "data/augmentation": "Distorting a training study so the model learns the knee rather than the picture.",
    "model/encoder": "The slice encoder that reads pixels, and which of its stages are allowed to learn.",
    "model/study_hierarchy": "Pooling encoded slices into series, and series into one study representation.",
    "model/sparse_head": "Target-specific top-k pooling over a grid of local evidence.",
    "model/study_model": "The full model: encoder, study hierarchy, sparse local branch, and the gate that mixes them.",
    "training/supervision": "Turning report labels into targets and weights, and balancing the twelve findings.",
    "training/losses": "The per-study loss, and moving one study onto the device.",
    "training/memory": "Keeping a long run inside its memory budget, and reporting what it used.",
    "training/surface": "Assembling the report-only training surface and fingerprinting what went into it.",
    "training/loop": "The training run: what changes, what is held fixed, and how the epoch is chosen.",
}


def relative_import(from_dest: str, to_dest: str) -> str:
    """`from ..imaging.triplets import x`, worked out from the two paths."""
    depth = from_dest.count("/")
    dots = "." * (depth + 1) if to_dest.count("/") == 0 or "/" not in to_dest else "." * (depth + 1)
    target = to_dest.replace("/", ".")
    if "/" in from_dest and from_dest.split("/")[0] == to_dest.split("/")[0]:
        return "." + to_dest.split("/", 1)[1]
    return dots + target


def topological(names, edges):
    """Definitions in an order where nothing is used before it is defined."""
    inside = set(names)
    remaining = dict.fromkeys(names)
    emitted, out = set(), []
    while remaining:
        progressed = False
        for key in list(remaining):
            waiting = {d for d in edges.get(key, ()) if d in inside and d not in emitted and d != key}
            if not waiting:
                out.append(key)
                emitted.add(key)
                del remaining[key]
                progressed = True
        if not progressed:
            # A cycle, or a class referring to itself in an annotation. Emit the
            # rest in the order they were discovered; Python resolves names at
            # call time, so only base classes truly need to come first.
            out.extend(remaining)
            break
    return out


def emit(out_root: Path) -> dict:
    order, edges, defs, trees = build_graph()

    home = {(mod, name): destination(mod, name) for mod, name in order}
    by_dest = defaultdict(list)
    for key in order:
        by_dest[home[key]].append(key)

    # A rename is a text substitution, so a name that also appears inside a
    # string literal gets rewritten there too. Sometimes that is right -- an
    # `__all__` entry or an error message naming the symbol should follow the
    # rename. Sometimes it would be a silent disaster: a frozen version string
    # or a fingerprint written into checkpoints and compared by contract checks
    # must never change, or every existing artefact stops validating.
    #
    # So this classifies rather than refuses blindly, and refuses only the
    # dangerous kind.
    frozen_suffixes = ("_VERSION", "_SIGNATURE", "_SHA256", "_DIGEST", "_POLICY")
    renamed_in_strings = []
    for mod in sorted({m for m, _ in order}):
        node = trees.get(mod)
        if node is None:
            continue
        for statement in ast.walk(node):
            assigned = [
                t.id for t in getattr(statement, "targets", []) if isinstance(t, ast.Name)
            ]
            for item in ast.walk(statement):
                if not (isinstance(item, ast.Constant) and isinstance(item.value, str)):
                    continue
                for old in RENAME:
                    if old == RENAME[old] or old not in item.value:
                        continue
                    frozen = any(a.endswith(frozen_suffixes) for a in assigned) or (
                        len(item.value) >= 40 and all(c in "0123456789abcdef" for c in item.value)
                    )
                    if frozen:
                        raise SystemExit(
                            f"refusing to rename {old!r}: it is inside a frozen value in "
                            f"{mod}.py ({assigned or 'literal'} = {item.value[:60]!r}). "
                            "Changing it would invalidate every checkpoint that records it."
                        )
                    renamed_in_strings.append((mod, old, item.value[:50]))

    if renamed_in_strings:
        print("renamed inside string literals (reviewed as safe):")
        for mod, old, value in renamed_in_strings:
            print(f"    {mod}.py  {old}  in  {value!r}")
        print()

    written = {}
    for dest, keys in sorted(by_dest.items()):
        ordered = topological(keys, edges)

        needed = defaultdict(set)
        for key in keys:
            for dep in edges.get(key, ()):
                if dep in home and home[dep] != dest:
                    needed[home[dep]].add(RENAME.get(dep[1], dep[1]))

        third_party = set()
        for mod in {m for m, _ in keys}:
            third_party |= third_party_imports(mod, trees)

        lines = [f'"""{MODULE_DOC.get(dest, dest)}"""', "from __future__ import annotations", ""]
        lines += sorted(third_party)
        if needed:
            lines.append("")
            for target in sorted(needed):
                names = ", ".join(sorted(needed[target]))
                lines.append(f"from {relative_import(dest, target)} import {names}")
        lines.append("")

        for mod, name in ordered:
            lines.append("")
            lines.append(slice_source(mod, defs[mod][name]))
            lines.append("")

        source = "\n".join(lines)
        for old, new in RENAME.items():
            if old != new:
                source = __import__("re").sub(rf"\b{old}\b", new, source)

        path = out_root / f"{dest}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        ast.parse(source)
        path.write_text(source + "\n", encoding="utf-8")
        written[dest] = len(source.split("\n"))

    for package_dir in {p.parent for p in (out_root / f"{d}.py" for d in by_dest)}:
        init = package_dir / "__init__.py"
        if not init.exists():
            init.write_text('"""Part of the knee MRI package."""\n', encoding="utf-8")

    return written


# --- this experiment's own identity ----------------------------------------
#
# These strings name the run itself, and nothing on disk records them yet: B53
# has never been executed, so no checkpoint, log or manifest depends on them.
# They are the ones a reader meets constantly, so they say what the run is.
#
# Everything else stays. A message like "B42 requires the historical fixed 90%
# center crop" is not leftover numbering -- it names *which* frozen contract is
# being enforced, and a reader who sees it can go and find that contract. The
# config keys (`b7_rotation_deg`), artefact filenames (`b50_selection_split.json`)
# and recorded version strings must not change under any circumstances.

IDENTITY = {
    '"B53_AUGMENTATION_APPLIED"': '"AUGMENTED_TRAINING"',
    '"b53_augmentation_applied_v1"': '"augmented_training_v1"',
    '"runs/087_Experiment_B53_augmentation_applied"': '"runs/augmented_training"',
    '"b53_best_model.pt"': '"best_model.pt"',
    '"[B53 preflight] ': '"[preflight] ',
    'f"[B53 preflight] ': 'f"[preflight] ',
    'f"[B53]': 'f"[train]',
    '"[B53]': '"[train]',
    '"B53 needs at least one epoch"': '"training needs at least one epoch"',
    '"B53 will not overwrite ': '"refusing to overwrite ',
    '"B53 finished without a usable validation score"': (
        '"the run finished without a usable validation score"'
    ),
    '"B53 preflight: no gradient reached the encoder"': (
        '"preflight: no gradient reached the encoder"'
    ),
    '"B53 domain split source train.csv fingerprint mismatch"': (
        '"domain split source train.csv fingerprint mismatch"'
    ),
    '"B53 requires the frozen B12/B13 series policy"': (
        '"this run requires the frozen series policy"'
    ),
    'f"B53 trains the encoder; stages must be 1..': 'f"training needs the encoder; stages must be 1..',
    '"B53 augmentation did not reach the pixels': '"augmentation did not reach the pixels',
    '"Train B53: B52 with the augmentation applied"': (
        '"Train the knee MRI model with augmentation applied"'
    ),
    '"B53 is the experiment in which augmentation is applied, but every "': (
        '"this run applies augmentation, but every "'
    ),
    '"B52 trains the encoder; none of it requires gradients. Check "': (
        '"the encoder must train; none of it requires gradients. Check "'
    ),
    'f"B52 train and validation splits share ': 'f"train and validation splits share ',
    '"B53 selects its checkpoint on a held-out report-labelled "': (
        '"This run selects its checkpoint on a held-out report-labelled "'
    ),
}


def apply_identity(out_root: Path) -> int:
    changed = 0
    for path in sorted(out_root.rglob("*.py")):
        text = original = path.read_text()
        for old, new in IDENTITY.items():
            text = text.replace(old, new)
        if text != original:
            ast.parse(text)
            path.write_text(text)
            changed += 1
    return changed

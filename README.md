# RSNA Knee

Twelve binary findings per knee MRI study, scored as macro ROC AUC.

```bash
pip install -e ".[test]"

rsna-knee-coverage --data-root /path/to/data          # is the data here?

rsna-knee-train \
  --config config/training.yaml \
  --data-root       /path/to/rsna-knee-abnormality-detection \
  --labels-root     /path/to/merged_label_export \
  --series-policy   /path/to/series_policy.json \
  --base-checkpoint /path/to/base_model.pt \
  --domain-split    /path/to/scanner_split \
  --out-root        runs/augmented_training \
  --epochs 6 --all-data --preflight-only
```

Drop `--preflight-only` to train. `docs/EXPERIMENT.md` says what the run
changes, what it is measured against, and how to read the result.

The five paths above are artefacts this repository does not carry — they are
large and unchanged. The run verifies each before spending a GPU: the series
policy by fingerprint, the base checkpoint by its recorded endpoint, the split
by the SHA-256 of the `train.csv` it was built from, and the data by counting
how many studies actually have readable series.

## What the model is

```text
DICOM volume
  -> percentile normalisation
  -> 32 slice centres, 2.5D triplets
  -> 90% native centre crop
  -> one aspect-preserving resize to constant pixel area
  -> ConvNeXt slice encoder
  -> series pooling -> study hierarchy   ->  base logits
  -> 6x6 local evidence grid, top-k=8    ->  local logits
  -> logits = base + tanh(gate) * local
```

Labels come from the radiology reports, not from the twelve label columns:
only 58 of 4,407 studies are expert-labelled, and those are held out entirely.
A report that does not mention a finding says nothing about it — silence is
never trained on as a negative, which is what the confidence column carries.

## Layout

```text
src/rsna_knee/
  constants.py          the twelve findings and the shared vocabulary
  runtime.py            device, precision, workers, seeding
  evaluation.py         macro ROC AUC over supervised cells only
  checkpoints.py        reading and fingerprinting the base checkpoint

  imaging/              pixels: DICOM in, model input out
    dicom_io.py           locating a series, sorting frames, normalising
    dicom_metadata.py     recovering plane / fluid / fat from the DICOM
    physical_scale.py     resampling so a millimetre means one thing
    crop_policy.py        the frozen 90% centre crop
    triplets.py           slice centres, 2.5D triplets, constant-area resize

  data/                 what the loader yields, and where it comes from
    tables.py             the competition tables and their metadata repair
    series_policy.py      which series may be used, and the pinned rule
    coverage.py           whether the DICOM folders are on this machine
    labels.py             report labels, confidences, and what silence means
    splits.py             the scanner-grouped split
    slice_selection.py    which slices represent a series
    dataset.py            one study, its series, its labels
    augmentation.py       distorting a training study

  model/
    encoder.py            the slice encoder, and which stages may learn
    study_hierarchy.py    slices -> series -> one study representation
    sparse_head.py        target-specific top-k over local evidence
    study_model.py        the whole model and its fusion gate

  training/
    supervision.py        report labels -> targets, weights, balance
    losses.py             the per-study loss
    memory.py             keeping a long run inside its budget
    surface.py            assembling and fingerprinting the training surface
    loop.py               the run: what changes, what is fixed, epoch choice
```

Nothing is named after an experiment. A test enforces that.

## Where this came from

Extracted from a 270-module research archive in which every module was named
after the experiment that first needed it. Half of what a run imported was
ancestry nothing called.

The migration was mechanical on purpose: definitions were copied verbatim,
regrouped by purpose, and renamed. Nothing was retyped, because this is
numerical code whose value is that its measurements can be trusted.

`tests/test_migration_equivalence.py` proves it. Given the original package it
runs both side by side on identical inputs and compares bit for bit — the
decoded pixels, the model's logits, the losses, the AUC, the split, the
augmentation draw. **25 of 25 checks passed.** Point `RSNA_KNEE_ORIGINAL` at the
original `developments/src` to re-run it.

One rule governed the whole rename: **class and module names are free,
`nn.Module` attributes are not.** Checkpoints are loaded by `load_state_dict`,
whose keys are attribute paths, so `self.encoder` and `self.global_projection`
had to survive untouched. A test pins the key list.

## What still carries a number, and why

Three kinds of string were deliberately left alone:

```text
config keys        b7_rotation_deg, b37_grid_size    read from config/training.yaml
artefact names     b50_selection_split.json          files that already exist
version strings    b36_..._residual_v1               recorded inside checkpoints
```

Renaming any of them would either fall back to a default in silence, or
invalidate every checkpoint that records it.

Inherited error messages also still name the contract they enforce — "requires
the historical fixed 90% center crop". That is not leftover numbering; it tells
a reader which frozen contract just refused them.

## Known gaps

- **No submission path.** Every inference launcher in the archive pins a
  specific checkpoint by SHA-256, and none of them names a checkpoint this
  trainer produces. A run from here cannot be submitted until a launcher pins it.
- **Untrained.** The experiment in `docs/EXPERIMENT.md` has not been run. The
  code is here and proven identical to its source; no result exists.
- **Artefacts not included.** The data, labels, series policy, base checkpoint
  and split live outside the repository.

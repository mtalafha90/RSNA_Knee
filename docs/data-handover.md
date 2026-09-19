# What is here, what is still needed

This session has 4 CPU cores, 15 GB of RAM, about 30 GB of free disk and **no
GPU**. It cannot hold the imaging data and cannot train anything. Training runs
on a separate GPU machine. The work here is analysis, label extraction and
pipeline design — which needs metadata and text, not pixels.

## Already committed

In `data/`:

| File | Rows | Notes |
|---|---:|---|
| `train.csv` | 4,407 | Reports in their original languages, plus labels for 58 studies. The only ground truth. |
| `hand_labels.csv` | 4,407 | **Excluded.** Unknown provenance, translated rather than source text. No code reads it. |
| `train_series.csv` | 24,371 | Series metadata. `Fat_Suppression` duplicates `Fluid_Sensitive` exactly. |
| `test.csv` | 3 | Placeholder. `StudyInstanceUID` only — no reports at test time. |
| `test_series.csv` | 15 | Same columns as `train_series.csv`. |
| `sample_submission.csv` | 3 | Thirteen columns, every probability 0.5. |

Read them through `scripts/load_data.py`. See `findings-01-metadata.md` for
what the numbers mean.

## Still needed

**1. The Kaggle page text.** The competition website is blocked by this
session's network policy, so five questions remain open: the notebook runtime
limit, the external data and pre-trained weight policies, the winning-solution
licence, the efficiency-track scoring formula, and the submission and team
limits. Copy the visible text of the Overview, Data and Rules pages into
`docs/kaggle-pages/` as plain Markdown.

**2. The DICOM header summary.** Run this where the images live and commit the
result:

```bash
python scripts/dump_dicom_headers.py \
    --dicom-root /path/to/train_series \
    --out data/dicom_headers_sample.csv \
    --studies 300
```

It reads headers only, never pixel data, so it is quick. The result is a few
megabytes describing the scanner mix, pulse sequences, slice geometry and image
sizes. That is what the preprocessing pipeline must be designed against. No
direct patient identifiers are collected.

**3. The GPU specification.** Model and VRAM, and roughly how many hours a day
it can run. With four weeks left, that decides whether the plan is 2.5D slice
models at 384px with five-fold cross-validation, or something leaner.

**4. A decision on rebuilding the 24 lost gold studies.** Excluding
`hand_labels.csv` takes the gold set from 82 studies to 58. Those 24 reports
could be read and labelled directly, which would be a worthwhile afternoon
given how thin 58 is.

**5. A normal-study sample.** None of the 58 gold studies is normal, so the
extractor's false-positive rate cannot be measured at all. A few dozen
hand-checked normal reports would close the largest measurement gap in the
project.

## Optionally, a handful of actual slices

For pixel-level questions — intensity ranges, windowing, whether any series are
colour or multi-frame — 20 to 30 individual `.dcm` files from a few different
sites are enough. Put them in `data/sample_dicom/` and force-add them, since
`.gitignore` excludes that directory:

```bash
git add -f data/sample_dicom/
```

## Do not commit

- The full DICOM corpus. It will not fit, and there is no GPU here to use it.
- Model checkpoints, cached tensors or preprocessed image arrays.
- Kaggle API tokens or any other credential.

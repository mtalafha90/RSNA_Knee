# RSNA Knee Abnormality Detection

Work towards the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: predicting twelve clinically important findings from knee
MRI studies, scored by macro-averaged AUC-ROC.

## Where things are

| Path | Contents |
|---|---|
| `docs/competition-brief.md` | Working understanding of the task, data, metric and rules, with a checklist of what still needs verifying. |
| `docs/data-handover.md` | Exactly which files to put in this repository, and which to keep out. |
| `docs/findings-01-metadata.md` | What the metadata actually contains. Read this first. |
| `scripts/load_data.py` | Loaders that handle the known data traps. Start here in any new script or notebook. |
| `scripts/inspect_data.py` | Schema-agnostic inspection of whatever CSVs are present. Writes `reports/schema_report.md`. |
| `scripts/dump_dicom_headers.py` | Summarises DICOM headers into one small, shareable CSV. Run it where the images live. |
| `data/` | Competition metadata CSVs. Small files only; no imaging data. |
| `data/sample_dicom/` | A handful of sample slices, if needed. Excluded from git by default. |
| `reports/` | Generated analysis output. |

## Getting started

```bash
pip install -r requirements.txt
python scripts/inspect_data.py
```

The metadata is already committed. See `docs/data-handover.md` for what is
still outstanding.

## The shape of the problem

Only 82 of the 4,407 training studies carry labels. For the other 98%, the
training signal is the free-text radiology report. Deriving the twelve labels
from those reports is the first task, and its quality caps everything the
imaging model can achieve afterwards.

## How the work is split

This repository is for analysis, pipeline design and code. Training runs on a
separate GPU machine; inference runs in a Kaggle notebook with no internet
access, so every weight and dependency must be pre-uploaded as a Kaggle
dataset.

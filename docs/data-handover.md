# What to put in this repository, and what to keep out

This session has 4 CPU cores, 15 GB of RAM, about 30 GB of free disk and **no
GPU**. It cannot hold the imaging data and cannot train anything. Training
happens on your own GPU machine. So the job here is analysis, pipeline design
and code — which needs metadata, not pixels.

## Commit these (small, and they unblock everything)

Place them in `data/raw/`:

| File | Why it is needed |
|---|---|
| `train.csv` | The labels. Gives prevalence, co-occurrence and the exact column names. |
| `train_series.csv` | Series-level metadata. Decides how studies are assembled into model inputs. |
| `sample_submission.csv` | Fixes the exact output format, column names and order. |
| the reports file | Whether it is a column in `train.csv` or its own file, the text drives the auxiliary-supervision design. |
| any other provided CSV | Nothing should be left out; small files cost nothing. |

These should total a handful of megabytes. If the reports file is large,
commit it anyway — text compresses well in git.

## Also paste across (because the pages are unreachable from here)

The competition website is blocked by this session's network policy. Copy the
visible text of these four pages into `docs/kaggle-pages/` as plain Markdown:

- Overview (including the Evaluation and Timeline tabs)
- Data
- Rules
- The efficiency-track description, if it is on a separate page

Without these, the brief in `competition-brief.md` stays unverified guesswork.

## Run this on the GPU machine, then commit its output

```bash
python scripts/dump_dicom_headers.py \
    --dicom-root /path/to/train_series \
    --out data/raw/dicom_headers_sample.csv \
    --studies 300
```

It reads headers only, never pixel data, so it is quick. The result is a single
CSV of a few megabytes describing the scanner mix, pulse sequences, slice
geometry and image sizes across 300 studies. That is what the preprocessing
pipeline needs to be designed against. Direct patient identifiers are not
collected.

## Optionally, a handful of actual slices

If pixel-level questions come up — intensity ranges, how the images are
windowed, whether any series are colour or multi-frame — then 20 to 30
individual `.dcm` files from a few different sites are enough. Put them in
`data/sample_dicom/` and force-add them, since `.gitignore` excludes that
directory by default:

```bash
git add -f data/sample_dicom/
```

## Do not commit

- The full DICOM corpus. It will not fit, and there is no GPU here to use it.
- Model checkpoints, cached tensors or preprocessed image arrays.
- Your Kaggle API token or any other credential.

## Once the files are in place

```bash
pip install -r requirements.txt
python scripts/inspect_data.py
```

This writes `reports/schema_report.md`, describing every column of every file
it finds: role, cardinality, missing values, label prevalence, findings per
study, and how series nest within studies. It assumes nothing about file or
column names, so it works whatever the real schema turns out to be.

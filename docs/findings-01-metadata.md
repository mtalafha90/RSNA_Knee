# Findings 1 — what the metadata actually contains

Everything below was measured from the committed CSVs. Reproduce with
`python scripts/inspect_data.py` and `python scripts/analyse_reports.py`.

`data/hand_labels.csv` is **excluded from this project**. Its provenance is
unknown and its English text is a translation rather than a source report, so a
mistranslation would corrupt the labels without leaving a trace. No code here
reads it. If a translated corpus is wanted, it will be built from `train.csv`
by a process we control and can re-run. The cost of that exclusion is recorded
below.

## The headline: the labels are almost entirely missing

`train.csv` holds 4,407 studies. **58 of them carry labels.** The other 4,349
have a report and nothing else.

The 58 are all-or-nothing: each has all twelve labels filled, giving 696 label
cells. This was checked against a CSV parsing failure and is genuine — a ragged
parse would have produced a scatter of partial rows, and there are none.

**So this is a weak-supervision problem before it is an imaging problem.** For
98.7% of the training data the signal is the free-text radiology report, not a
label column. Deriving the twelve labels from those reports is the first task,
and its quality caps everything downstream.

The pipeline follows from that:

1. **Report → twelve labels**, validated on the 58 gold studies, producing
   silver labels for all 4,407.
2. **Images → twelve labels**, trained on those silver labels.

Stage 2 cannot be better than stage 1.

## The reports are genuinely multilingual

14 languages were detected across the 4,407 reports. English is a minority.

| Language | Studies | Share | Gold studies |
|---|---:|---:|---:|
| English | 1,735 | 39.4% | 28 |
| Spanish | 640 | 14.5% | 10 |
| Turkish | 546 | 12.4% | 6 |
| Greek | 321 | 7.3% | 3 |
| Bosnian | 318 | 7.2% | 2 |
| German | 262 | 5.9% | 2 |
| Bulgarian | 220 | 5.0% | 3 |
| Dutch | 153 | 3.5% | 2 |
| Croatian | 88 | 2.0% | 2 |
| French | 81 | 1.8% | **0** |
| Rarer or misdetected | 43 | 1.0% | 0 |

Bosnian and Croatian together account for 9.2% and are close enough to be
handled as one. The smallest buckets — Extremaduran, Aragonese, Galician,
Latin — are near-certain misidentifications of Spanish or Italian, which is a
useful reminder that these counts are approximate.

Two consequences. First, an extractor must work across at least ten languages,
either by translating locally first or by using a multilingual model directly.
Second, **French has no gold example at all**, and nor do the rarer buckets:
124 studies, 2.8% of the corpus, on which extraction quality simply cannot be
measured.

This stage runs at training time on your own machine, so the competition's
no-internet rule does not constrain it. Any local model may be used. Report
text must still never leave your machine.

## The gold set is small, enriched, and contains no normal studies

58 studies, all twelve labels each.

| Label | Positive | Rate |
|---|---:|---:|
| Effusion | 35 | 60.3% |
| Synovitis | 27 | 46.6% |
| Medial Meniscus | 26 | 44.8% |
| ACL | 24 | 41.4% |
| Lateral Meniscus | 23 | 39.7% |
| PF OA | 21 | 36.2% |
| Contusion | 19 | 32.8% |
| Fracture | 18 | 31.0% |
| Medial OA | 15 | 25.9% |
| Baker's | 12 | 20.7% |
| Lateral OA | 11 | 19.0% |
| MCL | 9 | 15.5% |

Mean 4.14 findings per study, maximum 9, and **not one of the 58 is normal**.

That last point is the sharpest limitation in the whole dataset. An extractor
that reads abnormalities into a normal report would score perfectly here and
still rank badly on a macro-AUC leaderboard, because AUC is decided by how
cleanly positives separate from negatives — and there are no negatives in this
sample to fail on. Measuring specificity needs a separate, deliberately normal
sample, hand-checked.

These prevalence rates are likewise not population rates. The 58 were plainly
chosen to be findings-rich.

Excluding `hand_labels.csv` costs 24 further labelled studies, taking the gold
set from 82 to 58. That is a real loss on an already thin set, and it may be
worth rebuilding those labels by reading the 24 reports directly.

## The twelve labels, with their exact column names

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`,
`Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`,
`Fracture`.

The apostrophe in `Baker's` and the spaces in the multi-word names must be
reproduced exactly in `submission.csv`. `scripts/load_data.py` holds this list
and checks it against `sample_submission.csv`.

## Series structure

24,371 series across 4,407 studies — a median of 5 per study, ranging from 3
to 14.

| Plane | Series | Mean per study | Studies with none |
|---|---:|---:|---:|
| Sagittal | 9,864 | 2.24 | 0 |
| Coronal | 8,609 | 1.95 | 0 |
| Axial | 5,898 | 1.34 | 0 |

Every study has all three planes, so no missing-plane fallback is needed.

**`Fluid_Sensitive` and `Fat_Suppression` are the same column.** Across all
24,371 series they agree perfectly: 10,361 series have both at 0, 14,010 have
both at 1, and none disagrees. They carry one bit between them, not two.

Both columns, and `Anatomical_Plane`, appear in `test_series.csv`, so this
metadata is available at inference time and can be used to route series within
the model.

## Submission format

`sample_submission.csv` has thirteen columns: `StudyInstanceUID` plus the
twelve labels, every probability at 0.5. One row per study.

`test.csv` contains only `StudyInstanceUID` — and no `Report` column,
confirming that report text exists at training time only.

## Where the risk sits

Three things, in order of how much damage they can do:

1. **The extractor's specificity is unmeasurable as things stand.** No normal
   studies in the gold set. Fixing this means hand-labelling a sample of
   reports that look normal.
2. **58 gold studies is thin.** Roughly nine positives for MCL and eleven for
   Lateral OA. Differences between two extractors will often be invisible
   beneath the noise, so comparisons need bootstrap confidence intervals, not
   point estimates.
3. **Two-stage error compounding.** Silver-label noise sets a ceiling on the
   imaging model that no amount of architecture work will lift.

# Findings 1 — what the metadata actually contains

Everything below was measured from the committed CSVs, not inferred. Reproduce
with `python scripts/inspect_data.py --raw-dir data`.

## The headline: the labels are almost entirely missing

| File | Studies | Studies with any label | Share |
|---|---:|---:|---:|
| `train.csv` | 4,407 | 58 | 1.3% |
| `hand_labels.csv` | 4,407 | 82 | 1.9% |
| **Union** | **4,407** | **82** | **1.9%** |

The 58 labelled in `train.csv` are a subset of the 82 in `hand_labels.csv`, and
the two files agree on every one of the 696 label cells they share — not a
single disagreement. So there is one gold-standard set of 82 studies, of which
76 carry all twelve labels and six are partially labelled.

This was checked against a parsing failure and is genuine. Rows are
all-or-nothing: 4,349 studies have zero of the twelve labels filled and 58 have
all twelve. A ragged parse would have produced a scatter of partial rows
instead.

**So this is a weak-supervision problem before it is an imaging problem.**
The training signal for 98% of the data is the radiology report, not a label
column. Deriving the twelve labels from free text is the first task, and its
quality caps everything downstream.

## `hand_labels.csv` is the English translation of every report

The two files cover the same 4,407 studies, but their `Report` columns match in
only 6.4% of cases — precisely the reports that were already in English.
Everywhere else, `hand_labels.csv` carries an English rendering of the same
report:

| `train.csv` | `hand_labels.csv` |
|---|---|
| `Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno…` | `Technique: MRI of the knee. Results: Medial meniscus tear…` |
| `МР находка: МР данни за ставен излив. Костите, формиращи…` | `MR findings: MR findings of joint effusion. The bones forming…` |

This matters a great deal: label extraction can be done in English throughout,
rather than needing a multilingual model.

Two caveats. First, the provenance of these translations needs establishing —
see the open question at the end. Second, `hand_labels.csv` is delimited with
`, ` rather than `,`, so every column name and identifier carries a leading
space. Read it with `skipinitialspace=True` and strip the identifier column, or
every join against the other files silently produces nothing.

One row of `hand_labels.csv` is corrupt: its identifier is truncated to
`1.2.826.0.1.3680043.8.498`. One study from `train.csv` therefore has no
translation.

## Language mix of the original reports

Measured by script and diacritics, so approximate — the first bucket contains
English along with any unaccented language.

| Script group | Studies | Share |
|---|---:|---:|
| Plain Latin (English or unaccented) | 1,952 | 44.3% |
| Latin with Spanish-like accents | 765 | 17.4% |
| Latin with French-like accents | 628 | 14.3% |
| Greek | 321 | 7.3% |
| Latin with German-like accents | 270 | 6.1% |
| Latin with Polish-like accents | 251 | 5.7% |
| Cyrillic | 220 | 5.0% |

Report structure is not standardised either. Across the English translations,
52% contain "Findings", 30% "Impression", 29% "Conclusion" and 27%
"Technique". No single template covers even half the corpus, so section-based
parsing will not generalise.

## The twelve labels, with their exact column names

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`,
`Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`,
`Fracture`.

Note `Baker's` carries an apostrophe, and several names contain spaces. They
must be reproduced exactly in `submission.csv`.

Prevalence within the 82 gold studies:

| Label | Positive | Labelled | Rate |
|---|---:|---:|---:|
| Effusion | 52 | 78 | 66.7% |
| Medial Meniscus | 37 | 78 | 47.4% |
| Synovitis | 29 | 77 | 37.7% |
| PF OA | 29 | 79 | 36.7% |
| ACL | 30 | 82 | 36.6% |
| Contusion | 28 | 78 | 35.9% |
| Lateral Meniscus | 27 | 77 | 35.1% |
| Fracture | 23 | 76 | 30.3% |
| Medial OA | 23 | 80 | 28.7% |
| Lateral OA | 16 | 77 | 20.8% |
| Baker's | 14 | 76 | 18.4% |
| MCL | 11 | 78 | 14.1% |

Mean 3.89 findings per study, maximum 10, and only four of the 82 are entirely
negative. These rates should **not** be taken as population prevalence: 82
studies is a small and possibly deliberately enriched sample.

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
both at 1, and no series disagrees. Whatever the intent, they carry one bit of
information between them, not two. Treat them as a single sequence descriptor.

Both columns, and `Anatomical_Plane`, are present in `test_series.csv` too, so
this metadata is available at inference time and can be used to route series
within the model.

## Submission format

`sample_submission.csv` has thirteen columns: `StudyInstanceUID` plus the
twelve labels, every probability set to 0.5. One row per study.

`test.csv` contains only `StudyInstanceUID`, with three placeholder rows — and
notably **no `Report` column**, confirming that report text exists at training
time only.

## What follows for the approach

The pipeline has two stages, and the first is the one that decides the outcome:

1. **Report to labels.** Build an extractor that turns an English report into
   twelve probabilities, validated against the 82 gold studies. This yields
   silver labels for all 4,407.
2. **Images to labels.** Train the imaging model on those silver labels.

Stage 2 cannot be better than stage 1. Noise introduced when reading the
reports propagates into every image model trained afterwards.

The thinness of the gold set is the central risk. With roughly 78 labelled
examples per label, an AUC estimate carries a confidence interval wide enough
to hide a real difference between two extractors. Validating the extractor
therefore needs care: bootstrap confidence intervals rather than point
estimates, and cross-checking against the leaderboard rather than trusting a
local number.

## Open questions

1. **Where did `hand_labels.csv` come from?** If it is an official competition
   file, it can be used freely. If the translations were produced by sending
   report text to an external service, that needs checking against the
   competition's data-security rules before anything is built on it.
2. **Are the 82 gold studies a random sample?** If they were chosen for being
   findings-rich, their prevalence rates are not a guide to the population and
   should not be used to calibrate anything.

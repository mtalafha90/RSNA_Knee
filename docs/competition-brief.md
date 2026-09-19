# RSNA Knee Abnormality Detection — working brief

**Status: UNVERIFIED.** `kaggle.com` and `rsna.org` are both blocked by this
session's network egress policy, so none of the official pages could be read
directly. Everything below was assembled from search-engine snippets and press
coverage. Every line must be checked against the real competition pages before
any of it is relied upon. The checklist at the end tracks that.

## The task

Predict, for each knee MRI **study**, the probability of twelve binary findings.

Reported label set (exact column names still to be confirmed):

| # | Finding | Group |
|---|---|---|
| 1 | ACL tear | Ligament |
| 2 | MCL tear | Ligament |
| 3 | Medial meniscus tear | Meniscus |
| 4 | Lateral meniscus tear | Meniscus |
| 5 | Medial compartment osteoarthritis | Osteoarthritis |
| 6 | Lateral compartment osteoarthritis | Osteoarthritis |
| 7 | Patellofemoral osteoarthritis | Osteoarthritis |
| 8 | Joint effusion | Inflammatory |
| 9 | Synovitis | Inflammatory |
| 10 | Baker's cyst | Inflammatory |
| 11 | Bone contusion | Osseous |
| 12 | Fracture | Osseous |

## Evaluation

Macro-averaged AUC-ROC across the twelve labels, each weighted equally. The
public leaderboard is computed on roughly 30% of the test data.

Two consequences follow from the metric:

- **Rare labels matter as much as common ones.** A label present in 2% of
  studies contributes exactly as much as one present in 40%. Per-label
  threshold tuning is irrelevant, because AUC only cares about ranking; what
  matters is the ranking quality within each label separately.
- **Calibration across labels is irrelevant.** Each label's AUC is computed
  independently, so there is no benefit in making the twelve outputs comparable
  with one another.

## Data

- More than 5,000 training exams from 16 sites worldwide.
- DICOM, organised as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`,
  one file per image slice.
- Each study is several series — different anatomical planes and pulse
  sequences from one scanning session.
- `train_series.csv`: one row per series, with `StudyInstanceUID`,
  `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression` and
  `Anatomical_Plane` (sagittal, coronal or axial).
- Free-text radiology reports are provided for training, in roughly a dozen
  languages.
- **The hidden test set has no reports.** Text is a training-time signal only.

## Submission

- Kaggle notebook submission, roughly a 9-hour runtime limit, internet access
  disabled during scoring.
- The notebook writes `submission.csv`: one row per study, thirteen columns
  (the study identifier plus twelve probabilities in [0, 1]).
- `sample_submission.csv` is a valid submission with every probability set
  to 0.5.

## Timeline and prizes

- Entry and team-merger deadline: 15 October 2026.
- Final submission deadline: 22 October 2026.
- $77,000 in prizes: ten leaderboard places plus a separate three-prize
  efficiency track.

## What this implies for the approach

1. **Internet is off at inference.** Every model weight, Python wheel and
   auxiliary file must be uploaded to Kaggle as a dataset beforehand. Anything
   that quietly downloads on first use — pre-trained checkpoints from
   `timm`, tokenisers, NLTK corpora — will fail in the scoring run.
2. **Reports are training-only.** They are useful for label verification, for
   auxiliary text-supervision objectives, or for teacher-student distillation
   where a text-aware teacher supervises an image-only student. They are
   multilingual, so any text handling needs local open-weight models. Report
   text must never be sent to an external API.
3. **Studies are variable-length collections of series.** The model must cope
   with a varying number of series per study, varying planes and varying slice
   counts. Series-level metadata is the natural way to route each series to the
   right branch of the model.
4. **Label groups differ in what they need.** Meniscus and ligament tears are
   localised findings best seen on particular planes and sequences;
   osteoarthritis grading is a global assessment; effusion and Baker's cysts
   are fluid-bright and favour fluid-sensitive sequences. One architecture will
   not suit all twelve equally.

## Verification checklist

Each of these needs confirming against the official pages before it drives a
decision:

- [ ] Exact label column names and their order.
- [ ] Exact CSV file names and their columns.
- [ ] Where the report text lives (a column in `train.csv` or a separate file).
- [ ] Whether labels contain missing values as well as 0 and 1.
- [ ] Exact submission column names and the study identifier column name.
- [ ] Notebook runtime limit, and whether it differs for CPU and GPU.
- [ ] External data policy and pre-trained weight policy.
- [ ] Winning-solution licence obligations.
- [ ] Efficiency-track scoring formula.
- [ ] Daily submission limit and maximum team size.

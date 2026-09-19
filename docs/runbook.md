# Runbook — from reports to a training target

Exact steps, in order. Steps 1 to 5 have working tooling in this repository.
Steps 6 to 8 are described but not yet built; they need decisions that depend
on what steps 1 to 5 produce.

Run everything from the repository root.

---

## Step 0 — set up (5 minutes, this machine or the GPU box)

```bash
pip install -r requirements.txt
python scripts/load_data.py
```

Expect:

```
train      : 4,407 studies
gold       : 58 studies, all twelve labels each
unlabelled : 4,349 studies to extract labels for
```

If the gold count is not 58, stop — something is wrong with the data files.

---

## Step 1 — enlarge the gold set

**This is the bottleneck, and it comes first.** With 58 studies the 95%
interval on macro-AUC is 0.108 wide, so one extractor has to beat another by
more than about 0.05 before the difference is real. You cannot tune against a
ruler that coarse. And because every one of the 58 is abnormal, the false
positive rate is not merely imprecise — it is unmeasured.

```bash
python scripts/make_worklist.py --n 120 --out data/worklist.xlsx
```

This selects 120 studies at random, stratified across ten language groups, and
excludes the 58 already labelled. It writes a spreadsheet with the report text
and twelve empty label columns.

Now fill it in. Three rules:

1. **Fill all twelve columns for every study**, with 1 or 0. No blanks.
2. **Do not skip a study because the report looks normal.** Normal studies are
   exactly what the gold set is missing. Around a third of the sample should
   come out all-zero; if none does, something is being read too generously.
3. **Label what the report says, not what you infer.** If the report does not
   mention the ACL, that is a 0 — you are recording the report's content, which
   is what the extractor will be judged against.

Budget two to three minutes per report: roughly four to six hours. It is the
highest-value time in the whole project.

Partial progress is fine — incomplete rows are skipped and can be finished
later.

---

## Step 2 — merge and confirm the measurement improved

```bash
python scripts/merge_gold.py data/worklist.xlsx
```

This validates the entries, writes `data/gold_labels.csv`, and prints how much
the interval narrowed. At 174 studies expect a width near 0.061, down from
0.108 — meaning a real difference between two extractors is now anything above
about 0.03.

Check two things in its output:

- **Normal studies is not zero.** If it is, the sample was read too
  generously and specificity is still unmeasurable.
- **No label is marked `<- thin`.** A label with fewer than ten positives or
  ten negatives cannot be measured reliably whatever the total.

Every script from here on picks up `data/gold_labels.csv` automatically.

Commit it:

```bash
git add data/gold_labels.csv && git commit -m "Enlarge the gold set to N studies" && git push
```

---

## Step 3 — inspect the extraction prompt

```bash
python scripts/extract_labels.py --dry-run
```

Read the prompt that will be sent. Check that the verdict vocabulary matches
how you labelled in step 1 — if you treated hedged wording differently from the
prompt's `probable`, change one of them so they agree.

---

## Step 4 — extract a 200-study slice and score it

Do not run all 4,407 first. A slice costs a few minutes and catches a bad
prompt before it costs hours.

```bash
python scripts/extract_labels.py \
    --model <model-id> \
    --limit 200 \
    --out data/verdicts_pilot.jsonl

python scripts/review_classification.py data/verdicts_pilot.jsonl
```

Only some of the 200 will be gold studies, so the score is noisy — that is
expected. What matters at this stage:

- Verdicts parse: few or no studies coming back all `not_mentioned`.
- Non-English reports get verdicts. If Bulgarian or Greek come back empty, the
  model is not reading them and a larger or different model is needed.
- Spot-read twenty verdicts against their reports yourself.

Fix the prompt and repeat until the slice looks sound. Delete the pilot file
between runs, or it will resume instead of starting over.

---

## Step 5 — full extraction

```bash
python scripts/extract_labels.py --model <model-id> --out data/verdicts.jsonl
python scripts/review_classification.py data/verdicts.jsonl
```

Resumable: re-running skips studies already written, so an interrupted run
continues where it stopped.

The number to beat is **0.643**, the current keyword-based label set. Beating
it by less than 0.03 (on the enlarged gold set) is not yet a real improvement.

---

## Step 6 — tune the verdict mapping *(not yet built)*

The expensive step is now done and on disk. Turning verdicts into
probabilities is arithmetic, so the mapping in `extract_labels.py` —
`DEFAULT_MAPPING` and `DEFAULT_SILENCE_PRIOR` — can be fitted against the gold
set without re-running the model. That is a small optimiser over five numbers
plus twelve priors.

One thing already measured: a per-label constant silence prior does **not**
change the extractor's own AUC, because every silent study for a label ties
with the others. It matters as a training target for the imaging model, where
a soft 0.15 carries more information than a hard 0.

---

## Step 7 — export silver labels *(not yet built)*

Join the tuned probabilities for all 4,407 studies to `train_series.csv` and
write the training target for the imaging model. Keep the gold studies flagged,
so they can be held out or weighted.

---

## Step 8 — the disagreement loop *(not yet built)*

Once the imaging model exists, rank studies by how strongly it disagrees with
its own silver label. Those are concentrated where the extraction went wrong,
not where the imaging is hard. Re-read a few hundred, correct them, retrain.

This is the cheapest accuracy left in the project and most teams never do it.

---

## Still blocking

- **GPU model and VRAM** — decides the extraction model size in step 4.
- **The Kaggle rules pages** — five questions remain open, including the
  external-data and pre-trained-weight policies.
- **The DICOM header dump** — needed before any imaging work starts.

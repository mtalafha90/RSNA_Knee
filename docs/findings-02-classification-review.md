# Findings 2 — reviewing `train_classified.xlsx`

A candidate label set covering all 4,407 training studies, with a `Language`
column. Reproduce the numbers with:

```bash
python scripts/review_classification.py data/train_classified.xlsx
```

## Where it agrees with the independent measurements

These are worth stating first, because they mean both sides are looking at the
same data:

- **4,407 studies, identical identifiers**, no duplicates, none missing, none extra.
- **Report text matches `train.csv` exactly**, character for character, in 100% of rows.
- The `classified` and `classified (2)` sheets are byte-identical duplicates.
- The language assignment agrees closely with independent detection by
  `py3langid`, and is better grouped: treating Bosnian and Croatian as one
  category is the right call, and `py3langid`'s split into 318 plus 88 is a
  distinction without a difference.

## Where the language labelling differs, and why

| Their label | Studies | Independent detection says |
|---|---:|---|
| English | 1,702 | 1,702 English — exact agreement |
| Spanish | 573 | 556 Spanish, plus 17 Extremaduran/Galician/Aragonese (misfires on Spanish) |
| Turkish | 546 | 546 Turkish — exact |
| Croatian/Serbian/Bosnian | 409 | 318 Bosnian, 88 Croatian, 3 German |
| Greek | 321 | 321 Greek — exact |
| German | 254 | 254 German — exact |
| Cyrillic | 220 | 220 Bulgarian — exact |
| Dutch | 153 | 153 Dutch — exact |
| **Unknown** | **140** | **84 Spanish, 27 English, 25 Extremaduran (i.e. Spanish), 3 German, 1 Latin** |
| French | 81 | 81 French — exact |
| Romanian | 8 | 6 English, 2 German — not Romanian |

Only two categories are genuinely in question.

**"Unknown" is not an unknown language.** Those 140 reports have a median
length of 147 characters against 977 for the corpus overall. They are simply
too short for a language detector to settle on, and roughly 109 of them are
Spanish and 27 English. They should be folded into those groups rather than
left aside — particularly since 84% of them currently receive no finding.

**"Romanian" is 8 studies and appears to be a misattribution.** Too small to
matter either way.

The `All Languages Stats` sheet has two arithmetic problems: its header row is
mangled, so the columns are unlabelled, and its Turkish total reads 573 where
the data holds 546 — Spanish's figure appears to have been copied across. Its
grand total therefore reads 4,434 rather than 4,407.

## The serious problem: the labels do not survive contact with ground truth

Scored against the 58 studies that carry real labels:

| | |
|---|---|
| **Macro-AUC if used directly as predictions** | **0.643** |
| Cell accuracy | 71.8% |
| Recall | 47.5% |
| Precision | 62.0% |

For scale, labelling every cell negative scores 65.5% accuracy on this set.
More than half of the true findings are missed.

| Label | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|
| MCL | 0.667 | 0.776 | 0.721 |
| Fracture | 0.444 | 0.950 | 0.697 |
| Baker's | 0.417 | 0.957 | 0.687 |
| Contusion | 0.421 | 0.949 | 0.685 |
| Medial Meniscus | 0.500 | 0.844 | 0.672 |
| Lateral Meniscus | 0.609 | 0.714 | 0.661 |
| ACL | 0.583 | 0.735 | 0.659 |
| Medial OA | 0.333 | 0.977 | 0.655 |
| Synovitis | 0.370 | 0.871 | 0.621 |
| Lateral OA | 0.273 | 0.894 | 0.583 |
| PF OA | 0.238 | 0.865 | 0.551 |
| **Effusion** | 0.657 | **0.391** | **0.524** |

Effusion is the worst: it is called on 37 of 58 studies where the truth is 35,
but the overlap is poor, and specificity of 0.391 means it is barely better
than a coin flip. PF OA and Lateral OA are close behind, both through very low
sensitivity.

## The cause: the extractor cannot read most of the languages

| Language | Studies | Mean findings | Studies with no finding |
|---|---:|---:|---:|
| French | 81 | 2.15 | 2% |
| English | 1,702 | 3.63 | 11% |
| Croatian/Serbian/Bosnian | 409 | 2.44 | 12% |
| German | 254 | 1.07 | 31% |
| Spanish | 573 | 1.07 | 41% |
| Dutch | 153 | 0.22 | 82% |
| Turkish | 546 | 0.19 | 84% |
| Unknown | 140 | 0.29 | 84% |
| Greek | 321 | 0.12 | 88% |
| **Cyrillic (Bulgarian)** | **220** | **0.00** | **100%** |

Not one finding was assigned across 220 Bulgarian reports. Greek, Turkish and
Dutch are close behind.

English studies are 38.6% of the corpus but carry 73.0% of every finding
assigned. Even there the extractor over-calls: on the English gold studies it
assigns 145 findings where the truth is 108.

The pattern fits an extractor built on Latin-script cognates. It reaches French
and Croatian because "épanchement" and "menisk" are close enough to the terms
it knows; it fails on Greek and Cyrillic because the script shares nothing, and
on Turkish because the vocabulary does not.

## Why this matters more than the numbers suggest

An all-negative label on a study that has findings is not a neutral gap — it is
wrong supervision. Training an imaging model on these labels would teach it
that 220 Bulgarian knees and the great majority of Turkish and Greek ones are
free of abnormalities that are visibly present in the scans. That is worse than
excluding those studies, and it corrupts the model for every language.

A macro-AUC of 0.643 in the labels sets a hard ceiling on the imaging model
trained from them, since the model can only learn the target it is given.

Two caveats on the 0.643 figure. It rests on 58 studies, so its confidence
interval is wide. And those 58 contain no normal study, so specificity here is
measured only against label-level negatives within abnormal knees — the real
false-positive rate on normal studies is still unmeasured.

## What to do instead

The fix is not more keywords. Two routes work, and they can be combined:

1. **Translate, then extract.** Run a local translation model over the 2,705
   non-English reports, then apply one extractor to uniform English. This is
   essentially rebuilding `hand_labels.csv` by a process we control.
2. **Extract directly with a multilingual model.** A local instruction-following
   model handles all fourteen languages without a translation step and can be
   asked for per-label confidence rather than a hard 0/1 — which macro-AUC
   rewards, since it ranks rather than thresholds.

Either way this runs at training time on your own GPU, so the competition's
no-internet rule does not apply. Report text must still never leave the machine.

Whichever route, the measurement problem stays: 58 gold studies and none of
them normal. Rebuilding the 24 gold studies lost with `hand_labels.csv`, and
hand-labelling a few dozen normal reports, would do more for the project than
any amount of extractor tuning — because without them there is no way to tell
whether a new extractor is better.

# Classification review

Candidate: `train_classified.xlsx` — 4,407 rows, covering 4,407 of the 4,407 training studies. 58 gold studies matched.

## Macro-AUC: **0.643**  (95% interval 0.590 to 0.698)

0.500 is a coin flip. The interval comes from resampling studies, and it is 0.108 wide — so a rival label set must beat this by more than about 0.05 before the difference is real rather than noise.

The candidate is binary 0/1, so each AUC below is the mean of that label's sensitivity and specificity. Graded confidences would score higher at identical decisions — see `docs/findings-02-classification-review.md`.

| label            |   gold positives |   predicted |   TP |   FN |   FP |   TN |   sensitivity |   specificity |   AUC |
|:-----------------|-----------------:|------------:|-----:|-----:|-----:|-----:|--------------:|--------------:|------:|
| ACL              |               24 |          23 |   14 |   10 |    9 |   25 |         0.583 |         0.735 | 0.659 |
| MCL              |                9 |          17 |    6 |    3 |   11 |   38 |         0.667 |         0.776 | 0.721 |
| Medial Meniscus  |               26 |          18 |   13 |   13 |    5 |   27 |         0.500 |         0.844 | 0.672 |
| Lateral Meniscus |               23 |          24 |   14 |    9 |   10 |   25 |         0.609 |         0.714 | 0.661 |
| Medial OA        |               15 |           6 |    5 |   10 |    1 |   42 |         0.333 |         0.977 | 0.655 |
| Lateral OA       |               11 |           8 |    3 |    8 |    5 |   42 |         0.273 |         0.894 | 0.583 |
| PF OA            |               21 |          10 |    5 |   16 |    5 |   32 |         0.238 |         0.865 | 0.551 |
| Effusion         |               35 |          37 |   23 |   12 |   14 |    9 |         0.657 |         0.391 | 0.524 |
| Synovitis        |               27 |          14 |   10 |   17 |    4 |   27 |         0.370 |         0.871 | 0.621 |
| Baker's          |               12 |           7 |    5 |    7 |    2 |   44 |         0.417 |         0.957 | 0.687 |
| Contusion        |               19 |          10 |    8 |   11 |    2 |   37 |         0.421 |         0.949 | 0.685 |
| Fracture         |               18 |          10 |    8 |   10 |    2 |   38 |         0.444 |         0.950 | 0.697 |

## Findings assigned, by report language

| Language                          |   studies |   mean_findings |   finding_free |
|:----------------------------------|----------:|----------------:|---------------:|
| English                           |   1702.00 |            3.63 |           0.11 |
| Spanish                           |    573.00 |            1.07 |           0.41 |
| Turkish                           |    546.00 |            0.19 |           0.84 |
| Croatian/Serbian/Bosnian          |    409.00 |            2.44 |           0.12 |
| Greek                             |    321.00 |            0.12 |           0.88 |
| German                            |    254.00 |            1.07 |           0.31 |
| Cyrillic (Bulgarian/Russian/etc.) |    220.00 |            0.00 |           1.00 |
| Dutch                             |    153.00 |            0.22 |           0.82 |
| Unknown                           |    140.00 |            0.29 |           0.84 |
| French                            |     81.00 |            2.15 |           0.02 |
| Romanian                          |      8.00 |            2.00 |           0.12 |

**5 languages look unreadable to this extractor**, covering 1,380 studies (31.3%): Turkish, Greek, Cyrillic (Bulgarian/Russian/etc.), Dutch, Unknown. More than three quarters of their studies come out with no finding at all.

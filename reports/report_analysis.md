# Report analysis

4,407 studies, of which 58 are labelled (1.3%).

Report length in characters: median 977, shortest 52, longest 4,743.

## Language of the reports

14 languages detected across 4,407 studies.

| Language | Studies | Share | Gold studies |
| --- | ---: | ---: | ---: |
| `en` | 1,735 | 39.4% | 28 |
| `es` | 640 | 14.5% | 10 |
| `tr` | 546 | 12.4% | 6 |
| `el` | 321 | 7.3% | 3 |
| `bs` | 318 | 7.2% | 2 |
| `de` | 262 | 5.9% | 2 |
| `bg` | 220 | 5.0% | 3 |
| `nl` | 153 | 3.5% | 2 |
| `hr` | 88 | 2.0% | 2 |
| `fr` | 81 | 1.8% | 0 |
| `ext` ⚠ | 39 | 0.9% | 0 |
| *3 rarer* | 4 | 0.1% | — |

**5 languages have no gold example at all** (`an`, `ext`, `fr`, `gl`, `la`), covering 124 studies (2.8%). Extraction quality on those is unmeasurable.

Entries marked ⚠ are near-certain misidentifications of a related language, which is a reminder that these counts are approximate.

## The gold set

58 studies, each carrying all twelve labels — 696 label cells in total.

| Label | Positive | Rate |
| --- | ---: | ---: |
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

Findings per study: mean 4.14, median 4, maximum 9. Studies with no positive finding: **0**.

Not one of these studies is normal. The gold set can therefore show whether an extractor finds the abnormalities that are present, but it cannot show how often the extractor invents findings in a normal report — and specificity is exactly what a macro-AUC metric punishes. Any claim about false positives needs a separate, deliberately normal sample.

# Schema report

Files found under `data/`: 6.

## `hand_labels.csv`

4,407 rows × 14 columns (5.1 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 4,407, missing: 0 (0.0%)
  - examples: ` 1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260`, ` 1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585`, ` 1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231`
- **` Report`** — role: *text*, dtype: `str`, distinct: 4,264, missing: 0 (0.0%)
  - length in characters: min 44, median 1,007, max 4,742
  - first value (truncated): ` Technique: MRI of the knee. Results: Medial meniscus tear. Sign of subchondral avascular necrosis in the medial femoral condyle. Medial femorotibial osteoarthritis. Effusion. Impression: Medial meniscus tear. Sign of subchondral avascular necrosis in the medial femoral condyle. Medial femorotibial `
- **` ACL`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,325 (98.1%)
  - positives: 30 (36.59% of non-missing)
- **` MCL`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,329 (98.2%)
  - positives: 11 (14.10% of non-missing)
- **` Medial Meniscus`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,329 (98.2%)
  - positives: 37 (47.44% of non-missing)
- **` Lateral Meniscus`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,330 (98.3%)
  - positives: 27 (35.06% of non-missing)
- **` Medial OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,327 (98.2%)
  - positives: 23 (28.75% of non-missing)
- **` Lateral OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,330 (98.3%)
  - positives: 16 (20.78% of non-missing)
- **` PF OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,328 (98.2%)
  - positives: 29 (36.71% of non-missing)
- **` Effusion`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,329 (98.2%)
  - positives: 52 (66.67% of non-missing)
- **` Synovitis`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,330 (98.3%)
  - positives: 29 (37.66% of non-missing)
- **` Baker's`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,331 (98.3%)
  - positives: 14 (18.42% of non-missing)
- **` Contusion`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,329 (98.2%)
  - positives: 28 (35.90% of non-missing)
- **` Fracture`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,331 (98.3%)
  - positives: 23 (30.26% of non-missing)

### Looks like the label table

| label | prevalence | positives |
| --- | ---: | ---: |
|  Effusion | 66.67% | 52 |
|  Medial Meniscus | 47.44% | 37 |
|  Synovitis | 37.66% | 29 |
|  PF OA | 36.71% | 29 |
|  ACL | 36.59% | 30 |
|  Contusion | 35.90% | 28 |
|  Lateral Meniscus | 35.06% | 27 |
|  Fracture | 30.26% | 23 |
|  Medial OA | 28.75% | 23 |
|  Lateral OA | 20.78% | 16 |
|  Baker's | 18.42% | 14 |
|  MCL | 14.10% | 11 |

Findings per study: mean 0.07, median 0, max 10. Studies with no positive finding: 4,329 (98.2%).

## `sample_submission.csv`

3 rows × 13 columns (0.0 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 3, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191`, `1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776`, `1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402`
- **`ACL`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`MCL`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Medial Meniscus`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Lateral Meniscus`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Medial OA`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Lateral OA`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`PF OA`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Effusion`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Synovitis`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Baker's`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Contusion`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5
- **`Fracture`** — role: *numeric*, dtype: `float64`, distinct: 1, missing: 0 (0.0%)
  - min 0.5, median 0.5, max 0.5, mean 0.5

## `test.csv`

3 rows × 1 columns (0.0 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 3, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191`, `1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776`, `1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402`

## `test_series.csv`

15 rows × 5 columns (0.0 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 3, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191`, `1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776`, `1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402`
- **`SeriesInstanceUID`** — role: *id*, dtype: `str`, distinct: 15, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947`, `1.2.826.0.1.3680043.8.498.17811502614030631664517622518906646132`, `1.2.826.0.1.3680043.8.498.30565395595045942404081022062489758495`
- **`Fluid_Sensitive`** — role: *binary*, dtype: `int64`, distinct: 2, missing: 0 (0.0%)
  - positives: 9 (60.00% of non-missing)
- **`Fat_Suppression`** — role: *binary*, dtype: `int64`, distinct: 2, missing: 0 (0.0%)
  - positives: 9 (60.00% of non-missing)
- **`Anatomical_Plane`** — role: *categorical*, dtype: `str`, distinct: 3, missing: 0 (0.0%)
  - values: `Sagittal` (7), `Axial` (4), `Coronal` (4)

### Nesting: `SeriesInstanceUID` within `StudyInstanceUID`

3 distinct `StudyInstanceUID` values, 15 distinct `SeriesInstanceUID` values.
Per `StudyInstanceUID`: min 5, median 5, mean 5.00, max 5.

## `train.csv`

4,407 rows × 14 columns (5.4 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 4,407, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260`, `1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585`, `1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231`
- **`Report`** — role: *text*, dtype: `str`, distinct: 4,276, missing: 0 (0.0%)
  - length in characters: min 52, median 977, max 4,743
  - first value (truncated): `Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. Signo de necrosis avascular subcondral en el cóndilo femoral medial. Artrosis femorotibial medial. Derrame. . Impresión: Rotura de menisco interno. Signo de necrosis avascular subcondral en el cóndilo femoral medial. Artrosis femorot`
- **`ACL`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 24 (41.38% of non-missing)
- **`MCL`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 9 (15.52% of non-missing)
- **`Medial Meniscus`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 26 (44.83% of non-missing)
- **`Lateral Meniscus`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 23 (39.66% of non-missing)
- **`Medial OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 15 (25.86% of non-missing)
- **`Lateral OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 11 (18.97% of non-missing)
- **`PF OA`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 21 (36.21% of non-missing)
- **`Effusion`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 35 (60.34% of non-missing)
- **`Synovitis`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 27 (46.55% of non-missing)
- **`Baker's`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 12 (20.69% of non-missing)
- **`Contusion`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 19 (32.76% of non-missing)
- **`Fracture`** — role: *binary*, dtype: `float64`, distinct: 2, missing: 4,349 (98.7%)
  - positives: 18 (31.03% of non-missing)

### Looks like the label table

| label | prevalence | positives |
| --- | ---: | ---: |
| Effusion | 60.34% | 35 |
| Synovitis | 46.55% | 27 |
| Medial Meniscus | 44.83% | 26 |
| ACL | 41.38% | 24 |
| Lateral Meniscus | 39.66% | 23 |
| PF OA | 36.21% | 21 |
| Contusion | 32.76% | 19 |
| Fracture | 31.03% | 18 |
| Medial OA | 25.86% | 15 |
| Baker's | 20.69% | 12 |
| Lateral OA | 18.97% | 11 |
| MCL | 15.52% | 9 |

Findings per study: mean 0.05, median 0, max 9. Studies with no positive finding: 4,349 (98.7%).

## `train_series.csv`

24,371 rows × 5 columns (3.3 MB on disk).

- **`StudyInstanceUID`** — role: *id*, dtype: `str`, distinct: 4,407, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260`, `1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585`, `1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231`
- **`SeriesInstanceUID`** — role: *id*, dtype: `str`, distinct: 24,371, missing: 0 (0.0%)
  - examples: `1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772`, `1.2.826.0.1.3680043.8.498.13821229744997220641575291927426543265`, `1.2.826.0.1.3680043.8.498.23084836536722595275828690293168736174`
- **`Fluid_Sensitive`** — role: *binary*, dtype: `int64`, distinct: 2, missing: 0 (0.0%)
  - positives: 14,010 (57.49% of non-missing)
- **`Fat_Suppression`** — role: *binary*, dtype: `int64`, distinct: 2, missing: 0 (0.0%)
  - positives: 14,010 (57.49% of non-missing)
- **`Anatomical_Plane`** — role: *categorical*, dtype: `str`, distinct: 3, missing: 0 (0.0%)
  - values: `Sagittal` (9,864), `Coronal` (8,609), `Axial` (5,898)

### Nesting: `SeriesInstanceUID` within `StudyInstanceUID`

4,407 distinct `StudyInstanceUID` values, 24,371 distinct `SeriesInstanceUID` values.
Per `StudyInstanceUID`: min 3, median 5, mean 5.53, max 14.

# Data Provenance Log

## Source
All five datasets (CM1, JM1, KC1, KC2, PC1) were sourced from a single GitHub mirror
providing the datasets in the standard 21-metric PROMISE/tera-PROMISE schema:
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(retrieved 2026-09-27, commit 3898d21e287c130c76d458faf65793d32483076d).

## Alternative source evaluated and rejected
The Shepperd et al. (2013) NASA MDP repository (klainfo/NASADefectDataset on GitHub)
was also evaluated, since it provides documented Original/D'/D'' cleaning levels and
is the canonical source behind the data-quality methodology this project's proposal
cites. It was rejected for two reasons:
1. It does not include KC2 (a tera-PROMISE dataset, not part of the original NASA MDP
   family Shepperd et al. worked with). KC2 is one of the five projects in this study,
   so this alone rules the repository out.
2. Its CM1/PC1 files use an unreduced 41-attribute metric set, and its JM1/KC1 files
   use a differently-named 22-attribute set. Neither matches the standard 21-metric
   schema this project (and the wider CPDP literature) relies on, and using a mixed
   schema across source/target projects would break the shared feature-space
   requirement cross-project transfer depends on.

## Known data-quality issues identified during validation

### 1. Instance count discrepancies with commonly-cited literature figures
| Dataset | This source (raw) | Commonly cited in literature |
|---|---|---|
| CM1 | 498 | 498 (matches) |
| JM1 | 13,204 | ~10,878-10,885 |
| KC1 | 2,109 | 2,109 (matches) |
| KC2 | 522 | 522 (matches) |
| PC1 | 1,109 | 1,109 (matches) |

JM1's raw count (13,204) differs from the ~10,878-10,885 commonly cited, and the
count after duplicate removal (8,906 rows) does not match it either, because the
cited file itself contains repeated rows. The mirror's JM1 holds every row of the
10,878-row NASA MDP file (compared on the 14 whole-number metrics and the label)
plus 2,326 extra rows: the two artefact rows (see 2 below), one further row, and
2,323 copies of rows already in the file. After duplicate removal both files give
the same set of distinct modules. The other four projects match the cited figures
directly.

### 2. Two artefact rows per dataset
Each raw file contains two artefact rows.
* A row with fractional values across several count-based metrics at once (e.g.
  loc=1.1, uniq_Op=1.2, branchCount=1.4), which is impossible for count metrics. It
  is the first data row and is present in all five of the mirror's ARFF files.
* A row in which all 21 metrics equal 1, labelled defective. It is present in all
  five of the mirror's CSV files (and in four of its ARFF files).

Neither row is in the NASA MDP files used by Shepperd et al. (2013): none of the 13
files in OriginalData/MDP contains a row with fractional count values or an all-ones
row. Both rows were therefore added by the mirror and are not part of the original
data. Both were removed from every dataset (see the cleaning logs below).

### 3. High exact-duplicate rates
| Dataset | Raw rows | Duplicates removed | % duplicated (of rows after the fractional row) |
|---|---|---|---|
| CM1 | 498 | 56 | 11.3% |
| JM1 | 13,204 | 4,296 | 32.5% |
| KC1 | 2,109 | 897 | 42.6% |
| KC2 | 522 | 147 | 28.2% |
| PC1 | 1,109 | 155 | 14.0% |

This is a substantial data-quality issue, especially for KC1 and JM1, and corroborates
Shepperd et al.'s (2013) documented concerns about duplication in the NASA MDP
datasets. Exact duplicate rows (identical across all 21 features and the target) were
removed.

### 4. Defect rate shift after duplicate removal
Removing duplicates increased the observed defect rate in every dataset, since
duplicate rows were disproportionately clean (non-defective) modules. The second
column below is the rate after exact-duplicate removal only; the strict level
(Step 5 below) removes further rows and changes these figures again.

| Dataset | Defect rate (raw) | Defect rate (after duplicate removal) |
|---|---|---|
| CM1 | 9.8% | 10.7% |
| JM1 | ~19% (typical lit. figure) | 22.5% |
| KC1 | ~15% (typical lit. figure) | 25.9% |
| KC2 | ~20% (typical lit. figure) | 27.9% |
| PC1 | ~7% (typical lit. figure) | 7.3% |

This is worth discussing explicitly in the thesis: it demonstrates that the
duplicate-removal decision is not neutral with respect to class balance, which in
turn affects SMOTE's oversampling target ratio and should be noted as a limitation
or at least an explicit methodological choice.

## Cleaning steps applied (in order)
1. Standardise column names and target encoding across sources
2. Remove the two artefact rows (the fractional-value row and the all-ones row)
3. Remove exact duplicate rows (all 21 features + target identical)
4. Verify: no missing values, no zero-variance features remaining
5. Strict level only (`src/clean_data.py`): remove rows that break any of
   R1 all-ones row (none remain at this point, so R1 removes 0 rows); R2 conflicting labels for identical metrics; R3 n = 0 while loc > 0;
   R4 n != total_Op + total_Opnd; R5 loc = 0.

Row counts, duplicate-removed level -> strict level: CM1 440 -> 438; JM1 8,906 -> 7,698
(R2 = 176, R3 = 971, R4 = 139); KC1 1,210 -> 1,147; KC2 373 -> 328; PC1 952 -> 934.
The per-rule counts overlap, so they need not sum to the number of rows removed.

## Where things are
* Implementation of the strict level: `src/clean_data.py`.
* Per-rule counts for the strict level: `experiments/results/cleaning_log.csv`.
* Counts for the duplicate-removal level: `data/processed/cleaning_log.csv`.
  These are two different files with the same name.
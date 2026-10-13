# Data Provenance Log

## Source
All five datasets (CM1, JM1, KC1, KC2, PC1) were sourced from a single GitHub mirror
providing the datasets in the standard 21-metric PROMISE/tera-PROMISE schema:
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(retrieved <2021-02-03>, commit <full 3898d21e287c130c76d458faf65793d32483076d>).

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

JM1's raw count (13,204) differs from the ~10,878-10,885 commonly cited. The cause could
not be established. It is consistent with different distributors reporting different
counts for the same project (Shepperd et al., 2013), but this was not verified, so it is
recorded as an unexplained limitation. The other four projects match the cited figures.

### 2. A systematic single-row artefact
Every one of the five raw files contains, as its first data row, a row with
fractional values across multiple count-based metrics simultaneously (e.g.
loc=1.1, uniq_Op=1.2, branchCount=1.4), which are logically impossible for
count-based static code metrics. The same anomaly, at the same position, was also
found in CM1's official ARFF file from the Shepperd et al. (2013) repository,
indicating this is a long-standing artefact of the original ARFF construction
process rather than corruption introduced by any one distributor. This single row
was removed from each dataset (see the cleaning logs below).

### 3. High exact-duplicate rates
| Dataset | Raw rows | Duplicates removed | % duplicated |
|---|---|---|---|
| CM1 | 498 | 56 | 11.2% |
| JM1 | 13,204 | 4,296 | 32.5% |
| KC1 | 2,109 | 897 | 42.5% |
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
| CM1 | 9.8% | 10.9% |
| JM1 | ~19% (typical lit. figure) | 22.5% |
| KC1 | ~15% (typical lit. figure) | 26.0% |
| KC2 | ~20% (typical lit. figure) | 28.1% |
| PC1 | ~7% (typical lit. figure) | 7.4% |

This is worth discussing explicitly in the thesis: it demonstrates that the
duplicate-removal decision is not neutral with respect to class balance, which in
turn affects SMOTE's oversampling target ratio and should be noted as a limitation
or at least an explicit methodological choice.

## Cleaning steps applied (in order)
1. Standardise column names and target encoding across sources
2. Remove the single systematic artefact row (fractional count-metric values)
3. Remove exact duplicate rows (all 21 features + target identical)
4. Verify: no missing values, no zero-variance features remaining
5. Strict level only (`src/clean_data.py`): remove rows that break any of
   R1 all-ones row; R2 conflicting labels for identical metrics; R3 n = 0 while loc > 0;
   R4 n != total_Op + total_Opnd; R5 loc = 0.

Row counts, duplicate-removed level -> strict level: CM1 441 -> 438; JM1 8,907 -> 7,698
(R2 = 176, R3 = 971, R4 = 139); KC1 1,211 -> 1,147; KC2 374 -> 328; PC1 953 -> 934.
The per-rule counts overlap, so they need not sum to the number of rows removed.

## Where things are
* Implementation of the strict level: `src/clean_data.py`.
* Per-rule counts for the strict level: `experiments/results/cleaning_log.csv`.
* Counts for the duplicate-removal level: `data/processed/cleaning_log.csv`.
  These are two different files with the same name.
# Data Provenance Log

## Source
All five datasets (CM1, JM1, KC1, KC2, PC1) were sourced from a single GitHub mirror
providing the datasets in the standard 21-metric PROMISE/tera-PROMISE schema:
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(retrieved [DATE — fill in when finalised]).

## Alternative source evaluated and rejected
The Shepperd et al. (2013) NASA MDP repository (klainfo/NASADefectDataset on GitHub)
was also evaluated, since it provides documented Original/D'/D'' cleaning levels and
is the canonical source behind the data-quality methodology this project's proposal
cites. It was rejected for two reasons:
1. It does not include KC2 (a tera-PROMISE dataset, not part of the original NASA MDP
   family Shepperd et al. worked with).
2. Its CM1/PC1 files use an unreduced 41-attribute metric set, and its JM1/KC1 files
   use a differently-named 22-attribute set — neither matches the standard 21-metric
   schema this project (and the wider CPDP literature) relies on, and using a mixed
   schema across source/target projects would break the shared feature-space
   requirement cross-project transfer depends on.

## Known data-quality issues identified during validation

### 1. Instance count discrepancies with commonly-cited literature figures
| Dataset | This source (raw) | Commonly cited in literature |
|---|---|---|
| CM1 | 498 | 498 (matches) |
| JM1 | 13,204 | ~10,878–10,885 |
| KC1 | 2,109 | 2,109 (matches) |
| KC2 | 522 | 522 (matches) |
| PC1 | 1,109 | 1,109 (matches) |

JM1's discrepancy is substantial and consistent with Shepperd et al.'s (2013) general
finding that different distributors report different instance counts for the same
nominal NASA MDP project.

### 2. A systematic single-row artefact
Every one of the five raw files contains, as its first data row, a row with
fractional values across multiple count-based metrics simultaneously (e.g.
loc=1.1, uniq_Op=1.2, branchCount=1.4), which are logically impossible for
count-based static code metrics. The same anomaly, at the same position, was also
found in CM1's official ARFF file from the Shepperd et al. (2013) repository,
indicating this is a long-standing artefact of the original ARFF construction
process rather than corruption introduced by any one distributor. This single row
was removed from each dataset (see cleaning_log.csv).

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

### 4. Defect rate shift after cleaning
Removing duplicates increased the observed defect rate in every dataset, since
duplicate rows were disproportionately clean (non-defective) modules:

| Dataset | Defect rate (raw) | Defect rate (post-cleaning) |
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
1. Standardize column names and target encoding across sources
2. Remove the single systematic artefact row (fractional count-metric values)
3. Remove exact duplicate rows (all 21 features + target identical)
4. Verify: no missing values, no zero-variance features remaining

See `src/data_processing/load_data.py` for the implementation and
`data/processed/cleaning_log.csv` for the exact per-dataset counts.

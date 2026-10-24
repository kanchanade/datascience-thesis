# SHAP-Consensus Feature Reweighting (SFRT) for Cross-Project Defect Prediction

IT9115 thesis project. Five NASA PROMISE datasets (CM1, JM1, KC1, KC2, PC1),
leave-one-project-out evaluation: train on the 4 other projects pooled, test on the 5th.

## Setup
Python 3.11.

    python -m venv venv
    venv\Scripts\activate          (Windows)
    pip install -r requirements.txt

`requirements-frozen.txt` lists the exact package versions used for the reported results.

## Data
The raw files are NOT in this repository (see .gitignore). Source: GitHub mirror
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(PROMISE software engineering repository). Retrieved 2026-09-27, mirror commit
3898d21e287c130c76d458faf65793d32483076d. Full provenance, known data-quality issues and
the reasons another source was rejected are in `docs/data_provenance.md`.

* `data/processed/{name}_cleaned.csv` - exact duplicates removed (committed).
* `data/final/{name}_strict.csv`      - strict cleaning, rules R1-R5 (written by src/clean_data.py).

Two files are called `cleaning_log.csv`, and they are different:
`data/processed/cleaning_log.csv` records the duplicate-removal level, and
`experiments/results/cleaning_log.csv` records the strict level (per-rule counts).

Two cleaning levels are used throughout, and result files carry the level in their name:
`strict` is the main analysis; `duplicates` is the sensitivity analysis.

## Run order
    python src/clean_data.py                           # strict cleaning + experiments/results/cleaning_log.csv
    python src/run_experiments.py                      # all methods, seeds 0-9, strict data (long run)
    python src/run_experiments.py --level duplicates   # sensitivity analysis

Results are written to `experiments/results/runs_<level>.csv`, one row per
(method, classifier, target, seed). Runs can be stopped and restarted; finished rows are skipped.

    python src/summarise_results.py --level strict      # mean +/- sd, paired changes, selection frequencies
    python src/summarise_results.py --level duplicates
    python src/analyse_rq2.py                           # pre-specified RQ2 statistics (Wilcoxon, Friedman)
    python src/run_within_project.py                    # RQ1: within-project 10-fold CV x 10 seeds, then within vs cross comparison
    python src/analyse_rq3.py                           # RQ3: SHAP stability (noise floor, across projects, across models, grouped)
    python src/run_source_balance.py                    # source-balance ablation: equal and size-matched training tables (strict data)
    python src/analyse_source_balance.py                # summaries of the ablation
    python src/make_figures.py                          # Figures 4.1 to 4.3 into experiments/figures/

Each script takes `--level strict|duplicates` (default `strict`) and, where relevant,
`--seeds` (default `0-9`). `run_source_balance.py` also takes `--jobs` (parallel workers) and
can be stopped and restarted. The ablation was run on the strict data only.

## Code layout
* `src/clean_data.py`         - the cleaning rules (R1-R5), documented in the file header.
* `src/evaluation.py`         - ONE evaluation protocol; methods are plug-in "feature steps".
* `src/run_experiments.py`    - runs every method x classifier x target x seed.
* `src/summarise_results.py`, `src/analyse_rq2.py` - tables and statistics from the runs file.
* `src/run_within_project.py` - RQ1, within-project evaluation on the same data and metrics.
* `src/analyse_rq3.py`        - RQ3, reads the cached SHAP vectors.
* `src/run_source_balance.py`, `src/analyse_source_balance.py` - the source-balance ablation
  (does JM1's dominance of the pooled training data explain the results?).
* `src/make_figures.py`       - draws the three results figures from the saved result files.
* `notebooks/Final EDA.ipynb` - exploratory data analysis (run on the duplicate-removed data).
* `notebooks/Archive/`        - the original within-project baseline notebook, kept for reference.

Methods: `unweighted`; `variance_k{5,10,15}`; `mutualinfo_k{5,10,15}`;
SFRT: `sfrt_hard_k{5,10,15}`, `sfrt_hardstab_k*` (stability-penalised),
`sfrt_soft`, `sfrt_softstab`, `sfrt_softsharp` (Logistic Regression and MLP only,
because tree models cannot be reweighted), and `xgb_sampling_sfrt` with its control
`xgb_sampling_uniform` (XGBoost only). Per-source SHAP vectors are cached in
`experiments/cache/shap/` (also the input for the RQ3 stability analysis).

## Results files (`experiments/results/`)
| File | Content |
|---|---|
| `runs_<level>.csv` | every individual run (3,000 per level) |
| `within_<level>.csv` | within-project runs (RQ1) |
| `summary_by_method_<level>.csv` | mean +/- sd over seeds per method and classifier |
| `summary_paired_vs_unweighted_<level>.csv` | paired change versus unweighted, seeds where better |
| `summary_per_target_ci_<level>.csv` | per-target intervals |
| `selection_frequency_<level>.csv`, `selection_distinct_sets_<level>.csv` | how stable the selected feature sets are across seeds |
| `rq1_within_vs_cross_<level>.csv` | within versus cross-project, per project, one-sided Wilcoxon |
| `rq2_wilcoxon_<level>.csv`, `rq2_friedman_<level>.csv` | pre-specified RQ2 statistics |
| `rq3_*_<level>.csv` | SHAP ranking agreement, noise floor, source outliers, consensus stability |
| `ablation_balance_<level>.csv` | every ablation run (equal and size-matched training tables) |
| `ablation_balance_summary_<level>.csv` | ablation: mean +/- sd per method, classifier and table |
| `ablation_balance_change_<level>.csv` | ablation: change versus unweighted inside each table |
| `ablation_balance_mix_effect_<level>.csv` | ablation: equal table minus size-matched control |
| `cleaning_log.csv` | strict-level cleaning counts per rule and project |

Figures are written to `experiments/figures/`.

## Protocol notes
* Leave-one-project-out; ten seeds (0-9); SMOTE on the training data only.
* Scaling only for Logistic Regression and MLP, fitted on the training data.
* No hyperparameter tuning: fixed defaults for all four classifiers.
* Primary metrics: AUC-PR and MCC. F1 and AUC-ROC are reported as secondary.
* With five target projects the smallest possible one-sided Wilcoxon p-value is 1/32 = 0.031.
* The single-source setting was dropped, because SFRT needs at least two source projects.
* Source-balance ablation: an `equal` table (328 rows from each source) and a `size_matched`
  control (1,312 rows sampled from the pooled table). The SFRT consensus is always built
  from the full source projects.
* Tree-model scores can differ slightly on other hardware, so a re-run should agree to
  within small differences, not to the last digit.

## Status
Done: cleaning (both levels), baselines, SFRT variants and ablations, multi-seed runner,
RQ1 within-project protocol, RQ2 statistics, RQ3 stability analysis, source-balance
ablation, results figures.

Older scripts from the first stage of the project (run_*.py, crossproject_*.py, baseline_*.py,
sfrt_*.py) and their result files have been removed; `src/` supersedes them.
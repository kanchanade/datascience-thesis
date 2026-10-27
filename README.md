# SHAP-Consensus Feature Reweighting (SFRT) for Cross-Project Defect Prediction

IT9115 thesis project. Five NASA PROMISE datasets (CM1, JM1, KC1, KC2, PC1).
Leave-one-project-out: train on the other 4 projects pooled, test on the 5th.

## Setup
Python 3.11.

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

`requirements-frozen.txt` has the exact versions used for the reported results.

## Data
Raw files are not in this repository. Source: the PROMISE GitHub mirror
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(retrieved 2026-09-27, commit 3898d21e287c130c76d458faf65793d32483076d).
Put the CSV files in `data/raw/promise_source/`. Provenance and known data-quality
issues: `docs/data_provenance.md`.

* `data/processed/` - exact duplicates removed (committed). Sensitivity analysis.
* `data/final/` - strict cleaning, rules R1-R5, written by `src/clean_data.py`. Main analysis.

## Run order
    python src/clean_data.py
    python src/run_experiments.py                      # strict data, seeds 0-9 (long; can be restarted)
    python src/run_experiments.py --level duplicates
    python src/summarise_results.py --level strict
    python src/summarise_results.py --level duplicates
    python src/analyse_rq2.py                          # RQ2 statistics
    python src/run_within_project.py                   # RQ1
    python src/analyse_rq3.py                          # RQ3
    python src/run_source_balance.py                   # ablation
    python src/analyse_source_balance.py
    python src/make_figures.py                         # Figures 4.1 to 4.3

Logistic Regression at C = 0.1 and 10 (exploratory), run in a scratch copy:

    mkdir lr_sweep
    Copy-Item data lr_sweep\data -Recurse
    python src/run_experiments.py --root lr_sweep --classifiers --lr-c-sweep --methods unweighted mutualinfo_k10 sfrt_soft sfrt_hard_k5 sfrt_hard_k10 variance_k10 --jobs 4
    python src/analyse_lr_c.py --sweep-root lr_sweep

(The empty `--classifiers` is intentional. If a run stops with `PermissionError` on
Windows, rerun with `--jobs 1`.)

## Code
* `src/evaluation.py` - the one evaluation protocol; every method is a plug-in feature step.
* `src/run_*.py` - run the experiments. `src/analyse_*.py`, `summarise_results.py` - statistics and tables.
* `src/make_figures.py` - figures from the saved results.
* `notebooks/Final EDA.ipynb` - EDA. Open it from `notebooks/` and use Restart and Run All.

## Results (`experiments/results/`)
One CSV per table, named by level (`strict` or `duplicates`). `runs_<level>.csv` holds
every run (one row per method, classifier, target and seed). Summaries start with
`summary_`, RQ statistics with `rq1_`, `rq2_`, `rq3_`, the ablation with `ablation_`,
the C sweep with `lr_c_sweep_`. SHAP vectors are cached in `experiments/cache/shap/`.

## Protocol notes
* Ten seeds (0-9); SMOTE on training data only; scaling for Logistic Regression and MLP only.
* No hyperparameter tuning (fixed defaults).
* Primary metrics: AUC-PR and MCC. Secondary: F1, AUC-ROC.
* Five targets, so the smallest one-sided Wilcoxon p is 1/32 = 0.031.
* Scores can differ slightly on other hardware.
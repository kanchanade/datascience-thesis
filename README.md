# SHAP-Consensus Feature Reweighting (SFRT) for Cross-Project Defect Prediction

IT9115 thesis project. Five NASA PROMISE datasets (CM1, JM1, KC1, KC2, PC1),
leave-one-project-out evaluation: train on 4 projects pooled, test on the 5th.

## Setup
    python -m venv venv
    venv\Scripts\activate          (Windows)
    pip install -r requirements.txt

## Data
The raw files are NOT in this repository (see .gitignore). Source: GitHub mirror
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
(PROMISE software engineering repository). Retrieval date: **FILL IN** and
the mirror's commit hash: **FILL IN** (see docs/data_provenance.md).

* `data/processed/{name}_cleaned.csv` - exact duplicates removed (committed).
* `data/final/{name}_strict.csv`      - strict cleaning (written by src/clean_data.py).

## Run order
    python src/clean_data.py                 # strict cleaning + experiments/results/cleaning_log.csv
    python src/run_experiments.py            # all methods, seeds 0-9, strict data  (long run)
    python src/run_experiments.py --level duplicates   # sensitivity analysis

Results: `experiments/results/runs_<level>.csv`, one row per
(method, classifier, target, seed). Runs can be stopped and restarted.

    python src/summarise_results.py          # mean +/- sd, paired changes, selection frequencies
    python src/analyse_rq2.py                # pre-specified RQ2 statistics (Wilcoxon, Friedman)
    python src/run_experiments.py --lr-c-sweep   # optional: Logistic Regression at C = 0.1 and 10
    python src/run_within_project.py         # RQ1: within-project 10-fold CV x 10 seeds, then within vs cross comparison
    python src/analyse_rq3.py                # RQ3: SHAP stability (noise floor, across projects, across models, grouped)

## Code layout
* `src/clean_data.py`      - the cleaning rules (R1-R5), documented in the file header.
* `src/evaluation.py`      - ONE evaluation protocol; methods are plug-in "feature steps".
* `src/run_experiments.py` - runs every method x classifier x target x seed.
* `src/summarise_results.py`, `src/analyse_rq2.py` - tables and statistics from the runs file.
* `src/run_within_project.py` (RQ1) and `src/analyse_rq3.py` (RQ3, reads the SHAP cache).

Methods: `unweighted`; `variance_k{5,10,15}`; `mutualinfo_k{5,10,15}`;
SFRT: `sfrt_hard_k{5,10,15}`, `sfrt_hardstab_k*` (stability-penalised),
`sfrt_soft`, `sfrt_softstab`, `sfrt_softsharp` (Logistic Regression and MLP only,
because tree models cannot be reweighted), `xgb_sampling_sfrt` and its control
`xgb_sampling_uniform` (XGBoost only). Per-source SHAP vectors are cached in
`experiments/cache/shap/` (also the input for the RQ3 stability analysis).

## Status
Done: cleaning, baselines, SFRT variants and ablations, multi-seed runner, RQ1 within-project
protocol, RQ2 statistics, RQ3 stability analysis.
Not yet: charts for the new results; source-imbalance (equal rows per source) ablation.
Decisions recorded: single-source setting dropped (SFRT needs >= 2 sources); no
hyperparameter tuning (fixed defaults; Logistic Regression also run at C = 0.1, 10).
Primary metrics: AUC-PR and MCC. Older scripts (run_*.py, crossproject_*.py,
baseline_*.py, sfrt_*.py) are superseded by src/.

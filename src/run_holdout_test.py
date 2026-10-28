"""
Confirmation test on NEW projects (hold-out test).

WHY. The pattern "SFRT top-5 helps Logistic Regression and the MLP" was noticed
AFTER the main results were seen, on the same five projects. A pattern found that
way cannot be confirmed on the same data. This script tests it once, on seven
NASA projects that the study has never used, with every choice fixed in advance.

THE PLAN BELOW WAS FIXED ON 28 OCTOBER 2026, BEFORE ANY HOLD-OUT RESULT EXISTED.
Do not change any setting after running. Commit this file to Git BEFORE the first
run, and quote that commit in Chapter 3.

Training data : the five main projects (CM1, JM1, KC1, KC2, PC1, strict level), pooled.
                The SFRT consensus is built from these five projects (five SHAP vectors).
                The hold-out project is never used for any fitted step.
Targets       : the projects kept by src/prepare_holdout_data.py
                (KC3, MC1, MC2, MW1, PC3, PC4, PC5).
Methods       : unweighted (21 features), mutualinfo_k5, sfrt_hard_k5.
Classifiers   : Logistic Regression, MLP (primary); Random Forest, XGBoost (secondary).
Seeds         : 0-9. Every other setting is exactly as in src/evaluation.py.

PRIMARY HYPOTHESIS (one per classifier, Logistic Regression and MLP):
  SFRT top-5 scores higher than the unweighted model on the hold-out targets,
  with the full training table.
  Test: one-sided Wilcoxon signed-rank over the targets, using each target's
  seed-averaged difference, separately for MCC and AUC-PR. A classifier counts as a
  success only if BOTH metrics pass, so its p-value is the larger of the two.
  The two classifiers are then corrected with Holm's method at 0.05.
  (With 7 targets the smallest possible one-sided p is 1/128 = 0.0078.)

SECONDARY (reported with their p-values; no correction, no confirmatory claim):
  S1  SFRT top-5 against mutual information top-5 (Logistic Regression, MLP).
  S2  The same comparisons with a small training table: 1,640 rows (5 x 328) drawn
      at random from the pooled training data, keeping its defect rate.
  S3  Random Forest and XGBoost: SFRT top-5 against unweighted. The expectation from
      the main study is NO gain for tree models.

Run:   python src/run_holdout_test.py            (runs, then analyses)
       python src/run_holdout_test.py --analyse  (analysis only, from the saved runs)
       optional: --jobs 4
Writes experiments/results/holdout_runs.csv, holdout_primary.csv, holdout_secondary.csv
Safe to stop and restart: finished runs are skipped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import evaluation as ev

METHODS = {"unweighted": ev.unweighted_step(),
           "mutualinfo_k5": ev.mutual_info_step(5),
           "sfrt_hard_k5": ev.sfrt_hard_step(5)}
PRIMARY_CLASSIFIERS = ["Logistic Regression", "MLP"]
SECONDARY_CLASSIFIERS = ["Random Forest", "XGBoost"]
TABLES = {"full": None, "small": 5 * 328}
SEEDS = list(range(10))
METRICS = ["MCC", "AUC-PR"]
ALPHA = 0.05


def load_holdout(root):
    folder = root / "data" / "holdout"
    data = {}
    for path in sorted(folder.glob("*_strict.csv")):
        frame = pd.read_csv(path)
        data[path.name.split("_")[0].upper()] = (frame[ev.FEATURE_COLUMNS], frame["defects"].astype(int))
    if not data:
        raise SystemExit("No hold-out files found. Run src/prepare_holdout_data.py first.")
    return data


def training_table(sources, table, seed):
    X = pd.concat([sources[n][0] for n in ev.DATASET_NAMES], ignore_index=True)
    y = pd.concat([sources[n][1] for n in ev.DATASET_NAMES], ignore_index=True)
    if TABLES[table] is None:
        return X, y
    rng = np.random.default_rng([seed, 7])
    return ev._stratified_sample(X, y, TABLES[table], rng)


def one_run(sources, X_target, y_target, target, table, method, classifier, seed):
    """Same steps as ev.evaluate, with the five main projects as sources."""
    start = time.time()
    np.random.seed(seed)
    X_source, y_source = training_table(sources, table, seed)
    source_parts = {n: sources[n] for n in ev.DATASET_NAMES}   # consensus from the full projects
    kept, weights, info = METHODS[method](X_source, y_source, source_parts, classifier, seed)

    Xs, Xt = X_source.values, X_target.values
    if kept is not None:
        Xs, Xt = Xs[:, kept], Xt[:, kept]
    if ev.family(classifier) in ev.NEEDS_SCALING:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.transform(Xt)
    Xb, yb = SMOTE(random_state=seed).fit_resample(Xs, y_source)
    model = ev.make_classifier(classifier, seed)
    model.fit(Xb, yb)
    predicted = model.predict(Xt)
    probability = model.predict_proba(Xt)[:, 1]
    kept_names = ev.FEATURE_COLUMNS if kept is None else [ev.FEATURE_COLUMNS[i] for i in kept]
    return {"Table": table, "Method": method, "Classifier": classifier, "Target": target, "Seed": seed,
            "F1": round(f1_score(y_target, predicted, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_target, predicted), 4),
            "AUC-ROC": round(roc_auc_score(y_target, probability), 4),
            "AUC-PR": round(average_precision_score(y_target, probability), 4),
            "Selected_Features": " ".join(kept_names), "Seconds": round(time.time() - start, 1)}


def run_all(root, jobs):
    sources = ev.load_all_data("strict", root)
    holdout = load_holdout(root)
    out_path = root / "experiments" / "results" / "holdout_runs.csv"
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.Table, old.Method, old.Classifier, old.Target, old.Seed))
    classifiers = PRIMARY_CLASSIFIERS + SECONDARY_CLASSIFIERS
    for seed in SEEDS:
        tasks = [(t, tab, m, c) for tab in TABLES for c in classifiers for m in METHODS for t in holdout
                 if (tab, m, c, t, seed) not in done]
        if not tasks:
            continue
        start = time.time()
        rows = Parallel(n_jobs=jobs)(
            delayed(one_run)(sources, *holdout[t], t, tab, m, c, seed) for t, tab, m, c in tasks)
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"seed {seed}: {len(rows)} runs in {time.time() - start:.0f}s")


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def paired(runs, table, classifier, method, baseline, metric):
    """Seed-averaged difference per target, then a one-sided Wilcoxon test."""
    frame = runs[(runs.Table == table) & (runs.Classifier == classifier)]
    cell = frame.groupby(["Method", "Target"])[metric].mean()
    diff = (cell.loc[method] - cell.loc[baseline]).dropna()
    p = 1.0 if (diff == 0).all() else stats.wilcoxon(diff, alternative="greater").pvalue
    return {"Targets": len(diff), "Targets_better": int((diff > 0).sum()),
            "Mean_difference": round(diff.mean(), 4), "p_one_sided": round(p, 4),
            "Rank_biserial": round(rank_biserial(diff), 2)}


def analyse(root):
    folder = root / "experiments" / "results"
    runs = pd.read_csv(folder / "holdout_runs.csv")
    expected = len(TABLES) * len(METHODS) * 4 * runs.Target.nunique() * len(SEEDS)
    if len(runs) < expected:
        print(f"WARNING: {len(runs)} of {expected} runs present; finish the runs before reporting.")

    # ---- primary: intersection-union over the two metrics, Holm over the two classifiers ----
    primary = []
    for classifier in PRIMARY_CLASSIFIERS:
        row = {"Classifier": classifier, "Comparison": "sfrt_hard_k5 vs unweighted", "Table": "full"}
        p_values = []
        for metric in METRICS:
            result = paired(runs, "full", classifier, "sfrt_hard_k5", "unweighted", metric)
            row.update({f"{metric}_{k}": v for k, v in result.items() if k != "Targets"})
            p_values.append(result["p_one_sided"])
        row["Targets"] = result["Targets"]
        row["p_classifier"] = max(p_values)
        primary.append(row)
    primary = pd.DataFrame(primary).sort_values("p_classifier").reset_index(drop=True)
    m = len(primary)
    passed, still_going = [], True
    for i, p in enumerate(primary.p_classifier):
        threshold = ALPHA / (m - i)
        still_going = still_going and p <= threshold
        passed.append(still_going)
    primary["Holm_threshold"] = [round(ALPHA / (m - i), 4) for i in range(m)]
    primary["Supported"] = passed
    primary.to_csv(folder / "holdout_primary.csv", index=False)

    # ---- secondary ----
    secondary = []
    plan = ([("full", c, "mutualinfo_k5") for c in PRIMARY_CLASSIFIERS] +
            [("small", c, b) for c in PRIMARY_CLASSIFIERS for b in ("unweighted", "mutualinfo_k5")] +
            [(t, c, "unweighted") for t in TABLES for c in SECONDARY_CLASSIFIERS])
    for table, classifier, baseline in plan:
        for metric in METRICS:
            secondary.append({"Table": table, "Classifier": classifier,
                              "Comparison": f"sfrt_hard_k5 vs {baseline}", "Metric": metric,
                              **paired(runs, table, classifier, "sfrt_hard_k5", baseline, metric)})
    secondary = pd.DataFrame(secondary)
    secondary.to_csv(folder / "holdout_secondary.csv", index=False)

    means = (runs.groupby(["Table", "Classifier", "Method"])[METRICS].mean().round(3).reset_index())
    means.to_csv(folder / "holdout_means.csv", index=False)

    print("PRIMARY (pre-specified): SFRT top-5 vs unweighted, full training table")
    print(primary.to_string(index=False))
    print("\nSECONDARY (no confirmatory claim)")
    print(secondary.to_string(index=False))
    print("\nMEANS over targets and seeds")
    print(means.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", action="store_true", help="analysis only")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()
    ev.CACHE_DIR = args.root / "experiments" / "cache" / "shap"
    if not args.analyse:
        run_all(args.root, args.jobs)
    analyse(args.root)


if __name__ == "__main__":
    main()

"""
Within-project evaluation (RQ1), on the same data, classifiers and metrics as
the cross-project experiments, so the two settings can be compared directly.

    RQ1: How do standard classifiers perform for within-project versus
         cross-project defect prediction?

Protocol (same as the original Step 2 notebook, now seed-controlled):
  * stratified 10-fold cross-validation INSIDE one project;
  * per fold: scale (Logistic Regression and MLP only), SMOTE on the TRAINING
    fold only, train, score on the untouched test fold;
  * the seed sets the fold shuffle, SMOTE and the classifier;
  * a project's score for one seed = the mean over its 10 folds.
Ten seeds give the same seed-to-seed spread as the cross-project runs.

Two comparability notes to state in the thesis:
  * a within-project model trains on ~90% of ONE project; a cross-project model
    trains on FOUR pooled projects (JM1-dominated). Sizes and mixes differ.
  * random folds can put near-duplicate modules on both sides of a split,
    which flatters within-project scores; strict cleaning removes conflicting
    duplicates but not near-duplicates.

Run:  python src/run_within_project.py [--level strict] [--seeds 0-9] [--jobs 2]
Writes experiments/results/within_<level>.csv  (one row per classifier/project/seed)
and prints the within- vs cross-project comparison if runs_<level>.csv exists.
Safe to stop and restart (finished rows are skipped).
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import evaluation as ev
from run_experiments import parse_seeds

METRICS = ["F1", "MCC", "AUC-ROC", "AUC-PR"]


def within_run(data, level, classifier_name, project, seed, folds=10):
    start = time.time()
    np.random.seed(seed)
    X, y = data[project]
    X, y = X.values, y.values
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_scores = []
    for train_rows, test_rows in splitter.split(X, y):
        Xtr, Xte, ytr, yte = X[train_rows], X[test_rows], y[train_rows], y[test_rows]
        if ev.family(classifier_name) in ev.NEEDS_SCALING:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
        model = ev.make_classifier(classifier_name, seed)
        model.fit(Xtr, ytr)
        predicted = model.predict(Xte)
        probability = model.predict_proba(Xte)[:, 1]
        fold_scores.append({
            "F1": f1_score(yte, predicted, zero_division=0),
            "MCC": matthews_corrcoef(yte, predicted),
            "AUC-ROC": roc_auc_score(yte, probability),
            "AUC-PR": average_precision_score(yte, probability),
        })
    mean_scores = pd.DataFrame(fold_scores).mean()
    return {"Level": level, "Setting": "within-project", "Classifier": classifier_name,
            "Target_Dataset": project.upper(), "Seed": seed,
            **{m: round(float(mean_scores[m]), 4) for m in METRICS},
            "Seconds": round(time.time() - start, 1)}


def compare_with_cross_project(folder, level, within):
    runs_path = folder / f"runs_{level}.csv"
    if not runs_path.exists():
        print("(runs file for cross-project not found - skipping the comparison)")
        return
    cross = pd.read_csv(runs_path)
    cross = cross[cross.Method == "unweighted"]
    seeds = sorted(set(within.Seed) & set(cross.Seed))
    within, cross = within[within.Seed.isin(seeds)], cross[cross.Seed.isin(seeds)]

    # per seed: average over projects, then mean +/- sd over seeds (same convention as the cross-project table)
    rows = []
    for clf in ev.CLASSIFIER_NAMES:
        for label, frame in (("within-project", within), ("cross-project", cross)):
            per_seed = frame[frame.Classifier == clf].groupby("Seed")[METRICS].mean()
            row = {"Classifier": clf, "Setting": label}
            for m in ("MCC", "AUC-PR", "AUC-ROC"):
                row[m] = f"{per_seed[m].mean():.3f} +/- {per_seed[m].std(ddof=1):.3f}"
            rows.append(row)
    print(f"\nWITHIN vs CROSS-PROJECT, unweighted baseline ({len(seeds)} seeds; mean +/- sd over seeds)")
    print(pd.DataFrame(rows).to_string(index=False))

    # per project: is within better than cross, on every project? (5 projects -> min one-sided p = 0.031)
    out = []
    for clf in ev.CLASSIFIER_NAMES:
        for m in ("MCC", "AUC-PR"):
            w = within[within.Classifier == clf].groupby("Target_Dataset")[m].mean()
            c = cross[cross.Classifier == clf].groupby("Target_Dataset")[m].mean()
            diff = (w - c).reindex(sorted(w.index))
            p = stats.wilcoxon(diff, alternative="greater").pvalue if (diff != 0).any() else np.nan
            out.append({"Classifier": clf, "Metric": m, "Projects_within_better": f"{int((diff > 0).sum())}/5",
                        "Mean_within_minus_cross": round(diff.mean(), 3), "p_one_sided": round(p, 4)})
    table = pd.DataFrame(out)
    table.to_csv(folder / f"rq1_within_vs_cross_{level}.csv", index=False)
    print("\nPER-PROJECT DIFFERENCE (within minus cross), one-sided Wilcoxon over 5 projects")
    print(table.to_string(index=False))
    print("\nReminder: with 5 projects the smallest possible p is 0.031.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--classifiers", nargs="*", default=ev.CLASSIFIER_NAMES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()

    folder = args.root / "experiments" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"within_{args.level}.csv"
    data = ev.load_all_data(args.level, args.root)

    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.Classifier, old.Target_Dataset, old.Seed))

    for seed in parse_seeds(args.seeds):
        tasks = [(c, p) for c in args.classifiers for p in ev.DATASET_NAMES
                 if (c, p.upper(), seed) not in done]
        if not tasks:
            print(f"seed {seed}: already complete")
            continue
        start = time.time()
        rows = Parallel(n_jobs=args.jobs)(
            delayed(within_run)(data, args.level, c, p, seed) for c, p in tasks)
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"seed {seed}: {len(rows)} runs in {time.time() - start:.0f}s -> {out_path.name}")

    compare_with_cross_project(folder, args.level, pd.read_csv(out_path))


if __name__ == "__main__":
    main()

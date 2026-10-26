"""
Logistic Regression at three regularisation strengths (C = 0.1, 1, 10).

Does the size of SFRT's effect on Logistic Regression depend on C? Reads the
main runs (C = 1) and the sweep runs (C = 0.1 and 10) and reports, for each C,
the unweighted score and the change of each method against it.

Run:  python src/analyse_lr_c.py [--level strict] [--sweep-root <folder>]
Writes experiments/results/lr_c_sweep_<level>.csv (exploratory, no significance claims).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import evaluation as ev

METHODS = ["mutualinfo_k10", "variance_k10", "sfrt_soft", "sfrt_hard_k5", "sfrt_hard_k10"]
LABELS = {"Logistic Regression": 1.0, "Logistic Regression C=0.1": 0.1, "Logistic Regression C=10": 10.0}


def seed_target(frame, method, metric):
    return frame[frame.Method == method].groupby(["Seed", "Target_Dataset"])[metric].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT, help="main results (C = 1)")
    parser.add_argument("--sweep-root", type=Path, default=None,
                        help="folder whose experiments/results holds the sweep runs (default: --root)")
    args = parser.parse_args()
    sweep_root = args.sweep_root or args.root

    main_runs = pd.read_csv(args.root / "experiments" / "results" / f"runs_{args.level}.csv")
    sweep_runs = pd.read_csv(sweep_root / "experiments" / "results" / f"runs_{args.level}.csv")
    runs = pd.concat([main_runs[main_runs.Classifier == "Logistic Regression"],
                      sweep_runs[sweep_runs.Classifier.isin(LABELS) & (sweep_runs.Classifier != "Logistic Regression")]])

    rows = []
    for name, c_value in sorted(LABELS.items(), key=lambda kv: kv[1]):
        frame = runs[runs.Classifier == name]
        for metric in ["MCC", "AUC-PR"]:
            base = seed_target(frame, "unweighted", metric)
            per_seed = base.groupby("Seed").mean()
            rows.append({"C": c_value, "Metric": metric, "Method": "unweighted",
                         "Score_or_change": round(per_seed.mean(), 4), "Seeds_better": "",
                         "Targets_improved_of_5": "", "p_one_sided": ""})
            for method in METHODS:
                diff = seed_target(frame, method, metric) - base
                seed_change = diff.groupby("Seed").mean()
                target_change = diff.groupby("Target_Dataset").mean()
                p = stats.wilcoxon(target_change, alternative="greater").pvalue if target_change.abs().sum() > 0 else 1.0
                rows.append({"C": c_value, "Metric": metric, "Method": method,
                             "Score_or_change": round(seed_change.mean(), 4),
                             "Seeds_better": f"{int((seed_change > 0).sum())}/{len(seed_change)}",
                             "Targets_improved_of_5": f"{int((target_change > 0).sum())}/5",
                             "p_one_sided": round(p, 4)})
    table = pd.DataFrame(rows)
    out = args.root / "experiments" / "results" / f"lr_c_sweep_{args.level}.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()

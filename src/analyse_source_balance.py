"""
Summarise the source-balance ablation (run_source_balance.py).

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import evaluation as ev

METRICS = ["MCC", "AUC-PR"]
ORDER = ["full", "size_matched", "equal"]


def seed_means(frame, metric):
    """Mean over targets within each seed, then mean +/- sd over seeds."""
    per_seed = frame.groupby("Seed")[metric].mean()
    return per_seed.mean(), per_seed.std(ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()
    folder = args.root / "experiments" / "results"

    abl = pd.read_csv(folder / f"ablation_balance_{args.level}.csv")
    full = pd.read_csv(folder / f"runs_{args.level}.csv")
    methods = [m for m in abl.Method.unique()]
    full = full[full.Method.isin(methods)].assign(Pool="full")
    seeds = sorted(set(abl.Seed) & set(full.Seed))
    everything = pd.concat([full, abl], ignore_index=True)
    everything = everything[everything.Seed.isin(seeds)]
    print(f"level = {args.level}; seeds = {seeds}; rows per source = {sorted(abl.Rows_per_source.unique())}")

    summary, change = [], []
    for clf in ev.CLASSIFIER_NAMES:
        for method in methods:
            for pool in ORDER:
                part = everything[(everything.Classifier == clf) & (everything.Method == method) & (everything.Pool == pool)]
                if part.empty:
                    continue
                row = {"Classifier": clf, "Method": method, "Pool": pool}
                for m in METRICS:
                    mean, sd = seed_means(part, m)
                    row[m] = f"{mean:.3f} +/- {sd:.3f}"
                    row[f"{m}_mean"] = round(mean, 4)
                summary.append(row)
        for pool in ORDER:                                     # change vs unweighted, inside each condition
            base = everything[(everything.Classifier == clf) & (everything.Method == "unweighted") & (everything.Pool == pool)]
            for method in methods:
                if method == "unweighted":
                    continue
                part = everything[(everything.Classifier == clf) & (everything.Method == method) & (everything.Pool == pool)]
                if part.empty:
                    continue
                row = {"Classifier": clf, "Method": method, "Pool": pool}
                for m in METRICS:
                    diff = part.groupby("Seed")[m].mean() - base.groupby("Seed")[m].mean()
                    row[f"{m}_change"] = round(diff.mean(), 3)
                    row[f"{m}_seeds_better"] = f"{int((diff > 0).sum())}/{len(diff)}"
                change.append(row)
    summary = pd.DataFrame(summary)
    change = pd.DataFrame(change)
    summary.drop(columns=[c for c in summary if c.endswith("_mean")]).to_csv(folder / f"ablation_balance_summary_{args.level}.csv", index=False)
    change.to_csv(folder / f"ablation_balance_change_{args.level}.csv", index=False)

    show = summary.drop(columns=[c for c in summary if c.endswith("_mean")])
    print("\nMEAN +/- SD OVER SEEDS (target-averaged)")
    print(show.to_string(index=False))
    print("\nCHANGE VS UNWEIGHTED, INSIDE EACH TRAINING TABLE (mean over seeds; seeds where better)")
    print(change.to_string(index=False))

    print("\nSOURCE-MIX EFFECT ON THE UNWEIGHTED BASELINE: equal minus size_matched, per target (one-sided Wilcoxon over 5 targets)")
    rows = []
    for clf in ev.CLASSIFIER_NAMES:
        for m in METRICS:
            e = abl[(abl.Pool == "equal") & (abl.Method == "unweighted") & (abl.Classifier == clf)].groupby("Target_Dataset")[m].mean()
            s = abl[(abl.Pool == "size_matched") & (abl.Method == "unweighted") & (abl.Classifier == clf)].groupby("Target_Dataset")[m].mean()
            diff = (e - s).dropna()
            p = stats.wilcoxon(diff, alternative="greater").pvalue if (diff != 0).any() else np.nan
            rows.append({"Classifier": clf, "Metric": m, "Targets_better": f"{int((diff > 0).sum())}/{len(diff)}",
                         "Mean_difference": round(diff.mean(), 3), "p_one_sided": round(p, 4)})
    table = pd.DataFrame(rows)
    table.to_csv(folder / f"ablation_balance_mix_effect_{args.level}.csv", index=False)
    print(table.to_string(index=False))
    print("\nReminder: with 5 targets the smallest possible p is 0.031; this analysis is exploratory.")


if __name__ == "__main__":
    main()

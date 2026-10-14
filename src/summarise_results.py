"""
Turns the long per-seed results file into the tables.

Reads  : experiments/results/runs_<level>.csv   (one row per method/classifier/target/seed)
Writes : experiments/results/summary_by_method_<level>.csv      mean +/- sd over seeds
         experiments/results/summary_paired_vs_unweighted_<level>.csv
                                                          change vs unweighted, seed by seed
         experiments/results/summary_per_target_ci_<level>.csv   per-target change with 95% interval
         experiments/results/selection_frequency_<level>.csv     how often each feature was chosen
         experiments/results/selection_distinct_sets_<level>.csv how many different feature sets appeared

How the numbers are built (same convention as the review's Table 8):
  1. For each seed, average a score over the 5 target projects.
  2. Then take the mean and standard deviation of that over the seeds.
Run:  python src/summarise_results.py [--level strict]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import evaluation as ev

METRICS = ["MCC", "AUC-PR", "F1", "AUC-ROC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()

    folder = args.root / "experiments" / "results"
    runs = pd.read_csv(folder / f"runs_{args.level}.csv")
    n_seeds = runs.Seed.nunique()
    print(f"{len(runs)} runs, {n_seeds} seeds, level = {args.level}\n")

    # ---- 1. mean +/- sd over seeds (each seed first averaged over targets) ----
    per_seed = runs.groupby(["Classifier", "Method", "Seed"])[METRICS].mean().reset_index()
    rows = []
    for (clf, method), g in per_seed.groupby(["Classifier", "Method"]):
        row = {"Classifier": clf, "Method": method, "Seeds": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = round(g[m].mean(), 3)
            row[f"{m}_sd"] = round(g[m].std(ddof=1), 3) if len(g) > 1 else np.nan
        rows.append(row)
    by_method = pd.DataFrame(rows)
    by_method.to_csv(folder / f"summary_by_method_{args.level}.csv", index=False)
    print("MEAN +/- SD OVER SEEDS (MCC and AUC-PR)")
    show = by_method.assign(
        MCC=lambda d: d.MCC_mean.map("{:.3f}".format) + " +/- " + d.MCC_sd.map("{:.3f}".format),
        AUC_PR=lambda d: d["AUC-PR_mean"].map("{:.3f}".format) + " +/- " + d["AUC-PR_sd"].map("{:.3f}".format),
    )[["Classifier", "Method", "MCC", "AUC_PR"]]
    print(show.to_string(index=False), "\n")

    # ---- 2. paired change vs unweighted, seed by seed ----
    base = per_seed[per_seed.Method == "unweighted"].set_index(["Classifier", "Seed"])
    rows = []
    for (clf, method), g in per_seed[per_seed.Method != "unweighted"].groupby(["Classifier", "Method"]):
        g = g.set_index("Seed")
        row = {"Classifier": clf, "Method": method}
        for m in ["MCC", "AUC-PR"]:
            diff = g[m] - base.loc[clf][m].reindex(g.index)
            row[f"{m}_mean_change"] = round(diff.mean(), 3)
            row[f"{m}_seeds_better"] = f"{int((diff > 0).sum())}/{len(diff)}"
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(folder / f"summary_paired_vs_unweighted_{args.level}.csv", index=False)
    print("PAIRED CHANGE VS UNWEIGHTED (mean over seeds; seeds where better)")
    print(paired.to_string(index=False), "\n")

    # ---- 3. per-target change with a 95% interval over seeds ----
    # (does the difference on THIS target exceed seed noise?  averaged over classifiers)
    per_target = runs.groupby(["Method", "Target_Dataset", "Seed"])[["MCC", "AUC-PR"]].mean().reset_index()
    base_t = per_target[per_target.Method == "unweighted"].set_index(["Target_Dataset", "Seed"])
    rows = []
    for (method, target), g in per_target[per_target.Method != "unweighted"].groupby(["Method", "Target_Dataset"]):
        g = g.set_index("Seed")
        row = {"Method": method, "Target": target}
        for m in ["MCC", "AUC-PR"]:
            diff = g[m] - base_t.loc[target][m].reindex(g.index)
            mean, n = diff.mean(), len(diff)
            half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            row[f"{m}_change"] = round(mean, 3)
            row[f"{m}_CI95_low"] = round(mean - half, 3)
            row[f"{m}_CI95_high"] = round(mean + half, 3)
            row[f"{m}_exceeds_seed_noise"] = bool(n > 1 and (mean - half > 0 or mean + half < 0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / f"summary_per_target_ci_{args.level}.csv", index=False)
    print(f"Per-target intervals saved to summary_per_target_ci_{args.level}.csv\n")

    # ---- 4. selection frequency. Only methods that actually drop features.
    #         Variance and mutual-information selection do not depend on the classifier
    #         (use one classifier's rows). SFRT selection DOES depend on the classifier
    #         (SHAP comes from that classifier's models), so it is kept per classifier. ----
    sel = runs[runs.Features_Kept < len(ev.FEATURE_COLUMNS)].copy()
    sel["Group"] = np.where(sel.Method.str.startswith("sfrt"), sel.Method + " [" + sel.Classifier + "]", sel.Method)
    sel = sel[sel.Method.str.startswith("sfrt") | (sel.Classifier == sel.Classifier.iloc[0])]
    sel = sel.drop_duplicates(["Group", "Target_Dataset", "Seed"])
    freq_rows, set_rows = [], []
    for (method, target), g in sel.groupby(["Group", "Target_Dataset"]):
        sets = g.Selected_Features.str.split()
        n = len(sets)
        for feature in ev.FEATURE_COLUMNS:
            freq_rows.append({"Method": method, "Target": target, "Feature": feature,
                              "Selected_in_seeds": int(sum(feature in s for s in sets)),
                              "Frequency": round(sum(feature in s for s in sets) / n, 2)})
        set_rows.append({"Method": method, "Target": target, "Seeds": n,
                         "Distinct_feature_sets": g.Selected_Features.nunique(),
                         "Always_selected": " ".join(f for f in ev.FEATURE_COLUMNS if all(f in s for s in sets))})
    pd.DataFrame(freq_rows).to_csv(folder / f"selection_frequency_{args.level}.csv", index=False)
    distinct = pd.DataFrame(set_rows)
    distinct.to_csv(folder / f"selection_distinct_sets_{args.level}.csv", index=False)
    print("SELECTION STABILITY (distinct feature sets over seeds; features chosen in every seed)")
    print(distinct.to_string(index=False))


if __name__ == "__main__":
    main()

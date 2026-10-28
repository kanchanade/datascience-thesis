"""
Exploratory tables for Chapter 4: where does SFRT help?

These tables were NOT part of the pre-specified RQ2 plan. They describe a pattern
that was noticed after the main results were seen, so the thesis must label them
exploratory and must not attach significance claims to them. The pattern is then
tested properly on new projects by src/run_holdout_test.py.

Reads  : experiments/results/runs_<level>.csv              (full training table)
         experiments/results/ablation_balance_<level>.csv  (equal and size-matched tables)
         experiments/results/rq3_consensus_stability_<level>.csv
Writes : experiments/results/exploratory_top5_<level>.csv
         experiments/results/exploratory_selection_stability_<level>.csv

Run:  python src/make_exploratory_tables.py [--level strict]

Table 1 compares SFRT top-5 (sfrt_hard_k5) with
  (a) the unweighted model (all 21 features), and
  (b) mutual information top-5 (mutualinfo_k5), the same number of features,
for every classifier and for the three training tables:
  full         all rows of the four source projects (the main protocol)
  size_matched 1,312 rows sampled from the pooled sources (natural mix, mostly JM1)
  equal        328 rows from each source project
For each comparison it reports the mean change over targets and seeds, on how
many of the 5 targets the change is positive (target = mean over 10 seeds), and on
how many of the 10 seeds it is positive (seed = mean over 5 targets).

Table 2 compares how often the same top-k features are chosen again when only the
seed changes (mean pairwise Jaccard overlap over the 10 seeds; 1 = always the same).
"""

import argparse
from pathlib import Path

import pandas as pd

import evaluation as ev

POOLS = ["full", "size_matched", "equal"]
COMPARISONS = [("sfrt_hard_k5", "unweighted"), ("sfrt_hard_k5", "mutualinfo_k5")]
METRICS = ["MCC", "AUC-PR"]


def load_runs(folder, level):
    full = pd.read_csv(folder / f"runs_{level}.csv")
    full["Pool"] = "full"
    ablation = pd.read_csv(folder / f"ablation_balance_{level}.csv")
    keep = ["Pool", "Method", "Classifier", "Target_Dataset", "Seed"] + METRICS
    return pd.concat([full[keep], ablation[keep]], ignore_index=True)


def compare(runs, classifier, pool, method, baseline, metric):
    frame = runs[(runs.Classifier == classifier) & (runs.Pool == pool)]
    a = frame[frame.Method == method].set_index(["Target_Dataset", "Seed"])[metric]
    b = frame[frame.Method == baseline].set_index(["Target_Dataset", "Seed"])[metric]
    diff = (a - b).dropna()
    if diff.empty:
        return None
    by_target = diff.groupby(level="Target_Dataset").mean()
    by_seed = diff.groupby(level="Seed").mean()
    return {"change": round(diff.mean(), 3),
            "targets_better": f"{int((by_target > 0).sum())}/{len(by_target)}",
            "seeds_better": f"{int((by_seed > 0).sum())}/{len(by_seed)}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()
    folder = args.root / "experiments" / "results"

    runs = load_runs(folder, args.level)
    rows = []
    for classifier in ev.CLASSIFIER_NAMES:
        for pool in POOLS:
            for method, baseline in COMPARISONS:
                row = {"Classifier": classifier, "Training_table": pool,
                       "Comparison": f"{method} vs {baseline}"}
                for metric in METRICS:
                    result = compare(runs, classifier, pool, method, baseline, metric)
                    for key, value in (result or {}).items():
                        row[f"{metric}_{key}"] = value
                rows.append(row)
    table1 = pd.DataFrame(rows)
    table1.to_csv(folder / f"exploratory_top5_{args.level}.csv", index=False)
    print("TABLE 1 (EXPLORATORY): SFRT top-5 against unweighted and mutual information top-5")
    print(table1.to_string(index=False), "\n")

    stability = pd.read_csv(folder / f"rq3_consensus_stability_{args.level}.csv")
    stability = stability[stability.Method.isin(["mutual information", "SFRT consensus (mean importance)"])]
    table2 = (stability.groupby(["Method", "Classifier", "k"])["Mean_pairwise_Jaccard_over_seeds"]
              .mean().round(2).unstack("k").reset_index())
    table2.to_csv(folder / f"exploratory_selection_stability_{args.level}.csv", index=False)
    print("TABLE 2: overlap of the selected feature sets across seeds (1 = identical every time)")
    print(table2.to_string(index=False))
    print("\nBoth tables are exploratory. Report them as such in Chapter 4.")


if __name__ == "__main__":
    main()

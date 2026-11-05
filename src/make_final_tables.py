"""
Tables for the SFRT top-5 results in Chapter 4 (development and new projects).

Reads the saved runs only; trains no models. Run from the repository root:

    python src/make_final_tables.py

Writes four files to experiments/results/:
  final_sfrt_top5.csv            SFRT top-5 against the full metric set and against
                                 mutual information top-5, development and new projects
  final_small_table.csv          the same comparison against the full metric set on the
                                 small training tables (1,312 and 1,640 rows)
  final_selection_stability.csv  how often the same five metrics are chosen again
                                 when the seed changes (mean Jaccard)
  final_selected_metrics.csv     how often each metric is in the SFRT top-5 on the
                                 new projects
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
METRICS = ["MCC", "AUC-PR"]
CLASSIFIERS = ["Logistic Regression", "MLP", "Random Forest", "XGBoost"]


def load():
    dev = pd.read_csv(RES / "runs_strict.csv").rename(columns={"Target_Dataset": "Target"})
    new = pd.read_csv(RES / "holdout_runs.csv")
    small_dev = pd.read_csv(RES / "ablation_balance_strict.csv").rename(columns={"Target_Dataset": "Target"})
    small_dev = small_dev[small_dev.Pool == "size_matched"]
    return dev, new[new.Table == "full"], small_dev, new[new.Table == "small"]


def target_differences(runs, clf, metric, method, baseline):
    """Seed-averaged score per target, then method minus baseline."""
    g = runs[runs.Classifier == clf].groupby(["Method", "Target"])[metric].mean().unstack(0)
    return (g[method] - g[baseline]).dropna()


def wilcoxon_greater(d):
    """One-sided Wilcoxon over targets and the rank-biserial effect size."""
    p = stats.wilcoxon(d, alternative="greater").pvalue
    ranks = stats.rankdata(np.abs(d))
    rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    return round(p, 3), round(rb, 2)


def compare(dev, new, baseline):
    rows = []
    for clf in CLASSIFIERS:
        for metric in METRICS:
            d_dev = target_differences(dev, clf, metric, "sfrt_hard_k5", baseline)
            d_new = target_differences(new, clf, metric, "sfrt_hard_k5", baseline)
            p, rb = wilcoxon_greater(d_new.values)
            rows.append({"Classifier": clf, "Compared_with": baseline, "Metric": metric,
                         "Dev_change": round(d_dev.mean(), 3),
                         "Dev_targets_better": f"{(d_dev > 0).sum()}/{len(d_dev)}",
                         "New_change": round(d_new.mean(), 3),
                         "New_targets_better": f"{(d_new > 0).sum()}/{len(d_new)}",
                         "New_p_one_sided": p, "New_rank_biserial": rb})
    return pd.DataFrame(rows)


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b)


def seed_stability(runs, method, clf):
    """Mean Jaccard between the sets chosen with different seeds, same target."""
    g = runs[(runs.Method == method) & (runs.Classifier == clf)]
    values = []
    for _, gt in g.groupby("Target"):
        sets = gt.sort_values("Seed").Selected_Features.tolist()
        values += [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return round(float(np.mean(values)), 2)


def main():
    dev, new, small_dev, small_new = load()

    top5 = pd.concat([compare(dev, new, "unweighted"), compare(dev, new, "mutualinfo_k5")])
    top5 = top5[(top5.Compared_with == "unweighted") | top5.Classifier.isin(["Logistic Regression", "MLP"])]
    top5.to_csv(RES / "final_sfrt_top5.csv", index=False)

    small = compare(small_dev, small_new, "unweighted")
    small = small[small.Classifier.isin(["Logistic Regression", "MLP"])]
    small.to_csv(RES / "final_small_table.csv", index=False)

    stab = []
    for clf in CLASSIFIERS:
        stab.append({"Classifier": clf,
                     "Dev_SFRT": seed_stability(dev, "sfrt_hard_k5", clf),
                     "Dev_MI": seed_stability(dev, "mutualinfo_k5", clf),
                     "New_SFRT": seed_stability(new, "sfrt_hard_k5", clf),
                     "New_MI": seed_stability(new, "mutualinfo_k5", clf)})
    pd.DataFrame(stab).to_csv(RES / "final_selection_stability.csv", index=False)

    freq = []
    for clf in CLASSIFIERS:
        g = new[(new.Method == "sfrt_hard_k5") & (new.Classifier == clf)]
        counts = pd.Series(" ".join(g.Selected_Features).split()).value_counts()
        for metric, n in counts.items():
            freq.append({"Classifier": clf, "Metric": metric, "Share_of_runs": round(n / len(g), 2)})
    pd.DataFrame(freq).to_csv(RES / "final_selected_metrics.csv", index=False)

    for f in ["final_sfrt_top5.csv", "final_small_table.csv",
              "final_selection_stability.csv", "final_selected_metrics.csv"]:
        print(f"\n== {f}\n{pd.read_csv(RES / f).to_string(index=False)}")


if __name__ == "__main__":
    main()

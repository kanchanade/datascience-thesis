"""
RQ2 / H2 statistics, following review (Section 9.6).

PRE-SPECIFIED ANALYSIS PLAN (written before any SFRT result was inspected)
  Primary metrics : AUC-PR and MCC.  F1 and AUC-ROC are secondary.
  Primary SFRT    : the variant that is meaningful for each classifier:
                      Random Forest, XGBoost -> sfrt_hard_k10 (trees cannot be reweighted)
                      Logistic Regression, MLP -> sfrt_soft
  Comparators     : unweighted, and mutual information at the same k
                    (mutualinfo_k10) - the strongest baseline in the review.
                    (sfrt_soft keeps all 21 features, so its fair partner is
                    unweighted; it is also compared with mutualinfo_k10 for reference.)
  Test            : for each classifier and metric, take each of the 5 target
                    projects, average (SFRT - comparator) over the seeds, giving 5
                    differences; apply a ONE-SIDED Wilcoxon signed-rank test
                    (H: SFRT better). Smallest attainable p = 1/32 = 0.031.
                    Seeds measure algorithm noise, NOT generalisation, so they are
                    averaged away and never treated as independent observations.
  Effect size     : rank-biserial correlation of the 5 differences.
  Per-target      : mean difference over seeds with a 95% interval, to show whether
                    the change on a given project exceeds seed noise
                    (see summarise_results.py -> summary_per_target_ci.csv).
  Everything else (other k, ablations, LR at other C, XGBoost sampling, duplicates-only
  data) is EXPLORATORY and is reported as such, with no claim of significance.
Also reported: Friedman test across {unweighted, variance_k10, mutualinfo_k10, SFRT-primary}
per classifier and metric (blocks = the 5 targets, seed-averaged).

Run:  python src/analyse_rq2.py [--level strict]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import evaluation as ev

PRIMARY_SFRT = {"Random Forest": "sfrt_hard_k10", "XGBoost": "sfrt_hard_k10",
                "Logistic Regression": "sfrt_soft", "MLP": "sfrt_soft"}
COMPARATORS = ["unweighted", "mutualinfo_k10"]
METRICS = ["AUC-PR", "MCC"]
SECONDARY = ["F1", "AUC-ROC"]


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()
    folder = args.root / "experiments" / "results"
    runs = pd.read_csv(folder / f"runs_{args.level}.csv")
    # one number per (method, classifier, target): the mean over seeds
    cell = runs.groupby(["Method", "Classifier", "Target_Dataset"])[METRICS + SECONDARY].mean()
    n_seeds = runs.Seed.nunique()
    print(f"level = {args.level}, seeds = {n_seeds}\n")

    wilcoxon_rows, friedman_rows = [], []
    for clf in ev.CLASSIFIER_NAMES:
        sfrt = PRIMARY_SFRT[clf]
        for metric in METRICS + SECONDARY:
            tag = "PRIMARY" if metric in METRICS else "secondary"
            sfrt_scores = cell.loc[(sfrt, clf), metric].reindex(ev.DATASET_NAMES_UPPER)
            for comp in COMPARATORS:
                diff = sfrt_scores - cell.loc[(comp, clf), metric].reindex(ev.DATASET_NAMES_UPPER)
                p = 1.0 if (diff == 0).all() else stats.wilcoxon(diff, alternative="greater").pvalue
                wilcoxon_rows.append({
                    "Classifier": clf, "Metric": metric, "Role": tag, "SFRT": sfrt, "Comparator": comp,
                    "Targets_improved_of_5": int((diff > 0).sum()),
                    "Mean_difference": round(diff.mean(), 4),
                    "p_one_sided": round(p, 4), "Effect_rank_biserial": round(rank_biserial(diff), 2),
                    "Significant_0.05": bool(p < 0.05)})
            methods = ["unweighted", "variance_k10", "mutualinfo_k10", sfrt]
            table = pd.DataFrame({m: cell.loc[(m, clf), metric].reindex(ev.DATASET_NAMES_UPPER) for m in methods})
            f = stats.friedmanchisquare(*[table[m] for m in methods])
            ranks = table.rank(axis=1, ascending=False).mean()
            friedman_rows.append({"Classifier": clf, "Metric": metric, "Role": tag,
                                  "Chi_square": round(f.statistic, 3), "p": round(f.pvalue, 4),
                                  **{f"MeanRank_{m}": round(ranks[m], 2) for m in methods}})

    w = pd.DataFrame(wilcoxon_rows)
    w.to_csv(folder / f"rq2_wilcoxon_{args.level}.csv", index=False)
    pd.DataFrame(friedman_rows).to_csv(folder / f"rq2_friedman_{args.level}.csv", index=False)
    print("PRIMARY COMPARISONS (one-sided Wilcoxon over 5 targets, seed-averaged)")
    print(w[w.Role == "PRIMARY"].drop(columns=["Role"]).to_string(index=False))
    print("\nReminder: with 5 targets, p can only be significant if SFRT improves on ALL 5 (p = 0.031).")
    print("Full tables saved to rq2_wilcoxon_*.csv and rq2_friedman_*.csv.")


if __name__ == "__main__":
    main()

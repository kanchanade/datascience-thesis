"""
RQ3: how stable are SHAP feature-importance rankings?

    RQ3: How stable are SHAP feature importance rankings across
         (a) different models trained on the same project, and
         (b) the same model type across different projects?

What this script measures (no new model training - it reuses the per-source
SHAP vectors cached by run_experiments.py; anything missing is computed):

  NOISE FLOOR   same model, same project, different seed.
                How much do the SHAP rankings move just because of random
                choices (SMOTE, model start-up, SHAP sampling)? Nothing can
                be expected to agree MORE than this.
  RQ3(b)        same model type, different projects (same seed).
  RQ3(a)        different model types, same project (same seed).
  RANDOM        two random rankings, as the "no agreement at all" reference.

Each comparison uses:
  Spearman rank correlation of the two normalised importance vectors, and
  the overlap (Jaccard) of the top-5 and top-10 features.
Reading the numbers: if cross-project agreement is close to the noise floor,
the projects agree about as well as a project agrees with ITSELF - the ranking
differences are mostly noise. If it is far below the noise floor, the projects
genuinely disagree about which features matter, and a consensus across them
has less to agree on.

Extra analyses:
  * GROUPED: the 21 metrics are heavily correlated (e.g. loc, lOCode, n, v, b).
    Highly correlated metrics are grouped (|Spearman| >= 0.9, average linkage)
    and importance is summed inside each group, so SHAP spreading credit among
    near-copies is not counted as disagreement.
  * CONSENSUS STABILITY: how much the SFRT top-k set changes between seeds,
    compared with mutual information. Also a rank-based (Borda) consensus
    against the mean-importance consensus.
  * OUTLIER SOURCE: how well each source project agrees with the others
    (JM1 is 77-83% of the pooled data, so it matters whether it is the odd one out).

Ties: tree models give exactly zero importance to unused features. Top-k sets
break such ties by feature order, so top-k overlaps are slightly rough for
Random Forest / XGBoost; Spearman handles ties properly.

Run:  python src/analyse_rq3.py [--level strict] [--seeds 0-9]
Writes experiments/results/rq3_*_<level>.csv
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

import evaluation as ev
from run_experiments import parse_seeds


def normalise(vector):
    total = vector.sum()
    return vector / total if total > 0 else np.full(len(vector), 1 / len(vector))


def top_set(vector, k):
    return set(np.argsort(vector, kind="stable")[::-1][:k])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def rank_correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:       # constant vector: correlation undefined
        return np.nan
    return spearmanr(a, b).correlation


def agreement(a, b, ks):
    row = {"Spearman": rank_correlation(a, b)}
    for k in ks:
        row[f"Top{k}_Jaccard"] = jaccard(top_set(a, k), top_set(b, k))
    return row


def correlated_groups(data, threshold=0.9):
    """Group the 21 metrics: |Spearman| >= threshold (average linkage) on all rows."""
    pooled = pd.concat([data[n][0] for n in ev.DATASET_NAMES], ignore_index=True)
    corr = np.abs(pooled.corr(method="spearman").fillna(0).values)
    distance = 1 - corr
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=1 - threshold, criterion="distance")
    return labels


def random_reference(ks, n_features, draws=5000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        a, b = rng.random(n_features), rng.random(n_features)
        rows.append(agreement(a, b, ks))
    return pd.DataFrame(rows).mean().to_dict()


def mean_pairwise_jaccard(sets):
    pairs = list(itertools.combinations(sets, 2))
    return np.mean([jaccard(a, b) for a, b in pairs]) if pairs else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()

    ev.CACHE_DIR = args.root / "experiments" / "cache" / "shap"
    folder = args.root / "experiments" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    data = ev.load_all_data(args.level, args.root)
    classifiers = ev.CLASSIFIER_NAMES
    sources = ev.DATASET_NAMES
    n_features = len(ev.FEATURE_COLUMNS)
    print(f"level = {args.level}, seeds = {seeds}\n(missing SHAP vectors are computed now; "
          f"the MLP ones are slow)\n")

    # ---- load every normalised per-source importance vector ----
    imp = {}
    for clf in classifiers:
        for src in sources:
            X, y = data[src]
            for seed in seeds:
                imp[(clf, src, seed)] = normalise(ev.source_importance(src, X, y, clf, seed))

    # ---- feature table (for a heat-map in the thesis) ----
    rows = []
    for clf in classifiers:
        for src in sources:
            stack = np.vstack([imp[(clf, src, s)] for s in seeds])
            for j, name in enumerate(ev.FEATURE_COLUMNS):
                rows.append({"Classifier": clf, "Source": src.upper(), "Feature": name,
                             "Mean_importance": stack[:, j].mean(), "SD_over_seeds": stack[:, j].std(ddof=1) if len(seeds) > 1 else np.nan})
    pd.DataFrame(rows).round(4).to_csv(folder / f"rq3_feature_importance_{args.level}.csv", index=False)

    # ---- the pairwise comparisons, at feature level and at group level ----
    labels = correlated_groups(data)
    groups = {g: [ev.FEATURE_COLUMNS[i] for i in np.where(labels == g)[0]] for g in np.unique(labels)}
    print(f"CORRELATED METRIC GROUPS (|Spearman| >= 0.9): {len(groups)} groups")
    for g, members in groups.items():
        print(f"  group {g}: {' '.join(members)}")
    print()

    def to_groups(vector):
        return np.array([vector[labels == g].sum() for g in np.unique(labels)])

    records = []
    for level_name, transform, ks in [("features", lambda v: v, (5, 10)), ("groups", to_groups, (3,))]:
        for clf in classifiers:
            for src in sources:                                    # noise floor
                for s1, s2 in itertools.combinations(seeds, 2):
                    records.append({"Space": level_name, "Comparison": "NOISE FLOOR: same model, same project, different seed",
                                    "Classifier": clf, "Source": src.upper(),
                                    **agreement(transform(imp[(clf, src, s1)]), transform(imp[(clf, src, s2)]), ks)})
            for seed in seeds:                                     # RQ3(b)
                for a, b in itertools.combinations(sources, 2):
                    records.append({"Space": level_name, "Comparison": "RQ3(b): same model, different projects",
                                    "Classifier": clf, "Source": f"{a.upper()} vs {b.upper()}",
                                    **agreement(transform(imp[(clf, a, seed)]), transform(imp[(clf, b, seed)]), ks)})
        for src in sources:                                        # RQ3(a)
            for seed in seeds:
                for c1, c2 in itertools.combinations(classifiers, 2):
                    records.append({"Space": level_name, "Comparison": "RQ3(a): different models, same project",
                                    "Classifier": f"{c1} vs {c2}", "Source": src.upper(),
                                    **agreement(transform(imp[(c1, src, seed)]), transform(imp[(c2, src, seed)]), ks)})
    pairs = pd.DataFrame(records)
    pairs.to_csv(folder / f"rq3_pairwise_{args.level}.csv", index=False)

    metrics = [c for c in ["Spearman", "Top5_Jaccard", "Top10_Jaccard", "Top3_Jaccard"] if c in pairs]
    summary = (pairs.groupby(["Space", "Comparison", "Classifier"])[metrics].mean().round(2).reset_index())
    summary.to_csv(folder / f"rq3_summary_{args.level}.csv", index=False)

    reference = random_reference((5, 10), n_features)
    print("RANDOM-RANKING REFERENCE (features): Spearman %.2f, top-5 Jaccard %.2f, top-10 Jaccard %.2f\n"
          % (reference["Spearman"], reference["Top5_Jaccard"], reference["Top10_Jaccard"]))

    for space in ("features", "groups"):
        print(f"AGREEMENT OF SHAP RANKINGS ({space}); mean over pairs")
        print(summary[summary.Space == space].drop(columns="Space").dropna(axis=1, how="all").to_string(index=False), "\n")

    # ---- which source project is the odd one out? (RQ3(b), feature level) ----
    rows = []
    for clf in classifiers:
        for src in sources:
            rhos = [rank_correlation(imp[(clf, src, s)], imp[(clf, other, s)])
                    for s in seeds for other in sources if other != src]
            rows.append({"Classifier": clf, "Source": src.upper(), "Mean_Spearman_with_other_sources": round(np.nanmean(rhos), 2)})
    outlier = pd.DataFrame(rows)
    outlier.to_csv(folder / f"rq3_source_outlier_{args.level}.csv", index=False)
    print("AGREEMENT OF EACH SOURCE WITH THE OTHER FOUR (mean Spearman)")
    print(outlier.pivot(index="Source", columns="Classifier", values="Mean_Spearman_with_other_sources").to_string(), "\n")

    # ---- stability of the SFRT selection across seeds, vs mutual information ----
    rows = []
    for target in sources:
        source_names = [s for s in sources if s != target]
        X_pool, y_pool, _ = ev.pool_sources(data, target)
        mi_vectors = {s: mutual_info_classif(X_pool.values, y_pool.values, random_state=s) for s in seeds}
        for k in (5, 10):
            rows.append({"Method": "mutual information", "Classifier": "(any)", "Target": target.upper(), "k": k,
                         "Mean_pairwise_Jaccard_over_seeds": mean_pairwise_jaccard([top_set(mi_vectors[s], k) for s in seeds])})
            rows.append({"Method": "variance", "Classifier": "(any)", "Target": target.upper(), "k": k,
                         "Mean_pairwise_Jaccard_over_seeds": 1.0})   # deterministic: no seed enters
        for clf in classifiers:
            mean_consensus, borda_consensus = {}, {}
            for s in seeds:
                stack = np.vstack([imp[(clf, src, s)] for src in source_names])
                mean_consensus[s] = stack.mean(axis=0)
                ranks = np.vstack([np.argsort(np.argsort(v, kind="stable"), kind="stable") for v in stack])
                borda_consensus[s] = ranks.mean(axis=0)
            for k in (5, 10):
                rows.append({"Method": "SFRT consensus (mean importance)", "Classifier": clf, "Target": target.upper(), "k": k,
                             "Mean_pairwise_Jaccard_over_seeds": mean_pairwise_jaccard([top_set(mean_consensus[s], k) for s in seeds])})
                rows.append({"Method": "SFRT consensus (rank / Borda)", "Classifier": clf, "Target": target.upper(), "k": k,
                             "Mean_pairwise_Jaccard_over_seeds": mean_pairwise_jaccard([top_set(borda_consensus[s], k) for s in seeds])})
                rows.append({"Method": "mean-vs-Borda agreement (same seed)", "Classifier": clf, "Target": target.upper(), "k": k,
                             "Mean_pairwise_Jaccard_over_seeds": np.mean([jaccard(top_set(mean_consensus[s], k), top_set(borda_consensus[s], k)) for s in seeds])})
    stability = pd.DataFrame(rows)
    stability.to_csv(folder / f"rq3_consensus_stability_{args.level}.csv", index=False)
    table = (stability.groupby(["Method", "Classifier", "k"])["Mean_pairwise_Jaccard_over_seeds"].mean().round(2).unstack("k"))
    print("SELECTION STABILITY ACROSS SEEDS (mean over the 5 targets; 1.00 = same set every seed)")
    print(table.to_string(), "\n")
    print(f"Saved rq3_*_{args.level}.csv to {folder}")


if __name__ == "__main__":
    main()

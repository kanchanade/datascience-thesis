"""
Shared evaluation module: one leave-one-project-out protocol used by every method.

For one target project, classifier, feature step and seed:
  1. Pool the 4 other projects as source data. The target is used only for scoring.
  2. The feature step sees the source data only and returns which features to keep
     and, optionally, per-feature weights.
  3. Scale (Logistic Regression and MLP only), apply any weights, then SMOTE.
  4. Train, predict on the target, and score F1, MCC, AUC-ROC and AUC-PR.

The seed controls every random step (SMOTE, classifiers, mutual information, SHAP).
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAMES = ["cm1", "jm1", "kc1", "kc2", "pc1"]
DATASET_NAMES_UPPER = [n.upper() for n in DATASET_NAMES]

# The 21 features. Always selected BY NAME, never by position, because the
# PC1 file stores two columns in a different order from the other four.
FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

CLASSIFIER_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "MLP"]
NEEDS_SCALING = {"Logistic Regression", "MLP"}

# Extra Logistic Regression settings (SFRT's effect on Logistic Regression
# depends on the regularisation strength C). Not used for the reported results.
LR_C_VARIANTS = ["Logistic Regression C=0.1", "Logistic Regression C=10"]


def family(classifier_name):
    """'Logistic Regression C=0.1' -> 'Logistic Regression'."""
    return classifier_name.split(" C=")[0]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_all_data(level="strict", root=PROJECT_ROOT):
    """Load the 5 datasets. level = 'strict' (data/final, all five cleaning
    rules applied) or 'duplicates' (data/processed, exact duplicates only)."""
    data = {}
    for name in DATASET_NAMES:
        if level == "strict":
            path = Path(root) / "data" / "final" / f"{name}_strict.csv"
        elif level == "duplicates":
            path = Path(root) / "data" / "processed" / f"{name}_cleaned.csv"
        else:
            raise ValueError("level must be 'strict' or 'duplicates'")
        df = pd.read_csv(path)
        data[name] = (df[FEATURE_COLUMNS], df["defects"].astype(int))
    return data


def _stratified_sample(X, y, n_rows, rng):
    """Random sample of n_rows rows that keeps the defect rate of (X, y)."""
    n_rows = min(n_rows, len(y))
    positives = np.flatnonzero(y.values == 1)
    negatives = np.flatnonzero(y.values == 0)
    n_pos = int(round(n_rows * len(positives) / len(y)))
    n_pos = min(max(n_pos, 1), len(positives))
    n_neg = min(n_rows - n_pos, len(negatives))
    rows = np.concatenate([rng.choice(positives, n_pos, replace=False),
                           rng.choice(negatives, n_neg, replace=False)])
    rows.sort()
    return X.iloc[rows].reset_index(drop=True), y.iloc[rows].reset_index(drop=True)


def smallest_source_size(data):
    """Rows in the smallest of the five projects (the default 'rows per source'
    for the source-balance ablation)."""
    return min(len(data[n][1]) for n in DATASET_NAMES)


def pool_sources(data, target_name, mode="full", seed=0, rows_per_source=None):
    """Stack the 4 non-target projects into one training table.

    mode = "full"          all rows of the four projects (the main experiments).
    mode = "equal"         source-balance ablation: every source contributes the
                           SAME number of rows (rows_per_source, default = size of
                           the smallest project), sampled at random but keeping
                           each source's own defect rate.
    mode = "size_matched"  control for the ablation: a random sample of the pooled
                           table with the SAME total size as "equal" (4 x
                           rows_per_source) but the natural, JM1-dominated mix.
    Comparing "equal" with "size_matched" separates the effect of the source mix
    from the effect of simply training on fewer rows."""
    source_names = [n for n in DATASET_NAMES if n != target_name]
    if mode == "full":
        X = pd.concat([data[n][0] for n in source_names], ignore_index=True)
        y = pd.concat([data[n][1] for n in source_names], ignore_index=True)
        return X, y, source_names
    n_each = rows_per_source or smallest_source_size(data)
    if mode == "equal":
        parts = []
        for name in source_names:
            rng = np.random.default_rng([seed, DATASET_NAMES.index(name), 1])
            parts.append(_stratified_sample(*data[name], n_each, rng))
        return (pd.concat([p[0] for p in parts], ignore_index=True),
                pd.concat([p[1] for p in parts], ignore_index=True), source_names)
    if mode == "size_matched":
        X = pd.concat([data[n][0] for n in source_names], ignore_index=True)
        y = pd.concat([data[n][1] for n in source_names], ignore_index=True)
        rng = np.random.default_rng([seed, DATASET_NAMES.index(target_name), 2])
        X, y = _stratified_sample(X, y, n_each * len(source_names), rng)
        return X, y, source_names
    raise ValueError("mode must be 'full', 'equal' or 'size_matched'")


# ----------------------------------------------------------------------
# Classifiers (same settings as the original scripts, but the seed is an argument)
# ----------------------------------------------------------------------
def make_classifier(name, seed, **overrides):
    """Build a classifier. `overrides` lets a method change a setting
    (used only by the XGBoost feature-sampling variant of SFRT)."""
    base = family(name)
    if base == "Logistic Regression":
        c_value = float(name.split(" C=")[1]) if " C=" in name else 1.0
        return LogisticRegression(max_iter=1000, C=c_value, random_state=seed)
    if base == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    if base == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=seed, **overrides)
    if base == "MLP":
        # more iterations + early stopping fix the ConvergenceWarning seen
        # with the defaults (Section 3.4.3). Note: early stopping holds out
        # 10% of the SMOTE-balanced data, which contains synthetic rows, so
        # the stopping signal is slightly optimistic (a stated limitation).
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                             early_stopping=True, random_state=seed)
    raise ValueError(name)


# ----------------------------------------------------------------------
# Feature steps. Each one is a function:
#     step(X_source, y_source, source_parts, classifier_name, seed)
#         -> (kept_indices or None, weights or None, info dict)
# It sees SOURCE data only. `source_parts` is the dict of the 4 individual
# source projects (needed later by SFRT; the baselines ignore it).
# ----------------------------------------------------------------------
def unweighted_step():
    def step(X_source, y_source, source_parts, classifier_name, seed):
        return None, None, {}
    return step


def variance_step(k_keep):
    """Keep the k features with the HIGHEST variance.
    The features are very right-skewed,
    so on min-max scaled RAW values a few huge modules shrink the variance of
    every other module toward zero. We take log(1 + x) first, then min-max
    scale, then compute the variance.
    Honest note: this removes the outlier effect, but loc can still rank low
    (its log-scale spread is smaller than the Halstead metrics, which span
    several orders of magnitude). Variance ignores the label and depends on
    scale, so it stays a weak baseline."""
    def step(X_source, y_source, source_parts, classifier_name, seed):
        logged = np.log1p(np.clip(X_source.values, 0, None))
        scaled = MinMaxScaler().fit_transform(logged)
        variances = scaled.var(axis=0)
        kept = np.sort(np.argsort(variances)[::-1][:k_keep])
        return kept, None, {}
    return step


def mutual_info_step(k_keep):
    """Keep the k features with the highest mutual information with the label.
    The estimator is randomised (nearest-neighbour based), so it uses the seed."""
    def step(X_source, y_source, source_parts, classifier_name, seed):
        scores = mutual_info_classif(X_source.values, y_source.values, random_state=seed)
        kept = np.sort(np.argsort(scores)[::-1][:k_keep])
        return kept, None, {}
    return step


# ----------------------------------------------------------------------
# SFRT: SHAP-Consensus Feature Reweighting for Transfer
#
# Idea in plain words: train one model on EACH source project, ask SHAP which
# features that model relied on, then keep the features the sources AGREE on.
#   1. per source project: train the classifier, get mean |SHAP| per feature
#   2. normalise each source's 21 numbers to add up to 1 (so a source with a
#      big SHAP scale cannot dominate - the plan averages NORMALISED vectors)
#   3. consensus c_j = mean over sources, optionally penalised for
#      disagreement:  c_j = mu_j / (1 + lambda * sd_j / mu_j)
#      (lambda = 0 is the plain mean from the plan; lambda = 1 is the
#      "stability-penalised" variant)
#   4. hard variant: keep the top-k features by consensus
#      soft variant: multiply features by weights w_j ~ c_j^gamma (mean 1)
# Only SOURCE projects are used - the target is never touched.
# ----------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 60        # rows explained per source model
KERNEL_BACKGROUND_SIZE = 10  # k-means centres summarising the MLP background
CACHE_DIR = PROJECT_ROOT / "experiments" / "cache" / "shap"


def _mean_abs_shap(model, base_name, X_background, X_explain):
    """Mean |SHAP| per feature. Fast exact explainers for trees and linear
    models; KernelExplainer (approximate, slow) for the MLP."""
    if base_name in ("Random Forest", "XGBoost"):
        values = shap.TreeExplainer(model).shap_values(X_explain)
    elif base_name == "Logistic Regression":
        values = shap.LinearExplainer(model, X_background).shap_values(X_explain)
    else:
        background = shap.kmeans(X_background, KERNEL_BACKGROUND_SIZE)
        predict_positive = lambda X: model.predict_proba(X)[:, 1]
        values = shap.KernelExplainer(predict_positive, background).shap_values(X_explain, silent=True)
    values = np.array(values)
    if values.ndim == 3:                       # (rows, features, classes) -> positive class
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return np.abs(values).mean(axis=0)


def source_importance(source_name, X, y, classifier_name, seed):
    """Raw mean |SHAP| vector of ONE source project for one classifier and seed.
    It does not depend on which project is the target, so it is computed once
    and cached on disk (experiments/cache/shap) and reused for every target."""
    digest = hashlib.md5(X.values.tobytes() + y.values.tobytes()).hexdigest()[:8]
    cache_file = Path(CACHE_DIR) / f"{source_name}_{classifier_name.replace(' ', '')}_s{seed}_{digest}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    np.random.seed(seed)  # SHAP's internal sampling uses numpy's global RNG
    Xs = X.values
    if family(classifier_name) in NEEDS_SCALING:
        Xs = StandardScaler().fit_transform(Xs)
    Xb, yb = SMOTE(random_state=seed).fit_resample(Xs, y)
    model = make_classifier(classifier_name, seed)
    model.fit(Xb, yb)
    rng = np.random.default_rng([seed, DATASET_NAMES.index(source_name)])
    rows = rng.choice(len(Xb), size=min(SHAP_SAMPLE_SIZE, len(Xb)), replace=False)
    importance = _mean_abs_shap(model, family(classifier_name), Xb, Xb[rows])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(f".{os.getpid()}.tmp.npy")   # write-then-rename so parallel
  
    np.save(temporary, importance)                                   # workers never read a half-written file
    try:
        os.replace(temporary, cache_file)
    except PermissionError:
        # Windows refuses the rename when another worker is saving the same
        # file at the same moment. Both hold the same vector, so the other
        # copy is as good as ours.
        temporary.unlink(missing_ok=True)
        if not cache_file.exists():
            raise
    return importance

def consensus_scores(source_parts, classifier_name, seed, lam=0.0):
    """Normalise each source's importance vector, then combine them. Returns
    (consensus, per_source dict of normalised vectors)."""
    per_source = {}
    for name, (X, y) in source_parts.items():
        raw = source_importance(name, X, y, classifier_name, seed)
        total = raw.sum()
        per_source[name] = raw / total if total > 0 else np.full(len(raw), 1 / len(raw))
    stacked = np.vstack(list(per_source.values()))
    mu, sd = stacked.mean(axis=0), stacked.std(axis=0)
    consensus = mu / (1 + lam * sd / np.maximum(mu, 1e-12))
    return consensus, per_source


def sfrt_hard_step(k_keep, lam=0.0):
    """SFRT selection variant: keep the k features with the highest consensus."""
    def step(X_source, y_source, source_parts, classifier_name, seed):
        consensus, _ = consensus_scores(source_parts, classifier_name, seed, lam)
        return np.sort(np.argsort(consensus)[::-1][:k_keep]), None, {}
    return step


def sfrt_soft_step(lam=0.0, gamma=1.0):
    """SFRT reweighting variant: scale each (standardised) feature by its weight.
    Only meaningful for Logistic Regression and the MLP - trees split on
    thresholds, so multiplying a feature by a constant changes nothing."""
    def step(X_source, y_source, source_parts, classifier_name, seed):
        consensus, _ = consensus_scores(source_parts, classifier_name, seed, lam)
        weights = consensus ** gamma
        return None, weights / weights.mean(), {}
    return step


def xgb_sampling_step(use_consensus=True):
    """A reweighting mechanism that DOES work for trees: XGBoost offers each split a random half of the features
    (colsample_bynode = 0.5), drawn in proportion to feature_weights. With
    use_consensus=False the weights are uniform - the control that separates
    the effect of the consensus from the effect of feature sub-sampling."""
    def step(X_source, y_source, source_parts, classifier_name, seed):
        if use_consensus:
            weights, _ = consensus_scores(source_parts, classifier_name, seed)
        else:
            weights = np.ones(len(FEATURE_COLUMNS))
        return None, None, {"_model_params": {"colsample_bynode": 0.5,
                                              "feature_weights": weights}}
    return step


# ----------------------------------------------------------------------
# The one protocol
# ----------------------------------------------------------------------
def evaluate(data, target_name, classifier_name, feature_step, seed,
             pool_mode="full", rows_per_source=None):
    """Run one leave-one-project-out evaluation. Returns a dict of scores.
    pool_mode / rows_per_source only change the TRAINING table (see pool_sources);
    the default ("full") is the protocol used for every main result. The SFRT
    consensus is always built from the full source projects, so a given SFRT
    feature choice does not depend on the pool mode."""
    np.random.seed(seed)  # also seeds libraries that use numpy's global RNG (SHAP)

    X_target, y_target = data[target_name]
    X_source, y_source, source_names = pool_sources(data, target_name, pool_mode, seed, rows_per_source)
    source_parts = {n: data[n] for n in source_names}

    kept, weights, info = feature_step(X_source, y_source, source_parts, classifier_name, seed)

    Xs = X_source.values
    Xt = X_target.values
    if kept is not None:
        Xs, Xt = Xs[:, kept], Xt[:, kept]

    if family(classifier_name) in NEEDS_SCALING:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.transform(Xt)

    if weights is not None:
        Xs, Xt = Xs * weights, Xt * weights

    Xs_balanced, ys_balanced = SMOTE(random_state=seed).fit_resample(Xs, y_source)

    model_params = info.pop("_model_params", {})   # set only by the XGBoost feature-sampling variant
    model = make_classifier(classifier_name, seed, **model_params)
    model.fit(Xs_balanced, ys_balanced)
    predicted = model.predict(Xt)
    probability = model.predict_proba(Xt)[:, 1]

    kept_names = FEATURE_COLUMNS if kept is None else [FEATURE_COLUMNS[i] for i in kept]
    return {
        "F1": f1_score(y_target, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_target, predicted),
        "AUC-ROC": roc_auc_score(y_target, probability),
        "AUC-PR": average_precision_score(y_target, probability),
        "Features_Kept": len(kept_names),
        "Selected_Features": " ".join(kept_names),
        **info,
    }


# Registry: method name -> feature step. Each step may carry a `.classifiers`
# attribute listing the classifier families it applies to (default: all four).
def _only(step, *families):
    step.classifiers = set(families)
    return step


def build_methods():
    methods = {"unweighted": unweighted_step()}
    for k in (5, 10, 15):
        methods[f"variance_k{k}"] = variance_step(k)
        methods[f"mutualinfo_k{k}"] = mutual_info_step(k)
    for k in (5, 10, 15):
        methods[f"sfrt_hard_k{k}"] = sfrt_hard_step(k)                    # plain consensus
        methods[f"sfrt_hardstab_k{k}"] = sfrt_hard_step(k, lam=1.0)       # stability-penalised
    linear_models = ("Logistic Regression", "MLP")
    methods["sfrt_soft"] = _only(sfrt_soft_step(), *linear_models)
    methods["sfrt_softstab"] = _only(sfrt_soft_step(lam=1.0), *linear_models)
    methods["sfrt_softsharp"] = _only(sfrt_soft_step(gamma=2.0), *linear_models)
    methods["xgb_sampling_uniform"] = _only(xgb_sampling_step(False), "XGBoost")   # control
    methods["xgb_sampling_sfrt"] = _only(xgb_sampling_step(True), "XGBoost")
    return methods


def applies(step, classifier_name):
    allowed = getattr(step, "classifiers", None)
    return allowed is None or family(classifier_name) in allowed

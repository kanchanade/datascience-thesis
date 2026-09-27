"""
Step 1 — Proof-of-concept pipeline: one classifier, one dataset, within-project.

Goal of this script: confirm the full chain (load -> split -> scale -> train ->
predict -> evaluate) works correctly end-to-end, before extending to all 4
classifiers and all 5 datasets (Step 2) and then cross-project evaluation
(Step 3 onward). This is deliberately minimal — it is a checkpoint, not the
final within-project baseline result.

Method: Logistic Regression, stratified 10-fold cross-validation, on CM1 only
(Proposal Section 3.3.5: within-project results use stratified ten-fold CV).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

RANDOM_SEED = 42  # fixed for reproducibility, per Section 3.4.4


def load_dataset(name: str, processed_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one cleaned dataset and split into features (X) and target (y).
    Assumes the schema produced by the project's own cleaning notebook: original
    PROMISE-style column names (e.g. 'v(g)', 'lOCode', 'branchCount'), with a
    boolean or 0/1 'defects' target column."""
    df = pd.read_csv(processed_dir / f"{name}_cleaned.csv")
    y = df["defects"].astype(int)
    X = df.drop(columns=["defects"])
    return X, y


def evaluate_logistic_regression_within_project(X: pd.DataFrame, y: pd.Series,
                                                   n_folds: int = 10,
                                                   seed: int = RANDOM_SEED) -> dict:
    """Stratified k-fold within-project evaluation of a single classifier.
    Logistic Regression is scale-sensitive (Proposal Section 3.2.4), so scaling
    is fit on each training fold only and applied to that fold's test split —
    never fit on data the model will be evaluated against."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_f1, fold_mcc, fold_auc = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
        fold_mcc.append(matthews_corrcoef(y_test, y_pred))
        fold_auc.append(roc_auc_score(y_test, y_proba))

        print(f"  Fold {fold_idx:2d}: F1={fold_f1[-1]:.3f}  MCC={fold_mcc[-1]:.3f}  AUC-ROC={fold_auc[-1]:.3f}")

    return {
        "f1_mean": np.mean(fold_f1), "f1_std": np.std(fold_f1),
        "mcc_mean": np.mean(fold_mcc), "mcc_std": np.std(fold_mcc),
        "auc_mean": np.mean(fold_auc), "auc_std": np.std(fold_auc),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    print("=" * 70)
    print("STEP 1: Logistic Regression on CM1 (within-project, 10-fold CV)")
    print("=" * 70)

    X, y = load_dataset("cm1", processed_dir)
    print(f"\nLoaded CM1: {X.shape[0]} modules, {X.shape[1]} features, "
          f"{y.mean()*100:.2f}% defect rate\n")

    results = evaluate_logistic_regression_within_project(X, y)

    print("\n" + "-" * 70)
    print("SUMMARY (mean ± std across 10 folds)")
    print("-" * 70)
    print(f"F1-score:  {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    print(f"MCC:       {results['mcc_mean']:.3f} ± {results['mcc_std']:.3f}")
    print(f"AUC-ROC:   {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")

"""
Runs Random Forest on all 5 datasets, within-project, using 10-fold
cross-validation and SMOTE to fix class imbalance.

Random Forest does NOT need feature scaling (it makes yes/no splits on each
feature, so the size of the numbers doesn't matter to it), so this script is
slightly simpler than the Logistic Regression one — no scaling step.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42

data_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\data\processed")
dataset_names = ["cm1", "jm1", "kc1", "kc2", "pc1"]

all_results = []

print("=" * 70)
print("RANDOM FOREST — all 5 datasets, within-project, with SMOTE")
print("=" * 70)

for dataset_name in dataset_names:

    df = pd.read_csv(data_folder / f"{dataset_name}_cleaned.csv")
    y = df["defects"].astype(int)
    X = df.drop(columns=["defects"])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    fold_f1_scores = []
    fold_mcc_scores = []
    fold_auc_scores = []
    fold_auc_pr_scores = []

    for train_rows, test_rows in skf.split(X, y):

        X_train = X.iloc[train_rows]
        X_test = X.iloc[test_rows]
        y_train = y.iloc[train_rows]
        y_test = y.iloc[test_rows]

        # No scaling needed for Random Forest — go straight to SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
        model.fit(X_train_balanced, y_train_balanced)

        y_predicted = model.predict(X_test)
        y_predicted_probability = model.predict_proba(X_test)[:, 1]

        fold_f1_scores.append(f1_score(y_test, y_predicted, zero_division=0))
        fold_mcc_scores.append(matthews_corrcoef(y_test, y_predicted))
        fold_auc_scores.append(roc_auc_score(y_test, y_predicted_probability))
        fold_auc_pr_scores.append(average_precision_score(y_test, y_predicted_probability))

    average_f1 = np.mean(fold_f1_scores)
    average_mcc = np.mean(fold_mcc_scores)
    average_auc = np.mean(fold_auc_scores)
    average_auc_pr = np.mean(fold_auc_pr_scores)

    print(f"{dataset_name.upper()}:  F1={average_f1:.3f}   MCC={average_mcc:.3f}   "
          f"AUC-ROC={average_auc:.3f}   AUC-PR={average_auc_pr:.3f}")

    all_results.append({
        "Dataset": dataset_name.upper(),
        "Classifier": "Random Forest",
        "F1": round(average_f1, 3),
        "MCC": round(average_mcc, 3),
        "AUC-ROC": round(average_auc, 3),
        "AUC-PR": round(average_auc_pr, 3),
    })

results_table = pd.DataFrame(all_results)
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
results_folder.mkdir(parents=True, exist_ok=True)
results_table.to_csv(results_folder / "results_random_forest.csv", index=False)

print("\nSaved to experiments\\results\\results_random_forest.csv")

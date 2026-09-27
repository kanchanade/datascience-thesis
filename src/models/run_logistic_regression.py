"""
Runs Logistic Regression on all 5 datasets, within-project, using 10-fold
cross-validation and SMOTE to fix class imbalance.

This is the SAME method as Step 1, just repeated across all 5 datasets instead
of just CM1, and with SMOTE added to fix the imbalance problem Step 1 revealed.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42

# Where our cleaned data lives
data_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\data\processed")
dataset_names = ["cm1", "jm1", "kc1", "kc2", "pc1"]

# This list will collect one row of results per dataset
all_results = []

print("=" * 70)
print("LOGISTIC REGRESSION — all 5 datasets, within-project, with SMOTE")
print("=" * 70)

for dataset_name in dataset_names:

    # Load this dataset
    df = pd.read_csv(data_folder / f"{dataset_name}_cleaned.csv")
    y = df["defects"].astype(int)
    X = df.drop(columns=["defects"])

    # Split into 10 folds, keeping the defect ratio similar in each fold
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

        # Logistic Regression needs scaled features (learned from training data only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Balance the training data only, using SMOTE (never touch test data with SMOTE)
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Train and predict
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train_balanced, y_train_balanced)

        y_predicted = model.predict(X_test)
        y_predicted_probability = model.predict_proba(X_test)[:, 1]

        fold_f1_scores.append(f1_score(y_test, y_predicted, zero_division=0))
        fold_mcc_scores.append(matthews_corrcoef(y_test, y_predicted))
        fold_auc_scores.append(roc_auc_score(y_test, y_predicted_probability))
        # AUC-PR (Area Under the Precision-Recall curve) focuses specifically on
        # how well the model finds the rare "defective" class, unlike AUC-ROC
        # which can look artificially good on imbalanced data (Proposal 3.3.6.3)
        fold_auc_pr_scores.append(average_precision_score(y_test, y_predicted_probability))

    average_f1 = np.mean(fold_f1_scores)
    average_mcc = np.mean(fold_mcc_scores)
    average_auc = np.mean(fold_auc_scores)
    average_auc_pr = np.mean(fold_auc_pr_scores)

    print(f"{dataset_name.upper()}:  F1={average_f1:.3f}   MCC={average_mcc:.3f}   "
          f"AUC-ROC={average_auc:.3f}   AUC-PR={average_auc_pr:.3f}")

    all_results.append({
        "Dataset": dataset_name.upper(),
        "Classifier": "Logistic Regression",
        "F1": round(average_f1, 3),
        "MCC": round(average_mcc, 3),
        "AUC-ROC": round(average_auc, 3),
        "AUC-PR": round(average_auc_pr, 3),
    })

# Save results to a CSV file
results_table = pd.DataFrame(all_results)
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
results_folder.mkdir(parents=True, exist_ok=True)
results_table.to_csv(results_folder / "results_logistic_regression.csv", index=False)

print("\nSaved to experiments\\results\\results_logistic_regression.csv")

"""
Cross-project version of the Neural Network (MLP): leave-one-project-out.
See crossproject_logistic_regression.py for a full explanation of what
"cross-project" and "leave-one-project-out" mean.

Same convergence fix as run_mlp.py: more iterations, and early stopping.
"""

import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42

data_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\data\processed")
dataset_names = ["cm1", "jm1", "kc1", "kc2", "pc1"]

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

all_data = {}
for name in dataset_names:
    df = pd.read_csv(data_folder / f"{name}_cleaned.csv")
    X = df[FEATURE_COLUMNS]
    y = df["defects"].astype(int)
    all_data[name] = (X, y)

all_results = []

print("=" * 70)
print("MLP (NEURAL NETWORK) — cross-project (leave-one-project-out)")
print("=" * 70)

for target_name in dataset_names:

    X_target, y_target = all_data[target_name]

    source_names = [name for name in dataset_names if name != target_name]
    X_source = pd.concat([all_data[name][0] for name in source_names], ignore_index=True)
    y_source = pd.concat([all_data[name][1] for name in source_names], ignore_index=True)

    # MLP needs scaled features, learned from the source data only
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)

    smote = SMOTE(random_state=RANDOM_SEED)
    X_source_balanced, y_source_balanced = smote.fit_resample(X_source_scaled, y_source)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )
    model.fit(X_source_balanced, y_source_balanced)

    y_predicted = model.predict(X_target_scaled)
    y_predicted_probability = model.predict_proba(X_target_scaled)[:, 1]

    f1 = f1_score(y_target, y_predicted, zero_division=0)
    mcc = matthews_corrcoef(y_target, y_predicted)
    auc = roc_auc_score(y_target, y_predicted_probability)
    auc_pr = average_precision_score(y_target, y_predicted_probability)

    print(f"Target={target_name.upper():5s} (trained on the other 4)  "
          f"F1={f1:.3f}   MCC={mcc:.3f}   AUC-ROC={auc:.3f}   AUC-PR={auc_pr:.3f}")

    all_results.append({
        "Target_Dataset": target_name.upper(),
        "Classifier": "MLP (Neural Network)",
        "F1": round(f1, 3),
        "MCC": round(mcc, 3),
        "AUC-ROC": round(auc, 3),
        "AUC-PR": round(auc_pr, 3),
    })

results_table = pd.DataFrame(all_results)
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
results_folder.mkdir(parents=True, exist_ok=True)
results_table.to_csv(results_folder / "crossproject_mlp.csv", index=False)

print("\nSaved to experiments\\results\\crossproject_mlp.csv")

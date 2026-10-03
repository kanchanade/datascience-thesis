# imports
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42 # my rands
# @TO DO change the paths to a project folder rather than absolute path here
data_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\data\processed")
dataset_names = ["cm1", "jm1", "kc1", "kc2", "pc1"]

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

N_FEATURES_TO_KEEP = 10

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                           early_stopping=True, random_state=RANDOM_SEED),
}
needs_scaling = ["Logistic Regression", "MLP (Neural Network)"]

all_data = {}
for name in dataset_names:
    df = pd.read_csv(data_folder / f"{name}_cleaned.csv")
    X = df[FEATURE_COLUMNS]
    y = df["defects"].astype(int)
    all_data[name] = (X, y)

all_results = []

print("=" * 70)
print(f"MUTUAL INFORMATION FEATURE SELECTION (top {N_FEATURES_TO_KEEP}/21) — cross project baseline")
print("=" * 70)

for target_name in dataset_names:

    X_target, y_target = all_data[target_name]

    source_names = [name for name in dataset_names if name != target_name]
    X_source = pd.concat([all_data[name][0] for name in source_names], ignore_index=True)
    y_source = pd.concat([all_data[name][1] for name in source_names], ignore_index=True)

    print(f"\n--- Target = {target_name.upper()} (trained on the other 4) ---")


    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_SEED),
        k=N_FEATURES_TO_KEEP,
    )
    X_source_selected_array = selector.fit_transform(X_source, y_source)
    X_target_selected_array = selector.transform(X_target)

    selected_mask = selector.get_support()
    selected_feature_names = [f for f, keep in zip(FEATURE_COLUMNS, selected_mask) if keep]
    print(f"Selected features: {selected_feature_names}")

    for classifier_name, classifier in classifiers.items():

        X_source_selected = X_source_selected_array
        X_target_selected = X_target_selected_array

        if classifier_name in needs_scaling:
            scaler = StandardScaler()
            X_source_selected = scaler.fit_transform(X_source_selected)
            X_target_selected = scaler.transform(X_target_selected)

        smote = SMOTE(random_state=RANDOM_SEED)
        X_source_balanced, y_source_balanced = smote.fit_resample(X_source_selected, y_source)

        classifier.fit(X_source_balanced, y_source_balanced)
        y_predicted = classifier.predict(X_target_selected)
        y_predicted_probability = classifier.predict_proba(X_target_selected)[:, 1]

        f1 = f1_score(y_target, y_predicted, zero_division=0)
        mcc = matthews_corrcoef(y_target, y_predicted)
        auc = roc_auc_score(y_target, y_predicted_probability)
        auc_pr = average_precision_score(y_target, y_predicted_probability)

        print(f"{classifier_name:22s}  F1={f1:.3f}   MCC={mcc:.3f}   "
              f"AUC-ROC={auc:.3f}   AUC-PR={auc_pr:.3f}")

        all_results.append({
            "Target_Dataset": target_name.upper(),
            "Classifier": classifier_name,
            "Features_Kept": N_FEATURES_TO_KEEP,
            "Features_Total": len(FEATURE_COLUMNS),
            "F1": round(f1, 3),
            "MCC": round(mcc, 3),
            "AUC-ROC": round(auc, 3),
            "AUC-PR": round(auc_pr, 3),
        })

results_table = pd.DataFrame(all_results)
# @TO DO change the paths to a project folder rather than absolute path here
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
results_folder.mkdir(parents=True, exist_ok=True)
results_table.to_csv(results_folder / "baseline_mutual_info_selection.csv", index=False)

print("\nSaved to experiments\\results\\baseline_mutual_info_selection.csv")

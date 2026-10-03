# imports
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
# @TO DO change the paths to a project folder rather than absolute path here
data_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\data\processed")
dataset_names = ["cm1", "jm1", "kc1", "kc2", "pc1"]

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

N_FEATURES_TO_REMOVE = 5

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                           early_stopping=True, random_state=RANDOM_SEED),
}
needs_scaling = ["Logistic Regression", "MLP (Neural Network)"]

# Load all 5 datasets
all_data = {}
for name in dataset_names:
    df = pd.read_csv(data_folder / f"{name}_cleaned.csv")
    X = df[FEATURE_COLUMNS]
    y = df["defects"].astype(int)
    all_data[name] = (X, y)

all_results = []

print("=" * 70)
print("Variance based feature selection —>> cross-project baseline")
print("=" * 70)

for target_name in dataset_names:

    X_target, y_target = all_data[target_name]

    source_names = [name for name in dataset_names if name != target_name]
    X_source = pd.concat([all_data[name][0] for name in source_names], ignore_index=True)
    y_source = pd.concat([all_data[name][1] for name in source_names], ignore_index=True)

    print(f"\n--- Target = {target_name.upper()} (trained on the other 4) ---")

    for classifier_name, classifier in classifiers.items():

        # Step 1: scale to a 0-1 range first, so variance is comparable across
        minmax_scaler = MinMaxScaler()
        X_source_for_selection = minmax_scaler.fit_transform(X_source)
        X_target_for_selection = minmax_scaler.transform(X_target)

        # Step 2: work out each features variance using the SOURCE data only,
        feature_variances = X_source_for_selection.var(axis=0)
        lowest_variance_indices = np.argsort(feature_variances)[:N_FEATURES_TO_REMOVE]
        keep_mask = np.ones(len(FEATURE_COLUMNS), dtype=bool)
        keep_mask[lowest_variance_indices] = False

        X_source_selected = X_source_for_selection[:, keep_mask]
        X_target_selected = X_target_for_selection[:, keep_mask]

        n_kept = keep_mask.sum()
        n_total = len(FEATURE_COLUMNS)

        # Step 3: scale again for classifiers that need it
        if classifier_name in needs_scaling:
            scaler = StandardScaler()
            X_source_selected = scaler.fit_transform(X_source_selected)
            X_target_selected = scaler.transform(X_target_selected)

        # Step 4: balance the SOURCE training data only, using SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_source_balanced, y_source_balanced = smote.fit_resample(X_source_selected, y_source)

        # Step 5: train on source, test on the untouched target
        classifier.fit(X_source_balanced, y_source_balanced)
        y_predicted = classifier.predict(X_target_selected)
        y_predicted_probability = classifier.predict_proba(X_target_selected)[:, 1]

        f1 = f1_score(y_target, y_predicted, zero_division=0)
        mcc = matthews_corrcoef(y_target, y_predicted)
        auc = roc_auc_score(y_target, y_predicted_probability)
        auc_pr = average_precision_score(y_target, y_predicted_probability)

        print(f"{classifier_name:22s}  features kept={n_kept}/{n_total}  "
              f"F1={f1:.3f}   MCC={mcc:.3f}   AUC-ROC={auc:.3f}   AUC-PR={auc_pr:.3f}")

        all_results.append({
            "Target_Dataset": target_name.upper(),
            "Classifier": classifier_name,
            "Features_Kept": n_kept,
            "Features_Total": n_total,
            "F1": round(f1, 3),
            "MCC": round(mcc, 3),
            "AUC-ROC": round(auc, 3),
            "AUC-PR": round(auc_pr, 3),
        })

results_table = pd.DataFrame(all_results)
# @TO DO change the paths to a project folder rather than absolute path here
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
results_folder.mkdir(parents=True, exist_ok=True)
results_table.to_csv(results_folder / "baseline_variance_selection.csv", index=False)

print("\nSaved to experiments\\results\\baseline_variance_selection.csv")

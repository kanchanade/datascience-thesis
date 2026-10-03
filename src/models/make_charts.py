# imports
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set where is the results CSVs are and where to save the graph images
# @TO DO change the paths to a project folder rather than absolute path here
results_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\experiments\results")
graphs_folder = Path(r"C:\Users\kanch\Documents\datascience-thesis\docs\figures")
graphs_folder.mkdir(parents=True, exist_ok=True)

# The 4 result files made earlier one per classifier
result_files = {
    "Logistic Regression": "results_logistic_regression.csv",
    "Random Forest": "results_random_forest.csv",
    "XGBoost": "results_xgboost.csv",
    "MLP (Neural Network)": "results_mlp.csv",
}

# This will collect every classifiers results into a table, which I need for Part 2 (the combined chart)
all_results = []

for classifier_name, file_name in result_files.items():

    # Load this classifier results
    df = pd.read_csv(results_folder / file_name)
    all_results.append(df)

    # Gen: a bar chart with 3 bars per dataset (F1, MCC, AUC-ROC side by side)
    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = range(len(df))   # one position per dataset
    bar_width = 0.25

    ax.bar([x - bar_width for x in x_positions], df["F1"], width=bar_width, label="F1")
    ax.bar(x_positions, df["MCC"], width=bar_width, label="MCC")
    ax.bar([x + bar_width for x in x_positions], df["AUC-ROC"], width=bar_width, label="AUC-ROC")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(df["Dataset"])
    ax.set_ylabel("Score")
    ax.set_title(f"{classifier_name} — Performance by Dataset")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save the chart as an image file. Turn the classifier name into a simple filename
    safe_name = classifier_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_path = graphs_folder / f"chart_{safe_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

# Combine every classifiers table into a table
combined = pd.concat(all_results, ignore_index=True)

# Pivot turns into a grid: rows = datasets, columns = classifiers,
# values = F1 score. This makes it easy to plot side by side bars.
pivot_table = combined.pivot(index="Dataset", columns="Classifier", values="F1")

fig, ax = plt.subplots(figsize=(10, 6))
pivot_table.plot(kind="bar", ax=ax)

ax.set_ylabel("F1-score")
ax.set_title("F1-Score Comparison: All 4 Classifiers Across All 5 Datasets")
ax.legend(title="Classifier")
ax.set_ylim(0, 1)
plt.xticks(rotation=0)

plt.tight_layout()
combined_output_path = graphs_folder / "chart_all_classifiers_comparison.png"
plt.savefig(combined_output_path, dpi=150)
plt.close()

print(f"Saved: {combined_output_path}")
print("\nAll charts saved to docs\\figures\\")

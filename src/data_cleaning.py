
from pathlib import Path
import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaning_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. DATASET FILENAMES
# ==================================================

DATASETS = {
    "CM1": "cm1_cleaned.csv",
    "JM1": "jm1_cleaned.csv",
    "KC1": "kc1_cleaned.csv",
    "KC2": "kc2_cleaned.csv",
    "PC1": "pc1_cleaned.csv",
}


# ==================================================
# 3. COMMON 21 SOFTWARE METRICS
# ==================================================

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)",
    "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank",
    "locCodeAndComment",
    "uniq_Op", "uniq_Opnd",
    "total_Op", "total_Opnd",
    "branchCount",
]


# ==================================================
# STEP 1: REMOVE ARTIFICIAL ALL-ONES ROWS
# ==================================================

print("\n" + "=" * 60)
print("NASA PROMISE - DATA CLEANING")
print("STEP 1: REMOVE ARTIFICIAL ALL-ONES ROWS")
print("=" * 60)

summary = []

for dataset_name, filename in DATASETS.items():

    input_file = INPUT_DIR / filename

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    # Check required columns
    required_columns = FEATURE_COLUMNS + ["defects"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: Missing columns: {missing_columns}"
        )

    original_rows = len(df)

    # Identify rows where all 21 features equal 1
    all_ones_mask = (
        df[FEATURE_COLUMNS] == 1
    ).all(axis=1)

    removed_count = int(all_ones_mask.sum())

    # Remove all-ones rows
    df_cleaned = df.loc[~all_ones_mask].copy()

    # Verify removal
    remaining_count = int(
        (df_cleaned[FEATURE_COLUMNS] == 1)
        .all(axis=1)
        .sum()
    )

    assert remaining_count == 0, (
        f"{dataset_name}: All-ones rows still exist!"
    )

    # Save to separate folder
    output_file = OUTPUT_DIR / filename

    df_cleaned.to_csv(
        output_file,
        index=False
    )

    final_rows = len(df_cleaned)

    summary.append({
        "Dataset": dataset_name,
        "Before": original_rows,
        "Removed": removed_count,
        "After": final_rows,
    })

    print(
        f"{dataset_name}: "
        f"{original_rows} -> {final_rows} rows; "
        f"removed {removed_count}"
    )


# Save Step 1 audit
summary_df = pd.DataFrame(summary)

summary_file = OUTPUT_DIR / "all_ones_removal_audit.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

print("\nSTEP 1 SUMMARY")
print(summary_df.to_string(index=False))

print("\nStep 1 completed.")
print("Original processed datasets unchanged.")


# ==================================================
# STEP 2: IDENTIFY CONFLICTING DEFECT LABELS
# ==================================================

print("\n" + "=" * 60)
print("STEP 2: CONFLICTING DEFECT LABELS")
print("=" * 60)

conflict_summary = []

for dataset_name, filename in DATASETS.items():

    # Read the Step 1 cleaned dataset
    input_file = OUTPUT_DIR / filename

    df = pd.read_csv(input_file)

    # Group identical feature combinations
    # and count their distinct defect labels
    label_counts = df.groupby(
        FEATURE_COLUMNS,
        dropna=False
    )["defects"].transform("nunique")

    # Identify rows with contradictory labels
    conflict_mask = label_counts > 1

    conflicting_rows = df.loc[conflict_mask].copy()

    conflict_count = len(conflicting_rows)

    # Count distinct conflicting feature combinations
    conflict_groups = (
        conflicting_rows[FEATURE_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )

    conflict_summary.append({
        "Dataset": dataset_name,
        "Total Rows": len(df),
        "Conflicting Rows": conflict_count,
        "Conflicting Groups": conflict_groups,
    })

    print(
        f"{dataset_name}: "
        f"{conflict_count} conflicting rows, "
        f"{conflict_groups} conflicting groups"
    )


# Save Step 2 audit
conflict_summary_df = pd.DataFrame(conflict_summary)

conflict_summary_file = (
    OUTPUT_DIR / "conflicting_labels_audit.csv"
)

conflict_summary_df.to_csv(
    conflict_summary_file,
    index=False
)

print("\nSTEP 2 SUMMARY")
print(conflict_summary_df.to_string(index=False))

print("\nStep 2 audit completed.")
print("No conflicting-label rows were removed.")


# ==================================================
# FINAL MESSAGE
# ==================================================

print("\n" + "=" * 60)
print("DATA CLEANING AUDIT COMPLETED")
print("=" * 60)

print(f"Output folder: {OUTPUT_DIR}")
print("Original datasets remain unchanged.")
print("Step 1: All-ones rows removed from review copies.")
print("Step 2: Conflicting labels identified only.")

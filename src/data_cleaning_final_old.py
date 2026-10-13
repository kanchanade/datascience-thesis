"""Reproducible NASA PROMISE cleaning from original source CSV files.

Source: data/raw/promise_source/{cm1,jm1,kc1,kc2,pc1}.csv
Output: data/final_primary/*_cleaned.csv and cleaning_audit.csv

Raw, processed, and intermediate datasets are never overwritten.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "promise_source"
OUT_DIR = ROOT / "data" / "final_primary"
DATASETS = ("cm1", "jm1", "kc1", "kc2", "pc1")
FEATURES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment", "uniq_Op",
    "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]
COLUMNS = FEATURES + ["defects"]


def standardise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "kc2":
        df = df.rename(columns={
            "lOCodeAndComment": "locCodeAndComment",
            "problems": "defects",
        })
    elif name == "pc1":
        df = df.rename(columns={
            "iv(G)": "iv(g)", "N": "n", "V": "v", "L": "l", "D": "d",
            "I": "i", "E": "e", "B": "b", "T": "t",
        })
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name.upper()}: Missing columns: {sorted(missing)}")
    if len(df.columns) != len(COLUMNS):
        raise ValueError(f"{name.upper()}: Unexpected columns present")
    df = df[COLUMNS].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    labels = df["defects"].astype(str).str.strip().str.lower()
    label_map = {"true": True, "false": False, "yes": True, "no": False,
                 "1": True, "0": False}
    if not labels.isin(label_map).all():
        bad = sorted(labels[~labels.isin(label_map)].unique().tolist())
        raise ValueError(f"{name.upper()}: Unexpected defect labels: {bad}")
    df["defects"] = labels.map(label_map).astype(bool)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    print("NASA PROMISE - REPRODUCIBLE FINAL CLEANING")
    print("=" * 65)
    for name in DATASETS:
        source = RAW_DIR / f"{name}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing raw file: {source}")
        df = standardise(pd.read_csv(source), name)
        raw_count = len(df)

        # Drop exact duplicates (features AND defect label identical).
        duplicate_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

        # Previously verified synthetic fractional artefact, one per dataset.
        fractional_mask = (
            df["loc"].eq(1.1) & df["v(g)"].eq(1.4) & df["n"].eq(1.3)
        )
        fractional_count = int(fractional_mask.sum())
        if fractional_count != 1:
            raise ValueError(
                f"{name.upper()}: Expected 1 fractional artefact, found {fractional_count}"
            )
        df = df.loc[~fractional_mask].copy()

        # Synthetic all-ones record: all 21 features equal 1.
        all_ones_mask = df[FEATURES].eq(1).all(axis=1)
        all_ones_count = int(all_ones_mask.sum())
        if all_ones_count != 1:
            raise ValueError(
                f"{name.upper()}: Expected 1 all-ones record, found {all_ones_count}"
            )
        df = df.loc[~all_ones_mask].copy()

        # Exclude every row in a contradictory-label feature group.
        label_variants = df.groupby(FEATURES, dropna=False)["defects"].transform("nunique")
        conflict_mask = label_variants > 1
        conflict_count = int(conflict_mask.sum())
        df = df.loc[~conflict_mask].copy()

        # Audit-only: preserve these records for later sensitivity analysis.
        zero_halstead = int((df["n"].eq(0) & df["loc"].gt(0)).sum())
        op_mismatch = int((df["n"] - df["total_Op"] - df["total_Opnd"]).abs().gt(1e-6).sum())
        zero_loc = int(df["loc"].eq(0).sum())
        final_count = len(df)
        assert final_count == raw_count - duplicate_count - fractional_count - all_ones_count - conflict_count
        output = OUT_DIR / f"{name}_cleaned.csv"
        df.to_csv(output, index=False)
        audit.append({
            "dataset": name.upper(), "raw_rows": raw_count,
            "exact_duplicates_removed": duplicate_count,
            "fractional_artefacts_removed": fractional_count,
            "all_ones_removed": all_ones_count,
            "conflicting_rows_removed": conflict_count,
            "final_rows": final_count,
            "retained_zero_halstead_with_positive_loc": zero_halstead,
            "retained_operator_sum_mismatches": op_mismatch,
            "retained_zero_loc": zero_loc,
        })
        print(f"{name.upper()}: {raw_count} -> {final_count} rows "
              f"(duplicates {duplicate_count}, fractional {fractional_count}, "
              f"all-ones {all_ones_count}, conflicts {conflict_count})")

    report = pd.DataFrame(audit)
    report.to_csv(OUT_DIR / "cleaning_audit.csv", index=False)
    print("\nAUDIT SUMMARY")
    print(report.to_string(index=False))
    print(f"\nFinal datasets and audit: {OUT_DIR}")
    print("Raw and existing processed datasets unchanged.")
    print("JM1 source provenance still requires independent verification.")


if __name__ == "__main__":
    main()

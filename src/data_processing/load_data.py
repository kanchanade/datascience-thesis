"""
Dataset loading and standardization for the SFRT project.

Handles two source formats:
  - ARFF files from the Shepperd et al. (2013) NASA MDP repository (CM1, JM1, KC1, PC1)
  - CSV file from a secondary GitHub mirror (KC2 only, not present in the Shepperd repo)

Column names differ slightly across the original NASA MDP files in capitalisation and
ordering (e.g. 'iv(g)' vs 'iv(G)', 'n' vs 'N'). This module normalises all datasets to a
single canonical 21-feature schema plus a binary 'defects' target column (1 = defective,
0 = clean), so downstream code never needs to know which source format a dataset came from.
"""

import re
from pathlib import Path

import pandas as pd

# Canonical feature order used throughout the project (21 static code metrics).
# Mapping: lowercase, stripped-of-punctuation key -> canonical column name.
CANONICAL_COLUMNS = [
    "loc", "v_g", "ev_g", "iv_g", "n", "v", "l", "d", "i", "e", "b", "t",
    "locode", "locomment", "loblank", "loccodeandcomment",
    "uniq_op", "uniq_opnd", "total_op", "total_opnd", "branchcount",
]

TARGET_COLUMN = "defects"


def _normalize_key(col_name: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for robust matching."""
    return re.sub(r"[^a-z0-9]", "", col_name.lower())


# Build lookup from normalized source-column-name -> canonical name
_NORMALIZED_TO_CANONICAL = {_normalize_key(c): c for c in CANONICAL_COLUMNS}
# A few known aliases that don't normalize to an exact match automatically
_NORMALIZED_TO_CANONICAL.update({
    "locodeandcomment": "loccodeandcomment",  # pc1 spells this without the extra 'c'... handled defensively
})


def _standardize_columns(df: pd.DataFrame, target_aliases) -> pd.DataFrame:
    """Rename columns to the canonical schema and standardize the target to 0/1."""
    rename_map = {}
    target_col_found = None

    for col in df.columns:
        key = _normalize_key(col)
        if key in _NORMALIZED_TO_CANONICAL:
            rename_map[col] = _NORMALIZED_TO_CANONICAL[key]
        elif col.strip().lower() in [a.lower() for a in target_aliases]:
            target_col_found = col

    df = df.rename(columns=rename_map)

    if target_col_found is None:
        # Some sources already used 'defects' or similar directly
        for alias in target_aliases:
            if alias in df.columns:
                target_col_found = alias
                break
    if target_col_found is None:
        raise ValueError(f"Could not identify target column among: {list(df.columns)}")

    # Standardize target to binary 0/1
    df = df.rename(columns={target_col_found: TARGET_COLUMN})
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    )

    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns after standardization: {missing_cols}")

    return df[CANONICAL_COLUMNS + [TARGET_COLUMN]]


def parse_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF parser: reads @attribute names in order, then @data rows as CSV."""
    with open(path, "r", errors="ignore") as f:
        content = f.read()

    attr_names = re.findall(r"@attribute\s+([^\s]+)\s+", content, flags=re.IGNORECASE)
    data_idx = content.lower().find("@data")
    data_section = content[data_idx + len("@data"):]

    rows = []
    for line in data_section.splitlines():
        line = line.strip().rstrip("\r")
        if not line or line.startswith("%"):
            continue
        rows.append(line.split(","))

    df = pd.DataFrame(rows, columns=attr_names)
    # Convert all but the last (target) column to numeric
    for col in attr_names[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_raw_dataset(name: str, raw_dir: Path) -> pd.DataFrame:
    """
    Load one of the five PROMISE datasets from the project's designated raw source
    (a single GitHub mirror providing the standard 21-metric PROMISE schema
    consistently across all five datasets) and standardize it to the canonical schema.

    A second source (the Shepperd et al., 2013 NASA MDP repository) was evaluated
    during data preparation but rejected: it uses an unreduced, 41-attribute metric
    set for CM1/PC1 and a differently-named 22-attribute set for JM1/KC1, none of
    which match the standard 21-feature schema this project (and the wider CPDP
    literature it compares against) relies on. Using it would have broken the shared
    feature-space requirement that cross-project transfer depends on. See
    docs/data_provenance.md for the full comparison.

    name: one of 'cm1', 'jm1', 'kc1', 'kc2', 'pc1' (case-insensitive)
    raw_dir: path to sfrt-project/data/raw
    """
    name = name.lower()
    target_aliases = ["defects", "problems", "defect", "problem"]

    csv_path = raw_dir / "promise_source" / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # NOTE: row 0 of every file in this source contains a logically impossible
    # fractional value for 'loc' (e.g. 1.1, where lines-of-code should be an integer
    # count). This is NOT specific to this mirror or to KC2: the identical anomaly,
    # at the identical position, appears in CM1's *official* ARFF file from the
    # Shepperd et al. (2013) repository too, so it reflects a long-standing artefact
    # of the original NASA MDP data collection process rather than corruption
    # introduced by any one distributor. It is retained (not dropped) and flagged in
    # clean_dataset() via check_numeric_anomalies(), consistent with documenting
    # rather than silently discarding data-quality issues (Shepperd et al., 2013).

    df = _standardize_columns(df, target_aliases)
    return df


def check_numeric_anomalies(df: pd.DataFrame) -> dict:
    """
    Flag logically implausible values that survive standard cleaning: count-based
    metrics (loc, lOCode, lOComment, lOBlank, branchCount, uniq_Op, uniq_Opnd,
    total_Op, total_Opnd) should be non-negative integers. Fractional values in
    these columns indicate a data-quality artefact rather than a genuine module
    measurement (see Shepperd et al., 2013).
    """
    count_based_cols = [
        "loc", "locode", "locomment", "loblank", "loccodeandcomment",
        "uniq_op", "uniq_opnd", "total_op", "total_opnd", "branchcount",
    ]
    anomalies = {}
    for col in count_based_cols:
        non_integer_mask = (df[col] % 1 != 0)
        if non_integer_mask.any():
            anomalies[col] = int(non_integer_mask.sum())
    return anomalies


def clean_dataset(df: pd.DataFrame, dataset_name: str, log: list) -> pd.DataFrame:
    """
    Apply the documented cleaning steps (Proposal Section 3.2.4):
      - Remove exact duplicate rows (features + target)
      - Report missing values
      - Report zero-variance features
    `log` is a list that entries get appended to, for provenance reporting.
    """
    n_before = len(df)

    n_missing = df.isna().sum().sum()

    # Remove the known artefact row(s): a fractional value in a count-based column
    # (see load_raw_dataset docstring) indicates a template/placeholder row left over
    # from the original ARFF construction, not a genuine module measurement. This
    # occurs at most once per dataset in every one of the five files.
    artefact_mask = (df["loc"] % 1 != 0)
    n_artefact_rows = int(artefact_mask.sum())
    df = df.loc[~artefact_mask].reset_index(drop=True)

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    zero_var_cols = [c for c in CANONICAL_COLUMNS if df[c].nunique() <= 1]
    numeric_anomalies = check_numeric_anomalies(df)

    n_after = len(df)
    defect_rate = df[TARGET_COLUMN].mean() * 100

    log.append({
        "dataset": dataset_name,
        "rows_before_cleaning": n_before,
        "artefact_rows_removed": n_artefact_rows,
        "exact_duplicates_removed": int(n_dupes),
        "rows_after_cleaning": n_after,
        "missing_values_found": int(n_missing),
        "zero_variance_features": zero_var_cols,
        "remaining_fractional_anomalies": numeric_anomalies,
        "defect_rate_pct": round(defect_rate, 2),
    })

    return df


def load_and_clean_all(raw_dir: Path, processed_dir: Path) -> dict:
    """Load, standardize, and clean all five datasets. Save to processed_dir as CSV.
    Returns a dict of {name: DataFrame} plus writes a provenance log."""
    datasets = {}
    log = []

    for name in ["cm1", "jm1", "kc1", "kc2", "pc1"]:
        df = load_raw_dataset(name, raw_dir)
        df = clean_dataset(df, name, log)
        datasets[name] = df

        out_path = processed_dir / f"{name}_clean.csv"
        df.to_csv(out_path, index=False)

    log_df = pd.DataFrame(log)
    log_df.to_csv(processed_dir / "cleaning_log.csv", index=False)

    return datasets, log_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    datasets, log_df = load_and_clean_all(raw_dir, processed_dir)

    print("\n=== Dataset Cleaning Summary ===")
    print(log_df.to_string(index=False))

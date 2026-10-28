"""
Prepare the HOLD-OUT projects for the confirmation test (src/run_holdout_test.py).

The hold-out projects are NASA MDP projects that were never used in this thesis so
far. They come from the "Original" NASA MDP files of Shepperd et al. (2013), kept on
GitHub by C. Tantithamthavorn (repository klainfo/NASADefectDataset, commit
9a25ad697a4f3177ea541392e9ed21005d5f7aa1, folder OriginalData/MDP). The files use
long column names (LOC_TOTAL, HALSTEAD_VOLUME, ...). MAPPING below turns them into the
21 short names used everywhere else in this project.

The mapping was checked on the three projects that exist in both sources: for JM1,
KC1 and PC1, every distinct row of the MDP file appears in the mirror used for the
main study when the 14 whole-number metrics are compared (CM1 differs because the
two CM1 files are different versions, a known PROMISE issue).

Cleaning is the same as for the five main projects:
  1. map the columns and the label (Y/N -> 1/0)
  2. remove rows where a count metric is not a whole number (the artefact-row check)
  3. remove exact duplicates (all 21 metrics and the label identical)
  4. strict rules R1-R5, imported from src/clean_data.py so they are identical

Inclusion rule (fixed BEFORE any result is seen; do not change it afterwards):
  a project is used only if, after cleaning, all 21 metrics vary (none is constant)
  AND it has at least 20 defective modules.
With the current files this keeps KC3, MC1, MC2, MW1, PC3, PC4 and PC5, and drops
KC4 (Halstead metrics all zero, no rows survive) and PC2 (lOBlank is constant).

Run:  python src/prepare_holdout_data.py
Reads  : data/holdout/raw/<NAME>.arff  (downloaded automatically if missing)
Writes : data/holdout/<name>_strict.csv
         experiments/results/holdout_cleaning_log.csv
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation as ev
from clean_data import find_rule_violations

CANDIDATES = ["KC3", "KC4", "MC1", "MC2", "MW1", "PC2", "PC3", "PC4", "PC5"]
SOURCE_URL = ("https://raw.githubusercontent.com/klainfo/NASADefectDataset/"
              "9a25ad697a4f3177ea541392e9ed21005d5f7aa1/OriginalData/MDP/{name}.arff")
MIN_DEFECTIVE = 20

MAPPING = {
    "loc": "LOC_TOTAL", "v(g)": "CYCLOMATIC_COMPLEXITY", "ev(g)": "ESSENTIAL_COMPLEXITY",
    "iv(g)": "DESIGN_COMPLEXITY", "n": "HALSTEAD_LENGTH", "v": "HALSTEAD_VOLUME",
    "l": "HALSTEAD_LEVEL", "d": "HALSTEAD_DIFFICULTY", "i": "HALSTEAD_CONTENT",
    "e": "HALSTEAD_EFFORT", "b": "HALSTEAD_ERROR_EST", "t": "HALSTEAD_PROG_TIME",
    "lOCode": "LOC_EXECUTABLE", "lOComment": "LOC_COMMENTS", "lOBlank": "LOC_BLANK",
    "locCodeAndComment": "LOC_CODE_AND_COMMENT", "uniq_Op": "NUM_UNIQUE_OPERATORS",
    "uniq_Opnd": "NUM_UNIQUE_OPERANDS", "total_Op": "NUM_OPERATORS",
    "total_Opnd": "NUM_OPERANDS", "branchCount": "BRANCH_COUNT",
}
# metrics that are counts and must be whole numbers
COUNT_METRICS = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "lOCode", "lOComment", "lOBlank",
                 "locCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]


def read_arff(path):
    """Minimal ARFF reader: attribute names and comma-separated data rows."""
    names, rows, in_data = [], [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        text = line.strip()
        if not text or text.startswith("%"):
            continue
        lower = text.lower()
        if lower.startswith("@attribute"):
            names.append(text.split()[1].strip("'\""))
        elif lower.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([value.strip() for value in text.split(",")])
    return pd.DataFrame(rows, columns=names)


def download_if_missing(name, raw_folder):
    path = raw_folder / f"{name}.arff"
    if not path.exists():
        raw_folder.mkdir(parents=True, exist_ok=True)
        url = SOURCE_URL.format(name=name)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def clean(raw):
    """Return (cleaned table, counts per step)."""
    label_column = raw.columns[-1]
    table = pd.DataFrame({short: pd.to_numeric(raw[long], errors="coerce")
                          for short, long in MAPPING.items()})
    table["defects"] = raw[label_column].str.strip().str.upper().str[0].eq("Y").astype(int)
    counts = {"raw_rows": len(table)}

    table = table.dropna()
    not_whole = (table[COUNT_METRICS] % 1 != 0).any(axis=1)
    counts["artefact_rows"] = int(not_whole.sum())
    table = table[~not_whole]

    before = len(table)
    table = table.drop_duplicates().reset_index(drop=True)
    counts["exact_duplicates"] = before - len(table)

    violations = find_rule_violations(table)
    remove = np.zeros(len(table), dtype=bool)
    for rule, flags in violations.items():
        counts[rule] = int(flags.sum())
        remove |= flags.values
    table = table[~remove].reset_index(drop=True)
    counts["strict_rows"] = len(table)
    counts["defective"] = int(table["defects"].sum())
    counts["defect_rate"] = round(table["defects"].mean(), 4) if len(table) else np.nan
    return table, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()
    raw_folder = args.root / "data" / "holdout" / "raw"
    out_folder = args.root / "data" / "holdout"

    log = []
    for name in CANDIDATES:
        table, counts = clean(read_arff(download_if_missing(name, raw_folder)))
        constant = [f for f in ev.FEATURE_COLUMNS if len(table) == 0 or table[f].nunique() <= 1]
        included = len(constant) == 0 and counts["defective"] >= MIN_DEFECTIVE
        counts.update({"project": name, "constant_metrics": " ".join(constant), "included": included})
        log.append(counts)
        if included:
            table[ev.FEATURE_COLUMNS + ["defects"]].to_csv(out_folder / f"{name.lower()}_strict.csv", index=False)

    log = pd.DataFrame(log)
    first = ["project", "included", "raw_rows", "artefact_rows", "exact_duplicates"]
    log = log[first + [c for c in log.columns if c not in first]]
    log_path = args.root / "experiments" / "results" / "holdout_cleaning_log.csv"
    log.to_csv(log_path, index=False)
    print(log.to_string(index=False))
    print(f"\nIncluded: {', '.join(log[log.included].project)}")
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()

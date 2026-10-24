"""

Strict cleaning of the five datasets (the only place the strict rules are defined).

Reads  : data/processed/{name}_cleaned.csv   (exact duplicates already removed)
Writes : data/final/{name}_strict.csv
         experiments/results/cleaning_log.csv   (rows removed per rule)

Rules (after Shepperd et al., 2013):
  R1  placeholder row: all 21 features equal 1
  R2  conflicting labels: identical features, opposite defect label (both removed)
  R3  Halstead length n = 0 although loc > 0
  R4  n not equal to total_Op + total_Opnd
  R5  loc = 0
A row that breaks several rules is removed once but counted under each rule.


"""

import argparse
from pathlib import Path

import pandas as pd

DATASET_NAMES = ["cm1", "jm1", "kc1", "kc2", "pc1"]

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]


def find_rule_violations(df):
    """Return a dict of rule name -> boolean Series (True = row breaks the rule)."""
    features = df[FEATURE_COLUMNS]
    label = df["defects"].astype(int)

    r1 = (features == 1).all(axis=1)

    # R2: group rows by their 21 feature values; a group containing both
    # labels is a conflict. (R1 rows are excluded so one artefact row is not
    # counted twice.)
    n_labels_in_group = label.groupby([features[c] for c in FEATURE_COLUMNS]).transform("nunique")
    r2 = (n_labels_in_group > 1) & ~r1

    r3 = (features["n"] == 0) & (features["loc"] > 0)
    r4 = (features["n"] != features["total_Op"] + features["total_Opnd"]) & ~r1
    r5 = features["loc"] == 0

    return {"R1_all_ones_row": r1, "R2_conflicting_labels": r2,
            "R3_zero_length_but_loc": r3, "R4_length_not_op_sum": r4,
            "R5_zero_loc": r5}


def main(root):
    in_folder = root / "data" / "processed"
    out_folder = root / "data" / "final"
    out_folder.mkdir(parents=True, exist_ok=True)
    results_folder = root / "experiments" / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for name in DATASET_NAMES:
        df = pd.read_csv(in_folder / f"{name}_cleaned.csv")
        violations = find_rule_violations(df)
        remove = pd.concat(violations.values(), axis=1).any(axis=1)
        strict = df[~remove].reset_index(drop=True)
        strict.to_csv(out_folder / f"{name}_strict.csv", index=False)

        row = {"Dataset": name.upper(), "Rows_before": len(df)}
        row.update({rule: int(mask.sum()) for rule, mask in violations.items()})
        row.update({
            "Rows_removed_total": int(remove.sum()),
            "Rows_after": len(strict),
            "Defect_rate_before_%": round(100 * df["defects"].astype(int).mean(), 2),
            "Defect_rate_after_%": round(100 * strict["defects"].astype(int).mean(), 2),
        })
        log_rows.append(row)

    log = pd.DataFrame(log_rows)
    log.to_csv(results_folder / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nWrote strict files to {out_folder}")
    print(f"Wrote log to {results_folder / 'cleaning_log.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="project folder (default: the folder above src/)")
    main(parser.parse_args().root)

"""Audit and remove all-ones artefact rows from existing processed NASA datasets.

This is an intermediate cleaning review, not the final raw-to-processed pipeline.
Original processed CSV files are never overwritten.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'cleaning_review'
DATASETS = {
    'CM1': 'cm1_cleaned.csv',
    'JM1': 'jm1_cleaned.csv',
    'KC1': 'kc1_cleaned.csv',
    'KC2': 'kc2_cleaned.csv',
    'PC1': 'pc1_cleaned.csv',
}
FEATURE_COLUMNS = [
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
    'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
    'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount',
]


def main():
    print('=' * 60)
    print('NASA PROMISE - DATA CLEANING')
    print('Step 1: Remove artificial all-ones rows')
    print('=' * 60)

    # Validate every input before writing any output.
    loaded = {}
    for name, filename in DATASETS.items():
        path = INPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f'{name}: Dataset not found: {path}')
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS + ['defects'] if c not in df.columns]
        if missing:
            raise ValueError(f'{name}: Missing columns: {missing}')
        loaded[name] = df

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, filename in DATASETS.items():
        df = loaded[name]
        mask = df[FEATURE_COLUMNS].eq(1).all(axis=1)
        cleaned = df.loc[~mask].copy()
        assert not cleaned[FEATURE_COLUMNS].eq(1).all(axis=1).any(), name
        out = OUTPUT_DIR / filename
        cleaned.to_csv(out, index=False)
        summary.append({
            'Dataset': name,
            'Before': len(df),
            'Removed': int(mask.sum()),
            'After': len(cleaned),
        })
        print(f'{name}: {len(df)} -> {len(cleaned)} rows; removed {int(mask.sum())}; saved {out}')

    summary_df = pd.DataFrame(summary)
    summary_file = OUTPUT_DIR / 'all_ones_removal_audit.csv'
    summary_df.to_csv(summary_file, index=False)
    print('\nCLEANING SUMMARY')
    print(summary_df.to_string(index=False))
    print('\nAll-ones removal completed. Original processed datasets unchanged.')
    print(f'Audit saved to: {summary_file}')


if __name__ == '__main__':
    main()

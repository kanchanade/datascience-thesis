"""
Source-balance ablation: does JM1's dominance of the pooled training data
drive the cross-project results?

"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import evaluation as ev
from run_experiments import parse_seeds

METHODS = ["unweighted", "variance_k10", "mutualinfo_k5", "mutualinfo_k10",
           "sfrt_hard_k5", "sfrt_hard_k10", "sfrt_soft"]
POOLS = ["equal", "size_matched"]


def one_run(data, level, pool, rows_each, method_name, step, classifier_name, target_name, seed):
    start = time.time()
    scores = ev.evaluate(data, target_name, classifier_name, step, seed,
                         pool_mode=pool, rows_per_source=rows_each)
    return {
        "Level": level, "Pool": pool, "Rows_per_source": rows_each, "Method": method_name,
        "Classifier": classifier_name, "Target_Dataset": target_name.upper(), "Seed": seed,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
        "Seconds": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--classifiers", nargs="*", default=ev.CLASSIFIER_NAMES)
    parser.add_argument("--rows-per-source", type=int, default=None,
                        help="default: size of the smallest project")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()

    ev.CACHE_DIR = args.root / "experiments" / "cache" / "shap"
    methods = ev.build_methods()
    data = ev.load_all_data(args.level, args.root)
    rows_each = args.rows_per_source or ev.smallest_source_size(data)
    out_path = args.root / "experiments" / "results" / f"ablation_balance_{args.level}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"level = {args.level}; {rows_each} rows per source project "
          f"({4 * rows_each} training rows per target)")

    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.Pool, old.Method, old.Classifier, old.Target_Dataset, old.Seed))

    for seed in parse_seeds(args.seeds):
        tasks = [(p, m, c, t) for p in POOLS for m in args.methods for c in args.classifiers
                 for t in ev.DATASET_NAMES
                 if ev.applies(methods[m], c) and (p, m, c, t.upper(), seed) not in done]
        if not tasks:
            print(f"seed {seed}: already complete")
            continue
        start = time.time()
        rows = Parallel(n_jobs=args.jobs)(
            delayed(one_run)(data, args.level, p, rows_each, m, methods[m], c, t, seed)
            for p, m, c, t in tasks)
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"seed {seed}: {len(rows)} runs in {time.time() - start:.0f}s -> {out_path.name}")


if __name__ == "__main__":
    main()

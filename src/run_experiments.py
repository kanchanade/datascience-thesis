"""
Runs the experiments: every (method x classifier x target x seed) combination,
saving ONE ROW PER RUN to a long-format CSV.

Examples (run from the project folder):
    python src/run_experiments.py                       # all methods, 10 seeds, strict data
    python src/run_experiments.py --seeds 0-2           # quick test with 3 seeds
    python src/run_experiments.py --level duplicates    # sensitivity analysis
    python src/run_experiments.py --methods unweighted mutualinfo_k10 --jobs 4

It is safe to stop and restart: runs already in the output file are skipped.
Results go to experiments/results/runs_<level>.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import evaluation as ev


def parse_seeds(text):
    """'0-9' -> [0..9]; '0,3,7' -> [0,3,7]"""
    if "-" in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(s) for s in text.split(",")]


def one_run(data, level, method_name, step, classifier_name, target_name, seed):
    start = time.time()
    scores = ev.evaluate(data, target_name, classifier_name, step, seed)
    return {
        "Level": level, "Method": method_name, "Classifier": classifier_name,
        "Target_Dataset": target_name.upper(), "Seed": seed,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
        "Seconds": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="strict", choices=["strict", "duplicates"])
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=ev.CLASSIFIER_NAMES)
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers")
    parser.add_argument("--lr-c-sweep", action="store_true",
                        help="also run Logistic Regression at C=0.1 and C=10")
    parser.add_argument("--root", type=Path, default=ev.PROJECT_ROOT)
    args = parser.parse_args()

    ev.CACHE_DIR = args.root / "experiments" / "cache" / "shap"
    classifiers = list(args.classifiers) + (ev.LR_C_VARIANTS if args.lr_c_sweep else [])
    methods = ev.build_methods()
    chosen = args.methods or list(methods)
    data = ev.load_all_data(args.level, args.root)
    out_path = args.root / "experiments" / "results" / f"runs_{args.level}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.Method, old.Classifier, old.Target_Dataset, old.Seed))

    for seed in parse_seeds(args.seeds):
        tasks = [(m, c, t) for m in chosen for c in classifiers for t in ev.DATASET_NAMES
                 if ev.applies(methods[m], c) and (m, c, t.upper(), seed) not in done]
        if not tasks:
            print(f"seed {seed}: already complete")
            continue
        start = time.time()
        rows = Parallel(n_jobs=args.jobs)(
            delayed(one_run)(data, args.level, m, methods[m], c, t, seed) for m, c, t in tasks
        )
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"seed {seed}: {len(rows)} runs in {time.time() - start:.0f}s -> {out_path.name}")


if __name__ == "__main__":
    main()

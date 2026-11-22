import argparse
from itertools import product
from pathlib import Path
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pandas as pd
import torch
from src.experiment import run_experiment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ETTh", "ETTm", "eld"],
        default=["ETTh", "ETTm"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["level", "delta", "delta_ridge"],
        default=["level", "delta", "delta_ridge"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--profile", choices=["quick", "main", "extended"], default="main"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output / args.profile
    rows = []
    for dataset, mode, seed in product(args.datasets, args.modes, args.seeds):
        folder = output / dataset / mode / str(seed)
        if not (folder / "run.json").is_file():
            print(dataset, mode, seed, flush=True)
            run_experiment(dataset, mode, seed, args.profile, folder, ROOT / "data")
        scores = pd.read_csv(folder / "metrics.csv")
        rows.append(scores.assign(dataset=dataset, target_mode=mode, seed=seed))
    scores = pd.concat(rows, ignore_index=True)
    metrics = [
        "MAE",
        "RMSE",
        "sMAPE_ratio",
        "abs_weighted_sMAPE",
        "MAE_train_z",
        "RMSE_train_z",
    ]
    summary = (
        scores.query("split == 'test'")
        .groupby(["dataset", "target_mode", "model"])[metrics]
        .agg(["mean", "std"])
    )
    scores.to_csv(output / "all_scores.csv", index=False)
    summary.to_csv(output / "summary.csv")
    print(summary[["MAE", "RMSE"]].round(4).to_string())


if __name__ == "__main__":
    main()

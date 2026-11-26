"""Tổng hợp sai số trên cùng các mốc dự báo."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def main(root):
    rows = []
    bands = []
    bridge = []
    coverage = []
    for p in sorted(root.glob("*/*/forecasts.npz")):
        folder = p.parent
        protocol = folder.parent.name
        dataset, h = folder.name.split("_h")
        h = int(h)
        sel = json.loads((folder / "selection.json").read_text())["selected"]
        origins = np.load(p)
        actual = origins["actual"]
        series = origins["series"]
        table = (
            pd.read_csv(folder / "scores.csv")
            .query("split=='test'")
            .set_index("method")
        )
        c = pd.read_csv(folder / "coverage.csv")
        coverage.append(
            dict(
                protocol=protocol,
                dataset=dataset,
                horizon=h,
                series=len(c),
                rows=int(c.observations.sum()),
                train_rows=int(c.train_rows.sum()),
                training_windows=int(c.train_windows.sum()),
                test_points=actual.size,
            )
        )
        all_pred = {k: origins[sel if k == "selected" else k] for k in table.index}
        for fname, archive, model in [
            ("td3_scores", "td3_weights", "TD3"),
            ("patch_scores", "patch_forecasts", "PatchTST-small"),
        ]:
            if (folder / (fname + ".csv")).exists() and (folder / (archive + ".npz")).exists():
                aux = np.load(folder / (archive + ".npz"))
                assert np.array_equal(series, aux["series"]) and np.array_equal(
                    origins["origin"], aux["origin"]
                )
                if "actual" in aux:
                    np.testing.assert_allclose(actual, aux["actual"])
                table.loc[model] = (
                    pd.read_csv(folder / (fname + ".csv"))
                    .query("split=='test' and seed=='ensemble'")
                    .iloc[0]
                )
                all_pred[model] = aux["ensemble"]
        for model, pred in all_pred.items():
            ae = np.abs(actual.astype(float) - pred)
            assert np.isclose(ae.mean(), table.loc[model, "MAE"], rtol=1e-5, atol=1e-4)
            assert np.isclose(
                np.sqrt((ae**2).mean()), table.loc[model, "RMSE"], rtol=1e-5, atol=1e-4
            )
            rows.append(
                dict(
                    protocol=protocol,
                    dataset=dataset,
                    horizon=h,
                    model=model,
                    selected_model=sel,
                    MAE=ae.mean(),
                    RMSE=np.sqrt((ae**2).mean()),
                    sMAPE=table.loc[model, "sMAPE"],
                    n=ae.size,
                )
            )
        if protocol == "bridge":
            ref = (
                pd.read_csv(folder / "midterm_matched.csv").groupby("method").MAE.mean()
            )
            for method in ["T2f", "Calibrated", "SeasonalNaive", "Persistence"]:
                bridge.append(
                    dict(
                        dataset=dataset,
                        baseline=method,
                        midterm=ref[method],
                        final=table.loc["selected", "MAE"],
                        reduction_percent=100
                        * (1 - table.loc["selected", "MAE"] / ref[method]),
                    )
                )
        else:
            # Gộp các trạm cùng thời điểm trước khi lấy mẫu theo khối 7 ngày.
            base = "daily" if dataset == "ELD" else "last"
            diff = (
                np.abs(all_pred["selected"] - actual) - np.abs(origins[base] - actual)
            ).mean(1)
            by_time = pd.Series(diff, index=origins["origin"]).groupby(level=0).mean()
            days_per_step = (
                pd.Timestamp(by_time.index[1]) - pd.Timestamp(by_time.index[0])
            ) / pd.Timedelta(days=1)
            block = min(len(by_time), max(1, round(7 / days_per_step)))
            rng = np.random.default_rng(2026)
            boot = []
            for _ in range(1000):
                starts = rng.integers(
                    len(by_time), size=int(np.ceil(len(by_time) / block))
                )
                indexes = ((starts[:, None] + np.arange(block)) % len(by_time)).ravel()[
                    : len(by_time)
                ]
                boot.append(by_time.to_numpy()[indexes].mean())
            lo, hi = np.quantile(boot, [0.025, 0.975])
            bands.append(
                dict(
                    dataset=dataset,
                    horizon=h,
                    baseline=base,
                    delta_MAE=diff.mean(),
                    lo95=lo,
                    hi95=hi,
                    block_origins=block,
                )
            )
    if not rows:
        raise FileNotFoundError(f"Chưa có tệp forecasts.npz trong {root}")

    for name, data in [
        ("comparison", rows),
        ("paired_midterm", bridge),
        ("coverage", coverage),
        ("paired_blocks", bands),
    ]:
        pd.DataFrame(data).to_csv(root / (name + ".csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/final",
    )
    main(p.parse_args().output)

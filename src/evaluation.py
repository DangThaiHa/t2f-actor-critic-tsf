import numpy as np
import pandas as pd


def forecast_metrics(series, col):
    y = series.actual.to_numpy()
    pred = series[col].to_numpy()
    error = np.abs(y - pred)
    denom = np.abs(y) + np.abs(pred)
    ratio = np.divide(2 * error, denom, out=np.zeros_like(error), where=denom > 0)
    w = series.weight.abs().to_numpy()
    return dict(
        MAE=error.mean(),
        RMSE=np.sqrt(np.mean(error**2)),
        sMAPE_ratio=ratio.mean(),
        abs_weighted_sMAPE=np.average(ratio, weights=w) if w.sum() else None,
        MAE_train_z=np.mean(error / series.scale),
        RMSE_train_z=np.sqrt(np.mean((error / series.scale) ** 2)),
        n=len(y),
    )


def rolling_forecast(
    model,
    groups,
    times,
    features,
    target,
    weight,
    horizon,
    period,
    origins,
    means,
    scales,
    level_scales,
    target_mode,
    splits,
):
    records = []
    for part, lo, hi in splits:
        starts = np.linspace(
            lo, hi - horizon, min(origins, (hi - lo) // horizon), dtype=int
        )
        assert len(starts) > 0 and np.all(np.diff(starts) >= horizon)
        for i in starts:
            windows = []
            for station, series in groups.items():
                window = series.iloc[i - max(32, horizon) : i][features].copy()
                window["station"] = station
                windows.append(window)
            history = (
                pd.concat(windows)
                .reset_index()
                .set_index(["station", "date"])
                .sort_index()
            )
            pred = model.predict(history).prediction
            for station, series in groups.items():
                future = times[i : i + horizon]
                assert (pred.loc[station].index == future).all(), (
                    pred.loc[station].index,
                    future,
                )
                forecast_values = (
                    pred.loc[station].to_numpy() * scales[station] + means[station]
                )
                if target_mode.startswith("delta"):
                    forecast_values = series[target].iloc[i - 1] + np.cumsum(
                        forecast_values
                    )
                for h, stamp in enumerate(future):
                    records.append(
                        (
                            part,
                            station,
                            times[i - 1],
                            stamp,
                            h + 1,
                            series.at[stamp, target],
                            series.at[stamp, weight],
                            forecast_values[h],
                            series[target].iloc[i - 1],
                            series[target].iloc[i + (h % period) - period],
                            level_scales[station],
                        )
                    )
    data = pd.DataFrame(
        records,
        columns=[
            "split",
            "station",
            "origin",
            "timestamp",
            "h",
            "actual",
            "weight",
            "T2f",
            "Persistence",
            "SeasonalNaive",
            "scale",
        ],
    )
    assert np.isfinite(data.select_dtypes("number")).all().all()
    return data


def save_results(data, out):
    val = data[data.split == "val"]
    alpha = min(
        np.linspace(0, 1, 21),
        key=lambda a: np.abs(
            val.actual - (a * val.T2f + (1 - a) * val.Persistence)
        ).mean(),
    )
    data["Calibrated"] = alpha * data.T2f + (1 - alpha) * data.Persistence
    result = []
    for part, series in data.groupby("split"):
        for method in ["T2f", "Persistence", "SeasonalNaive", "Calibrated"]:
            result.append(
                dict(split=part, model=method, **forecast_metrics(series, method))
            )
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(result)
    metrics.to_csv(out / "metrics.csv", index=False)
    data.to_csv(out / "predictions.csv", index=False)
    return float(alpha)

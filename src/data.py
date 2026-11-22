import numpy as np
import pandas as pd


def load_series(dataset, data_dir):
    target, weight, features = (
        "OT",
        "HULL",
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
    )
    horizon, global_len, period = {
        "ETTh": (2, 1, 24),
        "ETTm": (30, 35, 96),
        "eld": (2, 3, 4),
    }[dataset]
    path = data_dir / f"{dataset.lower()}.parquet"
    if dataset == "eld":
        frame = pd.read_parquet(path, filters=[("id", "==", "MT_211")]).rename(
            columns={"id": "station", "timestamp": "date"}
        )
        target, weight, features = "load", "cost_kwh", ["tmax", "tmin", "tmed"]
    else:
        frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame.date)
    frame["station"] = frame.station.astype(str)
    cols = list(dict.fromkeys(features + [target, weight]))
    groups = {
        station: series.set_index("date")[cols].sort_index().dropna()
        for station, series in frame.groupby("station")
    }
    if dataset == "eld":
        groups = {
            station: series.resample("6h", closed="right", label="right")
            .mean()
            .dropna()
            for station, series in groups.items()
        }
    times = None
    for series in groups.values():
        assert series.index.is_unique
        times = series.index if times is None else times.intersection(series.index)
    times = times.sort_values()
    delta = pd.Timedelta(days=1) / period
    runs = times.to_series().diff().ne(delta).cumsum()
    times = max(
        (pd.DatetimeIndex(series) for _, series in times.to_series().groupby(runs)),
        key=len,
    )
    groups = {station: series.loc[times].copy() for station, series in groups.items()}
    return groups, times, target, weight, features, horizon, global_len, period


def build_training_set(groups, target, start, rows, target_mode):
    train = []
    scales = {}
    means = {}
    level_scales = {}
    for station, series in groups.items():
        sample = series.iloc[start - rows : start].copy()
        level_scales[station] = float(sample[target].std(ddof=0))
        if target_mode.startswith("delta"):
            sample[target] = series[target].diff().iloc[start - rows : start]
        means[station] = float(sample[target].mean())
        scales[station] = float(sample[target].std(ddof=0))
        if not scales[station] > 1e-8:
            raise ValueError(f"Constant target in training for {station}")
        sample[target] = (sample[target] - means[station]) / scales[station]
        sample["station"] = station
        train.append(sample)
    train = pd.concat(train).reset_index().set_index(["station", "date"]).sort_index()
    assert np.isfinite(train).all().all()
    return train, means, scales, level_scales

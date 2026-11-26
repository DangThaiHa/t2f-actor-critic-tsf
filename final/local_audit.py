"""Đánh giá từng mô hình thành phần ở các mốc kiểm tra giữa kỳ."""

import argparse
from pathlib import Path
import random
import pandas as pd
import numpy as np
import torch
from src.data import load_series, build_training_set
from src.model import core, RidgeVAR, logger
from .compare import ROOT, metrics


def audit(ds, mode, output):
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    groups, times, target, weight, features, h, global_length, period = load_series(
        ds, ROOT / "data"
    )
    v, t = int(0.7 * len(times)), int(0.8 * len(times))
    train, mu, sd, _ = build_training_set(groups, target, v, 512, mode)
    model = core.T2f(
        len(features),
        1,
        16,
        1,
        h,
        global_length,
        list(groups),
        freq=None,
        logger=logger,
        batch_size=16,
        timestamp_col="date",
        series_id_col="station",
        epochs=8,
    )
    if mode == "delta_ridge":
        model.local_models[0] = RidgeVAR()
    model.fit(train[features], train[[target]], train[[weight]])
    members = ["var", "lstm", "gru", "relu", "tanh"]
    local = []
    weights = []
    original = model.ensure_prediction_shape

    def capture(pred):
        z = original(pred)
        local.append(z.ravel().copy())
        return z

    model.ensure_prediction_shape = capture
    select = model.global_model.select_action

    def action(state):
        w = select(state)
        weights.append(w.ravel().copy())
        return w

    model.global_model.select_action = action
    rows = []
    for i in np.linspace(t, len(times) - h, 48, dtype=int):
        for sid, g in groups.items():
            window = g.iloc[i - 32 : i][features]
            history = pd.concat({sid: window}, names=["station", "date"])
            local.clear()
            weights.clear()
            mixture = model.predict(history).prediction.to_numpy()
            assert len(local) == 5 and len(weights) == 1
            np.testing.assert_allclose(
                np.sum(np.array(local) * weights[0][:, None], axis=0),
                mixture,
                rtol=1e-6,
                atol=1e-5,
            )
            forecasts = dict(zip(members, local))
            forecasts.update(uniform=np.mean(local, axis=0), actor=mixture)
            for name, pred in forecasts.items():
                yhat = pred * sd[sid] + mu[sid]
                if mode.startswith("delta"):
                    yhat = g[target].iloc[i - 1] + np.cumsum(yhat)
                for j in range(h):
                    rows.append(
                        (
                            sid,
                            times[i - 1],
                            j + 1,
                            name,
                            float(yhat[j]),
                            g[target].iloc[i + j],
                            weights[0][members.index(name)]
                            if name in members
                            else np.nan,
                        )
                    )
    d = pd.DataFrame(
        rows,
        columns=[
            "station",
            "origin",
            "h",
            "model",
            "prediction",
            "actual",
            "actor_weight",
        ],
    )
    dest = output / ds / mode
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "components.csv.gz", index=False)
    s = pd.DataFrame(
        [
            dict(
                model=k,
                peak_abs=g.prediction.abs().max(),
                **metrics(g.actual.to_numpy(), g.prediction.to_numpy()),
            )
            for k, g in d.groupby("model")
        ]
    )
    s.to_csv(dest / "scores.csv", index=False)
    ref = pd.read_csv(
        ROOT / "results/reference" / ds / mode / "42/predictions.csv",
        dtype={"station": str},
        parse_dates=["origin"],
    ).query("split=='test'")
    joined = d[d.model == "actor"].merge(
        ref, on=["station", "origin", "h"], validate="one_to_one"
    )
    assert len(joined) == len(ref)
    np.testing.assert_allclose(joined.prediction, joined.T2f, rtol=2e-5, atol=1e-4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["eld", "ETTh", "ETTm"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/final/local_audit")
    a = p.parse_args()
    for mode in ["level", "delta_ridge"]:
        audit(a.dataset, mode, a.output)

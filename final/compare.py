import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from upstream.t2f_base import Actor
from .experts import NAMES, fit_experts, predict_experts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/final"
SEEDS = [42, 43, 44]


def origins(lo, hi, h, step):
    return np.arange(lo, hi - h + 1, step)


def assemble(dataset, h, bridge):
    period = {"ETTh": 24, "ETTm": 96, "ELD": 4}[dataset]
    frames = {}
    if bridge:
        from src.data import load_series

        groups, _, target, _, features, _, _, _ = load_series(
            dataset if dataset != "ELD" else "eld", ROOT / "data"
        )
        frames = {str(k): g[[target] + features] for k, g in groups.items()}
    elif dataset == "ELD":
        panel = pd.read_parquet(DATA / "eld_6h.parquet")
        frames = {k: panel[[k]] for k in panel}
    else:
        for i in [1, 2]:
            g = pd.read_csv(DATA / f"{dataset}{i}.csv", parse_dates=["date"]).set_index(
                "date"
            )
            frames[str(i)] = g[["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]]
    chunks = {k: [] for k in ["meta", "val", "test"]}
    counts = []
    for sid, g in frames.items():
        y = g.to_numpy(dtype=float)
        times = g.index
        n = len(g)
        assert np.isfinite(y).all() and np.all(
            np.diff(times.values) == pd.Timedelta(days=1) / period
        )
        v, t = int(0.7 * n), int(0.8 * n)
        cut = (v + t) // 2
        mod, nfit = fit_experts(y, times, v, h, period)
        short, nshort = fit_experts(y, times, v, h, period, 512)
        scale = max(float(y[:v, 0].std()), 1e-4)
        counts.append(
            dict(
                series=sid,
                observations=n,
                train_end=str(times[v - 1]),
                train_rows=v,
                train_windows=nfit,
                short_windows=nshort,
                selection_start=str(times[cut]),
                test_start=str(times[t]),
                end=str(times[-1]),
            )
        )
        for split, lo, hi in [("meta", v, cut), ("val", cut, t), ("test", t, n)]:
            step = max(h, 1) if split == "test" else max(1, h // 2)
            idx = origins(lo, hi, h, step)
            if bridge and split == "test":
                idx = np.linspace(t, n - h, min(48, (n - t) // h), dtype=int)
            pred, ctx = predict_experts(mod, y, times, idx, h, period, scale)
            reduced, _ = predict_experts(short, y, times, idx, h, period, scale)
            assert idx.min() >= lo and idx.max() + h <= hi
            chunks[split].append(
                dict(
                    series=np.repeat(sid, len(idx)),
                    origin=times[idx - 1].to_numpy(),
                    ctx=ctx,
                    pred=pred,
                    target=y[idx[:, None] + np.arange(h), 0].astype("float32"),
                    anchor=y[idx - 1, 0].astype("float32"),
                    scale=np.full(len(idx), scale, dtype="float32"),
                    short=reduced[:, :, -1],
                )
            )
    return {
        k: {f: np.concatenate([part[f] for part in lst]) for f in lst[0]}
        for k, lst in chunks.items()
    }, counts


def actor_predictions(data, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    meta = data["meta"]
    model = Actor(meta["ctx"].shape[1], len(NAMES), 1, logging.getLogger("actor"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(250):
        i = rng.integers(len(meta["ctx"]), size=128)
        weights = torch.softmax(4 * model(torch.from_numpy(meta["ctx"][i])), dim=1)
        errors = (meta["pred"][i] - meta["target"][i, :, None]) / meta["scale"][
            i, None, None
        ]
        loss = (weights[:, None, :] * torch.from_numpy(errors)).sum(-1).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    result = {}
    with torch.no_grad():
        for split in ["val", "test"]:
            x = data[split]
            w = torch.softmax(4 * model(torch.from_numpy(x["ctx"])), dim=1).numpy()
            result[split] = ((w[:, None, :] * x["pred"]).sum(-1), w)
    return result


def metrics(y, p):
    e = y.astype(float) - p
    den = np.abs(y) + np.abs(p)
    return dict(
        MAE=np.abs(e).mean(),
        RMSE=np.sqrt((e**2).mean()),
        sMAPE=np.divide(2 * np.abs(e), den, out=np.zeros_like(e), where=den > 0).mean(),
        points=e.size,
    )


def evaluate(dataset, h, bridge, out):
    torch.set_num_threads(1)
    data, counts = assemble(dataset, h, bridge)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(out / "coverage.csv", index=False)
    # Chuẩn hóa sai số theo từng chuỗi trước khi học trọng số.
    m = data["meta"]
    errors = (m["pred"] - m["target"][:, :, None]) / m["scale"][:, None, None]
    err = errors.reshape(-1, len(NAMES)).astype("float64")

    def objective(w):
        residual = err @ w
        return np.abs(residual).mean(), (np.sign(residual) @ err) / len(err)

    fit = minimize(
        objective,
        np.full(len(NAMES), 1 / len(NAMES)),
        jac=True,
        method="SLSQP",
        bounds=[(0, 1)] * len(NAMES),
        constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)],
        options={"ftol": 1e-9, "maxiter": 150},
    )
    if not fit.success:
        raise ValueError(fit.message)
    w = fit.x
    actors = [actor_predictions(data, k) for k in SEEDS]
    scores = []
    per_series = []
    choice = None
    for split in ["val", "test"]:
        d = data[split]
        predictions = {name: d["pred"][:, :, i] for i, name in enumerate(NAMES)}
        predictions.update(
            ridge_512=d["short"], uniform=d["pred"].mean(-1), stacked=d["pred"] @ w
        )
        for seed, model in zip(SEEDS, actors):
            predictions[f"actor_{seed}"] = model[split][0]
        predictions["actor_mean"] = np.mean([a[split][0] for a in actors], axis=0)
        y = d["target"]
        if split == "val":
            candidates = NAMES + ["uniform", "stacked", "actor_mean"]
            choice = min(candidates, key=lambda k: metrics(y, predictions[k])["MAE"])
        predictions["selected"] = predictions[choice]
        for name, p in predictions.items():
            assert np.isfinite(p).all() and p.shape == y.shape
            scores.append(dict(split=split, method=name, **metrics(y, p)))
            if split == "test":
                for sid in np.unique(d["series"]):
                    mask = d["series"] == sid
                    per_series.append(
                        dict(series=sid, method=name, **metrics(y[mask], p[mask]))
                    )
        if split == "test":
            np.savez_compressed(
                out / "forecasts.npz",
                series=d["series"],
                origin=d["origin"],
                actual=y,
                **{
                    k: p.astype("float32")
                    for k, p in predictions.items()
                    if k != "selected"
                },
            )
            np.savez_compressed(
                out / "actor_weights.npz",
                names=NAMES,
                **{str(k): a[split][1] for k, a in zip(SEEDS, actors)},
            )
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(per_series).to_csv(out / "by_series.csv", index=False)
    (out / "selection.json").write_text(
        json.dumps(
            dict(selected=choice, weights=dict(zip(NAMES, w)), horizon=h, seeds=SEEDS),
            indent=2,
        )
    )
    if bridge:
        mid = (
            ROOT
            / "results/reference"
            / ("eld" if dataset == "ELD" else dataset)
            / "delta_ridge"
        )
        a = np.load(out / "forecasts.npz")
        base = pd.DataFrame(
            dict(
                series=np.repeat(a["series"], h),
                origin=np.repeat(a["origin"], h),
                h=np.tile(np.arange(1, h + 1), len(a["origin"])),
                actual=a["actual"].ravel(),
            )
        )
        rows = []
        for seed in SEEDS:
            ref = (
                pd.read_csv(
                    mid / str(seed) / "predictions.csv",
                    dtype={"station": str},
                    parse_dates=["origin"],
                )
                .query("split=='test'")
                .rename(columns={"station": "series"})
            )
            matched = base.merge(
                ref,
                on=["series", "origin", "h"],
                validate="one_to_one",
                suffixes=("", "_old"),
            )
            assert len(matched) == len(ref) == len(base) and np.allclose(
                matched.actual, matched.actual_old, atol=1e-5
            )
            for name in ["T2f", "Calibrated", "Persistence", "SeasonalNaive"]:
                rows.append(
                    dict(
                        seed=seed,
                        method=name,
                        **metrics(
                            matched.actual_old.to_numpy(), matched[name].to_numpy()
                        ),
                    )
                )
        pd.DataFrame(rows).to_csv(out / "midterm_matched.csv", index=False)
    print(dataset, h, "bridge" if bridge else "full", choice, scores[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ETTh", "ETTm", "ELD"], required=True)
    p.add_argument("--horizon", type=int)
    p.add_argument("--bridge", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/final")
    args = p.parse_args()
    h = args.horizon or {"ETTh": 2, "ETTm": 30, "ELD": 2}[args.dataset]
    evaluate(
        args.dataset,
        h,
        args.bridge,
        args.output / ("bridge" if args.bridge else "full") / f"{args.dataset}_h{h}",
    )

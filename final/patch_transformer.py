"""Dự báo OT bằng PatchTST-small trên hai trạm ETT."""

import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from .compare import ROOT, DATA, SEEDS, metrics, origins


class PatchForecaster(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.position = nn.Parameter(torch.randn(11, 32) * 0.01)
        self.encoder = nn.TransformerEncoderLayer(
            32, 4, 64, 0.1, batch_first=True, activation="gelu"
        )
        self.head = nn.Linear(11 * 32, horizon)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        patches = ((x - mean) / sd).unfold(1, 16, 8)
        encoded = self.encoder(self.embed(patches) + self.position)
        return self.head(encoded.flatten(1)) * sd + mean


def prepare(dataset, h):
    parts = {k: [[], [], [], []] for k in ["train", "meta", "val", "test"]}
    for sid in [1, 2]:
        g = pd.read_csv(DATA / f"{dataset}{sid}.csv", parse_dates=["date"])
        y = g.OT.to_numpy(dtype="float32")
        n = len(y)
        v, t = int(0.7 * n), int(0.8 * n)
        c = (v + t) // 2
        for part, lo, hi, step in [
            ("train", 96, v, max(2, h // 4)),
            ("meta", v, c, max(1, h // 2)),
            ("val", c, t, max(1, h // 2)),
            ("test", t, n, h),
        ]:
            idx = origins(lo, hi, h, step)
            p = parts[part]
            for stack, arr in zip(
                p,
                [
                    y[idx[:, None] + np.arange(-96, 0)],
                    y[idx[:, None] + np.arange(h)],
                    np.full(len(idx), str(sid)),
                    g.date.to_numpy()[idx - 1],
                ],
            ):
                stack.append(arr)
    return {
        k: [
            torch.from_numpy(np.concatenate(x)) if j < 2 else np.concatenate(x)
            for j, x in enumerate(v)
        ]
        for k, v in parts.items()
    }


def run(dataset, h, dest):
    torch.set_num_threads(1)
    d = prepare(dataset, h)
    x, y, *_ = d["train"]
    scores = []
    runs = {}
    history = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = PatchForecaster(h)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        best = float("inf")
        for epoch in range(1, 7):
            model.train()
            for i in torch.randperm(len(x)).split(128):
                yh = model(x[i])
                loss = (yh - y[i]).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(a) for a in d["meta"][0].split(256)])
                mae = (pred - d["meta"][1]).abs().mean().item()
            history.append([seed, epoch, mae])
            if mae < best:
                best = mae
                checkpoint = copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint)
        model.eval()
        for split in ["val", "test"]:
            with torch.no_grad():
                pred = torch.cat([model(a) for a in d[split][0].split(256)]).numpy()
            runs[seed, split] = pred
            scores.append(
                dict(split=split, seed=seed, **metrics(d[split][1].numpy(), pred))
            )
    dest.mkdir(exist_ok=True, parents=True)
    for split in ["val", "test"]:
        pred = np.mean([runs[seed, split] for seed in SEEDS], axis=0)
        scores.append(
            dict(split=split, seed="ensemble", **metrics(d[split][1].numpy(), pred))
        )
    np.savez_compressed(
        dest / "patch_forecasts.npz",
        series=d["test"][2],
        origin=d["test"][3],
        actual=d["test"][1].numpy(),
        ensemble=pred,
        **{str(seed): runs[seed, "test"] for seed in SEEDS},
    )
    pd.DataFrame(history, columns=["seed", "epoch", "meta_MAE"]).to_csv(
        dest / "patch_training.csv", index=False
    )
    pd.DataFrame(scores).to_csv(dest / "patch_scores.csv", index=False)
    print(dataset, h, scores[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ETTh", "ETTm"], required=True)
    p.add_argument("--horizon", type=int)
    p.add_argument("--output", type=type(ROOT), default=ROOT / "results/final")
    a = p.parse_args()
    h = a.horizon or (2 if a.dataset == "ETTh" else 30)
    run(a.dataset, h, a.output / "full" / f"{a.dataset}_h{h}")

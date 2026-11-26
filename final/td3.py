"""Đối chứng TD3 với sáu dự báo thành phần."""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from upstream.t2f_base import TD3
from .compare import assemble, metrics, ROOT, SEEDS
from .experts import NAMES


def train_policy(data, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = data["meta"]
    x = torch.from_numpy(m["ctx"])
    residual = torch.from_numpy(
        (m["pred"] - m["target"][:, :, None]) / m["scale"][:, None, None]
    )
    td = TD3(x.shape[1], len(NAMES), 1, logging.getLogger("td3"))
    # Kết thúc một lượt khi chuyển sang chuỗi khác.
    ns = np.minimum(np.arange(len(x)) + 1, len(x) - 1)
    done = (m["series"] != m["series"][ns]) | (ns == np.arange(len(x)))
    for step in range(400):
        idx = rng.integers(len(x), size=128)
        state = x[idx]
        successor = x[ns[idx]]
        with torch.no_grad():
            behavior = td.actor(state)
            action = (behavior + torch.randn_like(behavior) * 0.5).clamp(-1, 1)
            action[::2].uniform_(-1, 1)
            weights = torch.softmax(4 * action, dim=1)
            reward = (
                -(weights[:, None, :] * residual[idx])
                .sum(-1)
                .abs()
                .mean(-1, keepdim=True)
            )
            a2 = (
                td.actor_target(successor)
                + (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            ).clamp(-1, 1)
            q2 = torch.minimum(
                td.critic_1_target(successor, a2), td.critic_2_target(successor, a2)
            )
            target = reward + 0.9 * torch.from_numpy(~done[idx])[:, None] * q2
        for critic, opt in [
            (td.critic_1, td.critic_1_optimizer),
            (td.critic_2, td.critic_2_optimizer),
        ]:
            loss = F.smooth_l1_loss(critic(state, action), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 2 == 0:
            loss = -td.critic_1(state, td.actor(state)).mean()
            td.actor_optimizer.zero_grad()
            loss.backward()
            td.actor_optimizer.step()
            with torch.no_grad():
                for net, tgt in [
                    (td.actor, td.actor_target),
                    (td.critic_1, td.critic_1_target),
                    (td.critic_2, td.critic_2_target),
                ]:
                    for a, b in zip(net.parameters(), tgt.parameters()):
                        b.lerp_(a, 0.005)
    with torch.no_grad():
        return {
            part: torch.softmax(
                4 * td.actor(torch.from_numpy(data[part]["ctx"])), dim=1
            ).numpy()
            for part in ["val", "test"]
        }


def run(dataset, h, bridge, out):
    torch.set_num_threads(1)
    d, _ = assemble(dataset, h, bridge)
    out.mkdir(exist_ok=True, parents=True)
    ws = [train_policy(d, s) for s in SEEDS]
    scores = []
    saved = {}
    for part in ["val", "test"]:
        predictions = []
        for w, seed in zip(ws, SEEDS):
            pred = (w[part][:, None, :] * d[part]["pred"]).sum(-1)
            predictions.append(pred)
            scores.append(
                dict(split=part, seed=str(seed), **metrics(d[part]["target"], pred))
            )
            if part == "test":
                saved[str(seed)] = w[part]
        yhat = np.mean(predictions, axis=0)
        scores.append(
            dict(split=part, seed="ensemble", **metrics(d[part]["target"], yhat))
        )
        if part == "test":
            np.savez_compressed(
                out / "td3_weights.npz",
                names=NAMES,
                series=d[part]["series"],
                origin=d[part]["origin"],
                ensemble=yhat,
                **saved,
            )
    pd.DataFrame(scores).to_csv(out / "td3_scores.csv", index=False)
    print(dataset, h, bridge, scores[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ETTh", "ETTm", "ELD"], required=True)
    p.add_argument("--horizon", type=int)
    p.add_argument("--bridge", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/final")
    a = p.parse_args()
    h = a.horizon or {"ETTh": 2, "ETTm": 30, "ELD": 2}[a.dataset]
    run(
        a.dataset,
        h,
        a.bridge,
        a.output / ("bridge" if a.bridge else "full") / f"{a.dataset}_h{h}",
    )

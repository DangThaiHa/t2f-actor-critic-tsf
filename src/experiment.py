import hashlib
import json
import platform
import random
import time

import numpy as np
import pandas as pd
import torch

from .data import build_training_set, load_series
from .evaluation import rolling_forecast, save_results
from .model import RidgeVAR, core, logger


def run_experiment(dataset, target_mode, seed, profile, out, data_dir):
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    groups, times, target, weight, features, horizon, global_len, period = load_series(
        dataset, data_dir
    )
    path = data_dir / f"{dataset.lower()}.parquet"
    n = len(times)
    val_start, test_start = int(0.70 * n), int(0.80 * n)
    rows, epochs, origins = {
        "quick": (128, 3, 12),
        "main": (512, 8, 48),
        "extended": (1024, 15, 96),
    }[profile]
    assert (
        val_start > rows
        and n - test_start > horizon
        and test_start - val_start > horizon
    ), (dataset, n, rows)
    train, means, scales, level_scales = build_training_set(
        groups, target, val_start, rows, target_mode
    )
    model = core.T2f(
        len(features),
        1,
        16,
        1,
        horizon,
        global_len,
        list(groups),
        freq=None,
        logger=logger,
        batch_size=16,
        timestamp_col="date",
        series_id_col="station",
        epochs=epochs,
    )
    if target_mode == "delta_ridge":
        model.local_models[0] = RidgeVAR()
    before = [p.detach().clone() for p in model.global_model.actor.parameters()]
    model.fit(train[features], train[[target]], train[[weight]])
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model.global_model.actor.parameters())
    )
    splits = [("val", val_start, test_start), ("test", test_start, n)]
    data = rolling_forecast(
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
    )
    alpha = save_results(data, out)
    info = dict(
        target_mode=target_mode,
        level_scales=level_scales,
        dataset=dataset,
        seed=seed,
        profile=profile,
        epochs=epochs,
        train_rows_per_series=rows,
        common_regular_steps=n,
        stations=list(groups),
        train_start=str(times[val_start - rows]),
        train_end=str(times[val_start - 1]),
        val_start=str(times[val_start]),
        test_start=str(times[test_start]),
        test_end=str(times[-1]),
        target=target,
        weight=weight,
        horizon=horizon,
        global_length=global_len,
        features=features,
        alpha=float(alpha),
        target_means=means,
        target_scales=scales,
        actor_changed=True,
        seconds=time.monotonic() - started,
        parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        pandas=pd.__version__,
        python=platform.python_version(),
    )
    (out / "run.json").write_text(json.dumps(info, indent=2))

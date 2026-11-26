"""Huấn luyện các mô hình dự báo riêng cho từng chuỗi."""

import numpy as np

NAMES = ["last", "daily", "weekly", "nlinear", "dlinear", "ridge_calendar"]
LOOKBACK = 96


def moving_average(history, kernel=25):
    padded = np.pad(history, ((0, 0), (kernel // 2, kernel // 2)), mode="edge")
    sums = np.cumsum(np.pad(padded, ((0, 0), (1, 0))), axis=1)
    return (sums[:, kernel:] - sums[:, :-kernel]) / kernel


def design(values, times, origins, horizon, period):
    y = values[:, 0]
    history = y[origins[:, None] + np.arange(-LOOKBACK, 0)]
    anchor = history[:, -1:]
    offsets = np.arange(horizon)
    daily = y[origins[:, None] - period + offsets % period]
    weekly = y[origins[:, None] - 7 * period + offsets % (7 * period)]
    trend = moving_average(history)
    calendar = times[origins]
    hour = calendar.hour.to_numpy() + calendar.minute.to_numpy() / 60
    day = calendar.dayofweek.to_numpy()
    angle = np.column_stack(
        [
            2 * np.pi * hour / 24,
            4 * np.pi * hour / 24,
            2 * np.pi * day / 7,
            4 * np.pi * day / 7,
        ]
    )
    lags = [1, 2, 3, 4, period, 2 * period, 7 * period]
    context = np.column_stack([values[origins - k] for k in lags])
    ridge = np.column_stack(
        [
            history - anchor,
            context,
            np.sin(angle),
            np.cos(angle),
            day >= 5,
            daily[:, [0, -1]],
            weekly[:, [0, -1]],
        ]
    )
    matrices = [
        (history - anchor)[:, :-1],
        np.column_stack([trend, history - trend]),
        ridge,
    ]
    return matrices, anchor, daily, weekly


def fit_experts(values, times, stop, horizon, period, budget=None):
    first = max(LOOKBACK, 7 * period)
    if budget:
        first = max(first, stop - budget)
    origins = np.arange(first, stop - horizon + 1)
    assert len(origins) > 0
    moments = []
    for start in range(0, len(origins), 1024):
        idx = origins[start : start + 1024]
        xs, anchor, _, _ = design(values, times, idx, horizon, period)
        target = values[idx[:, None] + np.arange(horizon), 0]
        for j, (x, base) in enumerate(zip(xs, [anchor, 0, anchor])):
            x = np.column_stack([np.ones(len(x)), x])
            yy = target - base
            if start == 0:
                moments.append([x.T @ x, x.T @ yy])
            else:
                moments[j][0] += x.T @ x
                moments[j][1] += x.T @ yy
    models = []
    for j, (gram, cross) in enumerate(moments):
        # Điều chỉnh mức phạt theo độ phân tán của từng đặc trưng.
        variances = np.maximum(np.diag(gram) - gram[:, 0] ** 2 / gram[0, 0], 1.0e-6)
        penalty = [1.0e-6, 1.0e-6, 0.01][j] * variances
        penalty[0] = 0
        models.append(np.linalg.lstsq(gram + np.diag(penalty), cross, rcond=1.0e-12)[0])
    return models, len(origins)


def predict_experts(models, values, times, idx, horizon, period, scale):
    xs, anchor, daily, weekly = design(values, times, idx, horizon, period)
    pred = [np.broadcast_to(anchor, daily.shape), daily, weekly]
    for x, b, base in zip(xs, models, [anchor, 0, anchor]):
        pred.append(b[0] + x @ b[1:] + base)
    forecasts = np.stack(pred, axis=-1)
    features = np.column_stack(
        [
            (anchor[:, 0] - values[idx - 2, 0]) / scale,
            (daily[:, 0] - anchor[:, 0]) / scale,
            (weekly[:, 0] - anchor[:, 0]) / scale,
            ((forecasts - anchor[:, :, None]) / scale).mean(axis=1),
            np.sin(2 * np.pi * times[idx].hour / 24),
            np.cos(2 * np.pi * times[idx].hour / 24),
            times[idx].dayofweek >= 5,
        ]
    )
    return forecasts.astype("float32"), np.clip(features, -10, 10).astype("float32")

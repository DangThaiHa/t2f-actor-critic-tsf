import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "original_code"))
sys.path.insert(0, str(ROOT / "original_code" / "t2f"))

from t2f.t2f_base import data_load, t2f_fit


def main() -> None:
    print("=== T2F demo reproduction ===")
    dataset = "eld"
    local_length = 2
    global_length = 3
    epochs = 1
    predict_rows = 64

    input_path = ROOT / "datasets" / "eld" / "diff_eld.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {input_path}. "
            "Please run setup_project.py or extract the archive first."
        )

    start = time.time()
    print("Loading dataset...")
    train_df, test_df, prepared_df, freq = data_load(
        str(input_path),
        "id",
        "timestamp",
        "%Y-%m-%d %H:%M:%S",
        ["tmax", "tmin", "tmed"],
        "cost_kwh",
        "load",
        0.2,
        "15min",
        id_frac=0.02,
        len_frac=0.02,
    )

    print(f"Data shape: train={train_df.shape}, test={test_df.shape}, freq={freq}")

    model = t2f_fit(
        train_df=train_df,
        test_df=test_df,
        prepared_df=prepared_df,
        freq=freq,
        series_id_col="id",
        timestamp_col="timestamp",
        covariate_cols=["tmax", "tmin", "tmed"],
        weight_col="cost_kwh",
        target_col="load",
        local_length=local_length,
        global_length=global_length,
        epochs=epochs,
        model_save_path=str(ROOT / "results" / "demo_model" / f"ep{epochs}"),
        hidden_size=32,
        num_layers=1,
        batch_size=8,
    )

    X_test = test_df[["tmax", "tmin", "tmed"]].head(predict_rows)
    pred = model.predict(X_test)
    pred_df = pred.reset_index()
    pred_df = pred_df[pred_df["timestamp"].notna()].copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df = pred_df.sort_values("timestamp")

    prediction_preview = pred_df.head(10).reset_index().copy()
    if "timestamp" in prediction_preview.columns:
        prediction_preview["timestamp"] = prediction_preview["timestamp"].map(
            lambda ts: ts.isoformat() if pd.notna(ts) else None
        )

    result_payload = {
        "dataset": dataset,
        "freq": freq,
        "epochs": epochs,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "local_length": local_length,
        "global_length": global_length,
        "hidden_size": 32,
        "num_layers": 1,
        "batch_size": 8,
        "predict_rows": predict_rows,
        "model_save_path": str(ROOT / "results" / "demo_model" / f"ep{epochs}"),
        "prediction_shape": list(pred.shape),
        "prediction_head": prediction_preview.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    json_path = RESULTS_DIR / "eld_demo_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, ensure_ascii=False, indent=2)

    if not pred_df.empty and pred_df["timestamp"].notna().any():
        plot_source = pred_df.copy()
        plot_source["timestamp"] = pd.to_datetime(plot_source["timestamp"])
        plot_source = plot_source.sort_values("timestamp")
        y_values = plot_source["prediction"]
        label = "T2f prediction"
        title = "T2f Demo Prediction on ELD subset"
    else:
        plot_source = test_df.reset_index()[["timestamp", "load"]].head(64).copy()
        plot_source["timestamp"] = pd.to_datetime(plot_source["timestamp"])
        plot_source = plot_source.sort_values("timestamp")
        y_values = plot_source["load"]
        label = "Observed load"
        title = "Observed load on ELD subset"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_source["timestamp"], y_values, color="royalblue", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = RESULTS_DIR / "eld_demo_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved metrics: {json_path}")
    print(f"Saved plot: {plot_path}")

    print("=== Finished ===")


if __name__ == "__main__":
    main()

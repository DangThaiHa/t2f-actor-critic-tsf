"""Chuẩn bị dữ liệu ETT và phụ tải ELD theo khung 6 giờ."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eld-zip", type=Path)
    args = parser.parse_args()
    out = ROOT / "data/final"
    out.mkdir(exist_ok=True, parents=True)
    sources = {}
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
        path = out / f"{name}.csv"
        if not path.exists():
            urlretrieve(url, path)
        sources[name] = {
            "url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    path = args.eld_zip or out / "uci.zip"
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    if not path.exists():
        urlretrieve(url, path)
    with ZipFile(path) as archive, archive.open("LD2011_2014.txt") as f:
        df = pd.read_csv(f, sep=";", decimal=",", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index)
    assert df.shape == (140256, 370) and df.index.is_unique
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert not df.isna().any().any()
    groups = df.resample("6h", closed="right", label="right")
    assert (groups.size() == 24).all()
    panel = groups.mean().astype("float32")
    panel.to_parquet(out / "eld_6h.parquet")
    sources["ELD"] = {
        "url": url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_shape": df.shape,
        "panel_shape": panel.shape,
    }
    (out / "sources.json").write_text(json.dumps(sources, indent=2))

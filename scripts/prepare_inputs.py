"""Restore the experiment inputs from the authors' Zenodo release."""

import argparse
from pathlib import Path
import tempfile
from zipfile import ZipFile
import py7zr

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
args = parser.parse_args()
(root / "data").mkdir(exist_ok=True)
(root / "upstream").mkdir(exist_ok=True)
with ZipFile(args.archive) as release:
    base = next(
        n.removesuffix("t2f/t2f_base.py")
        for n in release.namelist()
        if n.endswith("/t2f/t2f_base.py")
    )
    for name in ["helpers.py", "t2f_base.py"]:
        code = release.read(base + "t2f/" + name).decode()
        if name == "t2f_base.py":
            code = code.replace("from helpers import", "from .helpers import")
            for key in ["state", "action", "next_state", "reward", "done"]:
                code = code.replace(
                    f"np.array({key}, copy=False)", f"np.asarray({key})"
                )
        (root / "upstream" / name).write_text(code.strip() + "\n")
    (root / "upstream/__init__.py").touch()
    for dataset in ["ETTh", "ETTm", "eld"]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.7z"
            path.write_bytes(
                release.read(f"{base}datasets/{dataset.lower()}/{dataset}.7z")
            )
            with py7zr.SevenZipFile(path) as data:
                data.extract(path=tmp, targets=[f"{dataset}.parquet"])
            (root / "data" / f"{dataset.lower()}.parquet").write_bytes(
                (Path(tmp) / f"{dataset}.parquet").read_bytes()
            )

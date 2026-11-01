from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
DATASET_DIR = ROOT / "datasets" / "eld"
ARCHIVE = DATASET_DIR / "eld.7z"
DATASET_FILE = DATASET_DIR / "diff_eld.parquet"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_python_311() -> str:
    candidates = [
        Path(sys.executable),
        Path(r"C:\Users\Jenny\AppData\Local\Programs\Python\Python311\python.exe"),
        Path(r"C:\Program Files\Python311\python.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            version = subprocess.run([str(candidate), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture_output=True, text=True, check=True).stdout.strip()
            if version == "3.11":
                return str(candidate)
    raise RuntimeError("Python 3.11 is required. Please install Python 3.11 and retry.")


def ensure_venv(python_path: str) -> str:
    if not VENV_DIR.exists():
        run([python_path, "-m", "venv", str(VENV_DIR)])
    venv_python = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        raise RuntimeError(f"Virtual environment Python not found at {venv_python}")
    return str(venv_python)


def install_requirements(venv_python: str) -> None:
    run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    run([venv_python, "-m", "pip", "install", "-r", str(REQ_FILE)])


def prepare_eld_dataset() -> None:
    if DATASET_FILE.exists():
        print("ELD parquet file already exists.")
        return

    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE}")

    try:
        import py7zr
    except Exception as exc:
        raise RuntimeError("py7zr is needed to unpack the ELD archive. Install requirements first.") from exc

    with py7zr.SevenZipFile(str(ARCHIVE), "r") as z:
        z.extractall(str(DATASET_DIR))

    parquet_candidates = list(DATASET_DIR.glob("*.parquet"))
    if not parquet_candidates:
        raise FileNotFoundError(f"No parquet file found after extraction in {DATASET_DIR}")
    src = parquet_candidates[0]
    shutil.copy2(src, DATASET_FILE)
    print(f"Prepared dataset: {DATASET_FILE}")


def main() -> None:
    print("== T2F environment bootstrap ==")
    python_path = ensure_python_311()
    print(f"Using Python: {python_path}")
    venv_python = ensure_venv(python_path)
    install_requirements(venv_python)
    prepare_eld_dataset()
    print("\nEnvironment is ready. Activate it with:")
    if os.name == "nt":
        print("  .\\.venv\\Scripts\\Activate.ps1")
    else:
        print("  source .venv/bin/activate")
    print("Then run the notebook from the repository root.")


if __name__ == "__main__":
    main()

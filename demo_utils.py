"""Small utilities shared by the reproducibility demos."""
from pathlib import Path


RESULTS_DIR = Path("results")


def result_path(filename):
    """Return a path under results/, creating the directory on demand."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename

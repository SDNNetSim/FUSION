from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "gnn_cached"
CACHE_DIR.mkdir(exist_ok=True)

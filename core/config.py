from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
INDEX_PATH = DATA_DIR / "index.json"

CHUNK_SIZE = 1024 * 1024  # 1 MiB for streaming

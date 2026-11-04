"""Create encoded chronological train/validation/test parquet files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_interactions, save_split


if __name__ == "__main__":
    save_split(load_interactions())


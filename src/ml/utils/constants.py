import os
from pathlib import Path

# project directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent
# data constants
DATA_DIR = os.path.join(ROOT_DIR, ".data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# embeddings constants
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, ".embeddings")
# experiment constants
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, ".experiments")

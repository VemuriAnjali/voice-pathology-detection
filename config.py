"""Project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = 42
EMBED_DIM = 128
BATCH_SIZE = 32
ENCODER_EPOCHS = 120
CONV_HSA_EPOCHS = 150
TEST_SIZE = 0.20

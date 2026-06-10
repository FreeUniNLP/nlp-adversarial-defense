"""
Configuration template for DagsHub team logging.

Copy this file to config.py and fill in your DagsHub credentials.
The actual config.py is ignored by git to prevent accidental token leaks.

Usage:
  1. cp config.example.py config.py
  2. Edit config.py and set your DagsHub credentials
  3. Run training: python scripts/train/train_model.py --mlflow
"""

# ================================================================
#  DagsHub Configuration (Team Logging)
# ================================================================

# Your DagsHub username / organization name
# Get from: https://dagshub.com/user/settings
DAGSHUB_REPO_OWNER = "your-dagshub-username"

# Repository name on DagsHub
DAGSHUB_REPO_NAME = "nlp-adversarial-defense"

# Your DagsHub API token (get from: https://dagshub.com/user/settings/tokens)
# WARNING: Keep this private! This file is ignored by git.
DAGSHUB_TOKEN = "your-dagshub-api-token"

# ================================================================
#  Optional: Feature Flags
# ================================================================

# Enable DagsHub logging by default (can still override with --mlflow flag)
USE_MLFLOW_BY_DEFAULT = False

# ================================================================
#  Project Paths (Auto-detected, no need to change)
# ================================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CORPUS_DIR = DATA_DIR / "raw" / "generated_texts"

__all__ = [
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TOKEN",
    "USE_MLFLOW_BY_DEFAULT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "CORPUS_DIR",
]


"""
Configuration file for DagsHub team logging.

This file stores your DagsHub credentials so you don't have to set environment variables
every time you run the training script.

For team collaboration: commit this file with repo name, but keep token private!
"""

# ================================================================
#  DagsHub Configuration (Team Logging)
# ================================================================

# Your DagsHub username / organization name
DAGSHUB_REPO_OWNER = "kende23"

# Repository name on DagsHub
DAGSHUB_REPO_NAME = "nlp-adversarial-defense"

# Your DagsHub API token (get from: https://dagshub.com/user/settings/tokens)
# WARNING: Keep this private! Add config.py to .gitignore if you commit a real token here.
DAGSHUB_TOKEN = "0305c6a2b20de4d5f1f1b16f1825405cfa52cac8"

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


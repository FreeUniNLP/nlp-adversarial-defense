"""
Shared fixtures for the test suite.
Loaded automatically by pytest from any test file in this folder.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
CORPUS_PATH     = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"
CKPT_PATH       = PROJECT_ROOT / "data" / "models" / "minigpt_corpus10000.pt"


def load_lexicon():
    from src.language.parsers import LexiconParser
    return LexiconParser.parse(WORDS_PATH)


def build_cfg(nouns, verbs, adjectives):
    from src.language.entities.cfg import CFG
    return CFG.from_json(
        file_path=str(TRANSITION_PATH),
        nouns=nouns, verbs=verbs, adjectives=adjectives,
    )


def build_validator(cfg):
    from src.language.entities.cfg_validator import CFGValidator
    return CFGValidator.from_cfg(cfg)

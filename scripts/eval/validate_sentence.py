"""Validate one (or more) complete sentences against the CFG grammar.

No model is involved — this just runs the sentence through CFGValidator and
reports whether it is grammatically valid, plus the error if it is not.

Usage:
    python scripts/eval/validate_sentence.py "FREE WOLF FALL"
    python scripts/eval/validate_sentence.py "FREE WOLF FALL" "BROKEN SENTENCE"
    python scripts/eval/validate_sentence.py            # interactive mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"


def build_validator() -> CFGValidator:
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        file_path=str(TRANSITION_PATH),
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives,
    )
    return CFGValidator.from_cfg(cfg)


def report(validator: CFGValidator, sentence: str) -> bool:
    sentence = sentence.strip().upper()
    result = validator.validate(sentence)
    if result.is_valid:
        print(f"[VALID]   {sentence}")
        return True
    print(f"[INVALID] {sentence}  -> {result.error}")
    return False


def interactive(validator: CFGValidator) -> None:
    print("Enter a sentence to validate (blank line or Ctrl-C to quit).")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            report(validator, line)
    except (EOFError, KeyboardInterrupt):
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate complete sentences against the CFG grammar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python scripts/eval/validate_sentence.py "FREE WOLF FALL"\n'
            "  python scripts/eval/validate_sentence.py            # interactive mode"
        ),
    )
    p.add_argument("sentences", nargs="*", help="One or more sentences to validate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validator = build_validator()

    if not args.sentences:
        interactive(validator)
        return

    all_valid = all(report(validator, s) for s in args.sentences)
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

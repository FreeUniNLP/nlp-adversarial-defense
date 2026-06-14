"""
Run the attacker — generate CFG-valid prefixes and display them.

Usage:
    python scripts/run_attacker.py
    python scripts/run_attacker.py --n 10
    python scripts/run_attacker.py --max-tokens 5 --temperature 0.5
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.parsers import LexiconParser
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.attacker.attacker import AttackerTransformer
from src.model.tokenizer import WordTokenizer

WORDS_PATH  = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Generate prefixes with the attacker")
    p.add_argument("--n",           type=int,   default=5,   help="Number of prefixes to generate (default: 5)")
    p.add_argument("--max-tokens",  type=int,   default=8,   dest="max_tokens", help="Max prefix length (default: 8)")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading lexicon and tokenizer...")
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)
    tracker   = CFGStateTracker(nouns, verbs, adjectives)
    attacker  = AttackerTransformer(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)

    print(f"\nGenerating {args.n} prefixes  |  max_tokens={args.max_tokens}  |  temperature={args.temperature}")
    print("=" * 60)

    for i in range(1, args.n + 1):
        ids, words = attacker.generate_prefix(
            bos_id      = tokenizer.bos_id,
            eos_id      = tokenizer.eos_id,
            cfg_tracker = tracker,
            token_to_id = tokenizer.token_to_id,
            id_to_token = tokenizer.id_to_token,
            max_tokens  = args.max_tokens,
            temperature = args.temperature,
        )
        prefix = " ".join(words) if words else "(empty)"
        print(f"  {i:2d}.  {prefix}")

    print("=" * 60)


if __name__ == "__main__":
    main()

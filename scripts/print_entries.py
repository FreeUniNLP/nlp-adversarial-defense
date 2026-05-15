#!/usr/bin/env python3
"""Simple CLI to print vocabulary and transition entries.

Usage examples:
  # Print counts only
  python3 scripts/print_entries.py --counts-only

  # Print full entries (may be large)
  python3 scripts/print_entries.py

  # Provide custom JSON paths
  python3 scripts/print_entries.py --words-path data/raw/words.json --transitions-path data/raw/transition.json
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be imported when this
# script is executed from the repository root or from inside the scripts/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Print vocabulary and transition entries")
    parser.add_argument("--words-path", type=Path, default=None, help="Path to words.json")
    parser.add_argument("--transitions-path", type=Path, default=None, help="Path to transition.json")
    parser.add_argument("--counts-only", action="store_true", help="Only print counts for each section")
    args = parser.parse_args()

    # Import here so the script remains lightweight until executed
    from src.language import load_vocabulary, load_transitions, print_all_entries

    if args.counts_only:
        words = load_vocabulary(args.words_path)
        pos_entries, cat1_entries, cat2_entries = load_transitions(args.transitions_path)

        print("WORD ENTRIES:", len(words))
        print("POS LEVEL ILLEGAL TRANSITIONS:", len(pos_entries))
        print("CATEGORY1 ILLEGAL TRANSITIONS:", len(cat1_entries))
        print("CATEGORY2 ILLEGAL TRANSITIONS:", len(cat2_entries))
    else:
        # Delegates to the package-level printer which prints everything
        print_all_entries(args.words_path, args.transitions_path)


if __name__ == "__main__":
    main()


"""Complete a sentence prefix with MiniGPT and validate the result with CFGValidator.

Usage:
    # Single prefix
    python scripts/complete_and_validate.py "MAN"

    # Interactive REPL
    python scripts/complete_and_validate.py

    # Multiple completions per prefix (to see variety)
    python scripts/complete_and_validate.py "MAN" --n 5

    # Adjust generation
    python scripts/complete_and_validate.py "FAST WOLF" --temperature 0.5 --max-tokens 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT
from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from scripts.eval.infer import complete_sentence

MODELS_DIR      = PROJECT_ROOT / "data" / "models"
CORPUS_DIR      = PROJECT_ROOT / "data" / "raw" / "generated_texts"
WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"


# ------------------------------------------------------------------ #
#  Setup                                                               #
# ------------------------------------------------------------------ #

def load_model(corpus_size: int, device: str = "cpu") -> tuple[MiniGPT, WordTokenizer]:
    corpus_path = CORPUS_DIR / f"generated_corpus_{corpus_size}.txt"
    ckpt_path   = MODELS_DIR / f"minigpt_corpus{corpus_size}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with: python scripts/train/train_model.py --corpus {corpus_size}"
        )
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    tokenizer = WordTokenizer.from_corpus(corpus_path)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MiniGPT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[OK] Model   : {ckpt_path.name}  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    print(f"[OK] Vocab   : {tokenizer.vocab_size} tokens")
    print(f"[OK] Device  : {device}")
    return model, tokenizer


def build_validator() -> CFGValidator:
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        file_path=str(TRANSITION_PATH),
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives,
    )
    return CFGValidator.from_cfg(cfg)


# ------------------------------------------------------------------ #
#  Core logic                                                          #
# ------------------------------------------------------------------ #

def run_once(
    prefix: str,
    model: MiniGPT,
    tokenizer: WordTokenizer,
    validator: CFGValidator,
    n: int,
    max_tokens: int,
    temperature: float,
    device: str,
) -> None:
    """Complete `prefix` n times, validate each, and print a summary."""
    print(f"\nPrefix  : {prefix.upper()}")
    print(f"Samples : {n}  |  temperature={temperature}  |  max_tokens={max_tokens}")
    print("-" * 60)

    valid_count   = 0
    invalid_count = 0

    for i in range(1, n + 1):
        sentence = complete_sentence(model, tokenizer, prefix.upper(), max_tokens, temperature, device)
        if sentence is None:
            break

        result = validator.validate(sentence)

        if result.is_valid:
            valid_count += 1
            status = "[VALID]  "
            detail = ""
        else:
            invalid_count += 1
            status = "[INVALID]"
            detail = f"  -> {result.error}"

        print(f"  {i:2d}. {status}  {sentence}{detail}")

    total = valid_count + invalid_count
    if total > 0:
        pct = 100 * valid_count / total
        print("-" * 60)
        print(f"  Result: {valid_count}/{total} valid  ({pct:.0f}%)")


# ------------------------------------------------------------------ #
#  Modes                                                               #
# ------------------------------------------------------------------ #

def single_mode(args: argparse.Namespace, model, tokenizer, validator) -> None:
    run_once(
        prefix      = args.prefix,
        model       = model,
        tokenizer   = tokenizer,
        validator   = validator,
        n           = args.n,
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
        device      = "cpu",
    )


def interactive_mode(args: argparse.Namespace, model, tokenizer, validator) -> None:
    print("\n" + "=" * 60)
    print("  Complete & Validate — Interactive Mode")
    print("=" * 60)
    print("  Enter a sentence prefix to complete and validate.")
    print("  Commands:")
    print("    temp <value>   change temperature (current: {:.1f})".format(args.temperature))
    print("    n <value>      change completions per prefix (current: {:d})".format(args.n))
    print("    quit           exit")
    print("=" * 60)

    temperature = args.temperature
    n           = args.n

    while True:
        try:
            raw = input("\nPrefix> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Commands
        if raw.lower().startswith("temp "):
            try:
                temperature = float(raw.split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: temp <float>  e.g. temp 0.5")
            continue

        if raw.lower().startswith("n "):
            try:
                n = int(raw.split()[1])
                print(f"  Completions per prefix set to {n}")
            except (IndexError, ValueError):
                print("  Usage: n <int>  e.g. n 5")
            continue

        run_once(
            prefix      = raw,
            model       = model,
            tokenizer   = tokenizer,
            validator   = validator,
            n           = n,
            max_tokens  = args.max_tokens,
            temperature = temperature,
            device      = "cpu",
        )


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete a sentence prefix with MiniGPT and validate with CFGValidator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/complete_and_validate.py "MAN"
  python scripts/complete_and_validate.py "FAST WOLF" --n 5 --temperature 0.5
  python scripts/complete_and_validate.py              # interactive mode
        """,
    )
    p.add_argument(
        "prefix", nargs="?", default=None,
        help="Sentence prefix to complete (omit for interactive mode)"
    )
    p.add_argument(
        "--corpus", type=int, default=10000, choices=[100, 500, 1000, 5000, 10000],
        help="Corpus size the model was trained on (default: 10000)"
    )
    p.add_argument(
        "--n", type=int, default=3,
        help="Number of completions to generate per prefix (default: 3)"
    )
    p.add_argument(
        "--max-tokens", type=int, default=15, dest="max_tokens",
        help="Maximum new tokens to generate (default: 15)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature: lower = more deterministic (default: 0.8)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    model, tokenizer = load_model(args.corpus)
    validator = build_validator()
    print()

    if args.prefix:
        single_mode(args, model, tokenizer, validator)
    else:
        interactive_mode(args, model, tokenizer, validator)

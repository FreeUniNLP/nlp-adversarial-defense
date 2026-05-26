"""Evaluate a trained MiniGPT by generating sentences and validating them with CFGValidator.

Usage:
    python scripts/evaluate_model.py                        # uses default checkpoint
    python scripts/evaluate_model.py --corpus 10000         # which corpus the model was trained on
    python scripts/evaluate_model.py --samples 200          # how many sentences to generate
    python scripts/evaluate_model.py --temperature 0.8      # sampling temperature
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT
from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"


def load_model(corpus_size: int, device: str) -> tuple[MiniGPT, WordTokenizer]:
    corpus_path = PROJECT_ROOT / "data" / "raw" / "generated_texts" / f"generated_corpus_{corpus_size}.txt"
    ckpt_path   = PROJECT_ROOT / "data" / "models" / f"minigpt_corpus{corpus_size}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. "
            f"Train first with: python scripts/train_model.py --corpus {corpus_size}"
        )

    tokenizer = WordTokenizer.from_corpus(corpus_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MiniGPT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return model, tokenizer


def build_validator() -> CFGValidator:
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        file_path=TRANSITION_PATH, nouns=nouns, verbs=verbs, adjectives=adjectives
    )
    return CFGValidator.from_cfg(cfg)


def evaluate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.corpus, device)
    validator = build_validator()

    print(f"Generating {args.samples} sentences (temperature={args.temperature})...\n")

    valid_sentences   = []
    invalid_sentences = []

    for _ in range(args.samples):
        ids      = model.generate(tokenizer.bos_id, tokenizer.eos_id,
                                  max_new_tokens=30, temperature=args.temperature, device=device)
        sentence = tokenizer.decode(ids)
        result   = validator.validate(sentence)

        if result.is_valid:
            valid_sentences.append(sentence)
        else:
            invalid_sentences.append((sentence, result.error))

    total      = len(valid_sentences) + len(invalid_sentences)
    valid_pct  = 100 * len(valid_sentences) / total if total else 0

    print(f"{'='*60}")
    print(f"  Valid:   {len(valid_sentences)}/{total}  ({valid_pct:.1f}%)")
    print(f"  Invalid: {len(invalid_sentences)}/{total}  ({100-valid_pct:.1f}%)")
    print(f"{'='*60}\n")

    n = args.show
    if valid_sentences:
        print(f"--- Valid samples (first {min(n, len(valid_sentences))}) ---")
        for s in valid_sentences[:n]:
            print(f"  {s}")

    if invalid_sentences:
        print(f"\n--- Invalid samples (first {min(n, len(invalid_sentences))}) ---")
        for s, err in invalid_sentences[:n]:
            print(f"  {s}")
            print(f"    reason: {err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MiniGPT sentence validity")
    p.add_argument("--corpus",      type=int,   default=10000,
                   choices=[100, 500, 1000, 5000, 10000])
    p.add_argument("--samples",     type=int,   default=200,
                   help="Number of sentences to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--show",        type=int,   default=10,
                   help="How many example sentences to print per category")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

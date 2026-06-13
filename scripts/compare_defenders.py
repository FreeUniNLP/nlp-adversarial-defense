"""Compare two defender (MiniGPT) checkpoints on free text generation quality.

Generates N sentences from each defender starting from BOS (no attacker prefix),
validates every sentence with the CFG validator, and prints a side-by-side
report showing grammar validity, sentence length, and diversity.

The key claim of adversarial training is that it should push the defender to
generate more grammatically valid completions. This script measures that.

Usage:
    python scripts/compare_defenders.py \\
        --baseline data/models/from_mlflow/MiniGPT/nosy-chimp-161/minigpt_corpus10000.pt \\
        --trained  data/models/cotrain/defender_round_final.pt

    python scripts/compare_defenders.py \\
        --baseline data/models/from_mlflow/MiniGPT/nosy-chimp-161/minigpt_corpus10000.pt \\
        --trained  data/models/cotrain/defender_round_final.pt \\
        --n 500 --min-tokens 5 --temperature 0.9
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacker.cfg_state_tracker import CFGStateTracker
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from src.language.parsers import LexiconParser
from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
DEFAULT_CORPUS  = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"


# ------------------------------------------------------------------ #
#  Model loading                                                       #
# ------------------------------------------------------------------ #

def load_defender(ckpt_path: Path, tokenizer: WordTokenizer, device: torch.device) -> MiniGPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MiniGPT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    # Support both raw model-state dicts and wrapped checkpoints
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------------ #
#  Generation                                                          #
# ------------------------------------------------------------------ #

def generate_sentence(
    model:          MiniGPT,
    tokenizer:      WordTokenizer,
    device:         torch.device,
    temperature:    float,
    max_tokens:     int,
    min_tokens:     int,
    prefix_ids:     list[int] | None = None,
) -> str:
    """Generate one sentence. If prefix_ids is None, starts from BOS only."""
    all_ids = [tokenizer.bos_id] + (prefix_ids or [])
    new_count = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([all_ids[-model.context_len:]], device=device)
            logits  = model(context)[:, -1, :] / temperature
            if new_count < min_tokens:
                logits[0, tokenizer.eos_id] = float("-inf")
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == tokenizer.eos_id:
                break
            all_ids.append(next_id)
            new_count += 1

    return tokenizer.decode(all_ids[1:])  # strip BOS


def generate_batch(
    model:       MiniGPT,
    tokenizer:   WordTokenizer,
    validator:   CFGValidator,
    tracker:     CFGStateTracker,
    device:      torch.device,
    n:           int,
    temperature: float,
    max_tokens:  int,
    min_tokens:  int,
    use_prefix:  bool,
) -> list[dict]:
    """Generate n sentences and return list of result dicts."""
    results = []
    for _ in range(n):
        prefix_ids = None
        if use_prefix:
            # Random 1-3 token CFG-valid prefix for more varied starts
            tracker.reset()
            ids, words = [], []
            for _ in range(3):
                valid, can_end = tracker.valid_next_words()
                if can_end and (not valid or len(ids) >= 1):
                    break
                if not valid:
                    break
                import random
                w = random.choice(valid)
                tracker.step(w)
                words.append(w)
                ids.append(tokenizer.token_to_id[w])
            prefix_ids = ids if ids else None

        sent = generate_sentence(
            model, tokenizer, device, temperature, max_tokens, min_tokens, prefix_ids)
        res  = validator.validate(sent)
        words = sent.split()
        results.append({
            "sentence": sent,
            "valid":    res.is_valid,
            "length":   len(words),
            "error":    res.error if not res.is_valid else None,
        })
    return results


# ------------------------------------------------------------------ #
#  Stats + reporting                                                   #
# ------------------------------------------------------------------ #

def compute_stats(results: list[dict]) -> dict:
    n          = len(results)
    n_valid    = sum(r["valid"] for r in results)
    lengths    = [r["length"] for r in results]
    sentences  = [r["sentence"] for r in results]
    unique     = len(set(sentences))
    error_ctr  = Counter(r["error"] for r in results if not r["valid"])
    return {
        "n":            n,
        "valid":        n_valid,
        "valid_pct":    n_valid / n * 100,
        "avg_len":      sum(lengths) / n,
        "min_len":      min(lengths),
        "max_len":      max(lengths),
        "unique":       unique,
        "unique_pct":   unique / n * 100,
        "top_errors":   error_ctr.most_common(3),
    }


def print_report(
    baseline_results: list[dict],
    trained_results:  list[dict],
    baseline_label:   str,
    trained_label:    str,
    n_samples:        int,
) -> None:
    bs = compute_stats(baseline_results)
    ts = compute_stats(trained_results)

    def delta(b, t, higher_is_better=True):
        d = t - b
        sign = "+" if d >= 0 else ""
        arrow = ("↑" if d > 0 else "↓") if d != 0 else "="
        if higher_is_better:
            tag = " ✓" if d > 0 else (" ✗" if d < 0 else "")
        else:
            tag = " ✓" if d < 0 else (" ✗" if d > 0 else "")
        return f"{sign}{d:.1f}{tag}"

    w = 30
    print()
    print("=" * 70)
    print("  DEFENDER COMPARISON REPORT")
    print("=" * 70)
    print(f"  {'Metric':<22}  {'Baseline':>14}  {'Trained':>14}  {'Delta':>10}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*10}")
    print(f"  {'Sentences generated':<22}  {bs['n']:>14}  {ts['n']:>14}")
    print(f"  {'Valid (%)':<22}  {bs['valid_pct']:>13.1f}%  {ts['valid_pct']:>13.1f}%  {delta(bs['valid_pct'], ts['valid_pct']):>10}")
    print(f"  {'Avg length (words)':<22}  {bs['avg_len']:>14.1f}  {ts['avg_len']:>14.1f}  {delta(bs['avg_len'], ts['avg_len']):>10}")
    print(f"  {'Min / Max length':<22}  {bs['min_len']:>6}/{bs['max_len']:<7}  {ts['min_len']:>6}/{ts['max_len']:<7}")
    print(f"  {'Unique sentences (%)':<22}  {bs['unique_pct']:>13.1f}%  {ts['unique_pct']:>13.1f}%  {delta(bs['unique_pct'], ts['unique_pct']):>10}")
    print()

    if bs["top_errors"] or ts["top_errors"]:
        print(f"  Top grammar errors (baseline):")
        for err, cnt in bs["top_errors"]:
            print(f"    [{cnt:3d}x] {err}")
        print(f"  Top grammar errors (trained):")
        for err, cnt in ts["top_errors"]:
            print(f"    [{cnt:3d}x] {err}")
        print()

    # Side-by-side sentence samples
    print(f"  {'─'*70}")
    print(f"  SAMPLE SENTENCES  (V=valid  X=invalid)")
    print(f"  {'─'*70}")
    print(f"  {'BASELINE':^33}  {'TRAINED':^33}")
    print(f"  {'─'*33}  {'─'*33}")
    for b, t in zip(baseline_results[:n_samples], trained_results[:n_samples]):
        b_tag = "V" if b["valid"] else "X"
        t_tag = "V" if t["valid"] else "X"
        b_str = f"[{b_tag}] {b['sentence']}"[:33]
        t_str = f"[{t_tag}] {t['sentence']}"[:33]
        print(f"  {b_str:<33}  {t_str:<33}")
    print("=" * 70)
    print()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    device = torch.device("cpu")

    # --- Language stack ---
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg       = CFG.from_json(str(TRANSITION_PATH), nouns=nouns, verbs=verbs, adjectives=adjectives)
    validator = CFGValidator.from_cfg(cfg)
    tracker   = CFGStateTracker(nouns, verbs, adjectives)

    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    tokenizer   = WordTokenizer.from_corpus(corpus_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # --- Load defenders ---
    baseline_path = Path(args.baseline)
    trained_path  = Path(args.trained)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_path}")

    print(f"Loading baseline : {baseline_path.name}")
    baseline = load_defender(baseline_path, tokenizer, device)
    print(f"Loading trained  : {trained_path.name}")
    trained  = load_defender(trained_path,  tokenizer, device)

    # --- Generate ---
    print(f"\nGenerating {args.n} sentences from each defender "
          f"(min_tokens={args.min_tokens}, max_tokens={args.max_tokens}, "
          f"temp={args.temperature}, prefix={'yes' if args.use_prefix else 'no'})...")

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    baseline_results = generate_batch(
        baseline, tokenizer, validator, tracker, device,
        args.n, args.temperature, args.max_tokens, args.min_tokens, args.use_prefix)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trained_results = generate_batch(
        trained, tokenizer, validator, tracker, device,
        args.n, args.temperature, args.max_tokens, args.min_tokens, args.use_prefix)

    # --- Report ---
    print_report(
        baseline_results, trained_results,
        baseline_label=baseline_path.parent.name or baseline_path.stem,
        trained_label=trained_path.parent.name or trained_path.stem,
        n_samples=args.show,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline vs adversarially trained defender on text generation quality")
    p.add_argument("--baseline",     required=True,
                   help="Path to baseline (pre-adversarial) defender checkpoint")
    p.add_argument("--trained",      required=True,
                   help="Path to adversarially trained defender checkpoint")
    p.add_argument("--n",            type=int,   default=200,
                   help="Sentences to generate per model (default: 200)")
    p.add_argument("--min-tokens",   type=int,   default=4,    dest="min_tokens",
                   help="Min new tokens before EOS is allowed (default: 4)")
    p.add_argument("--max-tokens",   type=int,   default=20,   dest="max_tokens",
                   help="Max tokens to generate (default: 20)")
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--use-prefix",   action="store_true", dest="use_prefix",
                   help="Start from a random CFG-valid prefix instead of pure BOS")
    p.add_argument("--show",         type=int,   default=10,
                   help="Number of sample sentences to show side-by-side (default: 10)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--corpus",       type=str,   default=None,
                   help="Corpus file for the tokenizer (default: corpus_10000.txt)")
    return p.parse_args()


if __name__ == "__main__":
    main()

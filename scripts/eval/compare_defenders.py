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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacker.attacker import AttackerTransformer
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
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------------ #
#  Generation                                                          #
# ------------------------------------------------------------------ #

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    """Divide logits of already-generated tokens by penalty (>1 = less likely to repeat)."""
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        logits[0, token_id] /= penalty
    return logits


def apply_no_repeat_ngram(logits: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
    """Block any token that would create a repeated n-gram (logit set to -inf).

    With ngram_size=2: blocks a token T if (prev_token, T) has already appeared.
    With ngram_size=3: blocks T if (prev-1, prev, T) has already appeared.
    """
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i: i + ngram_size - 1]) == prefix:
            logits[0, generated_ids[i + ngram_size - 1]] = float("-inf")
    return logits


def generate_sentence(
    model:                MiniGPT,
    tokenizer:            WordTokenizer,
    device:               torch.device,
    temperature:          float,
    max_tokens:           int,
    min_tokens:           int,
    prefix_ids:           list[int] | None = None,
    repetition_penalty:   float = 1.0,
    no_repeat_ngram_size: int   = 0,
) -> str:
    """Generate one sentence. If prefix_ids is None, starts from BOS only."""
    all_ids = [tokenizer.bos_id] + (prefix_ids or [])
    new_count = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([all_ids[-model.context_len:]], device=device)
            logits  = model(context)[:, -1, :] / temperature
            logits  = apply_repetition_penalty(logits, all_ids[1:], repetition_penalty)
            logits  = apply_no_repeat_ngram(logits, all_ids[1:], no_repeat_ngram_size)
            if new_count < min_tokens:
                logits[0, tokenizer.eos_id] = float("-inf")
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == tokenizer.eos_id:
                break
            all_ids.append(next_id)
            new_count += 1

    return tokenizer.decode(all_ids[1:])  # strip BOS


def build_prefixes(
    n:           int,
    tokenizer:   WordTokenizer,
    tracker:     CFGStateTracker,
    device:      torch.device,
    use_prefix:  bool,
    attacker:    AttackerTransformer | None = None,
    max_prefix:  int   = 6,
    atk_temp:    float = 1.0,
) -> list[list[int] | None]:
    """Build a list of n prefixes ONCE so every defender is evaluated on the
    exact same prefixes (apples-to-apples).

    Prefix source priority:
      * attacker is not None -> prefix generated by the trained attacker
      * use_prefix           -> random CFG-valid prefix
      * otherwise            -> None (pure BOS generation)
    """
    import random
    prefixes: list[list[int] | None] = []
    for _ in range(n):
        if attacker is not None:
            with torch.no_grad():
                p_ids, _p_words = attacker.generate_prefix(
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    cfg_tracker=tracker,
                    token_to_id=tokenizer.token_to_id,
                    id_to_token=tokenizer.id_to_token,
                    max_tokens=max_prefix,
                    temperature=atk_temp,
                    device=str(device),
                )
            prefixes.append(p_ids if p_ids else None)
        elif use_prefix:
            tracker.reset()
            ids, words = [], []
            for _ in range(3):
                valid, can_end = tracker.valid_next_words()
                if can_end and (not valid or len(ids) >= 1):
                    break
                if not valid:
                    break
                w = random.choice(valid)
                tracker.step(w)
                words.append(w)
                ids.append(tokenizer.token_to_id[w])
            prefixes.append(ids if ids else None)
        else:
            prefixes.append(None)
    return prefixes


def generate_batch(
    model:                MiniGPT,
    tokenizer:            WordTokenizer,
    validator:            CFGValidator,
    device:               torch.device,
    prefixes:             list[list[int] | None],
    temperature:          float,
    max_tokens:           int,
    min_tokens:           int,
    repetition_penalty:   float = 1.0,
    no_repeat_ngram_size: int   = 0,
) -> list[dict]:
    """Complete each of the given prefixes with `model`. Returns result dicts.

    The prefixes are precomputed (see build_prefixes) so two defenders can be
    run on identical prefixes.
    """
    results = []
    for prefix_ids in prefixes:
        sent = generate_sentence(
            model, tokenizer, device, temperature, max_tokens, min_tokens, prefix_ids,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)
        res  = validator.validate(sent)
        words = sent.split()
        results.append({
            "sentence": sent,
            "valid":    res.is_valid,
            "length":   len(words),
            "error":    res.error if not res.is_valid else None,
            "prefix":   tokenizer.decode(prefix_ids) if prefix_ids else "",
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

    # --- vocabulary diversity across the whole batch ---
    all_words   = [w for r in results for w in r["sentence"].split()]
    word_ctr    = Counter(all_words)
    total_words = len(all_words)
    distinct    = len(word_ctr)
    # share of all generated words taken by the single most common word
    top_word, top_count = word_ctr.most_common(1)[0] if word_ctr else ("-", 0)
    top_share   = top_count / total_words * 100 if total_words else 0.0
    # share taken by the top-3 words (how concentrated is the vocabulary)
    top3_share  = sum(c for _, c in word_ctr.most_common(3)) / total_words * 100 if total_words else 0.0

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
        "distinct_words": distinct,
        "top_word":     top_word,
        "top_share":    top_share,
        "top3_share":   top3_share,
        "top_words":    word_ctr.most_common(6),
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
        if higher_is_better:
            tag = " (better)" if d > 0 else (" (worse)" if d < 0 else "")
        else:
            tag = " (better)" if d < 0 else (" (worse)" if d > 0 else "")
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
    print(f"  {'Distinct words used':<22}  {bs['distinct_words']:>14}  {ts['distinct_words']:>14}  {delta(bs['distinct_words'], ts['distinct_words']):>10}")
    print(f"  {'Top word share (%)':<22}  {bs['top_share']:>13.1f}%  {ts['top_share']:>13.1f}%  {delta(bs['top_share'], ts['top_share'], higher_is_better=False):>10}")
    print(f"  {'Top-3 word share (%)':<22}  {bs['top3_share']:>13.1f}%  {ts['top3_share']:>13.1f}%  {delta(bs['top3_share'], ts['top3_share'], higher_is_better=False):>10}")
    print()
    print(f"  Most-used words (baseline): " + ", ".join(f"{w}:{c}" for w, c in bs['top_words']))
    print(f"  Most-used words (trained) : " + ", ".join(f"{w}:{c}" for w, c in ts['top_words']))
    print()

    if bs["top_errors"] or ts["top_errors"]:
        print(f"  Top grammar errors (baseline):")
        for err, cnt in bs["top_errors"]:
            print(f"    [{cnt:3d}x] {err}")
        print(f"  Top grammar errors (trained):")
        for err, cnt in ts["top_errors"]:
            print(f"    [{cnt:3d}x] {err}")
        print()

    # Side-by-side sentence samples. Both defenders share the same prefix, so
    # show the prefix once and each defender's full sentence beside it.
    has_prefix = any(b.get("prefix") for b in baseline_results[:n_samples])
    max_b = max(len(f"[{'V' if r['valid'] else 'X'}] {r['sentence']}") for r in baseline_results[:n_samples])
    max_t = max(len(f"[{'V' if r['valid'] else 'X'}] {r['sentence']}") for r in trained_results[:n_samples])
    col_w = max(max_b, max_t, 33)

    print(f"  {'-'*(col_w*2+4)}")
    print(f"  SAMPLE SENTENCES  (V=valid  X=invalid; same prefix for both)")
    print(f"  {'-'*(col_w*2+4)}")
    print(f"  {'BASELINE':^{col_w}}  {'TRAINED':^{col_w}}")
    print(f"  {'-'*col_w}  {'-'*col_w}")
    for b, t in zip(baseline_results[:n_samples], trained_results[:n_samples]):
        if has_prefix:
            print(f"  prefix> {b.get('prefix') or '(none)'}")
        b_str = f"[{'V' if b['valid'] else 'X'}] {b['sentence']}"
        t_str = f"[{'V' if t['valid'] else 'X'}] {t['sentence']}"
        print(f"  {b_str:<{col_w}}  {t_str:<{col_w}}")
    print("=" * (col_w*2+6))


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

    # --- Optional attacker for prefix generation ---
    attacker = None
    if args.attacker_ckpt:
        atk_path = Path(args.attacker_ckpt)
        if not atk_path.exists():
            raise FileNotFoundError(f"Attacker checkpoint not found: {atk_path}")
        print(f"Loading attacker : {atk_path.name}")
        atk_ckpt = torch.load(atk_path, map_location=device, weights_only=False)
        attacker = AttackerTransformer(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
        attacker.load_state_dict(atk_ckpt["model_state"])
        attacker.eval()

    prefix_mode = ("attacker" if attacker is not None
                   else "random" if args.use_prefix else "none (pure BOS)")

    # --- Generate ---
    print(f"\nGenerating {args.n} sentences from each defender "
          f"(min_tokens={args.min_tokens}, max_tokens={args.max_tokens}, "
          f"temp={args.temperature}, prefix={prefix_mode})...")

    import random

    # Build the prefixes ONCE so both defenders are evaluated on identical prefixes.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prefixes = build_prefixes(
        args.n, tokenizer, tracker, device, args.use_prefix,
        attacker=attacker, max_prefix=args.max_prefix, atk_temp=args.atk_temp)

    # Reseed before each defender so completions draw from the same RNG stream
    # too — any difference is then purely the model, not randomness.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    baseline_results = generate_batch(
        baseline, tokenizer, validator, device, prefixes,
        args.temperature, args.max_tokens, args.min_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    trained_results = generate_batch(
        trained, tokenizer, validator, device, prefixes,
        args.temperature, args.max_tokens, args.min_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size)

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
    p.add_argument("--temperature",         type=float, default=0.8)
    p.add_argument("--repetition-penalty",    type=float, default=1.3, dest="repetition_penalty",
                   help="Penalize repeated tokens (1.0=off, 1.3=moderate, 2.0=strong, default: 1.3)")
    p.add_argument("--no-repeat-ngram-size",  type=int,   default=2,   dest="no_repeat_ngram_size",
                   help="Hard-block any n-gram that already appeared (0=off, 2=no bigram repeats, default: 2)")
    p.add_argument("--use-prefix",   action="store_true", dest="use_prefix",
                   help="Start from a random CFG-valid prefix instead of pure BOS")
    p.add_argument("--attacker-ckpt", type=str, default=None, dest="attacker_ckpt",
                   help="Path to an attacker checkpoint. If given, prefixes are generated by the "
                        "attacker (same setup as attack_and_complete) instead of random/BOS")
    p.add_argument("--max-prefix",   type=int,   default=6,    dest="max_prefix",
                   help="Max attacker prefix tokens when --attacker-ckpt is used (default: 6)")
    p.add_argument("--atk-temp",     type=float, default=1.0,  dest="atk_temp",
                   help="Attacker temperature when --attacker-ckpt is used (default: 1.0)")
    p.add_argument("--show",         type=int,   default=10,
                   help="Number of sample sentences to show side-by-side (default: 10)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--corpus",       type=str,   default=None,
                   help="Corpus file for the tokenizer (default: corpus_10000.txt)")
    return p.parse_args()


if __name__ == "__main__":
    main()

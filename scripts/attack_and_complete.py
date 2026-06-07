"""
AttackPipeline — full adversarial pipeline:

  1. Attacker generates a CFG-valid prefix
  2. MiniGPT completes the prefix into a full sentence
  3. CFGValidator checks if the completed sentence is grammatically correct
  4. RewardComputer assigns a reward signal to the attacker

Reward levels:
  A) Grammar failure  → highest reward (1.0 * w_grammar)
  B) Topic mismatch   → medium reward  (mismatch * w_mismatch)
  C) Topic consistent → low/zero reward

Usage:
    python scripts/attack_and_complete.py
    python scripts/attack_and_complete.py --n 10
    python scripts/attack_and_complete.py --max-prefix 4 --temperature 0.9
    python scripts/attack_and_complete.py --verbose   # show reward breakdown
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.attacker.attacker import AttackerTransformer
from src.attacker.reward import RewardComputer, RewardConfig
from src.attacker.reward_function import RewardFunction, RewardWeights
from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
CORPUS_PATH     = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"
CKPT_PATH       = PROJECT_ROOT / "data" / "models" / "minigpt_corpus10000.pt"


# ------------------------------------------------------------------ #
#  Pipeline class                                                      #
# ------------------------------------------------------------------ #

class AttackPipeline:
    """
    Runs the full attack loop:
      Attacker -> prefix -> MiniGPT completion -> CFGValidator
    """

    def __init__(
        self,
        max_prefix_tokens: int   = 6,
        max_completion_tokens: int = 15,
        attacker_temperature: float = 1.0,
        defender_temperature: float = 0.8,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.max_prefix_tokens     = max_prefix_tokens
        self.max_completion_tokens = max_completion_tokens
        self.attacker_temperature  = attacker_temperature
        self.defender_temperature  = defender_temperature
        self.device                = device
        self.verbose               = verbose

        print("Loading lexicon...")
        nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)

        print("Building CFG and validator...")
        cfg = CFG.from_json(
            str(TRANSITION_PATH),
            nouns=nouns, verbs=verbs, adjectives=adjectives,
        )
        self.validator = CFGValidator.from_cfg(cfg)
        self.tracker   = CFGStateTracker(nouns, verbs, adjectives)

        print("Loading tokenizer...")
        self.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)

        print("Loading MiniGPT (defender)...")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        self.defender = MiniGPT(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        self.defender.load_state_dict(ckpt["model_state"])
        self.defender.to(device)
        self.defender.eval()
        print(f"  Loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

        print("Building attacker (untrained)...")
        self.attacker = AttackerTransformer(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        self.attacker.eval()

        print("Building reward computer...")
        self.reward_computer = RewardComputer(
            nouns=nouns, verbs=verbs, adjectives=adjectives,
            config=RewardConfig(),
        )
        self.reward_function = RewardFunction(
            nouns=nouns, verbs=verbs, adjectives=adjectives,
            weights=RewardWeights(),
        )

        print("Ready.\n")

    def _complete(self, prefix_ids: list[int]) -> list[int]:
        """
        Feed prefix_ids into MiniGPT and generate a completion.
        Returns all_ids (prefix + generated), BOS excluded.
        """
        all_ids = [self.tokenizer.bos_id] + prefix_ids

        with torch.no_grad():
            for _ in range(self.max_completion_tokens):
                context = torch.tensor(
                    [all_ids[-self.defender.context_len:]], device=self.device
                )
                logits  = self.defender(context)[:, -1, :] / self.defender_temperature
                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id == self.tokenizer.eos_id:
                    break

                all_ids.append(next_id)

        return all_ids[1:]  # strip BOS

    def run_once(self) -> dict:
        """
        Run one full attack iteration.
        Returns a dict with prefix, full_sentence, and validation result.
        """
        # Step 1 — attacker generates prefix
        prefix_ids, prefix_words = self.attacker.generate_prefix(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=self.max_prefix_tokens,
            temperature=self.attacker_temperature,
        )

        # Step 2 — MiniGPT completes the prefix
        full_ids   = self._complete(prefix_ids)
        full_words = self.tokenizer.decode(full_ids).split()

        # Step 3 — CFGValidator checks the full sentence
        full_sentence = " ".join(full_words)
        result        = self.validator.validate(full_sentence)

        # Step 4 — Reward (structured)
        prefix_part, suffix_part = self.reward_function.split_sentence(full_words, prefix_words)
        rf_output = self.reward_function.compute(
            prefix_words  = prefix_part,
            suffix_words  = suffix_part,
            full_sentence = full_sentence,
            is_valid      = result.is_valid,
            cfg_error     = result.error if not result.is_valid else None,
        )

        return {
            "prefix":        " ".join(prefix_words) if prefix_words else "(empty)",
            "full_sentence": full_sentence,
            "is_valid":      result.is_valid,
            "error":         result.error if not result.is_valid else "",
            "reward":        rf_output,
        }

    def run(self, n: int = 5) -> None:
        """Run n attack iterations and print a summary table."""

        print("=" * 70)
        print(f"  ATTACK PIPELINE  |  n={n}  |  prefix_max={self.max_prefix_tokens}  "
              f"|  T_atk={self.attacker_temperature}  |  T_def={self.defender_temperature}")
        print("=" * 70)

        valid_count    = 0
        invalid_count  = 0
        total_reward   = 0.0

        for i in range(1, n + 1):
            r  = self.run_once()
            rw = r["reward"]

            verdict = "VALID  " if r["is_valid"] else "INVALID"
            if r["is_valid"]:
                valid_count += 1
            else:
                invalid_count += 1
            total_reward += rw.reward

            print(f"\n  [{i:2d}] {verdict}  |  reward={rw.reward:.4f}")
            print(f"       Prefix    : {r['prefix']}")
            print(f"       Full      : {r['full_sentence']}")
            if r["error"]:
                print(f"       Reason    : {r['error']}")
            print(f"       Reward    : grammar={rw.grammar_reward:.2f}"
                  f"  noun_tag={rw.distances.noun_tag_dist:.3f}"
                  f"  verb_tag={rw.distances.verb_tag_dist:.3f}"
                  f"  adj_tag={rw.distances.adjective_tag_dist:.3f}"
                  f"  axis={rw.distances.axis_distance:.3f}"
                  f"  total={rw.reward:.4f}")
            if self.verbose:
                print()
                print(rw.summary())

        total   = valid_count + invalid_count
        pct     = 100 * valid_count / total if total else 0
        avg_rw  = total_reward / total if total else 0.0

        print(f"\n{'=' * 70}")
        print(f"  VALID       : {valid_count}/{total}  ({pct:.0f}%)")
        print(f"  INVALID     : {invalid_count}/{total}  ({100 - pct:.0f}%)")
        print(f"  AVG REWARD  : {avg_rw:.4f}")
        print("=" * 70)


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Attacker prefix + MiniGPT completion + CFGValidator")
    p.add_argument("--n",             type=int,   default=100,   help="Number of attack iterations (default: 5)")
    p.add_argument("--max-prefix",    type=int,   default=6,   dest="max_prefix",    help="Max attacker prefix tokens (default: 6)")
    p.add_argument("--max-completion",type=int,   default=15,  dest="max_completion",help="Max MiniGPT completion tokens (default: 15)")
    p.add_argument("--atk-temp",      type=float, default=1.0, dest="atk_temp",      help="Attacker temperature (default: 1.0)")
    p.add_argument("--verbose",       action="store_true",                           help="Show full reward breakdown per sentence")
    p.add_argument("--def-temp",      type=float, default=0.8, dest="def_temp",      help="Defender temperature (default: 0.8)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = AttackPipeline(
        max_prefix_tokens     = args.max_prefix,
        max_completion_tokens = args.max_completion,
        attacker_temperature  = args.atk_temp,
        defender_temperature  = args.def_temp,
        verbose               = args.verbose,
    )
    pipeline.run(n=args.n)

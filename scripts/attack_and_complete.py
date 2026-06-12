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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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

try:
    from config import DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
except ImportError:
    DAGSHUB_REPO_OWNER = None
    DAGSHUB_REPO_NAME = None
    DAGSHUB_TOKEN = None

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

try:
    import dagshub
    HAS_DAGSHUB = True
except ImportError:
    HAS_DAGSHUB = False
    dagshub = None

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
CORPUS_PATH     = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"
CKPT_PATH       = PROJECT_ROOT / "data" / "models" / "minigpt_corpus10000.pt"


def setup_mlflow(use_mlflow: bool) -> bool:
    if not use_mlflow:
        return False
    if not HAS_MLFLOW:
        print("[WARN] --mlflow requested but 'mlflow' not installed.")
        return False
    repo_owner = DAGSHUB_REPO_OWNER or os.getenv("DAGSHUB_REPO_OWNER")
    repo_name  = DAGSHUB_REPO_NAME  or os.getenv("DAGSHUB_REPO_NAME")
    token      = DAGSHUB_TOKEN      or os.getenv("DAGSHUB_USER_TOKEN")
    if repo_owner and repo_name and HAS_DAGSHUB:
        if token:
            os.environ["DAGSHUB_USER_TOKEN"] = token
            os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print(f"[OK] Connected to DagsHub: {repo_owner}/{repo_name}")
        except Exception as e:
            # DagsHub down / unreachable -> fall back to local MLflow tracking
            local_uri = (PROJECT_ROOT / "mlruns").as_uri()
            mlflow.set_tracking_uri(local_uri)
            print(f"[WARN] DagsHub unreachable ({type(e).__name__}). "
                  f"Falling back to LOCAL MLflow tracking.")
            print(f"  Local tracking dir: {PROJECT_ROOT / 'mlruns'}")
    else:
        print("[WARN] MLflow enabled (local tracking only).")
    mlflow.set_experiment("AttackBenchmark")
    return True


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
        min_completion_tokens: int = 1,
        attacker_temperature: float = 1.0,
        defender_temperature: float = 0.8,
        device: str = "cpu",
        verbose: bool = False,
        attacker_ckpt: str | None = None,
        defender_ckpt: str | None = None,
    ):
        self.max_prefix_tokens     = max_prefix_tokens
        self.max_completion_tokens = max_completion_tokens
        self.min_completion_tokens = min_completion_tokens
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

        defender_path = Path(defender_ckpt) if defender_ckpt else CKPT_PATH
        print(f"Loading MiniGPT (defender) from {defender_path.name}...")
        if not defender_path.exists():
            raise FileNotFoundError(
                f"Defender checkpoint not found: {defender_path}\n"
                f"Train one first with: python scripts/train/train_model.py"
            )
        ckpt = torch.load(defender_path, map_location=device, weights_only=False)
        self.defender = MiniGPT(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        self.defender.load_state_dict(ckpt["model_state"])
        self.defender.to(device)
        self.defender.eval()
        self.defender_ckpt_path = defender_path
        epoch = ckpt.get("epoch", "?")
        loss  = ckpt.get("loss")
        avg_rw = ckpt.get("avg_reward")
        if loss is not None:
            print(f"  Loaded epoch {epoch}, loss {loss:.4f}")
        elif avg_rw is not None:
            print(f"  Loaded episode {ckpt.get('episode', '?')}, avg_reward {avg_rw:.4f}")
        else:
            print(f"  Loaded checkpoint keys: {list(ckpt.keys())}")

        self.attacker = AttackerTransformer(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        if attacker_ckpt:
            print(f"Loading attacker checkpoint: {attacker_ckpt}")
            atk_ckpt = torch.load(attacker_ckpt, map_location=device, weights_only=False)
            self.attacker.load_state_dict(atk_ckpt["model_state"])
            print(f"  Loaded episode {atk_ckpt.get('episode', '?')}, avg_reward {atk_ckpt.get('avg_reward', float('nan')):.4f}")
        else:
            print("Building attacker (untrained)...")
        self.attacker.to(device)
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

        The defender must add at least `min_completion_tokens` new words:
        EOS is masked out until that minimum is reached, so the completion
        is never just the unchanged prefix.
        """
        all_ids   = [self.tokenizer.bos_id] + prefix_ids
        new_count = 0

        with torch.no_grad():
            for _ in range(self.max_completion_tokens):
                context = torch.tensor(
                    [all_ids[-self.defender.context_len:]], device=self.device
                )
                logits  = self.defender(context)[:, -1, :] / self.defender_temperature

                # Forbid EOS until the defender has added the minimum new words
                if new_count < self.min_completion_tokens:
                    logits[0, self.tokenizer.eos_id] = float("-inf")

                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id == self.tokenizer.eos_id:
                    break

                all_ids.append(next_id)
                new_count += 1

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

    def run(
        self,
        n: int = 5,
        use_mlflow: bool = False,
        attacker_ckpt_path: str | None = None,
        run_name: str | None = None,
        quiet: bool = False,
        log_every: int = 50,
    ) -> None:
        """Run n attack iterations and print a summary table."""
        import csv

        if use_mlflow:
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "n":                     n,
                "max_prefix_tokens":     self.max_prefix_tokens,
                "max_completion_tokens": self.max_completion_tokens,
                "attacker_temperature":  self.attacker_temperature,
                "defender_temperature":  self.defender_temperature,
                "attacker_ckpt":         attacker_ckpt_path or "untrained",
                "defender_ckpt":         str(self.defender_ckpt_path),
                "run_name":              run_name or "default",
            })

        print("=" * 70)
        print(f"  ATTACK PIPELINE  |  n={n}  |  prefix_max={self.max_prefix_tokens}  "
              f"|  T_atk={self.attacker_temperature}  |  T_def={self.defender_temperature}")
        print("=" * 70)

        valid_count    = 0
        invalid_count  = 0
        total_reward   = 0.0
        total_grammar  = 0.0
        total_noun     = 0.0
        total_verb     = 0.0
        total_adj      = 0.0
        total_axis     = 0.0

        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        csv_stem = run_name or "benchmark_iterations"
        csv_path = logs_dir / f"{csv_stem}.csv"
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "iteration", "prefix", "full_sentence", "is_valid", "reward",
            "grammar_reward", "noun_tag_dist", "verb_tag_dist",
            "adjective_tag_dist", "axis_distance", "error",
        ])

        for i in range(1, n + 1):
            r  = self.run_once()
            rw = r["reward"]

            verdict = "VALID  " if r["is_valid"] else "INVALID"
            if r["is_valid"]:
                valid_count += 1
            else:
                invalid_count += 1
            total_reward  += rw.reward
            total_grammar += rw.grammar_reward
            total_noun    += rw.distances.noun_tag_dist
            total_verb    += rw.distances.verb_tag_dist
            total_adj     += rw.distances.adjective_tag_dist
            total_axis    += rw.distances.axis_distance

            csv_writer.writerow([
                i, r["prefix"], r["full_sentence"], int(r["is_valid"]),
                f"{rw.reward:.4f}", f"{rw.grammar_reward:.4f}",
                f"{rw.distances.noun_tag_dist:.4f}",
                f"{rw.distances.verb_tag_dist:.4f}",
                f"{rw.distances.adjective_tag_dist:.4f}",
                f"{rw.distances.axis_distance:.4f}",
                r["error"],
            ])

            if use_mlflow:
                mlflow.log_metric("iter_reward", rw.reward, step=i)
                mlflow.log_metric("iter_invalid", 0 if r["is_valid"] else 1, step=i)

            if not quiet or i == 1 or i == n or i % log_every == 0:
                print(f"\n  [{i:4d}] {verdict}  |  reward={rw.reward:.4f}")
                if not quiet:
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
                elif i % log_every == 0:
                    pct_so_far = 100 * valid_count / i
                    print(f"       progress: valid={valid_count}/{i} ({pct_so_far:.0f}%)  "
                          f"avg_reward={total_reward/i:.4f}")

        csv_file.close()

        total   = valid_count + invalid_count
        pct     = 100 * valid_count / total if total else 0
        avg_rw  = total_reward / total if total else 0.0

        print(f"\n{'=' * 70}")
        print(f"  VALID       : {valid_count}/{total}  ({pct:.0f}%)")
        print(f"  INVALID     : {invalid_count}/{total}  ({100 - pct:.0f}%)")
        print(f"  AVG REWARD  : {avg_rw:.4f}")
        print("=" * 70)

        if use_mlflow and total:
            mlflow.log_metric("valid_count",         valid_count)
            mlflow.log_metric("invalid_count",       invalid_count)
            mlflow.log_metric("invalid_rate",        invalid_count / total)
            mlflow.log_metric("avg_reward",          avg_rw)
            mlflow.log_metric("avg_grammar_reward", total_grammar / total)
            mlflow.log_metric("avg_noun_tag_dist",  total_noun    / total)
            mlflow.log_metric("avg_verb_tag_dist",  total_verb    / total)
            mlflow.log_metric("avg_adj_tag_dist",   total_adj     / total)
            mlflow.log_metric("avg_axis_distance", total_axis    / total)
            mlflow.log_artifact(str(csv_path))
            mlflow.end_run()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Attacker prefix + MiniGPT completion + CFGValidator")
    p.add_argument("--n",             type=int,   default=100,   help="Number of attack iterations (default: 5)")
    p.add_argument("--max-prefix",    type=int,   default=6,   dest="max_prefix",    help="Max attacker prefix tokens (default: 6)")
    p.add_argument("--max-completion",type=int,   default=15,  dest="max_completion",help="Max MiniGPT completion tokens (default: 15)")
    p.add_argument("--min-completion",type=int,   default=1,   dest="min_completion",help="Min new words the defender must add (default: 1)")
    p.add_argument("--atk-temp",      type=float, default=1.0, dest="atk_temp",      help="Attacker temperature (default: 1.0)")
    p.add_argument("--verbose",       action="store_true",                           help="Show full reward breakdown per sentence")
    p.add_argument("--def-temp",      type=float, default=0.8, dest="def_temp",      help="Defender temperature (default: 0.8)")
    p.add_argument("--attacker-ckpt", type=str,   default=None, dest="attacker_ckpt", help="Path to attacker checkpoint (default: untrained attacker)")
    p.add_argument("--defender-ckpt", type=str,   default=None, dest="defender_ckpt", help="Path to defender checkpoint (default: minigpt_corpus10000.pt)")
    p.add_argument("--run-name",      type=str,   default=None, dest="run_name",      help="MLflow run name and CSV log stem (default: benchmark_iterations)")
    p.add_argument("--quiet",         action="store_true",                           help="Suppress per-iteration detail (recommended for large n)")
    p.add_argument("--log-every",     type=int,   default=50,  dest="log_every",     help="Progress interval when --quiet (default: 50)")
    p.add_argument("--mlflow",        action="store_true",                            help="Log benchmark to MLflow / DagsHub")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_tracking = setup_mlflow(args.mlflow)
    pipeline = AttackPipeline(
        max_prefix_tokens     = args.max_prefix,
        max_completion_tokens = args.max_completion,
        min_completion_tokens = args.min_completion,
        attacker_temperature  = args.atk_temp,
        defender_temperature  = args.def_temp,
        verbose               = args.verbose,
        attacker_ckpt         = args.attacker_ckpt,
        defender_ckpt         = args.defender_ckpt,
    )
    pipeline.run(
        n=args.n,
        use_mlflow=use_tracking,
        attacker_ckpt_path=args.attacker_ckpt,
        run_name=args.run_name,
        quiet=args.quiet,
        log_every=args.log_every,
    )

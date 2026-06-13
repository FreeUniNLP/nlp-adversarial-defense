"""Adversarial co-training: alternate REINFORCE between attacker and defender.

Takes the best attacker and best defender checkpoints and trains them
AGAINST EACH OTHER in alternating phases:

    round 1:  attacker trains X episodes   (defender frozen)
              defender trains X episodes   (attacker frozen)
    round 2:  attacker trains X episodes   (defender frozen)
              defender trains X episodes   (attacker frozen)
    ...

Both sides use the SAME RewardFunction:

    R = w_grammar * grammar_failure + w_tag * tag_mismatch + w_axis * axis_distance

    attacker maximizes  R      (wants invalid / off-topic completions)
    defender maximizes -R      (wants valid / on-topic completions)

Lessons from earlier single-sided runs are baked in:
  * attacker phases use an entropy bonus (--entropy-coef) so the attacker
    cannot mode-collapse onto a single exploit prefix
  * defender phases mix in random CFG prefixes (--mix-random) so the
    defender stays general instead of memorizing the attacker's habits
  * the CFGStateTracker guarantees every attacker prefix is completable,
    so neither side can win through unanswerable dead-end prefixes

After each round both models are frozen and evaluated head-to-head, so you
can watch the arms race round by round.

Usage:
    python scripts/train/train_adversarial.py --rounds 5 -x 1000
    python scripts/train/train_adversarial.py --rounds 3 -x 2000 --mlflow
    python scripts/train/train_adversarial.py --rounds 2 -x 200          # quick test
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacker.attacker import AttackerTransformer
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.attacker.reward_function import RewardFunction, RewardWeights
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from src.language.parsers import LexiconParser
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
DEFAULT_CORPUS  = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"

DEFAULT_ATTACKER_CKPT = PROJECT_ROOT / "data" / "models" / "from_mlflow" / "AttackerREINFORCE" / "enthused-sheep-51" / "attacker_best.pt"
DEFAULT_DEFENDER_CKPT = PROJECT_ROOT / "data" / "models" / "from_mlflow" / "MiniGPT" / "nosy-chimp-161" / "minigpt_corpus10000.pt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "cotrain"


# ------------------------------------------------------------------ #
#  Experiment tracking                                                 #
# ------------------------------------------------------------------ #

def setup_mlflow(use_mlflow: bool) -> bool:
    """Initialize MLflow + DagsHub. Falls back to local tracking if DagsHub is down."""
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
            local_uri = (PROJECT_ROOT / "mlruns").as_uri()
            mlflow.set_tracking_uri(local_uri)
            print(f"[WARN] DagsHub unreachable ({type(e).__name__}). "
                  f"Falling back to LOCAL MLflow tracking.")
            print(f"  Local tracking dir: {PROJECT_ROOT / 'mlruns'}")
    else:
        print("[WARN] MLflow enabled (local tracking only).")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_experiment("AdversarialCoTraining")
    return True


# ------------------------------------------------------------------ #
#  Rollout helpers                                                     #
# ------------------------------------------------------------------ #

def sample_random_prefix(
    tracker: CFGStateTracker,
    token_to_id: dict,
    max_tokens: int,
    stop_prob: float = 0.3,
) -> tuple[list[int], list[str]]:
    """Uniformly sample a CFG-valid prefix (defender exploration)."""
    tracker.reset()
    ids, words = [], []
    for _ in range(max_tokens):
        valid, can_end = tracker.valid_next_words()
        if can_end and (not valid or random.random() < stop_prob):
            break
        if not valid:
            break
        w = random.choice(valid)
        tracker.step(w)
        words.append(w)
        ids.append(token_to_id[w])
    return ids, words


def _no_repeat_ngram_block(logits: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i: i + ngram_size - 1]) == prefix:
            logits[0, generated_ids[i + ngram_size - 1]] = float("-inf")
    return logits


def defender_complete(
    defender:             MiniGPT,
    prefix_ids:           list[int],
    bos_id:               int,
    eos_id:               int,
    max_new_tokens:       int,
    temperature:          float,
    device:               torch.device,
    min_new_tokens:       int   = 1,
    repetition_penalty:   float = 1.0,
    no_repeat_ngram_size: int   = 0,
    with_grad:      bool = False,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Defender completion. If with_grad, each sampled token's log-prob and
    the full per-step entropy are tracked for REINFORCE; otherwise runs under
    no_grad.

    EOS is masked until min_new_tokens words have been added (same
    environment as AttackPipeline).
    Returns (full_ids_without_bos, log_probs, entropies).
    """
    all_ids   = [bos_id] + list(prefix_ids)
    log_probs = []
    entropies = []
    new_count = 0

    for _ in range(max_new_tokens):
        context = torch.tensor([all_ids[-defender.context_len:]], device=device)

        with torch.set_grad_enabled(with_grad):
            logits = defender(context)[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for tid in set(all_ids[1:]):
                    logits[0, tid] /= repetition_penalty
            logits = _no_repeat_ngram_block(logits, all_ids[1:], no_repeat_ngram_size)
            if new_count < min_new_tokens:
                mask = torch.zeros_like(logits)
                mask[0, eos_id] = float("-inf")
                logits = logits + mask
            log_dist = F.log_softmax(logits, dim=-1)

        # Sample from a clean detached probability tensor (avoids log→exp NaN path)
        probs = F.softmax(logits.detach(), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if with_grad:
            log_probs.append(log_dist[0, next_id])
            # True per-step entropy: H = -sum(p * log p); nan_to_num handles 0*-inf (masked EOS)
            entropies.append(-(probs * log_dist).nan_to_num(0.0).sum())

        if next_id == eos_id:
            break
        all_ids.append(next_id)
        new_count += 1

    lp = torch.stack(log_probs) if log_probs else torch.zeros(0, device=device)
    ent = torch.stack(entropies) if entropies else torch.zeros(0, device=device)
    return all_ids[1:], lp, ent


# ------------------------------------------------------------------ #
#  Co-trainer                                                          #
# ------------------------------------------------------------------ #

class AdversarialCoTrainer:
    """Alternating REINFORCE between attacker and defender."""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args   = args
        self.logger = logger
        self.device = torch.device(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

        # --- Language stack ---
        nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
        cfg = CFG.from_json(str(TRANSITION_PATH),
                            nouns=nouns, verbs=verbs, adjectives=adjectives)
        self.validator = CFGValidator.from_cfg(cfg)
        self.tracker   = CFGStateTracker(nouns, verbs, adjectives)

        corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
        self.tok = WordTokenizer.from_corpus(corpus_path)
        logger.info(f"Vocab size: {self.tok.vocab_size}")

        # --- Models ---
        self.attacker = self._load_attacker(args.attacker_ckpt)
        self.defender = self._load_defender(args.defender_ckpt)

        # --- Reward ---
        self.reward_fn = RewardFunction(
            nouns=nouns, verbs=verbs, adjectives=adjectives,
            weights=RewardWeights(w_grammar=args.w_grammar,
                                  w_tag=args.w_tag, w_axis=args.w_axis,
                                  w_repeat=args.w_repeat),
        )

        # --- Optimizers (persist across rounds) ---
        self.opt_atk = torch.optim.AdamW(self.attacker.parameters(), lr=args.lr_attacker)
        self.opt_def = torch.optim.AdamW(self.defender.parameters(), lr=args.lr_defender)

        # --- EMA baselines (persist across rounds) ---
        self.baseline_atk = 0.0
        self.baseline_def = 0.0

        self.global_episode = 0

    # ------------------------------------------------------------------ #

    def _load_attacker(self, ckpt_path: str | None) -> AttackerTransformer:
        path = Path(ckpt_path) if ckpt_path else DEFAULT_ATTACKER_CKPT
        if not path.exists():
            raise FileNotFoundError(f"Attacker checkpoint not found: {path}")
        self.logger.info(f"Loading attacker: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = AttackerTransformer(
            vocab_size=self.tok.vocab_size, pad_id=self.tok.pad_id).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    def _load_defender(self, ckpt_path: str | None) -> MiniGPT:
        path = Path(ckpt_path) if ckpt_path else DEFAULT_DEFENDER_CKPT
        if not path.exists():
            raise FileNotFoundError(f"Defender checkpoint not found: {path}")
        self.logger.info(f"Loading defender: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = MiniGPT(
            vocab_size=self.tok.vocab_size, pad_id=self.tok.pad_id).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()  # dropout off; gradients still flow when needed
        return model

    # ------------------------------------------------------------------ #
    #  One full episode (attack -> complete -> validate -> reward)        #
    # ------------------------------------------------------------------ #

    def _episode(self, train_side: str):
        """Run one episode. train_side in {'attacker', 'defender'}.

        Returns (attacker_reward, valid, atk_log_probs, def_log_probs,
                 prefix_words, suffix_words) — log_probs are empty tensors
        for the frozen side.
        """
        a = self.args

        # --- prefix ---
        atk_lp = torch.zeros(0, device=self.device)
        if train_side == "defender" and random.random() < a.mix_random:
            prefix_ids, prefix_words = sample_random_prefix(
                self.tracker, self.tok.token_to_id, a.max_prefix)
        elif train_side == "attacker":
            prefix_ids, prefix_words, atk_lp = self.attacker.generate_prefix_with_log_probs(
                bos_id=self.tok.bos_id, eos_id=self.tok.eos_id,
                cfg_tracker=self.tracker,
                token_to_id=self.tok.token_to_id, id_to_token=self.tok.id_to_token,
                max_tokens=a.max_prefix, temperature=a.atk_temp, device=str(self.device),
            )
        else:
            with torch.no_grad():
                prefix_ids, prefix_words = self.attacker.generate_prefix(
                    bos_id=self.tok.bos_id, eos_id=self.tok.eos_id,
                    cfg_tracker=self.tracker,
                    token_to_id=self.tok.token_to_id, id_to_token=self.tok.id_to_token,
                    max_tokens=a.max_prefix, temperature=a.atk_temp, device=str(self.device),
                )
        if not prefix_ids:
            return None

        # --- completion ---
        full_ids, def_lp, def_ent = defender_complete(
            self.defender, prefix_ids,
            bos_id=self.tok.bos_id, eos_id=self.tok.eos_id,
            max_new_tokens=a.max_completion, temperature=a.def_temp,
            device=self.device, with_grad=(train_side == "defender"),
            min_new_tokens=a.min_new_tokens,
            repetition_penalty=a.repetition_penalty,
            no_repeat_ngram_size=a.no_repeat_ngram_size,
        )

        full_words    = self.tok.decode(full_ids).split()
        full_sentence = " ".join(full_words)

        # --- validate + reward ---
        result = self.validator.validate(full_sentence)
        prefix_part, suffix_part = self.reward_fn.split_sentence(full_words, prefix_words)
        rw = self.reward_fn.compute(
            prefix_words=prefix_part, suffix_words=suffix_part,
            full_sentence=full_sentence,
            is_valid=result.is_valid,
            cfg_error=result.error if not result.is_valid else None,
        )
        return float(rw.reward), result.is_valid, atk_lp, def_lp, def_ent, prefix_words, suffix_part

    # ------------------------------------------------------------------ #
    #  Phases                                                              #
    # ------------------------------------------------------------------ #

    def train_phase(self, side: str, episodes: int, round_idx: int,
                    csv_writer, use_tracking: bool) -> dict:
        """Train one side for `episodes` episodes; the other side is frozen.

        Attacker early stopping: if the rolling valid rate falls below
        args.min_valid for a full log_every window the phase ends early so
        the defender can respond before the attacker fully collapses.
        """
        a = self.args
        reward_win = deque(maxlen=a.window)
        valid_win  = deque(maxlen=a.window)

        for ep in range(1, episodes + 1):
            out = self._episode(train_side=side)
            if out is None:
                continue
            attacker_reward, valid, atk_lp, def_lp, def_ent, prefix_words, suffix_words = out
            self.global_episode += 1

            if side == "attacker":
                if atk_lp.numel() == 0:
                    continue
                reward    = attacker_reward
                advantage = reward - self.baseline_atk
                policy_loss = -(advantage * atk_lp.sum())
                entropy_term = -atk_lp.mean()
                loss = policy_loss - a.entropy_coef * entropy_term

                self.opt_atk.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.attacker.parameters(), a.grad_clip)
                self.opt_atk.step()
                self.baseline_atk = (1 - a.baseline_alpha) * self.baseline_atk \
                                    + a.baseline_alpha * reward
            else:
                if def_lp.numel() == 0:
                    continue
                reward    = -attacker_reward          # defender minimizes attacker reward
                advantage = reward - self.baseline_def
                policy_loss = -(advantage * def_lp.sum())
                entropy_term = def_ent.mean() if def_ent.numel() > 0 else torch.tensor(0.0, device=self.device)
                loss = policy_loss - a.entropy_coef_defender * entropy_term

                self.opt_def.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.defender.parameters(), a.grad_clip)
                self.opt_def.step()
                self.baseline_def = (1 - a.baseline_alpha) * self.baseline_def \
                                    + a.baseline_alpha * reward

            reward_win.append(attacker_reward)
            valid_win.append(1.0 if valid else 0.0)

            csv_writer.writerow([
                self.global_episode, round_idx, side,
                " ".join(prefix_words), " ".join(suffix_words),
                int(valid), f"{attacker_reward:.4f}", f"{float(loss.item()):.4f}",
            ])

            if ep % a.log_every == 0:
                avg_r = sum(reward_win) / len(reward_win)
                vr    = sum(valid_win) / len(valid_win)
                self.logger.info(
                    f"  [round {round_idx} | {side:8}] ep {ep:5d}/{episodes}  "
                    f"atk_reward={avg_r:.4f}  valid={vr*100:.1f}%")
                if use_tracking:
                    mlflow.log_metric(f"{side}_phase_atk_reward", avg_r, step=self.global_episode)
                    mlflow.log_metric(f"{side}_phase_valid_rate", vr,   step=self.global_episode)

                # Early stop: attacker has collapsed → let defender respond now
                if (side == "attacker"
                        and a.min_valid > 0.0
                        and len(valid_win) >= a.log_every
                        and vr < a.min_valid):
                    self.logger.info(
                        f"  [round {round_idx} | attacker] valid={vr*100:.1f}% < "
                        f"{a.min_valid*100:.0f}% threshold — ending phase early at ep {ep}")
                    break

        return {
            "avg_attacker_reward": sum(reward_win) / len(reward_win) if reward_win else 0.0,
            "valid_rate":          sum(valid_win)  / len(valid_win)  if valid_win  else 0.0,
        }

    # ------------------------------------------------------------------ #

    def evaluate(self, episodes: int) -> dict:
        """Freeze both sides and play `episodes` head-to-head games."""
        n_valid, total_reward = 0, 0.0
        n = 0
        for _ in range(episodes):
            out = self._episode(train_side="eval")  # both frozen paths
            if out is None:
                continue
            attacker_reward, valid, *_rest = out
            n += 1
            n_valid      += int(valid)
            total_reward += attacker_reward
        return {
            "eval_valid_rate": n_valid / n if n else 0.0,
            "eval_atk_reward": total_reward / n if n else 0.0,
            "n": n,
        }

    def save(self, round_idx: int | str) -> tuple[Path, Path]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        atk_path = OUTPUT_DIR / f"attacker_round{round_idx}.pt"
        def_path = OUTPUT_DIR / f"defender_round{round_idx}.pt"
        torch.save({"round": round_idx, "model_state": self.attacker.state_dict(),
                    "args": vars(self.args)}, atk_path)
        torch.save({"round": round_idx, "model_state": self.defender.state_dict(),
                    "args": vars(self.args)}, def_path)
        return atk_path, def_path


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"train_adversarial_seed{args.seed}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logger = logging.getLogger(__name__)
    logger.info(f"Adversarial co-training | rounds={args.rounds} "
                f"episodes_per_phase={args.x} seed={args.seed}")

    use_tracking = setup_mlflow(args.mlflow)
    if use_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "rounds":             args.rounds,
            "episodes_per_phase": args.x,
            "lr_attacker":        args.lr_attacker,
            "lr_defender":        args.lr_defender,
            "entropy_coef":          args.entropy_coef,
            "entropy_coef_defender": args.entropy_coef_defender,
            "mix_random":            args.mix_random,
            "atk_temp":           args.atk_temp,
            "def_temp":           args.def_temp,
            "w_grammar":          args.w_grammar,
            "w_tag":              args.w_tag,
            "w_axis":             args.w_axis,
            "seed":               args.seed,
            "attacker_ckpt":      args.attacker_ckpt or str(DEFAULT_ATTACKER_CKPT.name),
            "defender_ckpt":      args.defender_ckpt or str(DEFAULT_DEFENDER_CKPT.name),
        })

    trainer = AdversarialCoTrainer(args, logger)

    episodes_csv = logs_dir / f"adversarial_episodes_seed{args.seed}.csv"
    csv_file = episodes_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["global_episode", "round", "trained_side",
                         "prefix", "completion", "valid", "attacker_reward", "loss"])

    # --- Round 0 baseline evaluation ---
    ev = trainer.evaluate(args.eval_episodes)
    logger.info(f"[round 0 | baseline ] valid={ev['eval_valid_rate']*100:.1f}%  "
                f"atk_reward={ev['eval_atk_reward']:.4f}  (n={ev['n']})")
    if use_tracking:
        mlflow.log_metric("round_valid_rate", ev["eval_valid_rate"], step=0)
        mlflow.log_metric("round_atk_reward", ev["eval_atk_reward"], step=0)

    t0 = time.time()
    for r in range(1, args.rounds + 1):
        logger.info(f"=== ROUND {r}/{args.rounds} — attacker phase ({args.x} episodes) ===")
        trainer.train_phase("attacker", args.x, r, csv_writer, use_tracking)

        logger.info(f"=== ROUND {r}/{args.rounds} — defender phase ({args.x} episodes) ===")
        trainer.train_phase("defender", args.x, r, csv_writer, use_tracking)

        # --- end-of-round head-to-head evaluation ---
        ev = trainer.evaluate(args.eval_episodes)
        logger.info(f"[round {r} | eval     ] valid={ev['eval_valid_rate']*100:.1f}%  "
                    f"atk_reward={ev['eval_atk_reward']:.4f}  (n={ev['n']})  "
                    f"elapsed={time.time()-t0:.0f}s")
        if use_tracking:
            mlflow.log_metric("round_valid_rate", ev["eval_valid_rate"], step=r)
            mlflow.log_metric("round_atk_reward", ev["eval_atk_reward"], step=r)

        atk_path, def_path = trainer.save(r)
        logger.info(f"  saved {atk_path.name}, {def_path.name}")

    # --- final checkpoints ---
    atk_path, def_path = trainer.save("_final")
    csv_file.close()
    logger.info(f"\nCo-training complete. Final checkpoints:")
    logger.info(f"  {atk_path}")
    logger.info(f"  {def_path}")

    # --- sample games with the final pair ---
    logger.info("\n--- Final attacker vs final defender (8 samples) ---")
    for _ in range(8):
        out = trainer._episode(train_side="eval")
        if out is None:
            continue
        reward, valid, _, _, _, prefix_words, suffix_words = out
        tag = "OK " if valid else "BAD"
        logger.info(f"  [{tag}] {' '.join(prefix_words)} | {' '.join(suffix_words)}  "
                    f"(R={reward:.3f})")

    if use_tracking:
        mlflow.log_artifact(str(episodes_csv))
        mlflow.log_artifact(str(atk_path))
        mlflow.log_artifact(str(def_path))
        mlflow.end_run()


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alternating adversarial co-training of attacker and defender")
    p.add_argument("--rounds",         type=int,   default=5,
                   help="Number of alternation rounds (default: 5)")
    p.add_argument("--episodes-per-phase", "-x", type=int, default=1000, dest="x",
                   help="Episodes per phase: attacker trains x, then defender trains x (default: 1000)")
    p.add_argument("--eval-episodes",  type=int,   default=100, dest="eval_episodes",
                   help="Frozen head-to-head games after each round (default: 100)")
    p.add_argument("--lr-attacker",    type=float, default=3e-4, dest="lr_attacker")
    p.add_argument("--lr-defender",    type=float, default=5e-5, dest="lr_defender",
                   help="Lower LR for the defender -- it is fine-tuned, not trained from scratch")
    p.add_argument("--entropy-coef",   type=float, default=0.05, dest="entropy_coef",
                   help="Attacker entropy bonus -- prevents mode collapse (default: 0.05)")
    p.add_argument("--entropy-coef-defender", type=float, default=0.01, dest="entropy_coef_defender",
                   help="Defender entropy bonus -- prevents mode collapse (default: 0.01)")
    p.add_argument("--mix-random",     type=float, default=0.3,  dest="mix_random",
                   help="Fraction of defender episodes using random CFG prefixes (default: 0.3)")
    p.add_argument("--max-prefix",     type=int,   default=6,    dest="max_prefix")
    p.add_argument("--max-completion", type=int,   default=20,   dest="max_completion")
    p.add_argument("--min-new-tokens",      type=int,   default=4,   dest="min_new_tokens",
                   help="Minimum new tokens the defender must generate before EOS is allowed (default: 4)")
    p.add_argument("--repetition-penalty",    type=float, default=1.3, dest="repetition_penalty",
                   help="Penalize repeated tokens in defender completion (1.0=off, default: 1.3)")
    p.add_argument("--no-repeat-ngram-size",  type=int,   default=2,   dest="no_repeat_ngram_size",
                   help="Hard-block repeated n-grams in defender completion (0=off, default: 2)")
    p.add_argument("--atk-temp",       type=float, default=1.0,  dest="atk_temp")
    p.add_argument("--def-temp",       type=float, default=0.8,  dest="def_temp")
    p.add_argument("--w-grammar",      type=float, default=1.0,  dest="w_grammar")
    p.add_argument("--w-tag",          type=float, default=0.30, dest="w_tag")
    p.add_argument("--w-axis",         type=float, default=0.20, dest="w_axis")
    p.add_argument("--w-repeat",       type=float, default=1.0,  dest="w_repeat",
                   help="Repetition penalty weight -- heavily penalizes the defender for "
                        "repeating words in its completion (default: 1.0)")
    p.add_argument("--min-valid",       type=float, default=0.0,  dest="min_valid",
                   help="Early-stop attacker phase if rolling valid rate drops below this (default: 0 = disabled)")
    p.add_argument("--baseline-alpha", type=float, default=0.05, dest="baseline_alpha")
    p.add_argument("--grad-clip",      type=float, default=1.0,  dest="grad_clip")
    p.add_argument("--window",         type=int,   default=100)
    p.add_argument("--log-every",      type=int,   default=100,  dest="log_every")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mlflow",         action="store_true")
    p.add_argument("--device",         choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--attacker-ckpt",  type=str, default=None, dest="attacker_ckpt",
                   help=f"Starting attacker (default: {DEFAULT_ATTACKER_CKPT})")
    p.add_argument("--defender-ckpt",  type=str, default=None, dest="defender_ckpt",
                   help=f"Starting defender (default: {DEFAULT_DEFENDER_CKPT})")
    p.add_argument("--corpus",         type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

"""Fine-tune the trained defender (MiniGPT) with REINFORCE against a frozen attacker.

This does NOT train a new defender. It loads an already-trained defender
checkpoint and modifies its weights with policy-gradient RL so it stops
making mistakes on the attacker's prefixes.

Roles:
  * Attacker (FROZEN)  -- the best RL-trained attacker. Generates CFG-valid
    prefixes. No gradients.
  * Defender (TRAINED) -- completes each prefix token by token. Each chosen
    token's log-probability is tracked so gradients flow.

Reward — same system the attacker was trained with (RewardFunction):

    R_attacker = w_grammar * grammar_failure
               + w_tag     * tag_mismatch
               + w_axis    * axis_distance

The defender wants the OPPOSITE outcome, so its reward is the negative:

    R_defender = -R_attacker

REINFORCE update per episode (EMA baseline for variance reduction):

    advantage = R_defender - baseline
    loss      = -(advantage * sum(log_probs of defender's tokens))

Episode mixing: with probability --mix-random the prefix is drawn uniformly
from the CFG instead of the attacker. The frozen attacker is mode-collapsed
(it repeats one exploit prefix), so training only on it would make the
defender forget how to complete everything else.

Usage:
    python scripts/train/train_defender_rl.py --episodes 1000
    python scripts/train/train_defender_rl.py --episodes 10000 --mlflow
    python scripts/train/train_defender_rl.py --episodes 5000 --mix-random 0.0
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

DEFAULT_DEFENDER_CKPT = PROJECT_ROOT / "data" / "models" / "from_mlflow" / "minigpt_corpus10000.pt"
DEFAULT_ATTACKER_CKPT = PROJECT_ROOT / "data" / "models" / "from_mlflow" / "attacker_best.pt"


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
    mlflow.set_experiment("DefenderRL")
    return True


# ------------------------------------------------------------------ #
#  Prefix sources                                                      #
# ------------------------------------------------------------------ #

def sample_random_prefix(
    tracker: CFGStateTracker,
    token_to_id: dict,
    max_tokens: int,
    stop_prob: float = 0.3,
) -> tuple[list[int], list[str]]:
    """Uniformly sample a CFG-valid prefix (exploration / anti-forgetting)."""
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


# ------------------------------------------------------------------ #
#  Defender completion with gradient-tracked log probs                  #
# ------------------------------------------------------------------ #

def _no_repeat_ngram_block(logits: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i: i + ngram_size - 1]) == prefix:
            logits[0, generated_ids[i + ngram_size - 1]] = float("-inf")
    return logits


def complete_with_log_probs(
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
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Autoregressive completion where each sampled token keeps its log-prob
    and the full per-step entropy with gradients attached (REINFORCE trajectory).

    EOS is masked until `min_new_tokens` words have been added — mirrors the
    AttackPipeline environment. The EOS choice itself is also a tracked action.
    Returns (full_ids_without_bos, log_probs_tensor, entropies_tensor).
    """
    all_ids   = [bos_id] + list(prefix_ids)
    log_probs = []
    entropies = []
    new_count = 0

    for _ in range(max_new_tokens):
        context = torch.tensor([all_ids[-defender.context_len:]], device=device)
        logits  = defender(context)[:, -1, :] / temperature

        if repetition_penalty != 1.0:
            for tid in set(all_ids[1:]):
                logits[0, tid] /= repetition_penalty
        logits = _no_repeat_ngram_block(logits, all_ids[1:], no_repeat_ngram_size)

        if new_count < min_new_tokens:
            mask = torch.zeros_like(logits)
            mask[0, eos_id] = float("-inf")
            logits = logits + mask

        log_dist = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits.detach(), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        log_probs.append(log_dist[0, next_id])
        # True per-step entropy: H = -sum(p * log p); nan_to_num handles 0*-inf (masked EOS)
        entropies.append(-(probs * log_dist).nan_to_num(0.0).sum())

        if next_id == eos_id:
            break
        all_ids.append(next_id)
        new_count += 1

    lp  = torch.stack(log_probs) if log_probs else torch.zeros(0, device=device)
    ent = torch.stack(entropies) if entropies else torch.zeros(0, device=device)
    return all_ids[1:], lp, ent


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logs_dir   = PROJECT_ROOT / "logs"
    output_dir = PROJECT_ROOT / "data" / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"train_defender_rl_seed{args.seed}.log"
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
    logger.info(f"Defender REINFORCE | episodes={args.episodes} seed={args.seed} "
                f"lr={args.lr} mix_random={args.mix_random}")

    use_tracking = setup_mlflow(args.mlflow)

    # --- Language stack ---
    logger.info("Loading lexicon, CFG, validator...")
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg       = CFG.from_json(str(TRANSITION_PATH), nouns=nouns, verbs=verbs, adjectives=adjectives)
    validator = CFGValidator.from_cfg(cfg)
    tracker   = CFGStateTracker(nouns, verbs, adjectives)

    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # --- Defender: load TRAINED checkpoint (this is the model being fine-tuned) ---
    defender_ckpt_path = Path(args.defender_ckpt) if args.defender_ckpt else DEFAULT_DEFENDER_CKPT
    if not defender_ckpt_path.exists():
        raise FileNotFoundError(f"Defender checkpoint not found: {defender_ckpt_path}")
    logger.info(f"Loading defender (to fine-tune): {defender_ckpt_path}")
    d_ckpt = torch.load(defender_ckpt_path, map_location=device, weights_only=False)
    defender = MiniGPT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    defender.load_state_dict(d_ckpt["model_state"])
    defender.eval()  # disable dropout; gradients still flow
    logger.info(f"  base checkpoint: epoch={d_ckpt.get('epoch','?')} loss={d_ckpt.get('loss', float('nan')):.4f}")

    # --- Attacker: FROZEN best checkpoint ---
    attacker_ckpt_path = Path(args.attacker_ckpt) if args.attacker_ckpt else DEFAULT_ATTACKER_CKPT
    if not attacker_ckpt_path.exists():
        raise FileNotFoundError(f"Attacker checkpoint not found: {attacker_ckpt_path}")
    logger.info(f"Loading attacker (frozen): {attacker_ckpt_path}")
    a_ckpt = torch.load(attacker_ckpt_path, map_location=device, weights_only=False)
    attacker = AttackerTransformer(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    attacker.load_state_dict(a_ckpt["model_state"])
    attacker.eval()
    for p in attacker.parameters():
        p.requires_grad = False
    logger.info(f"  attacker avg_reward at save: {a_ckpt.get('avg_reward', float('nan')):.4f}")

    # --- Reward (same system as the attacker's) ---
    reward_fn = RewardFunction(
        nouns=nouns, verbs=verbs, adjectives=adjectives,
        weights=RewardWeights(w_grammar=args.w_grammar, w_tag=args.w_tag, w_axis=args.w_axis,
                              w_repeat=args.w_repeat),
    )

    optimizer = torch.optim.AdamW(defender.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if use_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "episodes":       args.episodes,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "max_prefix":     args.max_prefix,
            "max_completion": args.max_completion,
            "atk_temp":       args.atk_temp,
            "def_temp":       args.def_temp,
            "mix_random":     args.mix_random,
            "w_grammar":      args.w_grammar,
            "w_tag":          args.w_tag,
            "w_axis":         args.w_axis,
            "entropy_coef":   args.entropy_coef,
            "baseline_alpha": args.baseline_alpha,
            "seed":           args.seed,
            "defender_ckpt":  str(defender_ckpt_path.name),
            "attacker_ckpt":  str(attacker_ckpt_path.name),
            "base_loss":      float(d_ckpt.get("loss", float("nan"))),
        })

    episodes_csv = logs_dir / f"defender_rl_episodes_seed{args.seed}.csv"
    csv_file = episodes_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "source", "prefix", "completion", "valid",
        "attacker_reward", "defender_reward", "loss", "baseline",
    ])

    # --- REINFORCE loop ---
    baseline    = 0.0
    alpha       = args.baseline_alpha
    reward_win  = deque(maxlen=args.window)   # defender rewards
    valid_win   = deque(maxlen=args.window)
    loss_win    = deque(maxlen=args.window)
    best_avg    = float("-inf")
    best_path   = output_dir / "defender_rl_best.pt"
    final_path  = output_dir / "defender_rl_final.pt"

    logger.info("Starting defender REINFORCE loop...")
    t_start = time.time()

    for ep in range(1, args.episodes + 1):
        # --- Step 1: prefix (frozen attacker, or random CFG for diversity) ---
        if random.random() < args.mix_random:
            source = "random"
            prefix_ids, prefix_words = sample_random_prefix(
                tracker, tokenizer.token_to_id, args.max_prefix)
        else:
            source = "attacker"
            with torch.no_grad():
                prefix_ids, prefix_words = attacker.generate_prefix(
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    cfg_tracker=tracker,
                    token_to_id=tokenizer.token_to_id,
                    id_to_token=tokenizer.id_to_token,
                    max_tokens=args.max_prefix,
                    temperature=args.atk_temp,
                    device=str(device),
                )
        if not prefix_ids:
            continue

        # --- Step 2: defender completes WITH log probs (gradients) ---
        full_ids, log_probs, entropies = complete_with_log_probs(
            defender, prefix_ids,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_completion,
            temperature=args.def_temp,
            device=device,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        if log_probs.numel() == 0:
            continue

        full_words    = tokenizer.decode(full_ids).split()
        full_sentence = " ".join(full_words)

        # --- Step 3: validate + reward (same system as attacker) ---
        result = validator.validate(full_sentence)
        prefix_part, suffix_part = reward_fn.split_sentence(full_words, prefix_words)
        rw = reward_fn.compute(
            prefix_words=prefix_part,
            suffix_words=suffix_part,
            full_sentence=full_sentence,
            is_valid=result.is_valid,
            cfg_error=result.error if not result.is_valid else None,
        )
        attacker_reward = float(rw.reward)
        defender_reward = -attacker_reward   # defender minimizes the attacker's reward

        # --- Step 4: REINFORCE update with entropy bonus ---
        advantage = defender_reward - baseline
        policy_loss = -(advantage * log_probs.sum())
        entropy_term = entropies.mean() if entropies.numel() > 0 else torch.tensor(0.0, device=device)
        loss = policy_loss - args.entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(defender.parameters(), args.grad_clip)
        optimizer.step()

        baseline = (1.0 - alpha) * baseline + alpha * defender_reward

        # --- Bookkeeping ---
        reward_win.append(defender_reward)
        valid_win.append(1.0 if result.is_valid else 0.0)
        loss_win.append(float(loss.item()))

        csv_writer.writerow([
            ep, source,
            " ".join(prefix_words),
            " ".join(suffix_part),
            int(result.is_valid),
            f"{attacker_reward:.4f}",
            f"{defender_reward:.4f}",
            f"{float(loss.item()):.4f}",
            f"{baseline:.4f}",
        ])

        if ep % args.log_every == 0:
            avg_rw     = sum(reward_win) / len(reward_win)
            valid_rate = sum(valid_win) / len(valid_win)
            avg_loss   = sum(loss_win) / len(loss_win)
            elapsed    = time.time() - t_start

            logger.info(
                f"ep {ep:6d}/{args.episodes}  "
                f"def_reward={avg_rw:.4f}  "
                f"valid_rate={valid_rate*100:.1f}%  "
                f"baseline={baseline:.4f}  "
                f"avg_loss={avg_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if use_tracking:
                mlflow.log_metric("avg_defender_reward", avg_rw,     step=ep)
                mlflow.log_metric("valid_rate",          valid_rate, step=ep)
                mlflow.log_metric("baseline",            baseline,   step=ep)
                mlflow.log_metric("avg_loss",            avg_loss,   step=ep)

            if len(reward_win) == args.window and avg_rw > best_avg:
                best_avg = avg_rw
                torch.save({
                    "episode":     ep,
                    "model_state": defender.state_dict(),
                    "avg_defender_reward": avg_rw,
                    "valid_rate":  valid_rate,
                    "base_ckpt":   str(defender_ckpt_path.name),
                    "args":        vars(args),
                }, best_path)
                logger.info(f"[BEST] def_reward={avg_rw:.4f} valid={valid_rate*100:.0f}% -> {best_path.name}")
                if use_tracking:
                    mlflow.log_metric("best_avg_defender_reward", best_avg, step=ep)

    # --- Final checkpoint ---
    torch.save({
        "episode":     args.episodes,
        "model_state": defender.state_dict(),
        "base_ckpt":   str(defender_ckpt_path.name),
        "args":        vars(args),
    }, final_path)
    logger.info(f"Final defender saved -> {final_path.name}")
    csv_file.close()

    # --- Show how the fine-tuned defender now answers the attacker ---
    logger.info("\n--- Fine-tuned defender vs frozen attacker (10 samples) ---")
    n_valid = 0
    for i in range(10):
        with torch.no_grad():
            p_ids, p_words = attacker.generate_prefix(
                bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                cfg_tracker=tracker,
                token_to_id=tokenizer.token_to_id, id_to_token=tokenizer.id_to_token,
                max_tokens=args.max_prefix, temperature=args.atk_temp, device=str(device),
            )
        f_ids, _, _ = complete_with_log_probs(
            defender, p_ids, tokenizer.bos_id, tokenizer.eos_id,
            args.max_completion, args.def_temp, device,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size)
        sent = tokenizer.decode(f_ids)
        ok = validator.validate(sent).is_valid
        n_valid += ok
        logger.info(f"  [{'OK ' if ok else 'BAD'}] {sent}")
    logger.info(f"  valid: {n_valid}/10")

    if use_tracking:
        mlflow.log_metric("final_sample_valid_rate", n_valid / 10)
        mlflow.log_artifact(str(episodes_csv))
        mlflow.log_artifact(str(final_path))
        if best_path.exists():
            mlflow.log_artifact(str(best_path))
        mlflow.end_run()


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REINFORCE fine-tuning for the defender against a frozen attacker")
    p.add_argument("--episodes",       type=int,   default=10000)
    p.add_argument("--lr",             type=float, default=5e-5,
                   help="Low LR -- we are fine-tuning a trained model, not training from scratch")
    p.add_argument("--weight-decay",   type=float, default=0.0,  dest="weight_decay")
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
    p.add_argument("--mix-random",     type=float, default=0.3,  dest="mix_random",
                   help="Fraction of episodes that use a random CFG prefix instead of the "
                        "attacker (the collapsed attacker repeats one prefix; mixing keeps "
                        "the defender general). 0 = attacker prefixes only.")
    p.add_argument("--w-grammar",      type=float, default=1.0,  dest="w_grammar")
    p.add_argument("--w-tag",          type=float, default=0.30, dest="w_tag")
    p.add_argument("--w-axis",         type=float, default=0.20, dest="w_axis")
    p.add_argument("--w-repeat",       type=float, default=1.0,  dest="w_repeat",
                   help="Repetition penalty weight -- heavily penalizes the defender for "
                        "repeating words in its completion (default: 1.0)")
    p.add_argument("--entropy-coef",   type=float, default=0.01, dest="entropy_coef",
                   help="Defender entropy bonus -- prevents mode collapse (default: 0.01)")
    p.add_argument("--baseline-alpha", type=float, default=0.05, dest="baseline_alpha")
    p.add_argument("--grad-clip",      type=float, default=1.0,  dest="grad_clip")
    p.add_argument("--window",         type=int,   default=100)
    p.add_argument("--log-every",      type=int,   default=100,  dest="log_every")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mlflow",         action="store_true")
    p.add_argument("--device",         choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--defender-ckpt",  type=str, default=None, dest="defender_ckpt",
                   help=f"Defender checkpoint to fine-tune (default: {DEFAULT_DEFENDER_CKPT.name} from MLflow)")
    p.add_argument("--attacker-ckpt",  type=str, default=None, dest="attacker_ckpt",
                   help=f"Frozen attacker checkpoint (default: {DEFAULT_ATTACKER_CKPT.name} from MLflow)")
    p.add_argument("--corpus",         type=str, default=None,
                   help="Corpus file for the tokenizer (default: corpus_10000.txt)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

"""Train the defender (MiniGPT) with REINFORCE policy gradient.

The defender completes attacker-generated prefixes token-by-token. The attacker
is frozen and serves as a deterministic environment. The defender reward
mirrors the attacker reward with inverted objectives:

    R = w_grammar * grammar_success
      + w_tag     * tag_closeness      (1 - tag_mismatch)
      + w_axis    * axis_closeness     (1 - axis_distance)

REINFORCE update (per episode):

    advantage = R - baseline      (EMA baseline)
    loss      = -(advantage * sum(log_probs))
    loss.backward()

No gradients flow through the attacker.

Usage:
    python scripts/train/train_defender.py --episodes 2000
    python scripts/train/train_defender.py --episodes 5000 --mlflow
    python scripts/train/train_defender.py --episodes 1000 --lr 1e-4
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacker.attacker import AttackerTransformer
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.defender.reward_function import DefenderRewardFunction, DefenderRewardWeights
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
DEFAULT_DEFENDER_CKPT = PROJECT_ROOT / "data" / "models" / "minigpt_corpus10000.pt"
DEFAULT_ATTACKER_CKPT = PROJECT_ROOT / "data" / "models" / "attacker_best.pt"


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
            local_uri = (PROJECT_ROOT / "mlruns").as_uri()
            mlflow.set_tracking_uri(local_uri)
            print(f"[WARN] DagsHub unreachable ({type(e).__name__}). "
                  f"Falling back to LOCAL MLflow tracking.")
    else:
        print("[WARN] MLflow enabled (local tracking only).")

    mlflow.set_experiment("DefenderREINFORCE")
    return True


def load_attacker(
    attacker: AttackerTransformer,
    ckpt_path: Path | None,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Load attacker weights if checkpoint exists; otherwise keep random init."""
    if ckpt_path and ckpt_path.exists():
        logger.info(f"Loading attacker checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        attacker.load_state_dict(ckpt["model_state"])
    else:
        if ckpt_path:
            logger.warning(f"Attacker checkpoint not found ({ckpt_path}); using random init.")
        else:
            logger.info("No attacker checkpoint specified; using random init.")


def train(args: argparse.Namespace) -> None:
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"train_defender_seed{args.seed}.log"
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
    logger.info(f"REINFORCE defender training | episodes={args.episodes} seed={args.seed}")

    use_tracking = setup_mlflow(args.mlflow)

    logger.info("Loading lexicon and CFG...")
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        str(TRANSITION_PATH),
        nouns=nouns, verbs=verbs, adjectives=adjectives,
    )
    validator = CFGValidator.from_cfg(cfg)
    tracker   = CFGStateTracker(nouns, verbs, adjectives)

    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    logger.info(f"Loading tokenizer from {corpus_path.name}...")
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    defender_ckpt = Path(args.defender_ckpt) if args.defender_ckpt else DEFAULT_DEFENDER_CKPT
    defender = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    ).to(device)
    if defender_ckpt.exists():
        logger.info(f"Loading defender init checkpoint: {defender_ckpt.name}")
        ckpt = torch.load(defender_ckpt, map_location=device, weights_only=False)
        defender.load_state_dict(ckpt["model_state"])
    else:
        logger.warning(f"Defender init checkpoint not found ({defender_ckpt}); training from scratch.")

    n_params = sum(p.numel() for p in defender.parameters() if p.requires_grad)
    logger.info(f"Defender trainable params: {n_params:,}")

    attacker = AttackerTransformer(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    ).to(device)
    attacker_ckpt = Path(args.attacker_ckpt) if args.attacker_ckpt else DEFAULT_ATTACKER_CKPT
    load_attacker(attacker, attacker_ckpt if attacker_ckpt.exists() else None, device, logger)
    attacker.eval()
    for p in attacker.parameters():
        p.requires_grad = False

    reward_fn = DefenderRewardFunction(
        nouns=nouns, verbs=verbs, adjectives=adjectives,
        weights=DefenderRewardWeights(
            w_grammar=args.w_grammar,
            w_tag=args.w_tag,
            w_axis=args.w_axis,
        ),
    )

    optimizer = torch.optim.AdamW(
        defender.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if use_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "episodes":         args.episodes,
            "lr":               args.lr,
            "weight_decay":     args.weight_decay,
            "max_prefix":       args.max_prefix,
            "max_completion":   args.max_completion,
            "atk_temp":         args.atk_temp,
            "def_temp":         args.def_temp,
            "w_grammar":        args.w_grammar,
            "w_tag":            args.w_tag,
            "w_axis":           args.w_axis,
            "baseline_alpha":   args.baseline_alpha,
            "entropy_coef":     args.entropy_coef,
            "seed":             args.seed,
            "param_count":      n_params,
        })

    episodes_csv = logs_dir / f"defender_episodes_seed{args.seed}.csv"
    csv_file = episodes_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "prefix", "completion", "valid", "reward",
        "grammar_reward", "tag_reward", "axis_reward",
        "noun_tag_dist", "verb_tag_dist", "adj_tag_dist",
        "loss", "baseline",
    ])

    baseline   = 0.0
    alpha      = args.baseline_alpha
    window     = deque(maxlen=args.window)
    valid_win  = deque(maxlen=args.window)
    loss_win   = deque(maxlen=args.window)
    best_avg   = float("-inf")
    best_path  = output_dir / "defender_rl_best.pt"
    final_path = output_dir / "defender_rl_final.pt"

    logger.info("Starting REINFORCE training loop...")
    t_start = time.time()

    for ep in range(1, args.episodes + 1):
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

        full_ids, log_probs = defender.complete_with_log_probs(
            prefix_ids=prefix_ids,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_completion,
            temperature=args.def_temp,
            device=str(device),
            min_new_tokens=1,
        )

        if log_probs.numel() == 0:
            continue

        full_words    = tokenizer.decode(full_ids).split()
        full_sentence = " ".join(full_words)
        result        = validator.validate(full_sentence)

        prefix_part, suffix_part = reward_fn.split_sentence(full_words, prefix_words)
        rw = reward_fn.compute(
            prefix_words  = prefix_part,
            suffix_words  = suffix_part,
            full_sentence = full_sentence,
            is_valid      = result.is_valid,
            cfg_error     = result.error if not result.is_valid else None,
        )
        reward = float(rw.reward)

        advantage   = reward - baseline
        policy_loss = -(advantage * log_probs.sum())
        entropy_term = -log_probs.mean() if args.entropy_coef > 0 else torch.tensor(0.0, device=device)
        loss = policy_loss - args.entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(defender.parameters(), args.grad_clip)
        optimizer.step()

        baseline = (1.0 - alpha) * baseline + alpha * reward

        window.append(reward)
        valid_win.append(1.0 if result.is_valid else 0.0)
        loss_win.append(float(loss.item()))

        csv_writer.writerow([
            ep,
            " ".join(prefix_words),
            " ".join(suffix_part),
            int(result.is_valid),
            f"{reward:.4f}",
            f"{rw.grammar_reward:.4f}",
            f"{rw.tag_reward:.4f}",
            f"{rw.axis_reward:.4f}",
            f"{rw.distances.noun_tag_dist:.4f}",
            f"{rw.distances.verb_tag_dist:.4f}",
            f"{rw.distances.adjective_tag_dist:.4f}",
            f"{float(loss.item()):.4f}",
            f"{baseline:.4f}",
        ])

        if ep % args.log_every == 0:
            avg_rw      = sum(window) / len(window)
            valid_rate  = sum(valid_win) / len(valid_win)
            avg_loss    = sum(loss_win) / len(loss_win)
            elapsed     = time.time() - t_start

            logger.info(
                f"ep {ep:5d}/{args.episodes}  "
                f"avg_reward={avg_rw:.4f}  "
                f"grammar_valid={valid_rate*100:.1f}%  "
                f"baseline={baseline:.4f}  "
                f"avg_loss={avg_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if use_tracking:
                mlflow.log_metric("avg_reward",       avg_rw,      step=ep)
                mlflow.log_metric("grammar_valid_rate", valid_rate, step=ep)
                mlflow.log_metric("baseline",         baseline,    step=ep)
                mlflow.log_metric("avg_loss",         avg_loss,    step=ep)
                mlflow.log_metric("episode_reward",   reward,      step=ep)

            if len(window) == args.window and avg_rw > best_avg:
                best_avg = avg_rw
                torch.save({
                    "episode":     ep,
                    "model_state": defender.state_dict(),
                    "avg_reward":  avg_rw,
                    "args":        vars(args),
                }, best_path)
                logger.info(f"[BEST] avg_reward={avg_rw:.4f} -> {best_path.name}")
                if use_tracking:
                    mlflow.log_metric("best_avg_reward", best_avg, step=ep)

    torch.save({
        "episode":     args.episodes,
        "model_state": defender.state_dict(),
        "args":        vars(args),
    }, final_path)
    logger.info(f"Final defender saved -> {final_path.name}")
    csv_file.close()

    logger.info("\n--- Sample completions from trained defender ---")
    defender.eval()
    for i in range(5):
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
            full_ids, _ = defender.complete_with_log_probs(
                prefix_ids=prefix_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_new_tokens=args.max_completion,
                temperature=args.def_temp,
                device=str(device),
            )
            suffix_words = tokenizer.decode(full_ids[len(prefix_ids):]).split()
        logger.info(f"  [{i+1}] {' '.join(prefix_words)} | {' '.join(suffix_words)}")

    if use_tracking:
        mlflow.log_artifact(str(episodes_csv))
        mlflow.log_artifact(str(final_path))
        if best_path.exists():
            mlflow.log_artifact(str(best_path))
        mlflow.end_run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REINFORCE training for the defender (MiniGPT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--episodes",       type=int,   default=2000)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight-decay",   type=float, default=0.0, dest="weight_decay")
    p.add_argument("--max-prefix",     type=int,   default=6,   dest="max_prefix")
    p.add_argument("--max-completion", type=int,   default=15,  dest="max_completion")
    p.add_argument("--atk-temp",       type=float, default=1.0, dest="atk_temp")
    p.add_argument("--def-temp",       type=float, default=0.8, dest="def_temp")
    p.add_argument("--w-grammar",      type=float, default=1.0, dest="w_grammar")
    p.add_argument("--w-tag",          type=float, default=0.30, dest="w_tag")
    p.add_argument("--w-axis",         type=float, default=0.20, dest="w_axis")
    p.add_argument("--baseline-alpha", type=float, default=0.05, dest="baseline_alpha")
    p.add_argument("--entropy-coef",   type=float, default=0.0, dest="entropy_coef")
    p.add_argument("--grad-clip",      type=float, default=1.0, dest="grad_clip")
    p.add_argument("--window",         type=int,   default=100)
    p.add_argument("--log-every",      type=int,   default=50, dest="log_every")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mlflow",         action="store_true")
    p.add_argument("--device",         choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--defender-ckpt",  type=str,   default=None, dest="defender_ckpt",
                   help="Init checkpoint for defender (default: minigpt_corpus10000.pt)")
    p.add_argument("--attacker-ckpt",  type=str,   default=None, dest="attacker_ckpt",
                   help="Frozen attacker checkpoint (default: attacker_best.pt if exists)")
    p.add_argument("--corpus",         type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

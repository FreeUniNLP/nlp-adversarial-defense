"""Train the adversarial attacker with REINFORCE policy gradient.

The attacker is a CFG-constrained causal Transformer that generates sentence
prefixes token-by-token. At each generation step, CFG-invalid tokens are
masked, so the policy only ever distributes probability over grammatically
legal actions. The defender (MiniGPT) is frozen and serves as a deterministic
environment: it completes the prefix into a full sentence. The reward function
scores the resulting sentence and returns a scalar signal:

    R = w_grammar * grammar_failure
      + w_tag     * tag_mismatch
      + w_axis    * axis_distance

REINFORCE update (per episode):

    advantage = R - baseline      (EMA baseline -- variance reduction)
    loss      = -(advantage * sum(log_probs))
    loss.backward()

No gradients flow through the defender.

Usage:
    python scripts/train/train_attacker.py --episodes 2000
    python scripts/train/train_attacker.py --episodes 5000 --mlflow
    python scripts/train/train_attacker.py --episodes 1000 --max-prefix 5 --lr 1e-4
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
from src.reward.reward_function import RewardFunction, RewardWeights
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


# ------------------------------------------------------------------ #
#  Experiment tracking                                                 #
# ------------------------------------------------------------------ #

def setup_mlflow(use_mlflow: bool) -> bool:
    """Initialize MLflow + DagsHub if requested. Mirrors train_model.py."""
    if not use_mlflow:
        return False
    if not HAS_MLFLOW:
        print("[WARN] --mlflow requested but 'mlflow' not installed.")
        return False

    repo_owner = DAGSHUB_REPO_OWNER or os.getenv("DAGSHUB_REPO_OWNER")
    repo_name  = DAGSHUB_REPO_NAME  or os.getenv("DAGSHUB_REPO_NAME")
    token      = DAGSHUB_TOKEN      or os.getenv("DAGSHUB_USER_TOKEN")

    if repo_owner and repo_name and HAS_DAGSHUB:
        # Wire the token into the env vars dagshub.init looks for. This avoids
        # an OAuth browser prompt when a token is present in config.py.
        if token:
            os.environ["DAGSHUB_USER_TOKEN"] = token
            os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print(f"[OK] Connected to DagsHub: {repo_owner}/{repo_name}")
            print(f"     MLflow URI: https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        except Exception as e:
            # DagsHub down / unreachable -> fall back to local MLflow tracking
            local_uri = (PROJECT_ROOT / "mlruns").as_uri()
            mlflow.set_tracking_uri(local_uri)
            print(f"[WARN] DagsHub unreachable ({type(e).__name__}). "
                  f"Falling back to LOCAL MLflow tracking.")
            print(f"  Local tracking dir: {PROJECT_ROOT / 'mlruns'}")
            print(f"  View with: mlflow ui --backend-store-uri \"{local_uri}\"")
    else:
        print("[WARN] MLflow enabled (local tracking only).")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_experiment("AttackerREINFORCE")
    return True


# ------------------------------------------------------------------ #
#  Defender completion (no grad)                                       #
# ------------------------------------------------------------------ #

def complete_with_defender(
    defender:        MiniGPT,
    prefix_ids:      list[int],
    bos_id:          int,
    eos_id:          int,
    max_new_tokens:  int,
    temperature:     float,
    device:          torch.device,
    min_new_tokens:  int = 1,
) -> list[int]:
    """Run the frozen defender to produce a completion. Returns full ids (BOS stripped).

    Mirrors AttackPipeline._complete: EOS is masked until the defender has
    added at least `min_new_tokens` new words, so the training environment
    matches the evaluation environment exactly.
    """
    all_ids   = [bos_id] + prefix_ids
    ctx_len   = defender.context_len
    new_count = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([all_ids[-ctx_len:]], device=device)
            logits  = defender(context)[:, -1, :] / temperature
            if new_count < min_new_tokens:
                logits[0, eos_id] = float("-inf")
            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == eos_id:
                break
            all_ids.append(next_id)
            new_count += 1
    return all_ids[1:]


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train(args: argparse.Namespace) -> None:
    # --- Device ---
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Logging setup ---
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"train_attacker_seed{args.seed}.log"
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
    logger.info(f"REINFORCE attacker training | episodes={args.episodes} seed={args.seed}")

    # --- MLflow ---
    use_tracking = setup_mlflow(args.mlflow)

    # --- Lexicon / CFG / validator ---
    logger.info("Loading lexicon and CFG...")
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        str(TRANSITION_PATH),
        nouns=nouns, verbs=verbs, adjectives=adjectives,
    )
    validator = CFGValidator.from_cfg(cfg)
    tracker   = CFGStateTracker(nouns, verbs, adjectives)

    # --- Tokenizer ---
    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    logger.info(f"Loading tokenizer from {corpus_path.name}...")
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # --- Defender (frozen) ---
    defender_ckpt = Path(args.defender_ckpt) if args.defender_ckpt else DEFAULT_DEFENDER_CKPT
    if not defender_ckpt.exists():
        raise FileNotFoundError(
            f"Defender checkpoint not found: {defender_ckpt}\n"
            f"Train one first with: python scripts/train/train_model.py"
        )
    logger.info(f"Loading defender checkpoint: {defender_ckpt.name}")
    ckpt = torch.load(defender_ckpt, map_location=device, weights_only=False)
    defender = MiniGPT.from_checkpoint(ckpt, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    defender.eval()
    for p in defender.parameters():
        p.requires_grad = False

    # --- Attacker (to train) ---
    attacker = AttackerTransformer(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in attacker.parameters() if p.requires_grad)
    logger.info(f"Attacker trainable params: {n_params:,}")

    # --- Reward function ---
    reward_fn = RewardFunction(
        nouns=nouns, verbs=verbs, adjectives=adjectives,
        weights=RewardWeights(
            w_grammar=args.w_grammar,
            w_tag=args.w_tag,
            w_axis=args.w_axis,
        ),
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        attacker.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- MLflow params ---
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

    # --- Episode CSV ---
    episodes_csv = logs_dir / f"attacker_episodes_seed{args.seed}.csv"
    csv_file = episodes_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "prefix", "completion", "valid", "reward",
        "grammar_reward", "tag_reward", "axis_reward",
        "noun_tag_dist", "verb_tag_dist", "adj_tag_dist",
        "loss", "baseline",
    ])

    # --- Training loop ---
    baseline      = 0.0
    alpha         = args.baseline_alpha
    window        = deque(maxlen=args.window)
    grammar_win   = deque(maxlen=args.window)
    loss_win      = deque(maxlen=args.window)
    best_avg      = float("-inf")
    best_path     = output_dir / "attacker_best.pt"
    final_path    = output_dir / "attacker_final.pt"

    logger.info("Starting REINFORCE training loop...")
    t_start = time.time()

    for ep in range(1, args.episodes + 1):
        # Step 1 -- attacker rolls out a prefix with log probs (gradient-tracked)
        prefix_ids, prefix_words, log_probs = attacker.generate_prefix_with_log_probs(
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            cfg_tracker=tracker,
            token_to_id=tokenizer.token_to_id,
            id_to_token=tokenizer.id_to_token,
            max_tokens=args.max_prefix,
            temperature=args.atk_temp,
            device=str(device),
        )

        if log_probs.numel() == 0:
            # Tracker hit a dead end before sampling any token; skip update.
            continue

        # Step 2 -- defender completes the prefix (no grad)
        full_ids   = complete_with_defender(
            defender, prefix_ids,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_completion,
            temperature=args.def_temp,
            device=device,
        )
        full_words    = tokenizer.decode(full_ids).split()
        full_sentence = " ".join(full_words)

        # Step 3 -- CFG validation
        result = validator.validate(full_sentence)

        # Step 4 -- reward
        prefix_part, suffix_part = reward_fn.split_sentence(full_words, prefix_words)
        rw = reward_fn.compute(
            prefix_words  = prefix_part,
            suffix_words  = suffix_part,
            full_sentence = full_sentence,
            is_valid      = result.is_valid,
            cfg_error     = result.error if not result.is_valid else None,
        )
        reward = float(rw.reward)

        # Step 5 -- REINFORCE update with EMA baseline
        advantage = reward - baseline
        # Sum over tokens in the trajectory; per-episode loss
        policy_loss = -(advantage * log_probs.sum())

        # Optional entropy bonus would require keeping the full distribution;
        # log_probs of the chosen action is the simplest surrogate.
        # entropy_term ~ -(log_probs.mean()) tends to discourage low-entropy choices.
        entropy_term = -log_probs.mean() if args.entropy_coef > 0 else torch.tensor(0.0, device=device)
        loss = policy_loss - args.entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(attacker.parameters(), args.grad_clip)
        optimizer.step()

        # Update EMA baseline AFTER computing advantage
        baseline = (1.0 - alpha) * baseline + alpha * reward

        # --- Bookkeeping ---
        window.append(reward)
        grammar_win.append(1.0 if not result.is_valid else 0.0)
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

        # --- Periodic logging ---
        if ep % args.log_every == 0:
            avg_rw    = sum(window) / len(window)
            fail_rate = sum(grammar_win) / len(grammar_win)
            avg_loss  = sum(loss_win) / len(loss_win)
            elapsed   = time.time() - t_start

            logger.info(
                f"ep {ep:5d}/{args.episodes}  "
                f"avg_reward={avg_rw:.4f}  "
                f"grammar_fail={fail_rate*100:.1f}%  "
                f"baseline={baseline:.4f}  "
                f"avg_loss={avg_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if use_tracking:
                mlflow.log_metric("avg_reward",        avg_rw,    step=ep)
                mlflow.log_metric("grammar_fail_rate", fail_rate, step=ep)
                mlflow.log_metric("baseline",          baseline,  step=ep)
                mlflow.log_metric("avg_loss",          avg_loss,  step=ep)
                mlflow.log_metric("episode_reward",    reward,    step=ep)

            # Save best
            if len(window) == args.window and avg_rw > best_avg:
                best_avg = avg_rw
                torch.save({
                    "episode":     ep,
                    "model_state": attacker.state_dict(),
                    "avg_reward":  avg_rw,
                    "args":        vars(args),
                }, best_path)
                logger.info(f"[BEST] avg_reward={avg_rw:.4f} -> {best_path.name}")
                if use_tracking:
                    mlflow.log_metric("best_avg_reward", best_avg, step=ep)

    # --- Final checkpoint ---
    torch.save({
        "episode":     args.episodes,
        "model_state": attacker.state_dict(),
        "args":        vars(args),
    }, final_path)
    logger.info(f"Final attacker saved -> {final_path.name}")
    csv_file.close()

    # --- Sample trained prefixes ---
    logger.info("\n--- Sample prefixes from trained attacker ---")
    attacker.eval()
    for i in range(5):
        ids, words = attacker.generate_prefix(
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            cfg_tracker=tracker,
            token_to_id=tokenizer.token_to_id,
            id_to_token=tokenizer.id_to_token,
            max_tokens=args.max_prefix,
            temperature=args.atk_temp,
            device=str(device),
        )
        logger.info(f"  [{i+1}] {' '.join(words)}")

    if use_tracking:
        mlflow.log_artifact(str(episodes_csv))
        mlflow.log_artifact(str(final_path))
        if best_path.exists():
            mlflow.log_artifact(str(best_path))
        mlflow.end_run()


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REINFORCE training for the adversarial attacker",
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
    p.add_argument("--baseline-alpha", type=float, default=0.05, dest="baseline_alpha",
                   help="EMA baseline smoothing rate")
    p.add_argument("--entropy-coef",   type=float, default=0.01, dest="entropy_coef",
                   help="Entropy bonus coefficient (encourage exploration)")
    p.add_argument("--grad-clip",      type=float, default=1.0, dest="grad_clip")
    p.add_argument("--window",         type=int,   default=100,
                   help="Rolling window size for avg-reward checkpointing")
    p.add_argument("--log-every",      type=int,   default=50, dest="log_every")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mlflow",         action="store_true")
    p.add_argument("--device",         choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--defender-ckpt",  type=str,   default=None, dest="defender_ckpt",
                   help="Path to defender checkpoint (default: data/models/minigpt_corpus10000.pt)")
    p.add_argument("--corpus",         type=str,   default=None,
                   help="Path to corpus file used to build the tokenizer (default: corpus_10000.txt)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

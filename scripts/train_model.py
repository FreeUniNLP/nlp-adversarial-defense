"""Train a small GPT-style decoder on the generated mini-language corpus.

Usage:
    python scripts/train_model.py                                    # local training only
    python scripts/train_model.py --corpus 5000 --epochs 20          # custom hyperparams
    python scripts/train_model.py --mlflow --corpus 10000            # with MLflow tracking
    DAGSHUB_REPO_OWNER=user DAGSHUB_REPO_NAME=repo python scripts/train_model.py --mlflow  # team logging
"""

from __future__ import annotations

import argparse
import sys
import time
import json
import logging
import csv
import random
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT

# Optional: import MLflow only if needed
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None  # noqa: F841

try:
    import dagshub
    HAS_DAGSHUB = True
except ImportError:
    HAS_DAGSHUB = False
    dagshub = None  # noqa: F841

# ------------------------------------------------------------------ #
#  Experiment Tracking Setup                                            #
# ------------------------------------------------------------------ #

def setup_mlflow(use_mlflow: bool) -> bool:
    """Initialize MLflow + DagsHub if requested and available.
    
    Args:
        use_mlflow: Whether to enable MLflow tracking
    
    Returns:
        True if MLflow initialized, False otherwise
    """
    if not use_mlflow:
        return False
    
    if not HAS_MLFLOW:
        print("⚠ --mlflow requested but 'mlflow' not installed. Skipping MLflow tracking.")
        print("  Install with: pip install mlflow dagshub")
        return False
    
    # Try to initialize DagsHub if credentials are available (team setup)
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if repo_owner and repo_name:
        if HAS_DAGSHUB:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print(f"✓ Connected to DagsHub: {repo_owner}/{repo_name}")
        else:
            print("⚠ DagsHub credentials found but 'dagshub' not installed. Using local MLflow.")
    else:
        print("ℹ MLflow enabled (local tracking). For team logging, set DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME env vars.")
    
    mlflow.set_experiment("MiniGPT")
    return True


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class SentenceDataset(Dataset):
    """Converts each sentence into overlapping (input, target) token pairs."""

    def __init__(self, corpus_path: Path, tokenizer: WordTokenizer, context_len: int = 32):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples: list[list[int]] = []

        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special=True)
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[idx]
        # Truncate to context_len + 1 so we always have a target
        ids = ids[: self.context_len + 1]
        # Pad to fixed length
        pad = self.tokenizer.pad_id
        padded = ids + [pad] * (self.context_len + 1 - len(ids))
        t = torch.tensor(padded, dtype=torch.long)
        return t[:-1], t[1:]  # input, target


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #



# ------------------------------------------------------------------ #
#  Training Loop                                                       #
# ------------------------------------------------------------------ #

def train(args: argparse.Namespace) -> None:
    # --- Setup ---
    # Device selection: default is CPU. Only use GPU if user explicitly requests it via --device cuda.
    if args.device == "cpu":
        device = torch.device("cpu")
        print("Device: cpu (forced by --device)")
    else:
        # User requested CUDA
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested via --device cuda but no CUDA-capable device is available. "
                "Re-run with --device cpu or set CUDA_VISIBLE_DEVICES= to force CPU."
            )
        try:
            prop = torch.cuda.get_device_properties(0)
            capability = prop.major + prop.minor / 10.0
            # Many modern PyTorch builds require compute capability >= 7.5 (sm_75)
            if capability < 7.5:
                raise RuntimeError(
                    f"CUDA device found (compute capability {capability}) is incompatible with this PyTorch build. "
                    "Re-run with --device cpu or install a PyTorch build matching your GPU.")
        except Exception as e:
            # Propagate a clear error
            raise RuntimeError(
                "Failed to query CUDA device properties or device incompatible. Re-run with --device cpu." +
                (f" Details: {e}" if e else ""))
        device = torch.device("cuda")
        print(f"Device: cuda (user requested)")
    
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Paths
    corpus_path = PROJECT_ROOT / "data" / "raw" / "generated_texts" / f"generated_corpus_{args.corpus}.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    output_dir = PROJECT_ROOT / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Logging setup ---
    log_file = logs_dir / f"train_corpus{args.corpus}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training corpus={args.corpus}, epochs={args.epochs}, seed={args.seed}")
    logger.info(f"Logs -> {log_file}")
    
    # --- Experiment tracking (optional) ---
    use_tracking = setup_mlflow(args.mlflow)
    mlflow_run = None
    
    # --- Load data ---
    logger.info(f"Loading corpus from {corpus_path.name}...")
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    
    # Save tokenizer for reproducible evaluation
    tokenizer_path = output_dir / f"tokenizer_corpus{args.corpus}.json"
    with tokenizer_path.open("w", encoding="utf-8") as f:
        json.dump({"vocab": list(tokenizer.token_to_id.keys())}, f, ensure_ascii=False, indent=2)
    
    # --- Dataset & DataLoader ---
    dataset = SentenceDataset(corpus_path, tokenizer, context_len=args.context_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    logger.info(f"Dataset: {len(dataset)} sentences | Batches/epoch: {len(loader)}")
    
    # --- Build model ---
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_len=args.context_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.embed_dim * 4,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {param_count:,} parameters")
    
    # --- Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # --- Setup tracking (if enabled) ---
    if use_tracking:
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "corpus": args.corpus,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "context_len": args.context_len,
            "seed": args.seed,
            "param_count": param_count,
        })
        mlflow.log_artifact(str(tokenizer_path))
    
    # --- Metrics tracking ---
    best_loss = float("inf")
    ckpt_path = output_dir / f"minigpt_corpus{args.corpus}.pt"
    metrics_path = logs_dir / f"metrics_corpus{args.corpus}.csv"
    
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["epoch", "avg_loss", "time_s"])
        
        # --- Training loop ---
        logger.info("Starting training...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            elapsed = time.time() - t0
            
            logger.info(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")
            metrics_writer.writerow([epoch, f"{avg_loss:.6f}", f"{elapsed:.3f}"])
            
            # Log to MLflow
            if use_tracking:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"✓ Best checkpoint saved (loss={avg_loss:.4f})")
                
                if use_tracking:
                    mlflow.log_artifact(str(ckpt_path))
                    mlflow.log_metric("best_loss", best_loss, step=epoch)
    
    # --- Final summary ---
    logger.info(f"\nTraining complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")
    logger.info(f"Metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    
    # --- Save embeddings for analysis ---
    embeddings_path = output_dir / f"embeddings_corpus{args.corpus}.npy"
    np.save(embeddings_path, model.token_emb.weight.detach().cpu().numpy())
    logger.info(f"Embeddings: {embeddings_path.relative_to(PROJECT_ROOT)}")
    
    if use_tracking:
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(embeddings_path))
    
    # --- Sample generation ---
    logger.info("\n--- Sample Generated Sentences ---")
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(5):
            ids = model.generate(
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_new_tokens=20,
                temperature=args.temperature,
                device=str(device),
            )
            text = tokenizer.decode(ids)
            samples.append(text)
            logger.info(f"  {text}")
    
    if use_tracking:
        mlflow.log_text("\n".join(samples), "generated_samples.txt")
        mlflow.end_run()



# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MiniGPT on the mini-language corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local training only
  python scripts/train_model.py --corpus 100 --epochs 5

  # With MLflow/DagsHub (if DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME are set)
  DAGSHUB_REPO_OWNER=youruser DAGSHUB_REPO_NAME=yourrepo python scripts/train_model.py --mlflow

  # Full 10k corpus with MLflow
  python scripts/train_model.py --mlflow --corpus 10000 --epochs 30
        """,
    )
    p.add_argument(
        "--corpus", type=int, default=10000, choices=[100, 500, 1000, 5000, 10000],
        help="Corpus size: 100, 500, 1000, 5000, or 10000 sentences (default: 10000)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, dest="batch_size",
        help="Training batch size (default: 64)",
    )
    p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--embed-dim", type=int, default=64, dest="embed_dim",
        help="Embedding dimension (default: 64)",
    )
    p.add_argument(
        "--num-heads", type=int, default=4, dest="num_heads",
        help="Number of attention heads (default: 4)",
    )
    p.add_argument(
        "--num-layers", type=int, default=4, dest="num_layers",
        help="Number of transformer layers (default: 4)",
    )
    p.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    p.add_argument(
        "--context-len", type=int, default=32, dest="context_len",
        help="Context length / max sequence length (default: 32)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature for generation (default: 0.8)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking (requires mlflow package; uses DagsHub if env vars set)",
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to run training on. Default: cpu. Set to 'cuda' to force GPU (will error if unavailable or incompatible).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

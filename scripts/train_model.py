"""Train a small GPT-style decoder on the generated mini-language corpus.

Usage:
    python scripts/train_model.py                        # default (10k corpus)
    python scripts/train_model.py --corpus 5000          # use 5k corpus
    python scripts/train_model.py --epochs 20 --lr 3e-4  # custom hyperparams
"""

from __future__ import annotations

import argparse
import sys
import time
import json
import logging
import csv
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT

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

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    corpus_path = PROJECT_ROOT / "data" / "raw" / "generated_texts" / f"generated_corpus_{args.corpus}.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    print(f"Building tokenizer from {corpus_path.name} ...")
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # prepare output & logging dirs
    output_dir = PROJECT_ROOT / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file + stdout
    log_file = logs_dir / f"train_corpus{args.corpus}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Training (corpus={args.corpus}) — logs -> {log_file}")

    # Save tokenizer vocab for reproducible evaluation
    tokenizer_path = output_dir / f"tokenizer_corpus{args.corpus}.json"
    with tokenizer_path.open("w", encoding="utf-8") as f:
        json.dump({"vocab": list(tokenizer.token_to_id.keys())}, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved tokenizer vocab -> {tokenizer_path}")

    dataset = SentenceDataset(corpus_path, tokenizer, context_len=args.context_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    logging.info(f"Dataset: {len(dataset)} sentences  |  Batches/epoch: {len(loader)}")

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
    logging.info(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    best_loss = float("inf")
    ckpt_path = output_dir / f"minigpt_corpus{args.corpus}.pt"

    # Prepare CSV metrics file
    metrics_path = logs_dir / f"metrics_corpus{args.corpus}.csv"
    metrics_f = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow(["epoch", "avg_loss", "time_s"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # logits: (B, T, V)  y: (B, T)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0
        logging.info(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")
        metrics_writer.writerow([epoch, f"{avg_loss:.6f}", f"{elapsed:.3f}"])

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"minigpt_corpus{args.corpus}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            }, ckpt_path)
            logging.info(f"Saved checkpoint -> {ckpt_path}")

    logging.info(f"\nBest loss: {best_loss:.4f}  checkpoint -> {ckpt_path.relative_to(PROJECT_ROOT)}")

    # close metrics file
    metrics_f.close()

    # Save embeddings for analysis
    embeddings_path = output_dir / f"embeddings_corpus{args.corpus}.npy"
    np.save(embeddings_path, model.token_emb.weight.detach().cpu().numpy())
    logging.info(f"Saved embeddings -> {embeddings_path}")

    # ---- sample a few generated sentences ----
    logging.info("\n--- Sample generated sentences ---")
    model.eval()
    for _ in range(5):
        ids = model.generate(
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=20,
            temperature=args.temperature,
            device=str(device),
        )
        logging.info(" %s", tokenizer.decode(ids))


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniGPT on the mini-language corpus")
    p.add_argument("--corpus", type=int, default=10000, choices=[100, 500, 1000, 5000, 10000],
                   help="Which generated corpus to use (number of sentences)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embed-dim", type=int, default=64, dest="embed_dim")
    p.add_argument("--num-heads", type=int, default=4, dest="num_heads")
    p.add_argument("--num-layers", type=int, default=4, dest="num_layers")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context-len", type=int, default=32, dest="context_len")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for generated examples")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

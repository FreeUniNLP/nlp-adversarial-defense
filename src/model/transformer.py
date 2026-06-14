"""Small GPT-style decoder-only Transformer for the mini-language."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        # causal mask — upper triangle is -inf
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only Transformer (GPT-style) sized for the mini-language."""

    def __init__(
        self,
        vocab_size: int,
        context_len: int = 32,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.context_len = context_len
        self.pad_id = pad_id
        # Self-describing architecture config so checkpoints can be reloaded at the
        # right size regardless of the constructor defaults.
        self.config = {
            "vocab_size":  vocab_size,
            "context_len": context_len,
            "embed_dim":   embed_dim,
            "num_heads":   num_heads,
            "num_layers":  num_layers,
            "ffn_dim":     ffn_dim,
            "dropout":     dropout,
        }

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def from_checkpoint(cls, ckpt: dict, vocab_size: int, pad_id: int) -> "MiniGPT":
        """Build a MiniGPT matching the architecture stored in a checkpoint.

        Looks for a 'model_config' dict first (newer checkpoints), then falls back
        to the training 'args' (train_model.py checkpoints), then to the original
        default architecture (embed 64 / 4 layers) for legacy checkpoints that
        recorded nothing.
        """
        cfg  = ckpt.get("model_config") or {}
        args = ckpt.get("args") or {}

        def pick(key, default):
            if key in cfg:
                return cfg[key]
            if key in args:
                return args[key]
            return default

        embed_dim = pick("embed_dim", 64)
        model = cls(
            vocab_size  = vocab_size,
            pad_id      = pad_id,
            context_len = pick("context_len", 32),
            embed_dim   = embed_dim,
            num_heads   = pick("num_heads", 4),
            num_layers  = pick("num_layers", 4),
            ffn_dim     = pick("ffn_dim", embed_dim * 4),
            dropout     = pick("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state"])
        return model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[int]:
        """Greedy / temperature-sampled autoregressive generation."""
        self.eval()
        ids = [bos_id]
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-self.context_len:]], device=device)
            logits = self(context)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == eos_id:
                break
            ids.append(next_id)
        return ids[1:]  # strip BOS

    def complete_with_log_probs(
        self,
        prefix_ids:     list[int],
        bos_id:         int,
        eos_id:         int,
        max_new_tokens: int   = 15,
        temperature:    float = 1.0,
        device:         str   = "cpu",
        min_new_tokens: int   = 1,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Complete an attacker prefix and return per-step log-probabilities.

        Returns
        -------
        full_ids  : complete token sequence (BOS stripped), prefix + suffix
        log_probs : shape (T_suffix,) log P(token | context) for suffix tokens
        """
        self.train()
        all_ids:   list[int]          = [bos_id] + list(prefix_ids)
        log_probs: list[torch.Tensor] = []
        new_count = 0

        for _ in range(max_new_tokens):
            context = torch.tensor([all_ids[-self.context_len:]], device=device)
            logits  = self(context)[:, -1, :] / temperature

            if new_count < min_new_tokens:
                logits = logits.clone()
                logits[0, eos_id] = float("-inf")

            log_prob_dist = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                probs   = torch.exp(log_prob_dist)
                next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == eos_id:
                break

            log_probs.append(log_prob_dist[0, next_id])
            all_ids.append(next_id)
            new_count += 1

        if log_probs:
            log_probs_tensor = torch.stack(log_probs)
        else:
            log_probs_tensor = torch.zeros(0, device=device)

        return all_ids[1:], log_probs_tensor

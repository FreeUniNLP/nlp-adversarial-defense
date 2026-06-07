"""
AttackerTransformer — causal decoder-only Transformer for prefix generation.

Architecture mirrors MiniGPT exactly (same building blocks).
The key difference is in generation: at every step, logits for
CFG-invalid tokens are masked to -inf before sampling.
This enforces hard structural constraints while the learned policy
distributes probability over the allowed actions.

Generation flow (per token):
    1. CFGStateTracker.valid_next_words()  → allowed surface words
    2. Map words to token IDs
    3. Forward pass → raw logits over full vocab
    4. Mask invalid IDs to -inf
    5. Softmax + sample
    6. CFGStateTracker.step(word)          → advance grammar state
    7. Repeat until EOS sampled or max_tokens reached

For RL training (to be added later):
    - generate_prefix() returns (token_ids, log_probs, words)
    - log_probs are the per-step log-probabilities under the current policy
    - REINFORCE updates: loss = -sum(log_prob * reward)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attacker.cfg_state_tracker import CFGStateTracker


# ------------------------------------------------------------------ #
#  Building blocks (identical to MiniGPT)                             #
# ------------------------------------------------------------------ #

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim,     bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) / scale
        mask  = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn  = attn.masked_fill(mask, float("-inf"))
        attn  = self.attn_drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ------------------------------------------------------------------ #
#  Attacker model                                                      #
# ------------------------------------------------------------------ #

class AttackerTransformer(nn.Module):
    """
    Decoder-only Transformer that generates CFG-valid sentence prefixes.

    Same architecture as MiniGPT so weights can be shared or transferred
    if needed.  At generation time, a CFGStateTracker filters the logits
    to hard-constrain the output to grammatically valid tokens.
    """

    def __init__(
        self,
        vocab_size:  int,
        context_len: int   = 32,
        embed_dim:   int   = 64,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        ffn_dim:     int   = 256,
        dropout:     float = 0.1,
        pad_id:      int   = 0,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.pad_id      = pad_id

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(context_len, embed_dim)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Standard causal forward pass. Returns logits (B, T, vocab_size)."""
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    # ------------------------------------------------------------------ #
    #  CFG-guided prefix generation                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_prefix(
        self,
        bos_id:      int,
        eos_id:      int,
        cfg_tracker: CFGStateTracker,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        max_tokens:  int   = 10,
        temperature: float = 1.0,
        device:      str   = "cpu",
    ) -> tuple[list[int], list[str]]:
        """
        Generate a CFG-valid prefix token by token.

        At each step:
          1. Ask CFGStateTracker for valid next words + can_end flag
          2. Map those words to token IDs
          3. Run forward pass, mask invalid logits to -inf
          4. Sample from the remaining distribution
          5. Advance the grammar state

        Args:
            bos_id      : BOS token id
            eos_id      : EOS token id
            cfg_tracker : CFGStateTracker instance (will be reset internally)
            token_to_id : word → id mapping from tokenizer
            id_to_token : id → word mapping from tokenizer
            max_tokens  : maximum prefix length (excluding BOS/EOS)
            temperature : sampling temperature
            device      : "cpu" or "cuda"

        Returns:
            (token_ids, words) — the generated prefix (BOS excluded)
        """
        self.eval()
        cfg_tracker.reset()

        all_ids: list[int] = [bos_id]

        for _ in range(max_tokens):
            valid_words, can_end = cfg_tracker.valid_next_words()

            # Collect valid token IDs
            valid_ids: list[int] = [
                token_to_id[w]
                for w in valid_words
                if w in token_to_id
            ]
            if can_end and len(all_ids) > 1:
                valid_ids.append(eos_id)

            if not valid_ids:
                break  # dead end — no valid continuation

            # Forward pass
            context = torch.tensor(
                [all_ids[-self.context_len:]], device=device
            )
            logits = self(context)[:, -1, :] / temperature  # (1, vocab_size)

            # Mask: keep only valid IDs
            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid_ids] = 0.0
            logits = logits + mask

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            # EOS → stop
            if next_id == eos_id:
                break

            word = id_to_token.get(next_id)
            if word is None:
                break

            # Advance grammar state
            if not cfg_tracker.step(word):
                break  # should not happen if masking is correct

            all_ids.append(next_id)

        prefix_ids   = all_ids[1:]           # strip BOS
        prefix_words = cfg_tracker.generated  # tracker holds the words
        return prefix_ids, prefix_words

    def generate_prefix_with_log_probs(
        self,
        bos_id:      int,
        eos_id:      int,
        cfg_tracker: CFGStateTracker,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        max_tokens:  int   = 10,
        temperature: float = 1.0,
        device:      str   = "cpu",
    ) -> tuple[list[int], list[str], torch.Tensor]:
        """
        Same as generate_prefix but also returns per-step log-probabilities.
        Required for REINFORCE-style RL training.

        Returns:
            (token_ids, words, log_probs)
            log_probs — shape (T,) tensor of log P(token | context, CFG-mask)
        """
        self.train()  # keep gradients flowing
        cfg_tracker.reset()

        all_ids:   list[int]          = [bos_id]
        log_probs: list[torch.Tensor] = []

        for _ in range(max_tokens):
            valid_words, can_end = cfg_tracker.valid_next_words()

            valid_ids = [token_to_id[w] for w in valid_words if w in token_to_id]
            if can_end and len(all_ids) > 1:
                valid_ids.append(eos_id)

            if not valid_ids:
                break

            context = torch.tensor([all_ids[-self.context_len:]], device=device)
            logits  = self(context)[:, -1, :] / temperature

            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid_ids] = 0.0
            logits = logits + mask

            log_prob_dist = F.log_softmax(logits, dim=-1)  # (1, vocab_size)

            # Sample (straight-through — no grad through multinomial)
            with torch.no_grad():
                probs   = torch.exp(log_prob_dist)
                next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == eos_id:
                break

            word = id_to_token.get(next_id)
            if word is None:
                break

            if not cfg_tracker.step(word):
                break

            # Record log-prob of the chosen action (with grad)
            log_probs.append(log_prob_dist[0, next_id])
            all_ids.append(next_id)

        prefix_ids   = all_ids[1:]
        prefix_words = cfg_tracker.generated

        if log_probs:
            log_probs_tensor = torch.stack(log_probs)
        else:
            log_probs_tensor = torch.zeros(0, device=device)

        return prefix_ids, prefix_words, log_probs_tensor

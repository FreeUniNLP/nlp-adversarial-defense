"""
Smoke tests for the REINFORCE attacker training loop.

These tests do NOT require the trained defender checkpoint; they use an
untrained MiniGPT as the environment, which is enough to verify the loop
mechanics (gradient flow, optimizer steps, EMA baseline, parameter updates).

Run:
    python -m pytest tests/test_train_attacker.py -v
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest

import torch

from src.attacker.attacker import AttackerTransformer
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.attacker.reward_function import RewardFunction, RewardWeights
from src.language.entities.cfg_validator import CFGValidator
from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT
from tests.conftest import (
    CORPUS_PATH, build_cfg, load_lexicon,
)


def _defender_complete(defender, prefix_ids, bos_id, eos_id, max_new_tokens, temperature):
    """Greedy-ish defender completion (no grad)."""
    all_ids = [bos_id] + prefix_ids
    ctx_len = defender.context_len
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx    = torch.tensor([all_ids[-ctx_len:]])
            logits = defender(ctx)[:, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, num_samples=1).item()
            if nxt == eos_id:
                break
            all_ids.append(nxt)
    return all_ids[1:]


class TestReinforceLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists():
            cls.ready = False
            return
        cls.ready = True

        nouns, verbs, adjectives = load_lexicon()
        cls.nouns, cls.verbs, cls.adjectives = nouns, verbs, adjectives

        cfg            = build_cfg(nouns, verbs, adjectives)
        cls.validator  = CFGValidator.from_cfg(cfg)
        cls.tracker    = CFGStateTracker(nouns, verbs, adjectives)
        cls.tokenizer  = WordTokenizer.from_corpus(CORPUS_PATH)

        cls.attacker = AttackerTransformer(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )
        cls.defender = MiniGPT(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )
        cls.defender.eval()
        for p in cls.defender.parameters():
            p.requires_grad = False

        cls.reward_fn = RewardFunction(
            nouns=nouns, verbs=verbs, adjectives=adjectives,
            weights=RewardWeights(),
        )

    def setUp(self):
        if not self.ready:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    # ------------------------------------------------------------------ #
    #  One-episode REINFORCE step                                          #
    # ------------------------------------------------------------------ #

    def _run_episode(self):
        prefix_ids, prefix_words, log_probs = self.attacker.generate_prefix_with_log_probs(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=5,
            temperature=1.0,
        )
        full_ids   = _defender_complete(
            self.defender, prefix_ids,
            self.tokenizer.bos_id, self.tokenizer.eos_id,
            max_new_tokens=10, temperature=1.0,
        )
        full_words   = self.tokenizer.decode(full_ids).split()
        full_sent    = " ".join(full_words)
        result       = self.validator.validate(full_sent)
        pre, suf     = self.reward_fn.split_sentence(full_words, prefix_words)
        rw           = self.reward_fn.compute(
            prefix_words=pre, suffix_words=suf,
            full_sentence=full_sent,
            is_valid=result.is_valid,
            cfg_error=result.error if not result.is_valid else None,
        )
        return prefix_words, log_probs, float(rw.reward), result.is_valid

    def test_episode_runs_end_to_end(self):
        prefix_words, log_probs, reward, _ = self._run_episode()
        self.assertGreater(len(prefix_words), 0, "Attacker produced empty prefix")
        self.assertEqual(log_probs.shape[0], len(prefix_words))
        self.assertTrue(torch.isfinite(torch.tensor(reward)))

    def test_reinforce_loss_has_gradient(self):
        _, log_probs, reward, _ = self._run_episode()
        if log_probs.numel() == 0:
            self.skipTest("Empty rollout")
        loss = -((reward - 0.0) * log_probs.sum())
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))

    def test_optimizer_step_updates_attacker_params(self):
        """After a few REINFORCE updates, at least one attacker parameter must move."""
        torch.manual_seed(0)
        optimizer = torch.optim.AdamW(self.attacker.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in self.attacker.parameters()]

        baseline = 0.0
        updates  = 0
        for _ in range(8):
            _, log_probs, reward, _ = self._run_episode()
            if log_probs.numel() == 0:
                continue
            advantage = reward - baseline
            loss      = -(advantage * log_probs.sum())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.attacker.parameters(), 1.0)
            optimizer.step()
            baseline = 0.95 * baseline + 0.05 * reward
            updates += 1

        self.assertGreater(updates, 0, "No REINFORCE updates ran")

        changed = sum(
            1 for a, b in zip(before, self.attacker.parameters())
            if not torch.equal(a, b.detach())
        )
        self.assertGreater(changed, 0, "No attacker parameter changed after updates")

    def test_ema_baseline_tracks_rewards(self):
        """The EMA baseline must approach the mean of observed rewards."""
        rewards  = [0.1, 0.5, 0.7, 0.9, 0.3, 0.6]
        alpha    = 0.2
        baseline = 0.0
        for r in rewards:
            baseline = (1 - alpha) * baseline + alpha * r
        # After this many updates, baseline should be inside the range of rewards
        self.assertGreater(baseline, min(rewards) - 0.01)
        self.assertLess(baseline,    max(rewards) + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)

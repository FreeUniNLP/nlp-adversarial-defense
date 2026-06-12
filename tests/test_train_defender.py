"""
Smoke tests for the REINFORCE defender training loop.

Uses an untrained attacker as the frozen environment and verifies gradient
flow through the defender completion path.

Run:
    python -m pytest tests/test_train_defender.py -v
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest

import torch

from src.attacker.attacker import AttackerTransformer
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.defender.reward_function import DefenderRewardFunction, DefenderRewardWeights
from src.language.entities.cfg_validator import CFGValidator
from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT
from tests.conftest import CORPUS_PATH, build_cfg, load_lexicon


class TestDefenderReinforceLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists():
            cls.ready = False
            return
        cls.ready = True

        nouns, verbs, adjectives = load_lexicon()
        cls.nouns, cls.verbs, cls.adjectives = nouns, verbs, adjectives

        cfg           = build_cfg(nouns, verbs, adjectives)
        cls.validator = CFGValidator.from_cfg(cfg)
        cls.tracker   = CFGStateTracker(nouns, verbs, adjectives)
        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)

        cls.attacker = AttackerTransformer(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )
        cls.attacker.eval()
        for p in cls.attacker.parameters():
            p.requires_grad = False

        cls.defender = MiniGPT(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )

        cls.reward_fn = DefenderRewardFunction(
            nouns=nouns, verbs=verbs, adjectives=adjectives,
            weights=DefenderRewardWeights(),
        )

    def setUp(self):
        if not self.ready:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    def _run_episode(self):
        with torch.no_grad():
            prefix_ids, prefix_words = self.attacker.generate_prefix(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                cfg_tracker=self.tracker,
                token_to_id=self.tokenizer.token_to_id,
                id_to_token=self.tokenizer.id_to_token,
                max_tokens=5,
                temperature=1.0,
            )

        full_ids, log_probs = self.defender.complete_with_log_probs(
            prefix_ids=prefix_ids,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=10,
            temperature=1.0,
        )

        full_words = self.tokenizer.decode(full_ids).split()
        full_sent  = " ".join(full_words)
        result     = self.validator.validate(full_sent)
        pre, suf   = self.reward_fn.split_sentence(full_words, prefix_words)
        rw         = self.reward_fn.compute(
            prefix_words=pre, suffix_words=suf,
            full_sentence=full_sent,
            is_valid=result.is_valid,
            cfg_error=result.error if not result.is_valid else None,
        )
        return prefix_words, log_probs, float(rw.reward), result.is_valid

    def test_episode_runs_end_to_end(self):
        prefix_words, log_probs, reward, _ = self._run_episode()
        self.assertGreater(len(prefix_words), 0)
        self.assertGreater(log_probs.numel(), 0)
        self.assertTrue(torch.isfinite(torch.tensor(reward)))

    def test_reinforce_loss_has_gradient(self):
        _, log_probs, reward, _ = self._run_episode()
        loss = -((reward - 0.0) * log_probs.sum())
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))

    def test_optimizer_step_updates_defender_params(self):
        torch.manual_seed(0)
        optimizer = torch.optim.AdamW(self.defender.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in self.defender.parameters()]

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
            torch.nn.utils.clip_grad_norm_(self.defender.parameters(), 1.0)
            optimizer.step()
            baseline = 0.95 * baseline + 0.05 * reward
            updates += 1

        self.assertGreater(updates, 0)
        changed = sum(
            1 for a, b in zip(before, self.defender.parameters())
            if not torch.equal(a, b.detach())
        )
        self.assertGreater(changed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

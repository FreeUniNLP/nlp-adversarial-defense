"""
Tests for AttackerTransformer — CFG-masked prefix generation.

Run:
    python -m pytest tests/test_attacker_transformer.py -v
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
from src.attacker.attacker import AttackerTransformer
from src.attacker.cfg_state_tracker import CFGStateTracker
from src.model.tokenizer import WordTokenizer
from tests.conftest import load_lexicon, CORPUS_PATH


class TestAttackerTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists():
            cls.attacker = None
            return

        nouns, verbs, adjectives = load_lexicon()
        cls.nouns      = nouns
        cls.verbs      = verbs
        cls.adjectives = adjectives
        cls.tokenizer  = WordTokenizer.from_corpus(CORPUS_PATH)
        cls.tracker    = CFGStateTracker(nouns, verbs, adjectives)
        cls.attacker   = AttackerTransformer(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )
        cls.attacker.eval()

    def setUp(self):
        if self.attacker is None:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    def _gen(self, max_tokens=8):
        return self.attacker.generate_prefix(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=max_tokens,
        )

    def _gen_with_lp(self, max_tokens=8):
        return self.attacker.generate_prefix_with_log_probs(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=max_tokens,
        )

    # --- forward pass ---

    def test_forward_output_shape(self):
        x = torch.tensor([[self.tokenizer.bos_id, 10, 20]])
        logits = self.attacker(x)
        B, T, V = logits.shape
        self.assertEqual(B, 1)
        self.assertEqual(T, 3)
        self.assertEqual(V, self.tokenizer.vocab_size)

    # --- generate_prefix ---

    def test_generate_prefix_returns_tuple(self):
        ids, words = self._gen()
        self.assertIsInstance(ids,   list)
        self.assertIsInstance(words, list)

    def test_generate_prefix_ids_match_words(self):
        ids, words = self._gen()
        self.assertEqual(len(ids), len(words))
        for i, w in zip(ids, words):
            self.assertEqual(self.tokenizer.id_to_token[i], w)

    def test_generate_prefix_respects_max_tokens(self):
        for max_t in (1, 3, 5):
            ids, _ = self._gen(max_tokens=max_t)
            self.assertLessEqual(len(ids), max_t, f"Exceeded max_tokens={max_t}")

    def test_generate_prefix_no_special_tokens(self):
        special = {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}
        for _ in range(5):
            ids, _ = self._gen()
            for i in ids:
                self.assertNotIn(i, special, f"Special token {i} in prefix")

    def test_generate_prefix_is_cfg_valid(self):
        """Every generated prefix must replay cleanly through the state tracker."""
        for _ in range(10):
            _, words = self._gen()
            check = CFGStateTracker(self.nouns, self.verbs, self.adjectives)
            for word in words:
                ok = check.step(word)
                self.assertTrue(ok, f"Invalid word '{word}' in prefix '{' '.join(words)}'")

    # --- generate_prefix_with_log_probs ---

    def test_log_probs_shape_matches_prefix_length(self):
        ids, _, log_probs = self._gen_with_lp()
        self.assertEqual(log_probs.shape[0], len(ids))

    def test_log_probs_are_non_positive(self):
        _, _, log_probs = self._gen_with_lp()
        if log_probs.numel() > 0:
            self.assertTrue(
                (log_probs <= 0).all(),
                f"Some log-probs are positive: {log_probs.tolist()}"
            )

    def test_log_probs_have_grad_fn(self):
        """log_probs must carry gradients for REINFORCE RL training."""
        _, _, log_probs = self._gen_with_lp()
        if log_probs.numel() > 0:
            self.assertTrue(
                log_probs.requires_grad or log_probs.grad_fn is not None,
                "log_probs must have a gradient function"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

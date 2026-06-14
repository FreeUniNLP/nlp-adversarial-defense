"""
Tests for MiniGPT — forward pass and generation.

Run:
    python -m pytest tests/test_minigpt.py -v
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT
from tests.conftest import CORPUS_PATH, CKPT_PATH


class TestMiniGPT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists() or not CKPT_PATH.exists():
            cls.model = None
            cls.tokenizer = None
            return

        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        cls.model = MiniGPT.from_checkpoint(
            ckpt, cls.tokenizer.vocab_size, cls.tokenizer.pad_id)
        cls.model.eval()

    def setUp(self):
        if self.model is None:
            self.skipTest("Model checkpoint or corpus not found")

    def test_forward_pass_output_shape(self):
        x = torch.tensor([[self.tokenizer.bos_id, 10, 20]])
        logits = self.model(x)
        B, T, V = logits.shape
        self.assertEqual(B, 1)
        self.assertEqual(T, 3)
        self.assertEqual(V, self.tokenizer.vocab_size)

    def test_generate_returns_list(self):
        ids = self.model.generate(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=10,
            temperature=1.0,
        )
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_generate_no_special_tokens_in_output(self):
        special = {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}
        for _ in range(5):
            ids = self.model.generate(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=15,
                temperature=1.0,
            )
            for i in ids:
                self.assertNotIn(i, special, f"Special token {i} found in output")

    def test_generate_respects_max_tokens(self):
        ids = self.model.generate(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=5,
            temperature=1.0,
        )
        self.assertLessEqual(len(ids), 5)

    def test_lower_temperature_is_more_deterministic(self):
        results = set()
        for _ in range(3):
            torch.manual_seed(0)
            ids = self.model.generate(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=10,
                temperature=0.01,
            )
            results.add(tuple(ids))
        self.assertEqual(len(results), 1, "Low temperature should produce deterministic output")


if __name__ == "__main__":
    unittest.main(verbosity=2)

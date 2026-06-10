"""
Tests for WordTokenizer — encoding, decoding, vocabulary.

Run:
    python -m pytest tests/test_tokenizer.py -v
"""

import unittest
from src.model.tokenizer import WordTokenizer
from tests.conftest import CORPUS_PATH


class TestWordTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists():
            cls.tokenizer = None
            return
        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)

    def setUp(self):
        if self.tokenizer is None:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    def test_vocab_size_greater_than_special_tokens(self):
        self.assertGreater(self.tokenizer.vocab_size, 4)

    def test_special_token_ids_unique(self):
        ids = [self.tokenizer.pad_id, self.tokenizer.bos_id,
               self.tokenizer.eos_id, self.tokenizer.unk_id]
        self.assertEqual(len(ids), len(set(ids)), "Special token IDs must be unique")

    def test_pad_id_is_zero(self):
        self.assertEqual(self.tokenizer.pad_id, 0)

    def test_encode_returns_list_of_ints(self):
        ids = self.tokenizer.encode("MAN RUN")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_encode_adds_bos_eos(self):
        ids = self.tokenizer.encode("MAN RUN", add_special=True)
        self.assertEqual(ids[0], self.tokenizer.bos_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_id)

    def test_encode_without_special(self):
        ids = self.tokenizer.encode("MAN RUN", add_special=False)
        self.assertNotEqual(ids[0], self.tokenizer.bos_id)
        self.assertNotEqual(ids[-1], self.tokenizer.eos_id)

    def test_decode_roundtrip(self):
        sentence = "MAN RUN"
        ids = self.tokenizer.encode(sentence, add_special=True)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, sentence)

    def test_unknown_word_maps_to_unk(self):
        ids = self.tokenizer.encode("UNKNOWNWORD", add_special=False)
        self.assertEqual(ids[0], self.tokenizer.unk_id)

    def test_decode_skips_special_tokens_by_default(self):
        ids = [self.tokenizer.bos_id, self.tokenizer.pad_id, self.tokenizer.eos_id]
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded.strip(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

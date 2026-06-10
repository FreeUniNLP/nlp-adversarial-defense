"""
Tests for CFGValidator — structural and semantic sentence validation.

Run:
    python -m pytest tests/test_cfg_validator.py -v
"""

import unittest
from tests.conftest import load_lexicon, build_cfg, build_validator, CORPUS_PATH


class TestCFGValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nouns, verbs, adjectives = load_lexicon()
        cfg = build_cfg(nouns, verbs, adjectives)
        cls.validator = build_validator(cfg)

    # --- valid sentences ---

    def test_valid_noun_verb(self):
        result = self.validator.validate("CHOICE SPREAD")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_noun_verb_noun(self):
        result = self.validator.validate("DRONE BREAK CLOCK")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_adj_noun_verb(self):
        result = self.validator.validate("FREE WOLF FALL")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_adj_noun_verb_adj_noun(self):
        result = self.validator.validate("CONNECTED BIRD BREAK SMALL STRONG WEAPON")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    # --- invalid sentences ---

    def test_invalid_empty_sentence(self):
        result = self.validator.validate("")
        self.assertFalse(result.is_valid)

    def test_invalid_unknown_word(self):
        result = self.validator.validate("MAN BLORP")
        self.assertFalse(result.is_valid)
        self.assertIn("Unknown word", result.error)

    def test_invalid_wrong_order_verb_first(self):
        result = self.validator.validate("RUN MAN")
        self.assertFalse(result.is_valid)
        self.assertIn("Skeleton invalid", result.error)

    def test_invalid_only_noun(self):
        result = self.validator.validate("MAN")
        self.assertFalse(result.is_valid)

    def test_invalid_only_verb(self):
        result = self.validator.validate("RUN")
        self.assertFalse(result.is_valid)

    def test_invalid_semantic_wrong_subject(self):
        result = self.validator.validate("STONE KNOW TRUTH")
        self.assertFalse(result.is_valid)
        self.assertIn("Semantic constraint violated", result.error)

    # --- corpus batch test ---

    def test_all_corpus_sentences_are_valid(self):
        if not CORPUS_PATH.exists():
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")
        invalid = []
        with CORPUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = self.validator.validate(line)
                if not result.is_valid:
                    invalid.append((line, result.error))
        self.assertEqual(
            len(invalid), 0,
            f"{len(invalid)} corpus sentences failed:\n" +
            "\n".join(f"  {s} -> {e}" for s, e in invalid[:5])
        )

    # --- ValidationResult helpers ---

    def test_validation_result_bool_true(self):
        result = self.validator.validate("MAN RUN")
        self.assertTrue(bool(result))

    def test_validation_result_bool_false(self):
        result = self.validator.validate("RUN MAN")
        self.assertFalse(bool(result))

    def test_validation_result_repr_valid(self):
        result = self.validator.validate("MAN RUN")
        self.assertIn("valid=True", repr(result))

    def test_validation_result_repr_invalid(self):
        result = self.validator.validate("RUN MAN")
        self.assertIn("valid=False", repr(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)

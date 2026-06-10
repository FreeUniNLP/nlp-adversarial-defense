"""
Tests for CFG — skeleton generation and sentence building.

Run:
    python -m pytest tests/test_cfg.py -v
"""

import unittest
from tests.conftest import load_lexicon, build_cfg


class TestCFG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nouns, verbs, adjectives = load_lexicon()
        cls.cfg = build_cfg(nouns, verbs, adjectives)

    def test_cfg_has_rules(self):
        self.assertGreater(len(self.cfg.rules), 0)

    def test_cfg_has_start_symbol(self):
        self.assertIn("START", self.cfg.rules)

    def test_generate_skeleton_returns_list(self):
        skeleton = self.cfg.generate_skeleton()
        self.assertIsInstance(skeleton, list)
        self.assertGreater(len(skeleton), 0)

    def test_skeleton_contains_only_pos_tags(self):
        valid_tags = {"NOUN", "VERB", "ADJ"}
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            for token in skeleton:
                self.assertIn(token, valid_tags, f"Unexpected token '{token}' in skeleton")

    def test_skeleton_always_has_noun_and_verb(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            self.assertIn("NOUN", skeleton)
            self.assertIn("VERB", skeleton)

    def test_build_sentence_from_skeleton_returns_string(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            try:
                sentence = self.cfg.build_sentence_from_skeleton(skeleton)
                self.assertIsInstance(sentence, str)
                self.assertGreater(len(sentence.strip()), 0)
            except ValueError:
                pass  # some skeletons legitimately fail semantic constraints

    def test_generated_sentence_words_are_uppercase(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            try:
                sentence = self.cfg.build_sentence_from_skeleton(skeleton)
                for word in sentence.split():
                    self.assertEqual(word, word.upper(), f"Word '{word}' should be uppercase")
            except ValueError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

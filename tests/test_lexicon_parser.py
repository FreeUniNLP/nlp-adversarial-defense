"""
Tests for LexiconParser — loading and parsing words.json.

Run:
    python -m pytest tests/test_lexicon_parser.py -v
"""

import unittest
from tests.conftest import load_lexicon


class TestLexiconParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()

    def test_nouns_non_empty(self):
        self.assertGreater(len(self.nouns), 0)

    def test_verbs_non_empty(self):
        self.assertGreater(len(self.verbs), 0)

    def test_adjectives_non_empty(self):
        self.assertGreater(len(self.adjectives), 0)

    def test_noun_counts(self):
        self.assertEqual(len(self.nouns), 69, f"Expected 69 nouns, got {len(self.nouns)}")

    def test_verb_counts(self):
        self.assertEqual(len(self.verbs), 32, f"Expected 32 verbs, got {len(self.verbs)}")

    def test_adjective_counts(self):
        self.assertEqual(len(self.adjectives), 41, f"Expected 41 adjectives, got {len(self.adjectives)}")

    def test_noun_has_word_tag_axis(self):
        noun = self.nouns[0]
        self.assertTrue(hasattr(noun, "word"))
        self.assertTrue(hasattr(noun, "tag"))
        self.assertTrue(hasattr(noun, "axis"))

    def test_verb_has_argument(self):
        verb = self.verbs[0]
        self.assertTrue(hasattr(verb, "verb_argument"))
        self.assertIsNotNone(verb.verb_argument.verb_to_subject_constraint)

    def test_adjective_has_constraint(self):
        adj = self.adjectives[0]
        self.assertTrue(hasattr(adj, "adjective_to_noun_constraint"))

    def test_noun_axis_values_in_range(self):
        for noun in self.nouns:
            for attr in ("agency", "physicality", "social", "system"):
                val = getattr(noun.axis, attr)
                self.assertGreaterEqual(val, 0, f"{noun.word}.{attr} should be >= 0")
                self.assertLessEqual(val, 5, f"{noun.word}.{attr} should be <= 5")

    def test_all_noun_words_uppercase(self):
        for noun in self.nouns:
            self.assertEqual(noun.word, noun.word.upper())

    def test_all_verb_words_uppercase(self):
        for verb in self.verbs:
            self.assertEqual(verb.word, verb.word.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for CFGStateTracker — grammar state machine and valid-word filtering.

Run:
    python -m pytest tests/test_cfg_state_tracker.py -v
"""

import unittest
from src.attacker.cfg_state_tracker import CFGStateTracker, GrammarState
from tests.conftest import load_lexicon


class TestCFGStateTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()
        cls.tracker = CFGStateTracker(cls.nouns, cls.verbs, cls.adjectives)

    def setUp(self):
        self.tracker.reset()

    # --- initial state ---

    def test_initial_state(self):
        self.assertEqual(self.tracker.state, GrammarState.SUBJECT_START)

    def test_initial_can_end_is_false(self):
        self.assertFalse(self.tracker.can_end)

    def test_initial_valid_words_includes_nouns_and_adjs(self):
        words, can_end = self.tracker.valid_next_words()
        noun_words = {n.word for n in self.nouns}
        adj_words  = {a.word for a in self.adjectives}
        self.assertTrue(noun_words & set(words), "Should include nouns")
        self.assertTrue(adj_words  & set(words), "Should include adjectives")
        self.assertFalse(can_end)

    def test_initial_valid_words_excludes_verbs(self):
        words, _ = self.tracker.valid_next_words()
        verb_words = {v.word for v in self.verbs}
        self.assertFalse(verb_words & set(words), "Verbs should not appear at subject start")

    # --- stepping with a noun ---

    def test_step_noun_moves_to_after_subject(self):
        self.tracker.step("MAN")
        self.assertEqual(self.tracker.state, GrammarState.AFTER_SUBJECT)

    def test_after_subject_valid_words_are_all_verbs(self):
        self.tracker.step("MAN")
        words, can_end = self.tracker.valid_next_words()
        verb_words = {v.word for v in self.verbs}
        self.assertTrue(set(words).issubset(verb_words), "After subject, only verbs allowed")
        self.assertTrue(can_end)

    def test_step_noun_records_subject(self):
        self.tracker.step("MAN")
        self.assertIsNotNone(self.tracker.subject_noun)
        self.assertEqual(self.tracker.subject_noun.word, "MAN")

    # --- stepping with an adjective ---

    def test_step_adj_moves_to_subject_after_adj(self):
        words, _ = self.tracker.valid_next_words()
        adj_words = {a.word for a in self.adjectives}
        first_adj = next(w for w in words if w in adj_words)
        self.tracker.step(first_adj)
        self.assertEqual(self.tracker.state, GrammarState.SUBJECT_AFTER_ADJ)

    def test_after_adj_only_nouns_allowed(self):
        adj_words = {a.word for a in self.adjectives}
        words, _ = self.tracker.valid_next_words()
        first_adj = next(w for w in words if w in adj_words)
        self.tracker.step(first_adj)
        next_words, _ = self.tracker.valid_next_words()
        verb_words = {v.word for v in self.verbs}
        adj_set    = {a.word for a in self.adjectives}
        self.assertFalse(verb_words & set(next_words), "No verbs after subject ADJ")
        self.assertFalse(adj_set   & set(next_words), "No adjs after subject ADJ")

    # --- intransitive verb ---

    def test_intransitive_verb_moves_to_after_intrans(self):
        intrans = [v for v in self.verbs if v.verb_argument.verb_to_object_constraint is None]
        self.assertTrue(intrans, "Need at least one intransitive verb")
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        intrans_names = {v.word for v in intrans}
        chosen = next((w for w in valid_verbs if w in intrans_names), None)
        if chosen is None:
            self.skipTest("No intransitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertEqual(self.tracker.state, GrammarState.AFTER_INTRANS_VERB)

    def test_after_intrans_verb_can_end(self):
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        intrans = {v.word for v in self.verbs if v.verb_argument.verb_to_object_constraint is None}
        chosen = next((w for w in valid_verbs if w in intrans), None)
        if chosen is None:
            self.skipTest("No intransitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertTrue(self.tracker.can_end)

    def test_after_intrans_no_further_words(self):
        """No verb chaining without an object in between."""
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        intrans = {v.word for v in self.verbs if v.verb_argument.verb_to_object_constraint is None}
        chosen = next((w for w in valid_verbs if w in intrans), None)
        if chosen is None:
            self.skipTest("No intransitive verb valid for MAN")
        self.tracker.step(chosen)
        next_words, _ = self.tracker.valid_next_words()
        self.assertEqual(next_words, [])

    # --- transitive verb ---

    def test_transitive_verb_moves_to_object_start(self):
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        trans = {v.word for v in self.verbs if v.verb_argument.verb_to_object_constraint is not None}
        chosen = next((w for w in valid_verbs if w in trans), None)
        if chosen is None:
            self.skipTest("No transitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertEqual(self.tracker.state, GrammarState.OBJECT_START)

    # --- invalid step ---

    def test_invalid_step_returns_false(self):
        result = self.tracker.step("RUN")   # verb at subject start = invalid
        self.assertFalse(result)

    def test_invalid_step_does_not_advance_state(self):
        self.tracker.step("RUN")
        self.assertEqual(self.tracker.state, GrammarState.SUBJECT_START)

    # --- sentence tracking ---

    def test_generated_records_words(self):
        self.tracker.step("MAN")
        self.assertEqual(self.tracker.generated, ["MAN"])

    def test_sentence_returns_space_joined(self):
        self.tracker.step("MAN")
        self.assertEqual(self.tracker.sentence(), "MAN")

    def test_reset_clears_state(self):
        self.tracker.step("MAN")
        self.tracker.reset()
        self.assertEqual(self.tracker.state, GrammarState.SUBJECT_START)
        self.assertEqual(self.tracker.generated, [])
        self.assertIsNone(self.tracker.subject_noun)


if __name__ == "__main__":
    unittest.main(verbosity=2)

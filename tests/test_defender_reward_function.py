"""
Tests for DefenderRewardFunction — inverted structured reward module.

Run:
    python -m pytest tests/test_defender_reward_function.py -v
"""

import unittest

from src.attacker.reward_function import RewardFunction
from src.defender.reward_function import (
    DefenderRewardFunction,
    DefenderRewardOutput,
    DefenderRewardWeights,
)
from tests.conftest import load_lexicon


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()
        cls.rf = DefenderRewardFunction(cls.nouns, cls.verbs, cls.adjectives)
        cls.atk_rf = RewardFunction(cls.nouns, cls.verbs, cls.adjectives)


class TestDefenderRewardCompute(_Base):

    def _compute(self, prefix, suffix, valid, error=None):
        full = " ".join(prefix + suffix)
        return self.rf.compute(
            prefix_words=prefix, suffix_words=suffix,
            full_sentence=full, is_valid=valid, cfg_error=error,
        )

    def test_returns_defender_reward_output(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIsInstance(out, DefenderRewardOutput)

    def test_grammar_reward_one_when_valid(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertEqual(out.grammar_reward, 1.0)

    def test_grammar_reward_zero_when_invalid(self):
        out = self._compute(["RIVER"], ["BURN"], valid=False)
        self.assertEqual(out.grammar_reward, 0.0)

    def test_tag_reward_is_closeness(self):
        out = self._compute(["MAN"], ["ALGORITHM"], valid=True)
        self.assertAlmostEqual(
            out.tag_reward, 1.0 - out.distances.tag_mismatch, places=6,
        )

    def test_axis_reward_is_closeness(self):
        out = self._compute(["MAN"], ["ALGORITHM"], valid=True)
        self.assertAlmostEqual(
            out.axis_reward, 1.0 - out.distances.axis_distance, places=6,
        )

    def test_valid_aligned_beats_invalid(self):
        valid = self._compute(["MAN", "WOMAN"], ["CHILD", "ELDER"], valid=True)
        invalid = self._compute(["RIVER"], ["BURN"], valid=False)
        self.assertGreater(valid.reward, invalid.reward)

    def test_high_alignment_beats_low_alignment(self):
        high = self._compute(["MAN", "WOMAN"], ["CHILD", "ELDER"], valid=True)
        low  = self._compute(["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], valid=True)
        self.assertGreater(high.reward, low.reward)

    def test_mirror_of_attacker_components(self):
        """Defender closeness + attacker distance should sum to 1 per component."""
        prefix, suffix = ["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"]
        full = " ".join(prefix + suffix)
        atk = self.atk_rf.compute(prefix, suffix, full, is_valid=True)
        defn = self.rf.compute(prefix, suffix, full, is_valid=True)
        self.assertAlmostEqual(atk.tag_reward + defn.tag_reward, 1.0, places=6)
        self.assertAlmostEqual(atk.axis_reward + defn.axis_reward, 1.0, places=6)

    def test_summary_returns_string(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        s = out.summary()
        self.assertIn("defender", s.lower())
        self.assertIn("tag_closeness", s)


class TestDefenderRewardWeights(unittest.TestCase):

    def test_default_weights_match_attacker(self):
        w = DefenderRewardWeights()
        self.assertAlmostEqual(w.w_grammar, 1.0)
        self.assertAlmostEqual(w.w_tag, 0.30)
        self.assertAlmostEqual(w.w_axis, 0.20)

    def test_max_reward_equals_weight_sum(self):
        w = DefenderRewardWeights()
        self.assertAlmostEqual(w.max_reward, w.w_grammar + w.w_tag + w.w_axis, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

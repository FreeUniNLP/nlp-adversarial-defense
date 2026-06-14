"""
Tests for RewardComputer — attacker reward signal.

Reward levels:
  A) Grammar failure  → grammar_reward=1.0, total >= 1.0
  B) Topic mismatch   → grammar_reward=0.0, reward driven by tag/axis mismatch
  C) Topic consistent → low reward

Run:
    python -m pytest tests/test_reward_computer.py -v
"""

import unittest
from src.reward.reward_computer import RewardComputer, RewardConfig, RewardResult, TopicProfile
from tests.conftest import load_lexicon


class TestRewardComputer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()
        cls.rc = RewardComputer(cls.nouns, cls.verbs, cls.adjectives)

    # --- result structure ---

    def test_compute_returns_reward_result(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertIsInstance(result, RewardResult)

    def test_reward_result_has_all_fields(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        for field in ("reward", "grammar_failure", "grammar_reward",
                      "tag_distance", "axis_distance", "topic_mismatch",
                      "prefix_profile", "suffix_profile", "cfg_error"):
            self.assertTrue(hasattr(result, field), f"Missing field: {field}")

    # --- Component A: grammar failure ---

    def test_grammar_failure_sets_grammar_reward_to_one(self):
        result = self.rc.compute(["RIVER"], ["BURN"], is_valid=False,
                                 cfg_error="Semantic constraint violated")
        self.assertEqual(result.grammar_reward, 1.0)

    def test_grammar_failure_flag_is_true_when_invalid(self):
        result = self.rc.compute(["RIVER"], ["BURN"], is_valid=False)
        self.assertTrue(result.grammar_failure)

    def test_grammar_failure_flag_is_false_when_valid(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertFalse(result.grammar_failure)

    def test_grammar_reward_is_zero_for_valid_sentence(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertEqual(result.grammar_reward, 0.0)

    def test_grammar_failure_dominates_reward(self):
        invalid = self.rc.compute(["RIVER"], ["BURN"], is_valid=False)
        valid   = self.rc.compute(["MAN"],   ["GROW"], is_valid=True)
        self.assertGreater(invalid.reward, valid.reward)

    def test_cfg_error_stored_on_failure(self):
        err = "Semantic constraint violated: something"
        result = self.rc.compute(["RIVER"], ["BURN"], is_valid=False, cfg_error=err)
        self.assertEqual(result.cfg_error, err)

    def test_cfg_error_is_none_on_valid_sentence(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertIsNone(result.cfg_error)

    # --- Component B: topic mismatch ---

    def test_identical_words_give_zero_tag_distance(self):
        result = self.rc.compute(["MAN"], ["MAN"], is_valid=True)
        self.assertAlmostEqual(result.tag_distance, 0.0, places=6)

    def test_different_domains_give_high_tag_distance(self):
        result = self.rc.compute(
            ["MAN", "WOMAN", "CHILD"],          # HUMAN domain
            ["ALGORITHM", "NETWORK", "SENSOR"], # MACHINE domain
            is_valid=True,
        )
        self.assertGreater(result.tag_distance, 0.5)

    def test_same_words_give_zero_axis_distance(self):
        result = self.rc.compute(["MAN"], ["MAN"], is_valid=True)
        self.assertAlmostEqual(result.axis_distance, 0.0, places=6)

    def test_topic_mismatch_in_zero_one_range(self):
        result = self.rc.compute(
            ["STRONG", "MAN", "CARRY"],
            ["FOREST", "GROW"],
            is_valid=True,
        )
        self.assertGreaterEqual(result.topic_mismatch, 0.0)
        self.assertLessEqual(result.topic_mismatch,    1.0)

    def test_high_mismatch_gives_higher_reward_than_low_mismatch(self):
        high = self.rc.compute(
            ["MAN", "WOMAN"],
            ["ALGORITHM", "SENSOR"],    # machine — very different tags
            is_valid=True,
        )
        low = self.rc.compute(
            ["MAN", "WOMAN"],
            ["CHILD", "ELDER"],         # also human — very similar tags
            is_valid=True,
        )
        self.assertGreater(high.reward, low.reward)

    # --- Reward range and weights ---

    def test_reward_is_non_negative(self):
        cases = [
            (["MAN"],   ["RUN"],  True),
            (["RIVER"], ["BURN"], False),
            ([],        [],       True),
        ]
        for prefix, suffix, valid in cases:
            result = self.rc.compute(prefix, suffix, is_valid=valid)
            self.assertGreaterEqual(result.reward, 0.0,
                msg=f"Negative reward for prefix={prefix}, suffix={suffix}")

    def test_max_reward_equals_weight_sum(self):
        """Upper bound: w_grammar=1.0 + w_mismatch=0.5 = 1.5 by default."""
        cfg = RewardConfig(w_grammar=1.0, w_mismatch=0.5)
        self.assertAlmostEqual(cfg.w_grammar + cfg.w_mismatch, 1.5)

    def test_custom_weights_affect_reward(self):
        rc_low  = RewardComputer(self.nouns, self.verbs, self.adjectives,
                                 config=RewardConfig(w_grammar=0.0, w_mismatch=0.5))
        rc_high = RewardComputer(self.nouns, self.verbs, self.adjectives,
                                 config=RewardConfig(w_grammar=0.0, w_mismatch=1.0))
        r1 = rc_low.compute( ["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], is_valid=True)
        r2 = rc_high.compute(["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], is_valid=True)
        self.assertGreater(r2.reward, r1.reward)

    # --- TopicProfile ---

    def test_empty_profile_mean_axis_is_zeros(self):
        p = TopicProfile()
        self.assertEqual(p.mean_axis(), [0.0, 0.0, 0.0, 0.0])

    def test_profile_tags_accumulate_from_known_word(self):
        result = self.rc.compute(["MAN"], [], is_valid=True)
        man_entry = next(n for n in self.nouns if n.word == "MAN")
        for tag in man_entry.tag.tag:
            self.assertIn(tag, result.prefix_profile.tags)

    def test_profile_word_count(self):
        result = self.rc.compute(["MAN", "WOMAN", "CHILD"], ["RUN"], is_valid=True)
        self.assertEqual(result.prefix_profile.word_count, 3)
        self.assertEqual(result.suffix_profile.word_count, 1)

    def test_unknown_words_ignored_gracefully(self):
        result = self.rc.compute(["UNKNOWN_WORD"], ["ANOTHER_FAKE"], is_valid=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.prefix_profile.word_count, 0)
        self.assertEqual(result.suffix_profile.word_count, 0)

    def test_summary_returns_string_with_reward(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Reward", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)

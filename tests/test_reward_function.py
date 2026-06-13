"""
Tests for RewardFunction — structured reward module.

Tests are organised into five groups:

  1. FeatureExtractor     — correct POS classification, tag accumulation, axis
  2. DistanceCalculator   — per-POS Jaccard distance, axis cosine distance
  3. RewardFunction.split_sentence  — sentence splitting logic
  4. RewardFunction.compute         — reward components and output structure
  5. RewardWeights        — weight configuration and max_reward

Run:
    python -m pytest tests/test_reward_function.py -v
"""

import math
import unittest

from src.attacker.reward_function import (
    RewardFunction,
    RewardWeights,
    RewardOutput,
    FeatureExtractor,
    DistanceCalculator,
    SegmentFeatures,
    DistanceScores,
)
from tests.conftest import load_lexicon


# ------------------------------------------------------------------ #
#  Shared setup                                                        #
# ------------------------------------------------------------------ #

class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()
        cls.extractor = FeatureExtractor(cls.nouns, cls.verbs, cls.adjectives)
        cls.rf        = RewardFunction(cls.nouns, cls.verbs, cls.adjectives)

        # Convenient word sets
        cls.noun_words = {n.word for n in cls.nouns}
        cls.verb_words = {v.word for v in cls.verbs}
        cls.adj_words  = {a.word for a in cls.adjectives}


# ================================================================== #
#  1. FeatureExtractor                                                 #
# ================================================================== #

class TestFeatureExtractor(_Base):

    def test_returns_segment_features(self):
        feat = self.extractor.extract(["MAN", "RUN"])
        self.assertIsInstance(feat, SegmentFeatures)

    def test_words_stored(self):
        words = ["MAN", "RUN", "BIG"]
        feat = self.extractor.extract(words)
        self.assertEqual(feat.words, words)

    # --- POS classification ---

    def test_noun_classified_as_noun(self):
        feat = self.extractor.extract(["MAN"])
        self.assertEqual(len(feat.nouns), 1)
        self.assertEqual(feat.nouns[0].word, "MAN")
        self.assertEqual(len(feat.verbs), 0)
        self.assertEqual(len(feat.adjectives), 0)

    def test_verb_classified_as_verb(self):
        feat = self.extractor.extract(["RUN"])
        self.assertEqual(len(feat.verbs), 1)
        self.assertEqual(feat.verbs[0].word, "RUN")
        self.assertEqual(len(feat.nouns), 0)
        self.assertEqual(len(feat.adjectives), 0)

    def test_adjective_classified_as_adjective(self):
        feat = self.extractor.extract(["BIG"])
        self.assertEqual(len(feat.adjectives), 1)
        self.assertEqual(feat.adjectives[0].word, "BIG")
        self.assertEqual(len(feat.nouns), 0)
        self.assertEqual(len(feat.verbs), 0)

    def test_mixed_pos_classified_correctly(self):
        feat = self.extractor.extract(["BIG", "MAN", "RUN"])
        self.assertEqual(len(feat.nouns),      1)
        self.assertEqual(len(feat.verbs),      1)
        self.assertEqual(len(feat.adjectives), 1)

    # --- Tag accumulation ---

    def test_noun_tags_populated(self):
        feat = self.extractor.extract(["MAN"])
        man  = next(n for n in self.nouns if n.word == "MAN")
        for tag in man.tag.tag:
            self.assertIn(tag, feat.noun_tags)

    def test_verb_tags_populated(self):
        feat = self.extractor.extract(["RUN"])
        run  = next(v for v in self.verbs if v.word == "RUN")
        for tag in run.tag.tag:
            self.assertIn(tag, feat.verb_tags)

    def test_adjective_tags_populated(self):
        feat = self.extractor.extract(["BIG"])
        big  = next(a for a in self.adjectives if a.word == "BIG")
        for tag in big.tag.tag:
            self.assertIn(tag, feat.adjective_tags)

    def test_all_tags_is_union_of_pos_tags(self):
        feat = self.extractor.extract(["BIG", "MAN", "RUN"])
        expected = feat.noun_tags | feat.verb_tags | feat.adjective_tags
        self.assertEqual(feat.all_tags, expected)

    def test_noun_tags_not_in_verb_tags(self):
        """Tags are stored per-POS, not mixed."""
        feat = self.extractor.extract(["MAN", "RUN"])
        # noun tags should only contain noun-origin tags
        for tag in feat.noun_tags:
            self.assertNotIn(tag, feat.verb_tags,
                msg=f"Tag '{tag}' leaked from noun into verb_tags")

    # --- Axis ---

    def test_word_count_correct(self):
        feat = self.extractor.extract(["BIG", "MAN", "RUN"])
        self.assertEqual(feat.word_count, 3)

    def test_empty_input_gives_zero_axis(self):
        feat = self.extractor.extract([])
        self.assertEqual(feat.mean_axis, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(feat.word_count, 0)

    def test_mean_axis_matches_manual_calculation(self):
        """mean_axis should be the elementwise average of all word axes."""
        man  = next(n for n in self.nouns  if n.word == "MAN")
        run  = next(v for v in self.verbs  if v.word == "RUN")
        feat = self.extractor.extract(["MAN", "RUN"])

        expected = [
            (man.axis.agency      + run.axis.agency)      / 2,
            (man.axis.physicality + run.axis.physicality) / 2,
            (man.axis.social      + run.axis.social)      / 2,
            (man.axis.system      + run.axis.system)      / 2,
        ]
        for a, b in zip(feat.mean_axis, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_unknown_words_ignored(self):
        feat = self.extractor.extract(["FAKEWORD123"])
        self.assertEqual(feat.word_count, 0)
        self.assertEqual(feat.noun_tags,       set())
        self.assertEqual(feat.verb_tags,       set())
        self.assertEqual(feat.adjective_tags,  set())
        self.assertEqual(feat.mean_axis,       [0.0, 0.0, 0.0, 0.0])

    def test_multiple_nouns_tags_accumulate(self):
        feat = self.extractor.extract(["MAN", "CHILD"])
        man   = next(n for n in self.nouns if n.word == "MAN")
        child = next(n for n in self.nouns if n.word == "CHILD")
        for tag in man.tag.tag + child.tag.tag:
            self.assertIn(tag, feat.noun_tags)


# ================================================================== #
#  2. DistanceCalculator                                               #
# ================================================================== #

class TestDistanceCalculator(_Base):

    def test_returns_distance_scores(self):
        pre = self.extractor.extract(["MAN"])
        suf = self.extractor.extract(["RUN"])
        scores = DistanceCalculator.compute(pre, suf)
        self.assertIsInstance(scores, DistanceScores)

    def test_identical_features_give_zero_distances(self):
        feat = self.extractor.extract(["MAN"])
        scores = DistanceCalculator.compute(feat, feat)
        self.assertAlmostEqual(scores.noun_tag_dist,      0.0, places=6)
        self.assertAlmostEqual(scores.verb_tag_dist,      0.0, places=6)
        self.assertAlmostEqual(scores.adjective_tag_dist, 0.0, places=6)
        self.assertAlmostEqual(scores.tag_mismatch,       0.0, places=6)
        self.assertAlmostEqual(scores.axis_distance,      0.0, places=6)

    def test_disjoint_noun_tags_give_max_distance(self):
        """Human nouns vs machine nouns should have noun_tag_dist close to 1."""
        pre = self.extractor.extract(["MAN", "WOMAN", "CHILD"])       # HUMAN
        suf = self.extractor.extract(["ALGORITHM", "NETWORK"])        # MACHINE
        scores = DistanceCalculator.compute(pre, suf)
        self.assertGreater(scores.noun_tag_dist, 0.5)

    def test_same_verb_in_both_gives_zero_verb_distance(self):
        pre = self.extractor.extract(["RUN"])
        suf = self.extractor.extract(["RUN"])
        scores = DistanceCalculator.compute(pre, suf)
        self.assertAlmostEqual(scores.verb_tag_dist, 0.0, places=6)

    def test_disjoint_verbs_give_high_distance(self):
        # pick two verbs with completely different tags
        pre = self.extractor.extract(["RUN"])       # PHYSICAL_ACTION, LOCOMOTION
        suf = self.extractor.extract(["COMPUTE"])   # PROCESSING / SYSTEM_ACTION
        scores = DistanceCalculator.compute(pre, suf)
        self.assertGreater(scores.verb_tag_dist, 0.0)

    def test_tag_mismatch_is_mean_of_three_pos_distances(self):
        pre = self.extractor.extract(["BIG", "MAN", "RUN"])
        suf = self.extractor.extract(["SMALL", "ALGORITHM", "COMPUTE"])
        scores = DistanceCalculator.compute(pre, suf)
        expected = (scores.noun_tag_dist + scores.verb_tag_dist + scores.adjective_tag_dist) / 3.0
        self.assertAlmostEqual(scores.tag_mismatch, expected, places=6)

    def test_axis_distance_in_zero_one_range(self):
        pre = self.extractor.extract(["MAN", "CARRY"])
        suf = self.extractor.extract(["ALGORITHM", "COMPUTE"])
        scores = DistanceCalculator.compute(pre, suf)
        self.assertGreaterEqual(scores.axis_distance, 0.0)
        self.assertLessEqual(scores.axis_distance,    1.0)

    def test_empty_segments_give_zero_distances(self):
        pre = self.extractor.extract([])
        suf = self.extractor.extract([])
        scores = DistanceCalculator.compute(pre, suf)
        self.assertAlmostEqual(scores.noun_tag_dist,      0.0, places=6)
        self.assertAlmostEqual(scores.tag_mismatch,       0.0, places=6)
        self.assertAlmostEqual(scores.axis_distance,      0.0, places=6)


# ================================================================== #
#  3. RewardFunction — split_sentence                                  #
# ================================================================== #

class TestSplitSentence(unittest.TestCase):

    def test_split_basic(self):
        full   = ["MAN", "CARRY", "CLOCK"]
        prefix = ["MAN", "CARRY"]
        pre, suf = RewardFunction.split_sentence(full, prefix)
        self.assertEqual(pre, ["MAN", "CARRY"])
        self.assertEqual(suf, ["CLOCK"])

    def test_split_prefix_is_full_sentence(self):
        full   = ["MAN", "RUN"]
        prefix = ["MAN", "RUN"]
        pre, suf = RewardFunction.split_sentence(full, prefix)
        self.assertEqual(pre, ["MAN", "RUN"])
        self.assertEqual(suf, [])

    def test_split_empty_prefix(self):
        full   = ["MAN", "RUN"]
        prefix = []
        pre, suf = RewardFunction.split_sentence(full, prefix)
        self.assertEqual(pre, [])
        self.assertEqual(suf, ["MAN", "RUN"])

    def test_split_empty_full_sentence(self):
        pre, suf = RewardFunction.split_sentence([], [])
        self.assertEqual(pre, [])
        self.assertEqual(suf, [])

    def test_split_prefix_longer_than_full(self):
        """Prefix longer than full sentence — suffix must be empty, no crash."""
        full   = ["MAN"]
        prefix = ["MAN", "RUN", "CARRY"]
        pre, suf = RewardFunction.split_sentence(full, prefix)
        self.assertEqual(pre, ["MAN"])
        self.assertEqual(suf, [])

    def test_split_single_word_prefix(self):
        full   = ["MAN", "RUN", "STONE"]
        prefix = ["MAN"]
        pre, suf = RewardFunction.split_sentence(full, prefix)
        self.assertEqual(pre, ["MAN"])
        self.assertEqual(suf, ["RUN", "STONE"])


# ================================================================== #
#  4. RewardFunction.compute                                           #
# ================================================================== #

class TestRewardFunctionCompute(_Base):

    def _compute(self, prefix, suffix, valid, error=None):
        full = " ".join(prefix + suffix)
        return self.rf.compute(
            prefix_words=prefix, suffix_words=suffix,
            full_sentence=full, is_valid=valid, cfg_error=error,
        )

    # --- output structure ---

    def test_returns_reward_output(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIsInstance(out, RewardOutput)

    def test_all_fields_present(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        for attr in ("reward", "grammar_failure", "grammar_reward",
                     "distances", "tag_reward", "axis_reward",
                     "prefix", "suffix", "cfg_error", "full_sentence"):
            self.assertTrue(hasattr(out, attr), f"Missing field: {attr}")

    def test_prefix_and_suffix_are_segment_features(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIsInstance(out.prefix, SegmentFeatures)
        self.assertIsInstance(out.suffix, SegmentFeatures)

    def test_distances_is_distance_scores(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIsInstance(out.distances, DistanceScores)

    def test_full_sentence_stored(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIn("MAN", out.full_sentence)
        self.assertIn("RUN", out.full_sentence)

    # --- grammar failure component ---

    def test_grammar_reward_one_when_invalid(self):
        out = self._compute(["RIVER"], ["BURN"], valid=False,
                            error="Semantic constraint violated")
        self.assertEqual(out.grammar_reward, 1.0)

    def test_grammar_reward_zero_when_valid(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertEqual(out.grammar_reward, 0.0)

    def test_grammar_failure_flag_true_when_invalid(self):
        out = self._compute(["RIVER"], ["BURN"], valid=False)
        self.assertTrue(out.grammar_failure)

    def test_grammar_failure_flag_false_when_valid(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertFalse(out.grammar_failure)

    def test_cfg_error_stored_when_invalid(self):
        err = "Semantic constraint violated: test"
        out = self._compute(["RIVER"], ["BURN"], valid=False, error=err)
        self.assertEqual(out.cfg_error, err)

    def test_cfg_error_none_when_valid(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertIsNone(out.cfg_error)

    # --- tag and axis reward components ---

    def test_tag_reward_equals_tag_mismatch(self):
        out = self._compute(["MAN"], ["ALGORITHM"], valid=True)
        self.assertAlmostEqual(out.tag_reward, out.distances.tag_mismatch, places=6)

    def test_axis_reward_equals_axis_distance(self):
        out = self._compute(["MAN"], ["ALGORITHM"], valid=True)
        self.assertAlmostEqual(out.axis_reward, out.distances.axis_distance, places=6)

    # --- reward range ---

    def test_reward_non_negative(self):
        for prefix, suffix, valid in [
            (["MAN"],   ["RUN"],       True),
            (["RIVER"], ["BURN"],      False),
            ([],        [],            True),
        ]:
            out = self._compute(prefix, suffix, valid)
            self.assertGreaterEqual(out.reward, 0.0)

    def test_grammar_failure_dominates(self):
        """Invalid sentence always scores higher than valid mismatched sentence."""
        invalid = self._compute(["RIVER"], ["BURN"],            valid=False)
        valid   = self._compute(["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], valid=True)
        self.assertGreater(invalid.reward, valid.reward)

    def test_high_mismatch_beats_low_mismatch(self):
        """Human prefix + machine suffix > human prefix + human suffix."""
        high = self._compute(["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], valid=True)
        low  = self._compute(["MAN", "WOMAN"], ["CHILD", "ELDER"],      valid=True)
        self.assertGreater(high.reward, low.reward)

    # --- prefix/suffix feature correctness ---

    def test_prefix_features_reflect_prefix_words(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertEqual(out.prefix.words, ["MAN"])
        self.assertEqual(len(out.prefix.nouns), 1)
        self.assertEqual(out.prefix.nouns[0].word, "MAN")

    def test_suffix_features_reflect_suffix_words(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        self.assertEqual(out.suffix.words, ["RUN"])
        self.assertEqual(len(out.suffix.verbs), 1)
        self.assertEqual(out.suffix.verbs[0].word, "RUN")

    # --- summary ---

    def test_summary_returns_string(self):
        out = self._compute(["MAN"], ["RUN"], valid=True)
        s = out.summary()
        self.assertIsInstance(s, str)
        self.assertIn("Sentence", s)
        self.assertIn("PREFIX", s)
        self.assertIn("SUFFIX", s)
        self.assertIn("Distance", s)
        self.assertIn("Reward", s)


# ================================================================== #
#  5. RewardWeights                                                    #
# ================================================================== #

class TestRewardWeights(unittest.TestCase):

    def test_default_weights(self):
        w = RewardWeights()
        self.assertAlmostEqual(w.w_grammar, 1.0)
        self.assertAlmostEqual(w.w_tag,     0.30)
        self.assertAlmostEqual(w.w_axis,    0.20)

    def test_max_reward_equals_weight_sum(self):
        w = RewardWeights()
        self.assertAlmostEqual(
            w.max_reward, w.w_grammar + w.w_tag + w.w_axis + w.w_repeat, places=6)

    def test_custom_weights_accepted(self):
        w = RewardWeights(w_grammar=2.0, w_tag=0.5, w_axis=0.5)
        self.assertAlmostEqual(w.w_grammar, 2.0)
        self.assertAlmostEqual(w.w_tag,     0.5)
        self.assertAlmostEqual(w.w_axis,    0.5)

    def test_higher_w_grammar_increases_invalid_reward(self):
        nouns, verbs, adjectives = load_lexicon()
        rf_low  = RewardFunction(nouns, verbs, adjectives,
                                 weights=RewardWeights(w_grammar=0.5, w_tag=0.0, w_axis=0.0))
        rf_high = RewardFunction(nouns, verbs, adjectives,
                                 weights=RewardWeights(w_grammar=2.0, w_tag=0.0, w_axis=0.0))
        r_low  = rf_low.compute( ["RIVER"], [], "RIVER BURN", is_valid=False)
        r_high = rf_high.compute(["RIVER"], [], "RIVER BURN", is_valid=False)
        self.assertGreater(r_high.reward, r_low.reward)

    def test_zero_weights_give_zero_reward_for_valid(self):
        nouns, verbs, adjectives = load_lexicon()
        rf = RewardFunction(nouns, verbs, adjectives,
                            weights=RewardWeights(w_grammar=0.0, w_tag=0.0, w_axis=0.0))
        out = rf.compute(["MAN"], ["RUN"], "MAN RUN", is_valid=True)
        self.assertAlmostEqual(out.reward, 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

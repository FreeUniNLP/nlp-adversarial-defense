"""
DefenderRewardFunction — structured reward module for the defender.

Mirrors the attacker RewardFunction with inverted objectives:

  Attacker wants:  grammar failure, tag mismatch, axis drift
  Defender wants:  grammar success, tag alignment, axis closeness

Reward formula
--------------
    R = w_grammar * grammar_reward          # 1.0 if CFG-valid, else 0.0
      + w_tag     * tag_closeness_score     # 1 - tag_mismatch (per-POS Jaccard)
      + w_axis    * axis_closeness_score    # 1 - axis_distance (cosine closeness)

Feature extraction and distance computation reuse the same components as the
attacker so both agents are scored on identical semantic measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.attacker.reward_function import (
    DistanceCalculator,
    DistanceScores,
    FeatureExtractor,
    RewardFunction,
    SegmentFeatures,
)
from src.language.entities.word_entity import AdjectiveEntry, NounEntry, VerbEntry


@dataclass
class DefenderRewardWeights:
    w_grammar: float = 1.0    # grammar success reward weight
    w_tag:     float = 0.30   # tag alignment weight
    w_axis:    float = 0.20   # axis closeness weight

    @property
    def max_reward(self) -> float:
        """Theoretical maximum: all components = 1.0."""
        return self.w_grammar + self.w_tag + self.w_axis


@dataclass
class DefenderRewardOutput:
    """
    Complete reward output for one attacker–defender exchange (defender view).

    tag_reward and axis_reward are *closeness* scores in [0, 1]:
      1 = fully aligned, 0 = maximally distant.
    """

    reward:          float
    grammar_valid:   bool
    grammar_reward:  float
    distances:       DistanceScores
    tag_reward:      float
    axis_reward:     float
    prefix:          SegmentFeatures
    suffix:          SegmentFeatures
    cfg_error:       Optional[str]
    full_sentence:   str

    def summary(self) -> str:
        lines = [
            f"Sentence      : {self.full_sentence}",
            f"Grammar valid : {self.grammar_valid}"
            + (f"  ({self.cfg_error})" if self.cfg_error else ""),
            f"",
            f"PREFIX features:",
            self.prefix.summary(),
            f"",
            f"SUFFIX features:",
            self.suffix.summary(),
            f"",
            f"Distance scores:",
            self.distances.summary(),
            f"",
            f"Reward breakdown (defender):",
            f"  grammar_reward : {self.grammar_reward:.4f}",
            f"  tag_closeness  : {self.tag_reward:.4f}",
            f"  axis_closeness : {self.axis_reward:.4f}",
            f"  TOTAL          : {self.reward:.4f}",
        ]
        return "\n".join(lines)


class DefenderRewardFunction:
    """
    Reward module for defender RL training.

    Uses the same feature extraction and distance metrics as the attacker,
    but inverts grammar and semantic distance into rewards the defender
    should maximize.
    """

    split_sentence = staticmethod(RewardFunction.split_sentence)

    def __init__(
        self,
        nouns:      list[NounEntry],
        verbs:      list[VerbEntry],
        adjectives: list[AdjectiveEntry],
        weights:    DefenderRewardWeights | None = None,
    ) -> None:
        self._weights   = weights or DefenderRewardWeights()
        self._extractor = FeatureExtractor(nouns, verbs, adjectives)
        self._distance  = DistanceCalculator()

    def compute(
        self,
        prefix_words:  list[str],
        suffix_words:  list[str],
        full_sentence: str,
        is_valid:      bool,
        cfg_error:     Optional[str] = None,
    ) -> DefenderRewardOutput:
        w = self._weights

        prefix_feat = self._extractor.extract(prefix_words)
        suffix_feat = self._extractor.extract(suffix_words)
        distances   = self._distance.compute(prefix_feat, suffix_feat)

        grammar_reward = 1.0 if is_valid else 0.0
        tag_reward     = 1.0 - distances.tag_mismatch
        axis_reward    = 1.0 - distances.axis_distance

        reward = (
            w.w_grammar * grammar_reward
            + w.w_tag   * tag_reward
            + w.w_axis  * axis_reward
        )

        return DefenderRewardOutput(
            reward         = reward,
            grammar_valid  = is_valid,
            grammar_reward = grammar_reward,
            distances      = distances,
            tag_reward     = tag_reward,
            axis_reward    = axis_reward,
            prefix         = prefix_feat,
            suffix         = suffix_feat,
            cfg_error      = cfg_error if not is_valid else None,
            full_sentence  = full_sentence,
        )

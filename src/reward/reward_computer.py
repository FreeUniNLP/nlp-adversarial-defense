"""
RewardComputer — computes the attacker reward after MiniGPT finishes a sentence.

Three reward levels (highest wins, all are combined with weights):

  A) GRAMMAR FAILURE  (weight=1.0)
     Full sentence is invalid under CFG → maximum reward.
     Attacker successfully made the defender produce bad grammar.

  B) TOPIC MISMATCH   (weight=0.5)
     Defender's completion drifts semantically from the attacker prefix.
     Measured by tag-Jaccard distance + cosine distance on axis vectors.

  C) TOPIC CONSISTENCY (weight=0.0 contribution)
     Completion stays aligned → near-zero reward on this axis.

Final reward:
    R = w_grammar * grammar_reward
      + w_mismatch * topic_mismatch_reward

Both component rewards are in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

@dataclass
class RewardConfig:
    w_grammar:  float = 1.0   # weight for grammar-failure reward
    w_mismatch: float = 0.5   # weight for topic-mismatch reward


# ------------------------------------------------------------------ #
#  Topic profile                                                        #
# ------------------------------------------------------------------ #

@dataclass
class TopicProfile:
    """Aggregated semantic profile of a sequence of words."""
    tags:       set[str]        = field(default_factory=set)
    axis:       list[float]     = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    word_count: int             = 0

    def add_word(self, entry) -> None:
        """Add a NounEntry, VerbEntry, or AdjectiveEntry to the profile."""
        if hasattr(entry, "tag") and entry.tag:
            self.tags.update(entry.tag.tag)
        if hasattr(entry, "axis") and entry.axis:
            ax = entry.axis
            self.axis[0] += ax.agency
            self.axis[1] += ax.physicality
            self.axis[2] += ax.social
            self.axis[3] += ax.system
        self.word_count += 1

    def mean_axis(self) -> list[float]:
        """Axis vector normalised by word count (avoids length bias)."""
        if self.word_count == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [v / self.word_count for v in self.axis]


# ------------------------------------------------------------------ #
#  Distance helpers                                                     #
# ------------------------------------------------------------------ #

def _tag_distance(a: set[str], b: set[str]) -> float:
    """
    1 - Jaccard similarity.
    0 = identical tag sets, 1 = completely disjoint.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union


def _cosine_distance(u: list[float], v: list[float]) -> float:
    """
    1 - cosine_similarity, clipped to [0, 1].
    0 = same direction, 1 = orthogonal/opposite.
    """
    dot  = sum(a * b for a, b in zip(u, v))
    nu   = math.sqrt(sum(a * a for a in u))
    nv   = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    cos  = dot / (nu * nv)
    return (1.0 - max(-1.0, min(1.0, cos))) / 2.0   # map [-1,1] → [0,1]


# ------------------------------------------------------------------ #
#  Result dataclass                                                     #
# ------------------------------------------------------------------ #

@dataclass
class RewardResult:
    reward:          float          # final scalar reward ∈ [0, ~1.5]
    grammar_failure: bool           # True if sentence was CFG-invalid
    grammar_reward:  float          # component A ∈ {0, 1}
    tag_distance:    float          # raw tag Jaccard distance ∈ [0, 1]
    axis_distance:   float          # raw cosine distance ∈ [0, 1]
    topic_mismatch:  float          # component B ∈ [0, 1]
    prefix_profile:  TopicProfile
    suffix_profile:  TopicProfile
    cfg_error:       Optional[str]  # validator error message (or None)

    def summary(self) -> str:
        lines = [
            f"  Reward        : {self.reward:.4f}",
            f"  Grammar fail  : {self.grammar_failure}"
            + (f"  ({self.cfg_error})" if self.cfg_error else ""),
            f"  Grammar reward: {self.grammar_reward:.2f}  (w={1.0})",
            f"  Tag distance  : {self.tag_distance:.4f}",
            f"  Axis distance : {self.axis_distance:.4f}",
            f"  Topic mismatch: {self.topic_mismatch:.4f}  (w={0.5})",
            f"  Prefix tags   : {sorted(self.prefix_profile.tags)}",
            f"  Suffix tags   : {sorted(self.suffix_profile.tags)}",
            f"  Prefix axis   : {[round(x,2) for x in self.prefix_profile.mean_axis()]}",
            f"  Suffix axis   : {[round(x,2) for x in self.suffix_profile.mean_axis()]}",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  RewardComputer                                                       #
# ------------------------------------------------------------------ #

class RewardComputer:
    """
    Computes attacker reward from a completed sentence.

    Usage:
        rc = RewardComputer(nouns, verbs, adjectives, config)

        result = rc.compute(
            prefix_words  = ["STRONG", "MAN", "CARRY"],
            suffix_words  = ["SMALL", "BOOK", "FALL"],
            is_valid      = False,        # from CFGValidator
            cfg_error     = "Semantic constraint violated: ...",
        )
        print(result.reward)
    """

    def __init__(
        self,
        nouns:      list[NounEntry],
        verbs:      list[VerbEntry],
        adjectives: list[AdjectiveEntry],
        config:     RewardConfig = None,
    ) -> None:
        self._config = config or RewardConfig()

        # word → entry lookup
        self._lookup: dict[str, NounEntry | VerbEntry | AdjectiveEntry] = {}
        for n in nouns:
            self._lookup[n.word] = n
        for v in verbs:
            self._lookup[v.word] = v
        for a in adjectives:
            self._lookup[a.word] = a

    # ------------------------------------------------------------------ #

    def _build_profile(self, words: list[str]) -> TopicProfile:
        profile = TopicProfile()
        for w in words:
            entry = self._lookup.get(w.upper())
            if entry is not None:
                profile.add_word(entry)
        return profile

    def compute(
        self,
        prefix_words: list[str],
        suffix_words:  list[str],
        is_valid:      bool,
        cfg_error:     Optional[str] = None,
    ) -> RewardResult:
        cfg = self._config

        # --- Component A: grammar failure ---
        grammar_reward  = 1.0 if not is_valid else 0.0

        # --- Component B: topic mismatch ---
        pre_profile = self._build_profile(prefix_words)
        suf_profile = self._build_profile(suffix_words)

        tag_dist  = _tag_distance(pre_profile.tags, suf_profile.tags)
        axis_dist = _cosine_distance(pre_profile.mean_axis(), suf_profile.mean_axis())

        # combine tag and axis into a single mismatch score ∈ [0, 1]
        topic_mismatch = 0.5 * tag_dist + 0.5 * axis_dist

        # --- Final reward ---
        reward = (
            cfg.w_grammar  * grammar_reward
            + cfg.w_mismatch * topic_mismatch
        )

        return RewardResult(
            reward          = reward,
            grammar_failure = not is_valid,
            grammar_reward  = grammar_reward,
            tag_distance    = tag_dist,
            axis_distance   = axis_dist,
            topic_mismatch  = topic_mismatch,
            prefix_profile  = pre_profile,
            suffix_profile  = suf_profile,
            cfg_error       = cfg_error if not is_valid else None,
        )

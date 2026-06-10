"""
RewardFunction — structured reward module for the adversarial attacker.

Responsibilities
----------------
1. Split a completed sentence into attacker prefix and defender suffix.
2. Extract structured features from each part:
     - noun tags, verb tags, adjective tags  (separate per POS)
     - axis vectors (agency, physicality, social, system)
3. Compute CFG validity of the full sentence.
4. Compute semantic distance between prefix and suffix.
5. Return a single scalar reward plus a fully-annotated breakdown.

Reward formula
--------------
    R = w_grammar  * grammar_reward          # 1.0 if CFG-invalid, else 0.0
      + w_tag      * tag_mismatch_score      # weighted Jaccard distance per POS
      + w_axis     * axis_distance_score     # cosine distance of mean axis vectors

Weights (RewardWeights dataclass, all defaults can be overridden):
    w_grammar = 1.0   (grammar failure dominates)
    w_tag     = 0.30  (tag mismatch — medium)
    w_axis    = 0.20  (axis drift — medium)

Tag mismatch score = average of three per-POS Jaccard distances:
    nouns_tag_dist, verbs_tag_dist, adjectives_tag_dist

This gives a fine-grained view of WHICH part of the vocabulary drifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry


# ------------------------------------------------------------------ #
#  Weights                                                             #
# ------------------------------------------------------------------ #

@dataclass
class RewardWeights:
    w_grammar: float = 1.0    # grammar failure reward weight
    w_tag:     float = 0.30   # tag mismatch weight
    w_axis:    float = 0.20   # axis distance weight

    @property
    def max_reward(self) -> float:
        """Theoretical maximum: all components = 1.0."""
        return self.w_grammar + self.w_tag + self.w_axis


# ------------------------------------------------------------------ #
#  Structured feature set for one sentence segment                     #
# ------------------------------------------------------------------ #

@dataclass
class SegmentFeatures:
    """
    Extracted semantic features for one segment (prefix or suffix).

    Attributes
    ----------
    words          : original surface words in this segment
    nouns          : NounEntry objects found in this segment
    verbs          : VerbEntry objects found in this segment
    adjectives     : AdjectiveEntry objects found in this segment
    noun_tags      : union of tags from all nouns
    verb_tags      : union of tags from all verbs
    adjective_tags : union of tags from all adjectives
    all_tags       : union of all tags (across all POS)
    mean_axis      : mean 4D axis vector over all words in segment
    word_count     : number of known words (unknown words excluded)
    """

    words:          list[str]               = field(default_factory=list)
    nouns:          list[NounEntry]         = field(default_factory=list)
    verbs:          list[VerbEntry]         = field(default_factory=list)
    adjectives:     list[AdjectiveEntry]    = field(default_factory=list)
    noun_tags:      set[str]                = field(default_factory=set)
    verb_tags:      set[str]                = field(default_factory=set)
    adjective_tags: set[str]                = field(default_factory=set)
    all_tags:       set[str]                = field(default_factory=set)
    mean_axis:      list[float]             = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    word_count:     int                     = 0

    def summary(self) -> str:
        return (
            f"  words       : {self.words}\n"
            f"  noun_tags   : {sorted(self.noun_tags)}\n"
            f"  verb_tags   : {sorted(self.verb_tags)}\n"
            f"  adj_tags    : {sorted(self.adjective_tags)}\n"
            f"  all_tags    : {sorted(self.all_tags)}\n"
            f"  mean_axis   : {[round(x, 3) for x in self.mean_axis]}"
        )


# ------------------------------------------------------------------ #
#  Pairwise distance scores                                            #
# ------------------------------------------------------------------ #

@dataclass
class DistanceScores:
    """
    All pairwise distance measures between prefix and suffix features.

    Each distance is in [0, 1]:
      0 = identical / fully aligned
      1 = completely disjoint / opposite
    """

    # per-POS tag Jaccard distances
    noun_tag_dist:      float = 0.0
    verb_tag_dist:      float = 0.0
    adjective_tag_dist: float = 0.0

    # combined tag mismatch (mean of the three above)
    tag_mismatch:       float = 0.0

    # axis cosine distance
    axis_distance:      float = 0.0

    def summary(self) -> str:
        return (
            f"  noun_tag_dist : {self.noun_tag_dist:.4f}\n"
            f"  verb_tag_dist : {self.verb_tag_dist:.4f}\n"
            f"  adj_tag_dist  : {self.adjective_tag_dist:.4f}\n"
            f"  tag_mismatch  : {self.tag_mismatch:.4f}  (mean of three)\n"
            f"  axis_distance : {self.axis_distance:.4f}"
        )


# ------------------------------------------------------------------ #
#  Full reward result                                                   #
# ------------------------------------------------------------------ #

@dataclass
class RewardOutput:
    """
    Complete reward output for one attacker–defender exchange.

    Attributes
    ----------
    reward          : final scalar reward
    grammar_failure : True if the full sentence failed CFG validation
    grammar_reward  : component A ∈ {0.0, 1.0}
    distances       : DistanceScores (all pairwise measures)
    tag_reward      : component B — tag mismatch reward ∈ [0, 1]
    axis_reward     : component C — axis distance reward ∈ [0, 1]
    prefix          : SegmentFeatures for the attacker prefix
    suffix          : SegmentFeatures for the defender suffix
    cfg_error       : CFG error message (None if sentence is valid)
    full_sentence   : the complete sentence that was evaluated
    """

    reward:          float
    grammar_failure: bool
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
            f"Grammar fail  : {self.grammar_failure}"
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
            f"Reward breakdown:",
            f"  grammar_reward : {self.grammar_reward:.4f}",
            f"  tag_reward     : {self.tag_reward:.4f}",
            f"  axis_reward    : {self.axis_reward:.4f}",
            f"  TOTAL          : {self.reward:.4f}",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def _jaccard_distance(a: set[str], b: set[str]) -> float:
    """
    1 - Jaccard similarity.
    Returns 0.0 if both sets are empty (identical vacuously).
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union


def _cosine_distance(u: list[float], v: list[float]) -> float:
    """
    (1 - cosine_similarity) / 2, mapped to [0, 1].
    Returns 0.0 if either vector is zero (no information to compare).
    """
    dot = sum(a * b for a, b in zip(u, v))
    nu  = math.sqrt(sum(a * a for a in u))
    nv  = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    cos = dot / (nu * nv)
    return (1.0 - max(-1.0, min(1.0, cos))) / 2.0


# ------------------------------------------------------------------ #
#  Feature extractor                                                   #
# ------------------------------------------------------------------ #

class FeatureExtractor:
    """
    Extracts SegmentFeatures from a list of surface words.

    Classifies each word by POS (noun / verb / adjective) using the
    lexicon lookup built at construction time. Unknown words are skipped.
    """

    def __init__(
        self,
        nouns:      list[NounEntry],
        verbs:      list[VerbEntry],
        adjectives: list[AdjectiveEntry],
    ) -> None:
        self._noun_map: dict[str, NounEntry]       = {n.word: n for n in nouns}
        self._verb_map: dict[str, VerbEntry]       = {v.word: v for v in verbs}
        self._adj_map:  dict[str, AdjectiveEntry]  = {a.word: a for a in adjectives}

    def extract(self, words: list[str]) -> SegmentFeatures:
        """Build a SegmentFeatures from a list of surface-form words."""
        feat = SegmentFeatures(words=list(words))

        axis_sum   = [0.0, 0.0, 0.0, 0.0]
        word_count = 0

        for w in words:
            w_upper = w.upper()

            if w_upper in self._noun_map:
                entry = self._noun_map[w_upper]
                feat.nouns.append(entry)
                feat.noun_tags.update(entry.tag.tag)
                feat.all_tags.update(entry.tag.tag)
                ax = entry.axis
                axis_sum[0] += ax.agency
                axis_sum[1] += ax.physicality
                axis_sum[2] += ax.social
                axis_sum[3] += ax.system
                word_count  += 1

            elif w_upper in self._verb_map:
                entry = self._verb_map[w_upper]
                feat.verbs.append(entry)
                feat.verb_tags.update(entry.tag.tag)
                feat.all_tags.update(entry.tag.tag)
                ax = entry.axis
                axis_sum[0] += ax.agency
                axis_sum[1] += ax.physicality
                axis_sum[2] += ax.social
                axis_sum[3] += ax.system
                word_count  += 1

            elif w_upper in self._adj_map:
                entry = self._adj_map[w_upper]
                feat.adjectives.append(entry)
                feat.adjective_tags.update(entry.tag.tag)
                feat.all_tags.update(entry.tag.tag)
                ax = entry.axis
                axis_sum[0] += ax.agency
                axis_sum[1] += ax.physicality
                axis_sum[2] += ax.social
                axis_sum[3] += ax.system
                word_count  += 1
            # else: unknown word — silently ignored

        feat.word_count = word_count
        if word_count > 0:
            feat.mean_axis = [v / word_count for v in axis_sum]
        else:
            feat.mean_axis = [0.0, 0.0, 0.0, 0.0]

        return feat


# ------------------------------------------------------------------ #
#  Distance calculator                                                 #
# ------------------------------------------------------------------ #

class DistanceCalculator:
    """
    Computes all pairwise semantic distances between two SegmentFeatures.

    Tag distances are computed per POS (noun / verb / adjective) so you
    can tell exactly which part of the vocabulary drifted.
    Combined tag_mismatch = mean of the three per-POS distances.
    """

    @staticmethod
    def compute(prefix: SegmentFeatures, suffix: SegmentFeatures) -> DistanceScores:
        noun_dist = _jaccard_distance(prefix.noun_tags,      suffix.noun_tags)
        verb_dist = _jaccard_distance(prefix.verb_tags,      suffix.verb_tags)
        adj_dist  = _jaccard_distance(prefix.adjective_tags, suffix.adjective_tags)

        # weight the three POS distances equally
        tag_mismatch = (noun_dist + verb_dist + adj_dist) / 3.0

        axis_dist = _cosine_distance(prefix.mean_axis, suffix.mean_axis)

        return DistanceScores(
            noun_tag_dist      = noun_dist,
            verb_tag_dist      = verb_dist,
            adjective_tag_dist = adj_dist,
            tag_mismatch       = tag_mismatch,
            axis_distance      = axis_dist,
        )


# ------------------------------------------------------------------ #
#  Main reward function                                                #
# ------------------------------------------------------------------ #

class RewardFunction:
    """
    Main reward module for the adversarial attack pipeline.

    Usage
    -----
        rf = RewardFunction(nouns, verbs, adjectives)

        output = rf.compute(
            prefix_words  = ["STRONG", "MAN", "CARRY"],
            suffix_words  = ["ALGORITHM", "SPREAD"],
            full_sentence = "STRONG MAN CARRY ALGORITHM SPREAD",
            is_valid      = False,
            cfg_error     = "Semantic constraint violated: ...",
        )

        print(output.reward)         # scalar
        print(output.summary())      # full breakdown
    """

    def __init__(
        self,
        nouns:      list[NounEntry],
        verbs:      list[VerbEntry],
        adjectives: list[AdjectiveEntry],
        weights:    RewardWeights = None,
    ) -> None:
        self._weights   = weights or RewardWeights()
        self._extractor = FeatureExtractor(nouns, verbs, adjectives)
        self._distance  = DistanceCalculator()

    # ------------------------------------------------------------------ #

    @staticmethod
    def split_sentence(
        full_words:   list[str],
        prefix_words: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Split full_words into (prefix, suffix) given the known prefix.

        The prefix is matched by length from the front; the remainder is
        the suffix. If the prefix is longer than the full sentence (edge
        case), suffix is empty.

        Parameters
        ----------
        full_words   : complete sentence as a list of tokens
        prefix_words : attacker-generated prefix tokens

        Returns
        -------
        (prefix_words, suffix_words) — both as lists of strings
        """
        n = len(prefix_words)
        return full_words[:n], full_words[n:]

    # ------------------------------------------------------------------ #

    def compute(
        self,
        prefix_words:  list[str],
        suffix_words:  list[str],
        full_sentence: str,
        is_valid:      bool,
        cfg_error:     Optional[str] = None,
    ) -> RewardOutput:
        """
        Compute the reward for one attacker–defender exchange.

        Parameters
        ----------
        prefix_words  : attacker-generated tokens (already split)
        suffix_words  : defender-generated tokens (already split)
        full_sentence : the joined sentence string (for display / bookkeeping)
        is_valid      : True if CFGValidator accepted the full sentence
        cfg_error     : error string from CFGValidator (or None)

        Returns
        -------
        RewardOutput with scalar reward and full feature breakdown
        """
        w = self._weights

        # --- Step 1: extract features from each segment ---
        prefix_feat = self._extractor.extract(prefix_words)
        suffix_feat = self._extractor.extract(suffix_words)

        # --- Step 2: compute pairwise distances ---
        distances = self._distance.compute(prefix_feat, suffix_feat)

        # --- Step 3: component rewards ---
        grammar_reward = 1.0 if not is_valid else 0.0
        tag_reward     = distances.tag_mismatch      # already ∈ [0, 1]
        axis_reward    = distances.axis_distance     # already ∈ [0, 1]

        # --- Step 4: weighted sum ---
        reward = (
            w.w_grammar * grammar_reward
            + w.w_tag   * tag_reward
            + w.w_axis  * axis_reward
        )

        return RewardOutput(
            reward          = reward,
            grammar_failure = not is_valid,
            grammar_reward  = grammar_reward,
            distances       = distances,
            tag_reward      = tag_reward,
            axis_reward     = axis_reward,
            prefix          = prefix_feat,
            suffix          = suffix_feat,
            cfg_error       = cfg_error if not is_valid else None,
            full_sentence   = full_sentence,
        )

"""
Demo script — shows what RewardFunction does step by step.

Runs three example cases to illustrate each reward level:
  Case A: Grammar failure   → highest reward
  Case B: Topic mismatch    → medium reward
  Case C: Topic consistent  → low reward

Usage:
    python scripts/demo_reward_function.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.parsers import LexiconParser
from src.reward.reward_function import RewardFunction, RewardWeights

WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"

# ------------------------------------------------------------------ #
#  Setup                                                               #
# ------------------------------------------------------------------ #

print("Loading lexicon...")
nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
rf = RewardFunction(nouns, verbs, adjectives)

SEP  = "=" * 65
SEP2 = "-" * 65

# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

def show_case(title, prefix, suffix, is_valid, cfg_error=None):
    full = " ".join(prefix + suffix)
    out  = rf.compute(
        prefix_words  = prefix,
        suffix_words  = suffix,
        full_sentence = full,
        is_valid      = is_valid,
        cfg_error     = cfg_error,
    )

    print(SEP)
    print(f"  {title}")
    print(SEP)

    # 1. Sentence split
    print(f"\n  FULL SENTENCE  : {full or '(empty)'}")
    print(f"  ATTACKER PREFIX: {' '.join(prefix) if prefix else '(empty)'}")
    print(f"  DEFENDER SUFFIX: {' '.join(suffix) if suffix else '(empty)'}")

    # 2. Extracted features
    print(f"\n  {SEP2}")
    print(f"  STEP 1 — Feature extraction")
    print(f"  {SEP2}")

    for seg_name, seg in [("PREFIX", out.prefix), ("SUFFIX", out.suffix)]:
        print(f"\n  [{seg_name}]")
        print(f"    nouns      : {[n.word for n in seg.nouns]}")
        print(f"    verbs      : {[v.word for v in seg.verbs]}")
        print(f"    adjectives : {[a.word for a in seg.adjectives]}")
        print(f"    noun_tags  : {sorted(seg.noun_tags)}")
        print(f"    verb_tags  : {sorted(seg.verb_tags)}")
        print(f"    adj_tags   : {sorted(seg.adjective_tags)}")
        print(f"    mean_axis  : agency={seg.mean_axis[0]:.2f}  "
              f"physicality={seg.mean_axis[1]:.2f}  "
              f"social={seg.mean_axis[2]:.2f}  "
              f"system={seg.mean_axis[3]:.2f}")

    # 3. Distances
    print(f"\n  {SEP2}")
    print(f"  STEP 2 — Semantic distances (prefix vs suffix)")
    print(f"  {SEP2}")
    d = out.distances
    print(f"\n    noun_tag_dist      : {d.noun_tag_dist:.4f}  "
          f"(1 - Jaccard of noun tags)")
    print(f"    verb_tag_dist      : {d.verb_tag_dist:.4f}  "
          f"(1 - Jaccard of verb tags)")
    print(f"    adjective_tag_dist : {d.adjective_tag_dist:.4f}  "
          f"(1 - Jaccard of adjective tags)")
    print(f"    tag_mismatch       : {d.tag_mismatch:.4f}  "
          f"(mean of three above)")
    print(f"    axis_distance      : {d.axis_distance:.4f}  "
          f"(cosine distance of mean axis vectors)")

    # 4. CFG validity
    print(f"\n  {SEP2}")
    print(f"  STEP 3 — CFG validity")
    print(f"  {SEP2}")
    if out.grammar_failure:
        print(f"\n    INVALID  ->  {out.cfg_error}")
    else:
        print(f"\n    VALID")

    # 5. Reward
    w = rf._weights
    print(f"\n  {SEP2}")
    print(f"  STEP 4 — Reward computation")
    print(f"  {SEP2}")
    print(f"\n    grammar_reward  = {out.grammar_reward:.2f}  x  w_grammar={w.w_grammar}  "
          f"=  {w.w_grammar * out.grammar_reward:.4f}")
    print(f"    tag_reward      = {out.tag_reward:.4f}  x  w_tag={w.w_tag}      "
          f"=  {w.w_tag * out.tag_reward:.4f}")
    print(f"    axis_reward     = {out.axis_reward:.4f}  x  w_axis={w.w_axis}     "
          f"=  {w.w_axis * out.axis_reward:.4f}")
    print(f"    {'-'*40}")
    print(f"    TOTAL REWARD    =  {out.reward:.4f}")
    print()


# ------------------------------------------------------------------ #
#  Case A — Grammar failure                                            #
# ------------------------------------------------------------------ #

show_case(
    title     = "CASE A — Grammar failure (highest reward)",
    prefix    = ["RIVER"],
    suffix    = ["BURN"],
    is_valid  = False,
    cfg_error = "Semantic constraint violated: Verb 'BURN' is incompatible with subject 'RIVER'",
)

# ------------------------------------------------------------------ #
#  Case B — Topic mismatch (valid sentence, very different domains)   #
# ------------------------------------------------------------------ #

show_case(
    title    = "CASE B — Topic mismatch (medium reward)",
    prefix   = ["MAN", "WOMAN", "CARRY"],   # human / physical domain
    suffix   = ["ALGORITHM", "SPREAD"],      # machine / system domain
    is_valid = True,
)

# ------------------------------------------------------------------ #
#  Case C — Topic consistent (low reward)                             #
# ------------------------------------------------------------------ #

show_case(
    title    = "CASE C — Topic consistent (low reward)",
    prefix   = ["MAN", "CARRY"],    # human carries something
    suffix   = ["CHILD"],           # also human — same domain
    is_valid = True,
)

# ------------------------------------------------------------------ #
#  Summary comparison                                                  #
# ------------------------------------------------------------------ #

print(SEP)
print("  SUMMARY — Reward comparison across cases")
print(SEP)
print()

cases = [
    ("A  Grammar failure",  ["RIVER"],             ["BURN"],              False, "Semantic constraint violated"),
    ("B  Topic mismatch",   ["MAN","WOMAN","CARRY"],["ALGORITHM","SPREAD"],True,  None),
    ("C  Topic consistent", ["MAN","CARRY"],        ["CHILD"],             True,  None),
]

print(f"  {'Case':<22} {'Grammar':>8} {'Tag':>8} {'Axis':>8} {'TOTAL':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for label, pre, suf, valid, err in cases:
    out = rf.compute(
        prefix_words=" ".join(pre).split(), suffix_words=" ".join(suf).split(),
        full_sentence=" ".join(pre + suf), is_valid=valid, cfg_error=err,
    )
    w = rf._weights
    print(f"  {label:<22} "
          f"{w.w_grammar*out.grammar_reward:>8.4f} "
          f"{w.w_tag*out.tag_reward:>8.4f} "
          f"{w.w_axis*out.axis_reward:>8.4f} "
          f"{out.reward:>8.4f}")

print()
print(f"  Weights:  w_grammar={rf._weights.w_grammar}  "
      f"w_tag={rf._weights.w_tag}  "
      f"w_axis={rf._weights.w_axis}  "
      f"max_possible={rf._weights.max_reward:.2f}")
print()

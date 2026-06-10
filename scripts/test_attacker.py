"""
Manual test script for CFGStateTracker and AttackerTransformer.

Runs a series of labelled checks and prints PASS / FAIL for each.

Usage:
    python scripts/test_attacker.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from src.attacker.cfg_state_tracker import CFGStateTracker, GrammarState
from src.attacker.attacker import AttackerTransformer
from src.model.tokenizer import WordTokenizer

WORDS_PATH  = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"

# ------------------------------------------------------------------ #
#  Tiny test runner                                                    #
# ------------------------------------------------------------------ #

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  [PASS]  {label}")
        passed += 1
    else:
        print(f"  [FAIL]  {label}" + (f"  ->  {detail}" if detail else ""))
        failed += 1

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ------------------------------------------------------------------ #
#  Setup                                                               #
# ------------------------------------------------------------------ #

print("Loading lexicon and tokenizer...")
nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)
tracker   = CFGStateTracker(nouns, verbs, adjectives)
attacker  = AttackerTransformer(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)

TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
cfg       = CFG.from_json(str(TRANSITION_PATH), nouns=nouns, verbs=verbs, adjectives=adjectives)
validator = CFGValidator.from_cfg(cfg)

noun_set  = {n.word for n in nouns}
verb_set  = {v.word for v in verbs}
adj_set   = {a.word for a in adjectives}
intrans   = {v.word for v in verbs if v.verb_argument.verb_to_object_constraint is None}
trans     = {v.word for v in verbs if v.verb_argument.verb_to_object_constraint is not None}


# ================================================================== #
#  CFGStateTracker tests                                               #
# ================================================================== #

section("CFGStateTracker — initial state")

tracker.reset()
words, can_end = tracker.valid_next_words()

check("State is SUBJECT_START",         tracker.state == GrammarState.SUBJECT_START)
check("can_end is False at start",      can_end == False)
check("can_end property is False",      tracker.can_end == False)
check("Valid words include nouns",      bool(noun_set & set(words)))
check("Valid words include adjectives", bool(adj_set  & set(words)))
check("Valid words exclude verbs",      not bool(verb_set & set(words)))
check("Generated is empty",            tracker.generated == [])
check("sentence() is empty string",    tracker.sentence() == "")


section("CFGStateTracker — step: NOUN as subject")

tracker.reset()
ok = tracker.step("MAN")
words, can_end = tracker.valid_next_words()

check("step('MAN') returns True",             ok)
check("State moves to AFTER_SUBJECT",         tracker.state == GrammarState.AFTER_SUBJECT)
check("subject_noun is MAN",                  tracker.subject_noun.word == "MAN")
check("subject_adjs is empty",                tracker.subject_adjs == [])
check("can_end is True after subject",        can_end)
check("Only verbs allowed after subject",     set(words).issubset(verb_set),
      f"Non-verbs found: {set(words) - verb_set}")
check("generated == ['MAN']",                 tracker.generated == ["MAN"])
check("sentence() == 'MAN'",                  tracker.sentence() == "MAN")


section("CFGStateTracker — step: ADJ then NOUN as subject")

tracker.reset()
# pick any adjective from valid words
adj_word = next(w for w in tracker.valid_next_words()[0] if w in adj_set)
tracker.step(adj_word)

check(f"After ADJ '{adj_word}': state is SUBJECT_AFTER_ADJ",
      tracker.state == GrammarState.SUBJECT_AFTER_ADJ)
check("subject_adjs has one entry",  len(tracker.subject_adjs) == 1)
check("can_end is True (prefix can stop after ADJ)", tracker.can_end)

after_adj_words, _ = tracker.valid_next_words()
check("After ADJ: only nouns allowed",  set(after_adj_words).issubset(noun_set),
      f"Non-nouns: {set(after_adj_words) - noun_set}")
check("After ADJ: no verbs",   not bool(verb_set & set(after_adj_words)))
check("After ADJ: no adjs",    not bool(adj_set  & set(after_adj_words)))

noun_word = after_adj_words[0]
tracker.step(noun_word)
check(f"After NOUN '{noun_word}': state is AFTER_SUBJECT",
      tracker.state == GrammarState.AFTER_SUBJECT)
check("subject_noun is set", tracker.subject_noun is not None)


section("CFGStateTracker — intransitive verb")

tracker.reset()
tracker.step("MAN")
valid_verbs, _ = tracker.valid_next_words()
intrans_choice = next((w for w in valid_verbs if w in intrans), None)

if intrans_choice:
    tracker.step(intrans_choice)
    check(f"After intrans '{intrans_choice}': state is AFTER_INTRANS_VERB",
          tracker.state == GrammarState.AFTER_INTRANS_VERB)
    check("can_end after intrans verb", tracker.can_end)
    next_words, can_end = tracker.valid_next_words()
    check("After intrans: no further words allowed (cannot chain verb without object)",
          len(next_words) == 0)
else:
    print("  [SKIP]  No intransitive verb valid for MAN")


section("CFGStateTracker — transitive verb -> object")

tracker.reset()
tracker.step("MAN")
valid_verbs, _ = tracker.valid_next_words()
trans_choice = next((w for w in valid_verbs if w in trans), None)

if trans_choice:
    tracker.step(trans_choice)
    check(f"After trans '{trans_choice}': state is OBJECT_START",
          tracker.state == GrammarState.OBJECT_START)
    check("can_end (prefix can stop before object)", tracker.can_end)

    obj_words, _ = tracker.valid_next_words()
    has_noun = bool(noun_set & set(obj_words))
    has_adj  = bool(adj_set  & set(obj_words))
    check("Object start: nouns allowed",      has_noun)
    check("Object start: adjectives allowed", has_adj)

    # place a noun object
    obj_noun = next(w for w in obj_words if w in noun_set)
    tracker.step(obj_noun)
    check(f"After object noun '{obj_noun}': state is AFTER_OBJECT",
          tracker.state == GrammarState.AFTER_OBJECT)
    check("can_end after object", tracker.can_end)
else:
    print("  [SKIP]  No transitive verb valid for MAN")


section("CFGStateTracker — invalid steps")

tracker.reset()
result = tracker.step("RUN")   # verb at subject start
check("VERB at subject start returns False",  result == False)
check("State unchanged after invalid step",   tracker.state == GrammarState.SUBJECT_START)
check("Generated unchanged after invalid",    tracker.generated == [])

tracker.reset()
tracker.step("MAN")
result = tracker.step("MAN")  # noun after subject noun = invalid
check("NOUN after NOUN (no verb) returns False", result == False)
check("State still AFTER_SUBJECT",               tracker.state == GrammarState.AFTER_SUBJECT)


section("CFGStateTracker — reset")

tracker.reset()
tracker.step("MAN")
tracker.reset()
check("State reset to SUBJECT_START",    tracker.state == GrammarState.SUBJECT_START)
check("generated cleared",               tracker.generated == [])
check("subject_noun cleared",            tracker.subject_noun is None)
check("subject_adjs cleared",            tracker.subject_adjs == [])
check("current_verb cleared",            tracker.current_verb is None)


section("CFGStateTracker — full valid sequence replay")

sequences = [
    ["MAN", "RUN"],
]
# build a longer one from valid corpus
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and len(line.split()) >= 4:
            sequences.append(line.split())
            break

for seq in sequences:
    tracker.reset()
    all_ok = True
    for word in seq:
        if not tracker.step(word):
            all_ok = False
            check(f"Replay '{' '.join(seq)}'", False, f"Failed at '{word}'")
            break
    if all_ok:
        check(f"Replay '{' '.join(seq)}'", True)


# ================================================================== #
#  AttackerTransformer tests                                           #
# ================================================================== #

section("AttackerTransformer — forward pass")

import torch
x = torch.tensor([[tokenizer.bos_id, 10, 20]])
logits = attacker(x)
B, T, V = logits.shape
check("Output shape B=1",               B == 1)
check("Output shape T=3",               T == 3)
check("Output shape V=vocab_size",      V == tokenizer.vocab_size)


section("AttackerTransformer — generate_prefix  (validated by CFGValidator)")

print()
print(f"  {'#':<4} {'VERDICT':<12} {'PREFIX'}")
print(f"  {'-'*4} {'-'*12} {'-'*40}")

for i in range(5):
    ids, words = attacker.generate_prefix(
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        cfg_tracker=tracker,
        token_to_id=tokenizer.token_to_id,
        id_to_token=tokenizer.id_to_token,
        max_tokens=8,
        temperature=1.0,
    )
    prefix_str = " ".join(words) if words else "(empty)"

    # basic checks
    ids_match  = all(tokenizer.id_to_token[t] == w for t, w in zip(ids, words))
    no_special = all(t not in {tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id} for t in ids)

    # CFGValidator check
    # A prefix is either:
    #   COMPLETE  — a full valid sentence (validator says True)
    #   PARTIAL   — incomplete but structurally sound (validator says "Skeleton invalid"
    #               or "No VERB found", meaning it's a valid partial parse, not a semantic error)
    #   INVALID   — violates semantic constraints (tag/axis mismatch) — a real error
    result = validator.validate(prefix_str) if words else None
    if result is None:
        verdict, structurally_ok, error_msg = "EMPTY",    False, "empty prefix"
    elif result.is_valid:
        verdict, structurally_ok, error_msg = "COMPLETE", True,  ""
    elif result.error and ("Skeleton invalid" in result.error or "No VERB" in result.error):
        verdict, structurally_ok, error_msg = "PARTIAL",  True,  result.error
    else:
        verdict, structurally_ok, error_msg = "INVALID",  False, result.error or ""

    print(f"  {i+1:<4} {verdict:<12} {prefix_str}")
    if error_msg and not structurally_ok:
        print(f"       {'':12} reason: {error_msg}")

    check(f"Prefix {i+1}: ids match words",                   ids_match)
    check(f"Prefix {i+1}: no special tokens",                 no_special)
    check(f"Prefix {i+1}: structurally valid (CFGValidator)", structurally_ok, error_msg)
    print()
    print()

check("max_tokens=1 respected", len(attacker.generate_prefix(
    tokenizer.bos_id, tokenizer.eos_id, tracker,
    tokenizer.token_to_id, tokenizer.id_to_token, max_tokens=1)[0]) <= 1)

check("max_tokens=3 respected", len(attacker.generate_prefix(
    tokenizer.bos_id, tokenizer.eos_id, tracker,
    tokenizer.token_to_id, tokenizer.id_to_token, max_tokens=3)[0]) <= 3)


section("AttackerTransformer — generate_prefix_with_log_probs")

ids, words, log_probs = attacker.generate_prefix_with_log_probs(
    bos_id=tokenizer.bos_id,
    eos_id=tokenizer.eos_id,
    cfg_tracker=tracker,
    token_to_id=tokenizer.token_to_id,
    id_to_token=tokenizer.id_to_token,
    max_tokens=8,
    temperature=1.0,
)

check("log_probs length == prefix length",  log_probs.shape[0] == len(ids))
check("log_probs are <= 0",
      log_probs.numel() == 0 or bool((log_probs <= 0).all()),
      f"Got: {log_probs.tolist()}")
check("log_probs have gradient",
      log_probs.numel() == 0 or log_probs.requires_grad or log_probs.grad_fn is not None)

print(f"\n  Prefix : '{' '.join(words)}'")
print(f"  IDs    : {ids}")
print(f"  Logprob: {[round(x, 3) for x in log_probs.detach().tolist()]}")


# ================================================================== #
#  Summary                                                             #
# ================================================================== #

total = passed + failed
print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed  |  {failed} failed")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)

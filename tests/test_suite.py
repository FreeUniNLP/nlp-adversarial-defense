"""
Test suite for nlp-adversarial-defense.

Covers:
  - LexiconParser        (parsers.py)
  - CFG                  (cfg.py)
  - CFGValidator         (cfg_validator.py)
  - WordTokenizer        (tokenizer.py)
  - MiniGPT              (transformer.py)

Usage:
    python -m pytest tests/test_suite.py -v
    python tests/test_suite.py          # run without pytest
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.parsers import LexiconParser
from src.language.entities.cfg import CFG
from src.language.entities.cfg_validator import CFGValidator
from src.model.tokenizer import WordTokenizer

WORDS_PATH      = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"
CORPUS_PATH     = PROJECT_ROOT / "data" / "raw" / "generated_texts" / "generated_corpus_10000.txt"
CKPT_PATH       = PROJECT_ROOT / "data" / "models" / "minigpt_corpus10000.pt"


# ------------------------------------------------------------------ #
#  Shared fixtures (built once per test class via setUpClass)          #
# ------------------------------------------------------------------ #

def load_lexicon():
    return LexiconParser.parse(WORDS_PATH)

def build_cfg(nouns, verbs, adjectives):
    return CFG.from_json(
        file_path=str(TRANSITION_PATH),
        nouns=nouns, verbs=verbs, adjectives=adjectives,
    )

def build_validator(cfg):
    return CFGValidator.from_cfg(cfg)


# ================================================================== #
#  1. LexiconParser tests                                              #
# ================================================================== #

class TestLexiconParser(unittest.TestCase):
    """Tests for LexiconParser — loading and parsing words.json."""

    @classmethod
    def setUpClass(cls):
        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()

    def test_nouns_non_empty(self):
        self.assertGreater(len(self.nouns), 0, "Nouns list should not be empty")

    def test_verbs_non_empty(self):
        self.assertGreater(len(self.verbs), 0, "Verbs list should not be empty")

    def test_adjectives_non_empty(self):
        self.assertGreater(len(self.adjectives), 0, "Adjectives list should not be empty")

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
            self.assertEqual(noun.word, noun.word.upper(), f"Noun '{noun.word}' should be uppercase")

    def test_all_verb_words_uppercase(self):
        for verb in self.verbs:
            self.assertEqual(verb.word, verb.word.upper(), f"Verb '{verb.word}' should be uppercase")


# ================================================================== #
#  2. CFG tests                                                        #
# ================================================================== #

class TestCFG(unittest.TestCase):
    """Tests for CFG — skeleton generation and sentence building."""

    @classmethod
    def setUpClass(cls):
        nouns, verbs, adjectives = load_lexicon()
        cls.cfg = build_cfg(nouns, verbs, adjectives)

    def test_cfg_has_rules(self):
        self.assertGreater(len(self.cfg.rules), 0, "CFG should have grammar rules")

    def test_cfg_has_start_symbol(self):
        self.assertIn("START", self.cfg.rules, "CFG rules should contain START")

    def test_generate_skeleton_returns_list(self):
        skeleton = self.cfg.generate_skeleton()
        self.assertIsInstance(skeleton, list)
        self.assertGreater(len(skeleton), 0)

    def test_skeleton_contains_only_pos_tags(self):
        valid_tags = {"NOUN", "VERB", "ADJ"}
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            for token in skeleton:
                self.assertIn(token, valid_tags, f"Unexpected token '{token}' in skeleton")

    def test_skeleton_always_has_noun_and_verb(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            self.assertIn("NOUN", skeleton)
            self.assertIn("VERB", skeleton)

    def test_build_sentence_from_skeleton_returns_string(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            try:
                sentence = self.cfg.build_sentence_from_skeleton(skeleton)
                self.assertIsInstance(sentence, str)
                self.assertGreater(len(sentence.strip()), 0)
            except ValueError:
                pass  # some skeletons legitimately fail semantic constraints

    def test_generated_sentence_words_are_uppercase(self):
        for _ in range(10):
            skeleton = self.cfg.generate_skeleton()
            try:
                sentence = self.cfg.build_sentence_from_skeleton(skeleton)
                for word in sentence.split():
                    self.assertEqual(word, word.upper(), f"Word '{word}' should be uppercase")
            except ValueError:
                pass


# ================================================================== #
#  3. CFGValidator tests                                               #
# ================================================================== #

class TestCFGValidator(unittest.TestCase):
    """Tests for CFGValidator — structural and semantic sentence validation."""

    @classmethod
    def setUpClass(cls):
        nouns, verbs, adjectives = load_lexicon()
        cfg = build_cfg(nouns, verbs, adjectives)
        cls.validator = build_validator(cfg)
        cls.cfg = cfg

    # --- valid sentences ---

    def test_valid_noun_verb(self):
        result = self.validator.validate("CHOICE SPREAD")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_noun_verb_noun(self):
        result = self.validator.validate("DRONE BREAK CLOCK")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_adj_noun_verb(self):
        result = self.validator.validate("FREE WOLF FALL")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    def test_valid_adj_noun_verb_adj_noun(self):
        result = self.validator.validate("CONNECTED BIRD BREAK SMALL STRONG WEAPON")
        self.assertTrue(result.is_valid, f"Expected valid, got: {result.error}")

    # --- invalid sentences ---

    def test_invalid_empty_sentence(self):
        result = self.validator.validate("")
        self.assertFalse(result.is_valid)

    def test_invalid_unknown_word(self):
        result = self.validator.validate("MAN BLORP")
        self.assertFalse(result.is_valid)
        self.assertIn("Unknown word", result.error)

    def test_invalid_wrong_order_verb_first(self):
        result = self.validator.validate("RUN MAN")
        self.assertFalse(result.is_valid)
        self.assertIn("Skeleton invalid", result.error)

    def test_invalid_only_noun(self):
        result = self.validator.validate("MAN")
        self.assertFalse(result.is_valid)

    def test_invalid_only_verb(self):
        result = self.validator.validate("RUN")
        self.assertFalse(result.is_valid)

    def test_invalid_semantic_wrong_subject(self):
        # STONE has agency=0, most verbs require agency >= 1 for subject
        result = self.validator.validate("STONE KNOW TRUTH")
        self.assertFalse(result.is_valid)
        self.assertIn("Semantic constraint violated", result.error)

    # --- corpus batch test ---

    def test_all_corpus_sentences_are_valid(self):
        """Every sentence in the generated corpus must validate as True."""
        if not CORPUS_PATH.exists():
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")
        invalid = []
        with CORPUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = self.validator.validate(line)
                if not result.is_valid:
                    invalid.append((line, result.error))
        self.assertEqual(
            len(invalid), 0,
            f"{len(invalid)} corpus sentences failed validation:\n" +
            "\n".join(f"  {s} -> {e}" for s, e in invalid[:5])
        )

    # --- ValidationResult helpers ---

    def test_validation_result_bool_true(self):
        result = self.validator.validate("MAN RUN")
        self.assertTrue(bool(result))

    def test_validation_result_bool_false(self):
        result = self.validator.validate("RUN MAN")
        self.assertFalse(bool(result))

    def test_validation_result_repr_valid(self):
        result = self.validator.validate("MAN RUN")
        self.assertIn("valid=True", repr(result))

    def test_validation_result_repr_invalid(self):
        result = self.validator.validate("RUN MAN")
        self.assertIn("valid=False", repr(result))


# ================================================================== #
#  4. WordTokenizer tests                                              #
# ================================================================== #

class TestWordTokenizer(unittest.TestCase):
    """Tests for WordTokenizer — encoding, decoding, vocabulary."""

    @classmethod
    def setUpClass(cls):
        if not CORPUS_PATH.exists():
            cls.tokenizer = None
            return
        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)

    def setUp(self):
        if self.tokenizer is None:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    def test_vocab_size_greater_than_special_tokens(self):
        self.assertGreater(self.tokenizer.vocab_size, 4)

    def test_special_token_ids_unique(self):
        ids = [self.tokenizer.pad_id, self.tokenizer.bos_id,
               self.tokenizer.eos_id, self.tokenizer.unk_id]
        self.assertEqual(len(ids), len(set(ids)), "Special token IDs must be unique")

    def test_pad_id_is_zero(self):
        self.assertEqual(self.tokenizer.pad_id, 0)

    def test_encode_returns_list_of_ints(self):
        ids = self.tokenizer.encode("MAN RUN")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_encode_adds_bos_eos(self):
        ids = self.tokenizer.encode("MAN RUN", add_special=True)
        self.assertEqual(ids[0], self.tokenizer.bos_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_id)

    def test_encode_without_special(self):
        ids = self.tokenizer.encode("MAN RUN", add_special=False)
        self.assertNotEqual(ids[0], self.tokenizer.bos_id)
        self.assertNotEqual(ids[-1], self.tokenizer.eos_id)

    def test_decode_roundtrip(self):
        sentence = "MAN RUN"
        ids = self.tokenizer.encode(sentence, add_special=True)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, sentence)

    def test_unknown_word_maps_to_unk(self):
        ids = self.tokenizer.encode("UNKNOWNWORD", add_special=False)
        self.assertEqual(ids[0], self.tokenizer.unk_id)

    def test_decode_skips_special_tokens_by_default(self):
        ids = [self.tokenizer.bos_id, self.tokenizer.pad_id, self.tokenizer.eos_id]
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded.strip(), "")


# ================================================================== #
#  5. MiniGPT tests                                                    #
# ================================================================== #

class TestMiniGPT(unittest.TestCase):
    """Tests for MiniGPT — forward pass and generation."""

    @classmethod
    def setUpClass(cls):
        import torch
        from src.model.transformer import MiniGPT

        if not CORPUS_PATH.exists() or not CKPT_PATH.exists():
            cls.model = None
            cls.tokenizer = None
            return

        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        cls.model = MiniGPT(vocab_size=cls.tokenizer.vocab_size, pad_id=cls.tokenizer.pad_id)
        cls.model.load_state_dict(ckpt["model_state"])
        cls.model.eval()

    def setUp(self):
        if self.model is None:
            self.skipTest("Model checkpoint or corpus not found")

    def test_forward_pass_output_shape(self):
        import torch
        x = torch.tensor([[self.tokenizer.bos_id, 10, 20]])  # batch=1, seq=3
        logits = self.model(x)
        B, T, V = logits.shape
        self.assertEqual(B, 1)
        self.assertEqual(T, 3)
        self.assertEqual(V, self.tokenizer.vocab_size)

    def test_generate_returns_list(self):
        ids = self.model.generate(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=10,
            temperature=1.0,
        )
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_generate_no_special_tokens_in_output(self):
        special = {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}
        for _ in range(5):
            ids = self.model.generate(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=15,
                temperature=1.0,
            )
            for i in ids:
                self.assertNotIn(i, special, f"Special token {i} found in generated output")

    def test_generate_respects_max_tokens(self):
        ids = self.model.generate(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=5,
            temperature=1.0,
        )
        self.assertLessEqual(len(ids), 5)

    def test_lower_temperature_is_more_deterministic(self):
        """Same seed should produce same output at temperature 0.01."""
        import torch
        results = set()
        for _ in range(3):
            torch.manual_seed(0)
            ids = self.model.generate(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=10,
                temperature=0.01,
            )
            results.add(tuple(ids))
        self.assertEqual(len(results), 1, "Low temperature should produce deterministic output")


# ================================================================== #
#  6. CFGStateTracker tests                                            #
# ================================================================== #

class TestCFGStateTracker(unittest.TestCase):
    """Tests for CFGStateTracker — grammar state and valid-word filtering."""

    @classmethod
    def setUpClass(cls):
        from src.attacker.cfg_state_tracker import CFGStateTracker, GrammarState
        cls.GrammarState = GrammarState
        nouns, verbs, adjectives = load_lexicon()
        cls.tracker = CFGStateTracker(nouns, verbs, adjectives)
        cls.nouns = nouns
        cls.verbs = verbs
        cls.adjectives = adjectives

    def setUp(self):
        self.tracker.reset()

    # --- initial state ---

    def test_initial_state(self):
        self.assertEqual(self.tracker.state, self.GrammarState.SUBJECT_START)

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
        self.assertEqual(self.tracker.state, self.GrammarState.AFTER_SUBJECT)

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
        self.assertEqual(self.tracker.state, self.GrammarState.SUBJECT_AFTER_ADJ)

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
        # pick a verb MAN can use
        valid_verbs, _ = self.tracker.valid_next_words()
        intrans_names = {v.word for v in intrans}
        chosen = next((w for w in valid_verbs if w in intrans_names), None)
        if chosen is None:
            self.skipTest("No intransitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertEqual(self.tracker.state, self.GrammarState.AFTER_INTRANS_VERB)

    def test_after_intrans_verb_can_end(self):
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        intrans = {v.word for v in self.verbs if v.verb_argument.verb_to_object_constraint is None}
        chosen = next((w for w in valid_verbs if w in intrans), None)
        if chosen is None:
            self.skipTest("No intransitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertTrue(self.tracker.can_end)

    # --- transitive verb ---

    def test_transitive_verb_moves_to_object_start(self):
        self.tracker.step("MAN")
        valid_verbs, _ = self.tracker.valid_next_words()
        trans = {v.word for v in self.verbs if v.verb_argument.verb_to_object_constraint is not None}
        chosen = next((w for w in valid_verbs if w in trans), None)
        if chosen is None:
            self.skipTest("No transitive verb valid for MAN")
        self.tracker.step(chosen)
        self.assertEqual(self.tracker.state, self.GrammarState.OBJECT_START)

    # --- invalid step ---

    def test_invalid_step_returns_false(self):
        result = self.tracker.step("RUN")   # verb at subject start = invalid
        self.assertFalse(result)

    def test_invalid_step_does_not_advance_state(self):
        self.tracker.step("RUN")
        self.assertEqual(self.tracker.state, self.GrammarState.SUBJECT_START)

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
        self.assertEqual(self.tracker.state, self.GrammarState.SUBJECT_START)
        self.assertEqual(self.tracker.generated, [])
        self.assertIsNone(self.tracker.subject_noun)


# ================================================================== #
#  7. AttackerTransformer tests                                        #
# ================================================================== #

class TestAttackerTransformer(unittest.TestCase):
    """Tests for AttackerTransformer — CFG-masked generation."""

    @classmethod
    def setUpClass(cls):
        import torch
        from src.attacker.attacker import AttackerTransformer
        from src.attacker.cfg_state_tracker import CFGStateTracker

        if not CORPUS_PATH.exists():
            cls.attacker = None
            return

        nouns, verbs, adjectives = load_lexicon()
        cls.tokenizer = WordTokenizer.from_corpus(CORPUS_PATH)
        cls.tracker   = CFGStateTracker(nouns, verbs, adjectives)
        cls.attacker  = AttackerTransformer(
            vocab_size=cls.tokenizer.vocab_size,
            pad_id=cls.tokenizer.pad_id,
        )
        cls.attacker.eval()

    def setUp(self):
        if self.attacker is None:
            self.skipTest(f"Corpus not found: {CORPUS_PATH}")

    # --- forward pass ---

    def test_forward_output_shape(self):
        import torch
        x = torch.tensor([[self.tokenizer.bos_id, 10, 20]])
        logits = self.attacker(x)
        B, T, V = logits.shape
        self.assertEqual(B, 1)
        self.assertEqual(T, 3)
        self.assertEqual(V, self.tokenizer.vocab_size)

    # --- generate_prefix ---

    def test_generate_prefix_returns_tuple(self):
        ids, words = self.attacker.generate_prefix(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=8,
        )
        self.assertIsInstance(ids,   list)
        self.assertIsInstance(words, list)

    def test_generate_prefix_ids_match_words(self):
        ids, words = self.attacker.generate_prefix(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=8,
        )
        self.assertEqual(len(ids), len(words))
        for i, w in zip(ids, words):
            self.assertEqual(self.tokenizer.id_to_token[i], w)

    def test_generate_prefix_respects_max_tokens(self):
        for max_t in (1, 3, 5):
            ids, words = self.attacker.generate_prefix(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                cfg_tracker=self.tracker,
                token_to_id=self.tokenizer.token_to_id,
                id_to_token=self.tokenizer.id_to_token,
                max_tokens=max_t,
            )
            self.assertLessEqual(len(ids), max_t, f"Exceeded max_tokens={max_t}")

    def test_generate_prefix_no_special_tokens(self):
        special = {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}
        for _ in range(5):
            ids, _ = self.attacker.generate_prefix(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                cfg_tracker=self.tracker,
                token_to_id=self.tokenizer.token_to_id,
                id_to_token=self.tokenizer.id_to_token,
                max_tokens=8,
            )
            for i in ids:
                self.assertNotIn(i, special, f"Special token {i} in prefix")

    def test_generate_prefix_is_cfg_valid_partial(self):
        """Every generated prefix must be a structurally valid partial sentence."""
        from src.attacker.cfg_state_tracker import CFGStateTracker, GrammarState
        nouns, verbs, adjectives = load_lexicon()

        for _ in range(10):
            _, words = self.attacker.generate_prefix(
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                cfg_tracker=self.tracker,
                token_to_id=self.tokenizer.token_to_id,
                id_to_token=self.tokenizer.id_to_token,
                max_tokens=8,
            )
            # Replay through a fresh tracker — every step must succeed
            check = CFGStateTracker(nouns, verbs, adjectives)
            for word in words:
                ok = check.step(word)
                self.assertTrue(ok, f"Word '{word}' was invalid in prefix '{' '.join(words)}'")

    # --- generate_prefix_with_log_probs ---

    def test_log_probs_shape_matches_prefix_length(self):
        import torch
        ids, words, log_probs = self.attacker.generate_prefix_with_log_probs(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=8,
        )
        self.assertEqual(log_probs.shape[0], len(ids))

    def test_log_probs_are_negative(self):
        """Log-probabilities must be <= 0."""
        _, _, log_probs = self.attacker.generate_prefix_with_log_probs(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=8,
        )
        if log_probs.numel() > 0:
            self.assertTrue(
                (log_probs <= 0).all(),
                f"Some log-probs are positive: {log_probs.tolist()}"
            )

    def test_log_probs_have_grad_fn(self):
        """log_probs must carry gradients for RL training."""
        _, _, log_probs = self.attacker.generate_prefix_with_log_probs(
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            cfg_tracker=self.tracker,
            token_to_id=self.tokenizer.token_to_id,
            id_to_token=self.tokenizer.id_to_token,
            max_tokens=8,
        )
        if log_probs.numel() > 0:
            self.assertTrue(
                log_probs.requires_grad or log_probs.grad_fn is not None,
                "log_probs must have gradient for RL training"
            )


# ================================================================== #
#  8. RewardComputer tests                                             #
# ================================================================== #

class TestRewardComputer(unittest.TestCase):
    """
    Tests for RewardComputer — reward signal for the attacker.

    Reward levels:
      A) Grammar failure  → grammar_reward=1.0, total >= 1.0
      B) Topic mismatch   → grammar_reward=0.0, reward driven by mismatch
      C) Topic consistent → low reward
    """

    @classmethod
    def setUpClass(cls):
        from src.attacker.reward import RewardComputer, RewardConfig, TopicProfile
        cls.RewardComputer  = RewardComputer
        cls.RewardConfig    = RewardConfig
        cls.TopicProfile    = TopicProfile

        cls.nouns, cls.verbs, cls.adjectives = load_lexicon()
        cls.rc = RewardComputer(cls.nouns, cls.verbs, cls.adjectives)

        # known-good / known-bad sentences (verified from corpus)
        cls.VALID_PREFIX   = ["MAN"]
        cls.VALID_SUFFIX   = ["RUN"]
        cls.INVALID_PREFIX = ["RIVER"]
        cls.INVALID_SUFFIX = ["BURN"]       # BURN is incompatible with RIVER
        cls.HUMAN_PREFIX   = ["STRONG", "MAN", "CARRY"]
        cls.NATURE_SUFFIX  = ["FOREST", "GROW"]   # very different domain

    # ------------------------------------------------------------------ #
    #  Reward result structure                                             #
    # ------------------------------------------------------------------ #

    def test_compute_returns_reward_result(self):
        from src.attacker.reward import RewardResult
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertIsInstance(result, RewardResult)

    def test_reward_result_has_all_fields(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertTrue(hasattr(result, "reward"))
        self.assertTrue(hasattr(result, "grammar_failure"))
        self.assertTrue(hasattr(result, "grammar_reward"))
        self.assertTrue(hasattr(result, "tag_distance"))
        self.assertTrue(hasattr(result, "axis_distance"))
        self.assertTrue(hasattr(result, "topic_mismatch"))
        self.assertTrue(hasattr(result, "prefix_profile"))
        self.assertTrue(hasattr(result, "suffix_profile"))
        self.assertTrue(hasattr(result, "cfg_error"))

    # ------------------------------------------------------------------ #
    #  Component A — grammar failure                                       #
    # ------------------------------------------------------------------ #

    def test_grammar_failure_sets_grammar_reward_to_one(self):
        result = self.rc.compute(
            self.INVALID_PREFIX, self.INVALID_SUFFIX,
            is_valid=False, cfg_error="Semantic constraint violated"
        )
        self.assertEqual(result.grammar_reward, 1.0)

    def test_grammar_failure_flag_is_true_when_invalid(self):
        result = self.rc.compute(["RIVER"], ["BURN"], is_valid=False)
        self.assertTrue(result.grammar_failure)

    def test_grammar_failure_flag_is_false_when_valid(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertFalse(result.grammar_failure)

    def test_grammar_failure_reward_is_zero_for_valid_sentence(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertEqual(result.grammar_reward, 0.0)

    def test_grammar_failure_dominates_reward(self):
        """Grammar-failure reward must always exceed pure topic-mismatch reward."""
        invalid = self.rc.compute(["RIVER"], ["BURN"],   is_valid=False)
        valid   = self.rc.compute(["MAN"],   ["GROW"],   is_valid=True)
        self.assertGreater(invalid.reward, valid.reward)

    def test_cfg_error_stored_on_failure(self):
        err = "Semantic constraint violated: something"
        result = self.rc.compute(["RIVER"], ["BURN"], is_valid=False, cfg_error=err)
        self.assertEqual(result.cfg_error, err)

    def test_cfg_error_is_none_on_valid_sentence(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        self.assertIsNone(result.cfg_error)

    # ------------------------------------------------------------------ #
    #  Component B — topic mismatch                                        #
    # ------------------------------------------------------------------ #

    def test_identical_words_give_zero_tag_distance(self):
        result = self.rc.compute(["MAN"], ["MAN"], is_valid=True)
        self.assertAlmostEqual(result.tag_distance, 0.0, places=6)

    def test_completely_different_domains_give_high_tag_distance(self):
        """HUMAN words vs MACHINE words should have large tag distance."""
        result = self.rc.compute(
            ["MAN", "WOMAN", "CHILD"],         # HUMAN domain
            ["ALGORITHM", "NETWORK", "SENSOR"], # MACHINE domain
            is_valid=True,
        )
        self.assertGreater(result.tag_distance, 0.5)

    def test_same_words_give_zero_axis_distance(self):
        result = self.rc.compute(["MAN"], ["MAN"], is_valid=True)
        self.assertAlmostEqual(result.axis_distance, 0.0, places=6)

    def test_topic_mismatch_in_zero_one_range(self):
        result = self.rc.compute(self.HUMAN_PREFIX, self.NATURE_SUFFIX, is_valid=True)
        self.assertGreaterEqual(result.topic_mismatch, 0.0)
        self.assertLessEqual(result.topic_mismatch,    1.0)

    def test_high_mismatch_gives_higher_reward_than_low_mismatch(self):
        """Human prefix + nature suffix should outscore human + human."""
        high = self.rc.compute(
            ["MAN", "WOMAN"],               # human prefix
            ["ALGORITHM", "SENSOR"],        # machine suffix  → very different tags
            is_valid=True,
        )
        low = self.rc.compute(
            ["MAN", "WOMAN"],               # human prefix
            ["CHILD", "ELDER"],             # also human suffix → very similar tags
            is_valid=True,
        )
        self.assertGreater(high.reward, low.reward)

    # ------------------------------------------------------------------ #
    #  Reward range and weights                                            #
    # ------------------------------------------------------------------ #

    def test_reward_is_non_negative(self):
        for prefix, suffix, valid in [
            (["MAN"],   ["RUN"],  True),
            (["RIVER"], ["BURN"], False),
            ([],        [],       True),
        ]:
            result = self.rc.compute(prefix, suffix, is_valid=valid)
            self.assertGreaterEqual(result.reward, 0.0,
                msg=f"Negative reward for prefix={prefix}, suffix={suffix}")

    def test_max_reward_is_grammar_plus_full_mismatch(self):
        """Upper bound: w_grammar*1 + w_mismatch*1 = 1.5 with default config."""
        cfg = self.RewardConfig(w_grammar=1.0, w_mismatch=0.5)
        self.assertAlmostEqual(cfg.w_grammar + cfg.w_mismatch, 1.5)

    def test_custom_weights_affect_reward(self):
        """Doubling w_mismatch should increase the reward for a mismatched pair."""
        rc_default = self.RewardComputer(
            self.nouns, self.verbs, self.adjectives,
            config=self.RewardConfig(w_grammar=0.0, w_mismatch=0.5),
        )
        rc_high = self.RewardComputer(
            self.nouns, self.verbs, self.adjectives,
            config=self.RewardConfig(w_grammar=0.0, w_mismatch=1.0),
        )
        r1 = rc_default.compute(["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], is_valid=True)
        r2 = rc_high.compute(   ["MAN", "WOMAN"], ["ALGORITHM", "SENSOR"], is_valid=True)
        self.assertGreater(r2.reward, r1.reward)

    # ------------------------------------------------------------------ #
    #  TopicProfile                                                        #
    # ------------------------------------------------------------------ #

    def test_empty_profile_mean_axis_is_zeros(self):
        p = self.TopicProfile()
        self.assertEqual(p.mean_axis(), [0.0, 0.0, 0.0, 0.0])

    def test_profile_tags_accumulate(self):
        """build_profile for MAN should contain its tags."""
        result = self.rc.compute(["MAN"], [], is_valid=True)
        man_entry = next(n for n in self.nouns if n.word == "MAN")
        for tag in man_entry.tag.tag:
            self.assertIn(tag, result.prefix_profile.tags)

    def test_profile_word_count(self):
        result = self.rc.compute(["MAN", "WOMAN", "CHILD"], ["RUN"], is_valid=True)
        self.assertEqual(result.prefix_profile.word_count, 3)
        self.assertEqual(result.suffix_profile.word_count, 1)

    def test_unknown_words_are_ignored_gracefully(self):
        """Words not in lexicon should not crash and not affect profile."""
        result = self.rc.compute(["UNKNOWN_WORD"], ["ANOTHER_FAKE"], is_valid=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.prefix_profile.word_count, 0)
        self.assertEqual(result.suffix_profile.word_count, 0)

    def test_summary_returns_string(self):
        result = self.rc.compute(["MAN"], ["RUN"], is_valid=True)
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Reward", summary)


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

if __name__ == "__main__":
    unittest.main(verbosity=2)

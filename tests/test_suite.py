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
#  Entry point                                                         #
# ================================================================== #

if __name__ == "__main__":
    unittest.main(verbosity=2)

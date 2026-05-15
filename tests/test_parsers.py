import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path so `src` package is importable when tests are
# executed from other working directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language import load_vocabulary, load_transitions


class ParsersTest(unittest.TestCase):

    def test_load_vocabulary_non_empty_and_contains_known_word(self):
        entries = load_vocabulary()
        self.assertTrue(len(entries) > 0, "Vocabulary should not be empty")

        # Check that a well-known adjective 'HAPPY' exists in ADJECTIVE -> EMOTION
        found = [e for e in entries if e.word == "HAPPY" and e.part_of_speech == "ADJECTIVE"]
        self.assertTrue(found, "Expected to find ADJECTIVE 'HAPPY' in the vocabulary")

    def test_load_transitions_pos_contains_start_to_verb(self):
        pos_entries, cat1_entries, cat2_entries = load_transitions()
        self.assertTrue(len(pos_entries) > 0, "POS transitions should not be empty")

        # Assert START -> VERB illegal transition exists (from transition.json)
        exists = any(p.part_of_speech_A == "START" and p.part_of_speech_B == "VERB" for p in pos_entries)
        self.assertTrue(exists, "Expected START -> VERB in pos_level transitions")

    def test_category1_contains_alive_to_device(self):
        _, cat1_entries, _ = load_transitions()
        exists = any(c.category_A == "ALIVE" and c.category_B == "DEVICE" for c in cat1_entries)
        self.assertTrue(exists, "Expected ALIVE -> DEVICE in noun category1 transitions")

    def test_category2_contains_subcategory_transitions(self):
        _, _, cat2_entries = load_transitions()
        self.assertTrue(len(cat2_entries) > 0, "Category2 transitions should not be empty")
        # Test for HUMAN -> ANIMAL (secondary level noun subcategory transition)
        exists = any(c.category_A == "HUMAN" and c.category_B == "ANIMAL" for c in cat2_entries)
        self.assertTrue(exists, "Expected HUMAN -> ANIMAL in noun category2 transitions")


if __name__ == "__main__":
    unittest.main()


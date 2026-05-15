from pathlib import Path
from typing import List, Tuple

from .models import (
	WordEntry,
	PartOfSpeechEntry,
	Category1Entry,
	Category2Entry,
)
from .parsers import WordParser, TransitionParser
from .reader import JsonReader


def load_vocabulary(path: str | Path = None) -> List[WordEntry]:
	if path is None:
		project_root = Path(__file__).resolve().parents[2]
		path = project_root / "data" / "raw" / "words.json"

	data = JsonReader.read(path)
	return WordParser.parse(data)


def load_transitions(path: str | Path = None) -> Tuple[list[PartOfSpeechEntry], list[Category1Entry], list[Category2Entry]]:
	if path is None:
		project_root = Path(__file__).resolve().parents[2]
		path = project_root / "data" / "raw" / "transition.json"

	data = JsonReader.read(path)
	return TransitionParser.parse_all(data)


def print_all_entries(words_path: str | Path = None, transitions_path: str | Path = None) -> None:
	"""Load and print word entries and transition entries in this order:
	1) WordEntry list
	2) PartOfSpeechEntry list
	3) Category1Entry list (primary category transitions like OBJECT->DEVICE)
	4) Category2Entry list (secondary category transitions like URBAN->CONCEPT)
	5) Category2Entry list (cross-category transitions like NOUN->ADJ)

	Useful for quick debugging and inspection from a REPL or a short script.
	"""

	words = load_vocabulary(words_path)
	pos_entries, cat1_entries, cat2_entries = load_transitions(transitions_path)

	print("--- WORD ENTRIES (count={}) ---".format(len(words)))
	for w in words:
		print(w)

	print("\n--- POS LEVEL ILLEGAL TRANSITIONS (count={}) ---".format(len(pos_entries)))
	for p in pos_entries:
		print(p)

	print("\n--- CATEGORY1 ILLEGAL TRANSITIONS (count={}) ---".format(len(cat1_entries)))
	for c in cat1_entries:
		print(c)

	print("\n--- CATEGORY2 ILLEGAL TRANSITIONS (count={}) ---".format(len(cat2_entries)))
	for c in cat2_entries:
		print(c)




__all__ = [
	"WordEntry",
	"PartOfSpeechEntry",
	"Category1Entry",
	"Category2Entry",
	"WordParser",
	"TransitionParser",
	"JsonReader",
	"load_vocabulary",
	"load_transitions",
]


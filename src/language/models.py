from dataclasses import dataclass

@dataclass
class WordEntry:
    part_of_speech: str
    category_1: str
    category_2: str
    word: str

@dataclass
class PartOfSpeechEntry:
    part_of_speech_A: str
    part_of_speech_B: str

@dataclass
class Category1Entry:
    category_A: str
    category_B: str

@dataclass
class Category2Entry:
    category_A: str
    category_B: str

@dataclass
class WordEntry:
    word_A: str
    word_B: str





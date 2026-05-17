from src.language.entities.cfg import CFG
from src.language.parsers import LexiconParser


class TextGenerator:
    def __init__(self):
        self.nouns, self.verbs, self.adjectives = LexiconParser.parse(
            "/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/words.json")
        self.cfg = CFG.from_json_file(
            "/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/transition.json")


    def generate_random_text(self, num_sentences: int) -> str:
        sentence_skeleton = self.cfg.generate()

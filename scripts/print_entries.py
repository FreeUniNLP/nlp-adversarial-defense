import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.entities.cfg import CFG
from src.language.lexicon_analyzer import LexiconAnalyzer
from src.language.parsers import LexiconParser

WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"


def main():
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)

    i = 0
    for noun in nouns:
        print(noun)
        i += 1
    print(i)
    i = 0

    for verb in verbs:
        print(verb)
        i += 1
    print(i)
    i = 0

    for adjective in adjectives:
        print(adjective)
        i += 1
    print(i)

    cfg = CFG.from_json(
        file_path=TRANSITION_PATH,
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives
    )
    print(cfg)

    # Retry on semantic dead-ends (some noun/verb combos have no valid match)
    for _ in range(20):
        try:
            skeleton = cfg.generate_skeleton()
            sentence = cfg.build_sentence_from_skeleton(skeleton)
            print(sentence)
            break
        except ValueError:
            continue

    LexiconAnalyzer.print_summary(nouns, verbs, adjectives)


if __name__ == "__main__":
    main()


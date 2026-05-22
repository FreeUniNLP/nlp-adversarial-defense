from src.language.entities.cfg import CFG
from src.language.lexicon_analyzer import LexiconAnalyzer
from src.language.parsers import LexiconParser


def main(text_generator=None):
    nouns, verbs, adjectives = LexiconParser.parse("/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/words.json")

    i = 0
    for noun in nouns:
        print(noun)
        i = i + 1

    print(i)
    i = 0

    for verb in verbs:
        print(verb)
        i = i + 1

    print(i)
    i = 0

    for adjective in adjectives:
        print(adjective)
        i = i + 1

    print(i)


    cfg = CFG.from_json_file("/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/transition.json")
    print(cfg)

    text_generator = cfg.generate()
    print(text_generator)

    # Delegate analysis explicitly to the Analyzer component
    LexiconAnalyzer.print_summary(nouns, verbs, adjectives)


if __name__ == "__main__":
    main()


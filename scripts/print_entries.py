from src.language.parsers import LexiconParser


def main():
    nouns, verbs, adjectives = LexiconParser.parse("/home/poz3/Documents/Projects/NLP/nlp-adversarial-defense/data/raw/word_centered_language/words.json")


    for noun in nouns:
        print(noun)

    for verb in verbs:
        print(verb)

    for adjective in adjectives:
        print(adjective)

if __name__ == "__main__":
    main()


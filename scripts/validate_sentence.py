import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.entities.cfg import CFG
from src.language.parsers import LexiconParser
from src.language.entities.cfg_validator import CFGValidator

WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "words.json"
TRANSITION_PATH = PROJECT_ROOT / "data" / "raw" / "word_centered_language" / "transition.json"


def build_validator() -> CFGValidator:
    nouns, verbs, adjectives = LexiconParser.parse(WORDS_PATH)
    cfg = CFG.from_json(
        file_path=TRANSITION_PATH,
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives,
    )
    return CFGValidator.from_cfg(cfg)


def validate_file(validator: CFGValidator, path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    valid = invalid = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        result = validator.validate(line)
        if result.is_valid:
            valid += 1
        else:
            invalid += 1
            print(f"  INVALID: {line!r}  ->  {result.error}")
    total = valid + invalid
    print(f"\nResults: {valid}/{total} valid ({invalid} invalid)")


def main():
    validator = build_validator()

    # Interactive mode: read sentences from stdin / args
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        path = Path(arg)
        if path.is_file():
            print(f"Validating file: {path}\n")
            validate_file(validator, path)
            return
        # Treat argument as a single sentence
        sentence = " ".join(sys.argv[1:])
        result = validator.validate(sentence)
        print(result)
        return

    # No args — interactive REPL
    print("Sentence Validator (type 'quit' to exit)")
    print("Loaded lexicon from:", WORDS_PATH.relative_to(PROJECT_ROOT))
    print("-" * 50)
    while True:
        try:
            sentence = input("sentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if sentence.lower() in ("quit", "exit", "q"):
            break
        if not sentence:
            continue
        result = validator.validate(sentence)
        print(f"  -> {result}\n")


if __name__ == "__main__":
    main()

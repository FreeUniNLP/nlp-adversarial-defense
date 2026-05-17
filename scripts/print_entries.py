import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.entities.cfg import CFG
from src.language.lexicon_analyzer import LexiconAnalyzer
from src.language.parsers import LexiconParser


def generate_dataset(cfg: CFG, target_count: int) -> list[str]:
    """Generates a list of valid sentences up to the target_count, handling dead-ends gracefully."""
    sentences = []
    attempts = 0
    max_attempts = target_count * 10  # Adaptive cap to prevent infinite loops if constraints are too tight

    while len(sentences) < target_count and attempts < max_attempts:
        attempts += 1
        try:
            # Step A: Layout the structural array
            skeleton = cfg.generate_skeleton(max_depth=9)

            # Step B: Bind text elements to constraints
            real_sentence = cfg.build_sentence_from_skeleton(skeleton)
            sentences.append(real_sentence)

        except ValueError:
            # Skip semantic dead-ends silently during bulk generation
            continue

    print(f"-> Target: {target_count} | Generated: {len(sentences)} (Took {attempts} structure layout attempts)")
    return sentences


def main():
    # 1. Parse your dataset strings into your structured entry objects
    nouns, verbs, adjectives = LexiconParser.parse(
        "/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/words.json"
    )

    print(f"Loaded Lexicon Component Counts -> Nouns: {len(nouns)}, Verbs: {len(verbs)}, Adjectives: {len(adjectives)}")

    # 2. Instantiate your CFG while injecting the parsed lexicon pools
    cfg = CFG.from_json_to_dataclass(
        file_path="/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/data/raw/word_centered_language/transition.json",
        nouns=nouns,
        verbs=verbs,
        adjectives=adjectives
    )

    # Define your required dataset cutoffs for model training evaluation
    dataset_sizes = [100, 500, 1000, 5000, 10000]

    # Establish and verify output path resolution based on your workspace tree
    output_dir = PROJECT_ROOT / "data" / "raw" / "generated_texts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Beginning Bulk Generation Pipeline ---")

    # Generate the maximum required corpus first to maintain statistical consistency across splits
    max_size = max(dataset_sizes)
    all_generated_sentences = generate_dataset(cfg, max_size)

    # 3. Slice data streams and write variants out to disk
    print("\n--- Writing Data Splits to Disk ---")
    for size in dataset_sizes:
        # Slice out only what is needed for this specific split file
        split_sentences = all_generated_sentences[:size]

        file_name = f"generated_corpus_{size}.txt"
        file_path = output_dir / file_name

        # Write one raw sentence per text file string newline separation
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(split_sentences) + "\n")

        print(f"Saved: {file_path.relative_to(PROJECT_ROOT)}")

    print("\n--- Vocabulary Distribution Summary ---")
    LexiconAnalyzer.print_summary(nouns, verbs, adjectives)


if __name__ == "__main__":
    main()
# AGENTS.md - AI Coding Guidelines for nlp-adversarial-defense

## Project Overview
Research project exploring adversarial attacks and robustness in NLP models using a custom synthetic mini-language, reinforcement learning, and attacker-defender architectures. The core data defines a constrained language with POS (parts-of-speech) transitions and category-level rules.

## Key Data Structures
- **Language Grammar**: `data/raw/transition.json` defines illegal transitions between POS tags (START, NOUN, VERB, ADJ, END) and subcategory constraints. Use this to validate sequence legality - any sequence containing an illegal transition is invalid.
- **Vocabulary**: `data/raw/words.json` categorizes words by POS and subcategories (e.g., NOUN: ALIVE/OBJECT/DEVICE/PLACE/ABSTRACT; VERB: PHYSICAL_ACTION/MENTAL_ACTION/SOCIAL_ACTION/SYSTEM_ACTION; ADJECTIVE: SIZE/QUALITY/EMOTION/STATE).
- **Sequence Format**: Valid sequences start with `<START>`, end with `<END>`, and must follow POS transition rules plus subcategory constraints within the same POS type.

## Architecture Components
- **`src/attacker/`**: Implement adversarial attack strategies to generate invalid sequences that appear plausible.
- **`src/defender/`**: Build defense mechanisms to detect and reject adversarial inputs.
- **`src/rl/`**: Use reinforcement learning to train agents that learn optimal attack/defense policies.
- **`src/evaluation/`**: Develop metrics and benchmarks to measure attack success rates and defense robustness.
- **`src/language/`**: Core language processing utilities for sequence generation, validation, and manipulation.

## Development Patterns
- **Sequence Validation**: Always check against `transition.json` rules - illegal POS transitions (e.g., START->VERB) and subcategory mismatches (e.g., NOUN ALIVE->DEVICE) make sequences invalid.
- **Category Constraints**: Within POS types, respect subcategory transition rules (e.g., NOUN->NOUN transitions must follow noun_category_illegal_transitions).
- **Data Loading**: Load vocabulary from `words.json` as nested dict: POS -> subcategory -> list of words.
- **Adversarial Examples**: Generate sequences that violate rules subtly, e.g., swap words between incompatible categories while maintaining POS flow.

## Workflows
- **Environment**: Use Python virtual environments; no specific requirements file yet - install common NLP libs (numpy, torch) as needed.
- **Data Processing**: Raw data in `data/raw/` should be processed into usable formats in `data/processed/` for training/evaluation.
- **Experiments**: Store experimental results and configurations in `experiments/` directory.
- **Notebooks**: Use `notebooks/` for exploratory analysis and prototyping before moving code to `src/`.

## Key Files
- `data/raw/transition.json`: Core grammar rules for sequence validation
- `data/raw/words.json`: Complete vocabulary with category structure
- `README.md`: High-level project description</content>
<parameter name="filePath">/home/konstantine/Documents/work/nlp/nlp-adversarial-defense/AGENTS.md
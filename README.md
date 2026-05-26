# nlp-adversarial-defense

Research project exploring adversarial attacks and robustness in NLP models using a custom mini-language, reinforcement learning, and attacker-defender architectures.

---

## Overview

The project has three main layers:

1. **Mini-language** — a controlled synthetic language with a fixed vocabulary (69 nouns, 32 verbs, 41 adjectives) and a Context-Free Grammar (CFG). Every word has semantic metadata: 4-dimensional axis values and tags that constrain which words can appear together.
2. **Sentence validator** — a CFG-based checker that verifies whether any sentence is structurally and semantically valid.
3. **Language model** — a small GPT-style decoder Transformer trained on the generated corpus.

The end goal is to train an adversarial attacker (via reinforcement learning) that tries to fool the language model into accepting invalid sentences.

---

## Project structure

```
nlp-adversarial-defense/
├── data/
│   ├── raw/
│   │   ├── word_centered_language/
│   │   │   ├── words.json          # Lexicon — all words, tags, axis values
│   │   │   └── transition.json     # CFG grammar rules
│   │   └── generated_texts/        # Pre-generated sentence corpora
│   │       ├── generated_corpus_100.txt
│   │       ├── generated_corpus_500.txt
│   │       ├── generated_corpus_1000.txt
│   │       ├── generated_corpus_5000.txt
│   │       └── generated_corpus_10000.txt
│   └── models/                     # Saved model checkpoints (created on first train)
├── src/
│   ├── language/
│   │   ├── entities/
│   │   │   ├── cfg.py              # CFG generation engine
│   │   │   └── word_entity.py      # Dataclasses: NounEntry, VerbEntry, AdjectiveEntry, ...
│   │   ├── parsers.py              # Parses words.json into dataclasses
│   │   ├── lexicon_analyzer.py     # Tag statistics and summaries
│   │   └── reader.py               # JSON file loader
│   ├── evaluation/
│   │   └── sentence_validator.py   # CFGValidator — structural + semantic validation
│   └── model/
│       ├── tokenizer.py            # Word-level tokenizer
│       └── transformer.py          # MiniGPT decoder model
└── scripts/
    ├── print_entries.py            # Inspect the lexicon
    ├── text_generator.py           # Generate sentence corpora
    ├── validate_sentence.py        # Validate sentences against the grammar
    ├── train_model.py              # Train the language model
    └── evaluate_model.py           # Validate model-generated sentences
```

---

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

All scripts are run from the **project root** directory.

---

## Scripts

### `scripts/print_entries.py` — Inspect the lexicon

Prints every noun, verb, and adjective with their tags and axis values, then generates one random valid sentence.

```bash
python scripts/print_entries.py
```

**Example output:**
```
NounEntry(word='MAN', tag=TagEntry(tag=['ALIVE', 'HUMAN']), axis=AxisEntry(agency=4, physicality=4, social=3, system=0))
...
69
...
FAST WOLF BURN DIRTY MAP
```

Use this to understand the semantic structure of the vocabulary before working with the grammar.

---

### `scripts/text_generator.py` — Generate sentence corpora

Generates valid sentences using the CFG and writes them to `data/raw/generated_texts/`. Creates corpus files at sizes 100, 500, 1000, 5000, and 10000.

```bash
python scripts/text_generator.py
```

**Example output:**
```
Loaded Lexicon Component Counts -> Nouns: 69, Verbs: 32, Adjectives: 41

--- Beginning Bulk Generation Pipeline ---
-> Target: 10000 | Generated: 10000 (Took 19992 structure layout attempts)

--- Writing Data Splits to Disk ---
Saved: data\raw\generated_texts\generated_corpus_100.txt
...
```

The attempt count shows how constrained the grammar is — roughly 50% of random skeletons fail semantic constraints and are retried silently.

---

### `scripts/validate_sentence.py` — Validate sentences

Checks whether a sentence conforms to the CFG grammar rules **and** the semantic constraints (tag overlap + axis bounds). Runs in three modes:

**Single sentence:**
```bash
python scripts/validate_sentence.py "MAN RUN"
# ValidationResult(valid=True)

python scripts/validate_sentence.py "ROBOT FAIL"
# ValidationResult(valid=False, error='Semantic constraint violated: Verb 'FAIL' is incompatible with subject 'ROBOT'...')

python scripts/validate_sentence.py "RUN MAN"
# ValidationResult(valid=False, error='Skeleton invalid: Token sequence ['VERB', 'NOUN'] cannot be derived from START.')
```

**Batch file** (one sentence per line):
```bash
python scripts/validate_sentence.py data/raw/generated_texts/generated_corpus_100.txt
# Results: 100/100 valid (0 invalid)
```

**Interactive REPL** (no arguments):
```bash
python scripts/validate_sentence.py
# sentence> MAN KNOW TRUTH
#   -> ValidationResult(valid=True)
# sentence> STONE KNOW TRUTH
#   -> ValidationResult(valid=False, error='Semantic constraint violated: ...')
# sentence> quit
```

**Validation checks (in order):**
1. Every word must exist in the lexicon
2. The POS sequence (NOUN/VERB/ADJ skeleton) must be derivable from the CFG grammar
3. Semantic constraints must be satisfied: tag overlap between subject/object and verb arguments, cumulative axis values within bounds

---

### `scripts/train_model.py` — Train the language model

Trains a small GPT-style decoder Transformer on one of the generated corpora. Saves the best checkpoint to `data/models/`.

```bash
# Default: 10k corpus, 30 epochs
python scripts/train_model.py

# Specific corpus size and epoch count
python scripts/train_model.py --corpus 10000 --epochs 30

# Quick test run
python scripts/train_model.py --corpus 100 --epochs 5
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `10000` | Corpus size: `100`, `500`, `1000`, `5000`, `10000` |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--embed-dim` | `64` | Transformer embedding dimension |
| `--num-heads` | `4` | Number of attention heads |
| `--num-layers` | `4` | Number of transformer blocks |
| `--dropout` | `0.1` | Dropout rate |
| `--context-len` | `32` | Max sequence length |
| `--temperature` | `0.8` | Sampling temperature for end-of-training examples |

**Example output:**
```
Device: cpu
Vocab size: 146
Dataset: 10000 sentences  |  Batches/epoch: 157
Model parameters: 208,960
Epoch   1/30  loss=4.7123  time=12.3s
...
Epoch  30/30  loss=1.8432  time=11.9s

Best loss: 1.7201  checkpoint -> data\models\minigpt_corpus10000.pt

--- Sample generated sentences ---
  MAN KNOW TRUTH
  FAST WOLF BURN DIRTY MAP
  ...
```

The checkpoint is saved whenever the epoch loss improves. Training on 10k sentences for 30 epochs takes roughly 5–10 minutes on CPU.

---

### `scripts/evaluate_model.py` — Evaluate model output

Loads a trained checkpoint, generates sentences, and validates each one with the CFGValidator. Reports the percentage of valid sentences.

```bash
# Default: 10k corpus, 200 samples
python scripts/evaluate_model.py

# Specific corpus / sample count
python scripts/evaluate_model.py --corpus 10000 --samples 500

# Lower temperature = less random = more structured output
python scripts/evaluate_model.py --corpus 10000 --samples 200 --temperature 0.5
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `10000` | Which checkpoint to load (must match training corpus) |
| `--samples` | `200` | Number of sentences to generate |
| `--temperature` | `0.8` | Sampling temperature (lower = more greedy) |
| `--show` | `10` | Number of example sentences to print per category |

**Example output:**
```
Loaded checkpoint: minigpt_corpus10000.pt  (epoch 28, loss 1.7201)
Generating 200 sentences (temperature=0.8)...

============================================================
  Valid:   142/200  (71.0%)
  Invalid: 58/200   (29.0%)
============================================================

--- Valid samples (first 10) ---
  MAN KNOW TRUTH
  FAST WOLF BURN DIRTY MAP
  ...

--- Invalid samples (first 10) ---
  STONE KNOW TRUTH
    reason: Semantic constraint violated: Verb 'KNOW' is incompatible with subject 'STONE'...
  ...
```

**Expected validity by training stage:**

| Loss | Valid % | What's happening |
|------|---------|------------------|
| ~4.7 | 0% | Random token output |
| ~3.0 | 5–15% | Learning rough word order |
| ~2.0 | 30–50% | Picking up grammar structure |
| ~1.5 | 60–80% | Mostly grammatical sentences |

---

## The grammar

The CFG in `data/raw/word_centered_language/transition.json` defines:

```
START        → SUBJECT_TERM
SUBJECT_TERM → SUBJECT VERB_TERM
SUBJECT      → NOUN | ADJ NOUN
VERB_TERM    → VERB | VERB OBJECT | VERB OBJECT VERB_TERM   (recursive)
OBJECT       → NOUN | ADJ NOUN | ADJ ADJ NOUN
```

This allows sentences like:
- `MAN RUN` — intransitive (NOUN VERB)
- `WOLF BURN SEED` — transitive (NOUN VERB NOUN)
- `FAST WOLF BURN DIRTY MAP` — with adjectives
- `MAN KNOW TRUTH WANT CLEAN RULE` — chained verbs

---

## The axis system

Each word has a 4-dimensional axis `(agency, physicality, social, system)` scored 0–5. Verb argument constraints specify min/max bounds on these axes for valid subjects and objects. Adjectives shift the axis values of the nouns they modify.

For example, a verb like `KNOW` might require its subject to have `agency >= 2` — so `STONE` (agency=0) cannot be the subject of `KNOW`, but `MAN` (agency=4) can.

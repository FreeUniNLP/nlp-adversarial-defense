# NLP Adversarial Defense

A research project that studies how a language model defends against an adversarial attacker in a controlled synthetic language. The attacker generates grammatically valid sentence prefixes designed to push the defender off-topic or into producing invalid grammar. The defender (MiniGPT) completes those prefixes. A structured reward signal tells us how successful the attack was.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Synthetic Language](#2-synthetic-language)
3. [Project Structure](#3-project-structure)
4. [Components](#4-components)
   - [Lexicon and Parser](#41-lexicon-and-parser)
   - [Context-Free Grammar](#42-context-free-grammar)
   - [CFG Validator](#43-cfg-validator)
   - [Word Tokenizer](#44-word-tokenizer)
   - [MiniGPT Defender](#45-minigpt-defender)
   - [CFG State Tracker](#46-cfg-state-tracker)
   - [Attacker Transformer](#47-attacker-transformer)
   - [Reward Simple](#48-reward-simple)
   - [Reward Function Structured](#49-reward-function-structured)
   - [Attack Pipeline](#410-attack-pipeline)
5. [Scripts](#5-scripts)
6. [Tests](#6-tests)
7. [Setup and Installation](#7-setup-and-installation)
8. [How to Run](#8-how-to-run)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Architecture Diagram](#10-architecture-diagram)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Project Overview

This project implements a full **adversarial attack and defense loop** over a synthetic mini-language:

```
Attacker  -->  CFG-valid prefix  -->  MiniGPT (defender)  -->  completed sentence
                                                                      |
                                                              CFGValidator checks
                                                                      |
                                                            RewardFunction scores
                                                                      |
                                                            reward signal (RL-ready)
```

**Goal of the attacker:** generate a prefix that causes the defender to complete the sentence in a way that is either grammatically invalid or semantically off-topic relative to the prefix.

**Goal of the defender:** complete any given prefix into a fluent, grammatically valid sentence according to the CFG rules.

**Why a synthetic language?** Using a fully controlled grammar lets us measure validity and semantic distance precisely — something impossible with real natural language.

---

## 2. Synthetic Language

The language is built from a hand-crafted lexicon and a context-free grammar.

### Lexicon

| Type | Count | Description |
|---|---|---|
| Nouns | 69 | Humans, animals, objects, places, abstract concepts, machines |
| Verbs | 32 | Actions — transitive (require object) and intransitive (no object) |
| Adjectives | 41 | Modifiers that change noun compatibility |

Every word carries two semantic properties:

**Tags** — semantic category labels, for example:
- Nouns: `ALIVE`, `HUMAN`, `ANIMAL`, `MACHINE`, `ABSTRACT`, `PLACE`
- Verbs: `PHYSICAL_ACTION`, `LOCOMOTION`, `DESTRUCTION`, `SYSTEM_ACTION`
- Adjectives: `SIZE`, `QUALITY`, `STATE`, `CONDITION`

**Axis values** — a 4-dimensional real-valued vector:

| Dimension | Meaning | Range |
|---|---|---|
| `agency` | capacity for autonomous action | 0 to 5 |
| `physicality` | physical concreteness | 0 to 5 |
| `social` | social or relational quality | 0 to 5 |
| `system` | machine or computational quality | 0 to 5 |

### Grammar (CFG)

```
START        -> SUBJECT_TERM
SUBJECT_TERM -> SUBJECT VERB_TERM
SUBJECT      -> NOUN | ADJ NOUN
VERB_TERM    -> VERB | VERB OBJECT | VERB OBJECT VERB_TERM   (recursive)
OBJECT       -> NOUN | ADJ NOUN | ADJ ADJ NOUN
```

**Semantic constraints** are layered on top of the skeleton:
- Every verb requires its subject to have compatible tags and axis values
- Transitive verbs require the object to satisfy additional tag and axis constraints
- Adjectives narrow down which nouns they can modify via tag overlap and axis bounds

Example valid sentences:
```
MAN RUN
FREE WOLF FALL
DRONE BREAK CLOCK
STRONG MAN CARRY SMALL BOOK FORGET LIE
```

---

## 3. Project Structure

```
nlp-adversarial-defense/
|
|-- data/
|   |-- raw/
|   |   |-- word_centered_language/
|   |   |   |-- words.json               full lexicon (nouns, verbs, adjectives)
|   |   |   `-- transition.json          CFG rules and semantic constraints
|   |   `-- generated_texts/
|   |       |-- generated_corpus_100.txt
|   |       |-- generated_corpus_1000.txt
|   |       |-- generated_corpus_5000.txt
|   |       `-- generated_corpus_10000.txt    training corpus (10k sentences)
|   `-- models/
|       `-- minigpt_corpus10000.pt            trained MiniGPT checkpoint
|
|-- src/
|   |-- language/
|   |   |-- parsers.py                   LexiconParser
|   |   |-- entities/
|   |   |   |-- word_entity.py           NounEntry, VerbEntry, AdjectiveEntry dataclasses
|   |   |   |-- cfg.py                   CFG -- skeleton generation and sentence building
|   |   |   |-- cfg_base.py              shared semantic constraint logic
|   |   |   `-- cfg_validator.py         CFGValidator -- full sentence validation
|   |   `-- reader.py
|   |
|   |-- model/
|   |   |-- tokenizer.py                 WordTokenizer -- word-level BOS/EOS/PAD/UNK
|   |   `-- transformer.py               MiniGPT -- decoder-only causal Transformer
|   |
|   `-- attacker/
|       |-- cfg_state_tracker.py         CFGStateTracker -- grammar state machine
|       |-- attacker.py                  AttackerTransformer -- CFG-masked generation
|       |-- reward.py                    RewardComputer -- simple reward module
|       `-- reward_function.py           RewardFunction -- structured reward module
|
|-- scripts/
|   |-- train/
|   |   `-- train_model.py               MiniGPT training loop with MLflow logging
|   |-- attack_and_complete.py           full AttackPipeline
|   |-- demo_reward_function.py          step-by-step reward function demo
|   |-- complete_and_validate.py         prefix -> MiniGPT -> CFGValidator
|   |-- infer.py                         interactive MiniGPT sentence completion
|   |-- run_attacker.py                  generate and display attacker prefixes
|   |-- test_attacker.py                 manual labelled test script
|   `-- validate_sentence.py             validate a sentence against the CFG
|
|-- tests/
|   |-- conftest.py                      shared fixtures and paths
|   |-- test_lexicon_parser.py           12 tests
|   |-- test_cfg.py                       7 tests
|   |-- test_cfg_validator.py            14 tests
|   |-- test_tokenizer.py                 9 tests
|   |-- test_minigpt.py                   5 tests
|   |-- test_cfg_state_tracker.py        17 tests
|   |-- test_attacker_transformer.py      9 tests
|   |-- test_reward_computer.py          22 tests
|   `-- test_reward_function.py          54 tests
|
|-- config.py                            MLflow / DagsHub credentials
`-- README.md
```

---

## 4. Components

### 4.1 Lexicon and Parser

**File:** `src/language/parsers.py`

Parses `words.json` into typed Python dataclasses.

```python
from src.language.parsers import LexiconParser

nouns, verbs, adjectives = LexiconParser.parse("data/raw/word_centered_language/words.json")

noun = nouns[0]
print(noun.word)              # "MAN"
print(noun.tag.tag)           # ["ALIVE", "HUMAN"]
print(noun.axis.agency)       # 4
print(noun.axis.physicality)  # 4

verb = verbs[0]
print(verb.word)              # "RUN"
print(verb.tag.tag)           # ["PHYSICAL_ACTION", "LOCOMOTION"]
print(verb.verb_argument.verb_to_subject_constraint)  # tag + axis constraint object
print(verb.verb_argument.verb_to_object_constraint)   # None if intransitive
```

---

### 4.2 Context-Free Grammar

**File:** `src/language/entities/cfg.py`

Generates random grammatically and semantically valid sentences.

```python
from src.language.entities.cfg import CFG

cfg = CFG.from_json(
    "data/raw/word_centered_language/transition.json",
    nouns=nouns, verbs=verbs, adjectives=adjectives,
)

skeleton = cfg.generate_skeleton()                    # ["NOUN", "VERB", "ADJ", "NOUN"]
sentence = cfg.build_sentence_from_skeleton(skeleton) # "MAN CARRY SMALL CLOCK"
```

Semantic constraints are enforced at generation time — the CFG never produces an invalid sentence.

---

### 4.3 CFG Validator

**File:** `src/language/entities/cfg_validator.py`

Validates any sentence string against all grammar and semantic rules.

```python
from src.language.entities.cfg_validator import CFGValidator

validator = CFGValidator.from_cfg(cfg)

result = validator.validate("FREE WOLF FALL")
print(result.is_valid)  # True

result = validator.validate("RIVER BURN")
print(result.is_valid)  # False
print(result.error)     # "Semantic constraint violated: Verb 'BURN' is incompatible with subject 'RIVER'"
```

Validation runs in three phases:

1. **Unknown word check** — all tokens must exist in the lexicon
2. **Skeleton check** — the POS sequence must match a derivable CFG pattern
3. **Semantic check** — all subject/verb/object constraints must be satisfied

---

### 4.4 Word Tokenizer

**File:** `src/model/tokenizer.py`

Word-level tokenizer built from a corpus file.

| Token | ID | Meaning |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<BOS>` | 1 | Beginning of sentence |
| `<EOS>` | 2 | End of sentence |
| `<UNK>` | 3 | Unknown word |

```python
from src.model.tokenizer import WordTokenizer

tokenizer = WordTokenizer.from_corpus("data/raw/generated_texts/generated_corpus_10000.txt")
# vocab_size = 146

ids  = tokenizer.encode("MAN RUN", add_special=True)  # [1, 5, 8, 2]
text = tokenizer.decode(ids)                           # "MAN RUN"
```

---

### 4.5 MiniGPT Defender

**File:** `src/model/transformer.py`

A decoder-only causal Transformer trained to generate valid sentences in the synthetic language.

| Parameter | Value |
|---|---|
| Architecture | Decoder-only causal Transformer |
| Layers | 4 |
| Embedding dim | 64 |
| Attention heads | 4 |
| Context length | 32 tokens |
| Vocabulary size | 146 |
| Total parameters | 210,432 |
| Training corpus | 10,000 sentences |
| Trained epochs | 32 |
| Final training loss | 2.7944 |

```python
import torch
from src.model.transformer import MiniGPT

model = MiniGPT(vocab_size=146, pad_id=0)
ckpt  = torch.load("data/models/minigpt_corpus10000.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Free generation
ids = model.generate(bos_id=1, eos_id=2, max_new_tokens=15, temperature=0.8)

# Prefix completion
x      = torch.tensor([[1, 5, 8]])  # BOS MAN RUN
logits = model(x)                   # shape (1, 3, 146)
```

---

### 4.6 CFG State Tracker

**File:** `src/attacker/cfg_state_tracker.py`

A grammar-aware finite state machine that tracks exactly which words are valid at every generation step. Used by the attacker to guarantee every generated prefix is a valid partial sentence.

**States:**

| State | Meaning |
|---|---|
| `SUBJECT_START` | Beginning — expect adjective or noun |
| `SUBJECT_AFTER_ADJ` | After subject adjective — expect noun only |
| `AFTER_SUBJECT` | Subject placed — expect verb |
| `AFTER_INTRANS_VERB` | Intransitive verb placed — can end, no further words |
| `OBJECT_START` | After transitive verb — expect adjective or noun |
| `OBJECT_AFTER_ADJ1` | After 1st object adjective — expect adjective or noun |
| `OBJECT_AFTER_ADJ2` | After 2nd object adjective — expect noun only |
| `AFTER_OBJECT` | Object placed — can add another verb or end |

```python
from src.attacker.cfg_state_tracker import CFGStateTracker

tracker = CFGStateTracker(nouns, verbs, adjectives)
tracker.reset()

words, can_end = tracker.valid_next_words()  # valid next tokens + whether EOS is allowed
ok = tracker.step("MAN")                     # True -- advances state
ok = tracker.step("CARRY")                   # True
print(tracker.sentence())                    # "MAN CARRY"
print(tracker.can_end)                       # True
```

Key constraint: after an intransitive verb the tracker returns no further valid words and `can_end=True`. Verb chaining is only allowed after a complete object has been placed.

---

### 4.7 Attacker Transformer

**File:** `src/attacker/attacker.py`

Identical architecture to MiniGPT (210,432 parameters). At each generation step, logits for all CFG-invalid tokens are masked to `-inf` before sampling — guaranteeing every generated prefix is a valid partial sentence according to the CFG rules.

```python
from src.attacker.attacker import AttackerTransformer

attacker = AttackerTransformer(vocab_size=146, pad_id=0)

# Generate a CFG-valid prefix
ids, words = attacker.generate_prefix(
    bos_id=tokenizer.bos_id,
    eos_id=tokenizer.eos_id,
    cfg_tracker=tracker,
    token_to_id=tokenizer.token_to_id,
    id_to_token=tokenizer.id_to_token,
    max_tokens=6,
    temperature=1.0,
)
# words -> e.g. ["STRONG", "MAN", "CARRY"]

# RL-ready: returns log-probabilities with gradients attached
ids, words, log_probs = attacker.generate_prefix_with_log_probs(...)

# REINFORCE policy gradient loss
loss = -(log_probs * reward).sum()
loss.backward()
```

---

### 4.8 Reward Simple

**File:** `src/attacker/reward.py`

A lightweight reward module that returns a scalar signal plus a basic breakdown.

```python
from src.attacker.reward import RewardComputer, RewardConfig

rc = RewardComputer(nouns, verbs, adjectives, config=RewardConfig())

result = rc.compute(
    prefix_words=["MAN", "CARRY"],
    suffix_words=["ALGORITHM", "SPREAD"],
    is_valid=False,
    cfg_error="Semantic constraint violated",
)

print(result.reward)          # scalar e.g. 1.22
print(result.grammar_reward)  # 1.0 (invalid) or 0.0 (valid)
print(result.tag_distance)    # Jaccard distance of combined tag sets
print(result.axis_distance)   # cosine distance of mean axis vectors
print(result.topic_mismatch)  # combined mismatch score
print(result.summary())       # human-readable breakdown
```

**Formula:** `R = 1.0 * grammar_reward + 0.5 * topic_mismatch`

---

### 4.9 Reward Function Structured

**File:** `src/attacker/reward_function.py`

The full structured reward module. Separates feature extraction, distance computation, and reward calculation into distinct classes. Tracks noun, verb, and adjective tags independently to give fine-grained visibility into where semantic drift occurred.

#### Sub-components

**`FeatureExtractor`** classifies each word by POS and extracts:
- `noun_tags` — tag set from all nouns in this segment
- `verb_tags` — tag set from all verbs
- `adjective_tags` — tag set from all adjectives
- `all_tags` — union of all three tag sets
- `mean_axis` — elementwise mean of all word axis vectors

**`DistanceCalculator`** computes pairwise distances between prefix and suffix features:
- `noun_tag_dist` — Jaccard distance of noun tag sets (0 = identical, 1 = disjoint)
- `verb_tag_dist` — Jaccard distance of verb tag sets
- `adjective_tag_dist` — Jaccard distance of adjective tag sets
- `tag_mismatch` — mean of the three POS distances
- `axis_distance` — cosine distance of mean axis vectors (0 = same direction, 1 = opposite)

**`RewardFunction`** orchestrates everything:

```python
from src.attacker.reward_function import RewardFunction, RewardWeights

rf = RewardFunction(nouns, verbs, adjectives, weights=RewardWeights())

# Split sentence into attacker prefix and defender suffix
prefix_part, suffix_part = rf.split_sentence(full_words, prefix_words)

# Compute structured reward
out = rf.compute(
    prefix_words=prefix_part,
    suffix_words=suffix_part,
    full_sentence="STRONG MAN CARRY ALGORITHM SPREAD",
    is_valid=False,
    cfg_error="Semantic constraint violated: ...",
)

print(out.reward)                        # final scalar
print(out.distances.noun_tag_dist)       # how different the nouns are
print(out.distances.verb_tag_dist)       # how different the verbs are
print(out.distances.axis_distance)       # how different the axis profiles are
print(out.prefix.noun_tags)              # {"ALIVE", "HUMAN"}
print(out.suffix.noun_tags)              # {"ABSTRACT", "SYSTEM"}
print(out.summary())                     # full human-readable breakdown
```

#### Reward formula

```
R = w_grammar * grammar_reward    (1.0 if CFG-invalid, 0.0 if valid)
  + w_tag     * tag_mismatch      (mean of 3 per-POS Jaccard distances)
  + w_axis    * axis_distance     (cosine distance of mean axis vectors)
```

#### Default weights

| Weight | Value | Role |
|---|---|---|
| `w_grammar` | 1.0 | Grammar failure dominates |
| `w_tag` | 0.30 | Tag mismatch — medium contribution |
| `w_axis` | 0.20 | Axis drift — medium contribution |
| Max reward | 1.50 | All components simultaneously = 1.0 |

#### Three reward levels

| Case | Grammar | Tag | Axis | Total |
|---|---|---|---|---|
| Grammar failure | 1.00 | ~0.20 | ~0.01 | ~1.21 |
| Topic mismatch | 0.00 | ~0.20 | ~0.08 | ~0.28 |
| Topic consistent | 0.00 | ~0.10 | ~0.00 | ~0.10 |

---

### 4.10 Attack Pipeline

**File:** `scripts/attack_and_complete.py`

Ties all components together into a single loop.

```
for each iteration:
  Step 1: Attacker generates a CFG-valid prefix   (CFGStateTracker enforces grammar)
  Step 2: MiniGPT completes the prefix             (defender)
  Step 3: CFGValidator checks the full sentence    (valid or invalid + error)
  Step 4: RewardFunction scores the result         (structured reward breakdown)
```

```python
from scripts.attack_and_complete import AttackPipeline

pipeline = AttackPipeline(
    max_prefix_tokens=6,
    max_completion_tokens=15,
    attacker_temperature=1.0,
    defender_temperature=0.8,
)

pipeline.run(n=100)
```

**Results on 100 iterations with untrained attacker:**
```
VALID   : 60/100  (60%)
INVALID : 40/100  (40%)
AVG REWARD : 0.64
```

---

## 5. Scripts

| Script | Purpose |
|---|---|
| `scripts/train/train_model.py` | Train MiniGPT on a corpus with MLflow logging |
| `scripts/train/train_attacker.py` | Train the attacker with REINFORCE policy gradient (frozen defender) |
| `scripts/attack_and_complete.py` | Run the full adversarial attack pipeline |
| `scripts/demo_reward_function.py` | Step-by-step walkthrough of the reward function |
| `scripts/infer.py` | Interactive MiniGPT sentence completion |
| `scripts/complete_and_validate.py` | Complete a prefix and validate the result |
| `scripts/run_attacker.py` | Generate and display attacker-generated prefixes |
| `scripts/test_attacker.py` | Manual labelled test script for attacker components |
| `scripts/validate_sentence.py` | Validate a single sentence against the CFG |

---

## 6. Tests

All tests use `unittest` and are compatible with `pytest`. Each component has its own dedicated file.

| File | Tests | What it covers |
|---|---|---|
| `test_lexicon_parser.py` | 12 | Word counts, field presence, axis range, uppercase |
| `test_cfg.py` | 7 | Skeleton generation, sentence building, POS tokens |
| `test_cfg_validator.py` | 14 | Valid sentences, invalid sentences, corpus batch, repr |
| `test_tokenizer.py` | 9 | Encode, decode, BOS/EOS, UNK, roundtrip |
| `test_minigpt.py` | 5 | Forward pass shape, generation, temperature |
| `test_cfg_state_tracker.py` | 17 | State transitions, valid words, invalid steps, reset |
| `test_attacker_transformer.py` | 9 | CFG-masked generation, log-prob gradients |
| `test_reward_computer.py` | 22 | Grammar component, mismatch component, TopicProfile |
| `test_reward_function.py` | 54 | FeatureExtractor, DistanceCalculator, split, compute, weights |
| `test_train_attacker.py` |  4 | REINFORCE loop smoke: gradient flow, optimizer steps, EMA baseline |
| **Total** | **155** | **all passing** |

Run all tests:

```powershell
python -m pytest tests/ --ignore=tests/test_suite.py -v
```

Run a single test class:

```powershell
python -m pytest tests/test_reward_function.py::TestFeatureExtractor -v
```

---

## 7. Setup and Installation

### Requirements

```
Python >= 3.10
torch
mlflow
dagshub
pytest
```

Install dependencies:

```powershell
pip install torch mlflow dagshub pytest
```

### Windows — OpenMP conflict

On Windows, PyTorch may raise:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

Fix by setting the environment variable before running any script:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

All scripts also set this automatically in code:

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

---

## 8. How to Run

### Train MiniGPT

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/train/train_model.py
```

### Train the attacker (REINFORCE)

Updates the attacker's policy to maximise reward against the *frozen* defender:

```
loss = -(reward - baseline) * sum(log_probs)
```

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 2000 episodes, local logging only
python scripts/train/train_attacker.py --episodes 2000

# Longer run with MLflow / DagsHub tracking
python scripts/train/train_attacker.py --episodes 10000 --mlflow
```

What it does each episode:
1. Attacker samples a CFG-valid prefix (gradients tracked through `log_probs`).
2. Frozen MiniGPT completes the prefix.
3. `CFGValidator` checks the full sentence; `RewardFunction` returns a scalar.
4. REINFORCE loss is computed with an EMA baseline for variance reduction.
5. Gradients flow only through the attacker; the defender is fully frozen.

Outputs:
- `data/models/attacker_best.pt`  — best rolling-avg-reward checkpoint
- `data/models/attacker_final.pt` — final-episode checkpoint
- `logs/attacker_episodes_seed{S}.csv` — per-episode prefix / completion / reward
- `logs/train_attacker_seed{S}.log` — training log
- MLflow metrics (if `--mlflow`): `avg_reward`, `grammar_fail_rate`, `baseline`, `avg_loss`

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 2000 | Number of REINFORCE updates |
| `--lr` | 3e-4 | AdamW learning rate |
| `--max-prefix` | 6 | Max attacker prefix length |
| `--atk-temp` | 1.0 | Attacker sampling temperature |
| `--def-temp` | 0.8 | Defender sampling temperature |
| `--w-grammar` / `--w-tag` / `--w-axis` | 1.0 / 0.30 / 0.20 | Reward component weights |
| `--baseline-alpha` | 0.05 | EMA smoothing rate for the reward baseline |
| `--entropy-coef` | 0.0 | Optional entropy bonus (encourage exploration) |
| `--window` | 100 | Rolling window for best-checkpoint averaging |

### Complete a sentence interactively

```powershell
python scripts/infer.py
```

### Validate a sentence

```powershell
python scripts/validate_sentence.py "FREE WOLF FALL"
```

### Generate attacker prefixes

```powershell
python scripts/run_attacker.py --n 10 --max-tokens 6 --temperature 1.0
```

### Run the full attack pipeline

```powershell
# 100 iterations, compact output
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/attack_and_complete.py --n 100

# 10 iterations with full reward breakdown per sentence
python scripts/attack_and_complete.py --n 10 --verbose
```

### Demo the reward function (step-by-step)

```powershell
python scripts/demo_reward_function.py
```

### Run all tests

```powershell
python -m pytest tests/ --ignore=tests/test_suite.py -v
```

---

## 9. Experiment Tracking

The training script logs to **DagsHub** via **MLflow**. All team members share one experiment view.

Configure credentials in `config.py`:

```python
DAGSHUB_USERNAME    = "your-username"
DAGSHUB_TOKEN       = "your-token"
DAGSHUB_REPO_NAME   = "your-repo-name"
MLFLOW_TRACKING_URI = "https://dagshub.com/your-username/your-repo-name.mlflow"
```

Each training run logs:

| Metric | Description |
|---|---|
| `train_loss` | Loss per epoch |
| `best_loss` | Best loss seen so far |
| `epoch` | Current epoch number |

Parameters logged per run: `vocab_size`, `embed_dim`, `n_heads`, `n_layers`, `context_len`, `corpus`

---

## 10. Architecture Diagram

```
+----------------------+       CFG-valid prefix        +----------------------+
|                      |  ---------------------------> |                      |
|  AttackerTransformer |                               |  MiniGPT (Defender)  |
|    210k parameters   |                               |    210k parameters   |
|                      |  <--------------------------- |                      |
+----------------------+       completed sentence       +----------------------+
         |                                                        |
         |                                                        |
         v                                                        v
+----------------------+                              +----------------------+
|   CFGStateTracker    |                              |    CFGValidator      |
|   (grammar FSM)      |                              |                      |
|                      |                              |  Phase 1: unknown    |
|   at each step:      |                              |  Phase 2: skeleton   |
|   filters valid      |                              |  Phase 3: semantics  |
|   next tokens        |                              +----------------------+
+----------------------+                                        |
                                                   valid / invalid + error msg
                                                                |
                                                                v
                                               +---------------------------------+
                                               |       RewardFunction            |
                                               |                                 |
                                               |  split_sentence                 |
                                               |    prefix | suffix              |
                                               |                                 |
                                               |  FeatureExtractor               |
                                               |    noun_tags  verb_tags         |
                                               |    adj_tags   mean_axis         |
                                               |                                 |
                                               |  DistanceCalculator             |
                                               |    noun_tag_dist                |
                                               |    verb_tag_dist                |
                                               |    adj_tag_dist                 |
                                               |    axis_distance                |
                                               |                                 |
                                               |  R = w_grammar * grammar_fail   |
                                               |    + w_tag     * tag_mismatch   |
                                               |    + w_axis    * axis_distance  |
                                               +---------------------------------+
                                                                |
                                                          scalar reward
                                                         (RL-ready signal)
                                                                |
                                                                v
                                               +---------------------------------+
                                               |     Future: Attacker Training   |
                                               |                                 |
                                               |  REINFORCE policy gradient:     |
                                               |  loss = -(log_probs * R).sum()  |
                                               |  loss.backward()                |
                                               +---------------------------------+
```

---

## 11. Key Design Decisions

**Why a synthetic language?**
Full control over grammar and semantics means validity is binary and objectively measurable. There is no ambiguity, no annotation cost, and no pre-trained resources required.

**Why CFG masking in the attacker?**
The attacker should never waste model capacity learning basic grammar rules — the state tracker enforces validity deterministically at every step. The model only learns which valid prefixes lead to high reward.

**Why per-POS tag distances?**
Aggregating all tags into a single set hides where drift occurred. Separating noun, verb, and adjective tag distances tells you exactly which grammatical slot diverged — actionable information for analysing and training the attacker.

**Why RL-ready log-probs?**
`generate_prefix_with_log_probs()` keeps gradients attached through `log_softmax` so a REINFORCE loss can be computed directly:

```python
loss = -(log_probs * reward).sum()
loss.backward()
```

The architecture is ready for adversarial reinforcement learning without any structural changes.

**Why the same architecture for attacker and defender?**
Keeping both models identical in size and structure creates a fair benchmark. Any difference in performance is due to training signal, not model capacity.

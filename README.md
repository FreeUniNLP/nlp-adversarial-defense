# NLP Adversarial Defense — Project & Report

A research project that studies how a small language model **defends** against an adversarial
**attacker** inside a fully controlled synthetic language. The attacker generates
grammatically valid sentence *prefixes* designed to push the defender into producing an
*invalid* or *off-topic* completion. The defender (MiniGPT) completes those prefixes. A
structured reward signal measures exactly how successful each attack was, and that signal
drives reinforcement learning on both sides.

Because the language is synthetic, **validity is binary and objectively measurable** — there
is no annotation cost, no ambiguity, and no dependence on pretrained resources. That is the
whole reason the project can give clean, quantitative answers about attack success and defense
robustness.

> This README doubles as the **project report**. The three report requirements are addressed
> directly:
> - **§5 — why this loss function and architecture**
> - **§6 — how training went, with graphs**
> - **§7 — evaluation tasks, why we chose them, and the results**

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [The synthetic language](#2-the-synthetic-language)
3. [System architecture](#3-system-architecture)
4. [Repository layout](#4-repository-layout)
5. [Design decisions: architecture & loss functions](#5-design-decisions-architecture--loss-functions)  ← *report pt.1*
6. [Training: how it went, with graphs](#6-training-how-it-went-with-graphs)  ← *report pt.2*
7. [Evaluation: tasks, rationale & results](#7-evaluation-tasks-rationale--results)  ← *report pt.3*
8. [Problems we hit and how we solved them](#8-problems-we-hit-and-how-we-solved-them)
9. [Component reference](#9-component-reference)
10. [How to run / reproduce](#10-how-to-run--reproduce)
11. [Experiment tracking (MLflow + DagsHub)](#11-experiment-tracking-mlflow--dagshub)
12. [Tests](#12-tests)
13. [Setup & installation](#13-setup--installation)

---

## 1. Project overview

```
Attacker ──► CFG-valid prefix ──► MiniGPT (defender) ──► completed sentence
                                                              │
                                                       CFGValidator checks
                                                              │
                                                     RewardFunction scores
                                                              │
                                              scalar reward  ──►  REINFORCE (both sides)
```

- **Attacker goal:** produce a prefix that makes the defender complete the sentence either
  *ungrammatically* or *off-topic* relative to the prefix.
- **Defender goal:** complete any prefix into a fluent, grammatically and semantically valid
  sentence.
- **Same reward, opposite signs:** the attacker maximizes `R`; the defender maximizes `−R`.

The project runs the full loop end-to-end: supervised pretraining of the defender, REINFORCE
training of the attacker, REINFORCE fine-tuning of the defender, and finally **adversarial
co-training** where the two alternate.

---

## 2. The synthetic language

### Lexicon

| Type | Count | Examples of categories |
|---|---|---|
| Nouns | 69 | `ALIVE`, `HUMAN`, `ANIMAL`, `MACHINE`, `ABSTRACT`, `PLACE` |
| Verbs | 32 | `PHYSICAL_ACTION`, `LOCOMOTION`, `DESTRUCTION`, `SYSTEM_ACTION` |
| Adjectives | 41 | `SIZE`, `QUALITY`, `STATE`, `CONDITION` |

Every word carries two semantic properties:

- **Tags** — discrete category labels (e.g. `MAN → {ALIVE, HUMAN}`).
- **Axis values** — a 4-D real vector that places the word in a continuous semantic space:

  | Axis | Meaning | Range |
  |---|---|---|
  | `agency` | capacity for autonomous action | 0–5 |
  | `physicality` | physical concreteness | 0–5 |
  | `social` | social / relational quality | 0–5 |
  | `system` | machine / computational quality | 0–5 |

### Grammar (context-free skeleton)

```
START        → SUBJECT_TERM
SUBJECT_TERM → SUBJECT VERB_TERM
SUBJECT      → NOUN | ADJ NOUN
VERB_TERM    → VERB | VERB OBJECT | VERB VERB_TERM | VERB OBJECT VERB_TERM
OBJECT       → NOUN | ADJ NOUN | ADJ ADJ NOUN
```

`VERB_TERM` is recursive: a new verb term can follow a complete object *or* a bare
(intransitive) verb, so verb chains like `MAN RUN FALL` are grammatical.

### Semantic constraints (layered on top of the skeleton)

- Every verb requires its **subject** to satisfy tag-overlap + axis-bound constraints.
- **Transitive** verbs additionally constrain their **object**.
- A bare verb must be **intransitive**; a transitive verb must place its object before the
  next verb can start.
- Adjectives narrow which nouns they can modify (tag overlap + axis bounds, which *stack*
  across multiple adjectives).

Valid examples: `MAN RUN` · `MAN RUN FALL` · `FREE WOLF FALL` · `DRONE BREAK CLOCK` ·
`STRONG MAN CARRY SMALL BOOK FORGET LIE`

Invalid example: `RIVER BURN` → *“Verb 'BURN' is incompatible with subject 'RIVER'.”*

---

## 3. System architecture

```
┌──────────────────────┐    CFG-valid prefix    ┌──────────────────────┐
│  AttackerTransformer  │ ─────────────────────► │  MiniGPT (Defender)  │
│   210k params         │                        │   210k params        │
│   CFG-masked decoding │ ◄───────────────────── │  free completion     │
└──────────┬───────────┘   completed sentence    └──────────┬───────────┘
           │                                                  │
           ▼                                                  ▼
┌──────────────────────┐                          ┌──────────────────────┐
│   CFGStateTracker     │                          │     CFGValidator      │
│  grammar FSM; filters │                          │  1 unknown-word check │
│  valid next tokens +  │                          │  2 skeleton check     │
│  viability pruning    │                          │  3 semantic check     │
└──────────────────────┘                          └──────────┬───────────┘
                                                              │ valid / invalid + reason
                                                              ▼
                                          ┌─────────────────────────────────────┐
                                          │            RewardFunction            │
                                          │  split → features → distances → R    │
                                          │  R = w_g·grammar + w_t·tag + w_a·axis │
                                          └─────────────────────────────────────┘
```

Attacker and defender share the **exact same architecture and parameter count** so that any
performance difference is attributable to the *training signal*, not model capacity.

---

## 4. Repository layout

```
nlp-adversarial-defense/
├── data/
│   ├── raw/word_centered_language/   words.json (lexicon) + transition.json (grammar)
│   ├── raw/generated_texts/          generated_corpus_{100,500,1000,5000,10000}.txt
│   └── models/                       trained checkpoints (+ from_mlflow/, cotrain/)
│
├── src/
│   ├── language/                     lexicon parsing, CFG, CFG validator
│   │   └── entities/                 word_entity, cfg, cfg_base, cfg_validator
│   ├── model/                        tokenizer, transformer (MiniGPT)
│   ├── attacker/                     attacker.py, cfg_state_tracker.py
│   ├── reward/                       ◄ shared reward logic (neutral package)
│   │   ├── reward_computer.py        simple scalar reward
│   │   ├── reward_function.py        structured per-POS reward (attacker view)
│   │   └── defender_reward.py        inverted reward (defender view)
│   └── defender/                     re-exports defender reward
│
├── scripts/
│   ├── train/                        train_model, train_attacker, train_defender,
│   │                                 train_defender_rl, train_adversarial
│   ├── eval/                         attack_and_complete, evaluate_model, infer,
│   │                                 complete_and_validate, run_attacker, test_attacker
│   ├── data/                         text_generator (corpus generation)
│   ├── demo/                         demo_reward_function
│   ├── downloads/                    download_best_attacker / _defender from MLflow
│   └── plot_report_figures.py        regenerates the graphs in this README
│
├── tests/                            269 unit tests (pytest)
├── docs/figures/                     report figures (PNG)
└── README.md                         ← this file
```

The reward logic lives in its **own neutral `src/reward/` package**. Earlier it sat inside
`src/attacker/` and the defender imported *up* into the attacker package — a backwards
dependency. Moving it out makes both agents depend only on shared, side-agnostic code.

---

## 5. Design decisions: architecture & loss functions

*(Report requirement 1 — justification of the model architecture and loss function.)*

### 5.1 Why a decoder-only causal Transformer

The core task is **autoregressive generation**: produce / complete a sentence left-to-right.
A GPT-style decoder-only Transformer is the canonical fit — it predicts the next token from
all previous tokens under a causal mask. We deliberately kept it **small**:

| Hyperparameter | Value | Reason |
|---|---|---|
| Layers | 4 | The language has 146 tokens and a shallow CFG; depth beyond this adds nothing. |
| Embedding dim | 64 | Enough to separate 146 tokens + positions; small enough to train on CPU. |
| Attention heads | 4 | head_dim = 16, a standard split. |
| Context length | 32 | Longest valid sentences are well under 32 tokens. |
| FFN dim | 256 | Conventional 4× expansion. |
| Dropout | 0.1 | Light regularization; corpus is large relative to model. |
| Weight tying | yes | Input embedding = output projection → fewer params, better generalization. |
| **Total params** | **210,432** | Tiny, fast, hard to overfit a controlled language. |

**Why identical attacker and defender.** Both are 210k-param MiniGPTs. Keeping them equal in
capacity turns the experiment into a clean test of *training signal*: if the attacker wins, it
is because its policy found exploits, not because it is a bigger model.

**Why CFG masking on the attacker.** The attacker should not waste capacity re-learning grammar
the `CFGStateTracker` already enforces. At each step, logits for grammar-illegal tokens are set
to `−∞` before sampling, so the policy only ever distributes probability over *valid* moves and
learns the one thing that matters: *which* valid prefix provokes a bad completion.

### 5.2 Why two different loss functions

This project has **two distinct learning problems**, and each needs the right objective.

**(a) Defender pretraining → token-level cross-entropy.**
Pretraining MiniGPT to speak the language is ordinary maximum-likelihood language modeling, so
we use next-token cross-entropy with padding ignored:

```python
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
loss = criterion(logits.reshape(-1, V), targets.reshape(-1))
```

`ignore_index=pad_id` keeps padding tokens from contributing gradient. Cross-entropy is the
standard, well-behaved MLE objective for autoregressive LMs — nothing exotic is warranted.

**(b) Adversarial training → REINFORCE policy gradient.**
The adversarial reward depends on (i) the **CFG validator’s** binary verdict and (ii)
**discrete sampled tokens**. Neither is differentiable — you cannot backpropagate through a
hard validity check or through `argmax`/sampling. So we estimate the gradient of *expected
reward* with the REINFORCE (score-function) estimator:

```
advantage = R − baseline                  # baseline = EMA of recent rewards
loss      = −(advantage × Σ log π(aₜ))     # sum of log-probs of the sampled tokens
loss.backward()
```

Design choices inside REINFORCE, and why:

| Choice | Why |
|---|---|
| **EMA baseline** | REINFORCE gradients are high-variance. Subtracting a running mean of reward (`baseline_alpha = 0.05`) centers the advantage and stabilizes learning without a learned critic. |
| **Entropy bonus** (`−β·H`) | Pure REINFORCE collapses onto a single high-reward prefix (mode collapse). An entropy term keeps the policy spread out. |
| **Gradient clipping** (1.0) | Caps occasional huge policy-gradient steps. |
| **Defender reward = −attacker reward** | A zero-sum framing: the same scalar trains both sides, so they optimize directly against each other. |

### 5.3 The reward function (the heart of the system)

A single structured reward scores every exchange. It encodes a **three-level priority**:
grammar failure ≫ topic mismatch ≫ topic drift.

```
R = w_grammar · grammar_reward      # 1.0 if CFG-invalid, else 0.0
  + w_tag     · tag_mismatch        # mean of 3 per-POS Jaccard distances (noun/verb/adj)
  + w_axis    · axis_distance       # cosine distance of mean 4-D axis vectors
```

| Weight | Value | Role |
|---|---|---|
| `w_grammar` | 1.0 | Grammar failure dominates — the biggest win for the attacker. |
| `w_tag` | 0.30 | Tag mismatch — medium signal. |
| `w_axis` | 0.20 | Axis drift — medium signal. |
| **max R** | **1.50** | all three components = 1.0 simultaneously. |

**Why per-POS tag distances** (noun / verb / adjective tracked separately rather than one pooled
set): it tells you *which* grammatical slot drifted, which is far more actionable both for
analysis and as a learning signal. **Why a continuous axis term on top of discrete tags:** two
words can share zero tags yet be semantically near (or vice-versa); the cosine distance on
axis vectors captures graded drift that the Jaccard term alone misses. This shaping gives the
attacker a smooth gradient to climb even on episodes where it has not yet achieved a full
grammar failure.

Worked reward levels:

| Case | grammar | tag | axis | total R |
|---|---|---|---|---|
| Grammar failure | 1.00 | ~0.20 | ~0.01 | **~1.21** |
| Topic mismatch | 0.00 | ~0.20 | ~0.08 | **~0.28** |
| Topic consistent | 0.00 | ~0.10 | ~0.00 | **~0.10** |

---

## 6. Training: how it went, with graphs

*(Report requirement 2 — the training process, illustrated.)*

All figures below are generated directly from the CSV logs in `logs/` by
`python scripts/plot_report_figures.py`.

### 6.1 Defender pretraining (cross-entropy)

MiniGPT trained on the 10,000-sentence corpus for 32 epochs (AdamW, lr 3e-4, batch 64,
grad-clip 1.0). Loss falls sharply in the first ~5 epochs, then settles into a smooth decline —
the expected MLE curve, with no sign of divergence or overfitting.

![MiniGPT training loss](docs/figures/fig_minigpt_loss.png)

- Loss: **4.095 → 2.819** (start → epoch 32); best team checkpoint reached **2.79**.
- Outcome: the defender reliably generates grammatical, on-topic sentences — a competent
  baseline opponent for the attacker.

### 6.2 Attacker REINFORCE (vs. frozen defender)

The attacker policy was trained with REINFORCE against the frozen defender. The two curves
below are the same run (2,000 episodes, EMA baseline): as the attacker’s **reward rises**, the
defender’s **valid-completion rate collapses** — direct evidence the attacker is learning to
break the defender.

![Attacker REINFORCE](docs/figures/fig_attacker_reward.png)

- Avg reward: **0.56 → 1.26**; grammar-failure rate: **0.32 → 1.00**.
- Clean crossover near episode ~700, where the attacker discovers prefix shapes that force a
  semantic violation almost every time.
- The team’s longer 20k-episode run (`attacker_best.pt`) reached **avg reward ≈ 1.31**.

### 6.3 Defender REINFORCE fine-tuning (vs. frozen attacker)

We then fine-tuned the defender with REINFORCE (reward = −attacker reward) for 10,000 episodes.
Crucially, **30% of prefixes were drawn at random** from the CFG rather than from the
mode-collapsed attacker, so the defender generalizes instead of memorizing one exploit.

![Defender RL fine-tuning](docs/figures/fig_defender_rl.png)

- Defender valid rate: **~76% → ~96%** and holds; defender reward rises from ~−0.55 to ~−0.34.
- The fine-tuned defender repairs the specific weaknesses the attacker exploited while staying
  general across the whole grammar.

### 6.4 Adversarial co-training (alternating)

Finally, both sides train in alternating phases (`scripts/train/train_adversarial.py`):
each round = X attacker episodes (defender frozen) then X defender episodes (attacker frozen),
followed by a frozen head-to-head evaluation. The graph shows the **arms race**: defender
validity (green) dips during every attacker phase and recovers during every defender phase.

![Adversarial co-training](docs/figures/fig_cotraining.png)

- The deep, narrow spikes are moments where the attacker briefly discovers a fresh exploit
  before the defender adapts in the following phase.
- Lessons baked in: attacker phases use an **entropy bonus** (anti–mode-collapse); defender
  phases **mix random prefixes** (anti-forgetting); the state tracker guarantees every prefix
  is answerable so neither side can “win” via dead-ends.

> **Known dynamic (honest note).** Co-training validity is *jumpy* — it starts high and
> oscillates hard within each round. That is inherent to alternating REINFORCE with a frozen
> opponent: the training side over-optimizes against a stationary target. We dampen it with the
> entropy bonus, random-prefix mixing, and an **early-stop** that ends an attacker phase once
> its rolling validity drops below a threshold (so the defender can respond sooner). Smoother
> curves would need a smaller LR or true simultaneous updates.

---

## 7. Evaluation: tasks, rationale & results

*(Report requirement 3 — what we evaluate, why, and the numbers.)*

**Why these metrics.** The synthetic language was chosen precisely so that **validity is a
ground-truth label**: the `CFGValidator` gives an exact valid/invalid verdict for any sentence,
with no annotation. So *validity rate* is our primary, unambiguous metric, and *average reward*
is the secondary metric because it captures graded semantic drift even when a sentence is
technically valid. All head-to-head evaluations **freeze both models** so we measure policy
quality, not a moving target.

### Task 1 — Defender language-modeling quality
*Question:* does the pretrained defender actually speak the language?
*Metric:* training cross-entropy + CFG validity of free generations.
*Result:* loss **2.82**; free samples are consistently grammatical and on-topic.

### Task 2 — Attack success (before vs. after REINFORCE)
*Question:* does REINFORCE make the attacker measurably better at breaking the defender?
*Metric:* defender valid-rate and attacker avg-reward against the **same** frozen defender,
same temperatures, same `min_completion_tokens=1` rule, at **n = 100,000** episodes.

| Metric | Untrained attacker | Trained `attacker_best.pt` | Δ |
|---|---|---|---|
| VALID completions | 44,898 / 100k (**44.9%**) | 5,295 / 100k (**5.3%**) | **−39.6 pts** |
| INVALID completions | 55,102 / 100k (**55.1%**) | 94,705 / 100k (**94.7%**) | **+39.6 pts** |
| Avg reward | **0.814** | **1.312** | **+0.50 (≈1.6×)** |

*Interpretation:* an untrained attacker emits ~uniform legal prefixes and the defender survives
~45% of the time; after training, the attacker concentrates on prefix shapes that force a
semantic violation ~95% of the time. Logged to DagsHub under `AttackBenchmark`.

### Task 3 — Defender robustness after RL fine-tuning
*Question:* can the defender be hardened against the trained attacker without forgetting the
rest of the language?
*Metric:* valid-rate vs. attacker prefixes **and** random CFG prefixes.
*Result:* valid-rate **~76% → ~96%**; generalization confirmed by the random-prefix mix
(validity stays high on prefixes the attacker never uses).

### Task 4 — Co-training dynamics
*Question:* what happens when both sides adapt?
*Metric:* per-round frozen head-to-head valid-rate.
*Result:* an oscillating equilibrium; at round *ends* (after the defender phase) the defender
holds **~95–99%** validity, i.e. it stays ahead across rounds while the attacker keeps probing.

---

## 8. Problems we hit and how we solved them

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | **Attacker mode collapse** — policy converged to a single prefix (`FAMOUS HOME`, `CITY …`). | Pure REINFORCE rewards exploitation of the single best exploit. | Added an **entropy bonus** to the attacker loss; in co-training, **early-stop** an attacker phase once validity collapses. |
| 2 | **Unanswerable dead-end prefixes** — 11 nouns (CITY, HOME, FOREST, …) have *zero* compatible verbs, so the defender *cannot* complete them validly. | The grammar allows a bare subject that no verb accepts. | `CFGStateTracker._precompute_viability()` prunes subjects/adjectives that lead to dead-ends, so every offered prefix is **completable**. |
| 3 | **Defender forgetting** during RL — it overfit to the collapsed attacker’s one prefix. | Training only on a degenerate prefix distribution. | **Mix 30% random CFG prefixes** (`--mix-random 0.3`) so the defender stays general (verified: validity high on unseen prefixes). |
| 4 | **Grammar change** — added `VERB_TERM → VERB VERB_TERM` (verb chaining, e.g. `MAN RUN FALL`). | New rule invalidated the corpus, the trained defender, and the state machine. | Regenerated the corpus, **retrained** the defender, and updated `CFGStateTracker` (new `AFTER_INTRANS_VERB` state offering verbs). |
| 5 | **DagsHub 401 auth** + **DagsHub downtime**. | DagsHub needs the token as **both** MLflow username *and* password; server occasionally unreachable. | Send token as user+pass; on any connection error, **fall back to local MLflow** (`mlruns/`). |
| 6 | **Windows crashes** — OpenMP `libiomp5md.dll` conflict and `cp1252` Unicode errors. | PyTorch OMP double-load; console can’t encode special glyphs. | Set `KMP_DUPLICATE_LIB_OK=TRUE` in every entry script; replace non-ASCII output with ASCII. |
| 7 | **Messy architecture** — reward code inside `src/attacker/`, defender importing *up* into attacker; duplicate `train_model.py`; flat `scripts/`. | Organic growth. | Extracted **`src/reward/`** shared package; grouped scripts into `train/ eval/ data/ demo/ downloads/`; deleted the duplicate. All 269 tests still pass. |

---

## 9. Component reference

| Component | File | Role |
|---|---|---|
| **LexiconParser** | `src/language/parsers.py` | Parse `words.json` into typed dataclasses. |
| **CFG** | `src/language/entities/cfg.py` | Generate grammatical+semantic sentences. |
| **CFGValidator** | `src/language/entities/cfg_validator.py` | 3-phase validation (unknown → skeleton → semantics). |
| **WordTokenizer** | `src/model/tokenizer.py` | Word-level tokenizer; `<PAD>=0 <BOS>=1 <EOS>=2 <UNK>=3`; vocab 146. |
| **MiniGPT** | `src/model/transformer.py` | 210k-param decoder-only Transformer (defender). |
| **CFGStateTracker** | `src/attacker/cfg_state_tracker.py` | Grammar FSM; exposes valid next tokens + viability pruning. |
| **AttackerTransformer** | `src/attacker/attacker.py` | Same arch as MiniGPT, with CFG-masked decoding + log-prob RL hooks. |
| **RewardComputer** | `src/reward/reward_computer.py` | Simple scalar reward (`grammar + topic_mismatch`). |
| **RewardFunction** | `src/reward/reward_function.py` | Structured per-POS reward (attacker view). |
| **DefenderRewardFunction** | `src/reward/defender_reward.py` | Inverted reward (defender view). |

---

## 10. How to run / reproduce

> On Windows, prefix any command with `$env:KMP_DUPLICATE_LIB_OK="TRUE"` (the scripts also set
> this in code).

```powershell
# 1. Pretrain the defender (cross-entropy)
python scripts/train/train_model.py --corpus 10000 --epochs 32

# 2. Train the attacker (REINFORCE, frozen defender)
python scripts/train/train_attacker.py --episodes 2000               # local
python scripts/train/train_attacker.py --episodes 20000 --mlflow     # tracked

# 3. Fine-tune the defender (REINFORCE, frozen attacker)
python scripts/train/train_defender_rl.py --episodes 10000 --mix-random 0.3 --mlflow

# 4. Adversarial co-training (alternating)
python scripts/train/train_adversarial.py --rounds 10 -x 1000 --mlflow

# Evaluate / inspect
python scripts/eval/attack_and_complete.py --n 100000 --mlflow       # benchmark
python scripts/eval/attack_and_complete.py --n 10 --verbose          # reward breakdown
python scripts/eval/infer.py                                         # interactive completion
python scripts/eval/validate_sentence.py "FREE WOLF FALL"            # validate one sentence
python scripts/demo/demo_reward_function.py                          # step-by-step reward demo

# Regenerate the report figures
python scripts/plot_report_figures.py
```

Key attacker/defender flags: `--lr`, `--max-prefix`, `--atk-temp`/`--def-temp`,
`--w-grammar`/`--w-tag`/`--w-axis`, `--baseline-alpha`, `--entropy-coef`, `--mix-random`,
`--window`, `--seed`.

---

## 11. Experiment tracking (MLflow + DagsHub)

Training logs to **MLflow**, backed by a shared **DagsHub** server so the whole team sees one
experiment view; if DagsHub is unreachable it transparently falls back to local `mlruns/`.

**Setup.** Copy the template and fill in your credentials (the file is git-ignored):

```powershell
cp config.example.py config.py
# then edit:
#   DAGSHUB_REPO_OWNER = "your-username"
#   DAGSHUB_REPO_NAME  = "nlp-adversarial-defense"
#   DAGSHUB_TOKEN      = "your-api-token"   # from https://dagshub.com/user/settings/tokens
```

Environment variables (`DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME`, `DAGSHUB_TOKEN`) work too and
take over if `config.py` is absent. Add `--mlflow` to any training command to enable tracking.

**Experiments:** `MiniGPT` (pretraining) · `AttackerREINFORCE` · `DefenderRL` ·
`AdversarialCoTraining` · `AttackBenchmark`. Each run logs params (lr, temps, weights, seed,
param_count), metrics (loss / reward / valid-rate / baseline), and artifacts (checkpoints,
tokenizer, episode CSVs).

**Pull the best team models:**

```powershell
python scripts/downloads/download_best_attacker.py   # → data/models/from_mlflow/
python scripts/downloads/download_best_defender.py
```

> DagsHub auth quirk: the access token must be sent as **both** the MLflow username **and**
> password — passing the repo owner as the username returns 401. The download scripts handle
> this automatically.

---

## 12. Tests

269 unit tests (`unittest`, pytest-compatible), one file per component — lexicon parser, CFG,
CFG validator, tokenizer, MiniGPT, state tracker, attacker, all three reward modules, and
REINFORCE smoke tests for both attacker and defender training loops.

```powershell
python -m pytest tests/ -q                                  # all
python -m pytest tests/test_reward_function.py -v           # one file
```

---

## 13. Setup & installation

```
Python ≥ 3.10
torch · mlflow · dagshub · pandas · matplotlib · pytest
```

```powershell
pip install torch mlflow dagshub pandas matplotlib pytest
# or: conda env create -f environment.yml
```

**Windows OpenMP fix** (also set in-code by every entry script):

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

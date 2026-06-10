# Team Setup: DagsHub + MLflow for Collaborative Experiment Tracking

This guide explains how to set up collaborative experiment tracking for the team using DagsHub and MLflow.

---

## Quick Start (Local Training Without DagsHub)

If you just want to train locally without experiment tracking:

```bash
python scripts/train_model.py --corpus 100 --epochs 5
```

Logs and metrics go to `logs/` and `data/models/` directories.

---

## Team Setup: DagsHub (Recommended)

### Step 1: Create a DagsHub Repo (Team Lead)

1. Go to [DagsHub.com](https://dagshub.com) and log in
2. Click **"New Repository"**
3. Enter repository name: `nlp-adversarial-defense`
4. Add team members as **Collaborators** (Settings > Members)
5. Invite them via email

### Step 2: Each Team Member — Initialize Local Repo

```bash
cd ~/Documents/work/nlp/nlp-adversarial-defense

# If not already a git repo:
git init
git remote add origin https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense.git
git fetch origin
git branch -u origin/main main

# Or if repo already exists, just update remote:
git remote set-url origin https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense.git
git fetch origin
```

### Step 3: Install MLflow + DagsHub

```bash
pip install mlflow dagshub
```

### Step 4: Train with Team Logging

Set environment variables and run training:

```bash
export DAGSHUB_REPO_OWNER="<your-dagshub-username>"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="<your-dagshub-api-token>"  # Optional, for remote push

python scripts/train_model.py --mlflow --corpus 100 --epochs 5
```

**To get your DagsHub API token:**
1. Go to [DagsHub Settings > Tokens](https://dagshub.com/user/settings/tokens)
2. Create a new token
3. Copy it and export as `DAGSHUB_TOKEN`

### Step 5: View Experiment Results

**Option A: DagsHub Web UI**
- Go to https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense/experiments
- All team members' runs appear here
- See metrics, parameters, artifacts (models, embeddings)

**Option B: Local MLflow UI**
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Persistent Environment Setup (Recommended)

To avoid setting env vars every time, add them to your shell config:

**For zsh (add to `~/.zshrc`):**
```bash
export DAGSHUB_REPO_OWNER="<your-dagshub-username>"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="<your-dagshub-api-token>"
```

Then:
```bash
source ~/.zshrc
python scripts/train_model.py --mlflow --corpus 10000
```

**For bash (add to `~/.bashrc`):**
```bash
export DAGSHUB_REPO_OWNER="<your-dagshub-username>"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="<your-dagshub-api-token>"
```

---

## Training Examples

### Local Training (No Tracking)
```bash
python scripts/train_model.py --corpus 100 --epochs 5
```

### Quick Team Experiment (With MLflow)
```bash
python scripts/train_model.py --mlflow --corpus 100 --epochs 3 --seed 42
```

### Full Training Run (Team Tracking)
```bash
python scripts/train_model.py --mlflow --corpus 10000 --epochs 30 --lr 3e-4 --seed 42
```

### Custom Hyperparameters
```bash
python scripts/train_model.py --mlflow \
  --corpus 5000 \
  --epochs 20 \
  --batch-size 32 \
  --lr 5e-4 \
  --embed-dim 128 \
  --num-heads 8 \
  --num-layers 6 \
  --dropout 0.2 \
  --seed 123
```

---

## What Gets Logged?

### Local (Always)
- **Logs:** `logs/train_corpus{SIZE}.log`
- **Metrics CSV:** `logs/metrics_corpus{SIZE}.csv` (epoch, loss, time)
- **Model Checkpoint:** `data/models/minigpt_corpus{SIZE}.pt`
- **Tokenizer:** `data/models/tokenizer_corpus{SIZE}.json`
- **Embeddings:** `data/models/embeddings_corpus{SIZE}.npy`
- **Sample Outputs:** Printed to log file

### DagsHub (With `--mlflow`)
- All of the above + **remote backup**
- **Metrics Dashboard:** Real-time loss tracking across all team runs
- **Artifacts:** Model checkpoints, tokenizer, embeddings stored remotely
- **Parameters:** All hyperparameters logged and searchable
- **Reproducibility:** Exact config saved in checkpoint for easy restoration

---

## Retrieving Models from DagsHub

If a teammate trained a model and pushed it to DagsHub:

```bash
# Option 1: Via Web UI
# Go to https://dagshub.com/<TEAM>/nlp-adversarial-defense/experiments
# Download the model artifact directly

# Option 2: Via MLflow (if you have access)
mlflow artifacts download -r <RUN_ID> --artifact-path models
```

---

## Troubleshooting

### Error: `DAGSHUB_REPO_OWNER not found`
→ Set environment variables (see Step 4 above)

### Error: `mlflow not installed`
```bash
pip install mlflow dagshub
```

### Error: `Connection refused` / auth fails
→ Check your `DAGSHUB_TOKEN` is valid (regenerate at https://dagshub.com/user/settings/tokens)

### Training works but MLflow doesn't log
→ Run with `--mlflow` flag. Without it, only local logging happens (this is intentional).

---

## Git Best Practices for the Team

Always commit your code changes:
```bash
git add -A
git commit -m "experiment: train with custom hyperparams (corpus 5k, lr 5e-4)"
git push origin main
```

Models are stored locally in `data/models/` but backed up on DagsHub if you use `--mlflow`. Don't commit model files to git (they're too large).

---

## Questions?

- DagsHub Docs: https://dagshub.com/docs
- MLflow Docs: https://mlflow.org/docs/


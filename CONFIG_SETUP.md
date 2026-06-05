# Configuration Setup Guide

This guide explains how to set up and use the `config.py` file for DagsHub team logging.

## Quick Setup (5 minutes)

### Option 1: Use Existing `config.py` (Easiest)

If you already have `config.py` configured in your workspace:

```bash
python scripts/train/train_model.py --mlflow --corpus 100 --epochs 5
```

The credentials will be automatically read from `config.py`.

---

### Option 2: Create `config.py` from Template

If you're setting up for the first time:

```bash
# Copy the example template
cp config.example.py config.py

# Edit with your DagsHub credentials
nano config.py  # or your preferred editor
```

Then set these values in `config.py`:

```python
DAGSHUB_REPO_OWNER = "your-dagshub-username"
DAGSHUB_REPO_NAME = "nlp-adversarial-defense"
DAGSHUB_TOKEN = "your-dagshub-api-token"
```

Save and run:

```bash
python scripts/train/train_model.py --mlflow --corpus 100 --epochs 5
```

---

## Getting Your DagsHub Credentials

### Step 1: DagsHub Username
1. Go to https://dagshub.com and log in
2. Click your profile icon (top right)
3. Your username is shown. Use it for `DAGSHUB_REPO_OWNER`

### Step 2: DagsHub API Token
1. Go to https://dagshub.com/user/settings/tokens
2. Click "Create New Token"
3. Name it (e.g., "local-dev", "team-training")
4. Click "Create"
5. Copy the token and paste into `config.py` as `DAGSHUB_TOKEN`

---

## Fallback: Environment Variables (if config.py not used)

You can still use environment variables if you prefer:

```bash
export DAGSHUB_REPO_OWNER="your-dagshub-username"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="your-dagshub-api-token"

python scripts/train/train_model.py --mlflow --corpus 10000 --epochs 30
```

The script checks for config.py first, then falls back to environment variables.

---

## Security

- **`config.py` is in `.gitignore`** — It will NOT be committed to git, so your token stays private
- **`config.example.py` is committed** — Shows the template for new team members
- **If you accidentally commit credentials** — Regenerate your token at https://dagshub.com/user/settings/tokens

---

## Verification

After setup, verify it works by running a quick training test:

```bash
python scripts/train/train_model.py --mlflow --corpus 100 --epochs 1
```

You should see output like:

```
✓ Connected to DagsHub: <your-username>/nlp-adversarial-defense
Experiments will be logged to DagsHub.
```

Then check your experiments at:
```
https://dagshub.com/<your-username>/nlp-adversarial-defense.mlflow/#/experiments
```

---

## Team Workflow

1. **Team Lead**: 
   - Create DagsHub repo: `nlp-adversarial-defense`
   - Invite team members

2. **Each Team Member**:
   - Clone repo
   - Copy `config.example.py` to `config.py`
   - Fill in your DAGSHUB credentials
   - Run training with `--mlflow` flag

3. **View All Experiments**:
   - Everyone sees all runs on DagsHub (if given repo access)
   - Dashboard: https://dagshub.com/&lt;team&gt;/nlp-adversarial-defense.mlflow

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Failed to connect to DagsHub" | Check `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME` match your actual DagsHub repo |
| "Invalid credentials" | Regenerate token at https://dagshub.com/user/settings/tokens and update `DAGSHUB_TOKEN` |
| "ImportError: config.py not found" | Run `cp config.example.py config.py` first |
| Experiments not appearing on DagsHub | Ensure you run with `--mlflow` flag |

---

## Files

- **`config.py`** — Your personal DagsHub credentials (git-ignored, never committed)
- **`config.example.py`** — Template showing what credentials you need (committed to git)
- **`.gitignore`** — Prevents `config.py` from being committed



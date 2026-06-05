# Quick Checklist: Team DagsHub Setup

Use this checklist to get started with collaborative experiment logging.

## Team Lead Setup (One Time)

- [ ] Create DagsHub account (if not already done)
- [ ] Create repository on DagsHub: `nlp-adversarial-defense`
- [ ] Invite team members (Settings > Members)
- [ ] Note your DagsHub username for step 2

## Each Team Member Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install mlflow dagshub
```

### 2. Set Environment Variables (Persistent)

**Option A: Add to ~/.zshrc (Permanent)**
```bash
# Add these lines to ~/.zshrc
export DAGSHUB_REPO_OWNER="<your-dagshub-username>"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="<your-dagshub-api-token>"
```

Then reload shell:
```bash
source ~/.zshrc
```

**Option B: Set Per Session (Temporary)**
```bash
export DAGSHUB_REPO_OWNER="<your-dagshub-username>"
export DAGSHUB_REPO_NAME="nlp-adversarial-defense"
export DAGSHUB_TOKEN="<your-dagshub-api-token>"
```

### 3. Clone Team Repository
```bash
git clone https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense.git
cd nlp-adversarial-defense
```

Or update existing repo:
```bash
git remote set-url origin https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense.git
```

### 4. Get Your DagsHub Token
1. Go to: https://dagshub.com/user/settings/tokens
2. Create a new API token (name it "local-dev")
3. Copy the token and save it somewhere safe
4. Use it as `DAGSHUB_TOKEN` above

---

## Run Training

### Local Training (No Tracking)
```bash
python scripts/train_model.py --corpus 100 --epochs 5
```
✓ Saves locally to `logs/` and `data/models/`

### Team Tracking (With DagsHub)
```bash
python scripts/train_model.py --mlflow --corpus 100 --epochs 5
```
✓ Saves locally **AND** uploads to DagsHub

### View Results
- **Local:** `mlflow ui` then open http://localhost:5000
- **Team:** https://dagshub.com/<TEAM_USERNAME>/nlp-adversarial-defense/experiments

---

## Common Commands

| Task | Command |
|------|---------|
| Quick test locally | `python scripts/train_model.py --corpus 100 --epochs 2` |
| Quick test with tracking | `python scripts/train_model.py --mlflow --corpus 100 --epochs 2` |
| Full training (no tracking) | `python scripts/train_model.py --corpus 10000 --epochs 30` |
| Full training with tracking | `python scripts/train_model.py --mlflow --corpus 10000 --epochs 30` |
| View local metrics | `mlflow ui` |
| Check available parameters | `python scripts/train_model.py --help` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Error: `DAGSHUB_REPO_OWNER not found` | Set env vars (see step 2) |
| Error: `mlflow not installed` | `pip install mlflow dagshub` |
| Error: `Connection refused` to DagsHub | Check `DAGSHUB_TOKEN` is valid; regenerate at https://dagshub.com/user/settings/tokens |
| Training runs but no MLflow logs | Add `--mlflow` flag |
| Can't see other team members' runs | Make sure you're logged into DagsHub and have repo access |

---

## Questions?

See full guide: `docs/TEAM_SETUP.md`


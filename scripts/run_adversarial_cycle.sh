#!/usr/bin/env bash
# Full adversarial cycle: pre-RL benchmark -> defender RL -> post-RL benchmark
set -euo pipefail

PYTHON="${PYTHON:-/home/konstantine/miniconda3/envs/nlp-adversarial/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs data/models
LOG="logs/adversarial_cycle_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo " Adversarial cycle started: $(date -Iseconds)"
echo " Python: $PYTHON"
echo " Log:    $LOG"
echo "=============================================="

BASE_DEF="data/models/minigpt_corpus10000.pt"
RL_BEST="data/models/defender_rl_best.pt"
RL_FINAL="data/models/defender_rl_final.pt"

if [[ ! -f "$BASE_DEF" ]]; then
  echo ""
  echo "=== Step 0: Train base MiniGPT (prerequisite) ==="
  "$PYTHON" scripts/train/train_model.py --mlflow --corpus 10000 --epochs 30 --seed 42
else
  echo ""
  echo "=== Step 0: Base MiniGPT already at $BASE_DEF — skipping ==="
fi

echo ""
echo "=== Step 1: Pre-RL attack benchmark (1000 iterations) ==="
"$PYTHON" scripts/attack_and_complete.py \
  --n 1000 --mlflow --quiet --log-every 100 \
  --run-name pre_rl_benchmark \
  --defender-ckpt "$BASE_DEF"

echo ""
echo "=== Step 2: Defender REINFORCE training (2000 episodes) ==="
"$PYTHON" scripts/train/train_defender.py \
  --episodes 2000 --mlflow --seed 42 \
  --defender-ckpt "$BASE_DEF"

DEF_CKPT="$RL_BEST"
if [[ ! -f "$DEF_CKPT" ]]; then
  DEF_CKPT="$RL_FINAL"
fi
echo "Using RL defender checkpoint: $DEF_CKPT"

echo ""
echo "=== Step 3: Post-RL attack benchmark (1000 iterations) ==="
"$PYTHON" scripts/attack_and_complete.py \
  --n 1000 --mlflow --quiet --log-every 100 \
  --run-name post_rl_benchmark \
  --defender-ckpt "$DEF_CKPT"

echo ""
echo "=============================================="
echo " Adversarial cycle finished: $(date -Iseconds)"
echo " Logs:  logs/pre_rl_benchmark.csv, logs/post_rl_benchmark.csv"
echo "        logs/defender_episodes_seed42.csv"
echo " MLflow: mlruns/ (local) + DagsHub if connected"
echo "=============================================="

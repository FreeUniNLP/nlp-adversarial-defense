"""
Download the BEST attacker checkpoint from the team MLflow / DagsHub server.

"Best" = the AttackerREINFORCE run with the highest `best_avg_reward`
(ties broken by the most training episodes). The script connects to the
shared DagsHub MLflow tracking server, finds that run, and downloads its
`attacker_best.pt` artifact into:

    data/models/from_mlflow/

Usage:
    python scripts/download_best_attacker.py
    python scripts/download_best_attacker.py --experiment AttackerREINFORCE
    python scripts/download_best_attacker.py --run-id 32b6b9dbc1174127ab2787e6289d6a96
    python scripts/download_best_attacker.py --artifact attacker_final.pt

Notes:
  * DagsHub auth: the personal token must be sent as BOTH the MLflow
    username and password — passing the repo owner as the username returns
    401. This script handles that automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

import mlflow
from mlflow.tracking import MlflowClient

try:
    from config import DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
except ImportError:
    DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
    DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

DEST_DIR = PROJECT_ROOT / "data" / "models" / "from_mlflow"


# ------------------------------------------------------------------ #
#  DagsHub connection                                                  #
# ------------------------------------------------------------------ #

def connect() -> str:
    """Point MLflow at the team DagsHub server and authenticate.

    DagsHub requires the access token as BOTH username and password.
    Returns the tracking URI.
    """
    if not (DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME and DAGSHUB_TOKEN):
        raise RuntimeError(
            "Missing DagsHub credentials. Set DAGSHUB_REPO_OWNER, "
            "DAGSHUB_REPO_NAME, and DAGSHUB_TOKEN in config.py or the environment."
        )
    # Token must be sent as username AND password (owner-as-username -> 401).
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
    return uri


# ------------------------------------------------------------------ #
#  Run selection                                                       #
# ------------------------------------------------------------------ #

def _episodes(run) -> int:
    try:
        return int(run.data.params.get("episodes", 0))
    except (TypeError, ValueError):
        return 0


def find_best_run(experiment_name: str):
    """Return the run with the highest best_avg_reward (ties -> most episodes)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found on the server.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    def score(run):
        reward = run.data.metrics.get("best_avg_reward")
        reward = reward if reward is not None else float("-inf")
        return (reward, _episodes(run))

    runs_sorted = sorted(runs, key=score, reverse=True)
    return runs_sorted, runs_sorted[0]


def print_table(runs) -> None:
    hdr = f"  {'run':24} {'user':12} {'episodes':>9} {'best_reward':>12} {'grammar_fail':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in runs:
        user   = r.data.tags.get("mlflow.user", "?")[:12]
        reward = r.data.metrics.get("best_avg_reward")
        gfail  = r.data.metrics.get("grammar_fail_rate")
        reward_s = f"{reward:.4f}" if reward is not None else "   -"
        gfail_s  = f"{gfail:.3f}" if gfail is not None else "   -"
        name = (r.info.run_name or r.info.run_id[:8])[:24]
        print(f"  {name:24} {user:12} {_episodes(r):>9} {reward_s:>12} {gfail_s:>13}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    print("Connecting to DagsHub MLflow...")
    uri = connect()
    print(f"  Tracking URI: {uri}\n")

    client = MlflowClient()

    # --- pick the run ---
    if args.run_id:
        run = client.get_run(args.run_id)
        print(f"Using pinned run: {run.info.run_name or args.run_id}\n")
    else:
        runs_sorted, run = find_best_run(args.experiment)
        print(f"Found {len(runs_sorted)} run(s) in '{args.experiment}':\n")
        print_table(runs_sorted)
        print()
        print(f"==> Best: '{run.info.run_name}'  "
              f"(episodes={_episodes(run)}, "
              f"best_avg_reward={run.data.metrics.get('best_avg_reward')})\n")

    # --- download ---
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{args.artifact}' -> {DEST_DIR}")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path=args.artifact,
        dst_path=str(DEST_DIR),
    )

    # --- save a sidecar metadata file so the team knows what this checkpoint is ---
    meta_path = DEST_DIR / (Path(args.artifact).stem + "_INFO.txt")
    meta_path.write_text(
        "Best attacker checkpoint downloaded from team MLflow / DagsHub\n"
        f"  experiment     : {args.experiment}\n"
        f"  run_name       : {run.info.run_name}\n"
        f"  run_id         : {run.info.run_id}\n"
        f"  user           : {run.data.tags.get('mlflow.user', '?')}\n"
        f"  episodes       : {_episodes(run)}\n"
        f"  best_avg_reward: {run.data.metrics.get('best_avg_reward')}\n"
        f"  grammar_fail   : {run.data.metrics.get('grammar_fail_rate')}\n"
        f"  artifact       : {args.artifact}\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Saved checkpoint: {local_path}")
    print(f"[OK] Saved metadata  : {meta_path}")
    print("\nUse it with the attack pipeline:")
    print(f"  python scripts/attack_and_complete.py --attacker-ckpt \"{local_path}\" --n 100")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the best attacker checkpoint from MLflow/DagsHub")
    p.add_argument("--experiment", default="AttackerREINFORCE",
                   help="MLflow experiment name (default: AttackerREINFORCE)")
    p.add_argument("--artifact", default="attacker_best.pt",
                   help="Artifact filename to download (default: attacker_best.pt)")
    p.add_argument("--run-id", default=None, dest="run_id",
                   help="Pin a specific run id instead of auto-selecting the best")
    return p.parse_args()


if __name__ == "__main__":
    main()

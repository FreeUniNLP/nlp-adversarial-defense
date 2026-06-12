"""
Download the BEST defender (MiniGPT) checkpoint from the team MLflow / DagsHub server.

"Best" = the MiniGPT run with the LOWEST `best_loss` (ties broken by the most
epochs). The script connects to the shared DagsHub MLflow tracking server,
finds that run, and downloads its checkpoint + tokenizer into:

    data/models/from_mlflow/

The tokenizer is downloaded alongside the checkpoint because the model's
vocabulary must match the tokenizer it was trained with.

Usage:
    python scripts/download_best_defender.py
    python scripts/download_best_defender.py --run-id 5dfceb980dff40cea7011baaab6ecfa9
    python scripts/download_best_defender.py --experiment MiniGPT

Notes:
  * DagsHub auth: the personal token must be sent as BOTH the MLflow
    username and password — passing the repo owner as the username returns
    401. This script handles that automatically.
  * Loss values are only comparable between runs trained on the SAME
    grammar/corpus version. If the grammar changed since a run was logged,
    a lower loss does not mean a better model for the current language.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

try:
    from config import DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
except ImportError:
    DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
    DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

DEST_DIR = PROJECT_ROOT / "data" / "models" / "from_mlflow"

CHECKPOINT_ARTIFACT = "minigpt_corpus10000.pt"
TOKENIZER_ARTIFACT  = "tokenizer_corpus10000.json"


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
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
    return uri


# ------------------------------------------------------------------ #
#  Run selection                                                       #
# ------------------------------------------------------------------ #

def _epochs(run) -> int:
    try:
        return int(run.data.params.get("epochs", 0))
    except (TypeError, ValueError):
        return 0


def _has_checkpoint(client: MlflowClient, run_id: str) -> bool:
    try:
        return any(a.path == CHECKPOINT_ARTIFACT for a in client.list_artifacts(run_id))
    except Exception:
        return False


def find_best_run(experiment_name: str, client: MlflowClient):
    """Return the finished run with the LOWEST best_loss that has a checkpoint.

    Ties are broken by the most epochs.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found on the server.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    scored = []
    for r in runs:
        loss = r.data.metrics.get("best_loss")
        if loss is None or r.info.status != "FINISHED":
            continue
        scored.append((loss, -_epochs(r), r))
    scored.sort(key=lambda t: (t[0], t[1]))

    # Walk from best to worst until we find a run with a checkpoint artifact
    for loss, _, run in scored:
        if _has_checkpoint(client, run.info.run_id):
            return runs, run
        print(f"  [skip] '{run.info.run_name}' (loss={loss:.4f}) has no checkpoint artifact")

    raise RuntimeError("No finished run with a checkpoint artifact was found.")


def print_table(runs) -> None:
    rows = []
    for r in runs:
        loss = r.data.metrics.get("best_loss")
        rows.append((loss if loss is not None else float("inf"), r))
    rows.sort(key=lambda t: t[0])

    hdr = f"  {'run':24} {'user':12} {'corpus':>7} {'epochs':>7} {'best_loss':>10} {'status':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for loss, r in rows:
        loss_s = f"{loss:.4f}" if loss != float("inf") else "    -"
        name = (r.info.run_name or r.info.run_id[:8])[:24]
        user = r.data.tags.get("mlflow.user", "?")[:12]
        print(f"  {name:24} {user:12} {r.data.params.get('corpus','?'):>7} "
              f"{r.data.params.get('epochs','?'):>7} {loss_s:>10} {r.info.status:>9}")


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
        all_runs, run = find_best_run(args.experiment, client)
        print(f"Found {len(all_runs)} run(s) in '{args.experiment}':\n")
        print_table(all_runs)
        print()
        print(f"==> Best: '{run.info.run_name}'  "
              f"(best_loss={run.data.metrics.get('best_loss'):.4f}, "
              f"epochs={_epochs(run)}, "
              f"user={run.data.tags.get('mlflow.user', '?')})\n")

    # --- download checkpoint + tokenizer ---
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for artifact in (CHECKPOINT_ARTIFACT, TOKENIZER_ARTIFACT):
        print(f"Downloading '{artifact}' -> {DEST_DIR}")
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=artifact,
                dst_path=str(DEST_DIR),
            )
            downloaded.append(local_path)
        except Exception as e:
            print(f"  [WARN] could not download '{artifact}': {type(e).__name__}")

    if not downloaded:
        raise RuntimeError("Nothing was downloaded.")

    # --- sidecar metadata file ---
    meta_path = DEST_DIR / "minigpt_best_INFO.txt"
    meta_path.write_text(
        "Best defender (MiniGPT) checkpoint downloaded from team MLflow / DagsHub\n"
        f"  experiment : {args.experiment}\n"
        f"  run_name   : {run.info.run_name}\n"
        f"  run_id     : {run.info.run_id}\n"
        f"  user       : {run.data.tags.get('mlflow.user', '?')}\n"
        f"  corpus     : {run.data.params.get('corpus', '?')}\n"
        f"  epochs     : {_epochs(run)}\n"
        f"  best_loss  : {run.data.metrics.get('best_loss')}\n"
        f"  artifacts  : {', '.join(Path(p).name for p in downloaded)}\n"
        "\n"
        "NOTE: loss values are only comparable between runs trained on the same\n"
        "grammar/corpus version. Verify this checkpoint matches the current grammar\n"
        "(transition.json) before using it as the project defender.\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Downloaded     : {', '.join(Path(p).name for p in downloaded)}")
    print(f"[OK] Saved metadata : {meta_path}")
    print(f"\nFiles are in: {DEST_DIR}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the best MiniGPT defender checkpoint from MLflow/DagsHub")
    p.add_argument("--experiment", default="MiniGPT",
                   help="MLflow experiment name (default: MiniGPT)")
    p.add_argument("--run-id", default=None, dest="run_id",
                   help="Pin a specific run id instead of auto-selecting the best")
    return p.parse_args()


if __name__ == "__main__":
    main()

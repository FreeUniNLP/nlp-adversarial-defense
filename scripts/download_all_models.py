"""
Download artifacts from every run across all MLflow experiments on the team
DagsHub server.

By default downloads every artifact from every run. Use flags to narrow scope.

Usage:
    python scripts/download_all_models.py
    python scripts/download_all_models.py --experiment MiniGPT
    python scripts/download_all_models.py --best-only
    python scripts/download_all_models.py --experiment AttackerREINFORCE --best-only
    python scripts/download_all_models.py --finished-only
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
    DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME")
    DAGSHUB_TOKEN      = os.getenv("DAGSHUB_TOKEN")

DEST_DIR = PROJECT_ROOT / "data" / "models" / "from_mlflow"


def connect() -> str:
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


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list[str]:
    """Return all leaf artifact paths under `path` for a given run."""
    results = []
    try:
        entries = client.list_artifacts(run_id, path)
    except Exception as e:
        print(f"    [WARN] could not list artifacts at '{path}': {e}")
        return results
    for entry in entries:
        if entry.is_dir:
            results.extend(list_artifacts_recursive(client, run_id, entry.path))
        else:
            results.append(entry.path)
    return results


def pick_best_run(runs: list, experiment_name: str):
    """Pick the single best run using experiment-specific heuristics."""
    # AttackerREINFORCE: highest best_avg_reward, tie-break by most episodes
    if "attacker" in experiment_name.lower() or "reinforce" in experiment_name.lower():
        def score(r):
            reward = r.data.metrics.get("best_avg_reward")
            reward = reward if reward is not None else float("-inf")
            try:
                eps = int(r.data.params.get("episodes", 0))
            except (TypeError, ValueError):
                eps = 0
            return (reward, eps)
        return max(runs, key=score)

    # MiniGPT / defender: lowest best_loss among FINISHED runs, tie-break by most epochs
    finished = [r for r in runs if r.info.status == "FINISHED"]
    candidates = finished if finished else runs

    def score_loss(r):
        loss = r.data.metrics.get("best_loss")
        loss = loss if loss is not None else float("inf")
        try:
            epochs = int(r.data.params.get("epochs", 0))
        except (TypeError, ValueError):
            epochs = 0
        return (loss, -epochs)

    return min(candidates, key=score_loss)


def download_run(client: MlflowClient, run, dest: Path) -> list[Path]:
    """Download all artifacts for a single run. Returns list of local paths."""
    run_id   = run.info.run_id
    run_name = run.info.run_name or run_id[:8]
    run_dir  = dest / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = list_artifacts_recursive(client, run_id)
    if not artifacts:
        print(f"    (no artifacts)")
        return []

    downloaded = []
    for artifact_path in artifacts:
        print(f"    Downloading '{artifact_path}' ...")
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=str(run_dir),
            )
            downloaded.append(Path(local))
        except Exception as e:
            print(f"    [WARN] failed: {type(e).__name__}: {e}")

    # Write a sidecar info file
    info = run_dir / "_RUN_INFO.txt"
    metrics_str = "\n".join(f"  {k}: {v}" for k, v in run.data.metrics.items())
    params_str  = "\n".join(f"  {k}: {v}" for k, v in run.data.params.items())
    info.write_text(
        f"run_name : {run_name}\n"
        f"run_id   : {run_id}\n"
        f"status   : {run.info.status}\n"
        f"user     : {run.data.tags.get('mlflow.user', '?')}\n"
        f"\nMetrics:\n{metrics_str}\n"
        f"\nParams:\n{params_str}\n",
        encoding="utf-8",
    )
    return downloaded


def main() -> None:
    args = parse_args()

    print("Connecting to DagsHub MLflow...")
    uri = connect()
    print(f"  Tracking URI: {uri}\n")

    client = MlflowClient()

    # Collect experiments to process
    if args.experiment:
        exp = mlflow.get_experiment_by_name(args.experiment)
        if exp is None:
            raise RuntimeError(f"Experiment '{args.experiment}' not found.")
        experiments = [exp]
    else:
        experiments = [e for e in client.search_experiments()
                       if e.name != "Default" or not args.skip_default]

    total_files = 0

    for exp in experiments:
        print(f"Experiment: '{exp.name}'")
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
        if not runs:
            print("  (no runs)\n")
            continue

        if args.finished_only:
            runs = [r for r in runs if r.info.status == "FINISHED"]
            if not runs:
                print("  (no finished runs)\n")
                continue

        if args.best_only:
            runs = [pick_best_run(runs, exp.name)]
            print(f"  Best-only mode: selected run '{runs[0].info.run_name or runs[0].info.run_id[:8]}'")

        exp_dest = DEST_DIR / exp.name
        for run in runs:
            run_name = run.info.run_name or run.info.run_id[:8]
            print(f"  Run: '{run_name}' [{run.info.status}]")
            files = download_run(client, run, exp_dest)
            total_files += len(files)
            print(f"    -> {len(files)} file(s) saved to {exp_dest / run_name}")
        print()

    print(f"Done. {total_files} file(s) downloaded to {DEST_DIR}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download all MLflow artifacts from the team DagsHub server")
    p.add_argument("--experiment", default=None,
                   help="Limit to a single experiment name (default: all experiments)")
    p.add_argument("--best-only", action="store_true",
                   help="Only download the best run per experiment instead of all runs")
    p.add_argument("--finished-only", action="store_true",
                   help="Skip runs that did not finish successfully")
    p.add_argument("--skip-default", action="store_true", dest="skip_default",
                   help="Skip the built-in 'Default' experiment")
    return p.parse_args()


if __name__ == "__main__":
    main()

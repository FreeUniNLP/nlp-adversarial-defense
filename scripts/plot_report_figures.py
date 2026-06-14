"""Generate the report figures from the training logs.

Reads the CSV logs in logs/ and writes PNG figures into docs/figures/.
Re-run this whenever the logs change to refresh the graphs in README.md.

    python scripts/plot_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"
FIGS = PROJECT_ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# muted, print-friendly palette
C_LOSS  = "#c0392b"
C_REW   = "#2e86de"
C_VALID = "#27ae60"
C_GRID  = "#dddddd"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 11,
    "axes.grid": True,
    "grid.color": C_GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> None:
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def fig_minigpt_loss() -> None:
    """Defender (MiniGPT) cross-entropy loss over training epochs."""
    df = pd.read_csv(LOGS / "metrics_corpus10000.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["avg_loss"], color=C_LOSS, marker="o", ms=3, lw=1.8)
    final = df["avg_loss"].iloc[-1]
    ax.annotate(f"final = {final:.3f}",
                xy=(df["epoch"].iloc[-1], final),
                xytext=(-90, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MiniGPT defender — training loss (10k corpus, 32 epochs)")
    _save(fig, "fig_minigpt_loss.png")


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(1, window // 5)).mean()


def fig_attacker_reward() -> None:
    """Attacker REINFORCE: reward rises as defender valid-rate collapses."""
    df = pd.read_csv(LOGS / "attacker_episodes_seed42.csv")
    w = 100
    ep = df["episode"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))

    ax1.plot(ep, _rolling(df["reward"], w), color=C_REW, lw=2,
             label="attacker reward")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("attacker reward (rolling 100)", color=C_REW)
    ax1.tick_params(axis="y", labelcolor=C_REW)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.plot(ep, _rolling(df["valid"], w) * 100, color=C_VALID, lw=2,
             label="defender valid %")
    ax2.set_ylabel("defender valid completions %", color=C_VALID)
    ax2.tick_params(axis="y", labelcolor=C_VALID)
    ax2.grid(False)

    ax1.set_title("Attacker REINFORCE — reward up, defender validity down (2k episodes)")
    _save(fig, "fig_attacker_reward.png")


def fig_cotraining() -> None:
    """Co-training arms race: validity oscillates as sides alternate."""
    df = pd.read_csv(LOGS / "adversarial_episodes_seed42.csv")
    w = 200
    ep = df["global_episode"]
    fig, ax1 = plt.subplots(figsize=(8, 4.2))

    ax1.plot(ep, _rolling(df["valid"], w) * 100, color=C_VALID, lw=1.8,
             label="defender valid %")
    ax1.set_xlabel("global episode")
    ax1.set_ylabel("defender valid completions %", color=C_VALID)
    ax1.tick_params(axis="y", labelcolor=C_VALID)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.plot(ep, _rolling(df["attacker_reward"], w), color=C_REW, lw=1.2,
             alpha=0.8, label="attacker reward")
    ax2.set_ylabel("attacker reward", color=C_REW)
    ax2.tick_params(axis="y", labelcolor=C_REW)
    ax2.grid(False)

    # round boundaries
    bounds = df.loc[df["round"].diff() != 0, "global_episode"].tolist()
    for b in bounds[1:]:
        ax1.axvline(b, color="#bbbbbb", ls=":", lw=0.8)

    ax1.set_title("Adversarial co-training — 10 rounds, attacker/defender alternating")
    _save(fig, "fig_cotraining.png")


def fig_defender_rl() -> None:
    """Defender RL fine-tuning: valid-rate climbs and stays high."""
    df = pd.read_csv(LOGS / "defender_rl_episodes_seed42.csv")
    w = 150
    ep = df["episode"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))

    ax1.plot(ep, _rolling(df["valid"], w) * 100, color=C_VALID, lw=2,
             label="defender valid %")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("defender valid completions %", color=C_VALID)
    ax1.tick_params(axis="y", labelcolor=C_VALID)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.plot(ep, _rolling(df["defender_reward"], w), color=C_REW, lw=1.5,
             label="defender reward")
    ax2.set_ylabel("defender reward (= -attacker reward)", color=C_REW)
    ax2.tick_params(axis="y", labelcolor=C_REW)
    ax2.grid(False)

    ax1.set_title("Defender REINFORCE fine-tuning (10k episodes, 30% random prefixes)")
    _save(fig, "fig_defender_rl.png")


def main() -> None:
    print("Generating report figures...")
    fig_minigpt_loss()
    fig_attacker_reward()
    fig_cotraining()
    fig_defender_rl()
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

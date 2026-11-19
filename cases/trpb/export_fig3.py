"""Export per-episode data underlying the TrpB numerical panels in Figure 3."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-panel", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    panel_episodes = pd.read_parquet(args.initial_panel / "episode_scores.parquet")
    panel_rounds = pd.read_parquet(args.initial_panel / "round_scores.parquet")
    archive_episodes = pd.read_parquet(args.archive / "episode_scores.parquet")
    archive_rounds = pd.read_parquet(args.archive / "round_scores.parquet")
    export_panels(archive_episodes, archive_rounds, panel_episodes, panel_rounds, args.out)


def export_panels(archive_episodes, archive_rounds, panel_episodes, panel_rounds, out: Path) -> None:
    full = "full_dast_llm_dual_ast"
    static = "static_dast"
    keys = ["episode_seed", "arm"]
    rounds = keys + ["phase", "round"]
    patch = panel_rounds.loc[panel_rounds["round"].isin([2, 7])].pivot(
        index=["episode_seed", "round"], columns="arm", values="batch_mean_normalized"
    )
    gains = (patch[full] - patch[static]).rename("full_minus_static")
    panels = {
        "Fig3A": archive_episodes[keys + ["ndcg_auc"]],
        "Fig3B": archive_rounds[rounds + ["prefix_ndcg"]],
        "Fig3C": panel_rounds[rounds + ["best_normalized"]],
        "Fig3D": gains.reset_index(),
        "Fig3E": panel_episodes[keys + ["function_retention"]],
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, data in panels.items():
        data.to_csv(out / f"{name}.csv", index=False)
    panel_auc = panel_episodes.groupby("arm").sample_efficiency_auc.mean()
    print(f"Archive Full mean prefix NDCG-AUC: {archive_episodes.loc[archive_episodes.arm == full, 'ndcg_auc'].mean():.6f}")
    print(f"Initial-panel Full minus Static mean AUC: {panel_auc[full] - panel_auc[static]:.6f}")
    print(f"Initial-panel mean paired revision gain: {gains.mean():.6f}")
    print(f"Initial-panel Full mean fitness retention: {panel_episodes.loc[panel_episodes.arm == full,'function_retention'].mean():.6f}")


if __name__ == "__main__":
    main()

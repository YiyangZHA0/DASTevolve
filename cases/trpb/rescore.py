"""Recompute the published panel values from experimental data and recorded query order."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import ast_controller
import modular_objective
import run_archive
import run_initial_panel as panel
from export_fig3 import export_panels
from phase_change import build_phase_b_ast


ROOT = Path(__file__).resolve().parent


def score_experiment(mode, records, rows, phase_asts, classes):
    protocol = json.loads((ROOT / "protocols" / f"{mode}.json").read_text())
    archive_mode = mode == "archive"
    if archive_mode:
        train = [row for row in rows if row["protocol_role"] == protocol["data"]["historical_archive_role"]]
        pool = [row for row in rows if row["protocol_role"] == protocol["data"]["hidden_candidate_role"]]
        seeds = protocol["development_episode_seeds"]
    else:
        pool = [row for row in rows if row["protocol_role"] in protocol["data"]["candidate_roles"]]
        seeds = protocol["protocol_freeze"]["confirmatory_episode_seeds"]

    ids = pd.Index([row["variant_id"] for row in pool])
    activity = np.array([row["target"] for row in pool])
    minimum = float(activity.min())
    evolution = protocol["evolution"]
    shift = evolution["environment_shift_before_round"]
    batch = evolution["batch_size"]
    objective = protocol["objective"]
    round_rows, episode_rows = [], []
    for seed in seeds:
        if not archive_mode:
            initial = panel.initial_panel_indices(pool, seed, protocol["data"]["initial_panel_size"])
        scores = {}
        for phase in ("a", "b"):
            offset = objective[f"phase_{phase}_program_seed_offset"]
            args = (seed + offset, phase_asts[phase], modular_objective, classes,
                    objective["experimental_activity_weight"])
            if archive_mode:
                _, scores[phase], _, _ = run_archive.objective(train, pool, *args)
            else:
                scores[phase], _, _ = panel.objective_arrays(pool, initial, seed, *args)

        for arm in protocol["arms"]:
            ordered = records.loc[(records.episode_seed == seed) & (records.arm == arm)].sort_values(["round", "rank"])
            picks = ids.get_indexer(ordered.variant_id)
            n_rounds = evolution["rounds"]
            assert len(picks) == batch * n_rounds and len(set(picks)) == len(picks) and min(picks) >= 0
            batches = picks.reshape(n_rounds, batch)
            values = []
            for i, selected in enumerate(batches):
                phase = "a" if i + 1 < shift else "b"
                prefix = batches[0 if phase == "a" else shift - 1 : i + 1].ravel()
                target = scores[phase]
                row = dict(episode_seed=seed, arm=arm, phase=phase, round=i + 1)
                if archive_mode:
                    value = run_archive.ndcg(target, prefix.tolist())
                    row["prefix_ndcg"] = value
                else:
                    value = panel.normalized_value(target, float(target[prefix].max()))
                    row["best_normalized"] = value
                    row["batch_mean_normalized"] = np.mean([panel.normalized_value(target, x) for x in target[selected]])
                values.append(value)
                round_rows.append(row)
            episode = dict(episode_seed=seed, arm=arm)
            if archive_mode:
                episode["ndcg_auc"] = np.mean(values)
            else:
                episode["sample_efficiency_auc"] = np.mean(values)
                best_a = activity[batches[:shift - 1]].max()
                best_b = activity[batches[shift - 1:]].max()
                episode["function_retention"] = np.clip((best_b - minimum) / max(best_a - minimum, 1e-12), 0, 1)
            episode_rows.append(episode)
    return pd.DataFrame(episode_rows), pd.DataFrame(round_rows)


def rescore(out: Path) -> None:
    inputs = ROOT / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    for entry in [*manifest["inputs"].values(), manifest["recorded_selections"]]:
        if panel.sha(inputs / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Input checksum mismatch for {entry['path']}")
    allowed = json.loads((inputs / "reference.json").read_text())["allowed_variable_positions_1based"]
    original = ast_controller.initial_ast(json.loads((inputs / "trpb_dual_ast_v1.json").read_text()), allowed)
    revised, _ = build_phase_b_ast(original, ast_controller, allowed)
    rows = pq.read_table(inputs / "visible_observations.parquet",
                         columns=["variant_id", "mutations", "target", "protocol_role"]).to_pylist()
    records = pd.read_parquet(inputs / manifest["recorded_selections"]["path"])
    tables = []
    for mode in ("archive", "initial_panel"):
        tables.extend(score_experiment(mode, records[records.experiment == mode], rows,
                                       {"a": original, "b": revised}, modular_objective.residue_classes()))
    export_panels(*tables, out)

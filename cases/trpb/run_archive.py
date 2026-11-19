#!/usr/bin/env python3
"""Run the archive-backed TrpB benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

import ast_controller as runner
import modular_objective as mechanism
import ridge_gpu as gpu
import trpb_features as features
from phase_change import build_phase_b_ast
from run_initial_panel import independent_functional_ast, sha, value_hash, write_json


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default=ROOT / "protocols" / "archive.json", type=Path)
    p.add_argument("--llm-patch-root", default=ROOT / "patches", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def scale(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    spread = float(train.std())
    if not spread > 0:
        raise ValueError("zero archive scale")
    return (values - float(train.mean())) / spread


def ndcg(values: np.ndarray, ordered_indices: Sequence[int]) -> float:
    if not ordered_indices:
        return 0.0
    relevance = values - float(values.min())
    gains = np.expm1(np.log(2.0) * np.clip(relevance, 0.0, 40.0))
    k = len(ordered_indices)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[np.asarray(ordered_indices, dtype=int)] / discounts))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def enrichment(values: np.ndarray, indices: Sequence[int], quantile: float) -> float:
    if not indices:
        return 0.0
    threshold = float(np.quantile(values, quantile))
    prevalence = float(np.mean(values >= threshold))
    selected = float(np.mean(values[np.asarray(indices, dtype=int)] >= threshold))
    return selected / prevalence if prevalence > 0 else 0.0


def objective(
    archive: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
    program_seed: int, ast: Mapping[str, Any], mechanism: Any,
    classes: Mapping[str, str], activity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive_real = np.asarray([float(row["target"]) for row in archive], dtype=np.float64)
    candidate_real = np.asarray([float(row["target"]) for row in candidates], dtype=np.float64)
    archive_program = np.asarray([
        mechanism.program_value(row, program_seed, ast, classes) for row in archive
    ], dtype=np.float64)
    candidate_program = np.asarray([
        mechanism.program_value(row, program_seed, ast, classes) for row in candidates
    ], dtype=np.float64)
    archive_composite = activity_weight * scale(archive_real, archive_real) + (1.0 - activity_weight) * scale(archive_program, archive_program)
    candidate_composite = activity_weight * scale(archive_real, candidate_real) + (1.0 - activity_weight) * scale(archive_program, candidate_program)
    return archive_composite, candidate_composite, archive_real, candidate_real


def main() -> int:
    args = arguments()
    if not args.device.startswith("cuda"):
        raise ValueError("CPU fitting is forbidden")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, mode=0o700)
    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != "dastbench.trpb.closed_loop_archive_protocol.dev.v1":
        raise ValueError("development archive protocol required")
    seeds = [int(x) for x in protocol["development_episode_seeds"]]
    source_path = ROOT / "inputs" / "manifest.json"
    source = json.loads(source_path.read_text())
    if source.get("official_test_labels_read") is not False:
        raise ValueError("source provenance invalid")
    inputs = source["inputs"]
    visible = (source_path.parent / inputs["visible_observations.parquet"]["path"])
    reference_path = (source_path.parent / inputs["reference.json"]["path"])
    prior_path = (source_path.parent / inputs["masked_marginals.parquet"]["path"])
    ast_path = (source_path.parent / inputs["trpb_dual_ast_v1.json"]["path"])

    for path in (visible, reference_path, prior_path, ast_path):
        if sha(path) != inputs[path.name]["sha256"]:
            raise ValueError(f"source input hash mismatch: {path.name}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    probe = torch.arange(4, device=device).square()
    torch.cuda.synchronize(device)
    if probe.cpu().tolist() != [0, 1, 4, 9]:
        raise RuntimeError("CUDA probe failed")
    del probe
    torch.cuda.reset_peak_memory_stats(device)

    allowed = tuple(int(x) for x in json.loads(reference_path.read_text())["allowed_variable_positions_1based"])
    reference_ast = runner.initial_ast(json.loads(ast_path.read_text()), allowed)
    starting_ast, starting_corruption = runner.corrupt_ast(reference_ast, 1.0, 2026080501)
    phase_b_ast, phase_b_locality = build_phase_b_ast(reference_ast, runner, allowed)
    target_asts = {"a": reference_ast, "b": phase_b_ast}
    patch_root = args.llm_patch_root.resolve()
    manifest = json.loads((patch_root / "PATCH_MANIFEST.json").read_text())
    receipts = json.loads((patch_root / "LLM_RECEIPTS.json").read_text())["calls"]
    if manifest.get("status") != "REAL_LLM_DUAL_AST_PATCHES_VALIDATED" or len(receipts) != 8:
        raise ValueError("real LLM patches invalid")
    llm_asts = {
        "a": runner.normalize_ast(json.loads((patch_root / "PHASE_A_AST.json").read_text()), allowed),
        "b": runner.normalize_ast(json.loads((patch_root / "PHASE_B_AST.json").read_text()), allowed),
    }
    if any(value_hash(llm_asts[p]) != value_hash(target_asts[p]) for p in ("a", "b")):
        raise ValueError("LLM patch mismatch")
    prior = gpu._load_prior(prior_path, allowed)
    rows = pq.read_table(visible, columns=["variant_id", "mutations", "target", "protocol_role"]).to_pylist()
    archive = [row for row in rows if row["protocol_role"] == protocol["data"]["historical_archive_role"]]
    candidates = [row for row in rows if row["protocol_role"] == protocol["data"]["hidden_candidate_role"]]
    candidate_ids = np.asarray([str(row["variant_id"]) for row in candidates], dtype=str)
    candidate_real = np.asarray([float(row["target"]) for row in candidates], dtype=np.float64)
    classes = mechanism.residue_classes()
    rounds = int(protocol["evolution"]["rounds"])
    batch = int(protocol["evolution"]["batch_size"])
    shift = int(protocol["evolution"]["environment_shift_before_round"])
    alpha = float(protocol["gpu"]["ridge_alpha"])
    activity_weight = float(protocol["objective"]["experimental_activity_weight"])
    quantile = float(protocol["metric"]["success_threshold_quantile"])
    arms = list(protocol["arms"])
    competitors = list(protocol["primary_competitors"])
    model_arms = {
        "full_dast_llm_dual_ast": "full_dast", "static_dast": "full_dast",
        "single_ast_llm": "full_dast", "flat_llm_memory": "flat_no_ast",
        "bayesian_ridge_onehot": "onehot_ridge", "oracle_dast_upper_bound": "full_dast",
    }
    phase_aware = {"full_dast_llm_dual_ast", "flat_llm_memory", "bayesian_ridge_onehot", "oracle_dast_upper_bound"}
    score_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    gpu_receipts: list[dict[str, Any]] = []
    fit_count = 0
    write_json(out / "PRECHECK.json", {
        "status": "READY", "development_only": True, "official_test_labels_read": False,
        "archive_rows": len(archive), "candidate_rows": len(candidates),
        "starting_corruption": starting_corruption, "phase_b_locality": phase_b_locality,
        "cuda": torch.cuda.get_device_name(device), "protocol_sha256": sha(protocol_path),
    })

    for episode_number, seed in enumerate(seeds, 1):
        print(f"[development episode {episode_number}/{len(seeds)}] seed={seed}", flush=True)
        archive_targets: dict[str, np.ndarray] = {}
        candidate_targets: dict[str, np.ndarray] = {}
        for phase, offset in (("a", "phase_a_program_seed_offset"), ("b", "phase_b_program_seed_offset")):
            archive_target, candidate_target, _archive_real, _candidate_real = objective(
                archive, candidates, seed + int(protocol["objective"][offset]), target_asts[phase],
                mechanism, classes, activity_weight,
            )
            archive_targets[phase] = archive_target
            candidate_targets[phase] = candidate_target
        audit_training = [{**row, "target": float(value)} for row, value in zip(archive, archive_targets["a"], strict=True)]
        for severity in protocol.get("corruption_levels", [0.0, 0.25, 0.5, 0.75, 1.0]):
            corrupted, achieved = runner.corrupt_ast(reference_ast, float(severity), 2026080800 + seed * 100 + int(float(severity) * 100))
            evaluator = runner.Evaluator(torch=torch, feature_module=features, gpu_module=gpu, prior=prior, allowed=allowed, device=device, alpha=alpha)
            prediction, fit = evaluator.predict(audit_training, candidates, "full_dast", corrupted)
            fit_count += evaluator.fit_count
            order = np.lexsort((candidate_ids, -prediction))[:192]
            corruption_rows.append({"episode_seed": seed, "severity": float(severity), "achieved": float(achieved), "ndcg": ndcg(candidate_targets["a"], order.tolist())})
            gpu_receipts.append({"episode_seed": seed, "round": 0, "arm": f"corruption_{severity}", "fit": fit})
        histories = {arm: [] for arm in arms}
        phase_queries = {arm: {"a": [], "b": []} for arm in arms}
        auc = {arm: [] for arm in arms}
        random_orders = {phase: np.random.default_rng(seed + (0 if phase == "a" else 10000)).permutation(len(candidates)).tolist() for phase in ("a", "b")}
        for round_number in range(1, rounds + 1):
            phase = "a" if round_number < shift else "b"
            hidden_target = candidate_targets[phase]
            threshold = float(np.quantile(hidden_target, quantile))
            for arm in arms:
                used = {int(item["candidate_index"]) for item in histories[arm]}
                available = [index for index in range(len(candidates)) if index not in used]
                ast_hash = None
                if arm == "random_search":
                    selected = [index for index in random_orders[phase] if index not in used][:batch]
                else:
                    if arm == "full_dast_llm_dual_ast":
                        ast = starting_ast if round_number == 1 else llm_asts[phase]
                    elif arm == "static_dast":
                        ast = starting_ast
                    elif arm == "single_ast_llm":
                        ast = independent_functional_ast(llm_asts[phase], runner, allowed)
                    elif arm == "oracle_dast_upper_bound":
                        ast = target_asts[phase]
                    else:
                        ast = starting_ast
                    if arm in phase_aware:
                        base_targets = archive_targets[phase]
                        visible_history = [item for item in histories[arm] if item["phase"] == phase]
                    else:
                        base_targets = archive_targets["a"]
                        visible_history = histories[arm]
                    training = [{**row, "target": float(value)} for row, value in zip(archive, base_targets, strict=True)]
                    training.extend({**candidates[int(item["candidate_index"])], "target": float(item["target"])} for item in visible_history)
                    evaluator = runner.Evaluator(torch=torch, feature_module=features, gpu_module=gpu, prior=prior, allowed=allowed, device=device, alpha=alpha)
                    prediction, fit = evaluator.predict(training, [candidates[index] for index in available], model_arms[arm], ast)
                    fit_count += evaluator.fit_count
                    available_ids = candidate_ids[np.asarray(available, dtype=int)]
                    rank_order = np.lexsort((available_ids, -prediction))
                    selected = [available[int(index)] for index in rank_order[:batch]]
                    ast_hash = value_hash(ast)
                    gpu_receipts.append({"episode_seed": seed, "round": round_number, "arm": arm, "fit": fit})
                for index in selected:
                    value = float(hidden_target[index])
                    histories[arm].append({"candidate_index": index, "phase": phase, "target": value})
                    phase_queries[arm][phase].append(index)
                    query_rows.append({"episode_seed": seed, "round": round_number, "phase": phase, "arm": arm, "variant_id": str(candidate_ids[index]), "objective": value, "experimental_activity": float(candidate_real[index]), "ast_value_hash": ast_hash})
                ordered = phase_queries[arm][phase]
                prefix_ndcg = ndcg(hidden_target, ordered)
                auc[arm].append(prefix_ndcg)
                selected_values = hidden_target[np.asarray(ordered, dtype=int)]
                score_rows.append({
                    "episode_seed": seed, "round": round_number, "phase": phase, "arm": arm,
                    "phase_queries": len(ordered), "prefix_ndcg": prefix_ndcg,
                    "enrichment": enrichment(hidden_target, ordered, quantile),
                    "success": bool(float(selected_values.max()) >= threshold),
                    "simple_regret": float(hidden_target.max() - selected_values.max()),
                    "batch_mean": float(hidden_target[np.asarray(selected, dtype=int)].mean()),
                })
        episode_scores = {}
        seed_scores = pd.DataFrame([row for row in score_rows if row["episode_seed"] == seed])
        for arm in arms:
            arm_scores = seed_scores[seed_scores.arm == arm]
            costs = {}
            for phase in ("a", "b"):
                hit = arm_scores[(arm_scores.phase == phase) & arm_scores.success]
                costs[phase] = int(hit.iloc[0].phase_queries) if len(hit) else 112
            phase_a = phase_queries[arm]["a"]
            phase_b = phase_queries[arm]["b"]
            low = float(candidate_real.min())
            best_a = max([float(candidate_real[index]) for index in phase_a], default=low)
            best_b = max([float(candidate_real[index]) for index in phase_b], default=low)
            retention = float(np.clip((best_b - low) / max(best_a - low, 1e-12), 0.0, 1.0))
            episode_scores[arm] = {"ndcg_auc": float(np.mean(auc[arm])), "cost_to_hit": costs["a"], "adaptation_budget": costs["b"], "function_retention": retention}
            episode_rows.append({"episode_seed": seed, "arm": arm, **episode_scores[arm]})
        strongest = max(competitors, key=lambda arm: episode_scores[arm]["ndcg_auc"])
        full_delta = episode_scores["full_dast_llm_dual_ast"]["ndcg_auc"] - episode_scores[strongest]["ndcg_auc"]
        for row in episode_rows[-len(arms):]:
            row["strongest_non_oracle"] = strongest
            row["full_delta_vs_strongest"] = full_delta
        write_json(out / "PROGRESS.json", {"completed": episode_number, "total": len(seeds), "gpu_fits": fit_count, "official_test_labels_read": False})

    score_path, query_path, episode_path, corruption_path = (out / "round_scores.parquet", out / "query_trajectories.parquet", out / "episode_scores.parquet", out / "corruption_curve.parquet")
    pq.write_table(pa.Table.from_pylist(score_rows), score_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(query_rows), query_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(corruption_rows), corruption_path, compression="zstd")
    write_json(out / "GPU_FIT_RECEIPTS.json", {"fits": gpu_receipts})
    episodes_frame = pd.DataFrame(episode_rows)
    means = episodes_frame.groupby("arm").ndcg_auc.mean()
    full = episodes_frame[episodes_frame.arm == "full_dast_llm_dual_ast"].set_index("episode_seed")
    competitor_auc = episodes_frame[episodes_frame.arm.isin(competitors)].pivot(index="episode_seed", columns="arm", values="ndcg_auc")
    strongest = competitor_auc.max(axis=1)
    corruption_frame = pd.DataFrame(corruption_rows).groupby("severity")[["achieved", "ndcg"]].mean()
    rho = float(spearmanr(corruption_frame.achieved, corruption_frame.ndcg).statistic)
    conditions = protocol["diagnostic_success_conditions"]
    checks = {
        "oracle_beats_random": float(means["oracle_dast_upper_bound"]) > float(means["random_search"]),
        "full_beats_static": float(means["full_dast_llm_dual_ast"]) > float(means["static_dast"]),
        "full_beats_flat": float(means["full_dast_llm_dual_ast"]) > float(means["flat_llm_memory"]),
        "episode_wins": int((full.ndcg_auc - strongest > 0).sum()) >= int(conditions["full_episode_wins_vs_strongest_out_of_4"]),
        "corruption_curve": rho <= float(conditions["corruption_performance_spearman_at_most"]),
    }
    result = {
        "schema_version": "dastbench.trpb.closed_loop_archive_development_run.v1",
        "status": "DEVELOPMENT_PASSED" if all(checks.values()) else "DEVELOPMENT_FAILED",
        "official_test_labels_read": False, "checks": checks,
        "arm_mean_ndcg_auc": {str(k): float(v) for k, v in means.items()},
        "episode_wins_vs_strongest": int((full.ndcg_auc - strongest > 0).sum()),
        "corruption_spearman": rho, "gpu_fits": fit_count,
        "cuda": {"device": torch.cuda.get_device_name(device), "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))},
        "runtime_seconds": time.perf_counter() - started,
        "inputs": {"protocol": sha(protocol_path), "input_manifest": sha(source_path), "runner": sha(Path(__file__)), "llm_receipts": sha(patch_root / "LLM_RECEIPTS.json")},
        "outputs": {path.name: sha(path) for path in (score_path, query_path, episode_path, corruption_path, out / "GPU_FIT_RECEIPTS.json")},
    }
    write_json(out / "RUN.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "DEVELOPMENT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

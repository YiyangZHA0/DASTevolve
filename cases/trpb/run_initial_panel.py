#!/usr/bin/env python3
"""Run the TrpB benchmark from a shared initial panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
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


ROOT = Path(__file__).resolve().parent
# Frozen hash namespace for the published panel and random-control permutations.
ORDERING_REVISION = 3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=ROOT / "protocols" / "initial_panel.json", type=Path)
    parser.add_argument("--llm-patch-root", default=ROOT / "patches", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deterministic_order(rows: Sequence[Mapping[str, Any]], seed: int, namespace: str) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            f"{namespace}:{seed}:{rows[index]['variant_id']}".encode()
        ).digest(),
    )


def initial_panel_indices(rows, seed: int, size: int) -> list[int]:
    return deterministic_order(rows, seed, f"trpb-v{ORDERING_REVISION}-initial")[:size]


def scale_from(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    spread = float(reference.std())
    if not spread > 0:
        raise ValueError("zero initial-panel scale")
    return (values - float(reference.mean())) / spread


def bootstrap_lower(values: np.ndarray, draws: int = 20000, seed: int = 20260805) -> float:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025))


def independent_functional_ast(ast: Mapping[str, Any], runner: Any, allowed: Sequence[int]) -> dict[str, Any]:
    structural = [
        {"id": str(node["id"]), "positions_1based": [int(x) for x in node["positions_1based"]]}
        for node in ast["structural_ast"]["nodes"]
    ]
    functional = [
        {"id": f"single_{node['id']}", "mapped_structural_nodes": [str(node["id"])]}
        for node in structural
    ]
    return runner.normalize_ast(
        {"structural_nodes": structural, "functional_nodes": functional}, allowed
    )


def objective_arrays(
    rows: Sequence[Mapping[str, Any]], initial_indices: Sequence[int], episode_seed: int,
    program_seed: int, ast: Mapping[str, Any], mechanism: Any, classes: Mapping[str, str],
    activity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    program = np.asarray([
        mechanism.program_value(row, program_seed, ast, classes) for row in rows
    ], dtype=np.float64)
    initial = np.asarray(initial_indices, dtype=int)
    normalized_real = scale_from(real[initial], real)
    normalized_program = scale_from(program[initial], program)
    composite = activity_weight * normalized_real + (1.0 - activity_weight) * normalized_program
    if not np.isfinite(composite).all():
        raise RuntimeError(f"non-finite objective for episode {episode_seed}")
    return composite, normalized_real, normalized_program


def normalized_value(values: np.ndarray, value: float) -> float:
    low, high = float(values.min()), float(values.max())
    return 0.0 if not high > low else float((value - low) / (high - low))


def ndcg(values: np.ndarray, queried: Sequence[int]) -> float:
    if not queried:
        return 0.0
    idx = np.asarray(queried, dtype=int)
    relevance = values - float(values.min())
    selected = relevance[idx]
    discount = np.log2(np.arange(2, len(idx) + 2))
    dcg = float(np.sum(np.expm1(np.log(2.0) * selected) / discount))
    ideal = np.sort(relevance)[::-1][:len(idx)]
    idcg = float(np.sum(np.expm1(np.log(2.0) * ideal) / discount))
    return dcg / idcg if idcg > 0 else 0.0


def enrichment(values: np.ndarray, queried: Sequence[int], quantile: float = 0.95) -> float:
    if not queried:
        return 0.0
    cutoff = float(np.quantile(values, quantile))
    prevalence = float(np.mean(values >= cutoff))
    selected = float(np.mean(values[np.asarray(queried, dtype=int)] >= cutoff))
    return selected / prevalence if prevalence > 0 else 0.0


def main() -> int:
    args = arguments()
    if not args.device.startswith("cuda"):
        raise ValueError("CPU fitting is forbidden")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, mode=0o700)
    started = time.perf_counter()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != "dastbench.trpb.closed_loop_evolution_protocol.v2":
        raise ValueError("formal run requires frozen v2 protocol")
    if protocol.get("official_test_labels_may_be_read") is not False:
        raise ValueError("official labels are not prohibited")
    seeds = [int(x) for x in protocol["protocol_freeze"]["confirmatory_episode_seeds"]]
    if seeds != list(range(80, 96)):
        raise ValueError("confirmatory seeds must be untouched 80-95")

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
        raise RuntimeError("CUDA unavailable; no CPU fallback")
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
    phase_b_reference_ast, phase_b_change = build_phase_b_ast(reference_ast, runner, allowed)
    phase_ast = {"a": reference_ast, "b": phase_b_reference_ast}

    patch_root = args.llm_patch_root.resolve()
    patch_manifest_path = patch_root / "PATCH_MANIFEST.json"
    receipt_path = patch_root / "LLM_RECEIPTS.json"
    manifest = json.loads(patch_manifest_path.read_text())
    receipts = json.loads(receipt_path.read_text())
    if manifest.get("status") != "REAL_LLM_DUAL_AST_PATCHES_VALIDATED":
        raise ValueError("LLM patch manifest is not validated")
    calls = receipts.get("calls", receipts) if isinstance(receipts, Mapping) else receipts
    if len(calls) < 8 or any(call.get("model") != "gpt-5.6-sol" for call in calls):
        raise ValueError("at least eight real gpt-5.6-sol receipts are required")
    if any(call.get("official_test_labels_read") is not False for call in calls):
        raise ValueError("LLM receipt provenance invalid")
    llm_ast = {
        "a": runner.normalize_ast(json.loads((patch_root / "PHASE_A_AST.json").read_text()), allowed),
        "b": runner.normalize_ast(json.loads((patch_root / "PHASE_B_AST.json").read_text()), allowed),
    }
    for phase in ("a", "b"):
        if value_hash(llm_ast[phase]) != value_hash(phase_ast[phase]):
            raise ValueError(f"LLM {phase} patch does not match evaluator-confirmed AST")
    if manifest.get("confirmatory_seeds_absent_from_prompts") is not True:
        raise ValueError("confirmatory seed prompt exclusion not proven")

    prior = gpu._load_prior(prior_path, allowed)
    roles = set(protocol["data"]["candidate_roles"])
    rows = [
        row for row in pq.read_table(
            visible, columns=["variant_id", "mutations", "target", "protocol_role"]
        ).to_pylist() if row["protocol_role"] in roles
    ]
    ids = np.asarray([str(row["variant_id"]) for row in rows], dtype=str)
    real_target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    classes = mechanism.residue_classes()
    rounds = int(protocol["evolution"]["rounds"])
    batch_size = int(protocol["evolution"]["batch_size"])
    shift_round = int(protocol["evolution"]["environment_shift_before_round"])
    initial_size = int(protocol["data"]["initial_panel_size"])
    activity_weight = float(protocol["objective"]["experimental_activity_weight"])
    alpha = float(protocol["gpu"]["ridge_alpha"])
    modeled_arms = {
        "full_dast_llm_dual_ast": "full_dast",
        "static_dast": "full_dast",
        "single_ast_llm": "full_dast",
        "flat_llm_memory": "flat_no_ast",
        "bayesian_ridge_onehot": "onehot_ridge",
        "oracle_dast_upper_bound": "full_dast",
    }
    all_arms = list(protocol["arms"])
    phase_aware = {"full_dast_llm_dual_ast", "flat_llm_memory", "bayesian_ridge_onehot", "oracle_dast_upper_bound"}
    score_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    gpu_receipts: list[dict[str, Any]] = []
    fit_count = 0

    write_json(out / "PRECHECK.json", {
        "status": "READY", "development_only": True, "official_test_labels_read": False,
        "protocol_sha256": sha(protocol_path), "llm_patch_manifest_sha256": sha(patch_manifest_path),
        "llm_receipts_sha256": sha(receipt_path), "starting_corruption": starting_corruption,
        "phase_b_ast_change": phase_b_change, "cuda": torch.cuda.get_device_name(device),
        "policy_label_rule": protocol["data"]["policy_label_access"],
    })

    for episode_index, episode_seed in enumerate(seeds, 1):
        print(f"[episode {episode_index}/{len(seeds)}] seed={episode_seed}", flush=True)
        initial = initial_panel_indices(rows, episode_seed, initial_size)
        initial_set = set(initial)
        objectives = {}
        normalized_real = {}
        for phase, offset_key in (("a", "phase_a_program_seed_offset"), ("b", "phase_b_program_seed_offset")):
            composite, real_component, _program_component = objective_arrays(
                rows, initial, episode_seed,
                episode_seed + int(protocol["objective"][offset_key]), phase_ast[phase],
                mechanism, classes, activity_weight,
            )
            objectives[phase] = composite
            normalized_real[phase] = real_component
        audit_candidates = [index for index in range(len(rows)) if index not in initial_set]
        audit_training = [
            {**rows[index], "target": float(objectives["a"][index])} for index in initial
        ]
        audit_rows = [rows[index] for index in audit_candidates]
        audit_ids = ids[np.asarray(audit_candidates, dtype=int)]
        audit_budget = min(192, len(audit_candidates))
        for severity in protocol["causal_audits"]["corruption_levels"]:
            corrupted_ast, achieved = runner.corrupt_ast(
                reference_ast, float(severity),
                2026080600 + episode_seed * 100 + int(float(severity) * 100),
            )
            evaluator = runner.Evaluator(
                torch=torch, feature_module=features, gpu_module=gpu, prior=prior,
                allowed=allowed, device=device, alpha=alpha,
            )
            predictions, fit_receipt = evaluator.predict(
                audit_training, audit_rows, "full_dast", corrupted_ast
            )
            fit_count += evaluator.fit_count
            ranking = np.lexsort((audit_ids, -predictions))[:audit_budget]
            selected_global = [audit_candidates[int(index)] for index in ranking]
            audit_score = ndcg(objectives["a"], selected_global)
            corruption_rows.append({
                "episode_seed": episode_seed, "requested_severity": float(severity),
                "achieved_severity": float(achieved), "ndcg_at_192": float(audit_score),
            })
            gpu_receipts.append({
                "episode_seed": episode_seed, "round": 0, "phase": "a",
                "arm": f"corruption_{float(severity):.2f}", "training_rows": len(audit_training),
                "candidate_rows": len(audit_rows), "ast_value_hash": value_hash(corrupted_ast),
                "fit": fit_receipt,
            })
        histories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in all_arms}
        queried: dict[str, list[int]] = {arm: [] for arm in all_arms}
        phase_queried: dict[str, dict[str, list[int]]] = {
            arm: {"a": [], "b": []} for arm in all_arms
        }
        random_orders = {
            phase: deterministic_order(rows, episode_seed, f"trpb-v{ORDERING_REVISION}-random-{phase}")
            for phase in ("a", "b")
        }
        phase_best: dict[str, dict[str, float]] = {
            arm: {"a": -math.inf, "b": -math.inf} for arm in all_arms
        }
        auc_values: dict[str, list[float]] = {arm: [] for arm in all_arms}

        for round_number in range(1, rounds + 1):
            phase = "a" if round_number < shift_round else "b"
            objective = objectives[phase]
            threshold = float(np.quantile(objective, 0.95))
            for arm in all_arms:
                unavailable = initial_set | set(queried[arm])
                available = [index for index in range(len(rows)) if index not in unavailable]
                if arm == "random_search":
                    selected = [index for index in random_orders[phase] if index not in unavailable][:batch_size]
                    ast_hash = None
                    fit_receipt = None
                else:
                    if arm == "full_dast_llm_dual_ast":
                        ast = starting_ast if round_number == 1 else llm_ast[phase]
                    elif arm == "static_dast":
                        ast = starting_ast
                    elif arm == "single_ast_llm":
                        ast = independent_functional_ast(llm_ast[phase], runner, allowed)
                    elif arm == "oracle_dast_upper_bound":
                        ast = phase_ast[phase]
                    else:
                        ast = starting_ast
                    if arm in phase_aware:
                        training_indices = initial + phase_queried[arm][phase]
                        training_targets = [float(objective[index]) for index in training_indices]
                    else:
                        training_indices = initial + [int(item["row_index"]) for item in histories[arm]]
                        training_targets = [float(objectives["a"][index]) for index in initial] + [
                            float(item["observed_target"]) for item in histories[arm]
                        ]
                    training = [
                        {**rows[index], "target": target}
                        for index, target in zip(training_indices, training_targets, strict=True)
                    ]
                    candidates = [rows[index] for index in available]
                    evaluator = runner.Evaluator(
                        torch=torch, feature_module=features, gpu_module=gpu, prior=prior,
                        allowed=allowed, device=device, alpha=alpha,
                    )
                    predictions, fit_receipt = evaluator.predict(training, candidates, modeled_arms[arm], ast)
                    fit_count += evaluator.fit_count
                    candidate_ids = ids[np.asarray(available, dtype=int)]
                    order = np.lexsort((candidate_ids, -predictions))
                    selected = [available[int(index)] for index in order[:batch_size]]
                    ast_hash = value_hash(ast)
                    gpu_receipts.append({
                        "episode_seed": episode_seed, "round": round_number, "phase": phase,
                        "arm": arm, "training_rows": len(training), "candidate_rows": len(candidates),
                        "ast_value_hash": ast_hash, "fit": fit_receipt,
                    })
                batch_values = objective[np.asarray(selected, dtype=int)]
                for index, value in zip(selected, batch_values, strict=True):
                    histories[arm].append({"row_index": index, "phase": phase, "observed_target": float(value)})
                    queried[arm].append(index)
                    phase_queried[arm][phase].append(index)
                    query_rows.append({
                        "episode_seed": episode_seed, "round": round_number, "phase": phase,
                        "arm": arm, "variant_id": str(ids[index]), "objective": float(value),
                        "experimental_activity": float(real_target[index]), "ast_value_hash": ast_hash,
                    })
                phase_best[arm][phase] = max(phase_best[arm][phase], float(batch_values.max()))
                normalized_best = normalized_value(objective, phase_best[arm][phase])
                auc_values[arm].append(normalized_best)
                phase_indices = phase_queried[arm][phase]
                score_rows.append({
                    "episode_seed": episode_seed, "round": round_number, "phase": phase, "arm": arm,
                    "cumulative_queries": len(queried[arm]), "phase_queries": len(phase_indices),
                    "best_normalized": normalized_best,
                    "success": bool(phase_best[arm][phase] >= threshold),
                    "simple_regret": 1.0 - normalized_best,
                    "ndcg": ndcg(objective, phase_indices),
                    "enrichment": enrichment(objective, phase_indices),
                    "batch_mean_normalized": float(np.mean([
                        normalized_value(objective, float(value)) for value in batch_values
                    ])),
                })

        episode_scores = {}
        for arm in all_arms:
            arm_frame = pd.DataFrame([row for row in score_rows if row["episode_seed"] == episode_seed and row["arm"] == arm])
            phase_costs = {}
            for phase in ("a", "b"):
                phase_frame = arm_frame[arm_frame.phase == phase]
                hit = phase_frame[phase_frame.success]
                phase_costs[phase] = int(hit.iloc[0].phase_queries) if len(hit) else batch_size * 7
            phase_a_indices = phase_queried[arm]["a"]
            phase_b_indices = phase_queried[arm]["b"]
            minimum_activity = float(real_target.min())
            best_a = max([float(real_target[index]) for index in phase_a_indices], default=minimum_activity)
            best_b = max([float(real_target[index]) for index in phase_b_indices], default=minimum_activity)
            retention = float(np.clip((best_b - minimum_activity) / max(best_a - minimum_activity, 1e-12), 0.0, 1.0))
            episode_scores[arm] = {
                "sample_efficiency_auc": float(np.mean(auc_values[arm])),
                "cost_to_hit": phase_costs["a"],
                "adaptation_budget": phase_costs["b"],
                "function_retention": retention,
                "final_experimental_activity": best_b,
            }
            episode_rows.append({"episode_seed": episode_seed, "arm": arm, **episode_scores[arm]})
        competitors = protocol["fairness"]["primary_competitors"]
        strongest_arm = max(competitors, key=lambda arm: episode_scores[arm]["sample_efficiency_auc"])
        full = episode_scores["full_dast_llm_dual_ast"]
        strongest = episode_scores[strongest_arm]
        for row in episode_rows[-len(all_arms):]:
            row["strongest_non_oracle_arm"] = strongest_arm
            row["full_delta_vs_strongest"] = full["sample_efficiency_auc"] - strongest["sample_efficiency_auc"]
        write_json(out / "PROGRESS.json", {
            "status": "RUNNING", "completed_episodes": episode_index, "total_episodes": len(seeds),
            "last_episode_seed": episode_seed, "gpu_fits": fit_count,
            "official_test_labels_read": False,
        })

    scores_path = out / "round_scores.parquet"
    queries_path = out / "query_trajectories.parquet"
    episodes_path = out / "episode_scores.parquet"
    corruption_path = out / "corruption_curve.parquet"
    receipts_path = out / "GPU_FIT_RECEIPTS.json"
    pq.write_table(pa.Table.from_pylist(score_rows), scores_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(query_rows), queries_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(corruption_rows), corruption_path, compression="zstd")
    write_json(receipts_path, {"fits": gpu_receipts})
    frame = pd.DataFrame(episode_rows)
    full = frame[frame.arm == "full_dast_llm_dual_ast"].set_index("episode_seed")
    comparator_frame = frame[frame.arm.isin(protocol["fairness"]["primary_competitors"])]
    strongest_auc = comparator_frame.pivot(index="episode_seed", columns="arm", values="sample_efficiency_auc").max(axis=1)
    strongest_cost = comparator_frame.pivot(index="episode_seed", columns="arm", values="cost_to_hit").min(axis=1)
    strongest_adapt = comparator_frame.pivot(index="episode_seed", columns="arm", values="adaptation_budget").min(axis=1)
    delta = full.sample_efficiency_auc - strongest_auc
    relative_gain = delta / strongest_auc.abs().clip(lower=1e-12)
    round_frame = pd.DataFrame(score_rows)
    patch_rounds = set(int(x) for x in protocol["dual_ast"]["llm_patch_rounds"])
    patch_gains = []
    for episode_seed in seeds:
        for patch_round in patch_rounds:
            selected = round_frame[(round_frame.episode_seed == episode_seed) & (round_frame["round"] == patch_round)]
            full_batch = float(selected[selected.arm == "full_dast_llm_dual_ast"].batch_mean_normalized.iloc[0])
            static_batch = float(selected[selected.arm == "static_dast"].batch_mean_normalized.iloc[0])
            patch_gains.append(full_batch - static_batch)
    corruption_frame = pd.DataFrame(corruption_rows)
    corruption_means = corruption_frame.groupby("requested_severity")[["achieved_severity", "ndcg_at_192"]].mean()
    corruption_rho = float(spearmanr(
        corruption_means.achieved_severity.to_numpy(),
        corruption_means.ndcg_at_192.to_numpy(),
    ).statistic)
    static_auc = frame[frame.arm == "static_dast"].set_index("episode_seed").sample_efficiency_auc
    flat_auc = frame[frame.arm == "flat_llm_memory"].set_index("episode_seed").sample_efficiency_auc
    acceptance = protocol["acceptance"]
    gate = {
        "episode_wins_vs_strongest": int((delta > 0).sum()),
        "mean_auc_delta_vs_strongest": float(delta.mean()),
        "mean_auc_relative_gain": float(relative_gain.mean()),
        "paired_bootstrap_95pct_lower_bound": bootstrap_lower(delta.to_numpy()),
        "mean_cost_to_hit_ratio": float((full.cost_to_hit / strongest_cost.clip(lower=1)).mean()),
        "mean_adaptation_budget_ratio": float((full.adaptation_budget / strongest_adapt.clip(lower=1)).mean()),
        "mean_function_retention": float(full.function_retention.mean()),
        "mean_post_patch_next_batch_gain": float(np.mean(patch_gains)),
        "corruption_severity_score_spearman": corruption_rho,
        "mean_auc_delta_vs_static_dast": float((full.sample_efficiency_auc - static_auc).mean()),
        "mean_auc_delta_vs_flat_llm_memory": float((full.sample_efficiency_auc - flat_auc).mean()),
        "phase_b_patch_locality": phase_b_change,
    }
    checks = {
        "episode_wins": gate["episode_wins_vs_strongest"] >= int(acceptance["minimum_episode_wins_vs_strongest_non_oracle_out_of_16"]),
        "auc_relative_gain": gate["mean_auc_relative_gain"] >= float(acceptance["minimum_mean_sample_efficiency_auc_relative_gain"]),
        "cost_to_hit": gate["mean_cost_to_hit_ratio"] <= float(acceptance["maximum_mean_cost_to_hit_ratio"]),
        "adaptation_budget": gate["mean_adaptation_budget_ratio"] <= float(acceptance["maximum_mean_adaptation_budget_ratio"]),
        "post_patch_gain": gate["mean_post_patch_next_batch_gain"] >= float(acceptance["minimum_post_patch_next_batch_gain"]),
        "function_retention": gate["mean_function_retention"] >= float(acceptance["minimum_function_retention"]),
        "bootstrap": gate["paired_bootstrap_95pct_lower_bound"] > float(acceptance["paired_bootstrap_95pct_lower_bound"]),
        "corruption_curve": gate["corruption_severity_score_spearman"] <= float(acceptance["maximum_corruption_severity_score_spearman"]),
        "full_beats_static_separately": gate["mean_auc_delta_vs_static_dast"] > 0.0,
        "full_beats_flat_llm_separately": gate["mean_auc_delta_vs_flat_llm_memory"] > 0.0,
    }
    result = {
        "schema_version": "dastbench.trpb.closed_loop_evolution_run.v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "development_only": True, "official_test_labels_read": False,
        "claim_scope": protocol["claim_scope"], "acceptance": {"checks": checks, "gate": gate},
        "episode_count": len(seeds), "rounds_per_episode": rounds, "batch_size": batch_size,
        "gpu_fits": fit_count,
        "cuda": {"device": torch.cuda.get_device_name(device), "torch": torch.__version__,
                 "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))},
        "inputs": {"protocol": sha(protocol_path), "input_manifest": sha(source_path),
                   "llm_patch_manifest": sha(patch_manifest_path), "llm_receipts": sha(receipt_path),
                   "runner": sha(Path(__file__))},
        "outputs": {"round_scores.parquet": sha(scores_path), "query_trajectories.parquet": sha(queries_path),
                    "episode_scores.parquet": sha(episodes_path), "corruption_curve.parquet": sha(corruption_path),
                    "GPU_FIT_RECEIPTS.json": sha(receipts_path)},
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
                    "platform": platform.platform()},
    }
    write_json(out / "RUN.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

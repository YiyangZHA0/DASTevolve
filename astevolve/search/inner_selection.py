

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from astevolve.evaluation.selection import select_feasibility_first
from astevolve.search.artifact_io import _seqs_hash, _write_json, _write_yaml
from astevolve.search.config import SAConfig
from astevolve.search.normalized_artifact import normalize_search_artifact
from astevolve.search.structure_pipeline import _has_structure_signal


def _structure_candidate_identity(
    candidate: Mapping[str, Any],
    index: int,
    used: Dict[str, int],
) -> str:


    base = str(
        candidate.get("variant_id")
        or candidate.get("seq_hash")
        or f"structure_candidate_{index + 1}"
    )
    occurrence = used.get(base, 0)
    used[base] = occurrence + 1
    return base if occurrence == 0 else f"{base}#{occurrence + 1}"


def _candidate_gate_sources(candidate: Mapping[str, Any]) -> Dict[str, Any]:


    raw_sources = candidate.get("feasibility_gate_sources")
    coverage = candidate.get("semantic_final_mutation_coverage")
    layered_coverage = bool(
        str(candidate.get("coverage_scope") or "").strip().lower()
        == "shortlist_set"
        or str(candidate.get("structure_shortlist_coverage_scope") or "")
        .strip()
        .lower()
        == "shortlist_set"
        or (
            isinstance(coverage, Mapping)
            and (
                str(coverage.get("gate_scope") or "").strip().lower()
                == "shortlist_set"
                or coverage.get("individual_hard_gate_enabled") is False
            )
        )
    )
    sources: Dict[str, Any] = {
        str(source): dict(payload) if isinstance(payload, Mapping) else payload
        for source, payload in (
            raw_sources.items() if isinstance(raw_sources, Mapping) else []
        )
        if not (
            layered_coverage
            and str(source)
            in {
                "semantic_final_mutation_coverage",
                "semantic_final_coverage",
            }
        )
    }
    if "fast_filter" not in sources and isinstance(candidate.get("fast_filter"), Mapping):
        sources["fast_filter"] = dict(candidate.get("fast_filter") or {})
    if isinstance(coverage, Mapping) and not layered_coverage:
        missing = list(coverage.get("missing_required_nodes_by_mutation", []) or [])
        sources["semantic_final_mutation_coverage"] = {
            "pass": bool(coverage.get("pass", True)),
            "reasons": [f"required_node_not_mutated:{node}" for node in missing],
        }
    return sources


def _candidate_feasibility_priority(candidate: Mapping[str, Any]) -> Dict[str, Any] | None:
    direct = candidate.get("feasibility_priority")
    if isinstance(direct, Mapping):
        return dict(direct)
    for payload in _candidate_gate_sources(candidate).values():
        if not isinstance(payload, Mapping):
            continue
        nested = payload.get("gate_status")
        for container in (payload, nested):
            if isinstance(container, Mapping) and isinstance(
                container.get("feasibility_priority"), Mapping
            ):
                return dict(container["feasibility_priority"])
    return None


def _select_final_structure_candidate(
    candidates: List[Dict[str, Any]],
    *,
    stepping_stone_enabled: bool = False,
    stepping_stone_max_energy_degradation: float = 0.0,
    stepping_stone_metrics: tuple[str, ...] = (),
    stepping_stone_min_metric_gain: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    if not candidates:
        raise ValueError("at least one structure candidate is required")

    used: Dict[str, int] = {}
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    selection_rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_id = _structure_candidate_identity(candidate, index, used)
        candidate["selection_candidate_id"] = candidate_id
        candidate_by_id[candidate_id] = candidate
        row = {
                "candidate_id": candidate_id,
                "raw_objective": float(
                    candidate.get(
                        "combined_energy",
                        candidate.get("combined_loss", candidate.get("fast_loss", 0.0)),
                    )
                ),
                "gate_sources": _candidate_gate_sources(candidate),
            }
        priority = _candidate_feasibility_priority(candidate)
        if priority is not None:
            row["feasibility_priority"] = priority
        selection_rows.append(row)

    decision = select_feasibility_first(selection_rows, direction="minimize")
    normalized_by_id = {
        str(row["candidate_id"]): row for row in decision.get("candidates", [])
    }

    def objective_key(candidate_id: str) -> Tuple[float, float, str]:
        candidate = candidate_by_id[candidate_id]
        loss = float(
            candidate.get(
                "combined_energy",
                candidate.get("combined_loss", candidate.get("fast_loss", 0.0)),
            )
        )
        try:
            plddt = float(candidate.get("plddt", 0.0) or 0.0)
        except (TypeError, ValueError):
            plddt = 0.0
        if not np.isfinite(plddt):
            plddt = 0.0
        return (loss, -plddt, candidate_id)

    feasible_ids = [
        candidate_id
        for candidate_id in candidate_by_id
        if bool(normalized_by_id[candidate_id].get("feasible"))
    ]
    infeasible_ids = [
        candidate_id
        for candidate_id in candidate_by_id
        if not bool(normalized_by_id[candidate_id].get("feasible"))
    ]
    if decision.get("all_infeasible"):
        ordered_ids = list(decision.get("ordered_ids", []))
    else:
        ordered_ids = sorted(feasible_ids, key=objective_key) + sorted(
            infeasible_ids, key=objective_key
        )
    eligible_set = set(decision.get("eligible_ids", []) or [])
    decision["ordered_ids"] = ordered_ids
    decision["eligible_ids"] = [
        candidate_id for candidate_id in ordered_ids if candidate_id in eligible_set
    ]
    decision["selected_candidate_id"] = ordered_ids[0]
    decision["candidates"] = [normalized_by_id[candidate_id] for candidate_id in ordered_ids]
    decision["objective"] = {
        "primary": "combined_energy",
        "primary_direction": "minimize",
        "tie_break": "plddt",
        "tie_break_direction": "maximize",
    }
    decision["diagnostic_fallback"] = bool(decision.get("all_infeasible"))

    def candidate_metric(candidate_id: str, metric: str) -> float | None:
        candidate = candidate_by_id[candidate_id]
        value = candidate.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        report = candidate.get("inner_evaluator_report")
        for term in (report.get("terms", []) if isinstance(report, Mapping) else []):
            if isinstance(term, Mapping) and str(term.get("name")) == metric:
                score = term.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    return float(score)
        return None

    scientific_best_id = ordered_ids[0]
    if stepping_stone_enabled and not decision.get("all_infeasible"):
        root_ids = [candidate_id for candidate_id in feasible_ids if bool(candidate_by_id[candidate_id].get("is_parent_baseline")) or str(candidate_by_id[candidate_id].get("candidate_role") or "") == "parent_baseline"]
        mutant_ids = [candidate_id for candidate_id in feasible_ids if candidate_id not in root_ids and candidate_by_id[candidate_id].get("seqs") != candidate_by_id[scientific_best_id].get("seqs")]
        if root_ids and scientific_best_id in root_ids and mutant_ids:
            root_energy = objective_key(scientific_best_id)[0]
            eligible_mutants = [candidate_id for candidate_id in mutant_ids if objective_key(candidate_id)[0] <= root_energy + float(stepping_stone_max_energy_degradation)]
            if stepping_stone_metrics:
                eligible_mutants = [candidate_id for candidate_id in eligible_mutants if any(candidate_metric(candidate_id, metric) is not None and candidate_metric(scientific_best_id, metric) is not None and candidate_metric(candidate_id, metric) >= candidate_metric(scientific_best_id, metric) + float(stepping_stone_min_metric_gain) for metric in stepping_stone_metrics)]
            if eligible_mutants:
                stepping_id = min(eligible_mutants, key=objective_key)
                decision["scientific_best_candidate_id"] = scientific_best_id
                decision["selected_candidate_id"] = stepping_id
                decision["reason"] = "bounded_mutant_stepping_stone"
                decision["stepping_stone"] = {"enabled": True, "root_energy": root_energy, "selected_energy": objective_key(stepping_id)[0], "max_energy_degradation": float(stepping_stone_max_energy_degradation)}

    for candidate_id, candidate in candidate_by_id.items():
        candidate["feasibility_selection"] = dict(normalized_by_id[candidate_id])
    selected = candidate_by_id[str(decision["selected_candidate_id"])]
    return selected, decision


def _merge_structure_selection_pool(
    screen_results: List[Dict[str, Any]],
    rerank_results: List[Dict[str, Any]],
    *,
    allow_low_fidelity_fallback: bool = True,
) -> List[Dict[str, Any]]:


    valid_reranks = [item for item in rerank_results if _has_structure_signal(item)]

    if not allow_low_fidelity_fallback:
        for item in valid_reranks:
            item["structure_selection_fidelity"] = "rerank"
        return valid_reranks
    if not valid_reranks:
        fallback = [item for item in screen_results if _has_structure_signal(item)]
        for item in fallback:
            item["structure_selection_fidelity"] = "low_fidelity_fallback"
        return fallback

    def identity(candidate: Mapping[str, Any]) -> str:
        return str(candidate.get("seq_hash") or candidate.get("variant_id") or "")

    replaced = {identity(item) for item in valid_reranks}
    retained_screen = [
        item for item in screen_results if identity(item) not in replaced
    ]
    for item in valid_reranks:
        item["structure_selection_fidelity"] = "rerank"
    for item in retained_screen:
        item["structure_selection_fidelity"] = "compatibility_screen_overlay"
    return valid_reranks + retained_screen


def _write_inner_loop_artifacts(
    cfg: SAConfig,
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    round_summary: Dict[str, Any],
) -> Dict[str, str]:


    out_dir = Path(cfg.mcts_output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    artifact_mode = str(getattr(cfg, "mcts_artifact_mode", "normalized")).lower()
    if artifact_mode == "legacy_full":
        if cfg.mcts_save_tree:
            tree_json = []
            for node in tree.values():
                item = {key: value for key, value in node.items() if key != "seqs"}
                if node.get("seqs") is not None:
                    item["seq_hash"] = _seqs_hash(node["seqs"])
                tree_json.append(item)
            path = out_dir / "mcts_tree.json"
            _write_json(path, {"nodes": tree_json, "root": "root"})
            paths["mcts_tree"] = str(path)

        if cfg.mcts_save_variants:
            path = out_dir / "evaluated_variants.json"
            _write_json(path, candidates)
            paths["evaluated_variants"] = str(path)
    elif cfg.mcts_save_tree or cfg.mcts_save_variants:
        path = out_dir / "mcts_search.normalized.json"
        artifact = normalize_search_artifact(
            tree=tree if cfg.mcts_save_tree else None,
            candidates=candidates if cfg.mcts_save_variants else None,
        )
        _write_json(path, artifact)
        paths["normalized_search"] = str(path)


        if cfg.mcts_save_tree:
            paths["mcts_tree"] = str(path)
        if cfg.mcts_save_variants:
            paths["evaluated_variants"] = str(path)

    path = out_dir / "round_summary.yaml"
    _write_yaml(path, round_summary)
    paths["round_summary"] = str(path)
    return paths


__all__ = [
    "_merge_structure_selection_pool",
    "_select_final_structure_candidate",
    "_write_inner_loop_artifacts",
]

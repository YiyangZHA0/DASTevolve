

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, MutableSet, Optional, Tuple

from astevolve.search.config import SAConfig
from astevolve.search.energy_reporting import (
    SEARCH_ENERGY_SCHEMA_VERSION,
    best_so_far_trace,
)


MCTS_WIDENING_NODE_FIELD = "mcts_widening_by_expansion_key"
MCTS_WIDENING_PAIR_SCHEMA_VERSION = "astevolve.mcts_widening_pair.v1"


def attach_structure_shortlist_health(
    search_artifacts: Dict[str, Any],
    stage_summaries: Mapping[str, Mapping[str, Any]],
    *,
    enabled: bool,
) -> None:


    round_summary = search_artifacts.get("round_summary")
    if not isinstance(round_summary, dict):
        return
    search_health = round_summary.get("search_health")
    if not isinstance(search_health, dict):
        search_health = {"schema_version": "ast_search_health_v1"}
        round_summary["search_health"] = search_health
    stages = {
        str(stage): dict(summary)
        for stage, summary in stage_summaries.items()
        if isinstance(summary, Mapping)
    }
    search_health["structure_shortlist"] = {
        "available": bool(stages),
        "enabled": bool(enabled),
        "latest_stage": next(reversed(stages), None),
        "covered_nodes": sorted(
            {
                str(node)
                for summary in stages.values()
                for node in (summary.get("covered_nodes", []) or [])
                if str(node)
            }
        ),
        "covered_depths": sorted(
            {
                int(depth)
                for summary in stages.values()
                for depth in (summary.get("covered_depths", []) or [])
                if isinstance(depth, int) and not isinstance(depth, bool)
            }
        ),
        "stages": stages,
    }


def _mcts_widening_capacity(parent: Mapping[str, Any], cfg: SAConfig) -> int:


    return int(
        math.ceil(
            float(getattr(cfg, "mcts_progressive_widening_c", 2.0))
            * max(1.0, float(parent.get("visits", 0)))
            ** float(getattr(cfg, "mcts_progressive_widening_alpha", 0.5))
        )
    )


def _mcts_widening_pair_state(
    parent: Dict[str, Any], expansion_key: str
) -> Dict[str, Any]:


    key = str(expansion_key or "")
    by_key = parent.setdefault(MCTS_WIDENING_NODE_FIELD, {})
    if not isinstance(by_key, dict):
        raise ValueError(f"{MCTS_WIDENING_NODE_FIELD} must be a mapping")
    pair = by_key.setdefault(
        key,
        {
            "schema_version": MCTS_WIDENING_PAIR_SCHEMA_VERSION,
            "expansion_key": key,
            "committed_children": 0,
            "capacity": 0,
            "available_slots": 0,
            "capacity_exhausted": False,
            "proposal_space_exhausted": False,
        },
    )
    if not isinstance(pair, dict):
        raise ValueError("MCTS widening pair state must be a mapping")
    return pair


def _refresh_mcts_widening_pair_state(
    parent: Dict[str, Any], expansion_key: str, cfg: SAConfig
) -> Dict[str, Any]:


    pair = _mcts_widening_pair_state(parent, expansion_key)
    base_capacity = _mcts_widening_capacity(parent, cfg)
    recovery_capacity_bonus = max(
        0, int(pair.get("recovery_capacity_bonus", 0) or 0)
    )
    capacity = base_capacity + recovery_capacity_bonus
    committed = max(0, int(pair.get("committed_children", 0) or 0))
    available = max(0, capacity - committed)
    pair.update(
        {
            "base_capacity": int(base_capacity),
            "recovery_capacity_bonus": int(recovery_capacity_bonus),
            "capacity": int(capacity),
            "available_slots": int(available),
            "capacity_exhausted": bool(available == 0),
            "parent_visits": int(parent.get("visits", 0) or 0),
        }
    )
    return pair


def _proposal_tier_history_fields(cfg: SAConfig) -> Dict[str, Any]:


    return {
        "proposal_tier_mode": str(
            getattr(cfg, "proposal_tier_mode", "fixed_node")
        ),


        "proposal_tier_counts": {},
        "proposal_tier_round_counts": {},
        "proposal_tier_attempt_counts": {},
        "proposal_tier_novel_counts": {},
        "proposal_tier_count_semantics": {
            "proposal_tier_round_counts": (
                "one count per realized MCTS expansion round / optimizer batch"
            ),
            "proposal_tier_attempt_counts": (
                "one count per logical candidate before duplicate detection"
            ),
            "proposal_tier_novel_counts": (
                "one count per novel tree child committed after duplicate detection"
            ),
            "proposal_tier_counts": (
                "backward-compatible alias of proposal_tier_novel_counts"
            ),
        },
    }


def _attach_proposal_tier_accounting(
    round_summary: Dict[str, Any],
    history: Mapping[str, Any],
) -> None:


    round_summary["proposal_tier_accounting"] = {
        "schema_version": "astevolve.proposal_tier_accounting.v1",
        "mode": history.get("proposal_tier_mode"),
        "round_counts": dict(history.get("proposal_tier_round_counts", {})),
        "attempt_counts": dict(history.get("proposal_tier_attempt_counts", {})),
        "novel_counts": dict(history.get("proposal_tier_novel_counts", {})),
        "legacy_mutation_history_counts": dict(
            history.get("proposal_tier_counts", {})
        ),
        "round_summary_proposal_tier_counts_unit": (
            "logical_candidate_before_duplicate_detection"
        ),
        "mutation_history_proposal_tier_counts_unit": (
            "novel_tree_child_after_duplicate_detection"
        ),
    }
    widening = history.get("mcts_progressive_widening")
    if isinstance(widening, Mapping):
        round_summary["mcts_progressive_widening"] = {
            **dict(widening),
            "duplicate_policy": {
                "fast_score": "short_circuit_before_provider",
                "backpropagation": "skipped",
                "capacity_effect": "none",
            },
        }


def _mcts_child_score(parent: Dict[str, Any], child: Dict[str, Any], cfg: SAConfig) -> float:
    q = 0.0 if child["visits"] == 0 else child["total_reward"] / child["visits"]
    u = (
        float(cfg.mcts_c_puct)
        * float(child.get("prior", 1.0))
        * math.sqrt(max(1.0, float(parent["visits"])))
        / (1.0 + float(child["visits"]))
    )
    return float(q + u)


def _mcts_select_leaf(
    tree: Dict[str, Dict[str, Any]],
    root_id: str,
    cfg: SAConfig,
    *,
    expansion_key: Optional[str] = None,
    deferred_expansions: Optional[MutableSet[Tuple[str, str]]] = None,
) -> str:
    optimizer_enabled = bool(getattr(cfg, "node_optimizer_enabled", False))
    key = str(expansion_key or "")

    if not optimizer_enabled:
        node_id = root_id
        while tree[node_id]["depth"] < cfg.mcts_max_depth:
            parent = tree[node_id]
            passing_children = [
                child_id
                for child_id in parent["children"]
                if tree[child_id].get("inner_structure_gate_pass") is not False
            ]
            bootstrap_children = [
                child_id
                for child_id in parent["children"]
                if tree[child_id].get("inner_structure_gate_pass") is False
                and bool(tree[child_id].get("bootstrap_expandable", False))
            ]
            if bootstrap_children and not passing_children:
                node_id = max(
                    bootstrap_children,
                    key=lambda child_id: (
                        float(
                            (tree[child_id].get("fast_filter") or {}).get(
                                "search_progress", 0.0
                            )
                            or 0.0
                        ),
                        _mcts_child_score(parent, tree[child_id], cfg),
                    ),
                )
                continue
            eligible_children = passing_children + bootstrap_children
            if not eligible_children:
                return node_id
            node_id = max(
                eligible_children,
                key=lambda child_id: _mcts_child_score(parent, tree[child_id], cfg),
            )
        return root_id if tree[node_id]["depth"] >= cfg.mcts_max_depth else node_id

    # Backtrack from terminal or exhausted high-PUCT paths.  Returning root as
    # soon as the best path reached max_depth could deadlock progressive
    # widening when root already had no slot: no child, no backprop, no new
    # visits, and therefore no future widening capacity.
    def _find_expandable(node_id: str, path: set[str]) -> Optional[str]:
        if node_id in path:
            return None
        node = tree[node_id]
        if int(node.get("depth", 0)) >= int(cfg.mcts_max_depth):
            return None

        next_path = set(path)
        next_path.add(node_id)
        passing_children = [
            child_id
            for child_id in node["children"]
            if tree[child_id].get("inner_structure_gate_pass") is not False
        ]
        bootstrap_children = [
            child_id
            for child_id in node["children"]
            if tree[child_id].get("inner_structure_gate_pass") is False
            and bool(tree[child_id].get("bootstrap_expandable", False))
        ]

        if bootstrap_children and not passing_children:
            ordered_bootstrap = sorted(
                bootstrap_children,
                key=lambda child_id: (
                    float(
                        (tree[child_id].get("fast_filter") or {}).get(
                            "search_progress", 0.0
                        )
                        or 0.0
                    ),
                    _mcts_child_score(node, tree[child_id], cfg),
                ),
                reverse=True,
            )
            for child_id in ordered_bootstrap:
                selected = _find_expandable(child_id, next_path)
                if selected is not None:
                    return selected

        pair = _refresh_mcts_widening_pair_state(node, key, cfg)
        exhausted_token = (node_id, key)
        proposal_space_exhausted = bool(
            pair.get("proposal_space_exhausted", False)
            or (
                deferred_expansions is not None
                and exhausted_token in deferred_expansions
            )
        )
        if (
            not proposal_space_exhausted
            and int(pair.get("available_slots", 0) or 0) > 0
        ):
            return node_id

        eligible_children = passing_children + bootstrap_children
        ordered_children = sorted(
            dict.fromkeys(eligible_children),
            key=lambda child_id: _mcts_child_score(node, tree[child_id], cfg),
            reverse=True,
        )
        for child_id in ordered_children:
            selected = _find_expandable(child_id, next_path)
            if selected is not None:
                return selected

        # If every traversable child is terminal/exhausted (or high-fidelity
        # gating removed the whole frontier), unlock exactly one additional
        # sibling for this parent/action pair.  Successful commitment consumes
        # the slot; another slot is opened only after a later selection again
        # proves that no descendant can be expanded.
        if not proposal_space_exhausted:
            pair["recovery_capacity_bonus"] = int(
                pair.get("recovery_capacity_bonus", 0) or 0
            ) + 1
            pair["recovery_unlocks"] = int(
                pair.get("recovery_unlocks", 0) or 0
            ) + 1
            pair = _refresh_mcts_widening_pair_state(node, key, cfg)
            if int(pair.get("available_slots", 0) or 0) > 0:
                return node_id
        return None

    selected = _find_expandable(root_id, set())
    return selected if selected is not None else root_id


def _mcts_backprop(
    tree: Dict[str, Dict[str, Any]],
    node_id: str,
    reward: float,
) -> None:


    cur: Optional[str] = node_id
    while cur is not None:
        node = tree[cur]
        node["visits"] += 1
        node["total_reward"] += float(reward)
        node["best_reward"] = max(float(node.get("best_reward", -1e9)), float(reward))
        cur = node.get("parent")


def _mcts_best_path(tree: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
    path = []
    cur: Optional[str] = node_id
    while cur is not None:
        path.append(cur)
        cur = tree[cur].get("parent")
    return list(reversed(path))


def _mcts_tree_quality_report(
    tree: Mapping[str, Mapping[str, Any]], cfg: SAConfig
) -> Dict[str, Any]:


    nodes = list(tree.values())
    root = tree.get("root", {})
    child_counts = [len(node.get("children", []) or []) for node in nodes]
    observed = {
        "nodes": len(nodes),
        "candidate_nodes": max(0, len(nodes) - 1),
        "root_children": len(root.get("children", []) or []),
        "branching_nodes": sum(count > 1 for count in child_counts),
        "leaves": sum(count == 0 for count in child_counts),
        "max_children": max(child_counts, default=0),
        "max_depth": max(
            (int(node.get("depth", 0) or 0) for node in nodes), default=0
        ),
    }
    required = {
        "root_children": int(getattr(cfg, "mcts_tree_min_root_children", 0)),
        "branching_nodes": int(
            getattr(cfg, "mcts_tree_min_branching_nodes", 0)
        ),
        "leaves": int(getattr(cfg, "mcts_tree_min_leaves", 0)),
        "max_depth": int(getattr(cfg, "mcts_tree_min_max_depth", 0)),
    }
    failures = [
        f"{name}:{observed[name]}<{minimum}"
        for name, minimum in required.items()
        if int(observed[name]) < int(minimum)
    ]
    return {
        "schema_version": "astevolve.mcts_tree_quality.v1",
        "required": bool(getattr(cfg, "mcts_tree_quality_required", False)),
        "pass": not failures,
        "observed": observed,
        "thresholds": required,
        "failures": failures,
        "interpretation": (
            "realized branching/depth topology after all MCTS expansions and "
            "high-fidelity reward corrections"
        ),
    }


def _candidate_has_changes(candidate: Dict[str, Any]) -> bool:
    move = candidate.get("move", {}) if isinstance(candidate.get("move"), dict) else {}
    return bool(move.get("changes"))


def _compact_candidate(candidate: Dict[str, Any], root_fast: float) -> Dict[str, Any]:
    move = candidate.get("move", {}) if isinstance(candidate.get("move"), dict) else {}
    plan = move.get("mutation_plan", {}) if isinstance(move.get("mutation_plan"), dict) else {}
    node_optimization = move.get("node_optimization", {}) if isinstance(move.get("node_optimization"), dict) else {}
    node_request = node_optimization.get("request", {}) if isinstance(node_optimization.get("request"), dict) else {}
    node_result = node_optimization.get("result", {}) if isinstance(node_optimization.get("result"), dict) else {}
    node_provenance = node_result.get("provenance", {}) if isinstance(node_result.get("provenance"), dict) else {}
    fast_filter = candidate.get("fast_filter", {}) if isinstance(candidate.get("fast_filter"), dict) else {}
    return {
        "variant_id": candidate.get("variant_id"),
        "parent_id": candidate.get("parent_id"),
        "seq_hash": candidate.get("seq_hash"),
        "seqs": candidate.get("seqs"),
        "fast_loss": float(candidate.get("fast_loss", 0.0)),
        "energy": candidate.get("energy"),
        "root_fast_loss": float(root_fast),
        "delta_vs_root": float(root_fast) - float(candidate.get("fast_loss", 0.0)),
        "is_better_than_root": bool(float(candidate.get("fast_loss", 0.0)) < float(root_fast)),
        "constraint_penalty": float(candidate.get("constraint_penalty", 0.0)),
        "progen_loglik_avg": float(candidate.get("progen_loglik_avg", 0.0)),
        "proposal_log_prior": float(candidate.get("proposal_log_prior", 0.0)),
        "reward": float(candidate.get("reward", 0.0)),
        "proposal_prior_evidence": {
            "role": "mcts_edge_prior_only",
            "optimizer_id": node_optimization.get("selected_optimizer_id"),
            "candidate_id": node_optimization.get("candidate_id"),
            "request_hash": node_request.get("request_hash"),
            "result_hash": node_result.get("result_hash"),
            "prior_scorer_id": node_provenance.get("prior_scorer_id"),
        },
        "fast_filter": fast_filter,
        "move": {
            "schema_version": move.get("schema_version"),
            "op": move.get("op"),
            "outcome": move.get("outcome"),
            "reason": move.get("reason"),
            "operator_spec": move.get("operator_spec"),
            "operator_selection": move.get("operator_selection"),
            "node": move.get("node"),
            "chain_id": move.get("chain_id"),
            "target_nodes": move.get("target_nodes", []),
            "attempted_positions": move.get("attempted_positions", {}),
            "positions": move.get("positions", {}),
            "changes": move.get("changes", []),
            "actual_delta": move.get("actual_delta"),
            "motif": move.get("motif"),
            "motif_source": move.get("motif_source"),
            "mapping_attribution": move.get("mapping_attribution"),
            "mutation_plan": {
                "schema_version": plan.get("schema_version"),
                "tier": plan.get("tier"),
                "intent": plan.get("intent"),
                "budget": plan.get("budget"),
                "op_weights": plan.get("op_weights"),
                "operator_weight_provenance": plan.get("operator_weight_provenance"),
                "target_node": plan.get("target_node"),
                "allowed_residue_classes": plan.get("allowed_residue_classes"),
                "forbidden_residues": plan.get("forbidden_residues"),
                "mapping_execution": plan.get("mapping_execution"),
                "ast_id": plan.get("ast_id"),
                "ast_revision": plan.get("ast_revision"),
                "edge_id": plan.get("edge_id"),
                "functional_node_id": plan.get("functional_node_id"),
                "structural_node_id": plan.get("structural_node_id"),
                "action_id": plan.get("action_id"),
                "measurement_id": plan.get("measurement_id"),
                "legal_positions": plan.get("legal_positions"),
            },
        },
    }


def _best_mutated_candidate(candidates: List[Dict[str, Any]], root_fast: float) -> Optional[Dict[str, Any]]:
    mutated = [cand for cand in candidates if _candidate_has_changes(cand)]
    if not mutated:
        return None
    passing = [cand for cand in mutated if (cand.get("fast_filter", {}) or {}).get("pass", True)]
    selected_pool = passing if passing else mutated
    best = min(selected_pool, key=lambda cand: float(cand.get("fast_loss", 1e18)))
    compact = _compact_candidate(best, root_fast)
    compact["selection_note"] = "best_fast_filter_pass_mutated_candidate" if passing else "best_mutated_candidate_including_fast_filter_failures"
    compact["num_mutated_candidates"] = len(mutated)
    compact["num_fast_filter_pass_mutated_candidates"] = len(passing)
    return compact


def _mcts_search_health(
    tree: Mapping[str, Mapping[str, Any]],
    candidates: List[Dict[str, Any]],
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:


    depths: List[int] = []
    for node in tree.values():
        try:
            depths.append(max(0, int(node.get("depth", 0) or 0)))
        except (TypeError, ValueError):
            continue

    aggregation = comparison.get("aggregation", {}) or {}
    raw_count = int(aggregation.get("raw_candidate_count", len(candidates)) or 0)
    unique_count = int(aggregation.get("unique_sequence_count", 0) or 0)
    duplicate_count = int(
        aggregation.get(
            "duplicate_rollout_count", max(0, raw_count - unique_count)
        )
        or 0
    )
    required_nodes = sorted(
        {
            str(node)
            for candidate in candidates
            for coverage in [candidate.get("semantic_final_coverage", {}) or {}]
            if isinstance(coverage, Mapping)
            for node in (coverage.get("required_nodes", []) or [])
            if str(node)
        }
    )
    semantic_joint = comparison.get("semantic_final_joint_coverage", {}) or {}
    joint_count = (
        int(semantic_joint.get("pass_count", 0) or 0)
        if required_nodes
        else None
    )
    return {
        "schema_version": "ast_search_health_v1",
        "mcts": {
            "tree_node_count": len(tree),
            "max_tree_depth": max(depths, default=0),
        },
        "candidate_population": {
            "raw_candidate_count": raw_count,
            "unique_sequence_count": unique_count,
            "duplicate_rollout_count": duplicate_count,
            "duplicate_rollout_rate": (
                float(duplicate_count) / float(raw_count) if raw_count else 0.0
            ),
            "required_nodes": required_nodes,
            "required_node_joint_candidate_count": joint_count,
        },
        "structure_shortlist": {
            "available": False,
            "enabled": None,
            "latest_stage": None,
            "covered_nodes": [],
            "covered_depths": [],
            "stages": {},
        },
    }


def _summarize_mcts_round(
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    best_node_id: str,
    root_fast: float,
) -> Dict[str, Any]:


    node_stats: Dict[str, Dict[str, Any]] = {}
    mutation_successes: List[Dict[str, Any]] = []
    tier_counts: Dict[str, int] = {}
    fast_filter_failures: Dict[str, int] = {}

    for cand in candidates:
        move = cand.get("move", {})
        tier = str((move.get("mutation_plan") or {}).get("tier", "unknown"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        fast_filter = cand.get("fast_filter", {}) or {}
        if not fast_filter.get("pass", True):
            for reason in fast_filter.get("reasons", []) or ["unknown"]:
                fast_filter_failures[reason] = fast_filter_failures.get(reason, 0) + 1
        node_name = str(move.get("node", "unknown"))
        stats = node_stats.setdefault(
            node_name,
            {
                "evaluated": 0,
                "improved": 0,
                "best_fast_loss": None,
                "mean_fast_loss": 0.0,
                "mean_reward": 0.0,
                "top_changes": [],
            },
        )
        stats["evaluated"] += 1
        stats["mean_fast_loss"] += float(cand["fast_loss"])
        stats["mean_reward"] += float(cand.get("reward", 0.0))
        if cand["fast_loss"] < root_fast:
            stats["improved"] += 1
            for change in move.get("changes", [])[:4]:
                mutation_successes.append({
                    **change,
                    "fast_loss": float(cand["fast_loss"]),
                    "reward": float(cand.get("reward", 0.0)),
                })
        best_loss = stats["best_fast_loss"]
        if best_loss is None or cand["fast_loss"] < best_loss:
            stats["best_fast_loss"] = float(cand["fast_loss"])
            stats["top_changes"] = move.get("changes", [])[:8]

    for stats in node_stats.values():
        n = max(1, int(stats["evaluated"]))
        stats["mean_fast_loss"] = float(stats["mean_fast_loss"] / n)
        stats["mean_reward"] = float(stats["mean_reward"] / n)
        stats["success_rate"] = float(stats["improved"] / n)

    promoted = sorted(
        node_stats,
        key=lambda n: (
            node_stats[n]["success_rate"],
            -float(node_stats[n]["best_fast_loss"] or 0.0),
        ),
        reverse=True,
    )[:5]
    suppressed = sorted(
        node_stats,
        key=lambda n: (node_stats[n]["success_rate"], node_stats[n]["mean_reward"]),
    )[:5]

    mutation_successes = sorted(
        mutation_successes,
        key=lambda x: (-float(x["reward"]), float(x["fast_loss"])),
    )[:25]
    best_mutated = _best_mutated_candidate(candidates, root_fast)
    comparison = _compare_inner_loop_candidates(candidates, root_fast)
    report = _build_experiment_analysis_report(
        comparison=comparison,
        node_stats=node_stats,
        root_fast=root_fast,
        best_node_id=best_node_id,
        best_mutated=best_mutated,
    )

    return {
        "created_at_unix": int(time.time()),
        "search_method": "mcts",
        "energy_schema_version": SEARCH_ENERGY_SCHEMA_VERSION,
        "energy_direction": "minimize",
        "root_fast_loss": float(root_fast),
        "root_energy": float(root_fast),
        "best_so_far_energy": min(
            [float(root_fast)]
            + [
                float(item["fast_loss"])
                for item in candidates
                if bool((item.get("fast_filter") or {}).get("pass", True))
            ]
        ),
        "energy_trace": best_so_far_trace(candidates, root_energy=root_fast),
        "best_node_id": best_node_id,
        "best_path": _mcts_best_path(tree, best_node_id),
        "best_final_is_root": bool(best_node_id == "root"),
        "best_mutated_candidate": best_mutated,
        "num_tree_nodes": len(tree),
        "num_evaluated_variants": len(candidates),
        "node_level_statistics": node_stats,
        "proposal_tier_counts": tier_counts,
        "fast_filter_failures": fast_filter_failures,
        "candidate_comparison": comparison,
        "search_health": _mcts_search_health(tree, candidates, comparison),
        "experiment_analysis_report": report,
        "internal_memory_update_suggestion": {
            "promote_nodes": promoted,
            "suppress_nodes": suppressed,
            "effective_mutations": mutation_successes,
            "note": "Use this summary to update adaptive_memory; the full MCTS tree is short-term search state.",
        },
    }


def _move_nodes(move: Mapping[str, Any]) -> List[str]:
    node = str(move.get("node") or move.get("segment") or "")
    nodes = [node] if node else []
    for change in move.get("changes", []) or []:
        if isinstance(change, Mapping) and change.get("node"):
            nodes.append(str(change.get("node")))
    return sorted(set(x for x in nodes if x))


def _candidate_digest(cand: Mapping[str, Any], root_fast: float) -> Dict[str, Any]:
    move = cand.get("move", {}) or {}
    fast_filter = cand.get("fast_filter", {}) or {}
    coverage = cand.get("semantic_final_coverage", {}) or {}
    fast_loss = float(cand.get("fast_loss", 0.0) or 0.0)
    return {
        "variant_id": cand.get("variant_id"),
        "parent_id": cand.get("parent_id"),
        "seq_hash": cand.get("seq_hash"),
        "fast_loss": fast_loss,
        "energy": cand.get("energy"),
        "delta_fast_loss_vs_root": float(fast_loss - float(root_fast)),
        "reward": float(cand.get("reward", 0.0) or 0.0),
        "constraint_penalty": float(cand.get("constraint_penalty", 0.0) or 0.0),
        "node": str(move.get("node") or move.get("segment") or ""),
        "nodes": _move_nodes(move),
        "op": str(move.get("op") or ""),
        "tier": str((move.get("mutation_plan") or {}).get("tier") or "unknown"),
        "num_changes": len(move.get("changes", []) or []),
        "changes": list(move.get("changes", []) or [])[:8],
        "fast_filter_pass": bool(fast_filter.get("pass", True)),
        "fast_filter_reasons": list(fast_filter.get("reasons", []) or []),
        "semantic_coverage_pass": bool(coverage.get("pass", True)),
        "semantic_missing_required_nodes": list(coverage.get("missing_required_nodes_by_mutation", []) or []),
        "mcts": cand.get("mcts", {}),
    }


def _candidate_sequence_identity(cand: Mapping[str, Any]) -> tuple:


    seqs = cand.get("seqs")
    if isinstance(seqs, Mapping) and seqs:
        return (
            "sequence",
            tuple(sorted((str(chain), str(sequence)) for chain, sequence in seqs.items())),
        )
    seq_hash = str(cand.get("seq_hash") or "").strip()
    if seq_hash:
        return ("seq_hash", seq_hash)
    return ("action_fallback", _candidate_action_identity(cand))


def _normalized_change_identity(change: Any) -> tuple:
    if not isinstance(change, Mapping):
        return (("value", str(change)),)
    preferred = ("chain_id", "position", "from", "to", "node")
    values = tuple((key, str(change.get(key))) for key in preferred if key in change)
    if values:
        return values
    return tuple(sorted((str(key), str(value)) for key, value in change.items()))


def _candidate_action_identity(cand: Mapping[str, Any]) -> tuple:


    move = cand.get("move", {}) or {}
    if not isinstance(move, Mapping):
        move = {}
    changes = tuple(
        sorted(_normalized_change_identity(change) for change in move.get("changes", []) or [])
    )
    positions = move.get("positions", {}) or move.get("attempted_positions", {}) or {}
    if isinstance(positions, Mapping):
        position_identity = tuple(
            sorted((str(chain), str(value)) for chain, value in positions.items())
        )
    else:
        position_identity = (str(positions),)
    return (
        str(move.get("node") or move.get("segment") or ""),
        str(move.get("op") or ""),
        tuple(sorted(str(node) for node in move.get("target_nodes", []) or [])),
        changes,
        position_identity,
    )


def _aggregate_candidate_group(
    group: List[Mapping[str, Any]],
    root_fast: float,
) -> Dict[str, Any]:


    passing = [
        cand
        for cand in group
        if bool((cand.get("fast_filter", {}) or {}).get("pass", True))
    ]
    representative_pool = passing or group
    representative = min(
        representative_pool,
        key=lambda cand: (
            float(
                cand.get("fast_loss")
                if cand.get("fast_loss") is not None
                else 1e18
            ),
            -float(cand.get("reward", 0.0) or 0.0),
            str(cand.get("variant_id") or ""),
        ),
    )
    row = _candidate_digest(representative, root_fast)

    action_groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for cand in group:
        action_groups.setdefault(_candidate_action_identity(cand), []).append(cand)

    action_support = []
    all_nodes = set()
    all_ops = set()
    all_tiers = set()
    for action_group in action_groups.values():
        action_rep = action_group[0]
        move = action_rep.get("move", {}) or {}
        nodes = _move_nodes(move)
        op = str(move.get("op") or "")
        tier = str((move.get("mutation_plan") or {}).get("tier") or "unknown")
        all_nodes.update(nodes)
        if op:
            all_ops.add(op)
        all_tiers.add(tier)
        action_support.append(
            {
                "node": str(move.get("node") or move.get("segment") or ""),
                "nodes": nodes,
                "op": op,
                "tier": tier,
                "changes": list(move.get("changes", []) or [])[:8],
                "raw_occurrence_count": len(action_group),
            }
        )
    action_support.sort(
        key=lambda action: (
            action["node"],
            action["op"],
            str(action["changes"]),
        )
    )

    fast_filter_pass = bool(passing)
    fast_filter_reasons = sorted(
        {
            str(reason)
            for cand in group
            for reason in ((cand.get("fast_filter", {}) or {}).get("reasons", []) or [])
        }
    )
    semantic_rows = [cand.get("semantic_final_coverage", {}) or {} for cand in group]
    semantic_coverage_pass = any(bool(item.get("pass", True)) for item in semantic_rows)
    semantic_missing = sorted(
        {
            str(node)
            for item in semantic_rows
            for node in (item.get("missing_required_nodes_by_mutation", []) or [])
        }
    )
    if semantic_coverage_pass:
        semantic_missing = []

    fast_loss = min(float(cand.get("fast_loss", 0.0) or 0.0) for cand in representative_pool)
    proxy_improvement = bool(fast_filter_pass and fast_loss < float(root_fast))
    row.update(
        {
            "fast_loss": fast_loss,
            "delta_fast_loss_vs_root": float(fast_loss - float(root_fast)),
            "fast_filter_pass": fast_filter_pass,
            "fast_filter_reasons": fast_filter_reasons,
            "semantic_coverage_pass": semantic_coverage_pass,
            "semantic_missing_required_nodes": semantic_missing,
            "physical_fast_improvement": proxy_improvement,
            "semantic_final_joint_coverage_pass": semantic_coverage_pass,
            "joint_success": bool(proxy_improvement and semantic_coverage_pass),
            "nodes": sorted(all_nodes),
            "ops": sorted(all_ops),
            "tiers": sorted(all_tiers),
            "raw_occurrence_count": len(group),
            "duplicate_occurrence_count": max(0, len(group) - 1),
            "unique_action_count": len(action_groups),
            "action_support": action_support[:12],
            "variant_ids": [str(cand.get("variant_id") or "") for cand in group[:12]],
        }
    )
    return row


def _unique_candidate_rows(
    candidates: List[Dict[str, Any]],
    root_fast: float,
) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for cand in candidates:
        groups.setdefault(_candidate_sequence_identity(cand), []).append(cand)
    return [_aggregate_candidate_group(group, root_fast) for group in groups.values()]


def _compare_inner_loop_candidates(candidates: List[Dict[str, Any]], root_fast: float) -> Dict[str, Any]:


    rows = _unique_candidate_rows(candidates, root_fast)
    successes: List[Dict[str, Any]] = []
    viable: List[Dict[str, Any]] = []
    hard_failures: List[Dict[str, Any]] = []
    proxy_improvements: List[Dict[str, Any]] = []
    non_improving: List[Dict[str, Any]] = []
    semantic_incomplete: List[Dict[str, Any]] = []
    constraint_failures: Dict[str, int] = {}
    semantic_failures: Dict[str, int] = {}
    node_candidates: Dict[str, int] = {}
    node_joint_success: Dict[str, int] = {}
    node_hard_failure: Dict[str, int] = {}
    node_proxy_improvement: Dict[str, int] = {}
    node_non_improving: Dict[str, int] = {}
    node_semantic_incomplete: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    op_counts: Dict[str, int] = {}
    fast_filter_pass_count = 0
    semantic_coverage_pass_count = 0
    loss_improvement_count = 0
    for row in rows:
        fast_pass = bool(row["fast_filter_pass"])
        semantic_pass = bool(row["semantic_coverage_pass"])
        improving = bool(fast_pass and row["fast_loss"] < root_fast)
        not_improving = bool(fast_pass and row["fast_loss"] >= root_fast)
        is_viable = fast_pass and semantic_pass
        is_success = is_viable and improving
        if is_viable:
            viable.append(row)
        if is_success:
            successes.append(row)
        if not fast_pass:
            hard_failures.append(row)
        if improving:
            proxy_improvements.append(row)
        if not_improving:
            non_improving.append(row)
        if not semantic_pass:
            semantic_incomplete.append(row)
        fast_filter_pass_count += int(fast_pass)
        semantic_coverage_pass_count += int(semantic_pass)
        loss_improvement_count += int(improving)
        for action in row.get("action_support", []) or []:
            tier = str(action.get("tier") or "unknown")
            op = str(action.get("op") or "")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if op:
                op_counts[op] = op_counts.get(op, 0) + 1
        if not fast_pass:
            for reason in row["fast_filter_reasons"]:
                constraint_failures[reason] = constraint_failures.get(reason, 0) + 1
        for node in row["semantic_missing_required_nodes"]:
            semantic_failures[node] = semantic_failures.get(node, 0) + 1
        for node in row["nodes"]:
            node_candidates[node] = node_candidates.get(node, 0) + 1
            if is_success:
                node_joint_success[node] = node_joint_success.get(node, 0) + 1
            if not fast_pass:
                node_hard_failure[node] = node_hard_failure.get(node, 0) + 1
            if improving:
                node_proxy_improvement[node] = node_proxy_improvement.get(node, 0) + 1
            if not_improving:
                node_non_improving[node] = node_non_improving.get(node, 0) + 1
            if not semantic_pass:
                node_semantic_incomplete[node] = node_semantic_incomplete.get(node, 0) + 1

    best = sorted(viable or rows, key=lambda row: (row["fast_loss"], -row["reward"]))[:8]
    worst_hard = sorted(
        hard_failures,
        key=lambda row: (-row["fast_loss"], row["reward"]),
    )[:8]
    best_proxy = sorted(
        proxy_improvements,
        key=lambda row: (row["fast_loss"], -row["reward"]),
    )[:8]
    worst_non_improving = sorted(
        non_improving,
        key=lambda row: (-row["fast_loss"], row["reward"]),
    )[:8]
    incomplete_examples = sorted(
        semantic_incomplete,
        key=lambda row: (not row["physical_fast_improvement"], row["fast_loss"]),
    )[:8]
    unique_action_count = sum(int(row.get("unique_action_count", 0) or 0) for row in rows)

    return {
        "schema_version": "ast_candidate_comparison_v2",
        "root_fast_loss": float(root_fast),
        "aggregation": {
            "primary_unit": "unique_final_sequence",
            "action_unit": "unique_sequence_action",
            "sequence_identity_precedence": ["seqs", "seq_hash", "action_fallback"],
            "raw_candidate_count": len(candidates),
            "unique_sequence_count": len(rows),
            "duplicate_rollout_count": max(0, len(candidates) - len(rows)),
            "unique_action_count": unique_action_count,
        },
        "outcome_semantics": {
            "success_count": "fast-filter-pass AND fast-loss-improvement AND final-joint-semantic-coverage",
            "failure_count": "explicit-fast-hard-filter-failure-only",
            "proxy_improvement": "fast-filter-pass AND fast_loss < root_fast_loss",
            "semantic_coverage": "instruction-following axis; not a physical-score verdict",
        },
        "raw_candidate_count": len(candidates),
        "candidate_count": len(rows),
        "success_count": len(successes),
        "viable_count": len(viable),
        "failure_count": len(hard_failures),
        "joint_ineligible_count": len(rows) - len(viable),
        "success_examples": sorted(successes, key=lambda row: (row["fast_loss"], -row["reward"]))[:8],
        "best_candidates": best,
        "failure_examples": worst_hard,
        "proxy_improvement_examples": best_proxy,
        "semantic_incomplete_examples": incomplete_examples,
        "non_improving_examples": worst_non_improving,
        "physical_fast_outcomes": {
            "basis": "unique final sequences passing the fast filter; lower fast_loss is better",
            "evidence_scope": "inexpensive fast proxy only; not final structure evaluation",
            "proxy_improvement_count": len(proxy_improvements),
            "non_improving_count": len(non_improving),
            "hard_filter_failure_count": len(hard_failures),
            "node_proxy_improvement_counts": dict(sorted(node_proxy_improvement.items())),
            "node_non_improving_counts": dict(sorted(node_non_improving.items())),
        },
        "semantic_final_joint_coverage": {
            "basis": "required-node mutation coverage of each unique final sequence",
            "pass_count": semantic_coverage_pass_count,
            "incomplete_count": len(semantic_incomplete),
            "missing_required_node_counts": dict(sorted(semantic_failures.items())),
            "node_incomplete_counts": dict(sorted(node_semantic_incomplete.items())),
        },
        "joint_outcomes": {
            "fast_improvement_and_semantic_coverage_count": len(successes),
            "fast_improvement_without_semantic_coverage_count": sum(
                1 for row in proxy_improvements if not row["semantic_coverage_pass"]
            ),
            "semantic_coverage_without_fast_improvement_count": sum(
                1
                for row in rows
                if row["fast_filter_pass"]
                and row["semantic_coverage_pass"]
                and not row["physical_fast_improvement"]
            ),
        },
        "effective_constraints": {
            "fast_filter_pass_count": fast_filter_pass_count,
            "semantic_coverage_pass_count": semantic_coverage_pass_count,
            "loss_improvement_count": loss_improvement_count,
        },
        "failed_constraints": {
            "fast_filter_reasons": dict(sorted(constraint_failures.items())),
            "semantic_missing_required_nodes": dict(sorted(semantic_failures.items())),
        },
        "constraint_failure_counts": dict(sorted(constraint_failures.items())),
        "semantic_missing_required_node_counts": dict(sorted(semantic_failures.items())),
        "node_candidate_counts": dict(sorted(node_candidates.items())),
        "node_success_counts": dict(sorted(node_joint_success.items())),
        "node_failure_counts": dict(sorted(node_hard_failure.items())),
        "node_proxy_improvement_counts": dict(sorted(node_proxy_improvement.items())),
        "node_non_improving_counts": dict(sorted(node_non_improving.items())),
        "node_semantic_incomplete_counts": dict(sorted(node_semantic_incomplete.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "op_counts": dict(sorted(op_counts.items())),
    }


def _build_experiment_analysis_report(
    comparison: Mapping[str, Any],
    node_stats: Mapping[str, Any],
    root_fast: float,
    best_node_id: str,
    best_mutated: Any,
) -> Dict[str, Any]:


    success_nodes = comparison.get("node_success_counts", {}) or {}
    proxy_improvement_nodes = comparison.get("node_proxy_improvement_counts", {}) or {}
    failure_nodes = comparison.get("node_failure_counts", {}) or {}
    node_candidate_counts = comparison.get("node_candidate_counts", {}) or {}
    constraint_failures = comparison.get("constraint_failure_counts", {}) or {}
    semantic_failures = comparison.get("semantic_missing_required_node_counts", {}) or {}

    success_patterns = []
    for node, count in sorted(success_nodes.items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
        stats = node_stats.get(node, {}) if isinstance(node_stats, Mapping) else {}
        unique_support = max(1, int(node_candidate_counts.get(node, 0) or 0))
        success_patterns.append({
            "node": node,
            "support": int(count),
            "success_rate": float(int(count) / unique_support),
            "best_fast_loss": stats.get("best_fast_loss"),
            "interpretation": "edits on this node produced at least one semantic-valid loss improvement",
        })

    proxy_improvement_patterns = []
    for node, count in sorted(
        proxy_improvement_nodes.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:8]:
        unique_support = max(1, int(node_candidate_counts.get(node, 0) or 0))
        proxy_improvement_patterns.append({
            "node": node,
            "support": int(count),
            "unique_sequence_rate": float(int(count) / unique_support),
            "joint_success_support": int(success_nodes.get(node, 0) or 0),
            "interpretation": (
                "edits on this node improved the fast physical proxy; joint "
                "required-node coverage is reported independently"
            ),
        })

    failure_patterns = []
    for node, count in sorted(failure_nodes.items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
        unique_support = max(1, int(node_candidate_counts.get(node, 0) or 0))
        failure_patterns.append({
            "node": node,
            "support": int(count),
            "hard_filter_failure_rate": float(int(count) / unique_support),
            "interpretation": "unique final sequences produced by this node failed an explicit fast hard filter",
        })

    likely_causes = []
    if constraint_failures:
        likely_causes.append({
            "type": "constraint_failure",
            "counts": dict(sorted(constraint_failures.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
            "explanation": "candidate generation is violating hard sequence/register/mutation-budget filters",
        })
    if semantic_failures:
        likely_causes.append({
            "type": "semantic_coverage_failure",
            "counts": dict(sorted(semantic_failures.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
            "explanation": "inner search is not touching all required AST nodes often enough",
        })
    if (
        not likely_causes
        and int(comparison.get("success_count", 0) or 0) == 0
        and not proxy_improvement_patterns
    ):
        likely_causes.append({
            "type": "weak_improvement_signal",
            "explanation": "candidates pass filters but do not improve root fast loss; search space or objective weights may need AST-level adjustment",
        })

    ast_recommendations = []
    if constraint_failures:
        ast_recommendations.append({
            "action": "repair_constraint_design",
            "target_constraints": list(constraint_failures.keys())[:6],
            "reason": "hard filters repeatedly reject generated variants; revise constraint emphasis, mutable masks, or node-specific residue rules",
        })
    if semantic_failures:
        ast_recommendations.append({
            "action": "rebalance_required_node_coverage",
            "target_nodes": list(semantic_failures.keys())[:6],
            "reason": "required structural nodes are under-sampled or absent from final variants",
        })
    if failure_patterns:
        ast_recommendations.append({
            "action": "narrow_or_retype_failure_nodes",
            "target_nodes": [item["node"] for item in failure_patterns[:5]],
            "reason": "these nodes generate unique sequences rejected by explicit hard filters; consider lowering mutation rate or redefining the node boundary",
        })
    if proxy_improvement_patterns:
        ast_recommendations.append({
            "action": "preserve_fast_proxy_improvement_signal",
            "target_nodes": [item["node"] for item in proxy_improvement_patterns[:5]],
            "reason": (
                "these nodes produced unique fast-proxy improvements; retain that "
                "signal while combining edits needed for final semantic coverage"
            ),
        })
    if success_patterns:
        ast_recommendations.append({
            "action": "expand_successful_structural_motifs",
            "target_nodes": [item["node"] for item in success_patterns[:5]],
            "reason": "successful mutations concentrate here; consider nearby loop/edge coupling or focused residue rules",
        })

    return {
        "schema_version": "ast_experiment_analysis_report_v2",
        "root_fast_loss": float(root_fast),
        "best_node_id": best_node_id,
        "best_mutated_candidate": best_mutated,
        "success_patterns": success_patterns,
        "proxy_improvement_patterns": proxy_improvement_patterns,
        "failure_patterns": failure_patterns,
        "likely_failure_causes": likely_causes,
        "key_design_rules": [
            "Report fast physical-proxy improvement independently from final joint semantic coverage, then seek candidates satisfying both.",
            "Treat repeated fixed/register violations as a node-boundary or constraint-design issue, not only a parameter issue.",
            "Treat unique-sequence semantic coverage misses as evidence to rebalance required nodes, region bindings, or search-space topology.",
        ],
        "ast_level_recommendations": ast_recommendations,
        "next_loop_bias": {
            "phase_hint": (
                "refine_ast"
                if int(comparison.get("success_count", 0) or 0) > 0
                or proxy_improvement_patterns
                else "explore_ast"
            ),
            "increase_inner_precision": bool(
                (
                    int(comparison.get("success_count", 0) or 0) > 0
                    or proxy_improvement_patterns
                )
                and not constraint_failures
            ),
            "decrease_mutation_budget": bool(constraint_failures),
        },
    }

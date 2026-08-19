

from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from astevolve.search.config import SAConfig
from astevolve.search.semantic_coverage import (
    _semantic_coverage_hard_enabled,
    _semantic_min_mutations,
    _semantic_required_nodes,
    _semantic_required_unavailable_nodes,
)


JOINT_COVERAGE_COMPLETION_VERSION = "astevolve.joint_coverage_completion.v1"


def _unique_strings(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [item.strip() for item in values.replace(";", ",").split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw = [str(item).strip() for item in values]
    else:
        raw = [str(values).strip()]
    return list(dict.fromkeys(item for item in raw if item))


def _segment_positions(segment: Any, positions: Sequence[int]) -> set[Tuple[str, int]]:
    chain_id = str(getattr(segment, "chain_id", "") or "")
    return {(chain_id, int(position)) for position in positions}


def _minimum_unique_mutations(
    required_nodes: Sequence[str],
    positions_by_node: Mapping[str, set[Tuple[str, int]]],
    minimum_per_node: int,
) -> Optional[int]:


    if minimum_per_node <= 0 or not required_nodes:
        return 0
    all_positions = sorted(
        {
            position
            for node in required_nodes
            for position in positions_by_node.get(node, set())
        }
    )
    target = tuple(int(minimum_per_node) for _node in required_nodes)
    start = tuple(0 for _node in required_nodes)
    costs: Dict[Tuple[int, ...], int] = {start: 0}
    for position in all_positions:
        covered_indices = [
            index
            for index, node in enumerate(required_nodes)
            if position in positions_by_node.get(node, set())
        ]
        if not covered_indices:
            continue
        updated = dict(costs)
        for state, cost in costs.items():
            next_state = list(state)
            for index in covered_indices:
                next_state[index] = min(target[index], next_state[index] + 1)
            next_key = tuple(next_state)
            updated[next_key] = min(updated.get(next_key, 10**9), cost + 1)
        costs = updated
    return costs.get(target)


def _mapping_actions_by_node(
    mapping_actions: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw_action in mapping_actions:
        action = dict(raw_action)
        node = str(action.get("compiled_segment_name") or "")
        if node:
            grouped.setdefault(node, []).append(action)
    for actions in grouped.values():
        actions.sort(key=lambda item: str(item.get("action_id") or ""))
    return grouped


def initialize_joint_coverage_completion(
    cfg: SAConfig,
    designable: Sequence[Tuple[Any, List[int]]],
    mapping_actions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:


    required = _semantic_required_nodes(cfg, list(designable))
    declared_required = _semantic_required_nodes(cfg)
    unavailable = _semantic_required_unavailable_nodes(cfg, list(designable))
    minimum = _semantic_min_mutations(cfg)
    hard_enabled = _semantic_coverage_hard_enabled(cfg)
    formal_joint_enabled = (
        str(getattr(cfg, "structure_shortlist_policy", "legacy_diverse") or "")
        .strip()
        .lower()
        == "formal_joint_novel"
    )
    formal_layered_enabled = (
        str(getattr(cfg, "structure_shortlist_policy", "legacy_diverse") or "")
        .strip()
        .lower()
        == "formal_layered_novel"
    )
    active_nodes = _unique_strings(getattr(cfg, "semantic_active_nodes", []))
    if not active_nodes:
        active_nodes = list(declared_required)
    declared_anchor_nodes = _unique_strings(
        getattr(cfg, "semantic_anchor_nodes", [])
    )
    if declared_anchor_nodes:
        anchor_nodes = declared_anchor_nodes
        anchor_source = "semantic_anchor_nodes"
    elif formal_layered_enabled and declared_required:


        anchor_nodes = list(declared_required)
        anchor_source = "semantic_required_nodes_compatibility_fallback"
    else:
        anchor_nodes = []
        anchor_source = "none"
    mapping_enabled = bool(mapping_actions)
    designable_by_node: Dict[str, List[Tuple[Any, List[int]]]] = {}
    positions_by_node: Dict[str, set[Tuple[str, int]]] = {}
    for segment, positions in designable:
        node = str(getattr(segment, "name", "") or "")
        if not node:
            continue
        designable_by_node.setdefault(node, []).append((segment, list(positions)))
        positions_by_node.setdefault(node, set()).update(
            _segment_positions(segment, positions)
        )

    actions_by_node = _mapping_actions_by_node(mapping_actions)
    executable_positions_by_node: Dict[str, set[Tuple[str, int]]] = {}
    max_step_capacity: Dict[str, int] = {}
    action_ids_by_node: Dict[str, List[str]] = {}
    impossible_reasons: List[str] = []
    minimum_steps_by_node: Dict[str, int] = {}

    for node in required:
        node_positions = set(positions_by_node.get(node, set()))
        executable_positions = set(node_positions)
        capacities: List[int] = []
        if mapping_enabled:
            node_actions = actions_by_node.get(node, [])
            action_ids_by_node[node] = [
                str(action.get("action_id") or "") for action in node_actions
            ]
            executable_positions = set()
            for action in node_actions:
                chain_id = str(action.get("chain_id") or "")
                legal = {
                    (chain_id, int(position))
                    for position in action.get("legal_positions", []) or []
                }
                usable = node_positions & legal
                budget = action.get("budget")
                maximum = (
                    int(budget.get("max", 0) or 0)
                    if isinstance(budget, Mapping)
                    else 0
                )
                capacity = min(len(usable), max(0, maximum))
                if capacity > 0:
                    executable_positions.update(usable)
                    capacities.append(capacity)
            if not node_actions:
                impossible_reasons.append(
                    f"required_node_without_executable_mapping_action:{node}"
                )
        else:


            capacities.append(len(executable_positions))

        executable_positions_by_node[node] = executable_positions
        capacity = max(capacities, default=0)
        max_step_capacity[node] = int(capacity)
        if minimum > 0 and len(executable_positions) < minimum:
            impossible_reasons.append(
                f"required_node_insufficient_executable_positions:{node}"
            )
        if minimum > 0 and capacity <= 0:
            impossible_reasons.append(
                f"required_node_zero_mutation_capacity:{node}"
            )
        if minimum > 0 and capacity > 0:
            minimum_steps_by_node[node] = int(ceil(minimum / capacity))
        else:
            minimum_steps_by_node[node] = 0

    for node in unavailable:
        impossible_reasons.append(f"required_node_not_designable:{node}")

    minimum_unique = _minimum_unique_mutations(
        required,
        executable_positions_by_node,
        minimum,
    )
    if minimum_unique is None and required and minimum > 0:
        impossible_reasons.append("joint_required_mutation_set_unreachable")


    overlap_exists = False
    for index, node in enumerate(required):
        for other in required[index + 1 :]:
            if executable_positions_by_node.get(node, set()) & executable_positions_by_node.get(other, set()):
                overlap_exists = True
                break
        if overlap_exists:
            break
    constructive_depth = int(sum(minimum_steps_by_node.values()))
    depth_lower_bound = (
        max(minimum_steps_by_node.values(), default=0)
        if overlap_exists
        else constructive_depth
    )
    maximum_depth = max(0, int(getattr(cfg, "mcts_max_depth", 0) or 0))
    iterations = max(0, int(getattr(cfg, "iterations", 0) or 0))
    if depth_lower_bound > maximum_depth:
        impossible_reasons.append("mcts_max_depth_below_joint_coverage_lower_bound")
    if depth_lower_bound > iterations:
        impossible_reasons.append("requested_expansions_below_joint_coverage_lower_bound")
    mutation_budget = max(0, int(getattr(cfg, "max_total_mutations", 0) or 0))
    if (
        mutation_budget > 0
        and minimum_unique is not None
        and int(minimum_unique) > mutation_budget
    ):
        impossible_reasons.append("max_total_mutations_below_joint_coverage_lower_bound")

    enabled = bool(
        not formal_layered_enabled
        and (hard_enabled or formal_joint_enabled)
        and declared_required
        and minimum > 0
    )
    if formal_layered_enabled:
        status = "delegated_to_shortlist_set"
    elif not enabled:
        status = "not_required"
    elif impossible_reasons:
        status = "impossible"
    else:
        status = "pending"

    activation = {
        "semantic_hard_coverage": bool(hard_enabled),
        "formal_joint_novel_shortlist": bool(formal_joint_enabled),
    }
    if formal_layered_enabled:
        activation["formal_layered_novel_shortlist"] = True
    report = {
        "schema_version": JOINT_COVERAGE_COMPLETION_VERSION,
        "enabled": enabled,
        "status": status,
        "pass": bool(
            status in {"not_required", "complete", "delegated_to_shortlist_set"}
        ),
        "basis": (
            "shortlist_set_mutation_coverage_vs_immutable_template"
            if formal_layered_enabled
            else "single_final_sequence_mutation_coverage_vs_immutable_template"
        ),
        "activation": activation,
        "required_nodes": declared_required,
        "active_nodes": active_nodes,
        "anchor_nodes": anchor_nodes,
        "anchor_source": anchor_source,
        "designable_required_nodes": required,
        "unavailable_required_nodes": unavailable,
        "minimum_mutations_per_node": int(minimum),
        "mapping_schedule_active": mapping_enabled,
        "mapping_action_ids_by_node": action_ids_by_node,
        "executable_position_counts_by_node": {
            node: len(executable_positions_by_node.get(node, set()))
            for node in declared_required
        },
        "max_step_mutation_capacity_by_node": max_step_capacity,
        "minimum_expansion_steps_by_node": minimum_steps_by_node,
        "minimum_unique_mutations_lower_bound": minimum_unique,
        "minimum_depth_lower_bound": int(depth_lower_bound),
        "constructive_reserved_expansions": int(constructive_depth),
        "reserved_expansion_budget": int(iterations),
        "mcts_max_depth": int(maximum_depth),
        "max_total_mutations": int(mutation_budget),
        "overlapping_required_node_positions": bool(overlap_exists),
        "feasibility": (
            {
                "status": "delegated_to_shortlist_set",
                "reasons": [],
            }
            if formal_layered_enabled
            else {
                "status": "impossible" if impossible_reasons else "declared_bounds_feasible",
                "reasons": list(dict.fromkeys(impossible_reasons)),
            }
        ),
        "prefix_node_id": "root",
        "prefix_depth": 0,
        "prefix_mutations_by_node": {
            node: 0 for node in declared_required
        },
        "covered_nodes": [],
        "missing_nodes": list(declared_required),
        "attempted_expansion_rounds": 0,
        "effective_expansion_rounds": 0,
        "stalled_expansion_rounds": 0,
        "attempts_by_node": {node: 0 for node in declared_required},
        "attempt_log": [],
        "completed_at_expansion_round": None,
        "completion_candidate_id": None,
        "underfill_reasons": (
            []
            if formal_layered_enabled
            else list(dict.fromkeys(impossible_reasons))
        ),
    }
    if formal_layered_enabled:
        report.update(
            {
                "coverage_scope": "shortlist_set",
                "individual_joint_prefix_required": False,
                "delegated_required_nodes": list(active_nodes),
                "delegated_anchor_nodes": list(anchor_nodes),
            }
        )
    return report


def joint_coverage_expansion_directive(
    report: Dict[str, Any],
    *,
    designable_by_name: Mapping[str, int],
    mapping_actions: Sequence[Mapping[str, Any]],
    tree: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:


    if not bool(report.get("enabled")) or report.get("status") != "pending":
        return None
    missing = [str(node) for node in report.get("missing_nodes", []) or []]
    if not missing:
        return None
    parent_id = str(report.get("prefix_node_id") or "root")
    parent = tree.get(parent_id)
    if not isinstance(parent, Mapping):
        report["status"] = "underfilled"
        report["pass"] = False
        report.setdefault("underfill_reasons", []).append(
            "coverage_prefix_node_missing_from_tree"
        )
        return None
    if int(parent.get("depth", 0) or 0) >= int(report.get("mcts_max_depth", 0) or 0):
        report["status"] = "underfilled"
        report["pass"] = False
        report.setdefault("underfill_reasons", []).append(
            "coverage_prefix_reached_mcts_max_depth"
        )
        return None

    target = missing[0]
    if target not in designable_by_name:
        report["status"] = "underfilled"
        report["pass"] = False
        report.setdefault("underfill_reasons", []).append(
            f"coverage_target_not_designable:{target}"
        )
        return None

    mapping_action: Optional[Dict[str, Any]] = None
    if bool(report.get("mapping_schedule_active")):
        choices = [
            dict(action)
            for action in mapping_actions
            if str(action.get("compiled_segment_name") or "") == target
        ]
        choices.sort(key=lambda item: str(item.get("action_id") or ""))
        if not choices:
            report["status"] = "underfilled"
            report["pass"] = False
            report.setdefault("underfill_reasons", []).append(
                f"coverage_target_without_runtime_mapping_action:{target}"
            )
            return None
        attempt_index = int((report.get("attempts_by_node") or {}).get(target, 0) or 0)
        mapping_action = choices[attempt_index % len(choices)]
        expansion_key = str(mapping_action.get("action_id") or f"segment:{target}")
    else:
        expansion_key = f"segment:{target}"

    return {
        "target_node": target,
        "segment_index": int(designable_by_name[target]),
        "parent_id": parent_id,
        "expansion_key": expansion_key,
        "mapping_action": mapping_action,
        "selection": {
            "source": "joint_required_node_coverage_completion",
            "required_node": target,
            "pending_required_nodes": missing,
            "parent_policy": "deterministic_joint_coverage_prefix",
            "coverage_prefix_node_id": parent_id,
            "coverage_prefix_depth": int(parent.get("depth", 0) or 0),
        },
    }


def _covered_nodes(
    coverage: Mapping[str, Any], required_nodes: Sequence[str], minimum: int
) -> List[str]:
    counts = coverage.get("mutations_by_node")
    counts = counts if isinstance(counts, Mapping) else {}
    return [
        node
        for node in required_nodes
        if int(counts.get(node, 0) or 0) >= int(minimum)
    ]


def update_joint_coverage_completion(
    report: Dict[str, Any],
    *,
    directive: Mapping[str, Any],
    new_candidates: Sequence[Mapping[str, Any]],
    tree: Mapping[str, Mapping[str, Any]],
    expansion_round: int,
    effective: bool,
) -> None:


    if report.get("status") != "pending":
        return
    target = str(directive.get("target_node") or "")
    report["attempted_expansion_rounds"] = int(
        report.get("attempted_expansion_rounds", 0) or 0
    ) + 1
    attempts_by_node = report.setdefault("attempts_by_node", {})
    attempts_by_node[target] = int(attempts_by_node.get(target, 0) or 0) + 1
    if effective:
        report["effective_expansion_rounds"] = int(
            report.get("effective_expansion_rounds", 0) or 0
        ) + 1
    else:
        report["stalled_expansion_rounds"] = int(
            report.get("stalled_expansion_rounds", 0) or 0
        ) + 1

    required = [str(node) for node in report.get("required_nodes", []) or []]
    minimum = int(report.get("minimum_mutations_per_node", 1) or 0)
    prior_counts = {
        node: min(
            minimum,
            int(
                (report.get("prefix_mutations_by_node") or {}).get(node, 0)
                or 0
            ),
        )
        for node in required
    }
    prior_progress = sum(prior_counts.values())
    progress_options: List[
        Tuple[int, int, float, float, str, str, int, List[str], Dict[str, int], bool]
    ] = []
    for candidate_index, candidate in enumerate(new_candidates):
        fast_filter = candidate.get("fast_filter") or {}
        fast_pass = bool(fast_filter.get("pass", True))
        bootstrap_expandable = bool(candidate.get("bootstrap_expandable", False))
        if not fast_pass and not bootstrap_expandable:
            continue
        coverage = candidate.get("semantic_final_coverage")
        if not isinstance(coverage, Mapping):
            continue
        raw_counts = coverage.get("mutations_by_node")
        raw_counts = raw_counts if isinstance(raw_counts, Mapping) else {}
        candidate_counts = {
            node: min(minimum, int(raw_counts.get(node, 0) or 0))
            for node in required
        }
        covered = _covered_nodes(coverage, required, minimum)
        if any(candidate_counts[node] < prior_counts[node] for node in required):
            continue
        candidate_progress = sum(candidate_counts.values())
        if candidate_progress <= prior_progress:
            continue
        node_id = str(
            candidate.get("transposition_target")
            if candidate.get("duplicate_sequence")
            else candidate.get("variant_id")
        )
        if node_id not in tree:
            continue
        progress_options.append(
            (
                -candidate_progress,
                -candidate_counts.get(target, 0),
                -float(fast_filter.get("search_progress", 0.0) or 0.0),
                float(candidate.get("fast_loss", 1e18)),
                str(candidate.get("seq_hash") or ""),
                node_id,
                int(candidate_index),
                covered,
                candidate_counts,
                fast_pass,
            )
        )

    event: Dict[str, Any] = {
        "expansion_round": int(expansion_round) + 1,
        "target_node": target,
        "parent_id": str(directive.get("parent_id") or ""),
        "expansion_key": str(directive.get("expansion_key") or ""),
        "candidate_count": len(new_candidates),
        "effective_expansion": bool(effective),
    }
    if progress_options:
        (
            _neg_count,
            _target_rank,
            _neg_search_progress,
            _loss,
            _seq_hash,
            node_id,
            _candidate_index,
            covered,
            candidate_counts,
            selected_fast_pass,
        ) = min(progress_options)
        report["prefix_node_id"] = node_id
        report["prefix_depth"] = int((tree.get(node_id) or {}).get("depth", 0) or 0)
        report["prefix_mutations_by_node"] = candidate_counts
        report["covered_nodes"] = [node for node in required if node in set(covered)]
        report["missing_nodes"] = [
            node for node in required if node not in set(report["covered_nodes"])
        ]
        event.update(
            {
                "outcome": "prefix_advanced",
                "selected_prefix_node_id": node_id,
                "prefix_mutations_by_node": dict(candidate_counts),
                "covered_nodes": list(report["covered_nodes"]),
                "missing_nodes": list(report["missing_nodes"]),
                "selected_candidate_fast_gate_pass": bool(selected_fast_pass),
                "bootstrap_path_only": not bool(selected_fast_pass),
            }
        )
        if not report["missing_nodes"]:
            report["status"] = "complete"
            report["pass"] = True
            report["completed_at_expansion_round"] = int(expansion_round) + 1
            report["completion_candidate_id"] = node_id
    else:
        moves = [candidate.get("move") or {} for candidate in new_candidates]
        if not new_candidates:
            reason = "no_candidate_emitted"
        elif all(bool(candidate.get("duplicate_sequence")) for candidate in new_candidates):
            reason = "duplicate_only_without_coverage_progress"
        elif all(not bool(move.get("changes")) for move in moves):
            reason = "noop_only_without_coverage_progress"
        elif all(
            not bool((candidate.get("fast_filter") or {}).get("pass", True))
            for candidate in new_candidates
        ):
            reason = "fast_filter_rejected_all_completion_candidates"
        else:
            reason = "target_node_mutation_not_realized_on_prefix"
        event.update({"outcome": "stalled", "reason": reason})

    report.setdefault("attempt_log", []).append(event)


def finalize_joint_coverage_completion(report: Dict[str, Any]) -> Dict[str, Any]:


    if bool(report.get("enabled")) and report.get("status") == "pending":
        report["status"] = "underfilled"
        report["pass"] = False
        reasons = [
            str(event.get("reason"))
            for event in report.get("attempt_log", []) or []
            if event.get("outcome") == "stalled" and event.get("reason")
        ]
        reasons.append("requested_expansion_budget_exhausted_before_joint_coverage")
        report["underfill_reasons"] = list(
            dict.fromkeys(
                [str(value) for value in report.get("underfill_reasons", []) or []]
                + reasons
            )
        )
    else:
        report["underfill_reasons"] = list(
            dict.fromkeys(str(value) for value in report.get("underfill_reasons", []) or [])
        )
    return report


def attach_joint_coverage_reporting(
    round_summary: Dict[str, Any],
    completion: Mapping[str, Any],
    expansion_accounting: Mapping[str, Any],
) -> None:


    round_summary["joint_coverage_completion"] = dict(completion)
    round_summary["expansion_round_accounting"] = dict(expansion_accounting)
    search_health = round_summary.get("search_health")
    if not isinstance(search_health, dict):
        return
    search_health["joint_coverage_completion"] = {
        "status": completion.get("status"),
        "pass": bool(completion.get("pass")),
        "coverage_scope": completion.get(
            "coverage_scope", "individual_sequence"
        ),
        "required_nodes": list(completion.get("required_nodes", []) or []),
        "active_nodes": list(completion.get("active_nodes", []) or []),
        "anchor_nodes": list(completion.get("anchor_nodes", []) or []),
        "covered_nodes": list(completion.get("covered_nodes", []) or []),
        "missing_nodes": list(completion.get("missing_nodes", []) or []),
        "completion_candidate_id": completion.get("completion_candidate_id"),
        "underfill_reasons": list(completion.get("underfill_reasons", []) or []),
    }


__all__ = [
    "JOINT_COVERAGE_COMPLETION_VERSION",
    "attach_joint_coverage_reporting",
    "finalize_joint_coverage_completion",
    "initialize_joint_coverage_completion",
    "joint_coverage_expansion_directive",
    "update_joint_coverage_completion",
]



from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, MutableMapping

from astevolve.domain.strategy import EDIT_CONTRACT_ACTIONS, EditContract


ACTION_TO_POLICY = {
    "optimize_node": {"priority_boost": 1.35, "mutation_rate": 0.075, "max_mutations_per_step": 4},
    "repair_node": {"priority_boost": 1.05, "mutation_rate": 0.025, "max_mutations_per_step": 2},
    "freeze_node": {"priority_boost": 0.2, "mutation_rate": 0.0, "max_mutations_per_step": 0},
    "expand_edit_scope": {"priority_boost": 1.2, "mutation_rate": 0.065, "max_mutations_per_step": 5},
    "increase_negative_design_weight": {"priority_boost": 1.45, "mutation_rate": 0.07, "max_mutations_per_step": 4},
    "modify_ast_node_definitions": {"priority_boost": 1.25, "mutation_rate": 0.055, "max_mutations_per_step": 4},
    "redesign_constraints": {"priority_boost": 1.1, "mutation_rate": 0.04, "max_mutations_per_step": 3},
    "adjust_search_space": {"priority_boost": 1.15, "mutation_rate": 0.06, "max_mutations_per_step": 5},
}

if frozenset(ACTION_TO_POLICY) != EDIT_CONTRACT_ACTIONS:
    raise RuntimeError("edit-contract action schema and policy registry are out of sync")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _first_action(item: Mapping[str, Any]) -> str:
    actions = item.get("recommended_actions")
    if isinstance(actions, Mapping) and actions:
        return str(max(actions.items(), key=lambda kv: kv[1])[0])
    return str(item.get("recommended_action") or "optimize_node")


def _target_names(state: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    target = state.get("target") if isinstance(state.get("target"), Mapping) else {}
    additional = state.get("additional_targets") if isinstance(state.get("additional_targets"), Mapping) else {}
    positives = [str(target.get("name") or target.get("id") or "target_peptide")] if target else ["target_peptide"]
    negatives = []
    for key, value in additional.items():
        if isinstance(value, Mapping):
            negatives.append(str(value.get("name") or key))
        else:
            negatives.append(str(key))
    if not negatives:
        negatives = ["source_peptide"]
    return positives, negatives


def _mutation_budget_for_action(action: str) -> Dict[str, int]:
    if action == "freeze_node":
        return {"min": 0, "max": 0}
    if action == "expand_edit_scope":
        return {"min": 2, "max": 6}
    return {
        "min": 1,
        "max": int(ACTION_TO_POLICY[action]["max_mutations_per_step"]),
    }


def generate_edit_contract(
    diagnosis: Mapping[str, Any],
    design_state: Mapping[str, Any],
    case_sheet: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    bottleneck_functional = _as_list(diagnosis.get("bottleneck_functional_nodes"))
    bottleneck_structural = _as_list(diagnosis.get("bottleneck_structural_nodes"))
    primary = bottleneck_functional[0] if bottleneck_functional and isinstance(bottleneck_functional[0], Mapping) else {}
    primary_functional = str(primary.get("node") or "ligand_recognition")
    action = _first_action(primary)

    structural_nodes: List[str] = []
    for item in bottleneck_structural:
        if isinstance(item, Mapping) and item.get("node"):
            try:
                severity = float(item.get("severity") or 0.0)
            except (TypeError, ValueError):
                severity = 0.0
            if severity <= 1e-8:
                continue
            structural_nodes.append(str(item["node"]))
    if not structural_nodes:
        design_points = design_state.get("design_points") if isinstance(design_state.get("design_points"), Mapping) else {}
        structural_nodes = [str(x) for x in _as_list(design_points.get("primary_design_nodes"))]
    structural_nodes = structural_nodes[:5]

    case_sheet = case_sheet or {}
    constraints = case_sheet.get("residue_level_constraints") if isinstance(case_sheet.get("residue_level_constraints"), Mapping) else {}
    frozen_nodes = [str(x) for x in _as_list(constraints.get("frozen_nodes"))]
    if not frozen_nodes:
        design_points = design_state.get("design_points") if isinstance(design_state.get("design_points"), Mapping) else {}
        frozen_nodes = [str(x) for x in _as_list(design_points.get("preserved_nodes"))]
    frozen_nodes = _unique([node for node in frozen_nodes if node])[:10]
    structural_nodes = [
        node for node in _unique(structural_nodes) if node not in set(frozen_nodes)
    ][:5]
    positives, negatives = _target_names(design_state)

    evaluator_focus = []
    for item in bottleneck_structural[:3]:
        if isinstance(item, Mapping):
            evaluator_focus.extend(str(term) for term in _as_list(item.get("terms")))
    evaluator_focus = sorted(set(evaluator_focus))[:8]
    negative_plan = [dict(item) for item in _as_list(diagnosis.get("negative_design_position_plan")) if isinstance(item, Mapping)]
    experiment_report = diagnosis.get("experiment_analysis_report") if isinstance(diagnosis.get("experiment_analysis_report"), Mapping) else {}
    ast_recommendations = [
        dict(item)
        for item in _as_list(experiment_report.get("ast_level_recommendations"))
        if isinstance(item, Mapping)
    ]
    if ast_recommendations and action == "optimize_node":
        first_action = str(ast_recommendations[0].get("action") or "")
        if first_action in {"narrow_or_retype_failure_nodes", "expand_successful_structural_motifs"}:
            action = "modify_ast_node_definitions"
        elif first_action == "rebalance_required_node_coverage":
            action = "adjust_search_space"
        elif first_action == "repair_constraint_design":
            action = "redesign_constraints"
    budget = _mutation_budget_for_action(action)
    contract = {
        "schema_version": "ast_edit_contract_v2",
        "action": action,
        "required_nodes": structural_nodes,
        "forbidden_nodes": frozen_nodes,
        "mutation_budget": budget,
        "rationale": {
            "bottleneck_functional_nodes": bottleneck_functional[:3],
            "bottleneck_structural_nodes": bottleneck_structural[:5],
            "hard_gate_pass": diagnosis.get("hard_gate_pass", True),
            "disqualification_reasons": diagnosis.get("disqualification_reasons", []),
        },
        "metadata": {
            "source": "semantic_graph_diagnosis",
            "primary_functional_node": primary_functional,
            "primary_design_node": structural_nodes[0] if structural_nodes else None,
            "functional_goal": _functional_goal(primary_functional, action),
            "positive_targets": positives,
            "negative_targets": negatives,
            "evaluator_focus": evaluator_focus,
            "negative_design_position_plan": negative_plan[:12],
            "candidate_comparison": diagnosis.get("candidate_comparison", {}),
            "experiment_analysis_report": experiment_report,
            "ast_level_recommendations": ast_recommendations[:8],
            "allowed_ast_edit_scope": [
                "layout_plan.secondary_structure_priors",
                "layout_plan.semantic_focus",
                "layout_plan.design_regions",
                "node definitions and bind_to mappings inside EVOLVE-BLOCK",
                "constraint emphasis and required_structural_nodes",
                "search_schedule and proposal fractions",
            ],
        },
    }
    return EditContract.from_mapping(contract).to_dict()


def _functional_goal(functional_node: str, action: str) -> str:
    goals = {
        "modify_ast_node_definitions": "change_ast_node_boundaries_or_functional_mappings",
        "redesign_constraints": "change_constraint_design_or_guardrails",
        "adjust_search_space": "rebalance_search_space_and_required_node_coverage",
        "increase_negative_design_weight": "strengthen_negative_design",
        "repair_node": "repair_declared_functional_constraint",
        "freeze_node": "preserve_declared_functional_constraint",
        "expand_edit_scope": "expand_declared_functional_search_scope",
        "optimize_node": "improve_declared_functional_objective",
    }
    return goals.get(action, f"address_declared_function:{functional_node}")


def apply_edit_contract_to_strategy(strategy: Mapping[str, Any]) -> Dict[str, Any]:


    updated = deepcopy(dict(strategy))
    contract = updated.get("edit_contract")
    if "edit_contract" not in updated or contract is None:
        return updated
    canonical = EditContract.from_mapping(contract).to_dict()
    updated["edit_contract"] = canonical
    nodes = list(canonical["required_nodes"])
    if nodes:
        current_order = [str(x) for x in _as_list(updated.get("preferred_edit_order")) if str(x)]
        updated["preferred_edit_order"] = _unique(nodes + current_order)
    policies: MutableMapping[str, Any] = dict(updated.get("node_edit_policies") or {})
    action = canonical["action"]
    policy_patch = ACTION_TO_POLICY[action]
    max_mut = canonical["mutation_budget"]["max"]
    for node in nodes:
        node_policy = dict(policies.get(node) or {})
        node_policy.update(policy_patch)
        node_policy["max_mutations_per_step"] = max_mut
        node_policy["edit_contract_action"] = action
        policies[node] = node_policy
    for node in canonical["forbidden_nodes"]:
        if not str(node):
            continue
        node_policy = dict(policies.get(str(node)) or {})
        node_policy.update(ACTION_TO_POLICY["freeze_node"])
        node_policy["edit_contract_action"] = "freeze_node"
        policies[str(node)] = node_policy
    if policies:
        updated["node_edit_policies"] = dict(policies)
        updated["_tree_policy_active"] = True
        updated["_edit_contract_applied"] = True
    if action == "increase_negative_design_weight":
        score_cfg = dict(updated.get("score_config") or {})
        score_cfg["weight_multistate"] = float(score_cfg.get("weight_multistate", 1.0)) * 1.15
        updated["score_config"] = score_cfg
    return updated


def _unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out

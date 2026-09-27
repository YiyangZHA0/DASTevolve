

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

EXECUTABLE_ISLAND_DIRECTIVE_VERSION = "astevolve.executable_island_directive.v1"
EXECUTABLE_ISLAND_SET_VERSION = "astevolve.executable_island_set.v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _specialty(role_id: str, island: int) -> str:
    text = role_id.lower()
    if any(token in text for token in ("fold", "interface", "global")):
        return "global_fold_rescue"
    if any(token in text for token in ("contact", "pose", "selectivity", "margin")):
        return "target_contact_pose_rescue"
    if any(token in text for token in ("region", "novel", "explor")):
        return "node_window_novelty"
    if any(token in text for token in ("robust", "safety", "pareto")):
        return "robust_pareto"
    return ("global_fold_rescue", "target_contact_pose_rescue", "node_window_novelty", "robust_pareto")[island % 4]


_PROFILES: Dict[str, Dict[str, Any]] = {
    "global_fold_rescue": {"portfolio": [2, 1, 4, 1], "fractions": [0.35, 0.10, 0.55], "temperature": 0.55, "depth": 4, "budget": [1, 8], "migrants": ["feasible_anchor", "robust_pareto"]},
    "target_contact_pose_rescue": {"portfolio": [3, 2, 2, 1], "fractions": [0.55, 0.20, 0.25], "temperature": 0.70, "depth": 5, "budget": [2, 12], "migrants": ["feasible_anchor", "contact_pose"]},
    "node_window_novelty": {"portfolio": [2, 1, 1, 4], "fractions": [0.20, 0.65, 0.15], "temperature": 1.10, "depth": 7, "budget": [3, 20], "migrants": ["feasible_anchor", "novel_module"]},
    "robust_pareto": {"portfolio": [2, 2, 3, 1], "fractions": [0.40, 0.15, 0.45], "temperature": 0.50, "depth": 4, "budget": [1, 10], "migrants": ["pareto_non_dominated"]},
}


def compile_executable_island_directives(strategy: Mapping[str, Any], roles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(roles) != 4:
        raise ValueError("executable island mode requires exactly four islands")
    supplied = {int(row["island"]): row for row in strategy.get("island_directives", []) if isinstance(row, Mapping) and isinstance(row.get("island"), int)}
    directives = []
    for island, role in enumerate(roles):
        role_id = str(role.get("role_id") or f"island_{island}")
        specialty = _specialty(role_id, island)
        profile = deepcopy(_PROFILES[specialty])
        source = supplied.get(island, {})
        portfolio = dict(zip(("primary", "secondary", "repair", "novelty_control"), profile["portfolio"]))
        fractions = dict(zip(("exploit", "explore", "repair"), profile["fractions"]))
        material = {
            "schema_version": EXECUTABLE_ISLAND_DIRECTIVE_VERSION,
            "island": island, "role_id": role_id, "specialty": specialty,
            "objective": str(source.get("objective") or role.get("focus") or specialty),
            "priority_metrics": [str(item) for item in source.get("priority_metrics", role.get("soft_objectives", []))],
            "architecture_targets": [str(item) for item in source.get("architecture_targets", [])],
            "portfolio_quotas": portfolio, "proposal_tier_fractions": fractions,
            "mcts_temperature": profile["temperature"], "mcts_max_depth": profile["depth"],
            "mutation_count_range": profile["budget"],
            "provider_quotas": {
                "classic_esmfold": "all_inner_candidates",
                "esmfold2": 10,
                "protenix": 10,
                "alphafold3": 4,
                "pyrosetta": 2,
            },
            "required_ablation": str(source.get("required_ablation") or "matched parent control"),
            "accepted_migrant_types": profile["migrants"],
            "parent_pool_scope": f"island:{island}:independent",
        }
        directives.append({**material, "directive_hash": _hash("executable_island_directive_sha256:", material)})
    if len({row["directive_hash"] for row in directives}) != 4:
        raise ValueError("four island effective directive hashes must be distinct")
    material = {"schema_version": EXECUTABLE_ISLAND_SET_VERSION, "directives": directives, "migration_source_policy": "pareto_non_dominated_only", "migration_rescore_policy": "target_island_acquisition", "global_best_retention": "immutable_history"}
    return {**material, "directive_set_hash": _hash("executable_island_set_sha256:", material)}


def directive_for_role(role_id: str, island: int = 0) -> Dict[str, Any]:
    roles = [{"role_id": f"placeholder_{index}"} for index in range(4)]
    roles[island % 4] = {"role_id": str(role_id)}
    return compile_executable_island_directives({}, roles)["directives"][island % 4]


def apply_executable_island_directive(cfg: Any, role_id: str, island: int = 0) -> Dict[str, Any]:
    if not bool(getattr(cfg, "executable_island_policy_enabled", False)):
        return {"enabled": False, "role_id": role_id, "reason": "compatibility_disabled"}
    directive = directive_for_role(role_id, island)
    fractions = directive["proposal_tier_fractions"]
    cfg.proposal_tier_mode = "mixed"
    cfg.proposal_exploit_frac, cfg.proposal_explore_frac, cfg.proposal_repair_frac = fractions["exploit"], fractions["explore"], fractions["repair"]
    cfg.node_optimizer_temperature = directive["mcts_temperature"]
    cfg.mcts_max_depth = directive["mcts_max_depth"]
    cfg.max_total_mutations = directive["mutation_count_range"][1]
    cfg.structure_prescreen_max_candidates = directive["provider_quotas"]["esmfold2"]
    cfg.structure_screen_max_candidates = directive["provider_quotas"]["protenix"]
    cfg.structure_rerank_max_candidates = directive["provider_quotas"]["alphafold3"]
    cfg.structure_physics_max_candidates = directive["provider_quotas"]["pyrosetta"]
    setattr(cfg, "effective_island_directive", deepcopy(directive))
    return {"enabled": True, "directive": directive, "directive_hash": directive["directive_hash"]}


__all__ = ["apply_executable_island_directive", "compile_executable_island_directives", "directive_for_role"]



from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def causal_identity(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    return {
        key: context.get(key)
        for key in (
            "generation_id", "proposal_id", "trial_id", "seed",
            "graph_patch_hash", "effective_contract_hash",
            "island_id", "island_role",


            "design_action_hash", "compiled_design_action_hash",
            "case_id", "parent_program_id", "parent_candidate_id",
            "parent_sequence_bundle_hash",
            "parent_effective_contract_hash", "parent_evolve_hash",
            "compiled_portfolio_request_hash",
        )
        if context.get(key) not in (None, "")
    }


def bind_move(
    move: Dict[str, Any], context: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    identity = causal_identity(context)
    if identity:
        move["causal_context"] = dict(identity)
    return move


def bind_candidate(
    candidate: Dict[str, Any], context: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    identity = causal_identity(context)
    if identity:
        candidate["causal_context"] = dict(identity)
        if isinstance(candidate.get("move"), dict):
            bind_move(candidate["move"], context)
    return candidate


def build_runtime(
    *,
    context: Optional[Mapping[str, Any]],
    root: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    structure_selected: Optional[Dict[str, Any]],
    fast_selected_variant_id: str,
) -> Optional[Dict[str, Any]]:
    identity = causal_identity(context)
    if not identity:
        return None
    by_id = {
        str(item.get("variant_id")): item
        for item in [root, *candidates]
        if item.get("variant_id") is not None
    }
    selected = structure_selected or by_id.get(fast_selected_variant_id, root)
    return {
        "identity": identity,
        "root_candidate": root,
        "candidates": candidates,
        "selected_candidate": selected,
        "selection_source": "structure" if structure_selected else "fast",
    }


__all__ = ["bind_candidate", "bind_move", "build_runtime", "causal_identity"]

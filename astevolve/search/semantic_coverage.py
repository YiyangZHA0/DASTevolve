

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astevolve.search.config import SAConfig


def _designable_segments(
    compiled: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> List[Tuple[Any, List[int]]]:


    out: List[Tuple[Any, List[int]]] = []
    for seg in compiled["segments"]:
        mask = masks.get(seg.chain_id)
        if mask is None:
            continue
        positions = [
            int(i)
            for i in seg.indices()
            if 0 <= int(i) < len(mask) and bool(mask[int(i)])
        ]
        if positions:
            out.append((seg, positions))
    return out


def _unique_strings(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [item.strip() for item in values.replace(";", ",").split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw = [str(item).strip() for item in values]
    else:
        raw = [str(values).strip()]
    out: List[str] = []
    for item in raw:
        if item and item not in out:
            out.append(item)
    return out


def _semantic_required_nodes(
    cfg: SAConfig,
    designable: Optional[List[Tuple[Any, List[int]]]] = None,
) -> List[str]:
    required = _unique_strings(getattr(cfg, "semantic_required_nodes", []))
    if designable is None:
        return required
    designable_names = {str(getattr(seg, "name", "")) for seg, _ in designable}
    return [node for node in required if node in designable_names]


def _semantic_required_unavailable_nodes(
    cfg: SAConfig,
    designable: List[Tuple[Any, List[int]]],
) -> List[str]:
    required = _unique_strings(getattr(cfg, "semantic_required_nodes", []))
    designable_names = {str(getattr(seg, "name", "")) for seg, _ in designable}
    return [node for node in required if node not in designable_names]


def _designable_index_by_node(designable: List[Tuple[Any, List[int]]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, (seg, _) in enumerate(designable):
        name = str(getattr(seg, "name", ""))
        if name:
            out.setdefault(name, []).append(int(idx))
    return out


def _semantic_min_visits(cfg: SAConfig) -> int:
    try:
        return max(0, int(getattr(cfg, "semantic_required_node_min_visits", 1) or 0))
    except (TypeError, ValueError):
        return 1


def _semantic_min_mutations(cfg: SAConfig) -> int:
    try:
        return max(0, int(getattr(cfg, "semantic_required_node_min_mutations", 1) or 0))
    except (TypeError, ValueError):
        return 1


def _semantic_coverage_hard_enabled(cfg: SAConfig) -> bool:
    return str(getattr(cfg, "semantic_coverage_mode", "soft") or "soft").lower() in {
        "hard",
        "gate",
        "required",
        "strict",
    }


def _semantic_force_steps(cfg: SAConfig) -> int:
    try:
        return max(0, int(getattr(cfg, "semantic_required_node_force_steps", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _select_semantic_segment(
    designable: List[Tuple[Any, List[int]]],
    norm_priors: np.ndarray,
    cfg: SAConfig,
    history: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[int, Dict[str, Any]]:
    required = _semantic_required_nodes(cfg, designable)
    index_by_node = _designable_index_by_node(designable)
    min_visits = _semantic_min_visits(cfg)
    min_mutations = _semantic_min_mutations(cfg)
    visit_counts = {
        str(k): int(v or 0)
        for k, v in (history.get("semantic_required_node_visits") or history.get("node_visit_counts") or {}).items()
    }
    mutation_counts = {
        str(k): int(v or 0)
        for k, v in (history.get("semantic_required_node_mutations") or {}).items()
    }

    pending: List[str] = []
    for node in required:
        if node not in index_by_node:
            continue
        needs_visit = min_visits > 0 and int(visit_counts.get(node, 0)) < min_visits
        needs_mutation = min_mutations > 0 and int(mutation_counts.get(node, 0)) < min_mutations
        if needs_visit or needs_mutation:
            pending.append(node)

    step = int(history.get("step", 0) or 0)
    force_steps = _semantic_force_steps(cfg)
    force_required = force_steps > 0 and step < force_steps and bool(required)
    forced_pending = [node for node in required if node in index_by_node]
    if force_required and not pending:
        pending = forced_pending

    if pending:
        if bool(getattr(cfg, "semantic_required_node_round_robin", True)):
            order = {node: idx for idx, node in enumerate(required)}
            node = sorted(
                pending,
                key=lambda n: (
                    int(mutation_counts.get(n, 0)),
                    int(visit_counts.get(n, 0)),
                    order.get(n, 9999),
                ),
            )[0]
        else:
            node = pending[int(rng.integers(0, len(pending)))]
        idxs = index_by_node.get(node, [])
        if idxs:
            weights = np.asarray([float(norm_priors[i]) for i in idxs], dtype=float)
            total = float(weights.sum())
            probs = weights / total if total > 0.0 and np.isfinite(total) else None
            selected = int(rng.choice(np.asarray(idxs), p=probs))
            return selected, {
                "source": "semantic_required_node",
                "required_node": node,
                "pending_required_nodes": list(pending),
                "min_visits": int(min_visits),
                "min_mutations": int(min_mutations),
                "force_steps": int(force_steps),
                "force_active": bool(force_required),
            }

    selected = int(rng.choice(len(designable), p=norm_priors))
    return selected, {
        "source": "prior_sampling",
        "required_nodes": list(required),
        "min_visits": int(min_visits),
        "min_mutations": int(min_mutations),
        "force_steps": int(force_steps),
        "force_active": bool(force_required),
    }


def _semantic_coverage_report(cfg: SAConfig, history: Optional[Dict[str, Any]]) -> Dict[str, Any]:


    history = history or {}
    required = _unique_strings(
        history.get("semantic_required_nodes", getattr(cfg, "semantic_required_nodes", []))
    )
    designable_required = _unique_strings(history.get("semantic_designable_required_nodes", required))
    unavailable = _unique_strings(history.get("semantic_unavailable_required_nodes", []))
    min_visits = _semantic_min_visits(cfg)
    min_mutations = _semantic_min_mutations(cfg)
    visit_counts = {
        str(k): int(v or 0)
        for k, v in (history.get("semantic_required_node_visits") or history.get("node_visit_counts") or {}).items()
    }
    mutation_counts = {
        str(k): int(v or 0)
        for k, v in (history.get("semantic_required_node_mutations") or {}).items()
    }
    missing_visit = [
        node for node in designable_required if min_visits > 0 and int(visit_counts.get(node, 0)) < min_visits
    ]
    missing_mutation = [
        node
        for node in designable_required
        if min_mutations > 0 and int(mutation_counts.get(node, 0)) < min_mutations
    ]
    passed = not missing_visit and not missing_mutation and not unavailable
    return {
        "schema_version": "ast_semantic_coverage_v1",
        "mode": str(getattr(cfg, "semantic_coverage_mode", "soft") or "soft"),
        "hard_gate_enabled": bool(_semantic_coverage_hard_enabled(cfg)),
        "pass": bool(passed),
        "required_nodes": required,
        "designable_required_nodes": designable_required,
        "unavailable_required_nodes": unavailable,
        "min_visits": int(min_visits),
        "min_mutations": int(min_mutations),
        "visits_by_node": {node: int(visit_counts.get(node, 0)) for node in required},
        "mutations_by_node": {node: int(mutation_counts.get(node, 0)) for node in required},
        "missing_required_nodes_by_visit": missing_visit,
        "missing_required_nodes_by_mutation": missing_mutation,
    }

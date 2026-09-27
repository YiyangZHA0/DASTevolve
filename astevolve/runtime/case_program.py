

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from astevolve.runtime.case_context import current_case_kwargs
from engine.case_builder import build_case_inputs, run_design_search


BaseStrategyFactory = Callable[[], Dict[str, Any]]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def merge_locked_runtime_defaults(
    proposed_strategy: Any,
    locked_defaults: Mapping[str, Any],
    base_strategy_factory: BaseStrategyFactory,
) -> Dict[str, Any]:


    merged = base_strategy_factory()
    if isinstance(proposed_strategy, dict):
        merged.update(proposed_strategy)
    merged.update(deepcopy(dict(locked_defaults)))
    return merged


def run_case_search(
    case_id: str,
    strategy: Dict[str, Any],
    *,
    case_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:


    return run_design_search(
        strategy,
        seed=seed,
        **current_case_kwargs(
            case_id,
            case_root=case_root,
            manifest_path=manifest_path,
        ),
    )


def build_runtime_preview(
    case_id: str,
    strategy: Dict[str, Any],
    locked_defaults: Mapping[str, Any],
    *,
    case_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:


    case_kwargs = current_case_kwargs(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    )
    blueprint, constraints, sa_cfg, masks, templates, fixed, score_cfg, state = (
        build_case_inputs(strategy, **case_kwargs)
    )
    compiled = blueprint.compile()
    return {
        "task_name": state["task_name"],
        "chain_lengths": compiled["chain_lengths"],
        "chain_order": compiled["chain_order"],
        "binder_domain_order": state["binder"].get("domain_order", []),
        "layout_summary": state.get("_layout_summary", {}),
        "strategy_schema_report": state.get("_strategy_schema_report", {}),
        "evaluator_plugin_resolution": state.get(
            "_evaluator_plugin_resolution",
            {},
        ),
        "node_policy_names": list(state.get("_node_edit_policies", {}).keys()),
        "locked_runtime_fields": sorted(locked_defaults),
        "complex_states": [
            item.get("name") for item in state.get("complex_states", [])
        ],
        "segments": [
            {
                "chain_id": segment.chain_id,
                "kind": segment.kind,
                "name": segment.name,
                "spans": segment.spans,
                "total_length": segment.total_length,
            }
            for segment in compiled["segments"]
        ],
        "constraint_specs": constraints,
        "sa_config": sa_cfg,
        "score_config": score_cfg,
        "mask_true_counts": {
            chain_id: int(sum(mask)) for chain_id, mask in masks.items()
        },
        "template_lengths": {
            chain_id: len(sequence) for chain_id, sequence in templates.items()
        },
        "fixed_residue_counts": {
            chain_id: len(residues) for chain_id, residues in fixed.items()
        },
    }

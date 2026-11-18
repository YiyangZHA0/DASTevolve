

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from astevolve.search.artifact_io import _write_json
from astevolve.search.config import SAConfig


def attach_structure_evaluation_summary(
    search_artifacts: Dict[str, Any],
    *,
    evaluated: Sequence[Mapping[str, Any]],
    selection_pool: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    cfg: SAConfig,
    best_confidence_metrics: Mapping[str, Any],
    best_structure_metrics: Mapping[str, Any],
    multistate_objectives: Mapping[str, Any],
    selection_decision: Mapping[str, Any] | None,
    parent_baseline_comparison: Mapping[str, Any],
) -> None:


    if not evaluated:
        return
    artifact_path: Path | None = None
    try:
        output_dir = Path(cfg.mcts_output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "structure_evaluated_variants.json"
        _write_json(artifact_path, list(evaluated))
        search_artifacts.setdefault("artifact_paths", {})[
            "structure_evaluated_variants"
        ] = str(artifact_path)
    except Exception:
        artifact_path = None

    evaluated_by_stage: Dict[str, int] = {}
    evaluated_by_provider: Dict[str, int] = {}
    for item in evaluated:
        stage = str(item.get("structure_stage") or "unknown")
        provider = str(item.get("structure_provider") or "unknown")
        evaluated_by_stage[stage] = evaluated_by_stage.get(stage, 0) + 1
        evaluated_by_provider[provider] = (
            evaluated_by_provider.get(provider, 0) + 1
        )
    structure = dict(best_structure_metrics or {})
    search_artifacts["structure_evaluation_summary"] = {
        "evaluated": len(evaluated),
        "selection_pool_size": len(selection_pool),
        "selected_stage": selected.get("structure_stage") if selection_pool else None,
        "selected_provider": selected.get("structure_provider") if selection_pool else None,
        "evaluated_by_stage": evaluated_by_stage,
        "evaluated_by_provider": evaluated_by_provider,
        "best_metrics": dict(best_confidence_metrics or {}),
        "best_interface": structure.get("interface") if structure else None,
        "best_node_summary": structure.get("node_summary") if structure else None,
        "dockq": structure.get("dockq") if structure else None,
        "dockq_proxy": structure.get("dockq_proxy") if structure else None,
        "multistate_objectives": dict(multistate_objectives or {}),
        "selection_decision": dict(selection_decision or {}),
        "parent_baseline_comparison": dict(parent_baseline_comparison),
        "artifact_path": str(artifact_path) if artifact_path else None,
    }


__all__ = ["attach_structure_evaluation_summary"]

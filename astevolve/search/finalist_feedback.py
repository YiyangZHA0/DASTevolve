

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

from astevolve.evaluation.selection import normalize_gate_payload
from astevolve.search.parent_baseline import is_parent_baseline
from astevolve.search.structure_pipeline import _has_structure_signal


STRUCTURE_FINALIST_FEEDBACK_VERSION = (
    "astevolve.structure_finalist_feedback.v1"
)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _direction_agreement(
    proxy_delta: float | None,
    structure_delta: float | None,
) -> bool | None:


    if proxy_delta is None or structure_delta is None:
        return None
    if proxy_delta == 0.0 or structure_delta == 0.0:
        return None
    return bool((proxy_delta < 0.0) == (structure_delta < 0.0))


def _compact_metrics(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    summary = candidate.get("structure_metrics")
    summary = summary if isinstance(summary, Mapping) else {}
    scalar = summary.get("scalar")
    scalar = scalar if isinstance(scalar, Mapping) else {}
    interface = summary.get("interface")
    interface = interface if isinstance(interface, Mapping) else {}
    scalar_keys = (
        "plddt",
        "ptm",
        "iptm",
        "gpde",
        "ranking_score",
        "has_clash",
        "clash_count",
    )
    interface_keys = (
        "interface_plddt_mean",
        "interface_plddt_min",
        "total_contact_count",
        "total_clash_count",
        "residue_pair_count",
    )
    compact_scalar = {
        key: value
        for key in scalar_keys
        if (value := _finite_or_none(scalar.get(key))) is not None
    }
    top_level_plddt = _finite_or_none(candidate.get("plddt"))
    if "plddt" not in compact_scalar and top_level_plddt is not None:
        compact_scalar["plddt"] = top_level_plddt
    compact_interface = {
        key: value
        for key in interface_keys
        if (value := _finite_or_none(interface.get(key))) is not None
    }
    states: Dict[str, Dict[str, float]] = {}
    raw_states = candidate.get("complex_state_confidence_metrics")
    if isinstance(raw_states, Mapping):
        for raw_name in sorted(raw_states, key=str):
            raw_metrics = raw_states[raw_name]
            if not isinstance(raw_metrics, Mapping):
                continue
            state_metrics = {
                key: value
                for key in ("plddt", "ptm", "iptm", "gpde", "ranking_score")
                if (value := _finite_or_none(raw_metrics.get(key))) is not None
            }
            if state_metrics:
                states[str(raw_name)] = state_metrics
    result: Dict[str, Any] = {"scalar": compact_scalar}
    if compact_interface:
        result["interface"] = compact_interface
    if states:
        result["states"] = states
    return result


def _gate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    sources: Dict[str, Any] = {}
    raw_sources = candidate.get("feasibility_gate_sources")
    if isinstance(raw_sources, Mapping):
        sources.update(raw_sources)
    if "fast_filter" not in sources and isinstance(
        candidate.get("fast_filter"), Mapping
    ):
        sources["fast_filter"] = candidate.get("fast_filter")
    normalized = []
    reasons: List[str] = []
    for raw_name in sorted(sources, key=str):
        decision = normalize_gate_payload(sources[raw_name])
        name = str(raw_name)
        normalized.append(
            {
                "source": name,
                "passed": bool(decision["passed"]),
            }
        )
        if not decision["passed"]:
            source_reasons = decision.get("reasons") or ["gate_failed"]
            reasons.extend(f"{name}:{reason}" for reason in source_reasons)
    return {
        "feasible": all(row["passed"] for row in normalized),
        "gate_sources": normalized,
        "gate_reasons": reasons,
    }


def _semantic_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    coverage = candidate.get("semantic_final_mutation_coverage")
    if not isinstance(coverage, Mapping):
        coverage = candidate.get("semantic_final_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    required = [str(value) for value in coverage.get("required_nodes", []) or []]
    missing = [
        str(value)
        for value in coverage.get(
            "missing_required_nodes_by_mutation", []
        )
        or []
    ]
    mutations = coverage.get("mutations_by_node")
    mutations = mutations if isinstance(mutations, Mapping) else {}
    return {
        "required_nodes": required,
        "joint_pass": bool(coverage.get("pass", not missing)),
        "missing_required_nodes": missing,
        "mutated_nodes": sorted(
            str(node)
            for node, count in mutations.items()
            if int(count or 0) > 0
        ),
    }


def _expensive_structure_result(candidate: Mapping[str, Any]) -> bool:
    provider = str(candidate.get("structure_provider") or "").lower()
    stage = str(candidate.get("structure_stage") or "").lower()
    return provider in {"protenix", "alphafold3", "af3"} or stage in {
        "rerank",
        "legacy",
    }


def build_structure_finalist_feedback(
    evaluated_results: Sequence[Mapping[str, Any]],
    *,
    selected_candidate: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    expensive = [
        candidate
        for candidate in evaluated_results
        if _expensive_structure_result(candidate)
    ]
    parent_candidates = [
        candidate for candidate in expensive if is_parent_baseline(candidate)
    ]
    parent = parent_candidates[-1] if parent_candidates else None
    root_hash = str((parent or {}).get("seq_hash") or "")
    root_energy = _finite_or_none(
        (parent or {}).get(
            "outer_aligned_energy",
            (parent or {}).get("combined_energy"),
        )
    )
    root_fast_energy = _finite_or_none((parent or {}).get("fast_loss"))
    selected_hash = str((selected_candidate or {}).get("seq_hash") or "")
    selected_stage = str(
        (selected_candidate or {}).get("structure_stage") or ""
    )
    selected_variant = str(
        (selected_candidate or {}).get("variant_id") or ""
    )

    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(expensive):
        candidate_id = str(
            candidate.get("selection_candidate_id")
            or candidate.get("variant_id")
            or f"finalist_{index + 1}"
        )
        seq_hash = str(candidate.get("seq_hash") or "")
        stage = str(candidate.get("structure_stage") or "unknown")
        aligned_energy = _finite_or_none(
            candidate.get(
                "outer_aligned_energy",
                candidate.get("combined_energy"),
            )
        )
        fast_energy = _finite_or_none(candidate.get("fast_loss"))
        fast_delta = (
            float(fast_energy - root_fast_energy)
            if fast_energy is not None and root_fast_energy is not None
            else None
        )
        aligned_delta = (
            float(aligned_energy - root_energy)
            if aligned_energy is not None and root_energy is not None
            else None
        )
        structure_signal_available = bool(_has_structure_signal(candidate))
        gate = _gate_summary(candidate)
        semantic = _semantic_summary(candidate)
        novel = bool(seq_hash and (not root_hash or seq_hash != root_hash))
        selected = bool(
            selected_candidate is not None
            and seq_hash == selected_hash
            and stage == selected_stage
            and str(candidate.get("variant_id") or "") == selected_variant
        )
        if selected:
            rejection_reason = None
        elif not structure_signal_available:
            rejection_reason = "structure_signal_unavailable"
        elif not gate["feasible"]:
            rejection_reason = "hard_gate_failed"
        elif not is_parent_baseline(candidate) and not novel:
            rejection_reason = "not_sequence_novel"
        elif not is_parent_baseline(candidate) and not semantic["joint_pass"]:
            rejection_reason = "semantic_joint_coverage_failed"
        elif bool(candidate.get("formal_rerank_eligible", True)) is False:
            rejection_reason = "diagnostic_not_formal_finalist"
        else:
            rejection_reason = "higher_final_energy_or_tie_break"
        rows.append(
            {
                "candidate_id": candidate_id,
                "seq_hash": seq_hash,
                "candidate_role": (
                    "parent_baseline"
                    if is_parent_baseline(candidate)
                    else "mutant"
                ),
                "stage": stage,
                "provider": str(
                    candidate.get("structure_provider") or "unknown"
                ),
                "aligned_energy": aligned_energy,
                "root_relative_aligned_energy": aligned_delta,
                "fast_energy": fast_energy,
                "root_relative_fast_energy": fast_delta,
                "proxy_structure_direction_agreement": (
                    _direction_agreement(fast_delta, aligned_delta)
                    if structure_signal_available
                    else None
                ),
                "feasible": bool(gate["feasible"]),
                "gate_reasons": gate["gate_reasons"],
                "semantic": semantic,
                "sequence_novel_vs_root": novel,
                "formal_rerank_eligible": bool(
                    candidate.get("formal_rerank_eligible", True)
                ),
                "structure_signal_available": structure_signal_available,
                "selected": selected,
                "rejection_reason": rejection_reason,
                "metrics": _compact_metrics(candidate),
                "_stable_order": index,
            }
        )

    rows.sort(
        key=lambda row: (
            0 if row["feasible"] else 1,
            (
                float(row["aligned_energy"])
                if row["aligned_energy"] is not None
                else math.inf
            ),
            -float(
                ((row.get("metrics") or {}).get("scalar") or {}).get(
                    "plddt", 0.0
                )
            ),
            str(row["candidate_id"]),
            int(row["_stable_order"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["scientific_rank"] = rank
        row.pop("_stable_order", None)

    alignment_rows = [
        row
        for row in rows
        if row["candidate_role"] == "mutant"
        and row["sequence_novel_vs_root"]
        and row["proxy_structure_direction_agreement"] is not None
    ]
    agreement_count = sum(
        bool(row["proxy_structure_direction_agreement"])
        for row in alignment_rows
    )
    disagreement_count = len(alignment_rows) - agreement_count

    return {
        "schema_version": STRUCTURE_FINALIST_FEEDBACK_VERSION,
        "direction": "minimize",
        "root_seq_hash": root_hash,
        "root_aligned_energy": root_energy,
        "root_fast_energy": root_fast_energy,
        "expensive_candidate_count": len(rows),
        "proxy_structure_alignment": {
            "comparison_count": len(alignment_rows),
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "agreement_rate": (
                float(agreement_count / len(alignment_rows))
                if alignment_rows
                else None
            ),
        },
        "selected_candidate_id": next(
            (row["candidate_id"] for row in rows if row["selected"]),
            None,
        ),
        "candidates": rows,
    }


__all__ = [
    "STRUCTURE_FINALIST_FEEDBACK_VERSION",
    "build_structure_finalist_feedback",
]



from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Literal


FEASIBILITY_SELECTION_VERSION = "astevolve.feasibility_selection.v2"

_PASS_KEYS = ("passed", "hard_gate_pass", "pass")
_REASON_KEYS = (
    "reasons",
    "disqualification_reasons",
    "hard_failures",
    "hard_gate_reasons",
)


class FeasibilitySelectionError(ValueError):
    pass


def _stable_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _boolean_decision(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise FeasibilitySelectionError(f"{location} must be a boolean")
    return value


def _level_decision(payload: Mapping[str, Any], *, location: str) -> bool | None:
    decisions = [
        (key, _boolean_decision(payload[key], location=f"{location}.{key}"))
        for key in _PASS_KEYS
        if key in payload
    ]
    if not decisions:
        return None
    first_key, first_value = decisions[0]
    for key, value in decisions[1:]:
        if value != first_value:
            raise FeasibilitySelectionError(
                f"gate decision conflict between {location}.{first_key} and {location}.{key}"
            )
    return first_value


def _reason_values(value: Any, *, location: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    elif isinstance(value, (set, frozenset)):
        values = tuple(sorted(value, key=str))
    else:
        raise FeasibilitySelectionError(f"{location} must be a string or a sequence of reasons")

    reasons: List[str] = []
    for item in values:
        if item is None:
            continue
        reason = str(item).strip()
        if reason:
            reasons.append(reason)
    return reasons


def _level_reasons(payload: Mapping[str, Any], *, location: str) -> List[str]:
    reasons: List[str] = []
    for key in _REASON_KEYS:
        if key in payload:
            reasons.extend(_reason_values(payload[key], location=f"{location}.{key}"))
    return reasons


def normalize_gate_payload(payload: object) -> Dict[str, Any]:


    if payload is None:
        return {"passed": True, "reasons": []}
    if isinstance(payload, bool):
        return {"passed": payload, "reasons": []}
    if not isinstance(payload, Mapping):
        raise FeasibilitySelectionError("gate payload must be a mapping, boolean, or null")

    top_decision = _level_decision(payload, location="gate")
    reasons = _level_reasons(payload, location="gate")

    raw_nested = payload.get("gate_status")
    nested_decision: bool | None = None
    if raw_nested is not None:
        if not isinstance(raw_nested, Mapping):
            raise FeasibilitySelectionError("gate.gate_status must be a mapping")
        nested_decision = _level_decision(raw_nested, location="gate.gate_status")
        reasons.extend(_level_reasons(raw_nested, location="gate.gate_status"))

    if top_decision is not None and nested_decision is not None and top_decision != nested_decision:
        raise FeasibilitySelectionError(
            "gate decision conflict between top-level payload and nested gate_status"
        )

    decision = top_decision if top_decision is not None else nested_decision
    return {
        "passed": True if decision is None else decision,
        "reasons": _stable_unique(reasons),
    }


def _normalize_gate_sources(value: object, *, candidate_id: str) -> List[Dict[str, Any]]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} gate_sources must be a mapping"
        )

    normalized: List[Dict[str, Any]] = []
    for raw_source in sorted(value, key=str):
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise FeasibilitySelectionError(
                f"candidate {candidate_id!r} gate source names must be non-empty strings"
            )
        gate = normalize_gate_payload(value[raw_source])
        normalized.append(
            {
                "source": raw_source,
                "passed": gate["passed"],
                "reasons": gate["reasons"],
            }
        )
    return normalized


def _objective(value: object, *, candidate_id: str) -> float:
    if isinstance(value, bool):
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} raw_objective must be a finite number"
        )
    try:
        objective = float(value)
    except (TypeError, ValueError) as exc:
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} raw_objective must be a finite number"
        ) from exc
    if not math.isfinite(objective):
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} raw_objective must be a finite number"
        )
    return objective


def _normalize_feasibility_priority(
    value: object, *, candidate_id: str
) -> Dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} feasibility_priority must be a mapping"
        )
    normalized: Dict[str, Any] = {}
    for key in ("hard_gate_pass_count", "hard_gate_total"):
        raw = value.get(key, 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise FeasibilitySelectionError(
                f"candidate {candidate_id!r} feasibility_priority.{key} "
                "must be a non-negative integer"
            )
        normalized[key] = raw
    if normalized["hard_gate_pass_count"] > normalized["hard_gate_total"]:
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} hard_gate_pass_count exceeds hard_gate_total"
        )
    for key in ("min_hard_margin", "joint_hard_margin"):
        normalized[key] = _objective(value.get(key, 0.0), candidate_id=candidate_id)
    margins = value.get("hard_gate_margins", {})
    if not isinstance(margins, Mapping):
        raise FeasibilitySelectionError(
            f"candidate {candidate_id!r} hard_gate_margins must be a mapping"
        )
    normalized["hard_gate_margins"] = {
        str(key): _objective(raw, candidate_id=candidate_id)
        for key, raw in sorted(margins.items(), key=lambda item: str(item[0]))
    }
    return normalized


def _candidate_row(candidate: object) -> Dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise FeasibilitySelectionError("each candidate must be a mapping")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise FeasibilitySelectionError("candidate_id must be a non-empty string")

    objective = _objective(candidate.get("raw_objective"), candidate_id=candidate_id)
    gate_sources = _normalize_gate_sources(
        candidate.get("gate_sources"),
        candidate_id=candidate_id,
    )
    feasible = all(source["passed"] for source in gate_sources)
    priority = _normalize_feasibility_priority(
        candidate.get("feasibility_priority"), candidate_id=candidate_id
    )
    gate_reasons: List[str] = []
    for source in gate_sources:
        if source["passed"]:
            continue
        source_reasons = source["reasons"] or ["gate_failed"]
        gate_reasons.extend(f"{source['source']}:{reason}" for reason in source_reasons)

    row = {
        "candidate_id": candidate_id,
        "raw_objective": objective,
        "feasible": feasible,
        "eligible": False,
        "gate_reasons": _stable_unique(gate_reasons),
        "gate_sources": gate_sources,
    }
    if priority is not None:
        row["feasibility_priority"] = priority
    return row


def select_feasibility_first(
    candidates: Sequence[Mapping[str, object]],
    *,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> Dict[str, Any]:


    if direction not in {"maximize", "minimize"}:
        raise FeasibilitySelectionError("direction must be 'maximize' or 'minimize'")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise FeasibilitySelectionError("candidates must be a sequence")
    if not candidates:
        raise FeasibilitySelectionError("at least one candidate is required")

    rows = [_candidate_row(candidate) for candidate in candidates]
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise FeasibilitySelectionError("duplicate candidate_id values are not allowed")

    def rank_key(row: Mapping[str, Any]) -> tuple[float, str]:
        objective = float(row["raw_objective"])
        primary = -objective if direction == "maximize" else objective
        return (primary, str(row["candidate_id"]))

    def infeasible_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        priority = row.get("feasibility_priority")
        if not isinstance(priority, Mapping):
            return (1, *rank_key(row))
        return (
            0,
            -int(priority["hard_gate_pass_count"]),
            -float(priority["min_hard_margin"]),
            -float(priority["joint_hard_margin"]),
            *rank_key(row),
        )

    feasible_rows = sorted((row for row in rows if row["feasible"]), key=rank_key)
    infeasible_rows = sorted(
        (row for row in rows if not row["feasible"]), key=infeasible_rank_key
    )
    all_infeasible = not feasible_rows

    if all_infeasible:
        ordered_rows = infeasible_rows
        eligible_rows = ordered_rows
        reason = (
            "all_infeasible_gate_margin_ranking"
            if any(row.get("feasibility_priority") is not None for row in rows)
            else "all_infeasible_raw_objective_fallback"
        )
    else:
        ordered_rows = feasible_rows + infeasible_rows
        eligible_rows = feasible_rows
        reason = "best_feasible_raw_objective"

    eligible_ids = [row["candidate_id"] for row in eligible_rows]
    eligible_set = set(eligible_ids)
    for row in ordered_rows:
        row["eligible"] = row["candidate_id"] in eligible_set

    return {
        "schema_version": FEASIBILITY_SELECTION_VERSION,
        "direction": direction,
        "selected_candidate_id": ordered_rows[0]["candidate_id"],
        "ordered_ids": [row["candidate_id"] for row in ordered_rows],
        "eligible_ids": eligible_ids,
        "all_infeasible": all_infeasible,
        "reason": reason,
        "counts": {
            "total": len(ordered_rows),
            "feasible": len(feasible_rows),
            "infeasible": len(infeasible_rows),
            "eligible": len(eligible_rows),
        },
        "candidates": ordered_rows,
    }


__all__ = [
    "FEASIBILITY_SELECTION_VERSION",
    "FeasibilitySelectionError",
    "normalize_gate_payload",
    "select_feasibility_first",
]

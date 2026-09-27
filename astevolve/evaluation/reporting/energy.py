

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence

from astevolve.evaluation.contracts import ScoreTerm


DESIGN_ENERGY_SCHEMA_VERSION = "astevolve.design_energy.v1"
DESIGN_ENERGY_POLICY_VERSION = "astevolve.design_energy_policy.v1"
DESIGN_ENERGY_DIRECTION = "minimize"


def _term_key(term: ScoreTerm) -> str:


    state = term.state or ""
    return "|".join(
        (str(term.provider), str(term.category), str(term.name), str(state))
    )


def _availability(term: ScoreTerm) -> tuple[str, bool, bool]:


    required = term.required
    if term.available:
        return "available", required, True
    if required:
        return "unavailable_required_worst_cost", True, True
    if term.explicitly_optional:
        return "unavailable_optional_excluded", False, False
    return "unavailable_unspecified_worst_cost", False, True


def _finite_weight(value: Any) -> float:


    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(weight):
        return 0.0
    return max(0.0, weight)


def _coverage_signature(rows: Sequence[Mapping[str, Any]]) -> str:


    basis = sorted(
        (
            {
                "term_key": str(row["term_key"]),
                "available": bool(row["available"]),
                "required": bool(row["required"]),
                "availability_semantics": str(row["availability_semantics"]),
                "included": bool(row["included"]),
                "weight": float(row["weight"]),
            }
            for row in rows
        ),
        key=lambda item: (
            item["term_key"],
            item["availability_semantics"],
            item["weight"],
            item["available"],
            item["included"],
        ),
    )
    encoded = json.dumps(
        {
            "energy_schema_version": DESIGN_ENERGY_SCHEMA_VERSION,
            "policy_version": DESIGN_ENERGY_POLICY_VERSION,
            "basis": basis,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def design_energy(terms: Sequence[ScoreTerm]) -> Dict[str, Any]:


    term_rows = []
    for term in terms:
        availability_semantics, required, included = _availability(term)
        reported_score = term.normalized_score
        score = reported_score if term.available else 0.0
        cost = 1.0 - score
        weight = _finite_weight(term.weight)
        try:
            declared_weight = float(term.weight)
        except (TypeError, ValueError):
            declared_weight = None
        if declared_weight is not None and not math.isfinite(declared_weight):
            declared_weight = None
        weighted_cost = weight * cost if included else 0.0
        term_rows.append(
            {
                "term_key": _term_key(term),
                "name": str(term.name),
                "category": str(term.category),
                "backend": str(term.backend),
                "provider": term.provider,
                "state": term.state,
                "available": bool(term.available),
                "required": required,
                "availability_semantics": availability_semantics,
                "included": included,
                "reported_score": float(reported_score),
                "score": float(score),
                "cost": float(cost),


                "declared_weight": declared_weight,
                "weight": float(weight),
                "weighted_cost": float(weighted_cost),
                "warnings": [str(item) for item in term.warnings],
            }
        )

    included_weight = sum(
        float(row["weight"]) for row in term_rows if row["included"]
    )
    weighted_cost = sum(float(row["weighted_cost"]) for row in term_rows)

    soft_energy = weighted_cost / included_weight if included_weight > 0.0 else 1.0

    categories: Dict[str, Dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in term_rows}):
        rows = [row for row in term_rows if row["category"] == category]
        category_weight = sum(
            float(row["weight"]) for row in rows if row["included"]
        )
        category_weighted_cost = sum(
            float(row["weighted_cost"]) for row in rows
        )
        categories[category] = {
            "energy": (
                float(category_weighted_cost / category_weight)
                if category_weight > 0.0
                else None
            ),
            "weighted_cost": float(category_weighted_cost),
            "weight": float(category_weight),
            "term_count": len(rows),
            "included_term_count": sum(bool(row["included"]) for row in rows),
            "available_term_count": sum(bool(row["available"]) for row in rows),
            "unavailable_required_term_count": sum(
                row["availability_semantics"]
                == "unavailable_required_worst_cost"
                for row in rows
            ),
            "unavailable_optional_term_count": sum(
                row["availability_semantics"]
                == "unavailable_optional_excluded"
                for row in rows
            ),
            "unavailable_unspecified_term_count": sum(
                row["availability_semantics"]
                == "unavailable_unspecified_worst_cost"
                for row in rows
            ),
        }

    declared_weight = sum(float(row["weight"]) for row in term_rows)
    available_weight = sum(
        float(row["weight"]) for row in term_rows if row["available"]
    )
    excluded_weight = sum(
        float(row["weight"]) for row in term_rows if not row["included"]
    )
    unavailable_required = [
        str(row["term_key"])
        for row in term_rows
        if row["availability_semantics"] == "unavailable_required_worst_cost"
    ]
    unavailable_optional = [
        str(row["term_key"])
        for row in term_rows
        if row["availability_semantics"] == "unavailable_optional_excluded"
    ]
    unavailable_unspecified = [
        str(row["term_key"])
        for row in term_rows
        if row["availability_semantics"]
        == "unavailable_unspecified_worst_cost"
    ]
    coverage = {
        "policy": {
            "policy_version": DESIGN_ENERGY_POLICY_VERSION,
            "aggregation": "flat_term_weighted_mean",
            "score_to_cost": "one_minus_clamped_score",
            "available": "include_reported_score_cost",
            "unavailable_required": "include_as_worst_cost",
            "unavailable_optional": "exclude_from_energy",
            "unavailable_unspecified": "include_as_worst_cost",
            "zero_or_invalid_weight": "report_without_numeric_effect",
            "hard_gate": "separate_feasibility_axis_not_a_finite_penalty",
            "comparison": "require_matching_coverage_signature",
        },
        "coverage_signature": _coverage_signature(term_rows),
        "term_count": len(term_rows),
        "included_term_count": sum(bool(row["included"]) for row in term_rows),
        "excluded_term_count": sum(not bool(row["included"]) for row in term_rows),
        "available_term_count": sum(bool(row["available"]) for row in term_rows),
        "unavailable_term_count": sum(not bool(row["available"]) for row in term_rows),
        "declared_weight": float(declared_weight),
        "included_weight": float(included_weight),
        "excluded_weight": float(excluded_weight),
        "available_weight": float(available_weight),
        "available_weight_fraction": (
            float(available_weight / declared_weight)
            if declared_weight > 0.0
            else 0.0
        ),
        "included_weight_fraction": (
            float(included_weight / declared_weight)
            if declared_weight > 0.0
            else 0.0
        ),
        "unavailable_required_terms": sorted(unavailable_required),
        "unavailable_optional_terms": sorted(unavailable_optional),
        "unavailable_unspecified_terms": sorted(unavailable_unspecified),
        "has_weighted_evidence": bool(included_weight > 0.0),
        "comparable_only_if_coverage_signature_matches": True,
    }

    return {
        "energy_schema_version": DESIGN_ENERGY_SCHEMA_VERSION,
        "direction": DESIGN_ENERGY_DIRECTION,
        "soft_energy": float(soft_energy),
        "total_energy": float(soft_energy),
        "term_energy_breakdown": term_rows,
        "category_energy_breakdown": categories,
        "energy_coverage": coverage,
    }


__all__ = [
    "DESIGN_ENERGY_DIRECTION",
    "DESIGN_ENERGY_POLICY_VERSION",
    "DESIGN_ENERGY_SCHEMA_VERSION",
    "design_energy",
]

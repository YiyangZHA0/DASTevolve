

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import as_list, safe_float


def endpoint_label(endpoint: Mapping[str, Any]) -> Dict[str, Any]:
    chain = endpoint.get("chain") or endpoint.get("asym") or endpoint.get("chain_id")
    position = endpoint.get("position")
    if position is None:
        position = endpoint.get("residue_index", endpoint.get("seq_id"))
    return {
        "chain": str(chain or ""),
        "position": position,
        "resname": endpoint.get("resname")
        or endpoint.get("residue_name")
        or endpoint.get("aa"),
    }


def pair_evidence_row(
    *,
    source: str,
    state: str,
    status: str,
    reason: str,
    binder: Mapping[str, Any],
    peptide: Mapping[str, Any],
    binder_node: str = "",
    min_distance: Any = None,
    interaction_classes: Any = None,
    score: Any = None,
    expected: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "state": state,
        "status": status,
        "reason": reason,
        "binder": endpoint_label(binder),
        "peptide": endpoint_label(peptide),
        "binder_node": str(binder_node or ""),
        "min_distance_angstrom": safe_float(min_distance),
        "interaction_classes": [str(item) for item in as_list(interaction_classes)],
        "score": safe_float(score),
        "expected": dict(expected or {}),
    }


def _normalized_rows(term: ScoreTerm) -> List[Dict[str, Any]]:
    details = term.details if isinstance(term.details, Mapping) else {}
    raw = details.get("residue_pair_evidence")
    if isinstance(raw, Mapping):
        rows: List[Dict[str, Any]] = []
        for bucket, values in raw.items():
            for value in as_list(values):
                if isinstance(value, Mapping):
                    row = dict(value)
                    row.setdefault("bucket", str(bucket))
                    rows.append(row)
        return rows
    return [dict(item) for item in as_list(raw) if isinstance(item, Mapping)]


def residue_pair_distance_evidence(
    terms: Sequence[ScoreTerm], max_examples: int = 16
) -> Dict[str, Any]:


    satisfied: List[Dict[str, Any]] = []
    unsatisfied: List[Dict[str, Any]] = []
    competitor: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    for term in terms:
        for raw_row in _normalized_rows(term):
            row = dict(raw_row)
            row.setdefault("source", term.name)
            row.setdefault("score", safe_float(term.score))
            bucket = str(row.pop("bucket", "") or "").lower()
            state = str(row.get("state") or "").lower()
            status = str(row.get("status") or "").lower()
            if bucket in {"unavailable", "missing_evidence"}:
                unavailable.append(row)
            elif bucket in {"competitor", "violating_competitor_pairs"} or state in {
                "negative",
                "competitor",
                "decoy",
            }:
                competitor.append(row)
            elif bucket in {"unsatisfied", "unsatisfied_or_missing_target_pairs"} or status in {
                "unsatisfied",
                "missing",
                "violating",
            }:
                unsatisfied.append(row)
            else:
                satisfied.append(row)

    key = lambda row: float(row.get("min_distance_angstrom") or 99.0)
    available = bool(satisfied or unsatisfied or competitor)
    return {
        "schema_version": "ast_residue_pair_distance_evidence_v1",
        "available": available,
        "reason": None
        if available
        else "no score term emitted normalized residue-pair evidence",
        "distance_units": "angstrom",
        "cutoff_reference": {},
        "satisfied_target_pairs": sorted(satisfied, key=key)[:max_examples],
        "unsatisfied_or_missing_target_pairs": unsatisfied[:max_examples],
        "violating_competitor_pairs": sorted(competitor, key=key)[:max_examples],
        "unavailable_modules": unavailable[:8],
        "llm_usage_hint": "Use declared pair evidence as one observation; do not treat a single distance as definitive physical energy.",
    }

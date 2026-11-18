

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import clamp01

from .energy import design_energy


def category_summary(terms: Sequence[ScoreTerm]) -> Dict[str, Dict[str, Any]]:


    by_category: Dict[str, List[ScoreTerm]] = {}
    for term in terms:
        by_category.setdefault(term.category, []).append(term)
    summary: Dict[str, Dict[str, Any]] = {}
    for category, items in by_category.items():
        total_weight = sum(max(0.0, float(item.weight)) for item in items)
        weighted = sum(clamp01(item.score) * max(0.0, float(item.weight)) for item in items)
        summary[category] = {
            "score": float(weighted / total_weight) if total_weight > 0 else 0.0,
            "weight": float(total_weight),
            "term_count": len(items),
            "available_term_count": sum(1 for item in items if item.available),
        }
    return summary


def term_dimension(term: ScoreTerm) -> str:


    raw = term.details.get("dimension") if isinstance(term.details, Mapping) else None
    if raw:
        return str(raw)
    if term.category in {"confidence", "fold_validity", "scaffold_preservation", "semantic_constraint", "pocket_geometry"}:
        return "quality"
    if term.category in {"interface", "interface_geometry", "interface_contact_graph", "interface_energy", "multistate", "task_specific", "task_correctness"}:
        return "correctness"
    return "other"


def dimension_summary(terms: Sequence[ScoreTerm]) -> Dict[str, Dict[str, Any]]:


    result: Dict[str, Dict[str, Any]] = {}
    for dimension in ("quality", "correctness", "other"):
        selected = [term for term in terms if term_dimension(term) == dimension]
        total_weight = sum(max(0.0, float(term.weight)) for term in selected)
        weighted = sum(clamp01(term.score) * max(0.0, float(term.weight)) for term in selected)
        result[dimension] = {
            "score": float(weighted / total_weight) if total_weight > 0 else 0.0,
            "weight": float(total_weight),
            "term_count": len(selected),
            "available_term_count": sum(1 for term in selected if term.available),
            "weakest_terms": [term.name for term in sorted(selected, key=lambda item: (clamp01(item.score), -float(item.weight)))[:5]],
        }
    return result


def layer_summary(terms: Sequence[ScoreTerm], categories: Sequence[str]) -> Dict[str, Any]:


    selected = [term for term in terms if term.category in set(categories)]
    total_weight = sum(max(0.0, float(term.weight)) for term in selected)
    weighted = sum(clamp01(term.score) * max(0.0, float(term.weight)) for term in selected)
    return {
        "score": float(weighted / total_weight) if total_weight > 0 else 0.0,
        "weight": float(total_weight),
        "enabled": bool(total_weight > 0),
        "available": any(bool(term.available) for term in selected),
        "term_count": len(selected),
        "available_term_count": sum(1 for term in selected if term.available),
    }


def scorer_layers(terms: Sequence[ScoreTerm]) -> Dict[str, Any]:


    return {
        "interface": {
            "proxy": {**layer_summary(terms, ("interface",)), "description": "fast residue-pair/contact/interface-confidence screen"},
            "geometry": {**layer_summary(terms, ("interface_geometry",)), "description": "atom-coordinate geometry screen for hbond/salt/hydrophobic contacts"},
            "contact_graph": {**layer_summary(terms, ("interface_contact_graph",)), "description": "getContacts non-covalent interaction graph for residue-pair coupling and interface quality"},
            "energy": {**layer_summary(terms, ("interface_energy",)), "description": "Rosetta/FoldX-style interface energy rerank layer"},
        },
        "pocket": {"geometry": {**layer_summary(terms, ("pocket_geometry",)), "description": "optional fpocket/POVME-style pocket geometry rerank layer"}},
        "scaffold": {"reference_rmsd": {**layer_summary(terms, ("scaffold_preservation",)), "description": "reference-backed CA RMSD/fold preservation terms when reference coordinates are configured"}},
    }


def overall_score(terms: Sequence[ScoreTerm]) -> float:


    total_weight = sum(max(0.0, float(term.weight)) for term in terms)
    if total_weight <= 0:
        return 0.0
    weighted = sum(clamp01(term.score) * max(0.0, float(term.weight)) for term in terms)
    return float(weighted / total_weight)


__all__ = [
    "category_summary",
    "design_energy",
    "dimension_summary",
    "layer_summary",
    "overall_score",
    "scorer_layers",
    "term_dimension",
]

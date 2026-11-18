

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SEARCH_ENERGY_SCHEMA_VERSION = "astevolve.search_energy.v1"
ENERGY_DIRECTION = "minimize"


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def fast_energy_record(
    *,
    fast_loss: Any,
    constraint_penalty: Any,
    progen_loglik_avg: Any,
    progen_weight: Any,
    hard_gate_pass: bool,
) -> dict[str, Any]:


    ranking = _finite(fast_loss, label="fast_loss")
    constraint = _finite(constraint_penalty, label="constraint_penalty")
    loglik = _finite(progen_loglik_avg, label="progen_loglik_avg")
    weight = _finite(progen_weight, label="progen_weight")
    if weight < 0.0:
        raise ValueError("progen_weight must be non-negative")
    prior = weight * (-loglik)
    passed = bool(hard_gate_pass)
    return {
        "schema_version": SEARCH_ENERGY_SCHEMA_VERSION,
        "direction": ENERGY_DIRECTION,
        "fidelity": "fast",
        "total_energy": ranking if passed else None,
        "legacy_ranking_energy": ranking,
        "soft_energy_available": passed,
        "hard_gate_pass": passed,
        "components": {
            "constraint_energy": constraint if passed else None,
            "sequence_prior_energy": prior if passed else None,
            "progen_nll_per_residue": -loglik if weight > 0.0 and passed else None,
            "progen_weight": weight,
        },
    }


def combined_energy_record(
    *,
    fast_energy: Any,
    structure_energy: Any,
    multistate_energy: Any,
    evaluator_energy: Any,
    total_energy: Any,
    hard_gate_pass: bool,
) -> dict[str, Any]:


    fast = _finite(fast_energy, label="fast_energy")
    structure = _finite(structure_energy, label="structure_energy")
    multistate = _finite(multistate_energy, label="multistate_energy")
    evaluator = _finite(evaluator_energy, label="evaluator_energy")
    total = _finite(total_energy, label="total_energy")
    return {
        "schema_version": SEARCH_ENERGY_SCHEMA_VERSION,
        "direction": ENERGY_DIRECTION,
        "fidelity": "structure",
        "total_energy": total,
        "hard_gate_pass": bool(hard_gate_pass),
        "components": {
            "fast_energy": fast,
            "structure_constraint_energy": structure,
            "multistate_energy": multistate,
            "evaluator_energy": evaluator,
        },
    }


def best_so_far_trace(
    candidates: Sequence[Mapping[str, Any]],
    *,
    root_energy: Any,
    root_gate_pass: bool = True,
    energy_field: str = "fast_loss",
) -> list[dict[str, Any]]:


    best = _finite(root_energy, label="root_energy")
    trace: list[dict[str, Any]] = [
        {
            "index": 0,
            "candidate_id": "root",
            "proposal_energy": best,
            "best_so_far_energy": best,
            "hard_gate_pass": bool(root_gate_pass),
        }
    ]
    for index, candidate in enumerate(candidates, start=1):
        raw_filter = candidate.get("fast_filter")
        fast_filter = raw_filter if isinstance(raw_filter, Mapping) else {}
        passed = bool(fast_filter.get("pass", True)) and candidate.get("inner_structure_gate_pass") is not False
        proposal = _finite(
            candidate.get(energy_field, candidate.get("fast_loss")),
            label=f"candidate {energy_field}",
        )
        if passed:
            best = min(best, proposal)
        trace.append(
            {
                "index": index,
                "candidate_id": str(
                    candidate.get("variant_id")
                    or candidate.get("candidate_id")
                    or f"candidate_{index}"
                ),
                "proposal_energy": proposal,
                "best_so_far_energy": best,
                "hard_gate_pass": passed,
            }
        )
    return trace


__all__ = [
    "ENERGY_DIRECTION",
    "SEARCH_ENERGY_SCHEMA_VERSION",
    "best_so_far_trace",
    "combined_energy_record",
    "fast_energy_record",
]

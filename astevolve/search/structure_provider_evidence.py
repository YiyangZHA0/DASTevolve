

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

from astevolve.search.artifact_io import _seqs_hash


STRUCTURE_PROVIDER_EVIDENCE_VERSION = "astevolve.structure_provider_evidence.v2"


def canonical_structure_provider(value: Any) -> str:


    provider = str(value or "").strip().lower()
    return {
        "af3": "alphafold3",
        "alpha-fold3": "alphafold3",
        "alpha_fold3": "alphafold3",
    }.get(provider, provider)


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_mapping(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, float] = {}
    for key, raw in value.items():
        number = _finite_float(raw)
        if number is not None:
            output[str(key)] = number
    return output


def _numeric_sequence_mapping(value: Any) -> Dict[str, list[float]]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, list[float]] = {}
    for key, raw_values in value.items():
        if hasattr(raw_values, "tolist"):
            raw_values = raw_values.tolist()
        if not isinstance(raw_values, (list, tuple)):
            continue
        numbers = [_finite_float(item) for item in raw_values]
        if numbers and all(item is not None for item in numbers):
            output[str(key)] = [float(item) for item in numbers if item is not None]
    return output


def _numeric_nested_mapping(value: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, Dict[str, float]] = {}
    for key, raw in value.items():
        nested = _numeric_mapping(raw)
        if nested:
            output[str(key)] = nested
    return output


def _compact_node_plddt(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_item in value.items():
        if not isinstance(raw_item, Mapping):
            continue
        item: Dict[str, Any] = {}
        for key in ("plddt_mean", "plddt_min", "plddt_max"):
            number = _finite_float(raw_item.get(key))
            if number is not None:
                item[key] = number
        for key in ("residue_count", "count"):
            number = _finite_float(raw_item.get(key))
            if number is not None:
                item[key] = int(number)
        chain_id = raw_item.get("chain_id")
        if chain_id is not None:
            item["chain_id"] = str(chain_id)
        if item:
            output[str(raw_name)] = item
    return output


def _compact_structure_metrics(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    interface = _numeric_mapping(value.get("interface"))
    for key in ("available", "reason"):
        raw = (
            (value.get("interface") or {}).get(key)
            if isinstance(value.get("interface"), Mapping)
            else None
        )
        if raw is not None:
            interface[key] = raw
    dockq_proxy = _finite_float(value.get("dockq_proxy"))
    output: Dict[str, Any] = {
        "scalar": _numeric_mapping(value.get("scalar")),
        "chain_plddt": _numeric_mapping(value.get("chain_plddt")),
        "node_summary": _numeric_mapping(value.get("node_summary")),
        "interface": interface,
    }
    if dockq_proxy is not None:
        output["dockq_proxy"] = dockq_proxy
    return output


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    value = str(candidate.get("seq_hash") or "").strip()
    if value:
        return value
    sequences = candidate.get("seqs")
    if isinstance(sequences, Mapping):
        return _seqs_hash({str(key): str(item) for key, item in sequences.items()})
    return ""


def _provider_record(candidate: Mapping[str, Any], provider: str) -> Dict[str, Any]:
    confidence = _numeric_mapping(candidate.get("confidence_metrics"))
    chain_plddt = _numeric_mapping(candidate.get("chain_plddt"))
    chain_pair_metrics = _numeric_nested_mapping(
        candidate.get("chain_pair_metrics")
    )
    node_plddt = _compact_node_plddt(candidate.get("node_plddt"))
    residue_plddt = _numeric_sequence_mapping(candidate.get("residue_plddt"))
    structure_metrics = _compact_structure_metrics(candidate.get("structure_metrics"))
    if not confidence:
        confidence = dict(structure_metrics.get("scalar") or {})
    if not chain_plddt:
        chain_plddt = dict(structure_metrics.get("chain_plddt") or {})
    has_signal = bool(
        confidence
        or chain_plddt
        or node_plddt
        or (structure_metrics.get("node_summary") or {})
    )
    record = {
        "provider": provider,
        "stage": str(candidate.get("structure_stage") or "unknown"),
        "model_name": candidate.get("structure_model_name"),
        "sequence_hash": _candidate_hash(candidate),
        "available": has_signal,
        "status": "ok" if has_signal else "unavailable",
        "confidence_metrics": confidence,
        "chain_plddt": chain_plddt,
        "chain_pair_metrics": chain_pair_metrics,
        "node_plddt": node_plddt,
        "residue_plddt": residue_plddt,
        "structure_metrics": structure_metrics,
    }
    seed = candidate.get("provider_seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        record["seed"] = seed
    if isinstance(candidate.get("provider_run_receipt"), Mapping):
        record["run_receipt"] = dict(candidate["provider_run_receipt"])
    if isinstance(candidate.get("provider_seed_aggregate"), Mapping):
        record["seed_aggregate"] = dict(candidate["provider_seed_aggregate"])
    return record


def _record_priority(record: Mapping[str, Any], ordinal: int) -> tuple[int, int, int]:
    stage = str(record.get("stage") or "").lower()
    return (
        int(bool(record.get("available"))),
        2 if stage == "rerank" else 1 if stage == "screen" else 0,
        ordinal,
    )


def build_structure_provider_evidence(
    evaluated: Iterable[Mapping[str, Any]],
    selected_sequences: Mapping[str, str],
    *,
    required_providers: Iterable[str] = ("alphafold3", "protenix"),
) -> Dict[str, Any]:


    evaluated_rows = list(evaluated)
    normalized_sequences = {
        str(chain_id): str(sequence)
        for chain_id, sequence in selected_sequences.items()
    }
    selected_hash = _seqs_hash(normalized_sequences)
    required = []
    for value in required_providers:
        provider = canonical_structure_provider(value)
        if provider and provider not in required:
            required.append(provider)

    providers: Dict[str, Dict[str, Any]] = {}
    provider_runs: Dict[str, list[Dict[str, Any]]] = {}
    priorities: Dict[str, tuple[int, int, int]] = {}
    matched_rows = 0
    for ordinal, candidate in enumerate(evaluated_rows):
        if (
            not isinstance(candidate, Mapping)
            or _candidate_hash(candidate) != selected_hash
        ):
            continue
        provider = canonical_structure_provider(candidate.get("structure_provider"))
        if not provider:
            continue
        matched_rows += 1
        record = _provider_record(candidate, provider)
        if "seed" in record:
            provider_runs.setdefault(provider, []).append(record)
        priority = _record_priority(record, ordinal)
        if provider not in priorities or priority > priorities[provider]:
            providers[provider] = record
            priorities[provider] = priority

    available = sorted(
        provider for provider, record in providers.items() if record.get("available")
    )
    missing = sorted(provider for provider in required if provider not in available)
    for provider, runs in provider_runs.items():
        runs.sort(key=lambda item: (int(item.get("seed", 0)), str(item.get("stage") or "")))


        selected = dict(providers.get(provider, runs[-1]))
        providers[provider] = selected
        selected["seed_runs"] = runs
        selected["seed_count"] = len({int(item["seed"]) for item in runs})
        selected["run_receipt_hashes"] = [
            item["run_receipt"]["receipt_hash"]
            for item in runs
            if isinstance(item.get("run_receipt"), Mapping)
            and item["run_receipt"].get("receipt_hash")
        ]
    disagreement = next(
        (
            dict(candidate["backend_disagreement"])
            for candidate in evaluated_rows
            if isinstance(candidate, Mapping)
            and _candidate_hash(candidate) == selected_hash
            and isinstance(candidate.get("backend_disagreement"), Mapping)
        ),
        None,
    )
    return {
        "schema_version": STRUCTURE_PROVIDER_EVIDENCE_VERSION,
        "source": "inner_structure_pipeline",
        "selected_sequence_hash": selected_hash,
        "required_providers": required,
        "available_providers": available,
        "missing_providers": missing,
        "complete": not missing,
        "matched_result_count": matched_rows,
        "providers": providers,
        "backend_disagreement": disagreement,
    }


__all__ = [
    "STRUCTURE_PROVIDER_EVIDENCE_VERSION",
    "build_structure_provider_evidence",
    "canonical_structure_provider",
]

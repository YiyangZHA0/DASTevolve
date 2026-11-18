

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

from astevolve.evaluation.support import evaluator_weight
from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.structure_provider_evidence import (
    STRUCTURE_PROVIDER_EVIDENCE_VERSION,
    canonical_structure_provider,
)


DUAL_STRUCTURE_EVALUATOR_VERSION = "cab_lys3.dual_structure_evaluator.v2"
_REQUIRED_PROVIDERS = ("alphafold3", "protenix")
_DESIGN_POSITIONS = (
    28,
    29,
    30,
    52,
    56,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
)
_COMPONENT_WEIGHTS = {
    "binder_confidence": 0.05,
    "design_region_confidence": 0.15,
    "interface_confidence": 0.30,
    "topology_confidence": 0.05,
    "interface_plddt": 0.15,
    "interface_pae_quality": 0.15,
    "contact_quality": 0.15,
}


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _confidence01(value: Any) -> Optional[float]:
    number = _finite_float(value)
    if number is None:
        return None
    return _clamp01(number / 100.0 if abs(number) > 1.5 else number)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _off_diagonal_pair_mean(record: Mapping[str, Any], *names: str) -> Optional[float]:
    pair_metrics = _mapping(record.get("chain_pair_metrics"))
    for name in names:
        values = []
        for pair, raw in _mapping(pair_metrics.get(name)).items():
            left, separator, right = str(pair).partition(":")
            number = _finite_float(raw)
            if separator and left != right and number is not None:
                values.append(number)
        if values:
            return float(sum(values) / len(values))
    return None


def _numeric_mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [
        number for value in values if (number := _finite_float(value)) is not None
    ]
    return float(sum(numbers) / len(numbers)) if numbers else None


def _node_confidence(record: Mapping[str, Any]) -> Optional[float]:
    residue_plddt = _mapping(record.get("residue_plddt"))
    chain_a = residue_plddt.get("A")
    if isinstance(chain_a, (list, tuple)) and len(chain_a) > max(_DESIGN_POSITIONS):
        exact_values = [chain_a[position] for position in _DESIGN_POSITIONS]
        exact_mean = _numeric_mean(exact_values)
        if exact_mean is not None:
            return _confidence01(exact_mean)

    local = _mapping(record.get("local_plddt"))
    value = _finite_float(local.get("benchmark_union"))
    if value is not None:
        return _confidence01(value)

    nodes = _mapping(record.get("node_plddt"))
    weighted_total = 0.0
    total_count = 0.0
    unweighted = []
    for name, item in nodes.items():
        if not isinstance(item, Mapping):
            continue
        chain_id = str(item.get("chain_id") or "A")
        token = str(name).lower()
        if chain_id != "A" or not ("cdr" in token or token in {"l0", "l1", "l2"}):
            continue
        mean = _finite_float(item.get("plddt_mean"))
        if mean is None:
            continue
        count = _finite_float(item.get("residue_count") or item.get("count"))
        if count is not None and count > 0.0:
            weighted_total += mean * count
            total_count += count
        unweighted.append(mean)
    if total_count > 0.0:
        return _confidence01(weighted_total / total_count)
    if unweighted:
        return _confidence01(_numeric_mean(unweighted))

    structure = _mapping(record.get("structure_metrics"))
    summary = _mapping(structure.get("node_summary"))
    return _confidence01(summary.get("node_plddt_mean"))


def _provider_components(record: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    structure = _mapping(record.get("structure_metrics"))
    scalar = dict(_mapping(structure.get("scalar")))
    scalar.update(_mapping(record.get("confidence_metrics")))
    chains = dict(_mapping(structure.get("chain_plddt")))
    chains.update(_mapping(record.get("chain_plddt")))
    interface = _mapping(structure.get("interface"))
    binder = _confidence01(chains.get("A"))
    raw_iptm = scalar.get("iptm_worst", scalar.get("iptm"))
    raw_ptm = scalar.get("ptm_worst", scalar.get("ptm"))
    interface_confidence = _confidence01(raw_iptm)
    pae = _off_diagonal_pair_mean(record, "pae_min", "gpde")
    pae_quality = (
        _clamp01(1.0 - float(pae) / 30.0)
        if pae is not None
        else interface_confidence
    )
    interface_plddt = _confidence01(interface.get("interface_plddt_mean"))
    if interface_plddt is None:
        interface_plddt = binder
    contact_quality = _confidence01(structure.get("dockq_proxy"))
    if contact_quality is None:
        contact_quality = interface_confidence
    return {
        "binder_confidence": binder,
        "design_region_confidence": _node_confidence(record),
        "interface_confidence": interface_confidence,
        "topology_confidence": _confidence01(raw_ptm),
        "interface_plddt": interface_plddt,
        "interface_pae_quality": pae_quality,
        "contact_quality": contact_quality,
    }


def _provider_quality(record: Mapping[str, Any]) -> Dict[str, Any]:
    components = _provider_components(record)
    missing = sorted(name for name, value in components.items() if value is None)
    available = bool(record.get("available", True)) and not missing
    score = (
        sum(
            _COMPONENT_WEIGHTS[name] * float(value)
            for name, value in components.items()
        )
        if available
        else 0.0
    )
    return {
        "available": available,
        "score": _clamp01(score),
        "components": components,
        "missing_components": missing,
        "component_weights": dict(_COMPONENT_WEIGHTS),
        "stage": record.get("stage"),
        "model_name": record.get("model_name"),
        "reference_analysis": dict(_mapping(record.get("reference_analysis"))),
    }


def _current_row_evidence(out: Mapping[str, Any]) -> Dict[str, Any]:
    provider = canonical_structure_provider(out.get("structure_provider"))
    if not provider:
        return {}
    structure = dict(_mapping(out.get("structure_metrics")))
    record = {
        "provider": provider,
        "stage": out.get("structure_stage"),
        "model_name": out.get("structure_model_name"),
        "available": True,
        "confidence_metrics": dict(_mapping(out.get("confidence_metrics"))),
        "chain_plddt": dict(_mapping(out.get("chain_plddt"))),
        "chain_pair_metrics": dict(_mapping(out.get("chain_pair_metrics"))),
        "node_plddt": dict(_mapping(out.get("node_plddt"))),
        "residue_plddt": dict(_mapping(out.get("residue_plddt"))),
        "structure_metrics": structure,
    }
    return {
        "schema_version": "cab_lys3.single_structure_row.v1",
        "source": "inner_provider_row",
        "providers": {provider: record},
        "complete": False,
    }


def _selected_sequence_hash(out: Mapping[str, Any]) -> Optional[str]:
    sequences = out.get("seqs") or out.get("best_seqs")
    if not isinstance(sequences, Mapping):
        return None
    return _seqs_hash(
        {str(chain): str(sequence) for chain, sequence in sequences.items()}
    )


def _resolved_evidence(out: Mapping[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
    raw = out.get("structure_provider_evidence")
    evidence = dict(raw) if isinstance(raw, Mapping) else _current_row_evidence(out)
    selected_hash = _selected_sequence_hash(out)
    evidence_hash = str(evidence.get("selected_sequence_hash") or "").strip()
    if selected_hash and evidence_hash and selected_hash != evidence_hash:
        evidence["providers"] = {}
        evidence["complete"] = False
        return evidence, (
            "structure_provider_evidence sequence hash does not match the "
            "final evaluator sequence"
        )
    return evidence, None


def structure_score_terms(
    out: Mapping[str, Any],
    score_config: Optional[Mapping[str, Any]],
    *,
    provider_name: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:


    score_cfg = score_config if isinstance(score_config, Mapping) else {}
    evidence, evidence_error = _resolved_evidence(out)
    provider_records = _mapping(evidence.get("providers"))
    qualities = {
        provider: _provider_quality(_mapping(provider_records.get(provider)))
        for provider in _REQUIRED_PROVIDERS
    }
    final_projection = evidence.get(
        "schema_version"
    ) == STRUCTURE_PROVIDER_EVIDENCE_VERSION or str(
        evidence.get("source") or ""
    ).endswith(
        "analysis_replay"
    )
    required = (
        bool(score_cfg.get("dual_structure_required", False)) and final_projection
    )

    def term(
        name: str,
        category: str,
        score: float,
        *,
        available: bool,
        weight_key: str,
        default_weight: float = 0.0,
        state: str = "positive",
        details: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Iterable[str]] = None,
        required_evidence: bool = False,
    ) -> Dict[str, Any]:
        weight = (
            evaluator_weight(score_cfg, weight_key, default_weight)
            if available or required_evidence
            else 0.0
        )
        metadata = dict(details or {})
        metadata.update(
            {
                "dimension": "structure_consensus",
                "required": bool(required_evidence),
            }
        )
        if not available and not required_evidence:
            metadata.update(
                {
                    "ignored_for_score": True,
                    "ignored_for_score_reason": "dual_structure_evidence_not_available_at_this_stage",
                }
            )
        return {
            "provider": provider_name,
            "state": state,
            "name": name,
            "category": category,
            "score": _clamp01(score),
            "weight": float(weight),
            "available": bool(available),
            "details": metadata,
            "warnings": [str(item) for item in (warnings or [])],
        }

    terms: list[Dict[str, Any]] = []
    provider_prefix = {"alphafold3": "af3", "protenix": "protenix"}
    for provider in _REQUIRED_PROVIDERS:
        result = qualities[provider]
        prefix = provider_prefix[provider]
        common_details = {
            "structure_provider": provider,
            "quality": result,
            "evidence_source": evidence.get("source"),
            "selected_sequence_hash": evidence.get("selected_sequence_hash"),
        }
        component_names = (
            ("binder_confidence", f"{prefix}_binder_confidence"),
            ("design_region_confidence", f"{prefix}_design_region_confidence"),
            ("interface_confidence", f"{prefix}_interface_confidence"),
            ("interface_plddt", f"{prefix}_interface_plddt"),
            ("interface_pae_quality", f"{prefix}_interface_pae_quality"),
            ("contact_quality", f"{prefix}_contact_quality"),
        )
        for component, name in component_names:
            value = result["components"].get(component)
            terms.append(
                term(
                    name,
                    f"{prefix}_structure_component",
                    float(value or 0.0),
                    available=value is not None and bool(result["available"]),
                    weight_key=f"eval_{name}",
                    details={**common_details, "component": component},
                )
            )
        terms.append(
            term(
                f"{prefix}_structure_quality",
                f"{prefix}_structure_quality",
                float(result["score"]),
                available=bool(result["available"]),
                weight_key=f"eval_{prefix}_structure_quality",
                details=common_details,
                warnings=(
                    [
                        f"{provider} structure evidence is incomplete: {result['missing_components']}"
                    ]
                    if not result["available"]
                    else []
                ),
            )
        )

    af3 = qualities["alphafold3"]
    protenix = qualities["protenix"]
    dual_available = bool(
        af3["available"] and protenix["available"] and not evidence_error
    )
    af3_weight = _finite_float(score_cfg.get("dual_structure_af3_weight"))
    af3_weight = _clamp01(0.70 if af3_weight is None else af3_weight)
    disagreement_weight = _finite_float(
        score_cfg.get("dual_structure_disagreement_weight")
    )
    disagreement_weight = max(
        0.0, 0.45 if disagreement_weight is None else disagreement_weight
    )
    worst_case_weight = _finite_float(
        score_cfg.get("dual_structure_worst_case_weight")
    )
    worst_case_weight = _clamp01(
        0.65 if worst_case_weight is None else worst_case_weight
    )
    if dual_available:
        consensus = af3_weight * af3["score"] + (1.0 - af3_weight) * protenix["score"]
        worst_case = min(af3["score"], protenix["score"])
        disagreement = sum(
            _COMPONENT_WEIGHTS[name]
            * abs(float(af3["components"][name]) - float(protenix["components"][name]))
            for name in _COMPONENT_WEIGHTS
        )
        agreement = _clamp01(1.0 - disagreement)
        pre_penalty = (1.0 - worst_case_weight) * consensus + worst_case_weight * worst_case
        dual_score = _clamp01(pre_penalty - disagreement_weight * disagreement)
    else:
        consensus = worst_case = agreement = dual_score = 0.0
        disagreement = 1.0
        pre_penalty = 0.0

    missing = sorted(
        provider
        for provider in _REQUIRED_PROVIDERS
        if not qualities[provider]["available"]
    )
    dual_details = {
        "schema_version": DUAL_STRUCTURE_EVALUATOR_VERSION,
        "af3_weight": af3_weight,
        "protenix_weight": 1.0 - af3_weight,
        "component_weights": dict(_COMPONENT_WEIGHTS),
        "raw_component_disagreement": disagreement,
        "agreement_score": agreement,
        "pre_penalty_score": pre_penalty,
        "worst_case_weight": worst_case_weight,
        "disagreement_penalty_weight": disagreement_weight,
        "missing_providers": missing,
        "evidence_error": evidence_error,
        "formula": "clip((1-worst_case_weight)*consensus + worst_case_weight*worst_case - disagreement_weight*disagreement)",
    }
    dual_specs = (
        ("dual_structure_consensus", "dual_structure_consensus", consensus),
        ("dual_structure_worst_case", "dual_structure_worst_case", worst_case),
        ("dual_structure_agreement", "dual_structure_disagreement", agreement),
        ("dual_structure_score", "dual_structure_objective", dual_score),
    )
    for name, category, value in dual_specs:
        is_objective = name == "dual_structure_score"
        terms.append(
            term(
                name,
                category,
                value,
                available=dual_available,
                weight_key=f"eval_{name}",
                default_weight=0.0,
                state="preserve" if name == "dual_structure_agreement" else "positive",
                details=dual_details,
                warnings=(
                    [evidence_error]
                    if evidence_error
                    else (
                        [f"dual structure evidence missing providers: {missing}"]
                        if missing
                        else []
                    )
                ),
                required_evidence=bool(required and is_objective),
            )
        )

    summary = {
        "schema_version": DUAL_STRUCTURE_EVALUATOR_VERSION,
        "evidence_schema_version": evidence.get("schema_version"),
        "evidence_source": evidence.get("source"),
        "selected_sequence_hash": evidence.get("selected_sequence_hash"),
        "required": required,
        "available": dual_available,
        "missing_providers": missing,
        "providers": qualities,
        "consensus_score": consensus,
        "worst_case_score": worst_case,
        "raw_disagreement_penalty": disagreement,
        "agreement_score": agreement,
        "dual_structure_score": dual_score,
        "formula": dual_details["formula"],
        "evidence_error": evidence_error,
    }
    return terms, summary


def _analysis_variant(document: Mapping[str, Any], provider: str) -> Mapping[str, Any]:
    variants = document.get("variants")
    rows = variants if isinstance(variants, list) else [document]
    canonical = canonical_structure_provider(provider)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_provider = canonical_structure_provider(row.get("provider") or canonical)
        if row_provider == canonical and str(row.get("status") or "ok").lower() == "ok":
            return row
    return {}


def _analysis_record(
    row: Mapping[str, Any], provider: str, sequence_hash: str
) -> Dict[str, Any]:
    local = dict(_mapping(row.get("local_plddt")))
    benchmark = _finite_float(local.get("benchmark_union"))
    node_summary = {"node_plddt_mean": benchmark} if benchmark is not None else {}
    node_plddt = {}
    for name in ("l0", "l1", "l2"):
        value = _finite_float(local.get(name))
        if value is not None:
            node_plddt[name.upper()] = {"plddt_mean": value}
    reference_analysis = {
        "aligned_rmsd_angstrom": dict(_mapping(row.get("aligned_rmsd_angstrom"))),
        "disulfide_sg_distance_angstrom": dict(
            _mapping(row.get("disulfide_sg_distance_angstrom"))
        ),
        "interface": dict(_mapping(row.get("interface"))),
    }
    return {
        "provider": provider,
        "stage": "historical_replay",
        "model_name": row.get("model_dir") or row.get("model_name"),
        "sequence_hash": sequence_hash,
        "available": bool(row),
        "status": str(row.get("status") or ("ok" if row else "unavailable")),
        "confidence_metrics": dict(_mapping(row.get("metrics"))),
        "chain_plddt": dict(_mapping(_mapping(row.get("chain_metrics")).get("plddt"))),
        "chain_pair_metrics": dict(_mapping(row.get("chain_pair_metrics"))),
        "node_plddt": node_plddt,
        "residue_plddt": dict(_mapping(row.get("residue_plddt"))),
        "local_plddt": local,
        "structure_metrics": {
            "scalar": dict(_mapping(row.get("metrics"))),
            "chain_plddt": dict(
                _mapping(_mapping(row.get("chain_metrics")).get("plddt"))
            ),
            "node_summary": node_summary,
            "interface": dict(_mapping(row.get("interface"))),
        },
        "reference_analysis": reference_analysis,
    }


def structure_provider_evidence_from_analysis(
    af3_analysis: Mapping[str, Any],
    protenix_analysis: Mapping[str, Any],
    selected_sequences: Mapping[str, str],
) -> Dict[str, Any]:


    sequences = {
        str(chain): str(sequence) for chain, sequence in selected_sequences.items()
    }
    sequence_hash = _seqs_hash(sequences)
    documents = {"alphafold3": af3_analysis, "protenix": protenix_analysis}
    providers: Dict[str, Any] = {}
    for provider, document in documents.items():
        row = _analysis_variant(_mapping(document), provider)
        providers[provider] = _analysis_record(row, provider, sequence_hash)
    available = sorted(
        provider for provider, record in providers.items() if record.get("available")
    )
    missing = sorted(
        provider for provider in _REQUIRED_PROVIDERS if provider not in available
    )
    return {
        "schema_version": STRUCTURE_PROVIDER_EVIDENCE_VERSION,
        "source": "historical_analysis_replay",
        "selected_sequence_hash": sequence_hash,
        "required_providers": list(_REQUIRED_PROVIDERS),
        "available_providers": available,
        "missing_providers": missing,
        "complete": not missing,
        "matched_result_count": len(available),
        "providers": providers,
    }


__all__ = [
    "DUAL_STRUCTURE_EVALUATOR_VERSION",
    "structure_provider_evidence_from_analysis",
    "structure_score_terms",
]

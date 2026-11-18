

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import statistics
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.structure_batch import evaluate_structure_candidates


PROVIDER_RUN_RECEIPT_VERSION = "astevolve.provider_run_receipt.v1"
PROVIDER_SEED_AGGREGATE_VERSION = "astevolve.provider_seed_aggregate.v1"
PROVIDER_FUNNEL_MANIFEST_VERSION = "astevolve.provider_funnel_manifest.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _provider(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"af3": "alphafold3", "alpha_fold3": "alphafold3"}.get(text, text)


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("seq_hash") or _seqs_hash(candidate["seqs"]))


def provider_seeds(cfg: Any, provider: str) -> Tuple[int, ...]:
    selected = _provider(provider)
    if not bool(getattr(cfg, "structure_multiseed_enabled", False)):
        return (int(getattr(cfg, "af3_seed", 202) if selected == "alphafold3" else getattr(cfg, "protenix_seed", 101)),)
    name = "structure_af3_seeds" if selected == "alphafold3" else "structure_protenix_seeds"
    values = tuple(int(seed) for seed in getattr(cfg, name, ()) or ())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid formal seed contract for {selected}")
    return values


def _protocol_material(cfg: Any, provider: str, stage: str, seed: int) -> Dict[str, Any]:
    selected = _provider(provider)
    common = {"provider": selected, "stage": str(stage), "seed": int(seed)}
    if selected == "protenix":
        common.update({
            "model": getattr(cfg, "protenix_model_name", None),
            "use_msa": getattr(cfg, "protenix_complex_use_msa", None),
            "cycle": getattr(cfg, "protenix_complex_cycle", None),
            "step": getattr(cfg, "protenix_complex_step", None),
            "sample": getattr(cfg, "protenix_complex_sample", None),
            "default_params": getattr(cfg, "protenix_complex_use_default_params", None),
        })
    elif selected == "alphafold3":
        common.update({
            "model": str(getattr(cfg, "af3_model_dir", None) or getattr(cfg, "structure_rerank_model_name", None) or "alphafold3"),
            "data_pipeline": bool(getattr(cfg, "af3_run_data_pipeline", False)),
            "recycles": int(getattr(cfg, "af3_num_recycles", 10)),
            "diffusion_samples": int(getattr(cfg, "af3_num_diffusion_samples", 1)),
            "flash_attention": str(getattr(cfg, "af3_flash_attention_implementation", "triton")),
        })
    return common


def _run_receipt(row: Mapping[str, Any], cfg: Any, provider: str, stage: str, seed: int) -> Dict[str, Any]:
    protocol = _protocol_material(cfg, provider, stage, seed)
    model_material = {"provider": _provider(provider), "declared_model": protocol.get("model")}
    status = "ok" if float(row.get("plddt", 0.0) or 0.0) > 0.0 else "unavailable"
    material = {
        "schema_version": PROVIDER_RUN_RECEIPT_VERSION,
        "candidate_sequence_hash": _candidate_hash(row),
        "provider": _provider(provider),
        "stage": str(stage),
        "seed": int(seed),
        "protocol_hash": _sha("provider_protocol_sha256:", protocol),


        "model_identity_hash": _sha("provider_model_identity_sha256:", model_material),
        "status": status,
    }
    return {**material, "receipt_hash": _sha("provider_run_receipt_sha256:", material)}


def _numeric(row: Mapping[str, Any], name: str) -> float | None:
    raw = row.get(name)
    if raw is None and isinstance(row.get("confidence_metrics"), Mapping):
        raw = row["confidence_metrics"].get(name)
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def aggregate_seed_rows(rows: Sequence[Mapping[str, Any]], *, expected_seeds: Sequence[int]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty provider seed group")
    ordered = sorted((deepcopy(dict(row)) for row in rows), key=lambda row: int(row["provider_seed"]))
    actual = [int(row["provider_seed"]) for row in ordered]
    if actual != sorted(int(seed) for seed in expected_seeds):
        raise ValueError(f"provider seed coverage mismatch: expected={sorted(expected_seeds)} actual={actual}")
    if len({_candidate_hash(row) for row in ordered}) != 1:
        raise ValueError("provider seed aggregate mixed sequence identities")


    def representative_key(row: Mapping[str, Any]) -> tuple[float, int]:
        energy = _numeric(row, "combined_energy")
        return (energy if energy is not None else float("inf"), int(row["provider_seed"]))

    representative = max(ordered, key=representative_key)
    aggregate = deepcopy(representative)
    metric_names = ("plddt", "ptm", "iptm", "combined_energy", "outer_aligned_energy", "multistate_score")
    statistics_payload: Dict[str, Any] = {}
    for name in metric_names:
        values = [value for row in ordered if (value := _numeric(row, name)) is not None]
        if not values:
            continue
        direction = "minimize" if name in {"combined_energy", "outer_aligned_energy"} else "maximize"
        worst = max(values) if direction == "minimize" else min(values)
        median = statistics.median(values)
        statistics_payload[name] = {"median": median, "worst": worst, "minimum": min(values), "maximum": max(values)}
        aggregate[name] = median
        if name in {"plddt", "ptm", "iptm"}:
            aggregate.setdefault("confidence_metrics", {})[name] = median
    gate_values = [bool((row.get("inner_evaluator_report") or {}).get("hard_gate_pass", True)) for row in ordered]
    material = {
        "schema_version": PROVIDER_SEED_AGGREGATE_VERSION,
        "candidate_sequence_hash": _candidate_hash(aggregate),
        "provider": _provider(aggregate.get("structure_provider")),
        "stage": str(aggregate.get("structure_stage") or ""),
        "expected_seeds": sorted(int(seed) for seed in expected_seeds),
        "successful_seeds": [int(row["provider_seed"]) for row in ordered if row["provider_run_receipt"]["status"] == "ok"],
        "representative_seed": int(representative["provider_seed"]),
        "all_seed_hard_gate_pass": all(gate_values),
        "statistics": statistics_payload,
        "run_receipt_hashes": [row["provider_run_receipt"]["receipt_hash"] for row in ordered],
    }
    aggregate["provider_seed_aggregate"] = {**material, "aggregate_hash": _sha("provider_seed_aggregate_sha256:", material)}
    aggregate["provider_seed"] = None
    aggregate["robust_hard_gate_pass"] = all(gate_values)
    return aggregate


def evaluate_multiseed_stage(
    candidates: Sequence[Dict[str, Any]], *, evaluator: Callable[..., Dict[str, Any]], evaluator_kwargs: Mapping[str, Any], provider: str, stage: str, cfg: Any, batch_size: int = 0, max_workers: int = 1,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    seeds = provider_seeds(cfg, provider)
    raw_rows: list[Dict[str, Any]] = []
    for seed in seeds:
        seeded_cfg = replace(cfg, protenix_seed=int(seed)) if _provider(provider) != "alphafold3" else replace(cfg, af3_seed=int(seed))
        kwargs = dict(evaluator_kwargs)
        kwargs["cfg"] = seeded_cfg
        evaluated = evaluate_structure_candidates(candidates, evaluator=evaluator, evaluator_kwargs=kwargs, provider=provider, stage=stage, batch_size=batch_size, max_workers=max_workers)
        for result in evaluated:
            row = dict(result)
            row["provider_seed"] = int(seed)
            row["provider_run_receipt"] = _run_receipt(row, seeded_cfg, provider, stage, seed)
            raw_rows.append(row)
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(_candidate_hash(row), []).append(row)
    aggregates = [aggregate_seed_rows(grouped[_candidate_hash(candidate)], expected_seeds=seeds) for candidate in candidates]
    material = {
        "schema_version": PROVIDER_FUNNEL_MANIFEST_VERSION,
        "provider": _provider(provider), "stage": str(stage),
        "candidate_count": len(candidates), "seeds": list(seeds),
        "physical_attempt_count": len(raw_rows),
        "candidate_sequence_hashes": [_candidate_hash(candidate) for candidate in candidates],
        "aggregate_hashes": [row["provider_seed_aggregate"]["aggregate_hash"] for row in aggregates],
    }
    return aggregates, raw_rows, {**material, "manifest_hash": _sha("provider_funnel_manifest_sha256:", material)}


def build_backend_disagreement(protenix: Mapping[str, Any], alphafold3: Mapping[str, Any], *, threshold: float) -> Dict[str, Any]:
    if _candidate_hash(protenix) != _candidate_hash(alphafold3):
        raise ValueError("backend disagreement requires the same sequence")
    metrics: Dict[str, Any] = {}
    disagreed = False
    for name in ("plddt", "ptm", "iptm"):
        left, right = _numeric(protenix, name), _numeric(alphafold3, name)
        if left is None or right is None:
            continue
        delta = right - left
        flagged = abs(delta) > float(threshold)
        disagreed |= flagged
        metrics[name] = {"protenix": left, "alphafold3": right, "delta": delta, "threshold_exceeded": flagged}
    gate_disagreement = bool(protenix.get("robust_hard_gate_pass", True)) != bool(alphafold3.get("robust_hard_gate_pass", True))
    disagreed |= gate_disagreement
    material = {"schema_version": "astevolve.backend_disagreement.v1", "candidate_sequence_hash": _candidate_hash(protenix), "threshold": float(threshold), "metrics": metrics, "hard_gate_disagreement": gate_disagreement, "disagreed": disagreed}
    return {**material, "disagreement_hash": _sha("backend_disagreement_sha256:", material)}


__all__ = ["aggregate_seed_rows", "build_backend_disagreement", "evaluate_multiseed_stage", "provider_seeds"]



from __future__ import annotations

from copy import deepcopy

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from astevolve.core.constraints import ChaiPlddtDeltaTerm, energy_breakdown
from astevolve.evaluation.evaluator_engine import evaluate_candidate
from astevolve.evaluation.energy_objective import compute_outer_energy_objective
from astevolve.evaluation.multistate_objectives import evaluate_multistate_objectives
from astevolve.evaluation.objectives.service import (
    validate_multistate_objective_specs,
)
from astevolve.evaluation.selection import normalize_gate_payload
from astevolve.metrics.structure import summarize_structure_metrics
from astevolve.providers.registry import (
    run_structure_confidence_complex_batch,
    run_structure_confidence_complex,
    run_structure_confidence_multichain,
)
from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.config import SAConfig
from astevolve.search.energy_reporting import combined_energy_record
from astevolve.search.runtime_evaluation import (
    evaluate_structure_evidence_persistently,
)
from astevolve.search.structure_failure_memory import (
    suppress_exact_structure_failures,
)
from astevolve.search.structure_evidence import (
    _aggregate_complex_state_metrics,
    _complex_entities_to_multichain,
    _infer_complex_entity_units,
    _residue_plddt_by_source_chain,
    _resolve_complex_entities,
    compute_node_plddt,
)
from engine.experiment_identity import SequenceBundleIdentity
from engine.memory_lifecycle import current_memory_execution_context


def _normalise_structure_provider(value: Optional[str]) -> str:


    provider = str(value or "protenix").strip().lower()
    return {
        "af3": "alphafold3",
        "alpha-fold3": "alphafold3",
        "alpha_fold3": "alphafold3",
        "structure_service": "service",
        "remote_service": "service",
    }.get(provider, provider)


def _service_backend(cfg: SAConfig) -> str:
    backend = _normalise_structure_provider(cfg.structure_service_backend)
    if backend == "service":
        raise ValueError("structure_service_backend cannot itself be service")
    return backend


def _service_transport_kwargs(cfg: SAConfig) -> Dict[str, Any]:
    return {
        "service_url": cfg.structure_service_url,
        "service_backend": _service_backend(cfg),
        "service_token": cfg.structure_service_token,
        "service_timeout": cfg.structure_service_timeout,
    }


class StructureEvidenceUnavailableError(RuntimeError):
    pass


_PROVIDER_EVIDENCE_VERSIONS = {
    "protenix": "astevolve.protenix_adapter.v2",
    "esmfold": "astevolve.esmfold_adapter.v1",
    "esmfold2": "astevolve.esmfold2_adapter.v2",
    "alphafold3": "astevolve.alphafold3_adapter.v1",
}


def _structure_model_cache_identity(
    provider: str,
    model_name: Optional[str],
    call_kwargs: Mapping[str, Any],
) -> Tuple[str, bool, Optional[str]]:


    selected = _normalise_structure_provider(provider)
    effective = str(call_kwargs.get("service_backend") or selected)
    effective = _normalise_structure_provider(effective)
    revision = str(
        call_kwargs.get("model_digest")
        or call_kwargs.get("model_revision")
        or ""
    ).strip()
    declared_model = model_name
    if selected == "service" and not str(declared_model or "").strip():
        if not revision:
            return (
                f"{selected}:{effective}:unspecified-service-model",
                False,
                "service_model_identity_missing",
            )
        declared_model = "revision-identified-service-model"
    if effective == "protenix" and not str(declared_model or "").strip():
        declared_model = "protenix_mini_esm_v0.5.0"
    if effective == "esmfold" and not str(declared_model or "").strip():
        declared_model = "facebook/esmfold_v1"
    if effective == "esmfold2" and (
        not str(declared_model or "").strip()
        or "protenix" in str(declared_model).lower()
    ):
        mode = str(
            call_kwargs.get("mode")
            or os.environ.get("ASTEVOLVE_ESMFOLD2_MODE", "local")
        ).strip().lower()
        declared_model = (
            os.environ.get("ASTEVOLVE_ESMFOLD2_MODEL")
            or (
                "esmfold2-fast-2026-05"
                if mode in {"api", "platform", "remote", "biohub"}
                else "biohub/ESMFold2"
            )
        )
    if effective == "alphafold3" and selected != "service":
        explicit_model_dir = call_kwargs.get("model_dir")
        candidate_model = str(declared_model or "").strip()
        candidate_path = Path(candidate_model).expanduser() if candidate_model else None
        if explicit_model_dir:
            declared_model = explicit_model_dir
        elif candidate_path is not None and (
            candidate_path.is_dir()
            or "/" in candidate_model
            or "\\" in candidate_model
        ):
            declared_model = candidate_path
        else:
            configured_model_dir = os.environ.get("ASTEVOLVE_AF3_MODEL_DIR")
            if configured_model_dir:
                declared_model = configured_model_dir
            else:
                model_root = os.environ.get("ASTEVOLVE_MODEL_ROOT")
                if model_root:
                    declared_model = Path(model_root) / "alphafold3"
                else:
                    runtime_root = os.environ.get("ASTEVOLVE_RUNTIME_ROOT")
                    if runtime_root:
                        declared_model = Path(runtime_root) / "models" / "alphafold3"
                    else:
                        declared_model = (
                            Path(__file__).resolve().parents[2].parent
                            / "DASTevolve_runtime"
                            / "models"
                            / "alphafold3"
                        )
        declared_model = Path(str(declared_model)).expanduser().resolve()
    declared = str(declared_model or effective).strip()
    path_like = Path(declared).is_absolute()
    stable_name = Path(declared).name if path_like else declared
    adapter = _PROVIDER_EVIDENCE_VERSIONS.get(effective, "astevolve.structure_adapter.v1")
    identity = f"{selected}:{effective}:{stable_name}:{revision or adapter}"
    if path_like and not revision:
        return identity, False, "absolute_model_path_has_no_immutable_revision"
    if (
        effective == "alphafold3"
        and bool(call_kwargs.get("run_data_pipeline"))
        and not str(call_kwargs.get("database_identity") or "").strip()
    ):
        return identity, False, "af3_data_pipeline_database_identity_missing"
    return identity, True, None


def _structure_scientific_settings(
    provider: str,
    call_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:


    selected = _normalise_structure_provider(provider)
    effective = _normalise_structure_provider(
        str(call_kwargs.get("service_backend") or selected)
    )
    settings: Dict[str, Any] = {
        "effective_provider": effective,
        "metric": str(call_kwargs.get("metric") or "plddt"),
        "provider_adapter_version": _PROVIDER_EVIDENCE_VERSIONS.get(
            effective, "astevolve.structure_adapter.v1"
        ),
    }
    if effective == "protenix":
        settings.update(
            {
                "use_msa": call_kwargs.get("use_msa"),
                "cycle": call_kwargs.get("cycle"),
                "step": call_kwargs.get("step"),
                "sample": call_kwargs.get("sample"),
                "use_default_params": call_kwargs.get("use_default_params"),
                "need_atom_confidence": str(
                    os.environ.get("ASTEVOLVE_PROTENIX_NEED_ATOM_CONFIDENCE", "1")
                ).strip().lower()
                in {"1", "true", "yes", "y", "on"},
            }
        )
    elif effective == "esmfold2":
        settings.update(
            {
                "mode": str(call_kwargs.get("mode") or "local"),
                "num_loops": int(call_kwargs.get("num_loops", 3)),
                "num_sampling_steps": int(
                    call_kwargs.get("num_sampling_steps", 32)
                ),
                "num_diffusion_samples": int(
                    call_kwargs.get("num_diffusion_samples", 1)
                ),
                "write_cif": bool(call_kwargs.get("write_cif", True)),
            }
        )
    elif effective == "alphafold3":
        settings.update(
            {
                "run_data_pipeline": bool(call_kwargs.get("run_data_pipeline")),
                "database_identity": call_kwargs.get("database_identity"),
                "num_recycles": int(call_kwargs.get("num_recycles", 10)),
                "num_diffusion_samples": int(
                    call_kwargs.get("num_diffusion_samples", 1)
                ),
                "flash_attention_implementation": str(
                    call_kwargs.get("flash_attention_implementation") or "triton"
                ),
            }
        )
    return settings


def _confidence_has_successful_evidence(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or "").strip().lower()
    if status and status not in {"ok", "success", "cache_hit", "salvaged"}:
        return False
    return bool(
        value.get("out_dir")
        or value.get("cif_path")
        or value.get("structure_path")
        or value.get("metrics")
        or value.get("residue_plddt")
    )


def _uncached_structure_evidence_artifact(reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "astevolve.persistent_evaluation_cache.v1",
        "enabled": False,
        "exact": False,
        "cache_key": None,
        "evaluator_descriptor_hash": None,
        "cache_hit": False,
        "evaluation_invoked": True,
        "outcome": "bypassed",
        "reason": str(reason),
    }


def _force_physical_structure_evidence() -> bool:


    return os.environ.get(
        "ASTEVOLVE_FORCE_PHYSICAL_STRUCTURE_EVIDENCE", "0"
    ).strip() == "1"


def _structure_occurrence_token(provider: str) -> str:


    normalized = _normalise_structure_provider(provider)
    if normalized != "alphafold3" and not _force_physical_structure_evidence():
        return ""
    context = current_memory_execution_context()
    if context is None:
        return ""
    components = (
        str(context.schema_version or ""),
        str(context.generation_id or ""),
        str(context.proposal_id or ""),
        str(context.trial_id or ""),
        str(context.scope_id or ""),
        str(context.design_action_json or ""),
    )
    if not any(components[1:]):
        return ""
    return hashlib.sha256("\0".join(components).encode("utf-8")).hexdigest()[:12]


def _cached_structure_evidence_call(
    sequence_bundle: Mapping[str, str],
    *,
    run_memory: Any,
    provider: str,
    model_name: Optional[str],
    operation: str,
    request_state: Any,
    call_kwargs: Mapping[str, Any],
    compute: Any,
    estimated_cost: float = 1.0,
) -> Tuple[Any, Dict[str, Any]]:


    model_identity, cache_safe, unsafe_reason = _structure_model_cache_identity(
        provider, model_name, call_kwargs
    )

    def validated_compute() -> Any:
        evidence = compute()
        values = evidence if isinstance(evidence, list) else [evidence]
        if not values or not all(
            _confidence_has_successful_evidence(item) for item in values
        ):
            raise StructureEvidenceUnavailableError(
                f"{provider}:{operation} returned incomplete structure evidence"
            )
        return evidence

    if _force_physical_structure_evidence():
        return validated_compute(), _uncached_structure_evidence_artifact(
            "controller_forced_physical_structure_evidence"
        )

    if not cache_safe:
        return validated_compute(), _uncached_structure_evidence_artifact(
            unsafe_reason or "structure_evidence_identity_not_exact"
        )

    evidence, artifact = evaluate_structure_evidence_persistently(
        sequence_bundle,
        run_memory=run_memory,
        provider=provider,
        model=model_identity,
        operation=operation,
        request_state=request_state,
        scientific_settings=_structure_scientific_settings(provider, call_kwargs),
        seed=int(call_kwargs.get("seed", 0)),
        compute=validated_compute,
        estimated_cost=float(estimated_cost),
    )
    return evidence, dict(artifact)


def _rebind_confidence_pred_name(
    confidence: Mapping[str, Any], pred_name: str
) -> Dict[str, Any]:
    rebound = dict(confidence)
    rebound["pred_name"] = str(pred_name)
    return rebound


def _structure_evidence_cache_summary(
    artifacts: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:


    unique: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any, Any]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        item = dict(artifact)
        identity = (
            item.get("cache_key"),
            item.get("evaluator_descriptor_hash"),
            item.get("reason"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    if not unique:
        return None
    effective_hits = [
        bool(
            item.get("cache_hit")
            or item.get("occurrence_cache_hit")
            or item.get("served_by_inner_run_cache")
        )
        for item in unique
    ]
    summary = dict(unique[0]) if len(unique) == 1 else {
        "schema_version": "astevolve.structure_evidence_cache_summary.v1",
        "enabled": all(bool(item.get("enabled")) for item in unique),
        "exact": all(bool(item.get("exact")) for item in unique),
        "cache_key": None,
        "evaluator_descriptor_hash": None,
        "outcome": "aggregate",
        "reason": None,
    }
    summary.update(
        {
            "cache_hit": bool(effective_hits and all(effective_hits)),
            "occurrence_cache_hit": bool(effective_hits and all(effective_hits)),
            "partial_cache_hit": bool(any(effective_hits) and not all(effective_hits)),
            "evaluation_invoked": any(
                bool(item.get("evaluation_invoked")) for item in unique
            ),
            "served_by_inner_run_cache": any(
                bool(item.get("served_by_inner_run_cache")) for item in unique
            ),
            "entry_count": len(unique),
            "entries": unique,
        }
    )
    return summary


def _evaluate_complex_states(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    cfg: SAConfig,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    pred_prefix: Optional[str] = None,
    stage: str = "structure",
    structure_evidence_run_memory: Any = None,
) -> Tuple[Optional[float], Dict[str, Any], Dict[str, Dict[str, Any]], Optional[str], Optional[str]]:


    state = compiled.get("_design_state", {}) or {}
    complex_states = state.get("complex_states", [])
    if not isinstance(complex_states, list) or not complex_states:
        return None, {}, {}, None, None
    objective_specs = (
        state.get("multistate_objectives") or state.get("objectives") or []
    )
    if cfg.multistate_objectives_enabled and objective_specs:
        objective_preflight = validate_multistate_objective_specs(
            objective_specs,
            evaluator_term_binding_available=False,
        )
        if not objective_preflight["valid"]:
            raise ValueError(
                "multistate_objective_preflight_failed:"
                + "|".join(objective_preflight["errors"])
            )
    selected_provider = _normalise_structure_provider(provider or cfg.structure_model)
    if selected_provider not in {
        "protenix", "esmfold", "esmfold2", "alphafold3", "service"
    }:
        return None, {}, {}, None, None
    effective_provider = (
        _service_backend(cfg) if selected_provider == "service" else selected_provider
    )
    if effective_provider not in {"protenix", "esmfold", "esmfold2", "alphafold3"}:
        return None, {}, {}, None, None

    state_results: List[Dict[str, Any]] = []
    first_out_dir: Optional[str] = None
    first_summary_json: Optional[str] = None
    confidence_metrics_by_state: Dict[str, Dict[str, Any]] = {}
    evidence_cache_artifacts: List[Dict[str, Any]] = []

    prepared_states: List[Dict[str, Any]] = []
    for index, raw_state in enumerate(complex_states, start=1):
        if not isinstance(raw_state, dict):
            continue
        name = str(raw_state.get("name") or f"state_{index}")
        raw_entities = list(raw_state.get("entities", []))
        entities = _resolve_complex_entities(raw_entities, seqs)
        if not entities:
            continue
        pred_name = f"{pred_prefix}__{name}" if pred_prefix else name
        metric = str(raw_state.get("metric", "plddt"))
        prepared_states.append(
            {
                "raw_state": raw_state,
                "name": name,
                "raw_entities": raw_entities,
                "entities": entities,
                "pred_name": pred_name,
                "metric": metric,
            }
        )


    batched_confidence: Dict[int, Dict[str, Any]] = {}
    batched_cache_artifacts: Dict[int, Dict[str, Any]] = {}
    if selected_provider == "protenix" and len(prepared_states) > 1:
        groups: Dict[str, List[int]] = {}
        for prepared_index, prepared in enumerate(prepared_states):
            groups.setdefault(str(prepared["metric"]), []).append(prepared_index)
        requested_batch_size = int(getattr(cfg, "structure_batch_size", 0) or 0)
        for batch_metric, indices in groups.items():
            if len(indices) < 2:
                continue
            chunk_size = requested_batch_size or len(indices)
            if chunk_size < 1:
                raise ValueError("structure_batch_size must be non-negative")
            jobs = [
                {
                    "pred_name": prepared_states[item_index]["pred_name"],
                    "entities": prepared_states[item_index]["entities"],
                    "constraint": prepared_states[item_index]["raw_state"].get(
                        "constraint"
                    ),
                    "covalent_bonds": prepared_states[item_index]["raw_state"].get(
                        "covalent_bonds"
                    ),
                }
                for item_index in indices
            ]
            resolved_batch_model = (
                model_name
                or cfg.structure_rerank_model_name
                or cfg.structure_model_name
                or cfg.protenix_model_name
            )
            batch_call_kwargs = {
                "metric": batch_metric,
                "seed": cfg.protenix_seed,
                "conda_env": cfg.protenix_conda_env,
                "use_msa": cfg.protenix_complex_use_msa,
                "cycle": cfg.protenix_complex_cycle,
                "step": cfg.protenix_complex_step,
                "sample": cfg.protenix_complex_sample,
                "use_default_params": cfg.protenix_complex_use_default_params,
                "timeout": cfg.protenix_complex_timeout,
            }

            def compute_batch() -> List[Dict[str, Any]]:
                computed: List[Dict[str, Any]] = []
                for offset in range(0, len(jobs), chunk_size):
                    chunk = jobs[offset : offset + chunk_size]
                    chunk_results = run_structure_confidence_complex_batch(
                        provider="protenix",
                        jobs=chunk,
                        model_name=resolved_batch_model,
                        **batch_call_kwargs,
                    )
                    if len(chunk_results) != len(chunk):
                        raise RuntimeError(
                            "Protenix batch returned an invalid result count"
                        )
                    computed.extend(chunk_results)
                return computed

            results, cache_artifact = _cached_structure_evidence_call(
                seqs,
                run_memory=structure_evidence_run_memory,
                provider="protenix",
                model_name=resolved_batch_model,
                operation="confidence_complex_batch",
                request_state={
                    "jobs": [
                        {
                            key: value
                            for key, value in job.items()
                            if key != "pred_name"
                        }
                        for job in jobs
                    ]
                },
                call_kwargs=batch_call_kwargs,
                compute=compute_batch,
                estimated_cost=float(math.ceil(len(jobs) / chunk_size)),
            )
            if len(results) != len(indices):
                raise RuntimeError("cached Protenix batch result count is invalid")
            for item_index, result, job in zip(indices, results, jobs):
                batched_confidence[item_index] = _rebind_confidence_pred_name(
                    result, str(job["pred_name"])
                )
                batched_cache_artifacts[item_index] = dict(cache_artifact)

    for prepared_index, prepared in enumerate(prepared_states):
        raw_state = prepared["raw_state"]
        name = str(prepared["name"])
        raw_entities = list(prepared["raw_entities"])
        entities = list(prepared["entities"])
        pred_name = str(prepared["pred_name"])
        metric = str(prepared["metric"])
        evidence_cache_artifact: Optional[Dict[str, Any]] = None
        if prepared_index in batched_confidence:
            confidence = batched_confidence[prepared_index]
            evidence_cache_artifact = batched_cache_artifacts.get(prepared_index)
            entity_units = _infer_complex_entity_units(
                raw_entities, confidence.get("entities", [])
            )
        elif effective_provider == "protenix":
            service_kwargs = (
                _service_transport_kwargs(cfg)
                if selected_provider == "service"
                else {}
            )
            resolved_call_model = (
                model_name
                or cfg.structure_rerank_model_name
                or cfg.structure_model_name
                or cfg.protenix_model_name
            )
            call_kwargs = {
                "metric": metric,
                "seed": cfg.protenix_seed,
                "conda_env": cfg.protenix_conda_env,
                "use_msa": cfg.protenix_complex_use_msa,
                "cycle": cfg.protenix_complex_cycle,
                "step": cfg.protenix_complex_step,
                "sample": cfg.protenix_complex_sample,
                "use_default_params": cfg.protenix_complex_use_default_params,
                "timeout": cfg.protenix_complex_timeout,
                **service_kwargs,
            }
            confidence, evidence_cache_artifact = _cached_structure_evidence_call(
                seqs,
                run_memory=structure_evidence_run_memory,
                provider=selected_provider,
                model_name=resolved_call_model,
                operation="confidence_complex",
                request_state={
                    "entities": entities,
                    "constraint": raw_state.get("constraint"),
                    "covalent_bonds": raw_state.get("covalent_bonds"),
                },
                call_kwargs=call_kwargs,
                compute=lambda: run_structure_confidence_complex(
                    provider=selected_provider,
                    pred_name=pred_name,
                    entities=entities,
                    constraint=raw_state.get("constraint"),
                    covalent_bonds=raw_state.get("covalent_bonds"),
                    model_name=resolved_call_model,
                    **call_kwargs,
                ),
            )
            confidence = _rebind_confidence_pred_name(confidence, pred_name)
            entity_units = _infer_complex_entity_units(raw_entities, confidence.get("entities", []))
        elif effective_provider == "alphafold3":
            service_kwargs = (
                _service_transport_kwargs(cfg)
                if selected_provider == "service"
                else {}
            )
            resolved_call_model = (
                model_name
                or cfg.af3_model_dir
                or cfg.structure_rerank_model_name
                or cfg.structure_model_name
            )
            call_kwargs = {
                "metric": metric,
                "seed": cfg.af3_seed,
                "model_dir": cfg.af3_model_dir,
                "conda_env": cfg.af3_conda_env,
                "timeout": cfg.af3_timeout,
                "run_data_pipeline": cfg.af3_run_data_pipeline,
                "db_dir": cfg.af3_db_dir,
                "num_recycles": cfg.af3_num_recycles,
                "num_diffusion_samples": cfg.af3_num_diffusion_samples,
                "flash_attention_implementation": cfg.af3_flash_attention_implementation,
                "gpu_device": cfg.af3_gpu_device,
                **service_kwargs,
            }
            confidence, evidence_cache_artifact = _cached_structure_evidence_call(
                seqs,
                run_memory=structure_evidence_run_memory,
                provider=selected_provider,
                model_name=resolved_call_model,
                operation="confidence_complex",
                request_state={
                    "entities": entities,
                    "constraint": raw_state.get("constraint"),
                    "covalent_bonds": raw_state.get("covalent_bonds"),
                },
                call_kwargs=call_kwargs,
                compute=lambda: run_structure_confidence_complex(
                    provider=selected_provider,
                    pred_name=pred_name,
                    entities=entities,
                    constraint=raw_state.get("constraint"),
                    covalent_bonds=raw_state.get("covalent_bonds"),
                    model_name=resolved_call_model,
                    **call_kwargs,
                ),
            )
            confidence = _rebind_confidence_pred_name(confidence, pred_name)
            entity_units = _infer_complex_entity_units(
                raw_entities,
                confidence.get("entities", []),
            )
        else:
            chains, entity_units, report_entities = _complex_entities_to_multichain(raw_entities, entities)
            if not chains:
                continue
            structure_kwargs: Dict[str, Any] = {
                "metric": metric,
                "seed": cfg.protenix_seed,
                "model_name": model_name or cfg.structure_screen_model_name or cfg.structure_model_name,
                "mode": cfg.esmfold2_mode,
                "num_loops": cfg.esmfold2_num_loops,
                "num_sampling_steps": cfg.esmfold2_num_sampling_steps,
                "num_diffusion_samples": cfg.esmfold2_num_diffusion_samples,
            }
            if selected_provider == "service":
                structure_kwargs.update(_service_transport_kwargs(cfg))
            else:
                structure_kwargs["conda_env"] = cfg.esmfold2_conda_env
            resolved_call_model = structure_kwargs.get("model_name")
            confidence, evidence_cache_artifact = _cached_structure_evidence_call(
                seqs,
                run_memory=structure_evidence_run_memory,
                provider=selected_provider,
                model_name=resolved_call_model,
                operation="confidence_multichain",
                request_state={"chains": [[cid, seq] for cid, seq in chains]},
                call_kwargs=structure_kwargs,
                compute=lambda: run_structure_confidence_multichain(
                    provider=selected_provider,
                    pred_name=pred_name,
                    chains=chains,
                    **structure_kwargs,
                ),
            )
            confidence = _rebind_confidence_pred_name(confidence, pred_name)
            confidence.setdefault("entities", report_entities)
        source_residue_plddt = _residue_plddt_by_source_chain(confidence, entity_units)
        node_plddt = compute_node_plddt(compiled, source_residue_plddt)
        summary = summarize_structure_metrics(confidence, node_plddt=node_plddt)
        state_result = {
            "name": name,
            "pred_name": pred_name,
            "provider": selected_provider,
            "structure_stage": stage,
            "role": raw_state.get("role"),
            "objective": raw_state.get("objective"),
            "entities": confidence.get("entities", []),
            "entity_units": entity_units,
            "polymer_units": confidence.get("polymer_units", []),
            "metric_units": confidence.get("metric_units", []),
            "confidence_metrics": dict(confidence.get("metrics", {}) or {}),
            "structure_metrics": summary,
            "cif_path": confidence.get("cif_path") or summary.get("cif_path"),
            "input_json": confidence.get("input_json"),
            "summary_json": confidence.get("summary_json"),
            "out_dir": confidence.get("out_dir"),
        }
        if evidence_cache_artifact is not None:
            evidence_cache_artifacts.append(dict(evidence_cache_artifact))
        if any(
            key in confidence
            for key in ("batch_index", "batch_status", "runner_status", "cache_hit")
        ):
            state_result["provider_batch"] = {
                key: confidence.get(key)
                for key in (
                    "batch_index",
                    "batch_status",
                    "runner_status",
                    "cache_hit",
                )
            }
        state_results.append(state_result)
        confidence_metrics_by_state[name] = state_result["confidence_metrics"]
        first_out_dir = first_out_dir or confidence.get("out_dir")
        first_summary_json = first_summary_json or confidence.get("summary_json")

    if not state_results:
        return None, {}, {}, first_out_dir, first_summary_json

    aggregate = _aggregate_complex_state_metrics(state_results)
    cache_summary = _structure_evidence_cache_summary(
        evidence_cache_artifacts
    )
    if cache_summary is not None:


        aggregate["_structure_evidence_cache"] = cache_summary
    if cfg.multistate_objectives_enabled and objective_specs:
        aggregate["multistate_objectives"] = evaluate_multistate_objectives(
            aggregate,
            objective_specs,
            compiled=compiled,
            design_state=state,
        )
    elif objective_specs:
        disable_contract = state.get("multistate_objective_aggregation", {})
        if not isinstance(disable_contract, Mapping):
            disable_contract = {}
        aggregate["multistate_objectives"] = {
            "enabled": False,
            "loss": 0.0,
            "normalized_score": None,
            "reason_code": str(
                disable_contract.get("reason_code")
                or "disabled_by_controller"
            ),
            "authoritative_provider": disable_contract.get(
                "authoritative_provider"
            ),
            "authoritative_runtime": disable_contract.get(
                "authoritative_runtime"
            ),
            "declared_objective_count": len(objective_specs),
        }
    plddt = aggregate.get("scalar", {}).get("plddt")
    return (
        float(plddt) if plddt is not None else 0.0,
        aggregate,
        confidence_metrics_by_state,
        first_out_dir,
        first_summary_json,
    )


def _install_confidence_cache(
    compiled: Dict[str, Any],
    seqs: Dict[str, str],
    chains: List[Tuple[str, str]],
    confidence: Dict[str, Any],
) -> None:


    struct_cache: Dict[Any, Any] = {}
    chain_ids = [cid for cid, _ in chains]
    struct_cache[("residue_plddt_signature", tuple(chain_ids))] = tuple(
        (cid, seqs.get(cid, "")) for cid in chain_ids
    )

    for metric_name, value in (confidence.get("metrics", {}) or {}).items():
        try:
            struct_cache[("scalar", str(metric_name), tuple(chain_ids))] = float(value)
        except (TypeError, ValueError):
            continue

    for cid, vals in (confidence.get("residue_plddt", {}) or {}).items():
        struct_cache[("residue_plddt", str(cid))] = [float(x) for x in vals]

    compiled["_struct_cache"] = struct_cache


def _extract_plddt_delta(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    terms_chai: List[tuple[float, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for _, t in terms_chai:
        if isinstance(t, ChaiPlddtDeltaTerm):
            try:
                mA = t._plddt_for(seqs, t.chains_A)
                mB = t._plddt_for(seqs, t.chains_B)
            except Exception:
                return None, None, None
            if mA is None or mB is None:
                return None, None, None

            return float(mA - mB), float(mA), float(mB)
    return None, None, None


def _structure_sequence_identity(
    candidate: Mapping[str, Any],
    fallback_index: int,
) -> Tuple[str, str]:


    seqs = candidate.get("seqs")
    if isinstance(seqs, Mapping) and seqs:
        normalized = {
            str(chain_id): str(sequence)
            for chain_id, sequence in seqs.items()
        }
        return ("sequence", _seqs_hash(normalized))
    seq_hash = str(candidate.get("seq_hash") or "").strip()
    if seq_hash:
        return ("seq_hash", seq_hash)
    variant_id = str(candidate.get("variant_id") or "").strip()
    return ("candidate", variant_id or str(fallback_index))


def _structure_candidate_nodes(
    candidate: Mapping[str, Any],
    semantic_required_nodes: Sequence[str],
) -> Tuple[str, ...]:


    required = {
        str(node).strip()
        for node in semantic_required_nodes
        if str(node).strip()
    }
    coverage = candidate.get("semantic_final_mutation_coverage")
    if not isinstance(coverage, Mapping):
        coverage = candidate.get("semantic_final_coverage")
    mutations: Any = {}
    if isinstance(coverage, Mapping):


        mutations = coverage.get("active_mutations_by_node")
        if not isinstance(mutations, Mapping):
            mutations = coverage.get("mutations_by_node", {})
    if required and isinstance(mutations, Mapping):
        try:
            min_mutations = max(1, int(coverage.get("min_mutations", 1) or 1))
        except (TypeError, ValueError):
            min_mutations = 1
        covered = {
            str(node)
            for node, count in mutations.items()
            if str(node) in required and int(count or 0) >= min_mutations
        }


        return tuple(sorted(covered))

    move = candidate.get("move")
    if not isinstance(move, Mapping):
        return ()
    move_nodes = {
        str(node).strip()
        for node in (move.get("target_nodes") or [])
        if str(node).strip()
    }
    node = str(move.get("node") or "").strip()
    if node:
        move_nodes.add(node)
    plan = move.get("mutation_plan")
    if isinstance(plan, Mapping):
        for key in ("structural_node_id", "target_node"):
            value = str(plan.get(key) or "").strip()
            if value:
                move_nodes.add(value)
    return tuple(sorted(move_nodes))


def _structure_candidate_depth(candidate: Mapping[str, Any]) -> Optional[int]:
    mcts = candidate.get("mcts")
    if not isinstance(mcts, Mapping):
        return None
    try:
        return int(mcts.get("depth"))
    except (TypeError, ValueError):
        path = mcts.get("path")
        if isinstance(path, list) and path:
            return max(0, len(path) - 1)
    return None


def _summarize_structure_shortlist(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    semantic_required_nodes: Optional[Sequence[str]] = None,
    semantic_anchor_nodes: Optional[Sequence[str]] = None,
    shortlist_policy: Optional[str] = None,
    selection_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    required = tuple(
        dict.fromkeys(
            str(node).strip()
            for node in (semantic_required_nodes or [])
            if str(node).strip()
        )
    )
    required_set = set(required)
    node_signatures = [
        _structure_candidate_nodes(candidate, required)
        for candidate in candidates
    ]
    depths = [
        depth
        for candidate in candidates
        if (depth := _structure_candidate_depth(candidate)) is not None
    ]
    unique_sequences = {
        _structure_sequence_identity(candidate, index)
        for index, candidate in enumerate(candidates)
    }
    summary = {
        "stage": str(stage),
        "candidate_count": len(candidates),
        "unique_sequence_count": len(unique_sequences),
        "covered_nodes": sorted(
            {node for signature in node_signatures for node in signature}
        ),
        "node_signatures": [
            list(signature) for signature in sorted(set(node_signatures))
        ],
        "node_signature_count": len(set(node_signatures)),
        "covered_depths": sorted(set(depths)),
        "depth_count": len(set(depths)),
        "unknown_depth_count": len(candidates) - len(depths),
        "required_nodes": list(required),
        "required_node_joint_candidate_count": (
            sum(required_set.issubset(set(signature)) for signature in node_signatures)
            if required_set
            else None
        ),
    }
    selection_roles = sorted(
        {
            str(candidate.get("structure_shortlist_role"))
            for candidate in candidates
            if str(candidate.get("structure_shortlist_role") or "").strip()
        }
    )
    if selection_roles:
        summary["selection_roles"] = selection_roles
        summary["formal_candidate_count"] = sum(
            str(candidate.get("structure_shortlist_role") or "").startswith(
                "formal_"
            )
            for candidate in candidates
        )
        summary["diagnostic_candidate_count"] = sum(
            str(candidate.get("structure_shortlist_role") or "").endswith(
                "diagnostic"
            )
            for candidate in candidates
        )
    policy = str(shortlist_policy or "").strip().lower()
    layered = policy == "formal_layered_novel" or any(
        str(candidate.get("structure_shortlist_role") or "").strip().lower()
        == "formal_layered_novel"
        for candidate in candidates
    )
    if layered:
        anchors = tuple(
            node
            for node in dict.fromkeys(
                str(node).strip()
                for node in (semantic_anchor_nodes or [])
                if str(node).strip()
            )
            if node in required_set
        )
        anchor_set = set(anchors)
        optional = tuple(node for node in required if node not in anchor_set)
        covered = {node for signature in node_signatures for node in signature}
        audit = selection_audit if isinstance(selection_audit, Mapping) else {}
        lanes = [
            str(candidate.get("lane") or candidate.get("structure_shortlist_lane") or "")
            for candidate in candidates
        ]
        lanes = [lane for lane in lanes if lane]
        requested = int(
            audit.get(
                "formal_quota_requested",
                max(
                    (
                        int(candidate.get("structure_shortlist_quota_requested", 0) or 0)
                        for candidate in candidates
                    ),
                    default=len(candidates),
                ),
            )
            or 0
        )
        underfill = bool(
            audit.get(
                "underfill",
                audit.get(
                    "formal_underfilled",
                    any(bool(candidate.get("underfill")) for candidate in candidates),
                ),
            )
        )
        summary.update(
            {
                "lane": lanes,
                "lane_counts": {
                    lane: lanes.count(lane) for lane in sorted(set(lanes))
                },
                "coverage_scope": "shortlist_set",
                "anchor_nodes": list(anchors),
                "optional_nodes": list(optional),
                "anchor_union_pass": anchor_set.issubset(covered),
                "optional_represented": bool(set(optional) & covered),
                "active_union_pass": required_set.issubset(covered),
                "missing_anchor_nodes": sorted(anchor_set - covered),
                "represented_optional_nodes": sorted(set(optional) & covered),
                "underfill": underfill,
                "underfill_reasons": list(
                    audit.get("underfill_reasons", audit.get("formal_underfill_reasons", []))
                    or []
                ),
                "mutant_quota_requested": requested,
            }
        )
    return summary


def _enforce_layered_shortlist_semantic_gate(
    semantic_audit: Dict[str, Any],
    stage_summaries: Mapping[str, Mapping[str, Any]],
    *,
    enabled: bool,
) -> Dict[str, Any]:


    if not enabled:
        return semantic_audit
    stage = next(
        (
            name
            for name in ("rerank", "screen", "legacy")
            if isinstance(stage_summaries.get(name), Mapping)
        ),
        None,
    )
    summary = dict(stage_summaries.get(stage, {}) or {}) if stage else {}
    suppression = summary.get("cross_round_failure_suppression")
    suppression = suppression if isinstance(suppression, Mapping) else {}
    requested = int(
        summary.get(
            "mutant_quota_requested",
            suppression.get("formal_quota_requested", 0),
        )
        or 0
    )
    selected = int(
        suppression.get("formal_selected_count", summary.get("candidate_count", 0))
        or 0
    )
    anchors_pass = bool(summary.get("anchor_union_pass", False))
    optional_nodes = list(summary.get("optional_nodes", []) or [])
    optional_represented = bool(summary.get("optional_represented", False))
    reasons: List[str] = []
    if stage is None:
        reasons.append("layered_shortlist_summary_missing")
    if selected != requested:
        reasons.append("layered_shortlist_quota_underfilled")
    if not anchors_pass:
        reasons.append("layered_shortlist_anchor_union_incomplete")
    if optional_nodes and not optional_represented:
        reasons.append("layered_shortlist_optional_node_unrepresented")
    report = {
        "schema_version": "astevolve.layered_shortlist_set_coverage.v1",
        "coverage_scope": "shortlist_set",
        "stage": stage,
        "selected_count": selected,
        "requested_count": requested,
        "quota_pass": selected == requested,
        "anchor_union_pass": anchors_pass,
        "optional_nodes": optional_nodes,
        "optional_represented": optional_represented,
        "underfill": bool(summary.get("underfill", selected != requested)),
        "pass": not reasons,
        "hard_gate_reasons": reasons,
        "stage_summary": summary,
    }
    semantic_audit["semantic_shortlist_set_coverage"] = report
    if reasons:
        semantic_audit["hard_gate_pass"] = False
        semantic_audit["hard_gate_reasons"] = list(
            dict.fromkeys(
                list(semantic_audit.get("hard_gate_reasons", []) or []) + reasons
            )
        )
    return semantic_audit


def _structure_candidate_stable_id(
    candidate: Mapping[str, Any],
    index: int,
) -> str:
    identity_kind, identity_value = _structure_sequence_identity(candidate, index)
    variant_id = str(candidate.get("variant_id") or "")
    return f"{identity_kind}:{identity_value}:{variant_id}"


def _structure_candidate_is_feasible(candidate: Mapping[str, Any]) -> bool:


    sources: Dict[str, Any] = {}
    raw_sources = candidate.get("feasibility_gate_sources")
    if isinstance(raw_sources, Mapping):
        sources.update(raw_sources)
    if "fast_filter" not in sources and isinstance(
        candidate.get("fast_filter"), Mapping
    ):
        sources["fast_filter"] = candidate.get("fast_filter")
    if "inner_structure" not in sources and "inner_structure_gate_pass" in candidate:
        sources["inner_structure"] = {"pass": bool(candidate.get("inner_structure_gate_pass"))}
    coverage = candidate.get("semantic_final_mutation_coverage")
    if isinstance(coverage, Mapping):
        sources["semantic_final_mutation_coverage"] = {
            "pass": bool(coverage.get("pass", True)),
            "reasons": list(
                coverage.get("missing_required_nodes_by_mutation", []) or []
            ),
        }
    return all(
        bool(normalize_gate_payload(payload)["passed"])
        for payload in sources.values()
    )


def _structure_candidate_is_layered_feasible(
    candidate: Mapping[str, Any],
) -> bool:


    raw_sources = candidate.get("feasibility_gate_sources")
    sources: Dict[str, Any] = {
        str(source): payload
        for source, payload in (
            raw_sources.items() if isinstance(raw_sources, Mapping) else []
        )
        if str(source) not in {
            "semantic_final_mutation_coverage",
            "semantic_final_coverage",
        }
    }
    if "fast_filter" not in sources and isinstance(
        candidate.get("fast_filter"), Mapping
    ):
        sources["fast_filter"] = candidate.get("fast_filter")
    if "inner_structure" not in sources and "inner_structure_gate_pass" in candidate:
        sources["inner_structure"] = {"pass": bool(candidate.get("inner_structure_gate_pass"))}
    return all(
        bool(normalize_gate_payload(payload)["passed"])
        for payload in sources.values()
    )


def _structure_candidate_feasibility_priority(
    candidate: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:


    direct = candidate.get("feasibility_priority")
    payloads: List[Any] = [direct]
    raw_sources = candidate.get("feasibility_gate_sources")
    if isinstance(raw_sources, Mapping):
        payloads.extend(raw_sources.values())
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        nested = payload.get("gate_status")
        direct_priority = (
            payload
            if "hard_gate_pass_count" in payload
            else payload.get("feasibility_priority")
        )
        for container in (direct_priority, nested):
            if not isinstance(container, Mapping):
                continue
            priority = (
                container.get("feasibility_priority")
                if "hard_gate_pass_count" not in container
                else container
            )
            if not isinstance(priority, Mapping):
                continue
            try:
                return {
                    "hard_gate_pass_count": int(
                        priority.get("hard_gate_pass_count", 0) or 0
                    ),
                    "hard_gate_total": int(
                        priority.get("hard_gate_total", 0) or 0
                    ),
                    "min_hard_margin": float(
                        priority.get("min_hard_margin", 0.0) or 0.0
                    ),
                    "joint_hard_margin": float(
                        priority.get("joint_hard_margin", 0.0) or 0.0
                    ),
                }
            except (TypeError, ValueError):
                return None
    return None


def _structure_candidate_is_rerank_rescue_eligible(
    candidate: Mapping[str, Any],
    *,
    layered_coverage_scope: bool,
) -> bool:


    priority = _structure_candidate_feasibility_priority(candidate)
    if priority is None:
        return False
    raw_sources = candidate.get("feasibility_gate_sources")
    sources: Dict[str, Any] = {
        str(source): payload
        for source, payload in (
            raw_sources.items() if isinstance(raw_sources, Mapping) else []
        )
    }
    if "fast_filter" not in sources and isinstance(
        candidate.get("fast_filter"), Mapping
    ):
        sources["fast_filter"] = candidate.get("fast_filter")
    for source, payload in sources.items():
        if source == "inner_evaluator":
            continue
        if layered_coverage_scope and source in {
            "semantic_final_mutation_coverage",
            "semantic_final_coverage",
        }:
            continue
        if not bool(normalize_gate_payload(payload)["passed"]):
            return False
    return True


def _formal_joint_semantic_candidate(
    candidate: Mapping[str, Any],
    required: Sequence[str],
) -> bool:
    required_set = {
        str(node).strip() for node in required if str(node).strip()
    }
    if not required_set:
        return False
    return required_set.issubset(
        set(_structure_candidate_nodes(candidate, tuple(required_set)))
    )


def _position_distribution_engagement(
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:


    empty = {
        "schema_version": "astevolve.position_distribution_engagement.v1",
        "validated_receipt": False,
        "incremental_change_count": 0,
        "soft_incremental_change_count": 0,
        "required_incremental_change_count": 0,
        "changed_distribution_hashes": [],
    }
    fast_filter = candidate.get("fast_filter")
    if not isinstance(fast_filter, Mapping):
        return empty
    validation = fast_filter.get("design_action_validation")
    if not isinstance(validation, Mapping):
        return empty
    receipt = validation.get("receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version")
        != "astevolve.candidate_design_action_validation.v2"
        or receipt.get("passed") is not True
    ):
        return empty
    incremental = receipt.get("incremental_mutations")
    realizations = receipt.get("position_distribution_realization")
    if not isinstance(incremental, Mapping) or not isinstance(realizations, list):
        return empty
    changed_positions = {
        (str(change.get("chain_id") or ""), int(change.get("position")))
        for change in (incremental.get("changes") or [])
        if isinstance(change, Mapping)
        and str(change.get("chain_id") or "")
        and isinstance(change.get("position"), int)
        and not isinstance(change.get("position"), bool)
    }
    changed_rows = [
        row
        for row in realizations
        if isinstance(row, Mapping)
        and isinstance(row.get("position"), int)
        and not isinstance(row.get("position"), bool)
        and (str(row.get("chain_id") or ""), int(row["position"]))
        in changed_positions
    ]
    changed_hashes = sorted(
        {
            str(row.get("distribution_hash") or "")
            for row in changed_rows
            if str(row.get("distribution_hash") or "")
        }
    )
    return {
        "schema_version": "astevolve.position_distribution_engagement.v1",
        "validated_receipt": True,
        "incremental_change_count": len(changed_rows),
        "soft_incremental_change_count": sum(
            row.get("required_mutation") is None for row in changed_rows
        ),
        "required_incremental_change_count": sum(
            row.get("required_mutation") is not None for row in changed_rows
        ),
        "changed_distribution_hashes": changed_hashes,
    }


_PORTFOLIO_REALIZATION_RECEIPT_VERSION = (
    "astevolve.portfolio_realization_receipt.v1"
)
_PORTFOLIO_REALIZATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "compiled_portfolio_request_hash",
        "slot_id",
        "slot_hash",
        "portfolio_id",
        "role",
        "generation_mode",
        "candidate_sequence_bundle_hash",
        "expected_sequence_bundle_hash",
        "exact_sequence_match",
        "atomic_requested_count",
        "atomic_realized_count",
        "partial_realization_count",
        "matched_pair_id",
        "receipt_hash",
    }
)
_PORTFOLIO_ROLE_PRIORITY = {
    role: priority
    for priority, role in enumerate(
        (
            "primary",
            "repair",
            "secondary",
            "novelty",
            "control",
            "matched_ablation",
        )
    )
}
_PORTFOLIO_GENERATION_MODES = frozenset(
    {"strict_sequence_seed", "module", "mcts", "matched_ablation", "control"}
)


def _is_prefixed_sha256(value: Any, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _validated_portfolio_realization_receipts(
    candidate: Mapping[str, Any],
) -> List[Dict[str, Any]]:


    request_hash = candidate.get("compiled_portfolio_request_hash")
    if not _is_prefixed_sha256(
        request_hash, "compiled_portfolio_request_sha256:"
    ):
        return []
    seqs = candidate.get("seqs")
    if not isinstance(seqs, Mapping) or not seqs:
        return []
    try:
        candidate_sequence_hash = SequenceBundleIdentity.create(
            {str(chain): str(sequence) for chain, sequence in seqs.items()}
        ).sequence_bundle_hash
    except (TypeError, ValueError):
        return []
    raw_receipts = candidate.get("portfolio_realization_receipts")
    if not isinstance(raw_receipts, list):
        return []

    validated: List[Dict[str, Any]] = []
    for receipt in raw_receipts:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != _PORTFOLIO_REALIZATION_RECEIPT_FIELDS
            or receipt.get("schema_version")
            != _PORTFOLIO_REALIZATION_RECEIPT_VERSION
            or receipt.get("compiled_portfolio_request_hash") != request_hash
            or not _is_prefixed_sha256(
                receipt.get("slot_hash"), "compiled_candidate_slot_sha256:"
            )
            or receipt.get("candidate_sequence_bundle_hash")
            != candidate_sequence_hash
            or receipt.get("expected_sequence_bundle_hash")
            != candidate_sequence_hash
            or receipt.get("exact_sequence_match") is not True
        ):
            continue
        if any(
            not isinstance(receipt.get(field), str)
            or not str(receipt.get(field)).strip()
            for field in ("slot_id", "portfolio_id")
        ):
            continue
        role = receipt.get("role")
        generation_mode = receipt.get("generation_mode")
        if (
            role not in _PORTFOLIO_ROLE_PRIORITY
            or generation_mode not in _PORTFOLIO_GENERATION_MODES
        ):
            continue
        matched_pair_id = receipt.get("matched_pair_id")
        if matched_pair_id is not None and (
            not isinstance(matched_pair_id, str) or not matched_pair_id.strip()
        ):
            continue
        requested_count = receipt.get("atomic_requested_count")
        realized_count = receipt.get("atomic_realized_count")
        partial_count = receipt.get("partial_realization_count")
        if (
            isinstance(requested_count, bool)
            or not isinstance(requested_count, int)
            or requested_count < 0
            or isinstance(realized_count, bool)
            or not isinstance(realized_count, int)
            or realized_count != requested_count
            or isinstance(partial_count, bool)
            or not isinstance(partial_count, int)
            or partial_count != 0
        ):
            continue
        semantic = {
            key: receipt[key]
            for key in _PORTFOLIO_REALIZATION_RECEIPT_FIELDS
            if key != "receipt_hash"
        }
        try:
            canonical = json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            continue
        expected_receipt_hash = "portfolio_realization_sha256:" + hashlib.sha256(
            (
                _PORTFOLIO_REALIZATION_RECEIPT_VERSION
                + "\0"
                + canonical
            ).encode("utf-8")
        ).hexdigest()
        if receipt.get("receipt_hash") != expected_receipt_hash:
            continue
        validated.append(dict(receipt))
    return sorted(
        validated,
        key=lambda receipt: (
            _PORTFOLIO_ROLE_PRIORITY[str(receipt["role"])],
            str(receipt["slot_id"]),
            str(receipt["slot_hash"]),
            str(receipt["receipt_hash"]),
        ),
    )


def _formal_candidate_order_key(
    candidate: Mapping[str, Any],
    index: int,
    *,
    ranking_source: str,
    layered_coverage_scope: bool = False,
) -> Tuple[Any, ...]:


    if ranking_source == "screen_combined_energy":
        objective = float(
            candidate.get(
                "combined_energy",
                candidate.get("outer_aligned_energy", candidate.get("fast_loss", 0.0)),
            )
        )
        plddt = float(candidate.get("plddt", 0.0) or 0.0)
    elif ranking_source == "selection_loss":
        objective = float(candidate.get("selection_loss", candidate.get("fast_loss", 0.0)))
        plddt = 0.0
    else:
        objective = float(candidate.get("fast_loss", 0.0))
        plddt = 0.0
    if not math.isfinite(objective):
        objective = math.inf
    if not math.isfinite(plddt):
        plddt = 0.0
    feasible = (
        _structure_candidate_is_layered_feasible(candidate)
        if layered_coverage_scope
        else _structure_candidate_is_feasible(candidate)
    )
    priority = _structure_candidate_feasibility_priority(candidate)
    if feasible:
        feasibility_key: Tuple[Any, ...] = (0, 0, 0.0, 0.0)
    elif priority is not None:
        feasibility_key = (
            1,
            -int(priority["hard_gate_pass_count"]),
            -float(priority["min_hard_margin"]),
            -float(priority["joint_hard_margin"]),
        )
    else:
        feasibility_key = (2, 0, 0.0, 0.0)
    return (
        *feasibility_key,
        objective,
        -plddt,
        _structure_candidate_stable_id(candidate, index),
    )


def _structure_sequence_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> int:


    left_seqs = left.get("seqs")
    right_seqs = right.get("seqs")
    if not isinstance(left_seqs, Mapping) or not isinstance(right_seqs, Mapping):
        return int(
            _structure_sequence_identity(left, -1)
            != _structure_sequence_identity(right, -2)
        )
    distance = 0
    chain_ids = sorted({str(key) for key in left_seqs} | {str(key) for key in right_seqs})
    for chain_id in chain_ids:
        left_sequence = str(left_seqs.get(chain_id, ""))
        right_sequence = str(right_seqs.get(chain_id, ""))
        overlap = min(len(left_sequence), len(right_sequence))
        distance += sum(
            left_sequence[index] != right_sequence[index]
            for index in range(overlap)
        )
        distance += abs(len(left_sequence) - len(right_sequence))
    return int(distance)


def _select_fast_proxy_diverse_formal_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    quota: int,
) -> List[Dict[str, Any]]:


    quota = min(max(0, int(quota)), len(candidates))
    if quota == 0:
        return []


    pool_size = min(len(candidates), max(quota, 4 * quota))
    pool = list(candidates[:pool_size])
    selected = [pool[0]]
    selected_ids = {id(pool[0])}
    while len(selected) < quota:
        remaining = [item for item in pool if id(item) not in selected_ids]
        if not remaining:
            break


        ranked = {id(item): index for index, item in enumerate(pool)}
        choice = max(
            remaining,
            key=lambda item: (
                min(
                    _structure_sequence_distance(item, chosen)
                    for chosen in selected
                ),
                -ranked[id(item)],
            ),
        )
        selected.append(choice)
        selected_ids.add(id(choice))
    return selected


def _select_layered_formal_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    quota: int,
    semantic_active_nodes: Sequence[str],
    semantic_anchor_nodes: Sequence[str],
    leader_count: int = 1,
    reserved_portfolio_candidates: Sequence[Dict[str, Any]] = (),
    reserved_position_distribution_candidates: Sequence[Dict[str, Any]] = (),
) -> List[Tuple[Dict[str, Any], str]]:


    quota = min(max(0, int(quota)), len(candidates))
    if quota == 0:
        return []
    active = tuple(
        dict.fromkeys(
            str(node).strip()
            for node in semantic_active_nodes
            if str(node).strip()
        )
    )
    active_set = set(active)
    anchors = tuple(
        node
        for node in dict.fromkeys(
            str(node).strip()
            for node in semantic_anchor_nodes
            if str(node).strip()
        )
        if node in active_set
    )
    anchor_set = set(anchors)
    optional_set = active_set - anchor_set
    nodes = {
        id(candidate): set(_structure_candidate_nodes(candidate, active))
        for candidate in candidates
    }
    rank = {id(candidate): index for index, candidate in enumerate(candidates)}

    selected: List[Tuple[Dict[str, Any], str]] = []
    selected_ids: set[int] = set()
    candidate_ids = {id(candidate) for candidate in candidates}
    for candidate in reserved_portfolio_candidates:
        if (
            len(selected) >= quota
            or id(candidate) not in candidate_ids
            or id(candidate) in selected_ids
        ):
            continue
        selected.append((candidate, "portfolio_contract"))
        selected_ids.add(id(candidate))
    for candidate in reserved_position_distribution_candidates:
        if (
            len(selected) >= quota
            or id(candidate) not in candidate_ids
            or id(candidate) in selected_ids
        ):
            continue
        selected.append((candidate, "position_distribution_engagement"))
        selected_ids.add(id(candidate))

    leader_count = min(
        max(0, int(leader_count)),
        max(0, quota - len(selected)),
    )
    leader_candidates = [
        candidate for candidate in candidates if id(candidate) not in selected_ids
    ][:leader_count]
    for index, candidate in enumerate(leader_candidates):
        selected.append(
            (
                candidate,
                "fast_leader" if index == 0 else "feasibility_rescue_leader",
            )
        )
        selected_ids.add(id(candidate))

    def selected_nodes(extra: Optional[Mapping[str, Any]] = None) -> set[str]:
        covered = {
            node
            for candidate, _lane in selected
            for node in nodes[id(candidate)]
        }
        if extra is not None:
            covered.update(nodes[id(extra)])
        return covered

    if len(selected) < quota:
        remaining = [
            candidate for candidate in candidates if id(candidate) not in selected_ids
        ]
        slots_after_anchor = quota - len(selected) - 1

        def anchor_choice_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
            covered = selected_nodes(candidate)
            future = [item for item in remaining if id(item) != id(candidate)]
            if slots_after_anchor > 0:
                attainable_anchor = anchor_set.issubset(covered) or any(
                    anchor_set.issubset(covered | nodes[id(item)])
                    for item in future
                )
                attainable_optional = bool(optional_set & covered) or any(
                    bool(optional_set & (covered | nodes[id(item)]))
                    for item in future
                )
                attainable_both = (
                    anchor_set.issubset(covered)
                    and bool(optional_set & covered)
                ) or any(
                    anchor_set.issubset(covered | nodes[id(item)])
                    and bool(optional_set & (covered | nodes[id(item)]))
                    for item in future
                )
            else:
                attainable_anchor = anchor_set.issubset(covered)
                attainable_optional = bool(optional_set & covered)
                attainable_both = attainable_anchor and attainable_optional
            already = selected_nodes()
            return (
                int(attainable_both),
                int(attainable_anchor),
                int(attainable_optional),
                len((nodes[id(candidate)] & anchor_set) - already),
                len(nodes[id(candidate)] & anchor_set),
                int(tuple(sorted(nodes[id(candidate)])) not in {
                    tuple(sorted(nodes[id(item)])) for item, _lane in selected
                }),
                min(
                    _structure_sequence_distance(candidate, item)
                    for item, _lane in selected
                ),
                -rank[id(candidate)],
            )

        if remaining:
            anchor_candidate = max(remaining, key=anchor_choice_key)
            selected.append((anchor_candidate, "anchor_coverage"))
            selected_ids.add(id(anchor_candidate))

    while len(selected) < quota:
        remaining = [
            candidate for candidate in candidates if id(candidate) not in selected_ids
        ]
        if not remaining:
            break
        covered = selected_nodes()
        signatures = {
            tuple(sorted(nodes[id(candidate)])) for candidate, _lane in selected
        }

        def optional_choice_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
            combined = covered | nodes[id(candidate)]
            signature = tuple(sorted(nodes[id(candidate)]))
            return (
                int(anchor_set.issubset(combined)),
                int(bool(optional_set & combined)),
                len((nodes[id(candidate)] & anchor_set) - covered),
                len((nodes[id(candidate)] & optional_set) - covered),
                len(nodes[id(candidate)] & optional_set),
                int(signature not in signatures),
                min(
                    _structure_sequence_distance(candidate, item)
                    for item, _lane in selected
                ),
                -rank[id(candidate)],
            )

        optional_candidate = max(remaining, key=optional_choice_key)
        selected.append((optional_candidate, "optional_diversity"))
        selected_ids.add(id(optional_candidate))

    return selected


def _select_single_node_structure_diagnostics(
    candidates: Sequence[Dict[str, Any]],
    *,
    quota: int,
    semantic_required_nodes: Sequence[str],
    reference_candidate: Optional[Mapping[str, Any]] = None,
    excluded_candidates: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:


    quota = max(0, int(quota))
    required = tuple(
        dict.fromkeys(
            str(node).strip()
            for node in semantic_required_nodes
            if str(node).strip()
        )
    )
    if quota == 0 or len(required) < 2:
        return []
    excluded_identities = {
        _structure_sequence_identity(item, index)
        for index, item in enumerate(excluded_candidates)
    }
    if reference_candidate is not None:
        excluded_identities.add(
            _structure_sequence_identity(reference_candidate, -1)
        )
    unique: List[Tuple[int, Dict[str, Any]]] = []
    seen = set(excluded_identities)
    for index, candidate in enumerate(candidates):
        identity = _structure_sequence_identity(candidate, index)
        if identity in seen:
            continue
        seen.add(identity)
        signature = _structure_candidate_nodes(candidate, required)
        if len(signature) != 1:
            continue
        unique.append((index, candidate))
    ordered = sorted(
        unique,
        key=lambda item: _formal_candidate_order_key(
            item[1], item[0], ranking_source="fast_loss"
        ),
    )
    selected = [candidate for _index, candidate in ordered[:quota]]
    for candidate in selected:
        candidate["structure_shortlist_role"] = "single_node_diagnostic"
        candidate["formal_rerank_eligible"] = False
    return selected


def _select_structure_candidates(
    candidates: List[Dict[str, Any]],
    *,
    top_frac: float,
    min_candidates: int,
    max_candidates: int,
    all_candidates: bool = False,
    semantic_required_nodes: Optional[Sequence[str]] = None,
    semantic_anchor_nodes: Optional[Sequence[str]] = None,
    shortlist_policy: str = "legacy_diverse",
    reference_candidate: Optional[Mapping[str, Any]] = None,
    ranking_source: str = "fast_loss",
    failure_memory: Optional[Mapping[str, Any]] = None,
    failure_context: Optional[Mapping[str, Any]] = None,
    suppression_audit: Optional[Dict[str, Any]] = None,
    allow_all_infeasible_rescue: bool = False,
    position_distribution_engagement_quota: int = 0,
    portfolio_contract_quota: int = 0,
) -> List[Dict[str, Any]]:


    policy = str(shortlist_policy).strip().lower()
    if policy not in {
        "legacy_diverse",
        "formal_joint_novel",
        "formal_layered_novel",
    }:
        raise ValueError(f"unsupported structure shortlist policy: {policy}")
    ranking_source = str(ranking_source).strip().lower()
    if ranking_source not in {"fast_loss", "selection_loss", "screen_combined_energy"}:
        raise ValueError(f"unsupported structure shortlist ranking: {ranking_source}")
    engagement_quota_configured = int(position_distribution_engagement_quota)
    if engagement_quota_configured < 0:
        raise ValueError(
            "position_distribution_engagement_quota must be non-negative"
        )
    if engagement_quota_configured and policy not in {
        "formal_joint_novel",
        "formal_layered_novel",
    }:
        raise ValueError(
            "position_distribution_engagement_quota requires a formal shortlist policy"
        )
    portfolio_quota_configured = int(portfolio_contract_quota)
    if portfolio_quota_configured < 0:
        raise ValueError("portfolio_contract_quota must be non-negative")
    if portfolio_quota_configured and policy not in {
        "formal_joint_novel",
        "formal_layered_novel",
    }:
        raise ValueError(
            "portfolio_contract_quota requires a formal shortlist policy"
        )

    if not candidates:
        if policy in {
            "formal_joint_novel",
            "formal_layered_novel",
        } and suppression_audit is not None:
            _empty, audit = suppress_exact_structure_failures(
                [],
                memory=failure_memory,
                context=(
                    failure_context
                    if policy == "formal_joint_novel"
                    or str((failure_context or {}).get("provider") or "").lower()
                    == "protenix"
                    else None
                ),
                root_seq_hash="",
            )
            suppression_audit.clear()
            suppression_audit.update(audit)
            suppression_audit.update(
                {
                    "formal_quota_requested": 0,
                    "formal_eligible_after_all_filters": 0,
                    "formal_selected_count": 0,
                    "formal_underfilled": False,
                    "formal_underfill_reasons": [],
                    "position_distribution_engagement_quota_configured": (
                        engagement_quota_configured
                    ),
                    "position_distribution_engagement_quota_requested": 0,
                    "position_distribution_engagement_available_count": 0,
                    "position_distribution_engagement_selected_count": 0,
                    "position_distribution_engagement_underfilled": False,
                    "position_distribution_engagement_underfill_reasons": [],
                    "position_distribution_engagement_selected": [],
                }
            )
            if portfolio_quota_configured:
                suppression_audit.update(
                    {
                        "portfolio_contract_quota_configured": (
                            portfolio_quota_configured
                        ),
                        "portfolio_contract_quota_requested": 0,
                        "portfolio_contract_available_count": 0,
                        "portfolio_contract_selected_count": 0,
                        "portfolio_contract_underfilled": False,
                        "portfolio_contract_underfill_reasons": [],
                        "portfolio_contract_selected": [],
                    }
                )
            if policy == "formal_layered_novel":
                suppression_audit.update(
                    {
                        "coverage_scope": "shortlist_set",
                        "anchor_union_pass": False,
                        "optional_represented": False,
                        "underfill": False,
                        "underfill_reasons": [],
                    }
                )
        return []

    indexed_candidates = list(enumerate(candidates))
    rescue_audit: Dict[str, Any] = {
        "all_infeasible_rescue_enabled": bool(allow_all_infeasible_rescue),
        "all_infeasible_rescue_used": False,
        "all_infeasible_rescue_eligible_count": 0,
    }
    if policy == "formal_joint_novel":


        feasible_candidates = [
            item
            for item in indexed_candidates
            if _structure_candidate_is_feasible(item[1])
        ]
        if feasible_candidates or not allow_all_infeasible_rescue:
            indexed_candidates = feasible_candidates
        else:
            indexed_candidates = [
                item
                for item in indexed_candidates
                if _structure_candidate_is_rerank_rescue_eligible(
                    item[1], layered_coverage_scope=False
                )
            ]
            rescue_audit.update(
                {
                    "all_infeasible_rescue_used": bool(indexed_candidates),
                    "all_infeasible_rescue_eligible_count": len(indexed_candidates),
                }
            )
        root_identity = (
            _structure_sequence_identity(reference_candidate, -1)[1]
            if reference_candidate is not None
            else ""
        )
        unsuppressed, audit = suppress_exact_structure_failures(
            [candidate for _index, candidate in indexed_candidates],
            memory=failure_memory,
            context=failure_context,
            root_seq_hash=root_identity,
        )
        indexed_candidates = list(enumerate(unsuppressed))
        if suppression_audit is not None:
            suppression_audit.clear()
            suppression_audit.update(audit)
            suppression_audit.update(rescue_audit)
    elif policy == "formal_layered_novel":


        feasible_candidates = [
            item
            for item in indexed_candidates
            if _structure_candidate_is_layered_feasible(item[1])
        ]
        if feasible_candidates or not allow_all_infeasible_rescue:
            indexed_candidates = feasible_candidates
        else:
            indexed_candidates = [
                item
                for item in indexed_candidates
                if _structure_candidate_is_rerank_rescue_eligible(
                    item[1], layered_coverage_scope=True
                )
            ]
            rescue_audit.update(
                {
                    "all_infeasible_rescue_used": bool(indexed_candidates),
                    "all_infeasible_rescue_eligible_count": len(indexed_candidates),
                }
            )
    if policy in {"formal_joint_novel", "formal_layered_novel"}:
        for _original_index, candidate in indexed_candidates:
            candidate["position_distribution_engagement"] = (
                _position_distribution_engagement(candidate)
            )
    candidates_sorted = sorted(
        indexed_candidates,
        key=lambda item: _formal_candidate_order_key(
            item[1],
            item[0],
            ranking_source=ranking_source,
            layered_coverage_scope=policy == "formal_layered_novel",
        ),
    )
    unique_candidates: List[Dict[str, Any]] = []
    seen_sequences: set[Tuple[str, str]] = set()
    for original_index, candidate in candidates_sorted:
        identity = _structure_sequence_identity(candidate, original_index)
        if identity in seen_sequences:
            continue
        seen_sequences.add(identity)
        unique_candidates.append(candidate)

    required = tuple(
        dict.fromkeys(
            str(node).strip()
            for node in (semantic_required_nodes or [])
            if str(node).strip()
        )
    )
    anchors = tuple(
        node
        for node in dict.fromkeys(
            str(node).strip()
            for node in (semantic_anchor_nodes or [])
            if str(node).strip()
        )
        if node in set(required)
    )
    if policy == "formal_joint_novel":
        if reference_candidate is not None:
            reference_identity = _structure_sequence_identity(
                reference_candidate, -1
            )
            unique_candidates = [
                candidate
                for index, candidate in enumerate(unique_candidates)
                if _structure_sequence_identity(candidate, index)
                != reference_identity
            ]
        unique_candidates = [
            candidate
            for candidate in unique_candidates
            if _formal_joint_semantic_candidate(candidate, required)
        ]
    elif policy == "formal_layered_novel":


        if reference_candidate is not None:
            reference_identity = _structure_sequence_identity(
                reference_candidate, -1
            )
            unique_candidates = [
                candidate
                for index, candidate in enumerate(unique_candidates)
                if _structure_sequence_identity(candidate, index)
                != reference_identity
            ]
        root_seq_hash = (
            _structure_sequence_identity(reference_candidate, -1)[1]
            if reference_candidate is not None
            else ""
        )
        provider_scope_match = (
            str((failure_context or {}).get("provider") or "").strip().lower()
            == "protenix"
        )
        unsuppressed, audit = suppress_exact_structure_failures(
            unique_candidates,
            memory=failure_memory,
            context=failure_context if provider_scope_match else None,
            root_seq_hash=root_seq_hash,
        )
        unique_candidates = list(unsuppressed)
        audit["provider_scope"] = "protenix"
        audit["provider_scope_match"] = provider_scope_match
        audit["sequence_novelty_applied_before_suppression"] = True
        if suppression_audit is not None:
            suppression_audit.clear()
            suppression_audit.update(audit)
            suppression_audit.update(rescue_audit)

    if all_candidates:
        k = len(unique_candidates)
    else:


        k = max(
            int(min_candidates),
            int(round(float(top_frac) * len(candidates_sorted))),
        )
    if int(max_candidates) > 0:
        k = min(k, int(max_candidates))
    k = min(max(k, 0), len(unique_candidates))
    if policy == "formal_joint_novel" and suppression_audit is not None:
        before_count = int(
            suppression_audit.get("candidate_count_before", len(candidates_sorted))
            or 0
        )
        if all_candidates:
            requested = before_count
        else:
            requested = max(
                int(min_candidates),
                int(round(float(top_frac) * before_count)),
            )
        if int(max_candidates) > 0:
            requested = min(requested, int(max_candidates))
        requested = max(0, requested)
        reasons: List[str] = []
        if int(suppression_audit.get("suppressed_candidate_count", 0) or 0):
            reasons.append("same_context_exact_failure_suppression")
        if len(unique_candidates) < requested:
            reasons.append("formal_semantic_feasibility_or_novelty_underfill")
        suppression_audit.update(
            {
                "formal_quota_requested": requested,
                "formal_eligible_after_all_filters": len(unique_candidates),
                "formal_selected_count": k,
                "formal_underfilled": bool(k < requested),
                "formal_underfill_reasons": reasons if k < requested else [],
            }
        )
    elif policy == "formal_layered_novel" and suppression_audit is not None:
        before_count = int(
            suppression_audit.get("candidate_count_before", len(candidates_sorted))
            or 0
        )
        if all_candidates:
            requested = before_count
        else:
            requested = max(
                int(min_candidates),
                int(round(float(top_frac) * before_count)),
            )
        if int(max_candidates) > 0:
            requested = min(requested, int(max_candidates))
        requested = max(0, requested)
        reasons: List[str] = []
        if int(suppression_audit.get("suppressed_candidate_count", 0) or 0):
            reasons.append("same_context_exact_protenix_failure_suppression")
        if len(unique_candidates) < requested:
            reasons.append("layered_feasibility_novelty_or_failure_underfill")
        suppression_audit.update(
            {
                "formal_quota_requested": requested,
                "formal_eligible_after_all_filters": len(unique_candidates),
                "formal_selected_count": k,
                "formal_underfilled": bool(k < requested),
                "formal_underfill_reasons": reasons if k < requested else [],
                "coverage_scope": "shortlist_set",
                "underfill": bool(k < requested),
                "underfill_reasons": reasons if k < requested else [],
            }
        )

    formal_selection_capacity = k
    portfolio_quota_requested = min(
        portfolio_quota_configured, formal_selection_capacity
    )
    portfolio_receipts: Dict[int, Dict[str, Any]] = {}
    if portfolio_quota_configured:
        for candidate in unique_candidates:
            receipts = _validated_portfolio_realization_receipts(candidate)
            if receipts:
                portfolio_receipts[id(candidate)] = receipts[0]
    proxy_order = {
        id(candidate): index
        for index, candidate in enumerate(unique_candidates)
    }
    portfolio_candidates = sorted(
        (
            candidate
            for candidate in unique_candidates
            if id(candidate) in portfolio_receipts
        ),
        key=lambda candidate: (
            _PORTFOLIO_ROLE_PRIORITY[
                str(portfolio_receipts[id(candidate)]["role"])
            ],
            proxy_order[id(candidate)],
        ),
    )
    reserved_portfolio_candidates = list(
        portfolio_candidates[:portfolio_quota_requested]
    )
    portfolio_underfilled = (
        len(portfolio_candidates) < portfolio_quota_requested
    )


    engagement_capacity = formal_selection_capacity
    reserved_portfolio_ids: set[int] = set()
    if portfolio_quota_configured:
        engagement_capacity = max(
            0, formal_selection_capacity - portfolio_quota_requested
        )
        reserved_portfolio_ids = {
            id(candidate) for candidate in reserved_portfolio_candidates
        }
    engagement_quota_requested = min(
        engagement_quota_configured, engagement_capacity
    )
    engagement_candidates = [
        candidate
        for candidate in unique_candidates
        if id(candidate) not in reserved_portfolio_ids
        if int(
            _position_distribution_engagement(candidate).get(
                "soft_incremental_change_count", 0
            )
            or 0
        )
        > 0
    ]
    engagement_underfilled = (
        len(engagement_candidates) < engagement_quota_requested
    )
    reserved_engagement_candidates = list(
        engagement_candidates[:engagement_quota_requested]
    )


    k = (
        formal_selection_capacity
        - (portfolio_quota_requested - len(reserved_portfolio_candidates))
        - (engagement_quota_requested - len(reserved_engagement_candidates))
    )

    def engagement_identity(candidate: Mapping[str, Any]) -> Dict[str, str]:
        return {
            "variant_id": str(candidate.get("variant_id") or ""),
            "seq_hash": str(
                candidate.get("seq_hash")
                or _structure_sequence_identity(candidate, -1)[1]
            ),
        }

    def portfolio_identity(candidate: Mapping[str, Any]) -> Dict[str, str]:
        receipt = portfolio_receipts[id(candidate)]
        return {
            "variant_id": str(candidate.get("variant_id") or ""),
            "seq_hash": str(receipt["candidate_sequence_bundle_hash"]),
            "compiled_portfolio_request_hash": str(
                receipt["compiled_portfolio_request_hash"]
            ),
            "slot_id": str(receipt["slot_id"]),
            "slot_hash": str(receipt["slot_hash"]),
            "portfolio_id": str(receipt["portfolio_id"]),
            "role": str(receipt["role"]),
            "receipt_hash": str(receipt["receipt_hash"]),
        }

    if suppression_audit is not None and policy in {
        "formal_joint_novel",
        "formal_layered_novel",
    }:
        suppression_audit.update(
            {
                "position_distribution_engagement_quota_configured": (
                    engagement_quota_configured
                ),
                "position_distribution_engagement_quota_requested": (
                    engagement_quota_requested
                ),
                "position_distribution_engagement_available_count": len(
                    engagement_candidates
                ),
                "position_distribution_engagement_selected_count": len(
                    reserved_engagement_candidates
                ),
                "position_distribution_engagement_underfilled": (
                    engagement_underfilled
                ),
                "position_distribution_engagement_underfill_reasons": (
                    ["position_distribution_engagement_quota_underfill"]
                    if engagement_underfilled
                    else []
                ),
                "position_distribution_engagement_selected": [
                    engagement_identity(candidate)
                    for candidate in reserved_engagement_candidates
                ],
            }
        )
        if portfolio_quota_configured:
            suppression_audit.update(
                {
                    "portfolio_contract_quota_configured": (
                        portfolio_quota_configured
                    ),
                    "portfolio_contract_quota_requested": (
                        portfolio_quota_requested
                    ),
                    "portfolio_contract_available_count": len(
                        portfolio_candidates
                    ),
                    "portfolio_contract_selected_count": len(
                        reserved_portfolio_candidates
                    ),
                    "portfolio_contract_underfilled": portfolio_underfilled,
                    "portfolio_contract_underfill_reasons": (
                        ["portfolio_contract_quota_underfill"]
                        if portfolio_underfilled
                        else []
                    ),
                    "portfolio_contract_selected": [
                        portfolio_identity(candidate)
                        for candidate in reserved_portfolio_candidates
                    ],
                }
            )
    if k == 0:
        if policy == "formal_layered_novel" and suppression_audit is not None:
            anchor_set = set(anchors)
            optional_set = set(required) - anchor_set
            requested = int(
                suppression_audit.get("formal_quota_requested", 0) or 0
            )
            reasons = list(suppression_audit.get("underfill_reasons", []) or [])
            if anchor_set:
                reasons.append("anchor_union_not_represented")
            if optional_set:
                reasons.append("optional_node_not_represented")
            if engagement_underfilled:
                reasons.append(
                    "position_distribution_engagement_quota_underfill"
                )
            if portfolio_underfilled:
                reasons.append("portfolio_contract_quota_underfill")
            suppression_audit.update(
                {
                    "formal_selected_count": 0,
                    "formal_underfilled": True,
                    "formal_underfill_reasons": list(dict.fromkeys(reasons)),
                    "anchor_nodes": list(anchors),
                    "active_nodes": list(required),
                    "optional_nodes": sorted(optional_set),
                    "anchor_union_pass": not anchor_set,
                    "optional_represented": False,
                    "active_union_pass": not set(required),
                    "underfill": bool(requested > 0 or anchor_set or optional_set),
                    "underfill_reasons": list(dict.fromkeys(reasons)),
                }
            )
        elif policy == "formal_joint_novel" and suppression_audit is not None:
            reasons = list(
                suppression_audit.get("formal_underfill_reasons", []) or []
            )
            if engagement_underfilled:
                reasons.append(
                    "position_distribution_engagement_quota_underfill"
                )
            if portfolio_underfilled:
                reasons.append("portfolio_contract_quota_underfill")
            suppression_audit.update(
                {
                    "formal_selected_count": 0,
                    "formal_underfilled": True,
                    "formal_underfill_reasons": list(dict.fromkeys(reasons)),
                }
            )
        return []
    if policy == "formal_layered_novel":
        layered = _select_layered_formal_candidates(
            unique_candidates,
            quota=k,
            semantic_active_nodes=required,
            semantic_anchor_nodes=anchors,
            reserved_portfolio_candidates=reserved_portfolio_candidates,
            reserved_position_distribution_candidates=(
                reserved_engagement_candidates
            ),


            leader_count=(
                min(2, k)
                if bool(rescue_audit.get("all_infeasible_rescue_used"))
                else 1
            ),
        )
        selected_formal = [candidate for candidate, _lane in layered]
        selected_coverage = {
            node
            for candidate in selected_formal
            for node in _structure_candidate_nodes(candidate, required)
        }
        anchor_set = set(anchors)
        optional_set = set(required) - anchor_set
        pool_coverage = {
            node
            for candidate in unique_candidates
            for node in _structure_candidate_nodes(candidate, required)
        }
        anchor_union_pass = anchor_set.issubset(selected_coverage)
        optional_represented = bool(optional_set & selected_coverage)
        active_union_pass = set(required).issubset(selected_coverage)
        requested = int(
            (suppression_audit or {}).get("formal_quota_requested", k) or 0
        )
        underfill_reasons = list(
            (suppression_audit or {}).get("underfill_reasons", []) or []
        )
        if not anchor_union_pass:
            underfill_reasons.append(
                "anchor_union_selection_underfill"
                if anchor_set.issubset(pool_coverage)
                else "anchor_union_unavailable_in_eligible_pool"
            )
        if optional_set and not optional_represented:
            underfill_reasons.append(
                "optional_selection_underfill"
                if bool(optional_set & pool_coverage)
                else "optional_node_unavailable_in_eligible_pool"
            )
        if engagement_underfilled:
            underfill_reasons.append(
                "position_distribution_engagement_quota_underfill"
            )
        if portfolio_underfilled:
            underfill_reasons.append("portfolio_contract_quota_underfill")
        underfill_reasons = list(dict.fromkeys(underfill_reasons))
        underfill = bool(len(selected_formal) < requested or underfill_reasons)
        ranks = {
            id(candidate): rank
            for rank, candidate in enumerate(unique_candidates, start=1)
        }
        basis = (
            "fast_leader_anchor_union_optional_signature_maxmin_diversity"
            if ranking_source == "fast_loss"
            else "screen_energy_leader_anchor_union_optional_signature_maxmin_diversity"
        )
        if engagement_quota_requested:
            basis = "position_distribution_engagement_lane_plus_" + basis
        if portfolio_quota_requested:
            basis = "portfolio_contract_lane_plus_" + basis
        for candidate, lane in layered:
            candidate["structure_shortlist_role"] = "formal_layered_novel"
            candidate["formal_rerank_eligible"] = True
            candidate["structure_shortlist_basis"] = basis
            candidate["structure_shortlist_proxy_rank"] = ranks[id(candidate)]
            candidate["lane"] = lane
            candidate["structure_shortlist_lane"] = lane
            candidate["coverage_scope"] = "shortlist_set"
            candidate["structure_shortlist_coverage_scope"] = "shortlist_set"
            candidate["anchor_union_pass"] = anchor_union_pass
            candidate["optional_represented"] = optional_represented
            candidate["active_union_pass"] = active_union_pass
            candidate["underfill"] = underfill
            candidate["structure_shortlist_quota_requested"] = requested
        if suppression_audit is not None:
            suppression_audit.update(
                {
                    "formal_selected_count": len(selected_formal),
                    "formal_underfilled": underfill,
                    "formal_underfill_reasons": underfill_reasons,
                    "anchor_nodes": list(anchors),
                    "active_nodes": list(required),
                    "optional_nodes": sorted(optional_set),
                    "selected_anchor_nodes": sorted(anchor_set & selected_coverage),
                    "represented_optional_nodes": sorted(optional_set & selected_coverage),
                    "anchor_union_pass": anchor_union_pass,
                    "optional_represented": optional_represented,
                    "active_union_pass": active_union_pass,
                    "underfill": underfill,
                    "underfill_reasons": underfill_reasons,
                    "lane": [lane for _candidate, lane in layered],
                    "feasibility_rescue_leader_count": sum(
                        lane == "feasibility_rescue_leader"
                        for _candidate, lane in layered
                    ),
                }
            )
        return selected_formal
    if all_candidates and not portfolio_quota_configured:
        selected_all = unique_candidates
        if policy == "formal_joint_novel":
            reserved_ids = {
                id(candidate) for candidate in reserved_engagement_candidates
            }
            for candidate in selected_all:
                candidate["structure_shortlist_role"] = "formal_joint_novel"
                candidate["formal_rerank_eligible"] = True
                if id(candidate) in reserved_ids:
                    candidate["lane"] = "position_distribution_engagement"
                    candidate["structure_shortlist_lane"] = (
                        "position_distribution_engagement"
                    )
        return selected_all

    if policy == "formal_joint_novel":
        selected_formal = list(
            [
                *reserved_portfolio_candidates,
                *reserved_engagement_candidates,
            ][:k]
        )
        selected_ids = {id(candidate) for candidate in selected_formal}
        remaining_pool = [
            candidate
            for candidate in unique_candidates
            if id(candidate) not in selected_ids
        ]
        remaining_quota = max(0, k - len(selected_formal))
        if ranking_source == "fast_loss" and remaining_quota > 1:
            selected_formal.extend(
                _select_fast_proxy_diverse_formal_candidates(
                    remaining_pool,
                    quota=remaining_quota,
                )
            )
            shortlist_basis = (
                "portfolio_contract_lane_plus_"
                if reserved_portfolio_candidates
                else ""
            ) + (
                "position_distribution_engagement_lane_plus_"
                if reserved_engagement_candidates
                else ""
            ) + "fast_proxy_leader_plus_sequence_diversity"
        else:
            selected_formal.extend(remaining_pool[:remaining_quota])
            shortlist_basis = (
                "portfolio_contract_lane_plus_"
                if reserved_portfolio_candidates
                else ""
            ) + (
                "position_distribution_engagement_lane_plus_"
                if reserved_engagement_candidates
                else ""
            ) + ranking_source
        fast_ranks = {
            id(candidate): rank
            for rank, candidate in enumerate(unique_candidates, start=1)
        }
        reserved_engagement_ids = {
            id(candidate) for candidate in reserved_engagement_candidates
        }
        reserved_contract_ids = {
            id(candidate) for candidate in reserved_portfolio_candidates
        }
        for candidate in selected_formal:
            candidate["structure_shortlist_role"] = "formal_joint_novel"
            candidate["formal_rerank_eligible"] = True
            candidate["structure_shortlist_basis"] = shortlist_basis
            candidate["structure_shortlist_proxy_rank"] = fast_ranks[id(candidate)]
            if id(candidate) in reserved_contract_ids:
                candidate["lane"] = "portfolio_contract"
                candidate["structure_shortlist_lane"] = "portfolio_contract"
            elif id(candidate) in reserved_engagement_ids:
                candidate["lane"] = "position_distribution_engagement"
                candidate["structure_shortlist_lane"] = (
                    "position_distribution_engagement"
                )
        if suppression_audit is not None:
            suppression_audit["formal_selected_count"] = len(selected_formal)
        return selected_formal

    required_set = set(required)
    candidate_nodes = {
        id(candidate): _structure_candidate_nodes(candidate, required)
        for candidate in unique_candidates
    }
    candidate_depths = {
        id(candidate): _structure_candidate_depth(candidate)
        for candidate in unique_candidates
    }

    selected: List[Dict[str, Any]] = []
    selected_ids: set[int] = set()


    if len(required_set) > 1:
        joint = next(
            (
                candidate
                for candidate in unique_candidates
                if required_set.issubset(set(candidate_nodes[id(candidate)]))
            ),
            None,
        )
        if joint is not None:
            selected.append(joint)
            selected_ids.add(id(joint))


    if len(selected) < k:
        best_fast = unique_candidates[0]
        if id(best_fast) not in selected_ids:
            selected.append(best_fast)
            selected_ids.add(id(best_fast))

    while len(selected) < k:
        used_nodes = {
            node
            for candidate in selected
            for node in candidate_nodes[id(candidate)]
        }
        used_signatures = {
            candidate_nodes[id(candidate)]
            for candidate in selected
        }
        used_depths = {
            candidate_depths[id(candidate)]
            for candidate in selected
            if candidate_depths[id(candidate)] is not None
        }
        remaining = [
            candidate
            for candidate in unique_candidates
            if id(candidate) not in selected_ids
        ]
        if not remaining:
            break
        rank = {id(candidate): index for index, candidate in enumerate(unique_candidates)}
        next_candidate = max(
            remaining,
            key=lambda candidate: (
                len(set(candidate_nodes[id(candidate)]) - used_nodes),
                int(candidate_nodes[id(candidate)] not in used_signatures),
                int(
                    candidate_depths[id(candidate)] is not None
                    and candidate_depths[id(candidate)] not in used_depths
                ),
                -rank[id(candidate)],
            ),
        )
        selected.append(next_candidate)
        selected_ids.add(id(next_candidate))

    return selected


def _has_structure_signal(candidate: Mapping[str, Any]) -> bool:


    metrics = candidate.get("structure_metrics") or {}
    if float(candidate.get("plddt", 0.0) or 0.0) > 0.0:
        return True
    if bool(metrics.get("structure_path") or metrics.get("cif_path")):
        return True
    if bool(candidate.get("structure_path") or candidate.get("cif_path")):
        return True
    return any(
        bool(
            ((state.get("structure_metrics") or {}).get("structure_path"))
            or ((state.get("structure_metrics") or {}).get("cif_path"))
            or state.get("structure_path")
            or state.get("cif_path")
        )
        for state in (metrics.get("states") or [])
        if isinstance(state, dict)
    )


def _get_chains_from_terms(terms_chai: List[tuple[float, Any]]) -> Optional[List[str]]:
    for _, t in terms_chai:
        if isinstance(t, ChaiPlddtDeltaTerm):
            return list(t.chains_B)
    return None


def _get_structure_metric(terms_chai: List[tuple[float, Any]]) -> str:
    for _, t in terms_chai:
        if isinstance(t, ChaiPlddtDeltaTerm):
            return getattr(t, "metric", "plddt")
    return "plddt"


def _structure_model_name_for_provider(cfg: SAConfig, provider: str, stage: str) -> Optional[str]:
    provider = _normalise_structure_provider(provider)
    if provider == "service":
        return _structure_model_name_for_provider(cfg, _service_backend(cfg), stage)
    if provider == "esmfold":
        return cfg.inner_structure_model_name or "facebook/esmfold_v1"
    if provider == "esmfold2":
        if stage.startswith("inner_loop_esmfold2"):
            return (
                getattr(cfg, "inner_esmfold2_model_name", None)
                or cfg.structure_prescreen_model_name
                or cfg.structure_model_name
            )
        if stage == "prescreen":
            return (
                cfg.structure_prescreen_model_name
                or cfg.structure_model_name
            )
        return cfg.structure_screen_model_name or cfg.structure_model_name
    if provider == "protenix":
        if stage == "rerank":
            return cfg.structure_rerank_model_name or cfg.structure_model_name or cfg.protenix_model_name
        return cfg.structure_model_name or cfg.protenix_model_name
    if provider == "alphafold3":
        if stage == "rerank":
            return cfg.af3_model_dir or cfg.structure_rerank_model_name or cfg.structure_model_name
        return cfg.af3_model_dir or cfg.structure_model_name
    return cfg.structure_model_name


def _score_config_for_structure_stage(
    score_config: Optional[Dict[str, Any]], stage: str
) -> Optional[Dict[str, Any]]:


    if not score_config:
        return score_config
    resolved = deepcopy(score_config)
    backends = resolved.get("evaluator_backends")
    if not isinstance(backends, Mapping):
        return resolved
    stage_name = str(stage).strip().lower()
    for name, raw in list(backends.items()):
        if not isinstance(raw, Mapping):
            continue
        required_stage = str(raw.get("defer_until_stage") or "").strip().lower()
        if required_stage and required_stage != stage_name:
            item = dict(raw)
            item["enabled"] = False
            item["required"] = False
            backends[name] = item
            weights = resolved.get("evaluator_weights")
            if isinstance(weights, dict):
                weights[f"eval_{name}"] = 0.0
    return resolved


def rescore_existing_structure_candidate(
    candidate: Mapping[str, Any],
    *,
    compiled: Dict[str, Any],
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    score_config: Dict[str, Any],
    design_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:


    out = dict(candidate)
    report = evaluate_candidate(
        out,
        compiled=compiled,
        design_state=design_state or {},
        masks=masks,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
        score_config=score_config,
    )
    evaluator_loss = float(report.get("total_energy", report.get("loss", 1.0)) or 0.0)
    evaluator_weight = float(score_config.get("inner_evaluator_loss_weight", score_config.get("weight_evaluator", 1.0)) or 0.0)
    evaluator_penalty = evaluator_weight * evaluator_loss
    hard_gate_pass = bool(report.get("hard_gate_pass", True))
    hard_gate_penalty = 0.0 if hard_gate_pass else float(score_config.get("inner_hard_gate_fail_penalty", 1000.0) or 1000.0)
    fast_loss = float(out.get("fast_loss", 0.0) or 0.0)
    struct_penalty = float(out.get("struct_penalty", 0.0) or 0.0)
    multistate_loss = float(out.get("multistate_loss", 0.0) or 0.0)
    legacy_design_energy = fast_loss + struct_penalty + evaluator_penalty
    objective_input = dict(out)
    objective_input["evaluator_report"] = report
    objective_input["evaluator_score"] = float(report.get("normalized_score", 0.0) or 0.0)
    objective_input["evaluator_soft_score"] = float(report.get("soft_score", objective_input["evaluator_score"]) or 0.0)
    objective = compute_outer_energy_objective(fast_loss, objective_input, score_config)
    aligned_energy = float(objective["final_energy"] )
    out.update({
        "inner_evaluator_report": report,
        "inner_evaluator_score": objective_input["evaluator_score"],
        "inner_evaluator_loss": evaluator_loss,
        "inner_evaluator_penalty": evaluator_penalty,
        "inner_evaluator_hard_gate_penalty_legacy": hard_gate_penalty,
        "combined_energy": aligned_energy,
        "outer_aligned_energy": aligned_energy,
        "outer_energy_objective": objective,
        "legacy_design_energy": legacy_design_energy,
        "structure_selection_objective": "outer_aligned",
        "combined_loss": legacy_design_energy + hard_gate_penalty,
        "legacy_hard_gate_penalty": hard_gate_penalty,
        "physics_evaluated": True,
        "physics_stage": "post_af3_topk",
    })
    out["energy"] = combined_energy_record(
        fast_energy=fast_loss,
        structure_energy=struct_penalty,
        multistate_energy=multistate_loss,
        evaluator_energy=evaluator_penalty,
        total_energy=aligned_energy,
        hard_gate_pass=hard_gate_pass,
    )
    out["energy"]["selection_objective"] = "outer_aligned"
    out["energy"]["outer_aligned_energy"] = aligned_energy
    out["energy"]["legacy_design_energy"] = legacy_design_energy
    return out


def _evaluate_structure_candidate(
    c: Dict[str, Any],
    compiled: Dict[str, Any],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    terms_chai: List[tuple[float, Any]],
    score_config: Optional[Dict[str, Any]],
    design_state: Optional[Dict[str, Any]],
    provider: str,
    stage: str,
    structure_evidence_run_memory: Any = None,
) -> Dict[str, Any]:


    provider = _normalise_structure_provider(provider or cfg.structure_model)
    model_name = _structure_model_name_for_provider(cfg, provider, stage)
    seq_hash = str(c.get("seq_hash") or _seqs_hash(c["seqs"]))
    variant_id = str(c.get("variant_id") or seq_hash)
    occurrence_token = _structure_occurrence_token(provider)
    occurrence_component = (
        f"__occ_{occurrence_token}" if occurrence_token else ""
    )
    pred_prefix = (
        f"{stage}{occurrence_component}__{variant_id}__{seq_hash}"
    )

    chains: List[Tuple[str, str]] = []
    confidence: Dict[str, Any] = {"metrics": {}, "chain_metrics": {}, "residue_plddt": {}}
    node_plddt: Dict[str, Dict[str, Any]] = {}
    chain_plddt: Dict[str, float] = {}
    structure_metrics: Dict[str, Any] = {}
    structure_out_dir: Optional[str] = None
    structure_summary_json: Optional[str] = None
    multistate_objectives: Dict[str, Any] = {}
    multistate_score = 0.0
    multistate_loss = 0.0
    plddt = 0.0
    c_complex_confidence_metrics: Dict[str, Dict[str, Any]] = {}
    evidence_cache_summary: Optional[Dict[str, Any]] = None

    try:
        complex_plddt, complex_summary, complex_confidence, complex_out_dir, complex_summary_json = _evaluate_complex_states(
            c["seqs"],
            compiled,
            cfg,
            provider=provider,
            model_name=model_name,
            pred_prefix=pred_prefix,
            stage=stage,
            structure_evidence_run_memory=structure_evidence_run_memory,
        )
        if complex_summary:
            plddt = float(complex_plddt or 0.0)
            confidence = {
                "metrics": dict(complex_summary.get("scalar", {}) or {}),
                "chain_metrics": {},
                "residue_plddt": {},
            }
            node_plddt = {}
            chain_plddt = {}
            structure_metrics = {
                key: value
                for key, value in complex_summary.items()
                if key != "_structure_evidence_cache"
            }
            structure_out_dir = complex_out_dir
            structure_summary_json = complex_summary_json
            c_complex_confidence_metrics = complex_confidence
            if isinstance(
                complex_summary.get("_structure_evidence_cache"), Mapping
            ):
                evidence_cache_summary = dict(
                    complex_summary["_structure_evidence_cache"]
                )
            multistate_objectives = dict(complex_summary.get("multistate_objectives", {}) or {})
            if multistate_objectives.get("enabled"):
                multistate_score = float(multistate_objectives.get("normalized_score") or 0.0)
                multistate_loss = float(cfg.multistate_objective_weight) * float(multistate_objectives.get("loss") or 0.0)
        else:
            chains_B = _get_chains_from_terms(terms_chai)
            if chains_B:
                chains = [(cid, c["seqs"][cid]) for cid in chains_B]
            else:
                chains = [(cid, c["seqs"][cid]) for cid in compiled["chain_order"]]

            metric = _get_structure_metric(terms_chai)
            pred_name = f"{pred_prefix}__" + ("__".join(cid for cid, _ in chains) or "pred")
            effective_provider = (
                _service_backend(cfg) if provider == "service" else provider
            )
            structure_kwargs: Dict[str, Any] = {
                "metric": metric,
                "seed": (
                    cfg.af3_seed
                    if effective_provider == "alphafold3"
                    else cfg.protenix_seed
                ),
                "model_name": model_name,
            }
            if effective_provider == "esmfold2":
                structure_kwargs.update(
                    {
                        "mode": cfg.esmfold2_mode,
                        "num_loops": cfg.esmfold2_num_loops,
                        "num_sampling_steps": cfg.esmfold2_num_sampling_steps,
                        "num_diffusion_samples": cfg.esmfold2_num_diffusion_samples,
                    }
                )
                if provider != "service":
                    structure_kwargs["conda_env"] = cfg.esmfold2_conda_env
            elif effective_provider == "protenix":
                structure_kwargs.update(
                    {
                        "conda_env": cfg.protenix_conda_env,
                        "timeout": cfg.protenix_complex_timeout,
                        "use_msa": cfg.protenix_complex_use_msa,
                        "cycle": cfg.protenix_complex_cycle,
                        "step": cfg.protenix_complex_step,
                        "sample": cfg.protenix_complex_sample,
                        "use_default_params": cfg.protenix_complex_use_default_params,
                    }
                )
            elif effective_provider == "alphafold3":
                structure_kwargs.update(
                    {
                        "model_dir": cfg.af3_model_dir,
                        "conda_env": cfg.af3_conda_env,
                        "timeout": cfg.af3_timeout,
                        "run_data_pipeline": cfg.af3_run_data_pipeline,
                        "db_dir": cfg.af3_db_dir,
                        "num_recycles": cfg.af3_num_recycles,
                        "num_diffusion_samples": cfg.af3_num_diffusion_samples,
                        "flash_attention_implementation": cfg.af3_flash_attention_implementation,
                        "gpu_device": cfg.af3_gpu_device,
                    }
                )
            if provider == "service":


                structure_kwargs.pop("conda_env", None)
                structure_kwargs.update(_service_transport_kwargs(cfg))

            confidence, evidence_cache_artifact = _cached_structure_evidence_call(
                c["seqs"],
                run_memory=structure_evidence_run_memory,
                provider=provider,
                model_name=model_name,
                operation="confidence_multichain",
                request_state={"chains": [[cid, seq] for cid, seq in chains]},
                call_kwargs=structure_kwargs,
                compute=lambda: run_structure_confidence_multichain(
                    pred_name=pred_name,
                    chains=chains,
                    provider=provider,
                    **structure_kwargs,
                ),
            )
            confidence = _rebind_confidence_pred_name(confidence, pred_name)
            evidence_cache_summary = _structure_evidence_cache_summary(
                [evidence_cache_artifact]
            )
            plddt = float(confidence.get("metrics", {}).get(metric, 0.0))
            chain_plddt = dict(confidence.get("chain_metrics", {}).get("plddt", {}) or {})
            node_plddt = compute_node_plddt(
                compiled,
                confidence.get("residue_plddt", {}) or {},
            )
            structure_metrics = summarize_structure_metrics(confidence, node_plddt=node_plddt)
            structure_out_dir = confidence.get("out_dir")
            structure_summary_json = confidence.get("summary_json")
    except Exception as exc:
        plddt = 0.0
        structure_metrics = summarize_structure_metrics(confidence, node_plddt=node_plddt)
        structure_metrics.setdefault("warnings", [])
        if isinstance(structure_metrics.get("warnings"), list):
            structure_metrics["warnings"].append(f"{stage}:{provider} failed: {type(exc).__name__}: {exc}")

    structure_constraint_energy = 0.0
    if terms_chai:
        compiled["_plddt"] = float(plddt)
        if chains:
            _install_confidence_cache(compiled, c["seqs"], chains, confidence)
        else:
            compiled["_struct_cache"] = {}
        structure_constraint_energy = energy_breakdown(
            c["seqs"], compiled, terms_chai
        )["total"]
        compiled["_plddt"] = None

    struct_pen = float(structure_constraint_energy) + float(multistate_loss)

    plddt_delta, plddt_A, plddt_B = _extract_plddt_delta(
        c["seqs"], compiled, terms_chai
    )
    compiled["_struct_cache"] = {}

    c2 = dict(c)
    c2["structure_stage"] = stage
    c2["structure_provider"] = provider
    c2["structure_model_name"] = model_name
    c2["plddt"] = float(plddt)
    c2["confidence_metrics"] = dict(confidence.get("metrics", {}) or {})
    c2["chain_pair_metrics"] = dict(
        confidence.get("chain_pair_metrics", {}) or {}
    )
    if c_complex_confidence_metrics:
        c2["complex_state_confidence_metrics"] = c_complex_confidence_metrics
    c2["chain_plddt"] = chain_plddt
    c2["node_plddt"] = node_plddt
    c2["residue_plddt"] = dict(
        confidence.get("residue_plddt", {}) or {}
    )
    c2["structure_metrics"] = structure_metrics
    c2["structure_out_dir"] = structure_out_dir
    c2["structure_summary_json"] = structure_summary_json
    structure_path = structure_metrics.get("structure_path") or structure_metrics.get("cif_path")
    if structure_path:
        c2["structure_path"] = structure_path
        if str(structure_path).lower().endswith(".cif"):
            c2["cif_path"] = structure_path

    c2["protenix_out_dir"] = structure_out_dir
    c2["protenix_summary_json"] = structure_summary_json
    c2["multistate_objectives"] = multistate_objectives
    c2["multistate_score"] = float(multistate_score)
    c2["multistate_loss"] = float(multistate_loss)
    if evidence_cache_summary is not None:
        c2["persistent_evaluation_cache"] = evidence_cache_summary
    gate_sources: Dict[str, Any] = {
        str(source): dict(payload) if isinstance(payload, Mapping) else payload
        for source, payload in (
            (c.get("feasibility_gate_sources") or {}).items()
            if isinstance(c.get("feasibility_gate_sources"), Mapping)
            else []
        )
    }
    fast_filter = c.get("fast_filter")
    if isinstance(fast_filter, Mapping):
        gate_sources.setdefault("fast_filter", dict(fast_filter))
    evaluator_penalty = 0.0
    legacy_hard_gate_penalty = 0.0
    evaluator_hard_gate_pass = True
    evaluator_report: Dict[str, Any] = {}
    active_score_config = _score_config_for_structure_stage(score_config, stage)
    if active_score_config:
        evaluator_report = evaluate_candidate(
            c2,
            compiled=compiled,
            design_state=design_state or {},
            masks=masks,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            score_config=active_score_config,
        )
        evaluator_loss = float(
            evaluator_report.get(
                "total_energy", evaluator_report.get("loss", 1.0)
            )
            or 0.0
        )
        evaluator_weight = float(active_score_config.get("inner_evaluator_loss_weight", active_score_config.get("weight_evaluator", 1.0)) or 0.0)
        evaluator_penalty = evaluator_weight * evaluator_loss
        evaluator_hard_gate_pass = bool(
            evaluator_report.get("hard_gate_pass", True)
        )
        if not evaluator_hard_gate_pass:
            legacy_hard_gate_penalty = float(
                active_score_config.get("inner_hard_gate_fail_penalty", 1000.0)
                or 1000.0
            )
        c2["inner_evaluator_report"] = evaluator_report
        c2["inner_evaluator_score"] = float(evaluator_report.get("normalized_score", 0.0) or 0.0)
        c2["inner_evaluator_loss"] = evaluator_loss
        c2["inner_evaluator_penalty"] = float(evaluator_penalty)
        c2["inner_evaluator_hard_gate_penalty_legacy"] = float(
            legacy_hard_gate_penalty
        )
        evaluator_gate_source: Dict[str, Any] = {}
        if "hard_gate_pass" in evaluator_report:
            evaluator_gate_source["hard_gate_pass"] = bool(
                evaluator_report.get("hard_gate_pass")
            )
        if "disqualification_reasons" in evaluator_report:
            evaluator_gate_source["disqualification_reasons"] = list(
                evaluator_report.get("disqualification_reasons", []) or []
            )
        if isinstance(evaluator_report.get("gate_status"), Mapping):
            evaluator_gate_source["gate_status"] = dict(
                evaluator_report.get("gate_status") or {}
            )
        gate_sources["inner_evaluator"] = evaluator_gate_source
    c2["feasibility_gate_sources"] = gate_sources
    c2["struct_penalty"] = float(struct_pen)
    legacy_design_energy = float(
        c["fast_loss"] + struct_pen + evaluator_penalty
    )
    objective_out = dict(c2)
    objective_out["evaluator_report"] = evaluator_report
    objective_out["evaluator_score"] = float(
        evaluator_report.get("normalized_score", 0.0) or 0.0
    )
    objective_out["evaluator_soft_score"] = float(
        evaluator_report.get(
            "soft_score",
            objective_out["evaluator_score"],
        )
        or 0.0
    )
    outer_objective = compute_outer_energy_objective(
        c["fast_loss"],
        objective_out,
        active_score_config,
    )
    aligned_energy = float(outer_objective["final_energy"])
    use_aligned_objective = (
        str(cfg.structure_selection_objective).strip().lower()
        == "outer_aligned"
    )
    combined_energy = (
        aligned_energy if use_aligned_objective else legacy_design_energy
    )
    c2["combined_energy"] = float(combined_energy)
    c2["outer_aligned_energy"] = aligned_energy
    c2["outer_energy_objective"] = outer_objective
    c2["legacy_design_energy"] = legacy_design_energy
    c2["structure_selection_objective"] = (
        "outer_aligned" if use_aligned_objective else "legacy_additive"
    )


    c2["combined_loss"] = float(
        legacy_design_energy + legacy_hard_gate_penalty
    )
    c2["legacy_hard_gate_penalty"] = float(legacy_hard_gate_penalty)
    c2["energy"] = combined_energy_record(
        fast_energy=c["fast_loss"],
        structure_energy=structure_constraint_energy,
        multistate_energy=multistate_loss,
        evaluator_energy=evaluator_penalty,
        total_energy=combined_energy,
        hard_gate_pass=bool(
            (fast_filter or {}).get("pass", True)
            if isinstance(fast_filter, Mapping)
            else True
        )
        and evaluator_hard_gate_pass,
    )
    c2["energy"]["selection_objective"] = c2[
        "structure_selection_objective"
    ]
    c2["energy"]["outer_aligned_energy"] = aligned_energy
    c2["energy"]["legacy_design_energy"] = legacy_design_energy
    c2["energy"]["outer_energy_components"] = list(
        outer_objective.get("energy_components", [])
    )
    c2["plddt_delta"] = plddt_delta
    c2["plddt_A"] = plddt_A
    c2["plddt_B"] = plddt_B
    return c2

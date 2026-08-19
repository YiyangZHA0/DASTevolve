

from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Mapping, Optional, List, Sequence, Tuple, Callable
import importlib
import os
import math
import numpy as np

from astevolve.core.constraints import build_terms_from_specs
from astevolve.search.causal_binding import (
    bind_candidate as _bind_causal_candidate,
    build_runtime as _build_causal_runtime,
)
from astevolve.search.candidate_validation import (
    _inner_loop_semantic_audit,
    _required_final_mutation_coverage,
    _semantic_prefilter_structure_candidates,
)
from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.config import SAConfig
from astevolve.search.candidate_wave_runtime import (
    CandidateWaveError,
    build_node_engagement_ledger,
    build_rerank_selection_receipt,
    freeze_candidate_wave,
    validate_candidate_wave_request,
)
from astevolve.search.energy_reporting import (
    SEARCH_ENERGY_SCHEMA_VERSION,
    best_so_far_trace,
    fast_energy_record,
)
from astevolve.search.joint_coverage_completion import (
    attach_joint_coverage_reporting,
    finalize_joint_coverage_completion,
    initialize_joint_coverage_completion,
    joint_coverage_expansion_directive,
    update_joint_coverage_completion,
)
from astevolve.search.finalist_feedback import build_structure_finalist_feedback
from astevolve.search.mcts_candidate_expansion import (
    MCTSExpansionState,
    PrebuiltProposal,
    expand_mcts_candidate_siblings,
    materialize_prebuilt_mcts_root_children,
)
from astevolve.search.mcts_fidelity_upgrade import (
    FIDELITY_UPGRADE_VERSION,
    apply_reward_delta as _apply_fidelity_reward_delta,
    refresh_best_reward as _refresh_fidelity_best_reward,
    select_fidelity_upgrade_cohort as _select_fidelity_upgrade_cohort,
    select_final_fidelity_cohort as _select_final_fidelity_cohort,
)
from astevolve.search.run_memory import InnerRunMemory
from astevolve.search.inner_selection import _merge_structure_selection_pool, _select_final_structure_candidate, _write_inner_loop_artifacts
from astevolve.search.proposal_engine import (
    _designable_segments,
    _mutate_node_seqs,
    _node_policy,
    _segment_prior,
    _select_semantic_segment,
    _semantic_coverage_hard_enabled,
    _semantic_coverage_report,
    _semantic_force_steps,
    _semantic_required_nodes,
    _semantic_required_unavailable_nodes,
    _unique_strings,
)
from astevolve.search.reporting import (
    _attach_proposal_tier_accounting, _mcts_best_path, _mcts_select_leaf,
    _mcts_tree_quality_report, _proposal_tier_history_fields,
    _summarize_mcts_round,
    attach_structure_shortlist_health as _attach_structure_shortlist_health,
)
from astevolve.search.scoring import _score_fast_candidate, compute_segment_scores
from astevolve.search.sequence_ops import init_seqs
from astevolve.search.parent_baseline import (
    build_parent_baseline_candidate as _build_parent_baseline_candidate,
    build_parent_baseline_comparison as _build_parent_baseline_comparison,
    evaluate_structure_candidate_dispatched as _record_structure_dispatch,
    include_parent_baseline as _include_parent_baseline,
    is_parent_baseline as _is_parent_baseline,
    public_chai_results as _public_chai_results,
)
from astevolve.search.structure_pipeline import (
    _enforce_layered_shortlist_semantic_gate,
    _evaluate_structure_candidate,
    _has_structure_signal,
    _select_single_node_structure_diagnostics,
    _select_structure_candidates,
    _summarize_structure_shortlist,
    rescore_existing_structure_candidate,
)
from astevolve.search.structure_batch import evaluate_structure_candidates
from astevolve.search.structure_failure_memory import attach_structure_failure_suppression_summary, build_structure_failure_context
from astevolve.search.structure_reporting import attach_structure_evaluation_summary
from astevolve.search.structure_provider_evidence import (
    build_structure_provider_evidence,
)
from astevolve.search.structure_multiseed import (
    build_backend_disagreement,
    evaluate_multiseed_stage,
)
from astevolve.search.result_adapter import optimize_multichain_result
from astevolve.search.mapping_schedule_runtime import (
    active_mapping_actions as _active_mapping_actions,
    bind_portfolio_mapping_components as _bind_portfolio_mapping_components,
    require_mapping_search_compatibility as _require_mapping_search_compatibility,
)
from astevolve.search.runtime_evaluation import (
    evaluate_structure_with_run_memory as _cached_structure_evaluation,
    score_fast_with_run_memory as _cached_fast_evaluation,
)
from astevolve.search.sa_search import run_sa_search as _run_sa_search
from engine.history_runtime import register_sequence_occurrence as _register_history_sequence


__all__ = ["optimize_multichain", "optimize_multichain_result"]

def _cheap_developability_audit(
    seqs: Mapping[str, str], template_seqs: Optional[Mapping[str, str]]
) -> Dict[str, Any]:
    hydrophobic = set("AILMFWVY")
    positive = set("KR")
    negative = set("DE")
    max_hydrophobic_run = 0
    max_homopolymer_run = 0
    net_charge = 0
    new_cysteines = 0
    for chain_id, sequence in seqs.items():
        template = str((template_seqs or {}).get(chain_id) or "")
        hydro_run = 0
        homo_run = 0
        previous = ""
        for index, residue in enumerate(str(sequence)):
            hydro_run = hydro_run + 1 if residue in hydrophobic else 0
            max_hydrophobic_run = max(max_hydrophobic_run, hydro_run)
            homo_run = homo_run + 1 if residue == previous else 1
            previous = residue
            max_homopolymer_run = max(max_homopolymer_run, homo_run)
            net_charge += int(residue in positive) - int(residue in negative)
            if residue == "C" and (index >= len(template) or template[index] != "C"):
                new_cysteines += 1
    passed = bool(
        max_hydrophobic_run <= 4
        and max_homopolymer_run <= 4
        and abs(net_charge) <= 18
        and new_cysteines == 0
    )
    return {
        "pass": passed,
        "max_hydrophobic_run": max_hydrophobic_run,
        "max_homopolymer_run": max_homopolymer_run,
        "net_charge": net_charge,
        "new_cysteines": new_cysteines,
    }


def _candidate_audit_record(
    candidate: Mapping[str, Any],
    *,
    template_seqs: Optional[Mapping[str, str]],
    causal_context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    seqs = dict(candidate.get("seqs") or {})
    changes = []
    for chain_id, sequence in seqs.items():
        template = str((template_seqs or {}).get(chain_id) or "")
        for index, residue in enumerate(str(sequence)):
            if index < len(template) and template[index] != residue:
                changes.append({
                    "chain_id": chain_id, "position": index,
                    "from": template[index], "to": residue,
                })
    move = candidate.get("move") if isinstance(candidate.get("move"), Mapping) else {}
    context = dict(causal_context or {})
    record = {
        "variant_id": candidate.get("variant_id"),
        "parent_id": candidate.get("parent_id"),
        "seq_hash": candidate.get("seq_hash") or _seqs_hash(seqs),
        "seqs": seqs,
        "mutation_count": len(changes),
        "mutations_from_template": changes,
        "node": move.get("node"),
        "node_resize": move.get("node_resize") or move.get("resize"),
        "developability": _cheap_developability_audit(seqs, template_seqs),
        "fast_filter": dict(candidate.get("fast_filter") or {}),
        "island_id": context.get("island_id"),
        "island_role": context.get("island_role") or context.get("role"),
    }
    if context.get("design_action_hash") and context.get(
        "compiled_design_action_hash"
    ):
        record["design_action_hash"] = context["design_action_hash"]
        record["compiled_design_action_hash"] = context[
            "compiled_design_action_hash"
        ]
        record["design_action_parent_binding"] = {
            field: context.get(field)
            for field in (
                "case_id",
                "parent_program_id",
                "parent_candidate_id",
                "parent_sequence_bundle_hash",
                "parent_effective_contract_hash",
                "parent_evolve_hash",
            )
        }
    for field in (
        "compiled_portfolio_request_hash",
        "portfolio_realization_receipts",
        "portfolio_pair_receipts",
        "portfolio_realization_summary",
        "candidate_wave_request_hash",
        "frozen_candidate_wave_hash",
        "candidate_wave_slot_id",
        "candidate_wave_role",
        "candidate_wave_slot_directive",
        "candidate_wave_slot_realization",
    ):
        if field in candidate:
            record[field] = candidate[field]
    return record


def _candidate_wave_protocol_preflight(cfg: SAConfig) -> None:


    if not bool(getattr(cfg, "candidate_wave_enabled", False)):
        return
    failures: List[str] = []
    if str(cfg.search_method).strip().lower() != "mcts":
        failures.append("search_method")
    if not bool(cfg.chai1_enabled):
        failures.append("chai1_enabled")
    if not bool(cfg.structure_screen_enabled):
        failures.append("structure_screen_enabled")
    if str(cfg.structure_screen_model).strip().lower() != "protenix":
        failures.append("structure_screen_model")
    if not bool(cfg.structure_rerank_enabled):
        failures.append("structure_rerank_enabled")
    if str(cfg.structure_rerank_model).strip().lower() not in {
        "alphafold3",
        "af3",
    }:
        failures.append("structure_rerank_model")
    if bool(cfg.structure_allow_low_fidelity_fallback):
        failures.append("structure_allow_low_fidelity_fallback")
    if int(cfg.portfolio_seed_refinement_rounds) != 0:
        failures.append("portfolio_seed_refinement_rounds")
    if failures:
        raise CandidateWaveError(
            "candidate_wave_provider_protocol_invalid", ",".join(failures)
        )


def _attach_frozen_candidate_wave(
    candidates: Sequence[Dict[str, Any]],
    manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:


    by_ref: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        reference = str(
            candidate.get("candidate_id")
            or candidate.get("node_id")
            or candidate.get("id")
            or candidate.get("variant_id")
            or ""
        )
        if reference:
            if reference in by_ref:
                raise CandidateWaveError(
                    "candidate_wave_candidate_ref_duplicate", reference
                )
            by_ref[reference] = candidate
    frozen: List[Dict[str, Any]] = []
    for member in manifest.get("members", []) or []:
        reference = str(member.get("candidate_ref") or "")
        candidate = by_ref.get(reference)
        if candidate is None:
            raise CandidateWaveError(
                "candidate_wave_frozen_candidate_missing", reference
            )
        if dict(candidate.get("seqs") or {}) != dict(member.get("seqs") or {}):
            raise CandidateWaveError(
                "candidate_wave_frozen_candidate_sequence_mismatch", reference
            )
        trusted = {
            "candidate_wave_request_hash": manifest[
                "candidate_wave_request_hash"
            ],
            "frozen_candidate_wave_hash": manifest[
                "frozen_candidate_wave_hash"
            ],
            "candidate_sequence_bundle_hash": member[
                "candidate_sequence_bundle_hash"
            ],
            "candidate_wave_slot_id": member["slot_id"],
            "candidate_wave_slot_hash": member["slot_hash"],
            "candidate_wave_portfolio_id": member["portfolio_id"],
            "candidate_wave_role": member["role"],
            "candidate_wave_generation_mode": member["generation_mode"],
            "candidate_wave_realization_kind": member["realization_kind"],
            "candidate_wave_exact": bool(member["exact"]),
            "candidate_wave_slot_directive": deepcopy(
                member.get("candidate_wave_slot_directive")
            ),
            "candidate_wave_slot_realization": deepcopy(
                member["candidate_wave_slot_realization"]
            ),
        }
        candidate.update(trusted)
        frozen.append(candidate)
    if len(frozen) != 8 or len(
        {item["candidate_sequence_bundle_hash"] for item in frozen}
    ) != 8:
        raise CandidateWaveError("candidate_wave_frozen_candidate_count_invalid")
    frozen.sort(key=lambda item: str(item["candidate_wave_slot_id"]))
    return frozen


def _select_candidate_wave_af3_subset(
    screen_mutants: Sequence[Dict[str, Any]],
    manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:


    expected_roles = {
        "matched_ablation": 2,
        "primary": 2,
        "repair": 2,
        "novelty": 1,
        "control": 1,
    }
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in screen_mutants:
        role = str(candidate.get("candidate_wave_role") or "")
        groups.setdefault(role, []).append(candidate)
    observed = {role: len(groups.get(role, [])) for role in expected_roles}
    if observed != expected_roles or len(screen_mutants) != 8:
        raise CandidateWaveError(
            "candidate_wave_screen_role_coverage_invalid", repr(observed)
        )

    selected = list(groups["matched_ablation"])
    role_decisions: Dict[str, Any] = {}
    for role in ("primary", "repair"):
        winner, decision = _select_final_structure_candidate(groups[role])
        selected.append(winner)
        role_decisions[role] = decision
    selected.sort(key=lambda item: str(item.get("candidate_wave_slot_id") or ""))
    if len(selected) != 4 or len(
        {item.get("candidate_sequence_bundle_hash") for item in selected}
    ) != 4:
        raise CandidateWaveError("candidate_wave_af3_subset_invalid")
    selected_roles: Dict[str, int] = {}
    for item in selected:
        role = str(item["candidate_wave_role"])
        selected_roles[role] = selected_roles.get(role, 0) + 1
    if selected_roles != {
        "matched_ablation": 2,
        "primary": 1,
        "repair": 1,
    }:
        raise CandidateWaveError(
            "candidate_wave_af3_role_quota_invalid", repr(selected_roles)
        )
    receipt = build_rerank_selection_receipt(manifest, selected)
    diagnostics = {
        "selection_policy": (
            "matched_pair_plus_feasibility_first_primary_repair"
        ),
        "role_selection_decisions": role_decisions,
    }
    return selected, receipt, diagnostics


def _evaluate_structure_candidate_dispatched(candidate: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return _record_structure_dispatch(candidate, evaluator=_evaluate_structure_candidate, **kwargs)


def _score_fast_with_run_memory(seqs: Dict[str, str], terms_fast: List[tuple[float, Any]], cfg: SAConfig, compiled: Dict[str, Any], run_memory: Optional[InnerRunMemory]) -> Tuple[Dict[str, float], Dict[str, float], float, bool]:
    return _cached_fast_evaluation(
        seqs, terms_fast, cfg, compiled, run_memory, scorer=_score_fast_candidate
    )


def _evaluate_structure_with_run_memory(candidate: Dict[str, Any], *, run_memory: Optional[InnerRunMemory], provider: str, stage: str, **kwargs: Any) -> Dict[str, Any]:
    return _cached_structure_evaluation(
        candidate, run_memory=run_memory, provider=provider, stage=stage,
        evaluator=_evaluate_structure_candidate_dispatched, **kwargs
    )


def _build_inner_structure_evaluator(
    *,
    compiled: Dict[str, Any],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    terms_chai: List[tuple[float, Any]],
    run_memory: Optional[InnerRunMemory],
    score_config: Optional[Mapping[str, Any]],
    design_state: Optional[Mapping[str, Any]],
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:


    if not bool(getattr(cfg, "inner_structure_enabled", False)):
        return None
    provider = str(getattr(cfg, "inner_structure_model", "esmfold2") or "esmfold2")
    weight = float(getattr(cfg, "inner_structure_weight", 1.0))
    failure_penalty = float(getattr(cfg, "inner_structure_failure_penalty", 1000.0))
    fail_closed = bool(getattr(cfg, "inner_structure_fail_closed", True))
    enforce_gate = bool(getattr(cfg, "inner_structure_hard_gate", True))
    esmfold2_enabled = bool(getattr(cfg, "inner_esmfold2_enabled", False))
    esmfold2_interval = int(getattr(cfg, "inner_esmfold2_interval", 10))
    esmfold2_weight = float(getattr(cfg, "inner_esmfold2_weight", 1.0))
    if weight < 0.0 or failure_penalty < 0.0:
        raise ValueError("inner structure weights and failure penalty must be non-negative")
    if esmfold2_interval < 1 or esmfold2_weight < 0.0:
        raise ValueError("invalid inner ESMFold2 cadence or weight")
    evaluation_count = 0
    esmfold2_count = 0

    def evaluate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal evaluation_count, esmfold2_count
        fast_loss = float(candidate.get("fast_loss", 0.0))
        try:
            evaluation_count += 1
            result = _evaluate_structure_with_run_memory(
                candidate,
                run_memory=run_memory,
                provider=provider,
                stage="inner_loop",
                compiled=compiled,
                cfg=cfg,
                masks=masks,
                template_seqs=template_seqs,
                fixed_residues=fixed_residues,
                terms_chai=terms_chai,
                score_config=dict(score_config) if score_config else None,
                design_state=dict(design_state) if design_state else None,
            )
            combined = float(result.get("combined_energy", result.get("combined_loss", fast_loss)))
            if not math.isfinite(combined) or not _has_structure_signal(result):
                metrics = result.get("structure_metrics") or {}
                warnings = (
                    list(metrics.get("warnings") or [])
                    if isinstance(metrics, Mapping)
                    else []
                )
                detail = "; ".join(str(item) for item in warnings[-3:])
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    "structure evaluator returned no finite structure signal"
                    + suffix
                )
            report = result.get("inner_evaluator_report") or {}
            energy = result.get("energy") or {}
            gate_pass = bool(report.get("hard_gate_pass", energy.get("hard_gate_pass", True)))
            selection_loss = fast_loss + weight * (combined - fast_loss)
            periodic = None
            if (
                esmfold2_enabled
                and evaluation_count % esmfold2_interval == 0
                and provider.lower() not in {"esmfold2", "esmfold2-fast"}
            ):
                esmfold2_count += 1
                try:
                    periodic_result = _evaluate_structure_with_run_memory(
                        candidate,
                        run_memory=run_memory,
                        provider="esmfold2",
                        stage="inner_loop_esmfold2_checkpoint",
                        compiled=compiled,
                        cfg=cfg,
                        masks=masks,
                        template_seqs=template_seqs,
                        fixed_residues=fixed_residues,
                        terms_chai=terms_chai,
                        score_config=dict(score_config) if score_config else None,
                        design_state=dict(design_state) if design_state else None,
                    )
                    periodic_combined = float(
                        periodic_result.get(
                            "combined_energy",
                            periodic_result.get("combined_loss", combined),
                        )
                    )
                    if math.isfinite(periodic_combined):
                        selection_loss += esmfold2_weight * (periodic_combined - fast_loss)
                    periodic = {
                        "status": "ok",
                        "provider": "esmfold2",
                        "combined_energy": periodic_combined,
                        "result": periodic_result,
                    }
                except Exception as periodic_exc:
                    periodic = {
                        "status": "failed",
                        "provider": "esmfold2",
                        "error": f"{type(periodic_exc).__name__}: {periodic_exc}",
                    }
            if enforce_gate and not gate_pass:
                selection_loss += failure_penalty
            return {
                "schema_version": "astevolve.inner_structure_evaluation.v1",
                "provider": provider,
                "status": "ok",
                "gate_pass": bool(gate_pass or not enforce_gate),
                "structure_combined_energy": combined,
                "selection_loss": float(selection_loss),
                "cache": deepcopy(result.get("structure_evidence_cache") or {}),
                "result": result,
                "esmfold2_checkpoint": periodic,
                "esmfold2_checkpoint_index": esmfold2_count if periodic is not None else None,
                "esmfold2_evaluation_count": evaluation_count,
            }
        except Exception as exc:
            gate_pass = not fail_closed
            return {
                "schema_version": "astevolve.inner_structure_evaluation.v1",
                "provider": provider,
                "status": "failed",
                "gate_pass": gate_pass,
                "structure_combined_energy": None,
                "selection_loss": float(fast_loss + (failure_penalty if fail_closed else 0.0)),
                "error": f"{type(exc).__name__}: {exc}",
                "esmfold2_evaluation_count": evaluation_count,
            }

    return evaluate


def _build_mcts_fidelity_upgrade_evaluator(
    *,
    compiled: Dict[str, Any],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    terms_chai: List[tuple[float, Any]],
    run_memory: Optional[InnerRunMemory],
    score_config: Optional[Mapping[str, Any]],
    design_state: Optional[Mapping[str, Any]],
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:


    if not bool(getattr(cfg, "mcts_fidelity_upgrade_enabled", False)):
        return None
    provider = str(cfg.mcts_fidelity_upgrade_provider).strip().lower()

    def evaluate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = _evaluate_structure_with_run_memory(
                candidate,
                run_memory=run_memory,
                provider=provider,


                stage="legacy",
                compiled=compiled,
                cfg=cfg,
                masks=masks,
                template_seqs=template_seqs,
                fixed_residues=fixed_residues,
                terms_chai=terms_chai,
                score_config=dict(score_config) if score_config else None,
                design_state=dict(design_state) if design_state else None,
            )
            combined = float(
                result.get("combined_energy", result.get("combined_loss", float("nan")))
            )
            if not math.isfinite(combined) or not _has_structure_signal(result):
                raise RuntimeError("high-fidelity evaluator returned no finite structure signal")
            report = result.get("inner_evaluator_report") or {}
            energy = result.get("energy") or {}
            return {
                "schema_version": FIDELITY_UPGRADE_VERSION,
                "provider": provider,
                "status": "ok",
                "selection_loss": combined,
                "gate_pass": bool(
                    report.get("hard_gate_pass", energy.get("hard_gate_pass", True))
                ),
                "result": result,
            }
        except Exception as exc:
            return {
                "schema_version": FIDELITY_UPGRADE_VERSION,
                "provider": provider,
                "status": "failed",
                "gate_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return evaluate


def _refresh_mcts_best_from_current_evidence(
    state: MCTSExpansionState,
    *,
    root_candidate: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    require_fidelity: bool,
) -> None:
    rows: List[Mapping[str, Any]] = [root_candidate, *candidates]
    if require_fidelity:
        rows = [
            row
            for row in rows
            if isinstance(row.get("mcts_fidelity_upgrade"), Mapping)
            and str(row["mcts_fidelity_upgrade"].get("status") or "") == "ok"
        ]
    eligible = [
        row
        for row in rows
        if not bool(row.get("duplicate_sequence"))
        and row.get("inner_structure_gate_pass") is not False
        and math.isfinite(float(row.get("selection_loss", float("inf"))))
    ]
    if not eligible:
        return
    selected = min(
        eligible,
        key=lambda row: (
            float(row.get("selection_loss", float("inf"))),
            str(row.get("variant_id") or "root"),
        ),
    )
    state.best_sequences = dict(selected.get("seqs") or state.best_sequences)
    state.best_fast = float(selected.get("fast_loss", state.best_fast))
    state.best_selection_loss = float(selected["selection_loss"])
    state.best_breakdown = {
        "total": float(selected.get("constraint_penalty", 0.0) or 0.0),
        "selection_source": (
            "high_fidelity" if require_fidelity else "current_best_available_fidelity"
        ),
    }
    state.best_progen = {
        "loglik_sum": float(selected.get("progen_loglik_sum", 0.0) or 0.0),
        "loglik_avg": float(selected.get("progen_loglik_avg", 0.0) or 0.0),
    }
    state.best_node_id = str(selected.get("variant_id") or "root")


def _run_mcts_fidelity_checkpoint(
    *,
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    root_candidate: Dict[str, Any],
    state: MCTSExpansionState,
    evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
    cfg: SAConfig,
    reward_scale: float,
    history: Dict[str, Any],
    checkpoint: int,
    wave_start: int,
    wave_end: int,
    final: bool = False,
) -> None:


    manifest = history.setdefault(
        "mcts_fidelity_upgrade",
        {
            "schema_version": FIDELITY_UPGRADE_VERSION,
            "enabled": True,
            "provider": str(cfg.mcts_fidelity_upgrade_provider),
            "interval": int(cfg.mcts_fidelity_upgrade_interval),
            "candidates_per_checkpoint": int(
                cfg.mcts_fidelity_upgrade_candidates
            ),
            "final_candidate_target": int(
                cfg.mcts_fidelity_upgrade_final_candidates
            ),
            "reward_update": "delta_backprop_visits_unchanged",
            "tree_persistence": "single_inner_run_only",
            "checkpoints": [],
        },
    )

    if str(getattr(cfg, "inner_structure_model", "") or "").strip().lower() in {
        "esmfold",
        "esmfold_v1",
        "classic_esmfold",
    }:
        from astevolve.providers.esmfold import release_esmfold_resources

        release_esmfold_resources()
        manifest["inline_provider_release_count"] = int(
            manifest.get("inline_provider_release_count", 0)
        ) + 1

    if not isinstance(root_candidate.get("mcts_fidelity_upgrade"), Mapping):
        root_filter_pass = bool(
            (root_candidate.get("fast_filter") or {}).get("pass", True)
        )
        root_upgrade = (
            evaluator(root_candidate)
            if root_filter_pass
            else {
                "schema_version": FIDELITY_UPGRADE_VERSION,
                "status": "skipped",
                "gate_pass": False,
                "selection_loss": float(root_candidate.get("fast_loss", 0.0)),
                "reason": "parent_failed_pre_model_sequence_gate",
            }
        )
        root_candidate["mcts_fidelity_upgrade"] = root_upgrade
        if str(root_upgrade.get("status") or "") not in {"ok", "skipped"}:
            if bool(cfg.mcts_fidelity_upgrade_required):
                raise RuntimeError(
                    "mcts_fidelity_root_upgrade_failed:"
                    + str(root_upgrade.get("error") or "unknown")
                )
        elif str(root_upgrade.get("status") or "") == "ok":
            root_loss = float(root_upgrade["selection_loss"])
            root_candidate.setdefault(
                "proxy_selection_loss", root_candidate.get("selection_loss")
            )
            root_candidate["selection_loss"] = root_loss
            root_candidate["inner_structure_gate_pass"] = bool(
                root_upgrade.get("gate_pass", False)
            )
            tree["root"]["selection_loss"] = root_loss
            tree["root"]["mcts_fidelity_upgrade"] = root_upgrade
            tree["root"]["inner_structure_gate_pass"] = bool(
                root_upgrade.get("gate_pass", False)
            )

    if final:
        selected = _select_final_fidelity_cohort(
            candidates,
            limit=int(cfg.mcts_fidelity_upgrade_final_candidates),
        )
    else:
        selected = _select_fidelity_upgrade_cohort(
            candidates,
            limit=int(cfg.mcts_fidelity_upgrade_candidates),
            wave_start=wave_start,
            wave_end=wave_end,
        )
    root_upgrade = root_candidate.get("mcts_fidelity_upgrade") or {}
    raw_root_loss = root_upgrade.get("selection_loss")
    if raw_root_loss is None:
        raw_root_loss = root_candidate.get("selection_loss")
    if raw_root_loss is None:
        raw_root_loss = root_candidate.get("fast_loss", 0.0)
    root_loss = float(raw_root_loss)
    event = {
        "checkpoint": int(checkpoint),
        "final": bool(final),
        "wave_start": int(wave_start),
        "wave_end": int(wave_end),
        "selected": [],
    }
    failures = []
    for candidate, lane in selected:
        node_id = str(candidate.get("variant_id") or "")
        old_reward = float(candidate.get("reward", 0.0) or 0.0)
        old_loss = float(candidate.get("selection_loss", candidate.get("fast_loss", 0.0)))
        upgrade = evaluator(candidate)
        receipt = {
            "variant_id": node_id,
            "lane": lane,
            "status": upgrade.get("status"),
            "old_selection_loss": old_loss,
        }
        if str(upgrade.get("status") or "") != "ok":
            candidate["mcts_fidelity_upgrade"] = upgrade
            receipt["error"] = upgrade.get("error")
            failures.append(receipt)
            event["selected"].append(receipt)
            continue

        new_loss = float(upgrade["selection_loss"])
        gate_pass = bool(upgrade.get("gate_pass", False))
        new_reward = (
            float(np.tanh((root_loss - new_loss) / reward_scale))
            if gate_pass
            else -1.0
        )
        delta = _apply_fidelity_reward_delta(
            tree,
            node_id,
            old_reward=old_reward,
            new_reward=new_reward,
        )
        upgrade = dict(upgrade)
        upgrade.update(
            {
                "checkpoint": int(checkpoint),
                "lane": lane,
                "proxy_selection_loss": old_loss,
                "old_reward": old_reward,
                "new_reward": new_reward,
                "reward_delta": delta,
                "visits_incremented": False,
            }
        )
        candidate.setdefault("proxy_selection_loss", old_loss)
        candidate["selection_loss"] = new_loss
        candidate["reward"] = new_reward
        candidate["inner_structure_gate_pass"] = gate_pass
        candidate["mcts_fidelity_upgrade"] = upgrade
        if isinstance(candidate.get("energy"), dict):
            candidate["energy"]["selection_loss"] = new_loss
            candidate["energy"]["fidelity_upgrade_provider"] = str(
                cfg.mcts_fidelity_upgrade_provider
            )
        node = tree[node_id]
        node.setdefault("proxy_selection_loss", old_loss)
        node["selection_loss"] = new_loss
        node["reward"] = new_reward
        node["inner_structure_gate_pass"] = gate_pass
        node["mcts_fidelity_upgrade"] = upgrade
        receipt.update(
            {
                "new_selection_loss": new_loss,
                "old_reward": old_reward,
                "new_reward": new_reward,
                "reward_delta": delta,
                "gate_pass": gate_pass,
            }
        )
        event["selected"].append(receipt)

    _refresh_fidelity_best_reward(tree)
    _refresh_mcts_best_from_current_evidence(
        state,
        root_candidate=root_candidate,
        candidates=candidates,
        require_fidelity=bool(final),
    )
    event["successful"] = sum(row.get("status") == "ok" for row in event["selected"])
    event["failed"] = len(failures)
    event["best_node_id_after_checkpoint"] = state.best_node_id
    event["best_selection_loss_after_checkpoint"] = state.best_selection_loss
    manifest["checkpoints"].append(event)
    if failures and bool(cfg.mcts_fidelity_upgrade_required):
        raise RuntimeError(
            "mcts_fidelity_upgrade_failed:"
            + ";".join(
                f"{row['variant_id']}:{row.get('error')}" for row in failures
            )
        )


def _completed_inline_evaluator_count(
    candidates: Sequence[Mapping[str, Any]],
) -> int:


    identities = set()
    for candidate in candidates:
        if bool(candidate.get("duplicate_sequence")):
            continue
        evaluation = candidate.get("inner_structure_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        if str(evaluation.get("status") or "") != "ok":
            continue
        result = evaluation.get("result")
        if not isinstance(result, Mapping):
            continue
        if not isinstance(result.get("inner_evaluator_report"), Mapping):
            continue
        identity = str(candidate.get("seq_hash") or "")
        if identity:
            identities.add(identity)
    return len(identities)


def _inline_evaluator_budget_diagnostics(
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    errors: List[str] = []
    for candidate in candidates:
        evaluation = candidate.get("inner_structure_evaluation")
        status = (
            str(evaluation.get("status") or "missing")
            if isinstance(evaluation, Mapping)
            else "missing"
        )
        statuses[status] = int(statuses.get(status, 0)) + 1
        if isinstance(evaluation, Mapping) and evaluation.get("error"):
            error = str(evaluation["error"])
            if error not in errors:
                errors.append(error)
    return {
        "candidate_rows": len(candidates),
        "unique_nonduplicate_rows": sum(
            not bool(candidate.get("duplicate_sequence"))
            for candidate in candidates
        ),
        "fast_filter_pass_rows": sum(
            bool((candidate.get("fast_filter") or {}).get("pass", True))
            for candidate in candidates
            if not bool(candidate.get("duplicate_sequence"))
        ),
        "inline_status_counts": dict(sorted(statuses.items())),
        "distinct_inline_errors": errors[:8],
    }


def _candidate_wave_free_attempt_specs(
    designable: Sequence[Tuple[Any, List[int]]],
    mapping_actions: Sequence[Mapping[str, Any]],
    *,
    slot_index: int,
) -> List[Tuple[int, Optional[Dict[str, Any]]]]:


    actions_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for raw in mapping_actions:
        action = deepcopy(dict(raw))
        node_name = str(action.get("compiled_segment_name") or "")
        if node_name:
            actions_by_node.setdefault(node_name, []).append(action)
    for actions in actions_by_node.values():
        actions.sort(key=lambda item: str(item.get("action_id") or ""))

    eligible: List[int] = []
    for index, (segment, _positions) in enumerate(designable):
        node_name = str(getattr(segment, "name", "") or "")
        if mapping_actions and not actions_by_node.get(node_name):
            continue
        eligible.append(index)
    if not eligible:
        return []
    offset = int(slot_index) % len(eligible)
    rotated = eligible[offset:] + eligible[:offset]
    attempts: List[Tuple[int, Optional[Dict[str, Any]]]] = []
    for segment_index in rotated:
        node_name = str(getattr(designable[segment_index][0], "name", "") or "")
        actions = actions_by_node.get(node_name, [])
        if not mapping_actions:
            attempts.append((segment_index, None))
            continue
        action_offset = int(slot_index) % len(actions)
        for action in actions[action_offset:] + actions[:action_offset]:
            attempts.append((segment_index, deepcopy(action)))
    return attempts


def _run_mcts_search(
    compiled: Dict[str, Any],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    rng: np.random.Generator,
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    internal_memory: Optional[Dict[str, Any]],
    run_memory: Optional[InnerRunMemory] = None,
    causal_context: Optional[Mapping[str, Any]] = None,
    prebuilt_proposals: Optional[Sequence[PrebuiltProposal]] = None,
    portfolio_materialization_accounting: Optional[Mapping[str, Any]] = None,
    portfolio_seed_refinement_directives: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    candidate_wave_free_slot_directives: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    inner_structure_evaluator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    mcts_fidelity_upgrade_evaluator: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ] = None,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], float, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:


    chain_lengths = compiled["chain_lengths"]
    root_seqs = init_seqs(
        chain_lengths, rng, template_seqs=template_seqs, fixed_residues=fixed_residues
    )
    _register_history_sequence(root_seqs, role="root", context_id="mcts:root", metadata={"search_method": "mcts"})
    root_break, root_progen, root_fast, root_fast_cache_hit = _score_fast_with_run_memory(
        root_seqs, terms_fast, cfg, compiled, run_memory
    )
    if run_memory is not None:
        run_memory.claim_sequence(root_seqs, node_id="root")
        run_memory.record_transposition(
            root_seqs,
            node_id="root",
            payload={"fast_loss": float(root_fast)},
        )
    root_candidate = _build_parent_baseline_candidate(
        root_seqs,
        root_break,
        root_progen,
        root_fast,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
        compiled=compiled,
        cfg=cfg,
    )
    _bind_causal_candidate(root_candidate, causal_context)
    root_inner_structure = (
        inner_structure_evaluator(root_candidate)
        if inner_structure_evaluator is not None
        and bool((root_candidate.get("fast_filter") or {}).get("pass", True))
        else None
    )
    root_selection_loss = float(root_fast)
    root_structure_gate_pass = bool(
        (root_candidate.get("fast_filter") or {}).get("pass", True)
    )
    if root_inner_structure is not None:
        root_selection_loss = float(root_inner_structure["selection_loss"])
        root_structure_gate_pass = bool(root_inner_structure.get("gate_pass", False))
        root_candidate["inner_structure_evaluation"] = root_inner_structure
        root_candidate["inner_structure_loss"] = root_inner_structure.get("structure_combined_energy")
        root_candidate["inner_structure_gate_pass"] = root_structure_gate_pass
        root_candidate["selection_loss"] = root_selection_loss

    designable = _designable_segments(compiled, masks)
    if not designable:
        mapping_actions = _active_mapping_actions(compiled, cfg)
        joint_completion = initialize_joint_coverage_completion(cfg, [], mapping_actions)
        joint_completion["underfill_reasons"] = list(
            dict.fromkeys(
                list(joint_completion.get("underfill_reasons", []) or [])
                + ["search_aborted_no_designable_positions"]
            )
        )
        finalize_joint_coverage_completion(joint_completion)
        requested_rounds = max(0, int(cfg.iterations))
        history = {
            "accepted_moves": [],
            "op_counts": {},
            "node_visit_counts": {},
            **_proposal_tier_history_fields(cfg),
            "fast_filter_failures": {},
            "semantic_required_nodes": _unique_strings(getattr(cfg, "semantic_required_nodes", [])),
            "semantic_designable_required_nodes": [],
            "semantic_unavailable_required_nodes": _unique_strings(getattr(cfg, "semantic_required_nodes", [])),
            "semantic_required_node_visits": {},
            "semantic_required_node_mutations": {},
            "expansion_rounds": 0,
            "effective_expansion_rounds": 0,
            "stalled_expansion_rounds": 0,
            "logical_candidates": 0,
            "joint_coverage_completion": joint_completion,
            "expansion_round_accounting": {
                "schema_version": "astevolve.mcts_expansion_round_accounting.v1",
                "requested_rounds": requested_rounds,
                "physical_rounds": 0,
                "effective_rounds": 0,
                "stalled_rounds": 0,
                "effective_definition": (
                    "at_least_one_novel_tree_child_committed"
                ),
                "physical_rounds_include_stalls": True,
                "aborted_before_expansion": True,
                "abort_reason": "no_designable_positions",
            },
        }
        return root_seqs, root_break, root_progen, root_fast, history, [], {
            "method": "mcts",
            "best_node_id": "root",
            "best_path": ["root"],
            "best_final_is_root": True,
            "root_candidate": root_candidate,
            "artifact_paths": {},
            "joint_coverage_completion": joint_completion,
            "expansion_round_accounting": history[
                "expansion_round_accounting"
            ],
        }

    raw_priors = np.array([
        _segment_prior(
            seg,
            internal_memory,
            _node_policy(cfg, seg),
        )
        for seg, _ in designable
    ], dtype=float)
    raw_priors = np.maximum(raw_priors, 1e-6)
    norm_priors = raw_priors / raw_priors.sum()

    tree: Dict[str, Dict[str, Any]] = {
        "root": {
            "id": "root",
            "parent": None,
            "children": [],
            "depth": 0,
            "visits": 0,
            "total_reward": 0.0,
            "best_reward": -1e9,
            "prior": 1.0,
            "move": None,
            "seqs": root_seqs,
            "fast_loss": float(root_fast),
            "selection_loss": float(root_selection_loss),
            "inner_structure_gate_pass": bool(root_structure_gate_pass),
            "inner_structure_evaluation": root_inner_structure,
            "constraint_penalty": float(root_break["total"]),
            "progen_loglik_avg": float(root_progen["loglik_avg"]),
            "energy": fast_energy_record(
                fast_loss=root_fast,
                constraint_penalty=root_break["total"],
                progen_loglik_avg=root_progen["loglik_avg"],
                progen_weight=cfg.progen_weight,
                hard_gate_pass=bool(root_structure_gate_pass),
            ),
        }
    }
    candidates: List[Dict[str, Any]] = []
    history: Dict[str, Any] = {
        "accepted_moves": [],
        "op_counts": {},
        "node_visit_counts": {},
        **_proposal_tier_history_fields(cfg),
        "fast_filter_failures": {},
        "search_method": "mcts",
        "semantic_required_nodes": _unique_strings(getattr(cfg, "semantic_required_nodes", [])),
        "semantic_designable_required_nodes": _semantic_required_nodes(cfg, designable),
        "semantic_unavailable_required_nodes": _semantic_required_unavailable_nodes(cfg, designable),
        "semantic_required_node_force_steps": _semantic_force_steps(cfg),
        "semantic_required_node_visits": {},
        "semantic_required_node_mutations": {},
        "duplicate_sequence_attempts": 0,
        "fast_cache_hits": int(root_fast_cache_hit),
    }

    expansion_state = MCTSExpansionState(
        candidate_serial=0,
        best_sequences=root_seqs,
        best_breakdown=root_break,
        best_progen=root_progen,
        best_fast=float(root_fast),
        best_selection_loss=float(root_selection_loss),
        best_node_id="root",
    )
    reward_scale = max(float(cfg.mcts_reward_scale), abs(float(root_fast)) * 0.05, 1.0)
    mapping_actions = _active_mapping_actions(compiled, cfg)
    designable_by_name = {
        str(segment.name): index
        for index, (segment, _positions) in enumerate(designable)
    }
    for action in mapping_actions:
        if not isinstance(action, Mapping):
            raise ValueError("executable mapping action spec must be a mapping")
        segment_name = str(action.get("compiled_segment_name") or "")
        if segment_name not in designable_by_name:
            raise ValueError(
                f"mapping action {action.get('action_id')!r} targets unavailable "
                f"segment {segment_name!r}"
            )


    if prebuilt_proposals and mapping_actions:
        prebuilt_proposals = _bind_portfolio_mapping_components(
            prebuilt_proposals,
            mapping_actions,
        )

    joint_completion = initialize_joint_coverage_completion(cfg, designable, mapping_actions)
    history["joint_coverage_completion"] = joint_completion
    history["expansion_rounds"] = 0
    history["effective_expansion_rounds"] = 0
    history["stalled_expansion_rounds"] = 0
    history["logical_candidates"] = 0
    prebuilt_segment, prebuilt_positions = designable[0]
    expansion_state = materialize_prebuilt_mcts_root_children(
        expansion_state,
        prebuilt_proposals=prebuilt_proposals,
        tree=tree,
        candidates=candidates,
        history=history,
        segment=prebuilt_segment,
        designable_positions=prebuilt_positions,
        rng=rng,
        cfg=cfg,
        masks=masks,
        internal_memory=internal_memory,
        fixed_residues=fixed_residues,
        compiled=compiled,
        template_seqs=template_seqs,
        terms_fast=terms_fast,
        root_fast=float(root_fast),
        reward_scale=reward_scale,
        run_memory=run_memory,
        causal_context=causal_context,
        score_fast=_score_fast_with_run_memory,
        inner_structure_evaluator=inner_structure_evaluator,
        candidate_evaluation_slots=(
            int(cfg.iterations)
            if cfg.mcts_iteration_unit == "evaluated_unique_candidates"
            else None
        ),
    )
    seed_refinement_schedule: List[Dict[str, Any]] = []
    exact_candidate_by_slot: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        for receipt in candidate.get("portfolio_realization_receipts", []) or []:
            if isinstance(receipt, Mapping) and receipt.get("slot_id"):
                exact_candidate_by_slot[str(receipt["slot_id"])] = candidate
    for directive in portfolio_seed_refinement_directives or ():
        if not isinstance(directive, Mapping):
            raise ValueError("portfolio seed refinement directive must be a mapping")
        slot_id = str(directive.get("slot_id") or "")
        seed_candidate = exact_candidate_by_slot.get(slot_id)
        if seed_candidate is None:
            continue
        raw_positions = directive.get("refinement_positions") or ()
        for segment_index, (segment, positions) in enumerate(designable):
            allowed = sorted(
                {
                    int(item.get("position"))
                    for item in raw_positions
                    if isinstance(item, Mapping)
                    and str(item.get("chain_id") or "")
                    == str(segment.chain_id)
                    and int(item.get("position", -1)) in positions
                }
            )
            if allowed:
                seed_refinement_schedule.append(
                    {
                        "seed_id": str(directive.get("seed_id") or ""),
                        "slot_id": slot_id,
                        "seed_node_id": str(seed_candidate.get("variant_id")),
                        "segment_index": segment_index,
                        "designable_positions": allowed,
                        "sequence_bundle_hash": directive.get(
                            "sequence_bundle_hash"
                        ),
                    }
                )
                break
    requested_seed_refinement_rounds = int(
        getattr(cfg, "portfolio_seed_refinement_rounds", 0) or 0
    )
    if requested_seed_refinement_rounds or portfolio_seed_refinement_directives:
        history["portfolio_seed_refinement"] = {
            "schema_version": "astevolve.portfolio_seed_refinement_accounting.v1",
            "requested_rounds": requested_seed_refinement_rounds,
            "available_directive_count": len(seed_refinement_schedule),
            "executed_rounds": 0,
            "descendant_candidate_ids": [],
            "underfill": bool(
                requested_seed_refinement_rounds > 0
                and not seed_refinement_schedule
            ),
        }
    free_slot_schedule = [
        deepcopy(dict(item))
        for item in (candidate_wave_free_slot_directives or ())
        if isinstance(item, Mapping)
    ]
    if len(free_slot_schedule) != len(candidate_wave_free_slot_directives or ()):
        raise ValueError("candidate wave free-slot directive must be a mapping")
    if len(free_slot_schedule) > int(cfg.iterations):
        raise ValueError(
            "candidate wave free-slot schedule exceeds physical MCTS rounds"
        )
    if requested_seed_refinement_rounds and free_slot_schedule:
        raise ValueError(
            "candidate wave free slots and seed refinement cannot share rounds"
        )
    if free_slot_schedule:
        history["candidate_wave_free_slot_schedule"] = {
            "schema_version": "astevolve.candidate_wave_free_slot_schedule.v1",
            "requested_count": len(free_slot_schedule),
            "slot_ids": [
                str(item.get("slot_id") or "") for item in free_slot_schedule
            ],
            "executed_count": 0,
            "scheduling_policy": (
                "distinct_segment_rotation_then_same_round_legal_fallback"
            ),
            "uniqueness_policy": "wave_global_full_sequence_bundle",
            "realization_attempts": [],
        }
    node_sweep_enabled = bool(getattr(cfg, "mcts_node_sweep_enabled", False))
    node_sweep_segment_indices = list(range(len(designable)))
    node_sweep_actions_by_segment: Dict[str, List[Dict[str, Any]]] = {}
    for raw_action in mapping_actions:
        segment_name = str(raw_action.get("compiled_segment_name") or "")
        node_sweep_actions_by_segment.setdefault(segment_name, []).append(dict(raw_action))
    if node_sweep_enabled:
        expected_candidates = int(cfg.mcts_node_sweep_count) * len(node_sweep_segment_indices)
        if int(cfg.iterations) != expected_candidates:
            raise ValueError(
                "node sweep candidate budget must equal sweep_count * designable_nodes: "
                f"{cfg.iterations} != {cfg.mcts_node_sweep_count} * {len(node_sweep_segment_indices)}"
            )
        missing_actions = [
            str(designable[index][0].name)
            for index in node_sweep_segment_indices
            if not node_sweep_actions_by_segment.get(str(designable[index][0].name))
        ]
        if missing_actions:
            raise ValueError(f"node sweep nodes lack executable mapping actions: {missing_actions}")
        history["node_sweep_summary"] = {
            "schema_version": "astevolve.node_sweep_summary.v1",
            "enabled": True,
            "sweep_count": int(cfg.mcts_node_sweep_count),
            "node_order": [str(designable[index][0].name) for index in node_sweep_segment_indices],
            "candidate_per_node_visit": 1,
            "parent_policy": "incumbent",
            "requested_candidates": expected_candidates,
            "nodes": {},
        }

    evaluated_budget_mode = (
        cfg.mcts_iteration_unit == "evaluated_unique_candidates"
    )
    requested_evaluated_candidates = (
        int(cfg.iterations) if evaluated_budget_mode else None
    )
    maximum_expansion_rounds = (
        int(cfg.iterations)
        * int(cfg.mcts_candidate_budget_max_round_multiplier)
        if evaluated_budget_mode
        else int(cfg.iterations)
    )
    next_fidelity_checkpoint = int(cfg.mcts_fidelity_upgrade_interval)
    last_fidelity_wave_end = 0
    for step in range(maximum_expansion_rounds):
        completed_before = _completed_inline_evaluator_count(candidates)
        if (
            requested_evaluated_candidates is not None
            and completed_before >= requested_evaluated_candidates
        ):
            break
        remaining_evaluation_slots = (
            requested_evaluated_candidates - completed_before
            if requested_evaluated_candidates is not None
            else None
        )
        history["expansion_rounds"] = int(history["expansion_rounds"]) + 1
        history["step"] = int(step)
        completion_directive = joint_coverage_expansion_directive(
            joint_completion,
            designable_by_name=designable_by_name,
            mapping_actions=mapping_actions,
            tree=tree,
        )
        mapping_action: Optional[Dict[str, Any]] = None
        if node_sweep_enabled:
            sweep_slot = int(completed_before)
            sweep_index = sweep_slot // len(node_sweep_segment_indices)
            node_slot = sweep_slot % len(node_sweep_segment_indices)
            seg_idx = node_sweep_segment_indices[node_slot]
            seg_for_sweep, _positions_for_sweep = designable[seg_idx]
            segment_name = str(seg_for_sweep.name)
            action_rows = node_sweep_actions_by_segment[segment_name]
            mapping_action = dict(action_rows[sweep_index % len(action_rows)])
            selection = {
                "source": "node_sweep",
                "sweep_index": int(sweep_index),
                "node_slot": int(node_slot),
                "ast_id": mapping_action.get("ast_id"),
                "ast_revision": mapping_action.get("ast_revision"),
                "edge_id": mapping_action.get("edge_id"),
                "functional_node_id": mapping_action.get("functional_node_id"),
                "structural_node_id": mapping_action.get("structural_node_id"),
                "action_id": mapping_action.get("action_id"),
            }
            expansion_key = (
                str(mapping_action.get("action_id") or f"segment:{segment_name}")
                + f"|node-sweep:{sweep_index}:{node_slot}"
            )
            parent_id = (
                str(expansion_state.best_node_id)
                if str(expansion_state.best_node_id) in tree
                else "root"
            )
            completion_directive = None
        elif completion_directive is not None:
            seg_idx = int(completion_directive["segment_index"])
            raw_mapping_action = completion_directive.get("mapping_action")
            mapping_action = (
                dict(raw_mapping_action)
                if isinstance(raw_mapping_action, Mapping)
                else None
            )
            selection = dict(completion_directive["selection"])
            if mapping_action is not None:
                selection.update(
                    {
                        "ast_id": mapping_action.get("ast_id"),
                        "ast_revision": mapping_action.get("ast_revision"),
                        "edge_id": mapping_action.get("edge_id"),
                        "functional_node_id": mapping_action.get(
                            "functional_node_id"
                        ),
                        "structural_node_id": mapping_action.get(
                            "structural_node_id"
                        ),
                        "action_id": mapping_action.get("action_id"),
                    }
                )
            expansion_key = str(completion_directive["expansion_key"])
            parent_id = str(completion_directive["parent_id"])
        else:
            if mapping_actions:
                mapping_action = dict(mapping_actions[step % len(mapping_actions)])
                seg_idx = designable_by_name[
                    str(mapping_action["compiled_segment_name"])
                ]
                selection = {
                    "source": "dual_ast_mapping_edge",
                    "ast_id": mapping_action.get("ast_id"),
                    "ast_revision": mapping_action.get("ast_revision"),
                    "edge_id": mapping_action.get("edge_id"),
                    "functional_node_id": mapping_action.get("functional_node_id"),
                    "structural_node_id": mapping_action.get("structural_node_id"),
                    "action_id": mapping_action.get("action_id"),
                }
            else:
                seg_idx, selection = _select_semantic_segment(
                    designable, norm_priors, cfg, history, rng
                )
            seg_for_key, _positions_for_key = designable[seg_idx]
            expansion_key = str(
                (mapping_action or {}).get("action_id")
                or f"segment:{seg_for_key.name}"
            )
            parent_id = _mcts_select_leaf(
                tree,
                "root",
                cfg,
                expansion_key=expansion_key,
                deferred_expansions=expansion_state.deferred_expansions,
            )
            if (
                _semantic_coverage_hard_enabled(cfg)
                and selection.get("source") == "semantic_required_node"
                and history.get("semantic_prefix_node_id") in tree
                and int(
                    tree[history["semantic_prefix_node_id"]].get("depth", 0)
                )
                < int(cfg.mcts_max_depth)
            ):
                parent_id = str(history["semantic_prefix_node_id"])
                selection["parent_policy"] = "semantic_prefix"
            elif (
                _semantic_coverage_hard_enabled(cfg)
                and selection.get("source") == "semantic_required_node"
            ):
                selection["parent_policy"] = "semantic_root"
        seed_refinement = None
        if step < requested_seed_refinement_rounds and seed_refinement_schedule:
            seed_refinement = seed_refinement_schedule[
                step % len(seed_refinement_schedule)
            ]
            seg_idx = int(seed_refinement["segment_index"])
            parent_id = str(seed_refinement["seed_node_id"])
            if parent_id not in tree:
                raise ValueError(
                    "portfolio seed refinement parent is absent from MCTS tree"
                )
            mapping_action = None
            completion_directive = None
            expansion_key = "portfolio-seed-refine:" + str(
                seed_refinement["seed_id"]
            )
            selection = {
                "source": "portfolio_seed_refinement",
                "seed_id": seed_refinement["seed_id"],
                "slot_id": seed_refinement["slot_id"],
                "seed_node_id": parent_id,
            }
        wave_slot_directive = (
            free_slot_schedule[step]
            if step < len(free_slot_schedule)
            else None
        )
        slot_id = ""
        if wave_slot_directive is not None:
            slot_id = str(wave_slot_directive.get("slot_id") or "")
            if not slot_id:
                raise ValueError("candidate wave free-slot directive has no slot_id")
        tree_nodes_before = len(tree)
        candidates_before = len(candidates)
        realized_parent_id = parent_id
        realized_completion_directive = completion_directive
        if wave_slot_directive is not None:
            schedule_accounting = history["candidate_wave_free_slot_schedule"]
            attempt_specs = _candidate_wave_free_attempt_specs(
                designable,
                mapping_actions,
                slot_index=step,
            )
            if not attempt_specs:
                raise CandidateWaveError(
                    "candidate_wave_free_slot_no_executable_segment", slot_id
                )
            slot_attempts: List[Dict[str, Any]] = []
            realized = False
            for attempt_index, (attempt_seg_idx, attempt_mapping) in enumerate(
                attempt_specs
            ):
                attempt_seg, attempt_positions = designable[attempt_seg_idx]
                attempt_action_id = str(
                    (attempt_mapping or {}).get("action_id")
                    or f"segment:{attempt_seg.name}"
                )
                attempt_expansion_key = (
                    f"{attempt_action_id}|candidate-wave-slot:{slot_id}"
                    f"|attempt:{attempt_index}"
                )
                attempt_parent_id = _mcts_select_leaf(
                    tree,
                    "root",
                    cfg,
                    expansion_key=attempt_expansion_key,
                    deferred_expansions=expansion_state.deferred_expansions,
                )
                attempt_selection = {
                    "source": "candidate_wave_free_slot",
                    "candidate_wave_slot_id": slot_id,
                    "candidate_wave_role": wave_slot_directive.get("role"),
                    "candidate_wave_forced_tier": wave_slot_directive.get(
                        "forced_tier"
                    ),
                    "candidate_wave_segment_rotation_index": int(step),
                    "candidate_wave_attempt_index": int(attempt_index),
                    "compiled_segment_name": str(attempt_seg.name),
                    "action_id": (
                        attempt_mapping.get("action_id")
                        if attempt_mapping is not None
                        else None
                    ),
                }
                attempt_candidates_before = len(candidates)
                expansion_state = expand_mcts_candidate_siblings(
                    expansion_state,
                    tree=tree,
                    candidates=candidates,
                    history=history,
                    parent_id=attempt_parent_id,
                    expansion_key=attempt_expansion_key,
                    segment=attempt_seg,
                    designable_positions=attempt_positions,
                    segment_prior=float(norm_priors[attempt_seg_idx]),
                    selection=attempt_selection,
                    expansion_round=step,
                    rng=rng,
                    cfg=cfg,
                    masks=masks,
                    internal_memory=internal_memory,
                    mapping_action=attempt_mapping,
                    fixed_residues=fixed_residues,
                    compiled=compiled,
                    template_seqs=template_seqs,
                    terms_fast=terms_fast,
                    root_fast=float(root_fast),
                    reward_scale=reward_scale,
                    run_memory=run_memory,
                    causal_context=causal_context,
                    score_fast=_score_fast_with_run_memory,
                    inner_structure_evaluator=inner_structure_evaluator,
                    candidate_wave_slot_directive=wave_slot_directive,
                    candidate_evaluation_slots=remaining_evaluation_slots,
                )
                new_unique = [
                    candidate
                    for candidate in candidates[attempt_candidates_before:]
                    if not bool(candidate.get("duplicate_sequence"))
                    and str(
                        (
                            candidate.get("candidate_wave_slot_directive")
                            or (candidate.get("move") or {}).get(
                                "candidate_wave_slot_directive"
                            )
                            or {}
                        ).get("slot_id")
                        or ""
                    )
                    == slot_id
                ]
                slot_attempts.append(
                    {
                        "attempt_index": int(attempt_index),
                        "physical_round": int(step),
                        "segment_name": str(attempt_seg.name),
                        "action_id": attempt_action_id,
                        "generated_unique_count": len(new_unique),
                        "status": "realized" if new_unique else "exhausted",
                    }
                )
                if new_unique:
                    realized = True
                    realized_parent_id = attempt_parent_id


                    realized_completion_directive = None
                    break
            schedule_accounting["realization_attempts"].append(
                {
                    "slot_id": slot_id,
                    "role": str(wave_slot_directive.get("role") or ""),
                    "forced_tier": str(
                        wave_slot_directive.get("forced_tier") or ""
                    ),
                    "physical_round": int(step),
                    "attempts": slot_attempts,
                    "status": "realized" if realized else "underfilled",
                }
            )
            if not realized:
                raise CandidateWaveError(
                    "candidate_wave_free_slot_generation_exhausted", slot_id
                )
            schedule_accounting["executed_count"] = int(
                schedule_accounting["executed_count"]
            ) + 1
        else:
            seg, positions = designable[seg_idx]
            if seed_refinement is not None:
                positions = list(seed_refinement["designable_positions"])
            expansion_state = expand_mcts_candidate_siblings(
                expansion_state,
                tree=tree,
                candidates=candidates,
                history=history,
                parent_id=parent_id,
                expansion_key=expansion_key,
                segment=seg,
                designable_positions=positions,
                segment_prior=float(norm_priors[seg_idx]),
                selection=selection,
                expansion_round=step,
                rng=rng,
                cfg=cfg,
                masks=masks,
                internal_memory=internal_memory,
                mapping_action=mapping_action,
                fixed_residues=fixed_residues,
                compiled=compiled,
                template_seqs=template_seqs,
                terms_fast=terms_fast,
                root_fast=float(root_fast),
                reward_scale=reward_scale,
                run_memory=run_memory,
                causal_context=causal_context,
                score_fast=_score_fast_with_run_memory,
                inner_structure_evaluator=inner_structure_evaluator,
                candidate_wave_slot_directive=None,
                candidate_evaluation_slots=remaining_evaluation_slots,
            )
        effective_expansion = len(tree) > tree_nodes_before
        if seed_refinement is not None:
            seed_accounting = history["portfolio_seed_refinement"]
            seed_accounting["executed_rounds"] = int(
                seed_accounting["executed_rounds"]
            ) + 1
            new_ids = [
                str(candidate.get("variant_id"))
                for candidate in candidates[candidates_before:]
                if str(candidate.get("parent_id")) == str(realized_parent_id)
                and not bool(candidate.get("duplicate_sequence"))
            ]
            seed_accounting["descendant_candidate_ids"].extend(new_ids)
            if not new_ids:
                seed_accounting["underfill"] = True
        if effective_expansion:
            history["effective_expansion_rounds"] = int(
                history["effective_expansion_rounds"]
            ) + 1
        else:
            history["stalled_expansion_rounds"] = int(
                history["stalled_expansion_rounds"]
            ) + 1
        if realized_completion_directive is not None:
            update_joint_coverage_completion(
                joint_completion,
                directive=realized_completion_directive,
                new_candidates=candidates[candidates_before:],
                tree=tree,
                expansion_round=step,
                effective=effective_expansion,
            )
        completed_after = _completed_inline_evaluator_count(candidates)
        while (
            mcts_fidelity_upgrade_evaluator is not None
            and completed_after >= next_fidelity_checkpoint
        ):
            _run_mcts_fidelity_checkpoint(
                tree=tree,
                candidates=candidates,
                root_candidate=root_candidate,
                state=expansion_state,
                evaluator=mcts_fidelity_upgrade_evaluator,
                cfg=cfg,
                reward_scale=reward_scale,
                history=history,
                checkpoint=next_fidelity_checkpoint,
                wave_start=last_fidelity_wave_end,
                wave_end=completed_after,
                final=False,
            )
            last_fidelity_wave_end = completed_after
            next_fidelity_checkpoint += int(cfg.mcts_fidelity_upgrade_interval)

    if mcts_fidelity_upgrade_evaluator is not None:
        completed_after = _completed_inline_evaluator_count(candidates)
        _run_mcts_fidelity_checkpoint(
            tree=tree,
            candidates=candidates,
            root_candidate=root_candidate,
            state=expansion_state,
            evaluator=mcts_fidelity_upgrade_evaluator,
            cfg=cfg,
            reward_scale=reward_scale,
            history=history,
            checkpoint=completed_after,
            wave_start=last_fidelity_wave_end,
            wave_end=completed_after,
            final=True,
        )

    if node_sweep_enabled:
        sweep_summary = history["node_sweep_summary"]
        incumbent_loss = float(root_selection_loss)
        accepted_total = 0
        for candidate in candidates:
            if bool(candidate.get("duplicate_sequence")):
                continue
            evaluation = candidate.get("inner_structure_evaluation")
            if not isinstance(evaluation, Mapping) or str(evaluation.get("status")) != "ok":
                continue
            move = candidate.get("move") or {}
            node_name = str(move.get("node") or "unknown")
            row = sweep_summary["nodes"].setdefault(
                node_name,
                {
                    "evaluated": 0, "gate_pass": 0, "accepted": 0,
                    "cumulative_improvement": 0.0, "best_step_improvement": 0.0,
                    "operators": {}, "residue_mutations": 0,
                },
            )
            row["evaluated"] += 1
            gate_pass = bool(candidate.get("inner_structure_gate_pass", False))
            row["gate_pass"] += int(gate_pass)
            operator = str(move.get("op") or "unknown")
            row["operators"][operator] = int(row["operators"].get(operator, 0)) + 1
            row["residue_mutations"] += len(move.get("changes", []) or [])
            candidate_loss = float(candidate.get("selection_loss", float("inf")))
            improvement = incumbent_loss - candidate_loss
            accepted = bool(gate_pass and math.isfinite(candidate_loss) and improvement > 0.0)
            if accepted:
                incumbent_loss = candidate_loss
                accepted_total += 1
                row["accepted"] += 1
                row["cumulative_improvement"] += float(improvement)
                row["best_step_improvement"] = max(
                    float(row["best_step_improvement"]), float(improvement)
                )
        completed_for_sweep = _completed_inline_evaluator_count(candidates)
        sweep_summary["completed_candidates"] = int(completed_for_sweep)
        sweep_summary["completed_sweeps"] = int(
            completed_for_sweep / max(1, len(node_sweep_segment_indices))
        )
        sweep_summary["accepted_incumbent_updates"] = int(accepted_total)
        sweep_summary["initial_selection_loss"] = float(root_selection_loss)
        sweep_summary["final_incumbent_selection_loss"] = float(incumbent_loss)
        for row in sweep_summary["nodes"].values():
            row["acceptance_rate"] = float(row["accepted"]) / max(1, int(row["evaluated"]))

    completed_evaluated_candidates = _completed_inline_evaluator_count(candidates)
    candidate_budget_accounting = {
        "schema_version": "astevolve.mcts_candidate_budget_accounting.v1",
        "iteration_unit": str(cfg.mcts_iteration_unit),
        "requested_evaluated_unique_candidates": requested_evaluated_candidates,
        "completed_evaluated_unique_candidates": completed_evaluated_candidates,
        "maximum_expansion_rounds": maximum_expansion_rounds,
        "physical_expansion_rounds": int(history["expansion_rounds"]),
        "underfilled": bool(
            requested_evaluated_candidates is not None
            and completed_evaluated_candidates < requested_evaluated_candidates
        ),
        "completion_definition": (
            "unique_nonduplicate_fast_legal_candidate_with_successful_inline_"
            "structure_result_and_case_evaluator_report"
        ),
        "diagnostics": _inline_evaluator_budget_diagnostics(candidates),
    }
    history["mcts_candidate_budget"] = candidate_budget_accounting
    if (
        candidate_budget_accounting["underfilled"]
        and bool(cfg.mcts_candidate_budget_fail_on_underfill)
    ):
        # Preserve the partial tree before raising.  Without this artifact an
        # underfilled run reports only aggregate candidate counts, hiding the
        # terminal/exhausted parent-action pairs that caused the stall.
        partial_summary = {
            "schema_version": "astevolve.mcts_underfill_diagnostics.v1",
            "status": "candidate_budget_underfilled",
            "mcts_candidate_budget": deepcopy(candidate_budget_accounting),
            "expansion_rounds": int(history["expansion_rounds"]),
            "effective_expansion_rounds": int(
                history["effective_expansion_rounds"]
            ),
            "stalled_expansion_rounds": int(
                history["stalled_expansion_rounds"]
            ),
            "mcts_progressive_widening": deepcopy(
                history.get("mcts_progressive_widening", {})
            ),
            "joint_coverage_completion": deepcopy(joint_completion),
            "mcts_tree_quality": _mcts_tree_quality_report(tree, cfg),
        }
        try:
            partial_paths = _write_inner_loop_artifacts(
                cfg,
                tree,
                candidates,
                partial_summary,
            )
            candidate_budget_accounting["partial_artifact_paths"] = dict(
                partial_paths
            )
        except Exception as artifact_error:
            candidate_budget_accounting["partial_artifact_error"] = (
                f"{type(artifact_error).__name__}:{artifact_error}"
            )
        raise RuntimeError(
            "mcts_evaluated_candidate_budget_underfilled:"
            f"requested={requested_evaluated_candidates}:"
            f"completed={completed_evaluated_candidates}:"
            f"rounds={history['expansion_rounds']}:"
            f"diagnostics={candidate_budget_accounting['diagnostics']}"
        )

    finalize_joint_coverage_completion(joint_completion)
    history["expansion_round_accounting"] = {
        "schema_version": "astevolve.mcts_expansion_round_accounting.v1",
        "requested_rounds": int(cfg.iterations),
        "physical_rounds": int(history["expansion_rounds"]),
        "effective_rounds": int(history["effective_expansion_rounds"]),
        "stalled_rounds": int(history["stalled_expansion_rounds"]),
        "effective_definition": "at_least_one_novel_tree_child_committed",
        "physical_rounds_include_stalls": True,
        "iteration_unit": str(cfg.mcts_iteration_unit),
    }

    best = expansion_state.best_sequences
    best_break = expansion_state.best_breakdown
    best_progen = expansion_state.best_progen
    best_fast = expansion_state.best_fast
    best_node_id = expansion_state.best_node_id

    final_coverage = _required_final_mutation_coverage(best, template_seqs, compiled, cfg)
    if _semantic_coverage_hard_enabled(cfg) and not final_coverage.get("pass", True):
        eligible = [
            cand
            for cand in candidates
            if not bool(cand.get("duplicate_sequence"))
            and (cand.get("semantic_final_coverage", {}) or {}).get("pass", False)
            and (cand.get("fast_filter", {}) or {}).get("pass", True)
            and cand.get("inner_structure_gate_pass") is not False
        ]
        if eligible:
            selected = min(eligible, key=lambda cand: float(cand.get("selection_loss", cand.get("fast_loss", 1e18))))
            best = selected["seqs"]
            best_fast = float(selected.get("fast_loss", best_fast))
            best_break = {"total": float(selected.get("constraint_penalty", 0.0)), "semantic_forced_final_selection": True}
            best_progen = {
                "loglik_sum": float(selected.get("progen_loglik_sum", 0.0)),
                "loglik_avg": float(selected.get("progen_loglik_avg", 0.0)),
            }
            best_node_id = str(selected.get("variant_id", best_node_id))
            history["semantic_forced_final_selection"] = {
                "selected_variant_id": best_node_id,
                "reason": "hard semantic coverage requires final sequence to include all required-node mutations",
                "coverage": selected.get("semantic_final_coverage", {}),
                "fast_loss": best_fast,
            }
        else:
            history["semantic_forced_final_selection"] = {
                "selected_variant_id": None,
                "reason": "no candidate satisfied hard semantic required-node final coverage",
                "coverage": final_coverage,
            }

    round_summary = _summarize_mcts_round(tree, candidates, best_node_id, float(root_fast))
    tree_quality = _mcts_tree_quality_report(tree, cfg)
    history["mcts_tree_quality"] = tree_quality
    round_summary["mcts_tree_quality"] = tree_quality
    _attach_proposal_tier_accounting(round_summary, history)
    round_summary["semantic_coverage"] = _semantic_coverage_report(cfg, history)
    round_summary["node_sweep_summary"] = deepcopy(
        history.get("node_sweep_summary", {})
    )
    attach_joint_coverage_reporting(
        round_summary,
        joint_completion,
        history["expansion_round_accounting"],
    )
    best_mutated = round_summary.get("best_mutated_candidate")
    artifact_paths = _write_inner_loop_artifacts(cfg, tree, candidates, round_summary)
    search_artifacts = {
        "method": "mcts",
        "best_node_id": best_node_id,
        "best_path": _mcts_best_path(tree, best_node_id),
        "best_final_is_root": bool(best_node_id == "root"),
        "best_mutated_candidate": best_mutated,
        "artifact_paths": artifact_paths,
        "round_summary": round_summary,
        "semantic_coverage": round_summary.get("semantic_coverage"),
        "joint_coverage_completion": dict(joint_completion),
        "expansion_round_accounting": dict(history["expansion_round_accounting"]),
        "mcts_candidate_budget": dict(candidate_budget_accounting),
        "mcts_tree_quality": dict(tree_quality),
        "node_sweep_summary": deepcopy(history.get("node_sweep_summary", {})),
        "root_candidate": root_candidate,
        "mcts_prebuilt_exact": dict(history.get("mcts_prebuilt_exact", {})),
        "portfolio_materialization_accounting": dict(
            portfolio_materialization_accounting or {}
        ),
        "portfolio_seed_refinement": dict(
            history.get("portfolio_seed_refinement", {})
        ),
        "candidate_wave_free_slot_schedule": dict(
            history.get("candidate_wave_free_slot_schedule", {})
        ),
        "inner_run_memory": run_memory.cache_summary() if run_memory is not None else None,
        "mcts_fidelity_upgrade": deepcopy(
            history.get("mcts_fidelity_upgrade", {})
        ),
    }
    if bool(tree_quality["required"]) and not bool(tree_quality["pass"]):
        raise RuntimeError(
            "mcts_tree_quality_failed:"
            + ",".join(str(item) for item in tree_quality["failures"])
            + f":artifact={artifact_paths.get('normalized_search', '')}"
        )
    return best, best_break, best_progen, best_fast, history, candidates, search_artifacts


def _fixed_prescreen_cohort_for_screen(
    candidates: Sequence[Mapping[str, Any]], *, max_candidates: int
) -> List[Dict[str, Any]]:


    return sorted(
        [dict(item) for item in candidates],
        key=lambda item: (
            float(
                item.get(
                    "screen_combined_energy",
                    item.get("combined_energy", float("inf")),
                )
            ),
            str(item.get("seq_hash") or ""),
        ),
    )[: max(0, int(max_candidates))]


def _periodic_progen_cohort_for_screen(
    candidates: Sequence[Mapping[str, Any]], *, batch_size: int
) -> List[Dict[str, Any]]:

    size = int(batch_size)
    if size <= 0:
        return []
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        if bool(item.get("duplicate_sequence")):
            continue
        identity = str(item.get("seq_hash") or item.get("variant_id") or "")
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        unique.append(dict(item))
    selected: List[Dict[str, Any]] = []
    for start in range(0, len(unique), size):
        batch = unique[start : start + size]
        if not batch:
            continue
        selected.append(
            min(
                batch,
                key=lambda item: (
                    float(item.get("fast_loss", float("inf"))),
                    str(item.get("seq_hash") or ""),
                ),
            )
        )
    return selected


def optimize_multichain(
    compiled: Dict[str, Any],
    constraint_specs: list[dict],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]] = None,
    fixed_residues: Optional[Dict[str, Dict[int, str]]] = None,
    internal_memory: Optional[Dict[str, Any]] = None,
    run_memory: Optional[InnerRunMemory] = None,
    score_config: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    causal_context: Optional[Mapping[str, Any]] = None,
    prebuilt_proposals: Optional[Sequence[PrebuiltProposal]] = None,
    portfolio_materialization_accounting: Optional[Mapping[str, Any]] = None,
    portfolio_seed_refinement_directives: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    candidate_wave_free_slot_directives: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    candidate_wave_request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    _require_mapping_search_compatibility(compiled, cfg)
    _candidate_wave_protocol_preflight(cfg)
    resolved_candidate_wave: Optional[Dict[str, Any]] = None
    if bool(getattr(cfg, "candidate_wave_enabled", False)):
        if candidate_wave_request is None:
            raise CandidateWaveError("candidate_wave_request_missing")
        resolved_candidate_wave = validate_candidate_wave_request(
            candidate_wave_request
        )
    elif candidate_wave_request is not None or candidate_wave_free_slot_directives:
        raise CandidateWaveError("candidate_wave_disabled_but_payload_present")

    rng = np.random.default_rng(cfg.seed)
    if prebuilt_proposals is None and str(
        getattr(cfg, "sequence_bootstrap_callable", "") or ""
    ).strip():
        module_name, function_name = str(cfg.sequence_bootstrap_callable).split(
            ":", 1
        )
        bootstrap_function = getattr(
            importlib.import_module(module_name), function_name
        )
        if not callable(bootstrap_function):
            raise TypeError("sequence bootstrap callable is not callable")
        generated_bootstrap = bootstrap_function(
            root_sequences=dict(template_seqs or {}),
            config=dict(cfg.sequence_bootstrap_config or {}),
            count=int(cfg.node_optimizer_candidate_count),
            seed=int(cfg.seed or 0),
        )
        if not isinstance(generated_bootstrap, list) or not generated_bootstrap:
            raise ValueError("sequence bootstrap callable returned no proposals")
        prebuilt_proposals = generated_bootstrap
    terms_fast = build_terms_from_specs(constraint_specs, stage="fast")
    terms_chai = build_terms_from_specs(constraint_specs, stage="chai")
    inner_structure_evaluator = _build_inner_structure_evaluator(
        compiled=compiled, cfg=cfg, masks=masks,
        template_seqs=template_seqs, fixed_residues=fixed_residues,
        terms_chai=terms_chai, run_memory=run_memory,
        score_config=score_config, design_state=design_state,
    )
    mcts_fidelity_upgrade_evaluator = _build_mcts_fidelity_upgrade_evaluator(
        compiled=compiled,
        cfg=cfg,
        masks=masks,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
        terms_chai=terms_chai,
        run_memory=run_memory,
        score_config=score_config,
        design_state=design_state,
    )

    search_artifacts: Dict[str, Any] = {}
    if str(cfg.search_method).lower() == "mcts":
        best, best_break, best_progen, best_fast, history, candidates, search_artifacts = _run_mcts_search(
            compiled=compiled,
            terms_fast=terms_fast,
            cfg=cfg,
            masks=masks,
            rng=rng,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            internal_memory=internal_memory,
            run_memory=run_memory,
            causal_context=causal_context,
            prebuilt_proposals=prebuilt_proposals,
            portfolio_materialization_accounting=(
                portfolio_materialization_accounting
            ),
            portfolio_seed_refinement_directives=(
                portfolio_seed_refinement_directives
            ),
            candidate_wave_free_slot_directives=(
                candidate_wave_free_slot_directives
            ),
            inner_structure_evaluator=inner_structure_evaluator,
            mcts_fidelity_upgrade_evaluator=mcts_fidelity_upgrade_evaluator,
        )
        fast_selected_variant_id = str(search_artifacts.get("best_node_id") or "root")
    else:
        (
            best,
            best_break,
            best_progen,
            best_fast,
            history,
            candidates,
            search_artifacts,
            fast_selected_variant_id,
        ) = _run_sa_search(
            compiled=compiled,
            terms_fast=terms_fast,
            cfg=cfg,
            masks=masks,
            rng=rng,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            run_memory=run_memory,
            causal_context=causal_context,
            score_fast=_score_fast_with_run_memory,
            inner_structure_evaluator=inner_structure_evaluator,
        )


    if (
        inner_structure_evaluator is not None
        and str(getattr(cfg, "inner_structure_model", "") or "").strip().lower()
        in {"esmfold", "esmfold_v1", "classic_esmfold"}
    ):
        from astevolve.providers.esmfold import release_esmfold_resources

        release_esmfold_resources()
        search_artifacts["inline_provider_resource_release"] = {
            "schema_version": "astevolve.provider_resource_release.v1",
            "provider": "esmfold",
            "released": True,
            "boundary": "inner_search_to_structure_prescreen",
        }

    inline_rows = [search_artifacts.get("root_candidate") or {}] + list(candidates)
    inline_winner_structure_result: Dict[str, Any] = {}
    if bool(getattr(cfg, "promote_inline_winner_structure_evidence", False)):
        selected_variant_id = str(
            search_artifacts.get("best_node_id") or fast_selected_variant_id or "root"
        )
        selected_inline_row: Mapping[str, Any] = (
            inline_rows[0] if selected_variant_id == "root" else {}
        )
        if selected_variant_id != "root":
            selected_inline_row = next(
                (
                    row
                    for row in candidates
                    if str(row.get("variant_id") or "") == selected_variant_id
                ),
                {},
            )
        selected_evaluation = selected_inline_row.get(
            "inner_structure_evaluation"
        )
        if isinstance(selected_evaluation, Mapping) and isinstance(
            selected_evaluation.get("result"), Mapping
        ):
            inline_winner_structure_result = deepcopy(
                dict(selected_evaluation["result"])
            )
            search_artifacts["inline_winner_structure_promotion"] = {
                "schema_version": "astevolve.inline_winner_structure_promotion.v1",
                "enabled": True,
                "selected_variant_id": selected_variant_id,
                "provider": selected_evaluation.get("provider"),
                "status": selected_evaluation.get("status"),
                "gate_pass": selected_evaluation.get("gate_pass"),
                "seq_hash": inline_winner_structure_result.get("seq_hash"),
            }
    inline_evaluations = [
        row.get("inner_structure_evaluation") for row in inline_rows
        if isinstance(row, Mapping) and isinstance(row.get("inner_structure_evaluation"), Mapping)
    ]
    search_artifacts["inner_structure_summary"] = {
        "schema_version": "astevolve.inner_structure_summary.v1",
        "enabled": inner_structure_evaluator is not None,
        "provider": str(getattr(cfg, "inner_structure_model", "esmfold2")),
        "candidate_count": len(candidates),
        "evaluated_count": len(inline_evaluations),
        "gate_pass_count": sum(bool(item.get("gate_pass", False)) for item in inline_evaluations),
        "gate_fail_count": sum(not bool(item.get("gate_pass", False)) for item in inline_evaluations),
        "failure_count": sum(str(item.get("status")) == "failed" for item in inline_evaluations),
        "selection_objective": "fast_plus_weighted_structure_delta",
    }
    history["inner_structure_summary"] = deepcopy(search_artifacts["inner_structure_summary"])

    root_energy = float(
        (search_artifacts.get("root_candidate") or {}).get("fast_loss", best_fast)
    )
    root_gate_pass = (
        (search_artifacts.get("root_candidate") or {}).get(
            "inner_structure_gate_pass"
        ) is not False
    )
    energy_trace = best_so_far_trace(
        candidates, root_energy=root_energy, root_gate_pass=root_gate_pass
    )
    selection_trace = None
    if inner_structure_evaluator is not None:
        root_selection_energy = float(
            (search_artifacts.get("root_candidate") or {}).get(
                "selection_loss", root_energy
            )
        )
        selection_trace = best_so_far_trace(
            candidates,
            root_energy=root_selection_energy,
            root_gate_pass=root_gate_pass,
            energy_field="selection_loss",
        )
    history["energy_schema_version"] = SEARCH_ENERGY_SCHEMA_VERSION
    history["energy_direction"] = "minimize"
    history["energy_trace"] = energy_trace
    if selection_trace is not None:
        history["selection_trace"] = selection_trace
    search_artifacts["energy_schema_version"] = SEARCH_ENERGY_SCHEMA_VERSION
    effective_island_directive = getattr(cfg, "effective_island_directive", None)
    if isinstance(effective_island_directive, Mapping):
        search_artifacts["executable_island_directive"] = deepcopy(
            dict(effective_island_directive)
        )
    search_artifacts["energy_direction"] = "minimize"
    search_artifacts["energy_trace"] = energy_trace
    if selection_trace is not None:
        search_artifacts["selection_trace"] = selection_trace
    frozen_candidate_wave: Optional[Dict[str, Any]] = None
    frozen_wave_candidates: List[Dict[str, Any]] = []
    if resolved_candidate_wave is not None:
        frozen_candidate_wave = freeze_candidate_wave(
            resolved_candidate_wave, candidates
        )
        frozen_wave_candidates = _attach_frozen_candidate_wave(
            candidates, frozen_candidate_wave
        )
        pre_provider_ledger = build_node_engagement_ledger(
            frozen_candidate_wave,
            candidates,
            resolved_candidate_wave["compiled_design_action"],
            {"node_changes": resolved_candidate_wave["node_changes"]},
            generated_minimum=int(
                cfg.candidate_wave_changed_node_min_generated_unique
            ),
            frozen_wave_minimum=int(
                cfg.candidate_wave_changed_node_min_frozen_unique
            ),
            protenix_attempt_minimum=int(
                cfg.candidate_wave_changed_node_min_protenix_attempts
            ),
        )
        if not (
            pre_provider_ledger.get("all_changed_nodes_generation_engaged")
            and pre_provider_ledger.get("all_changed_nodes_frozen_wave_engaged")
        ):
            raise CandidateWaveError(
                "candidate_wave_changed_node_engagement_underfilled"
            )
        search_artifacts["compiled_candidate_wave_request"] = deepcopy(
            resolved_candidate_wave
        )
        search_artifacts["frozen_candidate_wave"] = deepcopy(
            frozen_candidate_wave
        )
        search_artifacts["node_engagement_pre_provider"] = pre_provider_ledger
    search_artifacts["candidate_audit"] = [
        _candidate_audit_record(
            candidate,
            template_seqs=template_seqs,
            causal_context=causal_context,
        )
        for candidate in candidates
    ]

    raw_parent_candidate = search_artifacts.get("root_candidate")
    if isinstance(raw_parent_candidate, Mapping):
        parent_candidate = dict(raw_parent_candidate)
    else:


        parent_candidate = _build_parent_baseline_candidate(
            best,
            best_break,
            best_progen,
            best_fast,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            compiled=compiled,
            cfg=cfg,
        )
        search_artifacts["root_candidate"] = parent_candidate


    chai_results: List[Dict[str, Any]] = []
    selection_results: List[Dict[str, Any]] = []
    structure_shortlist_health: Dict[str, Dict[str, Any]] = {}
    structure_failure_context = build_structure_failure_context(cfg, score_config)
    structure_semantic_nodes = list(cfg.semantic_required_nodes)
    if cfg.structure_shortlist_policy == "formal_layered_novel":
        declared_active = list(getattr(cfg, "semantic_active_nodes", []) or [])
        if declared_active:
            structure_semantic_nodes = declared_active
    best_plddt: Optional[float] = None
    if cfg.chai1_enabled:
        structure_evaluator_kwargs = {
            "run_memory": run_memory,
            "compiled": compiled,
            "cfg": cfg,
            "masks": masks,
            "template_seqs": template_seqs,
            "fixed_residues": fixed_residues,
            "terms_chai": terms_chai,
            "score_config": score_config,
            "design_state": design_state,
        }
        if resolved_candidate_wave is not None and frozen_candidate_wave is not None:
            structure_evaluator_kwargs.update(
                {
                    "candidate_wave_request": resolved_candidate_wave,
                    "frozen_candidate_wave": frozen_candidate_wave,
                }
            )
        prescreen_results: List[Dict[str, Any]] = []
        prescreen_physical_results: List[Dict[str, Any]] = []
        prescreen_parent_result = parent_candidate
        prescreen_mutant_results: List[Dict[str, Any]] = []
        if bool(cfg.structure_prescreen_enabled):
            prescreen_pool = (
                list(frozen_wave_candidates)
                if frozen_candidate_wave is not None
                else _semantic_prefilter_structure_candidates(
                    candidates,
                    template_seqs,
                    compiled,
                    cfg,
                )
            )
            if inner_structure_evaluator is not None:
                prescreen_pool = [
                    candidate
                    for candidate in prescreen_pool
                    if candidate.get("inner_structure_gate_pass") is not False
                ]
            prescreen_mutant_top = _select_structure_candidates(
                prescreen_pool,
                top_frac=cfg.structure_prescreen_top_frac,
                min_candidates=cfg.structure_prescreen_min_candidates,
                max_candidates=cfg.structure_prescreen_max_candidates,
                all_candidates=False,
                semantic_required_nodes=structure_semantic_nodes,
                semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                shortlist_policy=cfg.structure_shortlist_policy,
                reference_candidate=parent_candidate,
                ranking_source=(
                    "selection_loss"
                    if inner_structure_evaluator is not None
                    else "fast_loss"
                ),
                failure_memory=internal_memory,
                failure_context=structure_failure_context,
                suppression_audit=(prescreen_suppression_audit := {}),
                position_distribution_engagement_quota=int(
                    cfg.structure_position_distribution_engagement_quota
                ),
                portfolio_contract_quota=int(
                    cfg.structure_portfolio_contract_quota
                ),
            )
            structure_shortlist_health["prescreen"] = (
                _summarize_structure_shortlist(
                    prescreen_mutant_top,
                    stage="prescreen",
                    semantic_required_nodes=structure_semantic_nodes,
                    semantic_anchor_nodes=getattr(
                        cfg, "semantic_anchor_nodes", []
                    ),
                    shortlist_policy=cfg.structure_shortlist_policy,
                    selection_audit=prescreen_suppression_audit,
                )
            )
            prescreen_top = _include_parent_baseline(
                prescreen_mutant_top,
                parent_candidate,
            )
            (
                prescreen_results,
                prescreen_physical_results,
                prescreen_funnel_manifest,
            ) = evaluate_multiseed_stage(
                prescreen_top,
                evaluator=_evaluate_structure_with_run_memory,
                evaluator_kwargs=structure_evaluator_kwargs,
                provider=cfg.structure_prescreen_model,
                stage="prescreen",
                cfg=cfg,
                batch_size=cfg.structure_batch_size,
                max_workers=cfg.structure_parallel_workers,
            )
            chai_results.extend(prescreen_physical_results)
            search_artifacts.setdefault(
                "provider_funnel_manifests", []
            ).append(prescreen_funnel_manifest)
            search_artifacts.setdefault("provider_run_receipts", []).extend(
                deepcopy(item["provider_run_receipt"])
                for item in prescreen_physical_results
                if isinstance(item.get("provider_run_receipt"), Mapping)
            )
            search_artifacts.setdefault("provider_seed_aggregates", []).extend(
                deepcopy(item["provider_seed_aggregate"])
                for item in prescreen_results
                if isinstance(item.get("provider_seed_aggregate"), Mapping)
            )
            prescreen_parent_result = next(
                item for item in prescreen_results if _is_parent_baseline(item)
            )
            prescreen_mutant_results = [
                item
                for item in prescreen_results
                if not _is_parent_baseline(item)
            ]
            selection_results = list(prescreen_results)
        if cfg.structure_screen_enabled:
            if frozen_candidate_wave is not None:
                structure_candidate_pool = (
                    list(prescreen_mutant_results)
                    if prescreen_results
                    else list(frozen_wave_candidates)
                )
                formal_screen_mutant_top = list(structure_candidate_pool)
                screen_diagnostics = []
                screen_suppression_audit = {
                    "selection_mode": "frozen_candidate_wave",
                    "candidate_wave_request_hash": frozen_candidate_wave[
                        "candidate_wave_request_hash"
                    ],
                    "frozen_candidate_wave_hash": frozen_candidate_wave[
                        "frozen_candidate_wave_hash"
                    ],
                    "requested_mutants": int(
                        cfg.candidate_wave_protenix_mutant_quota
                    ),
                    "selected_mutants": len(formal_screen_mutant_top),
                }
            else:
                structure_candidate_pool = (
                    list(prescreen_mutant_results)
                    if prescreen_results
                    else _semantic_prefilter_structure_candidates(
                        candidates,
                        template_seqs,
                        compiled,
                        cfg,
                    )
                )
                if (
                    prescreen_results
                    and bool(cfg.structure_prescreen_forward_all_to_screen)
                ):


                    formal_screen_mutant_top = _fixed_prescreen_cohort_for_screen(
                        structure_candidate_pool,
                        max_candidates=cfg.structure_screen_max_candidates,
                    )
                    screen_diagnostics = []
                    screen_suppression_audit = {
                        "selection_mode": "prescreen_fixed_cohort_forward",
                        "requested_mutants": int(
                            cfg.structure_prescreen_max_candidates
                        ),
                        "available_mutants": len(structure_candidate_pool),
                        "selected_mutants": len(formal_screen_mutant_top),
                        "underfilled": len(formal_screen_mutant_top)
                        < int(cfg.structure_prescreen_max_candidates),
                    }
                else:
                    formal_screen_mutant_top = None
                if inner_structure_evaluator is not None and not prescreen_results:
                    structure_candidate_pool = [
                        candidate for candidate in structure_candidate_pool
                        if candidate.get("inner_structure_gate_pass") is not False
                    ]
                periodic_batch = int(
                    getattr(cfg, "structure_screen_progen_batch_size", 0)
                )
                if formal_screen_mutant_top is None and periodic_batch > 0:
                    formal_screen_mutant_top = _periodic_progen_cohort_for_screen(
                        structure_candidate_pool, batch_size=periodic_batch
                    )
                    screen_diagnostics = []
                    screen_suppression_audit = {
                        "selection_mode": "periodic_progen_batch_winners",
                        "progen_batch_size": periodic_batch,
                        "progen_candidates": len(structure_candidate_pool),
                        "selected_mutants": len(formal_screen_mutant_top),
                        "underfilled": len(formal_screen_mutant_top)
                        < max(1, len(structure_candidate_pool) // periodic_batch),
                    }
                if formal_screen_mutant_top is None:
                    formal_screen_mutant_top = _select_structure_candidates(
                        structure_candidate_pool,
                        top_frac=cfg.structure_screen_top_frac,
                        min_candidates=cfg.structure_screen_min_candidates,
                        max_candidates=cfg.structure_screen_max_candidates,
                        all_candidates=cfg.structure_screen_all_candidates,
                        semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                        shortlist_policy=cfg.structure_shortlist_policy,
                        reference_candidate=prescreen_parent_result,
                        ranking_source=(
                            "screen_combined_energy"
                            if prescreen_results
                            else (
                                "selection_loss"
                                if inner_structure_evaluator is not None
                                else "fast_loss"
                            )
                        ),
                        failure_memory=internal_memory,
                        failure_context=structure_failure_context,
                        suppression_audit=(screen_suppression_audit := {}),
                        position_distribution_engagement_quota=int(
                            cfg.structure_position_distribution_engagement_quota
                        ),
                        portfolio_contract_quota=int(
                            cfg.structure_portfolio_contract_quota
                        ),
                    )
                    screen_diagnostics = _select_single_node_structure_diagnostics(
                        structure_candidate_pool,
                        quota=(
                            cfg.structure_screen_single_node_diagnostic_quota
                            if cfg.structure_shortlist_policy == "formal_joint_novel"
                            else 0
                        ),
                        semantic_required_nodes=structure_semantic_nodes,
                        reference_candidate=parent_candidate,
                        excluded_candidates=formal_screen_mutant_top,
                    )
            screen_mutant_top = formal_screen_mutant_top + screen_diagnostics
            structure_shortlist_health["screen"] = _summarize_structure_shortlist(
                screen_mutant_top,
                stage="screen",
                semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                shortlist_policy=cfg.structure_shortlist_policy,
                selection_audit=screen_suppression_audit,
            )
            structure_shortlist_health["screen"]["cross_round_failure_suppression"] = screen_suppression_audit
            screen_top = _include_parent_baseline(
                screen_mutant_top,
                prescreen_parent_result,
            )
            screen_results, screen_physical_results, screen_funnel_manifest = evaluate_multiseed_stage(
                screen_top,
                evaluator=_evaluate_structure_with_run_memory,
                evaluator_kwargs=structure_evaluator_kwargs,
                provider=cfg.structure_screen_model,
                stage="screen",
                cfg=cfg,
                batch_size=cfg.structure_batch_size,
                max_workers=cfg.structure_parallel_workers,
            )
            chai_results.extend(screen_physical_results)
            search_artifacts.setdefault("provider_funnel_manifests", []).append(
                screen_funnel_manifest
            )
            search_artifacts.setdefault("provider_run_receipts", []).extend(
                deepcopy(item["provider_run_receipt"])
                for item in screen_physical_results
                if isinstance(item.get("provider_run_receipt"), Mapping)
            )
            search_artifacts.setdefault("provider_seed_aggregates", []).extend(
                deepcopy(item["provider_seed_aggregate"])
                for item in screen_results
                if isinstance(item.get("provider_seed_aggregate"), Mapping)
            )
            selection_results = [
                item
                for item in screen_results
                if item.get("structure_shortlist_role")
                != "single_node_diagnostic"
            ]

            if cfg.structure_rerank_enabled and screen_results:
                parent_screen_result = next(
                    item for item in screen_results if _is_parent_baseline(item)
                )
                screen_mutant_results = [
                    item
                    for item in screen_results
                    if not _is_parent_baseline(item)
                    and (
                        frozen_candidate_wave is not None
                        or bool(item.get("formal_rerank_eligible", True))
                    )
                ]
                if frozen_candidate_wave is not None:
                    (
                        rerank_mutant_top,
                        candidate_wave_rerank_receipt,
                        candidate_wave_rerank_diagnostics,
                    ) = _select_candidate_wave_af3_subset(
                        screen_mutant_results,
                        frozen_candidate_wave,
                    )
                    rerank_suppression_audit = {
                        "selection_mode": "frozen_candidate_wave",
                        "requested_mutants": int(
                            cfg.candidate_wave_af3_mutant_quota
                        ),
                        "selected_mutants": len(rerank_mutant_top),
                        "rerank_selection_receipt_hash": (
                            candidate_wave_rerank_receipt[
                                "rerank_selection_receipt_hash"
                            ]
                        ),
                    }
                    structure_evaluator_kwargs[
                        "candidate_wave_rerank_selection_receipt"
                    ] = candidate_wave_rerank_receipt
                    search_artifacts[
                        "candidate_wave_rerank_selection"
                    ] = deepcopy(candidate_wave_rerank_receipt)
                    search_artifacts[
                        "candidate_wave_rerank_selection_diagnostics"
                    ] = deepcopy(candidate_wave_rerank_diagnostics)
                else:
                    rerank_mutant_top = _select_structure_candidates(
                        screen_mutant_results,
                        top_frac=cfg.structure_rerank_top_frac,
                        min_candidates=cfg.structure_rerank_min_candidates,
                        max_candidates=cfg.structure_rerank_max_candidates,
                        all_candidates=False,
                        semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                        shortlist_policy=cfg.structure_shortlist_policy,
                        reference_candidate=parent_screen_result,
                        ranking_source="screen_combined_energy",
                        failure_memory=internal_memory,
                        failure_context=structure_failure_context,
                        suppression_audit=(rerank_suppression_audit := {}),
                        allow_all_infeasible_rescue=bool(
                            cfg.structure_rerank_all_infeasible_rescue
                        ),
                        position_distribution_engagement_quota=int(
                            cfg.structure_position_distribution_engagement_quota
                        ),
                        portfolio_contract_quota=int(
                            cfg.structure_portfolio_contract_quota
                        ),
                    )
                structure_shortlist_health["rerank"] = _summarize_structure_shortlist(
                    rerank_mutant_top,
                    stage="rerank",
                    semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                    shortlist_policy=cfg.structure_shortlist_policy,
                    selection_audit=rerank_suppression_audit,
                )
                structure_shortlist_health["rerank"]["cross_round_failure_suppression"] = rerank_suppression_audit
                rerank_top = _include_parent_baseline(
                    rerank_mutant_top,
                    parent_screen_result,
                )
                rerank_results, rerank_physical_results, rerank_funnel_manifest = evaluate_multiseed_stage(
                    rerank_top,
                    evaluator=_evaluate_structure_with_run_memory,
                    evaluator_kwargs=structure_evaluator_kwargs,
                    provider=cfg.structure_rerank_model,
                    stage="rerank",
                    cfg=cfg,
                    batch_size=cfg.structure_batch_size,
                    max_workers=cfg.structure_parallel_workers,
                )
                for item in rerank_results:
                    item["screen_parent_seq_hash"] = item.get("seq_hash")
                chai_results.extend(rerank_physical_results)
                search_artifacts.setdefault("provider_funnel_manifests", []).append(
                    rerank_funnel_manifest
                )
                search_artifacts.setdefault("provider_run_receipts", []).extend(
                    deepcopy(item["provider_run_receipt"])
                    for item in rerank_physical_results
                    if isinstance(item.get("provider_run_receipt"), Mapping)
                )
                search_artifacts.setdefault("provider_seed_aggregates", []).extend(
                    deepcopy(item["provider_seed_aggregate"])
                    for item in rerank_results
                    if isinstance(item.get("provider_seed_aggregate"), Mapping)
                )
                screen_by_hash = {
                    str(item.get("seq_hash") or _seqs_hash(item["seqs"])): item
                    for item in screen_results
                }
                disagreement_rows = []
                for item in rerank_results:
                    identity = str(item.get("seq_hash") or _seqs_hash(item["seqs"]))
                    if identity in screen_by_hash:
                        disagreement = build_backend_disagreement(
                            screen_by_hash[identity],
                            item,
                            threshold=float(cfg.structure_disagreement_threshold),
                        )
                        item["backend_disagreement"] = disagreement
                        disagreement_rows.append(disagreement)
                search_artifacts["backend_disagreement"] = {
                    "schema_version": "astevolve.backend_disagreement_manifest.v1",
                    "candidate_count": len(disagreement_rows),
                    "disagreement_count": sum(bool(row.get("disagreed")) for row in disagreement_rows),
                    "candidates": disagreement_rows,
                }

                physics_results: List[Dict[str, Any]] = []
                physics_cap = max(0, int(cfg.structure_physics_max_candidates))
                if physics_cap and rerank_results:
                    physics_shortlist = sorted(
                        [item for item in rerank_results if _has_structure_signal(item)],
                        key=lambda item: (
                            float(item.get("combined_energy", float("inf"))),
                            str(item.get("seq_hash") or ""),
                        ),
                    )[:physics_cap]
                    for finalist in physics_shortlist:
                        identity = str(finalist.get("seq_hash") or _seqs_hash(finalist["seqs"]))
                        paired_rows = [
                            item
                            for item in [*screen_physical_results, *rerank_physical_results]
                            if str(item.get("seq_hash") or _seqs_hash(item["seqs"])) == identity
                        ]
                        with_evidence = dict(finalist)
                        with_evidence["structure_provider_evidence"] = build_structure_provider_evidence(
                            paired_rows,
                            finalist["seqs"],
                            required_providers=(
                                str(cfg.structure_screen_model),
                                str(cfg.structure_rerank_model),
                            ),
                        )
                        physics_results.append(
                            rescore_existing_structure_candidate(
                                with_evidence,
                                compiled=compiled,
                                masks=masks,
                                template_seqs=template_seqs,
                                fixed_residues=fixed_residues,
                                score_config=score_config,
                                design_state=design_state,
                            )
                        )
                    physics_receipts = []
                    for item in physics_results:
                        terms = (item.get("inner_evaluator_report") or {}).get("terms", [])
                        pyrosetta_terms = [
                            term for term in terms
                            if isinstance(term, Mapping)
                            and str(term.get("backend") or "").lower() == "pyrosetta"
                        ]
                        available = any(bool(term.get("available")) for term in pyrosetta_terms)
                        physics_receipts.append(
                            {
                                "schema_version": "astevolve.pyrosetta_finalist_receipt.v1",
                                "candidate_sequence_hash": str(
                                    item.get("seq_hash") or _seqs_hash(item["seqs"])
                                ),
                                "attempted": bool(pyrosetta_terms),
                                "available": available,
                                "term_count": len(pyrosetta_terms),
                            }
                        )
                    if bool(cfg.structure_pyrosetta_required) and (
                        len(physics_receipts) != physics_cap
                        or not all(row["available"] for row in physics_receipts)
                    ):
                        raise RuntimeError(
                            "formal PyRosetta finalist quota was not satisfied"
                        )
                    search_artifacts["pyrosetta_finalist_receipts"] = {
                        "schema_version": "astevolve.pyrosetta_finalist_manifest.v1",
                        "quota": physics_cap,
                        "receipts": physics_receipts,
                        "complete": len(physics_receipts) == physics_cap
                        and all(row["available"] for row in physics_receipts),
                    }
                    chai_results.extend(physics_results)

                formal_rerank_results = physics_results if physics_cap else rerank_results
                selection_results = _merge_structure_selection_pool(
                    screen_results,
                    formal_rerank_results,
                    allow_low_fidelity_fallback=bool(
                        cfg.structure_allow_low_fidelity_fallback
                    ),
                )
                if rerank_results:
                    invalid_rerank_results = [
                        item for item in rerank_results if not _has_structure_signal(item)
                    ]
                    for item in invalid_rerank_results:
                        item.setdefault("warnings", [])
                        if isinstance(item["warnings"], list):
                            item["warnings"].append(
                                "rerank produced no valid structure signal; "
                                + (
                                    "screen evidence is eligible only if no rerank has signal"
                                    if bool(cfg.structure_allow_low_fidelity_fallback)
                                    else "formal selection excludes its screen evidence"
                                )
                            )
        else:
            structure_candidate_pool = _semantic_prefilter_structure_candidates(
                candidates,
                template_seqs,
                compiled,
                cfg,
            )
            if inner_structure_evaluator is not None:
                structure_candidate_pool = [
                    candidate for candidate in structure_candidate_pool
                    if candidate.get("inner_structure_gate_pass") is not False
                ]
            mutant_top = _select_structure_candidates(
                structure_candidate_pool,
                top_frac=cfg.chai1_top_frac,
                min_candidates=cfg.chai1_min_candidates,
                max_candidates=cfg.chai1_max_candidates,
                all_candidates=False,
                semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                shortlist_policy=cfg.structure_shortlist_policy,
                reference_candidate=parent_candidate,
                ranking_source=("selection_loss" if inner_structure_evaluator is not None else "fast_loss"),
                failure_memory=internal_memory,
                failure_context=structure_failure_context,
                suppression_audit=(legacy_suppression_audit := {}),
                position_distribution_engagement_quota=int(
                    cfg.structure_position_distribution_engagement_quota
                ),
                portfolio_contract_quota=int(
                    cfg.structure_portfolio_contract_quota
                ),
            )
            structure_shortlist_health["legacy"] = _summarize_structure_shortlist(
                mutant_top,
                stage="legacy",
                semantic_required_nodes=structure_semantic_nodes, semantic_anchor_nodes=getattr(cfg, "semantic_anchor_nodes", []),
                shortlist_policy=cfg.structure_shortlist_policy,
                selection_audit=legacy_suppression_audit,
            )
            structure_shortlist_health["legacy"]["cross_round_failure_suppression"] = legacy_suppression_audit
            top = _include_parent_baseline(mutant_top, parent_candidate)
            legacy_results, legacy_physical_results, legacy_funnel_manifest = evaluate_multiseed_stage(
                top,
                evaluator=_evaluate_structure_with_run_memory,
                evaluator_kwargs=structure_evaluator_kwargs,
                provider=cfg.structure_model,
                stage="legacy",
                cfg=cfg,
                batch_size=cfg.structure_batch_size,
                max_workers=cfg.structure_parallel_workers,
            )
            chai_results.extend(legacy_physical_results)
            search_artifacts.setdefault("provider_funnel_manifests", []).append(
                legacy_funnel_manifest
            )
            search_artifacts.setdefault("provider_run_receipts", []).extend(
                deepcopy(item["provider_run_receipt"])
                for item in legacy_physical_results
                if isinstance(item.get("provider_run_receipt"), Mapping)
            )
            search_artifacts.setdefault("provider_seed_aggregates", []).extend(
                deepcopy(item["provider_seed_aggregate"])
                for item in legacy_results
                if isinstance(item.get("provider_seed_aggregate"), Mapping)
            )
            selection_results = legacy_results

        for item in chai_results:
            plddt = float(item.get("plddt", 0.0) or 0.0)
            if best_plddt is None or plddt > best_plddt:
                best_plddt = plddt
    if frozen_candidate_wave is not None and resolved_candidate_wave is not None:
        protenix_mutants = [
            item
            for item in chai_results
            if not _is_parent_baseline(item)
            and str(item.get("structure_provider") or "").lower() == "protenix"
        ]
        af3_mutants = [
            item
            for item in chai_results
            if not _is_parent_baseline(item)
            and str(item.get("structure_provider") or "").lower()
            in {"alphafold3", "af3"}
        ]
        protenix_mutant_identities = {
            str(item.get("seq_hash") or _seqs_hash(item["seqs"]))
            for item in protenix_mutants
        }
        af3_mutant_identities = {
            str(item.get("seq_hash") or _seqs_hash(item["seqs"]))
            for item in af3_mutants
        }
        if len(protenix_mutant_identities) != int(
            cfg.candidate_wave_protenix_mutant_quota
        ) or len(af3_mutant_identities) != int(cfg.candidate_wave_af3_mutant_quota):
            raise CandidateWaveError(
                "candidate_wave_provider_quota_mismatch",
                f"protenix={len(protenix_mutant_identities)};af3={len(af3_mutant_identities)}",
            )
        post_provider_ledger = build_node_engagement_ledger(
            frozen_candidate_wave,
            candidates,
            resolved_candidate_wave["compiled_design_action"],
            {"node_changes": resolved_candidate_wave["node_changes"]},
            chai_results,
            generated_minimum=int(
                cfg.candidate_wave_changed_node_min_generated_unique
            ),
            frozen_wave_minimum=int(
                cfg.candidate_wave_changed_node_min_frozen_unique
            ),
            protenix_attempt_minimum=int(
                cfg.candidate_wave_changed_node_min_protenix_attempts
            ),
        )
        if not post_provider_ledger.get("all_changed_nodes_protenix_engaged"):
            raise CandidateWaveError(
                "candidate_wave_changed_node_protenix_engagement_underfilled"
            )
        search_artifacts["node_engagement"] = post_provider_ledger
        search_artifacts["candidate_wave_provider_attempts"] = {
            "schema_version": "astevolve.candidate_wave_provider_attempts.v1",
            "frozen_candidate_wave_hash": frozen_candidate_wave[
                "frozen_candidate_wave_hash"
            ],
            "protenix_mutant_candidates": len(protenix_mutant_identities),
            "alphafold3_mutant_candidates": len(af3_mutant_identities),
            "protenix_mutant_attempts": len(protenix_mutants),
            "alphafold3_mutant_attempts": len(af3_mutants),
            "protenix_parent_attempts": sum(
                _is_parent_baseline(item)
                and str(item.get("structure_provider") or "").lower()
                == "protenix"
                for item in chai_results
            ),
            "alphafold3_parent_attempts": sum(
                _is_parent_baseline(item)
                and str(item.get("structure_provider") or "").lower()
                in {"alphafold3", "af3"}
                for item in chai_results
            ),
            "receipts": [
                {
                    "variant_id": item.get("variant_id"),
                    "candidate_sequence_bundle_hash": item.get(
                        "candidate_sequence_bundle_hash"
                    ),
                    "candidate_wave_slot_id": item.get(
                        "candidate_wave_slot_id"
                    ),
                    "candidate_wave_role": item.get("candidate_wave_role"),
                    "is_parent_baseline": bool(_is_parent_baseline(item)),
                    "provider": item.get("structure_provider"),
                    "stage": item.get("structure_stage"),
                    "structure_model_name": item.get("structure_model_name"),
                    "plddt": item.get("plddt"),
                    "confidence_metrics": deepcopy(
                        item.get("confidence_metrics", {})
                    ),
                    "structure_metrics": deepcopy(
                        item.get("structure_metrics", {})
                    ),
                    "persistent_evaluation_cache": deepcopy(
                        item.get("persistent_evaluation_cache")
                    ),
                    "structure_batch_dispatch": deepcopy(
                        item.get("structure_batch_dispatch")
                    ),
                    "provider_seed": item.get("provider_seed"),
                    "provider_run_receipt": deepcopy(
                        item.get("provider_run_receipt")
                    ),
                }
                for item in chai_results
            ],
        }
    _attach_structure_shortlist_health(
        search_artifacts, structure_shortlist_health, enabled=bool(cfg.chai1_enabled)
    )
    attach_structure_failure_suppression_summary(
        search_artifacts,
        structure_shortlist_health,
        context=structure_failure_context,
        enabled=cfg.structure_shortlist_policy
        in {"formal_joint_novel", "formal_layered_novel"},
    )
    final = best
    final_break = best_break
    final_progen = best_progen
    final_fast = best_fast
    final_plddt = best_plddt
    final_struct = 0.0
    final_combined_energy = float(best_fast)
    final_combined_loss = float(best_fast)
    final_legacy_design_energy = float(best_fast)
    final_outer_energy_objective: Dict[str, Any] = {}
    final_structure_selection_objective = "fast_only"
    final_delta = None
    final_plddt_A = None
    final_plddt_B = None
    final_chain_plddt: Dict[str, float] = {}
    final_node_plddt: Dict[str, Dict[str, Any]] = {}
    final_confidence_metrics: Dict[str, float] = {}
    final_structure_metrics: Dict[str, Any] = {}
    final_protenix_out_dir = None
    final_protenix_summary_json = None
    final_multistate_objectives: Dict[str, Any] = {}
    final_multistate_score = 0.0
    final_multistate_loss = 0.0
    final_inner_evaluator_report: Dict[str, Any] = {}
    structure_selection_decision: Optional[Dict[str, Any]] = None
    chai_best: Dict[str, Any] = {}

    if inline_winner_structure_result:
        inline = inline_winner_structure_result
        final_plddt = inline.get("plddt")
        final_struct = float(inline.get("struct_penalty", 0.0) or 0.0)
        final_combined_energy = float(
            inline.get("combined_energy", final_combined_energy)
        )
        final_combined_loss = float(
            inline.get("combined_loss", final_combined_energy)
        )
        final_legacy_design_energy = float(
            inline.get("legacy_design_energy", final_combined_loss)
        )
        final_outer_energy_objective = dict(
            inline.get("outer_energy_objective", {}) or {}
        )
        final_structure_selection_objective = str(
            inline.get("structure_selection_objective") or "inline_winner"
        )
        final_delta = inline.get("plddt_delta")
        final_plddt_A = inline.get("plddt_A")
        final_plddt_B = inline.get("plddt_B")
        final_confidence_metrics = dict(
            inline.get("confidence_metrics", {}) or {}
        )
        final_chain_plddt = dict(inline.get("chain_plddt", {}) or {})
        final_node_plddt = dict(inline.get("node_plddt", {}) or {})
        final_structure_metrics = dict(
            inline.get("structure_metrics", {}) or {}
        )
        final_protenix_out_dir = inline.get("protenix_out_dir")
        final_protenix_summary_json = inline.get("protenix_summary_json")
        final_multistate_objectives = dict(
            inline.get("multistate_objectives", {}) or {}
        )
        final_multistate_score = float(
            inline.get("multistate_score", 0.0) or 0.0
        )
        final_multistate_loss = float(
            inline.get("multistate_loss", 0.0) or 0.0
        )
        final_inner_evaluator_report = dict(
            inline.get("inner_evaluator_report", {}) or {}
        )

    if selection_results:
        if _semantic_coverage_hard_enabled(cfg):
            for item in selection_results:
                final_coverage = _required_final_mutation_coverage(
                    item.get("seqs", {}),
                    template_seqs,
                    compiled,
                    cfg,
                )
                item["semantic_final_mutation_coverage"] = final_coverage
            if not any(
                bool(
                    (item.get("semantic_final_mutation_coverage") or {}).get(
                        "pass", True
                    )
                )
                for item in selection_results
            ):
                search_artifacts.setdefault("warnings", []).append(
                    "no structure-evaluated candidate satisfied required final semantic-node mutation coverage; final selection will be hard-gated"
                )
        chai_best, structure_selection_decision = _select_final_structure_candidate(
            selection_results,
            stepping_stone_enabled=bool(cfg.structure_stepping_stone_enabled),
            stepping_stone_max_energy_degradation=float(cfg.structure_stepping_stone_max_energy_degradation),
            stepping_stone_metrics=tuple(cfg.structure_stepping_stone_metrics),
            stepping_stone_min_metric_gain=float(cfg.structure_stepping_stone_min_metric_gain),
        )
        search_artifacts["structure_selection_decision"] = structure_selection_decision
        final = chai_best["seqs"]
        final_break = {"total": chai_best["constraint_penalty"]}
        final_progen = {
            "loglik_avg": chai_best["progen_loglik_avg"],
            "loglik_sum": chai_best["progen_loglik_sum"],
        }
        final_fast = chai_best["fast_loss"]
        final_plddt = chai_best["plddt"]
        final_struct = chai_best.get("struct_penalty", 0.0)
        final_combined_energy = chai_best.get(
            "combined_energy", chai_best.get("combined_loss", float(final_fast))
        )
        final_combined_loss = chai_best.get(
            "combined_loss", float(final_combined_energy)
        )
        final_legacy_design_energy = float(
            chai_best.get("legacy_design_energy", final_combined_loss)
        )
        final_outer_energy_objective = dict(
            chai_best.get("outer_energy_objective", {}) or {}
        )
        final_structure_selection_objective = str(
            chai_best.get("structure_selection_objective") or "legacy_additive"
        )
        final_delta = chai_best.get("plddt_delta", None)
        final_plddt_A = chai_best.get("plddt_A", None)
        final_plddt_B = chai_best.get("plddt_B", None)
        final_confidence_metrics = chai_best.get("confidence_metrics", {}) or {}
        final_chain_plddt = chai_best.get("chain_plddt", {}) or {}
        final_node_plddt = chai_best.get("node_plddt", {}) or {}
        final_structure_metrics = chai_best.get("structure_metrics", {}) or {}
        final_protenix_out_dir = chai_best.get("protenix_out_dir")
        final_protenix_summary_json = chai_best.get("protenix_summary_json")
        final_multistate_objectives = chai_best.get("multistate_objectives", {}) or {}
        final_multistate_score = float(chai_best.get("multistate_score", 0.0) or 0.0)
        final_multistate_loss = float(chai_best.get("multistate_loss", 0.0) or 0.0)
        final_inner_evaluator_report = chai_best.get("inner_evaluator_report", {}) or {}

    parent_baseline_comparison = _build_parent_baseline_comparison(
        parent_candidate,
        chai_results,
        selection_results,
        structure_selection_decision,
    )
    search_artifacts["parent_baseline_comparison"] = parent_baseline_comparison
    structure_finalist_feedback = build_structure_finalist_feedback(
        chai_results,
        selected_candidate=(chai_best if selection_results else None),
    )
    structure_finalist_feedback["scientific_context"] = dict(structure_failure_context)
    search_artifacts["structure_finalist_feedback"] = (
        structure_finalist_feedback
    )

    attach_structure_evaluation_summary(
        search_artifacts, evaluated=chai_results, selection_pool=selection_results,
        selected=chai_best, cfg=cfg,
        best_confidence_metrics=final_confidence_metrics,
        best_structure_metrics=final_structure_metrics,
        multistate_objectives=final_multistate_objectives,
        selection_decision=structure_selection_decision,
        parent_baseline_comparison=parent_baseline_comparison,
    )

    semantic_audit = _inner_loop_semantic_audit(final, template_seqs, fixed_residues, compiled, cfg, history=history)
    _enforce_layered_shortlist_semantic_gate(
        semantic_audit, structure_shortlist_health,
        enabled=cfg.structure_shortlist_policy == "formal_layered_novel")
    if search_artifacts is not None:
        search_artifacts["semantic_audit"] = semantic_audit

    structure_required_providers: List[str] = []
    if cfg.chai1_enabled:
        if cfg.structure_screen_enabled:
            structure_required_providers.append(str(cfg.structure_screen_model))
            if cfg.structure_rerank_enabled:
                structure_required_providers.append(str(cfg.structure_rerank_model))
        else:
            structure_required_providers.append(str(cfg.structure_model))

    out = {
        "seqs": final,
        "fast_loss": float(final_fast),
        "constraint_penalty": float(final_break["total"]),
        "progen_loglik_avg": float(final_progen["loglik_avg"]),
        "progen_loglik_sum": float(final_progen["loglik_sum"]),
        "chai_plddt": float(final_plddt) if final_plddt is not None else None,
        "chai_struct_penalty": float(final_struct),


        "chai_combined_loss": float(final_combined_loss),
        "chai_combined_energy": float(final_combined_energy),
        "outer_aligned_energy": float(
            final_outer_energy_objective.get(
                "final_energy", final_combined_energy
            )
        ),
        "outer_energy_objective": final_outer_energy_objective,
        "scientific_final_energy": float(final_combined_energy),
        "legacy_design_energy": float(final_legacy_design_energy),
        "design_energy": float(final_legacy_design_energy),
        "structure_selection_objective": final_structure_selection_objective,
        "energy_schema_version": SEARCH_ENERGY_SCHEMA_VERSION,
        "energy_direction": "minimize",
        "energy": (
            dict(chai_best.get("energy") or {})
            if selection_results and isinstance(chai_best.get("energy"), Mapping)
            else fast_energy_record(
                fast_loss=final_fast,
                constraint_penalty=final_break["total"],
                progen_loglik_avg=final_progen["loglik_avg"],
                progen_weight=cfg.progen_weight,
                hard_gate_pass=bool(semantic_audit.get("hard_gate_pass", True)),
            )
        ),
        "chai_evaluated": len(chai_results),
        "chai_results": _public_chai_results(chai_results, limit=10),
        "structure_provider_evidence": build_structure_provider_evidence(
            chai_results,
            final,
            required_providers=structure_required_providers,
        ),
        "plddt_delta": final_delta,
        "plddt_A": final_plddt_A,
        "plddt_B": final_plddt_B,
        "confidence_metrics": final_confidence_metrics,
        "chain_plddt": final_chain_plddt,
        "node_plddt": final_node_plddt,
        "structure_metrics": final_structure_metrics,
        "protenix_out_dir": final_protenix_out_dir,
        "protenix_summary_json": final_protenix_summary_json,
        "multistate_objectives": final_multistate_objectives,
        "multistate_score": float(final_multistate_score),
        "multistate_loss": float(final_multistate_loss),
        "inner_evaluator_report": final_inner_evaluator_report,
        "structure_selection_decision": structure_selection_decision,
        "parent_baseline_comparison": parent_baseline_comparison,
        "structure_finalist_feedback": structure_finalist_feedback,
        "mutation_history": history,
        "inner_loop_semantic_audit": semantic_audit,
        "segment_scores": compute_segment_scores(final, compiled),
        "search_method": str(cfg.search_method).lower(),
        "search_artifacts": search_artifacts,
        "inner_run_memory": run_memory.cache_summary() if run_memory is not None else None,
    }
    causal_runtime = _build_causal_runtime(
        context=causal_context,
        root=parent_candidate,
        candidates=candidates,
        structure_selected=chai_best if selection_results else None,
        fast_selected_variant_id=str(fast_selected_variant_id),
    )
    if causal_runtime is not None:
        out["_causal_runtime"] = causal_runtime
    if str(os.environ.get("ASTEVOLVE_RELEASE_MODEL_CACHE_AFTER_EVAL", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        try:
            from astevolve.providers.esmfold2 import clear_esmfold2_model_cache

            clear_esmfold2_model_cache(clear_confidence_cache=False)
        except Exception:
            pass
        try:
            from astevolve.providers.progen import clear_progen_model_cache

            clear_progen_model_cache(clear_score_cache=False)
        except Exception:
            pass
    return out



from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
import hashlib
import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from astevolve.domain.compiled import (
    CompiledDesignAction,
    CompiledPortfolioOptimizationRequest,
)
from astevolve.search.portfolio_runtime import (
    PortfolioRuntimeError,
    validate_candidate_portfolio_realization,
)
from astevolve.search.candidate_wave_runtime import (
    CandidateWaveError,
    validate_candidate_wave_request,
    validate_frozen_candidate_wave,
    validate_rerank_selection_receipt,
)
from astevolve.search.mapping_schedule_runtime import (
    PortfolioMappingComponentError,
    validate_portfolio_mapping_components,
)
from engine.design_action_compiler import (
    DesignActionCompileError,
    validate_candidate_against_compiled_action,
)
from engine.experiment_identity import SequenceBundleIdentity


STRUCTURE_BATCH_DISPATCH_VERSION = "astevolve.structure_batch_dispatch.v1"
STRUCTURE_DESIGN_ACTION_GATE_VERSION = (
    "astevolve.structure_design_action_gate.v1"
)
MAPPING_COMPONENT_DISPATCH_VALIDATION_VERSION = (
    "astevolve.mapping_component_dispatch_validation.v1"
)
CANDIDATE_WAVE_BATCH_DISPATCH_VERSION = (
    "astevolve.candidate_wave_batch_dispatch.v1"
)
CANDIDATE_WAVE_CANDIDATE_DISPATCH_VERSION = (
    "astevolve.candidate_wave_candidate_dispatch.v1"
)
StructureEvaluator = Callable[..., Dict[str, Any]]


class StructureDispatchValidationError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _dispatch_fail(code: str, detail: str = "") -> None:
    raise StructureDispatchValidationError(code, detail)


_DESIGN_ACTION_PARENT_BINDING_FIELDS = (
    "case_id",
    "parent_program_id",
    "parent_candidate_id",
    "parent_sequence_bundle_hash",
    "parent_effective_contract_hash",
    "parent_evolve_hash",
)


def _strict_compiled_design_action(
    evaluator_kwargs: Mapping[str, Any],
) -> CompiledDesignAction | None:


    compiled = evaluator_kwargs.get("compiled")
    if not isinstance(compiled, Mapping):
        return None
    raw_action = compiled.get("compiled_design_action")
    if raw_action is None:
        return None
    if not isinstance(raw_action, Mapping):
        _dispatch_fail("compiled_design_action_artifact_invalid", "mapping required")
    try:
        return CompiledDesignAction.from_artifact(raw_action)
    except Exception as error:
        _dispatch_fail("compiled_design_action_artifact_invalid", str(error))


def _strict_compiled_portfolio_request(
    evaluator_kwargs: Mapping[str, Any],
) -> CompiledPortfolioOptimizationRequest | None:
    compiled = evaluator_kwargs.get("compiled")
    if not isinstance(compiled, Mapping):
        return None
    raw_request = compiled.get("compiled_portfolio_optimization_request")
    if raw_request is None:
        return None
    if not isinstance(raw_request, Mapping):
        _dispatch_fail("compiled_portfolio_request_artifact_invalid")
    try:
        return CompiledPortfolioOptimizationRequest.from_artifact(raw_request)
    except Exception as error:
        _dispatch_fail("compiled_portfolio_request_artifact_invalid", str(error))


def _validate_portfolio_action_binding(
    artifact: CompiledDesignAction,
    request: CompiledPortfolioOptimizationRequest,
) -> None:


    shared_fields = (
        "design_action_hash",
        "compiled_design_action_hash",
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "immutable_sequence_bundle_hash",
        "reconciled_root_sequence_bundle_hash",
    )
    for field in shared_fields:
        observed = str(getattr(request, field, "") or "")
        expected = str(getattr(artifact, field, "") or "")
        if observed != expected:
            _dispatch_fail(
                "compiled_portfolio_action_binding_mismatch",
                f"{field}:{observed}/{expected}",
            )

    request_root = {
        str(chain): str(sequence)
        for chain, sequence in request.reconciled_root_sequences.items()
    }
    action_root = {
        str(chain): str(sequence)
        for chain, sequence in artifact.reconciled_root_sequences.items()
    }
    if request_root != action_root:
        _dispatch_fail("compiled_portfolio_root_sequence_mismatch")

    active_owners = {
        (str(chain), int(position)): str(owner)
        for chain, entries in artifact.active_position_owners.items()
        for position, owner in entries.items()
    }

    def require_owner(change: Any, *, source: str) -> None:
        key = (str(change.chain_id), int(change.position))
        expected_owner = active_owners.get(key)
        if expected_owner is None:
            _dispatch_fail(
                "compiled_portfolio_position_outside_active_selector",
                f"{source}:{key[0]}:{key[1]}",
            )
        observed_owner = str(change.owner_node_id or "")
        if observed_owner != expected_owner:
            _dispatch_fail(
                "compiled_portfolio_position_owner_mismatch",
                f"{source}:{key[0]}:{key[1]}:{observed_owner}/{expected_owner}",
            )

    def validate_exact_sequences(
        sequences: Mapping[str, str], *, source: str
    ) -> None:
        try:
            validate_candidate_against_compiled_action(
                artifact,
                sequences,
                artifact.compiled_design_action_hash,
            )
        except DesignActionCompileError as error:
            _dispatch_fail(
                "compiled_portfolio_sequence_contract_violation",
                f"{source}:{error.code}:{error.detail}",
            )

    for module in request.mutation_modules:
        for change in module.mutations:
            require_owner(change, source=f"module:{module.module_id}")
        if module.operation != "matched_module_ablation":
            module_sequences = {
                chain: list(sequence) for chain, sequence in request_root.items()
            }
            for change in module.mutations:
                module_sequences[change.chain_id][change.position] = (
                    change.to_residue
                )
            validate_exact_sequences(
                {
                    chain: "".join(residues)
                    for chain, residues in sorted(module_sequences.items())
                },
                source=f"module:{module.module_id}",
            )

    for seed in request.sequence_seeds:
        for change in seed.root_delta.changes:
            require_owner(change, source=f"seed:{seed.seed_id}:delta")
        for chain, position, owner in seed.refinement_positions:
            key = (str(chain), int(position))
            expected_owner = active_owners.get(key)
            if expected_owner is None:
                _dispatch_fail(
                    "compiled_portfolio_position_outside_active_selector",
                    f"seed:{seed.seed_id}:refinement:{key[0]}:{key[1]}",
                )
            if str(owner) != expected_owner:
                _dispatch_fail(
                    "compiled_portfolio_position_owner_mismatch",
                    f"seed:{seed.seed_id}:refinement:{key[0]}:{key[1]}:"
                    f"{owner}/{expected_owner}",
                )
        validate_exact_sequences(seed.sequences, source=f"seed:{seed.seed_id}")

    for slot in request.candidate_slots:
        for change in slot.root_delta.changes:
            require_owner(change, source=f"slot:{slot.slot_id}")
        if slot.exact:
            validate_exact_sequences(slot.exact_sequences, source=f"slot:{slot.slot_id}")


def _preflight_design_action_candidate(
    candidate: Mapping[str, Any],
    artifact: CompiledDesignAction,
) -> Dict[str, Any]:


    context = candidate.get("causal_context")
    if not isinstance(context, Mapping):
        _dispatch_fail("candidate_design_action_provenance_missing")

    expected_identity = {
        "design_action_hash": artifact.design_action_hash,
        "compiled_design_action_hash": artifact.compiled_design_action_hash,
        **{
            field: str(getattr(artifact, field))
            for field in _DESIGN_ACTION_PARENT_BINDING_FIELDS
        },
    }
    missing = sorted(
        field
        for field in expected_identity
        if context.get(field) in (None, "")
    )
    if missing:
        _dispatch_fail(
            "candidate_design_action_provenance_missing",
            ",".join(missing),
        )
    for field, expected in expected_identity.items():
        observed = str(context.get(field))
        if observed != str(expected):
            code = (
                "candidate_design_action_hash_mismatch"
                if field == "design_action_hash"
                else "candidate_compiled_action_hash_mismatch"
                if field == "compiled_design_action_hash"
                else "candidate_parent_binding_mismatch"
            )
            _dispatch_fail(code, f"{field}:{observed}/{expected}")

    sequences = candidate.get("seqs")
    if not isinstance(sequences, Mapping):
        _dispatch_fail("candidate_sequence_bundle_missing")
    parent_role_claimed = bool(candidate.get("is_parent_baseline")) or (
        str(candidate.get("candidate_role") or "") == "parent_baseline"
    ) or str(candidate.get("variant_id") or "") == "root"
    explicit_parent_role = (
        candidate.get("is_parent_baseline") is True
        and str(candidate.get("candidate_role") or "") == "parent_baseline"
        and str(candidate.get("variant_id") or "") == "root"
        and candidate.get("parent_id") is None
        and candidate.get("move") is None
    )
    if parent_role_claimed and not explicit_parent_role:
        _dispatch_fail("candidate_parent_baseline_identity_invalid")
    if explicit_parent_role:
        try:
            observed_root_hash = SequenceBundleIdentity.create(
                {str(chain): str(sequence) for chain, sequence in sequences.items()}
            ).sequence_bundle_hash
        except (TypeError, ValueError) as error:
            _dispatch_fail("candidate_sequence_bundle_invalid", str(error))
        if observed_root_hash != artifact.reconciled_root_sequence_bundle_hash:
            _dispatch_fail("candidate_parent_baseline_sequence_mismatch")
    try:
        receipt = validate_candidate_against_compiled_action(
            artifact,
            sequences,
            str(context["compiled_design_action_hash"]),
            allow_reconciled_root=explicit_parent_role,
        )
    except DesignActionCompileError as error:
        _dispatch_fail(error.code, error.detail)

    provenance = {
        "schema_version": STRUCTURE_DESIGN_ACTION_GATE_VERSION,
        **expected_identity,
        "candidate_role": str(candidate.get("candidate_role") or "candidate"),
        "reconciled_root_allowed": explicit_parent_role,
        "validation_receipt_hash": receipt["validation_receipt_hash"],
    }
    prepared = dict(candidate)
    prepared["design_action_provenance"] = provenance
    prepared["design_action_validation"] = receipt
    return prepared


def _candidate_claims_portfolio(candidate: Mapping[str, Any]) -> bool:
    move = candidate.get("move")
    sensitive_fields = {
        "compiled_portfolio_request_hash",
        "portfolio_realization_receipts",
        "portfolio_pair_receipts",
        "portfolio_realization_summary",
        "prebuilt_exact_duplicate_receipts",
    }
    return bool(
        any(field in candidate for field in sensitive_fields)
        or (
            isinstance(move, Mapping)
            and (
                str(move.get("op") or "") == "portfolio_exact"
                or any(field in move for field in sensitive_fields)
            )
        )
    )


def _preflight_portfolio_candidate(
    candidate: Mapping[str, Any],
    request: CompiledPortfolioOptimizationRequest,
) -> Dict[str, Any]:
    context = candidate.get("causal_context")
    if not isinstance(context, Mapping):
        _dispatch_fail("candidate_portfolio_provenance_missing")
    observed = str(context.get("compiled_portfolio_request_hash") or "")
    if not observed:
        _dispatch_fail("candidate_portfolio_provenance_missing")
    if observed != request.compiled_portfolio_request_hash:
        _dispatch_fail(
            "candidate_portfolio_request_hash_mismatch",
            f"{observed}/{request.compiled_portfolio_request_hash}",
        )
    try:
        validation = validate_candidate_portfolio_realization(candidate, request)
    except PortfolioRuntimeError as error:
        _dispatch_fail(error.code, error.detail)
    prepared = dict(candidate)
    prepared["portfolio_dispatch_validation"] = validation
    if validation.get("claimed") is True:


        prepared["portfolio_realization_summary"] = deepcopy(
            validation["trusted_realization_summary"]
        )
    return prepared


def _mapping_component_claimed(candidate: Mapping[str, Any]) -> bool:


    move = candidate.get("move")
    return bool(
        any(
            field in candidate
            for field in ("mapping_components", "mapping_component_set_hash")
        )
        or (
            isinstance(move, Mapping)
            and any(
                field in move
                for field in ("mapping_components", "mapping_component_set_hash")
            )
        )
    )


def _mapping_dispatch_policy(
    evaluator_kwargs: Mapping[str, Any],
) -> tuple[bool, list[Any]]:


    compiled = evaluator_kwargs.get("compiled")
    compiled = compiled if isinstance(compiled, Mapping) else {}
    raw_plan = compiled.get("executable_mapping_plan")
    raw_schedule = compiled.get("effective_mapping_schedule")
    plan = raw_plan if isinstance(raw_plan, Mapping) else {}
    schedule = raw_schedule if isinstance(raw_schedule, Mapping) else {}
    plan_enabled = bool(plan.get("execution_enabled"))
    schedule_enabled = bool(schedule.get("execution_enabled"))
    full = bool(
        plan_enabled
        and schedule_enabled
        and str(plan.get("execution_mode") or "") == "full"
        and str(schedule.get("execution_mode") or "") == "full"
    )
    if plan_enabled != schedule_enabled:
        _dispatch_fail("mapping_component_execution_policy_mismatch")
    raw_actions = schedule.get("active_action_specs", []) or []
    if isinstance(raw_actions, (str, bytes)) or not isinstance(
        raw_actions, Sequence
    ):
        _dispatch_fail("mapping_component_active_actions_invalid")
    actions = list(raw_actions)
    if full and not actions:
        _dispatch_fail("mapping_component_active_actions_missing")
    return full, actions


def _mapping_component_dispatch_receipt(
    candidate: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    move = candidate["move"]
    positions = sorted(
        {
            (str(change["chain_id"]), int(change["position"]))
            for component in components
            for change in component.get("changes", [])
            if isinstance(change, Mapping)
        }
    )
    semantic = {
        "schema_version": MAPPING_COMPONENT_DISPATCH_VALIDATION_VERSION,
        "validated": True,
        "mapping_mode": "full",
        "compiled_portfolio_request_hash": candidate.get(
            "compiled_portfolio_request_hash"
        ),
        "mapping_component_set_hash": move.get("mapping_component_set_hash"),
        "component_count": len(components),
        "component_hashes": [
            str(component["mapping_component_hash"])
            for component in components
        ],
        "action_ids": [
            str(component["mapping_attribution"]["action_id"])
            for component in components
        ],
        "owner_node_ids": sorted(
            {str(component["owner_node_id"]) for component in components}
        ),
        "position_count": len(positions),
        "positions": [
            {"chain_id": chain, "position": position}
            for chain, position in positions
        ],
    }
    digest = hashlib.sha256(
        (
            MAPPING_COMPONENT_DISPATCH_VALIDATION_VERSION
            + "\0"
            + json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        ).encode("utf-8")
    ).hexdigest()
    return {
        **semantic,
        "receipt_hash": "mapping_component_dispatch_sha256:" + digest,
    }


def _preflight_portfolio_mapping_candidate(
    candidate: Mapping[str, Any],
    *,
    full_mapping: bool,
    mapping_actions: Sequence[Any],
) -> Dict[str, Any]:


    prepared = deepcopy(dict(candidate))
    move = prepared.get("move")
    if any(
        field in prepared
        for field in ("mapping_components", "mapping_component_set_hash")
    ):
        _dispatch_fail("candidate_mapping_components_top_level_forbidden")
    exact_mapping_move = bool(
        isinstance(move, Mapping)
        and str(move.get("op") or "") in {
            "portfolio_exact",
            "case_bootstrap_exact",
        }
    )
    claimed = _mapping_component_claimed(prepared)
    if not full_mapping:
        if claimed:
            _dispatch_fail("candidate_mapping_components_forbidden")
        return prepared
    if not exact_mapping_move:


        if claimed:
            _dispatch_fail("candidate_mapping_components_nonportfolio")
        return prepared
    if "mapping_components" not in move or "mapping_component_set_hash" not in move:
        _dispatch_fail("candidate_mapping_components_missing")
    trusted_summary = prepared.get("portfolio_realization_summary")
    if not isinstance(trusted_summary, Mapping) and isinstance(move, Mapping):
        trusted_summary = move.get("mapping_realization_summary")
    if not isinstance(trusted_summary, Mapping):
        _dispatch_fail("candidate_mapping_owner_summary_missing")
    trusted_move = deepcopy(dict(move))


    if str(trusted_move.get("op") or "") == "portfolio_exact":
        trusted_move["portfolio_realization_summary"] = deepcopy(trusted_summary)
    else:
        trusted_move["mapping_realization_summary"] = deepcopy(trusted_summary)
    try:
        components = validate_portfolio_mapping_components(
            trusted_move,
            mapping_actions=mapping_actions,
        )
    except PortfolioMappingComponentError as error:
        _dispatch_fail(error.code, error.detail)
    prepared["move"] = trusted_move
    prepared["mapping_component_dispatch_validation"] = (
        _mapping_component_dispatch_receipt(prepared, components)
    )
    return prepared


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _dispatch_fail("candidate_wave_not_canonical_json", str(error))


def _dispatch_hash(prefix: str, domain: str, semantic: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        (domain + "\0" + _canonical_json(semantic)).encode("utf-8")
    ).hexdigest()
    return prefix + digest


def _cfg_value(cfg: Any, field: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(field, default)
    return getattr(cfg, field, default)


def _wave_artifact(
    evaluator_kwargs: Mapping[str, Any], *fields: str
) -> Any:
    compiled = evaluator_kwargs.get("compiled")
    compiled = compiled if isinstance(compiled, Mapping) else {}
    values = []
    for field in fields:
        for container in (evaluator_kwargs, compiled):
            value = container.get(field)
            if value is not None:
                values.append((field, value))
    if not values:
        return None
    canonical = _canonical_json(values[0][1])
    if any(_canonical_json(value) != canonical for _field, value in values[1:]):
        _dispatch_fail(
            "candidate_wave_artifact_fields_conflict",
            ",".join(sorted({field for field, _value in values})),
        )
    return values[0][1]


def _candidate_wave_enabled(evaluator_kwargs: Mapping[str, Any]) -> bool:
    cfg = evaluator_kwargs.get("cfg")
    enabled = _cfg_value(cfg, "candidate_wave_enabled", False)
    if enabled is not True and enabled is not False:
        _dispatch_fail("candidate_wave_enabled_invalid")
    artifacts_present = any(
        _wave_artifact(evaluator_kwargs, field) is not None
        for field in ("candidate_wave_request", "frozen_candidate_wave")
    )
    if artifacts_present and enabled is not True:
        _dispatch_fail("candidate_wave_artifacts_present_while_disabled")
    return enabled is True


def _validate_candidate_wave_config(evaluator_kwargs: Mapping[str, Any]) -> None:
    cfg = evaluator_kwargs.get("cfg")
    expected = {
        "candidate_wave_size": 8,
        "candidate_wave_protenix_mutant_quota": 8,
        "candidate_wave_af3_mutant_quota": 4,
    }
    for field, required in expected.items():
        observed = _cfg_value(cfg, field, None)
        if observed != required:
            _dispatch_fail(
                "candidate_wave_dispatch_config_mismatch",
                f"{field}:{observed}/{required}",
            )
    if _cfg_value(cfg, "candidate_wave_fail_on_underfill", None) is not True:
        _dispatch_fail(
            "candidate_wave_dispatch_config_mismatch",
            "candidate_wave_fail_on_underfill",
        )


def _normalized_wave_provider(value: Any) -> str:
    provider = str(value or "").strip().lower()
    return {
        "af3": "alphafold3",
        "alpha-fold3": "alphafold3",
        "alpha_fold3": "alphafold3",
    }.get(provider, provider)


def _wave_sequence_hash(candidate: Mapping[str, Any]) -> str:
    sequences = candidate.get("seqs")
    if not isinstance(sequences, Mapping):
        _dispatch_fail("candidate_wave_candidate_sequences_missing")
    try:
        return SequenceBundleIdentity.create(
            {str(chain): str(sequence) for chain, sequence in sequences.items()}
        ).sequence_bundle_hash
    except (TypeError, ValueError) as error:
        _dispatch_fail("candidate_wave_sequence_bundle_invalid", str(error))


def _explicit_wave_parent(candidate: Mapping[str, Any], root_hash: str) -> bool:
    explicit = bool(
        candidate.get("is_parent_baseline") is True
        and str(candidate.get("candidate_role") or "") == "parent_baseline"
        and str(candidate.get("variant_id") or "") == "root"
        and candidate.get("parent_id") is None
        and candidate.get("move") is None
    )
    return explicit and _wave_sequence_hash(candidate) == root_hash


def _trusted_wave_candidate(
    candidate: Mapping[str, Any],
    member: Mapping[str, Any],
    *,
    wave: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> Dict[str, Any]:
    slot_id = str(member["slot_id"])
    observed_frozen_hash = str(candidate.get("frozen_candidate_wave_hash") or "")
    if observed_frozen_hash != frozen["frozen_candidate_wave_hash"]:
        _dispatch_fail(
            "candidate_wave_candidate_manifest_hash_mismatch", slot_id
        )
    observed_realization = candidate.get("candidate_wave_slot_realization")
    if _canonical_json(observed_realization) != _canonical_json(
        member["candidate_wave_slot_realization"]
    ):
        _dispatch_fail("candidate_wave_slot_realization_mismatch", slot_id)
    expected_directive = member["candidate_wave_slot_directive"]
    observed_directive = candidate.get("candidate_wave_slot_directive")
    if expected_directive is not None:
        if _canonical_json(observed_directive) != _canonical_json(expected_directive):
            _dispatch_fail("candidate_wave_slot_directive_mismatch", slot_id)
    elif observed_directive is not None:
        _dispatch_fail("candidate_wave_exact_directive_forbidden", slot_id)
    for field in ("candidate_wave_role", "portfolio_role"):
        claimed = candidate.get(field)
        if claimed is not None and str(claimed) != str(member["role"]):
            _dispatch_fail("candidate_wave_candidate_role_mismatch", slot_id)
    trusted_claims = {
        "candidate_sequence_bundle_hash": member[
            "candidate_sequence_bundle_hash"
        ],
        "candidate_wave_slot_id": slot_id,
        "candidate_wave_slot_hash": member["slot_hash"],
        "candidate_wave_portfolio_id": member["portfolio_id"],
        "candidate_wave_generation_mode": member["generation_mode"],
        "candidate_wave_realization_kind": member["realization_kind"],
        "candidate_wave_exact": bool(member["exact"]),
    }
    for field, expected in trusted_claims.items():
        if field in candidate and candidate[field] != expected:
            _dispatch_fail(
                "candidate_wave_candidate_claim_mismatch", f"{slot_id}:{field}"
            )
    prepared = deepcopy(dict(candidate))
    prepared.update(
        {
            "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
            "frozen_candidate_wave_hash": frozen["frozen_candidate_wave_hash"],
            "candidate_wave_role": member["role"],
            "candidate_wave_slot_directive": deepcopy(expected_directive),
            "candidate_wave_slot_realization": deepcopy(
                member["candidate_wave_slot_realization"]
            ),
            **trusted_claims,
        }
    )
    return prepared


def _candidate_wave_batch_receipt(
    *,
    wave: Mapping[str, Any],
    frozen: Mapping[str, Any],
    provider: str,
    stage: str,
    parent_hash: str,
    mutant_members: Sequence[Mapping[str, Any]],
    rerank_receipt: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    semantic = {
        "schema_version": CANDIDATE_WAVE_BATCH_DISPATCH_VERSION,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "frozen_candidate_wave_hash": frozen["frozen_candidate_wave_hash"],
        "provider": provider,
        "stage": stage,
        "parent_sequence_bundle_hash": parent_hash,
        "parent_count": 1,
        "mutant_count": len(mutant_members),
        "mutant_sequence_bundle_hashes": [
            member["candidate_sequence_bundle_hash"]
            for member in mutant_members
        ],
        "mutant_slot_ids": [member["slot_id"] for member in mutant_members],
        "mutant_roles": [member["role"] for member in mutant_members],
        "rerank_selection_receipt_hash": (
            rerank_receipt["rerank_selection_receipt_hash"]
            if rerank_receipt is not None
            else None
        ),
        "complete_batch_validated_before_provider": True,
    }
    return {
        **semantic,
        "candidate_wave_batch_dispatch_hash": _dispatch_hash(
            "candidate_wave_batch_dispatch_sha256:",
            CANDIDATE_WAVE_BATCH_DISPATCH_VERSION,
            semantic,
        ),
    }


def _candidate_wave_candidate_receipt(
    batch_receipt: Mapping[str, Any],
    *,
    sequence_hash: str,
    member: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    semantic = {
        "schema_version": CANDIDATE_WAVE_CANDIDATE_DISPATCH_VERSION,
        "candidate_wave_batch_dispatch_hash": batch_receipt[
            "candidate_wave_batch_dispatch_hash"
        ],
        "candidate_sequence_bundle_hash": sequence_hash,
        "dispatch_role": "parent_baseline" if member is None else "mutant",
        "slot_id": member["slot_id"] if member is not None else None,
        "candidate_wave_role": member["role"] if member is not None else None,
        "slot_realization_receipt_hash": (
            member["candidate_wave_slot_realization"][
                "realization_receipt_hash"
            ]
            if member is not None
            else None
        ),
    }
    return {
        **semantic,
        "candidate_wave_candidate_dispatch_hash": _dispatch_hash(
            "candidate_wave_candidate_dispatch_sha256:",
            CANDIDATE_WAVE_CANDIDATE_DISPATCH_VERSION,
            semantic,
        ),
    }


def _preflight_candidate_wave_batch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evaluator_kwargs: Mapping[str, Any],
    provider: Any,
    stage: Any,
) -> List[Dict[str, Any]]:
    _validate_candidate_wave_config(evaluator_kwargs)
    raw_wave = _wave_artifact(evaluator_kwargs, "candidate_wave_request")
    raw_frozen = _wave_artifact(evaluator_kwargs, "frozen_candidate_wave")
    if not isinstance(raw_wave, Mapping) or not isinstance(raw_frozen, Mapping):
        _dispatch_fail("candidate_wave_dispatch_artifacts_missing")
    try:
        wave = validate_candidate_wave_request(raw_wave)
        frozen = validate_frozen_candidate_wave(raw_frozen, wave)
    except CandidateWaveError as error:
        _dispatch_fail(error.code, error.detail)
    resolved_provider = _normalized_wave_provider(provider)
    resolved_stage = str(stage or "").strip().lower()
    if (resolved_stage, resolved_provider) not in {
        ("screen", "protenix"),
        ("rerank", "alphafold3"),
    }:
        _dispatch_fail(
            "candidate_wave_dispatch_provider_stage_invalid",
            f"{resolved_stage}:{resolved_provider}",
        )
    parent_indices = [
        index
        for index, candidate in enumerate(candidates)
        if _explicit_wave_parent(
            candidate, wave["reconciled_root_sequence_bundle_hash"]
        )
    ]
    if parent_indices != [0]:
        _dispatch_fail(
            "candidate_wave_parent_batch_identity_invalid",
            repr(parent_indices),
        )
    parent_hash = _wave_sequence_hash(candidates[0])
    mutant_candidates = list(candidates[1:])
    mutant_hashes = [_wave_sequence_hash(candidate) for candidate in mutant_candidates]
    if len(mutant_hashes) != len(set(mutant_hashes)):
        _dispatch_fail("candidate_wave_dispatch_duplicate_mutant")
    manifest_by_hash = {
        member["candidate_sequence_bundle_hash"]: member
        for member in frozen["members"]
    }
    rerank_receipt: Mapping[str, Any] | None = None
    if resolved_stage == "screen":
        if _wave_artifact(
            evaluator_kwargs,
            "rerank_selection_receipt",
            "candidate_wave_rerank_selection_receipt",
        ) is not None:
            _dispatch_fail("candidate_wave_screen_has_rerank_receipt")
        expected_hashes = set(manifest_by_hash)
        if len(mutant_candidates) != 8:
            _dispatch_fail(
                "candidate_wave_screen_mutant_count_invalid",
                str(len(mutant_candidates)),
            )
    else:
        raw_rerank = _wave_artifact(
            evaluator_kwargs,
            "rerank_selection_receipt",
            "candidate_wave_rerank_selection_receipt",
        )
        if not isinstance(raw_rerank, Mapping):
            _dispatch_fail("candidate_wave_rerank_selection_receipt_missing")
        try:
            rerank_receipt = validate_rerank_selection_receipt(
                raw_rerank, frozen
            )
        except CandidateWaveError as error:
            _dispatch_fail(error.code, error.detail)
        expected_hashes = {
            member["candidate_sequence_bundle_hash"]
            for member in rerank_receipt["selected_members"]
        }
        if len(mutant_candidates) != 4:
            _dispatch_fail(
                "candidate_wave_rerank_mutant_count_invalid",
                str(len(mutant_candidates)),
            )
    observed_hashes = set(mutant_hashes)
    if observed_hashes != expected_hashes:
        missing = sorted(expected_hashes - observed_hashes)
        extra = sorted(observed_hashes - expected_hashes)
        _dispatch_fail(
            "candidate_wave_dispatch_membership_mismatch",
            f"missing={missing};extra={extra}",
        )
    trusted_mutants = [
        _trusted_wave_candidate(
            candidate,
            manifest_by_hash[sequence_hash],
            wave=wave,
            frozen=frozen,
        )
        for candidate, sequence_hash in zip(mutant_candidates, mutant_hashes)
    ]
    selected_members = [
        manifest_by_hash[sequence_hash] for sequence_hash in mutant_hashes
    ]
    batch_receipt = _candidate_wave_batch_receipt(
        wave=wave,
        frozen=frozen,
        provider=resolved_provider,
        stage=resolved_stage,
        parent_hash=parent_hash,
        mutant_members=selected_members,
        rerank_receipt=rerank_receipt,
    )
    trusted_parent = deepcopy(dict(candidates[0]))
    trusted_parent.update(
        {
            "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
            "frozen_candidate_wave_hash": frozen["frozen_candidate_wave_hash"],
        }
    )
    prepared = [trusted_parent, *trusted_mutants]
    for candidate, member in zip(prepared, [None, *selected_members]):
        sequence_hash = _wave_sequence_hash(candidate)
        candidate["candidate_wave_batch_dispatch_validation"] = deepcopy(
            batch_receipt
        )
        candidate["candidate_wave_dispatch_validation"] = (
            _candidate_wave_candidate_receipt(
                batch_receipt,
                sequence_hash=sequence_hash,
                member=member,
            )
        )
        if rerank_receipt is not None:
            candidate["rerank_selection_receipt"] = deepcopy(rerank_receipt)
    return prepared


def preflight_structure_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evaluator_kwargs: Mapping[str, Any],
    provider: Any = None,
    stage: Any = None,
) -> list[Dict[str, Any]]:


    items = [dict(candidate) for candidate in candidates]
    artifact = _strict_compiled_design_action(evaluator_kwargs)
    portfolio_request = _strict_compiled_portfolio_request(evaluator_kwargs)
    if portfolio_request is not None and artifact is None:
        _dispatch_fail("compiled_portfolio_design_action_missing")
    if portfolio_request is not None and artifact is not None:
        _validate_portfolio_action_binding(artifact, portfolio_request)
    if portfolio_request is None and any(
        _candidate_claims_portfolio(candidate) for candidate in items
    ):
        _dispatch_fail("candidate_portfolio_request_missing")
    prepared = (
        [
            _preflight_design_action_candidate(candidate, artifact)
            for candidate in items
        ]
        if artifact is not None
        else items
    )
    if portfolio_request is not None:
        prepared = [
            _preflight_portfolio_candidate(candidate, portfolio_request)
            for candidate in prepared
        ]
    full_mapping, mapping_actions = _mapping_dispatch_policy(evaluator_kwargs)
    prepared = [
        _preflight_portfolio_mapping_candidate(
            candidate,
            full_mapping=full_mapping,
            mapping_actions=mapping_actions,
        )
        for candidate in prepared
    ]
    if _candidate_wave_enabled(evaluator_kwargs):
        prepared = _preflight_candidate_wave_batch(
            prepared,
            evaluator_kwargs=evaluator_kwargs,
            provider=provider,
            stage=stage,
        )
    return prepared


_COMPILED_NESTED_SCRATCH_FIELDS = ("_struct_cache",)
_DISPATCH_ONLY_EVALUATOR_FIELDS = frozenset(
    {
        "candidate_wave_request",
        "frozen_candidate_wave",
        "candidate_wave_rerank_selection_receipt",
        "rerank_selection_receipt",
    }
)


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


def _task_kwargs(
    shared: Mapping[str, Any], *, isolate_compiled: bool
) -> Dict[str, Any]:


    resolved = {
        key: value
        for key, value in shared.items()
        if key not in _DISPATCH_ONLY_EVALUATOR_FIELDS
    }
    if isolate_compiled and "compiled" in resolved:
        compiled = resolved["compiled"]
        if not isinstance(compiled, Mapping):
            raise TypeError("compiled evaluator state must be a mapping")


        task_compiled = dict(compiled)
        for field in _COMPILED_NESTED_SCRATCH_FIELDS:
            if field in compiled:
                task_compiled[field] = deepcopy(compiled[field])
        resolved["compiled"] = task_compiled
    return resolved


def _evaluate_batches(
    items: list[Dict[str, Any]],
    *,
    evaluator: StructureEvaluator,
    evaluator_kwargs: Mapping[str, Any],
    provider: str,
    stage: str,
    requested_batch_size: int,
    resolved_batch_size: int,
    workers: int,
    executor: Optional[ThreadPoolExecutor],
) -> list[Dict[str, Any]]:


    results: list[Dict[str, Any]] = []
    execution_mode = "threaded" if workers > 1 else "sequential"

    def invoke(candidate: Dict[str, Any]) -> Dict[str, Any]:


        kwargs = _task_kwargs(
            evaluator_kwargs,
            isolate_compiled=True,
        )
        return evaluator(
            candidate,
            provider=provider,
            stage=stage,
            **kwargs,
        )

    for batch_index, offset in enumerate(
        range(0, len(items), resolved_batch_size), start=1
    ):
        batch = items[offset : offset + resolved_batch_size]
        if executor is None or len(batch) == 1:
            batch_results = [invoke(candidate) for candidate in batch]
        else:
            futures = [
                executor.submit(copy_context().run, invoke, candidate)
                for candidate in batch
            ]


            batch_results = [future.result() for future in futures]

        for index_in_batch, result in enumerate(batch_results, start=1):
            normalized = dict(result)
            candidate = batch[index_in_batch - 1]
            for field in (
                "design_action_provenance",
                "design_action_validation",
                "compiled_portfolio_request_hash",
                "portfolio_realization_receipts",
                "portfolio_pair_receipts",
                "portfolio_realization_summary",
                "portfolio_dispatch_validation",
                "mapping_component_dispatch_validation",
                "candidate_wave_request_hash",
                "frozen_candidate_wave_hash",
                "candidate_wave_slot_id",
                "candidate_sequence_bundle_hash",
                "candidate_wave_role",
                "candidate_wave_slot_hash",
                "candidate_wave_portfolio_id",
                "candidate_wave_generation_mode",
                "candidate_wave_realization_kind",
                "candidate_wave_exact",
                "candidate_wave_slot_directive",
                "candidate_wave_slot_realization",
                "candidate_wave_batch_dispatch_validation",
                "candidate_wave_dispatch_validation",
                "rerank_selection_receipt",
            ):
                if field in candidate:


                    normalized[field] = deepcopy(candidate[field])
            if "candidate_wave_dispatch_validation" in candidate:


                for field in (
                    "seqs",
                    "variant_id",
                    "parent_id",
                    "candidate_role",
                    "is_parent_baseline",
                    "causal_context",
                    "move",
                ):
                    if field in candidate:
                        normalized[field] = deepcopy(candidate[field])
            if "mapping_component_dispatch_validation" in candidate:


                normalized["move"] = deepcopy(candidate["move"])
            normalized["structure_batch_dispatch"] = {
                "schema_version": STRUCTURE_BATCH_DISPATCH_VERSION,
                "provider": str(provider),
                "stage": str(stage),
                "batch_index": batch_index,
                "index_in_batch": index_in_batch,
                "candidates_in_batch": len(batch),
                "requested_batch_size": requested_batch_size,
                "max_workers": workers,
                "execution_mode": execution_mode,
                "outer_publication_barrier_preserved": True,
            }
            provenance = normalized.get("design_action_provenance")
            if isinstance(provenance, Mapping):
                normalized["structure_batch_dispatch"].update(
                    {
                        "design_action_hash": provenance.get(
                            "design_action_hash"
                        ),
                        "compiled_design_action_hash": provenance.get(
                            "compiled_design_action_hash"
                        ),
                        "design_action_validation_receipt_hash": provenance.get(
                            "validation_receipt_hash"
                        ),
                    }
                )
            wave_dispatch = normalized.get("candidate_wave_dispatch_validation")
            if isinstance(wave_dispatch, Mapping):
                normalized["structure_batch_dispatch"].update(
                    {
                        "frozen_candidate_wave_hash": normalized.get(
                            "frozen_candidate_wave_hash"
                        ),
                        "candidate_wave_batch_dispatch_hash": wave_dispatch.get(
                            "candidate_wave_batch_dispatch_hash"
                        ),
                        "candidate_wave_candidate_dispatch_hash": (
                            wave_dispatch.get(
                                "candidate_wave_candidate_dispatch_hash"
                            )
                        ),
                    }
                )
            results.append(normalized)
    return results


def evaluate_structure_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    evaluator: StructureEvaluator,
    evaluator_kwargs: Mapping[str, Any],
    provider: str,
    stage: str,
    batch_size: int = 0,
    max_workers: int = 1,
) -> list[Dict[str, Any]]:


    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    workers = _positive_int(max_workers, name="max_workers")
    requested_batch_size = _positive_int(batch_size, name="batch_size", allow_zero=True)


    items = preflight_structure_candidates(
        candidates,
        evaluator_kwargs=evaluator_kwargs,
        provider=provider,
        stage=stage,
    )
    if not items:
        return []
    resolved_batch_size = requested_batch_size or len(items)
    arguments = {
        "items": items,
        "evaluator": evaluator,
        "evaluator_kwargs": evaluator_kwargs,
        "provider": provider,
        "stage": stage,
        "requested_batch_size": requested_batch_size,
        "resolved_batch_size": resolved_batch_size,
        "workers": workers,
    }
    if workers == 1 or len(items) == 1 or resolved_batch_size == 1:
        return _evaluate_batches(**arguments, executor=None)


    with ThreadPoolExecutor(
        max_workers=min(workers, resolved_batch_size, len(items)),
        thread_name_prefix=f"structure-{stage}",
    ) as executor:
        return _evaluate_batches(**arguments, executor=executor)


__all__ = [
    "STRUCTURE_BATCH_DISPATCH_VERSION",
    "STRUCTURE_DESIGN_ACTION_GATE_VERSION",
    "MAPPING_COMPONENT_DISPATCH_VALIDATION_VERSION",
    "CANDIDATE_WAVE_BATCH_DISPATCH_VERSION",
    "CANDIDATE_WAVE_CANDIDATE_DISPATCH_VERSION",
    "StructureDispatchValidationError",
    "evaluate_structure_candidates",
    "preflight_structure_candidates",
]

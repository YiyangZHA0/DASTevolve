

from __future__ import annotations

import json
import hashlib
import pprint
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from astevolve.domain.design_action import (
    DESIGN_ACTION_VERSION,
    DesignAction,
    DesignActionError,
)
from astevolve.runtime.edit_contract_lifecycle import (
    EDIT_CONTRACT_RESPONSE_VERSION,
    EditContractEnvelope,
    EditContractLifecycleError,
    validate_contract_response,
)
from outerloop.structured_ast_proposal import (
    _plan_location,
    extract_current_ast_revision_plan,
    parent_evolve_hash,
)
from outerloop.utils.code_utils import CandidateDiffError


STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE = "structured_design_action_v1"
STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION = (
    "astevolve.structured_design_action_proposal.v1"
)
STRUCTURED_DESIGN_ACTION_CONTROLS: Dict[str, str] = {
    "proposal_mode": STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE,
    "mutation_surface": "design_action_v1",
    "design_action_schema_version": DESIGN_ACTION_VERSION,
    "rewrite_target": "propose_strategy.ast_revision_plan",
}

_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "parent_evolve_hash",
        "controls",
        "edit_contract_response",
        "design_action",
    }
)
STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS = frozenset(
    {
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "parent_effective_contract_hash",
        "parent_evolve_hash",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StructuredDesignActionEnvelope:


    parent_evolve_hash: str
    edit_contract_response: Mapping[str, str] | None
    design_action: DesignAction
    schema_version: str = STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION


@dataclass(frozen=True)
class StructuredDesignActionApplication:


    code: str
    parent_evolve_hash: str
    design_action: DesignAction
    edit_contract_response: Mapping[str, str] | None
    ast_revision_plan: Mapping[str, Any]
    changes_summary: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def structured_design_action_artifacts(
    application: StructuredDesignActionApplication,
    *,
    expected_parent_binding: Mapping[str, str],
) -> Dict[str, Any]:


    binding = _validated_expected_parent_binding(expected_parent_binding)
    action = application.design_action
    if action.parent.to_dict() != binding:
        raise CandidateDiffError(
            "structured_design_action_parent_binding_mismatch",
            "sealed DesignAction does not match the controller parent binding",
        )
    action_artifact = action.to_dict()
    action_payload = action.to_payload()
    audit_core = {
        "schema_version": (
            "astevolve.structured_design_action_proposal_audit_artifact.v1"
        ),
        "proposal_mode": STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE,
        "parent_evolve_hash": application.parent_evolve_hash,
        "design_action_hash": action.design_action_hash,
        "parent_binding": deepcopy(binding),
        "ast_revision_plan": deepcopy(dict(application.ast_revision_plan)),
        "hypothesis": deepcopy(action_payload["hypothesis"]),
        "falsification_predicates": deepcopy(
            action_payload["falsification_predicates"]
        ),
        "rollback_actions": deepcopy(action_payload["rollback_actions"]),
        "contains_hidden_chain_of_thought": False,
        "sealed_before_evaluation": True,
    }
    audit_hash = "structured_design_action_audit_sha256:" + hashlib.sha256(
        (
            "astevolve.structured_design_action_proposal_audit_artifact.v1\0"
            + _canonical_json(audit_core)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "design_action": deepcopy(action_artifact),
        "design_action_parent_binding": deepcopy(binding),
        "structured_design_action_proposal_audit": {
            **audit_core,
            "audit_hash": audit_hash,
        },
    }


def _strict_json(text: Any, *, max_bytes: int = 256 * 1024) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise CandidateDiffError(
            "structured_design_action_empty_response",
            "response must be one non-empty JSON object",
        )
    if len(text.encode("utf-8")) > max_bytes:
        raise CandidateDiffError(
            "structured_design_action_response_too_large",
            f"response exceeds {max_bytes} UTF-8 bytes",
        )
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise CandidateDiffError(
            "structured_design_action_json_only",
            "response must contain only the JSON object, without fences or prose",
        )

    def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise CandidateDiffError(
                    "structured_design_action_duplicate_key",
                    f"JSON object repeats key {key!r}",
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise CandidateDiffError(
            "structured_design_action_nonfinite_number",
            f"JSON contains non-finite value {value}",
        )

    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except CandidateDiffError:
        raise
    except json.JSONDecodeError as error:
        raise CandidateDiffError(
            "structured_design_action_invalid_json",
            f"response is invalid JSON at line {error.lineno} column {error.colno}",
        ) from error
    if not isinstance(parsed, Mapping):
        raise CandidateDiffError(
            "structured_design_action_invalid_type",
            "proposal envelope must be a JSON object",
        )
    keys = set(parsed)
    unknown = sorted(keys - _ENVELOPE_FIELDS)
    missing = sorted(_ENVELOPE_FIELDS - keys)
    if unknown or missing:
        details: Dict[str, Any] = {}
        if unknown:
            details["unknown_fields"] = unknown
        if missing:
            details["missing_fields"] = missing
        raise CandidateDiffError(
            "structured_design_action_closed_schema_violation",
            "proposal envelope must contain exactly "
            + ", ".join(sorted(_ENVELOPE_FIELDS)),
            details=details,
        )
    return parsed


def _design_action_error(error: DesignActionError) -> CandidateDiffError:
    raw_code = str(getattr(error, "code", "invalid") or "invalid")
    safe_code = re.sub(r"[^a-z0-9_]+", "_", raw_code.lower()).strip("_")
    path = str(getattr(error, "path", "") or "")
    detail = str(getattr(error, "detail", "") or str(error))
    return CandidateDiffError(
        f"structured_design_action_{safe_code or 'invalid'}",
        detail,
        details={"design_action_error_code": raw_code, "path": path},
    )


def _validate_edit_contract_response(
    value: Any,
    *,
    edit_contract_envelope_json: str,
) -> Mapping[str, str] | None:
    if not edit_contract_envelope_json:
        if value is not None:
            raise CandidateDiffError(
                "structured_design_action_unoffered_edit_contract_response",
                "edit_contract_response must be null when no contract was offered",
            )
        return None
    try:
        offered = EditContractEnvelope.from_json(edit_contract_envelope_json)


        if isinstance(value, Mapping):
            for legacy_key in ("last_contract_response", "response", "edit_contract_response"):
                nested = value.get(legacy_key)
                if isinstance(nested, Mapping):
                    value = nested
                    break
        return validate_contract_response(value, offered)
    except EditContractLifecycleError as error:
        raise CandidateDiffError(
            "structured_design_action_edit_contract_response_invalid",
            str(error),
            details={"edit_contract_error_code": str(error.code)},
        ) from error


def parse_structured_design_action_response(
    response: str,
    *,
    edit_contract_envelope_json: str = "",
) -> StructuredDesignActionEnvelope:


    envelope = _strict_json(response)
    if envelope.get("schema_version") != STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION:
        raise CandidateDiffError(
            "structured_design_action_invalid_schema",
            "schema_version must be "
            f"{STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION!r}",
        )
    supplied_hash = envelope.get("parent_evolve_hash")
    if not isinstance(supplied_hash, str) or not _SHA256.fullmatch(supplied_hash):
        raise CandidateDiffError(
            "structured_design_action_invalid_parent_hash",
            "parent_evolve_hash must be 64 lowercase hexadecimal characters",
        )
    if envelope.get("controls") != STRUCTURED_DESIGN_ACTION_CONTROLS:
        raise CandidateDiffError(
            "structured_design_action_controls_mismatch",
            "controls must exactly match the controller-owned required_controls",
            details={
                "required_controls": deepcopy(STRUCTURED_DESIGN_ACTION_CONTROLS)
            },
        )
    edit_response = _validate_edit_contract_response(
        envelope.get("edit_contract_response"),
        edit_contract_envelope_json=edit_contract_envelope_json,
    )
    raw_action = envelope.get("design_action")
    try:


        if isinstance(raw_action, Mapping) and "design_action_hash" in raw_action:
            action = DesignAction.from_mapping(raw_action)
        else:
            action = DesignAction.from_payload(raw_action)
    except DesignActionError as error:
        raise _design_action_error(error) from error
    return StructuredDesignActionEnvelope(
        parent_evolve_hash=supplied_hash,
        edit_contract_response=deepcopy(edit_response),
        design_action=action,
    )


def _action_parent_payload(action: DesignAction) -> Mapping[str, Any]:
    payload = action.to_payload()
    parent = payload.get("parent")
    if not isinstance(parent, Mapping):
        raise CandidateDiffError(
            "structured_design_action_parent_binding_missing",
            "validated DesignAction has no parent binding",
        )
    return parent


def _validated_expected_parent_binding(
    value: Mapping[str, str],
) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("expected_parent_binding must be a mapping")
    fields = set(value)
    if fields != STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS:
        raise CandidateDiffError(
            "structured_design_action_expected_parent_binding_invalid",
            "controller parent binding must contain exactly all DesignAction parent fields",
            details={
                "unknown_fields": sorted(
                    fields - STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS
                ),
                "missing_fields": sorted(
                    STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS - fields
                ),
            },
        )
    output: Dict[str, str] = {}
    for field in sorted(STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS):
        item = value[field]
        if not isinstance(item, str) or not item.strip():
            raise CandidateDiffError(
                "structured_design_action_expected_parent_binding_invalid",
                f"controller parent binding field {field!r} must be non-empty text",
                details={"field": field},
            )
        output[field] = item
    return output


def validate_structured_design_action_proposal(
    parent_code: str,
    response: str,
    *,
    expected_parent_binding: Mapping[str, str],
    edit_contract_envelope_json: str = "",
) -> StructuredDesignActionEnvelope:


    expected_binding = _validated_expected_parent_binding(expected_parent_binding)
    parsed = parse_structured_design_action_response(
        response,
        edit_contract_envelope_json=edit_contract_envelope_json,
    )
    expected_evolve_hash = parent_evolve_hash(parent_code)
    if parsed.parent_evolve_hash != expected_evolve_hash:
        raise CandidateDiffError(
            "structured_design_action_stale_parent_hash",
            "parent_evolve_hash does not match the unchanged selected parent",
            details={"expected_parent_evolve_hash": expected_evolve_hash},
        )
    parent = _action_parent_payload(parsed.design_action)
    action_evolve_hash = parent.get("parent_evolve_hash")
    if action_evolve_hash != expected_evolve_hash:
        raise CandidateDiffError(
            "structured_design_action_parent_binding_mismatch",
            "DesignAction parent_evolve_hash does not match the selected parent",
            details={
                "field": "parent_evolve_hash",
                "expected": expected_evolve_hash,
                "actual": action_evolve_hash,
            },
        )
    if edit_contract_envelope_json:
        try:
            offered = EditContractEnvelope.from_json(edit_contract_envelope_json)
        except EditContractLifecycleError as error:
            raise CandidateDiffError(
                "structured_design_action_edit_contract_response_invalid",
                str(error),
                details={"edit_contract_error_code": str(error.code)},
            ) from error
        if offered.parent_program_id != parent.get("parent_program_id"):
            raise CandidateDiffError(
                "structured_design_action_edit_contract_parent_mismatch",
                "the offered edit contract and DesignAction bind different parents",
                details={
                    "contract_parent_program_id": offered.parent_program_id,
                    "design_action_parent_program_id": parent.get(
                        "parent_program_id"
                    ),
                },
            )
    for field, expected in sorted(expected_binding.items()):
        if parent.get(field) != expected:
            raise CandidateDiffError(
                "structured_design_action_parent_binding_mismatch",
                f"DesignAction parent binding field {field!r} does not match",
                details={
                    "field": str(field),
                    "expected": expected,
                    "actual": parent.get(field),
                },
            )
    return parsed


def structured_design_action_repair_instruction(
    rejection: Exception,
    *,
    parent_code: str,
    expected_parent_binding: Mapping[str, str],
    edit_contract_envelope_json: str = "",
    previous_errors: Sequence[Exception] = (),
    expected_mutation_count_range: Tuple[int, int] | None = None,
    expected_minimum_hamming_distance: int | None = None,
    require_sequence_reconciliation: bool = False,
    minimum_position_distribution_count: int = 0,
    minimum_soft_position_distribution_count: int = 0,
    minimum_required_position_distribution_count: int = 0,
    max_positive_substitution_support_per_position: int | None = None,
    position_distribution_site_context: Mapping[str, Mapping[str, Any]] | None = None,
    require_portfolio_capabilities: bool = False,
    minimum_atomic_module_count: int = 0,
    minimum_cross_node_atomic_module_count: int = 0,
    minimum_sequence_seed_count: int = 0,
    minimum_module_portfolio_count: int = 0,
    minimum_strict_sequence_seed_portfolio_count: int = 0,
    minimum_matched_ablation_portfolio_count: int = 0,
    portfolio_position_site_context: Mapping[str, Mapping[str, Any]] | None = None,
    portfolio_sequence_context: Mapping[str, Mapping[str, str]] | None = None,
    portfolio_controller_mode: str = "step3_exact",
    require_frozen_candidate_wave: bool = False,
    step4_required_role_histogram: Mapping[str, int] | None = None,
    step4_required_portfolio_entry_shape: Mapping[str, int] | None = None,
    step4_free_slot_forced_tiers: Mapping[str, str] | None = None,
    step4_require_live_node_change: bool = False,
    step4_changed_node_min_generated_unique: int = 0,
    step4_changed_node_min_frozen_unique: int = 0,
    step4_changed_node_min_protenix_attempts: int = 0,
    step4_current_ast_revision_plan: Mapping[str, Any] | None = None,
) -> str:


    binding = _validated_expected_parent_binding(expected_parent_binding)
    expected_hash = parent_evolve_hash(parent_code)
    if binding["parent_evolve_hash"] != expected_hash:
        raise CandidateDiffError(
            "structured_design_action_parent_binding_mismatch",
            "controller parent binding does not match the supplied parent code",
            details={
                "field": "parent_evolve_hash",
                "expected": expected_hash,
                "actual": binding["parent_evolve_hash"],
            },
        )
    rows = []
    seen: set[Tuple[str, str]] = set()
    for error in (*previous_errors, rejection):
        code = str(getattr(error, "code", None) or type(error).__name__).strip()
        message = str(
            getattr(error, "message", None) or str(error)
        ).strip()
        code = code[:128]
        message = message[:600]
        identity = (code, message)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"code": code, "message": message})
        if len(rows) >= 16:
            break
    rejection_codes = {str(row.get("code") or "") for row in rows}
    rejection_messages = " ".join(str(row.get("message") or "") for row in rows)
    allowed_ast_evidence_ids: set[str] = set()
    current_plan_for_repair = extract_current_ast_revision_plan(parent_code)
    for surface in (
        current_plan_for_repair.get("structural_nodes", []) or [],
        current_plan_for_repair.get("mapping_edges", []) or [],
        [current_plan_for_repair.get("decision_record")],
    ):
        for item in surface:
            if isinstance(item, Mapping):
                allowed_ast_evidence_ids.update(
                    str(value) for value in item.get("evidence_refs", []) or []
                )
    for site_context in (
        position_distribution_site_context,
        portfolio_position_site_context,
    ):
        if isinstance(site_context, Mapping):
            for item in site_context.values():
                if isinstance(item, Mapping) and item.get("evidence_id"):
                    allowed_ast_evidence_ids.add(str(item["evidence_id"]))
    if edit_contract_envelope_json:
        try:
            offered = EditContractEnvelope.from_json(edit_contract_envelope_json)
        except EditContractLifecycleError as error:
            raise CandidateDiffError(
                "structured_design_action_edit_contract_response_invalid",
                str(error),
                details={"edit_contract_error_code": str(error.code)},
            ) from error
        if offered.parent_program_id != binding["parent_program_id"]:
            raise CandidateDiffError(
                "structured_design_action_edit_contract_parent_mismatch",
                "the offered edit contract and repair parent binding differ",
                details={
                    "contract_parent_program_id": offered.parent_program_id,
                    "design_action_parent_program_id": binding[
                        "parent_program_id"
                    ],
                },
            )


        response_example = {
            "schema_version": EDIT_CONTRACT_RESPONSE_VERSION,
            "contract_id": offered.contract_id,
            "decision": "accept",
            "reason": "Accept the exact parent-bound edit contract.",
        }
        response_rule = (
            "edit_contract_response must itself be exactly one four-field JSON "
            "object with schema_version, contract_id, decision, and reason; "
            "no wrapper and no last_contract_response, response, accepted, or "
            "rejected fields. Use this valid closed shape (decision may instead "
            "be exactly reject when evidence requires it): "
            + json.dumps(
                response_example,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "."
        )
    else:
        response_rule = "edit_contract_response must be JSON null."
    execution_rule = ""
    if any("replacement_residue_forbidden" in code for code in rejection_codes):
        execution_rule += (
            " LINEAGE UNION REPAIR: replacement_residue must be JSON null for "
            "retain, reassign_node, and revert_to_wt. Never copy an action token "
            "such as revert_to_wt into replacement_residue. Only action=replace "
            "may contain a one-letter canonical amino acid there."
        )
    if (
        "every executable functional node requires at least one mapping edge"
        in rejection_messages
        or "uncovered:" in rejection_messages
    ):
        current_plan = extract_current_ast_revision_plan(parent_code)
        current_edges = current_plan.get("mapping_edges", [])
        required_functional_ids = sorted(
            {
                str(edge.get("functional_node_id"))
                for edge in current_edges
                if isinstance(edge, Mapping) and edge.get("functional_node_id")
            }
        )
        execution_rule += (
            " DUAL-AST COVERAGE REPAIR: ast_revision_plan is a complete snapshot. "
            "Its mapping_edges must cover every required functional node at least "
            "once. Preserve every unrelated current edge byte-for-byte; if a "
            "structural target was removed, explicitly provide a valid replacement "
            "edge rather than omitting the functional objective. Required functional "
            "node IDs are "
            + json.dumps(
                required_functional_ids,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "."
        )
    if (
        "unknown evidence ids" in rejection_messages.lower()
        or "unknown evidence id" in rejection_messages.lower()
    ):
        execution_rule += (
            " RESIDUE EVIDENCE REPAIR: every evidence_refs item inside "
            "ast_revision_plan structural_nodes, mapping_edges, and "
            "decision_record must come from this exact case-owned whitelist: "
            + json.dumps(
                sorted(allowed_ast_evidence_ids),
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + ". Never use program:*, island:*, objective-vector, measurement, "
            "or other run-artifact identifiers on those AST surfaces."
        )
    if expected_mutation_count_range is not None:
        execution_rule += (
            " Set design_action.mutation_count_range exactly to "
            + json.dumps(list(expected_mutation_count_range), separators=(",", ":"))
            + "."
        )
    if expected_minimum_hamming_distance is not None:
        execution_rule += (
            " Set design_action.minimum_hamming_distance exactly to "
            + str(expected_minimum_hamming_distance)
            + "."
        )
    if require_sequence_reconciliation:
        execution_rule += (
            " This Step1 acceptance requires at least one lineage action with "
            "action=replace or action=revert_to_wt so the reconciled sequence "
            "root differs from the selected parent."
        )
    del position_distribution_site_context
    del portfolio_position_site_context
    del portfolio_sequence_context
    del step4_current_ast_revision_plan
    del step4_changed_node_min_generated_unique
    del step4_changed_node_min_frozen_unique
    del step4_changed_node_min_protenix_attempts
    if minimum_position_distribution_count:
        execution_rule += (
            " Emit at least "
            + str(minimum_position_distribution_count)
            + " executable position_distributions from active_position_catalog, "
            "including at least "
            + str(minimum_soft_position_distribution_count)
            + " executable soft entries with required_mutation=null and at "
            "least two residues having positive probability (a one-point "
            "distribution is not counted as soft), and at least "
            + str(minimum_required_position_distribution_count)
            + " entries with a non-null required_mutation. Only narrow or "
            "reweight the published hard alphabet."
        )
        if minimum_required_position_distribution_count:
            execution_rule += (
                " If this proposal creates, migrates, or resizes any Node, place "
                "at least one non-null required_mutation on a position owned by "
                "one of those changed Nodes so the Node edit necessarily reaches MCTS."
            )
    if max_positive_substitution_support_per_position is not None:
        execution_rule += (
            " For every soft distribution, after applying its lineage action, "
            "the number of positive-probability residues different from the "
            "reconciled root must be at most "
            + str(max_positive_substitution_support_per_position)
            + "; this is enforced by the controller before MCTS."
        )
    if require_frozen_candidate_wave:
        execution_rule += (
            " Step4 frozen-wave quotas are exact. Emit exactly three atomic "
            "modules (one primary module, one distinct matched treatment module, "
            "and one proper-subset matched_module_ablation), exactly one novel "
            "full-chain sequence seed, and exactly seven required entries: "
            "primary module count=1; primary mcts count=1; repair "
            "strict_sequence_seed count=1; repair mcts count=1; "
            "matched_ablation count=1; novelty mcts count=1; control control "
            "count=1. This compiles to exactly eight physical slots with role "
            "histogram "
            + json.dumps(
                dict(step4_required_role_histogram or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            + ". All entries set required=true. The four exact sequences must "
            "be pairwise distinct and differ from the reconciled root. Create, "
            "migrate, resize, or shift at least one live structural Node onto at "
            "least one newly selected coordinate from prospective_position_catalog; "
            "use proposed AST ownership for that coordinate and make at least two "
            "of the four exact candidates actually mutate every changed live Node. "
            "Free-slot tiers are controller-forced as "
            + json.dumps(
                dict(step4_free_slot_forced_tiers or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            + ". Do not add, omit, relabel, resize, or substitute entries."
        )
    elif require_portfolio_capabilities:
        execution_rule += (
            " Step3 executable portfolio acceptance quotas are exact. Emit exactly "
            + str(minimum_atomic_module_count)
            + " atomic modules, including exactly the required treatment and "
            "proper-subset ablation, with at least "
            + str(minimum_cross_node_atomic_module_count)
            + " non-ablation atomic module spanning distinct owner_node_id values; "
            "emit exactly "
            + str(minimum_sequence_seed_count)
            + " exact full-chain sequence seed; and emit only required count-one "
            "portfolio entries with exactly "
            + str(minimum_module_portfolio_count)
            + " module, "
            + str(minimum_strict_sequence_seed_portfolio_count)
            + " strict_sequence_seed, and "
            + str(minimum_matched_ablation_portfolio_count)
            + " matched_ablation entries. Mutation-module operation is a closed "
            "enum: module_insert, module_replace, module_revert, or "
            "matched_module_ablation. For this bounded acceptance, set the "
            "non-ablation treatment operation exactly to module_insert and the "
            "ablation operation exactly to matched_module_ablation; never emit "
            "substitution, point, apply_mutations, or another invented operation. "
            "Every module must set atomic=true and every mutation row must contain "
            "exactly residue, from_residue, and to_residue. Do not add optional, "
            "mcts, or control "
            "entries. Set their roles exactly to module=primary, "
            "strict_sequence_seed=repair, and matched_ablation=matched_ablation. "
            "A matched entry references one treatment "
            "module and one matched_module_ablation module as an unordered source set; "
            "the ablation rows are a non-empty proper subset of treatment rows and use "
            "from=treatment residue, to=reconciled-root residue. Exact seeds copy the "
            "published full chain set/lengths and must be distinct from both treatment "
            "and matched control; each seed needs a non-empty refinement_positions "
            "list with at least one legal alternative residue."
        )
    return (
        "STRUCTURED DESIGN ACTION REPAIR REQUIRED. Regenerate from the unchanged "
        "selected parent and resolve every rejection. previous_rejections="
        + json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + ". Return ONLY one "
        + STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION
        + " JSON object with exactly schema_version, parent_evolve_hash, controls, "
        "edit_contract_response, and design_action. Copy parent_evolve_hash="
        + expected_hash
        + ", controls="
        + json.dumps(
            STRUCTURED_DESIGN_ACTION_CONTROLS,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + ", and design_action.parent="
        + json.dumps(
            binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + " exactly. "
        + response_rule
        + execution_rule
        + " The DesignAction must remain closed v1; compatibility.v0 is never "
        "upgraded. "
        + "Every falsification_predicate.aggregation must be exactly one of "
        "single, median, worst_seed, mean (never minimum/maximum/min/max), and "
        "operator must be exactly one of <, <=, >, >=, ==. "
        + (
            "mutation_modules, sequence_seeds, and candidate_portfolio are "
            "required executable Step3 fields; no partial or unresolved intent is allowed. "
            if require_portfolio_capabilities
            else "mutation_modules, sequence_seeds, and candidate_portfolio must remain empty. "
        )
        + "position_distributions are executable exact priors. Use only evidence "
        "IDs copied verbatim from the supplied residue catalogs; program:* "
        "artifact IDs are not residue evidence and must never appear in AST "
        "decision_record or mapping edge evidence_refs. "
        "Account for every "
        "parent non-WT residue with exactly one lineage action. No Markdown, "
        "SEARCH/REPLACE blocks, source code, or prose outside JSON."
    )


def _rewrite_validated_ast_plan(
    parent_code: str,
    plan: Mapping[str, Any],
    *,
    max_code_length: int | None,
) -> str:
    code = str(parent_code).replace("\r\n", "\n").replace("\r", "\n")
    _node, start, end, start_column = _plan_location(code)
    rendered = pprint.pformat(
        deepcopy(dict(plan)),
        indent=4,
        width=240,
        sort_dicts=False,
    )
    if "\n" in rendered:
        rendered = rendered.replace("\n", "\n" + (" " * start_column))
    candidate = code[:start] + rendered + code[end:]
    if max_code_length is not None:
        if (
            isinstance(max_code_length, bool)
            or not isinstance(max_code_length, int)
            or max_code_length <= 0
        ):
            raise TypeError("max_code_length must be a positive integer or None")
        if len(candidate) > max_code_length:
            raise CandidateDiffError(
                "structured_design_action_generated_source_too_large",
                "validated AST plan renders to "
                f"{len(candidate)} characters; controller limit is {max_code_length}",
            )
    try:
        compile(candidate, "<structured-design-action-proposal>", "exec")
    except SyntaxError as error:
        raise CandidateDiffError(
            "structured_design_action_generated_invalid_python",
            "deterministic AST plan rendering produced invalid Python",
        ) from error
    if extract_current_ast_revision_plan(candidate) != plan:
        raise CandidateDiffError(
            "structured_design_action_generated_plan_mismatch",
            "the applied AST plan differs from the canonical DesignAction plan",
        )
    return candidate


def _validate_step3_portfolio_surface(
    action: DesignAction,
    *,
    require_portfolio_capabilities: bool,
    minimum_atomic_module_count: int,
    minimum_cross_node_atomic_module_count: int,
    minimum_sequence_seed_count: int,
    minimum_module_portfolio_count: int,
    minimum_strict_sequence_seed_portfolio_count: int,
    minimum_matched_ablation_portfolio_count: int,
    position_site_context: Mapping[str, Mapping[str, Any]] | None,
    sequence_context: Mapping[str, Mapping[str, str]] | None,
    portfolio_controller_mode: str = "step3_exact",
    require_frozen_candidate_wave: bool = False,
    step4_required_role_histogram: Mapping[str, int] | None = None,
    step4_required_portfolio_entry_shape: Mapping[str, int] | None = None,
    step4_require_live_node_change: bool = False,
    step4_current_ast_revision_plan: Mapping[str, Any] | None = None,
) -> None:


    quota_values = {
        "minimum_atomic_module_count": minimum_atomic_module_count,
        "minimum_cross_node_atomic_module_count": (
            minimum_cross_node_atomic_module_count
        ),
        "minimum_sequence_seed_count": minimum_sequence_seed_count,
        "minimum_module_portfolio_count": minimum_module_portfolio_count,
        "minimum_strict_sequence_seed_portfolio_count": (
            minimum_strict_sequence_seed_portfolio_count
        ),
        "minimum_matched_ablation_portfolio_count": (
            minimum_matched_ablation_portfolio_count
        ),
    }
    for name, value in quota_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CandidateDiffError(
                "structured_design_action_step3_quota_invalid",
                f"controller supplied invalid {name}",
            )

    modules = tuple(action.mutation_modules)
    seeds = tuple(action.sequence_seeds)
    portfolio = tuple(action.candidate_portfolio)
    if not require_portfolio_capabilities:
        if modules or seeds or portfolio:
            raise CandidateDiffError(
                "structured_design_action_unsupported_portfolio_capability",
                "mutation modules, sequence seeds and candidate portfolio are "
                "disabled by the controller for this proposal",
            )
        return

    controller_mode = str(portfolio_controller_mode or "").strip()
    expected_mode = (
        "step4_frozen_wave" if require_frozen_candidate_wave else "step3_exact"
    )
    if controller_mode != expected_mode:
        raise CandidateDiffError(
            "structured_design_action_portfolio_controller_mode_mismatch",
            f"{controller_mode}/{expected_mode}",
        )


    if len(modules) != minimum_atomic_module_count:
        raise CandidateDiffError(
            "structured_design_action_exact_module_count_mismatch",
            "bounded acceptance requires the exact atomic module count",
            details={
                "expected": minimum_atomic_module_count,
                "actual": len(modules),
            },
        )
    if len(seeds) != minimum_sequence_seed_count:
        raise CandidateDiffError(
            "structured_design_action_exact_sequence_seed_count_mismatch",
            "bounded acceptance requires the exact sequence seed count",
            details={
                "expected": minimum_sequence_seed_count,
                "actual": len(seeds),
            },
        )
    expected_portfolio_count = (
        minimum_module_portfolio_count
        + minimum_strict_sequence_seed_portfolio_count
        + minimum_matched_ablation_portfolio_count
    )
    if not require_frozen_candidate_wave and (
        len(portfolio) != expected_portfolio_count
        or any(not item.required or item.count != 1 for item in portfolio)
    ):
        raise CandidateDiffError(
            "structured_design_action_exact_portfolio_count_mismatch",
            "Step3 acceptance requires only exact required count-one entries",
            details={
                "expected": expected_portfolio_count,
                "actual": len(portfolio),
            },
        )

    if not isinstance(position_site_context, Mapping) or not position_site_context:
        raise CandidateDiffError(
            "structured_design_action_step3_position_context_missing",
            "Step3 requires the controller-owned active position catalog",
        )
    if not isinstance(sequence_context, Mapping):
        raise CandidateDiffError(
            "structured_design_action_step3_sequence_context_missing",
            "Step3 requires the controller-owned full sequence context",
        )
    raw_parent = sequence_context.get("parent_sequences")
    raw_immutable = sequence_context.get("immutable_reference_sequences")
    if not isinstance(raw_parent, Mapping) or not isinstance(raw_immutable, Mapping):
        raise CandidateDiffError(
            "structured_design_action_step3_sequence_context_invalid",
            "parent and immutable sequence bundles must both be mappings",
        )
    parent_sequences = {
        str(chain): str(sequence) for chain, sequence in raw_parent.items()
    }
    immutable_sequences = {
        str(chain): str(sequence) for chain, sequence in raw_immutable.items()
    }
    if set(parent_sequences) != set(immutable_sequences) or any(
        len(parent_sequences[chain]) != len(immutable_sequences[chain])
        for chain in parent_sequences
    ):
        raise CandidateDiffError(
            "structured_design_action_step3_sequence_context_invalid",
            "parent and immutable sequence bundle shapes differ",
        )

    site_context = {
        str(key): dict(value)
        for key, value in position_site_context.items()
        if isinstance(value, Mapping)
    }

    planned_coverage: Dict[str, set[Tuple[str, int]]] = {}
    raw_nodes = action.ast_revision_plan.get("structural_nodes")
    if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes)):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping):
                continue
            node_id = str(raw_node.get("node_id") or "")
            selector = raw_node.get("selector")
            if not node_id or not isinstance(selector, Mapping):
                continue
            chain_id = str(selector.get("chain_id") or "")
            spans = selector.get("spans")
            if not chain_id or not isinstance(spans, Sequence):
                continue
            positions: set[Tuple[str, int]] = set()
            for span in spans:
                if (
                    isinstance(span, Sequence)
                    and not isinstance(span, (str, bytes))
                    and len(span) == 2
                ):
                    start, end = int(span[0]), int(span[1])
                    positions.update(
                        (chain_id, position)
                        for position in range(start, end)
                    )
            planned_coverage[node_id] = positions

    if require_frozen_candidate_wave and step4_require_live_node_change:
        before_nodes: Dict[str, set[Tuple[str, int]]] = {}
        raw_before_nodes = (
            step4_current_ast_revision_plan.get("structural_nodes")
            if isinstance(step4_current_ast_revision_plan, Mapping)
            else None
        )
        if isinstance(raw_before_nodes, Sequence) and not isinstance(
            raw_before_nodes, (str, bytes)
        ):
            for raw_node in raw_before_nodes:
                if not isinstance(raw_node, Mapping):
                    continue
                node_id = str(raw_node.get("node_id") or "")
                selector = raw_node.get("selector")
                if not node_id or not isinstance(selector, Mapping):
                    continue
                chain_id = str(selector.get("chain_id") or "")
                positions: set[Tuple[str, int]] = set()
                for span in selector.get("spans", []) or []:
                    if (
                        isinstance(span, Sequence)
                        and not isinstance(span, (str, bytes))
                        and len(span) == 2
                    ):
                        start, end = int(span[0]), int(span[1])
                        positions.update(
                            (chain_id, position)
                            for position in range(start, end)
                        )
                before_nodes[node_id] = positions
        live_changes = {
            node_id: positions
            for node_id, positions in planned_coverage.items()
            if node_id not in before_nodes or positions != before_nodes[node_id]
        }
        additions = {
            node_id: positions - before_nodes.get(node_id, set())
            for node_id, positions in live_changes.items()
        }
        if not live_changes or not any(additions.values()):
            raise CandidateDiffError(
                "structured_design_action_step4_live_node_change_missing",
                "Step4 requires a created/migrated/resized/shifted live Node "
                "with at least one newly selected coordinate",
            )

    def residue_key(residue: Any) -> str:
        return f"{residue.chain_id}:{residue.residue_number}"


    root_lists = {
        chain: list(sequence) for chain, sequence in parent_sequences.items()
    }
    for lineage in action.lineage_actions:
        chain = lineage.residue.chain_id
        position = lineage.residue.zero_based_position
        if chain not in root_lists or not (0 <= position < len(root_lists[chain])):
            raise CandidateDiffError(
                "structured_design_action_step3_lineage_position_invalid",
                residue_key(lineage.residue),
            )
        if lineage.action == "revert_to_wt":
            root_lists[chain][position] = immutable_sequences[chain][position]
        elif lineage.action == "replace":
            root_lists[chain][position] = str(lineage.replacement_residue or "")
    for distribution in action.position_distributions:
        if distribution.required_mutation is not None:
            chain = distribution.residue.chain_id
            position = distribution.residue.zero_based_position
            root_lists[chain][position] = distribution.required_mutation
    root_sequences = {
        chain: "".join(residues) for chain, residues in root_lists.items()
    }

    def site_row(key: str) -> Mapping[str, Any]:
        row = site_context.get(key)
        if not isinstance(row, Mapping):
            raise CandidateDiffError(
                "structured_design_action_step3_residue_outside_active_catalog",
                key,
            )
        return row

    def require_planned_owner_coordinate(
        chain: str,
        position: int,
        key: str,
        row: Mapping[str, Any],
    ) -> str:
        coordinate = (str(chain), int(position))
        if require_frozen_candidate_wave or bool(row.get("prospective", False)):
            planned_owners = sorted(
                node_id
                for node_id, positions in planned_coverage.items()
                if coordinate in positions
            )
            if len(planned_owners) != 1:
                raise CandidateDiffError(
                    "structured_design_action_step4_planned_owner_ambiguous",
                    f"{key}:{planned_owners}",
                )
            return planned_owners[0]
        owner = str(row.get("owner_node_id") or "")
        if coordinate not in planned_coverage.get(owner, set()):
            raise CandidateDiffError(
                "structured_design_action_step3_planned_owner_mismatch",
                f"{key}:{owner}",
            )
        return owner

    def require_planned_owner(residue: Any, row: Mapping[str, Any]) -> str:
        return require_planned_owner_coordinate(
            str(residue.chain_id),
            int(residue.zero_based_position),
            residue_key(residue),
            row,
        )

    def hard_alphabet(row: Mapping[str, Any], key: str) -> set[str]:
        raw = row.get("hard_allowed_residues")
        if not isinstance(raw, list) or not raw:
            raise CandidateDiffError(
                "structured_design_action_step3_hard_alphabet_missing", key
            )
        return {str(item) for item in raw}

    distribution_by_site = {
        residue_key(item.residue): item for item in action.position_distributions
    }

    def validate_output_residue(key: str, residue: str) -> None:
        row = site_row(key)
        if residue not in hard_alphabet(row, key):
            raise CandidateDiffError(
                "structured_design_action_step3_hard_alphabet_violation",
                f"{key}:{residue}",
            )
        distribution = distribution_by_site.get(key)
        if distribution is None:
            return
        if distribution.required_mutation is not None:
            if residue != distribution.required_mutation:
                raise CandidateDiffError(
                    "structured_design_action_step3_required_distribution_violation",
                    f"{key}:{residue}/{distribution.required_mutation}",
                )
        elif (
            residue not in distribution.probabilities
            or float(distribution.probabilities[residue]) <= 0.0
        ):
            raise CandidateDiffError(
                "structured_design_action_step3_position_distribution_violation",
                f"{key}:{residue}",
            )

    modules_by_id = {item.module_id: item for item in modules}
    atomic_modules = [item for item in modules if item.atomic]
    if len(atomic_modules) < minimum_atomic_module_count:
        raise CandidateDiffError(
            "structured_design_action_atomic_module_underfill",
            "not enough all-or-nothing mutation modules",
            details={
                "expected_minimum": minimum_atomic_module_count,
                "actual": len(atomic_modules),
            },
        )
    if any(not item.atomic for item in modules):
        raise CandidateDiffError(
            "structured_design_action_nonatomic_module_forbidden",
            "Step3 executable modules must be atomic",
        )

    cross_node_count = 0
    for module in modules:
        owners: set[str] = set()
        for mutation in module.mutations:
            key = residue_key(mutation.residue)
            row = site_row(key)
            owners.add(require_planned_owner(mutation.residue, row))
            chain = mutation.residue.chain_id
            position = mutation.residue.zero_based_position
            root_residue = root_sequences[chain][position]
            if module.operation != "matched_module_ablation":
                if mutation.from_residue != root_residue:
                    raise CandidateDiffError(
                        "structured_design_action_module_source_mismatch",
                        f"{module.module_id}:{key}:{mutation.from_residue}/{root_residue}",
                    )
                validate_output_residue(key, mutation.to_residue)
        if module.operation != "matched_module_ablation" and len(owners) >= 2:
            cross_node_count += 1
    if cross_node_count < minimum_cross_node_atomic_module_count:
        raise CandidateDiffError(
            "structured_design_action_cross_node_module_underfill",
            "no required atomic treatment module spans multiple Node owners",
            details={
                "expected_minimum": minimum_cross_node_atomic_module_count,
                "actual": cross_node_count,
            },
        )

    if len(seeds) < minimum_sequence_seed_count:
        raise CandidateDiffError(
            "structured_design_action_sequence_seed_underfill",
            "not enough exact full-chain sequence seeds",
            details={
                "expected_minimum": minimum_sequence_seed_count,
                "actual": len(seeds),
            },
        )
    seed_sequence_keys: set[Tuple[Tuple[str, str], ...]] = set()
    for seed in seeds:
        sequences = dict(seed.sequences)
        if set(sequences) != set(root_sequences) or any(
            len(sequences[chain]) != len(root_sequences[chain])
            for chain in root_sequences
        ):
            raise CandidateDiffError(
                "structured_design_action_sequence_seed_shape_mismatch",
                seed.seed_id,
            )
        differences = []
        for chain in sorted(root_sequences):
            for position, (before, after) in enumerate(
                zip(root_sequences[chain], sequences[chain])
            ):
                if before == after:
                    continue
                key = f"{chain}:{position + 1}"
                row = site_row(key)
                require_planned_owner_coordinate(
                    chain, position, key, row
                )
                validate_output_residue(key, after)
                differences.append(key)
        if not differences:
            raise CandidateDiffError(
                "structured_design_action_sequence_seed_not_novel",
                seed.seed_id,
            )
        if not seed.refinement_positions:
            raise CandidateDiffError(
                "structured_design_action_sequence_seed_refinement_missing",
                seed.seed_id,
            )
        refinement_has_alternative = False
        for residue in seed.refinement_positions:
            row = site_row(residue_key(residue))
            require_planned_owner(residue, row)
            key = residue_key(residue)
            distribution = distribution_by_site.get(key)
            if distribution is None:
                enabled = hard_alphabet(row, key)
            elif distribution.required_mutation is not None:
                enabled = {distribution.required_mutation}
            else:
                enabled = {
                    amino_acid
                    for amino_acid, probability in distribution.probabilities.items()
                    if float(probability) > 0.0
                }
            current = sequences[residue.chain_id][
                residue.zero_based_position
            ]
            if any(amino_acid != current for amino_acid in enabled):
                refinement_has_alternative = True
        if not refinement_has_alternative:
            raise CandidateDiffError(
                "structured_design_action_sequence_seed_refinement_exhausted",
                seed.seed_id,
            )
        sequence_key = tuple(sorted(sequences.items()))
        if sequence_key in seed_sequence_keys:
            raise CandidateDiffError(
                "structured_design_action_sequence_seed_duplicate",
                seed.seed_id,
            )
        seed_sequence_keys.add(sequence_key)

    required_entries = [item for item in portfolio if item.required]
    if require_frozen_candidate_wave:
        expected_shape = {
            "module_primary": 1,
            "mcts_primary": 1,
            "strict_sequence_seed_repair": 1,
            "mcts_repair": 1,
            "matched_ablation": 1,
            "mcts_novelty": 1,
            "control": 1,
        }
        if dict(step4_required_portfolio_entry_shape or {}) != expected_shape:
            raise CandidateDiffError(
                "structured_design_action_step4_entry_shape_policy_invalid",
                "controller-owned Step4 entry shape is not canonical",
            )
        observed_shape = {
            "module_primary": sum(
                item.generation_mode == "module"
                and item.role == "primary"
                and item.count == 1
                for item in portfolio
            ),
            "mcts_primary": sum(
                item.generation_mode == "mcts"
                and item.role == "primary"
                and item.count == 1
                for item in portfolio
            ),
            "strict_sequence_seed_repair": sum(
                item.generation_mode == "strict_sequence_seed"
                and item.role == "repair"
                and item.count == 1
                for item in portfolio
            ),
            "mcts_repair": sum(
                item.generation_mode == "mcts"
                and item.role == "repair"
                and item.count == 1
                for item in portfolio
            ),
            "matched_ablation": sum(
                item.generation_mode == "matched_ablation"
                and item.role == "matched_ablation"
                and item.count == 1
                for item in portfolio
            ),
            "mcts_novelty": sum(
                item.generation_mode == "mcts"
                and item.role == "novelty"
                and item.count == 1
                for item in portfolio
            ),
            "control": sum(
                item.generation_mode == "control"
                and item.role == "control"
                and item.count == 1
                for item in portfolio
            ),
        }
        if (
            observed_shape != expected_shape
            or len(portfolio) != 7
            or len(required_entries) != 7
        ):
            raise CandidateDiffError(
                "structured_design_action_step4_entry_shape_mismatch",
                "Step4 requires seven exact portfolio entries and no substitutes",
                details={
                    "expected": expected_shape,
                    "actual": observed_shape,
                    "entry_count": len(portfolio),
                    "required_count": len(required_entries),
                },
            )
        physical_roles: Dict[str, int] = {}
        for item in portfolio:
            physical_count = 2 if item.generation_mode == "matched_ablation" else item.count
            physical_roles[item.role] = physical_roles.get(item.role, 0) + physical_count
        expected_roles = dict(step4_required_role_histogram or {})
        if physical_roles != expected_roles:
            raise CandidateDiffError(
                "structured_design_action_step4_role_histogram_mismatch",
                "compiled physical role histogram differs from the frozen wave",
                details={"expected": expected_roles, "actual": physical_roles},
            )
    module_entries = [
        item for item in required_entries if item.generation_mode == "module"
    ]
    seed_entries = [
        item
        for item in required_entries
        if item.generation_mode == "strict_sequence_seed"
    ]
    matched_entries = [
        item
        for item in required_entries
        if item.generation_mode == "matched_ablation"
    ]
    mode_quotas = (
        (
            "structured_design_action_module_portfolio_underfill",
            module_entries,
            minimum_module_portfolio_count,
        ),
        (
            "structured_design_action_sequence_seed_portfolio_underfill",
            seed_entries,
            minimum_strict_sequence_seed_portfolio_count,
        ),
        (
            "structured_design_action_matched_ablation_portfolio_underfill",
            matched_entries,
            minimum_matched_ablation_portfolio_count,
        ),
    )
    for code, entries, minimum in mode_quotas:
        if len(entries) != minimum:
            raise CandidateDiffError(
                code,
                "required executable portfolio role count is not exact",
                details={"expected": minimum, "actual": len(entries)},
            )


    expected_roles = {
        "module": "primary",
        "strict_sequence_seed": "repair",
        "matched_ablation": "matched_ablation",
    }
    for entry in portfolio:
        expected_role = expected_roles.get(entry.generation_mode)
        if expected_role is not None and entry.role != expected_role:
            raise CandidateDiffError(
                "structured_design_action_exact_portfolio_role_mismatch",
                entry.portfolio_id,
                details={
                    "generation_mode": entry.generation_mode,
                    "expected": expected_role,
                    "actual": entry.role,
                },
            )

    seed_ids = {item.seed_id for item in seeds}
    for entry in portfolio:
        refs = set(entry.source_refs)
        if entry.generation_mode == "module":
            if len(refs) != 1 or not refs.issubset(modules_by_id):
                raise CandidateDiffError(
                    "structured_design_action_module_portfolio_reference_invalid",
                    entry.portfolio_id,
                )
            module = modules_by_id[next(iter(refs))]
            if module.operation == "matched_module_ablation":
                raise CandidateDiffError(
                    "structured_design_action_module_portfolio_reference_invalid",
                    entry.portfolio_id,
                )
        elif entry.generation_mode == "strict_sequence_seed":
            if len(refs) != 1 or not refs.issubset(seed_ids):
                raise CandidateDiffError(
                    "structured_design_action_sequence_seed_portfolio_reference_invalid",
                    entry.portfolio_id,
                )
        elif entry.generation_mode == "matched_ablation":
            if len(refs) != 2 or not refs.issubset(modules_by_id):
                raise CandidateDiffError(
                    "structured_design_action_matched_ablation_reference_invalid",
                    entry.portfolio_id,
                )
            referenced = [modules_by_id[ref] for ref in refs]
            treatments = [
                item
                for item in referenced
                if item.operation != "matched_module_ablation"
            ]
            ablations = [
                item
                for item in referenced
                if item.operation == "matched_module_ablation"
            ]
            if len(treatments) != 1 or len(ablations) != 1:
                raise CandidateDiffError(
                    "structured_design_action_matched_ablation_reference_invalid",
                    entry.portfolio_id,
                )
            treatment = treatments[0]
            ablation = ablations[0]
            treatment_rows = {
                residue_key(item.residue): item for item in treatment.mutations
            }
            ablation_rows = {
                residue_key(item.residue): item for item in ablation.mutations
            }
            if not ablation_rows or not set(ablation_rows) < set(treatment_rows):
                raise CandidateDiffError(
                    "structured_design_action_matched_ablation_not_proper_subset",
                    entry.portfolio_id,
                )
            for key, ablation_row in ablation_rows.items():
                treatment_row = treatment_rows[key]
                chain = ablation_row.residue.chain_id
                position = ablation_row.residue.zero_based_position
                root_residue = root_sequences[chain][position]
                if (
                    ablation_row.from_residue != treatment_row.to_residue
                    or ablation_row.to_residue != root_residue
                ):
                    raise CandidateDiffError(
                        "structured_design_action_matched_ablation_delta_invalid",
                        f"{entry.portfolio_id}:{key}",
                    )
            treatment_lists = {
                chain: list(sequence) for chain, sequence in root_sequences.items()
            }
            for row in treatment.mutations:
                treatment_lists[row.residue.chain_id][
                    row.residue.zero_based_position
                ] = row.to_residue
            treatment_sequences = {
                chain: "".join(residues)
                for chain, residues in sorted(treatment_lists.items())
            }
            control_lists = {
                chain: list(sequence)
                for chain, sequence in treatment_sequences.items()
            }
            for row in ablation.mutations:
                control_lists[row.residue.chain_id][
                    row.residue.zero_based_position
                ] = root_sequences[row.residue.chain_id][
                    row.residue.zero_based_position
                ]
            exact_module_keys = {
                tuple(sorted(treatment_sequences.items())),
                tuple(
                    sorted(
                        (chain, "".join(residues))
                        for chain, residues in control_lists.items()
                    )
                ),
            }
            if seed_sequence_keys & exact_module_keys:
                raise CandidateDiffError(
                    "structured_design_action_sequence_seed_portfolio_duplicate",
                    "the exact seed must differ from treatment and matched control",
                )
        elif entry.generation_mode in {"mcts", "control"}:
            if refs:
                raise CandidateDiffError(
                    "structured_design_action_free_portfolio_reference_invalid",
                    entry.portfolio_id,
                )

    if require_frozen_candidate_wave:
        def apply_module_sequences(module: Any) -> Dict[str, str]:
            lists = {
                chain: list(sequence) for chain, sequence in root_sequences.items()
            }
            for mutation in module.mutations:
                lists[mutation.residue.chain_id][
                    mutation.residue.zero_based_position
                ] = mutation.to_residue
            return {
                chain: "".join(residues)
                for chain, residues in sorted(lists.items())
            }

        exact_sequences: list[Dict[str, str]] = []
        for entry in portfolio:
            refs = set(entry.source_refs)
            if entry.generation_mode == "module":
                exact_sequences.append(
                    apply_module_sequences(modules_by_id[next(iter(refs))])
                )
            elif entry.generation_mode == "strict_sequence_seed":
                seed = next(
                    item for item in seeds if item.seed_id == next(iter(refs))
                )
                exact_sequences.append(dict(seed.sequences))
            elif entry.generation_mode == "matched_ablation":
                referenced = [modules_by_id[ref] for ref in refs]
                treatment = next(
                    item
                    for item in referenced
                    if item.operation != "matched_module_ablation"
                )
                ablation = next(
                    item
                    for item in referenced
                    if item.operation == "matched_module_ablation"
                )
                treatment_sequences = apply_module_sequences(treatment)
                control_lists = {
                    chain: list(sequence)
                    for chain, sequence in treatment_sequences.items()
                }
                for mutation in ablation.mutations:
                    control_lists[mutation.residue.chain_id][
                        mutation.residue.zero_based_position
                    ] = root_sequences[mutation.residue.chain_id][
                        mutation.residue.zero_based_position
                    ]
                exact_sequences.extend(
                    [
                        treatment_sequences,
                        {
                            chain: "".join(residues)
                            for chain, residues in sorted(control_lists.items())
                        },
                    ]
                )
        exact_keys = [tuple(sorted(item.items())) for item in exact_sequences]
        root_key = tuple(sorted(root_sequences.items()))
        if (
            len(exact_keys) != 4
            or len(set(exact_keys)) != 4
            or root_key in set(exact_keys)
        ):
            raise CandidateDiffError(
                "structured_design_action_step4_exact_sequence_collision",
                "the four exact Step4 slots must be novel and pairwise unique",
                details={
                    "exact_count": len(exact_keys),
                    "unique_count": len(set(exact_keys)),
                    "root_collision": root_key in set(exact_keys),
                },
            )
        if step4_require_live_node_change:
            underfilled_nodes = []
            for node_id, _positions in live_changes.items():
                owned_positions = planned_coverage.get(node_id, set())
                engaged = sum(
                    any(
                        candidate[chain][position]
                        != root_sequences[chain][position]
                        for chain, position in owned_positions
                    )
                    for candidate in exact_sequences
                )
                if engaged < 2:
                    underfilled_nodes.append(
                        {"node_id": node_id, "exact_engaged_count": engaged}
                    )
            if underfilled_nodes:
                raise CandidateDiffError(
                    "structured_design_action_step4_changed_node_exact_underfill",
                    "each changed live Node needs two exact candidate realizations",
                    details={"nodes": underfilled_nodes},
                )


def apply_structured_design_action_proposal(
    parent_code: str,
    response: str,
    *,
    expected_parent_binding: Mapping[str, str],
    max_code_length: int | None = None,
    edit_contract_envelope_json: str = "",
    expected_mutation_count_range: Tuple[int, int] | None = None,
    expected_minimum_hamming_distance: int | None = None,
    require_sequence_reconciliation: bool = False,
    minimum_position_distribution_count: int = 0,
    minimum_soft_position_distribution_count: int = 0,
    minimum_required_position_distribution_count: int = 0,
    max_positive_substitution_support_per_position: int | None = None,
    position_distribution_site_context: Mapping[str, Mapping[str, Any]] | None = None,
    require_portfolio_capabilities: bool = False,
    minimum_atomic_module_count: int = 0,
    minimum_cross_node_atomic_module_count: int = 0,
    minimum_sequence_seed_count: int = 0,
    minimum_module_portfolio_count: int = 0,
    minimum_strict_sequence_seed_portfolio_count: int = 0,
    minimum_matched_ablation_portfolio_count: int = 0,
    portfolio_position_site_context: Mapping[str, Mapping[str, Any]] | None = None,
    portfolio_sequence_context: Mapping[str, Mapping[str, str]] | None = None,
    portfolio_controller_mode: str = "step3_exact",
    require_frozen_candidate_wave: bool = False,
    step4_required_role_histogram: Mapping[str, int] | None = None,
    step4_required_portfolio_entry_shape: Mapping[str, int] | None = None,
    step4_free_slot_forced_tiers: Mapping[str, str] | None = None,
    step4_require_live_node_change: bool = False,
    step4_changed_node_min_generated_unique: int = 0,
    step4_changed_node_min_frozen_unique: int = 0,
    step4_changed_node_min_protenix_attempts: int = 0,
    step4_current_ast_revision_plan: Mapping[str, Any] | None = None,
) -> StructuredDesignActionApplication:


    validated = validate_structured_design_action_proposal(
        parent_code,
        response,
        edit_contract_envelope_json=edit_contract_envelope_json,
        expected_parent_binding=expected_parent_binding,
    )
    if (
        expected_mutation_count_range is not None
        and validated.design_action.mutation_count_range
        != tuple(expected_mutation_count_range)
    ):
        raise CandidateDiffError(
            "structured_design_action_mutation_count_range_mismatch",
            "DesignAction mutation_count_range differs from the exact "
            "controller-owned execution range",
            details={
                "expected": list(expected_mutation_count_range),
                "actual": list(validated.design_action.mutation_count_range),
            },
        )
    if (
        expected_minimum_hamming_distance is not None
        and validated.design_action.minimum_hamming_distance
        != expected_minimum_hamming_distance
    ):
        raise CandidateDiffError(
            "structured_design_action_minimum_hamming_distance_mismatch",
            "DesignAction minimum_hamming_distance differs from the exact "
            "controller-owned execution value",
            details={
                "expected": expected_minimum_hamming_distance,
                "actual": validated.design_action.minimum_hamming_distance,
            },
        )
    if require_sequence_reconciliation and not any(
        item.action in {"replace", "revert_to_wt"}
        for item in validated.design_action.lineage_actions
    ):
        raise CandidateDiffError(
            "structured_design_action_step1_sequence_reconciliation_required",
            "Step1 acceptance requires at least one replace or revert_to_wt "
            "lineage action before sequence search and GPU evaluation",
        )
    distributions = validated.design_action.position_distributions


    soft_count = sum(
        item.required_mutation is None
        and sum(float(weight) > 0.0 for weight in item.probabilities.values()) >= 2
        for item in distributions
    )
    required_count = sum(item.required_mutation is not None for item in distributions)
    if max_positive_substitution_support_per_position is not None:
        if (
            isinstance(max_positive_substitution_support_per_position, bool)
            or not isinstance(max_positive_substitution_support_per_position, int)
            or max_positive_substitution_support_per_position < 1
        ):
            raise CandidateDiffError(
                "structured_design_action_position_distribution_support_cap_invalid",
                "controller supplied an invalid exact-position support cap",
            )
        if not isinstance(position_distribution_site_context, Mapping):
            raise CandidateDiffError(
                "structured_design_action_position_distribution_site_context_missing",
                "controller did not supply the parent residue context needed "
                "for pre-MCTS support validation",
            )
        lineage_by_site = {
            f"{item.residue.chain_id}:{item.residue.residue_number}": item
            for item in validated.design_action.lineage_actions
        }
        support_violations = []
        for item in distributions:
            if item.required_mutation is not None:


                continue
            key = f"{item.residue.chain_id}:{item.residue.residue_number}"
            raw_site = position_distribution_site_context.get(key)
            if not isinstance(raw_site, Mapping):
                raise CandidateDiffError(
                    "structured_design_action_position_distribution_site_unknown",
                    f"position distribution site {key} is absent from the "
                    "controller-owned active catalog",
                )
            parent_residue = str(raw_site.get("parent_residue") or "")
            immutable_residue = str(raw_site.get("immutable_residue") or "")
            root_residue = parent_residue
            lineage = lineage_by_site.get(key)
            if lineage is not None:
                if lineage.action == "revert_to_wt":
                    root_residue = immutable_residue
                elif lineage.action == "replace":
                    root_residue = str(lineage.replacement_residue or "")
                else:
                    root_residue = parent_residue
            positive_substitutions = sorted(
                residue
                for residue, probability in item.probabilities.items()
                if float(probability) > 0.0 and residue != root_residue
            )
            if (
                len(positive_substitutions)
                > max_positive_substitution_support_per_position
            ):
                support_violations.append(
                    {
                        "site": key,
                        "reconciled_root_residue": root_residue,
                        "positive_substitution_support": positive_substitutions,
                        "actual": len(positive_substitutions),
                        "maximum": max_positive_substitution_support_per_position,
                    }
                )
        if support_violations:
            raise CandidateDiffError(
                "structured_design_action_position_distribution_top_k_truncation",
                "one or more soft policies exceed the controller-owned "
                "per-position optimizer support cap",
                details={"violations": support_violations},
            )
    if len(distributions) < minimum_position_distribution_count:
        raise CandidateDiffError(
            "structured_design_action_position_distribution_underfill",
            "DesignAction did not satisfy the controller-owned exact-position quota",
            details={
                "expected_minimum": minimum_position_distribution_count,
                "actual": len(distributions),
            },
        )
    if soft_count < minimum_soft_position_distribution_count:
        raise CandidateDiffError(
            "structured_design_action_soft_position_distribution_underfill",
            "DesignAction did not include enough executable soft exact-position "
            "policies with at least two positive-probability residues",
            details={
                "expected_minimum": minimum_soft_position_distribution_count,
                "actual": soft_count,
            },
        )
    if required_count < minimum_required_position_distribution_count:
        raise CandidateDiffError(
            "structured_design_action_required_position_distribution_underfill",
            "DesignAction did not include enough required exact mutations",
            details={
                "expected_minimum": minimum_required_position_distribution_count,
                "actual": required_count,
            },
        )
    if minimum_required_position_distribution_count:
        def _node_coverage(plan_value: Mapping[str, Any]) -> Dict[str, set[Tuple[str, int]]]:
            coverage: Dict[str, set[Tuple[str, int]]] = {}
            for raw_node in plan_value.get("structural_nodes", []) or []:
                if not isinstance(raw_node, Mapping):
                    continue
                node_id = str(raw_node.get("node_id") or "")
                selector = raw_node.get("selector")
                if not node_id or not isinstance(selector, Mapping):
                    continue
                chain_id = str(selector.get("chain_id") or "")
                positions: set[Tuple[str, int]] = set()
                for span in selector.get("spans", []) or []:
                    if (
                        isinstance(span, Sequence)
                        and not isinstance(span, (str, bytes))
                        and len(span) == 2
                    ):
                        start, end = int(span[0]), int(span[1])
                        positions.update(
                            (chain_id, position) for position in range(start, end)
                        )
                coverage[node_id] = positions
            return coverage

        before_coverage = _node_coverage(
            extract_current_ast_revision_plan(parent_code)
        )
        after_coverage = _node_coverage(
            validated.design_action.ast_revision_plan
        )
        changed_nodes = {
            node_id
            for node_id, positions in after_coverage.items()
            if before_coverage.get(node_id) != positions
        }
        if changed_nodes:
            required_coordinates = {
                (item.residue.chain_id, item.residue.zero_based_position)
                for item in distributions
                if item.required_mutation is not None
            }
            engaged_changed_nodes = {
                node_id
                for node_id in changed_nodes
                if after_coverage[node_id] & required_coordinates
            }
            if not engaged_changed_nodes:
                raise CandidateDiffError(
                    "structured_design_action_changed_node_required_mutation_missing",
                    "a created, migrated, or resized Node must own at least one "
                    "required_mutation so the AST change necessarily reaches "
                    "the downstream sequence search",
                    details={"changed_nodes": sorted(changed_nodes)},
                )
    _validate_step3_portfolio_surface(
        validated.design_action,
        require_portfolio_capabilities=require_portfolio_capabilities,
        minimum_atomic_module_count=minimum_atomic_module_count,
        minimum_cross_node_atomic_module_count=(
            minimum_cross_node_atomic_module_count
        ),
        minimum_sequence_seed_count=minimum_sequence_seed_count,
        minimum_module_portfolio_count=minimum_module_portfolio_count,
        minimum_strict_sequence_seed_portfolio_count=(
            minimum_strict_sequence_seed_portfolio_count
        ),
        minimum_matched_ablation_portfolio_count=(
            minimum_matched_ablation_portfolio_count
        ),
        position_site_context=portfolio_position_site_context,
        sequence_context=portfolio_sequence_context,
        portfolio_controller_mode=portfolio_controller_mode,
        require_frozen_candidate_wave=require_frozen_candidate_wave,
        step4_required_role_histogram=step4_required_role_histogram,
        step4_required_portfolio_entry_shape=(
            step4_required_portfolio_entry_shape
        ),
        step4_require_live_node_change=step4_require_live_node_change,
        step4_current_ast_revision_plan=step4_current_ast_revision_plan,
    )
    action_payload = validated.design_action.to_payload()
    plan = action_payload["ast_revision_plan"]
    candidate = _rewrite_validated_ast_plan(
        parent_code,
        plan,
        max_code_length=max_code_length,
    )
    current_plan = extract_current_ast_revision_plan(parent_code)
    summary = (
        "Structured DesignAction proposal: "
        f"design_action_hash={validated.design_action.design_action_hash}; "
        f"ast_changed={plan != current_plan}; "
        f"parent_evolve_hash={validated.parent_evolve_hash[:12]}"
    )
    return StructuredDesignActionApplication(
        code=candidate,
        parent_evolve_hash=validated.parent_evolve_hash,
        design_action=validated.design_action,
        edit_contract_response=deepcopy(validated.edit_contract_response),
        ast_revision_plan=deepcopy(plan),
        changes_summary=summary,
    )


__all__ = [
    "STRUCTURED_DESIGN_ACTION_CONTROLS",
    "STRUCTURED_DESIGN_ACTION_PARENT_BINDING_FIELDS",
    "STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE",
    "STRUCTURED_DESIGN_ACTION_PROPOSAL_VERSION",
    "StructuredDesignActionApplication",
    "StructuredDesignActionEnvelope",
    "apply_structured_design_action_proposal",
    "parse_structured_design_action_response",
    "structured_design_action_artifacts",
    "structured_design_action_repair_instruction",
    "validate_structured_design_action_proposal",
]

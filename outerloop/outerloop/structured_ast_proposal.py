

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import pprint
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from astevolve.semantic_graph.ast_revision import (
    ASTRevisionError,
    GLOBAL_AST_DECISION_ACTIONS,
    GLOBAL_AST_REVISION_PLAN_VERSION,
    normalize_global_ast_revision_plan,
)
from astevolve.runtime.edit_contract_lifecycle import (
    EDIT_CONTRACT_RESPONSE_VERSION,
    EditContractEnvelope,
    EditContractLifecycleError,
    validate_contract_response,
)
from outerloop.island_capabilities import (
    requires_outside_incumbent_exploration,
)
from outerloop.utils.code_utils import CandidateDiffError


STRUCTURED_AST_PROPOSAL_MODE = "structured_strategy_v1"
LEGACY_DIFF_PROPOSAL_MODE = "legacy_diff"
STRUCTURED_AST_PROPOSAL_VERSION = "astevolve.structured_ast_proposal.v1"
STRUCTURED_AST_AUDIT_VERSION = "astevolve.structured_ast_audit.v1"
STRUCTURED_AST_AUDIT_V2_VERSION = "astevolve.structured_ast_audit.v2"
STRUCTURED_AST_PROPOSAL_CONSTRAINTS_VERSION = (
    "astevolve.structured_ast_proposal_constraints.v1"
)


STRUCTURED_AST_DECISION_ACTIONS = GLOBAL_AST_DECISION_ACTIONS
STRUCTURED_AST_CONTROLS: Dict[str, str] = {
    "proposal_mode": STRUCTURED_AST_PROPOSAL_MODE,
    "mutation_surface": "ast_revision_plan_only",
    "plan_schema_version": GLOBAL_AST_REVISION_PLAN_VERSION,
    "rewrite_target": "propose_strategy.ast_revision_plan",
}

_EVOLVE_START = "# EVOLVE-BLOCK-START"
_EVOLVE_END = "# EVOLVE-BLOCK-END"
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "parent_evolve_hash",
        "controls",
        "audit",
        "ast_revision_plan",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "rationale",
        "hypothesis",
        "expected_effects",
        "failure_condition",
        "rollback_condition",
        "edit_contract_response",
    }
)
_AUDIT_V2_FIELDS = frozenset(
    {
        "schema_version",
        "strategy_epoch_id",
        "island_role_id",
        "hypothesis_id",
        "rationale",
        "hypothesis",
        "target_node_ids",
        "target_segments",
        "target_residue_refs",
        "coordinate_system",
        "evidence_refs",
        "expected_effects",
        "absolute_quality_floors",
        "failure_condition",
        "falsification_test",
        "rollback_condition",
        "rollback_target",
        "planned_ablation",
        "edit_contract_response",
    }
)
_EXPECTED_EFFECT_FIELDS = frozenset(
    {"metric_id", "direction", "effect_size", "confidence", "evidence_refs"}
)
_QUALITY_FLOOR_FIELDS = frozenset(
    {"metric_id", "comparison", "direction", "threshold"}
)
_COORDINATE_FIELDS = frozenset(
    {
        "chain_id_scheme",
        "residue_numbering",
        "index_base",
        "segment_source",
        "protected_residues_respected",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


@dataclass(frozen=True)
class StructuredASTApplication:


    code: str
    parent_evolve_hash: str
    plan: Mapping[str, Any]
    audit: Mapping[str, Any]
    edit_contract_response: Mapping[str, Any] | None
    changes_summary: str


def _normalize_newlines(value: str) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _closed_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateDiffError(
            "structured_invalid_type", f"{label} must be a JSON object"
        )
    keys = {str(key) for key in value}
    unknown = sorted(keys - fields)
    missing = sorted(fields - keys)
    if unknown or missing:
        details: Dict[str, Any] = {}
        if unknown:
            details["unknown_fields"] = unknown
        if missing:
            details["missing_fields"] = missing
        raise CandidateDiffError(
            "structured_closed_schema_violation",
            f"{label} must contain exactly {', '.join(sorted(fields))}",
            details=details,
        )
    return value


def _strict_json(text: Any, *, max_bytes: int = 128 * 1024) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise CandidateDiffError(
            "structured_empty_response", "response must be one non-empty JSON object"
        )
    if len(text.encode("utf-8")) > max_bytes:
        raise CandidateDiffError(
            "structured_response_too_large",
            f"response exceeds {max_bytes} UTF-8 bytes",
        )
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise CandidateDiffError(
            "structured_json_only",
            "response must contain only the JSON object, without fences or prose",
        )

    def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise CandidateDiffError(
                    "structured_duplicate_key",
                    f"JSON object repeats key {key!r}",
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise CandidateDiffError(
            "structured_nonfinite_number", f"JSON contains non-finite value {value}"
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
            "structured_invalid_json",
            f"response is invalid JSON at line {error.lineno} column {error.colno}",
        ) from error
    return _closed_mapping(parsed, fields=_ENVELOPE_FIELDS, label="proposal envelope")


def _bounded_text(value: Any, label: str, *, maximum: int = 1500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} must be a non-empty string"
        )
    normalized = value.strip()
    if len(normalized) > maximum:
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} exceeds {maximum} characters"
        )
    return normalized


def _validated_edit_contract_response(
    value: Any,
    *,
    edit_contract_envelope_json: str,
) -> Dict[str, str] | None:


    if not edit_contract_envelope_json:
        if value is not None:
            raise CandidateDiffError(
                "structured_unoffered_contract_response",
                "audit.edit_contract_response must be null when no parent contract is offered",
            )
        return None
    try:
        envelope = EditContractEnvelope.from_json(edit_contract_envelope_json)
    except EditContractLifecycleError as error:
        raise CandidateDiffError(
            "structured_invalid_runtime_contract_envelope",
            "controller-owned parent contract envelope is invalid",
        ) from error
    try:
        return validate_contract_response(value, envelope)
    except EditContractLifecycleError as error:
        raise CandidateDiffError(
            f"structured_{error.code}",
            str(error).split(": ", 1)[-1],
            details={"expected_contract_id": envelope.contract_id},
        ) from error


def _audit(
    value: Any,
    *,
    edit_contract_envelope_json: str,
    hierarchical_design: Mapping[str, Any] | None = None,
    expected_island_role_id: str = "",
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateDiffError(
            "structured_invalid_type", "proposal audit must be a JSON object"
        )
    version = value.get("schema_version")
    if version == STRUCTURED_AST_AUDIT_V2_VERSION:
        return _audit_v2(
            value,
            edit_contract_envelope_json=edit_contract_envelope_json,
            hierarchical_design=hierarchical_design,
            expected_island_role_id=expected_island_role_id,
        )
    raw = _closed_mapping(value, fields=_AUDIT_FIELDS, label="proposal audit")
    if version != STRUCTURED_AST_AUDIT_VERSION:
        raise CandidateDiffError(
            "structured_invalid_audit_schema",
            "audit.schema_version must be either "
            f"{STRUCTURED_AST_AUDIT_VERSION!r} or "
            f"{STRUCTURED_AST_AUDIT_V2_VERSION!r}",
        )
    effects = raw.get("expected_effects")
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise CandidateDiffError(
            "structured_invalid_audit", "audit.expected_effects must be a JSON list"
        )
    normalized_effects = [
        _bounded_text(item, f"audit.expected_effects[{index}]", maximum=500)
        for index, item in enumerate(effects)
    ]
    if not normalized_effects or len(normalized_effects) > 8:
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.expected_effects must contain 1 through 8 concise effects",
        )
    if len(normalized_effects) != len(set(normalized_effects)):
        raise CandidateDiffError(
            "structured_invalid_audit", "audit.expected_effects contains duplicates"
        )
    return {
        "schema_version": STRUCTURED_AST_AUDIT_VERSION,
        "rationale": _bounded_text(raw["rationale"], "audit.rationale"),
        "hypothesis": _bounded_text(raw["hypothesis"], "audit.hypothesis"),
        "expected_effects": normalized_effects,
        "failure_condition": _bounded_text(
            raw["failure_condition"], "audit.failure_condition"
        ),
        "rollback_condition": _bounded_text(
            raw["rollback_condition"], "audit.rollback_condition"
        ),
        "edit_contract_response": _validated_edit_contract_response(
            raw["edit_contract_response"],
            edit_contract_envelope_json=edit_contract_envelope_json,
        ),
    }


def _public_id(value: Any, label: str) -> str:
    text = _bounded_text(value, label, maximum=128)
    if not _PUBLIC_ID.fullmatch(text):
        raise CandidateDiffError(
            "structured_invalid_audit",
            f"{label} must match {_PUBLIC_ID.pattern!r}",
        )
    return text


def _unique_text_list(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = 32,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} must be a JSON list"
        )
    normalized = [
        _public_id(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]
    if not minimum <= len(normalized) <= maximum:
        raise CandidateDiffError(
            "structured_invalid_audit",
            f"{label} must contain {minimum} through {maximum} items",
        )
    if len(normalized) != len(set(normalized)):
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} contains duplicates"
        )
    return normalized


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} must be a finite JSON number"
        )
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise CandidateDiffError(
            "structured_invalid_audit", f"{label} must be a finite JSON number"
        )
    return numeric


def _controller_evidence_ids(
    hierarchical_design: Mapping[str, Any] | None,
) -> set[str]:


    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str) and _PUBLIC_ID.fullmatch(evidence_id):
                found.add(evidence_id)
            values = item.get("evidence_ids")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    if isinstance(value, str) and _PUBLIC_ID.fullmatch(value):
                        found.add(value)
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    if isinstance(hierarchical_design, Mapping):
        visit(hierarchical_design)
    return found


def _strategy_epoch_id(
    hierarchical_design: Mapping[str, Any] | None,
) -> str:
    if not isinstance(hierarchical_design, Mapping):
        return ""
    for source in (
        hierarchical_design,
        hierarchical_design.get("strategy_plan"),
        hierarchical_design.get("global_strategy"),
    ):
        if not isinstance(source, Mapping):
            continue
        for key in ("strategy_epoch_id", "epoch_id"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _audit_v2(
    value: Mapping[str, Any],
    *,
    edit_contract_envelope_json: str,
    hierarchical_design: Mapping[str, Any] | None,
    expected_island_role_id: str,
) -> Dict[str, Any]:
    raw = _closed_mapping(value, fields=_AUDIT_V2_FIELDS, label="proposal audit v2")
    epoch_id = _public_id(raw["strategy_epoch_id"], "audit.strategy_epoch_id")
    controller_epoch = _strategy_epoch_id(hierarchical_design)
    if controller_epoch and epoch_id != controller_epoch:
        raise CandidateDiffError(
            "structured_stale_strategy_epoch",
            "audit.strategy_epoch_id does not match the frozen generation strategy",
            details={"expected_strategy_epoch_id": controller_epoch},
        )
    island_role_id = _public_id(raw["island_role_id"], "audit.island_role_id")
    expected_role = str(expected_island_role_id or "").strip()
    if expected_role and island_role_id != expected_role:
        raise CandidateDiffError(
            "structured_island_role_mismatch",
            "audit.island_role_id does not match the reserved island role",
            details={"expected_island_role_id": expected_role},
        )
    evidence_refs = _unique_text_list(
        raw["evidence_refs"], "audit.evidence_refs", minimum=1, maximum=32
    )
    published_evidence = _controller_evidence_ids(hierarchical_design)
    unknown = sorted(set(evidence_refs) - published_evidence)
    if published_evidence and unknown:
        raise CandidateDiffError(
            "structured_unknown_evidence_ref",
            "audit.evidence_refs contains identities absent from controller evidence",
            details={"unknown_evidence_refs": unknown},
        )

    effects_value = raw["expected_effects"]
    if isinstance(effects_value, (str, bytes)) or not isinstance(
        effects_value, Sequence
    ):
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.expected_effects must be a JSON list",
        )
    effects = []
    for index, item in enumerate(effects_value):
        effect = _closed_mapping(
            item,
            fields=_EXPECTED_EFFECT_FIELDS,
            label=f"audit.expected_effects[{index}]",
        )
        direction = str(effect["direction"])
        if direction not in {"increase", "decrease", "maintain"}:
            raise CandidateDiffError(
                "structured_invalid_audit",
                f"audit.expected_effects[{index}].direction is invalid",
            )
        effect_size = str(effect["effect_size"])
        if effect_size not in {"small", "medium", "large", "unknown"}:
            raise CandidateDiffError(
                "structured_invalid_audit",
                f"audit.expected_effects[{index}].effect_size is invalid",
            )
        confidence = _finite_number(
            effect["confidence"], f"audit.expected_effects[{index}].confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise CandidateDiffError(
                "structured_invalid_audit",
                f"audit.expected_effects[{index}].confidence must be in [0, 1]",
            )
        effect_refs = _unique_text_list(
            effect["evidence_refs"],
            f"audit.expected_effects[{index}].evidence_refs",
            minimum=1,
            maximum=16,
        )
        if not set(effect_refs) <= set(evidence_refs):
            raise CandidateDiffError(
                "structured_unbound_effect_evidence",
                f"audit.expected_effects[{index}] cites evidence absent from audit.evidence_refs",
            )
        effects.append(
            {
                "metric_id": _public_id(
                    effect["metric_id"],
                    f"audit.expected_effects[{index}].metric_id",
                ),
                "direction": direction,
                "effect_size": effect_size,
                "confidence": confidence,
                "evidence_refs": effect_refs,
            }
        )
    if not 1 <= len(effects) <= 8:
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.expected_effects must contain 1 through 8 effects",
        )

    floors_value = raw["absolute_quality_floors"]
    if isinstance(floors_value, (str, bytes)) or not isinstance(
        floors_value, Sequence
    ):
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.absolute_quality_floors must be a JSON list",
        )
    floors = []
    for index, item in enumerate(floors_value):
        floor = _closed_mapping(
            item,
            fields=_QUALITY_FLOOR_FIELDS,
            label=f"audit.absolute_quality_floors[{index}]",
        )
        comparison = str(floor["comparison"])
        direction = str(floor["direction"])
        if comparison not in {"absolute", "relative_to_parent"}:
            raise CandidateDiffError(
                "structured_invalid_audit",
                f"audit.absolute_quality_floors[{index}].comparison is invalid",
            )
        if direction not in {"min", "max"}:
            raise CandidateDiffError(
                "structured_invalid_audit",
                f"audit.absolute_quality_floors[{index}].direction is invalid",
            )
        floors.append(
            {
                "metric_id": _public_id(
                    floor["metric_id"],
                    f"audit.absolute_quality_floors[{index}].metric_id",
                ),
                "comparison": comparison,
                "direction": direction,
                "threshold": _finite_number(
                    floor["threshold"],
                    f"audit.absolute_quality_floors[{index}].threshold",
                ),
            }
        )
    if not 1 <= len(floors) <= 8:
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.absolute_quality_floors must contain 1 through 8 floors",
        )

    rollback_target = str(raw["rollback_target"])
    if rollback_target != "selected_parent":
        raise CandidateDiffError(
            "structured_invalid_audit",
            "audit.rollback_target must be 'selected_parent'",
        )
    coordinates = _closed_mapping(
        raw["coordinate_system"],
        fields=_COORDINATE_FIELDS,
        label="audit.coordinate_system",
    )
    expected_coordinates = {
        "chain_id_scheme": "runtime_chain_id",
        "residue_numbering": "case_sequence",
        "index_base": 1,
        "segment_source": "migration_frontier",
        "protected_residues_respected": True,
    }
    if dict(coordinates) != expected_coordinates:
        raise CandidateDiffError(
            "structured_coordinate_protocol_mismatch",
            "audit.coordinate_system must exactly match the controller coordinate protocol",
            details={"required_coordinate_system": expected_coordinates},
        )
    return {
        "schema_version": STRUCTURED_AST_AUDIT_V2_VERSION,
        "strategy_epoch_id": epoch_id,
        "island_role_id": island_role_id,
        "hypothesis_id": _public_id(raw["hypothesis_id"], "audit.hypothesis_id"),
        "rationale": _bounded_text(raw["rationale"], "audit.rationale"),
        "hypothesis": _bounded_text(raw["hypothesis"], "audit.hypothesis"),
        "target_node_ids": _unique_text_list(
            raw["target_node_ids"],
            "audit.target_node_ids",
            minimum=1,
            maximum=16,
        ),
        "target_segments": _unique_text_list(
            raw["target_segments"], "audit.target_segments", minimum=1, maximum=16
        ),
        "target_residue_refs": _unique_text_list(
            raw["target_residue_refs"],
            "audit.target_residue_refs",
            minimum=1,
            maximum=32,
        ),
        "coordinate_system": expected_coordinates,
        "evidence_refs": evidence_refs,
        "expected_effects": effects,
        "absolute_quality_floors": floors,
        "failure_condition": _bounded_text(
            raw["failure_condition"], "audit.failure_condition"
        ),
        "falsification_test": _bounded_text(
            raw["falsification_test"], "audit.falsification_test"
        ),
        "rollback_condition": _bounded_text(
            raw["rollback_condition"], "audit.rollback_condition"
        ),
        "rollback_target": rollback_target,
        "planned_ablation": _bounded_text(
            raw["planned_ablation"], "audit.planned_ablation"
        ),
        "edit_contract_response": _validated_edit_contract_response(
            raw["edit_contract_response"],
            edit_contract_envelope_json=edit_contract_envelope_json,
        ),
    }


def _evolve_block_span(code: str) -> Tuple[int, int, str]:


    lines = code.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines) if line.strip() == _EVOLVE_START
    ]
    ends = [index for index, line in enumerate(lines) if line.strip() == _EVOLVE_END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise CandidateDiffError(
            "structured_invalid_evolve_block",
            "parent must contain exactly one ordered EVOLVE block",
        )
    for index, line in enumerate(lines):
        if "EVOLVE-BLOCK-START" in line and line.strip() != _EVOLVE_START:
            raise CandidateDiffError(
                "structured_invalid_evolve_block",
                f"invalid EVOLVE start marker on line {index + 1}",
            )
        if "EVOLVE-BLOCK-END" in line and line.strip() != _EVOLVE_END:
            raise CandidateDiffError(
                "structured_invalid_evolve_block",
                f"invalid EVOLVE end marker on line {index + 1}",
            )
    start = sum(len(line) for line in lines[: starts[0]])
    end = sum(len(line) for line in lines[: ends[0] + 1])
    return start, end, code[start:end]


def parent_evolve_hash(parent_code: str) -> str:


    code = _normalize_newlines(parent_code)
    _start, _end, block = _evolve_block_span(code)
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def _source_offsets(code: str, node: ast.AST) -> Tuple[int, int, int]:
    if not all(
        hasattr(node, field)
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        raise CandidateDiffError(
            "structured_source_location_missing",
            "Python parser did not expose an exact ast_revision_plan source location",
        )
    lines = code.splitlines(keepends=True)

    def offset(line_number: int, byte_column: int) -> Tuple[int, int]:
        prefix = lines[line_number - 1].encode("utf-8")[:byte_column].decode("utf-8")
        return sum(len(line) for line in lines[: line_number - 1]) + len(prefix), len(prefix)

    start, start_column = offset(int(node.lineno), int(node.col_offset))
    end, _end_column = offset(int(node.end_lineno), int(node.end_col_offset))
    return start, end, start_column


def _plan_location(code: str) -> Tuple[ast.AST, int, int, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise CandidateDiffError(
            "structured_invalid_parent_python", "parent program is not valid Python"
        ) from error
    block_start, block_end, _block = _evolve_block_span(code)
    candidates: list[Tuple[ast.AST, int, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value != "ast_revision_plan":
                continue
            start, end, column = _source_offsets(code, value)
            if start >= block_start and end <= block_end:
                candidates.append((value, start, end, column))
    if len(candidates) != 1:
        raise CandidateDiffError(
            "structured_ambiguous_plan_target",
            "parent EVOLVE block must contain exactly one literal ast_revision_plan value",
            details={"match_count": len(candidates)},
        )
    return candidates[0]


def extract_current_ast_revision_plan(parent_code: str) -> Dict[str, Any]:


    code = _normalize_newlines(parent_code)
    value_node, _start, _end, _column = _plan_location(code)
    try:
        raw = ast.literal_eval(value_node)
    except (TypeError, ValueError) as error:
        raise CandidateDiffError(
            "structured_nonliteral_plan",
            "parent ast_revision_plan must be a Python literal",
        ) from error
    try:
        return normalize_global_ast_revision_plan(raw)
    except ASTRevisionError as error:
        raise CandidateDiffError(
            "structured_invalid_parent_plan", str(error)
        ) from error


def _artifact_mapping(value: Any) -> Mapping[str, Any] | None:


    if isinstance(value, Mapping):
        return value
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _artifact_policy_limits(
    program_artifacts: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:


    if not isinstance(program_artifacts, Mapping):
        return None
    reports = [_artifact_mapping(program_artifacts.get("ast_revision_report"))]
    summary = _artifact_mapping(program_artifacts.get("llm_feedback_summary"))
    if summary is not None:
        reports.append(_artifact_mapping(summary.get("ast_revision_report")))
    for report in reports:
        limits = report.get("policy_limits") if report is not None else None
        if isinstance(limits, Mapping):
            return limits
    return None


def _constraint_integer(value: Any, *, minimum: int = 1) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _constraint_names(value: Any) -> list[str] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    names = [str(item).strip() for item in value]
    if not names or any(not item or len(item) > 128 for item in names):
        return None
    if len(names) != len(set(names)):
        return None
    return names


def _constraint_action_profiles(value: Any) -> Dict[str, Any] | None:


    if not isinstance(value, Mapping) or not value:
        return None
    profiles: Dict[str, Any] = {}
    for raw_name, raw_profile in value.items():
        name = str(raw_name).strip()
        if not name or len(name) > 128 or name in profiles:
            return None


        actions_value = (
            raw_profile.get("allowed_actions")
            if isinstance(raw_profile, Mapping)
            else raw_profile
        )
        if isinstance(actions_value, (str, bytes)) or not isinstance(
            actions_value, Sequence
        ):
            return None
        projected_actions: list[Dict[str, Any]] = []
        seen_operators: set[str] = set()
        for raw_action in actions_value:
            if not isinstance(raw_action, Mapping):
                return None
            operator = str(raw_action.get("operator") or "").strip()
            budget = raw_action.get("budget")
            if (
                not operator
                or len(operator) > 64
                or operator in seen_operators
                or not isinstance(budget, Mapping)
            ):
                return None
            minimum = _constraint_integer(budget.get("min"))
            maximum = _constraint_integer(budget.get("max"))
            if minimum is None or maximum is None or minimum > maximum:
                return None
            seen_operators.add(operator)
            projected_actions.append(
                {
                    "operator": operator,
                    "budget": {"min": minimum, "max": maximum},
                }
            )
        if not projected_actions:
            return None
        profiles[name] = projected_actions
    return profiles


def structured_ast_proposal_constraints(
    program_artifacts: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    projection: Dict[str, Any] = {
        "schema_version": STRUCTURED_AST_PROPOSAL_CONSTRAINTS_VERSION,
        "decision_actions": list(STRUCTURED_AST_DECISION_ACTIONS),
    }
    limits = _artifact_policy_limits(program_artifacts)
    if limits is None:
        return projection

    node_pattern = limits.get("node_id_pattern")
    edge_pattern = limits.get("edge_id_pattern")
    functional_ids = _constraint_names(limits.get("allowed_functional_node_ids"))
    action_profiles = _constraint_action_profiles(limits.get("action_profiles"))
    budget_fields = (
        "min_total_editable_positions",
        "max_total_editable_positions",
        "min_active_structural_nodes",
        "max_active_structural_nodes",
        "max_positions_per_node",
        "max_spans_per_node",
        "max_mapping_edges",
    )
    budgets = {
        field: _constraint_integer(limits.get(field)) for field in budget_fields
    }
    if (
        not isinstance(node_pattern, str)
        or not node_pattern
        or len(node_pattern) > 256
        or not isinstance(edge_pattern, str)
        or not edge_pattern
        or len(edge_pattern) > 256
        or functional_ids is None
        or action_profiles is None
        or any(value is None for value in budgets.values())
    ):
        return projection
    try:
        re.compile(node_pattern)
        re.compile(edge_pattern)
    except re.error:
        return projection
    if (
        budgets["min_total_editable_positions"]
        > budgets["max_total_editable_positions"]
        or budgets["min_active_structural_nodes"]
        > budgets["max_active_structural_nodes"]
    ):
        return projection

    projection.update(
        {
            "node_id_pattern": node_pattern,
            "edge_id_pattern": edge_pattern,
            "allowed_functional_node_ids": functional_ids,
            "action_profiles": action_profiles,
            "budgets": budgets,
        }
    )
    immutable_ids = limits.get("immutable_structural_node_ids")
    if isinstance(immutable_ids, Sequence) and not isinstance(
        immutable_ids, (str, bytes)
    ):
        normalized_immutable = [str(item).strip() for item in immutable_ids]
        if all(normalized_immutable) and len(normalized_immutable) == len(
            set(normalized_immutable)
        ):
            projection["immutable_structural_node_ids"] = normalized_immutable
    safety = {
        "require_catalog_eligibility": limits.get("require_catalog_eligibility"),
        "require_position_evidence_refs": limits.get(
            "require_position_evidence_refs"
        ),
        "min_position_safety_score": limits.get("min_position_safety_score"),
    }
    if (
        isinstance(safety["require_catalog_eligibility"], bool)
        and isinstance(safety["require_position_evidence_refs"], bool)
        and isinstance(safety["min_position_safety_score"], (int, float))
        and not isinstance(safety["min_position_safety_score"], bool)
        and 0.0 <= float(safety["min_position_safety_score"]) <= 1.0
    ):
        safety["min_position_safety_score"] = float(
            safety["min_position_safety_score"]
        )
        projection["residue_safety"] = safety
    return projection


def structured_ast_prompt_context(
    parent_code: str,
    *,
    contract_handoff: Mapping[str, Any] | None = None,
    proposal_constraints: Mapping[str, Any] | None = None,
) -> str:


    payload = {
        "parent_evolve_hash": parent_evolve_hash(parent_code),
        "required_controls": deepcopy(STRUCTURED_AST_CONTROLS),
        "proposal_constraints": deepcopy(
            dict(
                proposal_constraints
                or structured_ast_proposal_constraints(None)
            )
        ),
        "current_ast_revision_plan": extract_current_ast_revision_plan(parent_code),
        "edit_contract_handoff": (
            deepcopy(dict(contract_handoff))
            if isinstance(contract_handoff, Mapping)
            else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def structured_ast_audit_prompt_contract(version: str) -> str:


    if version == STRUCTURED_AST_AUDIT_V2_VERSION:
        return (
            "`audit`: a concise closed object using schema "
            f"`{STRUCTURED_AST_AUDIT_V2_VERSION}` with exactly: "
            "`schema_version`, `strategy_epoch_id`, `island_role_id`, "
            "`hypothesis_id`, `rationale`, `hypothesis`, `target_node_ids`, "
            "`target_segments`, `target_residue_refs`, `coordinate_system`, `evidence_refs`, "
            "`expected_effects`, `absolute_quality_floors`, "
            "`failure_condition`, `falsification_test`, `rollback_condition`, "
            "`rollback_target`, `planned_ablation`, and "
            "`edit_contract_response`. Copy the frozen strategy epoch and "
            "assigned role exactly. Cite only supplied evidence IDs. Each "
            "free-text audit field (`rationale`, `hypothesis`, "
            "`failure_condition`, `falsification_test`, `rollback_condition`, "
            "and `planned_ablation`) must be concise and at most 1500 "
            "characters. Each "
            "expected effect must contain exactly `metric_id`, `direction` "
            "(`increase|decrease|maintain`), `effect_size` "
            "(`small|medium|large|unknown`), numeric `confidence` in [0,1], "
            "and `evidence_refs`. Each absolute quality floor must contain "
            "exactly `metric_id`, `comparison` "
            "(`absolute|relative_to_parent`), `direction` (`min|max`), and a "
            "finite numeric `threshold`. For `relative_to_parent`, threshold "
            "is the allowed child/parent ratio. Target node IDs and residues "
            "must match the submitted AST plan exactly. Set `coordinate_system` "
            "exactly to {\"chain_id_scheme\":\"runtime_chain_id\","
            "\"residue_numbering\":\"case_sequence\",\"index_base\":1,"
            "\"segment_source\":\"migration_frontier\","
            "\"protected_residues_respected\":true}. Set `rollback_target` exactly to "
            "`selected_parent`. AST selector spans remain 0-based half-open; audit "
            "residue refs and the frozen migration frontier are 1-based. For example, "
            "selector span [70,72) binds audit residues P:71 and P:72."
        )
    if version != STRUCTURED_AST_AUDIT_VERSION:
        raise ValueError(f"unsupported structured audit version: {version!r}")
    return (
        "`audit`: a concise closed object using schema "
        f"`{STRUCTURED_AST_AUDIT_VERSION}` with exactly `schema_version`, "
        "`rationale`, `hypothesis`, `expected_effects`, "
        "`failure_condition`, `rollback_condition`, and "
        "`edit_contract_response`. Keep every free-text audit field concise "
        "and at most 1500 characters."
    )


def structured_ast_repair_instruction(
    error: BaseException,
    *,
    parent_code: str,
    edit_contract_envelope_json: str = "",
    proposal_constraints: Mapping[str, Any] | None = None,
    previous_errors: Sequence[BaseException] = (),
    required_audit_version: str = "",
) -> str:


    rejections = []
    seen_rejections: set[tuple[str, str]] = set()
    for rejection in (*tuple(previous_errors), error):
        code = str(
            getattr(rejection, "code", None) or type(rejection).__name__
        ).strip()
        message = str(
            getattr(rejection, "message", None) or str(rejection)
        ).strip()
        if len(code) > 128:
            code = code[:125] + "..."
        if len(message) > 600:
            message = message[:597] + "..."
        identity = (code, message)
        if identity in seen_rejections:
            continue
        seen_rejections.add(identity)
        rejections.append({"code": code, "message": message})
    controls = json.dumps(
        STRUCTURED_AST_CONTROLS, sort_keys=True, separators=(",", ":")
    )
    constraints = json.dumps(
        dict(proposal_constraints or structured_ast_proposal_constraints(None)),
        sort_keys=True,
        separators=(",", ":"),
    )
    rejection_text = json.dumps(
        rejections, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if edit_contract_envelope_json:
        offered = EditContractEnvelope.from_json(edit_contract_envelope_json)
        response_rule = (
            "audit.edit_contract_response must contain exactly schema_version="
            f"{EDIT_CONTRACT_RESPONSE_VERSION}, contract_id={offered.contract_id}, "
            "decision, and reason."
        )
    else:
        response_rule = "audit.edit_contract_response must be null (no offer)."
    audit_rule = (
        f"audit.schema_version must be exactly {required_audit_version}. "
        if required_audit_version
        else "audit.schema_version must use one supported structured audit schema. "
    )
    return (
        "STRUCTURED AST PROPOSAL REPAIR REQUIRED. Every unique rejection in this "
        f"proposal round is listed here: previous_rejections={rejection_text}. "
        "Resolve all of them and regenerate from the unchanged parent. "
        f"Use parent_evolve_hash={parent_evolve_hash(parent_code)} and controls={controls}. "
        f"Obey proposal_constraints={constraints} exactly. decision_record.action "
        "must be one of keep/create/delete/migrate/resize/revise/rewire/mixed/revert; "
        "keep is valid, retain is not. "
        f"Return ONLY one {STRUCTURED_AST_PROPOSAL_VERSION} JSON object with the "
        "closed envelope fields shown in the original prompt. ast_revision_plan "
        "must contain exactly schema_version, structural_nodes, mapping_edges, and "
        "decision_record; never put last_contract_response or edit_contract_response "
        "When revising AST, preserve every existing functional-node mapping and "
        "every existing valid mapping edge unless the action explicitly deletes "
        "its target; never emit a partial mapping list. Evidence refs must be "
        "copied verbatim from the supplied catalog/evidence IDs, never invented. "
        f"inside that plan or Python source. {audit_rule}{response_rule} No Markdown, prose, "
        "SEARCH/REPLACE blocks, hidden chain-of-thought, or provider/runtime data."
    )


def _validate_envelope(
    parent_code: str,
    response: str,
    *,
    edit_contract_envelope_json: str,
    hierarchical_design: Mapping[str, Any] | None = None,
    expected_island_role_id: str = "",
    required_audit_version: str = "",
    proposal_critic_enabled: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    envelope = _strict_json(response)
    if envelope.get("schema_version") != STRUCTURED_AST_PROPOSAL_VERSION:
        raise CandidateDiffError(
            "structured_invalid_schema",
            f"schema_version must be {STRUCTURED_AST_PROPOSAL_VERSION!r}",
        )
    supplied_hash = envelope.get("parent_evolve_hash")
    if not isinstance(supplied_hash, str) or not _SHA256.fullmatch(supplied_hash):
        raise CandidateDiffError(
            "structured_invalid_parent_hash",
            "parent_evolve_hash must be 64 lowercase hexadecimal characters",
        )
    expected_hash = parent_evolve_hash(parent_code)
    if supplied_hash != expected_hash:
        raise CandidateDiffError(
            "structured_stale_parent_hash",
            "parent_evolve_hash does not match the unchanged selected parent",
            details={"expected_parent_evolve_hash": expected_hash},
        )
    controls = envelope.get("controls")
    if controls != STRUCTURED_AST_CONTROLS:
        raise CandidateDiffError(
            "structured_controls_mismatch",
            "controls must exactly match the controller-owned required_controls",
            details={"required_controls": deepcopy(STRUCTURED_AST_CONTROLS)},
        )
    audit = _audit(
        envelope.get("audit"),
        edit_contract_envelope_json=edit_contract_envelope_json,
        hierarchical_design=hierarchical_design,
        expected_island_role_id=expected_island_role_id,
    )
    if required_audit_version and audit["schema_version"] != required_audit_version:
        raise CandidateDiffError(
            "structured_required_audit_schema_mismatch",
            "audit.schema_version does not match the controller-required version",
            details={"required_audit_version": required_audit_version},
        )
    try:
        plan = normalize_global_ast_revision_plan(envelope.get("ast_revision_plan"))
    except ASTRevisionError as error:
        raise CandidateDiffError(
            "structured_invalid_ast_plan", str(error)
        ) from error
    if (
        audit["schema_version"] == STRUCTURED_AST_AUDIT_V2_VERSION
        and proposal_critic_enabled
    ):
        _critic_v2(
            audit,
            parent_plan=extract_current_ast_revision_plan(parent_code),
            candidate_plan=plan,
            hierarchical_design=hierarchical_design,
        )
    decision = plan["decision_record"]


    public_effects = audit["expected_effects"]
    if audit["schema_version"] == STRUCTURED_AST_AUDIT_V2_VERSION:
        public_effects = [
            (
                f"{effect['metric_id']} {effect['direction']} "
                f"({effect['effect_size']}; confidence={effect['confidence']:.3f}; "
                f"evidence={','.join(effect['evidence_refs'])})"
            )
            for effect in audit["expected_effects"]
        ]
    bindings = {
        "hypothesis": audit["hypothesis"],
        "expected_effects": public_effects,
        "failure_condition": audit["failure_condition"],
    }
    decision.update(deepcopy(bindings))
    decision["rationale"] = audit["rationale"]
    decision["rollback_condition"] = audit["rollback_condition"]
    try:
        plan = normalize_global_ast_revision_plan(plan)
    except ASTRevisionError as error:
        raise CandidateDiffError(
            "structured_invalid_bound_ast_plan", str(error)
        ) from error
    return plan, audit


def _critic_v2(
    audit: Mapping[str, Any],
    *,
    parent_plan: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    hierarchical_design: Mapping[str, Any] | None,
) -> None:


    def node_map(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
        return {
            str(node["node_id"]): node
            for node in plan.get("structural_nodes", [])
            if isinstance(node, Mapping) and node.get("node_id")
        }

    parent_nodes = node_map(parent_plan)
    candidate_nodes = node_map(candidate_plan)
    nodes = {**parent_nodes, **candidate_nodes}
    targets = set(map(str, audit.get("target_node_ids", [])))
    unknown_nodes = sorted(targets - set(nodes))
    if unknown_nodes:
        raise CandidateDiffError(
            "structured_critic_unknown_target_node",
            "audit target nodes are absent from both parent and candidate plans",
            details={"unknown_target_node_ids": unknown_nodes},
        )

    def related_edges(plan: Mapping[str, Any], node_id: str) -> list[Mapping[str, Any]]:
        return sorted(
            [
                dict(edge)
                for edge in plan.get("mapping_edges", [])
                if isinstance(edge, Mapping)
                and str(edge.get("structural_node_id") or "") == node_id
            ],
            key=lambda edge: str(edge.get("edge_id") or ""),
        )

    changed_targets = {
        node_id
        for node_id in targets
        if parent_nodes.get(node_id) != candidate_nodes.get(node_id)
        or related_edges(parent_plan, node_id)
        != related_edges(candidate_plan, node_id)
    }
    if not changed_targets:
        raise CandidateDiffError(
            "structured_critic_target_node_unchanged",
            "audit target nodes and their mapping edges are unchanged from the parent",
            details={"target_node_ids": sorted(targets)},
        )

    frontier = _find_migration_frontier(hierarchical_design)
    if frontier:
        incumbent_segments = set(map(str, frontier.get("incumbent_segments", [])))
        new_segments = set(map(str, frontier.get("ranked_new_segments", [])))
        segment_rows = [
            item
            for item in frontier.get("segments", [])
            if isinstance(item, Mapping)
        ]
        known_segments = incumbent_segments | new_segments
        known_segments.update(
            str(item.get("compiled_segment"))
            for item in segment_rows
            if item.get("compiled_segment")
        )
        known_segments.update(
            f"{item.get('chain_id')}:{item.get('compiled_segment')}"
            for item in segment_rows
            if item.get("chain_id") and item.get("compiled_segment")
        )

        def known_segment(value: str) -> bool:
            return value in known_segments or any(
                item.endswith(":" + value) for item in known_segments
            )

        target_segments = set(map(str, audit.get("target_segments", [])))
        unknown_segments = sorted(
            segment for segment in target_segments if not known_segment(segment)
        )
        if unknown_segments:
            raise CandidateDiffError(
                "structured_critic_unknown_target_segment",
                "audit target segments are absent from the frozen migration frontier",
                details={"unknown_target_segments": unknown_segments},
            )

        audit_role_id = str(audit.get("island_role_id"))
        if requires_outside_incumbent_exploration(audit_role_id):
            if not any(
                segment in new_segments
                or any(item.endswith(":" + segment) for item in new_segments)
                for segment in target_segments
            ):
                raise CandidateDiffError(
                    "structured_critic_missing_non_incumbent_segment",
                    f"{audit_role_id} must target a ranked non-incumbent segment",
                    details={"ranked_new_segments": sorted(new_segments)},
                )
            position_segments = dict(
                frontier.get("compiled_segment_by_position", {}) or {}
            )
            residue_refs = [
                ":".join(str(item).split(":")[-2:])
                for item in audit.get("target_residue_refs", [])
            ]
            frontier_indexing = str(frontier.get("indexing") or "one_based")

            def frontier_reference(reference: str) -> str:
                fields = reference.rsplit(":", 1)
                if len(fields) != 2:
                    return reference
                try:
                    position = int(fields[1])
                except ValueError:
                    return reference
                if frontier_indexing == "zero_based":
                    position -= 1
                return f"{fields[0]}:{position}"

            outside = 0
            for reference in residue_refs:
                segment = position_segments.get(frontier_reference(reference))
                chain = reference.split(":", 1)[0]
                if segment and (
                    str(segment) in new_segments
                    or f"{chain}:{segment}" in new_segments
                ):
                    outside += 1
                if segment and not (
                    str(segment) in target_segments
                    or f"{chain}:{segment}" in target_segments
                ):
                    raise CandidateDiffError(
                        "structured_critic_residue_segment_mismatch",
                        "audit target residue does not belong to a claimed target segment",
                        details={
                            "target_residue_ref": reference,
                            "frontier_segment": segment,
                            "target_segments": sorted(target_segments),
                        },
                    )
            quota = _island_outside_quota(
                hierarchical_design, str(audit.get("island_role_id"))
            )
            if residue_refs and outside / len(residue_refs) + 1.0e-12 < quota:
                raise CandidateDiffError(
                    "structured_critic_non_incumbent_quota",
                    f"{audit_role_id} target residues do not meet the frozen outside-incumbent quota",
                    details={
                        "outside_count": outside,
                        "target_count": len(residue_refs),
                        "required_quota": quota,
                    },
                )

    selected_positions: set[tuple[str, int]] = set()
    for node_id in targets:
        selector = candidate_nodes.get(node_id, {}).get("selector") or {}
        if not isinstance(selector, Mapping):
            continue
        chain = str(selector.get("chain_id") or "")
        for span in selector.get("spans", []):
            if (
                isinstance(span, Sequence)
                and not isinstance(span, (str, bytes))
                and len(span) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            ):
                selected_positions.update(
                    (chain, position)
                    for position in range(int(span[0]), int(span[1]))
                )
    unbound_residues = []
    for reference in audit.get("target_residue_refs", []):
        fields = str(reference).split(":")
        try:


            identity = (fields[-2], int(fields[-1]) - 1)
        except (IndexError, ValueError):
            unbound_residues.append(str(reference))
            continue
        if identity not in selected_positions:
            unbound_residues.append(str(reference))
    if unbound_residues:
        raise CandidateDiffError(
            "structured_critic_unbound_target_residue",
            "audit target residues are not selected by its target AST nodes",
            details={"unbound_target_residue_refs": sorted(unbound_residues)},
        )

    strategy = (
        hierarchical_design.get("strategy_plan")
        if isinstance(hierarchical_design, Mapping)
        else None
    )
    selection = strategy.get("final_selection") if isinstance(strategy, Mapping) else None
    hard_metrics = {
        str(name)
        for name in (selection.get("hard_gate_metrics", []) if isinstance(selection, Mapping) else [])
        if str(name) != "hard_gate_pass"
    }
    floor_metrics = {
        str(item.get("metric_id"))
        for item in audit.get("absolute_quality_floors", [])
        if isinstance(item, Mapping)
    }
    if hard_metrics and not floor_metrics & hard_metrics:
        raise CandidateDiffError(
            "structured_critic_missing_global_quality_floor",
            "audit quality floors must cover at least one controller global hard-quality metric",
            details={"required_one_of": sorted(hard_metrics)},
        )


def _find_migration_frontier(
    value: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    found: Dict[str, Any] = {}

    def visit(item: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(item, Mapping):
            if item.get("schema_version") == "astevolve.migration_frontier.v1":
                found = deepcopy(dict(item))
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    if isinstance(value, Mapping):
        visit(value)
    return found


def _island_outside_quota(
    hierarchical_design: Mapping[str, Any] | None,
    role_id: str,
) -> float:
    strategy = (
        hierarchical_design.get("strategy_plan")
        if isinstance(hierarchical_design, Mapping)
        else None
    )
    directives = (
        strategy.get("island_directives", [])
        if isinstance(strategy, Mapping)
        else []
    )
    for directive in directives:
        if not isinstance(directive, Mapping) or directive.get("role_id") != role_id:
            continue
        value = directive.get("outside_incumbent_quota")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(1.0, float(value)))
    return 0.0


def apply_structured_ast_proposal(
    parent_code: str,
    response: str,
    *,
    max_code_length: int | None = None,
    edit_contract_envelope_json: str = "",
    hierarchical_design: Mapping[str, Any] | None = None,
    expected_island_role_id: str = "",
    required_audit_version: str = "",
    proposal_critic_enabled: bool = True,
) -> StructuredASTApplication:


    code = _normalize_newlines(parent_code)
    plan, audit = _validate_envelope(
        code,
        response,
        edit_contract_envelope_json=edit_contract_envelope_json,
        hierarchical_design=hierarchical_design,
        expected_island_role_id=expected_island_role_id,
        required_audit_version=required_audit_version,
        proposal_critic_enabled=proposal_critic_enabled,
    )
    current = extract_current_ast_revision_plan(code)

    def scientific_value(value: Mapping[str, Any]) -> Dict[str, Any]:
        detached = deepcopy(dict(value))
        decision = detached.get("decision_record")
        if isinstance(decision, dict):
            decision.pop("rationale", None)
            decision.pop("rollback_condition", None)
        return detached

    if scientific_value(plan) == scientific_value(current):
        raise CandidateDiffError(
            "structured_no_effect",
            "ast_revision_plan has no scientific change from the selected parent plan",
        )
    _node, start, end, start_column = _plan_location(code)
    rendered = pprint.pformat(
        plan,
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
                "structured_generated_source_too_large",
                "validated plan renders to "
                f"{len(candidate)} characters; controller limit is {max_code_length}. "
                "Shorten audit/intent text or reduce redundant nodes and edges",
            )
    try:
        compile(candidate, "<structured-ast-proposal>", "exec")
    except SyntaxError as error:
        raise CandidateDiffError(
            "structured_generated_invalid_python",
            "deterministic AST plan rendering produced invalid Python",
        ) from error
    if extract_current_ast_revision_plan(candidate) != plan:
        raise CandidateDiffError(
            "structured_generated_plan_mismatch",
            "deterministic rewrite did not round-trip the validated plan",
        )
    decision = plan["decision_record"]
    summary = (
        "Structured AST proposal: "
        f"action={decision['action']}; nodes={len(plan['structural_nodes'])}; "
        f"edges={len(plan['mapping_edges'])}; parent_evolve_hash={parent_evolve_hash(code)[:12]}"
    )
    return StructuredASTApplication(
        code=candidate,
        parent_evolve_hash=parent_evolve_hash(code),
        plan=deepcopy(plan),
        audit=deepcopy(audit),
        edit_contract_response=deepcopy(audit["edit_contract_response"]),
        changes_summary=summary,
    )


__all__ = [
    "LEGACY_DIFF_PROPOSAL_MODE",
    "STRUCTURED_AST_AUDIT_VERSION",
    "STRUCTURED_AST_AUDIT_V2_VERSION",
    "STRUCTURED_AST_CONTROLS",
    "STRUCTURED_AST_PROPOSAL_MODE",
    "STRUCTURED_AST_PROPOSAL_CONSTRAINTS_VERSION",
    "STRUCTURED_AST_PROPOSAL_VERSION",
    "STRUCTURED_AST_DECISION_ACTIONS",
    "StructuredASTApplication",
    "apply_structured_ast_proposal",
    "extract_current_ast_revision_plan",
    "parent_evolve_hash",
    "structured_ast_prompt_context",
    "structured_ast_audit_prompt_contract",
    "structured_ast_proposal_constraints",
    "structured_ast_repair_instruction",
]

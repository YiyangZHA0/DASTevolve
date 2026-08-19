

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from astevolve.domain import EditContract


EDIT_CONTRACT_ENVELOPE_VERSION = "astevolve.edit_contract_envelope.v1"
EDIT_CONTRACT_RESPONSE_VERSION = "astevolve.edit_contract_response.v1"
EDIT_CONTRACT_LIFECYCLE_VERSION = "astevolve.edit_contract_lifecycle.v1"
EDIT_CONTRACT_PROMPT_PROJECTION_VERSION = (
    "astevolve.edit_contract_prompt_projection.v1"
)


class EditContractLifecycleError(ValueError):


    def __init__(self, code: str, message: str, *, artifact: Optional[Mapping[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.artifact = deepcopy(dict(artifact or {}))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EditContractEnvelope:


    contract_id: str
    contract_hash: str
    parent_program_id: str
    generated_generation_id: str
    generated_proposal_id: str
    canonical_contract_json: str
    envelope_hash: str
    schema_version: str = EDIT_CONTRACT_ENVELOPE_VERSION

    @classmethod
    def create(
        cls,
        contract: Mapping[str, Any],
        *,
        parent_program_id: str,
        generated_generation_id: str = "",
        generated_proposal_id: str = "",
    ) -> "EditContractEnvelope":
        parent_id = str(parent_program_id or "").strip()
        if not parent_id:
            raise EditContractLifecycleError(
                "parent_program_id_missing",
                "an edit-contract envelope must be bound to a parent program",
            )
        canonical = EditContract.from_mapping(contract).to_dict()
        contract_hash = _sha256_json(canonical)
        contract_id = "ec_" + _sha256_json(
            {
                "parent_program_id": parent_id,
                "contract_hash": contract_hash,
            }
        )[:24]
        core = {
            "schema_version": EDIT_CONTRACT_ENVELOPE_VERSION,
            "contract_id": contract_id,
            "contract_hash": contract_hash,
            "parent_program_id": parent_id,
            "generated_generation_id": str(generated_generation_id or ""),
            "generated_proposal_id": str(generated_proposal_id or ""),
            "contract": canonical,
        }
        return cls(
            contract_id=contract_id,
            contract_hash=contract_hash,
            parent_program_id=parent_id,
            generated_generation_id=core["generated_generation_id"],
            generated_proposal_id=core["generated_proposal_id"],
            canonical_contract_json=_canonical_json(canonical),
            envelope_hash=_sha256_json(core),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_parent_program_id: str = "",
    ) -> "EditContractEnvelope":
        if not isinstance(value, Mapping):
            raise EditContractLifecycleError(
                "contract_envelope_not_mapping", "edit-contract envelope must be a mapping"
            )
        allowed = {
            "schema_version",
            "contract_id",
            "contract_hash",
            "parent_program_id",
            "generated_generation_id",
            "generated_proposal_id",
            "contract",
            "envelope_hash",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise EditContractLifecycleError(
                "contract_envelope_unknown_fields",
                "unknown envelope fields: " + ", ".join(unknown),
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise EditContractLifecycleError(
                "contract_envelope_incomplete",
                "missing envelope fields: " + ", ".join(missing),
            )
        if value.get("schema_version") != EDIT_CONTRACT_ENVELOPE_VERSION:
            raise EditContractLifecycleError(
                "contract_envelope_version_unsupported",
                f"unsupported envelope version {value.get('schema_version')!r}",
            )
        rebuilt = cls.create(
            value.get("contract") if isinstance(value.get("contract"), Mapping) else {},
            parent_program_id=str(value.get("parent_program_id") or ""),
            generated_generation_id=str(value.get("generated_generation_id") or ""),
            generated_proposal_id=str(value.get("generated_proposal_id") or ""),
        )
        for field in ("contract_id", "contract_hash", "envelope_hash"):
            if str(value.get(field) or "") != getattr(rebuilt, field):
                raise EditContractLifecycleError(
                    "contract_envelope_hash_mismatch",
                    f"{field} does not match canonical envelope content",
                )
        expected_parent = str(expected_parent_program_id or "")
        if expected_parent and rebuilt.parent_program_id != expected_parent:
            raise EditContractLifecycleError(
                "contract_envelope_parent_mismatch",
                f"envelope belongs to {rebuilt.parent_program_id!r}, not {expected_parent!r}",
            )
        return rebuilt

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        expected_parent_program_id: str = "",
    ) -> "EditContractEnvelope":
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise EditContractLifecycleError(
                "contract_envelope_json_invalid", "envelope JSON is invalid"
            ) from exc
        return cls.from_mapping(
            parsed,
            expected_parent_program_id=expected_parent_program_id,
        )

    @property
    def contract(self) -> Dict[str, Any]:
        return deepcopy(json.loads(self.canonical_contract_json))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "parent_program_id": self.parent_program_id,
            "generated_generation_id": self.generated_generation_id,
            "generated_proposal_id": self.generated_proposal_id,
            "contract": self.contract,
            "envelope_hash": self.envelope_hash,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def envelope_from_parent_artifacts(
    artifacts: Optional[Mapping[str, Any]],
    *,
    parent_program_id: str,
) -> Optional[EditContractEnvelope]:


    if not isinstance(artifacts, Mapping):
        return None
    value = artifacts.get("edit_contract_envelope")
    if value is None:
        lifecycle = artifacts.get("edit_contract_lifecycle")
        if isinstance(lifecycle, Mapping):
            value = lifecycle.get("generated")
    if value is None:
        return None
    if isinstance(value, str):
        return EditContractEnvelope.from_json(
            value, expected_parent_program_id=parent_program_id
        )
    return EditContractEnvelope.from_mapping(
        value, expected_parent_program_id=parent_program_id
    )


def finalize_generated_contract_artifacts(
    artifacts: Optional[Mapping[str, Any]],
    *,
    parent_program_id: str,
    generated_generation_id: str = "",
    generated_proposal_id: str = "",
) -> Dict[str, Any]:


    output = deepcopy(dict(artifacts or {}))
    contract = output.get("edit_contract")
    if not isinstance(contract, Mapping):
        return output
    envelope = EditContractEnvelope.create(
        contract,
        parent_program_id=parent_program_id,
        generated_generation_id=generated_generation_id,
        generated_proposal_id=generated_proposal_id,
    )
    generated = envelope.to_dict()
    output["edit_contract_envelope"] = generated
    lifecycle = output.get("edit_contract_lifecycle")
    lifecycle = deepcopy(dict(lifecycle)) if isinstance(lifecycle, Mapping) else {}
    previous_generated = lifecycle.get("generated")
    if (
        isinstance(previous_generated, Mapping)
        and previous_generated.get("contract_id") != generated.get("contract_id")
    ):


        lifecycle["offered"] = deepcopy(dict(previous_generated))
    lifecycle.update(
        {
            "schema_version": EDIT_CONTRACT_LIFECYCLE_VERSION,
            "generated": generated,
        }
    )
    lifecycle.setdefault("response", None)
    lifecycle.setdefault("accepted", None)
    lifecycle.setdefault("rejected", None)
    lifecycle.setdefault("applied", None)
    output["edit_contract_lifecycle"] = lifecycle
    return output


def validate_contract_response(
    value: Any, envelope: EditContractEnvelope
) -> Dict[str, str]:


    if not isinstance(envelope, EditContractEnvelope):
        raise TypeError("envelope must be an EditContractEnvelope")
    if not isinstance(value, Mapping):
        raise EditContractLifecycleError(
            "contract_response_missing",
            "last_contract_response is required for the offered parent contract",
        )
    allowed = {"schema_version", "contract_id", "decision", "reason"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EditContractLifecycleError(
            "contract_response_unknown_fields",
            "unknown response fields: " + ", ".join(unknown),
        )
    missing = sorted(allowed - set(value))
    if missing:
        raise EditContractLifecycleError(
            "contract_response_incomplete",
            "missing response fields: " + ", ".join(missing),
        )
    if value.get("schema_version") != EDIT_CONTRACT_RESPONSE_VERSION:
        raise EditContractLifecycleError(
            "contract_response_version_unsupported",
            f"unsupported response version {value.get('schema_version')!r}",
        )
    contract_id = str(value.get("contract_id") or "")
    if contract_id != envelope.contract_id:
        raise EditContractLifecycleError(
            "contract_response_stale",
            f"response targets {contract_id!r}, expected {envelope.contract_id!r}",
        )
    decision = str(value.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject"}:
        raise EditContractLifecycleError(
            "contract_response_decision_invalid",
            "decision must be exactly 'accept' or 'reject'",
        )
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise EditContractLifecycleError(
            "contract_response_reason_missing", "response reason must be non-empty"
        )
    return {
        "schema_version": EDIT_CONTRACT_RESPONSE_VERSION,
        "contract_id": contract_id,
        "decision": decision,
        "reason": reason,
    }


def canonical_contract_response_json(
    value: Any, envelope_json: str
) -> str:


    envelope = EditContractEnvelope.from_json(envelope_json)
    return _canonical_json(validate_contract_response(value, envelope))


def inject_controller_contract_response(
    strategy: Mapping[str, Any], response: Mapping[str, Any] | None
) -> Dict[str, Any]:


    output = deepcopy(dict(strategy))
    if response is not None:
        output["last_contract_response"] = deepcopy(dict(response))
    return output


def _prompt_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 8,
    max_text: int = 320,
) -> Any:


    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= max_text else text[: max_text - 3] + "..."
    if depth >= max_depth:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return {
            "content_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "summary": "details omitted by bounded prompt projection",
        }
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        result = {
            str(key): _prompt_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_text=max_text,
            )
            for key, item in ordered[:max_items]
        }
        if len(ordered) > max_items:
            result["omitted_field_count"] = len(ordered) - max_items
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = [
            _prompt_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_text=max_text,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            result.append({"omitted_item_count": len(value) - max_items})
        return result
    return _prompt_value(str(value), max_text=max_text)


def edit_contract_prompt_projection(
    envelope: EditContractEnvelope,
    *,
    max_bytes: int = 8 * 1024,
) -> Dict[str, Any]:


    if not isinstance(envelope, EditContractEnvelope):
        raise TypeError("envelope must be an EditContractEnvelope")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
        raise ValueError("max_bytes must be an integer of at least 1024")
    contract = envelope.contract
    metadata = contract.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    recommendations = metadata.get("ast_level_recommendations")
    if isinstance(recommendations, (str, bytes)) or not isinstance(
        recommendations, Sequence
    ):
        recommendations = []
    projection: Dict[str, Any] = {
        "schema_version": EDIT_CONTRACT_PROMPT_PROJECTION_VERSION,
        "contract_id": envelope.contract_id,
        "contract_hash": envelope.contract_hash,
        "parent_program_id": envelope.parent_program_id,
        "envelope_hash": envelope.envelope_hash,
        "contract": {
            "schema_version": contract.get("schema_version"),
            "action": contract.get("action"),
            "required_nodes": deepcopy(contract.get("required_nodes", [])),
            "forbidden_nodes": deepcopy(contract.get("forbidden_nodes", [])),
            "mutation_budget": deepcopy(contract.get("mutation_budget", {})),
            "rationale": _prompt_value(contract.get("rationale")),
            "ast_level_recommendations": _prompt_value(
                list(recommendations), max_depth=3, max_items=8, max_text=320
            ),
        },
        "runtime_validation": "full_canonical_envelope",
    }

    def size() -> int:
        return len(_canonical_json(projection).encode("utf-8"))

    if size() > max_bytes:
        rationale = contract.get("rationale")
        rationale_json = json.dumps(
            rationale, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        projection["contract"]["rationale"] = {
            "content_sha256": hashlib.sha256(
                rationale_json.encode("utf-8")
            ).hexdigest(),
            "summary": "rationale omitted by bounded prompt projection",
        }
        projection["contract"]["ast_level_recommendations"] = _prompt_value(
            list(recommendations), max_depth=2, max_items=4, max_text=120
        )
    if size() > max_bytes:
        raise EditContractLifecycleError(
            "contract_prompt_projection_too_large",
            f"decision-bearing prompt projection exceeds {max_bytes} UTF-8 bytes",
        )
    return projection


def resolve_parent_contract(
    strategy: Mapping[str, Any],
    envelope_json: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    output = deepcopy(dict(strategy))
    envelope = EditContractEnvelope.from_json(envelope_json)
    if "edit_contract" in output and output.get("edit_contract") is not None:
        raise EditContractLifecycleError(
            "child_supplied_edit_contract_forbidden",
            "the parent envelope is authoritative; child code may only respond to it",
        )
    response = validate_contract_response(
        output.get("last_contract_response"), envelope
    )
    decision = response["decision"]
    if decision == "accept":
        output["edit_contract"] = envelope.contract
    else:
        output.pop("edit_contract", None)
    lifecycle = {
        "schema_version": EDIT_CONTRACT_LIFECYCLE_VERSION,
        "generated": envelope.to_dict(),
        "response": response,
        "accepted": envelope.to_dict() if decision == "accept" else None,
        "rejected": (
            {
                "contract_id": envelope.contract_id,
                "parent_program_id": envelope.parent_program_id,
                "reason": response["reason"],
            }
            if decision == "reject"
            else None
        ),
        "applied": None,
    }
    return output, lifecycle


def policy_projection(strategy: Mapping[str, Any]) -> Dict[str, Any]:


    return {
        "preferred_edit_order": deepcopy(strategy.get("preferred_edit_order", [])),
        "node_edit_policies": deepcopy(strategy.get("node_edit_policies", {})),
        "score_config": deepcopy(strategy.get("score_config", {})),
    }


def summarize_contract_response(strategy: Mapping[str, Any]) -> Dict[str, Any]:


    raw = strategy.get("last_contract_response")
    if not isinstance(raw, Mapping):
        return {
            "present": False,
            "usable": False,
            "accepted": [],
            "rejected": [],
            "reason": "",
            "warnings": ["strategy.last_contract_response is missing"],
        }
    if raw.get("schema_version") == EDIT_CONTRACT_RESPONSE_VERSION:
        contract_id = str(raw.get("contract_id") or "")
        decision = str(raw.get("decision") or "").strip().lower()
        accepted = [contract_id] if contract_id and decision == "accept" else []
        rejected = [contract_id] if contract_id and decision == "reject" else []
    else:
        def names(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(item) for item in value if str(item)]
            return []

        accepted = names(raw.get("accepted"))
        rejected = names(raw.get("rejected"))
    reason = str(raw.get("reason") or raw.get("rationale") or "").strip()
    warnings = []
    if not accepted and not rejected:
        warnings.append("last_contract_response has no accepted/rejected items")
    if not reason:
        warnings.append("last_contract_response.reason is empty")
    return {
        "present": True,
        "usable": bool(reason and (accepted or rejected)),
        "accepted": accepted,
        "rejected": rejected,
        "reason": reason,
        "raw": dict(raw),
        "warnings": warnings,
    }


def finalize_applied_contract(
    lifecycle: Mapping[str, Any],
    *,
    before_policy: Mapping[str, Any],
    after_policy: Mapping[str, Any],
    before_ast_hash: str = "",
    after_ast_hash: str = "",
) -> Dict[str, Any]:


    result = deepcopy(dict(lifecycle))
    accepted = result.get("accepted")
    if not isinstance(accepted, Mapping):
        result["applied"] = None
        return result
    before_hash = _sha256_json(dict(before_policy))
    after_hash = _sha256_json(dict(after_policy))
    accepted_contract = accepted.get("contract")
    accepted_action = str(
        accepted_contract.get("action")
        if isinstance(accepted_contract, Mapping)
        else ""
    )
    requires_ast_delta = accepted_action == "modify_ast_node_definitions"
    normalized_before_ast = str(before_ast_hash or "")
    normalized_after_ast = str(after_ast_hash or "")
    if requires_ast_delta and (
        not normalized_before_ast
        or not normalized_after_ast
        or normalized_before_ast == normalized_after_ast
    ):
        result["applied"] = {
            "contract_id": accepted.get("contract_id"),
            "status": "no_ast_delta",
            "before_policy_hash": before_hash,
            "after_policy_hash": after_hash,
            "before_ast_hash": normalized_before_ast or None,
            "after_ast_hash": normalized_after_ast or None,
        }
        raise EditContractLifecycleError(
            "accepted_contract_no_ast_delta",
            "modify_ast_node_definitions requires an observable child AST delta",
            artifact=result,
        )
    if not requires_ast_delta and before_hash == after_hash:
        result["applied"] = {
            "contract_id": accepted.get("contract_id"),
            "status": "no_op",
            "before_policy_hash": before_hash,
            "after_policy_hash": after_hash,
            "before_ast_hash": normalized_before_ast or None,
            "after_ast_hash": normalized_after_ast or None,
        }
        raise EditContractLifecycleError(
            "accepted_contract_no_policy_delta",
            "accepted parent contract produced no observable policy delta",
            artifact=result,
        )
    result["applied"] = {
        "contract_id": accepted.get("contract_id"),
        "status": "applied",
        "before_policy_hash": before_hash,
        "after_policy_hash": after_hash,
        "before_ast_hash": normalized_before_ast or None,
        "after_ast_hash": normalized_after_ast or None,
    }
    return result


def finalize_selected_parent_contract(
    lifecycle: Mapping[str, Any],
    *,
    before_policy: Mapping[str, Any],
    after_policy: Mapping[str, Any],
    causal_envelope: Mapping[str, Any],
    ast_revision_report: Mapping[str, Any],
) -> Dict[str, Any]:


    raw_parent = causal_envelope.get("parent_effective_contract")
    semantic = raw_parent.get("semantic", {}) if isinstance(raw_parent, Mapping) else {}
    state_context = semantic.get("state_context", {}) if isinstance(semantic, Mapping) else {}
    parent_ast = state_context.get("executable_ast", {}) if isinstance(state_context, Mapping) else {}
    parent_ast_hash = (
        str(parent_ast.get("effective_ast_hash") or "")
        if isinstance(parent_ast, Mapping)
        else ""
    )
    return finalize_applied_contract(
        lifecycle,
        before_policy=before_policy,
        after_policy=after_policy,
        before_ast_hash=parent_ast_hash,
        after_ast_hash=str(ast_revision_report.get("effective_ast_hash") or ""),
    )

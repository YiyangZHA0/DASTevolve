

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from astevolve.domain.compiled import (
    CompiledCandidateSlot,
    CompiledDesignAction,
    CompiledPortfolioOptimizationRequest,
)
from astevolve.search.portfolio_runtime import (
    strict_compiled_portfolio_request,
    validate_candidate_portfolio_realization,
)
from engine.design_action_compiler import (
    DesignActionCompileError,
    validate_candidate_against_compiled_action,
)
from engine.experiment_identity import SequenceBundleIdentity


CANDIDATE_WAVE_REQUEST_VERSION = "astevolve.candidate_wave_request.v1"
CANDIDATE_WAVE_SLOT_DIRECTIVE_VERSION = (
    "astevolve.candidate_wave_slot_directive.v1"
)
CANDIDATE_WAVE_SLOT_REALIZATION_VERSION = (
    "astevolve.candidate_wave_slot_realization.v1"
)
FROZEN_CANDIDATE_WAVE_VERSION = "astevolve.frozen_candidate_wave.v1"
CANDIDATE_WAVE_UNDERFILL_VERSION = "astevolve.candidate_wave_underfill.v1"
NODE_ENGAGEMENT_LEDGER_VERSION = "astevolve.node_engagement_ledger.v1"
RERANK_SELECTION_RECEIPT_VERSION = (
    "astevolve.candidate_wave_rerank_selection_receipt.v1"
)

DEFAULT_ROLE_HISTOGRAM: Mapping[str, int] = {
    "primary": 2,
    "repair": 2,
    "matched_ablation": 2,
    "novelty": 1,
    "control": 1,
}
_LIVE_NODE_CHANGE_KINDS = frozenset({"created", "migrated", "resized", "shifted"})
_KNOWN_NODE_CHANGE_KINDS = _LIVE_NODE_CHANGE_KINDS | {
    "deleted",
    "action_profile_changed",
}
RERANK_ROLE_HISTOGRAM: Mapping[str, int] = {
    "matched_ablation": 2,
    "primary": 1,
    "repair": 1,
}
CandidateBinding = Tuple[
    str,
    str,
    Dict[str, str],
    Mapping[str, Any],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]


class CandidateWaveError(ValueError):


    def __init__(
        self,
        code: str,
        detail: str = "",
        *,
        underfill_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.underfill_receipt = (
            deepcopy(dict(underfill_receipt))
            if underfill_receipt is not None
            else None
        )

        self.receipt = self.underfill_receipt
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(
    code: str,
    detail: str = "",
    *,
    underfill_receipt: Mapping[str, Any] | None = None,
) -> None:
    raise CandidateWaveError(
        code, detail, underfill_receipt=underfill_receipt
    )


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
        _fail("candidate_wave_not_canonical_json", str(error))


def _domain_hash(prefix: str, domain: str, semantic: Any) -> str:
    digest = hashlib.sha256(
        (domain + "\0" + _canonical_json(semantic)).encode("utf-8")
    ).hexdigest()
    return prefix + digest


def _closed(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("candidate_wave_mapping_required", label)
    expected = set(fields)
    observed = {str(key) for key in value}
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        _fail("candidate_wave_unknown_fields", f"{label}:{','.join(unknown)}")
    if missing:
        _fail("candidate_wave_fields_missing", f"{label}:{','.join(missing)}")
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _sequence_hash(sequences: Mapping[str, str]) -> str:
    try:
        return SequenceBundleIdentity.create(
            {str(chain): str(sequence) for chain, sequence in sequences.items()}
        ).sequence_bundle_hash
    except (TypeError, ValueError) as error:
        _fail("candidate_wave_sequence_bundle_invalid", str(error))


def _strict_action(
    value: CompiledDesignAction | Mapping[str, Any],
) -> CompiledDesignAction:
    try:
        artifact = value.to_artifact() if isinstance(value, CompiledDesignAction) else value
        if not isinstance(artifact, Mapping):
            _fail("compiled_design_action_artifact_invalid")
        return CompiledDesignAction.from_artifact(artifact)
    except CandidateWaveError:
        raise
    except Exception as error:
        _fail(
            str(getattr(error, "code", "compiled_design_action_artifact_invalid")),
            str(getattr(error, "detail", "") or error),
        )


def _normalize_histogram(value: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        _fail("candidate_wave_role_histogram_invalid")
    result: Dict[str, int] = {}
    for role, count in value.items():
        name = _text(role, "candidate_wave_role_invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("candidate_wave_role_count_invalid", name)
        if count:
            result[name] = count
    return {role: result[role] for role in sorted(result)}


def _positions_by_owner(
    owners: Mapping[str, Mapping[int, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for chain, entries in owners.items():
        for position, owner in entries.items():
            result.setdefault(str(owner), []).append(
                {"chain_id": str(chain), "position": int(position)}
            )
    for node_id in result:
        result[node_id].sort(key=lambda item: (item["chain_id"], item["position"]))
    return result


def _raw_node_changes(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Sequence[Any]:
    if isinstance(value, Mapping):
        raw = value.get("node_changes")
    else:
        raw = value
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        _fail("candidate_wave_node_changes_invalid")
    return raw


def _normalize_node_changes(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    action: CompiledDesignAction,
) -> List[Dict[str, Any]]:
    parent = _positions_by_owner(action.parent_position_owners)
    active = _positions_by_owner(action.active_position_owners)
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in _raw_node_changes(value):
        if not isinstance(raw, Mapping):
            _fail("candidate_wave_node_change_invalid")
        node_id = _text(raw.get("node_id"), "candidate_wave_node_id_invalid")
        reported_kind = _text(
            raw.get("change"), "candidate_wave_node_change_kind_invalid"
        )
        if reported_kind not in _KNOWN_NODE_CHANGE_KINDS:
            _fail(
                "candidate_wave_node_change_kind_invalid",
                f"{node_id}:{reported_kind}",
            )
        if node_id in seen:
            _fail("candidate_wave_node_change_duplicate", node_id)
        seen.add(node_id)
        before_positions = parent.get(node_id, [])
        after_positions = active.get(node_id, [])


        kind = reported_kind
        if not before_positions and after_positions:
            kind = "created"
        elif before_positions and not after_positions:
            kind = "deleted"
        if kind == "created" and (before_positions or not after_positions):
            _fail("candidate_wave_node_change_owner_mismatch", f"{node_id}:{kind}")
        if kind == "deleted" and (not before_positions or after_positions):
            _fail("candidate_wave_node_change_owner_mismatch", f"{node_id}:{kind}")
        if kind in {"migrated", "resized", "shifted", "action_profile_changed"} and (
            not before_positions or not after_positions
        ):
            _fail("candidate_wave_node_change_owner_mismatch", f"{node_id}:{kind}")
        before_keys = {
            (item["chain_id"], item["position"]) for item in before_positions
        }
        after_keys = {
            (item["chain_id"], item["position"]) for item in after_positions
        }
        records.append(
            {
                "node_id": node_id,
                "change": kind,
                "parent_positions": deepcopy(before_positions),
                "active_positions": deepcopy(after_positions),
                "added_positions": [
                    {"chain_id": chain, "position": position}
                    for chain, position in sorted(after_keys - before_keys)
                ],
                "removed_positions": [
                    {"chain_id": chain, "position": position}
                    for chain, position in sorted(before_keys - after_keys)
                ],
            }
        )
    records.sort(key=lambda item: item["node_id"])
    if not any(item["change"] in _LIVE_NODE_CHANGE_KINDS for item in records):
        _fail("candidate_wave_live_node_change_required")
    return records


def _validate_action_request_binding(
    request: CompiledPortfolioOptimizationRequest,
    action: CompiledDesignAction,
) -> None:
    for field in (
        "design_action_hash",
        "compiled_design_action_hash",
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "immutable_sequence_bundle_hash",
        "reconciled_root_sequence_bundle_hash",
    ):
        if str(getattr(request, field)) != str(getattr(action, field)):
            _fail("candidate_wave_action_binding_mismatch", field)
    if dict(request.reconciled_root_sequences) != dict(action.reconciled_root_sequences):
        _fail("candidate_wave_root_sequence_mismatch")


def _slot_summary(slot: CompiledCandidateSlot) -> Dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "slot_hash": slot.slot_hash,
        "portfolio_id": slot.portfolio_id,
        "role": slot.role,
        "generation_mode": slot.generation_mode,
        "realization_kind": slot.realization_kind,
        "required": slot.required,
        "exact": slot.exact,
        "exact_sequence_bundle_hash": slot.exact_sequence_bundle_hash,
    }


_WAVE_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "compiled_portfolio_request",
        "compiled_portfolio_request_hash",
        "compiled_design_action",
        "compiled_design_action_hash",
        "design_action_hash",
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "reconciled_root_sequence_bundle_hash",
        "role_histogram",
        "wave_size",
        "candidate_slots",
        "node_changes",
        "node_change_set_hash",
        "candidate_wave_request_hash",
    }
)


def _wave_request_semantic(
    request: CompiledPortfolioOptimizationRequest,
    action: CompiledDesignAction,
    node_changes: Sequence[Mapping[str, Any]],
    histogram: Mapping[str, int],
) -> Dict[str, Any]:
    slots = [_slot_summary(slot) for slot in request.candidate_slots]
    return {
        "compiled_portfolio_request": request.to_artifact(),
        "compiled_portfolio_request_hash": request.compiled_portfolio_request_hash,
        "compiled_design_action": action.to_artifact(),
        "compiled_design_action_hash": action.compiled_design_action_hash,
        "design_action_hash": action.design_action_hash,
        "case_id": action.case_id,
        "parent_program_id": action.parent_program_id,
        "parent_candidate_id": action.parent_candidate_id,
        "reconciled_root_sequence_bundle_hash": (
            action.reconciled_root_sequence_bundle_hash
        ),
        "role_histogram": dict(histogram),
        "wave_size": sum(histogram.values()),
        "candidate_slots": slots,
        "node_changes": [deepcopy(dict(item)) for item in node_changes],
        "node_change_set_hash": _domain_hash(
            "candidate_wave_node_changes_sha256:",
            "astevolve.candidate_wave_node_changes.v1",
            node_changes,
        ),
    }


def build_candidate_wave_request(
    request: CompiledPortfolioOptimizationRequest | Mapping[str, Any],
    compiled_design_action: CompiledDesignAction | Mapping[str, Any],
    ast_revision_report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    role_histogram: Mapping[str, int] = DEFAULT_ROLE_HISTOGRAM,
) -> Dict[str, Any]:


    try:
        compiled_request = strict_compiled_portfolio_request(request)
    except Exception as error:
        _fail(
            str(getattr(error, "code", "compiled_portfolio_request_invalid")),
            str(getattr(error, "detail", "") or error),
        )
    action = _strict_action(compiled_design_action)
    _validate_action_request_binding(compiled_request, action)
    histogram = _normalize_histogram(role_histogram)
    expected = _normalize_histogram(DEFAULT_ROLE_HISTOGRAM)
    if histogram != expected:
        _fail("candidate_wave_role_histogram_not_step4", repr(histogram))
    observed: Dict[str, int] = {}
    for slot in compiled_request.candidate_slots:
        observed[slot.role] = observed.get(slot.role, 0) + 1
        if slot.required is not True:
            _fail("candidate_wave_optional_slot_forbidden", slot.slot_id)
    observed = {role: observed[role] for role in sorted(observed)}
    if observed != histogram:
        _fail(
            "candidate_wave_role_histogram_mismatch",
            f"expected={histogram};observed={observed}",
        )
    if len(compiled_request.candidate_slots) != 8:
        _fail("candidate_wave_slot_count_invalid", str(len(compiled_request.candidate_slots)))
    exact_hashes = [
        str(slot.exact_sequence_bundle_hash)
        for slot in compiled_request.candidate_slots
        if slot.exact
    ]
    if len(exact_hashes) != len(set(exact_hashes)):
        _fail("candidate_wave_exact_sequence_duplicate")
    if compiled_request.reconciled_root_sequence_bundle_hash in exact_hashes:
        _fail("candidate_wave_exact_slot_is_root")
    node_changes = _normalize_node_changes(ast_revision_report, action)
    semantic = _wave_request_semantic(
        compiled_request, action, node_changes, histogram
    )
    artifact = {
        "schema_version": CANDIDATE_WAVE_REQUEST_VERSION,
        **semantic,
        "candidate_wave_request_hash": _domain_hash(
            "candidate_wave_request_sha256:",
            CANDIDATE_WAVE_REQUEST_VERSION,
            semantic,
        ),
    }
    return validate_candidate_wave_request(artifact)


def validate_candidate_wave_request(value: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _closed(value, _WAVE_REQUEST_FIELDS, "candidate_wave_request")
    if raw["schema_version"] != CANDIDATE_WAVE_REQUEST_VERSION:
        _fail("candidate_wave_schema_version_invalid", "request")
    try:
        request = strict_compiled_portfolio_request(raw["compiled_portfolio_request"])
    except Exception as error:
        _fail(
            str(getattr(error, "code", "compiled_portfolio_request_invalid")),
            str(getattr(error, "detail", "") or error),
        )
    action = _strict_action(raw["compiled_design_action"])
    _validate_action_request_binding(request, action)
    histogram = _normalize_histogram(raw["role_histogram"])
    if histogram != _normalize_histogram(DEFAULT_ROLE_HISTOGRAM):
        _fail("candidate_wave_role_histogram_not_step4")
    observed: Dict[str, int] = {}
    exact_hashes: List[str] = []
    for slot in request.candidate_slots:
        observed[slot.role] = observed.get(slot.role, 0) + 1
        if slot.required is not True:
            _fail("candidate_wave_optional_slot_forbidden", slot.slot_id)
        if slot.exact:
            exact_hashes.append(str(slot.exact_sequence_bundle_hash))
    observed = {role: observed[role] for role in sorted(observed)}
    if observed != histogram:
        _fail("candidate_wave_role_histogram_mismatch")
    if len(request.candidate_slots) != 8:
        _fail("candidate_wave_slot_count_invalid", str(len(request.candidate_slots)))
    if len(exact_hashes) != len(set(exact_hashes)):
        _fail("candidate_wave_exact_sequence_duplicate")
    if request.reconciled_root_sequence_bundle_hash in exact_hashes:
        _fail("candidate_wave_exact_slot_is_root")
    node_changes = _normalize_node_changes(
        {"node_changes": raw["node_changes"]}, action
    )
    semantic = _wave_request_semantic(request, action, node_changes, histogram)
    expected = {
        "schema_version": CANDIDATE_WAVE_REQUEST_VERSION,
        **semantic,
        "candidate_wave_request_hash": _domain_hash(
            "candidate_wave_request_sha256:",
            CANDIDATE_WAVE_REQUEST_VERSION,
            semantic,
        ),
    }
    if _canonical_json(raw) != _canonical_json(expected):
        _fail("candidate_wave_request_mismatch")
    return deepcopy(expected)


_DIRECTIVE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_wave_request_hash",
        "compiled_portfolio_request_hash",
        "slot_id",
        "slot_hash",
        "portfolio_id",
        "role",
        "generation_mode",
        "realization_kind",
        "forced_tier",
        "directive_hash",
    }
)


def _forced_tier(role: str) -> str:
    if role in {"primary", "control"}:
        return "exploit"
    if role == "repair":
        return "repair"


    return "explore"


def _directive(
    wave: Mapping[str, Any], slot: Mapping[str, Any]
) -> Dict[str, Any]:
    semantic = {
        "schema_version": CANDIDATE_WAVE_SLOT_DIRECTIVE_VERSION,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "compiled_portfolio_request_hash": wave[
            "compiled_portfolio_request_hash"
        ],
        "slot_id": slot["slot_id"],
        "slot_hash": slot["slot_hash"],
        "portfolio_id": slot["portfolio_id"],
        "role": slot["role"],
        "generation_mode": slot["generation_mode"],
        "realization_kind": slot["realization_kind"],
        "forced_tier": _forced_tier(str(slot["role"])),
    }
    return {
        **semantic,
        "directive_hash": _domain_hash(
            "candidate_wave_slot_directive_sha256:",
            CANDIDATE_WAVE_SLOT_DIRECTIVE_VERSION,
            semantic,
        ),
    }


def build_free_slot_directives(
    wave_request: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    wave = validate_candidate_wave_request(wave_request)
    return [
        _directive(wave, slot)
        for slot in wave["candidate_slots"]
        if slot["exact"] is False
    ]


def _strict_directive(
    value: Mapping[str, Any], wave: Mapping[str, Any]
) -> Dict[str, Any]:
    raw = _closed(value, _DIRECTIVE_FIELDS, "candidate_wave_slot_directive")
    slots = {slot["slot_id"]: slot for slot in wave["candidate_slots"]}
    slot = slots.get(str(raw.get("slot_id") or ""))
    if slot is None or slot["exact"] is not False:
        _fail("candidate_wave_slot_directive_slot_invalid", str(raw.get("slot_id")))
    expected = _directive(wave, slot)
    if _canonical_json(raw) != _canonical_json(expected):
        _fail("candidate_wave_slot_directive_mismatch", str(raw.get("slot_id")))
    return expected


def _candidate_sequences(candidate: Mapping[str, Any]) -> Dict[str, str]:
    seqs = candidate.get("seqs")
    alternate = candidate.get("sequences")
    if seqs is not None and alternate is not None and dict(seqs) != dict(alternate):
        _fail("candidate_wave_candidate_sequence_fields_conflict")
    raw = seqs if seqs is not None else alternate
    if not isinstance(raw, Mapping):
        _fail("candidate_wave_candidate_sequences_missing")
    sequences = {str(chain): str(sequence) for chain, sequence in raw.items()}
    _sequence_hash(sequences)
    return {chain: sequences[chain] for chain in sorted(sequences)}


def _candidate_ref(candidate: Mapping[str, Any], sequence_hash: str) -> str:
    for field in ("candidate_id", "node_id", "id", "variant_id", "program_id"):
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return sequence_hash


def _extract_directive(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    top = candidate.get("candidate_wave_slot_directive")
    move = candidate.get("move")
    nested = move.get("candidate_wave_slot_directive") if isinstance(move, Mapping) else None
    if top is not None and nested is not None and _canonical_json(top) != _canonical_json(nested):
        _fail("candidate_wave_slot_directive_fields_conflict")
    value = top if top is not None else nested
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _fail("candidate_wave_slot_directive_invalid")
    return value


def _portfolio_candidate_view(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(candidate))
    move = candidate.get("move")
    if isinstance(move, Mapping):
        for field in (
            "compiled_portfolio_request_hash",
            "portfolio_realization_receipts",
            "portfolio_pair_receipts",
            "portfolio_realization_summary",
        ):
            if field not in result and field in move:
                result[field] = deepcopy(move[field])
    return result


def _validate_candidate_against_action(
    candidate: Mapping[str, Any], action: CompiledDesignAction
) -> Tuple[Dict[str, str], str, str]:
    sequences = _candidate_sequences(candidate)
    try:
        validate_candidate_against_compiled_action(
            action,
            sequences,
            action.compiled_design_action_hash,


            allow_reconciled_root=True,
        )
    except DesignActionCompileError as error:
        _fail(
            "candidate_wave_candidate_contract_invalid",
            f"{error.code}:{error.detail}",
        )
    sequence_hash = _sequence_hash(sequences)
    return sequences, sequence_hash, _candidate_ref(candidate, sequence_hash)


def _realization_receipt(
    wave: Mapping[str, Any],
    slot: Mapping[str, Any],
    *,
    sequences: Mapping[str, str],
    candidate_ref: str,
    directive: Mapping[str, Any] | None,
    portfolio_receipts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source = "compiled_exact" if slot["exact"] else "directed_mcts"
    portfolio_hashes = sorted(
        str(receipt.get("receipt_hash") or "") for receipt in portfolio_receipts
    )
    if any(not item for item in portfolio_hashes):
        _fail("candidate_wave_portfolio_receipt_hash_missing", slot["slot_id"])
    semantic = {
        "schema_version": CANDIDATE_WAVE_SLOT_REALIZATION_VERSION,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "compiled_portfolio_request_hash": wave[
            "compiled_portfolio_request_hash"
        ],
        "slot_id": slot["slot_id"],
        "slot_hash": slot["slot_hash"],
        "portfolio_id": slot["portfolio_id"],
        "role": slot["role"],
        "generation_mode": slot["generation_mode"],
        "exact": slot["exact"],
        "realization_source": source,
        "candidate_ref": candidate_ref,
        "candidate_sequence_bundle_hash": _sequence_hash(sequences),
        "slot_directive_hash": (
            directive["directive_hash"] if directive is not None else None
        ),
        "portfolio_realization_receipt_hashes": portfolio_hashes,
    }
    return {
        **semantic,
        "realization_receipt_hash": _domain_hash(
            "candidate_wave_slot_realization_sha256:",
            CANDIDATE_WAVE_SLOT_REALIZATION_VERSION,
            semantic,
        ),
    }


_REALIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_wave_request_hash",
        "compiled_portfolio_request_hash",
        "slot_id",
        "slot_hash",
        "portfolio_id",
        "role",
        "generation_mode",
        "exact",
        "realization_source",
        "candidate_ref",
        "candidate_sequence_bundle_hash",
        "slot_directive_hash",
        "portfolio_realization_receipt_hashes",
        "realization_receipt_hash",
    }
)


def _underfill_receipt(
    wave: Mapping[str, Any],
    *,
    realized_slots: Sequence[str],
    realized_roles: Mapping[str, int],
    missing_slots: Sequence[str],
    sequence_collision_slots: Sequence[str],
    candidate_pool_size: int,
) -> Dict[str, Any]:
    semantic = {
        "schema_version": CANDIDATE_WAVE_UNDERFILL_VERSION,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "compiled_portfolio_request_hash": wave[
            "compiled_portfolio_request_hash"
        ],
        "required_wave_size": wave["wave_size"],
        "candidate_pool_size": candidate_pool_size,
        "realized_slot_count": len(realized_slots),
        "realized_slot_ids": sorted(realized_slots),
        "missing_slot_ids": sorted(missing_slots),
        "sequence_collision_slot_ids": sorted(set(sequence_collision_slots)),
        "expected_role_histogram": dict(wave["role_histogram"]),
        "realized_role_histogram": {
            role: int(realized_roles.get(role, 0))
            for role in sorted(wave["role_histogram"])
        },
        "provider_calls_authorized": False,
    }
    return {
        **semantic,
        "underfill_receipt_hash": _domain_hash(
            "candidate_wave_underfill_sha256:",
            CANDIDATE_WAVE_UNDERFILL_VERSION,
            semantic,
        ),
    }


_MEMBER_FIELDS = frozenset(
    {
        "slot_id",
        "slot_hash",
        "portfolio_id",
        "role",
        "generation_mode",
        "realization_kind",
        "exact",
        "candidate_ref",
        "seqs",
        "candidate_sequence_bundle_hash",
        "candidate_wave_slot_directive",
        "portfolio_realization_receipts",
        "portfolio_pair_receipts",
        "candidate_wave_slot_realization",
    }
)


def freeze_candidate_wave(
    wave_request: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:


    wave = validate_candidate_wave_request(wave_request)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        _fail("candidate_wave_candidates_invalid")
    request = strict_compiled_portfolio_request(
        wave["compiled_portfolio_request"]
    )
    action = _strict_action(wave["compiled_design_action"])
    slots = {slot["slot_id"]: slot for slot in wave["candidate_slots"]}
    exact_by_hash = {
        slot["exact_sequence_bundle_hash"]: slot
        for slot in wave["candidate_slots"]
        if slot["exact"]
    }
    directives = {
        item["slot_id"]: item for item in build_free_slot_directives(wave)
    }
    exact_candidates: Dict[str, List[CandidateBinding]] = {}
    free_candidates: Dict[str, List[CandidateBinding]] = {}

    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            _fail("candidate_wave_candidate_invalid")
        sequences, sequence_hash, candidate_ref = _validate_candidate_against_action(
            raw_candidate, action
        )
        if sequence_hash == wave["reconciled_root_sequence_bundle_hash"]:


            continue
        directive_raw = _extract_directive(raw_candidate)
        exact_slot = exact_by_hash.get(sequence_hash)
        portfolio_receipts: List[Mapping[str, Any]] = []
        pair_receipts: List[Mapping[str, Any]] = []
        if exact_slot is not None:


            portfolio_view = _portfolio_candidate_view(raw_candidate)
            has_portfolio_claim = any(
                portfolio_view.get(field) not in (None, "", [], {}, ())
                for field in (
                    "compiled_portfolio_request_hash",
                    "portfolio_realization_receipts",
                    "portfolio_pair_receipts",
                    "portfolio_realization_summary",
                    "portfolio_id",
                    "portfolio_slot_id",
                )
            )
            if (
                raw_candidate.get("duplicate_sequence") is True
                and directive_raw is None
                and not has_portfolio_claim
            ):
                continue
            if directive_raw is not None:
                _fail("candidate_wave_exact_candidate_has_free_directive", exact_slot["slot_id"])
            view = portfolio_view
            try:
                validation = validate_candidate_portfolio_realization(view, request)
            except Exception as error:
                _fail(
                    str(getattr(error, "code", "candidate_wave_exact_receipt_invalid")),
                    str(getattr(error, "detail", "") or error),
                )
            if exact_slot["slot_id"] not in validation.get("validated_slot_ids", []):
                _fail("candidate_wave_exact_slot_receipt_missing", exact_slot["slot_id"])
            portfolio_receipts = list(view.get("portfolio_realization_receipts", []))
            pair_receipts = list(view.get("portfolio_pair_receipts", []))
            exact_candidates.setdefault(exact_slot["slot_id"], []).append(
                (
                    sequence_hash,
                    candidate_ref,
                    sequences,
                    {},
                    portfolio_receipts,
                    pair_receipts,
                )
            )
            continue
        if directive_raw is None:

            continue
        directive = _strict_directive(directive_raw, wave)
        slot_id = directive["slot_id"]
        free_candidates.setdefault(slot_id, []).append(
            (sequence_hash, candidate_ref, sequences, directive, [], [])
        )

    bound: Dict[str, CandidateBinding] = {}
    used_hashes: set[str] = set()
    collisions: List[str] = []
    for slot_id in sorted(slots):
        slot = slots[slot_id]
        pool = (
            exact_candidates.get(slot_id, [])
            if slot["exact"]
            else free_candidates.get(slot_id, [])
        )
        ordered = sorted(pool, key=lambda item: (item[0], item[1]))
        selected = next((item for item in ordered if item[0] not in used_hashes), None)
        if selected is None:
            if ordered and all(item[0] in used_hashes for item in ordered):
                collisions.append(slot_id)
            continue
        bound[slot_id] = selected
        used_hashes.add(selected[0])

    missing = sorted(set(slots) - set(bound))
    realized_roles: Dict[str, int] = {}
    for slot_id in bound:
        role = str(slots[slot_id]["role"])
        realized_roles[role] = realized_roles.get(role, 0) + 1
    if missing:
        receipt = _underfill_receipt(
            wave,
            realized_slots=sorted(bound),
            realized_roles=realized_roles,
            missing_slots=missing,
            sequence_collision_slots=collisions,
            candidate_pool_size=len(candidates),
        )
        _fail(
            "candidate_wave_underfilled",
            ",".join(missing),
            underfill_receipt=receipt,
        )

    members: List[Dict[str, Any]] = []
    for slot_id in sorted(slots):
        slot = slots[slot_id]
        sequence_hash, candidate_ref, sequences, directive, receipts, pairs = bound[slot_id]
        realization = _realization_receipt(
            wave,
            slot,
            sequences=sequences,
            candidate_ref=candidate_ref,
            directive=directive or None,
            portfolio_receipts=receipts,
        )
        members.append(
            {
                "slot_id": slot_id,
                "slot_hash": slot["slot_hash"],
                "portfolio_id": slot["portfolio_id"],
                "role": slot["role"],
                "generation_mode": slot["generation_mode"],
                "realization_kind": slot["realization_kind"],
                "exact": slot["exact"],
                "candidate_ref": candidate_ref,
                "seqs": sequences,
                "candidate_sequence_bundle_hash": sequence_hash,
                "candidate_wave_slot_directive": deepcopy(dict(directive)) if directive else None,
                "portfolio_realization_receipts": deepcopy(receipts),
                "portfolio_pair_receipts": deepcopy(pairs),
                "candidate_wave_slot_realization": realization,
            }
        )
    semantic = {
        "candidate_wave_request": wave,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "compiled_portfolio_request_hash": wave[
            "compiled_portfolio_request_hash"
        ],
        "wave_size": wave["wave_size"],
        "role_histogram": dict(wave["role_histogram"]),
        "members": members,
        "unique_sequence_count": len(used_hashes),
        "complete": True,
        "provider_calls_authorized": True,
    }
    manifest = {
        "schema_version": FROZEN_CANDIDATE_WAVE_VERSION,
        **semantic,
        "frozen_candidate_wave_hash": _domain_hash(
            "frozen_candidate_wave_sha256:",
            FROZEN_CANDIDATE_WAVE_VERSION,
            semantic,
        ),
    }
    return validate_frozen_candidate_wave(manifest, wave)


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_wave_request",
        "candidate_wave_request_hash",
        "compiled_portfolio_request_hash",
        "wave_size",
        "role_histogram",
        "members",
        "unique_sequence_count",
        "complete",
        "provider_calls_authorized",
        "frozen_candidate_wave_hash",
    }
)


def validate_frozen_candidate_wave(
    manifest: Mapping[str, Any], wave_request: Mapping[str, Any]
) -> Dict[str, Any]:
    wave = validate_candidate_wave_request(wave_request)
    raw = _closed(manifest, _MANIFEST_FIELDS, "frozen_candidate_wave")
    if raw["schema_version"] != FROZEN_CANDIDATE_WAVE_VERSION:
        _fail("candidate_wave_schema_version_invalid", "manifest")
    if _canonical_json(raw["candidate_wave_request"]) != _canonical_json(wave):
        _fail("candidate_wave_manifest_request_mismatch")
    if raw["complete"] is not True or raw["provider_calls_authorized"] is not True:
        _fail("candidate_wave_manifest_not_dispatchable")
    members_raw = raw["members"]
    if isinstance(members_raw, (str, bytes)) or not isinstance(members_raw, Sequence):
        _fail("candidate_wave_manifest_members_invalid")
    request = strict_compiled_portfolio_request(wave["compiled_portfolio_request"])
    action = _strict_action(wave["compiled_design_action"])
    slots = {slot["slot_id"]: slot for slot in wave["candidate_slots"]}
    directives = {item["slot_id"]: item for item in build_free_slot_directives(wave)}
    normalized_members: List[Dict[str, Any]] = []
    seen_slots: set[str] = set()
    seen_sequences: set[str] = set()
    roles: Dict[str, int] = {}
    for item in members_raw:
        member = _closed(item, _MEMBER_FIELDS, "candidate_wave_member")
        slot_id = str(member.get("slot_id") or "")
        slot = slots.get(slot_id)
        if slot is None:
            _fail("candidate_wave_manifest_slot_unknown", slot_id)
        if slot_id in seen_slots:
            _fail("candidate_wave_manifest_slot_duplicate", slot_id)
        seen_slots.add(slot_id)
        for field in (
            "slot_hash",
            "portfolio_id",
            "role",
            "generation_mode",
            "realization_kind",
            "exact",
        ):
            if member[field] != slot[field]:
                _fail("candidate_wave_manifest_slot_mismatch", f"{slot_id}:{field}")
        sequences, sequence_hash, candidate_ref = _validate_candidate_against_action(
            member, action
        )
        if member["candidate_ref"] != candidate_ref:


            candidate_ref = _text(member["candidate_ref"], "candidate_wave_candidate_ref_invalid")
        if member["candidate_sequence_bundle_hash"] != sequence_hash:
            _fail("candidate_wave_manifest_sequence_hash_mismatch", slot_id)
        if sequence_hash in seen_sequences:
            _fail("candidate_wave_manifest_sequence_duplicate", slot_id)
        seen_sequences.add(sequence_hash)
        directive: Mapping[str, Any] | None
        receipts = member["portfolio_realization_receipts"]
        pairs = member["portfolio_pair_receipts"]
        if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
            _fail("candidate_wave_manifest_portfolio_receipts_invalid", slot_id)
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            _fail("candidate_wave_manifest_pair_receipts_invalid", slot_id)
        if slot["exact"]:
            if member["candidate_wave_slot_directive"] is not None:
                _fail("candidate_wave_manifest_exact_directive_forbidden", slot_id)
            if sequence_hash != slot["exact_sequence_bundle_hash"]:
                _fail("candidate_wave_manifest_exact_sequence_mismatch", slot_id)
            view = {
                "seqs": sequences,
                "compiled_portfolio_request_hash": wave[
                    "compiled_portfolio_request_hash"
                ],
                "portfolio_realization_receipts": receipts,
                "portfolio_pair_receipts": pairs,
            }
            try:
                validation = validate_candidate_portfolio_realization(view, request)
            except Exception as error:
                _fail(
                    str(getattr(error, "code", "candidate_wave_exact_receipt_invalid")),
                    str(getattr(error, "detail", "") or error),
                )
            if slot_id not in validation.get("validated_slot_ids", []):
                _fail("candidate_wave_exact_slot_receipt_missing", slot_id)
            directive = None
        else:
            if receipts or pairs:
                _fail("candidate_wave_manifest_free_portfolio_receipt_forbidden", slot_id)
            directive = _strict_directive(
                member["candidate_wave_slot_directive"], wave
            )
            if directive != directives[slot_id]:
                _fail("candidate_wave_manifest_directive_slot_mismatch", slot_id)
        expected_realization = _realization_receipt(
            wave,
            slot,
            sequences=sequences,
            candidate_ref=candidate_ref,
            directive=directive,
            portfolio_receipts=receipts,
        )
        realization = _closed(
            member["candidate_wave_slot_realization"],
            _REALIZATION_FIELDS,
            "candidate_wave_slot_realization",
        )
        if _canonical_json(realization) != _canonical_json(expected_realization):
            _fail("candidate_wave_slot_realization_mismatch", slot_id)
        roles[slot["role"]] = roles.get(slot["role"], 0) + 1
        normalized_members.append(deepcopy(dict(member)))
    if seen_slots != set(slots):
        _fail("candidate_wave_manifest_slot_coverage_mismatch")
    if roles != dict(wave["role_histogram"]):
        _fail("candidate_wave_manifest_role_histogram_mismatch")
    if len(normalized_members) != 8 or len(seen_sequences) != 8:
        _fail("candidate_wave_manifest_physical_size_invalid")
    normalized_members.sort(key=lambda member: member["slot_id"])
    semantic = {
        "candidate_wave_request": wave,
        "candidate_wave_request_hash": wave["candidate_wave_request_hash"],
        "compiled_portfolio_request_hash": wave[
            "compiled_portfolio_request_hash"
        ],
        "wave_size": 8,
        "role_histogram": dict(wave["role_histogram"]),
        "members": normalized_members,
        "unique_sequence_count": 8,
        "complete": True,
        "provider_calls_authorized": True,
    }
    expected = {
        "schema_version": FROZEN_CANDIDATE_WAVE_VERSION,
        **semantic,
        "frozen_candidate_wave_hash": _domain_hash(
            "frozen_candidate_wave_sha256:",
            FROZEN_CANDIDATE_WAVE_VERSION,
            semantic,
        ),
    }
    if _canonical_json(raw) != _canonical_json(expected):
        _fail("candidate_wave_manifest_mismatch")
    return deepcopy(expected)


_RERANK_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_wave_request_hash",
        "frozen_candidate_wave_hash",
        "source_provider",
        "target_provider",
        "selection_policy",
        "required_role_histogram",
        "selected_members",
        "role_selection_decisions",
        "rerank_selection_receipt_hash",
    }
)
_RERANK_MEMBER_FIELDS = frozenset(
    {
        "slot_id",
        "slot_hash",
        "role",
        "candidate_sequence_bundle_hash",
        "slot_realization_receipt_hash",
    }
)
_RERANK_DECISION_FIELDS = frozenset(
    {
        "role",
        "required_count",
        "selected_slot_ids",
        "selected_sequence_bundle_hashes",
        "selection_basis",
    }
)


def _selected_sequence_hash(value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        if not value.startswith("sequence_sha256:"):
            _fail("candidate_wave_rerank_sequence_hash_invalid", value)
        return value
    if not isinstance(value, Mapping):
        _fail("candidate_wave_rerank_selection_member_invalid")
    observed = _row_sequence_hash(value)
    if observed is None:
        _fail("candidate_wave_rerank_selection_sequence_missing")
    return observed


def _rerank_member(member: Mapping[str, Any]) -> Dict[str, Any]:
    realization = member.get("candidate_wave_slot_realization")
    if not isinstance(realization, Mapping):
        _fail("candidate_wave_rerank_slot_realization_missing")
    receipt_hash = _text(
        realization.get("realization_receipt_hash"),
        "candidate_wave_rerank_slot_realization_hash_missing",
    )
    return {
        "slot_id": member["slot_id"],
        "slot_hash": member["slot_hash"],
        "role": member["role"],
        "candidate_sequence_bundle_hash": member[
            "candidate_sequence_bundle_hash"
        ],
        "slot_realization_receipt_hash": receipt_hash,
    }


def _rerank_semantic(
    frozen: Mapping[str, Any], selected_members: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    selected = [deepcopy(dict(member)) for member in selected_members]
    selected.sort(key=lambda item: item["slot_id"])
    decisions = []
    for role, count in sorted(RERANK_ROLE_HISTOGRAM.items()):
        role_members = [member for member in selected if member["role"] == role]
        decisions.append(
            {
                "role": role,
                "required_count": count,
                "selected_slot_ids": sorted(
                    member["slot_id"] for member in role_members
                ),
                "selected_sequence_bundle_hashes": sorted(
                    member["candidate_sequence_bundle_hash"]
                    for member in role_members
                ),
                "selection_basis": "upstream_preregistered_role_quota",
            }
        )
    return {
        "candidate_wave_request_hash": frozen["candidate_wave_request_hash"],
        "frozen_candidate_wave_hash": frozen["frozen_candidate_wave_hash"],
        "source_provider": "protenix",
        "target_provider": "alphafold3",
        "selection_policy": "step4_role_quota_v1",
        "required_role_histogram": dict(RERANK_ROLE_HISTOGRAM),
        "selected_members": selected,
        "role_selection_decisions": decisions,
    }


def build_rerank_selection_receipt(
    manifest: Mapping[str, Any],
    selected_candidates: Sequence[Mapping[str, Any] | str],
) -> Dict[str, Any]:


    embedded_wave = manifest.get("candidate_wave_request") if isinstance(
        manifest, Mapping
    ) else None
    if not isinstance(embedded_wave, Mapping):
        _fail("candidate_wave_manifest_request_missing")
    frozen = validate_frozen_candidate_wave(manifest, embedded_wave)
    if isinstance(selected_candidates, (str, bytes)) or not isinstance(
        selected_candidates, Sequence
    ):
        _fail("candidate_wave_rerank_selection_invalid")
    selected_hashes = [
        _selected_sequence_hash(candidate) for candidate in selected_candidates
    ]
    if len(selected_hashes) != 4:
        _fail("candidate_wave_rerank_selection_count_invalid", str(len(selected_hashes)))
    if len(selected_hashes) != len(set(selected_hashes)):
        _fail("candidate_wave_rerank_selection_duplicate")
    by_hash = {
        member["candidate_sequence_bundle_hash"]: member
        for member in frozen["members"]
    }
    unknown = sorted(set(selected_hashes) - set(by_hash))
    if unknown:
        _fail("candidate_wave_rerank_selection_outside_wave", ",".join(unknown))
    selected_members = [_rerank_member(by_hash[item]) for item in selected_hashes]
    observed = Counter(member["role"] for member in selected_members)
    if dict(observed) != dict(RERANK_ROLE_HISTOGRAM):
        _fail(
            "candidate_wave_rerank_role_histogram_mismatch",
            repr(dict(observed)),
        )
    semantic = _rerank_semantic(frozen, selected_members)
    receipt = {
        "schema_version": RERANK_SELECTION_RECEIPT_VERSION,
        **semantic,
        "rerank_selection_receipt_hash": _domain_hash(
            "candidate_wave_rerank_selection_sha256:",
            RERANK_SELECTION_RECEIPT_VERSION,
            semantic,
        ),
    }
    return validate_rerank_selection_receipt(receipt, frozen)


def validate_rerank_selection_receipt(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    embedded_wave = manifest.get("candidate_wave_request") if isinstance(
        manifest, Mapping
    ) else None
    if not isinstance(embedded_wave, Mapping):
        _fail("candidate_wave_manifest_request_missing")
    frozen = validate_frozen_candidate_wave(manifest, embedded_wave)
    raw = _closed(
        receipt, _RERANK_SELECTION_FIELDS, "candidate_wave_rerank_selection"
    )
    if raw["schema_version"] != RERANK_SELECTION_RECEIPT_VERSION:
        _fail("candidate_wave_schema_version_invalid", "rerank_selection")
    members_raw = raw["selected_members"]
    if isinstance(members_raw, (str, bytes)) or not isinstance(
        members_raw, Sequence
    ):
        _fail("candidate_wave_rerank_selected_members_invalid")
    members: List[Dict[str, Any]] = []
    for item in members_raw:
        member = _closed(item, _RERANK_MEMBER_FIELDS, "rerank_selected_member")
        members.append(deepcopy(dict(member)))
    if len(members) != 4:
        _fail("candidate_wave_rerank_selection_count_invalid", str(len(members)))
    hashes = [member["candidate_sequence_bundle_hash"] for member in members]
    slots = [member["slot_id"] for member in members]
    if len(hashes) != len(set(hashes)) or len(slots) != len(set(slots)):
        _fail("candidate_wave_rerank_selection_duplicate")
    by_hash = {
        member["candidate_sequence_bundle_hash"]: member
        for member in frozen["members"]
    }
    if any(sequence_hash not in by_hash for sequence_hash in hashes):
        _fail("candidate_wave_rerank_selection_outside_wave")
    expected_members = [_rerank_member(by_hash[item]) for item in hashes]
    expected_members.sort(key=lambda item: item["slot_id"])
    if _canonical_json(sorted(members, key=lambda item: item["slot_id"])) != (
        _canonical_json(expected_members)
    ):
        _fail("candidate_wave_rerank_selected_member_mismatch")
    decisions_raw = raw["role_selection_decisions"]
    if isinstance(decisions_raw, (str, bytes)) or not isinstance(
        decisions_raw, Sequence
    ):
        _fail("candidate_wave_rerank_role_decisions_invalid")
    for decision in decisions_raw:
        _closed(decision, _RERANK_DECISION_FIELDS, "rerank_role_decision")
    semantic = _rerank_semantic(frozen, expected_members)
    expected = {
        "schema_version": RERANK_SELECTION_RECEIPT_VERSION,
        **semantic,
        "rerank_selection_receipt_hash": _domain_hash(
            "candidate_wave_rerank_selection_sha256:",
            RERANK_SELECTION_RECEIPT_VERSION,
            semantic,
        ),
    }
    if _canonical_json(raw) != _canonical_json(expected):
        _fail("candidate_wave_rerank_selection_receipt_mismatch")
    return deepcopy(expected)


def _sequence_delta(
    root: Mapping[str, str], sequences: Mapping[str, str]
) -> set[Tuple[str, int]]:
    if set(root) != set(sequences):
        _fail("candidate_wave_sequence_shape_mismatch")
    result: set[Tuple[str, int]] = set()
    for chain in root:
        if len(root[chain]) != len(sequences[chain]):
            _fail("candidate_wave_sequence_shape_mismatch", chain)
        result.update(
            (chain, position)
            for position, (before, after) in enumerate(
                zip(root[chain], sequences[chain])
            )
            if before != after
        )
    return result


def _row_sequence_hash(row: Mapping[str, Any]) -> str | None:
    for field in (
        "candidate_sequence_bundle_hash",
        "sequence_bundle_hash",
        "seq_hash",
    ):
        value = row.get(field)
        if isinstance(value, str) and value.startswith("sequence_sha256:"):
            return value
    try:
        return _sequence_hash(_candidate_sequences(row))
    except CandidateWaveError:
        return None


def _provider_name(row: Mapping[str, Any]) -> str:
    containers = [row]
    for field in ("structure_evidence", "provider_receipt", "evaluation"):
        value = row.get(field)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for field in ("provider", "dispatch_provider", "structure_provider", "backend"):
            value = container.get(field)
            if isinstance(value, str) and value.strip():
                name = value.strip().lower()
                if name in {"af3", "alphafold_3", "alphafold-3"}:
                    return "alphafold3"
                return name
    return "unknown"


def _provider_succeeded(row: Mapping[str, Any]) -> bool:
    explicit = row.get("provider_call_succeeded")
    if isinstance(explicit, bool):
        return explicit
    for field in ("error", "provider_error", "failure", "failure_reason"):
        if row.get(field):
            return False
    status = str(row.get("status") or row.get("provider_status") or "").lower()
    return status not in {"failed", "failure", "error", "timeout", "cancelled"}


def build_node_engagement_ledger(
    manifest: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    compiled_action: CompiledDesignAction | Mapping[str, Any],
    node_changes: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    provider_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    generated_minimum: int = 2,
    frozen_wave_minimum: int = 2,
    protenix_attempt_minimum: int = 1,
) -> Dict[str, Any]:


    thresholds = {
        "generated_unique_mutant_candidates": generated_minimum,
        "frozen_wave_unique_mutant_candidates": frozen_wave_minimum,
        "protenix_attempted_unique_candidates": protenix_attempt_minimum,
    }
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _fail("candidate_wave_engagement_threshold_invalid", name)

    embedded_wave = (
        manifest.get("candidate_wave_request")
        if isinstance(manifest, Mapping)
        else None
    )
    if not isinstance(embedded_wave, Mapping):
        _fail("candidate_wave_manifest_request_missing")
    frozen = validate_frozen_candidate_wave(manifest, embedded_wave)
    action = _strict_action(compiled_action)
    if action.compiled_design_action_hash != embedded_wave["compiled_design_action_hash"]:
        _fail("candidate_wave_engagement_action_mismatch")
    normalized_changes = _normalize_node_changes(node_changes, action)
    if _canonical_json(normalized_changes) != _canonical_json(embedded_wave["node_changes"]):
        _fail("candidate_wave_engagement_node_changes_mismatch")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        _fail("candidate_wave_candidates_invalid")
    if provider_rows is not None and (
        isinstance(provider_rows, (str, bytes)) or not isinstance(provider_rows, Sequence)
    ):
        _fail("candidate_wave_provider_rows_invalid")
    root = dict(action.reconciled_root_sequences)
    owner_by_position = {
        (str(chain), int(position)): str(owner)
        for chain, entries in action.active_position_owners.items()
        for position, owner in entries.items()
    }
    generated_by_node: Dict[str, set[str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _fail("candidate_wave_candidate_invalid")
        sequences, sequence_hash, _ref = _validate_candidate_against_action(candidate, action)
        for position in _sequence_delta(root, sequences):
            owner = owner_by_position.get(position)
            if owner:
                generated_by_node.setdefault(owner, set()).add(sequence_hash)
    member_hashes = {
        str(member["candidate_sequence_bundle_hash"]): dict(member["seqs"])
        for member in frozen["members"]
    }
    frozen_by_node: Dict[str, set[str]] = {}
    for sequence_hash, sequences in member_hashes.items():
        for position in _sequence_delta(root, sequences):
            owner = owner_by_position.get(position)
            if owner:
                frozen_by_node.setdefault(owner, set()).add(sequence_hash)
    provider_attempts: Dict[str, Dict[str, set[str]]] = {}
    provider_successes: Dict[str, Dict[str, set[str]]] = {}
    provider_call_counts: Dict[str, Dict[str, int]] = {}
    if provider_rows is not None:
        for row in provider_rows:
            if not isinstance(row, Mapping):
                _fail("candidate_wave_provider_row_invalid")
            sequence_hash = _row_sequence_hash(row)
            if sequence_hash not in member_hashes:
                continue
            provider = _provider_name(row)
            sequences = member_hashes[sequence_hash]
            owners = {
                owner_by_position[position]
                for position in _sequence_delta(root, sequences)
                if position in owner_by_position
            }
            for owner in owners:
                provider_attempts.setdefault(owner, {}).setdefault(
                    provider, set()
                ).add(sequence_hash)
                provider_call_counts.setdefault(owner, {})[provider] = (
                    provider_call_counts.setdefault(owner, {}).get(provider, 0) + 1
                )
                if _provider_succeeded(row):
                    provider_successes.setdefault(owner, {}).setdefault(
                        provider, set()
                    ).add(sequence_hash)
    entries: List[Dict[str, Any]] = []
    for change in normalized_changes:
        if change["change"] not in _LIVE_NODE_CHANGE_KINDS:
            continue
        node_id = change["node_id"]
        protenix_attempted = len(
            provider_attempts.get(node_id, {}).get("protenix", set())
        )
        entry = {
            "node_id": node_id,
            "change": change["change"],
            "active_positions": deepcopy(change["active_positions"]),
            "added_positions": deepcopy(change["added_positions"]),
            "generated_unique_mutant_candidate_count": len(generated_by_node.get(node_id, set())),
            "frozen_wave_unique_mutant_candidate_count": len(frozen_by_node.get(node_id, set())),
            "provider_attempted_unique_candidates": {
                provider: len(hashes)
                for provider, hashes in sorted(provider_attempts.get(node_id, {}).items())
            },
            "provider_succeeded_unique_candidates": {
                provider: len(hashes)
                for provider, hashes in sorted(provider_successes.get(node_id, {}).items())
            },
            "provider_call_counts": dict(sorted(provider_call_counts.get(node_id, {}).items())),
            "generation_engagement_pass": (
                len(generated_by_node.get(node_id, set())) >= generated_minimum
            ),
            "frozen_wave_engagement_pass": (
                len(frozen_by_node.get(node_id, set())) >= frozen_wave_minimum
            ),
            "protenix_engagement_pass": (
                protenix_attempted >= protenix_attempt_minimum
                if provider_rows is not None
                else None
            ),
        }
        entries.append(entry)
    all_generation = bool(entries) and all(
        entry["generation_engagement_pass"] for entry in entries
    )
    all_frozen = bool(entries) and all(
        entry["frozen_wave_engagement_pass"] for entry in entries
    )
    all_protenix = (
        bool(entries)
        and all(entry["protenix_engagement_pass"] for entry in entries)
        if provider_rows is not None
        else None
    )
    semantic = {
        "frozen_candidate_wave_hash": frozen["frozen_candidate_wave_hash"],
        "compiled_design_action_hash": action.compiled_design_action_hash,
        "node_change_set_hash": embedded_wave["node_change_set_hash"],
        "engagement_thresholds": thresholds,
        "provider_rows_present": provider_rows is not None,
        "changed_node_count": len(entries),
        "entries": entries,
        "all_changed_nodes_generation_engaged": all_generation,
        "all_changed_nodes_frozen_wave_engaged": all_frozen,
        "all_changed_nodes_protenix_engaged": all_protenix,
        "primary_node_engagement": 1.0 if all_generation and all_frozen else 0.0,
    }
    return {
        "schema_version": NODE_ENGAGEMENT_LEDGER_VERSION,
        **semantic,
        "node_engagement_ledger_hash": _domain_hash(
            "node_engagement_ledger_sha256:",
            NODE_ENGAGEMENT_LEDGER_VERSION,
            semantic,
        ),
    }


__all__ = [
    "CANDIDATE_WAVE_REQUEST_VERSION",
    "CANDIDATE_WAVE_SLOT_DIRECTIVE_VERSION",
    "CANDIDATE_WAVE_SLOT_REALIZATION_VERSION",
    "FROZEN_CANDIDATE_WAVE_VERSION",
    "CANDIDATE_WAVE_UNDERFILL_VERSION",
    "NODE_ENGAGEMENT_LEDGER_VERSION",
    "RERANK_SELECTION_RECEIPT_VERSION",
    "DEFAULT_ROLE_HISTOGRAM",
    "RERANK_ROLE_HISTOGRAM",
    "CandidateWaveError",
    "build_candidate_wave_request",
    "build_free_slot_directives",
    "freeze_candidate_wave",
    "validate_candidate_wave_request",
    "validate_frozen_candidate_wave",
    "build_rerank_selection_receipt",
    "validate_rerank_selection_receipt",
    "build_node_engagement_ledger",
]



from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from astevolve.domain.compiled import (
    CompiledCandidateSlot,
    CompiledPortfolioOptimizationRequest,
)
from astevolve.search.mutation_move import finalize_mutation_move
from engine.experiment_identity import SequenceBundleIdentity


PORTFOLIO_REALIZATION_RECEIPT_VERSION = (
    "astevolve.portfolio_realization_receipt.v1"
)
PORTFOLIO_PAIR_RECEIPT_VERSION = "astevolve.portfolio_pair_receipt.v1"
PORTFOLIO_MATERIALIZATION_VERSION = "astevolve.portfolio_materialization.v1"

_REALIZATION_FIELDS = frozenset(
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
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "compiled_portfolio_request_hash",
        "matched_pair_id",
        "treatment_slot_id",
        "treatment_slot_hash",
        "treatment_sequence_bundle_hash",
        "control_slot_id",
        "control_slot_hash",
        "control_sequence_bundle_hash",
        "declared_ablation_positions",
        "expected_hamming_distance",
        "actual_hamming_distance",
        "background_difference_count",
        "pair_receipt_hash",
    }
)
_PORTFOLIO_CLAIM_FIELDS = frozenset(
    {
        "compiled_portfolio_request_hash",
        "portfolio_realization_receipts",
        "portfolio_pair_receipts",
        "portfolio_realization_summary",
        "prebuilt_exact_duplicate_receipts",
    }
)


class PortfolioRuntimeError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise PortfolioRuntimeError(code, detail)


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
        _fail("portfolio_not_canonical_json", str(error))


def _domain_hash(prefix: str, domain: str, semantic: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        (domain + "\0" + _canonical_json(semantic)).encode("utf-8")
    ).hexdigest()
    return prefix + digest


def _sequence_hash(sequences: Mapping[str, str]) -> str:
    try:
        return SequenceBundleIdentity.create(
            {str(chain): str(sequence) for chain, sequence in sequences.items()}
        ).sequence_bundle_hash
    except (TypeError, ValueError) as error:
        _fail("portfolio_sequence_bundle_invalid", str(error))


def strict_compiled_portfolio_request(
    value: CompiledPortfolioOptimizationRequest | Mapping[str, Any],
) -> CompiledPortfolioOptimizationRequest:


    try:
        if isinstance(value, CompiledPortfolioOptimizationRequest):
            return CompiledPortfolioOptimizationRequest.from_artifact(
                value.to_artifact()
            )
        return CompiledPortfolioOptimizationRequest.from_artifact(value)
    except Exception as error:
        _fail(
            str(getattr(error, "code", "compiled_portfolio_request_invalid")),
            str(getattr(error, "detail", "") or error),
        )


def compiled_portfolio_request_from_search_state(
    compiled: Mapping[str, Any],
) -> CompiledPortfolioOptimizationRequest | None:
    raw = compiled.get("compiled_portfolio_optimization_request")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _fail("compiled_portfolio_request_artifact_invalid", "mapping required")
    request = strict_compiled_portfolio_request(raw)
    action = compiled.get("compiled_design_action")
    if not isinstance(action, Mapping):
        _fail("compiled_portfolio_design_action_missing")
    for field in (
        "design_action_hash",
        "compiled_design_action_hash",
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "reconciled_root_sequence_bundle_hash",
    ):
        request_field = (
            "compiled_portfolio_request_hash"
            if field == "compiled_portfolio_request_hash"
            else field
        )
        observed = getattr(request, request_field, None)
        if observed is None and field == "reconciled_root_sequence_bundle_hash":
            observed = request.reconciled_root_sequence_bundle_hash
        if str(action.get(field) or "") != str(observed or ""):
            _fail("compiled_portfolio_action_binding_mismatch", field)
    return request


def build_realization_receipt(
    request: CompiledPortfolioOptimizationRequest,
    slot: CompiledCandidateSlot,
) -> Dict[str, Any]:
    if not slot.exact or slot.exact_sequence_bundle_hash is None:
        _fail("portfolio_slot_not_exact", slot.slot_id)
    pair_id = (
        slot.portfolio_id if slot.generation_mode == "matched_ablation" else None
    )
    atomic_count = int(slot.root_delta.mutation_count)
    semantic = {
        "schema_version": PORTFOLIO_REALIZATION_RECEIPT_VERSION,
        "compiled_portfolio_request_hash": request.compiled_portfolio_request_hash,
        "slot_id": slot.slot_id,
        "slot_hash": slot.slot_hash,
        "portfolio_id": slot.portfolio_id,
        "role": slot.role,
        "generation_mode": slot.generation_mode,
        "candidate_sequence_bundle_hash": slot.exact_sequence_bundle_hash,
        "expected_sequence_bundle_hash": slot.exact_sequence_bundle_hash,
        "exact_sequence_match": True,
        "atomic_requested_count": atomic_count,
        "atomic_realized_count": atomic_count,
        "partial_realization_count": 0,
        "matched_pair_id": pair_id,
    }
    return {
        **semantic,
        "receipt_hash": _domain_hash(
            "portfolio_realization_sha256:",
            PORTFOLIO_REALIZATION_RECEIPT_VERSION,
            semantic,
        ),
    }


def _sequence_differences(
    left: Mapping[str, str], right: Mapping[str, str]
) -> set[Tuple[str, int]]:
    if set(left) != set(right):
        _fail("portfolio_pair_chain_set_mismatch")
    differences: set[Tuple[str, int]] = set()
    for chain in sorted(left):
        if len(left[chain]) != len(right[chain]):
            _fail("portfolio_pair_sequence_length_mismatch", chain)
        differences.update(
            (chain, position)
            for position, (a, b) in enumerate(zip(left[chain], right[chain]))
            if a != b
        )
    return differences


def _matched_pair_slots(
    request: CompiledPortfolioOptimizationRequest,
) -> Dict[str, Tuple[CompiledCandidateSlot, CompiledCandidateSlot]]:
    grouped: Dict[str, Dict[str, CompiledCandidateSlot]] = {}
    for slot in request.candidate_slots:
        if slot.generation_mode != "matched_ablation":
            continue
        grouped.setdefault(slot.portfolio_id, {})[slot.realization_kind] = slot
    pairs: Dict[str, Tuple[CompiledCandidateSlot, CompiledCandidateSlot]] = {}
    for pair_id, members in sorted(grouped.items()):
        if set(members) != {"matched_treatment", "matched_control"}:
            _fail("matched_pair_members_incomplete", pair_id)
        pairs[pair_id] = (
            members["matched_treatment"],
            members["matched_control"],
        )
    return pairs


def build_pair_receipt(
    request: CompiledPortfolioOptimizationRequest,
    pair_id: str,
    treatment: CompiledCandidateSlot,
    control: CompiledCandidateSlot,
) -> Dict[str, Any]:
    differences = _sequence_differences(
        treatment.exact_sequences, control.exact_sequences
    )
    treatment_positions = {
        (item.chain_id, item.position) for item in treatment.root_delta.changes
    }
    control_positions = {
        (item.chain_id, item.position) for item in control.root_delta.changes
    }
    declared_ablation = treatment_positions - control_positions
    background_difference_count = len(differences - declared_ablation)
    semantic = {
        "schema_version": PORTFOLIO_PAIR_RECEIPT_VERSION,
        "compiled_portfolio_request_hash": request.compiled_portfolio_request_hash,
        "matched_pair_id": pair_id,
        "treatment_slot_id": treatment.slot_id,
        "treatment_slot_hash": treatment.slot_hash,
        "treatment_sequence_bundle_hash": treatment.exact_sequence_bundle_hash,
        "control_slot_id": control.slot_id,
        "control_slot_hash": control.slot_hash,
        "control_sequence_bundle_hash": control.exact_sequence_bundle_hash,
        "declared_ablation_positions": [
            {"chain_id": chain, "position": position}
            for chain, position in sorted(declared_ablation)
        ],
        "expected_hamming_distance": len(declared_ablation),
        "actual_hamming_distance": len(differences),
        "background_difference_count": background_difference_count,
    }
    if (
        not declared_ablation
        or len(differences) != len(declared_ablation)
        or background_difference_count != 0
    ):
        _fail("matched_pair_background_mismatch", pair_id)
    return {
        **semantic,
        "pair_receipt_hash": _domain_hash(
            "portfolio_pair_sha256:", PORTFOLIO_PAIR_RECEIPT_VERSION, semantic
        ),
    }


def _position_owners(
    request: CompiledPortfolioOptimizationRequest,
    slot: CompiledCandidateSlot,
) -> List[Dict[str, Any]]:
    module_owners = {
        (change.chain_id, change.position): change.owner_node_id
        for module in request.mutation_modules
        for change in module.mutations
    }
    seed_owners = {
        (change.chain_id, change.position): change.owner_node_id
        for seed in request.sequence_seeds
        for change in seed.root_delta.changes
    }
    owners = {**seed_owners, **module_owners}
    return [
        {
            "chain_id": change.chain_id,
            "position": change.position,
            "owner_node_id": owners.get((change.chain_id, change.position)),
        }
        for change in slot.root_delta.changes
    ]


def _trusted_realization_summary(
    request: CompiledPortfolioOptimizationRequest,
    slots: Sequence[CompiledCandidateSlot],
) -> Dict[str, Any]:
    owners: Dict[Tuple[str, int], str] = {}
    for slot in slots:
        for row in _position_owners(request, slot):
            key = (str(row["chain_id"]), int(row["position"]))
            owner = str(row.get("owner_node_id") or "")
            if not owner:
                _fail(
                    "portfolio_position_owner_missing",
                    f"{key[0]}:{key[1]}",
                )
            prior = owners.get(key)
            if prior is not None and prior != owner:
                _fail(
                    "portfolio_position_owner_conflict",
                    f"{key[0]}:{key[1]}:{prior}/{owner}",
                )
            owners[key] = owner
    return {
        "schema_version": PORTFOLIO_MATERIALIZATION_VERSION,
        "slot_ids": sorted(slot.slot_id for slot in slots),
        "required": bool(slots) and all(slot.required for slot in slots),
        "exact_sequence_match": True,
        "position_owners": [
            {
                "chain_id": chain,
                "position": position,
                "owner_node_id": owner,
            }
            for (chain, position), owner in sorted(owners.items())
        ],
    }


def build_portfolio_prebuilt_proposals(
    value: CompiledPortfolioOptimizationRequest | Mapping[str, Any],
) -> Tuple[List[Tuple[Dict[str, str], Dict[str, Any]]], Dict[str, Any], List[Dict[str, Any]]]:


    request = strict_compiled_portfolio_request(value)
    pair_receipts = {
        pair_id: build_pair_receipt(request, pair_id, treatment, control)
        for pair_id, (treatment, control) in _matched_pair_slots(request).items()
    }
    proposals: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    seed_directives: List[Dict[str, Any]] = []
    seed_by_id = {seed.seed_id: seed for seed in request.sequence_seeds}
    for slot in request.candidate_slots:
        if not slot.exact:
            continue
        receipt = build_realization_receipt(request, slot)
        pair_receipt = (
            [pair_receipts[slot.portfolio_id]]
            if slot.portfolio_id in pair_receipts
            else []
        )
        synthetic_node = f"portfolio::{slot.portfolio_id}"
        move = finalize_mutation_move(
            request.reconciled_root_sequences,
            slot.exact_sequences,
            {
                "op": "portfolio_exact",
                "node": synthetic_node,
                "attempted_positions": {
                    chain: sorted(
                        change.position
                        for change in slot.root_delta.changes
                        if change.chain_id == chain
                    )
                    for chain in sorted(slot.exact_sequences)
                },
                "proposal_log_prior": 0.0,
                "mutation_plan": {
                    "tier": "portfolio_exact",
                    "slot_id": slot.slot_id,
                    "portfolio_id": slot.portfolio_id,
                    "generation_mode": slot.generation_mode,
                },
                "compiled_portfolio_request_hash": (
                    request.compiled_portfolio_request_hash
                ),
                "portfolio_realization_receipts": [receipt],
                "portfolio_pair_receipts": pair_receipt,
                "portfolio_realization_summary": _trusted_realization_summary(
                    request, [slot]
                ),
                "portfolio_id": slot.portfolio_id,
                "portfolio_role": slot.role,
                "portfolio_slot_id": slot.slot_id,
                "portfolio_required": bool(slot.required),
                "matched_pair_id": receipt["matched_pair_id"],
                "matched_pair_member": (
                    slot.realization_kind
                    if slot.generation_mode == "matched_ablation"
                    else None
                ),
            },
        )
        proposals.append((dict(slot.exact_sequences), move))
        if slot.generation_mode == "strict_sequence_seed":
            seed_id = next(iter(slot.source_refs))
            seed = seed_by_id[seed_id]
            seed_directives.append(
                {
                    "schema_version": "astevolve.portfolio_seed_refinement.v1",
                    "seed_id": seed.seed_id,
                    "slot_id": slot.slot_id,
                    "sequence_bundle_hash": seed.sequence_bundle_hash,
                    "refinement_positions": [
                        {
                            "chain_id": chain,
                            "position": position,
                            "owner_node_id": owner,
                        }
                        for chain, position, owner in seed.refinement_positions
                    ],
                }
            )
    unique_hashes = {
        _sequence_hash(sequences) for sequences, _move in proposals
    }
    accounting = {
        "schema_version": PORTFOLIO_MATERIALIZATION_VERSION,
        "compiled_portfolio_request_hash": request.compiled_portfolio_request_hash,
        "candidate_slot_count": len(request.candidate_slots),
        "exact_slot_count": len(proposals),
        "exact_unique_sequence_count": len(unique_hashes),
        "exact_duplicate_slot_count": len(proposals) - len(unique_hashes),
        "required_exact_slot_count": sum(
            slot.required and slot.exact for slot in request.candidate_slots
        ),
        "matched_pair_count": len(pair_receipts),
        "sequence_seed_count": len(seed_directives),
        "counts_as_physical_expansion_round": False,
    }
    return proposals, accounting, seed_directives


def _slot_mapping(
    request: CompiledPortfolioOptimizationRequest,
) -> Dict[str, CompiledCandidateSlot]:
    return {slot.slot_id: slot for slot in request.candidate_slots}


def validate_realization_receipt(
    candidate_sequences: Mapping[str, str],
    receipt: Mapping[str, Any],
    request: CompiledPortfolioOptimizationRequest,
) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != _REALIZATION_FIELDS:
        _fail("portfolio_realization_receipt_schema_invalid")
    slot = _slot_mapping(request).get(str(receipt.get("slot_id") or ""))
    if slot is None:
        _fail("portfolio_realization_slot_unknown", str(receipt.get("slot_id")))
    expected = build_realization_receipt(request, slot)
    if dict(receipt) != expected:
        _fail("portfolio_realization_receipt_mismatch", slot.slot_id)
    observed_hash = _sequence_hash(candidate_sequences)
    if observed_hash != slot.exact_sequence_bundle_hash:
        _fail("portfolio_realization_sequence_mismatch", slot.slot_id)
    return expected


def validate_pair_receipt(
    receipt: Mapping[str, Any],
    request: CompiledPortfolioOptimizationRequest,
) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != _PAIR_FIELDS:
        _fail("portfolio_pair_receipt_schema_invalid")
    pair_id = str(receipt.get("matched_pair_id") or "")
    pairs = _matched_pair_slots(request)
    if pair_id not in pairs:
        _fail("portfolio_pair_unknown", pair_id)
    expected = build_pair_receipt(request, pair_id, *pairs[pair_id])
    if dict(receipt) != expected:
        _fail("portfolio_pair_receipt_mismatch", pair_id)
    return expected


def validate_candidate_portfolio_realization(
    candidate: Mapping[str, Any],
    value: CompiledPortfolioOptimizationRequest | Mapping[str, Any],
) -> Dict[str, Any]:


    request = strict_compiled_portfolio_request(value)
    move = candidate.get("move")
    claimed = bool(
        any(field in candidate for field in _PORTFOLIO_CLAIM_FIELDS)
        or (
            isinstance(move, Mapping)
            and (
                str(move.get("op") or "") == "portfolio_exact"
                or any(field in move for field in _PORTFOLIO_CLAIM_FIELDS)
            )
        )
    )
    if not claimed:
        return {"claimed": False, "validated_receipt_count": 0}
    sequences = candidate.get("seqs")
    if not isinstance(sequences, Mapping):
        _fail("portfolio_candidate_sequences_missing")
    if candidate.get("compiled_portfolio_request_hash") != (
        request.compiled_portfolio_request_hash
    ):
        _fail("portfolio_candidate_request_hash_mismatch")
    raw_receipts = candidate.get("portfolio_realization_receipts")
    if (
        isinstance(raw_receipts, (str, bytes))
        or not isinstance(raw_receipts, Sequence)
        or not raw_receipts
    ):
        _fail("portfolio_realization_receipts_missing")
    receipts = [
        validate_realization_receipt(sequences, receipt, request)
        for receipt in raw_receipts
    ]
    receipt_slot_ids = [str(receipt["slot_id"]) for receipt in receipts]
    if len(receipt_slot_ids) != len(set(receipt_slot_ids)):
        duplicate = next(
            slot_id
            for index, slot_id in enumerate(receipt_slot_ids)
            if slot_id in receipt_slot_ids[:index]
        )
        _fail("portfolio_realization_slot_duplicate", duplicate)
    receipt_hashes = [str(receipt["receipt_hash"]) for receipt in receipts]
    if len(receipt_hashes) != len(set(receipt_hashes)):
        duplicate = next(
            receipt_hash
            for index, receipt_hash in enumerate(receipt_hashes)
            if receipt_hash in receipt_hashes[:index]
        )
        _fail("portfolio_realization_receipt_hash_duplicate", duplicate)
    receipt_slot_map = _slot_mapping(request)
    receipt_slots = [receipt_slot_map[str(receipt["slot_id"])] for receipt in receipts]
    raw_pairs = candidate.get("portfolio_pair_receipts", [])
    if isinstance(raw_pairs, (str, bytes)) or not isinstance(raw_pairs, Sequence):
        _fail("portfolio_pair_receipts_invalid")
    pairs = [validate_pair_receipt(receipt, request) for receipt in raw_pairs]
    pair_ids = [str(receipt["matched_pair_id"]) for receipt in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        duplicate = next(
            pair_id
            for index, pair_id in enumerate(pair_ids)
            if pair_id in pair_ids[:index]
        )
        _fail("portfolio_pair_id_duplicate", duplicate)
    pair_hashes = [str(receipt["pair_receipt_hash"]) for receipt in pairs]
    if len(pair_hashes) != len(set(pair_hashes)):
        duplicate = next(
            pair_hash
            for index, pair_hash in enumerate(pair_hashes)
            if pair_hash in pair_hashes[:index]
        )
        _fail("portfolio_pair_receipt_hash_duplicate", duplicate)
    required_pair_ids = {
        str(receipt["matched_pair_id"])
        for receipt in receipts
        if receipt.get("matched_pair_id") is not None
    }
    observed_pair_ids = {str(receipt["matched_pair_id"]) for receipt in pairs}
    if required_pair_ids != observed_pair_ids:
        _fail(
            "portfolio_pair_receipt_coverage_mismatch",
            f"{sorted(required_pair_ids)}/{sorted(observed_pair_ids)}",
        )
    return {
        "claimed": True,
        "compiled_portfolio_request_hash": request.compiled_portfolio_request_hash,
        "candidate_sequence_bundle_hash": _sequence_hash(sequences),
        "validated_receipt_count": len(receipts),
        "validated_slot_ids": sorted(receipt["slot_id"] for receipt in receipts),
        "validated_pair_count": len(pairs),
        "validated_pair_ids": sorted(observed_pair_ids),
        "trusted_realization_summary": _trusted_realization_summary(
            request, receipt_slots
        ),
    }


__all__ = [
    "PORTFOLIO_MATERIALIZATION_VERSION",
    "PORTFOLIO_PAIR_RECEIPT_VERSION",
    "PORTFOLIO_REALIZATION_RECEIPT_VERSION",
    "PortfolioRuntimeError",
    "build_pair_receipt",
    "build_portfolio_prebuilt_proposals",
    "build_realization_receipt",
    "compiled_portfolio_request_from_search_state",
    "strict_compiled_portfolio_request",
    "validate_candidate_portfolio_realization",
    "validate_pair_receipt",
    "validate_realization_receipt",
]

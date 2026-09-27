

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import hashlib
import json
import math
import random
import re
import threading
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable
import weakref


SEQUENCE_GENERATION_REQUEST_VERSION = "astevolve.sequence_generation_request.v1"
SEQUENCE_GENERATION_RESULT_VERSION = "astevolve.sequence_generation_result.v1"
CONSTRAINT_AWARE_GENERATOR_ID = "deterministic_constraint_aware_v1"
UNIFORM_RANDOM_BASELINE_ID = "uniform_random_baseline_v1"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "parent_sequences",
        "mapping_edge_id",
        "action_id",
        "structural_node_id",
        "operator",
        "target_chain",
        "write_positions",
        "mutation_budget",
        "mutable_masks",
        "fixed_residues",
        "allowed_residues",
        "soft_residue_weights",
        "seed",
        "step",
        "structure_condition_refs",
        "state_condition_refs",
        "request_hash",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "request_hash",
        "generator_id",
        "generator_version",
        "sequences",
        "changed_positions",
        "proposal_log_likelihood",
        "confidence",
        "heuristic_score",
        "constraint_violations",
        "provenance",
        "result_hash",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "algorithm",
        "deterministic",
        "baseline",
        "seed",
        "step",
        "mapping_edge_id",
        "action_id",
        "structural_node_id",
        "operator",
        "condition_refs_available",
        "condition_refs_consumed",
    }
)
_CONDITION_FIELDS = frozenset({"structure", "state"})
_VIOLATION_FIELDS = frozenset({"code", "chain_id", "position", "detail"})
_PARENT_SEQUENCE_RE = re.compile(r"[A-Z]+\Z")
_FACTORY_CONTRACT_STATES: Dict[
    int, Tuple[weakref.ReferenceType[Any], str, str, str]
] = {}
_FACTORY_CONTRACT_LOCK = threading.RLock()


class SequenceGenerationError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: Any = "") -> None:
    raise SequenceGenerationError(code, str(detail))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail("not_canonical_json", exc)


def _freeze_json(value: Any) -> Any:


    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:


    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _digest(domain: str, value: Any) -> str:
    encoded = f"{domain}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remember_factory_contract(contract: Any, digest_name: str) -> Any:


    identity = id(contract)

    def discard(reference: weakref.ReferenceType[Any]) -> None:
        with _FACTORY_CONTRACT_LOCK:
            current = _FACTORY_CONTRACT_STATES.get(identity)
            if current is not None and current[0] is reference:
                _FACTORY_CONTRACT_STATES.pop(identity, None)

    reference = weakref.ref(contract, discard)
    state = (
        reference,
        str(contract._payload_json),
        str(getattr(contract, digest_name)),
        str(contract.schema_version),
    )
    with _FACTORY_CONTRACT_LOCK:
        _FACTORY_CONTRACT_STATES[identity] = state
    return contract


def _is_untampered_factory_contract(contract: Any, digest_name: str) -> bool:
    with _FACTORY_CONTRACT_LOCK:
        state = _FACTORY_CONTRACT_STATES.get(id(contract))
    return bool(
        state is not None
        and state[0]() is contract
        and state[1] == getattr(contract, "_payload_json", None)
        and state[2] == getattr(contract, digest_name, None)
        and state[3] == getattr(contract, "schema_version", None)
    )


def _mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("sequence_required", label)
    return value


def _closed(value: Mapping[Any, Any], expected: frozenset[str], label: str) -> None:
    keys = {str(key) for key in value}
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    if unknown:
        _fail("unknown_fields", f"{label}:{','.join(unknown)}")
    if missing:
        _fail("fields_missing", f"{label}:{','.join(missing)}")


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _integer(value: Any, code: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(code, repr(value))
    if minimum is not None and value < minimum:
        _fail(code, repr(value))
    return value


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, repr(value))
    resolved = float(value)
    if not math.isfinite(resolved):
        _fail(code, repr(value))
    return resolved


def _normalize_parent_sequences(value: Any) -> Dict[str, str]:
    raw = _mapping(value, "parent_sequences")
    if not raw:
        _fail("parent_sequences_empty")
    out: Dict[str, str] = {}
    for raw_chain, raw_sequence in raw.items():
        if not isinstance(raw_chain, str):
            _fail("chain_id_invalid", repr(raw_chain))
        chain = _text(raw_chain, "chain_id_invalid")
        if chain in out:
            _fail("chain_id_duplicate", chain)
        if not isinstance(raw_sequence, str) or _PARENT_SEQUENCE_RE.fullmatch(raw_sequence) is None:
            _fail("parent_sequence_invalid", chain)
        out[chain] = raw_sequence
    return dict(sorted(out.items()))


def _normalize_positions(value: Any, label: str, *, allow_empty: bool = False) -> Tuple[int, ...]:
    raw = _sequence(value, label)
    positions = tuple(_integer(item, f"{label}_invalid", minimum=0) for item in raw)
    if not positions and not allow_empty:
        _fail(f"{label}_empty")
    if len(set(positions)) != len(positions):
        _fail(f"{label}_duplicate")
    return tuple(sorted(positions))


def _normalize_refs(value: Any, label: str) -> Tuple[str, ...]:
    raw = _sequence(value, label)
    refs = tuple(_text(item, f"{label}_invalid") for item in raw)
    if len(set(refs)) != len(refs):
        _fail(f"{label}_duplicate")
    return tuple(sorted(refs))


def _normalize_masks(value: Any, parents: Mapping[str, str]) -> Dict[str, list[bool]]:
    raw = _mapping(value, "mutable_masks")
    if {str(key) for key in raw} != set(parents):
        _fail("mutable_mask_chains_mismatch")
    out: Dict[str, list[bool]] = {}
    for chain, parent in parents.items():
        mask = _sequence(raw[chain], f"mutable_masks.{chain}")
        if len(mask) != len(parent) or any(not isinstance(flag, bool) for flag in mask):
            _fail("mutable_mask_invalid", chain)
        out[chain] = list(mask)
    return out


def _position_key(value: Any, *, chain: str, label: str) -> int:
    if isinstance(value, bool):
        _fail(f"{label}_position_invalid", f"{chain}:{value!r}")
    try:
        position = int(value)
    except (TypeError, ValueError):
        _fail(f"{label}_position_invalid", f"{chain}:{value!r}")
    if str(position) != str(value) and not isinstance(value, int):
        _fail(f"{label}_position_invalid", f"{chain}:{value!r}")
    if position < 0:
        _fail(f"{label}_position_invalid", f"{chain}:{position}")
    return position


def _normalize_fixed(value: Any, parents: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    raw = _mapping(value, "fixed_residues")
    unknown_chains = sorted(str(key) for key in raw if str(key) not in parents)
    if unknown_chains:
        _fail("fixed_residue_chain_unknown", ",".join(unknown_chains))
    out: Dict[str, Dict[str, str]] = {}
    for chain in sorted(str(key) for key in raw):
        if chain not in raw:

            _fail("fixed_residue_chain_invalid", chain)
        entries = _mapping(raw[chain], f"fixed_residues.{chain}")
        clean: Dict[str, str] = {}
        for raw_position, raw_residue in entries.items():
            position = _position_key(raw_position, chain=chain, label="fixed_residue")
            if position >= len(parents[chain]):
                _fail("fixed_residue_position_out_of_range", f"{chain}:{position}")
            if not isinstance(raw_residue, str) or len(raw_residue) != 1:
                _fail("fixed_residue_invalid", f"{chain}:{position}")
            if parents[chain][position] != raw_residue:
                _fail("fixed_residue_parent_mismatch", f"{chain}:{position}")
            key = str(position)
            if key in clean:
                _fail("fixed_residue_position_duplicate", f"{chain}:{position}")
            clean[key] = raw_residue
        out[chain] = dict(sorted(clean.items(), key=lambda item: int(item[0])))
    return out


def _normalize_allowed(
    value: Any,
    *,
    target_chain: str,
    write_positions: Tuple[int, ...],
) -> Dict[str, Dict[str, list[str]]]:
    raw = _mapping(value, "allowed_residues")
    if set(raw) != {target_chain}:
        _fail("allowed_residue_chains_mismatch")
    entries = _mapping(raw[target_chain], f"allowed_residues.{target_chain}")
    clean: Dict[str, list[str]] = {}
    for raw_position, raw_residues in entries.items():
        position = _position_key(raw_position, chain=target_chain, label="allowed_residue")
        residues = _sequence(raw_residues, f"allowed_residues.{target_chain}.{position}")
        normalized = []
        for residue in residues:
            if not isinstance(residue, str) or residue not in CANONICAL_AMINO_ACIDS:
                _fail("allowed_residue_invalid", f"{target_chain}:{position}:{residue!r}")
            normalized.append(residue)
        if not normalized:
            _fail("allowed_residues_empty", f"{target_chain}:{position}")
        if len(set(normalized)) != len(normalized):
            _fail("allowed_residue_duplicate", f"{target_chain}:{position}")
        key = str(position)
        if key in clean:
            _fail("allowed_position_duplicate", f"{target_chain}:{position}")
        clean[key] = sorted(normalized)
    if {int(key) for key in clean} != set(write_positions):
        _fail("allowed_positions_mismatch")
    return {
        target_chain: dict(sorted(clean.items(), key=lambda item: int(item[0])))
    }


def _normalize_soft_weights(
    value: Any,
    *,
    target_chain: str,
    allowed: Mapping[str, Mapping[str, Sequence[str]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    raw = _mapping(value, "soft_residue_weights")
    if set(raw) - {target_chain}:
        _fail("soft_weight_chain_unknown")
    entries = _mapping(raw.get(target_chain, {}), f"soft_residue_weights.{target_chain}")
    clean: Dict[str, Dict[str, float]] = {}
    allowed_positions = allowed[target_chain]
    for raw_position, raw_weights in entries.items():
        position = _position_key(raw_position, chain=target_chain, label="soft_weight")
        key = str(position)
        if key not in allowed_positions:
            _fail("soft_weight_position_not_allowed", f"{target_chain}:{position}")
        weights = _mapping(raw_weights, f"soft_residue_weights.{target_chain}.{position}")
        residue_weights: Dict[str, float] = {}
        for raw_residue, raw_weight in weights.items():
            residue = str(raw_residue)
            if residue not in allowed_positions[key]:
                _fail("soft_weight_residue_not_allowed", f"{target_chain}:{position}:{residue}")
            weight = _finite(raw_weight, "soft_weight_invalid")
            if weight < 0.0:
                _fail("soft_weight_invalid", f"{target_chain}:{position}:{weight}")
            residue_weights[residue] = weight
        clean[key] = dict(sorted(residue_weights.items()))
    return {
        target_chain: dict(sorted(clean.items(), key=lambda item: int(item[0])))
    }


def _normalize_budget(value: Any, position_count: int) -> Dict[str, int]:
    raw = _mapping(value, "mutation_budget")
    _closed(raw, frozenset({"min", "max"}), "mutation_budget")
    minimum = _integer(raw["min"], "mutation_budget_invalid", minimum=0)
    maximum = _integer(raw["max"], "mutation_budget_invalid", minimum=0)
    if minimum > maximum or maximum > position_count:
        _fail("mutation_budget_invalid", f"{minimum}:{maximum}:{position_count}")
    return {"min": minimum, "max": maximum}


def _normalize_request_payload(
    *,
    parent_sequences: Any,
    mapping_edge_id: Any,
    action_id: Any,
    structural_node_id: Any,
    operator: Any,
    target_chain: Any,
    write_positions: Any,
    mutation_budget: Any,
    mutable_masks: Any,
    fixed_residues: Any,
    allowed_residues: Any,
    soft_residue_weights: Any,
    seed: Any,
    step: Any,
    structure_condition_refs: Any,
    state_condition_refs: Any,
) -> Dict[str, Any]:
    parents = _normalize_parent_sequences(parent_sequences)
    chain = _text(target_chain, "target_chain_required")
    if chain not in parents:
        _fail("target_chain_unknown", chain)
    positions = _normalize_positions(write_positions, "write_positions")
    for position in positions:
        if position >= len(parents[chain]):
            _fail("write_position_out_of_range", f"{chain}:{position}")
    budget = _normalize_budget(mutation_budget, len(positions))
    masks = _normalize_masks(mutable_masks, parents)
    fixed = _normalize_fixed(fixed_residues, parents)
    for position in positions:
        if not masks[chain][position]:
            _fail("write_position_frozen", f"{chain}:{position}")
        if str(position) in fixed.get(chain, {}):
            _fail("write_position_fixed", f"{chain}:{position}")
    allowed = _normalize_allowed(
        allowed_residues,
        target_chain=chain,
        write_positions=positions,
    )
    soft = _normalize_soft_weights(
        soft_residue_weights,
        target_chain=chain,
        allowed=allowed,
    )
    parent = parents[chain]
    mandatory = 0
    mutable_alternatives = 0
    for position in positions:
        choices = allowed[chain][str(position)]
        current = parent[position]
        if current not in choices:
            mandatory += 1
        if any(residue != current for residue in choices):
            mutable_alternatives += 1
    if mandatory > budget["max"] or mutable_alternatives < budget["min"]:
        _fail(
            "mutation_budget_infeasible",
            f"mandatory={mandatory},alternatives={mutable_alternatives},budget={budget}",
        )
    return {
        "parent_sequences": parents,
        "mapping_edge_id": _text(mapping_edge_id, "mapping_edge_id_required"),
        "action_id": _text(action_id, "action_id_required"),
        "structural_node_id": _text(structural_node_id, "structural_node_id_required"),
        "operator": _text(operator, "operator_required"),
        "target_chain": chain,
        "write_positions": list(positions),
        "mutation_budget": budget,
        "mutable_masks": masks,
        "fixed_residues": fixed,
        "allowed_residues": allowed,
        "soft_residue_weights": soft,
        "seed": _integer(seed, "seed_invalid"),
        "step": _integer(step, "step_invalid", minimum=0),
        "structure_condition_refs": list(
            _normalize_refs(structure_condition_refs, "structure_condition_refs")
        ),
        "state_condition_refs": list(
            _normalize_refs(state_condition_refs, "state_condition_refs")
        ),
    }


@dataclass(frozen=True)
class SequenceGenerationRequest:


    _payload_json: str
    request_hash: str
    schema_version: str = SEQUENCE_GENERATION_REQUEST_VERSION

    @classmethod
    def create(cls, **values: Any) -> "SequenceGenerationRequest":
        payload = _normalize_request_payload(**values)
        return _remember_factory_contract(
            cls(
                _payload_json=_canonical_json(payload),
                request_hash=_digest(SEQUENCE_GENERATION_REQUEST_VERSION, payload),
            ),
            "request_hash",
        )

    @cached_property
    def _payload(self) -> Mapping[str, Any]:


        return _freeze_json(json.loads(self._payload_json))

    def __getstate__(self) -> Dict[str, Any]:


        state = dict(self.__dict__)
        state.pop("_payload", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    @property
    def parent_sequences(self) -> Dict[str, str]:
        return dict(self._payload["parent_sequences"])

    @property
    def mapping_edge_id(self) -> str:
        return self._payload["mapping_edge_id"]

    @property
    def action_id(self) -> str:
        return self._payload["action_id"]

    @property
    def structural_node_id(self) -> str:
        return self._payload["structural_node_id"]

    @property
    def operator(self) -> str:
        return self._payload["operator"]

    @property
    def target_chain(self) -> str:
        return self._payload["target_chain"]

    @property
    def write_positions(self) -> Tuple[int, ...]:
        return tuple(self._payload["write_positions"])

    @property
    def mutation_budget(self) -> Dict[str, int]:
        return dict(self._payload["mutation_budget"])

    @property
    def mutable_masks(self) -> Dict[str, list[bool]]:
        return {
            chain: list(values)
            for chain, values in self._payload["mutable_masks"].items()
        }

    @property
    def fixed_residues(self) -> Dict[str, Dict[int, str]]:
        return {
            chain: {int(position): residue for position, residue in values.items()}
            for chain, values in self._payload["fixed_residues"].items()
        }

    @property
    def allowed_residues(self) -> Dict[str, Dict[int, Tuple[str, ...]]]:
        return {
            chain: {
                int(position): tuple(residues)
                for position, residues in values.items()
            }
            for chain, values in self._payload["allowed_residues"].items()
        }

    @property
    def soft_residue_weights(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        return {
            chain: {
                int(position): dict(weights)
                for position, weights in values.items()
            }
            for chain, values in self._payload["soft_residue_weights"].items()
        }

    @property
    def seed(self) -> int:
        return self._payload["seed"]

    @property
    def step(self) -> int:
        return self._payload["step"]

    @property
    def structure_condition_refs(self) -> Tuple[str, ...]:
        return tuple(self._payload["structure_condition_refs"])

    @property
    def state_condition_refs(self) -> Tuple[str, ...]:
        return tuple(self._payload["state_condition_refs"])

    def to_dict(self) -> Dict[str, Any]:


        payload = json.loads(self._payload_json)
        return {
            "schema_version": self.schema_version,
            **payload,
            "request_hash": self.request_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequenceGenerationRequest":
        raw = _mapping(value, "sequence_generation_request")
        _closed(raw, _REQUEST_FIELDS, "sequence_generation_request")
        if raw.get("schema_version") != SEQUENCE_GENERATION_REQUEST_VERSION:
            _fail("schema_version_invalid", raw.get("schema_version"))
        contract = cls.create(
            **{
                key: raw[key]
                for key in _REQUEST_FIELDS
                if key not in {"schema_version", "request_hash"}
            }
        )
        if raw.get("request_hash") != contract.request_hash:
            _fail("hash_mismatch", "sequence_generation_request")
        return contract


def _changed_positions(
    parent_sequences: Mapping[str, str], sequences: Mapping[str, str]
) -> Dict[str, list[int]]:
    out: Dict[str, list[int]] = {}
    for chain in sorted(set(parent_sequences) | set(sequences)):
        before = str(parent_sequences.get(chain, ""))
        after = str(sequences.get(chain, ""))
        changed = [
            position
            for position in range(max(len(before), len(after)))
            if (before[position] if position < len(before) else None)
            != (after[position] if position < len(after) else None)
        ]
        if changed:
            out[chain] = changed
    return out


def _violation(
    code: str,
    *,
    chain_id: Optional[str] = None,
    position: Optional[int] = None,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "code": str(code),
        "chain_id": chain_id,
        "position": position,
        "detail": str(detail),
    }


def _normalize_violations(value: Any) -> Tuple[Dict[str, Any], ...]:
    raw = _sequence(value, "constraint_violations")
    clean = []
    for index, item in enumerate(raw):
        entry = _mapping(item, f"constraint_violations[{index}]")
        _closed(entry, _VIOLATION_FIELDS, f"constraint_violations[{index}]")
        chain = entry["chain_id"]
        if chain is not None:
            chain = _text(chain, "violation_chain_invalid")
        position = entry["position"]
        if position is not None:
            position = _integer(position, "violation_position_invalid", minimum=0)
        clean.append(
            _violation(
                _text(entry["code"], "violation_code_required"),
                chain_id=chain,
                position=position,
                detail=str(entry["detail"]),
            )
        )
    return tuple(
        sorted(
            clean,
            key=lambda item: (
                item["code"],
                item["chain_id"] or "",
                -1 if item["position"] is None else item["position"],
                item["detail"],
            ),
        )
    )


def inspect_generation_constraints(
    request: SequenceGenerationRequest,
    sequences: Mapping[str, str],
) -> Tuple[Dict[str, Any], ...]:


    return _inspect_generation_constraints_with_delta(request, sequences)[1]


def _inspect_generation_constraints_with_delta(
    request: SequenceGenerationRequest,
    sequences: Mapping[str, str],
) -> Tuple[Dict[str, list[int]], Tuple[Dict[str, Any], ...]]:


    if not isinstance(request, SequenceGenerationRequest):
        _fail("request_type_invalid")
    raw_sequences = _mapping(sequences, "generated_sequences")
    parents = request.parent_sequences
    normalized_sequences = {
        str(key): value
        for key, value in raw_sequences.items()
        if isinstance(value, str)
    }
    changed = _changed_positions(parents, normalized_sequences)
    observed_chains = {str(key) for key in raw_sequences}
    violations = []
    for chain in sorted(set(parents) - observed_chains):
        violations.append(_violation("sequence_chain_missing", chain_id=chain))
    for chain in sorted(observed_chains - set(parents)):
        violations.append(_violation("sequence_chain_extra", chain_id=chain))

    masks = request.mutable_masks
    fixed = request.fixed_residues
    allowed = request.allowed_residues[request.target_chain]
    write_set = set(request.write_positions)
    for chain, parent in parents.items():
        raw = raw_sequences.get(chain)
        if not isinstance(raw, str):
            if chain in observed_chains:
                violations.append(_violation("generated_sequence_invalid", chain_id=chain))
            continue
        generated = raw
        if _PARENT_SEQUENCE_RE.fullmatch(generated) is None:
            violations.append(_violation("generated_sequence_invalid", chain_id=chain))
        if len(generated) != len(parent):
            violations.append(
                _violation(
                    "sequence_length_changed",
                    chain_id=chain,
                    detail=f"{len(parent)}->{len(generated)}",
                )
            )
        for position in range(max(len(parent), len(generated))):
            before = parent[position] if position < len(parent) else None
            after = generated[position] if position < len(generated) else None
            if before == after:
                continue
            if chain != request.target_chain or position not in write_set:
                violations.append(
                    _violation("change_outside_write_set", chain_id=chain, position=position)
                )
            if position >= len(masks.get(chain, [])) or not masks[chain][position]:
                violations.append(
                    _violation("frozen_position_modified", chain_id=chain, position=position)
                )
        for position, expected in fixed.get(chain, {}).items():
            actual = generated[position] if position < len(generated) else None
            if actual != expected:
                violations.append(
                    _violation(
                        "fixed_residue_modified",
                        chain_id=chain,
                        position=position,
                        detail=f"expected={expected},actual={actual}",
                    )
                )

    target = raw_sequences.get(request.target_chain)
    if isinstance(target, str):
        for position in request.write_positions:
            residue = target[position] if position < len(target) else None
            if residue not in allowed[position]:
                violations.append(
                    _violation(
                        "residue_not_allowed",
                        chain_id=request.target_chain,
                        position=position,
                        detail=str(residue),
                    )
                )
    change_count = sum(len(values) for values in changed.values())
    budget = request.mutation_budget
    if change_count < budget["min"]:
        violations.append(
            _violation(
                "mutation_budget_underflow",
                chain_id=request.target_chain,
                detail=f"{change_count}<{budget['min']}",
            )
        )
    if change_count > budget["max"]:
        violations.append(
            _violation(
                "mutation_budget_overflow",
                chain_id=request.target_chain,
                detail=f"{change_count}>{budget['max']}",
            )
        )
    return changed, _normalize_violations(violations)


def _normalize_condition_blob(value: Any, label: str) -> Dict[str, list[str]]:
    raw = _mapping(value, label)
    _closed(raw, _CONDITION_FIELDS, label)
    return {
        "structure": list(_normalize_refs(raw["structure"], f"{label}.structure")),
        "state": list(_normalize_refs(raw["state"], f"{label}.state")),
    }


def _normalize_provenance(value: Any) -> Dict[str, Any]:
    raw = _mapping(value, "provenance")
    _closed(raw, _PROVENANCE_FIELDS, "provenance")
    deterministic = raw["deterministic"]
    baseline = raw["baseline"]
    if not isinstance(deterministic, bool) or not isinstance(baseline, bool):
        _fail("provenance_boolean_invalid")
    available = _normalize_condition_blob(raw["condition_refs_available"], "condition_refs_available")
    consumed = _normalize_condition_blob(raw["condition_refs_consumed"], "condition_refs_consumed")
    for kind in _CONDITION_FIELDS:
        if not set(consumed[kind]).issubset(available[kind]):
            _fail("consumed_condition_not_available", kind)
    return {
        "algorithm": _text(raw["algorithm"], "algorithm_required"),
        "deterministic": deterministic,
        "baseline": baseline,
        "seed": _integer(raw["seed"], "seed_invalid"),
        "step": _integer(raw["step"], "step_invalid", minimum=0),
        "mapping_edge_id": _text(raw["mapping_edge_id"], "mapping_edge_id_required"),
        "action_id": _text(raw["action_id"], "action_id_required"),
        "structural_node_id": _text(raw["structural_node_id"], "structural_node_id_required"),
        "operator": _text(raw["operator"], "operator_required"),
        "condition_refs_available": available,
        "condition_refs_consumed": consumed,
    }


@dataclass(frozen=True)
class SequenceGenerationResult:


    _payload_json: str
    result_hash: str
    schema_version: str = SEQUENCE_GENERATION_RESULT_VERSION

    @classmethod
    def create(
        cls,
        *,
        request: SequenceGenerationRequest,
        generator_id: str,
        generator_version: str,
        algorithm: str,
        deterministic: bool,
        baseline: bool,
        sequences: Mapping[str, str],
        proposal_log_likelihood: float,
        confidence: float,
        heuristic_score: float,
        constraint_violations: Sequence[Mapping[str, Any]],
        consumed_structure_condition_refs: Sequence[str],
        consumed_state_condition_refs: Sequence[str],
    ) -> "SequenceGenerationResult":
        if not isinstance(request, SequenceGenerationRequest):
            _fail("request_type_invalid")
        clean_sequences = _normalize_parent_sequences(sequences)
        log_likelihood = _finite(proposal_log_likelihood, "proposal_log_likelihood_invalid")
        if log_likelihood > 1e-12:
            _fail("proposal_log_likelihood_invalid", log_likelihood)
        resolved_confidence = _finite(confidence, "confidence_invalid")
        if not 0.0 <= resolved_confidence <= 1.0:
            _fail("confidence_invalid", resolved_confidence)
        provenance = _normalize_provenance(
            {
                "algorithm": algorithm,
                "deterministic": deterministic,
                "baseline": baseline,
                "seed": request.seed,
                "step": request.step,
                "mapping_edge_id": request.mapping_edge_id,
                "action_id": request.action_id,
                "structural_node_id": request.structural_node_id,
                "operator": request.operator,
                "condition_refs_available": {
                    "structure": list(request.structure_condition_refs),
                    "state": list(request.state_condition_refs),
                },
                "condition_refs_consumed": {
                    "structure": list(consumed_structure_condition_refs),
                    "state": list(consumed_state_condition_refs),
                },
            }
        )
        payload = {
            "request_hash": request.request_hash,
            "generator_id": _text(generator_id, "generator_id_required"),
            "generator_version": _text(generator_version, "generator_version_required"),
            "sequences": clean_sequences,
            "changed_positions": _changed_positions(request.parent_sequences, clean_sequences),
            "proposal_log_likelihood": log_likelihood,
            "confidence": resolved_confidence,
            "heuristic_score": _finite(heuristic_score, "heuristic_score_invalid"),
            "constraint_violations": list(_normalize_violations(constraint_violations)),
            "provenance": provenance,
        }
        return _remember_factory_contract(
            cls(
                _payload_json=_canonical_json(payload),
                result_hash=_digest(SEQUENCE_GENERATION_RESULT_VERSION, payload),
            ),
            "result_hash",
        )

    @cached_property
    def _payload(self) -> Mapping[str, Any]:


        return _freeze_json(json.loads(self._payload_json))

    def __getstate__(self) -> Dict[str, Any]:


        state = dict(self.__dict__)
        state.pop("_payload", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    @property
    def request_hash(self) -> str:
        return self._payload["request_hash"]

    @property
    def generator_id(self) -> str:
        return self._payload["generator_id"]

    @property
    def generator_version(self) -> str:
        return self._payload["generator_version"]

    @property
    def sequences(self) -> Dict[str, str]:
        return dict(self._payload["sequences"])

    @property
    def changed_positions(self) -> Dict[str, Tuple[int, ...]]:
        return {
            chain: tuple(positions)
            for chain, positions in self._payload["changed_positions"].items()
        }

    @property
    def proposal_log_likelihood(self) -> float:
        return self._payload["proposal_log_likelihood"]

    @property
    def confidence(self) -> float:
        return self._payload["confidence"]

    @property
    def heuristic_score(self) -> float:
        return self._payload["heuristic_score"]

    @property
    def constraint_violations(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(
            dict(item) for item in self._payload["constraint_violations"]
        )

    @property
    def provenance(self) -> Dict[str, Any]:
        return _thaw_json(self._payload["provenance"])

    def to_dict(self) -> Dict[str, Any]:
        payload = json.loads(self._payload_json)
        return {
            "schema_version": self.schema_version,
            **payload,
            "result_hash": self.result_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequenceGenerationResult":
        raw = _mapping(value, "sequence_generation_result")
        _closed(raw, _RESULT_FIELDS, "sequence_generation_result")
        if raw.get("schema_version") != SEQUENCE_GENERATION_RESULT_VERSION:
            _fail("schema_version_invalid", raw.get("schema_version"))
        sequences = _normalize_parent_sequences(raw["sequences"])
        changed_raw = _mapping(raw["changed_positions"], "changed_positions")
        changed = {
            str(chain): list(_normalize_positions(positions, "changed_positions", allow_empty=True))
            for chain, positions in changed_raw.items()
        }
        if any(not values for values in changed.values()):
            _fail("changed_positions_empty_entry")
        provenance = _normalize_provenance(raw["provenance"])
        log_likelihood = _finite(raw["proposal_log_likelihood"], "proposal_log_likelihood_invalid")
        confidence = _finite(raw["confidence"], "confidence_invalid")
        if log_likelihood > 1e-12 or not 0.0 <= confidence <= 1.0:
            _fail("result_probability_invalid")
        payload = {
            "request_hash": _text(raw["request_hash"], "request_hash_required"),
            "generator_id": _text(raw["generator_id"], "generator_id_required"),
            "generator_version": _text(raw["generator_version"], "generator_version_required"),
            "sequences": sequences,
            "changed_positions": dict(sorted(changed.items())),
            "proposal_log_likelihood": log_likelihood,
            "confidence": confidence,
            "heuristic_score": _finite(raw["heuristic_score"], "heuristic_score_invalid"),
            "constraint_violations": list(_normalize_violations(raw["constraint_violations"])),
            "provenance": provenance,
        }
        result = _remember_factory_contract(
            cls(
                _payload_json=_canonical_json(payload),
                result_hash=_digest(SEQUENCE_GENERATION_RESULT_VERSION, payload),
            ),
            "result_hash",
        )
        if raw.get("result_hash") != result.result_hash:
            _fail("hash_mismatch", "sequence_generation_result")
        return result


def validate_generation_result(
    request: SequenceGenerationRequest,
    result: SequenceGenerationResult,
) -> SequenceGenerationResult:


    return _validate_generation_result(request, result, reparse=True)


def _validate_generation_result(
    request: SequenceGenerationRequest,
    result: SequenceGenerationResult,
    *,
    reparse: bool,
) -> SequenceGenerationResult:


    if not isinstance(request, SequenceGenerationRequest):
        _fail("request_type_invalid")
    if not isinstance(result, SequenceGenerationResult):
        _fail("result_type_invalid")
    if reparse:


        SequenceGenerationRequest.from_mapping(request.to_dict())
        SequenceGenerationResult.from_mapping(result.to_dict())
    if result.request_hash != request.request_hash:
        _fail("request_hash_mismatch")
    provenance = result.provenance
    identity = {
        "seed": request.seed,
        "step": request.step,
        "mapping_edge_id": request.mapping_edge_id,
        "action_id": request.action_id,
        "structural_node_id": request.structural_node_id,
        "operator": request.operator,
        "condition_refs_available": {
            "structure": list(request.structure_condition_refs),
            "state": list(request.state_condition_refs),
        },
    }
    for key, expected in identity.items():
        if provenance.get(key) != expected:
            _fail("provenance_mismatch", key)
    actual_changed, actual_violations = _inspect_generation_constraints_with_delta(
        request, result.sequences
    )
    serialized_changed = {
        chain: list(positions) for chain, positions in result.changed_positions.items()
    }
    if serialized_changed != actual_changed:
        _fail("changed_positions_mismatch")
    if result.constraint_violations != actual_violations:
        _fail("violation_report_mismatch")
    if actual_violations:
        codes = ",".join(sorted({item["code"] for item in actual_violations}))
        _fail("constraint_violation", codes)
    return result


def _residue_probability(
    residues: Sequence[str], weights: Mapping[str, float], selected: str
) -> float:
    values = [float(weights.get(residue, 1.0)) for residue in residues]
    total = sum(values)
    if total <= 0.0:
        return 1.0 / len(residues)
    return float(weights.get(selected, 1.0)) / total


def _stable_tie(request: SequenceGenerationRequest, position: int, residue: str) -> str:
    return hashlib.sha256(
        f"{request.request_hash}\0{position}\0{residue}".encode("utf-8")
    ).hexdigest()


def _weighted_residue_sample(
    request: SequenceGenerationRequest,
    position: int,
    residues: Sequence[str],
    weights: Mapping[str, float],
) -> str:


    ordered = tuple(sorted(str(residue) for residue in residues))
    if not ordered:
        _fail("residue_sampling_support_empty", position)
    categorical = [float(weights.get(residue, 1.0)) for residue in ordered]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in categorical):
        _fail("residue_sampling_weight_invalid", position)
    total = math.fsum(categorical)
    if total <= 0.0:
        categorical = [1.0] * len(ordered)
        total = float(len(ordered))
    seed_material = _digest(
        "constraint_weighted_residue_sample.v1",
        {
            "request_hash": request.request_hash,
            "seed": request.seed,
            "step": request.step,
            "position": int(position),
            "support": list(ordered),
            "weights": categorical,
        },
    )
    draw = random.Random(int(seed_material, 16)).random() * total
    cumulative = 0.0
    for residue, weight in zip(ordered, categorical):
        cumulative += weight
        if draw < cumulative:
            return residue
    return ordered[-1]


def _target_change_count(request: SequenceGenerationRequest) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    parent = request.parent_sequences[request.target_chain]
    allowed = request.allowed_residues[request.target_chain]
    mandatory = tuple(
        position
        for position in request.write_positions
        if parent[position] not in allowed[position]
    )
    available = tuple(
        position
        for position in request.write_positions
        if any(residue != parent[position] for residue in allowed[position])
    )
    budget = request.mutation_budget
    count = min(budget["max"], len(available))
    count = max(count, budget["min"], len(mandatory))
    if count > len(available) or count > budget["max"]:
        _fail("mutation_budget_infeasible")
    return count, mandatory, available


class DeterministicConstraintAwareGeneratorV1:


    generator_id = CONSTRAINT_AWARE_GENERATOR_ID
    generator_version = "2"

    def generate(self, request: SequenceGenerationRequest) -> SequenceGenerationResult:
        if not isinstance(request, SequenceGenerationRequest):
            _fail("request_type_invalid")
        count, mandatory, available = _target_change_count(request)
        parent = request.parent_sequences[request.target_chain]
        allowed = request.allowed_residues[request.target_chain]
        weights = request.soft_residue_weights.get(request.target_chain, {})
        selections: Dict[int, Tuple[str, float, float, float]] = {}
        for position in available:
            alternatives = tuple(
                residue for residue in allowed[position] if residue != parent[position]
            )
            position_weights = weights.get(position, {})
            selected = _weighted_residue_sample(
                request,
                position,
                alternatives,
                position_weights,
            )
            conditional_probability = _residue_probability(
                alternatives, position_weights, selected
            )
            substitution_mass = _residue_probability(
                allowed[position], position_weights, selected
            )
            selections[position] = (
                selected,
                conditional_probability,
                float(position_weights.get(selected, 1.0)),
                substitution_mass,
            )
        mandatory_set = set(mandatory)
        optional = sorted(
            (position for position in available if position not in mandatory_set),
            key=lambda position: (
                -selections[position][3],
                _stable_tie(request, position, selections[position][0]),
                position,
            ),
        )
        chosen = list(mandatory) + optional[: max(0, count - len(mandatory))]
        generated = list(parent)
        probabilities = []
        heuristic_score = 0.0
        for position in chosen:
            residue, probability, score, _substitution_mass = selections[position]
            generated[position] = residue
            probabilities.append(max(probability, 1e-300))
            heuristic_score += score
        sequences = request.parent_sequences
        sequences[request.target_chain] = "".join(generated)
        log_likelihood = sum(math.log(value) for value in probabilities)
        confidence = (
            math.exp(log_likelihood / len(probabilities)) if probabilities else 1.0
        )
        result = SequenceGenerationResult.create(
            request=request,
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            algorithm="constraint_weighted_seeded_sampling_v2",
            deterministic=True,
            baseline=False,
            sequences=sequences,
            proposal_log_likelihood=log_likelihood,
            confidence=confidence,
            heuristic_score=heuristic_score,


            constraint_violations=(),
            consumed_structure_condition_refs=request.structure_condition_refs,
            consumed_state_condition_refs=request.state_condition_refs,
        )


        return result


class UniformRandomBaselineGenerator:


    generator_id = UNIFORM_RANDOM_BASELINE_ID
    generator_version = "1"

    def generate(self, request: SequenceGenerationRequest) -> SequenceGenerationResult:
        if not isinstance(request, SequenceGenerationRequest):
            _fail("request_type_invalid")
        count, mandatory, available = _target_change_count(request)
        seed_material = _digest(
            "uniform_random_baseline.seed.v1",
            {
                "request_hash": request.request_hash,
                "seed": request.seed,
                "step": request.step,
            },
        )
        rng = random.Random(int(seed_material, 16))
        mandatory_set = set(mandatory)
        optional = [position for position in available if position not in mandatory_set]
        rng.shuffle(optional)
        chosen = list(mandatory) + optional[: max(0, count - len(mandatory))]
        parent = request.parent_sequences[request.target_chain]
        allowed = request.allowed_residues[request.target_chain]
        generated = list(parent)
        log_likelihood = 0.0
        for position in chosen:
            alternatives = sorted(
                residue for residue in allowed[position] if residue != parent[position]
            )
            residue = alternatives[rng.randrange(len(alternatives))]
            generated[position] = residue
            log_likelihood += math.log(1.0 / len(alternatives))
        sequences = request.parent_sequences
        sequences[request.target_chain] = "".join(generated)
        confidence = math.exp(log_likelihood / len(chosen)) if chosen else 1.0
        result = SequenceGenerationResult.create(
            request=request,
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            algorithm="uniform_random_substitution_v1",
            deterministic=True,
            baseline=True,
            sequences=sequences,
            proposal_log_likelihood=log_likelihood,
            confidence=confidence,
            heuristic_score=0.0,
            constraint_violations=(),
            consumed_structure_condition_refs=(),
            consumed_state_condition_refs=(),
        )
        return result


@runtime_checkable
class SequenceGenerator(Protocol):


    generator_id: str
    generator_version: str

    def generate(self, request: SequenceGenerationRequest) -> SequenceGenerationResult:
        pass


class SequenceGeneratorRegistry:


    def __init__(self) -> None:
        self._generators: Dict[str, SequenceGenerator] = {}
        self._frozen = False

    def register(self, generator: SequenceGenerator) -> None:
        if self._frozen:
            _fail("registry_frozen")
        generator_id = _text(
            getattr(generator, "generator_id", None), "generator_id_required"
        )
        _text(getattr(generator, "generator_version", None), "generator_version_required")
        if not callable(getattr(generator, "generate", None)):
            _fail("generator_protocol_invalid", generator_id)
        if generator_id in self._generators:
            _fail("generator_already_registered", generator_id)
        self._generators[generator_id] = generator

    def freeze(self) -> "SequenceGeneratorRegistry":
        self._frozen = True
        return self

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._generators))

    def resolve(self, generator_id: str) -> SequenceGenerator:
        name = _text(generator_id, "generator_id_required")
        try:
            return self._generators[name]
        except KeyError:
            _fail("generator_not_registered", name)

    def generate(
        self,
        generator_id: str,
        request: SequenceGenerationRequest,
    ) -> SequenceGenerationResult:
        generator = self.resolve(generator_id)
        result = generator.generate(request)
        if not isinstance(result, SequenceGenerationResult):
            _fail("generator_result_type_invalid", generator_id)
        if result.generator_id != generator_id:
            _fail(
                "generator_identity_mismatch",
                f"requested={generator_id},reported={result.generator_id}",
            )
        if result.generator_version != str(generator.generator_version):
            _fail("generator_version_mismatch", generator_id)
        reparse = not (
            _is_untampered_factory_contract(request, "request_hash")
            and _is_untampered_factory_contract(result, "result_hash")
        )
        return _validate_generation_result(request, result, reparse=reparse)


def build_default_sequence_generator_registry() -> SequenceGeneratorRegistry:


    registry = SequenceGeneratorRegistry()
    registry.register(DeterministicConstraintAwareGeneratorV1())
    registry.register(UniformRandomBaselineGenerator())
    return registry


@lru_cache(maxsize=1)
def default_sequence_generator_registry() -> SequenceGeneratorRegistry:


    return build_default_sequence_generator_registry().freeze()


__all__ = [
    "CANONICAL_AMINO_ACIDS",
    "CONSTRAINT_AWARE_GENERATOR_ID",
    "SEQUENCE_GENERATION_REQUEST_VERSION",
    "SEQUENCE_GENERATION_RESULT_VERSION",
    "UNIFORM_RANDOM_BASELINE_ID",
    "DeterministicConstraintAwareGeneratorV1",
    "SequenceGenerationError",
    "SequenceGenerationRequest",
    "SequenceGenerationResult",
    "SequenceGenerator",
    "SequenceGeneratorRegistry",
    "UniformRandomBaselineGenerator",
    "build_default_sequence_generator_registry",
    "default_sequence_generator_registry",
    "inspect_generation_constraints",
    "validate_generation_result",
]

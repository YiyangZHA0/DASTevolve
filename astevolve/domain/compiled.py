

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from ._mapping import copied_mapping, copied_sequence_mapping


COMPILED_RESIDUE_CHANGE_VERSION = "astevolve.compiled_residue_change.v1"
MUTATION_OWNERSHIP_ENTRY_VERSION = "astevolve.mutation_ownership_entry.v1"
MUTATION_OWNERSHIP_LEDGER_VERSION = "astevolve.mutation_ownership_ledger.v1"
COMPILED_MUTATION_SET_VERSION = "astevolve.compiled_mutation_set.v1"
COMPILED_POSITION_DISTRIBUTION_VERSION = (
    "astevolve.compiled_position_distribution.v1"
)
LEGACY_COMPILED_DESIGN_ACTION_VERSION = "astevolve.compiled_design_action.v1"
COMPILED_DESIGN_ACTION_VERSION = "astevolve.compiled_design_action.v2"
COMPILED_MUTATION_MODULE_VERSION = (
    "astevolve.compiled_mutation_module.v1"
)
COMPILED_SEQUENCE_SEED_VERSION = "astevolve.compiled_sequence_seed.v1"
COMPILED_CANDIDATE_SLOT_VERSION = "astevolve.compiled_candidate_slot.v1"
COMPILED_PORTFOLIO_OPTIMIZATION_REQUEST_VERSION = (
    "astevolve.compiled_portfolio_optimization_request.v1"
)


class CompiledDesignActionSchemaError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _compiled_fail(code: str, detail: str = "") -> None:
    raise CompiledDesignActionSchemaError(code, detail)


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
        _compiled_fail("not_canonical_json", str(error))


def _compiled_digest(domain: str, value: Any) -> str:
    payload = domain.encode("utf-8") + b"\0" + _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _closed_mapping(
    value: Any,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", label)
    observed = {str(key) for key in value}
    unknown = sorted(observed - fields)
    missing = sorted(fields - observed)
    if unknown:
        _compiled_fail("unknown_fields", f"{label}:{','.join(unknown)}")
    if missing:
        _compiled_fail("fields_missing", f"{label}:{','.join(missing)}")
    return value


def _text(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _compiled_fail(code)
    return value.strip()


def _position(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _compiled_fail("position_invalid", repr(value))
    return value


def _residue(value: Any, *, code: str = "residue_invalid") -> str:
    residue = _text(value, code=code).upper()
    if len(residue) != 1 or residue not in "ACDEFGHIKLMNPQRSTVWY":
        _compiled_fail(code, residue)
    return residue


def _sequence_bundle_hash(sequences: Mapping[str, str]) -> str:
    payload = [
        {"chain_id": chain, "sequence": sequence}
        for chain, sequence in sorted(sequences.items())
    ]
    return f"sequence_sha256:{_compiled_digest('sequence_record.v1', payload)}"


@dataclass(frozen=True)
class CompiledSegment:


    chain_id: str
    name: str
    kind: str
    spans: Tuple[Tuple[int, int], ...]
    properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def props(self) -> Mapping[str, Any]:


        return self.properties

    @property
    def start(self) -> int:


        return self.spans[0][0] if self.spans else 0

    @property
    def end(self) -> int:


        return self.spans[-1][1] if self.spans else 0

    @property
    def total_length(self) -> int:
        return sum(end - start for start, end in self.spans)

    @property
    def is_contiguous(self) -> bool:
        return all(
            self.spans[index][1] == self.spans[index + 1][0]
            for index in range(len(self.spans) - 1)
        )

    def indices(self) -> list[int]:


        return [
            position
            for start, end in self.spans
            for position in range(start, end)
        ]

    def extract(self, sequence: str) -> str:


        return "".join(sequence[position] for position in self.indices())

    @classmethod
    def from_legacy(cls, value: Any) -> "CompiledSegment":
        spans = getattr(value, "spans", None)
        if spans is None and isinstance(value, Mapping):
            spans = value.get("spans", ())
        get = value.get if isinstance(value, Mapping) else lambda key, default=None: getattr(value, key, default)
        return cls(
            chain_id=str(get("chain_id", "")),
            name=str(get("name", "")),
            kind=str(get("kind", "")),
            spans=tuple((int(start), int(end)) for start, end in (spans or ())),
            properties=copied_mapping(get("props", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "kind": self.kind,
            "spans": [list(span) for span in self.spans],
            "props": copied_mapping(self.properties),
        }


@dataclass(frozen=True)
class CompiledDesign:


    chain_order: Tuple[str, ...]
    chain_lengths: Mapping[str, int]
    segments: Tuple[CompiledSegment, ...]
    template_sequences: Mapping[str, str] = field(default_factory=dict)
    masks: Mapping[str, Sequence[bool]] = field(default_factory=dict)
    fixed_residues: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "astevolve.compiled_design.v1"

    @classmethod
    def from_legacy(
        cls,
        compiled: Mapping[str, Any],
        *,
        template_sequences: Mapping[str, str] | None = None,
        masks: Mapping[str, Sequence[bool]] | None = None,
        fixed_residues: Mapping[str, Mapping[int, str]] | None = None,
    ) -> "CompiledDesign":
        return cls(
            chain_order=tuple(str(item) for item in compiled.get("chain_order", ()) or ()),
            chain_lengths={str(key): int(value) for key, value in (compiled.get("chain_lengths", {}) or {}).items()},
            segments=tuple(CompiledSegment.from_legacy(item) for item in compiled.get("segments", ()) or ()),
            template_sequences=copied_sequence_mapping(template_sequences),
            masks={str(key): tuple(bool(item) for item in value) for key, value in (masks or {}).items()},
            fixed_residues={
                str(chain): {int(position): str(aa) for position, aa in residues.items()}
                for chain, residues in (fixed_residues or {}).items()
            },
            metadata={
                key: item
                for key, item in copied_mapping(compiled).items()
                if key not in {"chain_order", "chain_lengths", "segments"}
            },
        )

    def to_legacy_dict(self) -> Dict[str, Any]:


        data = copied_mapping(self.metadata)
        data.update(
            {
                "chain_order": list(self.chain_order),
                "chain_lengths": dict(self.chain_lengths),

                "segments": list(self.segments),
            }
        )
        return data


@dataclass(frozen=True)
class CompiledResidueChange:


    chain_id: str
    position: int
    from_residue: str
    to_residue: str
    owner_node_id: str | None
    schema_version: str = COMPILED_RESIDUE_CHANGE_VERSION

    @classmethod
    def create(
        cls,
        *,
        chain_id: str,
        position: int,
        from_residue: str,
        to_residue: str,
        owner_node_id: str | None = None,
    ) -> "CompiledResidueChange":
        source = _residue(from_residue, code="from_residue_invalid")
        target = _residue(to_residue, code="to_residue_invalid")
        if source == target:
            _compiled_fail("residue_change_no_op", f"{chain_id}:{position}")
        owner = None
        if owner_node_id not in (None, ""):
            owner = _text(owner_node_id, code="owner_node_id_invalid")
        return cls(
            chain_id=_text(chain_id, code="chain_id_required"),
            position=_position(position),
            from_residue=source,
            to_residue=target,
            owner_node_id=owner,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "position": self.position,
            "from_residue": self.from_residue,
            "to_residue": self.to_residue,
            "owner_node_id": self.owner_node_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledResidueChange":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version",
                "chain_id",
                "position",
                "from_residue",
                "to_residue",
                "owner_node_id",
            },
            label="compiled_residue_change",
        )
        if raw["schema_version"] != COMPILED_RESIDUE_CHANGE_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_residue_change")
        return cls.create(
            chain_id=raw["chain_id"],
            position=raw["position"],
            from_residue=raw["from_residue"],
            to_residue=raw["to_residue"],
            owner_node_id=raw["owner_node_id"],
        )


@dataclass(frozen=True)
class MutationOwnershipEntry:


    lineage_action_id: str
    chain_id: str
    position: int
    immutable_residue: str
    parent_residue: str
    reconciled_residue: str
    action: str
    source_node_id: str
    requested_node_id: str | None
    effective_owner_node_id: str | None
    schema_version: str = MUTATION_OWNERSHIP_ENTRY_VERSION

    @classmethod
    def create(
        cls,
        *,
        lineage_action_id: str,
        chain_id: str,
        position: int,
        immutable_residue: str,
        parent_residue: str,
        reconciled_residue: str,
        action: str,
        source_node_id: str,
        requested_node_id: str | None,
        effective_owner_node_id: str | None,
    ) -> "MutationOwnershipEntry":
        normalized_action = str(action or "").strip()
        if normalized_action not in {
            "retain",
            "reassign_node",
            "replace",
            "revert_to_wt",
        }:
            _compiled_fail("lineage_action_invalid", normalized_action)
        source = _text(source_node_id, code="source_node_id_invalid")
        immutable = _residue(immutable_residue, code="immutable_residue_invalid")
        parent = _residue(parent_residue, code="parent_residue_invalid")
        reconciled = _residue(
            reconciled_residue, code="reconciled_residue_invalid"
        )
        if parent == immutable:
            _compiled_fail("ownership_entry_not_parent_mutation")
        requested = None
        if requested_node_id not in (None, ""):
            requested = _text(
                requested_node_id, code="requested_node_id_invalid"
            )
        effective = None
        if effective_owner_node_id not in (None, ""):
            effective = _text(
                effective_owner_node_id, code="effective_owner_node_id_invalid"
            )
        if normalized_action == "revert_to_wt":
            if reconciled != immutable or effective is not None:
                _compiled_fail("revert_ownership_inconsistent")
        else:
            if effective is None:
                _compiled_fail("effective_owner_required", normalized_action)
            if reconciled == immutable:
                _compiled_fail("active_ownership_reverted", normalized_action)
        if (
            normalized_action in {"retain", "reassign_node", "replace"}
            and requested is None
        ):
            _compiled_fail("requested_node_required", normalized_action)
        if requested is not None and effective != requested:
            _compiled_fail("requested_owner_mismatch", requested)
        if normalized_action == "retain" and source != effective:
            _compiled_fail("retain_source_owner_mismatch", source)
        if normalized_action == "reassign_node" and source == effective:
            _compiled_fail("reassign_source_owner_unchanged", source)
        return cls(
            lineage_action_id=_text(
                lineage_action_id, code="lineage_action_id_required"
            ),
            chain_id=_text(chain_id, code="chain_id_required"),
            position=_position(position),
            immutable_residue=immutable,
            parent_residue=parent,
            reconciled_residue=reconciled,
            action=normalized_action,
            source_node_id=source,
            requested_node_id=requested,
            effective_owner_node_id=effective,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lineage_action_id": self.lineage_action_id,
            "chain_id": self.chain_id,
            "position": self.position,
            "immutable_residue": self.immutable_residue,
            "parent_residue": self.parent_residue,
            "reconciled_residue": self.reconciled_residue,
            "action": self.action,
            "source_node_id": self.source_node_id,
            "requested_node_id": self.requested_node_id,
            "effective_owner_node_id": self.effective_owner_node_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MutationOwnershipEntry":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version",
                "lineage_action_id",
                "chain_id",
                "position",
                "immutable_residue",
                "parent_residue",
                "reconciled_residue",
                "action",
                "source_node_id",
                "requested_node_id",
                "effective_owner_node_id",
            },
            label="mutation_ownership_entry",
        )
        if raw["schema_version"] != MUTATION_OWNERSHIP_ENTRY_VERSION:
            _compiled_fail("schema_version_invalid", "mutation_ownership_entry")
        return cls.create(
            lineage_action_id=raw["lineage_action_id"],
            chain_id=raw["chain_id"],
            position=raw["position"],
            immutable_residue=raw["immutable_residue"],
            parent_residue=raw["parent_residue"],
            reconciled_residue=raw["reconciled_residue"],
            action=raw["action"],
            source_node_id=raw["source_node_id"],
            requested_node_id=raw["requested_node_id"],
            effective_owner_node_id=raw["effective_owner_node_id"],
        )


@dataclass(frozen=True)
class MutationOwnershipLedger:


    entries: Tuple[MutationOwnershipEntry, ...]
    parent_mutation_count: int
    covered_mutation_count: int
    ledger_hash: str
    schema_version: str = MUTATION_OWNERSHIP_LEDGER_VERSION

    @classmethod
    def create(
        cls,
        entries: Sequence[MutationOwnershipEntry],
        *,
        parent_mutation_count: int,
    ) -> "MutationOwnershipLedger":
        normalized = tuple(
            sorted(entries, key=lambda item: (item.chain_id, item.position))
        )
        positions = [(item.chain_id, item.position) for item in normalized]
        action_ids = [item.lineage_action_id for item in normalized]
        if len(positions) != len(set(positions)):
            _compiled_fail("ownership_position_duplicate")
        if len(action_ids) != len(set(action_ids)):
            _compiled_fail("lineage_action_id_duplicate")
        if (
            isinstance(parent_mutation_count, bool)
            or not isinstance(parent_mutation_count, int)
            or parent_mutation_count < 0
        ):
            _compiled_fail("parent_mutation_count_invalid")
        if len(normalized) != parent_mutation_count:
            _compiled_fail(
                "ownership_accounting_incomplete",
                f"{len(normalized)}/{parent_mutation_count}",
            )
        semantic = {
            "entries": [item.to_dict() for item in normalized],
            "parent_mutation_count": parent_mutation_count,
            "covered_mutation_count": len(normalized),
        }
        return cls(
            normalized,
            parent_mutation_count,
            len(normalized),
            "ownership_ledger_sha256:"
            + _compiled_digest("astevolve.mutation_ownership_ledger.v1", semantic),
        )

    @property
    def coverage_fraction(self) -> float:
        if self.parent_mutation_count == 0:
            return 1.0
        return self.covered_mutation_count / self.parent_mutation_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [item.to_dict() for item in self.entries],
            "parent_mutation_count": self.parent_mutation_count,
            "covered_mutation_count": self.covered_mutation_count,
            "ledger_hash": self.ledger_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MutationOwnershipLedger":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version",
                "entries",
                "parent_mutation_count",
                "covered_mutation_count",
                "ledger_hash",
            },
            label="mutation_ownership_ledger",
        )
        if raw["schema_version"] != MUTATION_OWNERSHIP_LEDGER_VERSION:
            _compiled_fail("schema_version_invalid", "mutation_ownership_ledger")
        if isinstance(raw["entries"], (str, bytes)) or not isinstance(
            raw["entries"], Sequence
        ):
            _compiled_fail("entries_invalid", "mutation_ownership_ledger")
        ledger = cls.create(
            tuple(MutationOwnershipEntry.from_mapping(item) for item in raw["entries"]),
            parent_mutation_count=raw["parent_mutation_count"],
        )
        if raw["covered_mutation_count"] != ledger.covered_mutation_count:
            _compiled_fail("covered_mutation_count_mismatch")
        if raw["ledger_hash"] != ledger.ledger_hash:
            _compiled_fail("hash_mismatch", "mutation_ownership_ledger")
        return ledger


@dataclass(frozen=True)
class CompiledMutationSet:


    comparison_base: str
    changes: Tuple[CompiledResidueChange, ...]
    mutation_count: int
    mutation_set_hash: str
    schema_version: str = COMPILED_MUTATION_SET_VERSION

    @classmethod
    def create(
        cls,
        *,
        comparison_base: str,
        changes: Sequence[CompiledResidueChange],
    ) -> "CompiledMutationSet":
        base = str(comparison_base or "").strip()
        if base not in {
            "immutable_reference",
            "selected_parent",
            "reconciled_root",
        }:
            _compiled_fail("comparison_base_invalid", base)
        normalized = tuple(
            sorted(changes, key=lambda item: (item.chain_id, item.position))
        )
        positions = [(item.chain_id, item.position) for item in normalized]
        if len(positions) != len(set(positions)):
            _compiled_fail("mutation_position_duplicate", base)
        semantic = {
            "comparison_base": base,
            "changes": [item.to_dict() for item in normalized],
            "mutation_count": len(normalized),
        }
        return cls(
            base,
            normalized,
            len(normalized),
            "mutation_set_sha256:"
            + _compiled_digest("astevolve.compiled_mutation_set.v1", semantic),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_base": self.comparison_base,
            "changes": [item.to_dict() for item in self.changes],
            "mutation_count": self.mutation_count,
            "mutation_set_hash": self.mutation_set_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledMutationSet":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version",
                "comparison_base",
                "changes",
                "mutation_count",
                "mutation_set_hash",
            },
            label="compiled_mutation_set",
        )
        if raw["schema_version"] != COMPILED_MUTATION_SET_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_mutation_set")
        if isinstance(raw["changes"], (str, bytes)) or not isinstance(
            raw["changes"], Sequence
        ):
            _compiled_fail("changes_invalid", "compiled_mutation_set")
        mutation_set = cls.create(
            comparison_base=raw["comparison_base"],
            changes=tuple(
                CompiledResidueChange.from_mapping(item) for item in raw["changes"]
            ),
        )
        if raw["mutation_count"] != mutation_set.mutation_count:
            _compiled_fail("mutation_count_mismatch", mutation_set.comparison_base)
        if raw["mutation_set_hash"] != mutation_set.mutation_set_hash:
            _compiled_fail("hash_mismatch", "compiled_mutation_set")
        return mutation_set


def _normalized_position_mapping(
    value: Mapping[str, Sequence[int]], *, label: str
) -> Dict[str, Tuple[int, ...]]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", label)
    normalized: Dict[str, Tuple[int, ...]] = {}
    for chain, raw_positions in value.items():
        if isinstance(raw_positions, (str, bytes)) or not isinstance(
            raw_positions, Sequence
        ):
            _compiled_fail("positions_invalid", f"{label}.{chain}")
        positions = tuple(_position(item) for item in raw_positions)
        if positions != tuple(sorted(set(positions))):
            _compiled_fail("positions_not_sorted_unique", f"{label}.{chain}")
        normalized[_text(str(chain), code="chain_id_required")] = positions
    return dict(sorted(normalized.items()))


def _normalized_fixed_mapping(
    value: Mapping[str, Mapping[int | str, str]], *, label: str
) -> Dict[str, Dict[int, str]]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", label)
    normalized: Dict[str, Dict[int, str]] = {}
    for chain, raw_values in value.items():
        if not isinstance(raw_values, Mapping):
            _compiled_fail("mapping_required", f"{label}.{chain}")
        chain_values: Dict[int, str] = {}
        for raw_position, raw_residue in raw_values.items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                _compiled_fail("position_invalid", repr(raw_position))
            position = _position(position)
            if position in chain_values:
                _compiled_fail("position_duplicate", f"{label}.{chain}:{position}")
            chain_values[position] = _residue(raw_residue)
        normalized[_text(str(chain), code="chain_id_required")] = dict(
            sorted(chain_values.items())
        )
    return dict(sorted(normalized.items()))


def _normalized_alphabet_mapping(
    value: Mapping[str, Mapping[int | str, Sequence[str]]],
) -> Dict[str, Dict[int, Tuple[str, ...]]]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", "hard_allowed_residues")
    normalized: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    for chain, raw_values in value.items():
        if not isinstance(raw_values, Mapping):
            _compiled_fail("mapping_required", f"hard_allowed_residues.{chain}")
        rules: Dict[int, Tuple[str, ...]] = {}
        for raw_position, raw_alphabet in raw_values.items():
            try:
                position = _position(int(raw_position))
            except (TypeError, ValueError):
                _compiled_fail("position_invalid", repr(raw_position))
            if isinstance(raw_alphabet, (str, bytes)) or not isinstance(
                raw_alphabet, Sequence
            ):
                _compiled_fail("alphabet_invalid", f"{chain}:{position}")
            alphabet = tuple(sorted({_residue(item) for item in raw_alphabet}))
            if not alphabet:
                _compiled_fail("alphabet_empty", f"{chain}:{position}")
            rules[position] = alphabet
        normalized[_text(str(chain), code="chain_id_required")] = dict(
            sorted(rules.items())
        )
    return dict(sorted(normalized.items()))


def _normalized_owner_mapping(
    value: Mapping[str, Mapping[int | str, str]], *, label: str
) -> Dict[str, Dict[int, str]]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", label)
    normalized: Dict[str, Dict[int, str]] = {}
    for raw_chain, raw_owners in value.items():
        chain = _text(str(raw_chain), code="chain_id_required")
        if not isinstance(raw_owners, Mapping):
            _compiled_fail("mapping_required", f"{label}.{chain}")
        owners: Dict[int, str] = {}
        for raw_position, raw_node_id in raw_owners.items():
            try:
                position = _position(int(raw_position))
            except (TypeError, ValueError):
                _compiled_fail("position_invalid", repr(raw_position))
            if position in owners:
                _compiled_fail("position_duplicate", f"{label}.{chain}:{position}")
            owners[position] = _text(
                raw_node_id, code="owner_node_id_invalid"
            )
        normalized[chain] = dict(sorted(owners.items()))
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class CompiledPositionDistribution:


    chain_id: str
    position: int
    owner_node_id: str
    immutable_residue: str
    reconciled_root_residue: str
    hard_allowed_residues: Tuple[str, ...]
    enabled_residues: Tuple[str, ...]
    forbidden_residues: Tuple[str, ...]
    requested_probabilities: Mapping[str, float]
    effective_probabilities: Mapping[str, float]
    required_mutation: str | None
    wt_revert_weight: float
    temperature: float
    distribution_hash: str
    schema_version: str = COMPILED_POSITION_DISTRIBUTION_VERSION

    @classmethod
    def create(
        cls,
        *,
        chain_id: str,
        position: int,
        owner_node_id: str,
        immutable_residue: str,
        reconciled_root_residue: str,
        hard_allowed_residues: Sequence[str],
        enabled_residues: Sequence[str],
        forbidden_residues: Sequence[str],
        requested_probabilities: Mapping[str, float],
        required_mutation: str | None,
        wt_revert_weight: float,
        temperature: float,
    ) -> "CompiledPositionDistribution":
        chain = _text(chain_id, code="chain_id_required")
        resolved_position = _position(position)
        owner = _text(owner_node_id, code="owner_node_id_invalid")
        immutable = _residue(
            immutable_residue, code="immutable_residue_invalid"
        )
        root = _residue(
            reconciled_root_residue, code="reconciled_root_residue_invalid"
        )

        def normalize_alphabet(value: Sequence[str], *, label: str) -> Tuple[str, ...]:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                _compiled_fail("alphabet_invalid", label)
            parsed = tuple(_residue(item) for item in value)
            if len(parsed) != len(set(parsed)):
                _compiled_fail("alphabet_duplicate", label)
            return tuple(sorted(parsed))

        hard = normalize_alphabet(
            hard_allowed_residues, label=f"hard_allowed_residues.{chain}:{resolved_position}"
        )
        enabled = normalize_alphabet(
            enabled_residues, label=f"enabled_residues.{chain}:{resolved_position}"
        )
        forbidden = normalize_alphabet(
            forbidden_residues, label=f"forbidden_residues.{chain}:{resolved_position}"
        )
        if not hard:
            _compiled_fail("hard_alphabet_empty", f"{chain}:{resolved_position}")
        if not enabled:
            _compiled_fail("position_distribution_enabled_empty", f"{chain}:{resolved_position}")
        if not set(enabled).issubset(hard):
            extra = sorted(set(enabled) - set(hard))
            _compiled_fail(
                "position_distribution_enabled_outside_hard",
                f"{chain}:{resolved_position}:{','.join(extra)}",
            )
        if not set(forbidden).issubset(hard):
            extra = sorted(set(forbidden) - set(hard))
            _compiled_fail(
                "position_distribution_forbidden_outside_hard",
                f"{chain}:{resolved_position}:{','.join(extra)}",
            )
        overlap = sorted(set(enabled) & set(forbidden))
        if overlap:
            _compiled_fail(
                "position_distribution_enabled_forbidden_overlap",
                f"{chain}:{resolved_position}:{','.join(overlap)}",
            )
        if not isinstance(requested_probabilities, Mapping):
            _compiled_fail(
                "position_distribution_probabilities_invalid",
                f"{chain}:{resolved_position}",
            )
        requested: Dict[str, float] = {}
        for raw_residue, raw_weight in requested_probabilities.items():
            aa = _residue(raw_residue)
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or float(raw_weight) < 0.0
            ):
                _compiled_fail(
                    "position_distribution_probability_invalid",
                    f"{chain}:{resolved_position}:{aa}:{raw_weight!r}",
                )
            requested[aa] = float(raw_weight)
        if set(requested) != set(enabled):
            _compiled_fail(
                "position_distribution_probability_alphabet_mismatch",
                f"{chain}:{resolved_position}",
            )
        requested_total = math.fsum(requested.values())
        if abs(requested_total - 1.0) > 1e-8:
            _compiled_fail(
                "position_distribution_probabilities_not_normalized",
                f"{chain}:{resolved_position}:{requested_total}",
            )
        if (
            isinstance(wt_revert_weight, bool)
            or not isinstance(wt_revert_weight, (int, float))
            or not math.isfinite(float(wt_revert_weight))
            or float(wt_revert_weight) < 0.0
        ):
            _compiled_fail(
                "position_distribution_wt_revert_weight_invalid",
                f"{chain}:{resolved_position}",
            )
        revert_weight = float(wt_revert_weight)
        if immutable not in enabled and abs(revert_weight - 1.0) > 1e-12:
            _compiled_fail(
                "position_distribution_wt_weight_without_wt",
                f"{chain}:{resolved_position}",
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            _compiled_fail(
                "position_distribution_temperature_invalid",
                f"{chain}:{resolved_position}",
            )
        resolved_temperature = float(temperature)

        adjusted: Dict[str, float] = {}
        for aa in enabled:
            weight = requested[aa]
            if aa == immutable:
                weight *= revert_weight
            adjusted[aa] = weight
        positive_logs = {
            aa: math.log(weight) / resolved_temperature
            for aa, weight in adjusted.items()
            if weight > 0.0
        }
        if not positive_logs:
            _compiled_fail(
                "position_distribution_all_zero", f"{chain}:{resolved_position}"
            )
        maximum_log = max(positive_logs.values())
        scaled = {
            aa: math.exp(log_weight - maximum_log)
            for aa, log_weight in positive_logs.items()
        }
        denominator = math.fsum(scaled.values())
        effective = {
            aa: (scaled[aa] / denominator if aa in scaled else 0.0)
            for aa in enabled
        }
        if abs(math.fsum(effective.values()) - 1.0) > 1e-12:
            _compiled_fail(
                "position_distribution_effective_not_normalized",
                f"{chain}:{resolved_position}",
            )

        required = None
        if required_mutation not in (None, ""):
            required = _residue(
                required_mutation, code="required_mutation_invalid"
            )
            if required == immutable:
                _compiled_fail(
                    "position_distribution_required_noop",
                    f"{chain}:{resolved_position}:{required}",
                )
            if required not in effective or effective[required] <= 0.0:
                _compiled_fail(
                    "position_distribution_required_not_enabled",
                    f"{chain}:{resolved_position}:{required}",
                )
            if root != required:
                _compiled_fail(
                    "position_distribution_required_root_mismatch",
                    f"{chain}:{resolved_position}:{root}/{required}",
                )
        semantic = {
            "chain_id": chain,
            "position": resolved_position,
            "owner_node_id": owner,
            "immutable_residue": immutable,
            "reconciled_root_residue": root,
            "hard_allowed_residues": list(hard),
            "enabled_residues": list(enabled),
            "forbidden_residues": list(forbidden),
            "requested_probabilities": dict(sorted(requested.items())),
            "effective_probabilities": dict(sorted(effective.items())),
            "required_mutation": required,
            "wt_revert_weight": revert_weight,
            "temperature": resolved_temperature,
        }
        distribution_hash = "position_distribution_sha256:" + _compiled_digest(
            COMPILED_POSITION_DISTRIBUTION_VERSION, semantic
        )
        return cls(
            chain_id=chain,
            position=resolved_position,
            owner_node_id=owner,
            immutable_residue=immutable,
            reconciled_root_residue=root,
            hard_allowed_residues=hard,
            enabled_residues=enabled,
            forbidden_residues=forbidden,
            requested_probabilities=dict(sorted(requested.items())),
            effective_probabilities=dict(sorted(effective.items())),
            required_mutation=required,
            wt_revert_weight=revert_weight,
            temperature=resolved_temperature,
            distribution_hash=distribution_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "position": self.position,
            "owner_node_id": self.owner_node_id,
            "immutable_residue": self.immutable_residue,
            "reconciled_root_residue": self.reconciled_root_residue,
            "hard_allowed_residues": list(self.hard_allowed_residues),
            "enabled_residues": list(self.enabled_residues),
            "forbidden_residues": list(self.forbidden_residues),
            "requested_probabilities": dict(self.requested_probabilities),
            "effective_probabilities": dict(self.effective_probabilities),
            "required_mutation": self.required_mutation,
            "wt_revert_weight": self.wt_revert_weight,
            "temperature": self.temperature,
            "distribution_hash": self.distribution_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "CompiledPositionDistribution":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version",
                "chain_id",
                "position",
                "owner_node_id",
                "immutable_residue",
                "reconciled_root_residue",
                "hard_allowed_residues",
                "enabled_residues",
                "forbidden_residues",
                "requested_probabilities",
                "effective_probabilities",
                "required_mutation",
                "wt_revert_weight",
                "temperature",
                "distribution_hash",
            },
            label="compiled_position_distribution",
        )
        if raw["schema_version"] != COMPILED_POSITION_DISTRIBUTION_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_position_distribution")
        compiled = cls.create(
            chain_id=raw["chain_id"],
            position=raw["position"],
            owner_node_id=raw["owner_node_id"],
            immutable_residue=raw["immutable_residue"],
            reconciled_root_residue=raw["reconciled_root_residue"],
            hard_allowed_residues=raw["hard_allowed_residues"],
            enabled_residues=raw["enabled_residues"],
            forbidden_residues=raw["forbidden_residues"],
            requested_probabilities=raw["requested_probabilities"],
            required_mutation=raw["required_mutation"],
            wt_revert_weight=raw["wt_revert_weight"],
            temperature=raw["temperature"],
        )


        for field in (
            "hard_allowed_residues",
            "enabled_residues",
            "forbidden_residues",
        ):
            supplied = raw[field]
            if (
                isinstance(supplied, (str, bytes))
                or not isinstance(supplied, Sequence)
                or list(supplied) != list(getattr(compiled, field))
            ):
                _compiled_fail("hash_mismatch", field)
        supplied_effective = raw["effective_probabilities"]
        if not isinstance(supplied_effective, Mapping) or set(supplied_effective) != set(
            compiled.effective_probabilities
        ):
            _compiled_fail("hash_mismatch", "effective_probabilities")
        for aa, expected in compiled.effective_probabilities.items():
            observed = supplied_effective[aa]
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or abs(float(observed) - expected) > 1e-15
            ):
                _compiled_fail("hash_mismatch", "effective_probabilities")
        if raw["distribution_hash"] != compiled.distribution_hash:
            _compiled_fail("hash_mismatch", "position_distribution")
        if _canonical_json(value) != _canonical_json(compiled.to_dict()):
            _compiled_fail("noncanonical_artifact", "position_distribution")
        return compiled


def compiled_position_distribution_set_hash(
    distributions: Sequence[CompiledPositionDistribution | Mapping[str, Any]],
) -> str:


    if isinstance(distributions, (str, bytes)) or not isinstance(
        distributions, Sequence
    ):
        _compiled_fail("position_distributions_invalid")
    normalized = [
        CompiledPositionDistribution.from_mapping(item)
        if isinstance(item, Mapping)
        else CompiledPositionDistribution.from_mapping(item.to_dict())
        for item in distributions
    ]
    ordered = sorted(normalized, key=lambda item: (item.chain_id, item.position))
    keys = [(item.chain_id, item.position) for item in ordered]
    if len(keys) != len(set(keys)):
        _compiled_fail("position_distribution_duplicate")
    return "position_distribution_set_sha256:" + _compiled_digest(
        "astevolve.position_distribution_set.v1",
        [item.distribution_hash for item in ordered],
    )


@dataclass(frozen=True)
class CompiledDesignAction:


    design_action_hash: str
    case_id: str
    parent_program_id: str
    parent_candidate_id: str
    parent_sequence_bundle_hash: str
    parent_effective_contract_hash: str
    parent_evolve_hash: str
    immutable_sequence_bundle_hash: str
    executable_node_plan_hash: str
    reconciled_root_sequences: Mapping[str, str]
    reconciled_root_sequence_bundle_hash: str
    active_selector_positions: Mapping[str, Tuple[int, ...]]
    active_position_owners: Mapping[str, Mapping[int, str]]
    parent_position_owners: Mapping[str, Mapping[int, str]]
    ownership_ledger: MutationOwnershipLedger
    absolute_mutations: CompiledMutationSet
    lineage_delta: CompiledMutationSet
    hard_allowed_residues: Mapping[str, Mapping[int, Tuple[str, ...]]]
    fixed_residues: Mapping[str, Mapping[int, str]]
    protected_residues: Mapping[str, Tuple[int, ...]]
    mutation_count_range: Tuple[int, int]
    minimum_hamming_distance: int
    max_total_mutations: int
    position_distributions: Tuple[CompiledPositionDistribution, ...]
    compiled_design_action_hash: str
    schema_version: str = COMPILED_DESIGN_ACTION_VERSION

    @property
    def position_distribution_set_hash(self) -> str:
        return compiled_position_distribution_set_hash(
            self.position_distributions
        )

    @classmethod
    def create(
        cls,
        *,
        design_action_hash: str,
        case_id: str,
        parent_program_id: str,
        parent_candidate_id: str,
        parent_sequence_bundle_hash: str,
        parent_effective_contract_hash: str,
        parent_evolve_hash: str,
        immutable_sequence_bundle_hash: str,
        executable_node_plan_hash: str,
        reconciled_root_sequences: Mapping[str, str],
        active_selector_positions: Mapping[str, Sequence[int]],
        active_position_owners: Mapping[str, Mapping[int | str, str]],
        parent_position_owners: Mapping[str, Mapping[int | str, str]],
        ownership_ledger: MutationOwnershipLedger,
        absolute_mutations: CompiledMutationSet,
        lineage_delta: CompiledMutationSet,
        hard_allowed_residues: Mapping[
            str, Mapping[int | str, Sequence[str]]
        ],
        fixed_residues: Mapping[str, Mapping[int | str, str]],
        protected_residues: Mapping[str, Sequence[int]],
        mutation_count_range: Sequence[int],
        minimum_hamming_distance: int,
        max_total_mutations: int,
        position_distributions: Sequence[
            CompiledPositionDistribution | Mapping[str, Any]
        ] = (),
    ) -> "CompiledDesignAction":
        sequences = {
            _text(str(chain), code="chain_id_required"): _text(
                sequence, code="sequence_required"
            )
            for chain, sequence in sorted(reconciled_root_sequences.items())
        }
        if not sequences:
            _compiled_fail("reconciled_root_sequences_required")
        for chain, sequence in sequences.items():
            invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
            if invalid:
                _compiled_fail(
                    "reconciled_root_residue_invalid",
                    f"{chain}:{','.join(invalid)}",
                )
        selector = _normalized_position_mapping(
            active_selector_positions, label="active_selector_positions"
        )
        active_owners = _normalized_owner_mapping(
            active_position_owners, label="active_position_owners"
        )
        parent_owners = _normalized_owner_mapping(
            parent_position_owners, label="parent_position_owners"
        )
        protected = _normalized_position_mapping(
            protected_residues, label="protected_residues"
        )
        fixed = _normalized_fixed_mapping(fixed_residues, label="fixed_residues")
        alphabet = _normalized_alphabet_mapping(hard_allowed_residues)
        if isinstance(position_distributions, (str, bytes)) or not isinstance(
            position_distributions, Sequence
        ):
            _compiled_fail("position_distributions_invalid")
        normalized_distributions: list[CompiledPositionDistribution] = []
        distribution_keys: set[Tuple[str, int]] = set()
        distribution_hashes: set[str] = set()
        for raw_distribution in position_distributions:
            distribution = (
                CompiledPositionDistribution.from_mapping(raw_distribution)
                if isinstance(raw_distribution, Mapping)
                else raw_distribution
            )
            if not isinstance(distribution, CompiledPositionDistribution):
                _compiled_fail("compiled_position_distribution_required")


            distribution = CompiledPositionDistribution.from_mapping(
                distribution.to_dict()
            )
            key = (distribution.chain_id, distribution.position)
            if key in distribution_keys:
                _compiled_fail(
                    "position_distribution_duplicate",
                    f"{distribution.chain_id}:{distribution.position}",
                )
            if distribution.distribution_hash in distribution_hashes:
                _compiled_fail(
                    "position_distribution_hash_duplicate",
                    distribution.distribution_hash,
                )
            distribution_keys.add(key)
            distribution_hashes.add(distribution.distribution_hash)
            normalized_distributions.append(distribution)
        distributions = tuple(
            sorted(
                normalized_distributions,
                key=lambda item: (item.chain_id, item.position),
            )
        )
        selector_keys = {
            (chain, position)
            for chain, positions in selector.items()
            for position in positions
        }
        active_owner_keys = {
            (chain, position)
            for chain, entries in active_owners.items()
            for position in entries
        }
        alphabet_keys = {
            (chain, position)
            for chain, entries in alphabet.items()
            for position in entries
        }
        if selector_keys != active_owner_keys:
            _compiled_fail("active_owner_selector_mismatch")
        if selector_keys != alphabet_keys:
            _compiled_fail("hard_alphabet_selector_mismatch")
        if not distribution_keys.issubset(selector_keys):
            escaped = sorted(distribution_keys - selector_keys)
            _compiled_fail(
                "position_distribution_not_active",
                ",".join(f"{chain}:{position}" for chain, position in escaped),
            )
        for distribution in distributions:
            chain = distribution.chain_id
            position = distribution.position
            if distribution.owner_node_id != active_owners[chain][position]:
                _compiled_fail(
                    "position_distribution_owner_mismatch",
                    f"{chain}:{position}",
                )
            if distribution.reconciled_root_residue != sequences[chain][position]:
                _compiled_fail(
                    "position_distribution_root_mismatch",
                    f"{chain}:{position}",
                )
            if tuple(distribution.hard_allowed_residues) != tuple(
                alphabet[chain][position]
            ):
                _compiled_fail(
                    "position_distribution_hard_alphabet_mismatch",
                    f"{chain}:{position}",
                )
        for chain, position in selector_keys:
            if chain not in sequences or position >= len(sequences[chain]):
                _compiled_fail(
                    "active_selector_out_of_bounds", f"{chain}:{position}"
                )
        for chain, entries in parent_owners.items():
            for position in entries:
                if chain not in sequences or position >= len(sequences[chain]):
                    _compiled_fail(
                        "parent_owner_out_of_bounds", f"{chain}:{position}"
                    )
        fixed_keys = {
            (chain, position)
            for chain, entries in fixed.items()
            for position in entries
        }
        protected_keys = {
            (chain, position)
            for chain, positions in protected.items()
            for position in positions
        }
        if selector_keys & fixed_keys:
            _compiled_fail("fixed_residue_in_active_selector")
        if selector_keys & protected_keys:
            _compiled_fail("protected_residue_in_active_selector")
        if (
            isinstance(mutation_count_range, (str, bytes))
            or not isinstance(mutation_count_range, Sequence)
            or len(mutation_count_range) != 2
        ):
            _compiled_fail("mutation_count_range_invalid")
        minimum, maximum = mutation_count_range
        for value in (minimum, maximum, minimum_hamming_distance, max_total_mutations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _compiled_fail("mutation_budget_invalid", repr(value))
        if minimum > maximum or maximum > max_total_mutations:
            _compiled_fail("mutation_budget_inconsistent")
        if minimum_hamming_distance > len(selector_keys):
            _compiled_fail("minimum_hamming_distance_unsatisfiable")
        if absolute_mutations.comparison_base != "immutable_reference":
            _compiled_fail("absolute_mutation_base_invalid")
        if lineage_delta.comparison_base != "selected_parent":
            _compiled_fail("lineage_delta_base_invalid")
        root_hash = _sequence_bundle_hash(sequences)
        implied_immutable_lists = {
            chain: list(sequence) for chain, sequence in sequences.items()
        }
        for change in absolute_mutations.changes:
            key = (change.chain_id, change.position)
            if key not in selector_keys:
                _compiled_fail(
                    "absolute_mutation_outside_active_selector",
                    f"{change.chain_id}:{change.position}",
                )
            if change.owner_node_id != active_owners[change.chain_id][change.position]:
                _compiled_fail(
                    "absolute_mutation_owner_mismatch",
                    f"{change.chain_id}:{change.position}",
                )
            if sequences[change.chain_id][change.position] != change.to_residue:
                _compiled_fail(
                    "absolute_mutation_target_mismatch",
                    f"{change.chain_id}:{change.position}",
                )
            implied_immutable_lists[change.chain_id][change.position] = (
                change.from_residue
            )
        implied_immutable = {
            chain: "".join(residues)
            for chain, residues in implied_immutable_lists.items()
        }
        expected_immutable_hash = _sequence_bundle_hash(implied_immutable)
        if expected_immutable_hash != immutable_sequence_bundle_hash:
            _compiled_fail("immutable_sequence_bundle_hash_mismatch")
        for distribution in distributions:
            if (
                distribution.immutable_residue
                != implied_immutable[distribution.chain_id][distribution.position]
            ):
                _compiled_fail(
                    "position_distribution_immutable_mismatch",
                    f"{distribution.chain_id}:{distribution.position}",
                )

        implied_parent_lists = {
            chain: list(sequence) for chain, sequence in sequences.items()
        }
        for change in lineage_delta.changes:
            if sequences[change.chain_id][change.position] != change.to_residue:
                _compiled_fail(
                    "lineage_delta_target_mismatch",
                    f"{change.chain_id}:{change.position}",
                )
            implied_parent_lists[change.chain_id][change.position] = (
                change.from_residue
            )
        implied_parent = {
            chain: "".join(residues) for chain, residues in implied_parent_lists.items()
        }
        if _sequence_bundle_hash(implied_parent) != parent_sequence_bundle_hash:
            _compiled_fail("parent_sequence_bundle_hash_mismatch")

        parent_mutation_positions = {
            (chain, position)
            for chain in implied_immutable
            for position, (wt, parent) in enumerate(
                zip(implied_immutable[chain], implied_parent[chain])
            )
            if wt != parent
        }
        ledger_positions = {
            (entry.chain_id, entry.position) for entry in ownership_ledger.entries
        }
        if ledger_positions != parent_mutation_positions:
            _compiled_fail("ownership_ledger_parent_mutation_mismatch")
        for entry in ownership_ledger.entries:
            chain, position = entry.chain_id, entry.position
            if (
                entry.immutable_residue != implied_immutable[chain][position]
                or entry.parent_residue != implied_parent[chain][position]
                or entry.reconciled_residue != sequences[chain][position]
            ):
                _compiled_fail(
                    "ownership_ledger_residue_mismatch", f"{chain}:{position}"
                )
            if parent_owners.get(chain, {}).get(position) != entry.source_node_id:
                _compiled_fail(
                    "ownership_ledger_source_owner_mismatch",
                    f"{chain}:{position}",
                )
            expected_active_owner = active_owners.get(chain, {}).get(position)
            if entry.action == "revert_to_wt":
                if entry.effective_owner_node_id is not None:
                    _compiled_fail("revert_effective_owner_forbidden")
            elif entry.effective_owner_node_id != expected_active_owner:
                _compiled_fail(
                    "ownership_ledger_effective_owner_mismatch",
                    f"{chain}:{position}",
                )
        for chain, entries in fixed.items():
            for position, residue in entries.items():
                if chain not in sequences or position >= len(sequences[chain]):
                    _compiled_fail("fixed_residue_out_of_bounds", f"{chain}:{position}")
                if sequences[chain][position] != residue:
                    _compiled_fail("fixed_residue_root_mismatch", f"{chain}:{position}")
        for chain, positions in protected.items():
            for position in positions:
                if chain not in sequences or position >= len(sequences[chain]):
                    _compiled_fail(
                        "protected_residue_out_of_bounds", f"{chain}:{position}"
                    )
                if sequences[chain][position] != implied_immutable[chain][position]:
                    _compiled_fail(
                        "protected_residue_root_mismatch", f"{chain}:{position}"
                    )
        if absolute_mutations.mutation_count > max_total_mutations:
            _compiled_fail("absolute_mutation_budget_exceeded")
        if absolute_mutations.mutation_count > maximum:
            _compiled_fail("absolute_mutation_range_exceeded")
        semantic = {
            "design_action_hash": _text(
                design_action_hash, code="design_action_hash_required"
            ),
            "case_id": _text(case_id, code="case_id_required"),
            "parent_program_id": _text(
                parent_program_id, code="parent_program_id_required"
            ),
            "parent_candidate_id": _text(
                parent_candidate_id, code="parent_candidate_id_required"
            ),
            "parent_sequence_bundle_hash": _text(
                parent_sequence_bundle_hash,
                code="parent_sequence_bundle_hash_required",
            ),
            "parent_effective_contract_hash": _text(
                parent_effective_contract_hash,
                code="parent_effective_contract_hash_required",
            ),
            "parent_evolve_hash": _text(
                parent_evolve_hash, code="parent_evolve_hash_required"
            ),
            "immutable_sequence_bundle_hash": _text(
                immutable_sequence_bundle_hash,
                code="immutable_sequence_bundle_hash_required",
            ),
            "executable_node_plan_hash": _text(
                executable_node_plan_hash,
                code="executable_node_plan_hash_required",
            ),
            "reconciled_root_sequences": sequences,
            "reconciled_root_sequence_bundle_hash": root_hash,
            "active_selector_positions": {
                chain: list(positions) for chain, positions in selector.items()
            },
            "active_position_owners": {
                chain: {str(position): node for position, node in entries.items()}
                for chain, entries in active_owners.items()
            },
            "parent_position_owners": {
                chain: {str(position): node for position, node in entries.items()}
                for chain, entries in parent_owners.items()
            },
            "ownership_ledger": ownership_ledger.to_dict(),
            "absolute_mutations": absolute_mutations.to_dict(),
            "lineage_delta": lineage_delta.to_dict(),
            "hard_allowed_residues": {
                chain: {
                    str(position): list(residues)
                    for position, residues in rules.items()
                }
                for chain, rules in alphabet.items()
            },
            "fixed_residues": {
                chain: {
                    str(position): residue for position, residue in residues.items()
                }
                for chain, residues in fixed.items()
            },
            "protected_residues": {
                chain: list(positions) for chain, positions in protected.items()
            },
            "mutation_count_range": [minimum, maximum],
            "minimum_hamming_distance": minimum_hamming_distance,
            "max_total_mutations": max_total_mutations,
        }
        if distributions:
            semantic["position_distributions"] = [
                item.to_dict() for item in distributions
            ]
        schema_version = (
            COMPILED_DESIGN_ACTION_VERSION
            if distributions
            else LEGACY_COMPILED_DESIGN_ACTION_VERSION
        )
        compiled_hash = "compiled_design_action_sha256:" + _compiled_digest(
            schema_version, semantic
        )
        return cls(
            design_action_hash=semantic["design_action_hash"],
            case_id=semantic["case_id"],
            parent_program_id=semantic["parent_program_id"],
            parent_candidate_id=semantic["parent_candidate_id"],
            parent_sequence_bundle_hash=semantic["parent_sequence_bundle_hash"],
            parent_effective_contract_hash=semantic[
                "parent_effective_contract_hash"
            ],
            parent_evolve_hash=semantic["parent_evolve_hash"],
            immutable_sequence_bundle_hash=semantic[
                "immutable_sequence_bundle_hash"
            ],
            executable_node_plan_hash=semantic["executable_node_plan_hash"],
            reconciled_root_sequences=sequences,
            reconciled_root_sequence_bundle_hash=root_hash,
            active_selector_positions=selector,
            active_position_owners=active_owners,
            parent_position_owners=parent_owners,
            ownership_ledger=ownership_ledger,
            absolute_mutations=absolute_mutations,
            lineage_delta=lineage_delta,
            hard_allowed_residues=alphabet,
            fixed_residues=fixed,
            protected_residues=protected,
            mutation_count_range=(minimum, maximum),
            minimum_hamming_distance=minimum_hamming_distance,
            max_total_mutations=max_total_mutations,
            position_distributions=distributions,
            compiled_design_action_hash=compiled_hash,
            schema_version=schema_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        artifact = {
            "schema_version": self.schema_version,
            "design_action_hash": self.design_action_hash,
            "case_id": self.case_id,
            "parent_program_id": self.parent_program_id,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_sequence_bundle_hash": self.parent_sequence_bundle_hash,
            "parent_effective_contract_hash": self.parent_effective_contract_hash,
            "parent_evolve_hash": self.parent_evolve_hash,
            "immutable_sequence_bundle_hash": self.immutable_sequence_bundle_hash,
            "executable_node_plan_hash": self.executable_node_plan_hash,
            "reconciled_root_sequences": dict(self.reconciled_root_sequences),
            "reconciled_root_sequence_bundle_hash": self.reconciled_root_sequence_bundle_hash,
            "active_selector_positions": {
                chain: list(positions)
                for chain, positions in self.active_selector_positions.items()
            },
            "active_position_owners": {
                chain: {
                    str(position): node for position, node in entries.items()
                }
                for chain, entries in self.active_position_owners.items()
            },
            "parent_position_owners": {
                chain: {
                    str(position): node for position, node in entries.items()
                }
                for chain, entries in self.parent_position_owners.items()
            },
            "ownership_ledger": self.ownership_ledger.to_dict(),
            "absolute_mutations": self.absolute_mutations.to_dict(),
            "lineage_delta": self.lineage_delta.to_dict(),
            "hard_allowed_residues": {
                chain: {
                    str(position): list(residues)
                    for position, residues in rules.items()
                }
                for chain, rules in self.hard_allowed_residues.items()
            },
            "fixed_residues": {
                chain: {
                    str(position): residue
                    for position, residue in residues.items()
                }
                for chain, residues in self.fixed_residues.items()
            },
            "protected_residues": {
                chain: list(positions)
                for chain, positions in self.protected_residues.items()
            },
            "mutation_count_range": list(self.mutation_count_range),
            "minimum_hamming_distance": self.minimum_hamming_distance,
            "max_total_mutations": self.max_total_mutations,
            "compiled_design_action_hash": self.compiled_design_action_hash,
        }
        if self.schema_version == COMPILED_DESIGN_ACTION_VERSION:
            artifact["position_distributions"] = [
                item.to_dict() for item in self.position_distributions
            ]
        return artifact

    def to_artifact(self) -> Dict[str, Any]:


        return self.to_dict()

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledDesignAction":
        if not isinstance(value, Mapping):
            _compiled_fail("mapping_required", "compiled_design_action")
        schema_version = value.get("schema_version")
        if schema_version not in {
            LEGACY_COMPILED_DESIGN_ACTION_VERSION,
            COMPILED_DESIGN_ACTION_VERSION,
        }:
            _compiled_fail("schema_version_invalid", "compiled_design_action")
        fields = {
            "schema_version",
            "design_action_hash",
            "case_id",
            "parent_program_id",
            "parent_candidate_id",
            "parent_sequence_bundle_hash",
            "parent_effective_contract_hash",
            "parent_evolve_hash",
            "immutable_sequence_bundle_hash",
            "executable_node_plan_hash",
            "reconciled_root_sequences",
            "reconciled_root_sequence_bundle_hash",
            "active_selector_positions",
            "active_position_owners",
            "parent_position_owners",
            "ownership_ledger",
            "absolute_mutations",
            "lineage_delta",
            "hard_allowed_residues",
            "fixed_residues",
            "protected_residues",
            "mutation_count_range",
            "minimum_hamming_distance",
            "max_total_mutations",
            "compiled_design_action_hash",
        }
        if schema_version == COMPILED_DESIGN_ACTION_VERSION:
            fields.add("position_distributions")
        raw = _closed_mapping(value, fields=fields, label="compiled_design_action")
        compiled = cls.create(
            design_action_hash=raw["design_action_hash"],
            case_id=raw["case_id"],
            parent_program_id=raw["parent_program_id"],
            parent_candidate_id=raw["parent_candidate_id"],
            parent_sequence_bundle_hash=raw["parent_sequence_bundle_hash"],
            parent_effective_contract_hash=raw["parent_effective_contract_hash"],
            parent_evolve_hash=raw["parent_evolve_hash"],
            immutable_sequence_bundle_hash=raw["immutable_sequence_bundle_hash"],
            executable_node_plan_hash=raw["executable_node_plan_hash"],
            reconciled_root_sequences=raw["reconciled_root_sequences"],
            active_selector_positions=raw["active_selector_positions"],
            active_position_owners=raw["active_position_owners"],
            parent_position_owners=raw["parent_position_owners"],
            ownership_ledger=MutationOwnershipLedger.from_mapping(
                raw["ownership_ledger"]
            ),
            absolute_mutations=CompiledMutationSet.from_mapping(
                raw["absolute_mutations"]
            ),
            lineage_delta=CompiledMutationSet.from_mapping(raw["lineage_delta"]),
            hard_allowed_residues=raw["hard_allowed_residues"],
            fixed_residues=raw["fixed_residues"],
            protected_residues=raw["protected_residues"],
            mutation_count_range=raw["mutation_count_range"],
            minimum_hamming_distance=raw["minimum_hamming_distance"],
            max_total_mutations=raw["max_total_mutations"],
            position_distributions=(
                [
                    CompiledPositionDistribution.from_mapping(item)
                    for item in raw["position_distributions"]
                ]
                if schema_version == COMPILED_DESIGN_ACTION_VERSION
                else ()
            ),
        )
        if compiled.schema_version != schema_version:
            _compiled_fail("schema_version_payload_mismatch", "compiled_design_action")
        if (
            raw["reconciled_root_sequence_bundle_hash"]
            != compiled.reconciled_root_sequence_bundle_hash
        ):
            _compiled_fail("hash_mismatch", "reconciled_root_sequence_bundle")
        if raw["compiled_design_action_hash"] != compiled.compiled_design_action_hash:
            _compiled_fail("hash_mismatch", "compiled_design_action")


        if (
            schema_version == COMPILED_DESIGN_ACTION_VERSION
            and _canonical_json(value) != _canonical_json(compiled.to_artifact())
        ):
            _compiled_fail("noncanonical_artifact", "compiled_design_action")
        return compiled

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> "CompiledDesignAction":
        return cls.from_mapping(value)


def _compiled_sequence_mapping(
    value: Mapping[str, str], *, allow_empty: bool = False
) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        _compiled_fail("mapping_required", "sequences")
    sequences: Dict[str, str] = {}
    for raw_chain, raw_sequence in value.items():
        chain = _text(str(raw_chain), code="chain_id_required")
        if chain != raw_chain:
            _compiled_fail("noncanonical_chain_id", repr(raw_chain))
        sequence = _text(raw_sequence, code="sequence_required")
        if sequence != sequence.upper() or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            _compiled_fail("sequence_residue_invalid", chain)
        sequences[chain] = sequence
    if not sequences and not allow_empty:
        _compiled_fail("sequences_required")
    return dict(sorted(sequences.items()))


@dataclass(frozen=True)
class CompiledMutationModule:


    module_id: str
    operation: str
    atomic: bool
    mutations: Tuple[CompiledResidueChange, ...]
    owner_node_ids: Tuple[str, ...]
    module_hash: str
    schema_version: str = COMPILED_MUTATION_MODULE_VERSION

    @classmethod
    def create(
        cls,
        *,
        module_id: str,
        operation: str,
        atomic: bool,
        mutations: Sequence[CompiledResidueChange | Mapping[str, Any]],
    ) -> "CompiledMutationModule":
        identifier = _text(module_id, code="module_id_required")
        resolved_operation = _text(operation, code="module_operation_required")
        if resolved_operation not in {
            "module_insert",
            "module_replace",
            "module_revert",
            "matched_module_ablation",
        }:
            _compiled_fail("module_operation_invalid", resolved_operation)
        if atomic is not True:
            _compiled_fail("module_not_atomic", identifier)
        if isinstance(mutations, (str, bytes)) or not isinstance(mutations, Sequence):
            _compiled_fail("module_mutations_invalid", identifier)
        parsed = tuple(
            CompiledResidueChange.from_mapping(item)
            if isinstance(item, Mapping)
            else CompiledResidueChange.from_mapping(item.to_dict())
            for item in mutations
        )
        if not parsed:
            _compiled_fail("module_mutations_empty", identifier)
        normalized = tuple(sorted(parsed, key=lambda item: (item.chain_id, item.position)))
        positions = [(item.chain_id, item.position) for item in normalized]
        if len(positions) != len(set(positions)):
            _compiled_fail("module_mutation_position_duplicate", identifier)
        owners = tuple(sorted({str(item.owner_node_id or "") for item in normalized}))
        if not owners or "" in owners:
            _compiled_fail("module_mutation_owner_missing", identifier)
        semantic = {
            "module_id": identifier,
            "operation": resolved_operation,
            "atomic": True,
            "mutations": [item.to_dict() for item in normalized],
            "owner_node_ids": list(owners),
        }
        return cls(
            module_id=identifier,
            operation=resolved_operation,
            atomic=True,
            mutations=normalized,
            owner_node_ids=owners,
            module_hash="compiled_mutation_module_sha256:"
            + _compiled_digest(COMPILED_MUTATION_MODULE_VERSION, semantic),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "operation": self.operation,
            "atomic": self.atomic,
            "mutations": [item.to_dict() for item in self.mutations],
            "owner_node_ids": list(self.owner_node_ids),
            "module_hash": self.module_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledMutationModule":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version", "module_id", "operation", "atomic",
                "mutations", "owner_node_ids", "module_hash",
            },
            label="compiled_mutation_module",
        )
        if raw["schema_version"] != COMPILED_MUTATION_MODULE_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_mutation_module")
        compiled = cls.create(
            module_id=raw["module_id"], operation=raw["operation"],
            atomic=raw["atomic"], mutations=raw["mutations"],
        )
        if list(raw["owner_node_ids"]) != list(compiled.owner_node_ids):
            _compiled_fail("owner_node_ids_mismatch", compiled.module_id)
        if raw["module_hash"] != compiled.module_hash:
            _compiled_fail("hash_mismatch", "compiled_mutation_module")
        if _canonical_json(value) != _canonical_json(compiled.to_dict()):
            _compiled_fail("noncanonical_artifact", "compiled_mutation_module")
        return compiled


@dataclass(frozen=True)
class CompiledSequenceSeed:


    seed_id: str
    sequences: Mapping[str, str]
    sequence_bundle_hash: str
    root_delta: CompiledMutationSet
    refinement_positions: Tuple[Tuple[str, int, str], ...]
    seed_hash: str
    schema_version: str = COMPILED_SEQUENCE_SEED_VERSION

    @classmethod
    def create(
        cls,
        *,
        seed_id: str,
        sequences: Mapping[str, str],
        root_delta: CompiledMutationSet | Mapping[str, Any],
        refinement_positions: Sequence[Sequence[Any] | Mapping[str, Any]],
    ) -> "CompiledSequenceSeed":
        identifier = _text(seed_id, code="seed_id_required")
        exact = _compiled_sequence_mapping(sequences)
        delta = (
            CompiledMutationSet.from_mapping(root_delta)
            if isinstance(root_delta, Mapping)
            else CompiledMutationSet.from_mapping(root_delta.to_dict())
        )
        if delta.comparison_base != "reconciled_root":
            _compiled_fail("seed_delta_base_invalid", identifier)
        if not delta.changes:
            _compiled_fail("sequence_seed_not_novel", identifier)
        if isinstance(refinement_positions, (str, bytes)) or not isinstance(
            refinement_positions, Sequence
        ):
            _compiled_fail("seed_refinement_positions_invalid", identifier)
        positions = []
        for item in refinement_positions:
            if isinstance(item, Mapping):
                raw = _closed_mapping(
                    item,
                    fields={"chain_id", "position", "owner_node_id"},
                    label="compiled_seed_refinement_position",
                )
                chain, position, owner = (
                    raw["chain_id"], raw["position"], raw["owner_node_id"]
                )
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 3:
                chain, position, owner = item
            else:
                _compiled_fail("seed_refinement_position_invalid", identifier)
            positions.append(
                (
                    _text(chain, code="chain_id_required"),
                    _position(position),
                    _text(owner, code="owner_node_id_invalid"),
                )
            )
        normalized_positions = tuple(sorted(positions))
        if len(normalized_positions) != len(set(normalized_positions)):
            _compiled_fail("seed_refinement_position_duplicate", identifier)
        semantic = {
            "seed_id": identifier,
            "sequences": exact,
            "sequence_bundle_hash": _sequence_bundle_hash(exact),
            "root_delta": delta.to_dict(),
            "refinement_positions": [
                {"chain_id": chain, "position": position, "owner_node_id": owner}
                for chain, position, owner in normalized_positions
            ],
        }
        return cls(
            seed_id=identifier,
            sequences=exact,
            sequence_bundle_hash=semantic["sequence_bundle_hash"],
            root_delta=delta,
            refinement_positions=normalized_positions,
            seed_hash="compiled_sequence_seed_sha256:"
            + _compiled_digest(COMPILED_SEQUENCE_SEED_VERSION, semantic),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed_id": self.seed_id,
            "sequences": dict(self.sequences),
            "sequence_bundle_hash": self.sequence_bundle_hash,
            "root_delta": self.root_delta.to_dict(),
            "refinement_positions": [
                {"chain_id": chain, "position": position, "owner_node_id": owner}
                for chain, position, owner in self.refinement_positions
            ],
            "seed_hash": self.seed_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledSequenceSeed":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version", "seed_id", "sequences", "sequence_bundle_hash",
                "root_delta", "refinement_positions", "seed_hash",
            },
            label="compiled_sequence_seed",
        )
        if raw["schema_version"] != COMPILED_SEQUENCE_SEED_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_sequence_seed")
        compiled = cls.create(
            seed_id=raw["seed_id"], sequences=raw["sequences"],
            root_delta=raw["root_delta"],
            refinement_positions=raw["refinement_positions"],
        )
        if raw["sequence_bundle_hash"] != compiled.sequence_bundle_hash:
            _compiled_fail("hash_mismatch", "compiled_sequence_seed_bundle")
        if raw["seed_hash"] != compiled.seed_hash:
            _compiled_fail("hash_mismatch", "compiled_sequence_seed")
        if _canonical_json(value) != _canonical_json(compiled.to_dict()):
            _compiled_fail("noncanonical_artifact", "compiled_sequence_seed")
        return compiled


@dataclass(frozen=True)
class CompiledCandidateSlot:


    slot_id: str
    portfolio_id: str
    role: str
    generation_mode: str
    realization_kind: str
    source_refs: Tuple[str, ...]
    required: bool
    exact: bool
    exact_sequences: Mapping[str, str]
    exact_sequence_bundle_hash: str | None
    root_delta: CompiledMutationSet
    slot_hash: str
    schema_version: str = COMPILED_CANDIDATE_SLOT_VERSION

    @classmethod
    def create(
        cls,
        *,
        slot_id: str,
        portfolio_id: str,
        role: str,
        generation_mode: str,
        realization_kind: str,
        source_refs: Sequence[str],
        required: bool,
        exact: bool,
        exact_sequences: Mapping[str, str],
        root_delta: CompiledMutationSet | Mapping[str, Any],
    ) -> "CompiledCandidateSlot":
        resolved_slot_id = _text(slot_id, code="slot_id_required")
        resolved_portfolio_id = _text(portfolio_id, code="portfolio_id_required")
        resolved_role = _text(role, code="portfolio_role_required")
        if resolved_role not in {
            "primary", "secondary", "repair", "novelty", "control",
            "matched_ablation",
        }:
            _compiled_fail("portfolio_role_invalid", resolved_role)
        mode = _text(generation_mode, code="generation_mode_required")
        if mode not in {"strict_sequence_seed", "module", "mcts", "matched_ablation", "control"}:
            _compiled_fail("generation_mode_invalid", mode)
        kind = _text(realization_kind, code="realization_kind_required")
        if kind not in {"module", "sequence_seed", "matched_treatment", "matched_control", "mcts", "control"}:
            _compiled_fail("realization_kind_invalid", kind)
        if required is not True and required is not False:
            _compiled_fail("portfolio_required_invalid", resolved_portfolio_id)
        if exact is not True and exact is not False:
            _compiled_fail("candidate_slot_exact_invalid", resolved_slot_id)
        if isinstance(source_refs, (str, bytes)) or not isinstance(source_refs, Sequence):
            _compiled_fail("candidate_slot_source_refs_invalid", resolved_slot_id)
        refs = tuple(sorted(_text(item, code="source_ref_invalid") for item in source_refs))
        if len(refs) != len(set(refs)):
            _compiled_fail("candidate_slot_source_ref_duplicate", resolved_slot_id)
        sequences = _compiled_sequence_mapping(exact_sequences, allow_empty=not exact)
        delta = (
            CompiledMutationSet.from_mapping(root_delta)
            if isinstance(root_delta, Mapping)
            else CompiledMutationSet.from_mapping(root_delta.to_dict())
        )
        if delta.comparison_base != "reconciled_root":
            _compiled_fail("candidate_slot_delta_base_invalid", resolved_slot_id)
        if exact:
            if not sequences:
                _compiled_fail("candidate_slot_exact_sequences_required", resolved_slot_id)
            bundle_hash: str | None = _sequence_bundle_hash(sequences)
        else:
            if sequences or delta.mutation_count:
                _compiled_fail("candidate_slot_free_has_exact_payload", resolved_slot_id)
            bundle_hash = None
        semantic = {
            "slot_id": resolved_slot_id,
            "portfolio_id": resolved_portfolio_id,
            "role": resolved_role,
            "generation_mode": mode,
            "realization_kind": kind,
            "source_refs": list(refs),
            "required": required,
            "exact": exact,
            "exact_sequences": sequences,
            "exact_sequence_bundle_hash": bundle_hash,
            "root_delta": delta.to_dict(),
        }
        return cls(
            slot_id=resolved_slot_id,
            portfolio_id=resolved_portfolio_id,
            role=resolved_role,
            generation_mode=mode,
            realization_kind=kind,
            source_refs=refs,
            required=required,
            exact=exact,
            exact_sequences=sequences,
            exact_sequence_bundle_hash=bundle_hash,
            root_delta=delta,
            slot_hash="compiled_candidate_slot_sha256:"
            + _compiled_digest(COMPILED_CANDIDATE_SLOT_VERSION, semantic),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot_id": self.slot_id,
            "portfolio_id": self.portfolio_id,
            "role": self.role,
            "generation_mode": self.generation_mode,
            "realization_kind": self.realization_kind,
            "source_refs": list(self.source_refs),
            "required": self.required,
            "exact": self.exact,
            "exact_sequences": dict(self.exact_sequences),
            "exact_sequence_bundle_hash": self.exact_sequence_bundle_hash,
            "root_delta": self.root_delta.to_dict(),
            "slot_hash": self.slot_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledCandidateSlot":
        raw = _closed_mapping(
            value,
            fields={
                "schema_version", "slot_id", "portfolio_id", "role",
                "generation_mode", "realization_kind", "source_refs", "required",
                "exact", "exact_sequences", "exact_sequence_bundle_hash",
                "root_delta", "slot_hash",
            },
            label="compiled_candidate_slot",
        )
        if raw["schema_version"] != COMPILED_CANDIDATE_SLOT_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_candidate_slot")
        compiled = cls.create(
            slot_id=raw["slot_id"], portfolio_id=raw["portfolio_id"],
            role=raw["role"], generation_mode=raw["generation_mode"],
            realization_kind=raw["realization_kind"],
            source_refs=raw["source_refs"], required=raw["required"],
            exact=raw["exact"], exact_sequences=raw["exact_sequences"],
            root_delta=raw["root_delta"],
        )
        if raw["exact_sequence_bundle_hash"] != compiled.exact_sequence_bundle_hash:
            _compiled_fail("hash_mismatch", "compiled_candidate_slot_bundle")
        if raw["slot_hash"] != compiled.slot_hash:
            _compiled_fail("hash_mismatch", "compiled_candidate_slot")
        if _canonical_json(value) != _canonical_json(compiled.to_dict()):
            _compiled_fail("noncanonical_artifact", "compiled_candidate_slot")
        return compiled


@dataclass(frozen=True)
class CompiledPortfolioOptimizationRequest:


    design_action_hash: str
    compiled_design_action_hash: str
    case_id: str
    parent_program_id: str
    parent_candidate_id: str
    immutable_sequences: Mapping[str, str]
    immutable_sequence_bundle_hash: str
    parent_sequences: Mapping[str, str]
    parent_sequence_bundle_hash: str
    reconciled_root_sequences: Mapping[str, str]
    reconciled_root_sequence_bundle_hash: str
    mutation_modules: Tuple[CompiledMutationModule, ...]
    sequence_seeds: Tuple[CompiledSequenceSeed, ...]
    candidate_slots: Tuple[CompiledCandidateSlot, ...]
    compiled_portfolio_request_hash: str
    schema_version: str = COMPILED_PORTFOLIO_OPTIMIZATION_REQUEST_VERSION

    @classmethod
    def create(
        cls,
        *,
        design_action_hash: str,
        compiled_design_action_hash: str,
        case_id: str,
        parent_program_id: str,
        parent_candidate_id: str,
        immutable_sequences: Mapping[str, str],
        immutable_sequence_bundle_hash: str,
        parent_sequences: Mapping[str, str],
        parent_sequence_bundle_hash: str,
        reconciled_root_sequences: Mapping[str, str],
        reconciled_root_sequence_bundle_hash: str,
        mutation_modules: Sequence[CompiledMutationModule | Mapping[str, Any]],
        sequence_seeds: Sequence[CompiledSequenceSeed | Mapping[str, Any]],
        candidate_slots: Sequence[CompiledCandidateSlot | Mapping[str, Any]],
    ) -> "CompiledPortfolioOptimizationRequest":
        immutable = _compiled_sequence_mapping(immutable_sequences)
        parent = _compiled_sequence_mapping(parent_sequences)
        root = _compiled_sequence_mapping(reconciled_root_sequences)
        if set(immutable) != set(parent) or set(parent) != set(root) or any(
            len(immutable[chain]) != len(parent[chain])
            or len(parent[chain]) != len(root[chain])
            for chain in immutable
        ):
            _compiled_fail("portfolio_sequence_shape_mismatch")
        computed_hashes = {
            "immutable_sequence_bundle_hash": _sequence_bundle_hash(immutable),
            "parent_sequence_bundle_hash": _sequence_bundle_hash(parent),
            "reconciled_root_sequence_bundle_hash": _sequence_bundle_hash(root),
        }
        supplied_hashes = {
            "immutable_sequence_bundle_hash": immutable_sequence_bundle_hash,
            "parent_sequence_bundle_hash": parent_sequence_bundle_hash,
            "reconciled_root_sequence_bundle_hash": reconciled_root_sequence_bundle_hash,
        }
        for field, expected in computed_hashes.items():
            if supplied_hashes[field] != expected:
                _compiled_fail("hash_mismatch", field)

        modules = tuple(sorted(
            (
                CompiledMutationModule.from_mapping(item)
                if isinstance(item, Mapping)
                else CompiledMutationModule.from_mapping(item.to_dict())
                for item in mutation_modules
            ),
            key=lambda item: item.module_id,
        ))
        seeds = tuple(sorted(
            (
                CompiledSequenceSeed.from_mapping(item)
                if isinstance(item, Mapping)
                else CompiledSequenceSeed.from_mapping(item.to_dict())
                for item in sequence_seeds
            ),
            key=lambda item: item.seed_id,
        ))
        slots = tuple(sorted(
            (
                CompiledCandidateSlot.from_mapping(item)
                if isinstance(item, Mapping)
                else CompiledCandidateSlot.from_mapping(item.to_dict())
                for item in candidate_slots
            ),
            key=lambda item: item.slot_id,
        ))
        for label, identifiers in (
            ("module_id", [item.module_id for item in modules]),
            ("seed_id", [item.seed_id for item in seeds]),
            ("slot_id", [item.slot_id for item in slots]),
        ):
            if len(identifiers) != len(set(identifiers)):
                _compiled_fail(f"{label}_duplicate")

        module_by_id = {item.module_id: item for item in modules}
        seed_by_id = {item.seed_id: item for item in seeds}

        def apply_treatment(module: CompiledMutationModule) -> Dict[str, str]:
            lists = {chain: list(sequence) for chain, sequence in root.items()}
            for change in module.mutations:
                if (
                    change.chain_id not in root
                    or change.position >= len(root[change.chain_id])
                    or change.from_residue != root[change.chain_id][change.position]
                ):
                    _compiled_fail(
                        "module_source_mismatch",
                        f"{module.module_id}:{change.chain_id}:{change.position}",
                    )
                lists[change.chain_id][change.position] = change.to_residue
            return {
                chain: "".join(residues) for chain, residues in sorted(lists.items())
            }

        for module in modules:
            if module.operation != "matched_module_ablation":
                apply_treatment(module)

        for seed in seeds:
            if set(seed.sequences) != set(root) or any(
                len(seed.sequences[chain]) != len(root[chain]) for chain in root
            ):
                _compiled_fail("sequence_seed_shape_mismatch", seed.seed_id)
            expected_seed_delta = {
                (chain, position): (before, after)
                for chain in root
                for position, (before, after) in enumerate(
                    zip(root[chain], seed.sequences[chain])
                )
                if before != after
            }
            observed_seed_delta = {
                (item.chain_id, item.position): (
                    item.from_residue,
                    item.to_residue,
                )
                for item in seed.root_delta.changes
            }
            if observed_seed_delta != expected_seed_delta:
                _compiled_fail("sequence_seed_delta_mismatch", seed.seed_id)
        for slot in slots:
            refs = set(slot.source_refs)
            if slot.generation_mode == "module":
                if len(refs) != 1 or not refs.issubset(module_by_id):
                    _compiled_fail("module_portfolio_reference_invalid", slot.portfolio_id)
                if module_by_id[next(iter(refs))].operation == "matched_module_ablation":
                    _compiled_fail("module_portfolio_reference_invalid", slot.portfolio_id)
            elif slot.generation_mode == "strict_sequence_seed":
                if len(refs) != 1 or not refs.issubset(seed_by_id):
                    _compiled_fail("sequence_seed_portfolio_reference_invalid", slot.portfolio_id)
            elif slot.generation_mode == "matched_ablation":
                referenced = [module_by_id[item] for item in refs if item in module_by_id]
                if len(refs) != 2 or len(referenced) != 2 or sum(
                    item.operation == "matched_module_ablation" for item in referenced
                ) != 1:
                    _compiled_fail("matched_ablation_reference_invalid", slot.portfolio_id)
            elif refs:
                _compiled_fail("free_portfolio_reference_invalid", slot.portfolio_id)
            if not slot.exact:
                continue
            if set(slot.exact_sequences) != set(root) or any(
                len(slot.exact_sequences[chain]) != len(root[chain]) for chain in root
            ):
                _compiled_fail("candidate_slot_sequence_shape_mismatch", slot.slot_id)
            expected = {
                (chain, position): (before, after)
                for chain in root
                for position, (before, after) in enumerate(
                    zip(root[chain], slot.exact_sequences[chain])
                )
                if before != after
            }
            observed = {
                (item.chain_id, item.position): (item.from_residue, item.to_residue)
                for item in slot.root_delta.changes
            }
            if observed != expected:
                _compiled_fail("candidate_slot_delta_mismatch", slot.slot_id)

        slots_by_portfolio: Dict[str, list[CompiledCandidateSlot]] = {}
        for slot in slots:
            slots_by_portfolio.setdefault(slot.portfolio_id, []).append(slot)
        referenced_ablation_ids: set[str] = set()
        for portfolio_id, portfolio_slots in slots_by_portfolio.items():
            modes = {item.generation_mode for item in portfolio_slots}
            if len(modes) != 1:
                _compiled_fail("portfolio_mode_mismatch", portfolio_id)
            mode = next(iter(modes))
            refs = set(portfolio_slots[0].source_refs)
            if any(set(item.source_refs) != refs for item in portfolio_slots):
                _compiled_fail("portfolio_source_refs_mismatch", portfolio_id)
            if mode == "module":
                if len(portfolio_slots) != 1:
                    _compiled_fail("portfolio_exact_count_invalid", portfolio_id)
                module = module_by_id[next(iter(refs))]
                slot = portfolio_slots[0]
                if (
                    not slot.exact
                    or slot.realization_kind != "module"
                    or dict(slot.exact_sequences) != apply_treatment(module)
                ):
                    _compiled_fail("module_slot_source_mismatch", portfolio_id)
            elif mode == "strict_sequence_seed":
                if len(portfolio_slots) != 1:
                    _compiled_fail("portfolio_exact_count_invalid", portfolio_id)
                seed = seed_by_id[next(iter(refs))]
                slot = portfolio_slots[0]
                if (
                    not slot.exact
                    or slot.realization_kind != "sequence_seed"
                    or dict(slot.exact_sequences) != dict(seed.sequences)
                ):
                    _compiled_fail("sequence_seed_slot_source_mismatch", portfolio_id)
            elif mode == "matched_ablation":
                if len(portfolio_slots) != 2:
                    _compiled_fail("matched_ablation_slot_count_invalid", portfolio_id)
                referenced = [module_by_id[item] for item in refs]
                treatments = [
                    item for item in referenced
                    if item.operation != "matched_module_ablation"
                ]
                ablations = [
                    item for item in referenced
                    if item.operation == "matched_module_ablation"
                ]
                if len(treatments) != 1 or len(ablations) != 1:
                    _compiled_fail("matched_ablation_reference_invalid", portfolio_id)
                treatment, ablation = treatments[0], ablations[0]
                referenced_ablation_ids.add(ablation.module_id)
                treatment_by_position = {
                    (item.chain_id, item.position): item
                    for item in treatment.mutations
                }
                ablation_by_position = {
                    (item.chain_id, item.position): item
                    for item in ablation.mutations
                }
                if (
                    not ablation_by_position
                    or not set(ablation_by_position) < set(treatment_by_position)
                ):
                    _compiled_fail("matched_ablation_not_proper_subset", portfolio_id)
                for key, change in ablation_by_position.items():
                    treatment_change = treatment_by_position[key]
                    if (
                        change.from_residue != treatment_change.to_residue
                        or change.to_residue != root[key[0]][key[1]]
                    ):
                        _compiled_fail("matched_ablation_delta_invalid", portfolio_id)
                expected_treatment = apply_treatment(treatment)
                control_lists = {
                    chain: list(sequence)
                    for chain, sequence in expected_treatment.items()
                }
                for chain, position in ablation_by_position:
                    control_lists[chain][position] = root[chain][position]
                expected_control = {
                    chain: "".join(residues)
                    for chain, residues in sorted(control_lists.items())
                }
                by_kind = {item.realization_kind: item for item in portfolio_slots}
                if set(by_kind) != {"matched_treatment", "matched_control"}:
                    _compiled_fail("matched_ablation_slot_kind_invalid", portfolio_id)
                if (
                    not all(item.exact for item in portfolio_slots)
                    or dict(by_kind["matched_treatment"].exact_sequences)
                    != expected_treatment
                    or dict(by_kind["matched_control"].exact_sequences)
                    != expected_control
                ):
                    _compiled_fail("matched_ablation_slot_source_mismatch", portfolio_id)
        declared_ablation_ids = {
            item.module_id
            for item in modules
            if item.operation == "matched_module_ablation"
        }
        if referenced_ablation_ids != declared_ablation_ids:
            _compiled_fail(
                "matched_ablation_module_reference_incomplete",
                ",".join(sorted(declared_ablation_ids - referenced_ablation_ids)),
            )

        semantic = {
            "design_action_hash": _text(design_action_hash, code="design_action_hash_required"),
            "compiled_design_action_hash": _text(
                compiled_design_action_hash, code="compiled_design_action_hash_required"
            ),
            "case_id": _text(case_id, code="case_id_required"),
            "parent_program_id": _text(parent_program_id, code="parent_program_id_required"),
            "parent_candidate_id": _text(parent_candidate_id, code="parent_candidate_id_required"),
            "immutable_sequences": immutable,
            "immutable_sequence_bundle_hash": computed_hashes["immutable_sequence_bundle_hash"],
            "parent_sequences": parent,
            "parent_sequence_bundle_hash": computed_hashes["parent_sequence_bundle_hash"],
            "reconciled_root_sequences": root,
            "reconciled_root_sequence_bundle_hash": computed_hashes["reconciled_root_sequence_bundle_hash"],
            "mutation_modules": [item.to_dict() for item in modules],
            "sequence_seeds": [item.to_dict() for item in seeds],
            "candidate_slots": [item.to_dict() for item in slots],
        }
        return cls(
            design_action_hash=semantic["design_action_hash"],
            compiled_design_action_hash=semantic["compiled_design_action_hash"],
            case_id=semantic["case_id"],
            parent_program_id=semantic["parent_program_id"],
            parent_candidate_id=semantic["parent_candidate_id"],
            immutable_sequences=immutable,
            immutable_sequence_bundle_hash=semantic[
                "immutable_sequence_bundle_hash"
            ],
            parent_sequences=parent,
            parent_sequence_bundle_hash=semantic["parent_sequence_bundle_hash"],
            reconciled_root_sequences=root,
            reconciled_root_sequence_bundle_hash=semantic[
                "reconciled_root_sequence_bundle_hash"
            ],
            mutation_modules=modules,
            sequence_seeds=seeds,
            candidate_slots=slots,
            compiled_portfolio_request_hash="compiled_portfolio_request_sha256:"
            + _compiled_digest(
                COMPILED_PORTFOLIO_OPTIMIZATION_REQUEST_VERSION, semantic
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_action_hash": self.design_action_hash,
            "compiled_design_action_hash": self.compiled_design_action_hash,
            "case_id": self.case_id,
            "parent_program_id": self.parent_program_id,
            "parent_candidate_id": self.parent_candidate_id,
            "immutable_sequences": dict(self.immutable_sequences),
            "immutable_sequence_bundle_hash": self.immutable_sequence_bundle_hash,
            "parent_sequences": dict(self.parent_sequences),
            "parent_sequence_bundle_hash": self.parent_sequence_bundle_hash,
            "reconciled_root_sequences": dict(self.reconciled_root_sequences),
            "reconciled_root_sequence_bundle_hash": self.reconciled_root_sequence_bundle_hash,
            "mutation_modules": [item.to_dict() for item in self.mutation_modules],
            "sequence_seeds": [item.to_dict() for item in self.sequence_seeds],
            "candidate_slots": [item.to_dict() for item in self.candidate_slots],
            "compiled_portfolio_request_hash": self.compiled_portfolio_request_hash,
        }

    def to_artifact(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledPortfolioOptimizationRequest":
        fields = {
            "schema_version", "design_action_hash", "compiled_design_action_hash",
            "case_id", "parent_program_id", "parent_candidate_id",
            "immutable_sequences", "immutable_sequence_bundle_hash",
            "parent_sequences", "parent_sequence_bundle_hash",
            "reconciled_root_sequences", "reconciled_root_sequence_bundle_hash",
            "mutation_modules", "sequence_seeds", "candidate_slots",
            "compiled_portfolio_request_hash",
        }
        raw = _closed_mapping(value, fields=fields, label="compiled_portfolio_request")
        if raw["schema_version"] != COMPILED_PORTFOLIO_OPTIMIZATION_REQUEST_VERSION:
            _compiled_fail("schema_version_invalid", "compiled_portfolio_request")
        compiled = cls.create(**{key: raw[key] for key in fields if key not in {
            "schema_version", "compiled_portfolio_request_hash"
        }})
        if raw["compiled_portfolio_request_hash"] != compiled.compiled_portfolio_request_hash:
            _compiled_fail("hash_mismatch", "compiled_portfolio_request")
        if _canonical_json(value) != _canonical_json(compiled.to_dict()):
            _compiled_fail("noncanonical_artifact", "compiled_portfolio_request")
        return compiled

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> "CompiledPortfolioOptimizationRequest":
        return cls.from_mapping(value)

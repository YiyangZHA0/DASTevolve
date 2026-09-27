

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

DESIGN_ACTION_VERSION = "astevolve.design_action.v1"
DESIGN_ACTION_HASH_PREFIX = "design_action_sha256:"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "parent",
        "coordinate_system",
        "ast_revision_plan",
        "position_distributions",
        "lineage_actions",
        "mutation_modules",
        "sequence_seeds",
        "candidate_portfolio",
        "mutation_count_range",
        "minimum_hamming_distance",
        "hypothesis",
        "falsification_predicates",
        "rollback_actions",
    }
)
_ARTIFACT_FIELDS = _PAYLOAD_FIELDS | {"design_action_hash"}
_PARENT_FIELDS = frozenset(
    {
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "parent_effective_contract_hash",
        "parent_evolve_hash",
    }
)
_COORDINATE_FIELDS = frozenset(
    {"chain_id_scheme", "residue_numbering", "index_base"}
)
_RESIDUE_FIELDS = frozenset({"chain_id", "residue_number"})
_POSITION_DISTRIBUTION_FIELDS = frozenset(
    {
        "residue",
        "enabled_residues",
        "forbidden_residues",
        "probabilities",
        "required_mutation",
        "wt_revert_weight",
        "temperature",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "lineage_action_id",
        "residue",
        "action",
        "expected_parent_residue",
        "replacement_residue",
        "source_node_id",
        "target_node_id",
        "rationale",
    }
)
_MUTATION_FIELDS = frozenset(
    {"residue", "from_residue", "to_residue"}
)
_MODULE_FIELDS = frozenset(
    {"module_id", "operation", "atomic", "mutations", "rationale"}
)
_SEED_FIELDS = frozenset(
    {"seed_id", "sequences", "refinement_positions", "rationale"}
)
_PORTFOLIO_FIELDS = frozenset(
    {
        "portfolio_id",
        "role",
        "count",
        "generation_mode",
        "source_refs",
        "required",
    }
)
_PREDICATE_FIELDS = frozenset(
    {"predicate_id", "metric", "provider", "aggregation", "operator", "threshold"}
)
_ROLLBACK_FIELDS = frozenset(
    {"rollback_action_id", "predicate_id", "action", "target_ref", "rationale"}
)
_HYPOTHESIS_FIELDS = frozenset(
    {"hypothesis_id", "statement", "rationale", "evidence_refs", "expected_effects"}
)

_LINEAGE_ACTIONS = frozenset(
    {"retain", "reassign_node", "replace", "revert_to_wt"}
)
_MODULE_OPERATIONS = frozenset(
    {"module_insert", "module_replace", "module_revert", "matched_module_ablation"}
)
_PORTFOLIO_ROLES = frozenset(
    {"primary", "secondary", "repair", "novelty", "control", "matched_ablation"}
)
_GENERATION_MODES = frozenset(
    {"strict_sequence_seed", "module", "mcts", "matched_ablation", "control"}
)
_AGGREGATIONS = frozenset({"single", "median", "worst_seed", "mean"})
_OPERATORS = frozenset({"<", "<=", ">", ">=", "=="})
_ROLLBACK_ACTIONS = frozenset(
    {"revert_to_parent", "revert_module", "activate_repair", "abstain"}
)
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+\-]{0,255}$")
_HASH_RE = re.compile(r"^design_action_sha256:[0-9a-f]{64}$")
_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


class DesignActionError(ValueError):


    def __init__(self, code: str, path: str = "$", detail: str = "") -> None:
        self.code = str(code)
        self.path = str(path or "$")
        self.detail = str(detail)
        suffix = f": {self.detail}" if self.detail else ""
        super().__init__(f"{self.code} at {self.path}{suffix}")


def _fail(code: str, path: str, detail: Any = "") -> None:
    raise DesignActionError(code, path, str(detail))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", path)
    non_strings = [repr(key) for key in value if not isinstance(key, str)]
    if non_strings:
        _fail("non_string_key", path, ",".join(non_strings))
    return value


def _closed(value: Any, fields: frozenset[str], path: str) -> Mapping[str, Any]:
    raw = _mapping(value, path)
    keys = set(raw)
    unknown = sorted(keys - fields)
    missing = sorted(fields - keys)
    if unknown:
        _fail("unknown_fields", path, ",".join(unknown))
    if missing:
        _fail("fields_missing", path, ",".join(missing))
    return raw


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("sequence_required", path)
    return value


def _text(
    value: Any,
    path: str,
    *,
    maximum: int = 4000,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail("text_required", path, repr(value))
    normalized = value.strip()
    if not normalized and not allow_empty:
        _fail("text_empty", path)
    if len(normalized) > maximum:
        _fail("text_too_long", path, maximum)
    return normalized


def _public_id(value: Any, path: str) -> str:
    resolved = _text(value, path, maximum=256)
    if _PUBLIC_ID_RE.fullmatch(resolved) is None:
        _fail("identifier_invalid", path, resolved)
    return resolved


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("integer_required", path, repr(value))
    if value < minimum:
        _fail("integer_out_of_range", path, value)
    return value


def _finite(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("finite_number_required", path, repr(value))
    resolved = float(value)
    if not math.isfinite(resolved):
        _fail("nonfinite_number", path, repr(value))
    if strictly_positive and resolved <= 0:
        _fail("number_out_of_range", path, resolved)
    if minimum is not None and resolved < minimum:
        _fail("number_out_of_range", path, resolved)
    return resolved


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail("boolean_required", path, repr(value))
    return value


def _optional_id(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _public_id(value, path)


def _amino_acid(value: Any, path: str) -> str:
    if not isinstance(value, str) or value not in CANONICAL_AMINO_ACIDS:
        _fail("amino_acid_invalid", path, repr(value))
    return value


def _optional_amino_acid(value: Any, path: str) -> str | None:
    return None if value is None else _amino_acid(value, path)


def _string_list(
    value: Any,
    path: str,
    *,
    public_ids: bool = False,
    maximum_items: int = 256,
) -> Tuple[str, ...]:
    raw = _sequence(value, path)
    if len(raw) > maximum_items:
        _fail("sequence_too_long", path, maximum_items)
    normalized = tuple(
        _public_id(item, f"{path}[{index}]")
        if public_ids
        else _text(item, f"{path}[{index}]", maximum=1000)
        for index, item in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        _fail("duplicate_value", path)
    return tuple(sorted(normalized))


def _amino_acid_list(value: Any, path: str) -> Tuple[str, ...]:
    raw = _sequence(value, path)
    normalized = tuple(
        _amino_acid(item, f"{path}[{index}]") for index, item in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        _fail("duplicate_value", path)
    return tuple(sorted(normalized))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("not_canonical_json", "$", error)


def _design_action_hash(payload: Mapping[str, Any]) -> str:
    encoded = f"{DESIGN_ACTION_VERSION}\0{_canonical_json(payload)}".encode("utf-8")
    return f"{DESIGN_ACTION_HASH_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def _strict_json(text: Any) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        _fail("json_text_required", "$")

    def object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                _fail("duplicate_json_key", "$", key)
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        _fail("nonfinite_number", "$", value)

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except DesignActionError:
        raise
    except json.JSONDecodeError as error:
        _fail("invalid_json", "$", f"line={error.lineno},column={error.colno}")
    return _mapping(parsed, "$")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ResidueRef:


    chain_id: str
    residue_number: int

    @property
    def zero_based_position(self) -> int:
        return self.residue_number - 1

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "residue") -> "ResidueRef":
        raw = _closed(value, _RESIDUE_FIELDS, path)
        return cls(
            chain_id=_public_id(raw["chain_id"], f"{path}.chain_id"),
            residue_number=_integer(
                raw["residue_number"], f"{path}.residue_number", minimum=1
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"chain_id": self.chain_id, "residue_number": self.residue_number}


@dataclass(frozen=True)
class ParentBinding:
    case_id: str
    parent_program_id: str
    parent_candidate_id: str
    parent_sequence_bundle_hash: str
    parent_effective_contract_hash: str
    parent_evolve_hash: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "parent") -> "ParentBinding":
        raw = _closed(value, _PARENT_FIELDS, path)
        return cls(
            **{
                field: _public_id(raw[field], f"{path}.{field}")
                for field in sorted(_PARENT_FIELDS)
            }
        )

    def to_dict(self) -> Dict[str, str]:
        return {field: str(getattr(self, field)) for field in sorted(_PARENT_FIELDS)}


@dataclass(frozen=True)
class CoordinateSystem:
    chain_id_scheme: str
    residue_numbering: str
    index_base: int

    @classmethod
    def from_mapping(
        cls, value: Any, *, path: str = "coordinate_system"
    ) -> "CoordinateSystem":
        raw = _closed(value, _COORDINATE_FIELDS, path)
        scheme = _public_id(raw["chain_id_scheme"], f"{path}.chain_id_scheme")
        numbering = _text(raw["residue_numbering"], f"{path}.residue_numbering")
        base = _integer(raw["index_base"], f"{path}.index_base", minimum=0)
        if numbering != "one_based" or base != 1:
            _fail(
                "coordinate_system_not_one_based",
                path,
                f"residue_numbering={numbering!r},index_base={base}",
            )
        return cls(scheme, numbering, base)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id_scheme": self.chain_id_scheme,
            "residue_numbering": self.residue_numbering,
            "index_base": self.index_base,
        }


@dataclass(frozen=True)
class PositionDistribution:
    residue: ResidueRef
    enabled_residues: Tuple[str, ...]
    forbidden_residues: Tuple[str, ...]
    probabilities: Mapping[str, float]
    required_mutation: str | None
    wt_revert_weight: float
    temperature: float

    @classmethod
    def from_mapping(
        cls, value: Any, *, path: str = "position_distribution"
    ) -> "PositionDistribution":
        raw = _closed(value, _POSITION_DISTRIBUTION_FIELDS, path)
        enabled = _amino_acid_list(raw["enabled_residues"], f"{path}.enabled_residues")
        if not enabled:
            _fail("enabled_residues_empty", f"{path}.enabled_residues")
        forbidden = _amino_acid_list(
            raw["forbidden_residues"], f"{path}.forbidden_residues"
        )
        overlap = sorted(set(enabled) & set(forbidden))
        if overlap:
            _fail("enabled_forbidden_overlap", path, ",".join(overlap))
        probability_raw = _mapping(raw["probabilities"], f"{path}.probabilities")
        probabilities: Dict[str, float] = {}
        for residue, weight in probability_raw.items():
            aa = _amino_acid(residue, f"{path}.probabilities.{residue}")
            probabilities[aa] = _finite(
                weight, f"{path}.probabilities.{aa}", minimum=0.0
            )
        if set(probabilities) != set(enabled):
            _fail(
                "probability_alphabet_mismatch",
                f"{path}.probabilities",
                f"expected={','.join(enabled)};actual={','.join(sorted(probabilities))}",
            )
        total = math.fsum(probabilities.values())
        if abs(total - 1.0) > 1e-8:
            _fail("probabilities_not_normalized", f"{path}.probabilities", total)
        required = _optional_amino_acid(
            raw["required_mutation"], f"{path}.required_mutation"
        )
        if required is not None and (
            required not in probabilities or probabilities[required] <= 0
        ):
            _fail("required_mutation_not_enabled", f"{path}.required_mutation", required)
        return cls(
            residue=ResidueRef.from_mapping(raw["residue"], path=f"{path}.residue"),
            enabled_residues=enabled,
            forbidden_residues=forbidden,
            probabilities=MappingProxyType(dict(sorted(probabilities.items()))),
            required_mutation=required,
            wt_revert_weight=_finite(
                raw["wt_revert_weight"], f"{path}.wt_revert_weight", minimum=0.0
            ),
            temperature=_finite(
                raw["temperature"], f"{path}.temperature", strictly_positive=True
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residue": self.residue.to_dict(),
            "enabled_residues": list(self.enabled_residues),
            "forbidden_residues": list(self.forbidden_residues),
            "probabilities": dict(self.probabilities),
            "required_mutation": self.required_mutation,
            "wt_revert_weight": self.wt_revert_weight,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class LineageAction:
    lineage_action_id: str
    residue: ResidueRef
    action: str
    expected_parent_residue: str
    replacement_residue: str | None
    source_node_id: str
    target_node_id: str | None
    rationale: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "lineage_action") -> "LineageAction":
        raw = _closed(value, _LINEAGE_FIELDS, path)
        action = _text(raw["action"], f"{path}.action")
        if action not in _LINEAGE_ACTIONS:
            _fail("lineage_action_invalid", f"{path}.action", action)
        replacement = _optional_amino_acid(
            raw["replacement_residue"], f"{path}.replacement_residue"
        )
        target = _optional_id(raw["target_node_id"], f"{path}.target_node_id")
        source = _public_id(raw["source_node_id"], f"{path}.source_node_id")
        expected = _amino_acid(
            raw["expected_parent_residue"], f"{path}.expected_parent_residue"
        )
        if action == "replace":
            if replacement is None:
                _fail("replacement_residue_required", f"{path}.replacement_residue")
            if replacement == expected:
                _fail("replacement_residue_unchanged", f"{path}.replacement_residue")
        elif replacement is not None:
            _fail("replacement_residue_forbidden", f"{path}.replacement_residue", action)
        if action in {"retain", "reassign_node", "replace"} and target is None:
            _fail("target_node_required", f"{path}.target_node_id", action)
        if action == "revert_to_wt" and target is not None:
            _fail("target_node_forbidden", f"{path}.target_node_id", action)
        if action == "retain" and target != source:
            _fail(
                "retain_owner_mismatch",
                f"{path}.target_node_id",
                f"source={source};target={target}",
            )
        if action == "reassign_node" and target == source:
            _fail("reassign_owner_unchanged", f"{path}.target_node_id", source)
        return cls(
            lineage_action_id=_public_id(
                raw["lineage_action_id"], f"{path}.lineage_action_id"
            ),
            residue=ResidueRef.from_mapping(raw["residue"], path=f"{path}.residue"),
            action=action,
            expected_parent_residue=expected,
            replacement_residue=replacement,
            source_node_id=source,
            target_node_id=target,
            rationale=_text(raw["rationale"], f"{path}.rationale", maximum=1500),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_action_id": self.lineage_action_id,
            "residue": self.residue.to_dict(),
            "action": self.action,
            "expected_parent_residue": self.expected_parent_residue,
            "replacement_residue": self.replacement_residue,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MutationSpec:
    residue: ResidueRef
    from_residue: str
    to_residue: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "mutation") -> "MutationSpec":
        raw = _closed(value, _MUTATION_FIELDS, path)
        source = _amino_acid(raw["from_residue"], f"{path}.from_residue")
        target = _amino_acid(raw["to_residue"], f"{path}.to_residue")
        if source == target:
            _fail("mutation_unchanged", path, source)
        return cls(
            residue=ResidueRef.from_mapping(raw["residue"], path=f"{path}.residue"),
            from_residue=source,
            to_residue=target,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residue": self.residue.to_dict(),
            "from_residue": self.from_residue,
            "to_residue": self.to_residue,
        }


@dataclass(frozen=True)
class MutationModule:
    module_id: str
    operation: str
    atomic: bool
    mutations: Tuple[MutationSpec, ...]
    rationale: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "mutation_module") -> "MutationModule":
        raw = _closed(value, _MODULE_FIELDS, path)
        operation = _text(raw["operation"], f"{path}.operation")
        if operation not in _MODULE_OPERATIONS:
            _fail("module_operation_invalid", f"{path}.operation", operation)
        mutation_values = _sequence(raw["mutations"], f"{path}.mutations")
        if not mutation_values:
            _fail("module_mutations_empty", f"{path}.mutations")
        mutations = tuple(
            MutationSpec.from_mapping(item, path=f"{path}.mutations[{index}]")
            for index, item in enumerate(mutation_values)
        )
        _reject_duplicate_residues(mutations, f"{path}.mutations")
        return cls(
            module_id=_public_id(raw["module_id"], f"{path}.module_id"),
            operation=operation,
            atomic=_boolean(raw["atomic"], f"{path}.atomic"),
            mutations=tuple(sorted(mutations, key=lambda item: _residue_key(item.residue))),
            rationale=_text(raw["rationale"], f"{path}.rationale", maximum=1500),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "operation": self.operation,
            "atomic": self.atomic,
            "mutations": [item.to_dict() for item in self.mutations],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SequenceSeed:
    seed_id: str
    sequences: Mapping[str, str]
    refinement_positions: Tuple[ResidueRef, ...]
    rationale: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "sequence_seed") -> "SequenceSeed":
        raw = _closed(value, _SEED_FIELDS, path)
        sequences_raw = _mapping(raw["sequences"], f"{path}.sequences")
        if not sequences_raw:
            _fail("seed_sequences_empty", f"{path}.sequences")
        sequences: Dict[str, str] = {}
        for chain, sequence in sequences_raw.items():
            clean_chain = _public_id(chain, f"{path}.sequences.{chain}")
            if not isinstance(sequence, str) or _SEQUENCE_RE.fullmatch(sequence) is None:
                _fail("seed_sequence_invalid", f"{path}.sequences.{clean_chain}")
            sequences[clean_chain] = sequence
        refs_raw = _sequence(raw["refinement_positions"], f"{path}.refinement_positions")
        refs = tuple(
            ResidueRef.from_mapping(item, path=f"{path}.refinement_positions[{index}]")
            for index, item in enumerate(refs_raw)
        )
        _reject_duplicate_residues(refs, f"{path}.refinement_positions")
        return cls(
            seed_id=_public_id(raw["seed_id"], f"{path}.seed_id"),
            sequences=MappingProxyType(dict(sorted(sequences.items()))),
            refinement_positions=tuple(sorted(refs, key=_residue_key)),
            rationale=_text(raw["rationale"], f"{path}.rationale", maximum=1500),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "sequences": dict(self.sequences),
            "refinement_positions": [item.to_dict() for item in self.refinement_positions],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CandidatePortfolioEntry:
    portfolio_id: str
    role: str
    count: int
    generation_mode: str
    source_refs: Tuple[str, ...]
    required: bool

    @classmethod
    def from_mapping(
        cls, value: Any, *, path: str = "candidate_portfolio"
    ) -> "CandidatePortfolioEntry":
        raw = _closed(value, _PORTFOLIO_FIELDS, path)
        role = _text(raw["role"], f"{path}.role")
        mode = _text(raw["generation_mode"], f"{path}.generation_mode")
        if role not in _PORTFOLIO_ROLES:
            _fail("portfolio_role_invalid", f"{path}.role", role)
        if mode not in _GENERATION_MODES:
            _fail("generation_mode_invalid", f"{path}.generation_mode", mode)
        refs = _string_list(raw["source_refs"], f"{path}.source_refs", public_ids=True)
        if mode in {"strict_sequence_seed", "module", "matched_ablation"} and not refs:
            _fail("portfolio_source_refs_empty", f"{path}.source_refs", mode)
        return cls(
            portfolio_id=_public_id(raw["portfolio_id"], f"{path}.portfolio_id"),
            role=role,
            count=_integer(raw["count"], f"{path}.count", minimum=1),
            generation_mode=mode,
            source_refs=refs,
            required=_boolean(raw["required"], f"{path}.required"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "role": self.role,
            "count": self.count,
            "generation_mode": self.generation_mode,
            "source_refs": list(self.source_refs),
            "required": self.required,
        }


@dataclass(frozen=True)
class MetricPredicate:
    predicate_id: str
    metric: str
    provider: str
    aggregation: str
    operator: str
    threshold: float

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "predicate") -> "MetricPredicate":
        raw = _closed(value, _PREDICATE_FIELDS, path)
        aggregation = _text(raw["aggregation"], f"{path}.aggregation")
        operator = _text(raw["operator"], f"{path}.operator")
        if aggregation not in _AGGREGATIONS:
            _fail("predicate_aggregation_invalid", f"{path}.aggregation", aggregation)
        if operator not in _OPERATORS:
            _fail("predicate_operator_invalid", f"{path}.operator", operator)
        return cls(
            predicate_id=_public_id(raw["predicate_id"], f"{path}.predicate_id"),
            metric=_public_id(raw["metric"], f"{path}.metric"),
            provider=_public_id(raw["provider"], f"{path}.provider"),
            aggregation=aggregation,
            operator=operator,
            threshold=_finite(raw["threshold"], f"{path}.threshold"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "metric": self.metric,
            "provider": self.provider,
            "aggregation": self.aggregation,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RollbackAction:
    rollback_action_id: str
    predicate_id: str
    action: str
    target_ref: str | None
    rationale: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "rollback_action") -> "RollbackAction":
        raw = _closed(value, _ROLLBACK_FIELDS, path)
        action = _text(raw["action"], f"{path}.action")
        if action not in _ROLLBACK_ACTIONS:
            _fail("rollback_action_invalid", f"{path}.action", action)
        target = _optional_id(raw["target_ref"], f"{path}.target_ref")
        if action == "revert_module" and target is None:
            _fail("rollback_target_required", f"{path}.target_ref", action)
        if action in {"revert_to_parent", "abstain"} and target is not None:
            _fail("rollback_target_forbidden", f"{path}.target_ref", action)
        return cls(
            rollback_action_id=_public_id(
                raw["rollback_action_id"], f"{path}.rollback_action_id"
            ),
            predicate_id=_public_id(raw["predicate_id"], f"{path}.predicate_id"),
            action=action,
            target_ref=target,
            rationale=_text(raw["rationale"], f"{path}.rationale", maximum=1500),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_action_id": self.rollback_action_id,
            "predicate_id": self.predicate_id,
            "action": self.action,
            "target_ref": self.target_ref,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class DesignHypothesis:
    hypothesis_id: str
    statement: str
    rationale: str
    evidence_refs: Tuple[str, ...]
    expected_effects: Tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "hypothesis") -> "DesignHypothesis":
        raw = _closed(value, _HYPOTHESIS_FIELDS, path)
        return cls(
            hypothesis_id=_public_id(raw["hypothesis_id"], f"{path}.hypothesis_id"),
            statement=_text(raw["statement"], f"{path}.statement", maximum=2000),
            rationale=_text(raw["rationale"], f"{path}.rationale", maximum=2000),
            evidence_refs=_string_list(
                raw["evidence_refs"], f"{path}.evidence_refs", public_ids=True
            ),
            expected_effects=_string_list(
                raw["expected_effects"], f"{path}.expected_effects", maximum_items=32
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
            "expected_effects": list(self.expected_effects),
        }


def _residue_key(value: Any) -> Tuple[str, int]:
    residue = value if isinstance(value, ResidueRef) else value.residue
    return residue.chain_id, residue.residue_number


def _reject_duplicate_residues(values: Sequence[Any], path: str) -> None:
    keys = [_residue_key(item) for item in values]
    if len(keys) != len(set(keys)):
        _fail("duplicate_residue", path)


def _objects(
    value: Any,
    path: str,
    factory: Any,
    *,
    sort_key: Any,
) -> Tuple[Any, ...]:
    raw = _sequence(value, path)
    normalized = tuple(
        factory(item, path=f"{path}[{index}]") for index, item in enumerate(raw)
    )
    return tuple(sorted(normalized, key=sort_key))


@dataclass(frozen=True)
class DesignAction:


    parent: ParentBinding
    coordinate_system: CoordinateSystem
    ast_revision_plan: Mapping[str, Any]
    position_distributions: Tuple[PositionDistribution, ...]
    lineage_actions: Tuple[LineageAction, ...]
    mutation_modules: Tuple[MutationModule, ...]
    sequence_seeds: Tuple[SequenceSeed, ...]
    candidate_portfolio: Tuple[CandidatePortfolioEntry, ...]
    mutation_count_range: Tuple[int, int]
    minimum_hamming_distance: int
    hypothesis: DesignHypothesis
    falsification_predicates: Tuple[MetricPredicate, ...]
    rollback_actions: Tuple[RollbackAction, ...]
    design_action_hash: str
    schema_version: str = DESIGN_ACTION_VERSION

    @classmethod
    def create(cls, **fields: Any) -> "DesignAction":


        return cls.from_payload({"schema_version": DESIGN_ACTION_VERSION, **fields})

    @classmethod
    def from_json(cls, text: Any) -> "DesignAction":


        return cls.from_mapping(_strict_json(text))

    @classmethod
    def from_payload(cls, value: Any) -> "DesignAction":


        raw = _closed(value, _PAYLOAD_FIELDS, "$")
        if raw["schema_version"] != DESIGN_ACTION_VERSION:
            _fail(
                "schema_version_unsupported",
                "$.schema_version",
                repr(raw["schema_version"]),
            )


        from astevolve.semantic_graph.ast_revision import (
            ASTRevisionError,
            normalize_global_ast_revision_plan,
        )

        try:
            plan = normalize_global_ast_revision_plan(raw["ast_revision_plan"])
        except ASTRevisionError as error:
            _fail("ast_revision_plan_invalid", "$.ast_revision_plan", error)

        positions = _objects(
            raw["position_distributions"],
            "$.position_distributions",
            PositionDistribution.from_mapping,
            sort_key=lambda item: _residue_key(item.residue),
        )
        _reject_duplicate_residues(positions, "$.position_distributions")
        lineage = _objects(
            raw["lineage_actions"],
            "$.lineage_actions",
            LineageAction.from_mapping,
            sort_key=lambda item: item.lineage_action_id,
        )
        _reject_duplicate_residues(lineage, "$.lineage_actions")
        modules = _objects(
            raw["mutation_modules"],
            "$.mutation_modules",
            MutationModule.from_mapping,
            sort_key=lambda item: item.module_id,
        )
        seeds = _objects(
            raw["sequence_seeds"],
            "$.sequence_seeds",
            SequenceSeed.from_mapping,
            sort_key=lambda item: item.seed_id,
        )
        portfolio = _objects(
            raw["candidate_portfolio"],
            "$.candidate_portfolio",
            CandidatePortfolioEntry.from_mapping,
            sort_key=lambda item: item.portfolio_id,
        )
        predicates = _objects(
            raw["falsification_predicates"],
            "$.falsification_predicates",
            MetricPredicate.from_mapping,
            sort_key=lambda item: item.predicate_id,
        )
        rollback = _objects(
            raw["rollback_actions"],
            "$.rollback_actions",
            RollbackAction.from_mapping,
            sort_key=lambda item: item.rollback_action_id,
        )

        range_raw = _sequence(raw["mutation_count_range"], "$.mutation_count_range")
        if len(range_raw) != 2:
            _fail("mutation_count_range_invalid", "$.mutation_count_range")
        minimum = _integer(range_raw[0], "$.mutation_count_range[0]", minimum=0)
        maximum = _integer(range_raw[1], "$.mutation_count_range[1]", minimum=0)
        if minimum > maximum:
            _fail("mutation_count_range_invalid", "$.mutation_count_range", "min>max")

        hypothesis = DesignHypothesis.from_mapping(raw["hypothesis"], path="$.hypothesis")
        _validate_unique_ids(hypothesis, lineage, modules, seeds, portfolio, predicates, rollback)
        predicate_ids = {item.predicate_id for item in predicates}
        for item in rollback:
            if item.predicate_id not in predicate_ids:
                _fail(
                    "rollback_predicate_unknown",
                    "$.rollback_actions",
                    item.predicate_id,
                )

        provisional = cls(
            parent=ParentBinding.from_mapping(raw["parent"], path="$.parent"),
            coordinate_system=CoordinateSystem.from_mapping(
                raw["coordinate_system"], path="$.coordinate_system"
            ),
            ast_revision_plan=_freeze_json(plan),
            position_distributions=positions,
            lineage_actions=lineage,
            mutation_modules=modules,
            sequence_seeds=seeds,
            candidate_portfolio=portfolio,
            mutation_count_range=(minimum, maximum),
            minimum_hamming_distance=_integer(
                raw["minimum_hamming_distance"],
                "$.minimum_hamming_distance",
                minimum=0,
            ),
            hypothesis=hypothesis,
            falsification_predicates=predicates,
            rollback_actions=rollback,
            design_action_hash="",
        )
        digest = _design_action_hash(provisional.to_payload())
        object.__setattr__(provisional, "design_action_hash", digest)
        return provisional

    @classmethod
    def from_mapping(cls, value: Any) -> "DesignAction":


        raw = _closed(value, _ARTIFACT_FIELDS, "$")
        supplied_hash = raw["design_action_hash"]
        if not isinstance(supplied_hash, str) or _HASH_RE.fullmatch(supplied_hash) is None:
            _fail("design_action_hash_invalid", "$.design_action_hash", supplied_hash)
        payload = {field: raw[field] for field in _PAYLOAD_FIELDS}
        action = cls.from_payload(payload)
        if supplied_hash != action.design_action_hash:
            _fail(
                "design_action_hash_mismatch",
                "$.design_action_hash",
                f"expected={action.design_action_hash};actual={supplied_hash}",
            )
        return action

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": DESIGN_ACTION_VERSION,
            "parent": self.parent.to_dict(),
            "coordinate_system": self.coordinate_system.to_dict(),
            "ast_revision_plan": _thaw_json(self.ast_revision_plan),
            "position_distributions": [item.to_dict() for item in self.position_distributions],
            "lineage_actions": [item.to_dict() for item in self.lineage_actions],
            "mutation_modules": [item.to_dict() for item in self.mutation_modules],
            "sequence_seeds": [item.to_dict() for item in self.sequence_seeds],
            "candidate_portfolio": [item.to_dict() for item in self.candidate_portfolio],
            "mutation_count_range": list(self.mutation_count_range),
            "minimum_hamming_distance": self.minimum_hamming_distance,
            "hypothesis": self.hypothesis.to_dict(),
            "falsification_predicates": [
                item.to_dict() for item in self.falsification_predicates
            ],
            "rollback_actions": [item.to_dict() for item in self.rollback_actions],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.to_payload(), "design_action_hash": self.design_action_hash}

    def canonical_json(self) -> str:


        return _canonical_json(self.to_payload())


def _validate_unique_ids(
    hypothesis: DesignHypothesis,
    lineage: Sequence[LineageAction],
    modules: Sequence[MutationModule],
    seeds: Sequence[SequenceSeed],
    portfolio: Sequence[CandidatePortfolioEntry],
    predicates: Sequence[MetricPredicate],
    rollback: Sequence[RollbackAction],
) -> None:
    records: list[tuple[str, str]] = [(hypothesis.hypothesis_id, "$.hypothesis")]
    records.extend((item.lineage_action_id, "$.lineage_actions") for item in lineage)
    records.extend((item.module_id, "$.mutation_modules") for item in modules)
    records.extend((item.seed_id, "$.sequence_seeds") for item in seeds)
    records.extend((item.portfolio_id, "$.candidate_portfolio") for item in portfolio)
    records.extend((item.predicate_id, "$.falsification_predicates") for item in predicates)
    records.extend((item.rollback_action_id, "$.rollback_actions") for item in rollback)
    seen: Dict[str, str] = {}
    for identifier, path in records:
        if identifier in seen:
            _fail("duplicate_id", path, f"{identifier};first={seen[identifier]}")
        seen[identifier] = path


__all__ = [
    "CANONICAL_AMINO_ACIDS",
    "DESIGN_ACTION_HASH_PREFIX",
    "DESIGN_ACTION_VERSION",
    "CandidatePortfolioEntry",
    "CoordinateSystem",
    "DesignAction",
    "DesignActionError",
    "DesignHypothesis",
    "LineageAction",
    "MetricPredicate",
    "MutationModule",
    "MutationSpec",
    "ParentBinding",
    "PositionDistribution",
    "ResidueRef",
    "RollbackAction",
    "SequenceSeed",
]

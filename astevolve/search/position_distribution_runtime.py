

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


POLICY_SCHEMA_VERSION = "astevolve.position_distribution_runtime_policy.v1"
EXECUTION_SCHEMA_VERSION = "astevolve.position_distribution_execution.v1"

_CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_WRAPPER_FIELDS = frozenset(
    {
        "schema_version",
        "compiled_design_action_hash",
        "distribution_set_hash",
        "rows",
    }
)
_ROW_FIELDS = frozenset(
    {
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
    }
)


class PositionDistributionRuntimeError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: Any = "") -> None:
    raise PositionDistributionRuntimeError(code, str(detail))


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


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(
        f"{domain}\0{_canonical_json(value)}".encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("sequence_required", label)
    return value


def _closed(value: Mapping[Any, Any], expected: frozenset[str], label: str) -> None:
    observed = {str(key) for key in value}
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        _fail("unknown_fields", f"{label}:{','.join(unknown)}")
    if missing:
        _fail("fields_missing", f"{label}:{','.join(missing)}")


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, repr(value))
    return value.strip()


def _position(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("position_invalid", repr(value))
    return value


def _residue(value: Any, label: str) -> str:
    if not isinstance(value, str):
        _fail("residue_invalid", f"{label}:{value!r}")
    residue = value.strip().upper()
    if len(residue) != 1 or residue not in _CANONICAL_AA:
        _fail("residue_invalid", f"{label}:{value!r}")
    return residue


def _residue_tuple(value: Any, label: str) -> Tuple[str, ...]:
    residues = tuple(_residue(item, label) for item in _sequence(value, label))
    if len(set(residues)) != len(residues):
        _fail("residue_duplicate", label)
    return tuple(sorted(residues))


def _probabilities(value: Any, label: str) -> Dict[str, float]:
    raw = _mapping(value, label)
    clean: Dict[str, float] = {}
    for raw_residue, raw_probability in raw.items():
        residue = _residue(raw_residue, label)
        if isinstance(raw_probability, bool) or not isinstance(
            raw_probability, (int, float)
        ):
            _fail("probability_invalid", f"{label}:{residue}")
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0.0:
            _fail("probability_invalid", f"{label}:{residue}:{probability}")
        if residue in clean:
            _fail("probability_residue_duplicate", f"{label}:{residue}")
        clean[residue] = probability
    if not clean:
        _fail("probabilities_empty", label)
    return dict(sorted(clean.items()))


@dataclass(frozen=True)
class RuntimePositionDistribution:


    schema_version: str
    chain_id: str
    position: int
    owner_node_id: str
    immutable_residue: str
    reconciled_root_residue: str
    hard_allowed_residues: Tuple[str, ...]
    enabled_residues: Tuple[str, ...]
    forbidden_residues: Tuple[str, ...]
    requested_probabilities: Tuple[Tuple[str, float], ...]
    effective_probabilities: Tuple[Tuple[str, float], ...]
    required_mutation: Optional[str]
    wt_revert_weight: float
    temperature: float
    distribution_hash: str

    @property
    def effective_probability_map(self) -> Dict[str, float]:


        return {
            residue: probability
            for residue, probability in self.effective_probabilities
            if probability > 0.0
        }

    @property
    def executable_residues(self) -> Tuple[str, ...]:
        if self.required_mutation is not None:
            return (self.required_mutation,)
        return tuple(self.effective_probability_map)

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
    def from_mapping(cls, value: Any) -> "RuntimePositionDistribution":
        raw = _mapping(value, "position_distribution")
        _closed(raw, _ROW_FIELDS, "position_distribution")
        hard = _residue_tuple(raw["hard_allowed_residues"], "hard_allowed_residues")
        enabled = _residue_tuple(raw["enabled_residues"], "enabled_residues")
        forbidden = _residue_tuple(raw["forbidden_residues"], "forbidden_residues")
        if not enabled or not set(enabled).issubset(hard):
            _fail("enabled_residues_widen_hard_alphabet")
        if set(enabled) & set(forbidden):
            _fail("enabled_forbidden_overlap")
        requested = _probabilities(
            raw["requested_probabilities"], "requested_probabilities"
        )
        effective = _probabilities(
            raw["effective_probabilities"], "effective_probabilities"
        )
        if set(requested) != set(enabled) or set(effective) != set(enabled):
            _fail("probability_support_mismatch")
        total = math.fsum(effective.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            _fail("effective_probability_not_normalized", repr(total))
        positive = {residue for residue, probability in effective.items() if probability > 0.0}
        if not positive:
            _fail("effective_probability_support_empty")
        required_raw = raw["required_mutation"]
        required = (
            None
            if required_raw is None
            else _residue(required_raw, "required_mutation")
        )
        if required is not None:
            if required not in positive:
                _fail("required_mutation_zero_probability", required)
            immutable = _residue(raw["immutable_residue"], "immutable_residue")
            root = _residue(
                raw["reconciled_root_residue"], "reconciled_root_residue"
            )
            if required == immutable:
                _fail("required_mutation_noop", required)
            if required != root:
                _fail("required_mutation_root_mismatch", f"{root}/{required}")
        wt_weight = raw["wt_revert_weight"]
        temperature = raw["temperature"]
        for number, code in (
            (wt_weight, "wt_revert_weight_invalid"),
            (temperature, "temperature_invalid"),
        ):
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                _fail(code, repr(number))
            if not math.isfinite(float(number)):
                _fail(code, repr(number))
        if float(wt_weight) < 0.0:
            _fail("wt_revert_weight_invalid", repr(wt_weight))
        if float(temperature) <= 0.0:
            _fail("temperature_invalid", repr(temperature))
        return cls(
            schema_version=_text(raw["schema_version"], "schema_version_invalid"),
            chain_id=_text(raw["chain_id"], "chain_id_invalid"),
            position=_position(raw["position"]),
            owner_node_id=_text(raw["owner_node_id"], "owner_node_id_invalid"),
            immutable_residue=_residue(raw["immutable_residue"], "immutable_residue"),
            reconciled_root_residue=_residue(
                raw["reconciled_root_residue"], "reconciled_root_residue"
            ),
            hard_allowed_residues=hard,
            enabled_residues=enabled,
            forbidden_residues=forbidden,
            requested_probabilities=tuple(requested.items()),
            effective_probabilities=tuple(effective.items()),
            required_mutation=required,
            wt_revert_weight=float(wt_weight),
            temperature=float(temperature),
            distribution_hash=_text(raw["distribution_hash"], "distribution_hash_invalid"),
        )


@dataclass(frozen=True)
class RuntimePositionDistributionPolicy:


    source_field: str
    source_schema_version: str
    compiled_design_action_hash: str
    distribution_set_hash: str
    rows: Tuple[RuntimePositionDistribution, ...]

    @property
    def formal(self) -> bool:
        return self.source_field == "compiled_position_distribution_policy"

    def get(self, chain_id: str, position: int) -> Optional[RuntimePositionDistribution]:
        key = (str(chain_id), int(position))
        return next(
            (row for row in self.rows if (row.chain_id, row.position) == key),
            None,
        )

    def for_node(
        self,
        chain_id: str,
        owner_node_id: str,
        positions: Optional[Iterable[int]] = None,
    ) -> Tuple[RuntimePositionDistribution, ...]:
        allowed_positions = None if positions is None else {int(item) for item in positions}
        return tuple(
            row
            for row in self.rows
            if row.chain_id == str(chain_id)
            and row.owner_node_id == str(owner_node_id)
            and (allowed_positions is None or row.position in allowed_positions)
        )

    def execution_receipt(
        self,
        rows: Iterable[RuntimePositionDistribution],
        *,
        parent_sequence: Optional[str] = None,
        final_sequence: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected = tuple(sorted(rows, key=lambda row: (row.chain_id, row.position)))
        realizations = []
        for row in selected:
            before = (
                parent_sequence[row.position]
                if parent_sequence is not None and row.position < len(parent_sequence)
                else None
            )
            after = (
                final_sequence[row.position]
                if final_sequence is not None and row.position < len(final_sequence)
                else None
            )
            realizations.append(
                {
                    "chain_id": row.chain_id,
                    "position": row.position,
                    "owner_node_id": row.owner_node_id,
                    "distribution_hash": row.distribution_hash,
                    "effective_probabilities": row.effective_probability_map,
                    "required_mutation": row.required_mutation,
                    "parent_residue": before,
                    "final_residue": after,
                    "required_realized": (
                        None
                        if row.required_mutation is None or after is None
                        else after == row.required_mutation
                    ),
                }
            )
        payload = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "source_field": self.source_field,
            "formal_compiler_wrapper": self.formal,
            "compiled_design_action_hash": self.compiled_design_action_hash,
            "distribution_set_hash": self.distribution_set_hash,
            "distribution_hashes": [row.distribution_hash for row in selected],
            "realizations": realizations,
        }
        return {
            **payload,
            "execution_hash": _digest(EXECUTION_SCHEMA_VERSION, payload),
        }


def _alias_wrapper(value: Any, source_field: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and "rows" in value:
        return value
    rows = list(_sequence(value, source_field))
    row_payloads = [dict(_mapping(row, source_field)) for row in rows]
    set_hash = _digest("astevolve.position_distribution_alias_set.v1", row_payloads)
    return {
        "schema_version": "astevolve.position_distribution_alias_wrapper.v1",
        "compiled_design_action_hash": "unbound-development-alias",
        "distribution_set_hash": set_hash,
        "rows": row_payloads,
    }


def resolve_position_distribution_policy(
    cfg_or_mapping: Any,
) -> RuntimePositionDistributionPolicy:


    source_field = "compiled_position_distribution_policy"
    value = (
        cfg_or_mapping.get(source_field)
        if isinstance(cfg_or_mapping, Mapping)
        else getattr(cfg_or_mapping, source_field, None)
    )
    if not value:


        for alias in ("compiled_position_distributions", "position_distributions"):
            candidate = (
                cfg_or_mapping.get(alias)
                if isinstance(cfg_or_mapping, Mapping)
                else getattr(cfg_or_mapping, alias, None)
            )
            if candidate:
                source_field, value = alias, candidate
                break
    if not value:
        return RuntimePositionDistributionPolicy(
            source_field=source_field,
            source_schema_version=POLICY_SCHEMA_VERSION,
            compiled_design_action_hash="",
            distribution_set_hash="",
            rows=(),
        )
    wrapper = _mapping(_alias_wrapper(value, source_field), source_field)
    _closed(wrapper, _WRAPPER_FIELDS, source_field)
    row_values: Sequence[Any] = _sequence(
        wrapper["rows"], f"{source_field}.rows"
    )
    if source_field == "compiled_position_distribution_policy":
        if wrapper["schema_version"] != (
            "astevolve.compiled_position_distribution_policy.v1"
        ):
            _fail("formal_policy_schema_version_invalid", wrapper["schema_version"])
        action_hash = _text(
            wrapper["compiled_design_action_hash"],
            "compiled_design_action_hash_invalid",
        )
        if not action_hash.startswith("compiled_design_action_sha256:"):
            _fail("compiled_design_action_hash_invalid", action_hash)
        if not row_values:
            _fail("formal_policy_rows_empty")


        try:
            from astevolve.domain.compiled import (
                CompiledPositionDistribution,
                compiled_position_distribution_set_hash,
            )

            compiled_rows = tuple(
                CompiledPositionDistribution.from_mapping(_mapping(item, "row"))
                for item in row_values
            )
            computed_set_hash = compiled_position_distribution_set_hash(
                compiled_rows
            )
        except Exception as exc:
            _fail("formal_position_distribution_invalid", exc)
        if wrapper["distribution_set_hash"] != computed_set_hash:
            _fail(
                "distribution_set_hash_mismatch",
                f"{wrapper['distribution_set_hash']}/{computed_set_hash}",
            )
        row_values = tuple(item.to_dict() for item in compiled_rows)
    rows = tuple(
        RuntimePositionDistribution.from_mapping(item)
        for item in row_values
    )
    keys = [(row.chain_id, row.position) for row in rows]
    if len(set(keys)) != len(keys):
        _fail("position_distribution_duplicate")
    hashes = [row.distribution_hash for row in rows]
    if len(set(hashes)) != len(hashes):
        _fail("distribution_hash_duplicate")
    return RuntimePositionDistributionPolicy(
        source_field=source_field,
        source_schema_version=_text(wrapper["schema_version"], "schema_version_invalid"),
        compiled_design_action_hash=_text(
            wrapper["compiled_design_action_hash"], "compiled_design_action_hash_invalid"
        ),
        distribution_set_hash=_text(
            wrapper["distribution_set_hash"], "distribution_set_hash_invalid"
        ),
        rows=tuple(sorted(rows, key=lambda row: (row.chain_id, row.position))),
    )


def sample_effective_residue(
    distribution: RuntimePositionDistribution, rng: Any
) -> str:


    probabilities = distribution.effective_probability_map
    if distribution.required_mutation is not None:
        return distribution.required_mutation
    residues = tuple(probabilities)
    weights = tuple(probabilities[residue] for residue in residues)
    if not residues or not math.isclose(
        math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        _fail("effective_probability_not_normalized")
    return str(rng.choice(residues, p=weights))


__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "PositionDistributionRuntimeError",
    "RuntimePositionDistribution",
    "RuntimePositionDistributionPolicy",
    "resolve_position_distribution_policy",
    "sample_effective_residue",
]

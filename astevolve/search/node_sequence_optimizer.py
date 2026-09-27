

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable


LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION = (
    "astevolve.node_sequence_optimization_request.v1"
)
PREVIOUS_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION = (
    "astevolve.node_sequence_optimization_request.v2"
)
NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION = (
    "astevolve.node_sequence_optimization_request.v3"
)
NODE_SEQUENCE_OPTIMIZATION_RESULT_VERSION = (
    "astevolve.node_sequence_optimization_result.v1"
)
NODE_SEQUENCE_CANDIDATE_VERSION = "astevolve.node_sequence_candidate.v1"
DEFAULT_NODE_SEQUENCE_OPTIMIZER_ID = "exhaustive_single_diverse_beam_v1"
SOFT_WEIGHT_PRIOR_SCORER_ID = "soft_residue_weight_logit_v1"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_PARENT_SEQUENCE_RE = re.compile(r"[A-Z]+\Z")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "parent_sequences",
        "structural_node_id",
        "mapping_edge_id",
        "action_id",
        "operator",
        "target_chain",
        "legal_positions",
        "allowed_residues",
        "mutation_budget",
        "soft_residue_weights",
        "exact_residue_probabilities",
        "position_distribution_hashes",
        "required_mutations",
        "candidate_count",
        "beam_width",
        "top_k_per_position",
        "temperature",
        "diversity_weight",
        "mutation_penalty",
        "seed",
        "excluded_sequences",
        "request_hash",
    }
)
_PREVIOUS_REQUEST_FIELDS = _REQUEST_FIELDS - {
    "exact_residue_probabilities",
    "position_distribution_hashes",
    "required_mutations",
}
_LEGACY_REQUEST_FIELDS = _PREVIOUS_REQUEST_FIELDS - {"excluded_sequences"}
_ACTION_FIELDS = frozenset(
    {"chain_id", "position", "from_residue", "to_residue", "log_prior"}
)
_CANDIDATE_FIELDS = frozenset(
    {"candidate_id", "actions", "sequence", "log_prior", "provenance"}
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "request_hash",
        "optimizer_id",
        "optimizer_version",
        "candidates",
        "provenance",
        "result_hash",
    }
)


class NodeSequenceOptimizationError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: Any = "") -> None:
    raise NodeSequenceOptimizationError(code, str(detail))


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
    encoded = f"{domain}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


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
    number = float(value)
    if not math.isfinite(number):
        _fail(code, repr(value))
    return number


def _position(value: Any, label: str) -> int:
    if isinstance(value, bool):
        _fail("position_invalid", f"{label}:{value!r}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        _fail("position_invalid", f"{label}:{value!r}")
    if result < 0:
        _fail("position_invalid", f"{label}:{result}")
    return result


def _residue(value: Any, label: str) -> str:
    residue = str(value)
    if len(residue) != 1 or residue not in CANONICAL_AMINO_ACIDS:
        _fail("residue_invalid", f"{label}:{residue}")
    return residue


def _normalize_parent_sequences(value: Any) -> Dict[str, str]:
    raw = _mapping(value, "parent_sequences")
    if not raw:
        _fail("parent_sequences_empty")
    clean: Dict[str, str] = {}
    for raw_chain, raw_sequence in raw.items():
        chain = _text(raw_chain, "chain_id_required")
        if chain in clean:
            _fail("chain_id_duplicate", chain)
        if not isinstance(raw_sequence, str) or not _PARENT_SEQUENCE_RE.fullmatch(
            raw_sequence
        ):
            _fail("parent_sequence_invalid", chain)
        clean[chain] = raw_sequence
    return dict(sorted(clean.items()))


def _unwrap_target_chain_mapping(
    value: Any, *, target_chain: str, label: str
) -> Mapping[Any, Any]:
    raw = _mapping(value, label)


    if target_chain in raw and isinstance(raw[target_chain], Mapping):
        unknown_chains = {str(key) for key in raw} - {target_chain}
        if unknown_chains:
            _fail("target_chain_mapping_ambiguous", label)
        return _mapping(raw[target_chain], f"{label}.{target_chain}")
    return raw


def _normalize_request_payload(
    *,
    parent_sequences: Any,
    structural_node_id: Any,
    mapping_edge_id: Any,
    action_id: Any,
    operator: Any,
    target_chain: Any,
    legal_positions: Any,
    allowed_residues: Any,
    mutation_budget: Any,
    soft_residue_weights: Any,
    exact_residue_probabilities: Any,
    position_distribution_hashes: Any,
    required_mutations: Any,
    candidate_count: Any,
    beam_width: Any,
    top_k_per_position: Any,
    temperature: Any,
    diversity_weight: Any,
    mutation_penalty: Any,
    seed: Any,
    excluded_sequences: Any,
) -> Dict[str, Any]:
    parents = _normalize_parent_sequences(parent_sequences)
    chain = _text(target_chain, "target_chain_required")
    if chain not in parents:
        _fail("target_chain_unknown", chain)

    positions = tuple(
        sorted(
            {
                _position(item, "legal_positions")
                for item in _sequence(legal_positions, "legal_positions")
            }
        )
    )
    if not positions:
        _fail("legal_positions_empty")
    if len(positions) != len(_sequence(legal_positions, "legal_positions")):
        _fail("legal_positions_duplicate")
    for item in positions:
        if item >= len(parents[chain]):
            _fail("legal_position_out_of_range", f"{chain}:{item}")

    raw_allowed = _unwrap_target_chain_mapping(
        allowed_residues, target_chain=chain, label="allowed_residues"
    )
    allowed: Dict[str, list[str]] = {}
    for raw_position, raw_residues in raw_allowed.items():
        position = _position(raw_position, "allowed_residues")
        residues = sorted(
            {
                _residue(item, f"allowed_residues.{position}")
                for item in _sequence(
                    raw_residues, f"allowed_residues.{position}"
                )
            }
        )
        if not residues:
            _fail("allowed_residues_empty", f"{chain}:{position}")
        if not any(residue != parents[chain][position] for residue in residues):
            _fail("position_without_substitution", f"{chain}:{position}")
        allowed[str(position)] = residues
    if set(map(int, allowed)) != set(positions):
        _fail("allowed_positions_mismatch")

    raw_soft = _unwrap_target_chain_mapping(
        soft_residue_weights, target_chain=chain, label="soft_residue_weights"
    )
    soft: Dict[str, Dict[str, float]] = {}
    for raw_position, raw_weights in raw_soft.items():
        position = _position(raw_position, "soft_residue_weights")
        if position not in positions:
            _fail("soft_weight_position_not_allowed", f"{chain}:{position}")
        weights = _mapping(raw_weights, f"soft_residue_weights.{position}")
        clean_weights: Dict[str, float] = {}
        for raw_residue, raw_weight in weights.items():
            residue = _residue(raw_residue, f"soft_residue_weights.{position}")
            if residue not in allowed[str(position)]:
                _fail(
                    "soft_weight_residue_not_allowed",
                    f"{chain}:{position}:{residue}",
                )
            weight = _finite(raw_weight, "soft_weight_invalid")
            if weight < 0.0:
                _fail("soft_weight_invalid", f"{chain}:{position}:{weight}")
            clean_weights[residue] = weight
        soft[str(position)] = dict(sorted(clean_weights.items()))

    raw_exact = _unwrap_target_chain_mapping(
        exact_residue_probabilities,
        target_chain=chain,
        label="exact_residue_probabilities",
    )
    exact: Dict[str, Dict[str, float]] = {}
    for raw_position, raw_probabilities in raw_exact.items():
        position = _position(raw_position, "exact_residue_probabilities")
        if position not in positions:
            _fail("exact_probability_position_not_allowed", f"{chain}:{position}")
        probabilities = _mapping(
            raw_probabilities, f"exact_residue_probabilities.{position}"
        )
        clean_probabilities: Dict[str, float] = {}
        for raw_residue, raw_probability in probabilities.items():
            residue = _residue(
                raw_residue, f"exact_residue_probabilities.{position}"
            )
            if residue not in allowed[str(position)]:
                _fail(
                    "exact_probability_residue_not_allowed",
                    f"{chain}:{position}:{residue}",
                )
            probability = _finite(raw_probability, "exact_probability_invalid")


            if probability <= 0.0:
                _fail(
                    "exact_probability_not_positive",
                    f"{chain}:{position}:{residue}:{probability}",
                )
            clean_probabilities[residue] = probability
        if set(clean_probabilities) != set(allowed[str(position)]):
            _fail("exact_probability_support_mismatch", f"{chain}:{position}")
        total = math.fsum(clean_probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            _fail(
                "exact_probability_not_normalized",
                f"{chain}:{position}:{total}",
            )
        exact[str(position)] = dict(sorted(clean_probabilities.items()))

    raw_hashes = _unwrap_target_chain_mapping(
        position_distribution_hashes,
        target_chain=chain,
        label="position_distribution_hashes",
    )
    distribution_hashes: Dict[str, str] = {}
    for raw_position, raw_hash in raw_hashes.items():
        position = _position(raw_position, "position_distribution_hashes")
        if position not in positions:
            _fail("distribution_hash_position_not_allowed", f"{chain}:{position}")
        distribution_hashes[str(position)] = _text(
            raw_hash, "position_distribution_hash_required"
        )
    if set(map(int, distribution_hashes)) != set(map(int, exact)):
        _fail("position_distribution_hashes_mismatch")

    raw_required = _unwrap_target_chain_mapping(
        required_mutations,
        target_chain=chain,
        label="required_mutations",
    )
    required: Dict[str, str] = {}
    for raw_position, raw_residue in raw_required.items():
        position = _position(raw_position, "required_mutations")
        if position not in positions:
            _fail("required_mutation_position_not_allowed", f"{chain}:{position}")
        residue = _residue(raw_residue, f"required_mutations.{position}")
        if position not in set(map(int, exact)):
            _fail("required_mutation_without_exact_distribution", f"{chain}:{position}")
        if residue not in exact[str(position)]:
            _fail("required_mutation_not_supported", f"{chain}:{position}:{residue}")
        if residue == parents[chain][position]:
            _fail("required_mutation_noop", f"{chain}:{position}:{residue}")


        if set(allowed[str(position)]) != {residue} or set(exact[str(position)]) != {
            residue
        }:
            _fail("required_mutation_not_locked", f"{chain}:{position}:{residue}")
        required[str(position)] = residue

    raw_budget = _mapping(mutation_budget, "mutation_budget")
    _closed(raw_budget, frozenset({"min", "max"}), "mutation_budget")
    minimum = _integer(raw_budget["min"], "mutation_budget_invalid", minimum=0)
    maximum = _integer(raw_budget["max"], "mutation_budget_invalid", minimum=1)
    if minimum > maximum or maximum > len(positions):
        _fail("mutation_budget_invalid", f"{minimum}:{maximum}:{len(positions)}")
    mandatory = {
        position
        for position in positions
        if parents[chain][position] not in allowed[str(position)]
    }
    if len(mandatory) > maximum or len(positions) < minimum:
        _fail("mutation_budget_infeasible")

    normalized_diversity_weight = _finite(
        diversity_weight, "diversity_weight_invalid"
    )
    if normalized_diversity_weight < 0.0:
        _fail("diversity_weight_invalid", normalized_diversity_weight)
    normalized_temperature = _finite(temperature, "temperature_invalid")
    if normalized_temperature <= 0.0:
        _fail("temperature_invalid", normalized_temperature)
    normalized_mutation_penalty = _finite(
        mutation_penalty, "mutation_penalty_invalid"
    )
    if normalized_mutation_penalty < 0.0:
        _fail("mutation_penalty_invalid", normalized_mutation_penalty)
    normalized_top_k = _integer(
        top_k_per_position, "top_k_per_position_invalid", minimum=1
    )
    for raw_position, probabilities in exact.items():
        position = int(raw_position)
        executable_substitutions = sum(
            residue != parents[chain][position] for residue in probabilities
        )
        if executable_substitutions > normalized_top_k:
            _fail(
                "position_distribution_top_k_truncation",
                f"{chain}:{position}:{executable_substitutions}/{normalized_top_k}",
            )

    raw_excluded = _sequence(excluded_sequences, "excluded_sequences")
    normalized_excluded = []
    for index, raw_sequence in enumerate(raw_excluded):
        if not isinstance(raw_sequence, str) or not _PARENT_SEQUENCE_RE.fullmatch(
            raw_sequence
        ):
            _fail("excluded_sequence_invalid", index)
        if len(raw_sequence) != len(parents[chain]):
            _fail("excluded_sequence_length_mismatch", index)
        normalized_excluded.append(raw_sequence)
    if len(set(normalized_excluded)) != len(normalized_excluded):
        _fail("excluded_sequences_duplicate")

    return {
        "parent_sequences": parents,
        "structural_node_id": _text(
            structural_node_id, "structural_node_id_required"
        ),
        "mapping_edge_id": _text(mapping_edge_id, "mapping_edge_id_required"),
        "action_id": _text(action_id, "action_id_required"),
        "operator": _text(operator, "operator_required"),
        "target_chain": chain,
        "legal_positions": list(positions),
        "allowed_residues": dict(
            sorted(allowed.items(), key=lambda item: int(item[0]))
        ),
        "mutation_budget": {"min": minimum, "max": maximum},
        "soft_residue_weights": dict(
            sorted(soft.items(), key=lambda item: int(item[0]))
        ),
        "exact_residue_probabilities": dict(
            sorted(exact.items(), key=lambda item: int(item[0]))
        ),
        "position_distribution_hashes": dict(
            sorted(distribution_hashes.items(), key=lambda item: int(item[0]))
        ),
        "required_mutations": dict(
            sorted(required.items(), key=lambda item: int(item[0]))
        ),
        "candidate_count": _integer(
            candidate_count, "candidate_count_invalid", minimum=1
        ),
        "beam_width": _integer(beam_width, "beam_width_invalid", minimum=1),
        "top_k_per_position": normalized_top_k,
        "temperature": normalized_temperature,
        "diversity_weight": normalized_diversity_weight,
        "mutation_penalty": normalized_mutation_penalty,
        "seed": _integer(seed, "seed_invalid"),
        "excluded_sequences": sorted(normalized_excluded),
    }


@dataclass(frozen=True)
class NodeSequenceOptimizationRequest:


    _payload_json: str
    request_hash: str
    schema_version: str = NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION

    @classmethod
    def create(
        cls,
        *,
        parent_sequences: Mapping[str, str],
        structural_node_id: str,
        mapping_edge_id: str,
        action_id: str,
        target_chain: str,
        legal_positions: Sequence[int],
        allowed_residues: Mapping[Any, Any],
        mutation_budget: Mapping[str, int],
        soft_residue_weights: Optional[Mapping[Any, Any]] = None,
        exact_residue_probabilities: Optional[Mapping[Any, Any]] = None,
        position_distribution_hashes: Optional[Mapping[Any, Any]] = None,
        required_mutations: Optional[Mapping[Any, Any]] = None,
        candidate_count: int = 8,
        beam_width: int = 16,
        top_k_per_position: int = 4,
        temperature: float = 0.8,
        diversity_weight: float = 0.15,
        mutation_penalty: float = 0.25,
        seed: int = 0,
        operator: str = "substitution",
        excluded_sequences: Sequence[str] = (),
    ) -> "NodeSequenceOptimizationRequest":
        payload = _normalize_request_payload(
            parent_sequences=parent_sequences,
            structural_node_id=structural_node_id,
            mapping_edge_id=mapping_edge_id,
            action_id=action_id,
            operator=operator,
            target_chain=target_chain,
            legal_positions=legal_positions,
            allowed_residues=allowed_residues,
            mutation_budget=mutation_budget,
            soft_residue_weights=(
                {} if soft_residue_weights is None else soft_residue_weights
            ),
            exact_residue_probabilities=(
                {}
                if exact_residue_probabilities is None
                else exact_residue_probabilities
            ),
            position_distribution_hashes=(
                {}
                if position_distribution_hashes is None
                else position_distribution_hashes
            ),
            required_mutations=(
                {} if required_mutations is None else required_mutations
            ),
            candidate_count=candidate_count,
            beam_width=beam_width,
            top_k_per_position=top_k_per_position,
            temperature=temperature,
            diversity_weight=diversity_weight,
            mutation_penalty=mutation_penalty,
            seed=seed,
            excluded_sequences=excluded_sequences,
        )
        return cls(
            _payload_json=_canonical_json(payload),
            request_hash=_digest(NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION, payload),
        )

    @cached_property
    def _payload(self) -> Mapping[str, Any]:
        return _freeze_json(json.loads(self._payload_json))

    @property
    def parent_sequences(self) -> Dict[str, str]:
        return dict(self._payload["parent_sequences"])

    @property
    def structural_node_id(self) -> str:
        return self._payload["structural_node_id"]

    @property
    def mapping_edge_id(self) -> str:
        return self._payload["mapping_edge_id"]

    @property
    def action_id(self) -> str:
        return self._payload["action_id"]

    @property
    def operator(self) -> str:
        return self._payload["operator"]

    @property
    def target_chain(self) -> str:
        return self._payload["target_chain"]

    @property
    def legal_positions(self) -> Tuple[int, ...]:
        return tuple(self._payload["legal_positions"])

    @property
    def allowed_residues(self) -> Dict[int, Tuple[str, ...]]:
        return {
            int(position): tuple(residues)
            for position, residues in self._payload["allowed_residues"].items()
        }

    @property
    def mutation_budget(self) -> Dict[str, int]:
        return dict(self._payload["mutation_budget"])

    @property
    def soft_residue_weights(self) -> Dict[int, Dict[str, float]]:
        return {
            int(position): dict(weights)
            for position, weights in self._payload["soft_residue_weights"].items()
        }

    @property
    def exact_residue_probabilities(self) -> Dict[int, Dict[str, float]]:
        return {
            int(position): dict(probabilities)
            for position, probabilities in self._payload.get(
                "exact_residue_probabilities", {}
            ).items()
        }

    @property
    def position_distribution_hashes(self) -> Dict[int, str]:
        return {
            int(position): str(distribution_hash)
            for position, distribution_hash in self._payload.get(
                "position_distribution_hashes", {}
            ).items()
        }

    @property
    def required_mutations(self) -> Dict[int, str]:
        return {
            int(position): str(residue)
            for position, residue in self._payload.get("required_mutations", {}).items()
        }

    @property
    def candidate_count(self) -> int:
        return self._payload["candidate_count"]

    @property
    def beam_width(self) -> int:
        return self._payload["beam_width"]

    @property
    def top_k_per_position(self) -> int:
        return self._payload["top_k_per_position"]

    @property
    def temperature(self) -> float:
        return self._payload["temperature"]

    @property
    def diversity_weight(self) -> float:
        return self._payload["diversity_weight"]

    @property
    def mutation_penalty(self) -> float:
        return self._payload["mutation_penalty"]

    @property
    def seed(self) -> int:
        return self._payload["seed"]

    @property
    def excluded_sequences(self) -> Tuple[str, ...]:
        return tuple(self._payload.get("excluded_sequences", ()))

    @cached_property
    def ranking_hash(self) -> str:


        payload = json.loads(self._payload_json)
        payload.pop("excluded_sequences", None)
        return _digest("astevolve.node_sequence_optimizer.ranking.v1", payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = json.loads(self._payload_json)
        return {
            "schema_version": self.schema_version,
            **payload,
            "request_hash": self.request_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "NodeSequenceOptimizationRequest":
        raw = _mapping(value, "node_sequence_optimization_request")
        schema_version = raw.get("schema_version")
        if schema_version == NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:
            _closed(raw, _REQUEST_FIELDS, "node_sequence_optimization_request")
        elif schema_version == PREVIOUS_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:
            _closed(
                raw,
                _PREVIOUS_REQUEST_FIELDS,
                "node_sequence_optimization_request",
            )
        elif schema_version == LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:


            observed = {str(key) for key in raw}
            hybrid_fields = _REQUEST_FIELDS - {"excluded_sequences"}
            _closed(
                raw,
                hybrid_fields if observed == hybrid_fields else _LEGACY_REQUEST_FIELDS,
                "node_sequence_optimization_request",
            )
        else:
            _fail("schema_version_invalid", schema_version)
        request = cls.create(
            parent_sequences=raw["parent_sequences"],
            structural_node_id=raw["structural_node_id"],
            mapping_edge_id=raw["mapping_edge_id"],
            action_id=raw["action_id"],
            operator=raw["operator"],
            target_chain=raw["target_chain"],
            legal_positions=raw["legal_positions"],
            allowed_residues=raw["allowed_residues"],
            mutation_budget=raw["mutation_budget"],
            soft_residue_weights=raw["soft_residue_weights"],
            exact_residue_probabilities=raw.get("exact_residue_probabilities", {}),
            position_distribution_hashes=raw.get("position_distribution_hashes", {}),
            required_mutations=raw.get("required_mutations", {}),
            candidate_count=raw["candidate_count"],
            beam_width=raw["beam_width"],
            top_k_per_position=raw["top_k_per_position"],
            temperature=raw["temperature"],
            diversity_weight=raw["diversity_weight"],
            mutation_penalty=raw["mutation_penalty"],
            seed=raw["seed"],
            excluded_sequences=raw.get("excluded_sequences", ()),
        )
        if schema_version in {
            LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION,
            PREVIOUS_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION,
        }:
            legacy_payload = json.loads(request._payload_json)
            preserve_step2_fields = (
                schema_version == LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION
                and "exact_residue_probabilities" in raw
            )
            if not preserve_step2_fields:
                for field in (
                    "exact_residue_probabilities",
                    "position_distribution_hashes",
                    "required_mutations",
                ):
                    legacy_payload.pop(field, None)
            if schema_version == LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:
                legacy_payload.pop("excluded_sequences", None)
            expected_hash = _digest(schema_version, legacy_payload)
            if raw.get("request_hash") != expected_hash:
                _fail("hash_mismatch", "node_sequence_optimization_request")
            return cls(
                _payload_json=_canonical_json(legacy_payload),
                request_hash=expected_hash,
                schema_version=schema_version,
            )
        if raw.get("request_hash") != request.request_hash:
            _fail("hash_mismatch", "node_sequence_optimization_request")
        return request


@dataclass(frozen=True)
class NodeSubstitutionAction:


    chain_id: str
    position: int
    from_residue: str
    to_residue: str
    log_prior: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "position": self.position,
            "from_residue": self.from_residue,
            "to_residue": self.to_residue,
            "log_prior": self.log_prior,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NodeSubstitutionAction":
        raw = _mapping(value, "node_substitution_action")
        _closed(raw, _ACTION_FIELDS, "node_substitution_action")
        source = _residue(raw["from_residue"], "from_residue")
        target = _residue(raw["to_residue"], "to_residue")
        if source == target:
            _fail("substitution_noop")
        return cls(
            chain_id=_text(raw["chain_id"], "chain_id_required"),
            position=_position(raw["position"], "substitution_action"),
            from_residue=source,
            to_residue=target,
            log_prior=_finite(raw["log_prior"], "log_prior_invalid"),
        )


@dataclass(frozen=True)
class NodeSequenceCandidate:


    candidate_id: str
    actions: Tuple[NodeSubstitutionAction, ...]
    sequence: str
    log_prior: float
    _provenance_json: str

    @property
    def provenance(self) -> Dict[str, Any]:
        return json.loads(self._provenance_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "actions": [action.to_dict() for action in self.actions],
            "sequence": self.sequence,
            "log_prior": self.log_prior,
            "provenance": self.provenance,
        }

    @classmethod
    def create(
        cls,
        *,
        request: NodeSequenceOptimizationRequest,
        actions: Sequence[NodeSubstitutionAction],
        sequence: str,
        log_prior: float,
        provenance: Mapping[str, Any],
    ) -> "NodeSequenceCandidate":
        canonical_actions = tuple(
            sorted(actions, key=lambda action: (action.position, action.to_residue))
        )
        payload = {
            "actions": [action.to_dict() for action in canonical_actions],
            "sequence": str(sequence),
            "log_prior": _finite(log_prior, "log_prior_invalid"),
            "provenance": dict(provenance),
        }
        return cls(
            candidate_id=_digest(
                NODE_SEQUENCE_CANDIDATE_VERSION,
                {"request_hash": request.request_hash, **payload},
            ),
            actions=canonical_actions,
            sequence=str(sequence),
            log_prior=payload["log_prior"],
            _provenance_json=_canonical_json(payload["provenance"]),
        )

    @classmethod
    def from_mapping(
        cls,
        request: NodeSequenceOptimizationRequest,
        value: Mapping[str, Any],
    ) -> "NodeSequenceCandidate":
        raw = _mapping(value, "node_sequence_candidate")
        _closed(raw, _CANDIDATE_FIELDS, "node_sequence_candidate")
        candidate = cls.create(
            request=request,
            actions=[
                NodeSubstitutionAction.from_mapping(item)
                for item in _sequence(raw["actions"], "candidate.actions")
            ],
            sequence=_text(raw["sequence"], "candidate_sequence_required"),
            log_prior=raw["log_prior"],
            provenance=_mapping(raw["provenance"], "candidate.provenance"),
        )
        if raw.get("candidate_id") != candidate.candidate_id:
            _fail("hash_mismatch", "node_sequence_candidate")
        return candidate


@dataclass(frozen=True)
class NodeSequenceOptimizationResult:


    request_hash: str
    optimizer_id: str
    optimizer_version: str
    candidates: Tuple[NodeSequenceCandidate, ...]
    _provenance_json: str
    result_hash: str
    schema_version: str = NODE_SEQUENCE_OPTIMIZATION_RESULT_VERSION

    @property
    def provenance(self) -> Dict[str, Any]:
        return json.loads(self._provenance_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_hash": self.request_hash,
            "optimizer_id": self.optimizer_id,
            "optimizer_version": self.optimizer_version,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": self.provenance,
            "result_hash": self.result_hash,
        }

    @classmethod
    def create(
        cls,
        *,
        request: NodeSequenceOptimizationRequest,
        optimizer_id: str,
        optimizer_version: str,
        candidates: Sequence[NodeSequenceCandidate],
        provenance: Mapping[str, Any],
    ) -> "NodeSequenceOptimizationResult":
        payload = {
            "request_hash": request.request_hash,
            "optimizer_id": _text(optimizer_id, "optimizer_id_required"),
            "optimizer_version": _text(
                optimizer_version, "optimizer_version_required"
            ),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "provenance": dict(provenance),
        }
        return cls(
            request_hash=request.request_hash,
            optimizer_id=payload["optimizer_id"],
            optimizer_version=payload["optimizer_version"],
            candidates=tuple(candidates),
            _provenance_json=_canonical_json(payload["provenance"]),
            result_hash=_digest(NODE_SEQUENCE_OPTIMIZATION_RESULT_VERSION, payload),
        )

    @classmethod
    def from_mapping(
        cls,
        request: NodeSequenceOptimizationRequest,
        value: Mapping[str, Any],
    ) -> "NodeSequenceOptimizationResult":
        raw = _mapping(value, "node_sequence_optimization_result")
        _closed(raw, _RESULT_FIELDS, "node_sequence_optimization_result")
        if raw.get("schema_version") != NODE_SEQUENCE_OPTIMIZATION_RESULT_VERSION:
            _fail("schema_version_invalid", raw.get("schema_version"))
        if raw.get("request_hash") != request.request_hash:
            _fail("request_hash_mismatch")
        result = cls.create(
            request=request,
            optimizer_id=raw["optimizer_id"],
            optimizer_version=raw["optimizer_version"],
            candidates=[
                NodeSequenceCandidate.from_mapping(request, item)
                for item in _sequence(raw["candidates"], "result.candidates")
            ],
            provenance=_mapping(raw["provenance"], "result.provenance"),
        )
        if raw.get("result_hash") != result.result_hash:
            _fail("hash_mismatch", "node_sequence_optimization_result")
        return result


@runtime_checkable
class ResiduePriorScorer(Protocol):


    scorer_id: str
    scorer_version: str

    def score(
        self,
        request: NodeSequenceOptimizationRequest,
        position: int,
        residue: str,
    ) -> float:
        pass


class SoftWeightResiduePriorScorer:


    scorer_id = SOFT_WEIGHT_PRIOR_SCORER_ID
    scorer_version = "1"

    def score(
        self,
        request: NodeSequenceOptimizationRequest,
        position: int,
        residue: str,
    ) -> float:
        exact = request.exact_residue_probabilities.get(position)
        if exact is not None:
            if residue not in exact:
                _fail(
                    "exact_probability_residue_not_executable",
                    f"{request.target_chain}:{position}:{residue}",
                )


            return request.temperature * math.log(float(exact[residue]))
        weight = request.soft_residue_weights.get(position, {}).get(residue, 1.0)


        return math.log(max(float(weight), 1e-300))


@runtime_checkable
class NodeSequenceOptimizer(Protocol):


    optimizer_id: str
    optimizer_version: str

    def optimize(
        self, request: NodeSequenceOptimizationRequest
    ) -> NodeSequenceOptimizationResult:
        pass


@dataclass(frozen=True)
class _BeamState:
    actions: Tuple[NodeSubstitutionAction, ...]
    log_prior: float

    @property
    def signature(self) -> Tuple[Tuple[int, str], ...]:
        return tuple((action.position, action.to_residue) for action in self.actions)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _tie_key(request: NodeSequenceOptimizationRequest, signature: Any) -> str:
    return _digest(
        "astevolve.node_sequence_optimizer.seeded_tie.v1",
        {


            "ranking_hash": request.ranking_hash,
            "seed": request.seed,
            "signature": signature,
        },
    )


def _state_distance(left: _BeamState, right: _BeamState) -> float:
    left_set = set(left.signature)
    right_set = set(right.signature)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - (len(left_set & right_set) / len(union))


def _select_diverse_states(
    request: NodeSequenceOptimizationRequest,
    states: Sequence[_BeamState],
    count: int,
    *,
    initial: Sequence[_BeamState] = (),
) -> list[_BeamState]:
    remaining = {state.signature: state for state in states}
    selected: list[_BeamState] = list(initial)
    added: list[_BeamState] = []
    while remaining and len(added) < count:
        def rank(state: _BeamState) -> Tuple[float, float, str]:
            diversity = (
                min(_state_distance(state, prior) for prior in selected)
                if selected
                else 0.0
            )
            utility = state.log_prior + request.diversity_weight * diversity
            return (-utility, -state.log_prior, _tie_key(request, state.signature))

        chosen = min(remaining.values(), key=rank)
        remaining.pop(chosen.signature)
        selected.append(chosen)
        added.append(chosen)
    return added


def _candidate_from_state(
    request: NodeSequenceOptimizationRequest,
    state: _BeamState,
    *,
    source: str,
    scorer_id: str,
    scorer_version: str,
) -> NodeSequenceCandidate:
    sequence = list(request.parent_sequences[request.target_chain])
    for action in state.actions:
        sequence[action.position] = action.to_residue
    return NodeSequenceCandidate.create(
        request=request,
        actions=state.actions,
        sequence="".join(sequence),
        log_prior=state.log_prior,
        provenance={
            "algorithm": source,
            "deterministic": True,
            "seed": request.seed,
            "request_hash": request.request_hash,
            "ranking_hash": request.ranking_hash,
            "structural_node_id": request.structural_node_id,
            "mapping_edge_id": request.mapping_edge_id,
            "action_id": request.action_id,
            "operator": request.operator,
            "depth": len(state.actions),
            "prior_scorer_id": scorer_id,
            "prior_scorer_version": scorer_version,
            "temperature": request.temperature,
            "mutation_penalty": request.mutation_penalty,
            "position_distribution_hashes": {
                str(position): distribution_hash
                for position, distribution_hash in sorted(
                    request.position_distribution_hashes.items()
                )
            },
            "required_mutations": {
                str(position): residue
                for position, residue in sorted(request.required_mutations.items())
            },
        },
    )


def _validate_candidate(
    request: NodeSequenceOptimizationRequest,
    candidate: NodeSequenceCandidate,
) -> None:
    if not isinstance(candidate, NodeSequenceCandidate):
        _fail("candidate_type_invalid")
    parent = request.parent_sequences[request.target_chain]
    if len(candidate.sequence) != len(parent) or not _PARENT_SEQUENCE_RE.fullmatch(
        candidate.sequence
    ):
        _fail("candidate_sequence_invalid", candidate.candidate_id)
    budget = request.mutation_budget
    if not budget["min"] <= len(candidate.actions) <= budget["max"]:
        _fail("candidate_budget_violation", candidate.candidate_id)
    positions = [action.position for action in candidate.actions]
    if positions != sorted(set(positions)):
        _fail("candidate_action_positions_invalid", candidate.candidate_id)
    legal = set(request.legal_positions)
    allowed = request.allowed_residues
    generated = list(parent)
    for action in candidate.actions:
        if action.chain_id != request.target_chain:
            _fail("candidate_chain_violation", candidate.candidate_id)
        if action.position not in legal:
            _fail("candidate_position_violation", candidate.candidate_id)
        if action.from_residue != parent[action.position]:
            _fail("candidate_parent_mismatch", candidate.candidate_id)
        if (
            action.to_residue not in allowed[action.position]
            or action.to_residue == action.from_residue
        ):
            _fail("candidate_residue_violation", candidate.candidate_id)
        generated[action.position] = action.to_residue
    mandatory = {
        position
        for position in request.legal_positions
        if parent[position] not in allowed[position]
    }
    if not mandatory.issubset(positions):
        _fail("candidate_mandatory_position_missing", candidate.candidate_id)
    for position, residue in request.required_mutations.items():
        if candidate.sequence[position] != residue:
            _fail("candidate_required_mutation_missing", candidate.candidate_id)
    if "".join(generated) != candidate.sequence:
        _fail("candidate_sequence_action_mismatch", candidate.candidate_id)
    if candidate.sequence in set(request.excluded_sequences):
        _fail("candidate_sequence_excluded", candidate.candidate_id)
    if not math.isclose(
        candidate.log_prior,
        sum(action.log_prior for action in candidate.actions),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail("candidate_log_prior_mismatch", candidate.candidate_id)
    restored = NodeSequenceCandidate.from_mapping(request, candidate.to_dict())
    if restored.candidate_id != candidate.candidate_id:
        _fail("candidate_hash_mismatch", candidate.candidate_id)


def validate_node_sequence_optimization_result(
    request: NodeSequenceOptimizationRequest,
    result: NodeSequenceOptimizationResult,
) -> NodeSequenceOptimizationResult:


    if not isinstance(request, NodeSequenceOptimizationRequest):
        _fail("request_type_invalid")
    if not isinstance(result, NodeSequenceOptimizationResult):
        _fail("result_type_invalid")
    NodeSequenceOptimizationRequest.from_mapping(request.to_dict())
    NodeSequenceOptimizationResult.from_mapping(request, result.to_dict())
    if result.request_hash != request.request_hash:
        _fail("request_hash_mismatch")
    if not result.candidates:
        _fail("candidates_empty")
    if len(result.candidates) > request.candidate_count:
        _fail("candidate_count_exceeded")
    candidate_ids: set[str] = set()
    sequences: set[str] = set()
    for candidate in result.candidates:
        _validate_candidate(request, candidate)
        if candidate.candidate_id in candidate_ids or candidate.sequence in sequences:
            _fail("candidate_duplicate")
        candidate_ids.add(candidate.candidate_id)
        sequences.add(candidate.sequence)
        provenance = candidate.provenance
        expected = {
            "seed": request.seed,
            "request_hash": request.request_hash,
            "structural_node_id": request.structural_node_id,
            "mapping_edge_id": request.mapping_edge_id,
            "action_id": request.action_id,
            "operator": request.operator,
        }
        if (
            request.schema_version
            != LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION
        ):
            expected["ranking_hash"] = request.ranking_hash
        for key, value in expected.items():
            if provenance.get(key) != value:
                _fail("candidate_provenance_mismatch", key)
        if request.schema_version == NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:
            expected_distributions = {
                str(position): distribution_hash
                for position, distribution_hash in sorted(
                    request.position_distribution_hashes.items()
                )
            }
            expected_required = {
                str(position): residue
                for position, residue in sorted(request.required_mutations.items())
            }
            if provenance.get("position_distribution_hashes") != expected_distributions:
                _fail("candidate_provenance_mismatch", "position_distribution_hashes")
            if provenance.get("required_mutations") != expected_required:
                _fail("candidate_provenance_mismatch", "required_mutations")
    provenance = result.provenance
    expected_result = {
        "deterministic": True,
        "seed": request.seed,
        "candidate_count_requested": request.candidate_count,
        "candidate_count_returned": len(result.candidates),
        "beam_width": request.beam_width,
        "top_k_per_position": request.top_k_per_position,
        "temperature": request.temperature,
        "diversity_weight": request.diversity_weight,
        "mutation_penalty": request.mutation_penalty,
    }
    if (
        request.schema_version
        != LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION
    ):
        expected_result.update(
            {
                "ranking_hash": request.ranking_hash,
                "excluded_sequence_count": len(request.excluded_sequences),
            }
        )
    if request.schema_version == NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION:
        expected_result.update(
            {
                "position_distribution_hashes": {
                    str(position): distribution_hash
                    for position, distribution_hash in sorted(
                        request.position_distribution_hashes.items()
                    )
                },
                "required_mutations": {
                    str(position): residue
                    for position, residue in sorted(
                        request.required_mutations.items()
                    )
                },
            }
        )
    for key, value in expected_result.items():
        if provenance.get(key) != value:
            _fail("result_provenance_mismatch", key)
    return result


class DefaultNodeSequenceOptimizer:


    optimizer_id = DEFAULT_NODE_SEQUENCE_OPTIMIZER_ID
    optimizer_version = "1"

    def __init__(
        self, residue_prior_scorer: Optional[ResiduePriorScorer] = None
    ) -> None:
        scorer = residue_prior_scorer or SoftWeightResiduePriorScorer()
        self._scorer = scorer
        self._scorer_id = _text(
            getattr(scorer, "scorer_id", None), "prior_scorer_id_required"
        )
        self._scorer_version = _text(
            getattr(scorer, "scorer_version", None),
            "prior_scorer_version_required",
        )
        if not callable(getattr(scorer, "score", None)):
            _fail("prior_scorer_protocol_invalid", self._scorer_id)

    def _atomic_actions(
        self, request: NodeSequenceOptimizationRequest
    ) -> Dict[int, Tuple[NodeSubstitutionAction, ...]]:
        parent = request.parent_sequences[request.target_chain]
        by_position: Dict[int, Tuple[NodeSubstitutionAction, ...]] = {}
        for position in request.legal_positions:
            residues = request.allowed_residues[position]
            exact = request.exact_residue_probabilities.get(position)
            if exact is not None:


                logits = [
                    request.temperature * math.log(float(exact[residue]))
                    for residue in residues
                ]
            else:
                logits = [
                    _finite(
                        self._scorer.score(request, position, residue),
                        "prior_scorer_value_invalid",
                    )
                    for residue in residues
                ]
            logits = [logit / request.temperature for logit in logits]
            normalizer = _logsumexp(logits)
            actions = []
            for residue, logit in zip(residues, logits):
                if residue == parent[position]:
                    continue
                actions.append(
                    NodeSubstitutionAction(
                        chain_id=request.target_chain,
                        position=position,
                        from_residue=parent[position],
                        to_residue=residue,
                        log_prior=(
                            logit - normalizer - request.mutation_penalty
                        ),
                    )
                )
            actions.sort(
                key=lambda action: (
                    -action.log_prior,
                    _tie_key(
                        request,
                        ((action.position, action.to_residue),),
                    ),
                )
            )
            by_position[position] = tuple(actions[: request.top_k_per_position])
        return by_position

    def optimize(
        self, request: NodeSequenceOptimizationRequest
    ) -> NodeSequenceOptimizationResult:
        if not isinstance(request, NodeSequenceOptimizationRequest):
            _fail("request_type_invalid")


        request = NodeSequenceOptimizationRequest.from_mapping(request.to_dict())
        actions_by_position = self._atomic_actions(request)
        budget = request.mutation_budget
        parent = request.parent_sequences[request.target_chain]
        mandatory = {
            position
            for position in request.legal_positions
            if parent[position] not in request.allowed_residues[position]
        }

        def feasible(state: _BeamState) -> bool:
            positions = {action.position for action in state.actions}
            return (
                budget["min"] <= len(state.actions) <= budget["max"]
                and mandatory.issubset(positions)
            )

        def viable(state: _BeamState) -> bool:
            positions = {action.position for action in state.actions}
            missing_mandatory = len(mandatory - positions)
            remaining_slots = budget["max"] - len(state.actions)
            return missing_mandatory <= remaining_slots

        excluded_sequences = set(request.excluded_sequences)

        def available(state: _BeamState) -> bool:
            sequence = list(parent)
            for action in state.actions:
                sequence[action.position] = action.to_residue
            return "".join(sequence) not in excluded_sequences

        single_states = [
            _BeamState(actions=(action,), log_prior=action.log_prior)
            for position in request.legal_positions
            for action in actions_by_position[position]
        ]
        feasible_singles = [state for state in single_states if feasible(state)]

        beam = _select_diverse_states(
            request,
            [state for state in single_states if viable(state)],
            request.beam_width,
        )
        multi_states: Dict[Tuple[Tuple[int, str], ...], _BeamState] = {}
        for depth in range(2, budget["max"] + 1):
            expanded: Dict[Tuple[Tuple[int, str], ...], _BeamState] = {}
            for state in beam:
                occupied = {action.position for action in state.actions}
                for position in request.legal_positions:
                    if position in occupied:
                        continue
                    for action in actions_by_position[position]:
                        actions = tuple(
                            sorted(
                                (*state.actions, action),
                                key=lambda item: (item.position, item.to_residue),
                            )
                        )
                        candidate = _BeamState(
                            actions=actions,
                            log_prior=sum(item.log_prior for item in actions),
                        )
                        if viable(candidate):
                            expanded[candidate.signature] = candidate
            if not expanded:
                break
            for signature, state in expanded.items():
                if feasible(state):
                    multi_states[signature] = state
            beam = _select_diverse_states(
                request,
                list(expanded.values()),
                request.beam_width,
            )
            if depth >= budget["max"]:
                break

        requested = request.candidate_count
        available_singles = [state for state in feasible_singles if available(state)]
        multi_pool = [state for state in multi_states.values() if available(state)]


        reserve_multi = bool(available_singles and multi_pool and requested >= 2)
        single_limit = requested - 1 if reserve_multi else requested
        selected_singles = _select_diverse_states(
            request,
            available_singles,
            min(single_limit, len(available_singles)),
        )
        remaining = requested - len(selected_singles)
        selected_multi = _select_diverse_states(
            request,
            multi_pool,
            remaining,
            initial=selected_singles,
        )
        selected = [*selected_singles, *selected_multi]
        if len(selected) < requested:
            selected_signatures = {state.signature for state in selected}
            unselected_singles = [
                state
                for state in available_singles
                if state.signature not in selected_signatures
            ]
            selected.extend(
                _select_diverse_states(
                    request,
                    unselected_singles,
                    requested - len(selected),
                    initial=selected,
                )
            )
        if not selected:
            if excluded_sequences and (feasible_singles or multi_states):
                _fail("candidate_space_exhausted")
            _fail("candidate_space_empty")

        single_signatures = {state.signature for state in feasible_singles}
        candidates = [
            _candidate_from_state(
                request,
                state,
                source=(
                    "exhaustive_single_substitution_v1"
                    if state.signature in single_signatures
                    else "diversity_aware_beam_v1"
                ),
                scorer_id=self._scorer_id,
                scorer_version=self._scorer_version,
            )
            for state in selected
        ]
        result = NodeSequenceOptimizationResult.create(
            request=request,
            optimizer_id=self.optimizer_id,
            optimizer_version=self.optimizer_version,
            candidates=candidates,
            provenance={
                "algorithm": "exhaustive_single_then_diversity_aware_beam_v1",
                "deterministic": True,
                "seed": request.seed,
                "ranking_hash": request.ranking_hash,
                "excluded_sequence_count": len(request.excluded_sequences),
                "prior_scorer_id": self._scorer_id,
                "prior_scorer_version": self._scorer_version,
                "exhaustive_single_count": len(single_states),
                "feasible_single_count": len(feasible_singles),
                "available_single_count": len(available_singles),
                "beam_candidate_count": len(multi_states),
                "available_beam_candidate_count": len(multi_pool),
                "candidate_count_requested": requested,
                "candidate_count_returned": len(candidates),
                "beam_width": request.beam_width,
                "top_k_per_position": request.top_k_per_position,
                "temperature": request.temperature,
                "diversity_weight": request.diversity_weight,
                "mutation_penalty": request.mutation_penalty,
                "position_distribution_hashes": {
                    str(position): distribution_hash
                    for position, distribution_hash in sorted(
                        request.position_distribution_hashes.items()
                    )
                },
                "required_mutations": {
                    str(position): residue
                    for position, residue in sorted(
                        request.required_mutations.items()
                    )
                },
            },
        )
        return validate_node_sequence_optimization_result(request, result)


__all__ = [
    "DEFAULT_NODE_SEQUENCE_OPTIMIZER_ID",
    "LEGACY_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION",
    "PREVIOUS_NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION",
    "NODE_SEQUENCE_CANDIDATE_VERSION",
    "NODE_SEQUENCE_OPTIMIZATION_REQUEST_VERSION",
    "NODE_SEQUENCE_OPTIMIZATION_RESULT_VERSION",
    "SOFT_WEIGHT_PRIOR_SCORER_ID",
    "DefaultNodeSequenceOptimizer",
    "NodeSequenceCandidate",
    "NodeSequenceOptimizationError",
    "NodeSequenceOptimizationRequest",
    "NodeSequenceOptimizationResult",
    "NodeSequenceOptimizer",
    "NodeSubstitutionAction",
    "ResiduePriorScorer",
    "SoftWeightResiduePriorScorer",
    "validate_node_sequence_optimization_result",
]

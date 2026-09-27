

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Optional

from .causal_flow import (
    CausalFlowContractError,
    EffectiveSearchContract,
    SequenceRecord,
    canonical_json,
)


CODE_IDENTITY_VERSION = "astevolve.code_identity.v1"
EFFECTIVE_CONTRACT_IDENTITY_VERSION = (
    "astevolve.effective_contract_identity.v1"
)
SEQUENCE_BUNDLE_IDENTITY_VERSION = "astevolve.sequence_bundle_identity.v1"
EVALUATOR_DESCRIPTOR_VERSION = "astevolve.evaluator_descriptor.v1"
EXACT_EVALUATION_KEY_VERSION = "astevolve.exact_evaluation_key.v1"

_RAW_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_CODE_HASH_RE = re.compile(r"code_sha256:[0-9a-f]{64}\Z")
_SEQUENCE_HASH_RE = re.compile(r"sequence_sha256:[0-9a-f]{64}\Z")
_DESCRIPTOR_HASH_RE = re.compile(r"evaluator_descriptor_sha256:[0-9a-f]{64}\Z")
_CACHE_KEY_RE = re.compile(r"evaluation_cache_sha256:[0-9a-f]{64}\Z")


class ExperimentIdentityError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise ExperimentIdentityError(code, detail)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _closed(value: Mapping[str, Any], fields: set[str]) -> None:
    unknown = sorted(str(key) for key in set(value) - fields)
    missing = sorted(str(key) for key in fields - set(value))
    if unknown:
        _fail("unknown_fields", ",".join(unknown))
    if missing:
        _fail("fields_missing", ",".join(missing))


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (CausalFlowContractError, TypeError, ValueError) as exc:
        _fail("not_canonical_json", f"{label}:{exc}")


def _digest(domain: str, payload: Any) -> str:
    encoded = f"{domain}\0{canonical_json(payload)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("seed_invalid", repr(value))
    return value


def _validate_raw_hash(value: Any, code: str) -> str:
    text = _required_text(value, code)
    if _RAW_HEX_RE.fullmatch(text) is None:
        _fail(code, text)
    return text


def _validate_sequence_hash(value: Any) -> str:
    text = _required_text(value, "sequence_bundle_hash_required")
    if _SEQUENCE_HASH_RE.fullmatch(text) is None:
        _fail("sequence_bundle_hash_invalid", text)
    return text


@dataclass(frozen=True)
class CodeIdentity:


    byte_length: int
    code_hash: str
    identity_hash: str
    schema_version: str = CODE_IDENTITY_VERSION

    @classmethod
    def create(cls, source: bytes | bytearray | memoryview) -> "CodeIdentity":
        if not isinstance(source, (bytes, bytearray, memoryview)):
            _fail("source_bytes_required", type(source).__name__)
        raw = bytes(source)
        code_hash = f"code_sha256:{hashlib.sha256(raw).hexdigest()}"
        semantic = {"byte_length": len(raw), "code_hash": code_hash}
        return cls(
            byte_length=len(raw),
            code_hash=code_hash,
            identity_hash=f"code_identity_sha256:{_digest('code_identity.v1', semantic)}",
        )

    @classmethod
    def from_text(cls, source: str, *, encoding: str = "utf-8") -> "CodeIdentity":
        if not isinstance(source, str):
            _fail("source_text_required", type(source).__name__)
        return cls.create(source.encode(encoding))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "byte_length": self.byte_length,
            "code_hash": self.code_hash,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodeIdentity":
        artifact = _mapping(value, "code_identity")
        _closed(
            artifact,
            {"schema_version", "byte_length", "code_hash", "identity_hash"},
        )
        if artifact.get("schema_version") != CODE_IDENTITY_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        length = artifact.get("byte_length")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            _fail("byte_length_invalid", repr(length))
        code_hash = _required_text(artifact.get("code_hash"), "code_hash_required")
        if _CODE_HASH_RE.fullmatch(code_hash) is None:
            _fail("code_hash_invalid", code_hash)
        semantic = {"byte_length": length, "code_hash": code_hash}
        expected = f"code_identity_sha256:{_digest('code_identity.v1', semantic)}"
        if artifact.get("identity_hash") != expected:
            _fail("hash_mismatch", "code_identity")
        return cls(length, code_hash, expected)


@dataclass(frozen=True)
class EffectiveContractIdentity:


    _effective_contract_json: str
    effective_contract_hash: str
    identity_hash: str
    schema_version: str = EFFECTIVE_CONTRACT_IDENTITY_VERSION

    @property
    def effective_contract(self) -> dict[str, Any]:
        return json.loads(self._effective_contract_json)

    @classmethod
    def create(
        cls, value: EffectiveSearchContract | Mapping[str, Any]
    ) -> "EffectiveContractIdentity":
        try:
            if isinstance(value, EffectiveSearchContract):
                parsed = EffectiveSearchContract.from_mapping(value.to_dict())
            elif isinstance(value, Mapping):
                parsed = EffectiveSearchContract.from_mapping(value)
            else:
                _fail("effective_contract_invalid", type(value).__name__)
        except ExperimentIdentityError:
            raise
        except (CausalFlowContractError, TypeError, ValueError) as exc:
            _fail("effective_contract_invalid", str(exc))


        normalized = EffectiveSearchContract.create(**parsed.semantic)
        artifact = normalized.to_dict()
        identity_hash = (
            "effective_contract_identity_sha256:"
            + _digest(
                "effective_contract_identity.v1",
                {"effective_contract_hash": normalized.contract_hash},
            )
        )
        return cls(
            canonical_json(artifact), normalized.contract_hash, identity_hash
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "effective_contract": self.effective_contract,
            "effective_contract_hash": self.effective_contract_hash,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "EffectiveContractIdentity":
        artifact = _mapping(value, "effective_contract_identity")
        _closed(
            artifact,
            {
                "schema_version",
                "effective_contract",
                "effective_contract_hash",
                "identity_hash",
            },
        )
        if artifact.get("schema_version") != EFFECTIVE_CONTRACT_IDENTITY_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        try:
            identity = cls.create(
                _mapping(artifact.get("effective_contract"), "effective_contract")
            )
        except ExperimentIdentityError as exc:
            if exc.code == "mapping_required":
                _fail("effective_contract_invalid", exc.detail)
            raise
        if artifact.get("effective_contract_hash") != identity.effective_contract_hash:
            _fail("hash_mismatch", "effective_contract_hash")
        if artifact.get("identity_hash") != identity.identity_hash:
            _fail("hash_mismatch", "effective_contract_identity")
        return identity


@dataclass(frozen=True)
class SequenceBundleIdentity:


    _chain_items: tuple[tuple[str, str], ...]
    sequence_bundle_hash: str
    schema_version: str = SEQUENCE_BUNDLE_IDENTITY_VERSION

    @property
    def chains(self) -> dict[str, str]:
        return dict(self._chain_items)

    @property
    def semantic_id(self) -> str:
        return self.sequence_bundle_hash

    @classmethod
    def create(
        cls, value: Mapping[str, str] | SequenceRecord | "SequenceBundleIdentity"
    ) -> "SequenceBundleIdentity":
        if isinstance(value, cls):
            return value
        try:
            record = value if isinstance(value, SequenceRecord) else SequenceRecord.create(value)
            record = SequenceRecord.from_mapping(record.to_dict())
        except (CausalFlowContractError, TypeError, ValueError) as exc:
            _fail("sequence_bundle_invalid", str(exc))
        return cls(tuple(sorted(record.chains.items())), record.semantic_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chains": [
                {"chain_id": chain, "sequence": sequence}
                for chain, sequence in self._chain_items
            ],
            "sequence_bundle_hash": self.sequence_bundle_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequenceBundleIdentity":
        artifact = _mapping(value, "sequence_bundle_identity")
        _closed(
            artifact, {"schema_version", "chains", "sequence_bundle_hash"}
        )
        if artifact.get("schema_version") != SEQUENCE_BUNDLE_IDENTITY_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        chains: dict[str, str] = {}
        raw_chains = artifact.get("chains")
        if not isinstance(raw_chains, list):
            _fail("chains_invalid")
        for raw in raw_chains:
            item = _mapping(raw, "chain")
            _closed(item, {"chain_id", "sequence"})
            chain_id = _required_text(item.get("chain_id"), "chain_id_required")
            if chain_id in chains:
                _fail("chain_duplicate", chain_id)
            sequence = _required_text(item.get("sequence"), "sequence_required")
            chains[chain_id] = sequence
        identity = cls.create(chains)
        observed = artifact.get("sequence_bundle_hash")
        if observed != identity.sequence_bundle_hash:
            _fail("hash_mismatch", "sequence_bundle")
        return identity


@dataclass(frozen=True)
class EvaluatorDescriptor:


    tool: str
    tool_version: str
    model: str
    _config_json: str
    _state_json: str
    seed: Optional[int]
    descriptor_hash: str
    schema_version: str = EVALUATOR_DESCRIPTOR_VERSION

    @property
    def config(self) -> Any:
        return json.loads(self._config_json)

    @property
    def state(self) -> Any:
        return json.loads(self._state_json)

    @classmethod
    def create(
        cls,
        *,
        tool: str,
        tool_version: str,
        model: str,
        config: Any,
        state: Any,
        seed: Optional[int],
    ) -> "EvaluatorDescriptor":
        resolved_tool = _required_text(tool, "tool_required")
        resolved_version = _required_text(tool_version, "tool_version_required")
        resolved_model = _required_text(model, "model_required")
        clean_config = _json_copy(config, "config")
        clean_state = _json_copy(state, "state")
        resolved_seed = _seed(seed)
        semantic = {
            "tool": resolved_tool,
            "tool_version": resolved_version,
            "model": resolved_model,
            "config": clean_config,
            "state": clean_state,
            "seed": resolved_seed,
        }
        descriptor_hash = (
            "evaluator_descriptor_sha256:"
            + _digest("evaluator_descriptor.v1", semantic)
        )
        return cls(
            resolved_tool,
            resolved_version,
            resolved_model,
            canonical_json(clean_config),
            canonical_json(clean_state),
            resolved_seed,
            descriptor_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "model": self.model,
            "config": self.config,
            "state": self.state,
            "seed": self.seed,
            "descriptor_hash": self.descriptor_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluatorDescriptor":
        artifact = _mapping(value, "evaluator_descriptor")
        _closed(
            artifact,
            {
                "schema_version",
                "tool",
                "tool_version",
                "model",
                "config",
                "state",
                "seed",
                "descriptor_hash",
            },
        )
        if artifact.get("schema_version") != EVALUATOR_DESCRIPTOR_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        descriptor = cls.create(
            tool=artifact.get("tool"),
            tool_version=artifact.get("tool_version"),
            model=artifact.get("model"),
            config=artifact.get("config"),
            state=artifact.get("state"),
            seed=artifact.get("seed"),
        )
        if artifact.get("descriptor_hash") != descriptor.descriptor_hash:
            _fail("hash_mismatch", "evaluator_descriptor")
        return descriptor


@dataclass(frozen=True)
class ExactEvaluationKey:


    sequence_bundle_hash: str
    _evaluator_descriptor_json: str
    evaluator_descriptor_hash: str
    cache_key: str
    schema_version: str = EXACT_EVALUATION_KEY_VERSION

    @property
    def evaluator_descriptor(self) -> dict[str, Any]:
        return json.loads(self._evaluator_descriptor_json)

    @classmethod
    def create(
        cls,
        sequence: Mapping[str, str] | SequenceRecord | SequenceBundleIdentity,
        evaluator: EvaluatorDescriptor | Mapping[str, Any],
    ) -> "ExactEvaluationKey":
        sequence_identity = SequenceBundleIdentity.create(sequence)
        try:
            descriptor = (
                evaluator
                if isinstance(evaluator, EvaluatorDescriptor)
                else EvaluatorDescriptor.from_mapping(evaluator)
            )
        except (ExperimentIdentityError, TypeError, ValueError) as exc:
            _fail("evaluator_descriptor_invalid", str(exc))
        descriptor = EvaluatorDescriptor.from_mapping(descriptor.to_dict())
        payload = {
            "sequence_bundle_hash": sequence_identity.sequence_bundle_hash,
            "evaluator_descriptor_hash": descriptor.descriptor_hash,
        }
        cache_key = "evaluation_cache_sha256:" + _digest(
            "exact_evaluation_key.v1", payload
        )
        return cls(
            sequence_identity.sequence_bundle_hash,
            canonical_json(descriptor.to_dict()),
            descriptor.descriptor_hash,
            cache_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence_bundle_hash": self.sequence_bundle_hash,
            "evaluator_descriptor": self.evaluator_descriptor,
            "evaluator_descriptor_hash": self.evaluator_descriptor_hash,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExactEvaluationKey":
        artifact = _mapping(value, "exact_evaluation_key")
        _closed(
            artifact,
            {
                "schema_version",
                "sequence_bundle_hash",
                "evaluator_descriptor",
                "evaluator_descriptor_hash",
                "cache_key",
            },
        )
        if artifact.get("schema_version") != EXACT_EVALUATION_KEY_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        sequence_hash = _validate_sequence_hash(
            artifact.get("sequence_bundle_hash")
        )
        descriptor = EvaluatorDescriptor.from_mapping(
            _mapping(artifact.get("evaluator_descriptor"), "evaluator_descriptor")
        )
        payload = {
            "sequence_bundle_hash": sequence_hash,
            "evaluator_descriptor_hash": descriptor.descriptor_hash,
        }
        expected = "evaluation_cache_sha256:" + _digest(
            "exact_evaluation_key.v1", payload
        )
        if artifact.get("evaluator_descriptor_hash") != descriptor.descriptor_hash:
            _fail("hash_mismatch", "evaluator_descriptor_hash")
        if artifact.get("cache_key") != expected:
            _fail("hash_mismatch", "exact_evaluation_key")
        return cls(
            sequence_hash,
            canonical_json(descriptor.to_dict()),
            descriptor.descriptor_hash,
            expected,
        )


__all__ = [
    "CODE_IDENTITY_VERSION",
    "EFFECTIVE_CONTRACT_IDENTITY_VERSION",
    "EVALUATOR_DESCRIPTOR_VERSION",
    "EXACT_EVALUATION_KEY_VERSION",
    "SEQUENCE_BUNDLE_IDENTITY_VERSION",
    "CodeIdentity",
    "EffectiveContractIdentity",
    "EvaluatorDescriptor",
    "ExactEvaluationKey",
    "ExperimentIdentityError",
    "SequenceBundleIdentity",
]

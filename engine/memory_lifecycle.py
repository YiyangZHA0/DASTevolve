

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from .memory_policy import MemoryPolicyConfig, MemoryScope


MEMORY_SNAPSHOT_VERSION = "astevolve.memory_snapshot.v1"
MEMORY_EXECUTION_CONTEXT_VERSION = "astevolve.memory_execution_context.v5"
LEGACY_MEMORY_EXECUTION_CONTEXT_VERSIONS = frozenset(
    {"astevolve.memory_execution_context.v4"}
)
MEMORY_COMMIT_MODES = frozenset({"deferred", "standalone"})
SCOPED_ADAPTIVE_PRIOR_VERSION = "astevolve.scoped_adaptive_prior.v1"
PROPOSAL_CAUSAL_ENVELOPE_VERSION = "astevolve.proposal_causal_envelope.v1"
PARENT_SEQUENCE_LINEAGE_VERSION = "astevolve.parent_sequence_lineage.v1"
PARENT_EFFECTIVE_AST_LINEAGE_VERSION = (
    "astevolve.parent_effective_ast_lineage.v1"
)


class MemorySnapshotError(ValueError):
    pass


class CausalEnvelopeError(ValueError):
    pass


def build_parent_sequence_lineage(
    parent_program_id: str,
    sequences: Mapping[str, Any] | None,
) -> str:


    parent_id = str(parent_program_id or "").strip()
    if not parent_id or not isinstance(sequences, Mapping) or not sequences:
        return ""
    normalized: Dict[str, str] = {}
    for raw_chain_id, raw_sequence in sequences.items():
        chain_id = str(raw_chain_id or "").strip()
        sequence = str(raw_sequence or "").strip().replace(" ", "").replace("\n", "").upper()
        if not chain_id or not sequence:
            raise CausalEnvelopeError(
                "parent sequence lineage requires non-empty chain IDs and sequences"
            )
        if any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence):
            raise CausalEnvelopeError(
                f"parent sequence lineage chain {chain_id!r} contains non-canonical residues"
            )
        if chain_id in normalized:
            raise CausalEnvelopeError(
                f"parent sequence lineage repeats chain {chain_id!r}"
            )
        normalized[chain_id] = sequence
    sequence_hash = _sha256(
        canonical_memory_content(dict(sorted(normalized.items()))).encode("utf-8")
    )
    return canonical_memory_content(
        {
            "schema_version": PARENT_SEQUENCE_LINEAGE_VERSION,
            "parent_program_id": parent_id,
            "sequences": dict(sorted(normalized.items())),
            "sequence_hash": sequence_hash,
        }
    )


def parse_parent_sequence_lineage(value: str) -> Dict[str, Any] | None:


    if not value:
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise CausalEnvelopeError("parent sequence lineage is invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise CausalEnvelopeError("parent sequence lineage must be a mapping")
    if set(raw) != {
        "schema_version",
        "parent_program_id",
        "sequences",
        "sequence_hash",
    }:
        raise CausalEnvelopeError("parent sequence lineage has an invalid closed schema")
    if raw.get("schema_version") != PARENT_SEQUENCE_LINEAGE_VERSION:
        raise CausalEnvelopeError("unsupported parent sequence lineage schema_version")
    rebuilt = build_parent_sequence_lineage(
        str(raw.get("parent_program_id") or ""), raw.get("sequences")
    )
    if rebuilt != value:
        raise CausalEnvelopeError("parent sequence lineage is not canonical or hash-matched")
    return deepcopy(dict(raw))


def _effective_dual_ast_hash(ast: Mapping[str, Any]) -> str:


    try:
        canonical = json.dumps(
            ast,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CausalEnvelopeError(
            f"parent effective AST is not canonical JSON: {error}"
        ) from error
    return _sha256(
        f"astevolve.executable_dual_ast.v1\0{canonical}".encode("utf-8")
    )


def _normalize_parent_ast_node_metadata(
    value: Mapping[str, Any] | None,
    *,
    editable_node_ids: set[str],
) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CausalEnvelopeError("parent effective AST node_metadata must be a mapping")
    normalized: Dict[str, Any] = {}
    for raw_node_id, raw_metadata in value.items():
        if not isinstance(raw_node_id, str) or not raw_node_id.strip():
            raise CausalEnvelopeError(
                "parent effective AST node_metadata keys must be non-empty strings"
            )
        node_id = raw_node_id.strip()
        if node_id != raw_node_id or node_id in normalized:
            raise CausalEnvelopeError(
                "parent effective AST node_metadata keys must be canonical and unique"
            )
        if node_id not in editable_node_ids:
            raise CausalEnvelopeError(
                f"parent effective AST node_metadata references non-editable node {node_id!r}"
            )
        if not isinstance(raw_metadata, Mapping):
            raise CausalEnvelopeError(
                f"parent effective AST node_metadata[{node_id!r}] must be a mapping"
            )
        try:
            normalized[node_id] = json.loads(
                canonical_memory_content(raw_metadata)
            )
        except MemorySnapshotError as error:
            raise CausalEnvelopeError(
                f"parent effective AST node_metadata[{node_id!r}] is invalid: {error}"
            ) from error
    return dict(sorted(normalized.items()))


def build_parent_effective_ast_lineage(
    parent_program_id: str,
    executable_dual_ast: Mapping[str, Any] | None,
    node_metadata: Mapping[str, Any] | None = None,
    *,
    expected_effective_ast_hash: str = "",
) -> str:


    parent_id = str(parent_program_id or "").strip()
    if not parent_id or executable_dual_ast is None:
        return ""
    if not isinstance(executable_dual_ast, Mapping):
        raise CausalEnvelopeError("parent effective AST must be a mapping")
    try:
        from astevolve.domain.dual_ast import ExecutableDualAST

        normalized_ast = ExecutableDualAST.from_mapping(
            executable_dual_ast
        ).to_dict()
    except (TypeError, ValueError) as error:
        raise CausalEnvelopeError(f"parent effective AST is invalid: {error}") from error
    editable_ids = {
        str(node["node_id"])
        for node in normalized_ast["structural_nodes"]
        if node["kind"] == "editable"
    }
    normalized_metadata = _normalize_parent_ast_node_metadata(
        node_metadata,
        editable_node_ids=editable_ids,
    )
    effective_ast_hash = _effective_dual_ast_hash(normalized_ast)
    claimed_hash = str(expected_effective_ast_hash or "").strip().lower()
    if claimed_hash and claimed_hash != effective_ast_hash:
        raise CausalEnvelopeError(
            "parent effective AST does not match its evaluated effective_ast_hash"
        )
    core = {
        "schema_version": PARENT_EFFECTIVE_AST_LINEAGE_VERSION,
        "parent_program_id": parent_id,
        "executable_dual_ast": normalized_ast,
        "node_metadata": normalized_metadata,
        "effective_ast_hash": effective_ast_hash,
    }
    lineage_hash = _sha256(canonical_memory_content(core).encode("utf-8"))
    return canonical_memory_content({**core, "lineage_hash": lineage_hash})


def parse_parent_effective_ast_lineage(value: str) -> Dict[str, Any] | None:


    if not value:
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise CausalEnvelopeError(
            "parent effective AST lineage is invalid JSON"
        ) from error
    if not isinstance(raw, Mapping):
        raise CausalEnvelopeError("parent effective AST lineage must be a mapping")
    if set(raw) != {
        "schema_version",
        "parent_program_id",
        "executable_dual_ast",
        "node_metadata",
        "effective_ast_hash",
        "lineage_hash",
    }:
        raise CausalEnvelopeError(
            "parent effective AST lineage has an invalid closed schema"
        )
    if raw.get("schema_version") != PARENT_EFFECTIVE_AST_LINEAGE_VERSION:
        raise CausalEnvelopeError(
            "unsupported parent effective AST lineage schema_version"
        )
    rebuilt = build_parent_effective_ast_lineage(
        str(raw.get("parent_program_id") or ""),
        raw.get("executable_dual_ast"),
        raw.get("node_metadata"),
        expected_effective_ast_hash=str(raw.get("effective_ast_hash") or ""),
    )
    if rebuilt != value:
        raise CausalEnvelopeError(
            "parent effective AST lineage is not canonical or hash-matched"
        )
    return deepcopy(dict(raw))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value, key=str)
    raise TypeError(f"memory contains a non-JSON value: {type(value).__name__}")


def canonical_memory_content(document: Mapping[str, Any]) -> str:


    if not isinstance(document, Mapping):
        raise MemorySnapshotError("memory document must be a mapping")

    def reject_nonfinite(value: Any, path: str = "memory") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise MemorySnapshotError(f"{path} contains a non-finite float")
        if isinstance(value, Mapping):
            for key, item in value.items():
                reject_nonfinite(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                reject_nonfinite(item, f"{path}[{index}]")

    reject_nonfinite(document)
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise MemorySnapshotError(f"memory is not canonically serializable: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MemorySnapshot:


    canonical_content: str
    content_hash: str
    raw_hash: str
    source_path: str
    raw_size: int
    source_exists: bool
    schema_version: str = MEMORY_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SNAPSHOT_VERSION:
            raise MemorySnapshotError(
                f"unsupported memory snapshot version: {self.schema_version!r}"
            )
        try:
            parsed = json.loads(self.canonical_content)
        except json.JSONDecodeError as exc:
            raise MemorySnapshotError("snapshot canonical content is invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise MemorySnapshotError("snapshot canonical content is not a mapping")
        if canonical_memory_content(parsed) != self.canonical_content:
            raise MemorySnapshotError("snapshot content is not in canonical form")
        actual_hash = _sha256(self.canonical_content.encode("utf-8"))
        if actual_hash != self.content_hash:
            raise MemorySnapshotError("snapshot content_hash does not match its content")
        if int(self.raw_size) < 0:
            raise MemorySnapshotError("snapshot raw_size cannot be negative")

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        source_path: str | Path = "",
        raw_content: Optional[bytes] = None,
        source_exists: bool = True,
    ) -> "MemorySnapshot":
        canonical = canonical_memory_content(document)
        canonical_bytes = canonical.encode("utf-8")
        raw = canonical_bytes if raw_content is None else bytes(raw_content)
        return cls(
            canonical_content=canonical,
            content_hash=_sha256(canonical_bytes),
            raw_hash=_sha256(raw),
            source_path=str(source_path),
            raw_size=len(raw),
            source_exists=bool(source_exists),
        )

    def materialize(self) -> Dict[str, Any]:


        value = json.loads(self.canonical_content)
        if not isinstance(value, dict):
            raise MemorySnapshotError("snapshot canonical content is not a mapping")
        return deepcopy(value)

    def to_artifact(self) -> Dict[str, Any]:


        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_exists": self.source_exists,
            "content_hash": self.content_hash,
            "raw_hash": self.raw_hash,
            "raw_size": self.raw_size,
        }


def capture_memory_snapshot(path: str | Path) -> MemorySnapshot:


    memory_path = Path(path)
    if not memory_path.exists():
        return MemorySnapshot.from_document(
            {},
            source_path=memory_path,
            raw_content=b"",
            source_exists=False,
        )
    try:
        raw = memory_path.read_bytes()
    except OSError as exc:
        raise MemorySnapshotError(f"failed to read memory {memory_path}: {exc}") from exc

    try:
        import yaml
    except Exception as exc:
        raise MemorySnapshotError("PyYAML is required to read memory.yaml") from exc
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise MemorySnapshotError(f"failed to parse memory {memory_path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise MemorySnapshotError(f"memory {memory_path} did not contain a mapping")
    return MemorySnapshot.from_document(
        loaded,
        source_path=memory_path,
        raw_content=raw,
        source_exists=True,
    )


@dataclass(frozen=True)
class ScopedAdaptivePriorSnapshot:


    scope: MemoryScope
    snapshot: MemorySnapshot
    schema_version: str = SCOPED_ADAPTIVE_PRIOR_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCOPED_ADAPTIVE_PRIOR_VERSION:
            raise MemorySnapshotError(
                f"unsupported scoped adaptive-prior version: {self.schema_version!r}"
            )
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("scoped adaptive prior requires a MemoryScope")
        if not isinstance(self.snapshot, MemorySnapshot):
            raise TypeError("scoped adaptive prior requires a MemorySnapshot")

    def effective_prior(
        self,
        policy: MemoryPolicyConfig,
        requested_scope: MemoryScope,
    ) -> Dict[str, Any]:


        if not isinstance(policy, MemoryPolicyConfig):
            raise TypeError("policy must be a MemoryPolicyConfig")
        if not policy.may_read_adaptive_prior:
            return {}
        requested_scope.require_compatible(self.scope, level="lineage")
        document = self.snapshot.materialize()
        adaptive = document.get("adaptive_memory", {})
        if not isinstance(adaptive, Mapping):
            raise MemorySnapshotError("adaptive_memory must be a mapping")
        return deepcopy(dict(adaptive))

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.to_artifact(),
            "snapshot": self.snapshot.to_artifact(),
        }


@dataclass(frozen=True)
class ProposalCausalEnvelope:


    parent_program_id: str
    parent_code_hash: str
    parent_effective_contract_json: str
    parent_patch_json: str
    envelope_hash: str
    schema_version: str = PROPOSAL_CAUSAL_ENVELOPE_VERSION

    @staticmethod
    def _canonical_optional_mapping(
        value: Optional[Mapping[str, Any]], field_name: str
    ) -> str:
        if value is None:
            return ""
        if not isinstance(value, Mapping):
            raise CausalEnvelopeError(f"{field_name} must be a mapping or None")
        try:
            return canonical_memory_content(value)
        except MemorySnapshotError as exc:
            raise CausalEnvelopeError(f"{field_name} is not canonical JSON: {exc}") from exc

    @staticmethod
    def _normalize_code_hash(value: str) -> str:
        code_hash = str(value or "").strip().lower()
        if code_hash and (
            len(code_hash) != 64 or any(ch not in "0123456789abcdef" for ch in code_hash)
        ):
            raise CausalEnvelopeError("parent_code_hash must be an SHA-256 hex digest")
        return code_hash

    @classmethod
    def create(
        cls,
        *,
        parent_program_id: str,
        parent_code_hash: str = "",
        parent_effective_contract: Optional[Mapping[str, Any]] = None,
        parent_patch: Optional[Mapping[str, Any]] = None,
    ) -> "ProposalCausalEnvelope":
        parent_id = str(parent_program_id or "").strip()
        if not parent_id:
            raise CausalEnvelopeError("parent_program_id must be non-empty")
        code_hash = cls._normalize_code_hash(parent_code_hash)
        contract_json = cls._canonical_optional_mapping(
            parent_effective_contract, "parent_effective_contract"
        )
        patch_json = cls._canonical_optional_mapping(parent_patch, "parent_patch")
        core = {
            "schema_version": PROPOSAL_CAUSAL_ENVELOPE_VERSION,
            "parent_program_id": parent_id,
            "parent_code_hash": code_hash or None,
            "parent_effective_contract": (
                json.loads(contract_json) if contract_json else None
            ),
            "parent_patch": json.loads(patch_json) if patch_json else None,
        }
        return cls(
            parent_program_id=parent_id,
            parent_code_hash=code_hash,
            parent_effective_contract_json=contract_json,
            parent_patch_json=patch_json,
            envelope_hash=_sha256(canonical_memory_content(core).encode("utf-8")),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalCausalEnvelope":
        if not isinstance(value, Mapping):
            raise CausalEnvelopeError("proposal causal envelope must be a mapping")
        allowed = {
            "schema_version",
            "parent_program_id",
            "parent_code_hash",
            "parent_effective_contract",
            "parent_patch",
            "envelope_hash",
        }
        unknown = sorted(str(key) for key in set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown:
            raise CausalEnvelopeError(
                "unknown proposal causal envelope fields: " + ", ".join(unknown)
            )
        if missing:
            raise CausalEnvelopeError(
                "missing proposal causal envelope fields: " + ", ".join(missing)
            )
        if value.get("schema_version") != PROPOSAL_CAUSAL_ENVELOPE_VERSION:
            raise CausalEnvelopeError(
                f"unsupported proposal causal envelope version: {value.get('schema_version')!r}"
            )
        for field_name in ("parent_effective_contract", "parent_patch"):
            field_value = value.get(field_name)
            if field_value is not None and not isinstance(field_value, Mapping):
                raise CausalEnvelopeError(
                    f"{field_name} must be a mapping or None"
                )
        rebuilt = cls.create(
            parent_program_id=str(value.get("parent_program_id") or ""),
            parent_code_hash=str(value.get("parent_code_hash") or ""),
            parent_effective_contract=(
                value.get("parent_effective_contract")
                if isinstance(value.get("parent_effective_contract"), Mapping)
                else None
            ),
            parent_patch=(
                value.get("parent_patch")
                if isinstance(value.get("parent_patch"), Mapping)
                else None
            ),
        )
        if str(value.get("envelope_hash") or "") != rebuilt.envelope_hash:
            raise CausalEnvelopeError(
                "proposal causal envelope hash does not match canonical content"
            )
        return rebuilt

    @classmethod
    def from_json(cls, value: str) -> "ProposalCausalEnvelope":
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise CausalEnvelopeError("proposal causal envelope JSON is invalid") from exc
        return cls.from_mapping(parsed)

    @property
    def parent_effective_contract(self) -> Optional[Dict[str, Any]]:
        return (
            deepcopy(json.loads(self.parent_effective_contract_json))
            if self.parent_effective_contract_json
            else None
        )

    @property
    def parent_patch(self) -> Optional[Dict[str, Any]]:
        return (
            deepcopy(json.loads(self.parent_patch_json))
            if self.parent_patch_json
            else None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_program_id": self.parent_program_id,
            "parent_code_hash": self.parent_code_hash or None,
            "parent_effective_contract": self.parent_effective_contract,
            "parent_patch": self.parent_patch,
            "envelope_hash": self.envelope_hash,
        }

    def to_json(self) -> str:
        return canonical_memory_content(self.to_dict())


@dataclass(frozen=True)
class MemoryExecutionContext:


    generation_id: str = ""
    proposal_id: str = ""
    trial_id: str = ""
    island_id: Optional[int] = None
    island_role: str = ""
    scope_id: str = ""
    logical_time: str = ""
    commit_mode: str = "deferred"
    snapshot: Optional[MemorySnapshot] = None
    target_path: str = ""
    output_dir: str = ""
    edit_contract_envelope_json: str = ""
    edit_contract_response_json: str = ""
    proposal_causal_envelope_json: str = ""
    parent_sequence_lineage_json: str = ""
    parent_effective_ast_lineage_json: str = ""
    design_action_json: str = ""
    design_action_parent_binding_json: str = ""
    memory_scope: Optional[MemoryScope] = None
    memory_policy: MemoryPolicyConfig = MemoryPolicyConfig()
    history_registry_path: str = ""
    history_scope: str = ""
    history_owner_token: str = ""
    history_lease_seconds: float = 300.0
    history_replicate_policy: str = "reject"
    schema_version: str = MEMORY_EXECUTION_CONTEXT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            MEMORY_EXECUTION_CONTEXT_VERSION,
            *LEGACY_MEMORY_EXECUTION_CONTEXT_VERSIONS,
        }:
            raise CausalEnvelopeError(
                f"unsupported memory execution context version: {self.schema_version!r}"
            )
        if (
            self.schema_version in LEGACY_MEMORY_EXECUTION_CONTEXT_VERSIONS
            and (self.design_action_json or self.design_action_parent_binding_json)
        ):
            raise CausalEnvelopeError(
                "legacy memory execution contexts cannot carry DesignAction v1"
            )
        if (
            self.island_id is not None
            and (
                isinstance(self.island_id, bool)
                or not isinstance(self.island_id, int)
                or self.island_id < 0
            )
        ):
            raise CausalEnvelopeError("island_id must be a non-negative integer or None")
        if self.proposal_causal_envelope_json:
            envelope = ProposalCausalEnvelope.from_json(
                self.proposal_causal_envelope_json
            )
            if self.proposal_causal_envelope_json != envelope.to_json():
                raise CausalEnvelopeError(
                    "proposal causal envelope JSON must be canonical"
                )
        if self.edit_contract_response_json:
            if not self.edit_contract_envelope_json:
                raise CausalEnvelopeError(
                    "edit-contract response requires a full runtime envelope"
                )
            from astevolve.runtime.edit_contract_lifecycle import (
                canonical_contract_response_json,
            )

            try:
                parsed_response = json.loads(self.edit_contract_response_json)
                canonical_response = canonical_contract_response_json(
                    parsed_response, self.edit_contract_envelope_json
                )
            except (TypeError, ValueError) as error:
                raise CausalEnvelopeError(
                    "edit-contract response is invalid or stale"
                ) from error
            if canonical_response != self.edit_contract_response_json:
                raise CausalEnvelopeError(
                    "edit-contract response JSON must be canonical"
                )
        if self.parent_sequence_lineage_json:
            parse_parent_sequence_lineage(self.parent_sequence_lineage_json)
        if self.parent_effective_ast_lineage_json:
            parse_parent_effective_ast_lineage(
                self.parent_effective_ast_lineage_json
            )
        if self.design_action_json or self.design_action_parent_binding_json:
            if not self.design_action_json or not self.design_action_parent_binding_json:
                raise CausalEnvelopeError(
                    "DesignAction and controller parent binding must be supplied together"
                )
            action = self._parse_design_action_json(self.design_action_json)
            binding = self._parse_design_action_parent_binding_json(
                self.design_action_parent_binding_json
            )
            if action.parent.to_dict() != binding:
                raise CausalEnvelopeError(
                    "DesignAction parent does not match controller-owned binding"
                )
        if self.history_registry_path:


            from .history_lifecycle import HistoryExecutionContext

            HistoryExecutionContext(
                registry_path=self.history_registry_path,
                scope=self.history_scope,
                owner_token=self.history_owner_token,
                lease_seconds=self.history_lease_seconds,
                replicate_policy=self.history_replicate_policy,
            )

    def proposal_causal_envelope(self) -> Optional[Dict[str, Any]]:
        if not self.proposal_causal_envelope_json:
            return None
        return ProposalCausalEnvelope.from_json(
            self.proposal_causal_envelope_json
        ).to_dict()

    def edit_contract_response(self) -> Optional[Dict[str, Any]]:


        if not self.edit_contract_response_json:
            return None
        return deepcopy(json.loads(self.edit_contract_response_json))

    def with_edit_contract_response(
        self, response: Mapping[str, Any] | None
    ) -> "MemoryExecutionContext":


        if response is None:
            if self.edit_contract_envelope_json:
                from astevolve.runtime.edit_contract_lifecycle import (
                    EditContractEnvelope,
                    validate_contract_response,
                )

                validate_contract_response(
                    None,
                    EditContractEnvelope.from_json(
                        self.edit_contract_envelope_json
                    ),
                )
            return replace(self, edit_contract_response_json="")
        if not self.edit_contract_envelope_json:
            raise CausalEnvelopeError(
                "edit-contract response was supplied without a parent offer"
            )
        from astevolve.runtime.edit_contract_lifecycle import (
            canonical_contract_response_json,
        )

        canonical = canonical_contract_response_json(
            response, self.edit_contract_envelope_json
        )
        return replace(self, edit_contract_response_json=canonical)

    def with_island_identity(
        self,
        island_id: Optional[int],
        island_role: Mapping[str, Any] | str = "",
    ) -> "MemoryExecutionContext":


        role_id = (
            str(island_role.get("role_id") or "")
            if isinstance(island_role, Mapping)
            else str(island_role or "")
        )
        return replace(
            self,
            island_id=island_id,
            island_role=role_id.strip(),
        )

    @staticmethod
    def _parse_design_action_json(value: str):


        from astevolve.domain.design_action import DesignAction, DesignActionError

        try:
            action = DesignAction.from_json(value)
        except (TypeError, ValueError, DesignActionError) as error:
            raise CausalEnvelopeError("DesignAction runtime JSON is invalid") from error
        canonical = json.dumps(
            action.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if value != canonical:
            raise CausalEnvelopeError("DesignAction runtime JSON must be canonical")
        return action

    def typed_design_action(self):


        if not self.design_action_json:
            return None
        return self._parse_design_action_json(self.design_action_json)

    @staticmethod
    def _parse_design_action_parent_binding_json(value: str) -> Dict[str, str]:
        fields = {
            "case_id",
            "parent_program_id",
            "parent_candidate_id",
            "parent_sequence_bundle_hash",
            "parent_effective_contract_hash",
            "parent_evolve_hash",
        }
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise CausalEnvelopeError(
                "DesignAction parent binding JSON is invalid"
            ) from error
        if not isinstance(parsed, Mapping) or set(parsed) != fields:
            raise CausalEnvelopeError(
                "DesignAction parent binding must contain exactly six fields"
            )
        normalized: Dict[str, str] = {}
        for field in sorted(fields):
            item = parsed.get(field)
            if not isinstance(item, str) or not item.strip() or item != item.strip():
                raise CausalEnvelopeError(
                    f"DesignAction parent binding field {field!r} is invalid"
                )
            normalized[field] = item
        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if value != canonical:
            raise CausalEnvelopeError(
                "DesignAction parent binding JSON must be canonical"
            )
        return normalized

    def design_action_parent_binding(self) -> Optional[Dict[str, str]]:
        if not self.design_action_parent_binding_json:
            return None
        return deepcopy(
            self._parse_design_action_parent_binding_json(
                self.design_action_parent_binding_json
            )
        )

    def design_action(self) -> Optional[Dict[str, Any]]:
        action = self.typed_design_action()
        return deepcopy(action.to_dict()) if action is not None else None

    def with_design_action(
        self,
        action: Any | None,
        *,
        parent_binding: Mapping[str, Any] | None = None,
    ) -> "MemoryExecutionContext":


        if action is None:
            return replace(
                self,
                design_action_json="",
                design_action_parent_binding_json="",
                schema_version=MEMORY_EXECUTION_CONTEXT_VERSION,
            )
        if parent_binding is None:
            raise CausalEnvelopeError(
                "controller-owned DesignAction parent binding is required"
            )
        from astevolve.domain.design_action import DesignAction, DesignActionError

        try:
            typed = (
                action
                if isinstance(action, DesignAction)
                else DesignAction.from_mapping(action)
            )
        except (TypeError, ValueError, DesignActionError) as error:
            raise CausalEnvelopeError("DesignAction runtime payload is invalid") from error
        canonical = json.dumps(
            typed.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        parent_canonical = json.dumps(
            dict(parent_binding),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        validated_binding = self._parse_design_action_parent_binding_json(
            parent_canonical
        )
        if typed.parent.to_dict() != validated_binding:
            raise CausalEnvelopeError(
                "DesignAction parent does not match controller-owned binding"
            )
        return replace(
            self,
            design_action_json=canonical,
            design_action_parent_binding_json=parent_canonical,
            schema_version=MEMORY_EXECUTION_CONTEXT_VERSION,
        )

    def parent_sequence_lineage(self) -> Optional[Dict[str, Any]]:
        return parse_parent_sequence_lineage(self.parent_sequence_lineage_json)

    def parent_effective_ast_lineage(self) -> Optional[Dict[str, Any]]:
        return parse_parent_effective_ast_lineage(
            self.parent_effective_ast_lineage_json
        )

    def history_execution_context(self):


        if not self.history_registry_path:
            return None
        from .history_lifecycle import HistoryExecutionContext

        return HistoryExecutionContext(
            registry_path=self.history_registry_path,
            scope=self.history_scope,
            owner_token=self.history_owner_token,
            lease_seconds=self.history_lease_seconds,
            replicate_policy=self.history_replicate_policy,
        )

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "proposal_id": self.proposal_id,
            "trial_id": self.trial_id,
            "island_id": self.island_id,
            "island_role": self.island_role or None,
            "scope_id": self.scope_id,
            "logical_time": self.logical_time or None,
            "commit_mode": self.commit_mode,
            "target_path": self.target_path or None,
            "output_dir": self.output_dir or None,
            "edit_contract_envelope_json": self.edit_contract_envelope_json or None,
            "edit_contract_response": self.edit_contract_response(),
            "proposal_causal_envelope": self.proposal_causal_envelope(),
            "parent_sequence_lineage": self.parent_sequence_lineage(),
            "parent_effective_ast_lineage": self.parent_effective_ast_lineage(),
            "design_action": self.design_action(),
            "design_action_parent_binding": self.design_action_parent_binding(),
            "proposal_causal_envelope_hash": (
                ProposalCausalEnvelope.from_json(
                    self.proposal_causal_envelope_json
                ).envelope_hash
                if self.proposal_causal_envelope_json
                else None
            ),
            "memory_scope": self.memory_scope.to_artifact() if self.memory_scope else None,
            "memory_policy": self.memory_policy.to_artifact(),
            "snapshot": self.snapshot.to_artifact() if self.snapshot else None,
            "history_execution_context": (
                self.history_execution_context().to_artifact()
                if self.history_execution_context() is not None
                else None
            ),
        }


_MEMORY_EXECUTION_CONTEXT: ContextVar[Optional[MemoryExecutionContext]] = ContextVar(
    "astevolve_memory_execution_context",
    default=None,
)


def current_memory_execution_context() -> Optional[MemoryExecutionContext]:


    return _MEMORY_EXECUTION_CONTEXT.get()


def _default_scope_id(generation_id: str, proposal_id: str, trial_id: str) -> str:
    return "/".join(item for item in (generation_id, proposal_id, trial_id) if item)


@contextmanager
def memory_execution_scope(
    *,
    generation_id: str = "",
    proposal_id: str = "",
    trial_id: str = "",
    island_id: Optional[int] = None,
    island_role: Mapping[str, Any] | str = "",
    scope_id: str = "",
    logical_time: str = "",
    commit_mode: str = "deferred",
    snapshot: Optional[MemorySnapshot] = None,
    target_path: str | Path = "",
    output_dir: str | Path = "",
    edit_contract_envelope_json: str = "",
    edit_contract_response_json: str = "",
    proposal_causal_envelope_json: str = "",
    parent_sequence_lineage_json: str = "",
    parent_effective_ast_lineage_json: str = "",
    design_action_json: str = "",
    design_action_parent_binding_json: str = "",
    memory_scope: Optional[MemoryScope] = None,
    memory_policy: Optional[MemoryPolicyConfig] = None,
    history_registry_path: str | Path = "",
    history_scope: str = "",
    history_owner_token: str = "",
    history_lease_seconds: float = 300.0,
    history_replicate_policy: str = "reject",
) -> Iterator[MemoryExecutionContext]:


    normalized_mode = str(commit_mode or "deferred").strip().lower()
    if normalized_mode not in MEMORY_COMMIT_MODES:
        raise ValueError(
            f"unsupported memory commit mode {commit_mode!r}; "
            f"expected one of {sorted(MEMORY_COMMIT_MODES)}"
        )
    resolved_scope = str(scope_id or _default_scope_id(
        str(generation_id), str(proposal_id), str(trial_id)
    ))
    context = MemoryExecutionContext(
        generation_id=str(generation_id),
        proposal_id=str(proposal_id),
        trial_id=str(trial_id),
        island_id=island_id,
        island_role=(
            str(island_role.get("role_id") or "")
            if isinstance(island_role, Mapping)
            else str(island_role or "")
        ),
        scope_id=resolved_scope,
        logical_time=str(logical_time),
        commit_mode=normalized_mode,
        snapshot=snapshot,
        target_path=str(target_path),
        output_dir=str(output_dir),
        edit_contract_envelope_json=str(edit_contract_envelope_json or ""),
        edit_contract_response_json=str(edit_contract_response_json or ""),
        proposal_causal_envelope_json=str(proposal_causal_envelope_json or ""),
        parent_sequence_lineage_json=str(parent_sequence_lineage_json or ""),
        parent_effective_ast_lineage_json=str(
            parent_effective_ast_lineage_json or ""
        ),
        design_action_json=str(design_action_json or ""),
        design_action_parent_binding_json=str(
            design_action_parent_binding_json or ""
        ),
        memory_scope=memory_scope,
        memory_policy=memory_policy or MemoryPolicyConfig(),
        history_registry_path=str(history_registry_path),
        history_scope=str(history_scope),
        history_owner_token=str(history_owner_token),
        history_lease_seconds=float(history_lease_seconds),
        history_replicate_policy=str(history_replicate_policy),
    )
    token: Token[Optional[MemoryExecutionContext]] = _MEMORY_EXECUTION_CONTEXT.set(context)
    try:
        yield context
    finally:
        _MEMORY_EXECUTION_CONTEXT.reset(token)


@contextmanager
def install_memory_execution_context(
    context: MemoryExecutionContext,
) -> Iterator[MemoryExecutionContext]:


    if not isinstance(context, MemoryExecutionContext):
        raise TypeError("context must be a MemoryExecutionContext")
    with memory_execution_scope(
        generation_id=context.generation_id,
        proposal_id=context.proposal_id,
        trial_id=context.trial_id,
        island_id=context.island_id,
        island_role=context.island_role,
        scope_id=context.scope_id,
        logical_time=context.logical_time,
        commit_mode=context.commit_mode,
        snapshot=context.snapshot,
        target_path=context.target_path,
        output_dir=context.output_dir,
        edit_contract_envelope_json=context.edit_contract_envelope_json,
        edit_contract_response_json=context.edit_contract_response_json,
        proposal_causal_envelope_json=context.proposal_causal_envelope_json,
        parent_sequence_lineage_json=context.parent_sequence_lineage_json,
        parent_effective_ast_lineage_json=(
            context.parent_effective_ast_lineage_json
        ),
        design_action_json=context.design_action_json,
        design_action_parent_binding_json=(
            context.design_action_parent_binding_json
        ),
        memory_scope=context.memory_scope,
        memory_policy=context.memory_policy,
        history_registry_path=context.history_registry_path,
        history_scope=context.history_scope,
        history_owner_token=context.history_owner_token,
        history_lease_seconds=context.history_lease_seconds,
        history_replicate_policy=context.history_replicate_policy,
    ) as installed:
        history_context = context.history_execution_context()
        if history_context is None:
            yield installed
            return
        from .history_lifecycle import history_execution_scope

        with history_execution_scope(context=history_context):
            yield installed



from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from astevolve.search.artifact_io import _ArtifactJSONEncoder
from astevolve.search.sequence_generator import (
    SEQUENCE_GENERATION_REQUEST_VERSION,
    SEQUENCE_GENERATION_RESULT_VERSION,
    SequenceGenerationRequest,
    SequenceGenerationResult,
    validate_generation_result,
)


NORMALIZED_SEARCH_ARTIFACT_VERSION = "astevolve.normalized_search_artifact.v1"
_RUNTIME_VERSION = "astevolve.sequence_generation_runtime.v1"
_REF_KEY = "$astevolve_ref"
_TABLE_NAMES = (
    "sequences",
    "generation_requests",
    "generation_results",
    "generation_runtime",
    "moves",
)


class NormalizedArtifactError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        cls=_ArtifactJSONEncoder,
    )


def _digest(namespace: str, value: Any) -> str:
    payload = f"{namespace}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reference(table: str, identity: str) -> Dict[str, Dict[str, str]]:
    return {_REF_KEY: {"table": table, "id": identity}}


def _sequence_mapping(value: Any) -> Optional[Dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        return None
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return None
    return {str(key): str(item) for key, item in sorted(value.items())}


class _Normalizer:
    def __init__(self) -> None:
        self.tables: Dict[str, Dict[str, Any]] = {
            table: {} for table in _TABLE_NAMES
        }


        self._canonical_entries: Dict[str, Dict[str, str]] = {
            table: {} for table in _TABLE_NAMES
        }


        self._object_identities: Dict[
            tuple[str, int], tuple[dict[Any, Any], str]
        ] = {}

    def _cached_object_identity(
        self, table: str, value: Mapping[str, Any]
    ) -> Optional[str]:
        if type(value) is not dict:
            return None
        cached = self._object_identities.get((table, id(value)))
        if cached is None or cached[0] is not value:
            return None
        return cached[1]

    def _remember_object_identity(
        self, table: str, value: Mapping[str, Any], identity: str
    ) -> None:
        if type(value) is dict:
            self._object_identities[(table, id(value))] = (value, identity)

    def _store(
        self,
        table: str,
        identity: str,
        value: Any,
        *,
        canonical: Optional[str] = None,
    ) -> None:
        encoded = canonical if canonical is not None else _canonical_json(value)
        if (
            identity in self.tables[table]
            and self._canonical_entries[table][identity] != encoded
        ):
            raise NormalizedArtifactError(
                f"conflicting {table} entry for identity {identity}"
            )
        if identity not in self.tables[table]:
            self.tables[table][identity] = value
            self._canonical_entries[table][identity] = encoded

    def _intern_sequence(self, value: Mapping[str, str]) -> Dict[str, Any]:
        cached = self._cached_object_identity("sequences", value)
        if cached is not None:
            return _reference("sequences", cached)
        sequence = {str(key): str(item) for key, item in sorted(value.items())}
        canonical = _canonical_json(sequence)
        identity = hashlib.sha256(
            f"astevolve.sequences.v1\0{canonical}".encode("utf-8")
        ).hexdigest()
        self._store("sequences", identity, sequence, canonical=canonical)
        self._remember_object_identity("sequences", value, identity)
        return _reference("sequences", identity)

    def _intern_contract(self, value: Mapping[str, Any], *, request: bool) -> Dict[str, Any]:
        table = "generation_requests" if request else "generation_results"
        cached = self._cached_object_identity(table, value)
        if cached is not None:
            return _reference(table, cached)
        identity_field = "request_hash" if request else "result_hash"
        identity = str(value.get(identity_field) or "")
        if not identity:
            raise NormalizedArtifactError(f"{identity_field} is required")
        normalized = {
            str(key): self.normalize(item, key=str(key))
            for key, item in value.items()
        }
        self._store(table, identity, normalized)
        self._remember_object_identity(table, value, identity)
        return _reference(table, identity)

    def _intern_runtime(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        cached = self._cached_object_identity("generation_runtime", value)
        if cached is not None:
            return _reference("generation_runtime", cached)
        normalized = {
            str(key): self.normalize(item, key=str(key))
            for key, item in value.items()
        }
        result = value.get("result")
        identity = str(result.get("result_hash") or "") if isinstance(result, Mapping) else ""
        if not identity:
            identity = _digest("astevolve.sequence_generation_runtime.v1", normalized)
        self._store("generation_runtime", identity, normalized)
        self._remember_object_identity("generation_runtime", value, identity)
        return _reference("generation_runtime", identity)

    def _intern_move(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        cached = self._cached_object_identity("moves", value)
        if cached is not None:
            return _reference("moves", cached)
        normalized = {
            str(key): self.normalize(item, key=str(key))
            for key, item in value.items()
        }
        canonical = _canonical_json(normalized)
        identity = hashlib.sha256(
            f"astevolve.search_move.v1\0{canonical}".encode("utf-8")
        ).hexdigest()
        self._store("moves", identity, normalized, canonical=canonical)
        self._remember_object_identity("moves", value, identity)
        return _reference("moves", identity)

    def normalize(self, value: Any, *, key: Optional[str] = None) -> Any:
        if isinstance(value, Mapping):
            schema_version = value.get("schema_version")
            if schema_version == SEQUENCE_GENERATION_REQUEST_VERSION:
                return self._intern_contract(value, request=True)
            if schema_version == SEQUENCE_GENERATION_RESULT_VERSION:
                return self._intern_contract(value, request=False)
            if schema_version == _RUNTIME_VERSION:
                return self._intern_runtime(value)
            if key == "move":
                return self._intern_move(value)
            if key in {"seqs", "sequences", "parent_sequences", "best_seqs"}:
                sequence = _sequence_mapping(value)
                if sequence is not None:


                    cached = self._cached_object_identity("sequences", value)
                    if cached is not None:
                        return _reference("sequences", cached)
                    reference = self._intern_sequence(sequence)
                    identity = str(reference[_REF_KEY]["id"])
                    self._remember_object_identity("sequences", value, identity)
                    return reference
            return {
                str(item_key): self.normalize(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.normalize(item, key=key) for item in value]
        return value


def normalize_search_artifact(
    *,
    tree: Optional[Mapping[str, Mapping[str, Any]]] = None,
    candidates: Optional[list[Mapping[str, Any]]] = None,
    root: str = "root",
) -> Dict[str, Any]:


    normalizer = _Normalizer()
    nodes = (
        [normalizer.normalize(dict(node)) for node in tree.values()]
        if tree is not None
        else None
    )
    candidate_rows = (
        [normalizer.normalize(dict(candidate)) for candidate in candidates]
        if candidates is not None
        else None
    )
    artifact: Dict[str, Any] = {
        "schema_version": NORMALIZED_SEARCH_ARTIFACT_VERSION,
        "root": str(root),
        "tables": {
            name: dict(sorted(entries.items()))
            for name, entries in normalizer.tables.items()
            if entries
        },
    }
    if nodes is not None:
        artifact["nodes"] = nodes
    if candidate_rows is not None:
        artifact["candidates"] = candidate_rows
    return artifact


def _require_tables(value: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    raw = value.get("tables")
    if not isinstance(raw, Mapping):
        raise NormalizedArtifactError("tables must be a mapping")
    unknown = set(raw) - set(_TABLE_NAMES)
    if unknown:
        raise NormalizedArtifactError(f"unknown normalized tables: {sorted(unknown)}")
    tables: Dict[str, Mapping[str, Any]] = {}
    for name in _TABLE_NAMES:
        table = raw.get(name, {})
        if not isinstance(table, Mapping):
            raise NormalizedArtifactError(f"table {name} must be a mapping")
        tables[name] = table
    return tables


def _verify_tables(tables: Mapping[str, Mapping[str, Any]]) -> None:
    for identity, sequence in tables["sequences"].items():
        normalized = _sequence_mapping(sequence)
        if normalized is None or _digest("astevolve.sequences.v1", normalized) != identity:
            raise NormalizedArtifactError(f"sequence digest mismatch: {identity}")
    for identity, move in tables["moves"].items():
        if _digest("astevolve.search_move.v1", move) != identity:
            raise NormalizedArtifactError(f"move digest mismatch: {identity}")


def rehydrate_search_artifact(value: Mapping[str, Any]) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        raise NormalizedArtifactError("artifact must be a mapping")
    if value.get("schema_version") != NORMALIZED_SEARCH_ARTIFACT_VERSION:
        raise NormalizedArtifactError("unsupported normalized artifact schema")
    allowed = {"schema_version", "root", "tables", "nodes", "candidates"}
    unknown = set(value) - allowed
    if unknown:
        raise NormalizedArtifactError(f"unknown artifact fields: {sorted(unknown)}")
    tables = _require_tables(value)
    _verify_tables(tables)
    active: set[tuple[str, str]] = set()

    def expand(item: Any) -> Any:
        if isinstance(item, Mapping) and set(item) == {_REF_KEY}:
            descriptor = item[_REF_KEY]
            if not isinstance(descriptor, Mapping) or set(descriptor) != {"table", "id"}:
                raise NormalizedArtifactError("malformed normalized reference")
            table = str(descriptor["table"])
            identity = str(descriptor["id"])
            if table not in tables or identity not in tables[table]:
                raise NormalizedArtifactError(f"dangling reference: {table}/{identity}")
            token = (table, identity)
            if token in active:
                raise NormalizedArtifactError(f"cyclic reference: {table}/{identity}")
            active.add(token)
            try:
                expanded = expand(tables[table][identity])
            finally:
                active.remove(token)
            if table == "generation_requests":
                try:
                    SequenceGenerationRequest.from_mapping(expanded)
                except (TypeError, ValueError) as exc:
                    raise NormalizedArtifactError(
                        f"invalid generation request {identity}: {exc}"
                    ) from exc
                if expanded.get("request_hash") != identity:
                    raise NormalizedArtifactError(f"request identity mismatch: {identity}")
            elif table == "generation_results":
                try:
                    SequenceGenerationResult.from_mapping(expanded)
                except (TypeError, ValueError) as exc:
                    raise NormalizedArtifactError(
                        f"invalid generation result {identity}: {exc}"
                    ) from exc
                if expanded.get("result_hash") != identity:
                    raise NormalizedArtifactError(f"result identity mismatch: {identity}")
            elif table == "generation_runtime":
                if not isinstance(expanded, Mapping):
                    raise NormalizedArtifactError(
                        f"generation runtime {identity} must be a mapping"
                    )
                request_raw = expanded.get("request")
                result_raw = expanded.get("result")
                if (
                    expanded.get("schema_version") != _RUNTIME_VERSION
                    or not isinstance(request_raw, Mapping)
                    or not isinstance(result_raw, Mapping)
                ):
                    raise NormalizedArtifactError(
                        f"invalid generation runtime {identity}"
                    )
                try:
                    request = SequenceGenerationRequest.from_mapping(request_raw)
                    result = SequenceGenerationResult.from_mapping(result_raw)
                    validate_generation_result(request, result)
                except (TypeError, ValueError) as exc:
                    raise NormalizedArtifactError(
                        f"invalid generation runtime {identity}: {exc}"
                    ) from exc
                if result.result_hash != identity:
                    raise NormalizedArtifactError(
                        f"generation runtime identity mismatch: {identity}"
                    )
                if expanded.get("selected_generator_id") != result.generator_id:
                    raise NormalizedArtifactError(
                        f"generation runtime generator mismatch: {identity}"
                    )
            return expanded
        if isinstance(item, Mapping):
            return {str(key): expand(child) for key, child in item.items()}
        if isinstance(item, list):
            return [expand(child) for child in item]
        return item


    for table in _TABLE_NAMES:
        for identity in tables[table]:
            expand(_reference(table, str(identity)))

    output: Dict[str, Any] = {
        "schema_version": NORMALIZED_SEARCH_ARTIFACT_VERSION,
        "root": str(value.get("root") or "root"),
    }
    if "nodes" in value:
        output["nodes"] = expand(value["nodes"])
    if "candidates" in value:
        output["candidates"] = expand(value["candidates"])
    return output


__all__ = [
    "NORMALIZED_SEARCH_ARTIFACT_VERSION",
    "NormalizedArtifactError",
    "normalize_search_artifact",
    "rehydrate_search_artifact",
]

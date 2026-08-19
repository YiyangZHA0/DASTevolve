

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence


RESIDUE_EVIDENCE_CATALOG_VERSION = "astevolve.residue_evidence_catalog.v1"
RESIDUE_EVIDENCE_INDEXING = "zero_based"
DEFAULT_RESIDUE_PROMPT_MAX_BYTES = 64 * 1024
MIGRATION_FRONTIER_VERSION = "astevolve.migration_frontier.v1"

_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "indexing",
        "sequences",
        "chains",
        "residues",
        "metadata",
        "catalog_hash",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "evidence_id",
        "chain_id",
        "position",
        "residue",
        "eligible",
        "safety_score",
        "structure",
        "interface",
        "conservation",
        "history",
        "metadata",
    }
)
_SIGNAL_FIELDS = ("structure", "interface", "conservation", "history")
_SAFE_ATOM = re.compile(r"^[A-Za-z0-9_.:+/-]+$")
_PROTEIN_SEQUENCE = re.compile(r"^[A-Z]+$")


class ResidueDesignContextError(ValueError):
    pass


def _strict_fields(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidueDesignContextError(f"{label} must be a mapping")
    keys = {str(key) for key in value}
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown:
        raise ResidueDesignContextError(
            f"{label} contains unknown fields: {', '.join(unknown)}"
        )
    if missing:
        raise ResidueDesignContextError(
            f"{label} is missing fields: {', '.join(missing)}"
        )
    return value


def _text(value: Any, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResidueDesignContextError(f"{label} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ResidueDesignContextError(f"{label} exceeds {maximum} characters")
    return result


def _sequence(value: Any, label: str) -> str:
    result = _text(value, label, maximum=1_000_000).upper()
    if not _PROTEIN_SEQUENCE.fullmatch(result):
        raise ResidueDesignContextError(
            f"{label} must contain contiguous uppercase one-letter residues"
        )
    return result


def _canonical_value(value: Any, label: str) -> Any:


    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ResidueDesignContextError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise ResidueDesignContextError(
                    f"{label} contains a non-string mapping key"
                )
            if key in normalized:
                raise ResidueDesignContextError(f"{label} contains duplicate key {key!r}")
            normalized[key] = _canonical_value(value[key], f"{label}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ResidueDesignContextError(
        f"{label} contains non-JSON value of type {type(value).__name__}"
    )


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
        raise ResidueDesignContextError(
            f"residue evidence is not canonical JSON: {error}"
        ) from error


def _register_sequence(
    sequences: Dict[str, str], chain_id: Any, sequence: Any, label: str
) -> None:
    chain = _text(chain_id, f"{label}.chain_id", maximum=64)
    normalized = _sequence(sequence, f"{label}.sequence")
    previous = sequences.get(chain)
    if previous is not None and previous != normalized:
        raise ResidueDesignContextError(
            f"design_state declares conflicting sequences for chain {chain!r}"
        )
    sequences[chain] = normalized


def _sequence_from_segments(rows: Any, label: str) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ""
    parts = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            raw = row.get("sequence") or row.get("seq")
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            raw = row[2] if len(row) >= 3 else None
        else:
            raw = None
        if raw is None:
            raise ResidueDesignContextError(
                f"{label}[{index}] has no segment sequence"
            )
        parts.append(_sequence(raw, f"{label}[{index}].sequence"))
    return "".join(parts)


def _entity_sequence(entity: Any, label: str) -> tuple[str, str] | None:
    if not isinstance(entity, Mapping):
        return None
    raw_sequence = entity.get("sequence")
    raw_chain = entity.get("chain_id") or entity.get("chain") or entity.get("id")
    if raw_sequence is None or raw_chain is None:
        return None
    return (
        _text(raw_chain, f"{label}.chain_id", maximum=64),
        _sequence(raw_sequence, f"{label}.sequence"),
    )


def _design_state_sequences(design_state: Mapping[str, Any]) -> Dict[str, str]:


    sequences: Dict[str, str] = {}
    for field in ("sequences", "sequence_by_chain"):
        raw = design_state.get(field)
        if isinstance(raw, Mapping):
            for chain_id, value in raw.items():
                sequence = value.get("sequence") if isinstance(value, Mapping) else value
                _register_sequence(sequences, chain_id, sequence, f"design_state.{field}")
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for index, value in enumerate(raw):
                resolved = _entity_sequence(value, f"design_state.{field}[{index}]")
                if resolved is not None:
                    _register_sequence(
                        sequences,
                        resolved[0],
                        resolved[1],
                        f"design_state.{field}[{index}]",
                    )

    raw_chains = design_state.get("chains")
    if isinstance(raw_chains, Mapping):
        for chain_id, value in raw_chains.items():
            sequence = value.get("sequence") if isinstance(value, Mapping) else value
            _register_sequence(sequences, chain_id, sequence, "design_state.chains")
    elif isinstance(raw_chains, Sequence) and not isinstance(
        raw_chains, (str, bytes, bytearray)
    ):
        for index, value in enumerate(raw_chains):
            resolved = _entity_sequence(value, f"design_state.chains[{index}]")
            if resolved is not None:
                _register_sequence(
                    sequences,
                    resolved[0],
                    resolved[1],
                    f"design_state.chains[{index}]",
                )

    binder = design_state.get("binder")
    if isinstance(binder, Mapping):
        chain_id = binder.get("chain_id") or binder.get("chain") or binder.get("id")
        binder_sequence = binder.get("sequence")
        if binder_sequence is None:
            raw_keys = binder.get("domain_segment_keys")
            segment_keys: Dict[str, str] = {}
            if isinstance(raw_keys, Mapping):
                segment_keys = {
                    str(domain): str(key)
                    for domain, key in raw_keys.items()
                    if isinstance(binder.get(str(key)), Sequence)
                }
            if not segment_keys:
                segment_keys = {
                    str(key)[:-9]: str(key)
                    for key, value in binder.items()
                    if isinstance(key, str)
                    and key.endswith("_segments")
                    and isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                }
            raw_order = binder.get("domain_order")
            order = (
                [str(item) for item in raw_order]
                if isinstance(raw_order, Sequence)
                and not isinstance(raw_order, (str, bytes, bytearray))
                else list(segment_keys)
            )
            order.extend(domain for domain in segment_keys if domain not in order)
            parts = []
            for domain in order:
                key = segment_keys.get(domain)
                if key:
                    parts.append(
                        _sequence_from_segments(
                            binder[key], f"design_state.binder.{key}"
                        )
                    )
            binder_sequence = "".join(parts) or None
        if chain_id is not None and binder_sequence is not None:
            _register_sequence(
                sequences, chain_id, binder_sequence, "design_state.binder"
            )

    for field in (
        "target",
        "counter_target",
        "design_protein",
        "target_protein",
        "counter_target_protein",
    ):
        resolved = _entity_sequence(design_state.get(field), f"design_state.{field}")
        if resolved is not None:
            _register_sequence(
                sequences, resolved[0], resolved[1], f"design_state.{field}"
            )
    return dict(sorted(sequences.items()))


def _catalog_sequences(
    raw: Mapping[str, Any],
    *,
    expected_sequences: Mapping[str, str],
) -> Dict[str, str]:
    if raw.get("sequences") is not None and raw.get("chains") is not None:
        raise ResidueDesignContextError(
            "residue_evidence_catalog must not declare both sequences and chains"
        )
    sequences: Dict[str, str] = {}
    raw_sequences = raw.get("sequences")
    raw_chains = raw.get("chains")
    if raw_sequences is not None:
        if not isinstance(raw_sequences, Mapping) or not raw_sequences:
            raise ResidueDesignContextError(
                "residue_evidence_catalog.sequences must be a non-empty mapping"
            )
        for chain_id, value in raw_sequences.items():
            sequence = value.get("sequence") if isinstance(value, Mapping) else value
            _register_sequence(sequences, chain_id, sequence, "catalog.sequences")
    elif raw_chains is not None:
        if isinstance(raw_chains, Mapping):
            items = list(raw_chains.items())
        elif isinstance(raw_chains, Sequence) and not isinstance(
            raw_chains, (str, bytes, bytearray)
        ):
            items = []
            for index, value in enumerate(raw_chains):
                resolved = _entity_sequence(value, f"catalog.chains[{index}]")
                if resolved is None:
                    raise ResidueDesignContextError(
                        f"catalog.chains[{index}] requires chain_id and sequence"
                    )
                items.append(resolved)
        else:
            raise ResidueDesignContextError(
                "residue_evidence_catalog.chains must be a mapping or list"
            )
        for chain_id, value in items:
            sequence = value.get("sequence") if isinstance(value, Mapping) else value
            _register_sequence(sequences, chain_id, sequence, "catalog.chains")
    else:
        raw_residues = raw.get("residues")
        chain_ids = sorted(
            {
                str(item.get("chain_id") or "").strip()
                for item in raw_residues or []
                if isinstance(item, Mapping) and str(item.get("chain_id") or "").strip()
            }
        )
        if not chain_ids:
            raise ResidueDesignContextError(
                "residue_evidence_catalog has no declared or inferable chains"
            )
        for chain_id in chain_ids:
            if chain_id not in expected_sequences:
                raise ResidueDesignContextError(
                    f"catalog chain {chain_id!r} has no sequence in design_state"
                )
            sequences[chain_id] = expected_sequences[chain_id]

    for chain_id, sequence in sequences.items():
        expected = expected_sequences.get(chain_id)
        if expected is not None and expected != sequence:
            raise ResidueDesignContextError(
                f"catalog sequence for chain {chain_id!r} does not match design_state"
            )
        if expected_sequences and expected is None:
            raise ResidueDesignContextError(
                f"catalog chain {chain_id!r} is absent from design_state sequences"
            )
    return dict(sorted(sequences.items()))


def _signal(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidueDesignContextError(f"{label} must be a mapping")
    if not isinstance(value.get("available"), bool):
        raise ResidueDesignContextError(f"{label}.available must be boolean")
    normalized = _canonical_value(value, label)
    assert isinstance(normalized, dict)
    return normalized


def _normalize_record(
    value: Any,
    *,
    index: int,
    sequences: Mapping[str, str],
) -> Dict[str, Any]:
    label = f"residue_evidence_catalog.residues[{index}]"
    raw = _strict_fields(
        value,
        label=label,
        allowed=_RECORD_FIELDS,
        required=frozenset(
            {
                "evidence_id",
                "chain_id",
                "position",
                "residue",
                "eligible",
                "safety_score",
                "structure",
                "interface",
                "conservation",
                "history",
            }
        ),
    )
    evidence_id = _text(raw["evidence_id"], f"{label}.evidence_id", maximum=256)
    chain_id = _text(raw["chain_id"], f"{label}.chain_id", maximum=64)
    if chain_id not in sequences:
        raise ResidueDesignContextError(
            f"{label}.chain_id {chain_id!r} is not declared by the catalog"
        )
    position = raw["position"]
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ResidueDesignContextError(f"{label}.position must be an integer >= 0")
    sequence = sequences[chain_id]
    if position >= len(sequence):
        raise ResidueDesignContextError(
            f"{label}.position {position} is outside chain {chain_id!r} length {len(sequence)}"
        )
    residue = _sequence(raw["residue"], f"{label}.residue")
    if len(residue) != 1:
        raise ResidueDesignContextError(f"{label}.residue must contain exactly one residue")
    expected = sequence[position]
    if residue != expected:
        raise ResidueDesignContextError(
            f"{label}.residue {residue!r} does not match chain {chain_id!r} "
            f"position {position} ({expected!r})"
        )
    if not isinstance(raw["eligible"], bool):
        raise ResidueDesignContextError(f"{label}.eligible must be boolean")
    safety = raw["safety_score"]
    if isinstance(safety, bool) or not isinstance(safety, (int, float)):
        raise ResidueDesignContextError(f"{label}.safety_score must be numeric")
    safety = float(safety)
    if not isfinite(safety) or not 0.0 <= safety <= 1.0:
        raise ResidueDesignContextError(
            f"{label}.safety_score must be finite and in [0, 1]"
        )
    normalized: Dict[str, Any] = {
        "evidence_id": evidence_id,
        "chain_id": chain_id,
        "position": int(position),
        "residue": residue,
        "eligible": bool(raw["eligible"]),
        "safety_score": safety,
    }
    for field in _SIGNAL_FIELDS:
        normalized[field] = _signal(raw[field], f"{label}.{field}")
    normalized["metadata"] = _canonical_value(raw.get("metadata", {}), f"{label}.metadata")
    if not isinstance(normalized["metadata"], dict):
        raise ResidueDesignContextError(f"{label}.metadata must be a mapping")
    return normalized


def _normalize_catalog_payload(
    raw_catalog: Any,
    *,
    expected_sequences: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    raw = _strict_fields(
        raw_catalog,
        label="residue_evidence_catalog",
        allowed=_CATALOG_FIELDS,
        required=frozenset({"schema_version", "residues"}),
    )
    if raw.get("schema_version") != RESIDUE_EVIDENCE_CATALOG_VERSION:
        raise ResidueDesignContextError(
            "unsupported residue_evidence_catalog schema_version: "
            f"{raw.get('schema_version')!r}"
        )
    indexing = raw.get("indexing", RESIDUE_EVIDENCE_INDEXING)
    if indexing != RESIDUE_EVIDENCE_INDEXING:
        raise ResidueDesignContextError(
            "residue_evidence_catalog.indexing must be exactly 'zero_based'"
        )
    expected = {
        str(chain_id): _sequence(sequence, f"expected_sequences.{chain_id}")
        for chain_id, sequence in (expected_sequences or {}).items()
    }
    sequences = _catalog_sequences(raw, expected_sequences=expected)
    raw_records = raw.get("residues")
    if not isinstance(raw_records, Sequence) or isinstance(
        raw_records, (str, bytes, bytearray)
    ):
        raise ResidueDesignContextError(
            "residue_evidence_catalog.residues must be a list"
        )
    if not raw_records:
        raise ResidueDesignContextError(
            "residue_evidence_catalog.residues must not be empty"
        )
    records = [
        _normalize_record(item, index=index, sequences=sequences)
        for index, item in enumerate(raw_records)
    ]
    evidence_ids: set[str] = set()
    positions: set[tuple[str, int]] = set()
    for record in records:
        evidence_id = record["evidence_id"]
        position_key = (record["chain_id"], record["position"])
        if evidence_id in evidence_ids:
            raise ResidueDesignContextError(
                f"residue_evidence_catalog contains duplicate evidence_id {evidence_id!r}"
            )
        if position_key in positions:
            raise ResidueDesignContextError(
                "residue_evidence_catalog contains duplicate residue position "
                f"{position_key[0]}:{position_key[1]}"
            )
        evidence_ids.add(evidence_id)
        positions.add(position_key)

    for chain_id, sequence in sequences.items():
        observed = {position for chain, position in positions if chain == chain_id}
        required = set(range(len(sequence)))
        missing = sorted(required - observed)
        if missing:
            preview = missing[:12]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ResidueDesignContextError(
                f"residue_evidence_catalog does not cover full chain {chain_id!r}; "
                f"missing positions {preview}{suffix}"
            )

    records.sort(key=lambda item: (item["chain_id"], item["position"]))
    metadata = _canonical_value(raw.get("metadata", {}), "catalog.metadata")
    if not isinstance(metadata, dict):
        raise ResidueDesignContextError("residue_evidence_catalog.metadata must be a mapping")
    return {
        "schema_version": RESIDUE_EVIDENCE_CATALOG_VERSION,
        "indexing": RESIDUE_EVIDENCE_INDEXING,
        "sequences": sequences,
        "residues": records,
        "metadata": metadata,
    }


def canonical_catalog_hash(
    raw_catalog: Mapping[str, Any],
    *,
    expected_sequences: Mapping[str, str] | None = None,
) -> str:


    payload = _normalize_catalog_payload(
        raw_catalog, expected_sequences=expected_sequences
    )
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_residue_evidence_catalog(
    raw_catalog: Mapping[str, Any],
    *,
    expected_sequences: Mapping[str, str] | None = None,
) -> Dict[str, Any]:


    payload = _normalize_catalog_payload(
        raw_catalog, expected_sequences=expected_sequences
    )
    computed = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    supplied = raw_catalog.get("catalog_hash")
    if supplied not in (None, ""):
        if not isinstance(supplied, str):
            raise ResidueDesignContextError(
                "residue_evidence_catalog.catalog_hash must be a SHA-256 string"
            )
        normalized_supplied = supplied.strip().lower()
        if normalized_supplied.startswith("sha256:"):
            normalized_supplied = normalized_supplied[7:]
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_supplied):
            raise ResidueDesignContextError(
                "residue_evidence_catalog.catalog_hash must contain 64 hex characters"
            )
        if normalized_supplied != computed:
            raise ResidueDesignContextError(
                "residue_evidence_catalog.catalog_hash does not match canonical content"
            )
    return {**payload, "catalog_hash": computed}


def _catalog_source_path(
    source: str | Path,
    *,
    design_state: Mapping[str, Any],
    design_state_path: str | Path | None,
) -> Path:
    path = Path(source).expanduser()
    if path.is_absolute():
        return path.resolve()
    raw_base = design_state_path or design_state.get("_path")
    if raw_base:
        base = Path(str(raw_base)).expanduser().resolve()
        base = base if base.is_dir() else base.parent
    else:
        base = Path.cwd()
    return (base / path).resolve()


def load_residue_evidence_catalog(
    design_state: Mapping[str, Any],
    *,
    design_state_path: str | Path | None = None,
) -> Dict[str, Any]:


    if not isinstance(design_state, Mapping):
        raise ResidueDesignContextError("design_state must be a mapping")
    source = design_state.get("residue_evidence_catalog")
    if source is None:
        raise ResidueDesignContextError(
            "design_state.residue_evidence_catalog is required"
        )
    raw_catalog: Any
    if isinstance(source, (str, Path)):
        path = _catalog_source_path(
            source,
            design_state=design_state,
            design_state_path=design_state_path,
        )
        try:
            raw_catalog = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResidueDesignContextError(
                f"cannot load residue evidence catalog JSON: {path}"
            ) from error
    elif isinstance(source, Mapping) and set(source) == {"path"}:
        path = _catalog_source_path(
            source["path"],
            design_state=design_state,
            design_state_path=design_state_path,
        )
        try:
            raw_catalog = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResidueDesignContextError(
                f"cannot load residue evidence catalog JSON: {path}"
            ) from error
    elif isinstance(source, Mapping):
        raw_catalog = source
    else:
        raise ResidueDesignContextError(
            "design_state.residue_evidence_catalog must be an inline mapping or JSON path"
        )
    expected_sequences = _design_state_sequences(design_state)
    return normalize_residue_evidence_catalog(
        raw_catalog, expected_sequences=expected_sequences
    )


def build_residue_evidence_state_patch(
    design_state: Mapping[str, Any],
    *,
    design_state_path: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:


    return {
        "_residue_evidence_catalog": load_residue_evidence_catalog(
            design_state, design_state_path=design_state_path
        )
    }


def _availability(signal: Mapping[str, Any]) -> int:
    return int(bool(signal.get("available")))


def _history_count(history: Mapping[str, Any]) -> int:
    for key in ("observations", "events", "attempts", "mutations"):
        value = history.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return len(value)
    for key in ("observation_count", "event_count", "count"):
        value = history.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _format_score(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == 1.0:
        return "1"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _atom(value: str) -> str:
    return value if _SAFE_ATOM.fullmatch(value) else json.dumps(value, ensure_ascii=True)


def _flatten_signal(value: Any, *, prefix: str = "") -> list[str]:
    tokens: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            if key == "available":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            child = value[key]
            if isinstance(child, Mapping):
                tokens.extend(_flatten_signal(child, prefix=path))
            elif isinstance(child, Sequence) and not isinstance(
                child, (str, bytes, bytearray)
            ):
                scalars = all(
                    item is None or isinstance(item, (str, bool, int, float))
                    for item in child
                )
                encoded = _canonical_json(child)
                if scalars and len(encoded) <= 48:
                    tokens.append(f"{path}={encoded}")
                else:
                    digest = sha256(encoded.encode("utf-8")).hexdigest()[:8]
                    tokens.append(f"{path}.n={len(child)}@{digest}")
            else:
                tokens.append(f"{path}={_canonical_json(child)}")
    return tokens


def _compact_signal(signal: Mapping[str, Any], budget: int) -> str:
    if budget <= 0:
        return ""
    tokens = _flatten_signal(signal)
    if not tokens:
        return ""
    full = ",".join(tokens)
    if len(full.encode("utf-8")) <= budget:
        return full
    marker = "~" + sha256(full.encode("utf-8")).hexdigest()[:8]
    if len(marker) > budget:
        return ""
    selected: list[str] = []
    for token in tokens:
        candidate = ",".join([*selected, token, marker])
        if len(candidate.encode("utf-8")) > budget:
            break
        selected.append(token)
    return ",".join([*selected, marker])


def _current_parent_sequences(
    catalog: Mapping[str, Any],
    current_parent_sequences: Mapping[str, Any] | None,
) -> tuple[Dict[str, str], str]:


    catalog_sequences = {
        str(chain_id): str(sequence)
        for chain_id, sequence in (catalog.get("sequences") or {}).items()
    }
    if current_parent_sequences is None:
        resolved = dict(catalog_sequences)
    else:
        if not isinstance(current_parent_sequences, Mapping):
            raise ResidueDesignContextError(
                "current_parent_sequences must be a mapping"
            )
        resolved = {}
        for chain_id, canonical_sequence in catalog_sequences.items():
            if chain_id not in current_parent_sequences:
                raise ResidueDesignContextError(
                    f"current parent sequence is missing catalog chain {chain_id!r}"
                )
            parent_sequence = _sequence(
                current_parent_sequences[chain_id],
                f"current_parent_sequences.{chain_id}",
            )
            if len(parent_sequence) != len(canonical_sequence):
                raise ResidueDesignContextError(
                    f"current parent sequence for chain {chain_id!r} has length "
                    f"{len(parent_sequence)}, expected {len(canonical_sequence)}"
                )
            resolved[chain_id] = parent_sequence
    parent_hash = sha256(
        _canonical_json(dict(sorted(resolved.items()))).encode("utf-8")
    ).hexdigest()
    return dict(sorted(resolved.items())), parent_hash


def _render_prompt_digest(
    catalog: Mapping[str, Any],
    detail_budget: int,
    *,
    current_parent_sequences: Mapping[str, str],
    current_parent_sequence_hash: str,
) -> str:
    chains = ",".join(
        f"{_atom(chain_id)}:{len(sequence)}"
        for chain_id, sequence in sorted(catalog["sequences"].items())
    )
    lines = [
        "RESIDUE_EVIDENCE "
        f"schema={catalog['schema_version']} hash={catalog['catalog_hash']} "
        f"parent_hash={current_parent_sequence_hash} "
        f"index=0 chains={chains} "
        "legend=cat:catalog_or_canonical_residue,cur:trusted_current_parent_residue,"
        "e:eligible,s:safety,st/if/cv:available,h:available/count"
    ]
    labels = {
        "structure": "st",
        "interface": "if",
        "conservation": "cv",
        "history": "h",
    }
    for record in catalog["residues"]:
        history = record["history"]
        chain_id = str(record["chain_id"])
        position = int(record["position"])
        current_residue = current_parent_sequences[chain_id][position]
        line = (
            f"{_atom(chain_id)}:{position} "
            f"cat={_atom(record['residue'])} cur={_atom(current_residue)} "
            f"id={_atom(record['evidence_id'])} e={int(record['eligible'])} "
            f"s={_format_score(record['safety_score'])} "
            f"st={_availability(record['structure'])} "
            f"if={_availability(record['interface'])} "
            f"cv={_availability(record['conservation'])} "
            f"h={_availability(history)}/{_history_count(history)}"
        )
        for field in _SIGNAL_FIELDS:
            details = _compact_signal(record[field], detail_budget)
            if details:
                line += f" {labels[field]}[{details}]"
        lines.append(line)
    return "\n".join(lines)


def build_residue_prompt_digest(
    catalog: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_RESIDUE_PROMPT_MAX_BYTES,
    max_detail_bytes_per_signal: int = 96,
    current_parent_sequences: Mapping[str, Any] | None = None,
) -> str:


    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ResidueDesignContextError("max_bytes must be an integer > 0")
    if (
        isinstance(max_detail_bytes_per_signal, bool)
        or not isinstance(max_detail_bytes_per_signal, int)
        or max_detail_bytes_per_signal < 0
    ):
        raise ResidueDesignContextError(
            "max_detail_bytes_per_signal must be an integer >= 0"
        )
    normalized = normalize_residue_evidence_catalog(catalog)
    parent_sequences, parent_hash = _current_parent_sequences(
        normalized, current_parent_sequences
    )
    minimal = _render_prompt_digest(
        normalized,
        0,
        current_parent_sequences=parent_sequences,
        current_parent_sequence_hash=parent_hash,
    )
    required = len(minimal.encode("utf-8"))
    if required > max_bytes:
        raise ResidueDesignContextError(
            "residue prompt byte budget cannot preserve full-chain coverage: "
            f"requires at least {required} bytes, received {max_bytes}"
        )
    for detail_budget in range(max_detail_bytes_per_signal, -1, -1):
        rendered = _render_prompt_digest(
            normalized,
            detail_budget,
            current_parent_sequences=parent_sequences,
            current_parent_sequence_hash=parent_hash,
        )
        if len(rendered.encode("utf-8")) <= max_bytes:
            if len(rendered.splitlines()) != len(normalized["residues"]) + 1:
                raise ResidueDesignContextError(
                    "residue prompt digest lost full-chain line coverage"
                )
            return rendered
    raise ResidueDesignContextError(
        "residue prompt byte budget cannot preserve full-chain coverage"
    )


def _state_segment_positions(
    design_state: Mapping[str, Any],
) -> Dict[tuple[str, int], str]:


    binder = design_state.get("binder")
    if not isinstance(binder, Mapping):
        return {}
    chain_id = str(
        binder.get("chain_id") or binder.get("chain") or binder.get("id") or ""
    ).strip()
    if not chain_id:
        return {}
    raw_keys = binder.get("domain_segment_keys")
    segment_keys: Dict[str, str] = {}
    if isinstance(raw_keys, Mapping):
        segment_keys = {
            str(domain): str(key)
            for domain, key in raw_keys.items()
            if isinstance(binder.get(str(key)), Sequence)
            and not isinstance(binder.get(str(key)), (str, bytes, bytearray))
        }
    if not segment_keys:
        segment_keys = {
            str(key)[:-9]: str(key)
            for key, value in binder.items()
            if isinstance(key, str)
            and key.endswith("_segments")
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        }
    raw_order = binder.get("domain_order")
    order = (
        [str(item) for item in raw_order]
        if isinstance(raw_order, Sequence)
        and not isinstance(raw_order, (str, bytes, bytearray))
        else list(segment_keys)
    )
    order.extend(domain for domain in segment_keys if domain not in order)
    positions: Dict[tuple[str, int], str] = {}
    cursor = 0
    for domain in order:
        key = segment_keys.get(domain)
        if not key:
            continue
        for index, row in enumerate(binder[key]):
            if isinstance(row, Mapping):
                raw_name = row.get("segment_id") or row.get("name") or row.get("id")
                raw_sequence = row.get("sequence") or row.get("seq")
            elif isinstance(row, Sequence) and not isinstance(
                row, (str, bytes, bytearray)
            ):
                raw_name = row[0] if len(row) >= 1 else None
                raw_sequence = row[2] if len(row) >= 3 else None
            else:
                raw_name = None
                raw_sequence = None
            if raw_name is None or raw_sequence is None:
                raise ResidueDesignContextError(
                    f"design_state.binder.{key}[{index}] requires segment ID and sequence"
                )
            segment_id = _text(
                raw_name,
                f"design_state.binder.{key}[{index}].segment_id",
                maximum=256,
            )
            sequence = _sequence(
                raw_sequence,
                f"design_state.binder.{key}[{index}].sequence",
            )
            for position in range(cursor, cursor + len(sequence)):
                positions[(chain_id, position)] = segment_id
            cursor += len(sequence)
    return positions


def _record_segment(record: Mapping[str, Any]) -> str | None:
    for container_name in ("structure", "metadata"):
        container = record.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in ("compiled_segment", "structural_node", "segment_id"):
            value = str(container.get(key) or "").strip()
            if value:
                return value
    return None


def _span_positions(value: Any) -> Dict[str, set[int]]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, set[int]] = {}
    for raw_chain, raw_spans in value.items():
        chain_id = str(raw_chain).strip()
        if not chain_id or not isinstance(raw_spans, Sequence) or isinstance(
            raw_spans, (str, bytes, bytearray)
        ):
            continue
        selected: set[int] = set()
        for span in raw_spans:
            if (
                not isinstance(span, Sequence)
                or isinstance(span, (str, bytes, bytearray))
                or len(span) != 2
            ):
                continue
            start, end = span
            if (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end
            ):
                selected.update(range(start, end))
        output[chain_id] = selected
    return output


def _active_ast_positions(raw_ast: Any) -> Dict[tuple[str, int], str]:
    if not isinstance(raw_ast, Mapping):
        return {}
    output: Dict[tuple[str, int], str] = {}
    for node in raw_ast.get("structural_nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("kind") != "editable":
            continue
        node_id = str(node.get("node_id") or "").strip()
        selector = node.get("selector")
        if not node_id or not isinstance(selector, Mapping):
            continue
        chain_id = str(selector.get("chain_id") or "").strip()
        for position in _span_positions({chain_id: selector.get("spans")}).get(
            chain_id, set()
        ):
            output[(chain_id, position)] = node_id
    return output


def _case_owned_frontier_alphabets(
    design_state: Mapping[str, Any],
) -> tuple[str | None, Dict[tuple[str, int], tuple[str, ...]]]:


    policy = design_state.get("case_owned_residue_policy")
    if not isinstance(policy, Mapping):
        return None, {}
    tiers = policy.get("tiers")
    if not isinstance(tiers, Mapping) or not tiers:
        return None, {}
    environment_name = str(policy.get("tier_environment_variable") or "").strip()
    requested = (
        str(os.environ.get(environment_name) or "").strip()
        if environment_name
        else ""
    )
    tier_name = requested or str(policy.get("default_tier") or "").strip()
    tier = tiers.get(tier_name)
    if not tier_name or not isinstance(tier, Mapping):
        raise ResidueDesignContextError(
            "case_owned_residue_policy selected an unavailable tier: "
            + repr(tier_name)
        )
    raw_contract = tier.get("residue_mutation_contract")
    if not isinstance(raw_contract, Mapping):
        raise ResidueDesignContextError(
            f"case_owned_residue_policy tier {tier_name!r} has no residue contract"
        )
    output: Dict[tuple[str, int], tuple[str, ...]] = {}
    for raw_chain, raw_rows in raw_contract.items():
        chain_id = str(raw_chain)
        if not isinstance(raw_rows, Mapping):
            raise ResidueDesignContextError(
                f"case-owned residue contract chain {chain_id!r} must be a mapping"
            )
        for raw_position, raw_alphabet in raw_rows.items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as error:
                raise ResidueDesignContextError(
                    "case-owned residue contract position must be an integer"
                ) from error
            if (
                isinstance(raw_alphabet, (str, bytes, bytearray))
                or not isinstance(raw_alphabet, Sequence)
                or not raw_alphabet
            ):
                raise ResidueDesignContextError(
                    f"case-owned residue alphabet is empty at {chain_id}:{position}"
                )
            residues = tuple(dict.fromkeys(str(item) for item in raw_alphabet))
            if any(len(item) != 1 or not item.isalpha() or not item.isupper() for item in residues):
                raise ResidueDesignContextError(
                    f"case-owned residue alphabet is invalid at {chain_id}:{position}"
                )
            output[(chain_id, position)] = residues
    return tier_name, output


def build_migration_frontier(
    design_state: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    current_parent_sequences: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    normalized = normalize_residue_evidence_catalog(catalog)
    parent_sequences, parent_hash = _current_parent_sequences(
        normalized,
        current_parent_sequences
        if current_parent_sequences is not None
        else design_state.get("_current_parent_sequences"),
    )
    policy = design_state.get("global_ast_evolution_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    allowed = _span_positions(policy.get("allowed_chain_spans"))
    protected = _span_positions(policy.get("protected_chain_spans"))
    raw_minimum = policy.get("min_position_safety_score", 0.0)
    minimum_safety = (
        float(raw_minimum)
        if isinstance(raw_minimum, (int, float)) and not isinstance(raw_minimum, bool)
        else 0.0
    )
    derived_segments = _state_segment_positions(design_state)
    active_positions = _active_ast_positions(design_state.get("executable_dual_ast"))
    resolved_tier, frontier_alphabets = _case_owned_frontier_alphabets(design_state)
    compiled_segment_by_position: Dict[str, str] = {}
    groups: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for record in normalized["residues"]:
        chain_id = str(record["chain_id"])
        position = int(record["position"])
        segment_id = _record_segment(record) or derived_segments.get(
            (chain_id, position)
        )
        if not segment_id:
            segment_id = f"{chain_id}:whole_chain"
        compiled_segment_by_position[f"{chain_id}:{position}"] = segment_id
        if allowed and position not in allowed.get(chain_id, set()):
            continue
        if position in protected.get(chain_id, set()):
            continue
        if not bool(record.get("eligible")) or float(record.get("safety_score", 0.0)) < minimum_safety:
            continue
        hard_alphabet = frontier_alphabets.get((chain_id, position))


        if resolved_tier is not None and hard_alphabet is None:
            continue
        structure = record.get("structure")
        structure = structure if isinstance(structure, Mapping) else {}
        interface = record.get("interface")
        interface = interface if isinstance(interface, Mapping) else {}
        a_contacts = interface.get("A_heavy_atom_contacts")
        b_contacts = interface.get("B_heavy_atom_contacts")
        contact_delta = (
            float(a_contacts) - float(b_contacts)
            if isinstance(a_contacts, (int, float))
            and not isinstance(a_contacts, bool)
            and isinstance(b_contacts, (int, float))
            and not isinstance(b_contacts, bool)
            else None
        )
        groups.setdefault((chain_id, segment_id), []).append(
            {
                "position": position,
                "catalog_residue": record["residue"],
                "current_parent_residue": parent_sequences[chain_id][position],


                "residue": record["residue"],
                "evidence_id": record["evidence_id"],
                "safety_score": float(record["safety_score"]),
                "relative_sasa": structure.get("relative_sasa"),
                "A_heavy_atom_contacts": a_contacts,
                "B_heavy_atom_contacts": b_contacts,
                "A_minus_B_contacts": contact_delta,
                "variant_confounded": interface.get("variant_confounded"),
                "incumbent_node_id": active_positions.get((chain_id, position)),
                "hard_allowed_residues": list(hard_alphabet or ()),
                "case_owned_residue_tier": resolved_tier,
            }
        )

    grouped: list[Dict[str, Any]] = []
    for (chain_id, segment_id), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[1][0]["position"])
    ):
        rows.sort(key=lambda item: item["position"])
        incumbent_nodes = sorted(
            {
                str(row["incumbent_node_id"])
                for row in rows
                if row.get("incumbent_node_id")
            }
        )
        contact_signal = sum(
            abs(float(row["A_minus_B_contacts"]))
            for row in rows
            if row.get("A_minus_B_contacts") is not None
        )
        grouped.append(
            {
                "chain_id": chain_id,
                "compiled_segment": segment_id,
                "incumbent": bool(incumbent_nodes),
                "incumbent_node_ids": incumbent_nodes,
                "legal_position_count": len(rows),
                "contact_difference_signal": contact_signal,
                "positions": rows,
            }
        )
    incumbent_segments = sorted(
        f"{item['chain_id']}:{item['compiled_segment']}"
        for item in grouped
        if item["incumbent"]
    )
    new_segments = [
        item for item in grouped if not item["incumbent"] and item["legal_position_count"]
    ]
    new_segments.sort(
        key=lambda item: (
            -float(item["contact_difference_signal"]),
            -max(float(row["safety_score"]) for row in item["positions"]),
            item["chain_id"],
            item["compiled_segment"],
        )
    )
    return {
        "schema_version": MIGRATION_FRONTIER_VERSION,
        "catalog_hash": normalized["catalog_hash"],
        "current_parent_sequence_hash": parent_hash,
        "indexing": RESIDUE_EVIDENCE_INDEXING,
        "constraints": {
            "min_active_positions": policy.get("min_total_editable_positions"),
            "max_active_positions": policy.get("max_total_editable_positions"),
            "node_must_stay_within_one_compiled_segment": True,
            "stagnation_new_segment_count": 1,
            "stagnation_retain_incumbent_anchor": True,
            "case_owned_residue_tier": resolved_tier,
            "prospective_positions_require_hard_alphabet": bool(
                resolved_tier is not None
            ),
        },
        "incumbent_segments": incumbent_segments,
        "ranked_new_segments": [
            f"{item['chain_id']}:{item['compiled_segment']}" for item in new_segments
        ],
        "residue_identity_by_position": {
            f"{record['chain_id']}:{record['position']}": {
                "catalog_residue": record["residue"],
                "current_parent_residue": parent_sequences[str(record["chain_id"])][
                    int(record["position"])
                ],
            }
            for record in normalized["residues"]
        },
        "segments": grouped,
        "compiled_segment_by_position": dict(sorted(compiled_segment_by_position.items())),
    }


def build_migration_frontier_prompt_digest(frontier: Mapping[str, Any]) -> str:


    if not isinstance(frontier, Mapping) or frontier.get("schema_version") != MIGRATION_FRONTIER_VERSION:
        raise ResidueDesignContextError("migration frontier has an unsupported schema")
    return "MIGRATION_FRONTIER " + _canonical_json(frontier)


def attach_residue_evidence_context(
    state: Mapping[str, Any],
    *,
    design_state_path: str | Path | None = None,
    current_parent_sequences: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    updated = deepcopy(dict(state))
    if updated.get("residue_evidence_catalog") is None:
        return updated
    updated.update(
        build_residue_evidence_state_patch(
            updated,
            design_state_path=design_state_path,
        )
    )
    parent_sequences, parent_hash = _current_parent_sequences(
        updated["_residue_evidence_catalog"], current_parent_sequences
    )
    updated["_current_parent_sequences"] = parent_sequences
    updated["_current_parent_sequence_hash"] = parent_hash
    updated = refresh_migration_frontier_context(updated)
    residue_digest = build_residue_prompt_digest(
        updated["_residue_evidence_catalog"],
        max_bytes=int(
            updated.get(
                "residue_evidence_prompt_max_bytes",
                DEFAULT_RESIDUE_PROMPT_MAX_BYTES,
            )
        ),
        current_parent_sequences=parent_sequences,
    )
    updated["_residue_evidence_prompt_digest"] = residue_digest
    return updated


def refresh_migration_frontier_context(state: Mapping[str, Any]) -> Dict[str, Any]:


    updated = deepcopy(dict(state))
    catalog = updated.get("_residue_evidence_catalog")
    if not isinstance(catalog, Mapping):
        return updated
    frontier = build_migration_frontier(
        updated,
        catalog,
        current_parent_sequences=updated.get("_current_parent_sequences"),
    )
    updated["_migration_frontier"] = frontier
    updated["_compiled_segment_by_position"] = deepcopy(
        frontier["compiled_segment_by_position"]
    )
    return updated


__all__ = [
    "DEFAULT_RESIDUE_PROMPT_MAX_BYTES",
    "MIGRATION_FRONTIER_VERSION",
    "RESIDUE_EVIDENCE_CATALOG_VERSION",
    "RESIDUE_EVIDENCE_INDEXING",
    "ResidueDesignContextError",
    "attach_residue_evidence_context",
    "build_residue_evidence_state_patch",
    "build_migration_frontier",
    "build_migration_frontier_prompt_digest",
    "build_residue_prompt_digest",
    "canonical_catalog_hash",
    "load_residue_evidence_catalog",
    "normalize_residue_evidence_catalog",
    "refresh_migration_frontier_context",
]

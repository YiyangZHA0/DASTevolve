

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_design_state_path() -> Path:


    from astevolve.cases import resolve_case

    return resolve_case().design_state_path


def state_domain_segment_keys(state: Dict[str, Any]) -> Dict[str, str]:


    binder = state.get("binder", {})
    custom = binder.get("domain_segment_keys")
    if isinstance(custom, dict) and custom:
        out: Dict[str, str] = {}
        for domain_name, segment_key in custom.items():
            if isinstance(segment_key, str) and isinstance(binder.get(segment_key), list):
                out[str(domain_name)] = segment_key
        if out:
            return out

    inferred: Dict[str, str] = {}
    for key, value in binder.items():
        if isinstance(key, str) and key.endswith("_segments") and isinstance(value, list):
            inferred[key[:-9]] = key
    return inferred


def state_domain_aliases(state: Dict[str, Any]) -> Dict[str, str]:


    raw = state.get("binder", {}).get("domain_aliases", {})
    if not isinstance(raw, dict):
        return {}
    segment_keys = state_domain_segment_keys(state)
    aliases: Dict[str, str] = {}
    for alias, canonical in raw.items():
        alias_text = str(alias)
        canonical_text = str(canonical)
        if canonical_text in segment_keys:
            aliases[alias_text] = canonical_text
    return aliases


def load_design_state(path: str | Path | None = None) -> Dict[str, Any]:


    state_path = Path(path) if path else default_design_state_path()
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    data = json.loads(state_path.read_text(encoding="utf-8"))
    validate_design_state(data)
    data["_path"] = str(state_path)
    return data


def validate_design_state(state: Dict[str, Any]) -> None:


    for key in ("task_name", "binder", "mutation_policy"):
        if key not in state:
            raise ValueError(f"design_state missing required key: {key}")

    target = state.get("target")
    if has_target(state):
        assert isinstance(target, dict)
        seq = str(target.get("sequence") or "")
    else:
        single_chain = bool(state.get("single_chain")) or "single_chain" in str(
            state.get("task_type") or ""
        ).lower()
        if not single_chain:
            raise ValueError(
                "design_state requires a non-empty target unless single_chain is explicit"
            )
        seq = ""

    if not state_domain_segment_keys(state):
        raise ValueError("design_state binder has no domain_segment_keys or *_segments lists")

    target_spec = target if isinstance(target, dict) else {}
    for span in target_spec.get("epitope_spans", []):
        if len(span) != 2:
            raise ValueError(f"Invalid epitope span: {span}")
        start, end = int(span[0]), int(span[1])
        if not (0 <= start < end <= len(seq)):
            raise ValueError(f"Epitope span out of range: {span}")


def has_target(state: Dict[str, Any]) -> bool:


    target = state.get("target")
    return (
        isinstance(target, dict)
        and target.get("enabled") is not False
        and bool(str(target.get("sequence") or ""))
    )


def segment_spans(parts: List[List[str]]) -> Dict[str, Tuple[int, int]]:


    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, _kind, seq in parts:
        spans[name] = (cursor, cursor + len(seq))
        cursor += len(seq)
    return spans


def binder_domain_order(state: Dict[str, Any]) -> List[str]:


    binder = state["binder"]
    segment_keys = state_domain_segment_keys(state)
    fallback_order = list(segment_keys.keys())
    raw_order = binder.get("domain_order", fallback_order)
    if not isinstance(raw_order, list):
        raw_order = fallback_order

    aliases = state_domain_aliases(state)
    order: List[str] = []
    for item in raw_order:
        text = str(item)
        key = aliases.get(text, text)
        if key in segment_keys and key not in order:
            order.append(key)

    for key in fallback_order:
        if key not in order:
            order.append(key)
    return order


def flatten_binder_parts(state: Dict[str, Any]) -> List[List[str]]:


    binder = state["binder"]
    segment_keys = state_domain_segment_keys(state)
    parts: List[List[str]] = []
    for domain_name in binder_domain_order(state):
        key = segment_keys.get(domain_name)
        if key:
            parts.extend(list(binder[key]))
    return parts


def binder_sequence(state: Dict[str, Any]) -> str:


    return "".join(part[2] for part in flatten_binder_parts(state))

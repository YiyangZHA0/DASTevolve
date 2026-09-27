

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:


    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:


    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def clamp01(value: Any) -> float:


    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def mean(values: Iterable[float]) -> Optional[float]:


    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def score_from_100(value: Any) -> float:


    numeric = safe_float(value)
    if numeric is None:
        return 0.0
    return clamp01(numeric / 100.0)


def score_at_least(value: Any, good: float, bad: float = 0.0) -> float:


    numeric = safe_float(value)
    if numeric is None:
        return 0.0
    if numeric >= good:
        return 1.0
    if numeric <= bad:
        return 0.0
    return clamp01((numeric - bad) / max(1e-8, good - bad))


def score_at_most(value: Any, good: float, bad: float) -> float:


    numeric = safe_float(value)
    if numeric is None:
        return 0.0
    if numeric <= good:
        return 1.0
    if numeric >= bad:
        return 0.0
    return clamp01(1.0 - (numeric - good) / max(1e-8, bad - good))


def normalize_name(value: Any) -> str:


    return str(value or "").strip()


def as_list(value: Any) -> List[Any]:


    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def as_str_list(value: Any) -> List[str]:


    return [normalize_name(item) for item in as_list(value) if normalize_name(item)]


def get_nested(mapping: Mapping[str, Any], *keys: str) -> Any:


    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def evaluator_weight(
    score_config: Optional[Mapping[str, Any]],
    key: str,
    default: float,
) -> float:


    weights = score_config.get("evaluator_weights", {}) if isinstance(score_config, Mapping) else {}
    if isinstance(weights, Mapping) and key in weights:
        value = safe_float(weights.get(key), default)
        return float(default if value is None else value)
    value = (
        safe_float((score_config or {}).get(f"weight_{key}"), default)
        if isinstance(score_config, Mapping)
        else default
    )
    return float(default if value is None else value)

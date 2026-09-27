

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


def copied_mapping(value: Any) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        return {}
    return deepcopy({str(key): item for key, item in value.items()})


def copied_sequence_mapping(value: Any) -> Dict[str, str]:


    if not isinstance(value, Mapping):
        return {}
    return {str(chain_id): str(sequence) for chain_id, sequence in value.items()}


def optional_float(value: Any) -> float | None:


    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

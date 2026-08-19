

from __future__ import annotations

from typing import Any, Mapping, Sequence


OUTSIDE_INCUMBENT_EXPLORATION = "outside_incumbent_exploration"

_OUTSIDE_INCUMBENT_ROLE_ALIASES = frozenset(
    {
        "region_exploration",
        "node_region_explorer",
    }
)


def _declared_capabilities(role: Mapping[str, Any]) -> set[str]:
    raw = role.get("capabilities") or role.get("capability_ids") or ()
    if isinstance(raw, Mapping):
        return {
            str(name).strip()
            for name, enabled in raw.items()
            if bool(enabled) and str(name).strip()
        }
    if isinstance(raw, str):
        return {raw.strip()} if raw.strip() else set()
    if isinstance(raw, Sequence):
        return {str(item).strip() for item in raw if str(item).strip()}
    return set()


def role_has_capability(role: Mapping[str, Any] | str, capability: str) -> bool:


    capability_id = str(capability or "").strip()
    if not capability_id:
        return False
    if isinstance(role, Mapping):
        if capability_id in _declared_capabilities(role):
            return True
        search_scope = role.get("search_scope")
        if (
            capability_id == OUTSIDE_INCUMBENT_EXPLORATION
            and isinstance(search_scope, Mapping)
            and bool(search_scope.get("require_non_incumbent_segment_probe"))
        ):
            return True
        role_id = str(role.get("role_id") or "").strip()
    else:
        role_id = str(role or "").strip()
    return bool(
        capability_id == OUTSIDE_INCUMBENT_EXPLORATION
        and role_id in _OUTSIDE_INCUMBENT_ROLE_ALIASES
    )


def requires_outside_incumbent_exploration(
    role: Mapping[str, Any] | str,
) -> bool:


    return role_has_capability(role, OUTSIDE_INCUMBENT_EXPLORATION)


__all__ = [
    "OUTSIDE_INCUMBENT_EXPLORATION",
    "requires_outside_incumbent_exploration",
    "role_has_capability",
]

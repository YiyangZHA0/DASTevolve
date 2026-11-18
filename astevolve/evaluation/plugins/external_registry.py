

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PLUGIN_REGISTRY_VERSION = "astevolve.evaluator_plugin_registry.v2"
PLUGIN_RESOLUTION_VERSION = "astevolve.plugin_resolution.v1"


class PluginConfigError(ValueError):
    pass


class PluginLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PluginConfigField:


    kind: str
    runtime_key: str


@dataclass(frozen=True)
class EvaluatorPluginSpec:


    name: str
    factory: str
    aliases: Tuple[str, ...] = ()
    config_fields: Mapping[str, PluginConfigField] = field(default_factory=dict)
    weight_fields: Tuple[str, ...] = ()


PLUGIN_REGISTRY: Dict[str, EvaluatorPluginSpec] = {}
PLUGIN_ALIASES: Dict[str, str] = {}


def register_plugin(
    spec: EvaluatorPluginSpec,
    *,
    replace: bool = False,
) -> None:


    if not isinstance(spec, EvaluatorPluginSpec):
        raise TypeError("spec must be EvaluatorPluginSpec")
    name = str(spec.name).strip().lower()
    if not name:
        raise PluginConfigError("plugin name must be non-empty")
    if name in PLUGIN_REGISTRY and not replace:
        raise PluginConfigError(f"Evaluator plugin {name!r} is already registered")
    aliases = tuple(str(value).strip().lower() for value in spec.aliases)
    if any(not value for value in aliases):
        raise PluginConfigError(f"Evaluator plugin {name!r} has an empty alias")
    conflicts = [
        alias
        for alias in aliases
        if alias in PLUGIN_REGISTRY
        or (
            alias in PLUGIN_ALIASES
            and PLUGIN_ALIASES[alias] != name
        )
    ]
    if conflicts:
        raise PluginConfigError(
            f"Evaluator plugin {name!r} has conflicting aliases: {conflicts}"
        )
    if replace and name in PLUGIN_REGISTRY:
        for alias, target in tuple(PLUGIN_ALIASES.items()):
            if target == name:
                del PLUGIN_ALIASES[alias]
    normalized = EvaluatorPluginSpec(
        name=name,
        factory=str(spec.factory).strip(),
        aliases=aliases,
        config_fields=dict(spec.config_fields),
        weight_fields=tuple(str(value) for value in spec.weight_fields),
    )
    PLUGIN_REGISTRY[name] = normalized
    PLUGIN_ALIASES.update({alias: name for alias in aliases})


def unregister_plugin(name: str) -> None:


    canonical = canonical_plugin_name(name)
    del PLUGIN_REGISTRY[canonical]
    for alias, target in tuple(PLUGIN_ALIASES.items()):
        if target == canonical:
            del PLUGIN_ALIASES[alias]


def plugin_registry_manifest() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "registry_version": PLUGIN_REGISTRY_VERSION,
            "name": name,
            "factory": spec.factory,
            "aliases": list(spec.aliases),
            "config_fields": sorted(spec.config_fields),
            "weight_fields": list(spec.weight_fields),
        }
        for name, spec in sorted(PLUGIN_REGISTRY.items())
    }


def canonical_plugin_name(name: Any) -> str:
    raw = str(name or "").strip().lower()
    canonical = PLUGIN_ALIASES.get(raw, raw)
    if canonical not in PLUGIN_REGISTRY:
        raise PluginConfigError(f"Unknown evaluator plugin {raw!r}")
    return canonical


def _request_items(value: Any) -> List[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    items: List[str] = []
    for item in raw:
        name = (
            item.get("name") or item.get("plugin")
            if isinstance(item, Mapping)
            else item
        )
        text = str(name or "").strip()
        if text:
            items.append(text)
    return items


def normalize_plugin_requests(value: Any) -> Tuple[List[str], List[str]]:
    requested = _request_items(value)
    resolved: List[str] = []
    unknown: List[str] = []
    for raw in requested:
        try:
            canonical = canonical_plugin_name(raw)
        except PluginConfigError:
            unknown.append(str(raw).strip().lower())
            continue
        if canonical not in resolved:
            resolved.append(canonical)
    if unknown:
        label = "plugin" if len(unknown) == 1 else "plugins"
        raise PluginConfigError(
            f"Unknown evaluator {label}: {', '.join(repr(v) for v in unknown)}"
        )
    return requested, resolved


def _normalize_value(path: str, value: Any, kind: str) -> Any:
    if kind == "float":
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PluginConfigError(f"{path} must be numeric") from exc
        if not isfinite(numeric):
            raise PluginConfigError(f"{path} must be finite")
        return numeric
    if kind == "int":
        if isinstance(value, bool):
            raise PluginConfigError(f"{path} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise PluginConfigError(f"{path} must be an integer") from exc
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise PluginConfigError(f"{path} must be boolean")
    if kind == "str":
        return str(value)
    if kind == "mapping":
        if not isinstance(value, Mapping):
            raise PluginConfigError(f"{path} must be a mapping")
        return dict(value)
    if kind == "sequence":
        if not isinstance(value, (list, tuple)):
            raise PluginConfigError(f"{path} must be a sequence")
        return list(value)
    raise PluginConfigError(f"{path} uses unsupported field kind {kind!r}")


def normalize_plugin_config(
    value: Any,
    *,
    requested_plugins: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PluginConfigError("score_config.plugin_config must be a mapping")
    requested = set(str(name) for name in (requested_plugins or []))
    normalized_config: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_config in value.items():
        canonical = canonical_plugin_name(raw_name)
        if requested and canonical not in requested:
            raise PluginConfigError(
                f"score_config.plugin_config.{raw_name} configures an "
                "unrequested plugin"
            )
        if canonical in normalized_config:
            raise PluginConfigError(
                f"score_config.plugin_config declares {canonical!r} twice"
            )
        if not isinstance(raw_config, Mapping):
            raise PluginConfigError(
                f"score_config.plugin_config.{raw_name} must be a mapping"
            )
        spec = PLUGIN_REGISTRY[canonical]
        scoped: Dict[str, Any] = {}
        for field_name, field_value in raw_config.items():
            field_name = str(field_name)
            path = f"score_config.plugin_config.{canonical}.{field_name}"
            if field_name == "evaluator_weights":
                if not isinstance(field_value, Mapping):
                    raise PluginConfigError(f"{path} must be a mapping")
                weights: Dict[str, float] = {}
                for weight_name, weight_value in field_value.items():
                    key = str(weight_name)
                    if key not in spec.weight_fields:
                        raise PluginConfigError(
                            f"{path}.{key} is not a registered plugin weight"
                        )
                    weights[key] = _normalize_value(
                        f"{path}.{key}",
                        weight_value,
                        "float",
                    )
                scoped[field_name] = weights
                continue
            field_spec = spec.config_fields.get(field_name)
            if field_spec is None:
                raise PluginConfigError(
                    f"Unknown field {path}; allowed fields are "
                    f"{sorted((*spec.config_fields, 'evaluator_weights'))}"
                )
            scoped[field_name] = _normalize_value(
                path,
                field_value,
                field_spec.kind,
            )
        normalized_config[canonical] = scoped
    return normalized_config


def resolve_plugin_plan(
    design_state: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> Dict[str, Any]:
    case_sheet = (
        design_state.get("_case_sheet")
        if isinstance(design_state.get("_case_sheet"), Mapping)
        else {}
    )
    declarations: List[Tuple[str, Any]] = []
    for source, mapping in (
        ("score_config.evaluator_plugins", score_config),
        ("design_state.evaluator_plugins", design_state),
        ("case_sheet.evaluator_plugins", case_sheet),
    ):
        if "evaluator_plugins" in mapping:
            declarations.append((source, mapping.get("evaluator_plugins")))
    if not declarations:
        return {
            "schema_version": PLUGIN_RESOLUTION_VERSION,
            "source": "none",
            "requested": [],
            "resolved": [],
            "unknown": [],
            "strict": False,
        }

    normalized = [
        (source, *normalize_plugin_requests(declaration))
        for source, declaration in declarations
    ]
    reference = normalized[0][2]
    if any(resolved != reference for _, _, resolved in normalized[1:]):
        raise PluginConfigError(
            "Conflicting evaluator plugin declarations: "
            + ", ".join(f"{source}={resolved}" for source, _, resolved in normalized)
        )
    configured = normalize_plugin_config(
        score_config.get("plugin_config"),
        requested_plugins=reference,
    )
    extra = sorted(set(configured) - set(reference))
    if extra:
        raise PluginConfigError(
            f"Plugin config exists for unrequested plugin(s): {', '.join(extra)}"
        )
    return {
        "schema_version": PLUGIN_RESOLUTION_VERSION,
        "source": normalized[0][0],
        "requested": normalized[0][1],
        "resolved": reference,
        "unknown": [],
        "strict": True,
    }


def plugin_runtime_score_config(
    score_config: Mapping[str, Any],
    plugin_name: str,
) -> Dict[str, Any]:
    canonical = canonical_plugin_name(plugin_name)
    out = dict(score_config)
    scoped = normalize_plugin_config(
        score_config.get("plugin_config")
    ).get(canonical, {})
    spec = PLUGIN_REGISTRY[canonical]
    for field_name, value in scoped.items():
        if field_name == "evaluator_weights":
            weights = dict(out.get("evaluator_weights") or {})
            weights.update(value)
            out["evaluator_weights"] = weights
        else:
            out[spec.config_fields[field_name].runtime_key] = value
    out["active_plugin"] = canonical
    out["active_plugin_config"] = dict(scoped)
    return out


def load_plugin(name: str) -> Any:
    canonical = canonical_plugin_name(name)
    spec = PLUGIN_REGISTRY[canonical]
    module_name, separator, attribute = spec.factory.partition(":")
    if not separator or not module_name or not attribute:
        raise PluginLoadError(
            f"Evaluator plugin {canonical!r} has invalid factory {spec.factory!r}"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
        plugin = factory()
    except Exception as exc:
        raise PluginLoadError(
            f"Failed to load requested evaluator plugin {canonical!r}: {exc}"
        ) from exc
    if str(getattr(plugin, "name", "")) != canonical:
        raise PluginLoadError(
            f"Plugin factory returned {getattr(plugin, 'name', None)!r}; "
            f"expected {canonical!r}"
        )
    return plugin


def preflight_evaluator_plugins(
    design_state: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> Dict[str, Any]:
    plan = resolve_plugin_plan(design_state, score_config)
    loaded = [name for name in plan["resolved"] if load_plugin(name)]
    return {
        "registry_version": PLUGIN_REGISTRY_VERSION,
        **plan,
        "loaded": loaded,
        "failed": [],
        "available": bool(loaded),
    }


__all__ = [
    "EvaluatorPluginSpec",
    "PLUGIN_ALIASES",
    "PLUGIN_REGISTRY",
    "PLUGIN_REGISTRY_VERSION",
    "PLUGIN_RESOLUTION_VERSION",
    "PluginConfigError",
    "PluginConfigField",
    "PluginLoadError",
    "canonical_plugin_name",
    "load_plugin",
    "normalize_plugin_config",
    "normalize_plugin_requests",
    "preflight_evaluator_plugins",
    "plugin_registry_manifest",
    "plugin_runtime_score_config",
    "register_plugin",
    "resolve_plugin_plan",
    "unregister_plugin",
]



from __future__ import annotations

from typing import Mapping

from .registry import (
    EvaluatorPluginSpec,
    PLUGIN_REGISTRY,
    PluginConfigError,
    PluginConfigField,
    PluginLoadError,
    load_plugin,
    register_plugin,
    resolve_plugin_plan,
    unregister_plugin,
)


def resolve_plugins(design_state: Mapping, score_config: Mapping):


    plan = resolve_plugin_plan(design_state, score_config)
    return [load_plugin(name) for name in plan["resolved"]]


__all__ = [
    "EvaluatorPluginSpec",
    "PLUGIN_REGISTRY",
    "PluginConfigError",
    "PluginConfigField",
    "PluginLoadError",
    "register_plugin",
    "resolve_plugins",
    "unregister_plugin",
]

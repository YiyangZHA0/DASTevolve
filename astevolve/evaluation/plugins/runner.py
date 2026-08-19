

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    PLUGIN_REGISTRY_VERSION,
    PluginLoadError,
    load_plugin,
    normalize_plugin_config,
    plugin_runtime_score_config,
    resolve_plugin_plan,
)


def plugin_terms(context: EvaluatorContext, terms: List[ScoreTerm]) -> Dict[str, Any]:


    plan = resolve_plugin_plan(context.design_state, context.score_config)
    status: Dict[str, Any] = {
        "registry_version": PLUGIN_REGISTRY_VERSION,
        **plan,
        "available": False,
        "loaded": [],
        "failed": [],
    }
    all_plugin_config = normalize_plugin_config(
        context.score_config.get("plugin_config")
        if isinstance(context.score_config, Mapping)
        else None
    )
    for name in plan["resolved"]:
        try:
            plugin = load_plugin(name)
            plugin_context = replace(
                context,
                score_config=plugin_runtime_score_config(
                    context.score_config, name
                ),
                plugin_name=name,
                plugin_config=dict(all_plugin_config.get(name, {})),
            )
            generated_terms = plugin.score_terms(plugin_context)
            if plan["strict"] and not generated_terms:
                raise PluginLoadError(
                    f"Requested evaluator plugin {name!r} produced zero score terms"
                )
            attributed_terms = []
            for term in generated_terms or []:
                details = dict(term.details) if isinstance(term.details, Mapping) else {}
                details["_plugin_name"] = name
                attributed_terms.append(replace(term, details=details))
            terms.extend(attributed_terms)
            status["loaded"].append({"name": name, "term_count": len(generated_terms or [])})
        except Exception as error:
            if plan["strict"]:
                if isinstance(error, PluginLoadError):
                    raise
                raise PluginLoadError(
                    f"Requested evaluator plugin {name!r} failed: {error}"
                ) from error
            status["failed"].append({"name": name, "error": str(error)})
            terms.append(
                ScoreTerm(
                    f"{name}_plugin_error",
                    "plugin",
                    0.0,
                    0.0,
                    {"plugin": name, "error": str(error), "dimension": "correctness"},
                    warnings=[f"{name} evaluator plugin failed: {error}"],
                    backend="plugin",
                    available=False,
                )
            )
    status["available"] = bool(status["loaded"]) and not status["failed"]
    if plan["strict"] and plan["resolved"] and len(status["loaded"]) != len(plan["resolved"]):
        raise PluginLoadError(
            "Requested evaluator plugin set did not fully load: "
            f"requested={plan['resolved']}, loaded={[row['name'] for row in status['loaded']]}"
        )
    return status

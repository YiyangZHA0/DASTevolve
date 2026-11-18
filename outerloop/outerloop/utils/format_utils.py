

from typing import Any, Dict

from outerloop.utils.metric_semantics import compare_metrics


def format_metrics_safe(metrics: Dict[str, Any]) -> str:

    if not metrics:
        return ""

    formatted_parts = []
    for name, value in metrics.items():

        if isinstance(value, (int, float)):
            try:

                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):

                formatted_parts.append(f"{name}={value}")
        else:

            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)


def format_directional_improvement_safe(
    parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]
) -> str:

    if not parent_metrics or not child_metrics:
        return ""

    improvement_parts = [
        f"{item.metric}={item.improvement_delta:+.4f}"
        for item in compare_metrics(parent_metrics, child_metrics)
        if item.comparable and item.improvement_delta is not None
    ]

    return ", ".join(improvement_parts)


def format_improvement_safe(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:


    return format_directional_improvement_safe(parent_metrics, child_metrics)

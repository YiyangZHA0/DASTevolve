

from typing import Any, Dict, List, Optional


def get_primary_objective(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:


    if not metrics:
        return None

    for key in ("final_energy", "combined_energy"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if numeric == numeric and numeric not in (float("inf"), float("-inf")):
                return {
                    "name": key,
                    "value": numeric,
                    "direction": "minimize",
                }

    value = metrics.get("combined_score")
    if value is not None and not isinstance(value, bool):
        try:
            return {
                "name": "combined_score",
                "value": float(value),
                "direction": "maximize",
            }
        except (ValueError, TypeError, OverflowError):
            pass

    return None


def safe_numeric_average(metrics: Dict[str, Any]) -> float:

    if not metrics:
        return 0.0

    numeric_values = []
    for value in metrics.values():


        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:

                float_val = float(value)
                if not (float_val != float_val):
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):

                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:

    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:

                float_val = float(value)
                if not (float_val != float_val):
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):

                continue

    return numeric_sum


def get_fitness_score(
    metrics: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
) -> float:

    if not metrics:
        return 0.0

    primary = get_primary_objective(metrics)
    if primary is not None:
        value = float(primary["value"])
        return -value if primary["direction"] == "minimize" else value


    feature_dimensions = feature_dimensions or []
    fitness_metrics = {}

    for key, value in metrics.items():

        if key not in feature_dimensions:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    float_val = float(value)
                    if not (float_val != float_val):
                        fitness_metrics[key] = float_val
                except (ValueError, TypeError, OverflowError):
                    continue


    if not fitness_metrics:
        return safe_numeric_average(metrics)

    return safe_numeric_average(fitness_metrics)


def format_feature_coordinates(metrics: Dict[str, Any], feature_dimensions: List[str]) -> str:

    feature_values = []
    for dim in feature_dimensions:
        if dim in metrics:
            value = metrics[dim]
            if isinstance(value, (int, float)):
                try:
                    float_val = float(value)
                    if not (float_val != float_val):
                        feature_values.append(f"{dim}={float_val:.2f}")
                except (ValueError, TypeError, OverflowError):
                    feature_values.append(f"{dim}={value}")
            else:
                feature_values.append(f"{dim}={value}")

    if not feature_values:
        return ""

    return ", ".join(feature_values)

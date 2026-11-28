from pathlib import Path

from astevolve.runtime.paths import data_path


def reference_path(*parts: str) -> Path:
    """Prefer an installed reference when available, otherwise use the case bundle."""
    external = data_path("cab_lys3_hewl_cdr_recovery", *parts)
    return external if external.exists() else Path(__file__).parent.joinpath("inputs", *parts)

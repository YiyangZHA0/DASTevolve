

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import clamp01

from .base import backend_enabled, neutralize_optional_unavailable_backend
from .foldx import foldx_backend_term
from .fpocket import fpocket_backend_term
from .getcontacts import getcontacts_backend_term
from .ipsae import ipsae_backend_term
from .povme import povme_backend_term
from .pyrosetta import pyrosetta_backend_term
from .rosetta import rosetta_backend_term


BackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], ScoreTerm]


BACKEND_FACTORIES: Tuple[BackendFactory, ...] = (
    rosetta_backend_term,
    pyrosetta_backend_term,
    getcontacts_backend_term,
    ipsae_backend_term,
    foldx_backend_term,
    fpocket_backend_term,
    povme_backend_term,
)


def optional_backend_terms(
    structure: Mapping[str, Any],
    terms: List[ScoreTerm],
    score_config: Mapping[str, Any],
) -> Dict[str, Any]:


    backends = {
        "rosetta": {"enabled": backend_enabled(score_config, "rosetta")},
        "pyrosetta": {"enabled": backend_enabled(score_config, "pyrosetta")},
        "getcontacts": {"enabled": backend_enabled(score_config, "getcontacts")},
        "ipsae": {"enabled": backend_enabled(score_config, "ipsae")},
        "foldx": {"enabled": backend_enabled(score_config, "foldx")},
        "fpocket": {"enabled": backend_enabled(score_config, "fpocket")},
        "povme": {"enabled": backend_enabled(score_config, "povme")},
    }
    for factory in BACKEND_FACTORIES:
        term = neutralize_optional_unavailable_backend(factory(structure, score_config))
        terms.append(term)
        backends[term.backend] = {
            "enabled": bool(term.details.get("enabled", True)) if term.weight > 0 or term.available else bool(term.details.get("enabled", False)),
            "available": bool(term.available),
            "required": bool(term.details.get("required", False)),
            "weight": float(term.weight),
            "score": float(clamp01(term.score)),
            "warnings": list(term.warnings),
        }
    return backends

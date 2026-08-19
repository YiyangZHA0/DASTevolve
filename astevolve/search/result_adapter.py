

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from astevolve.domain import SearchResult
from astevolve.search.config import SAConfig
from astevolve.search.run_memory import InnerRunMemory


def optimize_multichain_result(
    compiled: Dict[str, Any],
    constraint_specs: list[dict],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]] = None,
    fixed_residues: Optional[Dict[str, Dict[int, str]]] = None,
    internal_memory: Optional[Dict[str, Any]] = None,
    run_memory: Optional[InnerRunMemory] = None,
    score_config: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    causal_context: Optional[Mapping[str, Any]] = None,
) -> SearchResult:
    from astevolve.search.inner_opt import optimize_multichain

    return SearchResult.from_legacy(
        optimize_multichain(
            compiled=compiled,
            constraint_specs=constraint_specs,
            cfg=cfg,
            masks=masks,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            internal_memory=internal_memory,
            run_memory=run_memory,
            score_config=score_config,
            design_state=design_state,
            causal_context=causal_context,
        )
    )


__all__ = ["optimize_multichain_result"]

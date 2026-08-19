

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astevolve.core.amino_acids import CHARGED, HYDROPHOBIC
from astevolve.core.constraints import energy_breakdown
from astevolve.providers.registry import sequence_loglikelihood
from astevolve.search.config import SAConfig


def _progen_score(
    seqs: Dict[str, str],
    chains: Optional[List[str]],
    reduce: str,
    model: str = "progen",
) -> Dict[str, float]:


    if chains is None:
        chains = list(seqs.keys())
    scores = []
    total_len = 0
    sum_loglik = 0.0
    for chain_id in chains:
        sequence = seqs.get(chain_id, "")
        if not sequence:
            continue
        result = sequence_loglikelihood(sequence, provider=model)
        loglik_sum = result["loglik_sum"]
        loglik_avg = result["loglik_avg"]
        length = len(sequence)
        scores.append((loglik_sum, loglik_avg, length))
        sum_loglik += loglik_sum
        total_len += length

    if not scores:
        return {"loglik_sum": 0.0, "loglik_avg": 0.0}

    if reduce == "mean":
        average = float(np.mean([item[1] for item in scores]))
        return {"loglik_sum": float(sum_loglik), "loglik_avg": average}

    average = float(sum_loglik / max(1, total_len))
    return {"loglik_sum": float(sum_loglik), "loglik_avg": average}


def compute_segment_scores(
    seqs: Dict[str, str], compiled: Dict[str, Any]
) -> List[Dict[str, Any]]:


    output = []
    for segment in compiled["segments"]:
        fragment = segment.extract(seqs.get(segment.chain_id, ""))
        if not fragment:
            continue
        length = len(fragment)
        hydrophobic = sum(1 for residue in fragment if residue in HYDROPHOBIC) / length
        charged = sum(1 for residue in fragment if residue in CHARGED) / length
        flexible = sum(1 for residue in fragment if residue in set("GS")) / length
        polar = sum(1 for residue in fragment if residue in set("STNQY")) / length
        output.append(
            {
                "chain_id": segment.chain_id,
                "kind": segment.kind,
                "name": segment.name,
                "spans": segment.spans,
                "total_length": segment.total_length,
                "is_contiguous": segment.is_contiguous,
                "length": length,
                "hydro_frac": float(hydrophobic),
                "charged_frac": float(charged),
                "flexible_frac": float(flexible),
                "polar_frac": float(polar),
            }
        )
    return output


def _score_fast_candidate(
    seqs: Dict[str, str],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    compiled: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], float]:


    breakdown = energy_breakdown(seqs, compiled, terms_fast)
    if float(cfg.progen_weight) <= 0.0:
        progen = {"loglik_sum": 0.0, "loglik_avg": 0.0}
    else:
        progen = _progen_score(
            seqs,
            cfg.progen_chains,
            cfg.progen_reduce,
            cfg.sequence_prior_model,
        )
    fast_loss = breakdown["total"] + cfg.progen_weight * (-progen["loglik_avg"])
    return breakdown, progen, float(fast_loss)


__all__ = ["_progen_score", "_score_fast_candidate", "compute_segment_scores"]

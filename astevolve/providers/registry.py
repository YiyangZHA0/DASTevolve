

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any, Dict, List, Optional, Tuple

from astevolve.runtime.provider_registry import ProviderRegistry


def _normalise_name(name: Optional[str], default: str) -> str:
    value = str(name or default).strip().lower()
    aliases = {
        "progen2": "progen",
        "progen2-small": "progen",
        "protenix_mini": "protenix",
        "protenix-mini": "protenix",
        "esmfold_v2": "esmfold2",
        "esmfold2-fast": "esmfold2",
        "esmfold2_fast": "esmfold2",
        "af3": "alphafold3",
        "alpha-fold3": "alphafold3",
        "alpha_fold3": "alphafold3",
        "structure_service": "service",
        "remote_service": "service",
    }
    return aliases.get(value, value)


def _registry_name(value: str) -> str:
    return _normalise_name(value, value)


SEQUENCE_MODELS: ProviderRegistry[Any] = ProviderRegistry(_registry_name)
RESIDUE_PRIORS: ProviderRegistry[Any] = ProviderRegistry(_registry_name)
STRUCTURE_MODELS: ProviderRegistry[Any] = ProviderRegistry(_registry_name)


def _progen_factory() -> Any:
    from .adapters import ProGenSequencePrior

    return ProGenSequencePrior()


def _protenix_factory() -> Any:
    from .adapters import ProtenixStructureModel

    return ProtenixStructureModel()


def _masked_lm_factory() -> Any:
    from .masked_lm import MaskedProteinLMPrior

    return MaskedProteinLMPrior()


def _esmfold2_factory() -> Any:
    from .adapters import ESMFold2StructureModel

    return ESMFold2StructureModel()


def _esmfold_factory() -> Any:
    from .adapters import ESMFoldStructureModel

    return ESMFoldStructureModel()


def _alphafold3_factory() -> Any:
    from .adapters import AlphaFold3StructureModel

    return AlphaFold3StructureModel()


def _structure_service_factory() -> Any:
    from .service import StructureServiceModel

    return StructureServiceModel()


SEQUENCE_MODELS.register(
    "progen",
    _progen_factory,
    aliases=("progen2", "progen2-small"),
)
RESIDUE_PRIORS.register(
    "masked_lm",
    _masked_lm_factory,
    aliases=("esm", "esm2", "protein_masked_lm"),
)
STRUCTURE_MODELS.register(
    "protenix",
    _protenix_factory,
    aliases=("protenix_mini", "protenix-mini"),
)
STRUCTURE_MODELS.register(
    "esmfold",
    _esmfold_factory,
    aliases=("esmfold_v1", "classic_esmfold"),
)
STRUCTURE_MODELS.register(
    "esmfold2",
    _esmfold2_factory,
    aliases=("esmfold_v2", "esmfold2-fast", "esmfold2_fast"),
)
STRUCTURE_MODELS.register(
    "alphafold3",
    _alphafold3_factory,
    aliases=("af3", "alpha-fold3", "alpha_fold3"),
)
STRUCTURE_MODELS.register(
    "service",
    _structure_service_factory,
    aliases=("structure_service", "remote_service"),
)


def register_sequence_model(
    name: str,
    factory: Callable[[], Any],
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> None:
    SEQUENCE_MODELS.register(name, factory, aliases=aliases, replace=replace)


def register_residue_prior(
    name: str,
    factory: Callable[[], Any],
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> None:
    RESIDUE_PRIORS.register(name, factory, aliases=aliases, replace=replace)


def register_structure_model(
    name: str,
    factory: Callable[[], Any],
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> None:
    STRUCTURE_MODELS.register(name, factory, aliases=aliases, replace=replace)


def available_model_interfaces() -> Dict[str, List[str]]:
    return {
        "sequence_prior": list(SEQUENCE_MODELS.available()),
        "residue_prior": list(RESIDUE_PRIORS.available()),
        "structure": list(STRUCTURE_MODELS.available()),
    }


def _sequence_model(name: Optional[str] = None) -> Any:
    provider = _normalise_name(
        name or os.environ.get("ASTEVOLVE_SEQUENCE_PRIOR_MODEL"),
        "progen",
    )
    return SEQUENCE_MODELS.create(provider)


def _structure_model(name: Optional[str] = None) -> Any:
    provider = _normalise_name(
        name or os.environ.get("ASTEVOLVE_STRUCTURE_MODEL"),
        "protenix",
    )
    return STRUCTURE_MODELS.create(provider)


def _residue_prior(name: Optional[str] = None) -> Any:
    provider = _normalise_name(
        name or os.environ.get("ASTEVOLVE_RESIDUE_PRIOR_MODEL"),
        "masked_lm",
    )
    return RESIDUE_PRIORS.create(provider)


def sequence_loglikelihood(
    seq: str,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    return _sequence_model(provider).sequence_loglikelihood(seq, **kwargs)


def sequence_masked_marginals(
    seq: str,
    positions: Iterable[int],
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:


    values = _residue_prior(provider).batch_masked_marginals(
        seq, tuple(int(position) for position in positions)
    )
    return [value.as_dict() for value in values]


def sequence_candidate_parent_deltas(
    parent_sequence: str,
    candidate_sequences: Iterable[str],
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:


    values = _residue_prior(provider).candidate_parent_deltas(
        parent_sequence, tuple(str(sequence) for sequence in candidate_sequences)
    )
    return [value.as_dict() for value in values]


def run_structure_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return _structure_model(provider).confidence_multichain(
        pred_name=pred_name,
        chains=chains,
        **kwargs,
    )


def run_structure_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> float:
    return _structure_model(provider).scalar_multichain(
        pred_name=pred_name,
        chains=chains,
        **kwargs,
    )


def run_structure_confidence_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    model = _structure_model(provider)
    confidence_complex = getattr(model, "confidence_complex", None)
    if not callable(confidence_complex):
        raise NotImplementedError(f"{model.name} does not support complex entities")
    return confidence_complex(
        pred_name=pred_name,
        entities=entities,
        **kwargs,
    )


def run_structure_confidence_complex_batch(
    jobs: List[Dict[str, Any]],
    provider: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    model = _structure_model(provider)
    native_batch = getattr(model, "confidence_complex_batch", None)
    if callable(native_batch):
        return list(native_batch(jobs=jobs, **kwargs))
    confidence_complex = getattr(model, "confidence_complex", None)
    if not callable(confidence_complex):
        raise NotImplementedError(f"{model.name} does not support complex entities")

    results: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("complex batch jobs must be dictionaries")
        call = dict(job)
        pred_name = call.pop("pred_name", call.pop("name", None))
        entities = call.pop("entities", None)
        results.append(
            confidence_complex(
                pred_name=pred_name,
                entities=entities,
                **call,
                **kwargs,
            )
        )
    return results


def run_structure_plddt_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> float:
    model = _structure_model(provider)
    scalar_complex = getattr(model, "scalar_complex", None)
    if not callable(scalar_complex):
        raise NotImplementedError(f"{model.name} does not support complex entities")
    return scalar_complex(
        pred_name=pred_name,
        entities=entities,
        **kwargs,
    )


__all__ = [
    "SEQUENCE_MODELS",
    "RESIDUE_PRIORS",
    "STRUCTURE_MODELS",
    "available_model_interfaces",
    "register_residue_prior",
    "register_sequence_model",
    "register_structure_model",
    "run_structure_confidence_complex",
    "run_structure_confidence_complex_batch",
    "run_structure_confidence_multichain",
    "run_structure_plddt_complex",
    "run_structure_plddt_multichain",
    "sequence_loglikelihood",
    "sequence_masked_marginals",
    "sequence_candidate_parent_deltas",
]

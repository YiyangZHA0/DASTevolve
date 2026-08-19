

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple


class SequencePriorModel(Protocol):


    name: str

    def sequence_loglikelihood(self, seq: str, **kwargs: Any) -> Dict[str, float]:
        ...


SequenceScorerModel = SequencePriorModel


class SequenceGeneratorModel(Protocol):


    name: str

    def generate(self, prompt: str, *, count: int = 1, **kwargs: Any) -> List[str]:
        ...


class StructureModel(Protocol):


    name: str

    def confidence_multichain(
        self,
        pred_name: Optional[str],
        chains: Optional[List[Tuple[str, str]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def scalar_multichain(
        self,
        pred_name: Optional[str],
        chains: Optional[List[Tuple[str, str]]],
        **kwargs: Any,
    ) -> float:
        ...

    def confidence_complex(
        self,
        pred_name: Optional[str],
        entities: Optional[List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def scalar_complex(
        self,
        pred_name: Optional[str],
        entities: Optional[List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> float:
        ...



from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ProGenSequencePrior:
    name = "progen"

    def sequence_loglikelihood(self, seq: str, **kwargs: Any) -> Dict[str, float]:
        from .progen import sequence_loglikelihood

        return sequence_loglikelihood(seq, **kwargs)


class ProtenixStructureModel:
    name = "protenix"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .protenix import run_protenix_confidence_multichain

        return run_protenix_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        from .protenix import run_protenix_plddt_multichain

        return run_protenix_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def confidence_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .protenix import run_protenix_confidence_complex

        return run_protenix_confidence_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )

    def confidence_complex_batch(
        self,
        jobs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        from .protenix import run_protenix_confidence_complex_batch

        return run_protenix_confidence_complex_batch(jobs=jobs, **kwargs)

    def scalar_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> float:
        from .protenix import run_protenix_plddt_complex

        return run_protenix_plddt_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )


class ESMFoldStructureModel:


    name = "esmfold"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .esmfold import run_esmfold_confidence_multichain

        return run_esmfold_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        from .esmfold import run_esmfold_plddt_multichain

        return run_esmfold_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )


class ESMFold2StructureModel:
    name = "esmfold2"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .esmfold2 import run_esmfold2_confidence_multichain

        return run_esmfold2_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        from .esmfold2 import run_esmfold2_plddt_multichain

        return run_esmfold2_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )


class AlphaFold3StructureModel:
    name = "alphafold3"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .alphafold3 import run_alphafold3_confidence_multichain

        return run_alphafold3_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        from .alphafold3 import run_alphafold3_plddt_multichain

        return run_alphafold3_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def confidence_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .alphafold3 import run_alphafold3_confidence_complex

        return run_alphafold3_confidence_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )

    def scalar_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> float:
        from .alphafold3 import run_alphafold3_plddt_complex

        return run_alphafold3_plddt_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )


__all__ = [
    "AlphaFold3StructureModel",
    "ESMFoldStructureModel",
    "ESMFold2StructureModel",
    "ProGenSequencePrior",
    "ProtenixStructureModel",
]

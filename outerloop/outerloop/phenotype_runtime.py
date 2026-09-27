

from __future__ import annotations

from typing import Any, Mapping

from outerloop.effective_phenotype import (
    AcceptedRuntimeArtifact,
    EffectivePhenotypeDescriptor,
    EffectivePhenotypeIdentity,
    PhenotypeDescriptorConfig,
)


def seal_effective_phenotype(
    source_code: str,
    runtime_artifacts: Mapping[str, Any],
    metrics: Mapping[str, Any],
    database_config: Any,
) -> dict[str, Any]:


    enabled = getattr(database_config, "outer_effective_phenotype_enabled", None)
    if enabled is None or enabled is False:
        return {}
    if enabled is not True:
        raise ValueError("outer_effective_phenotype_enabled must be boolean")
    accepted = AcceptedRuntimeArtifact.create(
        source_code=source_code,
        runtime_artifacts=runtime_artifacts,
        metrics=metrics,
    )
    identity = EffectivePhenotypeIdentity.create(accepted)
    configured = getattr(
        database_config,
        "outer_effective_descriptor_dimensions",
        None,
    )
    descriptor_config = PhenotypeDescriptorConfig.create(
        components=configured if configured else None,
    )
    descriptor = EffectivePhenotypeDescriptor.create(
        accepted,
        config=descriptor_config,
    )
    return {
        "accepted_runtime_artifact": accepted.to_dict(),
        "effective_phenotype_identity": identity.to_dict(),
        "effective_phenotype_descriptor": descriptor.to_dict(),
    }


__all__ = ["seal_effective_phenotype"]

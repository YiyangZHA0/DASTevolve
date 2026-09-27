

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional


EXTERNAL_KNOWLEDGE_POLICY_VERSION = "astevolve.external_knowledge_policy.v1"

EXTERNAL_KB_FIELDS = (
    "external_kb_enabled",
    "external_kb_path",
    "external_kb_weight",
    "external_kb_embedding_manifest",
    "external_kb_retrieval_enabled",
    "external_kb_retrieval_top_k",
    "external_kb_retrieval_weight",
    "external_kb_device",
    "external_kb_max_length",
)

EXTERNAL_KB_ENV_VARS = (
    "ASTEVOLVE_EXTERNAL_KB",
    "ASTEVOLVE_ENABLE_EXTERNAL_KB",
    "ASTEVOLVE_EXTERNAL_RETRIEVAL",
    "ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL",
    "ASTEVOLVE_EXTERNAL_KB_PATH",
    "ASTEVOLVE_EXTERNAL_KB_EMBEDDING_MANIFEST",
)


def build_external_knowledge_policy(
    strategy: Optional[Mapping[str, Any]],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:


    requested = strategy if isinstance(strategy, Mapping) else {}
    environment = os.environ if environ is None else environ
    requested_fields = [name for name in EXTERNAL_KB_FIELDS if name in requested]
    requested_environment_variables = [
        name
        for name in EXTERNAL_KB_ENV_VARS
        if name in environment and str(environment.get(name, "")) != ""
    ]
    return {
        "schema_version": EXTERNAL_KNOWLEDGE_POLICY_VERSION,
        "status": "unsupported",
        "external_kb_used": False,
        "provider_loaded": False,
        "disposition": "deprecated_unsupported",
        "requested_fields": requested_fields,
        "requested_environment_variables": requested_environment_variables,
        "memory_classification": "internal_run_memory_is_not_external_knowledge",
    }


__all__ = [
    "EXTERNAL_KB_ENV_VARS",
    "EXTERNAL_KB_FIELDS",
    "EXTERNAL_KNOWLEDGE_POLICY_VERSION",
    "build_external_knowledge_policy",
]

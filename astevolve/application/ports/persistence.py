

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from astevolve.domain import ExperimentResult, RunContext


class MemoryStore(Protocol):


    def load(self, context: RunContext) -> Mapping[str, Any]:
        ...

    def update(self, result: ExperimentResult, context: RunContext) -> Mapping[str, Any]:
        ...


class ArtifactStore(Protocol):


    def write_json(self, relative_path: str, value: Any, context: RunContext) -> Path:
        ...

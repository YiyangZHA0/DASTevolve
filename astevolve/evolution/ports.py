

from __future__ import annotations

from typing import Optional, Protocol

from .domain import GenerationCommit


class GenerationPublisher(Protocol):


    def publish_atomic(self, commit: GenerationCommit) -> GenerationCommit: ...

    def get(self, generation_id: str) -> Optional[GenerationCommit]: ...


__all__ = ["GenerationPublisher"]

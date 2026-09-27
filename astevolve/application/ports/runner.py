

from __future__ import annotations

from typing import Mapping, Protocol

from astevolve.domain import RunContext


class DesignSearchRunner(Protocol):


    def run(self, strategy: Mapping[str, object], context: RunContext) -> Mapping[str, object]:
        ...

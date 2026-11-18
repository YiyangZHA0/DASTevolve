

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DesignCase:


    case_id: str
    root: Path
    manifest_path: Optional[Path]
    design_state_path: Path
    memory_path: Path
    output_root: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> Dict[str, str]:


        return {
            "design_state_path": str(self.design_state_path),
            "memory_path": str(self.memory_path),
        }

    def to_spec(self):


        from astevolve.domain import CaseSpec

        return CaseSpec.from_resolved_case(self)

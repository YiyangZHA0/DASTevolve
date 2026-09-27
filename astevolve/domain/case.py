

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ._mapping import copied_mapping


@dataclass(frozen=True)
class CaseSpec:


    case_id: str
    root: Path
    design_state_path: Path
    memory_path: Path
    output_root: Path
    manifest_path: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "astevolve.case.v1"

    @classmethod
    def from_resolved_case(cls, case: Any) -> "CaseSpec":


        return cls(
            case_id=str(case.case_id),
            root=Path(case.root),
            design_state_path=Path(case.design_state_path),
            memory_path=Path(case.memory_path),
            output_root=Path(case.output_root),
            manifest_path=Path(case.manifest_path) if case.manifest_path else None,
            metadata=copied_mapping(getattr(case, "metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:


        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "root": str(self.root),
            "design_state_path": str(self.design_state_path),
            "memory_path": str(self.memory_path),
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "metadata": copied_mapping(self.metadata),
        }

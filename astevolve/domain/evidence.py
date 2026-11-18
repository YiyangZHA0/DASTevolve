

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Tuple

from ._mapping import copied_mapping


@dataclass(frozen=True)
class EvidenceRecord:


    source: str
    kind: str
    available: bool = True
    value: Any = None
    details: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "available": self.available,
            "value": self.value,
            "details": copied_mapping(self.details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvidenceBundle:


    records: Tuple[EvidenceRecord, ...] = ()
    schema_version: str = "astevolve.evidence.v1"

    @classmethod
    def of(cls, records: Iterable[EvidenceRecord]) -> "EvidenceBundle":
        return cls(records=tuple(records))

    def available(self, kind: str) -> Tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.kind == kind and record.available)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
        }



import json
from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class EvaluationResult:


    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":

        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:

        return self.metrics

    def has_artifacts(self) -> bool:

        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:

        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:

        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:

        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())

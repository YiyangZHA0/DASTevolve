

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class Node:


    id: str
    kind: str = "node"
    role: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:


        return {"id": self.id, "kind": self.kind, "role": self.role, **self.metadata}


@dataclass
class StructuralNode(Node):


    kind: str = "structural"
    editable: bool = False
    frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:


        data = super().to_dict()
        data.update({"editable": self.editable, "frozen": self.frozen})
        return data


@dataclass
class FunctionalNode(Node):


    kind: str = "functional"
    maps_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:


        data = super().to_dict()
        data["maps_to"] = list(self.maps_to)
        return data


@dataclass
class Edge:


    id: str
    source: str
    target: str
    edge_type: str = "functional_coupling"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:


        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "weight": float(self.weight),
            **self.metadata,
        }


def normalize_graph_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:

    structural = summary.get("structural_nodes") if isinstance(summary.get("structural_nodes"), Mapping) else {}
    functional = summary.get("functional_nodes") if isinstance(summary.get("functional_nodes"), Mapping) else {}
    mapping = summary.get("functional_to_structural") if isinstance(summary.get("functional_to_structural"), Mapping) else {}
    return {
        "enabled": bool(summary.get("enabled")),
        "structural_nodes": {str(k): dict(v) if isinstance(v, Mapping) else {"id": str(k)} for k, v in structural.items()},
        "functional_nodes": {str(k): dict(v) if isinstance(v, Mapping) else {"id": str(k)} for k, v in functional.items()},
        "functional_to_structural": {str(k): [str(x) for x in v] for k, v in mapping.items() if isinstance(v, list)},
        "outer_loop_contract": dict(summary.get("outer_loop_contract") or {}) if isinstance(summary.get("outer_loop_contract"), Mapping) else {},
    }

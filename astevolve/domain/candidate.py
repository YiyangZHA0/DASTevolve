

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from ._mapping import copied_mapping, copied_sequence_mapping, optional_float


@dataclass(frozen=True)
class Mutation:


    chain_id: str
    position: int
    old_residue: str
    new_residue: str
    structural_node: str = ""
    functional_nodes: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Mutation":


        functional = value.get("functional_nodes") or value.get("functions") or ()
        if isinstance(functional, str):
            functional = (functional,)
        raw_position = value.get("position", value.get("index", value.get("pos", 0)))
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            position = 0
        return cls(
            chain_id=str(value.get("chain_id") or value.get("chain") or ""),
            position=position,
            old_residue=str(
                value.get("old_residue") or value.get("old") or value.get("from") or ""
            ),
            new_residue=str(
                value.get("new_residue") or value.get("new") or value.get("to") or ""
            ),
            structural_node=str(
                value.get("structural_node") or value.get("node") or ""
            ),
            functional_nodes=tuple(str(item) for item in functional),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "position": self.position,
            "old_residue": self.old_residue,
            "new_residue": self.new_residue,
            "structural_node": self.structural_node,
            "functional_nodes": list(self.functional_nodes),
        }


@dataclass(frozen=True)
class Candidate:


    candidate_id: str
    sequences: Mapping[str, str]
    parent_id: str | None = None
    mutations: Tuple[Mutation, ...] = ()
    fast_loss: float | None = None
    generator: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _from_legacy: bool = field(default=False, repr=False, compare=False)
    schema_version: str = "astevolve.candidate.v1"

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any]) -> "Candidate":
        candidate_id = value.get("variant_id") or value.get("candidate_id") or value.get("seq_hash") or "candidate"
        raw_mutations = value.get("mutations") or ()
        known = {
            "variant_id",
            "candidate_id",
            "seq_hash",
            "seqs",
            "sequences",
            "parent_id",
            "mutations",
            "fast_loss",
            "generator",
        }
        return cls(
            candidate_id=str(candidate_id),
            sequences=copied_sequence_mapping(value.get("seqs") or value.get("sequences")),
            parent_id=str(value["parent_id"]) if value.get("parent_id") is not None else None,
            mutations=tuple(
                Mutation.from_mapping(item)
                for item in raw_mutations
                if isinstance(item, Mapping)
            ),
            fast_loss=optional_float(value.get("fast_loss")),
            generator=str(value.get("generator") or value.get("proposal_engine") or ""),
            metadata={key: item for key, item in copied_mapping(value).items() if key not in known},
            raw=copied_mapping(value),
            _from_legacy=True,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:


        if self._from_legacy:
            return copied_mapping(self.raw)
        data = copied_mapping(self.metadata)
        data.update(
            {
                "variant_id": self.candidate_id,
                "seqs": copied_sequence_mapping(self.sequences),
                "parent_id": self.parent_id,
                "fast_loss": self.fast_loss,
                "generator": self.generator,
            }
        )
        if self.mutations:
            data["mutations"] = [mutation.to_dict() for mutation in self.mutations]
        return data



from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable, Tuple


@dataclass
class ConstraintSpec:


    kind: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:


    kind: str
    name: str = ""
    length: Optional[int] = None
    repeat: int = 1
    children: List["Node"] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    props: Dict[str, Any] = field(default_factory=dict)

    residue_spans: Optional[List[Tuple[int, int]]] = None

    def iter_nodes(self) -> Iterable["Node"]:
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def has_explicit_spans(self) -> bool:

        if self.residue_spans is not None:
            return True
        return any(c.has_explicit_spans() for c in self.children)


@dataclass
class Segment:

    kind: str
    name: str
    chain_id: str
    spans: List[Tuple[int, int]]
    props: Dict[str, Any] = field(default_factory=dict)


    @property
    def start(self) -> int:
        return self.spans[0][0] if self.spans else 0

    @property
    def end(self) -> int:
        return self.spans[-1][1] if self.spans else 0

    def indices(self) -> List[int]:

        idx: List[int] = []
        for s, e in self.spans:
            idx.extend(range(s, e))
        return idx

    def extract(self, seq: str) -> str:

        return "".join(seq[i] for i in self.indices())

    def write_into(self, seq_list: List[str], fragment: str) -> None:

        idxs = self.indices()
        if len(fragment) != len(idxs):
            raise ValueError(
                f"Fragment length {len(fragment)} != segment index count {len(idxs)}"
            )
        for i, idx in enumerate(idxs):
            seq_list[idx] = fragment[i]

    @property
    def total_length(self) -> int:
        return sum(e - s for s, e in self.spans)

    @property
    def is_contiguous(self) -> bool:
        if len(self.spans) <= 1:
            return True
        for i in range(len(self.spans) - 1):
            if self.spans[i][1] != self.spans[i + 1][0]:
                return False
        return True


@dataclass
class Blueprint:


    root: Node
    constraints: List[ConstraintSpec] = field(default_factory=list)

    def compile(self) -> Dict[str, Any]:

        segments: List[Segment] = []
        chain_lengths: Dict[str, int] = {}
        chain_order: List[str] = []

        def emit_chain(chain_node: Node):
            if chain_node.kind != "chain":
                raise ValueError("Children of complex must be chain nodes")
            chain_id = chain_node.props.get("chain_id", chain_node.name)
            if not chain_id:
                raise ValueError("chain_id is required")

            chain_order.append(chain_id)


            uses_explicit = any(
                n.residue_spans is not None
                for n in chain_node.iter_nodes()
                if n is not chain_node
            )

            if uses_explicit:
                new_segs = _emit_chain_explicit(chain_node, chain_id)
                segments.extend(new_segs)

                if chain_node.length is not None:
                    chain_lengths[chain_id] = chain_node.length
                else:
                    max_end = 0
                    for seg in new_segs:
                        for _, e in seg.spans:
                            max_end = max(max_end, e)
                    chain_lengths[chain_id] = max_end
            else:
                new_segs, cursor = _emit_chain_sequential(chain_node, chain_id)
                segments.extend(new_segs)
                chain_lengths[chain_id] = cursor

        if self.root.kind == "chain":
            emit_chain(self.root)
        elif self.root.kind == "complex":
            for ch in self.root.children:
                emit_chain(ch)
        else:
            raise ValueError("Blueprint.root.kind must be 'chain' or 'complex'")

        return {
            "segments": segments,
            "chain_lengths": chain_lengths,
            "chain_order": chain_order,
            "blueprint": self,
        }


def _emit_chain_sequential(
    chain_node: Node, chain_id: str
) -> Tuple[List[Segment], int]:

    segments: List[Segment] = []
    cursor = 0

    def emit_subtree(node: Node, repeat_group: Optional[str] = None):
        nonlocal cursor
        for r in range(node.repeat):
            if node.children:
                for ch in node.children:
                    emit_subtree(
                        ch,
                        repeat_group=repeat_group or node.props.get("repeat_group"),
                    )
            else:
                if node.length is None or node.length <= 0:
                    raise ValueError(f"Leaf node must have positive length: {node}")
                seg = Segment(
                    kind=node.kind,
                    name=node.name or node.kind,
                    chain_id=chain_id,
                    spans=[(cursor, cursor + node.length)],
                    props={
                        **node.props,
                        "repeat_group": repeat_group,
                        "repeat_index": r,
                    },
                )
                segments.append(seg)
                cursor += node.length

    for ch in chain_node.children:
        emit_subtree(ch)

    return segments, cursor


def _emit_chain_explicit(
    chain_node: Node, chain_id: str
) -> List[Segment]:

    segments: List[Segment] = []

    def emit_subtree(
        node: Node,
        inherited_spans: Optional[List[Tuple[int, int]]] = None,
        repeat_group: Optional[str] = None,
    ):
        for r in range(node.repeat):

            effective_spans = (
                node.residue_spans if node.residue_spans is not None else inherited_spans
            )

            if node.children:
                for ch in node.children:
                    emit_subtree(
                        ch,
                        inherited_spans=effective_spans,
                        repeat_group=repeat_group or node.props.get("repeat_group"),
                    )
            else:

                if effective_spans is None:
                    raise ValueError(
                        f"In explicit mode, leaf node must have residue_spans "
                        f"(directly or inherited): {node}"
                    )
                seg = Segment(
                    kind=node.kind,
                    name=node.name or node.kind,
                    chain_id=chain_id,
                    spans=list(effective_spans),
                    props={
                        **node.props,
                        "repeat_group": repeat_group,
                        "repeat_index": r,
                    },
                )
                segments.append(seg)

    for ch in chain_node.children:
        emit_subtree(ch)


    all_indices: set = set()
    for seg in segments:
        for idx in seg.indices():
            if idx in all_indices:
                raise ValueError(
                    f"Overlapping spans in chain {chain_id}: index {idx} "
                    f"appears in multiple segments"
                )
            all_indices.add(idx)

    return segments

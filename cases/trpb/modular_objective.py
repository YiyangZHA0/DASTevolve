from __future__ import annotations

import hashlib
from typing import Any, Mapping


def coefficient(seed: int, level: str, node: str, left: str, right: str) -> float:
    payload = f"trpb-program-v1:{seed}:{level}:{node}:{left}:{right}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 2.0 * integer / float(2**64 - 1) - 1.0


def aa_classes(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        residue: name
        for name, residues in protocol["program"]["amino_acid_classes"].items()
        for residue in residues
    }


def residue_classes() -> dict[str, str]:
    return aa_classes({"program": {"amino_acid_classes": {
        "hydrophobic": "AILMV", "aromatic": "FWY", "polar": "CNQST",
        "charged": "DEHKR", "special": "GP"
    }}})


def program_value(row: Mapping[str, Any], seed: int, ast: Mapping[str, Any], classes: Mapping[str, str]) -> float:
    position_to_structural = {
        int(position): str(node["id"])
        for node in ast["structural_ast"]["nodes"]
        for position in node["positions_1based"]
    }
    structural_to_functional = {
        str(child): str(node["id"])
        for node in ast["functional_ast"]["nodes"]
        for child in node["mapped_structural_nodes"]
    }
    mutations = sorted(row["mutations"], key=lambda x: int(x["position_1based"]))
    value = 0.0
    for left_index in range(len(mutations)):
        left = mutations[left_index]
        left_structural = position_to_structural[int(left["position_1based"])]
        left_functional = structural_to_functional[left_structural]
        for right_index in range(left_index + 1, len(mutations)):
            right = mutations[right_index]
            right_structural = position_to_structural[int(right["position_1based"])]
            right_functional = structural_to_functional[right_structural]
            left_class = classes[str(left["alt_aa"])]
            right_class = classes[str(right["alt_aa"])]
            if left_structural == right_structural:
                value += coefficient(seed, "structural", left_structural, left_class, right_class)
            if left_functional == right_functional:
                value += 0.5 * coefficient(seed, "functional", left_functional, left_class, right_class)
    return value

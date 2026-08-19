

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_POLYMER_KINDS = {"protein", "dna", "rna"}
_KIND_ALIASES = {
    "protein": "protein",
    "proteinchain": "protein",
    "protein_chain": "protein",
    "peptide": "protein",
    "polymer": "protein",
    "dna": "dna",
    "dnasequence": "dna",
    "dna_sequence": "dna",
    "rna": "rna",
    "rnasequence": "rna",
    "rna_sequence": "rna",
    "ligand": "ligand",
    "small_molecule": "ligand",
    "small-molecule": "ligand",
    "molecule": "ligand",
    "ion": "ion",
}
_DIRECT_KEYS = {
    "protein": "protein",
    "proteinChain": "protein",
    "dna": "dna",
    "dnaSequence": "dna",
    "rna": "rna",
    "rnaSequence": "rna",
    "ligand": "ligand",
    "ion": "ion",
}


@dataclass(frozen=True)
class _EntitySpec:


    kind: str
    label: str
    count: int
    sequence: str
    token_length: int
    payload: Dict[str, Any]


def _normalise_count(value: Any, *, fallback: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _clean_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def _scalar_label(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _copy_present(
    target: Dict[str, Any],
    source: Mapping[str, Any],
    keys: Iterable[str],
) -> None:
    for key in keys:
        if key in source and source[key] is not None:
            target[key] = source[key]


def _af3_chain_id(index: int) -> str:


    if index < 0:
        raise ValueError("AlphaFold chain index must be non-negative")
    value = index + 1
    chars: List[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _unwrap_entity(
    entity: Mapping[str, Any],
    index: int,
) -> Tuple[str, Dict[str, Any], str]:


    if not isinstance(entity, Mapping):
        raise TypeError(f"Entity #{index} must be a mapping")

    outer_label = _scalar_label(entity.get("id")) or _scalar_label(entity.get("name"))
    for direct_key, kind in _DIRECT_KEYS.items():
        if direct_key not in entity:
            continue
        raw_payload = entity[direct_key]
        if isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)
        elif direct_key in {"ligand", "ion"}:
            payload = {direct_key: raw_payload}
        else:
            raise TypeError(f"Entity #{index} field {direct_key!r} must be a mapping")
        inner_id = payload.get("id")
        inner_label = _scalar_label(inner_id) or _scalar_label(payload.get("name"))
        label = outer_label or inner_label or f"entity{index}"
        return kind, payload, label

    raw_kind = str(entity.get("type") or entity.get("kind") or "").strip().lower()
    try:
        kind = _KIND_ALIASES[raw_kind]
    except KeyError as exc:
        label = outer_label or f"entity{index}"
        raise ValueError(
            f"Unknown AlphaFold 3 entity type for {label!r}; "
            "use protein, dna, rna, ligand, or ion"
        ) from exc
    return kind, dict(entity), outer_label or f"entity{index}"


def _msa_value(payload: Mapping[str, Any], camel_name: str, snake_name: str) -> Any:
    if camel_name in payload:
        return payload[camel_name]
    if snake_name in payload:
        return payload[snake_name]
    return None


def _polymer_payload(
    kind: str,
    payload: Mapping[str, Any],
    label: str,
    *,
    run_data_pipeline: bool,
) -> Tuple[Dict[str, Any], str]:
    sequence = _clean_sequence(payload.get("sequence"))
    if not sequence:
        raise ValueError(f"{kind.upper()} entity {label!r} needs a sequence")

    native: Dict[str, Any] = {"sequence": sequence}
    _copy_present(native, payload, ("modifications", "description"))

    if kind == "protein":
        unpaired = _msa_value(payload, "unpairedMsa", "unpaired_msa")
        if unpaired is None and isinstance(payload.get("msa"), str):
            unpaired = payload["msa"]
        paired = _msa_value(payload, "pairedMsa", "paired_msa")
        templates = payload.get("templates")
        if not run_data_pipeline:
            native["unpairedMsa"] = "" if unpaired is None else unpaired
            native["pairedMsa"] = "" if paired is None else paired
            native["templates"] = [] if templates is None else templates
        else:
            if unpaired is not None:
                native["unpairedMsa"] = unpaired
            if paired is not None:
                native["pairedMsa"] = paired
            if templates is not None:
                native["templates"] = templates
    elif kind == "rna":
        unpaired = _msa_value(payload, "unpairedMsa", "unpaired_msa")
        if unpaired is None and isinstance(payload.get("msa"), str):
            unpaired = payload["msa"]
        if not run_data_pipeline:
            native["unpairedMsa"] = "" if unpaired is None else unpaired
        elif unpaired is not None:
            native["unpairedMsa"] = unpaired

    return native, sequence


def _strip_ccd_prefix(value: Any) -> str:
    code = str(value or "").strip().upper()
    return code[4:] if code.startswith("CCD_") else code


def _ccd_codes(value: Any) -> List[str]:
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    codes = [_strip_ccd_prefix(item) for item in raw_values]
    return [code for code in codes if code]


def _ligand_payload(
    kind: str,
    payload: Mapping[str, Any],
    label: str,
) -> Tuple[Dict[str, Any], int]:
    native: Dict[str, Any] = {}
    if kind == "ion":
        code = _strip_ccd_prefix(
            payload.get("ion")
            or payload.get("ccd")
            or payload.get("code")
            or payload.get("id")
            or label
        )
        if not code:
            raise ValueError(f"Ion entity {label!r} needs an ion or CCD code")
        native["ccdCodes"] = [code]
    else:
        raw_codes = payload.get("ccdCodes")
        if raw_codes is None:
            raw_codes = payload.get("ccd") or payload.get("code")
        smiles = payload.get("smiles")
        legacy_ligand = payload.get("ligand")
        if raw_codes is None and smiles is None and legacy_ligand is not None:
            legacy_text = str(legacy_ligand).strip()
            if legacy_text.upper().startswith("CCD_"):
                raw_codes = legacy_text
            elif legacy_text.upper().startswith("FILE_"):
                raise ValueError(
                    f"Ligand entity {label!r} uses a Protenix FILE_ reference, "
                    "which is not an AlphaFold 3 ligand format"
                )
            else:
                smiles = legacy_text

        codes = _ccd_codes(raw_codes) if raw_codes is not None else []
        if codes and smiles:
            raise ValueError(f"Ligand entity {label!r} cannot specify both CCD and SMILES")
        if codes:
            native["ccdCodes"] = codes
        elif smiles:
            native["smiles"] = str(smiles)
        else:
            raise ValueError(f"Ligand entity {label!r} needs ccd/ccdCodes or smiles")

    if payload.get("description") is not None:
        native["description"] = payload["description"]
    return native, max(1, len(native.get("ccdCodes", [])))


def _normalise_entity(
    entity: Mapping[str, Any],
    index: int,
    *,
    run_data_pipeline: bool,
) -> _EntitySpec:
    kind, payload, label = _unwrap_entity(entity, index)
    raw_ids = payload.get("id")
    inferred_count = len(raw_ids) if isinstance(raw_ids, list) else 1
    count = _normalise_count(payload.get("count", entity.get("count")), fallback=inferred_count)

    if kind in _POLYMER_KINDS:
        native, sequence = _polymer_payload(
            kind,
            payload,
            label,
            run_data_pipeline=run_data_pipeline,
        )
        return _EntitySpec(kind, label, count, sequence, len(sequence), native)

    native, token_length = _ligand_payload(kind, payload, label)
    if label == f"entity{index}" and kind == "ion":
        label = str(native["ccdCodes"][0])
    return _EntitySpec(kind, label, count, "", token_length, native)


def _normalise_seeds(seed: int | Sequence[int]) -> List[int]:
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        seeds = [int(value) for value in seed]
    else:
        seeds = [int(seed)]
    if not seeds:
        raise ValueError("AlphaFold 3 needs at least one model seed")
    return seeds


def _atom_tuple(value: Any) -> Tuple[str, int, str]:
    if isinstance(value, Mapping):
        label = value.get("chain") or value.get("entity") or value.get("id")
        residue = value.get("residue") or value.get("residue_id") or value.get("position")
        atom = value.get("atom") or value.get("atom_name")
        value = (label, residue, atom)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Bond atom must be [entity, residue, atom], got {value!r}")
    label, residue, atom = value
    try:
        residue_id = int(residue)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bond residue index must be an integer, got {residue!r}") from exc
    if residue_id < 1:
        raise ValueError("Bond residue indices are 1-based and must be positive")
    if not str(label or "").strip() or not str(atom or "").strip():
        raise ValueError(f"Bond atom is missing entity or atom name: {value!r}")
    return str(label).strip(), residue_id, str(atom).strip()


def _bond_pair(value: Any) -> Tuple[Any, Any]:
    if isinstance(value, Mapping):
        for left_key, right_key in (
            ("left", "right"),
            ("source", "target"),
            ("atom1", "atom2"),
            ("begin", "end"),
        ):
            if left_key in value and right_key in value:
                return value[left_key], value[right_key]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"Bond must contain exactly two atoms, got {value!r}")


def _map_bond_label(
    label: str,
    *,
    assigned_ids: set[str],
    source_unit_to_id: Mapping[str, str],
    entity_to_ids: Mapping[str, List[str]],
) -> str:
    if label in source_unit_to_id:
        return source_unit_to_id[label]
    if label in assigned_ids:
        return label
    ids = entity_to_ids.get(label)
    if ids and len(ids) == 1:
        return ids[0]
    if ids:
        raise ValueError(
            f"Bond entity {label!r} has {len(ids)} copies; address one as "
            f"{label}_1, {label}_2, ..."
        )
    raise ValueError(f"Bond refers to unknown entity or chain {label!r}")


def _normalise_bonds(
    bonds: Optional[Sequence[Any]],
    *,
    assigned_ids: set[str],
    source_unit_to_id: Mapping[str, str],
    entity_to_ids: Mapping[str, List[str]],
) -> List[List[List[Any]]]:
    out: List[List[List[Any]]] = []
    for raw_bond in bonds or []:
        left_raw, right_raw = _bond_pair(raw_bond)
        pair: List[List[Any]] = []
        for raw_atom in (left_raw, right_raw):
            source_label, residue, atom_name = _atom_tuple(raw_atom)
            af3_id = _map_bond_label(
                source_label,
                assigned_ids=assigned_ids,
                source_unit_to_id=source_unit_to_id,
                entity_to_ids=entity_to_ids,
            )
            pair.append([af3_id, residue, atom_name])
        out.append(pair)
    return out


def build_alphafold3_input(
    pred_name: str,
    entities: Sequence[Mapping[str, Any]],
    *,
    seed: int | Sequence[int] = 101,
    covalent_bonds: Optional[Sequence[Any]] = None,
    bonded_atom_pairs: Optional[Sequence[Any]] = None,
    run_data_pipeline: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    if covalent_bonds is not None and bonded_atom_pairs is not None:
        raise ValueError("Specify only one of covalent_bonds or bonded_atom_pairs")

    specs = [
        _normalise_entity(entity, index, run_data_pipeline=run_data_pipeline)
        for index, entity in enumerate(entities, start=1)
    ]
    labels = [spec.label for spec in specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"AlphaFold 3 source entity labels must be unique: {duplicates}")

    sequences: List[Dict[str, Any]] = []
    entity_report: List[Dict[str, Any]] = []
    polymer_units: List[Tuple[str, str]] = []
    metric_units: List[Tuple[str, str]] = []
    af3_polymer_units: List[Tuple[str, str]] = []
    af3_metric_units: List[Tuple[str, str]] = []
    source_chain_map: Dict[str, str] = {}
    chain_id_map: Dict[str, str] = {}
    entity_id_map: Dict[str, List[str]] = {}

    next_chain_index = 0
    for entity_index, spec in enumerate(specs, start=1):
        af3_ids = [
            _af3_chain_id(next_chain_index + copy_index)
            for copy_index in range(spec.count)
        ]
        next_chain_index += spec.count
        entity_id_map[spec.label] = list(af3_ids)

        payload = dict(spec.payload)
        payload["id"] = af3_ids[0] if len(af3_ids) == 1 else af3_ids
        native_kind = "ligand" if spec.kind == "ion" else spec.kind
        sequences.append({native_kind: payload})
        entity_report.append(
            {
                "entity": entity_index,
                "label": spec.label,
                "source_label": spec.label,
                "kind": spec.kind,
                "count": spec.count,
                "token_length": spec.token_length,
                "af3_ids": list(af3_ids),
            }
        )

        for copy_index, af3_id in enumerate(af3_ids, start=1):
            source_unit = spec.label if spec.count == 1 else f"{spec.label}_{copy_index}"
            source_chain_map[af3_id] = source_unit
            chain_id_map[source_unit] = af3_id
            metric_sequence = spec.sequence if spec.kind in _POLYMER_KINDS else "X"
            metric_units.append((source_unit, metric_sequence))
            af3_metric_units.append((af3_id, metric_sequence))
            if spec.kind in _POLYMER_KINDS:
                polymer_units.append((source_unit, spec.sequence))
                af3_polymer_units.append((af3_id, spec.sequence))

    selected_bonds = bonded_atom_pairs if bonded_atom_pairs is not None else covalent_bonds
    native_bonds = _normalise_bonds(
        selected_bonds,
        assigned_ids=set(source_chain_map),
        source_unit_to_id=chain_id_map,
        entity_to_ids=entity_id_map,
    )

    job: Dict[str, Any] = {
        "name": str(pred_name or "alphafold3_prediction"),
        "modelSeeds": _normalise_seeds(seed),
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }
    if native_bonds:
        job["bondedAtomPairs"] = native_bonds

    preview = {
        "entities": entity_report,
        "polymer_units": polymer_units,
        "metric_units": metric_units,
        "af3_polymer_units": af3_polymer_units,
        "af3_metric_units": af3_metric_units,
        "source_chain_map": source_chain_map,
        "chain_id_map": chain_id_map,
        "entity_id_map": entity_id_map,
        "run_data_pipeline": bool(run_data_pipeline),
    }
    return job, preview


def write_alphafold3_input_json(
    path: Path | str,
    pred_name: str,
    entities: Sequence[Mapping[str, Any]],
    *,
    seed: int | Sequence[int] = 101,
    covalent_bonds: Optional[Sequence[Any]] = None,
    bonded_atom_pairs: Optional[Sequence[Any]] = None,
    run_data_pipeline: bool = False,
) -> Dict[str, Any]:


    job, preview = build_alphafold3_input(
        pred_name,
        entities,
        seed=seed,
        covalent_bonds=covalent_bonds,
        bonded_atom_pairs=bonded_atom_pairs,
        run_data_pipeline=run_data_pipeline,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return {"input_json": str(output_path), **preview}


__all__ = ["build_alphafold3_input", "write_alphafold3_input_json"]

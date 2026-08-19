

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _default_pred_name(chains: List[Tuple[str, str]]) -> str:
    names = [str(cid) for cid, _ in chains if cid]
    return "__".join(names) if names else "pred"


def _normalise_inputs(
    pred_name: Optional[str],
    chains: Optional[List[Tuple[str, str]]],
) -> Tuple[str, List[Tuple[str, str]]]:
    if chains is None and isinstance(pred_name, list):
        chains = pred_name
        pred_name = None

    norm_chains = [(str(cid), str(seq)) for cid, seq in (chains or [])]
    return str(pred_name or _default_pred_name(norm_chains)), norm_chains


def _cache_key(
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
) -> str:
    seq_key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    return f"{pred_name}|{seq_key}|{metric}|{seed}|{model_name}"


def _json_cache_key(
    pred_name: str,
    entities: List[Dict[str, Any]],
    constraint: Optional[Dict[str, Any]],
    covalent_bonds: Optional[List[Dict[str, Any]]],
    metric: str,
    seed: int,
    model_name: str,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "pred_name": pred_name,
        "entities": entities,
        "constraint": constraint,
        "covalent_bonds": covalent_bonds,
        "metric": metric,
        "seed": int(seed),
        "model_name": model_name,
        "runtime_options": runtime_options or {},
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _copy_optional(payload: Dict[str, Any], source: Dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key in source and source[key] is not None:
            payload[key] = source[key]


def _normalise_count(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _ion_code(entity: Dict[str, Any], label: str) -> str:
    ion = str(
        entity.get("ion")
        or entity.get("ccd")
        or entity.get("code")
        or entity.get("id")
        or entity.get("name")
        or ""
    ).strip().upper()
    if ion.startswith("CCD_"):
        ion = ion[4:]
    if not ion:
        raise ValueError(f"Ion entity '{label}' needs an ion code")
    return ion


def _ligand_value(entity: Dict[str, Any]) -> str:
    if entity.get("ligand"):
        return str(entity["ligand"])
    if entity.get("smiles"):
        return str(entity["smiles"])
    if entity.get("ccd"):
        ccd = str(entity["ccd"])
        return ccd if ccd.startswith("CCD_") else f"CCD_{ccd}"
    if entity.get("file"):
        value = str(entity["file"])
        return value if value.startswith("FILE_") else f"FILE_{value}"
    raise ValueError("Ligand entity needs one of: ligand, smiles, ccd, file")


def _normalise_entity(
    entity: Dict[str, Any],
    index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    if not isinstance(entity, dict):
        raise TypeError(f"Entity #{index} must be a dictionary")

    direct_keys = ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
    for key in direct_keys:
        if key in entity and (isinstance(entity[key], dict) or key == "ion"):
            payload = dict(entity[key]) if isinstance(entity[key], dict) else {"ion": entity[key]}
            count = _normalise_count(payload.get("count", 1))
            payload["count"] = count
            label = str(entity.get("id") or entity.get("name") or payload.pop("id", None) or f"entity{index}")
            payload.pop("name", None)
            if key == "ion":
                payload["ion"] = _ion_code({**entity, **payload}, label)
            entry = {key: payload}
            return entry, _entity_info(key, payload, label, count)

    kind = str(entity.get("type") or entity.get("kind") or "").strip().lower()
    label = str(entity.get("id") or entity.get("name") or f"entity{index}")
    count = _normalise_count(entity.get("count", 1))

    if kind in {"protein", "proteinchain", "protein_chain"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"Protein entity '{label}' needs a sequence")
        payload: Dict[str, Any] = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications", "msa"])
        return {"proteinChain": payload}, _entity_info("proteinChain", payload, label, count)

    if kind in {"dna", "dnasequence", "dna_sequence"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"DNA entity '{label}' needs a sequence")
        payload = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications"])
        return {"dnaSequence": payload}, _entity_info("dnaSequence", payload, label, count)

    if kind in {"rna", "rnasequence", "rna_sequence"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"RNA entity '{label}' needs a sequence")
        payload = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications"])
        return {"rnaSequence": payload}, _entity_info("rnaSequence", payload, label, count)

    if kind in {"ligand", "small_molecule", "small-molecule", "molecule"}:
        payload = {"ligand": _ligand_value(entity), "count": count}
        return {"ligand": payload}, _entity_info("ligand", payload, label, count)

    if kind == "ion":
        ion = _ion_code(entity, label)
        payload = {"ion": ion, "count": count}
        return {"ion": payload}, _entity_info("ion", payload, label, count)

    raise ValueError(
        f"Unknown Protenix entity type for '{label}'. Use protein, dna, rna, ligand, or ion."
    )


def _entity_info(key: str, payload: Dict[str, Any], label: str, count: int) -> Dict[str, Any]:
    sequence = str(payload.get("sequence") or "")
    token_length = len(sequence) if sequence else 1
    return {
        "kind": key,
        "label": label,
        "count": count,
        "token_length": token_length,
        "sequence": sequence,
    }


def _normalise_entities(
    entities: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    sequences: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    polymer_units: List[Tuple[str, str]] = []
    metric_units: List[Tuple[str, str]] = []

    for index, raw_entity in enumerate(entities, start=1):
        entry, info = _normalise_entity(raw_entity, index)
        sequences.append(entry)
        report.append({
            "entity": index,
            "label": info["label"],
            "kind": info["kind"],
            "count": info["count"],
            "token_length": info["token_length"],
        })

        for copy_idx in range(1, int(info["count"]) + 1):
            label = info["label"] if int(info["count"]) == 1 else f"{info['label']}_{copy_idx}"
            if info["kind"] in {"proteinChain", "dnaSequence", "rnaSequence"}:
                polymer_units.append((label, info["sequence"]))
                metric_units.append((label, info["sequence"]))
            else:
                metric_units.append((label, "X"))

    return sequences, report, polymer_units, metric_units


def _normalise_constraint(constraint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not constraint:
        return None
    if "constraint" in constraint and isinstance(constraint["constraint"], dict):
        return dict(constraint["constraint"])
    out: Dict[str, Any] = {}
    for key in ("contact", "pocket"):
        if key in constraint and constraint[key]:
            out[key] = constraint[key]
    return out or dict(constraint)


def _write_input_json(path: Path, pred_name: str, chains: List[Tuple[str, str]]) -> None:
    entities = [
        {"type": "protein", "id": cid, "sequence": seq, "count": 1}
        for cid, seq in chains
    ]
    write_protenix_complex_input_json(path, pred_name, entities)


def write_protenix_complex_input_json(
    path: Path | str,
    pred_name: str,
    entities: List[Dict[str, Any]],
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:


    job, preview = _complex_job_payload(
        pred_name,
        entities,
        constraint=constraint,
        covalent_bonds=covalent_bonds,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([job], indent=2), encoding="utf-8")
    preview["input_json"] = str(out_path)
    return preview


def _batch_pred_name(job: Dict[str, Any], index: int) -> str:


    raw_name = job.get("pred_name", job.get("name"))
    name = str(raw_name).strip() if raw_name is not None else ""
    if not name:
        name = f"protenix_complex_{index:04d}"
    if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(
            f"Protenix batch job #{index} has an unsafe pred_name: {name!r}"
        )
    return name


def _complex_job_payload(
    pred_name: str,
    entities: List[Dict[str, Any]],
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sequences, report, polymer_units, metric_units = _normalise_entities(entities)
    job_payload: Dict[str, Any] = {"sequences": sequences, "name": pred_name}
    if covalent_bonds:
        job_payload["covalent_bonds"] = covalent_bonds
    norm_constraint = _normalise_constraint(constraint)
    if norm_constraint:
        job_payload["constraint"] = norm_constraint
    preview = {
        "pred_name": pred_name,
        "entities": report,
        "polymer_units": polymer_units,
        "metric_units": metric_units,
    }
    return job_payload, preview


def write_protenix_complex_batch_input_json(
    path: Path | str,
    jobs: List[Dict[str, Any]],
) -> Dict[str, Any]:


    payload: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    seen_names = set()
    for index, raw_job in enumerate(jobs):
        if not isinstance(raw_job, dict):
            raise TypeError(f"Protenix batch job #{index} must be a dictionary")
        pred_name = _batch_pred_name(raw_job, index)
        if pred_name in seen_names:
            raise ValueError(f"Duplicate Protenix batch pred_name: {pred_name!r}")
        seen_names.add(pred_name)

        entities = raw_job.get("entities")
        if not isinstance(entities, list) or not entities:
            raise ValueError(
                f"Protenix batch job {pred_name!r} needs a non-empty entities list"
            )
        job_payload, preview = _complex_job_payload(
            pred_name,
            entities,
            constraint=raw_job.get("constraint"),
            covalent_bonds=raw_job.get("covalent_bonds"),
        )
        payload.append(job_payload)
        previews.append(preview)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "input_json": str(out_path),
        "job_count": len(payload),
        "jobs": previews,
    }

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _template_root() -> Path:
    raw = str(os.environ.get("ASTEVOLVE_PROTENIX_MSA_TEMPLATE_ROOT") or "").strip()
    if not raw:
        raise RuntimeError(
            "ASTEVOLVE_PROTENIX_MSA_TEMPLATE_ROOT is required when "
            "ASTevolve runs Protenix-v2 with MSA reuse"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Protenix MSA template root does not exist: {root}")
    return root


def _rewrite_query(template: Path, output: Path, sequence: str) -> None:
    candidate = str(sequence).strip().upper()
    if not candidate or any(residue not in _AMINO_ACIDS for residue in candidate):
        raise ValueError("MSA query sequence must contain canonical amino acids only")
    lines = template.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise RuntimeError(f"invalid A3M template: {template}")
    template_query = lines[1].replace("-", "")
    if len(template_query) != len(candidate):
        raise RuntimeError(
            f"MSA template/query length mismatch for {template.name}: "
            f"{len(template_query)} != {len(candidate)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join([">query", candidate, *lines[2:]]) + "\n",
        encoding="utf-8",
    )


def attach_reused_remote_msa(
    entities: Iterable[Dict[str, Any]],
    *,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    """Attach candidate-query A3Ms derived from frozen remote ColabFold results.

    The homolog rows and pair assignments are reused from a remote search on the
    same fixed-length CAB VHH--HEWL system.  Only the first (query) row is changed
    for each candidate, so Protenix receives the exact candidate sequence while
    avoiding a separate public MMseqs2 request for every MCTS rollout.
    """

    root = _template_root()
    prepared: List[Dict[str, Any]] = []
    for index, raw in enumerate(entities):
        entity = dict(raw)
        kind = str(entity.get("type") or entity.get("kind") or "").lower()
        if kind not in {"protein", "proteinchain", "protein_chain"}:
            prepared.append(entity)
            continue
        label = str(entity.get("id") or entity.get("name") or f"entity{index + 1}")
        sequence = str(entity.get("sequence") or "").strip().upper()
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:20]
        msa_dir = run_dir / "candidate_msa"
        for mode, json_key in (
            ("unpaired", "unpairedMsaPath"),
            ("paired", "pairedMsaPath"),
        ):
            template = root / f"{label}.{mode}.template.a3m"
            if not template.is_file():
                raise RuntimeError(f"missing frozen remote MSA template: {template}")
            output = msa_dir / f"{label}.{digest}.{mode}.a3m"
            if not output.is_file():
                _rewrite_query(template, output, sequence)
            entity[json_key] = str(output.resolve())
        prepared.append(entity)
    return prepared


__all__ = ["attach_reused_remote_msa"]

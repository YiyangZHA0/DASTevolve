

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import as_list, evaluator_weight, mean, safe_float, score_at_least

from .base import backend_enabled, backend_item, backend_required, backend_timeout, run_backend_command


def ipsae_candidate_files_from_value(value: Any) -> List[Path]:


    paths: List[Path] = []
    for item in as_list(value):
        if not item:
            continue
        path = Path(str(item))
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*ipsae*.txt"))[:20])
            paths.extend(sorted(path.rglob("*_scores*.txt"))[:20])
            paths.extend(sorted(path.rglob("*_[0-9][0-9]_[0-9][0-9].txt"))[:20])
            paths.extend(sorted(path.rglob("*ipsae*.json"))[:20])
    return paths


def ipsae_score_files(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> List[Path]:


    config = backend_item(score_config, "ipsae")
    candidates: List[Path] = []
    for key in ("score_txt", "scores", "score_file", "path", "json", "output_dir"):
        candidates.extend(ipsae_candidate_files_from_value(config.get(key)))
    for env_key in ("ASTEVOLVE_IPSAE_SCORES", "ASTEVOLVE_IPSAE_SCORE_TXT", "ASTEVOLVE_IPSAE_OUTPUT_DIR"):
        candidates.extend(ipsae_candidate_files_from_value(os.environ.get(env_key)))

    def visit(value: Any) -> None:


        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).lower()
                if "ipsae" in key_text or key_text in {"out_dir", "output_dir"}:
                    candidates.extend(ipsae_candidate_files_from_value(item))
                elif key_text in {"states", "structure_metrics"}:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(structure)
    seen = set()
    result: List[Path] = []
    for path in candidates:
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        if path.name.endswith("_ipsae_af2.json"):
            continue
        if path.name.endswith("_byres.txt"):
            continue
        if path.suffix.lower() not in {".txt", ".json"}:
            continue
        seen.add(resolved)
        result.append(path)
    return result[:30]


def truthy(value: Any, default: bool = False) -> bool:


    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ipsae_auto_generate_score_files(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> Tuple[List[Path], Dict[str, Any], List[str]]:


    config = backend_item(score_config, "ipsae")
    auto_generate = config.get("auto_generate")
    if auto_generate is None:
        auto_generate = os.environ.get("ASTEVOLVE_IPSAE_AUTO_GENERATE")
    if not truthy(auto_generate, default=True):
        return [], {"auto_generate": False}, []

    output_root = (
        structure.get("protenix_out_dir")
        or structure.get("structure_out_dir")
        or structure.get("out_dir")
        or structure.get("protenix_output_dir")
    )
    if not output_root:
        return [], {"auto_generate": True, "reason": "no Protenix output directory available"}, []
    protenix_out = Path(str(output_root))
    if not protenix_out.is_dir():
        return [], {"auto_generate": True, "protenix_out_dir": str(protenix_out), "reason": "Protenix output directory missing"}, []


    project_root = Path(__file__).resolve().parents[3]
    converter = project_root / "scripts" / "protenix_to_ipsae.py"
    if not converter.exists():
        return [], {"auto_generate": True, "reason": f"converter not found: {converter}"}, []

    output_dir = config.get("output_dir") or os.environ.get("ASTEVOLVE_IPSAE_OUTPUT_DIR")
    if output_dir:
        out_dir = Path(str(output_dir))
    else:
        out_dir = protenix_out.parent / "ipsae_auto" / protenix_out.name
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(converter),
        "--protenix-out-dir",
        str(protenix_out),
        "--output-dir",
        str(out_dir),
        "--run",
    ]
    summary_json = structure.get("protenix_summary_json") or structure.get("structure_summary_json") or structure.get("summary_json")
    if summary_json:
        args.extend(["--summary-json", str(summary_json)])
    model_name = structure.get("seq_hash") or structure.get("variant_id")
    if model_name:
        args.extend(["--model-name", str(model_name)])

    command_result = run_backend_command(args, backend_timeout(score_config, "ipsae", 240), cwd=project_root)
    files = ipsae_candidate_files_from_value(out_dir)
    details = {
        "auto_generate": True,
        "protenix_out_dir": str(protenix_out),
        "output_dir": str(out_dir),
        **command_result,
        "generated_score_files": [str(path) for path in files],
    }
    warnings = [] if command_result.get("ok") and files else ["IPSAE auto-generation failed or produced no score file"]
    return files, details, warnings


def parse_ipsae_rows_from_text(text: str) -> List[Dict[str, Any]]:


    rows: List[Dict[str, Any]] = []
    header: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[:2] == ["Chn1", "Chn2"]:
            header = parts
            continue
        if not header or len(parts) < len(header):
            continue
        row: Dict[str, Any] = {}
        for key, value in zip(header, parts):
            if key in {"Chn1", "Chn2", "Type", "Model"}:
                row[key] = value
            else:
                row[key] = safe_float(value, value)
        rows.append(row)
    return rows


def load_ipsae_rows(path: Path) -> List[Dict[str, Any]]:


    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, Mapping) and ("pae" in data or "predicted_aligned_error" in data):
                return []
            if isinstance(data, list):
                return [dict(item) for item in data if isinstance(item, Mapping)]
            if isinstance(data, Mapping):
                for key in ("rows", "ipsae_rows", "ipsae_scores", "scores"):
                    raw = data.get(key)
                    if isinstance(raw, list):
                        return [dict(item) for item in raw if isinstance(item, Mapping)]
                return [dict(data)]
        return parse_ipsae_rows_from_text(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return []


def ipsae_best_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:


    if not rows:
        return None
    preferred = [row for row in rows if str(row.get("Type") or "").lower() == "max"]
    pool = preferred or list(rows)
    return dict(
        max(
            pool,
            key=lambda row: (
                safe_float(row.get("ipSAE"), 0.0) or 0.0,
                safe_float(row.get("pDockQ2"), 0.0) or 0.0,
                safe_float(row.get("LIS"), 0.0) or 0.0,
            ),
        )
    )


def ipsae_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "ipsae"):
        return ScoreTerm("ipsae_interface_confidence", "interface_confidence", 0.0, 0.0, {"enabled": False}, backend="ipsae", available=False)
    config = backend_item(score_config, "ipsae")
    files = ipsae_score_files(structure, score_config)
    auto_details: Dict[str, Any] = {}
    auto_warnings: List[str] = []
    if not files:
        files, auto_details, auto_warnings = ipsae_auto_generate_score_files(structure, score_config)
    if not files:
        return ScoreTerm(
            "ipsae_backend",
            "interface_confidence",
            0.0,
            evaluator_weight(score_config, "eval_ipsae", 0.0),
            {
                "enabled": True,
                "required": backend_required(score_config, "ipsae"),
                "analysis_role": config.get("analysis_role"),
                "reason": "IPSAE backend enabled but no IPSAE score txt/json was found",
                "auto_generation": auto_details,
            },
            auto_warnings or ["IPSAE score file unavailable; run scripts/protenix_to_ipsae.py or provide evaluator_backends.ipsae.score_txt"],
            backend="ipsae",
            available=False,
        )

    rows: List[Dict[str, Any]] = []
    loaded_files: List[str] = []
    for path in files:
        parsed = load_ipsae_rows(path)
        if parsed:
            rows.extend(parsed)
            loaded_files.append(str(path))
    best = ipsae_best_row(rows)
    if not best:
        return ScoreTerm(
            "ipsae_interface_confidence",
            "interface_confidence",
            0.0,
            evaluator_weight(score_config, "eval_ipsae", 0.0),
            {
                "enabled": True,
                "analysis_role": config.get("analysis_role"),
                "score_files": [str(path) for path in files],
                "auto_generation": auto_details,
                "reason": "no parseable IPSAE rows",
            },
            auto_warnings + ["IPSAE files were found but no parseable score rows were available"],
            backend="ipsae",
            available=False,
        )

    ipsae = safe_float(best.get("ipSAE"))
    pdockq = safe_float(best.get("pDockQ"))
    pdockq2 = safe_float(best.get("pDockQ2"))
    lis = safe_float(best.get("LIS"))
    iptm_d0chn = safe_float(best.get("ipTM_d0chn"))
    score = mean(
        [
            score_at_least(ipsae, 0.60, 0.20) if ipsae is not None else None,
            score_at_least(pdockq2, 0.40, 0.05) if pdockq2 is not None else None,
            score_at_least(lis, 0.50, 0.10) if lis is not None else None,
            score_at_least(iptm_d0chn, 0.60, 0.20) if iptm_d0chn is not None else None,
            score_at_least(pdockq, 0.40, 0.05) if pdockq is not None else None,
        ]
    )
    details = {
        "enabled": True,
        "analysis_role": config.get("analysis_role"),
        "score_files": loaded_files,
        "row_count": len(rows),
        "selected_row": best,
        "ipSAE": ipsae,
        "pDockQ": pdockq,
        "pDockQ2": pdockq2,
        "LIS": lis,
        "ipTM_d0chn": iptm_d0chn,
        "method": "IPSAE/pDockQ2/LIS score-file parser; expects output from an external IPSAE tool or scripts/protenix_to_ipsae.py",
    }
    if auto_details:
        details["auto_generation"] = auto_details
    return ScoreTerm(
        "ipsae_interface_confidence",
        "interface_confidence",
        score if score is not None else 0.0,
        evaluator_weight(score_config, "eval_ipsae", 0.0),
        details,
        auto_warnings,
        backend="ipsae",
        available=True,
    )

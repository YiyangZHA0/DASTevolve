"""Build the TIAM1 strict preflight bundle from this checkout and a fresh reference prediction."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from astevolve.providers.protenix_runner import _protenix_command, _protenix_env
from engine.case_preflight import (
    CHECK_REGISTRY, CPU_ONLY_CLAIMS, PREFLIGHT_MANIFEST_VERSION,
    check_registry_hash, run_case_preflight, verify_case_readiness,
)


MODEL = "protenix_mini_esm_v0.5.0"
REFERENCE = "tiam1_reference_apo"


def _write(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_bundle(bundle):
    manifest = bundle / "case.json"
    verify_case_readiness(bundle / "readiness.json", manifest)
    evidence = json.loads((bundle / "recorded_evidence.json").read_text(encoding="utf-8"))
    for item in evidence["provenance"]["source_files"]:
        path = (bundle / item["relative_path"]).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or _sha(path) != item["sha256"]:
            raise ValueError(f"Missing or changed TIAM1 reference artifact: {item['role']}")
    return manifest


def _record_evidence(bundle, manifest):
    state = json.loads((bundle / manifest["design_state_path"]).read_text(encoding="utf-8"))
    sequence = "".join(segment[2] for segment in state["binder"]["pdz_segments"])
    probe = bundle / "reference"
    probe.mkdir()
    request = probe / "input.json"
    _write(request, [{"name": REFERENCE, "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}]}])
    cmd = _protenix_command(
        input_json_path=request, out_dir=probe, seed=101, model_name=MODEL, conda_env=None,
        use_msa=False, cycle=1, step=20, sample=1, use_default_params=False,
    )
    env = _protenix_env()
    # Use the worker's own interpreter for this check, including its optional conda prefix.
    worker_index = next(i for i, value in enumerate(cmd) if Path(value).name == "protenix_worker.py")
    subprocess.run([*cmd[:worker_index], "-c", "import torch; assert torch.cuda.is_available(), 'TIAM1 reference requires CUDA'"],
                   cwd=probe, env=env, check=True, timeout=120)
    subprocess.run(cmd, cwd=probe, env=env, check=True, timeout=1800)
    predictions = probe / REFERENCE / "seed_101" / "predictions"
    files = {
        "input": request,
        "prediction_cif": predictions / f"{REFERENCE}_sample_0.cif",
        "summary_confidence": predictions / f"{REFERENCE}_summary_confidence_sample_0.json",
    }
    confidence = json.loads(files["summary_confidence"].read_text(encoding="utf-8"))
    metrics = {"ptm": float(confidence["ptm"]), "plddt_divided_by_100": float(confidence["plddt"]) / 100}
    if not all(0 <= value <= 1 for value in metrics.values()) or not files["prediction_cif"].stat().st_size:
        raise ValueError("Invalid TIAM1 reference confidence or empty prediction CIF")
    sample_id = _sha(files["summary_confidence"])
    observed = [{"tool": MODEL, "metric": name, "value": score} for name, score in metrics.items()]
    states = [{"tool": MODEL, "metric": name, "state": "apo", "score": score, "available": True}
              for name, score in metrics.items()]
    records = [{
        "record_id": "reference_apo", "evidence_kind": "recorded_tool_smoke",
        "provenance_ref": sample_id, "injection": None, "expected_decision": "accept",
        "evaluator_states": states, "reason": "Measured confidence scores for the reference apo structure.",
    }]
    for kind, scores in (("score_conflict", [1.0, 0.0]), ("missing_evidence", [None])):
        records.append({
            "record_id": kind, "evidence_kind": "deterministic_failure_injection",
            "provenance_ref": f"failure_injection:{kind}", "expected_decision": "abstain",
            "reason": "Deterministic parser control.",
            "injection": {"schema_version": "astevolve.deterministic_failure_injection.v1",
                          "injection_id": kind, "kind": kind, "deterministic": True},
            "evaluator_states": [{"tool": "control", "metric": f"value_{i}", "state": "control",
                                  "score": score, "available": score is not None} for i, score in enumerate(scores)],
        })
    evidence = bundle / "recorded_evidence.json"
    _write(evidence, {
        "schema_version": "astevolve.recorded_evidence_pack.v1", "case_id": manifest["case_id"],
        "source": "TIAM1 reference prediction from the current run",
        "claims": {"biological_success": False, "current_gpu_executed": False,
                   "historical_gpu_artifact_replayed": True, "preflight_execution_cpu_only": True},
        "provenance": {
            "schema_version": "astevolve.recorded_tool_provenance.v1", "provenance_id": sample_id,
            "artifact_kind": "protenix_apo_reference", "historical_execution": "gpu",
            "replay_mode": "frozen_values_no_tool_execution", "observed_metrics": observed,
            "source_files": [{"role": role, "relative_path": path.relative_to(bundle).as_posix(),
                              "sha256": _sha(path)} for role, path in files.items()],
        },
        "records": records,
    })
    return evidence


def prepare_run(manifest: Path, output: Path) -> Path:
    """Create and verify the checked reference pack once per run, outside the checkout."""
    output = output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Choose a TIAM1 output directory outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tiam1-preflight"
    final_manifest = destination / "case.json"
    if final_manifest.exists():
        return _verify_bundle(destination)
    source = manifest.resolve().parent
    config = json.loads(manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=".tiam1-preflight-", dir=output) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name in ("initial_program.py", "tiam1_evaluator.py", "design_state.json", "case_sheet.json",
                     "memory.yaml", "config.yaml", "residue_evidence_catalog.json", "__init__.py"):
            shutil.copyfile(source / name, bundle / name)
        evidence = _record_evidence(bundle, config)
        config["recorded_evidence_path"] = evidence.name
        config["preflight"] = {
            "schema_version": PREFLIGHT_MANIFEST_VERSION, "intended_stage": "cpu_contract_smoke",
            "required_checks": sorted(check.check_id for check in CHECK_REGISTRY),
            "check_registry_hash": check_registry_hash(), "recorded_evidence_sha256": _sha(evidence),
            "timeout_seconds": 90, "claims": CPU_ONLY_CLAIMS,
        }
        _write(bundle / "case.json", config)
        report = run_case_preflight(bundle / "case.json", bundle / "readiness.json")
        if not report.ready_for_smoke:
            raise ValueError(f"TIAM1 preflight failed: {report.failure_codes}")
        os.rename(bundle, destination)
    return _verify_bundle(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="External run directory")
    args = parser.parse_args()
    print(prepare_run(Path(__file__).with_name("case.json"), args.output))

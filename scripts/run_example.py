"""Run a packaged two-level search from a fresh, repository-external output."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "outerloop")]
CASES = {
    "asyn": "asyn_c_terminal_hotspot_binder",
    "cab": "cab_lys3_hewl_cdr_recovery",
    "tiam1": "tiam1_pdz_caspr4_vs_sdc1_selectivity",
    "ttgr": "ttgr_naltrexone_l1_blind",
}


def external(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(PROJECT):
        raise ValueError(f"Choose a directory outside the Git checkout: {path}")
    return path


def stage_inputs(manifest: Path, data_root: Path):
    """Install the packaged immutable inputs, preserving existing external data."""
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    prefix = f"data://{metadata['case_id']}/"
    for entry in metadata.get("required_assets", []):
        path = entry["path"] if isinstance(entry, dict) else entry
        if not path.startswith(prefix):
            continue
        relative = Path(path[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid packaged input path: {path}")
        source = manifest.parent / "inputs" / relative
        if not source.is_file():
            continue
        target = data_root / metadata["case_id"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(source, temporary)
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError(f"Existing input differs from packaged input: {target}. Choose a fresh data root.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--output", type=Path, help="A new output directory")
    parser.add_argument("--iterations", type=int, default=2, help="Number of accepted outer revisions")
    parser.add_argument("--max-attempts", type=int, help="Proposal limit, default 3 per requested revision")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Show paths without installing or running anything")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare the static case inputs and exit")
    args = parser.parse_args()
    attempts = args.max_attempts if args.max_attempts is not None else 3 * args.iterations
    if args.iterations < 1 or attempts < args.iterations:
        parser.error("iterations must be positive and max-attempts must cover the requested revisions")

    slug = CASES[args.case]
    runtime = external(os.getenv("ASTEVOLVE_RUNTIME_ROOT", PROJECT.parent / "DASTevolve_runtime"))
    data = external(os.getenv("ASTEVOLVE_DATA_ROOT", runtime / "datasets"))
    models = external(os.getenv("ASTEVOLVE_MODEL_ROOT", runtime / "models"))
    case_dir = PROJECT / "cases" / slug
    manifest = case_dir / "case.json"
    runs = external(os.getenv("ASTEVOLVE_ARTIFACT_ROOT", runtime / "runs"))
    if args.output:
        args.output = external(args.output)
    if args.plan:
        print(json.dumps(dict(case=slug, manifest=str(manifest), datasets=str(data), models=str(models),
                              output=str(args.output or runs / slug), outer_revisions=args.iterations,
                              inner_rounds=10, proposal_limit=attempts), indent=2))
        return
    os.environ.update(ASTEVOLVE_PROJECT_ROOT=str(PROJECT), ASTEVOLVE_CASE_ID=slug,
                      ASTEVOLVE_CASE_MANIFEST=str(manifest), ASTEVOLVE_RUNTIME_ROOT=str(runtime),
                      ASTEVOLVE_DATA_ROOT=str(data), ASTEVOLVE_MODEL_ROOT=str(models))
    stage_inputs(manifest, data)
    inputs = None
    if (case_dir / "prepare_inputs.py").is_file():
        inputs = importlib.import_module(f"cases.{slug}.prepare_inputs")
        if hasattr(inputs, "prepare"):
            inputs.prepare(data)
    from check_assets import check_case_assets
    assets = check_case_assets(manifest_path=manifest)
    if not assets["ready"]:
        missing = [c["path"] for c in assets["checks"] if c["required"] and not c["exists"]]
        raise ValueError("Missing case inputs:\n" + "\n".join(missing))
    if args.prepare_only:
        print(f"Case inputs ready: {slug}")
        return

    for key in ("ASTEVOLVE_LLM_API_BASE", "ASTEVOLVE_LLM_API_KEY", "ASTEVOLVE_LLM_MODEL"):
        if not os.getenv(key):
            raise ValueError(f"Set {key} before running a live example")
    if args.case == "ttgr":
        # The original TtgR bundle uses the OpenAI SDK's variable names.
        os.environ["OPENAI_API_BASE"] = os.environ["ASTEVOLVE_LLM_API_BASE"]
        os.environ["OPENAI_API_KEY"] = os.environ["ASTEVOLVE_LLM_API_KEY"]
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        output = args.output
    else:
        parent = runs / slug
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    tmp = output / "tmp"
    tmp.mkdir()
    os.environ.update(
        ASTEVOLVE_OPENEVOLVE_SERIAL="1", ASTEVOLVE_INNER_ITERATIONS="10",
        ASTEVOLVE_ARTIFACT_ROOT=str(output / "artifacts"), ASTEVOLVE_RUN_ROOT=str(output / "artifacts"),
        ASTEVOLVE_TMP_ROOT=str(tmp), ASTEVOLVE_MCTS_OUTPUT_DIR=str(output / "inner"),
        ASTEVOLVE_PROTENIX_TMP=str(output / "structures" / "protenix"),
        ASTEVOLVE_ESMFOLD_TMP=str(output / "structures" / "esmfold"), TMPDIR=str(tmp),
    )
    if args.case == "cab":
        os.environ["ASTEVOLVE_FORMAL_INNER_ITERATIONS"] = "10"
    if inputs and hasattr(inputs, "prepare_run"):
        manifest = inputs.prepare_run(manifest, output)
        os.environ["ASTEVOLVE_CASE_MANIFEST"] = str(manifest)

    from astevolve.cases.example_lineage import ExampleLineageRun
    from outerloop.config import load_config
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    bundle = manifest.parent
    config = load_config(bundle / metadata.get("config_path", "config.yaml"))
    config.checkpoint_interval = 1
    config.database.proposal_wave_size = 1
    config.evaluator.parallel_evaluations = 1
    config.evolution_trace.enabled = True
    config.evolution_trace.buffer_size = 1
    evaluator = (bundle / metadata["outer_evaluator_path"]
                 if metadata.get("outer_evaluator_path") else PROJECT / "evaluator.py")
    run = ExampleLineageRun(
        initial_program_path=str(bundle / metadata.get("entry_program", "initial_program.py")),
        evaluation_file=str(evaluator), config=config, output_dir=str(output / "outer"),
    )
    run.inner_rounds = 10
    run.proposal_limit = attempts
    print(f"Running {args.case} in {output}", flush=True)
    asyncio.run(run.run(iterations=args.iterations))
    print(f"Verified lineage: {output / 'outer' / 'verified-rounds.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, RuntimeError) as error:
        sys.exit(str(error))

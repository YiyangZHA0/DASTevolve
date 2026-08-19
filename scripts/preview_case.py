#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.cases import resolve_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    case = resolve_case(manifest_path=args.manifest)
    manifest = case.manifest_path
    if manifest is None:
        raise SystemExit("resolved case has no manifest")
    case_root = manifest.parent
    entry = case_root / str(case.metadata.get("entry_program", "initial_program.py"))
    os.environ["ASTEVOLVE_CASE_ID"] = case.case_id
    os.environ["ASTEVOLVE_CASE_MANIFEST"] = str(manifest)
    if str(case_root) not in sys.path:
        sys.path.insert(0, str(case_root))
    outer_evaluator = case.metadata.get("outer_evaluator_path")
    if outer_evaluator:
        evaluator_path = case_root / str(outer_evaluator)
        evaluator_spec = importlib.util.spec_from_file_location(
            f"dastevolve_preview_evaluator_{case.case_id}", evaluator_path
        )
        if evaluator_spec is None or evaluator_spec.loader is None:
            raise SystemExit(f"cannot load outer evaluator: {evaluator_path}")
        evaluator_module = importlib.util.module_from_spec(evaluator_spec)
        evaluator_spec.loader.exec_module(evaluator_module)
    spec = importlib.util.spec_from_file_location(
        f"dastevolve_preview_{case.case_id}", entry
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load entry program: {entry}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preview = getattr(module, "preview_case", None)
    if not callable(preview):
        raise SystemExit(f"entry program has no preview_case(): {entry}")
    print(json.dumps(preview(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

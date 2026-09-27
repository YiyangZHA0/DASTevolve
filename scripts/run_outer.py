

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.cases import resolve_case
from astevolve.loops.outer import OuterLoopRun
from astevolve.runtime.paths import artifact_path


def main(argv: Optional[Sequence[str]] = None) -> int:


    parser = argparse.ArgumentParser(
        description="Launch OuterLoop for an external ASTevolve case bundle."
    )
    parser.add_argument("--case", default=None, help="Optional case id.")
    parser.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Bundle directory containing case.json, or a collection root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Exact path to case.json; takes precedence over --case-root.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        case = resolve_case(
            args.case,
            case_root=args.case_root,
            manifest_path=args.manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output = args.output or str(artifact_path("outerloop", case.case_id))
    runner = OuterLoopRun(
        project_root=PROJECT_ROOT,
        case_id=case.case_id,
        case_root=args.case_root,
        manifest_path=args.manifest or case.manifest_path,
        output_dir=output,
    )
    cmd = runner.command(iterations=args.iterations)
    if args.conda_env:
        cmd = ["conda", "run", "-n", args.conda_env, *cmd]
    else:


        cmd[0] = sys.executable

    print(" ".join(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["ASTEVOLVE_CASE_ID"] = case.case_id
    if case.manifest_path:
        env["ASTEVOLVE_CASE_MANIFEST"] = str(case.manifest_path)
    env.setdefault("ASTEVOLVE_PROJECT_ROOT", str(PROJECT_ROOT))
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

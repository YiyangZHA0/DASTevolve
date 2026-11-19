"""Recompute Figure 3A-E from recorded queries, or rerun both TrpB searches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results",
                        help="New directory for both searches and Figure 3 panel tables")
    parser.add_argument("--rerun-search", action="store_true", help="Run fresh searches using CUDA")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not args.rerun_search:
        from rescore import rescore
        rescore(out / "figure3")
        return

    for experiment, failed_status in (("archive", "DEVELOPMENT_FAILED"), ("initial_panel", "FAILED")):
        run_dir = out / experiment
        command = [sys.executable, str(ROOT / f"run_{experiment}.py"), "--output-dir", str(run_dir)]
        run = subprocess.run(command)
        if run.returncode == 2 and (run_dir / "RUN.json").is_file():
            report = json.loads((run_dir / "RUN.json").read_text())
            if report["status"] == failed_status:
                print(f"{experiment} completed with unmet study criteria. See {run_dir / 'RUN.json'}.", flush=True)
                continue
        run.check_returncode()

    subprocess.run([sys.executable, str(ROOT / "export_fig3.py"),
                    "--archive", str(out / "archive"), "--initial-panel", str(out / "initial_panel"),
                    "--out", str(out / "figure3")], check=True)


if __name__ == "__main__":
    main()

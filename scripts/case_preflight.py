#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.case_preflight import (
    CHECK_REGISTRY,
    CHECK_REGISTRY_VERSION,
    CasePreflightError,
    check_registry_hash,
    run_case_preflight,
    verify_case_readiness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static case readiness; this does not execute models or claim biological success."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listed = commands.add_parser("list-checks", help="print the closed C0-C3 registry")
    listed.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    run = commands.add_parser("run", help="run checks and always write a report")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--report", required=True, type=Path)
    run.add_argument("--timeout-seconds", type=float)

    verify = commands.add_parser("verify", help="verify a current, ready report")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-checks":
        payload = {
            "registry_version": CHECK_REGISTRY_VERSION,
            "registry_hash": check_registry_hash(),
            "checks": [item.to_dict() for item in CHECK_REGISTRY],
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"{payload['registry_version']} {payload['registry_hash']}")
            for item in CHECK_REGISTRY:
                print(f"{item.level}\t{item.scope}\t{item.check_id}\t{item.description}")
        return 0
    try:
        if args.command == "run":
            report = run_case_preflight(
                args.manifest,
                args.report,
                timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(report.to_dict(), sort_keys=True))
            return 0 if report.ready_for_smoke else 2
        report = verify_case_readiness(args.report, args.manifest)
        print(
            json.dumps(
                {
                    "verified": True,
                    "case_id": report.case_id,
                    "report_hash": report.report_hash,
                    "ready_for_smoke": report.ready_for_smoke,
                    "claims": dict(report.claims),
                },
                sort_keys=True,
            )
        )
        return 0
    except (CasePreflightError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "verified": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

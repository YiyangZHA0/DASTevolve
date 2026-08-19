

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.cases import resolve_case


def _declared_assets(
    bundle_root: Path,
    manifest: Mapping[str, Any],
) -> Iterable[tuple[str, Path, bool]]:


    for index, value in enumerate(manifest.get("required_assets") or []):
        if isinstance(value, str):
            yield f"required_asset:{index}", bundle_root / value, True
            continue
        if not isinstance(value, Mapping) or not value.get("path"):
            raise ValueError(
                "required_assets entries must be paths or objects with path"
            )
        path = Path(str(value["path"])).expanduser()
        if not path.is_absolute():
            path = bundle_root / path
        yield (
            str(value.get("role") or f"required_asset:{index}"),
            path,
            bool(value.get("required", True)),
        )


def check_case_assets(
    *,
    case_id: str | None = None,
    case_root: Path | None = None,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:


    case = resolve_case(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    )
    if case.manifest_path is None:
        raise ValueError("resolved case has no manifest")
    manifest = json.loads(
        case.manifest_path.read_text(encoding="utf-8-sig")
    )
    bundle_root = case.manifest_path.parent
    entry_program = bundle_root / str(
        case.metadata.get("entry_program", "initial_program.py")
    )
    rows = [
        ("manifest", case.manifest_path, True),
        ("design_state", case.design_state_path, True),
        ("memory", case.memory_path, False),
        ("entry_program", entry_program, True),
        *_declared_assets(bundle_root, manifest),
    ]
    checks = [
        {
            "role": role,
            "path": str(path.expanduser().resolve()),
            "required": required,
            "exists": path.expanduser().exists(),
        }
        for role, path, required in rows
    ]
    missing_required = [
        row["role"]
        for row in checks
        if row["required"] and not row["exists"]
    ]
    return {
        "schema_version": "astevolve.case_asset_check.v1",
        "case_id": case.case_id,
        "manifest": str(case.manifest_path),
        "ready": not missing_required,
        "missing_required": missing_required,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check assets declared by an external ASTevolve case."
    )
    parser.add_argument("--case", default=None)
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = check_case_assets(
            case_id=args.case,
            case_root=args.case_root,
            manifest_path=args.manifest,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())



from __future__ import annotations

from pathlib import Path

from astevolve.evaluation.outerloop import evaluate, validate_candidate


def _default_program(
    case_id: str | None = None,
    *,
    case_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> str:
    from astevolve.cases import resolve_case

    case = resolve_case(
        case_id,
        case_root=case_root,
        manifest_path=manifest_path,
    )
    entry = str(case.metadata.get("entry_program", "initial_program.py"))
    bundle_root = case.manifest_path.parent if case.manifest_path else case.root
    return str(bundle_root / entry)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate an ASTevolve program through the OuterLoop shim."
    )
    parser.add_argument("program", nargs="?")
    parser.add_argument("--case", default=None)
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    program = args.program or _default_program(
        args.case,
        case_root=args.case_root,
        manifest_path=args.manifest,
    )
    print(evaluate(program))


if __name__ == "__main__":
    main()

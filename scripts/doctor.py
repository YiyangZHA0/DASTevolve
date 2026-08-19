#!/usr/bin/env python3


from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTERLOOP_SOURCE_ROOT = PROJECT_ROOT / "outerloop"
MINIMUM_PYTHON = (3, 10)
MAXIMUM_PYTHON = (3, 14)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    required: bool
    detail: str
    hint: str | None = None


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        name: str,
        status: str,
        detail: str,
        *,
        required: bool = False,
        hint: str | None = None,
    ) -> None:
        if status not in {"ok", "warning", "error"}:
            raise ValueError(f"unsupported status: {status}")
        self.checks.append(
            Check(
                name=name,
                status=status,
                required=required,
                detail=detail,
                hint=hint,
            )
        )

    @property
    def errors(self) -> list[Check]:
        return [item for item in self.checks if item.status == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [item for item in self.checks if item.status == "warning"]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "dastevolve.doctor_report.v1",
            "project_root": str(PROJECT_ROOT),
            "ok": not self.errors,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": [asdict(item) for item in self.checks],
        }


def _resolved_env_path(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    return (
        Path(raw).expanduser().resolve()
        if raw and raw.strip()
        else fallback.expanduser().resolve()
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _check_python(report: Report) -> None:
    current = sys.version_info[:3]
    supported = MINIMUM_PYTHON <= current < MAXIMUM_PYTHON
    report.add(
        "python.version",
        "ok" if supported else "error",
        f"{sys.version.split()[0]} at {sys.executable}",
        required=True,
        hint="Use CPython 3.10, 3.11, 3.12, or 3.13." if not supported else None,
    )


def _check_layout(report: Report) -> None:
    required_paths = (
        "README.md",
        "pyproject.toml",
        "astevolve/__init__.py",
        "engine/__init__.py",
        "outerloop/outerloop/__init__.py",
        "scripts/run_outer.py",
    )
    missing = [name for name in required_paths if not (PROJECT_ROOT / name).is_file()]
    report.add(
        "repository.layout",
        "error" if missing else "ok",
        "missing: " + ", ".join(missing) if missing else "required source files present",
        required=True,
    )
    configured = os.environ.get("ASTEVOLVE_PROJECT_ROOT")
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        report.add(
            "repository.configured_root",
            "ok" if configured_path == PROJECT_ROOT else "error",
            f"configured={configured_path}; detected={PROJECT_ROOT}",
            required=True,
            hint="Set ASTEVOLVE_PROJECT_ROOT to this checkout."
            if configured_path != PROJECT_ROOT
            else None,
        )


def _check_dependencies(report: Report) -> None:
    modules = {
        "dacite": "dacite",
        "numpy": "numpy",
        "openai": "openai",
        "PyYAML": "yaml",
    }
    missing = [
        distribution
        for distribution, module in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    report.add(
        "python.core_dependencies",
        "error" if missing else "ok",
        "missing: " + ", ".join(missing) if missing else "all core dependencies discoverable",
        required=True,
        hint='Run: python -m pip install -e "."' if missing else None,
    )


def _check_imports(report: Report) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(OUTERLOOP_SOURCE_ROOT))
    modules = (
        "astevolve.domain",
        "astevolve.evolution",
        "astevolve.search",
        "astevolve.evaluation",
        "astevolve.providers",
        "engine.case_builder",
        "outerloop",
    )
    failures: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    report.add(
        "python.core_imports",
        "error" if failures else "ok",
        "; ".join(failures) if failures else "core and compatibility imports passed",
        required=True,
        hint="Inspect the first import failure; core imports must remain CPU-safe."
        if failures
        else None,
    )


def _runtime_roots() -> dict[str, Path]:
    runtime = _resolved_env_path(
        "ASTEVOLVE_RUNTIME_ROOT", PROJECT_ROOT.parent / "DASTevolve_runtime"
    )
    artifact = _resolved_env_path("ASTEVOLVE_ARTIFACT_ROOT", runtime / "runs")
    return {
        "runtime": runtime,
        "cases": _resolved_env_path("ASTEVOLVE_CASE_ROOT", runtime / "cases"),
        "models": _resolved_env_path("ASTEVOLVE_MODEL_ROOT", runtime / "models"),
        "tools": _resolved_env_path("ASTEVOLVE_TOOL_ROOT", runtime / "tools"),
        "datasets": _resolved_env_path("ASTEVOLVE_DATA_ROOT", runtime / "datasets"),
        "runs": _resolved_env_path("ASTEVOLVE_RUN_ROOT", artifact),
        "tmp": _resolved_env_path("ASTEVOLVE_TMP_ROOT", runtime / "tmp"),
        "containers": _resolved_env_path(
            "ASTEVOLVE_CONTAINER_ROOT", runtime / "containers"
        ),
    }


def _check_runtime(report: Report, *, strict: bool) -> dict[str, Path]:
    roots = _runtime_roots()
    runtime = roots["runtime"]
    external = not _is_within(runtime, PROJECT_ROOT)
    report.add(
        "runtime.external_root",
        "ok" if external else "error",
        str(runtime),
        required=True,
        hint="Choose a runtime root outside the Git checkout." if not external else None,
    )
    missing = [name for name, path in roots.items() if not path.is_dir()]
    report.add(
        "runtime.directories",
        "error" if strict and missing else ("warning" if missing else "ok"),
        "missing: " + ", ".join(missing) if missing else "external runtime layout present",
        required=strict,
        hint="Run scripts/bootstrap_linux.sh or create the listed directories."
        if missing
        else None,
    )
    existing_parent = runtime
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    writable = existing_parent.is_dir() and os.access(existing_parent, os.W_OK)
    report.add(
        "runtime.writable",
        "ok" if writable else "error",
        f"nearest existing parent: {existing_parent}",
        required=True,
        hint="Choose a runtime location writable by the current user."
        if not writable
        else None,
    )
    return roots


def _case_manifests(case_root: Path) -> list[Path]:
    explicit = os.environ.get("ASTEVOLVE_CASE_MANIFEST")
    if explicit and explicit.strip():
        return [Path(explicit).expanduser().resolve()]
    if (case_root / "case.json").is_file():
        return [case_root / "case.json"]
    if not case_root.is_dir():
        return []
    return sorted(path for path in case_root.glob("*/case.json") if path.is_file())


def _check_cases(
    report: Report, roots: dict[str, Path], *, require_case: bool
) -> None:
    manifests = _case_manifests(roots["cases"])
    malformed: list[str] = []
    for path in manifests:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise TypeError("top level is not an object")
        except (OSError, ValueError, TypeError) as exc:
            malformed.append(f"{path}: {exc}")
    if malformed:
        report.add(
            "cases.manifests",
            "error",
            "; ".join(malformed),
            required=True,
            hint="Fix or remove the invalid selected manifest.",
        )
    elif manifests:
        report.add(
            "cases.manifests",
            "ok",
            f"{len(manifests)} manifest(s): "
            + ", ".join(str(path) for path in manifests[:3]),
            required=require_case,
        )
    else:
        report.add(
            "cases.manifests",
            "error" if require_case else "warning",
            "no external case manifest selected",
            required=require_case,
            hint="Set ASTEVOLVE_CASE_MANIFEST or ASTEVOLVE_CASE_ROOT.",
        )


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _check_optional_tools(report: Report, roots: dict[str, Path]) -> None:
    tool_root = roots["tools"]
    tools: dict[str, tuple[Path | None, str]] = {
        "IPSAE": (
            _first_existing(
                (
                    Path(os.environ["ASTEVOLVE_IPSAE_ROOT"]).expanduser()
                    if os.environ.get("ASTEVOLVE_IPSAE_ROOT")
                    else tool_root / "IPSAE",
                    tool_root / "ipsae",
                )
            ),
            "resources/external-tools.lock.json",
        ),
        "getContacts": (
            _first_existing(
                (
                    Path(os.environ["ASTEVOLVE_GETCONTACTS_ROOT"]).expanduser()
                    if os.environ.get("ASTEVOLVE_GETCONTACTS_ROOT")
                    else tool_root / "getcontacts",
                )
            )
            or (
                Path(found)
                if (found := shutil.which("get_static_contacts.py"))
                else None
            ),
            "resources/external-tools.lock.json",
        ),
        "AlphaFold 3": (
            _first_existing(
                (
                    Path(os.environ["ASTEVOLVE_AF3_ROOT"]).expanduser()
                    if os.environ.get("ASTEVOLVE_AF3_ROOT")
                    else tool_root / "alphafold3",
                )
            ),
            "resources/external-tools.lock.json and upstream license terms",
        ),
    }
    available = [name for name, (path, _) in tools.items() if path is not None]
    absent = [name for name, (path, _) in tools.items() if path is None]
    detail = "available: " + (", ".join(available) if available else "<none>")
    if absent:
        detail += "; unavailable: " + ", ".join(absent)
    report.add(
        "tools.optional",
        "ok" if available else "warning",
        detail,
        required=False,
        hint="Optional tools are installed only when a selected case requires them.",
    )


def _check_gpu(
    report: Report, roots: dict[str, Path], *, require_gpu: bool
) -> None:
    modules = ("torch", "transformers", "protenix")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    status = "error" if require_gpu and missing else ("warning" if missing else "ok")
    report.add(
        "gpu.python_packages",
        status,
        "missing: " + ", ".join(missing) if missing else "GPU provider packages discoverable",
        required=require_gpu,
        hint='Install a CUDA-compatible PyTorch build and then ".[gpu]".'
        if missing
        else None,
    )

    cuda_detail = "not checked"
    cuda_ok = False
    if require_gpu and "torch" not in missing:
        try:
            torch = importlib.import_module("torch")
            cuda_ok = bool(torch.cuda.is_available())
            cuda_detail = (
                f"torch={torch.__version__}; cuda_available={cuda_ok}; "
                f"device_count={torch.cuda.device_count()}"
            )
        except Exception as exc:
            cuda_detail = f"{type(exc).__name__}: {exc}"
    report.add(
        "gpu.cuda",
        ("ok" if cuda_ok else "error") if require_gpu else "warning",
        cuda_detail if require_gpu else "not required; heavyweight import skipped",
        required=require_gpu,
        hint="Run this check on an allocated GPU node with the correct CUDA build."
        if require_gpu and not cuda_ok
        else None,
    )

    model_candidates = {
        "ProGen2": Path(
            os.environ.get(
                "ASTEVOLVE_PROGEN_MODEL_DIR", roots["models"] / "progen2-small"
            )
        ).expanduser(),
        "Protenix": Path(
            os.environ.get(
                "ASTEVOLVE_PROTENIX_ROOT", roots["models"] / "protenix"
            )
        ).expanduser(),
        "ESMFold2": Path(
            os.environ.get(
                "ASTEVOLVE_ESMFOLD2_MODEL",
                roots["models"] / "biohub" / "ESMFold2-Fast",
            )
        ).expanduser(),
        "AlphaFold 3": Path(
            os.environ.get(
                "ASTEVOLVE_AF3_MODEL_DIR", roots["models"] / "alphafold3"
            )
        ).expanduser(),
    }
    present = [name for name, path in model_candidates.items() if path.exists()]
    report.add(
        "models.optional_assets",
        "ok" if present else "warning",
        "present: " + (", ".join(present) if present else "<none>"),
        required=False,
        hint="Populate only case-required assets using a private copy of resources/model-assets.example.json.",
    )


def _check_llm(report: Report, *, require_llm: bool) -> None:
    base = os.environ.get("ASTEVOLVE_LLM_API_BASE", "").strip()
    key = os.environ.get("ASTEVOLVE_LLM_API_KEY", "").strip()
    configured = bool(base and key)
    report.add(
        "llm.configuration",
        "ok" if configured else ("error" if require_llm else "warning"),
        "OpenAI-compatible base and key configured"
        if configured
        else "LLM endpoint/key not configured",
        required=require_llm,
        hint="Export ASTEVOLVE_LLM_API_BASE and ASTEVOLVE_LLM_API_KEY at runtime."
        if require_llm and not configured
        else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a DASTevolve clone and external runtime without downloading assets."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require the complete external runtime directory skeleton",
    )
    parser.add_argument(
        "--require-case",
        action="store_true",
        help="fail when no valid external case manifest is selected",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="require GPU packages and an available CUDA device",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="require an OpenAI-compatible endpoint and key",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = Report()
    _check_python(report)
    _check_layout(report)
    _check_dependencies(report)
    _check_imports(report)
    roots = _check_runtime(report, strict=args.strict)
    _check_cases(report, roots, require_case=args.require_case)
    _check_optional_tools(report, roots)
    _check_gpu(report, roots, require_gpu=args.require_gpu)
    _check_llm(report, require_llm=args.require_llm)

    payload = report.payload()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        labels = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
        for item in report.checks:
            print(f"[{labels[item.status]:5}] {item.name}: {item.detail}")
            if item.hint:
                print(f"        hint: {item.hint}")
        print(
            f"\nDASTevolve doctor: "
            f"{'ready' if not report.errors else 'not ready'} "
            f"({len(report.errors)} error(s), {len(report.warnings)} warning(s))"
        )
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

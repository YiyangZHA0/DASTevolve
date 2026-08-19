

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_command(configured: str | None, candidates: list[str]) -> str | None:
    if configured:
        path = shutil.which(configured) or configured
        if shutil.which(path) or Path(path).exists():
            return path
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _run(args: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        text = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode == 0, text[-4000:]
    except Exception as exc:
        return False, str(exc)


def check_rosetta() -> bool:


    command = _find_command(
        os.environ.get("ASTEVOLVE_ROSETTA_INTERFACE_ANALYZER"),
        [
            "InterfaceAnalyzer",
            "InterfaceAnalyzer.default.linuxgccrelease",
            "interface_analyzer.default.linuxgccrelease",
        ],
    )
    if not command:
        print("[rosetta] missing: set ASTEVOLVE_ROSETTA_INTERFACE_ANALYZER to the InterfaceAnalyzer executable")
        return False
    ok, text = _run([command, "-help"], timeout=30)
    print(f"[rosetta] command: {command}")
    print(f"[rosetta] help_ok: {ok}")
    if not ok:
        print(text)
    return ok


def check_pyrosetta() -> bool:


    python_bin = os.environ.get("ASTEVOLVE_PYROSETTA_PYTHON") or sys.executable
    ok, text = _run(
        [
            python_bin,
            "-c",
            (
                "import os; import pyrosetta; "
                "pyrosetta.init(os.environ.get('ASTEVOLVE_PYROSETTA_INIT_OPTS', '-mute all')); "
                "print('pyrosetta ok')"
            ),
        ],
        timeout=60,
    )
    print(f"[pyrosetta] python: {python_bin}")
    print(f"[pyrosetta] import_init_ok: {ok}")
    if not ok:
        print(text)
    return ok


def check_getcontacts() -> bool:


    command = _find_command(
        os.environ.get("ASTEVOLVE_GET_STATIC_CONTACTS_BIN") or os.environ.get("ASTEVOLVE_GETCONTACTS_BIN"),
        ["get_static_contacts.py", "get_static_contacts"],
    )
    if not command:
        print("[getcontacts] missing: set ASTEVOLVE_GET_STATIC_CONTACTS_BIN to get_static_contacts.py")
        return False
    args = [command, "--help"]
    path = Path(command)
    if path.suffix.lower() == ".py" and not os.access(str(path), os.X_OK):
        args = [sys.executable, *args]
    import_ok, import_text = (
        _run(args[:1] + ["-c", "import vmd; import numpy; print('vmd ok')"], timeout=30)
        if Path(args[0]).name.startswith("python")
        else (None, "")
    )
    print(f"[getcontacts] command: {command}")
    if import_ok is not None:
        print(f"[getcontacts] vmd_python_ok: {import_ok}")
        if not import_ok:
            print(import_text)
            return False
    ok, text = _run(args, timeout=30)
    print(f"[getcontacts] help_ok: {ok}")
    if not ok:
        print(text)
    return ok


def main() -> int:


    parser = argparse.ArgumentParser(description="Check ASTevolve physical evaluator backends.")
    parser.add_argument("--require-rosetta", action="store_true")
    parser.add_argument("--require-pyrosetta", action="store_true", default=False)
    parser.add_argument("--require-getcontacts", action="store_true")
    args = parser.parse_args()

    rosetta_ok = check_rosetta()
    pyrosetta_ok = check_pyrosetta()
    getcontacts_ok = check_getcontacts()
    failed = []
    if args.require_rosetta and not rosetta_ok:
        failed.append("rosetta")
    if args.require_pyrosetta and not pyrosetta_ok:
        failed.append("pyrosetta")
    if args.require_getcontacts and not getcontacts_ok:
        failed.append("getcontacts")
    if failed:
        print(f"missing required backends: {', '.join(failed)}")
        return 2
    print("all required physics backends are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

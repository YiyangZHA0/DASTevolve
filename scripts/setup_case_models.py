"""Fetch or verify the exact model assets used by the four portable cases."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "case_models.json"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def acquire(path, url, expected):
    import fcntl  # Downloads use a Linux file lock. Offline checks also work on Windows.

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if digest(path) != expected:
                raise ValueError(f"SHA-256 mismatch for {path}; move the existing file aside before retrying")
            return
        part = path.with_name(path.name + ".part")
        curl = [
            "curl", "--fail", "--location", "--silent", "--show-error", "--retry", "4",
            "--retry-all-errors", "--connect-timeout", "30", "--speed-limit", "512",
            "--speed-time", "150", "--output", str(part),
        ]
        if part.exists() and digest(part) == expected:
            part.replace(path)
            return
        if part.exists():
            download = subprocess.run([*curl, "--continue-at", "-", url])
            if download.returncode in (22, 33, 36):
                # A stale/complete range or an origin with no range support.
                subprocess.run([*curl, url], check=True)
            else:
                download.check_returncode()
        else:
            subprocess.run([*curl, url], check=True)
        if digest(part) != expected:
            part.unlink()
            raise ValueError(f"Downloaded SHA-256 did not match for {path.name}")
        part.replace(path)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=manifest["cases"], required=True)
    parser.add_argument("--runtime-root", type=Path, required=True, help="External DASTevolve runtime directory")
    parser.add_argument("--model-root", type=Path, help="Overrides ASTEVOLVE_MODEL_ROOT and RUNTIME/models")
    parser.add_argument("--check-only", action="store_true", help="Check all hashes offline without writing files")
    args = parser.parse_args()
    root = args.runtime_root.expanduser().resolve()
    if root == ROOT or ROOT in root.parents:
        parser.error("--runtime-root must be outside the Git checkout")
    model_root = (args.model_root or Path(os.environ.get("ASTEVOLVE_MODEL_ROOT") or root / "models")).expanduser().resolve()
    if model_root == ROOT or ROOT in model_root.parents:
        parser.error("model storage must be outside the Git checkout")

    errors = []
    for group in manifest["cases"][args.case]:
        model = manifest["models"][group]
        for name, sha in model["files"].items():
            path = model_root / model["folder"] / name
            print(f"{'Checking' if args.check_only else 'Preparing'} {path}", flush=True)
            if args.check_only:
                if not path.is_file() or digest(path) != sha:
                    errors.append(str(path))
            else:
                url = model["url"] + name
                if group == "protenix-v2":
                    url = os.environ.get("ASTEVOLVE_PROTENIX_V2_URL") or url
                try:
                    acquire(path, url, sha)
                except subprocess.CalledProcessError as exc:
                    if group == "protenix-v2":
                        raise ValueError(
                            f"Could not download Protenix-v2 (curl status {exc.returncode}). "
                            f"Place an authorized checkpoint at {path} or set "
                            "ASTEVOLVE_PROTENIX_V2_URL to its authorized download URL. "
                            f"Required SHA-256: {sha}."
                        ) from None
                    raise
    if errors:
        print("Missing or incorrect files:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))

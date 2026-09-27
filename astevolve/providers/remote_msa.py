from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_MSA_SERVER = "https://api.colabfold.com"
DEFAULT_USER_AGENT = "DASTevolve/0.1 remote-msa"
_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


class RemoteMSAError(RuntimeError):
    pass


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalise_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).split()).upper()
    invalid = sorted(set(sequence) - _VALID_AA)
    if not sequence or invalid:
        raise ValueError(f"Invalid protein sequence; unsupported symbols: {invalid!r}")
    return sequence


def _cache_root() -> Path:
    configured = os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_CACHE")
    if configured:
        root = Path(configured)
    else:
        runtime_root = os.environ.get("ASTEVOLVE_RUNTIME_ROOT")
        root = (
            Path(runtime_root) / "datasets" / "msa_cache"
            if runtime_root
            else Path.home() / ".cache" / "dastevolve" / "msa"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _request(
    url: str,
    *,
    data: Optional[Mapping[str, str]] = None,
    timeout: float,
    user_agent: str,
) -> bytes:
    body = urllib.parse.urlencode(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": user_agent},
        method="POST" if body is not None else "GET",
    )
    retries = max(1, int(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_NETWORK_RETRIES", "5")))
    last_error: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))
    raise RemoteMSAError(f"Remote MSA request failed for {url}: {last_error}") from last_error


def _json_request(
    url: str,
    *,
    data: Optional[Mapping[str, str]] = None,
    timeout: float,
    user_agent: str,
) -> Dict[str, Any]:
    raw = _request(url, data=data, timeout=timeout, user_agent=user_agent)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RemoteMSAError(f"Remote MSA server returned non-JSON data for {url}") from exc
    if not isinstance(payload, dict):
        raise RemoteMSAError(f"Remote MSA server returned an invalid response for {url}")
    return payload


def _iter_a3m_entries(text: str) -> Iterable[Tuple[str, str]]:
    header: Optional[str] = None
    sequence: List[str] = []
    for raw_line in text.replace("\x00", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(sequence)
            header, sequence = line[1:], []
        elif header is not None:
            sequence.append(line)
    if header is not None:
        yield header, "".join(sequence)


def _format_a3m(entries: Iterable[Tuple[str, str]]) -> str:
    return "".join(f">{header}\n{sequence}\n" for header, sequence in entries)


def _read_result_members(payload: bytes, use_env: bool) -> str:
    wanted = ["uniref.a3m"]
    if use_env:
        wanted.append("bfd.mgnify30.metaeuk30.smag30.a3m")
    contents: List[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            regular = {
                Path(member.name).name: member
                for member in archive.getmembers()
                if member.isfile()
            }
            for filename in wanted:
                member = regular.get(filename)
                if member is None:
                    if filename == "uniref.a3m":
                        raise RemoteMSAError(f"Remote MSA archive is missing {filename}")
                    continue
                stream = archive.extractfile(member)
                if stream is not None:
                    contents.append(stream.read().decode("utf-8"))
    except (tarfile.TarError, UnicodeDecodeError) as exc:
        raise RemoteMSAError("Remote MSA server returned an invalid result archive") from exc

    merged: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for file_index, content in enumerate(contents):
        for entry_index, entry in enumerate(_iter_a3m_entries(content)):
            if file_index > 0 and entry_index == 0:
                continue
            if entry not in seen:
                merged.append(entry)
                seen.add(entry)
    if not merged:
        raise RemoteMSAError("Remote MSA result did not contain any aligned sequences")
    return _format_a3m(merged)


def fetch_colabfold_msa(
    sequence: str,
    *,
    cache_dir: Optional[str | Path] = None,
    host_url: Optional[str] = None,
    use_env: Optional[bool] = None,
    use_filter: Optional[bool] = None,
    user_agent: Optional[str] = None,
    request_timeout: Optional[float] = None,
    overall_timeout: Optional[float] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Fetch one unpaired A3M from the public ColabFold MMseqs2 service."""

    sequence = _normalise_sequence(sequence)
    host = str(host_url or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_URL") or DEFAULT_MSA_SERVER).rstrip("/")
    include_env = use_env if use_env is not None else _as_bool(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_USE_ENV"), True)
    filter_hits = use_filter if use_filter is not None else _as_bool(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_USE_FILTER"), True)
    mode = "env" if include_env else "all"
    if not filter_hits:
        mode = "env-nofilter" if include_env else "nofilter"
    agent = str(user_agent or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_USER_AGENT") or DEFAULT_USER_AGENT)
    per_request = float(request_timeout or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_REQUEST_TIMEOUT", "30"))
    total = float(overall_timeout or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_TIMEOUT", "1800"))
    root = Path(cache_dir) if cache_dir else _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{host}\0{mode}\0{sequence}".encode("utf-8")).hexdigest()
    a3m_path = root / f"{digest}.a3m"
    metadata_path = root / f"{digest}.json"
    if a3m_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {**metadata, "path": str(a3m_path), "cached": True}

    query = f">101\n{sequence}\n"
    deadline = time.monotonic() + total
    attempt = 0
    while True:
        response = _json_request(
            f"{host}/ticket/msa",
            data={"q": query, "mode": mode},
            timeout=per_request,
            user_agent=agent,
        )
        status = str(response.get("status", "ERROR")).upper()
        if status not in {"UNKNOWN", "RATELIMIT"}:
            break
        if time.monotonic() >= deadline:
            raise RemoteMSAError(f"Timed out while submitting remote MSA query ({status})")
        attempt += 1
        time.sleep(min(30.0, 5.0 + attempt + random.random() * 4.0))

    if status in {"ERROR", "MAINTENANCE"} or not response.get("id"):
        raise RemoteMSAError(f"Remote MSA submission failed with status {status}: {response}")
    ticket = str(response["id"])
    while status in {"UNKNOWN", "RUNNING", "PENDING"}:
        if time.monotonic() >= deadline:
            raise RemoteMSAError(f"Timed out waiting for remote MSA ticket {ticket}")
        time.sleep(5.0 + random.random() * 5.0)
        response = _json_request(
            f"{host}/ticket/{urllib.parse.quote(ticket, safe='')}",
            timeout=per_request,
            user_agent=agent,
        )
        status = str(response.get("status", "ERROR")).upper()
    if status != "COMPLETE":
        raise RemoteMSAError(f"Remote MSA ticket {ticket} ended with status {status}")

    archive = _request(
        f"{host}/result/download/{urllib.parse.quote(ticket, safe='')}",
        timeout=max(per_request, 120.0),
        user_agent=agent,
    )
    a3m = _read_result_members(archive, include_env)
    entries = list(_iter_a3m_entries(a3m))
    query_match = "".join(char for char in entries[0][1] if not char.islower() and char != "-")
    if query_match.upper() != sequence:
        raise RemoteMSAError("Remote MSA query row does not match the submitted sequence")
    tmp_path = a3m_path.with_suffix(".a3m.tmp")
    tmp_path.write_text(a3m, encoding="utf-8")
    tmp_path.replace(a3m_path)
    metadata = {
        "provider": "colabfold",
        "host_url": host,
        "mode": mode,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "ticket": ticket,
        "depth": len(entries),
        "path": str(a3m_path),
        "cached": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def resolve_chain_msa_paths(
    chains: List[Tuple[str, str]],
    *,
    msa_mode: Optional[str] = None,
    msa_paths: Optional[Mapping[str, str] | str] = None,
    cache_dir: Optional[str | Path] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    mode = str(msa_mode or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_MODE", "off")).strip().lower()
    if mode in {"", "none", "disabled", "false", "0"}:
        mode = "off"
    if isinstance(msa_paths, str):
        msa_paths = json.loads(msa_paths)
    if msa_paths is None:
        raw = os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_PATHS", "").strip()
        msa_paths = json.loads(raw) if raw else {}
    paths = {str(key): str(value) for key, value in dict(msa_paths).items()}
    resolved: Dict[str, str] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for chain_id, sequence in chains:
        explicit = paths.get(chain_id)
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"MSA for chain {chain_id!r} does not exist: {path}")
            resolved[chain_id] = str(path)
            metadata[chain_id] = {"provider": "file", "path": str(path), "cached": True}
        elif mode == "remote":
            result = fetch_colabfold_msa(sequence, cache_dir=cache_dir)
            resolved[chain_id] = str(result["path"])
            metadata[chain_id] = result
        elif mode in {"file", "local"}:
            raise ValueError(f"MSA mode {mode!r} requires an MSA path for chain {chain_id!r}")
        elif mode not in {"off", "auto"}:
            raise ValueError(f"Unknown ESMFold2 MSA mode: {mode}")
    return resolved, metadata


def _main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache an unpaired ColabFold A3M")
    parser.add_argument("sequence")
    parser.add_argument("--cache-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = fetch_colabfold_msa(args.sequence, cache_dir=args.cache_dir, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

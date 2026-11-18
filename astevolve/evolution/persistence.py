

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

from .domain import (
    EvolutionContractError,
    GenerationCommit,
    GenerationManifest,
)


class GenerationPersistenceError(RuntimeError):
    pass


class GenerationPersistenceCorruption(GenerationPersistenceError):
    pass


class GenerationPublicationConflict(GenerationPersistenceError):
    pass


class GenerationManifestNotRegistered(GenerationPersistenceError):
    pass


@dataclass(frozen=True)
class PublishedGeneration:


    manifest: GenerationManifest
    commit: GenerationCommit

    def __post_init__(self) -> None:
        self.manifest.verify()
        self.commit.verify(self.manifest)


_Record = TypeVar("_Record", GenerationManifest, GenerationCommit)


def _canonical_record_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GenerationPersistenceError(
            "generation record is not finite JSON data"
        ) from exc


def _generation_key(generation_id: str) -> str:
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or generation_id != generation_id.strip()
    ):
        raise ValueError("generation_id must be a non-empty normalized string")
    return hashlib.sha256(generation_id.encode("utf-8")).hexdigest()


class FileGenerationLedger:


    def __init__(self, root: os.PathLike[str] | str) -> None:
        self._root = Path(root)
        self._manifest_dir = self._root / "manifests"
        self._commit_dir = self._root / "commits"
        try:
            self._manifest_dir.mkdir(parents=True, exist_ok=True)
            self._commit_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GenerationPersistenceError(
                f"cannot initialize generation ledger at {self._root}"
            ) from exc
        if not self._manifest_dir.is_dir() or not self._commit_dir.is_dir():
            raise GenerationPersistenceError(
                f"generation ledger paths are not directories: {self._root}"
            )

    @property
    def root(self) -> Path:
        return self._root

    def _manifest_path(self, generation_id: str) -> Path:
        return self._manifest_dir / f"{_generation_key(generation_id)}.json"

    def _commit_path(self, generation_id: str) -> Path:
        return self._commit_dir / f"{_generation_key(generation_id)}.json"

    @staticmethod
    def _load_record(
        path: Path,
        parser: Callable[[Mapping[str, Any]], _Record],
        *,
        label: str,
    ) -> Optional[_Record]:
        try:
            raw_bytes = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GenerationPersistenceError(f"cannot read {label}: {path}") from exc
        try:
            decoded = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise EvolutionContractError(f"{label} must contain a JSON object")
            record = parser(decoded)
            expected_bytes = _canonical_record_bytes(record.to_dict())
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            EvolutionContractError,
            TypeError,
            ValueError,
        ) as exc:
            raise GenerationPersistenceCorruption(
                f"visible {label} is invalid: {path}"
            ) from exc
        if raw_bytes != expected_bytes:
            raise GenerationPersistenceCorruption(
                f"visible {label} is not canonical: {path}"
            )
        return record

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd: Optional[int] = None
        try:
            directory_fd = os.open(path, flags)
            os.fsync(directory_fd)
        except OSError as exc:
            raise GenerationPersistenceError(
                f"cannot durably publish generation record in {path}"
            ) from exc
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    @classmethod
    def _install_once(cls, path: Path, payload: bytes) -> bool:


        descriptor: Optional[int] = None
        temporary_path: Optional[Path] = None
        try:
            descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix=f".{path.stem}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary_path = Path(raw_temporary_path)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                return False
            cls._fsync_directory(path.parent)
            return True
        except GenerationPersistenceError:
            raise
        except OSError as exc:
            raise GenerationPersistenceError(
                f"cannot atomically install generation record: {path}"
            ) from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:


                    pass

    def _load_manifest_path(self, path: Path) -> Optional[GenerationManifest]:
        return self._load_record(
            path,
            GenerationManifest.from_mapping,
            label="generation manifest",
        )

    def _load_commit_path(self, path: Path) -> Optional[GenerationCommit]:
        return self._load_record(
            path,
            GenerationCommit.from_mapping,
            label="generation commit",
        )

    def register_manifest(self, manifest: GenerationManifest) -> GenerationManifest:


        if not isinstance(manifest, GenerationManifest):
            raise TypeError("manifest must be GenerationManifest")
        manifest.verify()
        path = self._manifest_path(manifest.generation_id)
        existing = self._load_manifest_path(path)
        if existing is not None:
            if existing != manifest:
                raise GenerationPublicationConflict(
                    f"generation {manifest.generation_id!r} has another manifest"
                )
            return existing

        installed = self._install_once(
            path, _canonical_record_bytes(manifest.to_dict())
        )
        if installed:
            return manifest
        raced = self._load_manifest_path(path)
        if raced is None:
            raise GenerationPersistenceCorruption(
                f"manifest CAS lost without a visible record: {path}"
            )
        if raced != manifest:
            raise GenerationPublicationConflict(
                f"generation {manifest.generation_id!r} has another manifest"
            )
        return raced

    def load_manifest(self, generation_id: str) -> Optional[GenerationManifest]:
        manifest = self._load_manifest_path(self._manifest_path(generation_id))
        if manifest is not None and manifest.generation_id != generation_id:
            raise GenerationPersistenceCorruption(
                "generation manifest does not match its content-addressed path"
            )
        return manifest

    def publish_atomic(self, commit: GenerationCommit) -> GenerationCommit:


        if not isinstance(commit, GenerationCommit):
            raise TypeError("commit must be GenerationCommit")

        commit.to_dict()
        manifest = self.load_manifest(commit.generation_id)
        if manifest is None:
            raise GenerationManifestNotRegistered(
                f"generation {commit.generation_id!r} has no registered manifest"
            )
        commit.verify(manifest)

        path = self._commit_path(commit.generation_id)
        existing = self._load_commit_path(path)
        if existing is not None:
            existing.verify(manifest)
            if existing != commit:
                raise GenerationPublicationConflict(
                    f"generation {commit.generation_id!r} is already committed"
                )
            return existing

        installed = self._install_once(path, _canonical_record_bytes(commit.to_dict()))
        if installed:
            return commit
        raced = self._load_commit_path(path)
        if raced is None:
            raise GenerationPersistenceCorruption(
                f"commit CAS lost without a visible record: {path}"
            )
        raced.verify(manifest)
        if raced != commit:
            raise GenerationPublicationConflict(
                f"generation {commit.generation_id!r} is already committed"
            )
        return raced

    def get(self, generation_id: str) -> Optional[GenerationCommit]:
        commit = self._load_commit_path(self._commit_path(generation_id))
        if commit is None:
            return None
        manifest = self.load_manifest(generation_id)
        if manifest is None:
            raise GenerationPersistenceCorruption(
                f"generation {generation_id!r} has a commit without a manifest"
            )
        commit.verify(manifest)
        return commit

    def load_published(self, generation_id: str) -> Optional[PublishedGeneration]:
        commit = self.get(generation_id)
        if commit is None:
            return None
        manifest = self.load_manifest(generation_id)
        if manifest is None:
            raise GenerationPersistenceCorruption(
                f"generation {generation_id!r} has a commit without a manifest"
            )
        return PublishedGeneration(manifest=manifest, commit=commit)

    def published_generation_ids(self) -> Tuple[str, ...]:
        generation_ids = []
        for path in sorted(self._commit_dir.glob("*.json")):
            commit = self._load_commit_path(path)
            if commit is None:
                continue
            if path != self._commit_path(commit.generation_id):
                raise GenerationPersistenceCorruption(
                    f"generation commit is stored under the wrong path: {path}"
                )

            self.get(commit.generation_id)
            generation_ids.append(commit.generation_id)
        if len(generation_ids) != len(set(generation_ids)):
            raise GenerationPersistenceCorruption(
                "generation ledger contains duplicate commit identities"
            )
        return tuple(sorted(generation_ids))

    def iter_published(self) -> Tuple[PublishedGeneration, ...]:
        values = []
        for generation_id in self.published_generation_ids():
            published = self.load_published(generation_id)
            if published is None:
                raise GenerationPersistenceCorruption(
                    f"published generation disappeared during recovery: {generation_id}"
                )
            values.append(published)
        return tuple(values)


__all__ = [
    "FileGenerationLedger",
    "GenerationManifestNotRegistered",
    "GenerationPersistenceCorruption",
    "GenerationPersistenceError",
    "GenerationPublicationConflict",
    "PublishedGeneration",
]

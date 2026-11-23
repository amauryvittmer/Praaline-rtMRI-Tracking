from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .corpus import make_corpus_entry


HISTORY_DIRECTORY = ".praaline-history"
_ANNOTATION_KINDS = {
    "trs": ".trs",
    "textgrid": ".textgrid",
}


class AnnotationHistoryError(RuntimeError):
    """Raised when an annotation revision cannot be saved or restored."""


@dataclass(frozen=True)
class AnnotationRevision:
    revision_id: str
    created_at: str
    corrected: bool
    corrected_by: str
    files: tuple[str, ...]


def _audio_history_directory(audio_path: str | Path, corpus_root: str | Path) -> Path:
    audio = Path(audio_path).expanduser().resolve()
    root = Path(corpus_root).expanduser().resolve()
    try:
        relative_audio = audio.relative_to(root)
    except ValueError as error:
        raise AnnotationHistoryError(
            "Le fichier audio doit se trouver dans le dossier du corpus."
        ) from error
    return root / HISTORY_DIRECTORY / relative_audio.parent / relative_audio.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AnnotationHistoryError(
            f"Impossible de lire l’annotation {path.name} : {error}"
        ) from error
    return digest.hexdigest()


def _read_manifest(
    revisions_directory: Path, revision_id: str, audio_stem: str
) -> tuple[AnnotationRevision, dict[str, dict[str, str]]]:
    if (
        not revision_id
        or revision_id in {".", ".."}
        or Path(revision_id).name != revision_id
    ):
        raise AnnotationHistoryError("Identifiant de version invalide.")
    revision_directory = revisions_directory / revision_id
    manifest_path = revision_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnnotationHistoryError(
            f"Impossible de lire la version {revision_id} : {error}"
        ) from error

    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("created_at"), str)
        or not isinstance(manifest.get("corrected"), bool)
        or not isinstance(manifest.get("corrected_by"), str)
        or not isinstance(manifest.get("files"), list)
    ):
        raise AnnotationHistoryError(f"Historique invalide : {manifest_path}")
    try:
        datetime.fromisoformat(manifest["created_at"])
    except ValueError as error:
        raise AnnotationHistoryError(f"Date invalide dans {manifest_path}") from error

    files: dict[str, dict[str, str]] = {}
    names: list[str] = []
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise AnnotationHistoryError(f"Historique invalide : {manifest_path}")
        kind = record.get("kind")
        name = record.get("name")
        checksum = record.get("sha256")
        if (
            not isinstance(kind, str)
            or kind not in _ANNOTATION_KINDS
            or kind in files
            or not isinstance(name, str)
            or Path(name).name != name
            or Path(name).stem.casefold() != audio_stem.casefold()
            or Path(name).suffix.casefold() != _ANNOTATION_KINDS[kind]
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise AnnotationHistoryError(f"Historique invalide : {manifest_path}")
        files[kind] = {
            "name": name,
            "sha256": checksum,
        }
        names.append(name)

    return (
        AnnotationRevision(
            revision_id=revision_id,
            created_at=manifest["created_at"],
            corrected=manifest["corrected"],
            corrected_by=manifest["corrected_by"],
            files=tuple(names),
        ),
        files,
    )


def list_annotation_revisions(
    audio_path: str | Path, corpus_root: str | Path
) -> tuple[AnnotationRevision, ...]:
    audio = Path(audio_path).expanduser().resolve()
    revisions_directory = _audio_history_directory(audio, corpus_root) / "revisions"
    if not revisions_directory.exists():
        return ()
    try:
        revision_directories = sorted(
            (
                path
                for path in revisions_directory.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError as error:
        raise AnnotationHistoryError(
            f"Impossible de lire l’historique des annotations : {error}"
        ) from error

    revisions = []
    for directory in revision_directories:
        revision, _files = _read_manifest(
            revisions_directory, directory.name, audio.stem
        )
        revisions.append(revision)
    return tuple(
        sorted(
            revisions,
            key=lambda revision: (
                datetime.fromisoformat(revision.created_at),
                revision.revision_id,
            ),
            reverse=True,
        )
    )


def record_annotation_revision(
    audio_path: str | Path,
    corpus_root: str | Path,
    *,
    corrected: bool,
    corrected_by: str,
) -> tuple[AnnotationRevision, bool]:
    audio = Path(audio_path).expanduser().resolve()
    entry = make_corpus_entry(audio)
    paths = {
        "trs": entry.trs_path,
        "textgrid": entry.textgrid_path,
    }
    current_files: dict[str, dict[str, str | Path]] = {}
    for kind, path in paths.items():
        if path is not None:
            if not path.is_file():
                raise AnnotationHistoryError(
                    f"L’annotation n’existe plus : {path}"
                )
            if path.parent != audio.parent:
                raise AnnotationHistoryError(
                    f"L’annotation doit se trouver à côté de l’audio : {path}"
                )
            current_files[kind] = {
                "name": path.name,
                "sha256": _sha256(path),
                "path": path,
            }

    revisions_directory = _audio_history_directory(audio, corpus_root) / "revisions"
    previous_revisions = list_annotation_revisions(audio, corpus_root)
    if previous_revisions:
        latest, latest_files = _read_manifest(
            revisions_directory, previous_revisions[0].revision_id, audio.stem
        )
        previous_state = {
            kind: (record["name"], record["sha256"])
            for kind, record in latest_files.items()
        }
        current_state = {
            kind: (str(record["name"]), str(record["sha256"]))
            for kind, record in current_files.items()
        }
        if current_state == previous_state:
            return latest, False

    now = datetime.now().astimezone()
    revision_id = (
        now.strftime("%Y%m%dT%H%M%S%f%z")
        + f"-{uuid4().hex[:8]}"
    )
    created_at = now.isoformat(timespec="microseconds")
    temporary_directory: Path | None = None
    try:
        revisions_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(prefix=".revision-", dir=revisions_directory)
        )
        annotation_directory = temporary_directory / "annotations"
        annotation_directory.mkdir()
        file_records = []
        for kind, record in current_files.items():
            path = Path(str(record["path"]))
            shutil.copy2(path, annotation_directory / path.name)
            file_records.append(
                {
                    "kind": kind,
                    "name": str(record["name"]),
                    "sha256": str(record["sha256"]),
                }
            )
        manifest_path = temporary_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "created_at": created_at,
                    "corrected": corrected,
                    "corrected_by": corrected_by,
                    "files": file_records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        revision_directory = revisions_directory / revision_id
        os.replace(temporary_directory, revision_directory)
        temporary_directory = None
    except OSError as error:
        raise AnnotationHistoryError(
            f"Impossible d’archiver les annotations : {error}"
        ) from error
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    revision, _files = _read_manifest(revisions_directory, revision_id, audio.stem)
    return revision, True


def restore_annotation_revision(
    audio_path: str | Path,
    corpus_root: str | Path,
    revision_id: str,
    *,
    corrected: bool,
    corrected_by: str,
) -> AnnotationRevision:
    audio = Path(audio_path).expanduser().resolve()
    audio_history_directory = _audio_history_directory(audio, corpus_root)
    revisions_directory = audio_history_directory / "revisions"
    revision, revision_files = _read_manifest(
        revisions_directory, revision_id, audio.stem
    )
    revision_directory = revisions_directory / revision_id

    for kind, record in revision_files.items():
        archived_file = revision_directory / "annotations" / record["name"]
        if not archived_file.is_file() or _sha256(archived_file) != record["sha256"]:
            raise AnnotationHistoryError(
                f"L’archive de {record['name']} est absente ou endommagée."
            )

    record_annotation_revision(
        audio,
        corpus_root,
        corrected=corrected,
        corrected_by=corrected_by,
    )
    current_entry = make_corpus_entry(audio)
    current_paths = {
        "trs": audio.parent / current_entry.trs_path.name
        if current_entry.trs_path is not None
        else None,
        "textgrid": audio.parent / current_entry.textgrid_path.name
        if current_entry.textgrid_path is not None
        else None,
    }
    staged_files: dict[str, tuple[Path, Path]] = {}
    try:
        for kind, record in revision_files.items():
            destination = audio.parent / record["name"]
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".praaline-restore-",
                suffix=destination.suffix,
                dir=audio.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            staged_files[kind] = (destination, temporary_path)
            shutil.copy2(
                revision_directory / "annotations" / record["name"],
                temporary_path,
            )
        for destination, temporary_path in staged_files.values():
            os.replace(temporary_path, destination)
        for kind, current_path in current_paths.items():
            restored_file = revision_files.get(kind)
            if current_path is not None and (
                restored_file is None or current_path.name != restored_file["name"]
            ):
                current_path.unlink(missing_ok=True)
    except OSError as error:
        raise AnnotationHistoryError(
            f"Impossible de restaurer la version {revision_id} : {error}"
        ) from error
    finally:
        for _destination, temporary_path in staged_files.values():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return revision

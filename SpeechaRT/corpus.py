from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}
)
ANNOTATION_EXTENSIONS = (".textgrid", ".trs")
CORPUS_METADATA_FILENAME = ".praaline-corpus.json"
CORPUS_INDEX_FILENAME = "praaline-corpus.csv"
_CorpusIndexField = Literal[
    "version", "audio", "trs", "textgrid", "corrected", "corrected_by"
]
_CORPUS_INDEX_FIELDS: tuple[_CorpusIndexField, ...] = (
    "version",
    "audio",
    "trs",
    "textgrid",
    "corrected",
    "corrected_by",
)


@dataclass(frozen=True)
class CorpusEntry:
    audio_path: Path
    trs_path: Path | None = None
    textgrid_path: Path | None = None

    @property
    def annotation_path(self) -> Path | None:
        return self.textgrid_path or self.trs_path


@dataclass(frozen=True)
class CorrectionInfo:
    corrected: bool = False
    corrected_by: str = ""


@dataclass(frozen=True)
class CorpusCatalog:
    entries: tuple[CorpusEntry, ...]
    corrections: dict[str, CorrectionInfo]
    missing_audio_count: int = 0
    missing_annotation_count: int = 0


def discover_audio_files(folder: str | Path) -> tuple[Path, ...]:
    """Find supported audio files recursively in a corpus folder."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Le dossier de corpus est introuvable : {root}")
    try:
        return tuple(sorted(
            (
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
            ),
            key=lambda path: str(path).casefold(),
        ))
    except OSError as error:
        raise ValueError(f"Impossible de parcourir le dossier de corpus : {error}") from error


def make_corpus_entry(audio_path: str | Path) -> CorpusEntry:
    """Pair an audio file with an adjacent TextGrid or TRS sharing its stem."""
    audio = Path(audio_path).expanduser().resolve()
    try:
        siblings = tuple(audio.parent.iterdir())
    except OSError as error:
        raise ValueError(f"Impossible de lire le dossier de l’audio : {error}") from error

    candidates = {
        path.suffix.casefold(): path
        for path in siblings
        if path.is_file()
        and path.stem.casefold() == audio.stem.casefold()
        and path.suffix.casefold() in ANNOTATION_EXTENSIONS
    }
    return CorpusEntry(
        audio,
        trs_path=candidates.get(".trs"),
        textgrid_path=candidates.get(".textgrid"),
    )


def _catalog_path(root: Path, relative_path: str, field: str) -> Path:
    if not relative_path or "\x00" in relative_path:
        raise ValueError(f"Chemin {field} invalide dans {CORPUS_INDEX_FILENAME}.")
    candidate = Path(relative_path.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate.parts[0].endswith(":")
    ):
        raise ValueError(f"Chemin absolu interdit pour {field} dans le catalogue.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Chemin hors corpus pour {field} : {relative_path}") from error
    return resolved


def load_corpus_catalog(folder: str | Path) -> CorpusCatalog:
    """Load a portable corpus index using paths relative to its root folder."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Le dossier de corpus est introuvable : {root}")
    catalog_path = root / CORPUS_INDEX_FILENAME
    try:
        with catalog_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _CORPUS_INDEX_FIELDS:
                raise ValueError(f"En-tête CSV invalide : {catalog_path}")
            entries: list[CorpusEntry] = []
            corrections: dict[str, CorrectionInfo] = {}
            seen_audio: set[str] = set()
            missing_audio_count = 0
            missing_annotation_count = 0
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Ligne CSV invalide : {catalog_path}")
                if row["version"] != "1":
                    raise ValueError(f"Version CSV non prise en charge : {catalog_path}")
                audio = _catalog_path(root, row["audio"], "audio")
                if audio.suffix.casefold() not in AUDIO_EXTENSIONS:
                    raise ValueError(f"Format audio invalide dans le catalogue : {audio}")
                audio_key = str(audio)
                if audio_key in seen_audio:
                    raise ValueError(f"Audio dupliqué dans le catalogue : {audio}")
                seen_audio.add(audio_key)

                corrected_value = row["corrected"].casefold()
                if corrected_value not in {"true", "false"}:
                    raise ValueError(f"État de correction invalide pour {audio}.")
                corrected_by = row["corrected_by"]
                if corrected_value == "true" and not corrected_by.strip():
                    raise ValueError(f"Nom du correcteur manquant pour {audio}.")
                corrections[audio_key] = CorrectionInfo(
                    corrected_value == "true",
                    corrected_by if corrected_value == "true" else "",
                )
                if not audio.is_file():
                    missing_audio_count += 1
                    continue

                annotation_paths: dict[str, Path | None] = {}
                for field, suffix in (("trs", ".trs"), ("textgrid", ".textgrid")):
                    value = row[field]
                    annotation = None
                    if value:
                        annotation = _catalog_path(root, value, field)
                        if annotation.suffix.casefold() != suffix:
                            raise ValueError(
                                f"Format d’annotation invalide dans le catalogue : {annotation}"
                            )
                        if not annotation.is_file():
                            annotation = None
                            missing_annotation_count += 1
                    annotation_paths[field] = annotation

                entries.append(
                    CorpusEntry(
                        audio_path=audio,
                        trs_path=annotation_paths["trs"],
                        textgrid_path=annotation_paths["textgrid"],
                    )
                )
    except (OSError, csv.Error) as error:
        raise ValueError(f"Impossible de lire le catalogue {catalog_path} : {error}") from error
    return CorpusCatalog(
        entries=tuple(entries),
        corrections=corrections,
        missing_audio_count=missing_audio_count,
        missing_annotation_count=missing_annotation_count,
    )


def save_corpus_catalog(
    folder: str | Path,
    entries: Iterable[CorpusEntry],
    corrections: dict[str, CorrectionInfo],
) -> None:
    """Atomically save a portable corpus index with root-relative paths."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Le dossier de corpus est introuvable : {root}")
    catalog_path = root / CORPUS_INDEX_FILENAME
    temporary_path: str | None = None
    seen_audio: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=root,
            prefix=f"{CORPUS_INDEX_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            writer = csv.DictWriter(temporary, fieldnames=_CORPUS_INDEX_FIELDS)
            writer.writeheader()
            for entry in sorted(entries, key=lambda item: str(item.audio_path).casefold()):
                audio = entry.audio_path.resolve()
                audio_relative = audio.relative_to(root).as_posix()
                if str(audio) in seen_audio:
                    continue
                seen_audio.add(str(audio))
                correction = corrections.get(str(audio), CorrectionInfo())
                row: dict[_CorpusIndexField, str] = {
                    "version": "1",
                    "audio": audio_relative,
                    "trs": "",
                    "textgrid": "",
                    "corrected": str(correction.corrected).lower(),
                    "corrected_by": correction.corrected_by if correction.corrected else "",
                }
                if entry.trs_path is not None:
                    row["trs"] = entry.trs_path.resolve().relative_to(root).as_posix()
                if entry.textgrid_path is not None:
                    row["textgrid"] = (
                        entry.textgrid_path.resolve().relative_to(root).as_posix()
                    )
                writer.writerow(row)
        os.replace(temporary_path, catalog_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Impossible d’enregistrer le catalogue : {error}") from error
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


def load_corpus_corrections(folder: str | Path) -> dict[str, CorrectionInfo]:
    """Load correction metadata from corpus folders and their subfolders."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Le dossier de corpus est introuvable : {root}")

    corrections: dict[str, CorrectionInfo] = {}
    try:
        metadata_files = sorted(root.rglob(CORPUS_METADATA_FILENAME))
    except OSError as error:
        raise ValueError(f"Impossible de chercher les métadonnées du corpus : {error}") from error

    for metadata_file in metadata_files:
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Impossible de lire les métadonnées {metadata_file}: {error}"
            ) from error
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"Format de métadonnées invalide : {metadata_file}")
        files = data.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"Liste de fichiers invalide : {metadata_file}")

        metadata_root = metadata_file.parent
        for relative_path, record in files.items():
            if (
                not isinstance(relative_path, str)
                or not isinstance(record, dict)
                or not isinstance(record.get("corrected"), bool)
                or not isinstance(record.get("corrected_by"), str)
            ):
                raise ValueError(f"Entrée de correction invalide : {metadata_file}")
            audio_path = (metadata_root / relative_path).resolve()
            corrections[str(audio_path)] = CorrectionInfo(
                record["corrected"], record["corrected_by"]
            )
    return corrections


def save_corpus_corrections(
    folder: str | Path,
    corrections: dict[str, CorrectionInfo],
) -> None:
    """Atomically save correction metadata relative to a corpus folder."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Le dossier de métadonnées est introuvable : {root}")

    files: dict[str, dict[str, object]] = {}
    try:
        for audio_path, correction in corrections.items():
            relative_path = Path(audio_path).resolve().relative_to(root).as_posix()
            if correction.corrected or correction.corrected_by:
                files[relative_path] = {
                    "corrected": correction.corrected,
                    "corrected_by": correction.corrected_by,
                }
    except ValueError as error:
        raise ValueError(
            "Les fichiers du corpus doivent se trouver dans le dossier du corpus."
        ) from error

    metadata_file = root / CORPUS_METADATA_FILENAME
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=f"{CORPUS_METADATA_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(
                {"version": 1, "files": files},
                temporary,
                ensure_ascii=False,
                indent=2,
            )
            temporary.write("\n")
        os.replace(temporary_path, metadata_file)
    except OSError as error:
        raise ValueError(f"Impossible d’enregistrer les corrections : {error}") from error
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


class AlignmentError(RuntimeError):
    """Raised when an MFA alignment cannot be completed."""


def _resolve_executable(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
    else:
        found = shutil.which(executable)
        resolved = Path(found).resolve() if found else candidate
    if not resolved.is_file():
        raise AlignmentError(f"Exécutable MFA introuvable : {executable}")
    if not os.access(resolved, os.X_OK):
        raise AlignmentError(f"L’exécutable MFA n’est pas exécutable : {resolved}")
    return resolved


def _process_environment(executable: Path) -> dict[str, str]:
    environment = os.environ.copy()
    executable_directory = str(executable.parent)
    path_entries = environment.get("PATH", "").split(os.pathsep)
    if executable_directory not in path_entries:
        environment["PATH"] = os.pathsep.join(
            [executable_directory, *filter(None, path_entries)]
        )
    return environment


def _read_transcript(transcript: Path) -> str:
    if transcript.suffix.casefold() == ".trs":
        try:
            root = ET.parse(transcript).getroot()
        except (OSError, ET.ParseError) as error:
            raise AlignmentError(
                f"Impossible de lire la transcription TRS : {error}"
            ) from error
        turns = (
            element
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "Turn"
        )
        return "\n".join(
            text
            for text in (
                " ".join(" ".join(turn.itertext()).split()) for turn in turns
            )
            if text
        )

    try:
        return transcript.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise AlignmentError(f"Impossible de lire la transcription : {error}") from error


def _run(command: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise AlignmentError(f"Impossible de lancer MFA : {error}") from error


def _failure_details(result: subprocess.CompletedProcess[str]) -> str:
    sections = []
    for label, output in (("sortie standard", result.stdout), ("erreurs", result.stderr)):
        output = output.strip()
        if not output:
            continue
        if len(output) > 6000:
            output = f"… sortie tronquée …\n{output[-6000:]}"
        sections.append(f"{label} :\n{output}")
    return "\n\n".join(sections)


def _mfa_log_details(
    temporary_directory: Path, result: subprocess.CompletedProcess[str]
) -> str:
    temporary_root = temporary_directory.resolve()
    referenced_paths = re.findall(
        r"(/[^\s\"']+\.log)", f"{result.stdout}\n{result.stderr}"
    )
    log_files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in referenced_paths:
        path = Path(raw_path.rstrip(".,;:"))
        resolved = path.resolve()
        if (
            resolved.is_relative_to(temporary_root)
            and path.is_file()
            and resolved not in seen
        ):
            log_files.append(path)
            seen.add(resolved)

    for path in sorted(temporary_directory.rglob("*.log")):
        resolved = path.resolve()
        if path.name == "pretrained_aligner.log" and resolved not in seen:
            log_files.append(path)
            seen.add(resolved)

    sections = []
    for path in log_files[:3]:
        try:
            size = path.stat().st_size
            with path.open("rb") as log_file:
                log_file.seek(max(0, size - 4000))
                content = log_file.read().decode("utf-8", errors="replace").strip()
        except OSError as error:
            sections.append(
                f"Impossible de lire le journal MFA {path.name} : {error}"
            )
            continue
        if content:
            if size > 4000:
                content = f"… début du journal tronqué …\n{content}"
            sections.append(f"journal MFA ({path.name}) :\n{content}")
    return "\n\n".join(sections)


def _write_failure_log(output: Path, audio: Path, details: str) -> Path:
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output,
            prefix=f"{audio.stem}.mfa-error-",
            suffix=".log",
            delete=False,
        ) as log_file:
            log_file.write(details)
            log_file.write("\n")
            return Path(log_file.name)
    except OSError as error:
        raise AlignmentError(
            f"Impossible d’enregistrer le journal d’échec MFA dans {output} : {error}"
        ) from error


def align_audio(
    audio_file: str | Path,
    transcript_file: str | Path,
    mfa_executable: str,
    dictionary_model: str,
    acoustic_model: str,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Align one audio file and its text or Transcriber transcript."""
    audio = Path(audio_file).expanduser().resolve()
    transcript = Path(transcript_file).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()

    if not audio.is_file():
        raise AlignmentError(f"Fichier audio introuvable : {audio}")
    if not transcript.is_file():
        raise AlignmentError(f"Fichier de transcription introuvable : {transcript}")
    transcript_text = _read_transcript(transcript)
    if not transcript_text:
        raise AlignmentError("La transcription est vide.")
    if not dictionary_model.strip() or not acoustic_model.strip():
        raise AlignmentError("Indiquez le dictionnaire et le modèle acoustique MFA.")

    target = output / f"{audio.stem}.TextGrid"
    if target.exists() and not overwrite:
        raise AlignmentError(f"Le fichier existe déjà : {target}")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AlignmentError(f"Impossible de créer le dossier de sortie : {error}") from error

    executable = _resolve_executable(mfa_executable.strip())
    environment = _process_environment(executable)
    version_result = _run([str(executable), "version"], environment)
    if version_result.returncode:
        details = _failure_details(version_result)
        raise AlignmentError(
            "La vérification de MFA a échoué."
            + (f"\n{details}" if details else "")
        )

    version_output = (version_result.stdout + "\n" + version_result.stderr).strip()
    version_match = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)?\b", version_output)
    if not version_match:
        raise AlignmentError(f"Version MFA non reconnue : {version_output or '(vide)'}")
    align_command = "align_legacy" if int(version_match.group(1)) >= 4 else "align"

    try:
        with tempfile.TemporaryDirectory(prefix="praaline-mfa-") as temporary:
            temporary_root = Path(temporary)
            corpus = temporary_root / "corpus" / "speaker_1"
            results = temporary_root / "results"
            mfa_temporary = temporary_root / "mfa-temporary"
            for directory in (corpus, results, mfa_temporary):
                directory.mkdir(parents=True)

            corpus_audio = corpus / audio.name
            corpus_transcript = corpus / f"{audio.stem}.lab"
            shutil.copy2(audio, corpus_audio)
            corpus_transcript.write_text(transcript_text + "\n", encoding="utf-8")

            command = [
                str(executable),
                align_command,
                "--clean",
                "--temporary_directory",
                str(mfa_temporary),
                str(temporary_root / "corpus"),
                dictionary_model.strip(),
                acoustic_model.strip(),
                str(results),
            ]
            result = _run(command, environment)
            if result.returncode:
                details = _failure_details(result)
                log_details = _mfa_log_details(mfa_temporary, result)
                if not log_details:
                    log_details = "Aucun journal interne MFA .log n’a été trouvé."
                details = "\n\n".join(
                    part for part in (details, log_details) if part
                )
                try:
                    failure_log = _write_failure_log(output, audio, details)
                except AlignmentError as error:
                    raise AlignmentError(
                        f"MFA a échoué (code {result.returncode}).\n"
                        f"{details}\n\n{error}"
                    ) from error
                raise AlignmentError(
                    f"MFA a échoué (code {result.returncode})."
                    f"\nJournal détaillé enregistré : {failure_log}"
                )

            textgrids = [
                path
                for path in results.rglob("*")
                if path.is_file()
                and path.suffix.casefold() == ".textgrid"
                and path.stem.casefold() == audio.stem.casefold()
            ]
            if not textgrids:
                raise AlignmentError(
                    "MFA a terminé, mais n’a produit aucun TextGrid pour cet audio."
                )

            if target.exists() and not overwrite:
                raise AlignmentError(f"Le fichier existe déjà : {target}")
            shutil.copy2(textgrids[0], target)
    except AlignmentError:
        raise
    except OSError as error:
        raise AlignmentError(f"Erreur de fichier pendant l’alignement : {error}") from error

    return target

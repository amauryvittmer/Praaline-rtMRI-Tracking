from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf


class SilenceDetectionError(RuntimeError):
    """Raised when silence detection or its Transcriber export fails."""


@dataclass(frozen=True)
class SilenceRange:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SilenceDetectionResult:
    sample_rate: int
    frame_count: int
    effective_threshold_db: float
    silences: tuple[SilenceRange, ...]


def detect_silences(
    audio_path: str | Path,
    threshold_db: float = -35.0,
    minimum_silence_seconds: float = 0.3,
) -> SilenceDetectionResult:
    """Detect pauses using 10 ms RMS windows and the Praaline adaptive threshold."""
    audio = Path(audio_path).expanduser()
    if not math.isfinite(threshold_db) or not -100.0 <= threshold_db <= 0.0:
        raise SilenceDetectionError("Le seuil doit être compris entre -100 et 0 dB.")
    if not math.isfinite(minimum_silence_seconds) or minimum_silence_seconds <= 0:
        raise SilenceDetectionError("La durée minimale doit être supérieure à 0.")

    try:
        with sf.SoundFile(audio) as source:
            sample_rate = int(source.samplerate)
            frame_count = int(source.frames)
            channels = int(source.channels)
            if sample_rate <= 0 or frame_count <= 0 or channels <= 0:
                raise SilenceDetectionError("Le fichier audio ne contient pas d’échantillons lisibles.")

            window_frames = max(1, round(sample_rate * 0.01))
            chunk_frames = max(
                window_frames,
                (sample_rate // window_frames) * window_frames,
            )
            levels_db: list[float] = []
            frames_read = 0
            while True:
                samples = source.read(
                    frames=chunk_frames,
                    dtype="float64",
                    always_2d=True,
                )
                if samples.size == 0:
                    break
                frames_read += len(samples)
                if not np.isfinite(samples).all():
                    raise SilenceDetectionError(
                        "Le fichier audio contient des échantillons non finis."
                    )
                for window_start in range(0, len(samples), window_frames):
                    window = samples[window_start : window_start + window_frames]
                    rms = float(np.sqrt(np.mean(np.square(window))))
                    levels_db.append(20.0 * math.log10(max(rms, 1.0e-12)))
            if frames_read != frame_count:
                raise SilenceDetectionError("Le fichier audio est incomplet ou tronqué.")
    except SilenceDetectionError:
        raise
    except (OSError, RuntimeError) as error:
        raise SilenceDetectionError(f"Impossible de lire l’audio : {error}") from error

    if not levels_db:
        raise SilenceDetectionError("Le fichier audio ne contient pas d’échantillons lisibles.")

    levels = np.asarray(levels_db)
    percentile_indices = (
        math.floor(0.20 * (len(levels) - 1)),
        math.floor(0.85 * (len(levels) - 1)),
    )
    sorted_levels = np.partition(levels, percentile_indices)
    noise_floor_db = float(sorted_levels[percentile_indices[0]])
    speech_level_db = float(sorted_levels[percentile_indices[1]])
    level_range_db = speech_level_db - noise_floor_db
    effective_threshold_db = threshold_db
    if level_range_db >= 8.0:
        adaptive_margin_db = min(12.0, max(6.0, level_range_db * 0.35))
        effective_threshold_db = min(threshold_db, noise_floor_db + adaptive_margin_db)

    enter_silence_db = effective_threshold_db - 2.0
    leave_silence_db = effective_threshold_db + 2.0
    silence_mask: list[bool] = []
    in_silence = levels_db[0] <= effective_threshold_db
    for level_db in levels_db:
        if not in_silence and level_db <= enter_silence_db:
            in_silence = True
        elif in_silence and level_db > leave_silence_db:
            in_silence = False
        silence_mask.append(in_silence)

    maximum_gap_windows = max(1, round(0.1 * sample_rate / window_frames))
    index = 0
    while index < len(silence_mask):
        if silence_mask[index]:
            index += 1
            continue
        gap_start = index
        while index < len(silence_mask) and not silence_mask[index]:
            index += 1
        if (
            gap_start > 0
            and index < len(silence_mask)
            and index - gap_start <= maximum_gap_windows
        ):
            silence_mask[gap_start:index] = [True] * (index - gap_start)

    minimum_silence_frames = math.ceil(minimum_silence_seconds * sample_rate)
    silences: list[SilenceRange] = []
    index = 0
    while index < len(silence_mask):
        if not silence_mask[index]:
            index += 1
            continue
        silence_start_window = index
        while index < len(silence_mask) and silence_mask[index]:
            index += 1
        start_frame = silence_start_window * window_frames
        end_frame = min(frame_count, index * window_frames)
        if end_frame - start_frame >= minimum_silence_frames:
            silences.append(SilenceRange(start_frame, end_frame))

    return SilenceDetectionResult(
        sample_rate,
        frame_count,
        effective_threshold_db,
        tuple(silences),
    )


def write_silence_trs(
    audio_path: str | Path,
    output_path: str | Path,
    result: SilenceDetectionResult,
    speaker_name: str,
) -> Path:
    """Write a fresh Transcriber file with detected pauses labeled ``#``."""
    audio = Path(audio_path).expanduser()
    destination = Path(output_path).expanduser()
    output = destination.parent.resolve() / destination.name
    name = speaker_name.strip()
    if not name:
        raise SilenceDetectionError("Indiquez le nom du locuteur.")

    if result.sample_rate <= 0 or result.frame_count <= 0:
        raise SilenceDetectionError("La durée du fichier audio n’est pas valide.")
    duration = result.frame_count / result.sample_rate
    previous_end = 0
    for silence in result.silences:
        if (
            silence.start_frame < previous_end
            or silence.start_frame < 0
            or silence.end_frame <= silence.start_frame
            or silence.end_frame > result.frame_count
        ):
            raise SilenceDetectionError("Les limites d’une pause sont invalides.")
        previous_end = silence.end_frame
    root = ET.Element(
        "Trans",
        {
            "scribe": "Praaline",
            "audio_filename": audio.stem,
            "version": "2",
            "version_date": date.today().isoformat(),
        },
    )
    topics = ET.SubElement(root, "Topics")
    ET.SubElement(topics, "Topic", {"id": "topic0", "desc": ""})
    speakers = ET.SubElement(root, "Speakers")
    ET.SubElement(
        speakers,
        "Speaker",
        {
            "id": "spk1",
            "name": name,
            "check": "no",
            "type": "unknown",
            "dialect": "",
            "scope": "local",
            "accent": "",
        },
    )
    episode = ET.SubElement(root, "Episode")
    section = ET.SubElement(
        episode,
        "Section",
        {
            "type": "report",
            "startTime": "0",
            "endTime": _format_time(duration),
            "topic": "topic0",
        },
    )
    turn = ET.SubElement(
        section,
        "Turn",
        {
            "speaker": "spk1",
            "startTime": "0",
            "endTime": _format_time(duration),
            "mode": "planned",
            "fidelity": "high",
            "channel": "studio",
        },
    )

    boundaries = sorted(
        {0, result.frame_count}
        | {
            boundary
            for silence in result.silences
            for boundary in (silence.start_frame, silence.end_frame)
        }
    )
    silence_ranges = {
        (silence.start_frame, silence.end_frame) for silence in result.silences
    }
    sync = ET.SubElement(turn, "Sync", {"time": "0"})
    for start_frame, end_frame in zip(boundaries, boundaries[1:]):
        is_silence = (start_frame, end_frame) in silence_ranges
        sync.tail = "\n" + ("#" if is_silence else "")
        sync = ET.SubElement(
            turn,
            "Sync",
            {"time": _format_time(end_frame / result.sample_rate)},
        )

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE Trans SYSTEM "trans-13.dtd">\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(content)
        os.replace(temporary_path, output)
    except OSError as error:
        raise SilenceDetectionError(
            f"Impossible d’enregistrer le fichier TRS : {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass
    return output


def _format_time(value: float) -> str:
    return format(value, ".15g")

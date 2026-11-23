from __future__ import annotations

import base64
import binascii
import codecs
import colorsys
import hashlib
import math
import os
import queue
import re
import struct
import tempfile
import threading
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import numpy as np
import soundfile as sf

from .video_player import VideoPlayer


TIER_COLORS = (
    ("#315dcc", "#edf2ff"),
    ("#b54708", "#fff3e8"),
    ("#027a48", "#ecfdf3"),
    ("#7a3e9d", "#f7edfc"),
    ("#0e7490", "#ecfeff"),
    ("#b42318", "#fef3f2"),
    ("#6941c6", "#f4f3ff"),
    ("#8a5b00", "#fffaeb"),
)
_MAX_UNDO_STEPS = 20


SPECTROGRAM_COLOUR_MAPS = (
    "Default",
    "Sunset",
    "White on Black",
    "Black on White",
    "Red on Blue",
    "Yellow on Black",
    "Blue on Black",
    "Fruit Salad",
    "Banded",
    "Highlight",
    "Printer",
    "High Gain",
)
SPECTROGRAM_COLOUR_SCALES = ("Linear", "Meter", "dBV^2", "dBV", "Phase")
SPECTROGRAM_WINDOW_SIZES = tuple(32 << power for power in range(11))
SPECTROGRAM_WINDOW_OVERLAPS = ("None", "25 %", "50 %", "75 %", "87.5 %", "93.75 %")
SPECTROGRAM_BIN_DISPLAYS = ("All Bins", "Peak Bins", "Frequencies")
SPECTROGRAM_FREQUENCY_SCALES = ("Linear", "Log")


@dataclass(frozen=True)
class SpectrogramSettings:
    colour_map: str = "Black on White"
    colour_scale: str = "dBV^2"
    window_size: int = 256
    window_hop_level: int = 5
    bin_display: str = "All Bins"
    frequency_scale: str = "Linear"
    normalize_columns: bool = False
    normalize_visible_area: bool = True
    grayscale: bool = True
    gain_db: int = 0
    threshold_db: int = -80
    colour_rotation: int = 0


SpectrogramKey = tuple[
    int, int, int, int, SpectrogramSettings, tuple[int, int] | None
]


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str
    source_turn_index: int | None = None
    source_speaker_id: str | None = None


@dataclass(frozen=True)
class TextGridTier:
    name: str
    intervals: tuple[Interval, ...]
    source_index: int | None = None
    source_start: float | None = None
    source_end: float | None = None
    source_speaker_id: str | None = None


@dataclass(frozen=True)
class _BoundaryDrag:
    tier_index: int
    interval_index: int
    side: Literal["internal", "start", "end"]


@dataclass(frozen=True)
class _AnnotationUndo:
    path: Path
    content: bytes
    resulting_digest: bytes


def parse_textgrid(path: str | Path) -> tuple[TextGridTier, ...]:
    """Read Praat's long TextGrid format, including its interval tiers."""
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Impossible de lire le TextGrid : {error}") from error

    return parse_textgrid_content(content)


def parse_textgrid_content(content: str) -> tuple[TextGridTier, ...]:
    """Parse Praat's long TextGrid format from already loaded text."""
    item_headers = list(re.finditer(r"(?m)^\s*item\s*\[\d+\]\s*:\s*$", content))
    tiers: list[TextGridTier] = []
    for item_index, item_header in enumerate(item_headers):
        item_end = (
            item_headers[item_index + 1].start()
            if item_index + 1 < len(item_headers)
            else len(content)
        )
        item = content[item_header.end():item_end]
        tier_class = _field(item, "class")
        name = _quoted_field(item, "name")
        if tier_class != "IntervalTier" or name is None:
            continue

        interval_headers = list(
            re.finditer(r"(?m)^\s*intervals\s*\[\d+\]\s*:\s*$", item)
        )
        intervals: list[Interval] = []
        for interval_index, interval_header in enumerate(interval_headers):
            interval_end = (
                interval_headers[interval_index + 1].start()
                if interval_index + 1 < len(interval_headers)
                else len(item)
            )
            block = item[interval_header.end():interval_end]
            start = _numeric_field(block, "xmin")
            end = _numeric_field(block, "xmax")
            text = _quoted_field(block, "text")
            if start is None or end is None or text is None:
                raise ValueError(f"Intervalle illisible dans le tier « {name} ».")
            if end < start:
                raise ValueError(f"Intervalle invalide dans le tier « {name} ».")
            intervals.append(Interval(start, end, text))
        tiers.append(
            TextGridTier(
                name,
                tuple(intervals),
                item_index,
                _numeric_field(item, "xmin"),
                _numeric_field(item, "xmax"),
            )
        )

    if not tiers:
        raise ValueError(
            "Aucun tier d’intervalles Praat lisible n’a été trouvé dans ce TextGrid."
        )
    return tuple(tiers)


def parse_trs(path: str | Path) -> tuple[TextGridTier, ...]:
    """Read Transcriber turns and synchronization points into speaker tiers."""
    try:
        root = ET.parse(Path(path).expanduser()).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"Impossible de lire le fichier TRS : {error}") from error

    speakers: dict[str, str] = {}
    for element in root.iter():
        if _xml_local_name(element.tag) == "Speaker":
            speaker_id = element.get("id")
            if speaker_id:
                speakers[speaker_id] = (
                    element.get("name", "").strip() or speaker_id
                )

    intervals_by_speaker: dict[str, list[Interval]] = {}
    turn_elements = [
        element for element in root.iter() if _xml_local_name(element.tag) == "Turn"
    ]
    for turn_index, turn in enumerate(turn_elements):
        start = _xml_time(turn.get("startTime"))
        end = _xml_time(turn.get("endTime"))
        if (
            start is None
            or end is None
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end < start
        ):
            raise ValueError("Horodatage de tour TRS manquant ou invalide.")

        speaker_ids = (turn.get("speaker") or "").split()
        speaker_id = speaker_ids[0] if speaker_ids else ""
        current_time = start
        text_by_speaker: dict[str, list[str]] = {}

        def flush(until: float) -> None:
            for segment_speaker, text_parts in text_by_speaker.items():
                text = " ".join(" ".join(text_parts).split())
                if text and until >= current_time:
                    intervals_by_speaker.setdefault(segment_speaker, []).append(
                        Interval(
                            current_time,
                            until,
                            text,
                            source_turn_index=turn_index,
                            source_speaker_id=segment_speaker,
                        )
                    )
            text_by_speaker.clear()

        for content, value in _trs_turn_contents(turn):
            if content == "sync":
                sync_time = _xml_time(value)
                if (
                    sync_time is None
                    or not math.isfinite(sync_time)
                    or not current_time <= sync_time <= end
                ):
                    raise ValueError("Point Sync horodaté invalide dans le fichier TRS.")
                flush(sync_time)
                current_time = sync_time
            elif content == "speaker":
                try:
                    speaker_number = int(value) - 1
                except ValueError as error:
                    raise ValueError("Balise Who invalide dans le fichier TRS.") from error
                if 0 <= speaker_number < len(speaker_ids):
                    speaker_id = speaker_ids[speaker_number]
            elif content == "text":
                text_by_speaker.setdefault(speaker_id, []).append(value)
            elif content == "event":
                text_by_speaker.setdefault(speaker_id, []).append(f"[{value}]")
        flush(end)

    speaker_ids = list(speakers)
    speaker_ids.extend(
        speaker_id
        for speaker_id in intervals_by_speaker
        if speaker_id not in speakers
    )
    tiers = tuple(
        TextGridTier(
            speakers.get(speaker_id, speaker_id or "Transcription"),
            tuple(
                sorted(
                    intervals_by_speaker.get(speaker_id, ()),
                    key=lambda interval: (interval.start, interval.end),
                )
            ),
            source_speaker_id=speaker_id,
        )
        for speaker_id in speaker_ids
    )
    if not tiers:
        raise ValueError("Aucun locuteur lisible n’a été trouvé dans ce fichier TRS.")
    return tiers


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_time(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _trs_turn_contents(turn: ET.Element) -> list[tuple[str, str]]:
    contents: list[tuple[str, str]] = []

    def visit(element: ET.Element) -> None:
        if element.text:
            contents.append(("text", element.text))
        for child in element:
            name = _xml_local_name(child.tag)
            if name == "Sync":
                contents.append(("sync", child.get("time", "")))
            elif name == "Who":
                contents.append(("speaker", child.get("nb", "")))
            elif name == "Event":
                description = child.get("desc", "").strip()
                if description:
                    contents.append(("event", description))
            else:
                visit(child)
            if child.tail:
                contents.append(("text", child.tail))

    visit(turn)
    return contents


def replace_tier_interval_range(
    tier: TextGridTier, start: float, end: float, text: str
) -> TextGridTier:
    """Split existing intervals and set the selected time range to new text."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("La sélection doit avoir une durée positive.")
    if not tier.intervals:
        if tier.source_start is None or tier.source_end is None:
            raise ValueError("Les limites temporelles de ce tier sont indisponibles.")
        start = max(start, tier.source_start)
        end = min(end, tier.source_end)
        if end <= start:
            raise ValueError("La sélection est en dehors des limites de ce tier.")
        return replace(tier, intervals=(Interval(start, end, text),))

    start = max(start, min(interval.start for interval in tier.intervals))
    end = min(end, max(interval.end for interval in tier.intervals))
    if end <= start:
        raise ValueError("La sélection est en dehors des intervalles de ce niveau.")

    result: list[Interval] = []
    covered_until = start
    for interval in tier.intervals:
        overlap_start = max(start, interval.start)
        overlap_end = min(end, interval.end)
        if overlap_end <= overlap_start:
            result.append(interval)
            continue
        if interval.start < start:
            result.append(replace(interval, end=start))
        if covered_until < overlap_start:
            result.append(Interval(covered_until, overlap_start, text))
        result.append(Interval(overlap_start, overlap_end, text))
        covered_until = max(covered_until, overlap_end)
        if interval.end > end:
            result.append(replace(interval, start=end))
    if covered_until < end:
        result.append(Interval(covered_until, end, text))
    return replace(
        tier,
        intervals=tuple(sorted(result, key=lambda interval: (interval.start, interval.end))),
    )


def clear_tier_interval(tier: TextGridTier, interval_index: int) -> TextGridTier:
    if not 0 <= interval_index < len(tier.intervals):
        raise ValueError("L’annotation sélectionnée n’existe plus.")
    intervals = list(tier.intervals)
    intervals[interval_index] = replace(intervals[interval_index], text="")
    return replace(tier, intervals=tuple(intervals))


def set_tier_interval_text(
    tier: TextGridTier, interval_index: int, text: str
) -> TextGridTier:
    if not 0 <= interval_index < len(tier.intervals):
        raise ValueError("L’annotation sélectionnée n’existe plus.")
    intervals = list(tier.intervals)
    intervals[interval_index] = replace(intervals[interval_index], text=text)
    return replace(tier, intervals=tuple(intervals))


def merge_tier_intervals(tier: TextGridTier, boundary_index: int) -> TextGridTier:
    """Remove one internal TextGrid boundary and merge its adjacent intervals."""
    if not 1 <= boundary_index < len(tier.intervals):
        raise ValueError("La barre sélectionnée n’existe pas.")
    left = tier.intervals[boundary_index - 1]
    right = tier.intervals[boundary_index]
    if not math.isclose(left.end, right.start, abs_tol=1e-6):
        raise ValueError("Les intervalles de part et d’autre ne sont pas contigus.")
    merged = replace(left, end=right.end, text=f"{left.text} {right.text}")
    intervals = (
        tier.intervals[:boundary_index - 1]
        + (merged,)
        + tier.intervals[boundary_index + 1:]
    )
    return replace(tier, intervals=intervals)


def move_tier_boundary(
    tier: TextGridTier, boundary_index: int, time: float
) -> TextGridTier:
    """Move one shared TextGrid boundary without changing its interval labels."""
    if not math.isfinite(time):
        raise ValueError("L’instant de la nouvelle barre est invalide.")
    if not 1 <= boundary_index < len(tier.intervals):
        raise ValueError("La barre sélectionnée n’existe pas.")
    left = tier.intervals[boundary_index - 1]
    right = tier.intervals[boundary_index]
    if not math.isclose(left.end, right.start, abs_tol=1e-6):
        raise ValueError("Les intervalles de part et d’autre ne sont pas contigus.")
    if not left.start < time < right.end:
        raise ValueError("La barre doit rester à l’intérieur des intervalles voisins.")
    intervals = list(tier.intervals)
    intervals[boundary_index - 1] = replace(left, end=time)
    intervals[boundary_index] = replace(right, start=time)
    return replace(tier, intervals=tuple(intervals))


def split_tier_interval(tier: TextGridTier, time: float) -> TextGridTier:
    """Insert a boundary and leave the original text in the first interval."""
    if not math.isfinite(time):
        raise ValueError("L’instant de la nouvelle barre est invalide.")
    for index, interval in enumerate(tier.intervals):
        if interval.start < time < interval.end:
            intervals = (
                tier.intervals[:index]
                + (
                    replace(interval, end=time),
                    Interval(time, interval.end, ""),
                )
                + tier.intervals[index + 1:]
            )
            return replace(tier, intervals=intervals)
    raise ValueError("Cliquez à l’intérieur d’un intervalle pour ajouter une barre.")


def save_textgrid(
    path: str | Path, source: str, tiers: tuple[TextGridTier, ...]
) -> str:
    newline = "\r\n" if "\r\n" in source else "\n"
    item_headers = list(re.finditer(r"(?m)^\s*item\s*\[\d+\]\s*:\s*$", source))
    updated = source
    for tier in sorted(
        (tier for tier in tiers if tier.source_index is not None),
        key=lambda item: item.source_index or 0,
        reverse=True,
    ):
        item_index = tier.source_index
        if item_index is None or item_index >= len(item_headers):
            raise ValueError(f"Le tier « {tier.name} » n’existe plus dans le TextGrid.")
        headers = list(re.finditer(r"(?m)^\s*item\s*\[\d+\]\s*:\s*$", updated))
        if item_index >= len(headers):
            raise ValueError(f"Le tier « {tier.name} » n’existe plus dans le TextGrid.")
        item_start = headers[item_index].end()
        item_end = headers[item_index + 1].start() if item_index + 1 < len(headers) else len(updated)
        item = updated[item_start:item_end]
        if _field(item, "class") != "IntervalTier" or _quoted_field(item, "name") != tier.name:
            raise ValueError(f"Le tier « {tier.name} » a changé depuis son ouverture.")
        intervals_header = re.search(
            r"(?m)^\s*intervals:\s*size\s*=\s*\d+\s*$", item
        )
        if intervals_header is None:
            raise ValueError(f"Les intervalles du tier « {tier.name} » sont illisibles.")

        interval_lines = [f"        intervals: size = {len(tier.intervals)}"]
        for index, interval in enumerate(tier.intervals, start=1):
            escaped_text = interval.text.replace('"', '""')
            interval_lines.extend(
                (
                    f"        intervals [{index}]:",
                    f"            xmin = {_praat_number(interval.start)}",
                    f"            xmax = {_praat_number(interval.end)}",
                    f'            text = "{escaped_text}"',
                )
            )
        replacement = newline.join(interval_lines) + newline
        replace_start = item_start + intervals_header.start()
        replace_end = item_end
        updated = updated[:replace_start] + replacement + updated[replace_end:]

    new_tiers = tuple(tier for tier in tiers if tier.source_index is None)
    if new_tiers:
        size_matches = list(
            re.finditer(r"(?m)^([ \t]*size[ \t]*=[ \t]*)(\d+)([ \t]*)$", updated)
        )
        if len(size_matches) != 1:
            raise ValueError("Le nombre de niveaux du TextGrid est illisible.")
        size_match = size_matches[0]
        tier_count = int(size_match.group(2))
        updated = (
            updated[:size_match.start()]
            + f"{size_match.group(1)}{tier_count + len(new_tiers)}{size_match.group(3)}"
            + updated[size_match.end():]
        )

        appended_items: list[str] = []
        for item_index, tier in enumerate(new_tiers, start=tier_count + 1):
            starts = [interval.start for interval in tier.intervals]
            ends = [interval.end for interval in tier.intervals]
            tier_start = (
                tier.source_start
                if tier.source_start is not None
                else min(starts, default=None)
            )
            tier_end = (
                tier.source_end
                if tier.source_end is not None
                else max(ends, default=None)
            )
            if (
                tier_start is None
                or tier_end is None
                or tier_end < tier_start
                or "\n" in tier.name
                or "\r" in tier.name
            ):
                raise ValueError(f"Les limites du tier « {tier.name} » sont invalides.")

            escaped_name = tier.name.replace('"', '""')
            item_lines = [
                f"    item [{item_index}]:",
                '        class = "IntervalTier"',
                f'        name = "{escaped_name}"',
                f"        xmin = {_praat_number(tier_start)}",
                f"        xmax = {_praat_number(tier_end)}",
                f"        intervals: size = {len(tier.intervals)}",
            ]
            for interval_index, interval in enumerate(tier.intervals, start=1):
                if (
                    interval.start < tier_start
                    or interval.end > tier_end
                    or interval.end < interval.start
                ):
                    raise ValueError(
                        f"Un intervalle du tier « {tier.name} » est hors limites."
                    )
                escaped_text = interval.text.replace('"', '""')
                item_lines.extend(
                    (
                        f"        intervals [{interval_index}]:",
                        f"            xmin = {_praat_number(interval.start)}",
                        f"            xmax = {_praat_number(interval.end)}",
                        f'            text = "{escaped_text}"',
                    )
                )
            appended_items.append(newline.join(item_lines))
        separator = "" if updated.endswith(("\n", "\r")) else newline
        updated += separator + newline.join(appended_items) + newline

    _atomic_write_text(path, updated)
    return updated


def clear_trs_interval(path: str | Path, interval: Interval) -> None:
    if interval.source_turn_index is None or interval.source_speaker_id is None:
        raise ValueError("Cette annotation TRS ne peut pas être modifiée.")
    tree = _load_trs_tree(path)
    root = tree.getroot()
    if root is None:
        raise ValueError("Le fichier TRS ne contient pas de document XML.")
    turns = [element for element in root.iter() if _xml_local_name(element.tag) == "Turn"]
    if interval.source_turn_index >= len(turns):
        raise ValueError("Le tour TRS sélectionné n’existe plus.")
    turn = turns[interval.source_turn_index]
    speaker_ids = (turn.get("speaker") or "").split()
    current_speaker = speaker_ids[0] if speaker_ids else ""
    start_time = _xml_time(turn.get("startTime"))
    end_time = _xml_time(turn.get("endTime"))
    if start_time is None or end_time is None:
        raise ValueError("Horodatage de tour TRS manquant ou invalide.")
    segment_start = start_time
    text_slots: list[tuple[ET.Element, str, str]] = []
    event_slots: list[tuple[ET.Element, ET.Element, str]] = []
    changed = False

    def flush(until: float) -> None:
        nonlocal changed
        if math.isclose(segment_start, interval.start, abs_tol=1e-6) and math.isclose(
            until, interval.end, abs_tol=1e-6
        ):
            for element, attribute, speaker_id in text_slots:
                if speaker_id == interval.source_speaker_id:
                    setattr(element, attribute, "")
                    changed = True
            for parent, event, speaker_id in event_slots:
                if speaker_id == interval.source_speaker_id:
                    parent.remove(event)
                    changed = True
        text_slots.clear()
        event_slots.clear()

    def visit(element: ET.Element) -> None:
        nonlocal current_speaker, segment_start
        if element.text:
            text_slots.append((element, "text", current_speaker))
        for child in list(element):
            name = _xml_local_name(child.tag)
            if name == "Sync":
                sync_time = _xml_time(child.get("time"))
                if sync_time is None:
                    raise ValueError("Point Sync horodaté invalide dans le fichier TRS.")
                flush(sync_time)
                segment_start = sync_time
            elif name == "Who":
                try:
                    speaker_number = int(child.get("nb", "")) - 1
                except ValueError as error:
                    raise ValueError("Balise Who invalide dans le fichier TRS.") from error
                if 0 <= speaker_number < len(speaker_ids):
                    current_speaker = speaker_ids[speaker_number]
            elif name == "Event":
                event_slots.append((element, child, current_speaker))
            else:
                visit(child)
            if child.tail:
                text_slots.append((child, "tail", current_speaker))

    visit(turn)
    flush(end_time)
    if not changed:
        raise ValueError("Le segment TRS sélectionné a changé depuis son ouverture.")
    _save_trs_tree(path, tree)


def set_trs_interval_text(
    path: str | Path, interval: Interval, text: str
) -> None:
    if interval.source_turn_index is None or interval.source_speaker_id is None:
        raise ValueError("Cette annotation TRS ne peut pas être modifiée.")
    tree = _load_trs_tree(path)
    root = tree.getroot()
    if root is None:
        raise ValueError("Le fichier TRS ne contient pas de document XML.")
    turns = [element for element in root.iter() if _xml_local_name(element.tag) == "Turn"]
    if interval.source_turn_index >= len(turns):
        raise ValueError("Le tour TRS sélectionné n’existe plus.")
    turn = turns[interval.source_turn_index]
    speaker_ids = (turn.get("speaker") or "").split()
    current_speaker = speaker_ids[0] if speaker_ids else ""
    start_time = _xml_time(turn.get("startTime"))
    end_time = _xml_time(turn.get("endTime"))
    if start_time is None or end_time is None:
        raise ValueError("Horodatage de tour TRS manquant ou invalide.")
    segment_start = start_time
    text_slots: list[tuple[ET.Element, str, str]] = []
    event_slots: list[tuple[ET.Element, ET.Element, str]] = []
    changed = False

    def flush(until: float) -> None:
        nonlocal changed
        if math.isclose(segment_start, interval.start, abs_tol=1e-6) and math.isclose(
            until, interval.end, abs_tol=1e-6
        ):
            speaker_slots = [
                (element, attribute)
                for element, attribute, speaker_id in text_slots
                if speaker_id == interval.source_speaker_id
            ]
            if not speaker_slots:
                raise ValueError("Le segment TRS sélectionné ne contient plus de texte.")
            for index, (element, attribute) in enumerate(speaker_slots):
                setattr(element, attribute, text if index == 0 else "")
            for parent, event, speaker_id in event_slots:
                if speaker_id == interval.source_speaker_id:
                    parent.remove(event)
            changed = True
        text_slots.clear()
        event_slots.clear()

    def visit(element: ET.Element) -> None:
        nonlocal current_speaker, segment_start
        text_slots.append((element, "text", current_speaker))
        for child in list(element):
            name = _xml_local_name(child.tag)
            if name == "Sync":
                sync_time = _xml_time(child.get("time"))
                if sync_time is None:
                    raise ValueError("Point Sync horodaté invalide dans le fichier TRS.")
                flush(sync_time)
                segment_start = sync_time
            elif name == "Who":
                try:
                    speaker_number = int(child.get("nb", "")) - 1
                except ValueError as error:
                    raise ValueError("Balise Who invalide dans le fichier TRS.") from error
                if 0 <= speaker_number < len(speaker_ids):
                    current_speaker = speaker_ids[speaker_number]
            elif name == "Event":
                event_slots.append((element, child, current_speaker))
            else:
                visit(child)
            text_slots.append((child, "tail", current_speaker))

    visit(turn)
    flush(end_time)
    if not changed:
        raise ValueError("Le segment TRS sélectionné a changé depuis son ouverture.")
    _save_trs_tree(path, tree)


def move_trs_boundary(
    path: str | Path,
    interval: Interval,
    side: Literal["start", "end"],
    new_time: float,
) -> None:
    if interval.source_turn_index is None or interval.source_speaker_id is None:
        raise ValueError("Cette annotation TRS ne peut pas être déplacée.")
    if not math.isfinite(new_time):
        raise ValueError("L’instant de la nouvelle barre est invalide.")
    if side not in {"start", "end"}:
        raise ValueError("Le côté de la barre TRS à déplacer est invalide.")
    if (
        not math.isfinite(interval.start)
        or not math.isfinite(interval.end)
        or interval.end <= interval.start
    ):
        raise ValueError("L’intervalle TRS sélectionné est invalide.")

    tree = _load_trs_tree(path)
    root = tree.getroot()
    if root is None:
        raise ValueError("Le fichier TRS ne contient pas de document XML.")
    turns = [element for element in root.iter() if _xml_local_name(element.tag) == "Turn"]
    if not 0 <= interval.source_turn_index < len(turns):
        raise ValueError("Le tour TRS sélectionné n’existe plus.")
    turn = turns[interval.source_turn_index]
    speakers = (turn.get("speaker") or "").split()
    if interval.source_speaker_id and interval.source_speaker_id not in speakers:
        raise ValueError("Le locuteur de l’annotation TRS a changé.")
    turn_start = _xml_time(turn.get("startTime"))
    turn_end = _xml_time(turn.get("endTime"))
    if (
        turn_start is None
        or turn_end is None
        or not math.isfinite(turn_start)
        or not math.isfinite(turn_end)
        or turn_end < turn_start
    ):
        raise ValueError("Les limites du tour TRS sont invalides.")

    syncs = [
        element for element in turn.iter() if _xml_local_name(element.tag) == "Sync"
    ]
    sync_times: list[float] = []
    for sync in syncs:
        sync_time = _xml_time(sync.get("time"))
        if (
            sync_time is None
            or not math.isfinite(sync_time)
            or not turn_start <= sync_time <= turn_end
        ):
            raise ValueError("Un point Sync du tour TRS est invalide.")
        sync_times.append(sync_time)
    if any(left > right for left, right in zip(sync_times, sync_times[1:])):
        raise ValueError("Les points Sync du tour TRS ne sont pas ordonnés.")
    boundary_time = interval.start if side == "start" else interval.end
    if math.isclose(new_time, boundary_time, abs_tol=1e-6):
        return
    matches = [
        index
        for index, sync_time in enumerate(sync_times)
        if math.isclose(sync_time, boundary_time, abs_tol=1e-6)
    ]
    if len(matches) > 1:
        raise ValueError("La barre TRS sélectionnée ne correspond pas à un point Sync unique.")
    if not matches:
        turn_boundary = turn_start if side == "start" else turn_end
        if not math.isclose(boundary_time, turn_boundary, abs_tol=1e-6):
            raise ValueError("La barre TRS sélectionnée ne correspond pas à un point Sync.")
        if len(speakers) != 1 or speakers[0] != interval.source_speaker_id:
            raise ValueError(
                "La limite d’un tour partagé entre plusieurs locuteurs ne peut pas "
                "être déplacée séparément."
            )

        section_bounds = [
            (
                _xml_time(section.get("startTime")),
                _xml_time(section.get("endTime")),
            )
            for section in root.iter()
            if _xml_local_name(section.tag) == "Section"
            and any(child is turn for child in section.iter())
        ]
        section_start, section_end = section_bounds[0] if section_bounds else (None, None)
        if (
            section_start is not None
            and not math.isfinite(section_start)
            or section_end is not None
            and not math.isfinite(section_end)
        ):
            raise ValueError("Les limites de la section TRS sont invalides.")

        neighboring_turns: list[tuple[float, float]] = []
        for other_turn in turns:
            if other_turn is turn or interval.source_speaker_id not in (
                other_turn.get("speaker") or ""
            ).split():
                continue
            other_start = _xml_time(other_turn.get("startTime"))
            other_end = _xml_time(other_turn.get("endTime"))
            if (
                other_start is None
                or other_end is None
                or not math.isfinite(other_start)
                or not math.isfinite(other_end)
            ):
                raise ValueError("Les limites d’un autre tour TRS sont invalides.")
            neighboring_turns.append((other_start, other_end))

        if side == "start":
            previous_ends = [
                other_end
                for other_start, other_end in neighboring_turns
                if other_end <= turn_start and other_start < turn_start
            ]
            sync_limit = min(
                (sync_time for sync_time in sync_times if sync_time > turn_start),
                default=turn_end,
            )
            lower_bound = max(
                0.0,
                section_start if section_start is not None else 0.0,
                *previous_ends,
            )
            upper_bound = min(sync_limit, interval.end)
            if not lower_bound <= new_time < upper_bound:
                raise ValueError(
                    "La limite de début doit rester avant le premier point Sync "
                    "et ne pas vider l’intervalle."
                )
            turn.set("startTime", _praat_number(new_time))
        else:
            next_starts = [
                other_start
                for other_start, other_end in neighboring_turns
                if other_start >= turn_end and other_end > turn_end
            ]
            sync_limit = max(
                (sync_time for sync_time in sync_times if sync_time < turn_end),
                default=turn_start,
            )
            lower_bound = max(sync_limit, interval.start)
            upper_bounds = [
                section_end if section_end is not None else float("inf"),
                *next_starts,
            ]
            upper_bound = min(upper_bounds)
            if not lower_bound < new_time <= upper_bound:
                raise ValueError(
                    "La limite de fin doit rester après le dernier point Sync "
                    "et ne pas vider l’intervalle."
                )
            turn.set("endTime", _praat_number(new_time))
        _save_trs_tree(path, tree)
        return

    sync_index = matches[0]
    sync_time = sync_times[sync_index]

    previous_time = (
        sync_times[sync_index - 1] if sync_index > 0 else turn_start
    )
    next_time = (
        sync_times[sync_index + 1]
        if sync_index + 1 < len(sync_times)
        else turn_end
    )
    if (
        previous_time is None
        or next_time is None
        or not previous_time < new_time < next_time
        or side == "start" and new_time >= interval.end
        or side == "end" and new_time <= interval.start
    ):
        raise ValueError(
            "La barre doit rester entre les points Sync voisins et ne pas vider "
            "l’intervalle."
        )
    syncs[sync_index].set("time", _praat_number(new_time))
    _save_trs_tree(path, tree)


def add_trs_turn(
    path: str | Path, speaker_id: str, start: float, end: float, text: str
) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("La sélection doit avoir une durée positive.")
    tiers = parse_trs(path)
    speaker_tier = next(
        (tier for tier in tiers if tier.source_speaker_id == speaker_id), None
    )
    if speaker_tier is None:
        raise ValueError("Le locuteur sélectionné n’existe plus dans ce fichier TRS.")
    if any(
        interval.start < end and start < interval.end
        for interval in speaker_tier.intervals
    ):
        raise ValueError(
            "Cette plage chevauche déjà une annotation de ce locuteur. "
            "Effacez-la ou choisissez une plage libre."
        )

    tree = _load_trs_tree(path)
    root = tree.getroot()
    if root is None:
        raise ValueError("Le fichier TRS ne contient pas de document XML.")
    for turn in root.iter():
        if _xml_local_name(turn.tag) != "Turn":
            continue
        turn_speakers = (turn.get("speaker") or "").split()
        turn_start = _xml_time(turn.get("startTime"))
        turn_end = _xml_time(turn.get("endTime"))
        if (
            speaker_id in turn_speakers
            and turn_start is not None
            and turn_end is not None
            and turn_start < end
            and start < turn_end
        ):
            raise ValueError(
                "Cette plage chevauche déjà un tour TRS de ce locuteur. "
                "Choisissez une plage libre."
            )
    declared_speakers = {
        element.get("id")
        for element in root.iter()
        if _xml_local_name(element.tag) == "Speaker"
    }
    if speaker_id not in declared_speakers:
        raise ValueError("Le locuteur sélectionné n’est pas déclaré dans le TRS.")

    sections = [
        element for element in root.iter() if _xml_local_name(element.tag) == "Section"
    ]
    section = next(
        (
            element
            for element in sections
            if (section_start := _xml_time(element.get("startTime"))) is not None
            and (section_end := _xml_time(element.get("endTime"))) is not None
            and section_start <= start
            and end <= section_end
        ),
        None,
    )
    if section is None:
        raise ValueError("La sélection doit se trouver dans une section TRS existante.")

    turn = ET.Element(
        _qualified_tag(section, "Turn"),
        {
            "startTime": _praat_number(start),
            "endTime": _praat_number(end),
            "speaker": speaker_id,
        },
    )
    sync = ET.SubElement(
        turn,
        _qualified_tag(section, "Sync"),
        {"time": _praat_number(start)},
    )
    sync.tail = text
    insert_at = len(section)
    for index, child in enumerate(section):
        if _xml_local_name(child.tag) == "Turn":
            turn_start = _xml_time(child.get("startTime"))
            if turn_start is not None and turn_start > start:
                insert_at = index
                break
    section.insert(insert_at, turn)
    _save_trs_tree(path, tree)


def _praat_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Un horodatage d’annotation est invalide.")
    return format(value, ".15g")


def _atomic_write_text(path: str | Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: str | Path, content: bytes) -> None:
    destination = Path(path).expanduser().resolve()
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(content)
        os.replace(temporary_path, destination)
    except OSError as error:
        raise ValueError(f"Impossible d’enregistrer {destination.name} : {error}") from error
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


def _load_trs_tree(path: str | Path) -> ET.ElementTree[ET.Element]:
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(Path(path).expanduser(), parser=parser)
        if tree.getroot() is None:
            raise ValueError("Le fichier TRS ne contient pas de document XML.")
        return tree
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"Impossible de lire le fichier TRS : {error}") from error


def _save_trs_tree(path: str | Path, tree: ET.ElementTree[ET.Element]) -> None:
    source_path = Path(path).expanduser()
    try:
        original_bytes = source_path.read_bytes()
    except OSError as error:
        raise ValueError(f"Impossible de lire le fichier TRS : {error}") from error
    if original_bytes.startswith(codecs.BOM_UTF16_LE) or original_bytes.startswith(
        codecs.BOM_UTF16_BE
    ):
        encoding = "utf-16"
    elif original_bytes.startswith(codecs.BOM_UTF8):
        encoding = "utf-8-sig"
    else:
        header = original_bytes[:512].decode("ascii", errors="ignore")
        declaration_encoding = re.search(
            r"""encoding\s*=\s*['"]([^'"]+)['"]""", header, flags=re.IGNORECASE
        )
        encoding = declaration_encoding.group(1) if declaration_encoding else "utf-8"
    try:
        original = original_bytes.decode(encoding)
    except (LookupError, UnicodeError) as error:
        raise ValueError(f"Encodage TRS non pris en charge : {error}") from error

    prolog = []
    declaration = re.search(r"<\?xml[^?]*\?>", original)
    doctype = re.search(r"<!DOCTYPE[^>]*>", original, flags=re.IGNORECASE)
    if declaration:
        prolog.append(declaration.group(0))
    if doctype:
        prolog.append(doctype.group(0))
    root = tree.getroot()
    if root is None:
        raise ValueError("Le fichier TRS ne contient pas de document XML.")
    body = ET.tostring(root, encoding="unicode")
    content = "\n".join((*prolog, body)) + "\n"
    try:
        encoded_content = content.encode(encoding)
    except (LookupError, UnicodeError) as error:
        raise ValueError(f"Encodage TRS non pris en charge : {error}") from error
    _atomic_write_bytes(source_path, encoded_content)


def _qualified_tag(parent: ET.Element, name: str) -> str:
    if parent.tag.startswith("{"):
        namespace = parent.tag.split("}", 1)[0] + "}"
        return namespace + name
    return name


def _field(block: str, field_name: str) -> str | None:
    match = re.search(
        rf'(?m)^\s*{re.escape(field_name)}\s*=\s*"([^"]*)"\s*$',
        block,
    )
    return match.group(1) if match else None


def _quoted_field(block: str, field_name: str) -> str | None:
    match = re.search(
        rf'(?m)^\s*{re.escape(field_name)}\s*=\s*("(?:[^"]|"")*")\s*$',
        block,
    )
    if not match:
        return None
    return match.group(1)[1:-1].replace('""', '"')


def _numeric_field(block: str, field_name: str) -> float | None:
    match = re.search(
        rf"(?m)^\s*{re.escape(field_name)}\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
        block,
    )
    return float(match.group(1)) if match else None


def waveform_peaks(samples: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a min/max envelope with at most one pair per horizontal pixel."""
    if width <= 0 or samples.size == 0:
        return np.empty(0), np.empty(0)
    frame_count = samples.shape[0]
    columns = min(width, frame_count)
    boundaries = np.linspace(0, frame_count, columns + 1, dtype=np.int64)
    lows = np.minimum.reduceat(samples, boundaries[:-1], axis=0)
    highs = np.maximum.reduceat(samples, boundaries[:-1], axis=0)
    if samples.ndim > 1:
        lows = np.min(lows, axis=1)
        highs = np.max(highs, axis=1)
    return lows.astype(np.float32), highs.astype(np.float32)


def waveform_ppm(
    samples: np.ndarray,
    width: int,
    height: int,
    background: tuple[int, int, int] = (251, 252, 255),
    foreground: tuple[int, int, int] = (52, 58, 104),
) -> bytes:
    """Rasterize a waveform envelope into one PPM image."""
    if width <= 0 or height <= 0 or samples.size == 0:
        raise ValueError("Dimensions ou signal invalides pour la forme d’onde.")
    lows, highs = waveform_peaks(samples, width)
    if lows.size == 0:
        raise ValueError("Aucun échantillon disponible pour la forme d’onde.")
    peak = max(float(np.max(np.abs(lows))), float(np.max(np.abs(highs))), 1e-4)
    gain = min(1.0 / peak, 8.0)
    center = height // 2
    amplitude = height * 0.42
    upper_rows = np.clip(
        np.ceil(center - highs * amplitude * gain), 0, height - 1
    ).astype(np.intp)
    lower_rows = np.clip(
        np.floor(center - lows * amplitude * gain), 0, height - 1
    ).astype(np.intp)
    source_columns = np.minimum(
        ((np.arange(width) + 0.5) * lows.size / width).astype(np.intp),
        lows.size - 1,
    )
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    image[center, :] = (170, 177, 187)
    rows = np.arange(height)[:, np.newaxis]
    signal_mask = (
        (rows >= upper_rows[source_columns])
        & (rows <= lower_rows[source_columns])
    )
    image[signal_mask] = foreground
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + image.tobytes()


def selection_zoom_scale(
    selection: tuple[float, float], usable_width: int, fit_scale: float
) -> float:
    """Return the pixels-per-second scale needed to fit a time selection."""
    start, end = selection
    if end <= start or usable_width <= 0:
        raise ValueError("La sélection doit avoir une durée et une largeur positives.")
    return max(fit_scale, usable_width / (end - start))


def playback_frame_range(
    sample_count: int,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> tuple[int, int]:
    """Convert a time range to a non-empty, audio-bounded frame range."""
    if sample_count <= 0 or sample_rate <= 0 or end_time <= start_time:
        raise ValueError("La plage de lecture audio est vide ou invalide.")
    start_frame = max(0, min(sample_count, int(start_time * sample_rate)))
    end_frame = max(
        start_frame,
        min(sample_count, math.ceil(end_time * sample_rate)),
    )
    if end_frame <= start_frame:
        raise ValueError("La plage de lecture audio est vide ou invalide.")
    return start_frame, end_frame


def selection_overlay_png(width: int, height: int, alpha: int = 64) -> bytes:
    """Create a blue RGBA overlay for the selected signal range."""
    if width <= 0 or height <= 0 or not 0 <= alpha <= 255:
        raise ValueError("Dimensions ou transparence invalides pour la sélection.")

    pixel = bytes((95, 125, 235, alpha))
    scanline = b"\x00" + pixel * width
    image_data = zlib.compress(scanline * height)

    def chunk(name: bytes, data: bytes) -> bytes:
        body = name + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", image_data)
        + chunk(b"IEND", b"")
    )


def spectrogram_ppm(
    samples: np.ndarray,
    sample_rate: int,
    width: int,
    height: int,
    settings: SpectrogramSettings | None = None,
    visible_area: tuple[int, int] | None = None,
) -> bytes:
    """Build a spectrogram image using settings similar to Praaline's layer."""
    if sample_rate <= 0 or width <= 0 or height <= 0 or samples.size == 0:
        raise ValueError("Impossible de calculer le spectrogramme de cet audio.")

    settings = settings or SpectrogramSettings()
    channels = samples if samples.ndim == 2 else samples[:, np.newaxis]
    if channels.shape[0] == 0:
        raise ValueError("L’audio est trop court pour calculer un spectrogramme.")

    fft_size = settings.window_size
    if settings.window_hop_level == 0:
        hop_size = fft_size
    elif settings.window_hop_level == 1:
        hop_size = (fft_size * 3) // 4
    else:
        hop_size = fft_size // (1 << (settings.window_hop_level - 1))
    sample_count = channels.shape[0]
    padded_channels = np.pad(
        channels, ((fft_size // 2, fft_size // 2), (0, 0))
    )
    frame_centers = np.arange(0, sample_count, hop_size, dtype=np.int64)
    window = np.hanning(fft_size + 1)[:-1].astype(np.float32)
    frequency_count = fft_size // 2 + 1
    pixel_magnitudes = np.zeros((width, frequency_count), dtype=np.float32)
    pixel_phase_vectors = (
        np.zeros((width, frequency_count), dtype=np.complex128)
        if settings.colour_scale == "Phase"
        else None
    )
    pixel_has_frames = np.zeros(width, dtype=bool)
    window_offsets = np.arange(fft_size)
    batch_size = max(1, min(256, 2_000_000 // (fft_size * channels.shape[1])))
    for batch_start in range(0, len(frame_centers), batch_size):
        centers = frame_centers[batch_start:batch_start + batch_size]
        frame_indices = centers[:, np.newaxis] + window_offsets
        frames = padded_channels[frame_indices]
        spectra = np.fft.rfft(frames * window[np.newaxis, :, np.newaxis], axis=1)
        channel_magnitudes = np.abs(spectra)
        magnitudes = np.sqrt(np.mean(channel_magnitudes**2, axis=2))
        if settings.bin_display != "All Bins" and frequency_count > 2:
            peaks = np.zeros_like(magnitudes, dtype=bool)
            peaks[:, 1:-1] = (
                (magnitudes[:, 1:-1] >= magnitudes[:, :-2])
                & (magnitudes[:, 1:-1] > magnitudes[:, 2:])
            )
            magnitudes = np.where(peaks, magnitudes, 0.0)
        pixel_indices = np.minimum(
            centers * width // sample_count, width - 1
        )
        group_starts_mask = np.empty(len(pixel_indices), dtype=bool)
        group_starts_mask[0] = True
        group_starts_mask[1:] = pixel_indices[1:] != pixel_indices[:-1]
        group_starts = np.flatnonzero(group_starts_mask)
        group_pixels = pixel_indices[group_starts]
        batch_magnitudes = np.maximum.reduceat(
            magnitudes, group_starts, axis=0
        )
        pixel_magnitudes[group_pixels] = np.maximum(
            pixel_magnitudes[group_pixels], batch_magnitudes
        )
        if pixel_phase_vectors is not None:
            batch_phase_vectors = np.add.reduceat(
                np.mean(spectra, axis=2), group_starts, axis=0
            )
            pixel_phase_vectors[group_pixels] += batch_phase_vectors
        pixel_has_frames[group_pixels] = True

    if not np.all(pixel_has_frames):
        pixel_positions = np.arange(width)
        populated_positions = pixel_positions[pixel_has_frames]
        for bin_index in range(frequency_count):
            pixel_magnitudes[:, bin_index] = np.interp(
                pixel_positions,
                populated_positions,
                pixel_magnitudes[pixel_has_frames, bin_index],
            )
    min_bin = 1
    max_bin = min(
        fft_size // 2,
        int(5000.0 * fft_size / sample_rate + 0.001),
    )
    max_bin = max(min_bin, max_bin)
    min_frequency = min_bin * sample_rate / fft_size
    max_frequency = max_bin * sample_rate / fft_size
    if settings.frequency_scale == "Log":
        frequency_edges = np.geomspace(
            min_frequency, max_frequency, height + 1
        )[::-1]
    else:
        frequency_edges = np.linspace(
            max_frequency, min_frequency, height + 1
        )
    frequency_bin_edges = frequency_edges * fft_size / sample_rate
    frequency_positions = (
        frequency_bin_edges[:-1] + frequency_bin_edges[1:]
    ) / 2
    lower_bins = np.floor(frequency_positions).astype(np.intp)
    upper_bins = np.ceil(frequency_positions).astype(np.intp)
    frequency_fraction = (frequency_positions - lower_bins)[:, np.newaxis]
    image_magnitude = (
        pixel_magnitudes[:, upper_bins].T * frequency_fraction
        + pixel_magnitudes[:, lower_bins].T * (1.0 - frequency_fraction)
    )
    for row_index, (first_edge, second_edge) in enumerate(
        zip(frequency_bin_edges, frequency_bin_edges[1:])
    ):
        if abs(first_edge - second_edge) < 1:
            continue
        first_bin = max(0, int(math.floor(min(first_edge, second_edge) + 0.001)))
        last_bin = min(
            frequency_count,
            int(math.ceil(max(first_edge, second_edge) + 0.001)),
        )
        if last_bin <= first_bin:
            last_bin = min(frequency_count, first_bin + 1)
        image_magnitude[row_index] = np.max(
            pixel_magnitudes[:, first_bin:last_bin], axis=1
        )
    image_magnitude /= fft_size / 2
    if settings.normalize_columns:
        column_maxima = np.maximum(
            np.max(image_magnitude, axis=0, keepdims=True),
            np.finfo(np.float32).tiny,
        )
        image_magnitude /= column_maxima
        magnitude_minimum, magnitude_maximum = 0.0, 1.0
    elif settings.normalize_visible_area:
        if visible_area is None:
            normalization_magnitudes = image_magnitude
        else:
            visible_start, visible_end = visible_area
            if not 0 <= visible_start < visible_end <= width:
                raise ValueError("La zone visible du spectrogramme est invalide.")
            normalization_magnitudes = image_magnitude[:, visible_start:visible_end]
        magnitude_minimum = float(np.min(normalization_magnitudes))
        magnitude_maximum = float(np.max(normalization_magnitudes))
    else:
        magnitude_minimum = 0.0
        magnitude_maximum = 0.1 if settings.colour_scale == "Linear" else 1.0
    if magnitude_maximum <= magnitude_minimum:
        magnitude_maximum = magnitude_minimum + np.finfo(np.float32).eps

    gain = 10 ** (settings.gain_db / 20)
    gained_magnitude = image_magnitude * gain
    if settings.colour_scale == "Phase":
        if pixel_phase_vectors is None:
            raise RuntimeError("Les phases du spectrogramme n’ont pas été calculées.")
        pixel_phases = np.angle(pixel_phase_vectors)
        image_phase = (
            pixel_phases[:, upper_bins].T * frequency_fraction
            + pixel_phases[:, lower_bins].T * (1.0 - frequency_fraction)
        )
        intensity = np.clip((image_phase + math.pi) / (2 * math.pi), 0.0, 1.0)
    else:
        normalized = np.clip(
            (gained_magnitude - magnitude_minimum)
            / (magnitude_maximum - magnitude_minimum),
            0.0,
            1.0,
        )
        if settings.colour_scale == "Meter":
            intensity = np.sqrt(normalized)
        elif settings.colour_scale == "dBV^2":
            decibels = 20 * np.log10(np.maximum(normalized, 1e-4))
            decibel_floor = -80.0
            if magnitude_minimum > 0:
                decibel_floor = max(
                    decibel_floor, 20 * math.log10(magnitude_minimum)
                )
            intensity = np.clip(
                (decibels - decibel_floor) / -decibel_floor, 0.0, 1.0
            )
        elif settings.colour_scale == "dBV":
            decibels = 10 * np.log10(np.maximum(normalized, 1e-8))
            decibel_floor = -80.0
            if magnitude_minimum > 0:
                decibel_floor = max(
                    decibel_floor, 10 * math.log10(magnitude_minimum)
                )
            intensity = np.clip(
                (decibels - decibel_floor) / -decibel_floor, 0.0, 1.0
            )
        else:
            intensity = normalized
        if settings.threshold_db > -80:
            threshold = 10 ** (settings.threshold_db / 20)
            intensity = np.where(normalized < threshold, 0.0, intensity)
    palette = _spectrogram_palette(settings.colour_map, settings.colour_rotation)
    if settings.grayscale and settings.colour_map not in {
        "Black on White",
        "White on Black",
    }:
        luminance = palette @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
        palette = np.repeat(luminance[:, np.newaxis], 3, axis=1)
    palette_indices = np.rint(intensity * 255).astype(np.uint8)
    rgb = np.rint(palette[palette_indices] * 255).astype(np.uint8)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb.tobytes()


def _spectrogram_palette(colour_map: str, rotation: int) -> np.ndarray:
    levels = np.linspace(0.0, 1.0, 256)
    palette: list[tuple[float, float, float]] = []
    banded_colours = (
        (0.0, 0.39, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.55),
        (0.0, 0.0, 1.0),
        (0.55, 0.55, 0.0),
        (1.0, 1.0, 0.0),
        (0.55, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    for level in levels:
        if colour_map == "Black on White":
            colour = (1 - level,) * 3
        elif colour_map == "White on Black":
            colour = (level,) * 3
        elif colour_map == "Red on Blue":
            colour = colorsys.hsv_to_rgb(0.5833 + level * 0.4167, 1.0, level)
        elif colour_map == "Yellow on Black":
            colour = (level, level, 0.0)
        elif colour_map == "Blue on Black":
            colour = colorsys.hsv_to_rgb(2 / 3, 1.0, min(1.0, level * 2))
        elif colour_map == "Sunset" or colour_map == "High Gain":
            adjusted = level
            if colour_map == "High Gain":
                adjusted = (
                    0.0
                    if level <= 1 / 256
                    else min(1.0, 0.1 + ((2 * level - 1) ** 3 + 1) / 2.081)
                )
            red = np.clip((adjusted - 0.24) * 2.38, 0.0, 1.0)
            green = np.clip((adjusted - 0.64) * 2.777, 0.0, 1.0)
            blue = 3.6 * adjusted if adjusted <= 0.277 else 2.0 - 3.6 * adjusted
            colour = (float(red), float(green), float(np.clip(blue, 0.0, 1.0)))
        elif colour_map == "Fruit Salad":
            colour = colorsys.hsv_to_rgb((0.7221 - level) % 1.0, 1.0, 1.0)
        elif colour_map == "Banded":
            colour = banded_colours[min(int(level * len(banded_colours)), 7)]
        elif colour_map == "Highlight":
            colour = (1.0, 1.0, 1.0) if level > 0.99 else (0.0, 0.0, 0.55)
        elif colour_map == "Printer":
            grayscale = 1.0 - (math.floor(level * 8) / 10 if level > 0.2 else 0.0)
            colour = (grayscale, grayscale, grayscale)
        else:
            colour = colorsys.hsv_to_rgb(
                (2 / 3) * (1 - level), 0.5 + level / 2, level
            )
        palette.append(colour)
    shift = rotation % 255
    if shift:
        palette[1:] = palette[1 + shift:] + palette[1:1 + shift]
    return np.asarray(palette, dtype=np.float32)


class AudioVisualizer(ttk.Frame):
    LEFT_MARGIN = 116
    AXIS_HEIGHT = 28
    WAVEFORM_HEIGHT = 140
    TIER_HEIGHT = 52
    HIDDEN_TIER_HEIGHT = 28
    SPECTROGRAM_HEIGHT = 230

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.samples: np.ndarray | None = None
        self.sample_rate = 0
        self.duration = 0.0
        self.tiers: tuple[TextGridTier, ...] = ()
        self._visible_tier_indices: set[int] = set()
        self.audio_path: Path | None = None
        self.annotation_path: Path | None = None
        self._annotation_format: str | None = None
        self._textgrid_source: str | None = None
        self.before_annotation_edit: Callable[[], None] | None = None
        self._undo_stack: list[_AnnotationUndo] = []
        self._edit_mode = "select"
        self._edit_drag_start: tuple[int, float] | None = None
        self._boundary_drag_start: _BoundaryDrag | None = None
        self._boundary_drag_preview: float | None = None
        self._tool_images: dict[str, tk.PhotoImage] = {}
        self._tool_buttons: dict[str, ttk.Button] = {}
        self.scale = 1.0
        self.fit_scale = 1.0
        self._zoomed = False
        self.cursor = 0.0
        self.selection: tuple[float, float] | None = None
        self._selection_start: float | None = None
        self._selected_annotation: tuple[int, int] | None = None
        self.spectrogram_settings = SpectrogramSettings()
        self._waveform_image: tk.PhotoImage | None = None
        self._waveform_key: tuple[int, int, int] | None = None
        self._spectrogram_image: tk.PhotoImage | None = None
        self._spectrogram_key: SpectrogramKey | None = None
        self._spectrogram_requested_key: SpectrogramKey | None = None
        self._spectrogram_pending_key: SpectrogramKey | None = None
        self._autoscale_visible_area: tuple[int, int] | None = None
        self._autoscale_redraw_pending = False
        self._selection_overlay: tk.PhotoImage | None = None
        self._selection_overlay_key: tuple[int, int] | None = None
        self._playback_active = False
        self._playback_stream_started_at = 0.0
        self._playback_output_latency = 0.0
        self._playback_start_time = 0.0
        self._playback_end_time = 0.0
        self._playback_poll_id: str | None = None
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._status = tk.StringVar(value="Choisissez un fichier audio pour l’afficher.")

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(3, 0))
        style = ttk.Style(self)
        style.configure(
            "Visualizer.TButton",
            font=("TkDefaultFont", 9),
            padding=(3, 1),
        )
        style.configure(
            "VisualizerActive.TButton",
            font=("TkDefaultFont", 9),
            padding=(3, 1),
            relief="sunken",
        )
        self.play_button = ttk.Button(
            toolbar,
            text="Play View",
            command=self._toggle_playback,
            style="Visualizer.TButton",
        )
        self.play_button.pack(side="left", padx=(0, 4))
        ttk.Button(
            toolbar, text="Fit", command=self.fit, style="Visualizer.TButton"
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="−",
            width=2,
            command=lambda: self.zoom(0.8),
            style="Visualizer.TButton",
        ).pack(
            side="left", padx=(4, 1)
        )
        ttk.Button(
            toolbar,
            text="+",
            width=2,
            command=lambda: self.zoom(1.25),
            style="Visualizer.TButton",
        ).pack(
            side="left"
        )
        ttk.Button(
            toolbar,
            text="Spectrogram Options…",
            command=self._open_spectrogram_options,
            style="Visualizer.TButton",
        ).pack(side="left", padx=(5, 0))
        ttk.Button(
            toolbar,
            text="Video",
            command=self._toggle_video_panel,
            style="Visualizer.TButton",
        ).pack(side="left", padx=(5, 0))
        tools_toolbar = ttk.Frame(self)
        tools_toolbar.pack(fill="x", pady=(2, 0))
        for mode, label, icon_name in (
            ("select", "Select", "select.png"),
            ("erase", "Erase", "erase.png"),
            ("delete_boundary", "Delete Boundary", "interval_delete.png"),
            ("add_boundary", "Add Boundary", "interval_add.png"),
            ("move_boundary", "Move Boundary", None),
            ("draw", "Annotate", "draw.png"),
        ):
            image = None
            if icon_name is not None:
                icon_path = Path(__file__).with_name("icons") / icon_name
                image = tk.PhotoImage(file=str(icon_path))
                self._tool_images[mode] = image
            button = ttk.Button(
                tools_toolbar,
                text=label,
                command=lambda selected_mode=mode: self._set_edit_mode(selected_mode),
                style="Visualizer.TButton",
            )
            if image is not None:
                button.configure(image=image, compound="left")
            button.pack(side="left", padx=(4, 0))
            self._tool_buttons[mode] = button
        self._update_tool_buttons()
        ttk.Label(self, textvariable=self._status, anchor="w").pack(
            fill="x", padx=4, pady=(2, 4)
        )

        self._content_pane = ttk.Panedwindow(self, orient="horizontal")
        self._content_pane.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(self._content_pane)
        self._content_pane.add(canvas_frame, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
            takefocus=True,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.horizontal_scroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self.canvas.xview
        )
        self.horizontal_scroll.grid(row=1, column=0, sticky="ew")
        self.vertical_scroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self.canvas.yview
        )
        self.vertical_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(
            xscrollcommand=self._on_horizontal_view_changed,
            yscrollcommand=self.vertical_scroll.set,
        )
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self._on_annotation_double_click)
        self.canvas.bind("<B1-Motion>", self._drag_selection)
        self.canvas.bind("<ButtonRelease-1>", self._end_selection)
        self.canvas.bind("<Return>", self._zoom_to_selection)
        self.canvas.bind("<KP_Enter>", self._zoom_to_selection)
        self.canvas.bind("<space>", self._on_playback_shortcut)
        self.canvas.bind("<Shift-Left>", self._on_annotation_navigation)
        self.canvas.bind("<Shift-Right>", self._on_annotation_navigation)
        self.canvas.bind("<Shift-Up>", self._on_annotation_navigation)
        self.canvas.bind("<Shift-Down>", self._on_annotation_navigation)
        self.winfo_toplevel().bind(
            "<Command-z>", self._on_undo_shortcut, add="+"
        )
        self.winfo_toplevel().bind(
            "<Control-z>", self._on_undo_shortcut, add="+"
        )
        self._video_player = VideoPlayer(
            self._content_pane,
            on_visibility_changed=self._set_video_panel_visible,
        )
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.after(100, self._poll_events)

    def load_audio(self, path: str | Path) -> None:
        audio_path = Path(path).expanduser().resolve()
        self._status.set(f"Chargement de {audio_path.name}…")

        def load() -> None:
            try:
                samples, sample_rate = sf.read(
                    audio_path, dtype="float32", always_2d=True
                )
                if sample_rate <= 0 or samples.shape[0] == 0:
                    raise ValueError("Le fichier audio est vide ou invalide.")
                self._events.put(("audio", (audio_path, samples, sample_rate)))
            except (OSError, RuntimeError, ValueError) as error:
                self._events.put(("error", f"Lecture audio impossible : {error}"))

        threading.Thread(target=load, daemon=True).start()

    def load_textgrid(self, path: str | Path) -> None:
        self._undo_stack.clear()
        self.tiers = ()
        self._visible_tier_indices.clear()
        self._selected_annotation = None
        self._update_tool_buttons()
        self.annotation_path = Path(path).expanduser().resolve()
        self._annotation_format = "textgrid"
        self._textgrid_source = None
        try:
            self.tiers = parse_textgrid(self.annotation_path)
            self._textgrid_source = self.annotation_path.read_text(encoding="utf-8-sig")
        except ValueError as error:
            self._status.set(str(error))
            self._redraw()
            return
        except (OSError, UnicodeError) as error:
            self._status.set(f"Impossible de lire le TextGrid : {error}")
            self._redraw()
            return
        self._visible_tier_indices = set(range(len(self.tiers)))
        self._status.set(f"{len(self.tiers)} tier(s) chargé(s) : {Path(path).name}")
        self._update_tool_buttons()
        self._redraw()

    def load_trs(self, path: str | Path) -> None:
        self._undo_stack.clear()
        self.tiers = ()
        self._visible_tier_indices.clear()
        self._selected_annotation = None
        self._update_tool_buttons()
        self.annotation_path = Path(path).expanduser().resolve()
        self._annotation_format = "trs"
        self._textgrid_source = None
        try:
            self.tiers = parse_trs(self.annotation_path)
        except ValueError as error:
            self._status.set(str(error))
            self._redraw()
            return
        self._visible_tier_indices = set(range(len(self.tiers)))
        self._status.set(f"{len(self.tiers)} tier(s) TRS chargé(s) : {Path(path).name}")
        self._update_tool_buttons()
        self._redraw()

    def clear_textgrid(self) -> None:
        self._undo_stack.clear()
        self.tiers = ()
        self._visible_tier_indices.clear()
        self._selected_annotation = None
        self.annotation_path = None
        self._annotation_format = None
        self._textgrid_source = None
        self._update_tool_buttons()
        self._redraw()

    def fit(self) -> None:
        self._zoomed = False
        self.scale = self.fit_scale
        self.canvas.xview_moveto(0)
        self._redraw()

    def zoom(self, factor: float) -> None:
        if self.duration <= 0:
            return
        self._zoomed = True
        self.scale = min(max(self.scale * factor, self.fit_scale * 0.5), self.fit_scale * 12)
        self._redraw()

    def _toggle_playback(self) -> None:
        if self._playback_active:
            self._stop_playback()
            return
        if self.samples is None or self.sample_rate <= 0:
            self._status.set("Ouvrez un fichier audio avant de lancer la lecture.")
            return

        if self.selection is not None and self.selection[1] > self.selection[0]:
            start_time, end_time = self.selection
        else:
            left = (self.canvas.canvasx(0) - self.LEFT_MARGIN) / self.scale
            right = (
                self.canvas.canvasx(self.canvas.winfo_width()) - self.LEFT_MARGIN
            ) / self.scale
            start_time = max(0.0, left)
            end_time = min(self.duration, right)

        try:
            start_frame, end_frame = playback_frame_range(
                self.samples.shape[0],
                self.sample_rate,
                start_time,
                end_time,
            )
        except ValueError as error:
            self._status.set(str(error))
            return

        try:
            import sounddevice as sd
        except ImportError as error:
            self._status.set(f"Bibliothèque de lecture audio indisponible : {error}")
            return
        try:
            sd.play(
                self.samples[start_frame:end_frame],
                self.sample_rate,
                blocking=False,
            )
        except (sd.PortAudioError, ValueError) as error:
            self._status.set(f"Lecture audio impossible : {error}")
            return

        try:
            stream = sd.get_stream()
            self._playback_stream_started_at = stream.time
            self._playback_output_latency = max(0.0, float(stream.latency))
        except (sd.PortAudioError, RuntimeError, ValueError) as error:
            try:
                sd.stop()
            except sd.PortAudioError as stop_error:
                self._status.set(
                    f"Synchronisation audio impossible : {error}. "
                    f"Arrêt de la lecture impossible : {stop_error}"
                )
            else:
                self._status.set(f"Synchronisation audio impossible : {error}")
            return

        self._playback_start_time = start_frame / self.sample_rate
        self._playback_end_time = end_frame / self.sample_rate
        self._playback_active = True
        self._video_player.set_timeline_position(self._playback_start_time, force=True)
        self._video_player.set_audio_playing(True)
        self.play_button.configure(text="Stop")
        self._status.set(
            f"Lecture : {self._playback_start_time:.2f}–"
            f"{self._playback_end_time:.2f} s"
        )
        self._poll_playback()

    def _on_playback_shortcut(self, _event: tk.Event) -> str:
        self._toggle_playback()
        return "break"

    def _on_annotation_navigation(self, event: tk.Event) -> str:
        selected = self._selected_annotation
        if (
            self._edit_mode != "select"
            or selected is None
            or self.selection is None
            or not self._selection_matches_annotation(*selected)
        ):
            self._selected_annotation = None
            self._status.set("Sélectionnez une annotation avant d’utiliser Maj+flèches.")
            return "break"

        tier_index, interval_index = selected
        interval = self.tiers[tier_index].intervals[interval_index]
        if event.keysym == "Left":
            target_index = interval_index - 1
            if target_index < 0:
                self._status.set("Aucune annotation précédente dans ce niveau.")
                return "break"
            self._select_annotation(tier_index, target_index, reveal=True)
        elif event.keysym == "Right":
            target_index = interval_index + 1
            if target_index >= len(self.tiers[tier_index].intervals):
                self._status.set("Aucune annotation suivante dans ce niveau.")
                return "break"
            self._select_annotation(tier_index, target_index, reveal=True)
        else:
            direction = -1 if event.keysym == "Up" else 1
            tier_indices = [
                index
                for index, tier in enumerate(self.tiers)
                if index in self._visible_tier_indices and tier.intervals
            ]
            if tier_index not in tier_indices:
                tier_indices.append(tier_index)
                tier_indices.sort()
            next_tier_position = tier_indices.index(tier_index) + direction
            if not 0 <= next_tier_position < len(tier_indices):
                self._status.set("Aucun niveau d’annotation dans cette direction.")
                return "break"
            target_tier_index = tier_indices[next_tier_position]
            reference_time = (interval.start + interval.end) / 2
            target_intervals = self.tiers[target_tier_index].intervals
            target_index = min(
                range(len(target_intervals)),
                key=lambda index: _distance_to_interval(
                    reference_time, target_intervals[index]
                ),
            )
            self._select_annotation(target_tier_index, target_index, reveal=True)
        return "break"

    def _selection_matches_annotation(
        self, tier_index: int, interval_index: int
    ) -> bool:
        if not (
            0 <= tier_index < len(self.tiers)
            and 0 <= interval_index < len(self.tiers[tier_index].intervals)
        ):
            return False
        interval = self.tiers[tier_index].intervals[interval_index]
        return self.selection == (interval.start, interval.end)

    def _select_annotation(
        self, tier_index: int, interval_index: int, *, reveal: bool = False
    ) -> None:
        interval = self.tiers[tier_index].intervals[interval_index]
        self._selected_annotation = (tier_index, interval_index)
        self.selection = (interval.start, interval.end)
        self.cursor = interval.start
        self._video_player.set_timeline_position(self.cursor, force=True)
        self._status.set(
            f"Annotation sélectionnée : {self.tiers[tier_index].name}"
            f" · {interval.start:.3f}–{interval.end:.3f} s"
        )
        self._redraw()
        if reveal:
            self._scroll_annotation_into_view(tier_index, interval)

    def _scroll_annotation_into_view(
        self, tier_index: int, interval: Interval
    ) -> None:
        self.canvas.update_idletasks()
        bounds = self.canvas.bbox("all")
        if bounds is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        content_width = max(1, bounds[2] - bounds[0])
        x1 = self._x_for_time(interval.start)
        x2 = self._x_for_time(interval.end)
        visible_left = self.canvas.canvasx(0)
        visible_right = self.canvas.canvasx(canvas_width)
        if x1 < visible_left:
            target_x = max(0, x1 - self.LEFT_MARGIN)
            self.canvas.xview_moveto(min(1.0, target_x / content_width))
        elif x2 > visible_right:
            target_x = max(0, x2 - canvas_width + self.LEFT_MARGIN)
            self.canvas.xview_moveto(min(1.0, target_x / content_width))

        tier_top = self.AXIS_HEIGHT + self.WAVEFORM_HEIGHT
        row_top, row_height = self._tier_row_layout()[tier_index]
        y1 = tier_top + row_top
        y2 = y1 + row_height
        canvas_height = max(1, self.canvas.winfo_height())
        if y1 < self.canvas.canvasy(0) or y2 > self.canvas.canvasy(canvas_height):
            content_height = max(1, bounds[3] - bounds[1])
            target_y = max(0, y1 - self.AXIS_HEIGHT)
            self.canvas.yview_moveto(min(1.0, target_y / content_height))

    def _stop_playback(self) -> None:
        if not self._playback_active:
            return
        try:
            import sounddevice as sd
        except ImportError as error:
            self._status.set(f"Bibliothèque de lecture audio indisponible : {error}")
            self._finish_playback()
            return
        try:
            sd.stop()
        except sd.PortAudioError as error:
            self._status.set(f"Arrêt de la lecture impossible : {error}")
        self._finish_playback()

    def _poll_playback(self) -> None:
        if not self._playback_active:
            return
        try:
            import sounddevice as sd
        except ImportError as error:
            self._status.set(f"Bibliothèque de lecture audio indisponible : {error}")
            self._finish_playback()
            return
        try:
            stream = sd.get_stream()
            active = stream is not None and stream.active
            stream_time = stream.time if active and stream is not None else 0.0
        except (sd.PortAudioError, RuntimeError) as error:
            self._status.set(f"État de lecture indisponible : {error}")
            self._finish_playback()
            return

        if not active:
            self.cursor = self._playback_end_time
            self._video_player.set_timeline_position(self.cursor, force=True)
            self._finish_playback()
            return

        elapsed = max(
            0.0,
            stream_time
            - self._playback_stream_started_at
            - self._playback_output_latency,
        )
        self.cursor = min(
            self._playback_start_time + elapsed,
            self._playback_end_time,
        )
        self._video_player.set_timeline_position(self.cursor)
        cursor_x = self._x_for_time(self.cursor)
        bounds = self.canvas.bbox("all")
        if bounds is not None:
            self.canvas.coords("playback-cursor", cursor_x, 0, cursor_x, bounds[3])
        self._playback_poll_id = self.after(50, self._poll_playback)

    def _finish_playback(self) -> None:
        self._playback_active = False
        self._playback_poll_id = None
        self._video_player.set_audio_playing(False)
        self.play_button.configure(text="Play View")
        self._redraw()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            if self._playback_active:
                self._stop_playback()
            self._video_player.shutdown()

    def _zoom_to_selection(self, _event: tk.Event | None = None) -> str:
        if self.selection is None or self.duration <= 0:
            return "break"

        start, end = self.selection
        usable_width = max(
            1, self.canvas.winfo_width() - self.LEFT_MARGIN - 12
        )
        self.scale = selection_zoom_scale(
            self.selection, usable_width, self.fit_scale
        )
        self._zoomed = True
        self._redraw()
        self.canvas.update_idletasks()
        content_width = self.LEFT_MARGIN + int(self.duration * self.scale) + 12
        target_offset = max(0.0, self._x_for_time(start) - self.LEFT_MARGIN)
        self.canvas.xview_moveto(
            min(1.0, target_offset / max(content_width, 1))
        )
        return "break"

    def _poll_events(self) -> None:
        try:
            kind, payload = self._events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_events)
            return

        if kind == "audio":
            if self._playback_active:
                self._stop_playback()
            audio_path, samples, sample_rate = payload
            self.audio_path = audio_path
            self._video_player.set_audio_file(audio_path)
            self.samples = samples
            self.sample_rate = sample_rate
            self.duration = samples.shape[0] / sample_rate
            self._waveform_image = None
            self._waveform_key = None
            self.cursor = 0.0
            self._video_player.set_timeline_position(0.0, force=True)
            self.selection = None
            self._spectrogram_key = None
            canvas_width = max(1, self.canvas.winfo_width())
            self.fit_scale = max(0.01, (canvas_width - self.LEFT_MARGIN - 12) / self.duration)
            self.scale = self.fit_scale
            self._zoomed = False
            self._status.set(
                f"{audio_path.name} · {self.duration:.2f} s · {sample_rate} Hz"
            )
            self.canvas.xview_moveto(0)
            self._redraw()
        elif kind == "spectrogram":
            key, ppm = payload
            if self._spectrogram_pending_key == key:
                self._spectrogram_pending_key = None
            if key == self._spectrogram_requested_key:
                try:
                    self._spectrogram_image = tk.PhotoImage(
                        data=ppm.decode("latin-1"), format="PPM"
                    )
                    self._spectrogram_key = key
                except tk.TclError as error:
                    self._spectrogram_image = None
                    self._spectrogram_key = None
                    self._status.set(f"Spectrogramme indisponible : {error}")
            self._redraw()
        elif kind == "spectrogram_error":
            key, error_message = payload
            if self._spectrogram_pending_key == key:
                self._spectrogram_pending_key = None
            if key == self._spectrogram_requested_key:
                self._status.set(f"Spectrogramme indisponible : {error_message}")
            else:
                self._redraw()
        elif kind == "error":
            self._status.set(str(payload))
        self.after(100, self._poll_events)

    def _on_configure(self, _event: tk.Event) -> None:
        if self.duration > 0 and not self._zoomed:
            width = max(1, self.canvas.winfo_width())
            self.fit_scale = max(
                0.01, (width - self.LEFT_MARGIN - 12) / self.duration
            )
            self.scale = self.fit_scale
        self._redraw()

    def _redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        if self.samples is None or self.duration <= 0:
            canvas.configure(scrollregion=(0, 0, max(1, canvas.winfo_width()), 1))
            canvas.create_text(
                18, 30, anchor="nw", fill="#6b7280",
                text="La forme d’onde, les annotations et le spectrogramme apparaîtront ici.",
            )
            return

        track_width = max(1, int(self.duration * self.scale))
        content_width = self.LEFT_MARGIN + track_width + 12
        tier_top = self.AXIS_HEIGHT + self.WAVEFORM_HEIGHT
        spectrogram_top = tier_top + self._annotation_area_height()
        content_height = spectrogram_top + self.SPECTROGRAM_HEIGHT + 16
        canvas.configure(scrollregion=(0, 0, content_width, content_height))
        canvas.create_rectangle(
            0, 0, content_width, content_height, fill="#ffffff", outline=""
        )
        self._draw_axis(track_width)
        self._draw_waveform(track_width)
        self._draw_tiers(track_width, tier_top)
        self._draw_boundary_drag_preview(tier_top)
        self._draw_spectrogram(track_width, spectrogram_top)
        self._draw_grid(track_width)
        self._draw_selection(content_height, track_width)
        self._draw_cursor(content_height)

    def _x_for_time(self, time: float) -> float:
        return self.LEFT_MARGIN + time * self.scale

    def _draw_boundary_drag_preview(self, tier_top: int) -> None:
        if (
            self._boundary_drag_start is None
            or self._boundary_drag_preview is None
        ):
            return
        tier_index = self._boundary_drag_start.tier_index
        row_top, row_height = self._tier_row_layout()[tier_index]
        x = self._x_for_time(self._boundary_drag_preview)
        top = tier_top + row_top
        self.canvas.create_line(
            x,
            top,
            x,
            top + row_height,
            fill="#2563eb",
            width=2,
            dash=(4, 2),
            tags=("boundary-drag-preview",),
        )

    def _tier_row_layout(self) -> tuple[tuple[float, float], ...]:
        tier_count = len(self.tiers)
        if not tier_count:
            return ()

        visible_count = len(self._visible_tier_indices)
        if not visible_count:
            heights = [self.TIER_HEIGHT] * tier_count
        else:
            hidden_count = tier_count - visible_count
            visible_height = (
                tier_count * self.TIER_HEIGHT
                - hidden_count * self.HIDDEN_TIER_HEIGHT
            ) / visible_count
            heights = [
                visible_height if index in self._visible_tier_indices
                else self.HIDDEN_TIER_HEIGHT
                for index in range(tier_count)
            ]

        rows = []
        top = 0.0
        for height in heights:
            rows.append((top, height))
            top += height
        return tuple(rows)

    def _annotation_area_height(self) -> int:
        rows = self._tier_row_layout()
        return round(sum(height for _top, height in rows)) if rows else self.TIER_HEIGHT

    def _draw_selection(self, content_height: int, track_width: int) -> None:
        if self.selection is None:
            return
        start, end = self.selection
        x1 = self._x_for_time(start)
        x2 = self._x_for_time(end)
        overlay_width = max(1, int(round(x2 - x1)))
        overlay_key = (overlay_width, content_height)
        if self._selection_overlay_key != overlay_key:
            png = selection_overlay_png(*overlay_key)
            self._selection_overlay = tk.PhotoImage(
                data=base64.b64encode(png).decode("ascii"),
                format="png",
            )
            self._selection_overlay_key = overlay_key
        if self._selection_overlay is not None:
            self.canvas.create_image(
                x1, 0, anchor="nw", image=self._selection_overlay
            )
        self.canvas.create_rectangle(
            x1, 0, x2, content_height,
            fill="", outline="#738be0",
        )

    def _draw_axis(self, track_width: int) -> None:
        canvas = self.canvas
        canvas.create_rectangle(
            0, 0, self.LEFT_MARGIN, self.AXIS_HEIGHT,
            fill="#f3f4f6", outline="#d1d5db",
        )
        canvas.create_text(
            12, self.AXIS_HEIGHT - 6, anchor="sw",
            text="Temps (s)", fill="#4b5563",
        )
        canvas.create_line(
            self.LEFT_MARGIN, self.AXIS_HEIGHT - 1,
            self.LEFT_MARGIN + track_width, self.AXIS_HEIGHT - 1,
            fill="#78808a",
        )
        step = _time_step(self.duration, track_width)
        tick = 0.0
        while tick <= self.duration:
            x = self._x_for_time(tick)
            canvas.create_line(x, 0, x, self.AXIS_HEIGHT, fill="#e1e4e8")
            canvas.create_line(x, self.AXIS_HEIGHT - 6, x, self.AXIS_HEIGHT, fill="#667085")
            canvas.create_text(
                x + 3, 4, anchor="nw", text=f"{tick:g}", fill="#4b5563"
            )
            tick += step

    def _draw_waveform(self, track_width: int) -> None:
        canvas = self.canvas
        top = self.AXIS_HEIGHT
        canvas.create_rectangle(
            0, top, self.LEFT_MARGIN, top + self.WAVEFORM_HEIGHT,
            fill="#f8fafc", outline="#d1d5db",
        )
        canvas.create_text(
            12, top + self.WAVEFORM_HEIGHT // 2,
            anchor="w", text="Forme d’onde", fill="#4b5563",
        )
        if self.samples is None:
            canvas.create_rectangle(
                self.LEFT_MARGIN, top, self.LEFT_MARGIN + track_width,
                top + self.WAVEFORM_HEIGHT, fill="#fbfcff", outline="#d1d5db",
            )
            return
        key = (id(self.samples), track_width, self.WAVEFORM_HEIGHT)
        if self._waveform_key != key:
            try:
                ppm = waveform_ppm(
                    self.samples, track_width, self.WAVEFORM_HEIGHT
                )
                self._waveform_image = tk.PhotoImage(
                    data=ppm.decode("latin-1"), format="PPM"
                )
                self._waveform_key = key
            except (ValueError, tk.TclError) as error:
                self._waveform_image = None
                self._waveform_key = None
                self._status.set(f"Forme d’onde indisponible : {error}")
        if self._waveform_image is not None:
            canvas.create_image(
                self.LEFT_MARGIN, top,
                anchor="nw", image=self._waveform_image,
            )
            canvas.create_rectangle(
                self.LEFT_MARGIN, top, self.LEFT_MARGIN + track_width,
                top + self.WAVEFORM_HEIGHT, fill="", outline="#d1d5db",
            )

    def _draw_tiers(self, track_width: int, tier_top: int) -> None:
        canvas = self.canvas
        if not self.tiers:
            y = tier_top
            canvas.create_rectangle(
                0, y, self.LEFT_MARGIN, y + self.TIER_HEIGHT,
                fill="#f8fafc", outline="#d1d5db",
            )
            canvas.create_text(
                12, y + self.TIER_HEIGHT / 2, anchor="w",
                text="Annotations", fill="#4b5563",
            )
            canvas.create_rectangle(
                self.LEFT_MARGIN, y, self.LEFT_MARGIN + track_width,
                y + self.TIER_HEIGHT, fill="#ffffff", outline="#d1d5db",
            )
            canvas.create_text(
                self.LEFT_MARGIN + 12, y + self.TIER_HEIGHT / 2,
                anchor="w", text="Ouvrez un TextGrid pour afficher ses tiers.",
                fill="#9ca3af",
            )
            return

        for tier_index, (tier, (relative_top, row_height)) in enumerate(
            zip(self.tiers, self._tier_row_layout())
        ):
            top = tier_top + relative_top
            bottom = top + row_height
            visible = tier_index in self._visible_tier_indices
            tier_color, tier_background = TIER_COLORS[tier_index % len(TIER_COLORS)]
            annotation_font_size = max(
                9, min(18, round(9 * row_height / self.TIER_HEIGHT))
            )
            canvas.create_rectangle(
                0, top, self.LEFT_MARGIN, bottom,
                fill=tier_background if visible else "#f3f4f6",
                outline=tier_color if visible else "#d1d5db",
            )
            checkbox_top = (top + bottom - 14) / 2
            canvas.create_rectangle(
                8,
                checkbox_top,
                22,
                checkbox_top + 14,
                fill="#ffffff",
                outline=tier_color if visible else "#9ca3af",
            )
            if visible:
                canvas.create_line(
                    11, checkbox_top + 7, 14, checkbox_top + 10,
                    fill=tier_color,
                    width=2,
                )
                canvas.create_line(
                    14, checkbox_top + 10, 20, checkbox_top + 3,
                    fill=tier_color,
                    width=2,
                )
            canvas.create_text(
                30,
                (top + bottom) / 2,
                anchor="w",
                text=tier.name,
                fill=tier_color if visible else "#9ca3af",
            )
            canvas.create_rectangle(
                self.LEFT_MARGIN, top, self.LEFT_MARGIN + track_width, bottom,
                fill=tier_background if visible else "#fafafa",
                outline=tier_color if visible else "#cbd5e1",
            )
            intervals = tier.intervals if visible else ()
            for interval in intervals:
                start = max(0.0, min(interval.start, self.duration))
                end = max(start, min(interval.end, self.duration))
                x1 = self._x_for_time(start)
                x2 = self._x_for_time(end)
                canvas.create_line(x1, top, x1, bottom, fill=tier_color, width=2)
                if self._annotation_format == "trs":
                    canvas.create_line(x2, top, x2, bottom, fill=tier_color, width=2)
                if interval.text and x2 - x1 > 14:
                    canvas.create_text(
                        (x1 + x2) / 2, (top + bottom) / 2,
                        text=interval.text,
                        fill="#3f4858",
                        font=("", annotation_font_size),
                        width=max(1, x2 - x1 - 4),
                    )
            canvas.create_line(
                self.LEFT_MARGIN + track_width, top,
                self.LEFT_MARGIN + track_width, bottom, fill=tier_color,
            )

    def _on_canvas_click(self, event: tk.Event) -> str | None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if x < self.LEFT_MARGIN:
            tier_index = self._tier_index_at_y(y)
            if (
                8 <= x <= 22
                and tier_index is not None
            ):
                if tier_index in self._visible_tier_indices:
                    self._visible_tier_indices.remove(tier_index)
                    state = "masqué"
                else:
                    self._visible_tier_indices.add(tier_index)
                    state = "affiché"
                self._status.set(f"Niveau « {self.tiers[tier_index].name} » {state}.")
                self._redraw()
            return "break"

        tier_index = self._tier_index_at_y(y)
        if self._edit_mode == "draw":
            if tier_index is None:
                self._status.set("Le crayon doit commencer dans une ligne d’annotation.")
                return "break"
            start = self._event_time(event)
            self._edit_drag_start = (tier_index, start)
            self.selection = (start, start)
            self._redraw()
            return "break"

        if self._edit_mode == "move_boundary":
            if tier_index is None:
                self._status.set("Cliquez sur une barre d’annotation.")
                return "break"
            time = self._event_time(event)
            tolerance = 5.0 / max(self.scale, 0.01)
            if self._annotation_format == "textgrid":
                boundary_index = self._boundary_index_at_time(
                    tier_index, time, tolerance
                )
                if boundary_index is None:
                    self._status.set("Aucune barre TextGrid proche du clic.")
                    return "break"
                drag = _BoundaryDrag(tier_index, boundary_index, "internal")
                boundary_time = self.tiers[tier_index].intervals[
                    boundary_index
                ].start
            elif self._annotation_format == "trs":
                trs_boundary = self._trs_boundary_at_time(
                    tier_index, time, tolerance
                )
                if trs_boundary is None:
                    self._status.set("Aucune barre TRS proche du clic.")
                    return "break"
                interval_index, side = trs_boundary
                drag = _BoundaryDrag(tier_index, interval_index, side)
                interval = self.tiers[tier_index].intervals[interval_index]
                boundary_time = interval.start if side == "start" else interval.end
            else:
                self._status.set("Le déplacement est disponible pour TextGrid et TRS.")
                return "break"
            self._boundary_drag_start = drag
            self._boundary_drag_preview = boundary_time
            self.selection = None
            self.canvas.focus_set()
            self._status.set("Faites glisser la barre vers sa nouvelle position.")
            self._redraw()
            return "break"

        if self._edit_mode == "delete_boundary":
            if tier_index is None:
                self._status.set("Cliquez sur une barre de séparation d’un TextGrid.")
                return "break"
            boundary_index = self._boundary_index_at_time(
                tier_index,
                self._event_time(event),
                5.0 / max(self.scale, 0.01),
            )
            if boundary_index is None:
                self._status.set("Aucune barre de séparation proche du clic.")
                return "break"
            self._delete_annotation_boundary(tier_index, boundary_index)
            return "break"

        if self._edit_mode == "add_boundary":
            if tier_index is None:
                self._status.set("Cliquez dans un niveau TextGrid pour ajouter une barre.")
                return "break"
            self._add_annotation_boundary(tier_index, self._event_time(event))
            return "break"

        if tier_index is not None:
            time = self._event_time(event)
            interval_index = self._interval_index_at_time(tier_index, time)
            if interval_index is not None:
                if self._edit_mode == "erase":
                    self._erase_annotation(tier_index, interval_index)
                else:
                    self.canvas.focus_set()
                    self._select_annotation(tier_index, interval_index)
                return "break"
            if self._edit_mode == "erase":
                self._status.set("Aucune annotation à effacer à cet endroit.")
                return "break"
        self._begin_selection(event)
        return None

    def _on_annotation_double_click(self, event: tk.Event) -> str:
        if self._edit_mode != "select":
            return "break"
        x = self.canvas.canvasx(event.x)
        tier_index = self._tier_index_at_y(self.canvas.canvasy(event.y))
        if x < self.LEFT_MARGIN or tier_index is None:
            return "break"
        interval_index = self._interval_index_at_time(
            tier_index, self._event_time(event)
        )
        if interval_index is not None:
            self._edit_annotation_text(tier_index, interval_index)
        return "break"

    def _on_undo_shortcut(self, event: tk.Event) -> str | None:
        widget = event.widget
        while widget is not None:
            if widget is self:
                return self._undo_last_annotation_edit(event)
            widget = getattr(widget, "master", None)
        return None

    def _undo_last_annotation_edit(self, _event: tk.Event | None = None) -> str:
        if not self._undo_stack:
            self._status.set("Aucune modification à annuler.")
            return "break"
        record = self._undo_stack[-1]
        if (
            self.annotation_path is None
            or self.annotation_path.resolve() != record.path
        ):
            self._undo_stack.clear()
            self._status.set("Le fichier d’annotation a changé ; annulation impossible.")
            return "break"
        if self._annotation_format not in {"textgrid", "trs"}:
            self._status.set("Aucun fichier d’annotation modifiable n’est ouvert.")
            return "break"

        try:
            current_content = record.path.read_bytes()
            if hashlib.sha256(current_content).digest() != record.resulting_digest:
                raise ValueError(
                    "Le fichier a été modifié depuis la dernière action."
                )
            if self.before_annotation_edit is not None:
                self.before_annotation_edit()
            _atomic_write_bytes(record.path, record.content)
            if self._annotation_format == "textgrid":
                tiers = parse_textgrid(record.path)
                textgrid_source = record.path.read_text(encoding="utf-8-sig")
            else:
                tiers = parse_trs(record.path)
                textgrid_source = None
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            self._status.set(f"Annulation impossible : {error}")
            return "break"

        self._undo_stack.pop()
        self.tiers = tiers
        self._textgrid_source = textgrid_source
        self._visible_tier_indices = set(range(len(tiers)))
        self.selection = None
        self._status.set("Dernière modification annulée.")
        self._update_tool_buttons()
        self._redraw()
        return "break"

    def _capture_annotation_edit(self) -> tuple[Path, bytes]:
        if self.annotation_path is None:
            raise ValueError("Aucun fichier d’annotation modifiable n’est ouvert.")
        path = self.annotation_path.resolve()
        try:
            previous_content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Impossible de préparer l’annulation : {error}") from error
        if self.before_annotation_edit is not None:
            self.before_annotation_edit()
        return path, previous_content

    def _commit_annotation_edit(
        self, snapshot: tuple[Path, bytes]
    ) -> None:
        path, previous_content = snapshot
        try:
            updated_content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Impossible de préparer l’annulation : {error}") from error
        if updated_content != previous_content:
            self._undo_stack.append(
                _AnnotationUndo(
                    path,
                    previous_content,
                    hashlib.sha256(updated_content).digest(),
                )
            )
            del self._undo_stack[:-_MAX_UNDO_STEPS]

    def _edit_annotation_text(self, tier_index: int, interval_index: int) -> None:
        interval = self.tiers[tier_index].intervals[interval_index]
        text = simpledialog.askstring(
            "Modifier l’annotation",
            f"Texte pour « {self.tiers[tier_index].name} » :",
            initialvalue=interval.text,
            parent=self.winfo_toplevel(),
        )
        if text is None:
            self._status.set("Modification d’annotation annulée.")
            return

        try:
            snapshot = self._capture_annotation_edit()
            if self._annotation_format == "textgrid":
                tier = set_tier_interval_text(
                    self.tiers[tier_index], interval_index, text
                )
                tiers = list(self.tiers)
                tiers[tier_index] = tier
                if self.annotation_path is None or self._textgrid_source is None:
                    raise ValueError("Le TextGrid source n’est plus disponible.")
                self._textgrid_source = save_textgrid(
                    self.annotation_path, self._textgrid_source, tuple(tiers)
                )
                self.tiers = tuple(tiers)
            elif self._annotation_format == "trs" and self.annotation_path is not None:
                set_trs_interval_text(self.annotation_path, interval, text)
                self.tiers = parse_trs(self.annotation_path)
                self._visible_tier_indices = set(range(len(self.tiers)))
            else:
                raise ValueError("Aucun fichier d’annotation modifiable n’est ouvert.")
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Modification impossible : {error}")
            return
        self._status.set("Annotation modifiée et enregistrée.")
        self._redraw()

    def _set_edit_mode(self, mode: str) -> None:
        self._edit_mode = mode
        self._edit_drag_start = None
        self._boundary_drag_start = None
        self._boundary_drag_preview = None
        self._selection_start = None
        for button_mode, button in self._tool_buttons.items():
            button.configure(
                style=(
                    "VisualizerActive.TButton"
                    if button_mode == mode
                    else "Visualizer.TButton"
                )
            )
        messages = {
            "select": "Sélectionnez une annotation ou faites glisser sur la forme d’onde.",
            "erase": "Cliquez sur une annotation pour en effacer le texte.",
            "delete_boundary": "Cliquez sur une barre pour fusionner les intervalles voisins.",
            "add_boundary": "Cliquez dans un intervalle pour ajouter une barre.",
            "move_boundary": "Faites glisser une barre TextGrid ou TRS pour la déplacer.",
            "draw": "Faites glisser sur une ligne pour ajouter une annotation.",
        }
        self._status.set(messages[mode])

    def _update_tool_buttons(self) -> None:
        has_annotations = self.annotation_path is not None and bool(self.tiers)
        for mode, button in self._tool_buttons.items():
            unavailable = mode == "delete_boundary" and (
                self._annotation_format != "textgrid"
                or not any(len(tier.intervals) > 1 for tier in self.tiers)
            )
            unavailable = unavailable or mode == "add_boundary" and (
                self._annotation_format != "textgrid"
                or not any(tier.intervals for tier in self.tiers)
            )
            unavailable = unavailable or mode == "move_boundary" and (
                self._annotation_format not in {"textgrid", "trs"}
                or not any(tier.intervals for tier in self.tiers)
                or self._annotation_format == "textgrid"
                and not any(len(tier.intervals) > 1 for tier in self.tiers)
            )
            button.configure(
                state=(
                    "disabled"
                    if unavailable or mode != "select" and not has_annotations
                    else "normal"
                )
            )

    def _toggle_video_panel(self) -> None:
        if str(self._video_player) in self._content_pane.panes():
            self._video_player.hide()
        else:
            self._set_video_panel_visible(True)

    def _set_video_panel_visible(self, visible: bool) -> None:
        pane = str(self._video_player)
        if visible and pane not in self._content_pane.panes():
            self._content_pane.add(self._video_player, weight=0)
        elif not visible and pane in self._content_pane.panes():
            self._content_pane.forget(self._video_player)

    def _tier_index_at_y(self, y: float) -> int | None:
        tier_top = self.AXIS_HEIGHT + self.WAVEFORM_HEIGHT
        return next(
            (
                index
                for index, (row_top, row_height) in enumerate(self._tier_row_layout())
                if tier_top + row_top <= y < tier_top + row_top + row_height
            ),
            None,
        )

    def _interval_index_at_time(self, tier_index: int, time: float) -> int | None:
        intervals = self.tiers[tier_index].intervals
        return next(
            (
                index
                for index, interval in enumerate(intervals)
                if interval.start <= time < interval.end
                or math.isclose(time, interval.end) and index == len(intervals) - 1
            ),
            None,
        )

    def _boundary_index_at_time(
        self, tier_index: int, time: float, tolerance: float
    ) -> int | None:
        intervals = self.tiers[tier_index].intervals
        if len(intervals) < 2:
            return None
        nearest_index, distance = min(
            (
                (index, abs(interval.start - time))
                for index, interval in enumerate(intervals[1:], start=1)
            ),
            key=lambda item: item[1],
        )
        return nearest_index if distance <= tolerance else None

    def _trs_boundary_at_time(
        self, tier_index: int, time: float, tolerance: float
    ) -> tuple[int, Literal["start", "end"]] | None:
        candidates: list[tuple[float, int, Literal["start", "end"]]] = []
        for interval_index, interval in enumerate(self.tiers[tier_index].intervals):
            candidates.append((abs(interval.start - time), interval_index, "start"))
            candidates.append((abs(interval.end - time), interval_index, "end"))
        if not candidates:
            return None
        distance, interval_index, side = min(candidates, key=lambda item: item[0])
        if distance > tolerance:
            return None
        return interval_index, side

    def _delete_annotation_boundary(
        self, tier_index: int, boundary_index: int
    ) -> None:
        try:
            if self._annotation_format != "textgrid":
                raise ValueError("La suppression de barre est disponible pour les TextGrid.")
            snapshot = self._capture_annotation_edit()
            tier = merge_tier_intervals(self.tiers[tier_index], boundary_index)
            tiers = list(self.tiers)
            tiers[tier_index] = tier
            if self.annotation_path is None or self._textgrid_source is None:
                raise ValueError("Le TextGrid source n’est plus disponible.")
            self._textgrid_source = save_textgrid(
                self.annotation_path, self._textgrid_source, tuple(tiers)
            )
            self.tiers = tuple(tiers)
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Suppression de barre impossible : {error}")
            return
        self.selection = None
        self._status.set("Barre supprimée ; les intervalles voisins sont fusionnés.")
        self._update_tool_buttons()
        self._redraw()

    def _move_annotation_boundary(
        self, tier_index: int, boundary_index: int, time: float
    ) -> None:
        try:
            if self._annotation_format != "textgrid":
                raise ValueError(
                    "Le déplacement de barre est disponible pour les TextGrid."
                )
            if self.annotation_path is None or self._textgrid_source is None:
                raise ValueError("Le TextGrid source n’est plus disponible.")
            previous_tier = self.tiers[tier_index]
            tier = move_tier_boundary(previous_tier, boundary_index, time)
            if tier == previous_tier:
                self._status.set("La barre est déjà à cette position.")
                self._redraw()
                return
            snapshot = self._capture_annotation_edit()
            tiers = list(self.tiers)
            tiers[tier_index] = tier
            self._textgrid_source = save_textgrid(
                self.annotation_path, self._textgrid_source, tuple(tiers)
            )
            self.tiers = tuple(tiers)
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Déplacement de barre impossible : {error}")
            self._redraw()
            return
        self.selection = None
        self._status.set("Barre déplacée ; les textes des intervalles sont conservés.")
        self._update_tool_buttons()
        self._redraw()

    def _move_trs_annotation_boundary(
        self,
        tier_index: int,
        interval_index: int,
        side: Literal["start", "end"],
        time: float,
    ) -> None:
        try:
            if self._annotation_format != "trs" or self.annotation_path is None:
                raise ValueError("Le fichier TRS source n’est plus disponible.")
            interval = self.tiers[tier_index].intervals[interval_index]
            old_time = interval.start if side == "start" else interval.end
            if math.isclose(time, old_time, abs_tol=1e-6):
                self._status.set("La barre est déjà à cette position.")
                self._redraw()
                return
            snapshot = self._capture_annotation_edit()
            move_trs_boundary(self.annotation_path, interval, side, time)
            self.tiers = parse_trs(self.annotation_path)
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Déplacement de barre impossible : {error}")
            self._redraw()
            return
        self.selection = None
        self._status.set("Barre TRS déplacée ; les textes des segments sont conservés.")
        self._update_tool_buttons()
        self._redraw()

    def _add_annotation_boundary(self, tier_index: int, time: float) -> None:
        try:
            if self._annotation_format != "textgrid":
                raise ValueError("L’ajout de barre est disponible pour les TextGrid.")
            snapshot = self._capture_annotation_edit()
            tier = split_tier_interval(self.tiers[tier_index], time)
            tiers = list(self.tiers)
            tiers[tier_index] = tier
            if self.annotation_path is None or self._textgrid_source is None:
                raise ValueError("Le TextGrid source n’est plus disponible.")
            self._textgrid_source = save_textgrid(
                self.annotation_path, self._textgrid_source, tuple(tiers)
            )
            self.tiers = tuple(tiers)
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Ajout de barre impossible : {error}")
            return
        self.selection = None
        self._status.set("Barre ajoutée ; le texte reste dans l’intervalle gauche.")
        self._update_tool_buttons()
        self._redraw()

    def _erase_annotation(self, tier_index: int, interval_index: int) -> None:
        try:
            snapshot = self._capture_annotation_edit()
            if self._annotation_format == "textgrid":
                tier = clear_tier_interval(self.tiers[tier_index], interval_index)
                tiers = list(self.tiers)
                tiers[tier_index] = tier
                if self.annotation_path is None or self._textgrid_source is None:
                    raise ValueError("Le TextGrid source n’est plus disponible.")
                self._textgrid_source = save_textgrid(
                    self.annotation_path, self._textgrid_source, tuple(tiers)
                )
                self.tiers = tuple(tiers)
            elif self._annotation_format == "trs" and self.annotation_path is not None:
                clear_trs_interval(
                    self.annotation_path,
                    self.tiers[tier_index].intervals[interval_index],
                )
                self.tiers = parse_trs(self.annotation_path)
                self._visible_tier_indices = set(range(len(self.tiers)))
            else:
                raise ValueError("Aucun fichier d’annotation modifiable n’est ouvert.")
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Effacement impossible : {error}")
            return
        self._status.set("Annotation effacée et enregistrée.")
        self._redraw()

    def _add_annotation(self, tier_index: int, start: float, end: float) -> None:
        text = simpledialog.askstring(
            "Nouvelle annotation",
            f"Texte pour « {self.tiers[tier_index].name} » :",
            parent=self.winfo_toplevel(),
        )
        if text is None:
            self._status.set("Ajout d’annotation annulé.")
            return
        if not text.strip():
            self._status.set("Le texte de l’annotation ne peut pas être vide.")
            return

        try:
            snapshot = self._capture_annotation_edit()
            if self._annotation_format == "textgrid":
                tier = replace_tier_interval_range(
                    self.tiers[tier_index], start, end, text
                )
                tiers = list(self.tiers)
                tiers[tier_index] = tier
                if self.annotation_path is None or self._textgrid_source is None:
                    raise ValueError("Le TextGrid source n’est plus disponible.")
                self._textgrid_source = save_textgrid(
                    self.annotation_path, self._textgrid_source, tuple(tiers)
                )
                self.tiers = tuple(tiers)
            elif (
                self._annotation_format == "trs"
                and self.annotation_path is not None
            ):
                speaker_id = self.tiers[tier_index].source_speaker_id
                if speaker_id is None:
                    raise ValueError("Le locuteur TRS de ce niveau est introuvable.")
                add_trs_turn(
                    self.annotation_path,
                    speaker_id,
                    start,
                    end,
                    text,
                )
                self.tiers = parse_trs(self.annotation_path)
                self._visible_tier_indices = set(range(len(self.tiers)))
            else:
                raise ValueError("Aucun fichier d’annotation modifiable n’est ouvert.")
            self._commit_annotation_edit(snapshot)
        except (OSError, RuntimeError, ValueError) as error:
            self._status.set(f"Ajout impossible : {error}")
            return
        self._status.set("Annotation ajoutée et enregistrée.")
        self._redraw()

    def _draw_spectrogram(self, track_width: int, top: int) -> None:
        canvas = self.canvas
        canvas.create_rectangle(
            0, top, self.LEFT_MARGIN, top + self.SPECTROGRAM_HEIGHT,
            fill="#f3f4f6", outline="#d1d5db",
        )
        canvas.create_text(
            12, top + self.SPECTROGRAM_HEIGHT / 2,
            anchor="w", text="Spectrogramme", fill="#4b5563",
        )
        canvas.create_rectangle(
            self.LEFT_MARGIN, top, self.LEFT_MARGIN + track_width,
            top + self.SPECTROGRAM_HEIGHT, fill="#f7f7f7", outline="#cbd5e1",
        )
        if self.samples is None or track_width <= 0:
            self._spectrogram_requested_key = None
            return
        image_width = track_width
        image_height = self.SPECTROGRAM_HEIGHT
        normalize_visible_area = (
            self.spectrogram_settings.normalize_visible_area
            and not self.spectrogram_settings.normalize_columns
        )
        visible_area = (
            self._visible_spectrogram_area(image_width)
            if normalize_visible_area
            else None
        )
        self._autoscale_visible_area = visible_area
        key = (
            id(self.samples),
            image_width,
            image_height,
            self.sample_rate,
            self.spectrogram_settings,
            visible_area,
        )
        self._spectrogram_requested_key = key
        if self._spectrogram_key == key and self._spectrogram_image is not None:
            canvas.create_image(
                self.LEFT_MARGIN, top,
                anchor="nw", image=self._spectrogram_image,
            )
            return
        if self._spectrogram_pending_key is None:
            samples = self.samples
            sample_rate = self.sample_rate
            settings = self.spectrogram_settings
            self._spectrogram_pending_key = key

            def render() -> None:
                try:
                    ppm = spectrogram_ppm(
                        samples,
                        sample_rate,
                        image_width,
                        image_height,
                        settings,
                        visible_area,
                    )
                except ValueError as error:
                    self._events.put(("spectrogram_error", (key, str(error))))
                    return
                self._events.put(("spectrogram", (key, ppm)))

            threading.Thread(target=render, daemon=True).start()

    def _visible_spectrogram_area(self, track_width: int) -> tuple[int, int]:
        canvas_width = max(1, self.canvas.winfo_width())
        visible_left = self.canvas.canvasx(0) - self.LEFT_MARGIN
        visible_right = self.canvas.canvasx(canvas_width) - self.LEFT_MARGIN
        start = min(max(math.floor(visible_left), 0), track_width - 1)
        end = min(max(math.ceil(visible_right), start + 1), track_width)
        return start, end

    def _on_horizontal_view_changed(self, first: str, last: str) -> None:
        self.horizontal_scroll.set(first, last)
        settings = self.spectrogram_settings
        if not settings.normalize_visible_area or settings.normalize_columns:
            self._autoscale_visible_area = None
            return
        track_width = max(1, int(self.duration * self.scale))
        visible_area = self._visible_spectrogram_area(track_width)
        if visible_area == self._autoscale_visible_area:
            return
        self._autoscale_visible_area = visible_area
        if not self._autoscale_redraw_pending:
            self._autoscale_redraw_pending = True
            self.after_idle(self._redraw_autoscaled_spectrogram)

    def _redraw_autoscaled_spectrogram(self) -> None:
        self._autoscale_redraw_pending = False
        if (
            self.spectrogram_settings.normalize_visible_area
            and not self.spectrogram_settings.normalize_columns
        ):
            self._redraw()

    def _open_spectrogram_options(self) -> None:
        settings = self.spectrogram_settings
        dialog = tk.Toplevel(self)
        dialog.title("Options du spectrogramme")
        dialog.transient(self.winfo_toplevel())
        dialog.resizable(False, False)
        dialog.grab_set()

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        variables: dict[str, tk.Variable] = {
            "colour_map": tk.StringVar(value=settings.colour_map),
            "colour_scale": tk.StringVar(value=settings.colour_scale),
            "window_size": tk.StringVar(value=str(settings.window_size)),
            "window_hop_level": tk.StringVar(
                value=SPECTROGRAM_WINDOW_OVERLAPS[settings.window_hop_level]
            ),
            "bin_display": tk.StringVar(value=settings.bin_display),
            "frequency_scale": tk.StringVar(value=settings.frequency_scale),
            "normalize_columns": tk.BooleanVar(value=settings.normalize_columns),
            "normalize_visible_area": tk.BooleanVar(
                value=settings.normalize_visible_area
            ),
            "grayscale": tk.BooleanVar(value=settings.grayscale),
            "gain_db": tk.IntVar(value=settings.gain_db),
            "threshold_db": tk.IntVar(value=settings.threshold_db),
            "colour_rotation": tk.IntVar(value=settings.colour_rotation),
        }

        def add_group(title: str, row: int) -> ttk.LabelFrame:
            group = ttk.LabelFrame(body, text=title, padding=8)
            group.grid(row=row, column=0, sticky="ew", pady=3)
            group.columnconfigure(1, weight=1)
            return group

        def add_choice(
            group: ttk.Widget,
            row: int,
            label: str,
            name: str,
            values: tuple[str, ...],
        ) -> None:
            ttk.Label(group, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=3
            )
            combobox = ttk.Combobox(
                group,
                textvariable=variables[name],
                values=values,
                state="readonly",
                width=20,
            )
            combobox.grid(row=row, column=1, sticky="ew", pady=3)
            combobox.bind("<<ComboboxSelected>>", lambda _event: apply_settings())

        def add_number(
            group: ttk.Widget,
            row: int,
            label: str,
            name: str,
            minimum: int,
            maximum: int,
        ) -> None:
            ttk.Label(group, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=3
            )
            spinbox = ttk.Spinbox(
                group,
                from_=minimum,
                to=maximum,
                textvariable=variables[name],
                width=8,
                command=lambda: apply_settings(),
            )
            spinbox.grid(row=row, column=1, sticky="w", pady=3)
            spinbox.bind("<Return>", lambda _event: apply_settings())
            spinbox.bind("<FocusOut>", lambda _event: apply_settings())

        colour_group = add_group("Couleur", 0)
        add_choice(colour_group, 0, "Palette", "colour_map", SPECTROGRAM_COLOUR_MAPS)
        add_number(colour_group, 1, "Rotation", "colour_rotation", 0, 254)

        scale_group = add_group("Échelle", 1)
        add_choice(scale_group, 0, "Type", "colour_scale", SPECTROGRAM_COLOUR_SCALES)
        add_number(scale_group, 1, "Gain (dB)", "gain_db", -60, 60)
        add_number(scale_group, 2, "Seuil (dB)", "threshold_db", -120, 0)

        window_group = add_group("Fenêtre", 2)
        add_choice(
            window_group,
            0,
            "Taille",
            "window_size",
            tuple(map(str, SPECTROGRAM_WINDOW_SIZES)),
        )
        add_choice(
            window_group,
            1,
            "Recouvrement",
            "window_hop_level",
            SPECTROGRAM_WINDOW_OVERLAPS,
        )

        bins_group = add_group("Bins", 3)
        add_choice(
            bins_group, 0, "Affichage", "bin_display", SPECTROGRAM_BIN_DISPLAYS
        )
        add_choice(
            bins_group,
            1,
            "Échelle fréquentielle",
            "frequency_scale",
            SPECTROGRAM_FREQUENCY_SCALES,
        )

        display_group = add_group("Affichage", 4)
        ttk.Checkbutton(
            display_group,
            text="Normalised",
            variable=variables["normalize_columns"],
            command=lambda: apply_settings(),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(
            display_group,
            text="Autoscaling",
            variable=variables["normalize_visible_area"],
            command=lambda: apply_settings(),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(
            display_group,
            text="Niveau de gris",
            variable=variables["grayscale"],
            command=lambda: apply_settings(),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        def apply_settings() -> None:
            try:
                updated = replace(
                    settings,
                    colour_map=str(variables["colour_map"].get()),
                    colour_scale=str(variables["colour_scale"].get()),
                    window_size=int(str(variables["window_size"].get())),
                    window_hop_level=SPECTROGRAM_WINDOW_OVERLAPS.index(
                        str(variables["window_hop_level"].get())
                    ),
                    bin_display=str(variables["bin_display"].get()),
                    frequency_scale=str(variables["frequency_scale"].get()),
                    normalize_columns=bool(variables["normalize_columns"].get()),
                    normalize_visible_area=bool(
                        variables["normalize_visible_area"].get()
                    ),
                    grayscale=bool(variables["grayscale"].get()),
                    gain_db=int(str(variables["gain_db"].get())),
                    threshold_db=int(str(variables["threshold_db"].get())),
                    colour_rotation=int(str(variables["colour_rotation"].get())),
                )
            except (ValueError, tk.TclError) as error:
                self._status.set(f"Options du spectrogramme invalides : {error}")
                return
            self.spectrogram_settings = updated
            self._spectrogram_key = None
            self._redraw()

        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Fermer", command=dialog.destroy).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    def _draw_cursor(self, content_height: int) -> None:
        x = self._x_for_time(min(max(self.cursor, 0), self.duration))
        self.canvas.create_line(
            x, 0, x, content_height,
            fill="#e34b4b", width=2, tags=("playback-cursor",),
        )

    def _draw_grid(self, track_width: int) -> None:
        step = _time_step(self.duration, track_width)
        tick = step
        while tick < self.duration:
            x = self._x_for_time(tick)
            self.canvas.create_line(
                x,
                self.AXIS_HEIGHT,
                x,
                self.AXIS_HEIGHT + self.WAVEFORM_HEIGHT,
                fill="#e8ebf0",
            )
            tick += step

    def _event_time(self, event: tk.Event) -> float:
        x = self.canvas.canvasx(event.x)
        return min(max((x - self.LEFT_MARGIN) / self.scale, 0.0), self.duration)

    def _begin_selection(self, event: tk.Event) -> None:
        if self.samples is None or self.canvas.canvasx(event.x) < self.LEFT_MARGIN:
            return
        self.canvas.focus_set()
        self._selected_annotation = None
        self._selection_start = self._event_time(event)
        self.selection = (self._selection_start, self._selection_start)
        self.cursor = self._selection_start
        self._video_player.set_timeline_position(self.cursor, force=True)
        self._redraw()

    def _drag_selection(self, event: tk.Event) -> None:
        if self._boundary_drag_start is not None:
            self._boundary_drag_preview = self._event_time(event)
            self._status.set(
                f"Nouvelle position : {self._boundary_drag_preview:.3f} s"
            )
            self._redraw()
            return
        if self._edit_drag_start is not None:
            _tier_index, start = self._edit_drag_start
            end = self._event_time(event)
            self.selection = (min(start, end), max(start, end))
            self._status.set(
                f"Nouvelle plage : {self.selection[0]:.3f}–{self.selection[1]:.3f} s"
            )
            self._redraw()
            return
        if self._selection_start is None:
            return
        start = self._selection_start
        end = self._event_time(event)
        selection = (min(start, end), max(start, end))
        self.selection = selection
        self._status.set(
            f"Sélection : {selection[0]:.3f} – {selection[1]:.3f} s"
        )
        self._redraw()

    def _end_selection(self, event: tk.Event) -> None:
        if self._boundary_drag_start is not None:
            drag = self._boundary_drag_start
            target_time = self._event_time(event)
            self._boundary_drag_start = None
            self._boundary_drag_preview = None
            if drag.side == "internal":
                self._move_annotation_boundary(
                    drag.tier_index, drag.interval_index, target_time
                )
            else:
                self._move_trs_annotation_boundary(
                    drag.tier_index,
                    drag.interval_index,
                    drag.side,
                    target_time,
                )
            return
        if self._edit_drag_start is not None:
            tier_index, start = self._edit_drag_start
            end = self._event_time(event)
            self._edit_drag_start = None
            if abs(end - start) < 0.025:
                self.selection = None
                self._status.set("La plage sélectionnée est trop courte.")
                self._redraw()
                return
            self._add_annotation(tier_index, min(start, end), max(start, end))
            return
        if self._selection_start is None:
            return
        end = self._event_time(event)
        start = self._selection_start
        self._selection_start = None
        if abs(end - start) < 0.025:
            self.cursor = end
            self._video_player.set_timeline_position(self.cursor, force=True)
            self.selection = None
            self._status.set(f"Curseur : {end:.3f} s")
        else:
            selection = (min(start, end), max(start, end))
            self.selection = selection
            self._status.set(
                f"Sélection : {selection[0]:.3f} – {selection[1]:.3f} s"
                " · Entrée pour zoomer"
            )
        self._redraw()


def _time_step(duration: float, width: int) -> float:
    desired = duration * 75 / max(width, 1)
    power = 10 ** math.floor(math.log10(max(desired, 1e-9)))
    for factor in (1, 2, 5, 10):
        step = factor * power
        if step >= desired:
            return step
    return 10 * power


def _distance_to_interval(time: float, interval: Interval) -> float:
    if time < interval.start:
        return interval.start - time
    if time > interval.end:
        return time - interval.end
    return 0.0

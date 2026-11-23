from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .visualization import Interval, TextGridTier


DEFAULT_VOWEL_PHONES = "i y u e ø o ɛ œ ɔ a ɑ ə ɐ ɪ ʊ ɛ̃ œ̃ ɑ̃ ɔ̃"


class CVSyllableSegmentationError(ValueError):
    """Raised when the phone/token tiers cannot be syllabified."""


@dataclass(frozen=True)
class CVSyllabificationResult:
    cv_tier: TextGridTier
    syllable_tier: TextGridTier
    tokens_without_nucleus: int


@dataclass(frozen=True)
class _PhoneSlice:
    text: str
    start: float
    end: float
    is_vowel: bool


def segment_cv_and_syllables(
    tiers: tuple[TextGridTier, ...],
    *,
    phone_level: str = "phone",
    token_level: str = "token",
    cv_level: str = "CV",
    syllable_level: str = "syll",
    vowel_phones: str = DEFAULT_VOWEL_PHONES,
    maximum_onset_consonants: int = 2,
) -> CVSyllabificationResult:
    """Build C/V runs and maximal-onset syllables from phone and token tiers."""
    phone_level = phone_level.strip()
    token_level = token_level.strip()
    cv_level = cv_level.strip()
    syllable_level = syllable_level.strip()
    if not phone_level or not token_level:
        raise CVSyllableSegmentationError("Indiquez les niveaux d’entrée phone et token.")
    if (
        not cv_level
        or not syllable_level
        or cv_level == syllable_level
        or cv_level in {phone_level, token_level}
        or syllable_level in {phone_level, token_level}
    ):
        raise CVSyllableSegmentationError(
            "Les niveaux de sortie doivent être distincts des niveaux d’entrée "
            "et l’un de l’autre."
        )

    phone_tier = _find_input_tier(tiers, phone_level, "phone")
    token_tier = _find_input_tier(tiers, token_level, "token")
    if phone_tier is None or token_tier is None:
        missing = [
            name
            for name, tier in ((phone_level, phone_tier), (token_level, token_tier))
            if tier is None
        ]
        available = ", ".join(tier.name for tier in tiers) or "(aucun)"
        raise CVSyllableSegmentationError(
            f"Niveau(x) d’entrée absent(s) du TextGrid : {', '.join(missing)}. "
            f"Niveaux disponibles : {available}."
        )

    tier_start, tier_end = _tier_bounds(phone_tier)
    vowels = {
        _normalize_phone(phone)
        for phone in vowel_phones.split()
        if _normalize_phone(phone)
    }
    cv_intervals = _build_cv_intervals(phone_tier, vowels)
    syllable_intervals, tokens_without_nucleus = _build_syllables(
        phone_tier,
        token_tier,
        vowels,
        min(3, max(1, maximum_onset_consonants)),
    )
    return CVSyllabificationResult(
        TextGridTier(
            cv_level,
            tuple(cv_intervals),
            source_start=tier_start,
            source_end=tier_end,
        ),
        TextGridTier(
            syllable_level,
            tuple(syllable_intervals),
            source_start=tier_start,
            source_end=tier_end,
        ),
        tokens_without_nucleus,
    )


def _find_input_tier(
    tiers: tuple[TextGridTier, ...], requested: str, kind: str
) -> TextGridTier | None:
    normalized = requested.casefold()
    if kind == "phone" and normalized in {"phone", "phones"}:
        candidates = ("phone", "phones")
    elif kind == "token" and normalized in {"token", "tokens", "word", "words"}:
        candidates = ("token", "tokens", "word", "words")
    else:
        candidates = (normalized,)
    return next(
        (
            tier
            for candidate in candidates
            for tier in tiers
            if tier.name.casefold() == candidate
        ),
        None,
    )


def _normalize_phone(phone: str) -> str:
    return unicodedata.normalize("NFC", phone.strip())


def _is_pause_phone(phone: str) -> bool:
    label = _normalize_phone(phone)
    lower_label = label.casefold()
    return (
        not label
        or label in {"_", "-", "*", "#"}
        or lower_label in {"sil", "sp", "spn", "pau"}
        or (label.startswith("[") and label.endswith("]"))
    )


def _tier_bounds(tier: TextGridTier) -> tuple[float, float]:
    starts = [interval.start for interval in tier.intervals]
    ends = [interval.end for interval in tier.intervals]
    start = tier.source_start if tier.source_start is not None else min(starts, default=None)
    end = tier.source_end if tier.source_end is not None else max(ends, default=None)
    if start is None or end is None or end < start:
        raise CVSyllableSegmentationError(
            f"Les limites temporelles du niveau « {tier.name} » sont indisponibles."
        )
    return start, end


def _build_cv_intervals(
    phone_tier: TextGridTier, vowels: set[str]
) -> list[Interval]:
    intervals: list[Interval] = []
    run_label: str | None = None
    run_start = 0.0
    run_end = 0.0
    for phone in phone_tier.intervals:
        label = ""
        if not _is_pause_phone(phone.text):
            label = "V" if _normalize_phone(phone.text) in vowels else "C"
        if (
            run_label is not None
            and label == run_label
            and phone.start <= run_end
        ):
            run_end = max(run_end, phone.end)
            continue
        if run_label is not None:
            intervals.append(Interval(run_start, run_end, run_label))
        run_start = phone.start
        run_end = phone.end
        run_label = label
    if run_label is not None:
        intervals.append(Interval(run_start, run_end, run_label))
    return intervals


def _build_syllables(
    phone_tier: TextGridTier,
    token_tier: TextGridTier,
    vowels: set[str],
    maximum_onset_consonants: int,
) -> tuple[list[Interval], int]:
    syllables: list[Interval] = []
    tokens_without_nucleus = 0
    for token in token_tier.intervals:
        if not token.text.strip():
            continue
        word_phones: list[_PhoneSlice] = []

        def flush_word_phones() -> None:
            nonlocal tokens_without_nucleus
            if not word_phones:
                return
            count_before = len(syllables)
            syllables.extend(
                _append_syllables(word_phones, maximum_onset_consonants)
            )
            if len(syllables) == count_before:
                tokens_without_nucleus += 1
            word_phones.clear()

        for phone in phone_tier.intervals:
            center = (phone.start + phone.end) / 2
            if center < token.start or center >= token.end:
                continue
            if _is_pause_phone(phone.text):
                flush_word_phones()
                continue
            start = max(token.start, phone.start)
            end = min(token.end, phone.end)
            if start >= end:
                continue
            label = _normalize_phone(phone.text)
            word_phones.append(
                _PhoneSlice(label, start, end, label in vowels)
            )
        flush_word_phones()
    return syllables, tokens_without_nucleus


def _append_syllables(
    phones: list[_PhoneSlice], maximum_onset_consonants: int
) -> list[Interval]:
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(phones):
        if not phones[index].is_vowel:
            index += 1
            continue
        start = index
        while index + 1 < len(phones) and phones[index + 1].is_vowel:
            index += 1
        nuclei.append((start, index))
        index += 1
    if not nuclei:
        return []

    syllables: list[Interval] = []
    syllable_start = 0
    for nucleus_index, nucleus in enumerate(nuclei):
        syllable_end = len(phones) - 1
        if nucleus_index + 1 < len(nuclei):
            next_nucleus = nuclei[nucleus_index + 1]
            cluster_start = nucleus[1] + 1
            cluster_end = next_nucleus[0] - 1
            cluster_length = max(0, cluster_end - cluster_start + 1)
            onset_length = min(maximum_onset_consonants, cluster_length)
            next_syllable_start = next_nucleus[0] - onset_length
            syllable_end = next_syllable_start - 1

        if syllable_start <= syllable_end:
            syllable_phones = phones[syllable_start : syllable_end + 1]
            label = "".join(phone.text for phone in syllable_phones)
            if label:
                syllables.append(
                    Interval(
                        syllable_phones[0].start,
                        syllable_phones[-1].end,
                        label,
                    )
                )
        if nucleus_index + 1 < len(nuclei):
            next_nucleus = nuclei[nucleus_index + 1]
            cluster_start = nucleus[1] + 1
            cluster_end = next_nucleus[0] - 1
            cluster_length = max(0, cluster_end - cluster_start + 1)
            syllable_start = next_nucleus[0] - min(
                maximum_onset_consonants, cluster_length
            )
    return syllables

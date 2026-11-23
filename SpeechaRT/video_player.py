from __future__ import annotations

import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

import numpy as np


_MAX_FRAME_SIZE = (480, 270)


@dataclass(frozen=True)
class _VideoOpened:
    generation: int
    duration: float
    frame_rate: float


@dataclass(frozen=True)
class _VideoFrame:
    generation: int
    duration: float
    frame_rate: float
    position: float
    pixels: np.ndarray


@dataclass(frozen=True)
class _VideoFailure:
    generation: int
    message: str


_VideoEvent = _VideoOpened | _VideoFrame | _VideoFailure


def video_associations_path() -> Path:
    if sys.platform == "darwin":
        settings_directory = Path.home() / "Library" / "Application Support" / "Praaline"
    elif os.name == "nt":
        settings_directory = Path(
            os.environ.get("APPDATA", str(Path.home()))
        ) / "Praaline"
    else:
        settings_directory = Path(
            os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        ) / "praaline"
    return settings_directory / "video-associations.json"


def read_video_associations(path: str | Path) -> dict[str, str]:
    associations_path = Path(path).expanduser()
    try:
        data = json.loads(associations_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Impossible de lire les associations vidéo : {error}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Le fichier des associations vidéo a un format invalide.")
    videos = data.get("videos")
    if not isinstance(videos, dict) or any(
        not isinstance(audio, str) or not isinstance(video, str)
        for audio, video in videos.items()
    ):
        raise ValueError("Le fichier des associations vidéo a un format invalide.")
    return videos


def write_video_associations(
    path: str | Path, associations: dict[str, str]
) -> None:
    associations_path = Path(path).expanduser()
    temporary_path: str | None = None
    try:
        associations_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=associations_path.parent,
            prefix=f".{associations_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(
                {"version": 1, "videos": associations},
                temporary,
                ensure_ascii=False,
                indent=2,
            )
            temporary.write("\n")
        os.replace(temporary_path, associations_path)
    except OSError as error:
        raise ValueError(
            f"Impossible d’enregistrer les associations vidéo : {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


class VideoPlayer(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_visibility_changed: Callable[[bool], None],
        associations_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent, padding=8)
        self._on_visibility_changed = on_visibility_changed
        self._associations_path = (
            Path(associations_path).expanduser()
            if associations_path is not None
            else video_associations_path()
        )
        self._audio_path: Path | None = None
        self._video_path: Path | None = None
        self._generation = 0
        self._duration = 0.0
        self._frame_rate = 0.0
        self._position = 0.0
        self._last_requested_position: float | None = None
        self._association_warning: str | None = None
        self._audio_playing = False
        self._manual_playing = False
        self._last_manual_tick = 0.0
        self._updating_controls = False
        self._photo: tk.PhotoImage | None = None
        self._poll_id: str | None = None
        self._lock = threading.Lock()
        self._wake_worker = threading.Event()
        self._stop_worker = threading.Event()
        self._pending_open: tuple[int, Path | None] | None = None
        self._pending_position: tuple[int, float] | None = None
        self._events: queue.Queue[_VideoEvent] = queue.Queue(maxsize=4)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        heading = ttk.Frame(self)
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="Vidéo", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.close_button = ttk.Button(
            heading, text="×", width=3, command=self.hide
        )
        self.close_button.grid(row=0, column=1, sticky="e")
        self.preview = tk.Label(
            self,
            background="#111111",
            foreground="#ffffff",
            text="Aucune vidéo sélectionnée",
        )
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew")
        self.open_button = ttk.Button(
            controls, text="Ouvrir une vidéo…", command=self.open_video_dialog
        )
        self.open_button.pack(side="left")
        self.play_button = ttk.Button(
            controls,
            text="Lire",
            command=self._toggle_manual_playback,
            state="disabled",
        )
        self.play_button.pack(side="left", padx=(5, 0))
        self.stop_button = ttk.Button(
            controls, text="Arrêter", command=self._stop_manual_playback, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(5, 0))
        self.time_label = ttk.Label(controls, text="00:00 / 00:00")
        self.time_label.pack(side="right", padx=(6, 0))
        self.position_scale = ttk.Scale(
            self,
            from_=0.0,
            to=0.0,
            orient="horizontal",
            command=self._on_scale_changed,
            state="disabled",
        )
        self.position_scale.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.status = tk.StringVar(value="Aucune vidéo sélectionnée.")
        ttk.Label(self, textvariable=self.status, anchor="w", wraplength=420).grid(
            row=4, column=0, sticky="ew", pady=(6, 0)
        )

        self._worker = threading.Thread(
            target=self._video_worker, name="Praaline video decoder", daemon=True
        )
        self._worker.start()
        self._poll_id = self.after(30, self._poll_video)

    @property
    def video_path(self) -> Path | None:
        return self._video_path

    def set_audio_file(self, path: str | Path) -> None:
        self._audio_path = Path(path).expanduser().resolve()
        self._audio_playing = False
        self._manual_playing = False
        try:
            associations = read_video_associations(self._associations_path)
        except ValueError as error:
            self.set_video_file(None)
            self.status.set(str(error))
            self._on_visibility_changed(True)
            return

        associated_video = associations.get(str(self._audio_path))
        if associated_video is None:
            self.set_video_file(None)
            return
        self.set_video_file(associated_video, remember=False)

    def open_video_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Ouvrir une vidéo",
            filetypes=(
                (
                    "Fichiers vidéo",
                    "*.mp4 *.mov *.m4v *.avi *.mkv *.webm",
                ),
                ("Tous les fichiers", "*"),
            ),
        )
        if filename:
            self.set_video_file(filename)

    def set_video_file(
        self, path: str | Path | None, *, remember: bool = True
    ) -> None:
        self._generation += 1
        generation = self._generation
        video_path = Path(path).expanduser().resolve() if path is not None else None
        self._video_path = video_path
        self._duration = 0.0
        self._frame_rate = 0.0
        self._position = 0.0
        self._last_requested_position = None
        self._association_warning = None
        self._manual_playing = False
        self._photo = None
        self.preview.configure(image="", text="Chargement de la vidéo…")
        self._update_position_controls()

        if video_path is None:
            self.status.set("Aucune vidéo associée à cet enregistrement.")
            with self._lock:
                self._pending_open = (generation, None)
                self._pending_position = None
            self._wake_worker.set()
            self._on_visibility_changed(False)
            return
        if not video_path.is_file():
            self.status.set(f"Fichier vidéo introuvable : {video_path}")
            with self._lock:
                self._pending_open = (generation, None)
                self._pending_position = None
            self._wake_worker.set()
            self._on_visibility_changed(True)
            return

        if remember and self._audio_path is not None:
            try:
                associations = read_video_associations(self._associations_path)
                associations[str(self._audio_path)] = str(video_path)
                write_video_associations(self._associations_path, associations)
            except ValueError as error:
                self._association_warning = str(error)
        if self._association_warning is not None:
            self.status.set(
                f"Chargement de {video_path.name} ; association non mémorisée."
            )
        else:
            self.status.set(f"Chargement de {video_path.name}…")

        with self._lock:
            self._pending_open = (generation, video_path)
            self._pending_position = (generation, 0.0)
        self._wake_worker.set()
        self._on_visibility_changed(True)

    def set_timeline_position(self, position: float, *, force: bool = False) -> None:
        if not math.isfinite(position):
            return
        self._position = max(0.0, position)
        if self._duration > 0:
            self._position = min(self._position, self._duration)
        self._update_position_controls()
        if self._video_path is None:
            return
        frame_interval = 1.0 / max(self._frame_rate, 1.0)
        if (
            not force
            and self._last_requested_position is not None
            and abs(self._position - self._last_requested_position)
            < frame_interval * 0.75
        ):
            return
        self._request_frame(self._position)

    def set_audio_playing(self, playing: bool) -> None:
        self._audio_playing = playing
        if playing:
            self._manual_playing = False
        self._update_position_controls()

    def hide(self) -> None:
        self._on_visibility_changed(False)

    def shutdown(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        self._stop_worker.set()
        self._wake_worker.set()
        if self._worker.is_alive():
            self._worker.join(timeout=0.5)

    def _request_frame(self, position: float) -> None:
        self._last_requested_position = position
        with self._lock:
            self._pending_position = (self._generation, position)
        self._wake_worker.set()

    def _video_worker(self) -> None:
        capture = None
        cv2_module = None
        current_generation = -1
        frame_rate = 0.0
        duration = 0.0
        while not self._stop_worker.is_set():
            self._wake_worker.wait()
            self._wake_worker.clear()
            if self._stop_worker.is_set():
                break
            with self._lock:
                open_request = self._pending_open
                self._pending_open = None
                position_request = self._pending_position
                self._pending_position = None

            if open_request is not None:
                if capture is not None:
                    capture.release()
                    capture = None
                current_generation, video_path = open_request
                if video_path is None:
                    continue
                try:
                    import cv2 as cv2_module
                except ImportError as error:
                    self._publish(
                        _VideoFailure(
                            current_generation,
                            "Lecture vidéo indisponible : installez opencv-python-headless.",
                        )
                    )
                    continue
                try:
                    capture = cv2_module.VideoCapture(str(video_path))
                    if not capture.isOpened():
                        capture.release()
                        capture = None
                        self._publish(
                            _VideoFailure(
                                current_generation,
                                f"Impossible de décoder la vidéo : {video_path.name}",
                            )
                        )
                        continue
                    frame_rate = float(capture.get(cv2_module.CAP_PROP_FPS))
                    frame_count = float(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
                    duration = (
                        frame_count / frame_rate
                        if frame_rate > 0 and frame_count > 0
                        else 0.0
                    )
                    self._publish(
                        _VideoOpened(current_generation, duration, frame_rate)
                    )
                    if position_request is None or position_request[0] != current_generation:
                        position_request = (current_generation, 0.0)
                except (OSError, RuntimeError, ValueError, cv2_module.error) as error:
                    if capture is not None:
                        capture.release()
                        capture = None
                    self._publish(_VideoFailure(current_generation, str(error)))
                    continue

            if (
                position_request is None
                or position_request[0] != current_generation
                or capture is None
                or cv2_module is None
            ):
                continue
            position = max(0.0, position_request[1])
            if duration > 0:
                position = min(position, duration)
            try:
                capture.set(cv2_module.CAP_PROP_POS_MSEC, position * 1000)
                success, frame = capture.read()
                if not success or frame is None:
                    self._publish(
                        _VideoFailure(
                            current_generation,
                            f"Impossible de lire l’image à {position:.2f} s.",
                        )
                    )
                    continue
                height, width = frame.shape[:2]
                scale = min(
                    _MAX_FRAME_SIZE[0] / width,
                    _MAX_FRAME_SIZE[1] / height,
                    1.0,
                )
                if scale < 1:
                    frame = cv2_module.resize(
                        frame,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                    )
                pixels = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
                self._publish(
                    _VideoFrame(
                        current_generation,
                        duration,
                        frame_rate,
                        position,
                        pixels,
                    )
                )
            except (OSError, RuntimeError, ValueError, cv2_module.error) as error:
                self._publish(_VideoFailure(current_generation, str(error)))

        if capture is not None:
            capture.release()

    def _publish(self, event: _VideoEvent) -> None:
        while not self._stop_worker.is_set():
            try:
                self._events.put_nowait(event)
                return
            except queue.Full:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    return

    def _poll_video(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event.generation != self._generation:
                continue
            if isinstance(event, _VideoFailure):
                self.status.set(event.message)
                continue
            self._duration = event.duration
            self._frame_rate = event.frame_rate
            if isinstance(event, _VideoOpened):
                self.status.set(self._ready_status())
                self._update_position_controls()
                continue
            height, width = event.pixels.shape[:2]
            ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + event.pixels.tobytes()
            try:
                self._photo = tk.PhotoImage(
                    data=ppm.decode("latin-1"), format="PPM"
                )
                self.preview.configure(image=self._photo, text="")
                self.status.set(self._ready_status())
            except tk.TclError as error:
                self.status.set(f"Affichage vidéo impossible : {error}")
            self._update_position_controls()

        if self._manual_playing and not self._audio_playing:
            now = time.monotonic()
            elapsed = max(0.0, now - self._last_manual_tick)
            self._last_manual_tick = now
            self.set_timeline_position(self._position + elapsed)
            if self._duration > 0 and self._position >= self._duration:
                self._manual_playing = False
                self._update_position_controls()
        self._poll_id = self.after(30, self._poll_video)

    def _ready_status(self) -> str:
        if self._association_warning is not None:
            return f"Vidéo prête ; association non mémorisée : {self._association_warning}"
        return "Vidéo prête."

    def _toggle_manual_playback(self) -> None:
        if self._video_path is None or self._audio_playing:
            return
        if self._manual_playing:
            self._manual_playing = False
        elif self._duration > 0:
            if self._position >= self._duration:
                self.set_timeline_position(0.0, force=True)
            self._manual_playing = True
            self._last_manual_tick = time.monotonic()
        else:
            self.status.set("La vidéo n’est pas encore prête à être lue.")
        self._update_position_controls()

    def _stop_manual_playback(self) -> None:
        self._manual_playing = False
        self.set_timeline_position(0.0, force=True)
        self._update_position_controls()

    def _on_scale_changed(self, value: str) -> None:
        if self._updating_controls:
            return
        self.set_timeline_position(float(value), force=True)

    def _update_position_controls(self) -> None:
        self._updating_controls = True
        self.position_scale.configure(to=max(self._duration, 0.0))
        self.position_scale.set(self._position)
        self._updating_controls = False
        self.time_label.configure(
            text=f"{self._format_time(self._position)} / "
            f"{self._format_time(self._duration)}"
        )
        has_video = self._video_path is not None
        self.play_button.configure(
            text="Pause" if self._manual_playing else "Lire",
            state=(
                "disabled"
                if not has_video or self._audio_playing or self._duration <= 0
                else "normal"
            ),
        )
        self.stop_button.configure(
            state="normal"
            if has_video and self._manual_playing and not self._audio_playing
            else "disabled"
        )
        self.position_scale.configure(
            state="disabled"
            if not has_video or self._audio_playing or self._duration <= 0
            else "normal"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        whole_seconds = max(0, int(seconds))
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        if hours:
            return f"{hours:02}:{minutes:02}:{seconds_part:02}"
        return f"{whole_seconds // 60:02}:{seconds_part:02}"

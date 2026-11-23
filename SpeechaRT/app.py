from __future__ import annotations

import math
import queue
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .annotation_history import (
    AnnotationHistoryError,
    AnnotationRevision,
    list_annotation_revisions,
    record_annotation_revision,
    restore_annotation_revision,
)
from .corpus import (
    CORPUS_INDEX_FILENAME,
    CorrectionInfo,
    CorpusEntry,
    discover_audio_files,
    load_corpus_catalog,
    load_corpus_corrections,
    make_corpus_entry,
    save_corpus_catalog,
    save_corpus_corrections,
)
from .cv_syllable_segmenter import (
    CVSyllabificationResult,
    DEFAULT_VOWEL_PHONES,
    segment_cv_and_syllables,
)
from .mfa import AlignmentError, align_audio
from .silence_detector import (
    SilenceDetectionError,
    detect_silences,
    write_silence_trs,
)
from .video_player import video_associations_path
from .visualization import (
    AudioVisualizer,
    TextGridTier,
    parse_textgrid_content,
    save_textgrid,
)


def _mfa_executable_settings_path() -> Path:
    return video_associations_path().with_name("mfa-executable.txt")


def _read_mfa_executable(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").rstrip("\r\n")
    except FileNotFoundError:
        return ""
    except OSError as error:
        raise OSError(f"Impossible de lire le chemin MFA enregistré : {error}") from error


def _write_mfa_executable(path: Path, executable: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(executable, encoding="utf-8")
    except OSError as error:
        raise OSError(f"Impossible d’enregistrer le chemin MFA : {error}") from error


class PraalineSimpleApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SpeechaRT")
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False
        self._status = tk.StringVar(value="Ouvrez un fichier WAV pour commencer.")
        mfa_settings_path = _mfa_executable_settings_path()
        try:
            saved_mfa_executable = _read_mfa_executable(mfa_settings_path)
        except OSError as error:
            saved_mfa_executable = ""
            self._status.set(str(error))
        self._saved_mfa_executable = saved_mfa_executable
        self._corpus_entries: dict[str, CorpusEntry] = {}
        self._correction_info: dict[str, CorrectionInfo] = {}
        self._corpus_folder = tk.StringVar(value="Aucun dossier de corpus sélectionné")
        self._corpus_root: Path | None = None
        self._current_entry: CorpusEntry | None = None
        self._alignment_entry: CorpusEntry | None = None
        self._silence_entry: CorpusEntry | None = None
        self._cv_entry: CorpusEntry | None = None
        self._cv_source_text: str | None = None
        self._cv_source_tiers: tuple[TextGridTier, ...] | None = None
        self._cv_result: CVSyllabificationResult | None = None
        self._active_plugin: str | None = None
        self._plugin_audio_key: str | None = None

        self._build_menu()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(6, 4))
        self._build_corpus_tab()
        self._build_plugins_tab()
        self.visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_tab, text="Visualisation")
        self.visualizer = AudioVisualizer(self.visualization_tab)
        self.visualizer.before_annotation_edit = self._archive_before_visualizer_edit
        self.visualizer.pack(fill="both", expand=True, padx=4, pady=4)

        status_bar = ttk.Label(
            root, textvariable=self._status, anchor="w", relief="sunken", padding=(6, 3)
        )
        status_bar.pack(fill="x", side="bottom", padx=8, pady=(0, 6))

        self.root.bind("<Command-o>", lambda _event: self.open_audio())
        self.root.bind("<Control-o>", lambda _event: self.open_audio())
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(480, min(1200, screen_width - 40))
        window_height = max(400, min(800, screen_height - 80))
        self.root.minsize(min(900, window_width), min(650, window_height))
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.after(100, self._poll_messages)

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(
            label="Ouvrir un fichier audio…",
            accelerator="⌘O" if sys.platform == "darwin" else "Ctrl+O",
            command=self.open_audio,
        )
        file_menu.add_command(
            label="Ouvrir un TextGrid…",
            command=self.open_textgrid,
        )
        file_menu.add_command(
            label="Ouvrir un fichier TRS…",
            command=self.open_trs,
        )
        file_menu.add_command(
            label="Fermer les annotations",
            command=self.close_textgrid,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.destroy)
        menu_bar.add_cascade(label="Fichier", menu=file_menu)

        plugins_menu = tk.Menu(menu_bar, tearoff=False)
        plugins_menu.add_command(
            label="Ouvrir les plugins",
            command=self.open_mfa_plugin,
        )
        menu_bar.add_cascade(label="Plugins", menu=plugins_menu)
        self.root.configure(menu=menu_bar)

    def _build_corpus_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.corpus_tab = tab
        self.notebook.add(tab, text="Corpus")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        ttk.Label(tab, textvariable=self._corpus_folder, anchor="w").grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )
        controls = ttk.Frame(tab)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            controls, text="Open Repertory Corpus", command=self.open_corpus_folder
        ).pack(side="left")
        ttk.Button(
            controls, text="Refresh", command=self.refresh_corpus
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls, text="Add Audio Files…", command=self.add_audio_files
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls, text="Remove from List", command=self.remove_corpus_entries
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls, text="Correction Status…", command=self.edit_selected_correction
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls, text="History…", command=self.open_annotation_history
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls, text="Open Selection", command=self.open_selected_corpus_entry
        ).pack(side="right")

        columns = ("audio", "trs", "textgrid", "corrected", "corrected_by")
        self.corpus_tree = ttk.Treeview(
            tab, columns=columns, show="headings", selectmode="extended"
        )
        self.corpus_tree.heading("audio", text="Fichier audio")
        self.corpus_tree.heading("trs", text="Annotation TRS")
        self.corpus_tree.heading("textgrid", text="Annotation TextGrid")
        self.corpus_tree.heading("corrected", text="Correction faite")
        self.corpus_tree.heading("corrected_by", text="Corrigée par")
        self.corpus_tree.column("audio", width=300, anchor="w")
        self.corpus_tree.column("trs", width=150, anchor="w")
        self.corpus_tree.column("textgrid", width=180, anchor="w")
        self.corpus_tree.column("corrected", width=120, anchor="center")
        self.corpus_tree.column("corrected_by", width=160, anchor="w")
        self.corpus_tree.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            tab, orient="vertical", command=self.corpus_tree.yview
        )
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.corpus_tree.configure(yscrollcommand=scrollbar.set)
        self.corpus_tree.bind(
            "<Double-1>", lambda _event: self.open_selected_corpus_entry()
        )
        self.corpus_tree.bind(
            "<Return>", lambda _event: self.open_selected_corpus_entry()
        )
        ttk.Label(
            tab,
            text=(
                "Les fichiers TRS et TextGrid ayant le même nom que l’audio "
                "sont associés et affichés séparément."
            ),
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_plugins_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=6)
        self.plugins_tab = tab
        self.notebook.add(tab, text="Plugins")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")
        recordings_panel = ttk.LabelFrame(panes, text="Enregistrements", padding=6)
        plugin_list_panel = ttk.LabelFrame(panes, text="Plugins", padding=6)
        self.plugin_parameters_panel = ttk.LabelFrame(
            panes, text="Paramètres", padding=6
        )
        panes.add(recordings_panel, weight=1)
        panes.add(plugin_list_panel, weight=1)
        panes.add(self.plugin_parameters_panel, weight=2)

        recordings_panel.rowconfigure(0, weight=1)
        recordings_panel.columnconfigure(0, weight=1)
        self.plugin_recordings_tree = ttk.Treeview(
            recordings_panel, show="tree", selectmode="browse"
        )
        self.plugin_recordings_tree.heading("#0", text="Fichier audio")
        self.plugin_recordings_tree.column("#0", width=250, anchor="w", stretch=True)
        self.plugin_recordings_tree.grid(row=0, column=0, sticky="nsew")
        recording_scrollbar = ttk.Scrollbar(
            recordings_panel,
            orient="vertical",
            command=self.plugin_recordings_tree.yview,
        )
        recording_scrollbar.grid(row=0, column=1, sticky="ns")
        self.plugin_recordings_tree.configure(
            yscrollcommand=recording_scrollbar.set
        )
        self.plugin_recordings_tree.bind(
            "<<TreeviewSelect>>", self._on_plugin_audio_selected
        )

        plugin_list_panel.rowconfigure(0, weight=1)
        plugin_list_panel.columnconfigure(0, weight=1)
        self.plugin_list_tree = ttk.Treeview(
            plugin_list_panel, show="tree", selectmode="browse"
        )
        self.plugin_list_tree.column("#0", width=240, anchor="w", stretch=True)
        self.plugin_list_tree.insert(
            "", "end", iid="mfa", text="Montreal Forced Aligner (MFA)"
        )
        self.plugin_list_tree.insert(
            "", "end", iid="silence_detector", text="Silence Detector"
        )
        self.plugin_list_tree.insert(
            "", "end", iid="cv_syllabifier", text="CV and Syllable Segmenter"
        )
        self.plugin_list_tree.grid(row=0, column=0, sticky="nsew")
        plugin_scrollbar = ttk.Scrollbar(
            plugin_list_panel,
            orient="vertical",
            command=self.plugin_list_tree.yview,
        )
        plugin_scrollbar.grid(row=0, column=1, sticky="ns")
        self.plugin_list_tree.configure(yscrollcommand=plugin_scrollbar.set)
        self.plugin_list_tree.bind("<<TreeviewSelect>>", self._on_plugin_selected)
        self.plugin_list_tree.selection_set("mfa")

        parameters = self.plugin_parameters_panel
        parameters.rowconfigure(0, weight=1)
        parameters.columnconfigure(0, weight=1)
        self.plugin_mfa_form = ttk.Frame(parameters)
        self.plugin_mfa_form.grid(row=0, column=0, sticky="nsew")
        self.plugin_mfa_form.columnconfigure(1, weight=1)
        self.plugin_audio = tk.StringVar()
        self.plugin_transcript = tk.StringVar()
        self.plugin_executable = tk.StringVar(
            value=self._saved_mfa_executable or shutil.which("mfa") or ""
        )
        self.plugin_dictionary = tk.StringVar(value="french_mfa")
        self.plugin_acoustic_model = tk.StringVar(value="french_mfa")
        self.plugin_output = tk.StringVar()
        ttk.Label(
            self.plugin_mfa_form,
            text="Aligne un enregistrement et sa transcription avec MFA.",
            wraplength=320,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 12))

        ttk.Label(self.plugin_mfa_form, text="Audio sélectionné").grid(
            row=1, column=0, sticky="w", padx=(8, 8), pady=5
        )
        ttk.Entry(
            self.plugin_mfa_form, textvariable=self.plugin_audio, state="readonly"
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=5)

        self._plugin_path_row(
            self.plugin_mfa_form,
            2,
            "Transcription",
            self.plugin_transcript,
            "Choisir la transcription",
            [
                ("Transcription", "*.txt *.lab *.trs"),
                ("Tous les fichiers", "*"),
            ],
        )
        self._plugin_path_row(
            self.plugin_mfa_form,
            3,
            "Exécutable MFA",
            self.plugin_executable,
            "Choisir MFA",
            [("Tous les fichiers", "*")],
            on_change=self._save_mfa_executable,
        )
        self._plugin_entry_row(
            self.plugin_mfa_form, 4, "Dictionnaire", self.plugin_dictionary
        )
        self._plugin_entry_row(
            self.plugin_mfa_form, 5, "Modèle acoustique", self.plugin_acoustic_model
        )
        self._plugin_path_row(
            self.plugin_mfa_form,
            6,
            "Dossier de sortie",
            self.plugin_output,
            "Choisir le dossier de sortie",
            directory=True,
        )

        self.plugin_silence_form = ttk.Frame(parameters)
        self.plugin_silence_form.grid(row=0, column=0, sticky="nsew")
        self.plugin_silence_form.columnconfigure(1, weight=1)
        self.plugin_silence_threshold = tk.StringVar(value="-35")
        self.plugin_silence_minimum = tk.StringVar(value="0.3")
        ttk.Label(
            self.plugin_silence_form,
            text=(
                "Repère les pauses et crée des frontières de phrase dans un "
                "fichier Transcriber (.trs). Les pauses sont marquées « # »."
            ),
            wraplength=320,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 12))
        ttk.Label(self.plugin_silence_form, text="Audio sélectionné").grid(
            row=1, column=0, sticky="w", padx=(8, 8), pady=5
        )
        ttk.Entry(
            self.plugin_silence_form, textvariable=self.plugin_audio, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=5)
        ttk.Label(self.plugin_silence_form, text="Seuil de silence (dB)").grid(
            row=2, column=0, sticky="w", padx=(8, 8), pady=5
        )
        ttk.Spinbox(
            self.plugin_silence_form,
            textvariable=self.plugin_silence_threshold,
            from_=-100,
            to=0,
            increment=1,
        ).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=5)
        ttk.Label(
            self.plugin_silence_form, text="Durée minimale de pause (secondes)"
        ).grid(row=3, column=0, sticky="w", padx=(8, 8), pady=5)
        ttk.Spinbox(
            self.plugin_silence_form,
            textvariable=self.plugin_silence_minimum,
            from_=0.01,
            to=3600,
            increment=0.1,
        ).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=5)

        self.plugin_cv_form = ttk.Frame(parameters)
        self.plugin_cv_form.grid(row=0, column=0, sticky="nsew")
        self.plugin_cv_form.columnconfigure(1, weight=1)
        self.plugin_cv_phone_level = tk.StringVar(value="phone")
        self.plugin_cv_token_level = tk.StringVar(value="token")
        self.plugin_cv_level = tk.StringVar(value="CV")
        self.plugin_syllable_level = tk.StringVar(value="syll")
        self.plugin_vowel_phones = tk.StringVar(value=DEFAULT_VOWEL_PHONES)
        self.plugin_maximum_onset = tk.StringVar(value="2")
        ttk.Label(
            self.plugin_cv_form,
            text=(
                "Le TextGrid associé doit contenir les niveaux de phones et de "
                "tokens (les niveaux MFA « phones » et « words » sont reconnus). "
                "Le plugin ajoute ou recalcule les niveaux C/V et syllabique."
            ),
            wraplength=320,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 12))
        ttk.Label(self.plugin_cv_form, text="Audio sélectionné").grid(
            row=1, column=0, sticky="w", padx=(8, 8), pady=5
        )
        ttk.Entry(
            self.plugin_cv_form, textvariable=self.plugin_audio, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=5)
        self._plugin_entry_row(
            self.plugin_cv_form, 2, "Niveau des phones", self.plugin_cv_phone_level
        )
        self._plugin_entry_row(
            self.plugin_cv_form, 3, "Niveau des tokens", self.plugin_cv_token_level
        )
        self._plugin_entry_row(
            self.plugin_cv_form, 4, "Niveau C/V en sortie", self.plugin_cv_level
        )
        self._plugin_entry_row(
            self.plugin_cv_form, 5, "Niveau syllabique en sortie", self.plugin_syllable_level
        )
        self._plugin_entry_row(
            self.plugin_cv_form, 6, "Phones vocaliques (séparés par des espaces)",
            self.plugin_vowel_phones,
        )
        ttk.Label(
            self.plugin_cv_form,
            text="Consonnes maximales en attaque",
        ).grid(row=7, column=0, sticky="w", padx=(14, 8), pady=5)
        ttk.Spinbox(
            self.plugin_cv_form,
            textvariable=self.plugin_maximum_onset,
            from_=1,
            to=3,
            increment=1,
        ).grid(row=7, column=1, sticky="ew", padx=(0, 14), pady=5)
        self.plugin_run_button = ttk.Button(
            parameters,
            text="Lancer l’alignement",
            command=self._run_selected_plugin,
            state="disabled",
        )
        self.plugin_run_button.grid(
            row=1, column=0, sticky="e", padx=8, pady=(12, 8)
        )
        self._on_plugin_selected()

    def _on_plugin_audio_selected(self, _event: tk.Event | None = None) -> None:
        selected = self.plugin_recordings_tree.selection()
        if not selected:
            return
        entry = self._corpus_entries.get(selected[0])
        if entry is not None:
            self._set_plugin_audio(entry)

    def _on_plugin_selected(self, _event: tk.Event | None = None) -> None:
        selected_plugins = self.plugin_list_tree.selection()
        selected_plugin = selected_plugins[0] if selected_plugins else None
        if selected_plugin == "mfa":
            self.plugin_mfa_form.tkraise()
            self.plugin_run_button.configure(text="Lancer l’alignement")
        elif selected_plugin == "silence_detector":
            self.plugin_silence_form.tkraise()
            self.plugin_run_button.configure(text="Détecter les silences")
        elif selected_plugin == "cv_syllabifier":
            self.plugin_cv_form.tkraise()
            self.plugin_run_button.configure(text="Générer les niveaux CV/syll")
        enabled = (
            "normal"
            if selected_plugin in {"mfa", "silence_detector", "cv_syllabifier"}
            and self._plugin_audio_key is not None
            and not self.busy
            else "disabled"
        )
        self.plugin_run_button.configure(state=enabled)

    def _set_plugin_audio(self, entry: CorpusEntry) -> None:
        key = str(entry.audio_path)
        if key == self._plugin_audio_key:
            return
        self._plugin_audio_key = key
        self.plugin_audio.set(key)
        transcript_candidates = (
            entry.audio_path.with_suffix(".txt"),
            entry.audio_path.with_suffix(".lab"),
            entry.trs_path or entry.audio_path.with_suffix(".trs"),
        )
        transcript = next(
            (
                path
                for path in transcript_candidates
                if path is not None and path.is_file()
            ),
            None,
        )
        self.plugin_transcript.set(str(transcript) if transcript else "")
        self.plugin_output.set(str(entry.audio_path.parent))
        self._on_plugin_selected()

    def _run_selected_plugin(self) -> None:
        selected = self.plugin_list_tree.selection()
        if not selected:
            self._status.set("Sélectionnez un plugin.")
            return
        if selected[0] == "mfa":
            self._run_selected_mfa()
        elif selected[0] == "silence_detector":
            self._run_selected_silence_detector()
        elif selected[0] == "cv_syllabifier":
            self._run_selected_cv_syllabifier()

    def _run_selected_mfa(self) -> None:
        if self.busy:
            self._status.set("Un alignement MFA est déjà en cours.")
            return
        selected = self.plugin_recordings_tree.selection()
        if not selected:
            self._status.set("Sélectionnez un enregistrement pour lancer MFA.")
            return
        entry = self._corpus_entries.get(selected[0])
        if entry is None:
            self._status.set("L’enregistrement sélectionné n’est plus dans le corpus.")
            return
        self._set_plugin_audio(entry)
        self._current_entry = entry
        if self.visualizer.audio_path != entry.audio_path:
            self.visualizer.load_audio(entry.audio_path)
        mfa_executable = self.plugin_executable.get().strip()
        self._save_mfa_executable(mfa_executable)
        self._start_alignment(
            self.plugin_parameters_panel,
            str(entry.audio_path),
            self.plugin_transcript.get().strip(),
            mfa_executable,
            self.plugin_dictionary.get().strip(),
            self.plugin_acoustic_model.get().strip(),
            self.plugin_output.get().strip(),
        )

    def _run_selected_silence_detector(self) -> None:
        if self.busy:
            self._status.set("Un traitement de plugin est déjà en cours.")
            return
        selected = self.plugin_recordings_tree.selection()
        if not selected:
            self._status.set("Sélectionnez un enregistrement pour détecter les silences.")
            return
        entry = self._corpus_entries.get(selected[0])
        if entry is None:
            self._status.set("L’enregistrement sélectionné n’est plus dans le corpus.")
            return
        try:
            threshold_db = float(self.plugin_silence_threshold.get())
            minimum_silence_seconds = float(self.plugin_silence_minimum.get())
        except ValueError:
            messagebox.showerror(
                "Paramètres invalides",
                "Le seuil et la durée minimale doivent être des nombres.",
                parent=self.plugin_silence_form,
            )
            return
        if (
            not math.isfinite(threshold_db)
            or not -100.0 <= threshold_db <= 0.0
            or not math.isfinite(minimum_silence_seconds)
            or minimum_silence_seconds <= 0.0
        ):
            messagebox.showerror(
                "Paramètres invalides",
                "Le seuil doit être compris entre -100 et 0 dB et la durée doit "
                "être positive.",
                parent=self.plugin_silence_form,
            )
            return

        output = entry.trs_path or entry.audio_path.with_suffix(".trs")
        overwrite = output.exists()
        if overwrite and not messagebox.askyesno(
            "Fichier Transcriber déjà présent",
            f"{output}\n\nLe contenu de ce fichier sera remplacé par une nouvelle "
            "segmentation. Voulez-vous continuer ?",
            parent=self.plugin_silence_form,
        ):
            return

        self._silence_entry = entry
        self._active_plugin = "silence_detector"
        self._current_entry = entry
        if self.visualizer.audio_path != entry.audio_path:
            self.visualizer.load_audio(entry.audio_path)
        correction = self._correction_info.get(
            str(entry.audio_path), CorrectionInfo()
        )
        self.busy = True
        self._on_plugin_selected()
        self._status.set(f"Détection des silences : {entry.audio_path.name}")
        threading.Thread(
            target=self._run_silence_detection,
            args=(
                entry.audio_path,
                output,
                threshold_db,
                minimum_silence_seconds,
                self._metadata_root_for_entry(entry),
                correction.corrected,
                correction.corrected_by,
                overwrite,
            ),
            daemon=True,
        ).start()

    def _run_silence_detection(
        self,
        audio_path: Path,
        output_path: Path,
        threshold_db: float,
        minimum_silence_seconds: float,
        metadata_root: Path,
        corrected: bool,
        corrected_by: str,
        archive_existing: bool,
    ) -> None:
        try:
            result = detect_silences(
                audio_path,
                threshold_db,
                minimum_silence_seconds,
            )
            if archive_existing:
                record_annotation_revision(
                    audio_path,
                    metadata_root,
                    corrected=corrected,
                    corrected_by=corrected_by,
                )
            write_silence_trs(audio_path, output_path, result, "Locuteur 1")
            self.messages.put(("silence_success", str(output_path)))
        except (SilenceDetectionError, AnnotationHistoryError, ValueError) as error:
            self.messages.put(("error", str(error)))

    def _run_selected_cv_syllabifier(self) -> None:
        if self.busy:
            self._status.set("Un traitement de plugin est déjà en cours.")
            return
        selected = self.plugin_recordings_tree.selection()
        if not selected:
            self._status.set("Sélectionnez un enregistrement pour le segmenter.")
            return
        entry = self._corpus_entries.get(selected[0])
        if entry is None:
            self._status.set("L’enregistrement sélectionné n’est plus dans le corpus.")
            return
        annotation_path = entry.textgrid_path
        if annotation_path is None or not annotation_path.is_file():
            messagebox.showerror(
                "TextGrid requis",
                "CV and Syllable Segmenter nécessite un TextGrid associé à cet audio "
                "avec des niveaux phone et token.",
                parent=self.plugin_cv_form,
            )
            return
        try:
            maximum_onset_consonants = int(self.plugin_maximum_onset.get())
        except ValueError:
            messagebox.showerror(
                "Paramètres invalides",
                "Le nombre maximal de consonnes en attaque doit être un entier de 1 à 3.",
                parent=self.plugin_cv_form,
            )
            return
        if not 1 <= maximum_onset_consonants <= 3:
            messagebox.showerror(
                "Paramètres invalides",
                "Le nombre maximal de consonnes en attaque doit être compris entre 1 et 3.",
                parent=self.plugin_cv_form,
            )
            return
        if not messagebox.askyesno(
            "Générer les niveaux CV et syllabique",
            "Le TextGrid actuel sera archivé avant l’enregistrement. Les niveaux "
            "de sortie existants portant ces noms seront remplacés. Continuer ?",
            parent=self.plugin_cv_form,
        ):
            return

        self._set_plugin_audio(entry)
        self._cv_entry = entry
        self._cv_source_text = None
        self._cv_source_tiers = None
        self._cv_result = None
        self._active_plugin = "cv_syllabifier"
        self._current_entry = entry
        if self.visualizer.audio_path != entry.audio_path:
            self.visualizer.load_audio(entry.audio_path)
        self.visualizer.load_textgrid(annotation_path)
        self.busy = True
        self._on_plugin_selected()
        self._status.set(f"Segmentation CV et syllabique : {entry.audio_path.name}")
        threading.Thread(
            target=self._run_cv_syllabifier,
            args=(
                annotation_path,
                self.plugin_cv_phone_level.get().strip(),
                self.plugin_cv_token_level.get().strip(),
                self.plugin_cv_level.get().strip(),
                self.plugin_syllable_level.get().strip(),
                self.plugin_vowel_phones.get(),
                maximum_onset_consonants,
            ),
            daemon=True,
        ).start()

    def _run_cv_syllabifier(
        self,
        annotation_path: Path,
        phone_level: str,
        token_level: str,
        cv_level: str,
        syllable_level: str,
        vowel_phones: str,
        maximum_onset_consonants: int,
    ) -> None:
        try:
            source_text = annotation_path.read_text(encoding="utf-8-sig")
            source_tiers = parse_textgrid_content(source_text)
            result = segment_cv_and_syllables(
                source_tiers,
                phone_level=phone_level,
                token_level=token_level,
                cv_level=cv_level,
                syllable_level=syllable_level,
                vowel_phones=vowel_phones,
                maximum_onset_consonants=maximum_onset_consonants,
            )
            self._cv_source_text = source_text
            self._cv_source_tiers = source_tiers
            self._cv_result = result
            self.messages.put(("cv_ready", str(annotation_path)))
        except (OSError, UnicodeError, ValueError) as error:
            self.messages.put(("error", str(error)))

    def _complete_cv_syllabifier(self, annotation_path: Path) -> None:
        entry = self._cv_entry
        source_text = self._cv_source_text
        source_tiers = self._cv_source_tiers
        result = self._cv_result
        self._cv_entry = None
        self._cv_source_text = None
        self._cv_source_tiers = None
        self._cv_result = None
        if entry is None or source_text is None or source_tiers is None or result is None:
            self._status.set("Le résultat de la segmentation CV est indisponible.")
            return

        try:
            current_source = annotation_path.read_text(encoding="utf-8-sig")
            if current_source != source_text:
                raise ValueError(
                    "Le TextGrid a été modifié pendant le calcul. Aucun résultat n’a "
                    "été enregistré ; relancez le plugin sur la version actuelle."
                )
            generated_tiers = (result.cv_tier, result.syllable_tier)
            replacements: dict[int, TextGridTier] = {}
            new_tiers: list[TextGridTier] = []
            for generated_tier in generated_tiers:
                matches = [
                    tier for tier in source_tiers if tier.name == generated_tier.name
                ]
                if len(matches) > 1:
                    raise ValueError(
                        f"Le niveau « {generated_tier.name} » apparaît plusieurs "
                        "fois dans le TextGrid."
                    )
                if matches:
                    source_index = matches[0].source_index
                    if source_index is None:
                        raise ValueError(
                            f"Le niveau « {generated_tier.name} » ne peut pas être remplacé."
                        )
                    replacements[source_index] = replace(
                        generated_tier, source_index=source_index
                    )
                else:
                    new_tiers.append(generated_tier)
            tiers_to_save = tuple(
                replacements.get(tier.source_index, tier)
                if tier.source_index is not None
                else tier
                for tier in source_tiers
            ) + tuple(new_tiers)
            correction = self._correction_info.get(
                str(entry.audio_path), CorrectionInfo()
            )
            record_annotation_revision(
                entry.audio_path,
                self._metadata_root_for_entry(entry),
                corrected=correction.corrected,
                corrected_by=correction.corrected_by,
            )
            save_textgrid(annotation_path, source_text, tiers_to_save)
        except (AnnotationHistoryError, OSError, UnicodeError, ValueError) as error:
            self._status.set("Échec du CV and Syllable Segmenter.")
            messagebox.showerror(
                "Échec du CV and Syllable Segmenter",
                str(error),
                parent=self.plugin_cv_form,
            )
            return

        if (
            self._current_entry is not None
            and self._current_entry.audio_path == entry.audio_path
        ):
            self._current_entry = self._corpus_entries.get(
                str(entry.audio_path), entry
            )
            self.visualizer.load_textgrid(annotation_path)
        self._status.set(
            f"Segmentation terminée : niveaux « {result.cv_tier.name} » et "
            f"« {result.syllable_tier.name} » enregistrés. "
            f"Segments sans noyau vocalique : {result.tokens_without_nucleus}."
        )

    @staticmethod
    def _audio_filetypes() -> list[tuple[str, str]]:
        return [
            ("Fichiers audio", "*.wav *.flac *.mp3 *.m4a *.ogg *.aiff *.aif"),
            ("Tous les fichiers", "*"),
        ]

    def open_audio(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Ouvrir un fichier audio",
            filetypes=self._audio_filetypes(),
        )
        if not path:
            return
        try:
            entry = make_corpus_entry(path)
            correction_info = load_corpus_corrections(entry.audio_path.parent)
        except ValueError as error:
            messagebox.showerror("Ouverture impossible", str(error), parent=self.root)
            return
        self._correction_info.update(correction_info)
        self._show_corpus_entry(entry)

    def open_corpus_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self.root, title="Ouvrir un dossier de corpus"
        )
        if not folder:
            return
        root = Path(folder).expanduser().resolve()
        catalog_warning: str | None = None
        try:
            catalog_path = root / CORPUS_INDEX_FILENAME
            if catalog_path.is_file():
                catalog = load_corpus_catalog(root)
                entries = catalog.entries
                correction_info = catalog.corrections
                status = (
                    f"Catalogue chargé : {len(entries)} fichier(s) audio "
                    "(utilisez Actualiser pour rescanner)."
                )
                if catalog.missing_audio_count or catalog.missing_annotation_count:
                    status += (
                        f" Fichiers absents ignorés : {catalog.missing_audio_count} audio(s), "
                        f"{catalog.missing_annotation_count} annotation(s)."
                    )
            else:
                audio_files = discover_audio_files(root)
                entries = tuple(make_corpus_entry(path) for path in audio_files)
                correction_info = load_corpus_corrections(root)
                try:
                    save_corpus_catalog(root, entries, correction_info)
                    status = (
                        f"Corpus chargé et catalogue créé : {len(entries)} fichier(s) audio."
                    )
                except ValueError as error:
                    catalog_warning = str(error)
                    status = (
                        f"Corpus chargé : {len(entries)} fichier(s) audio ; "
                        "catalogue non enregistré."
                    )
        except ValueError as error:
            messagebox.showerror("Ouverture du corpus impossible", str(error), parent=self.root)
            return
        self._corpus_folder.set(str(root))
        self._corpus_root = root
        self._correction_info = correction_info
        self._replace_corpus_entries(entries)
        self._status.set(status)
        if catalog_warning is not None:
            messagebox.showwarning(
                "Catalogue non enregistré", catalog_warning, parent=self.root
            )

    def refresh_corpus(self) -> None:
        root = self._corpus_root
        if root is None:
            self._status.set("Ouvrez un dossier de corpus avant de l’actualiser.")
            return
        try:
            audio_files = discover_audio_files(root)
            entries = tuple(make_corpus_entry(path) for path in audio_files)
            corrections = load_corpus_corrections(root)
            catalog_path = root / CORPUS_INDEX_FILENAME
            if catalog_path.is_file():
                corrections.update(load_corpus_catalog(root).corrections)
        except ValueError as error:
            messagebox.showerror("Actualisation impossible", str(error), parent=self.root)
            return
        try:
            save_corpus_catalog(root, entries, corrections)
        except ValueError as error:
            self._correction_info = corrections
            self._replace_corpus_entries(entries)
            self._status.set(
                f"Corpus actualisé : {len(entries)} fichier(s) audio ; "
                "catalogue non enregistré."
            )
            messagebox.showwarning(
                "Catalogue non enregistré", str(error), parent=self.root
            )
            return
        self._correction_info = corrections
        self._replace_corpus_entries(entries)
        self._status.set(
            f"Corpus actualisé : {len(entries)} fichier(s) audio ; catalogue enregistré."
        )

    def add_audio_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Ajouter des fichiers audio au corpus",
            filetypes=self._audio_filetypes(),
        )
        if not paths:
            return
        try:
            entries = [make_corpus_entry(path) for path in paths]
            corrections = {}
            for entry in entries:
                corrections.update(load_corpus_corrections(entry.audio_path.parent))
        except ValueError as error:
            messagebox.showerror("Ajout impossible", str(error), parent=self.root)
            return
        self._correction_info.update(corrections)
        for entry in entries:
            self._insert_corpus_entry(entry)
        if (
            self._corpus_root is not None
            and all(entry.audio_path.is_relative_to(self._corpus_root) for entry in entries)
        ):
            try:
                save_corpus_catalog(
                    self._corpus_root,
                    tuple(self._corpus_entries.values()),
                    self._correction_info,
                )
            except ValueError as error:
                messagebox.showerror("Catalogue non enregistré", str(error), parent=self.root)
        else:
            self._corpus_root = None
            self._corpus_folder.set("Corpus constitué à partir de fichiers sélectionnés")
        self._status.set(f"{len(entries)} fichier(s) audio ajouté(s) au corpus.")

    def _replace_corpus_entries(self, entries: tuple[CorpusEntry, ...]) -> None:
        self.corpus_tree.delete(*self.corpus_tree.get_children())
        self.plugin_recordings_tree.delete(
            *self.plugin_recordings_tree.get_children()
        )
        self._corpus_entries.clear()
        self._plugin_audio_key = None
        self.plugin_audio.set("")
        self.plugin_transcript.set("")
        self.plugin_output.set("")
        for entry in entries:
            self._insert_corpus_entry(entry)
        self._on_plugin_selected()

    def _insert_corpus_entry(self, entry: CorpusEntry) -> None:
        key = str(entry.audio_path)
        if key in self._corpus_entries:
            return
        self._corpus_entries[key] = entry
        correction = self._correction_info.get(key, CorrectionInfo())
        display_audio = entry.audio_path.name
        if self._corpus_root is not None:
            try:
                display_audio = str(entry.audio_path.relative_to(self._corpus_root))
            except ValueError:
                pass
        self.corpus_tree.insert(
            "",
            "end",
            iid=key,
            values=(
                display_audio,
                entry.trs_path.name if entry.trs_path else "—",
                entry.textgrid_path.name if entry.textgrid_path else "—",
                "Oui" if correction.corrected else "Non",
                correction.corrected_by,
            ),
        )
        self.plugin_recordings_tree.insert("", "end", iid=key, text=display_audio)

    def _metadata_root_for_entry(self, entry: CorpusEntry) -> Path:
        metadata_root = self._corpus_root
        if metadata_root is None or not entry.audio_path.is_relative_to(metadata_root):
            metadata_root = entry.audio_path.parent
        return metadata_root

    def _archive_before_visualizer_edit(self) -> None:
        entry = self._current_entry
        annotation_path = self.visualizer.annotation_path
        if entry is None or annotation_path is None:
            return
        entry_annotations = (entry.trs_path, entry.textgrid_path)
        if not any(
            candidate is not None
            and candidate.resolve() == annotation_path.resolve()
            for candidate in entry_annotations
        ):
            return
        correction = self._correction_info.get(
            str(entry.audio_path), CorrectionInfo()
        )
        record_annotation_revision(
            entry.audio_path,
            self._metadata_root_for_entry(entry),
            corrected=correction.corrected,
            corrected_by=correction.corrected_by,
        )

    def edit_selected_correction(self) -> None:
        selected = self.corpus_tree.selection()
        if len(selected) != 1:
            self._status.set("Sélectionnez un seul fichier pour modifier sa correction.")
            return
        audio_key = selected[0]
        entry = self._corpus_entries.get(audio_key)
        if entry is None:
            return
        current = self._correction_info.get(audio_key, CorrectionInfo())

        dialog = tk.Toplevel(self.root)
        dialog.title("État de la correction")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        corrected = tk.BooleanVar(value=current.corrected)
        corrected_by = tk.StringVar(value=current.corrected_by)
        ttk.Label(dialog, text=entry.audio_path.name, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 8)
        )
        ttk.Checkbutton(
            dialog,
            text="La correction est terminée",
            variable=corrected,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=5)
        ttk.Label(dialog, text="Corrigée par").grid(
            row=2, column=0, sticky="w", padx=(14, 8), pady=5
        )
        ttk.Entry(dialog, textvariable=corrected_by, width=32).grid(
            row=2, column=1, sticky="ew", padx=(0, 14), pady=5
        )

        def save() -> None:
            name = corrected_by.get().strip()
            if corrected.get() and not name:
                messagebox.showerror(
                    "Nom manquant",
                    "Indiquez qui a effectué la correction.",
                    parent=dialog,
                )
                return
            info = CorrectionInfo(corrected.get(), name if corrected.get() else "")
            new_corrections = dict(self._correction_info)
            new_corrections[audio_key] = info
            metadata_root = self._metadata_root_for_entry(entry)
            metadata_to_save = {
                path: correction
                for path, correction in new_corrections.items()
                if Path(path).is_relative_to(metadata_root)
            }
            try:
                updated_entry = make_corpus_entry(entry.audio_path)
                revision, archived = record_annotation_revision(
                    entry.audio_path,
                    metadata_root,
                    corrected=info.corrected,
                    corrected_by=info.corrected_by,
                )
                save_corpus_corrections(metadata_root, metadata_to_save)
                if (
                    self._corpus_root is not None
                    and entry.audio_path.is_relative_to(self._corpus_root)
                ):
                    catalog_entries = tuple(
                        corpus_entry
                        for corpus_entry in self._corpus_entries.values()
                        if corpus_entry.audio_path.is_relative_to(metadata_root)
                    )
                    catalog_entries = tuple(
                        updated_entry
                        if corpus_entry.audio_path == entry.audio_path
                        else corpus_entry
                        for corpus_entry in catalog_entries
                    )
                    save_corpus_catalog(
                        metadata_root, catalog_entries, new_corrections
                    )
            except (AnnotationHistoryError, ValueError) as error:
                messagebox.showerror(
                    "Enregistrement impossible", str(error), parent=dialog
                )
                return
            self._correction_info = new_corrections
            self._corpus_entries[audio_key] = updated_entry
            self.corpus_tree.set(
                audio_key,
                "trs",
                updated_entry.trs_path.name if updated_entry.trs_path else "—",
            )
            self.corpus_tree.set(
                audio_key,
                "textgrid",
                updated_entry.textgrid_path.name
                if updated_entry.textgrid_path
                else "—",
            )
            if (
                self._current_entry is not None
                and self._current_entry.audio_path == entry.audio_path
            ):
                self._current_entry = updated_entry
            self.corpus_tree.set(audio_key, "corrected", "Oui" if info.corrected else "Non")
            self.corpus_tree.set(audio_key, "corrected_by", info.corrected_by)
            if archived:
                self._status.set(
                    f"Correction enregistrée et version archivée "
                    f"({revision.revision_id})."
                )
            else:
                self._status.set(
                    "Correction enregistrée ; annotations inchangées, "
                    "aucune nouvelle version."
                )
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", padx=14, pady=14)
        ttk.Button(buttons, text="Annuler", command=dialog.destroy).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(buttons, text="Enregistrer", command=save).pack(side="right")
        dialog.grab_set()
        dialog.wait_window()

    def open_annotation_history(self) -> None:
        selected = self.corpus_tree.selection()
        if len(selected) != 1:
            self._status.set("Sélectionnez un seul fichier pour consulter son historique.")
            return
        audio_key = selected[0]
        entry = self._corpus_entries.get(audio_key)
        if entry is None:
            return
        metadata_root = self._metadata_root_for_entry(entry)
        try:
            revisions = list_annotation_revisions(entry.audio_path, metadata_root)
        except AnnotationHistoryError as error:
            messagebox.showerror("Historique indisponible", str(error), parent=self.root)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Historique des annotations — {entry.audio_path.name}")
        dialog.transient(self.root)
        dialog.geometry("720x340")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        history_tree = ttk.Treeview(
            dialog,
            columns=("date", "completion", "corrector", "files"),
            show="headings",
            selectmode="browse",
        )
        history_tree.heading("date", text="Date")
        history_tree.heading("completion", text="Correction")
        history_tree.heading("corrector", text="Corrigée par")
        history_tree.heading("files", text="Annotations archivées")
        history_tree.column("date", width=190, anchor="w")
        history_tree.column("completion", width=110, anchor="center")
        history_tree.column("corrector", width=140, anchor="w")
        history_tree.column("files", width=280, anchor="w")
        history_tree.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=12)
        scrollbar = ttk.Scrollbar(
            dialog, orient="vertical", command=history_tree.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=12)
        history_tree.configure(yscrollcommand=scrollbar.set)
        ttk.Label(
            dialog,
            text="Restaurer une version remplace les annotations actuelles.",
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)

        def refresh(revisions_to_show: tuple[AnnotationRevision, ...]) -> None:
            history_tree.delete(*history_tree.get_children())
            for revision in revisions_to_show:
                created_at = datetime.fromisoformat(revision.created_at)
                files = ", ".join(revision.files) or "Aucune annotation"
                history_tree.insert(
                    "",
                    "end",
                    iid=revision.revision_id,
                    values=(
                        created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Terminée" if revision.corrected else "En cours",
                        revision.corrected_by,
                        files,
                    ),
                )

        refresh(revisions)

        def restore_selected() -> None:
            selected_revision = history_tree.selection()
            if not selected_revision:
                self._status.set("Sélectionnez une version à restaurer.")
                return
            revision_id = selected_revision[0]
            if not messagebox.askyesno(
                "Restaurer les annotations",
                "Les annotations actuelles seront conservées dans l’historique "
                "avant la restauration. Continuer ?",
                parent=dialog,
            ):
                return
            correction = self._correction_info.get(audio_key, CorrectionInfo())
            try:
                revision = restore_annotation_revision(
                    entry.audio_path,
                    metadata_root,
                    revision_id,
                    corrected=correction.corrected,
                    corrected_by=correction.corrected_by,
                )
                restored_entry = make_corpus_entry(entry.audio_path)
                revisions_after_restore = list_annotation_revisions(
                    entry.audio_path, metadata_root
                )
            except (AnnotationHistoryError, ValueError) as error:
                messagebox.showerror("Restauration impossible", str(error), parent=dialog)
                return

            self._corpus_entries[audio_key] = restored_entry
            self.corpus_tree.set(
                audio_key,
                "trs",
                restored_entry.trs_path.name if restored_entry.trs_path else "—",
            )
            self.corpus_tree.set(
                audio_key,
                "textgrid",
                restored_entry.textgrid_path.name
                if restored_entry.textgrid_path
                else "—",
            )
            if (
                self._current_entry is not None
                and self._current_entry.audio_path == entry.audio_path
            ):
                self._current_entry = restored_entry
                if restored_entry.annotation_path is None:
                    self.visualizer.clear_textgrid()
                elif restored_entry.annotation_path.suffix.casefold() == ".trs":
                    self.visualizer.load_trs(restored_entry.annotation_path)
                else:
                    self.visualizer.load_textgrid(restored_entry.annotation_path)
            if (
                self._corpus_root is not None
                and entry.audio_path.is_relative_to(self._corpus_root)
            ):
                try:
                    save_corpus_catalog(
                        metadata_root,
                        tuple(self._corpus_entries.values()),
                        self._correction_info,
                    )
                except ValueError as error:
                    messagebox.showerror(
                        "Catalogue non enregistré", str(error), parent=dialog
                    )
                    return
            refresh(revisions_after_restore)
            self._status.set(
                f"Annotations restaurées depuis la version {revision.revision_id}."
            )

        buttons = ttk.Frame(dialog)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=12, pady=12)
        ttk.Button(buttons, text="Fermer", command=dialog.destroy).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(
            buttons, text="Restaurer cette version…", command=restore_selected
        ).pack(side="right")
        dialog.grab_set()

    def remove_corpus_entries(self) -> None:
        selected = self.corpus_tree.selection()
        for item_id in selected:
            self.corpus_tree.delete(item_id)
            if self.plugin_recordings_tree.exists(item_id):
                self.plugin_recordings_tree.delete(item_id)
            self._corpus_entries.pop(item_id, None)
            if self._plugin_audio_key == item_id:
                self._plugin_audio_key = None
                self.plugin_audio.set("")
                self.plugin_transcript.set("")
                self.plugin_output.set("")
                self._on_plugin_selected()
        if self._corpus_root is not None:
            try:
                save_corpus_catalog(
                    self._corpus_root,
                    tuple(self._corpus_entries.values()),
                    self._correction_info,
                )
            except ValueError as error:
                messagebox.showerror("Catalogue non enregistré", str(error), parent=self.root)

    def open_selected_corpus_entry(self) -> None:
        selected = self.corpus_tree.selection()
        if not selected:
            self._status.set("Sélectionnez un fichier audio dans le corpus.")
            return
        entry = self._corpus_entries.get(selected[0])
        if entry is not None:
            self._show_corpus_entry(entry)

    def _show_corpus_entry(self, entry: CorpusEntry) -> None:
        self._insert_corpus_entry(entry)
        key = str(entry.audio_path)
        self.plugin_recordings_tree.selection_set(key)
        self._set_plugin_audio(entry)
        self._current_entry = entry
        self.visualizer.load_audio(entry.audio_path)
        if entry.annotation_path is None:
            self.visualizer.clear_textgrid()
        elif entry.annotation_path.suffix.casefold() == ".trs":
            self.visualizer.load_trs(entry.annotation_path)
        else:
            self.visualizer.load_textgrid(entry.annotation_path)
        self.notebook.select(self.visualization_tab)
        self._status.set(f"Ouverture : {entry.audio_path.name}")

    def open_textgrid(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Ouvrir un TextGrid",
            filetypes=[
                ("TextGrid Praat", "*.TextGrid *.textgrid"),
                ("Tous les fichiers", "*"),
            ],
        )
        if not path:
            return
        self.visualizer.load_textgrid(path)
        self.notebook.select(self.visualization_tab)
        if self.visualizer.tiers:
            self._status.set(f"Annotations chargées indépendamment : {Path(path).name}")
        else:
            self._status.set("Le TextGrid n’a pas pu être affiché.")

    def open_trs(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Ouvrir un fichier TRS",
            filetypes=[
                ("Transcriber", "*.trs"),
                ("Tous les fichiers", "*"),
            ],
        )
        if not path:
            return
        self.visualizer.load_trs(path)
        self.notebook.select(self.visualization_tab)
        if self.visualizer.tiers:
            self._status.set(f"Annotations TRS chargées : {Path(path).name}")
        else:
            self._status.set("Le fichier TRS n’a pas pu être affiché.")

    def close_textgrid(self) -> None:
        self.visualizer.clear_textgrid()
        self._status.set("Annotations fermées ; le signal audio reste affiché.")

    def open_mfa_plugin(self) -> None:
        if self.busy:
            messagebox.showinfo(
                "Alignement en cours",
                "Attendez la fin de l’alignement en cours avant d’en démarrer un autre.",
                parent=self.root,
            )
            return
        self.notebook.select(self.plugins_tab)
        self.plugin_list_tree.selection_set("mfa")
        selected = self.corpus_tree.selection()
        key = selected[0] if selected else None
        entry = self._corpus_entries.get(key) if key else self._current_entry
        if entry is not None:
            audio_key = str(entry.audio_path)
            if self.plugin_recordings_tree.exists(audio_key):
                self.plugin_recordings_tree.selection_set(audio_key)
                self._set_plugin_audio(entry)

    @staticmethod
    def _plugin_path_row(
        parent: tk.Misc,
        row: int,
        label: str,
        variable: tk.StringVar,
        title: str,
        filetypes: list[tuple[str, str]] | None = None,
        *,
        directory: bool = False,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(14, 8), pady=5
        )
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        if on_change is not None:
            entry.bind("<FocusOut>", lambda _event: on_change(variable.get()))

        def browse() -> None:
            if directory:
                selected = filedialog.askdirectory(parent=parent, title=title)
            else:
                selected = filedialog.askopenfilename(
                    parent=parent, title=title, filetypes=filetypes or []
                )
            if selected:
                variable.set(selected)
                if on_change is not None:
                    on_change(selected)

        ttk.Button(parent, text="Parcourir…", command=browse).grid(
            row=row, column=2, sticky="e", padx=(8, 14), pady=5
        )

    def _save_mfa_executable(self, executable: str) -> None:
        if executable == self._saved_mfa_executable:
            return
        try:
            _write_mfa_executable(_mfa_executable_settings_path(), executable)
        except OSError as error:
            self._status.set(str(error))
            return
        self._saved_mfa_executable = executable
        self._status.set("Chemin de l’exécutable MFA enregistré.")

    @staticmethod
    def _plugin_entry_row(
        parent: tk.Misc, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(14, 8), pady=5
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=5
        )

    def _start_alignment(
        self,
        dialog: tk.Misc,
        audio: str,
        transcript: str,
        executable: str,
        dictionary: str,
        acoustic_model: str,
        output: str,
    ) -> None:
        if not all((audio, transcript, executable, dictionary, acoustic_model, output)):
            messagebox.showerror(
                "Champs manquants",
                "Indiquez l’audio, la transcription, MFA, le dictionnaire, "
                "le modèle acoustique et le dossier de sortie.",
                parent=dialog,
            )
            return

        audio_path = Path(audio).expanduser()
        target = Path(output).expanduser() / f"{audio_path.stem}.TextGrid"
        overwrite = False
        if target.exists():
            overwrite = messagebox.askyesno(
                "Fichier déjà présent",
                f"{target}\n\nVoulez-vous le remplacer ?",
                parent=dialog,
            )
            if not overwrite:
                return

        arguments = (
            str(audio_path),
            transcript,
            executable,
            dictionary,
            acoustic_model,
            output,
        )
        self._alignment_entry = self._corpus_entries.get(str(audio_path))
        self._active_plugin = "mfa"
        self.busy = True
        self._status.set(f"Alignement MFA en cours : {Path(audio).name}")
        self._on_plugin_selected()
        threading.Thread(
            target=self._run_alignment,
            args=(arguments, overwrite),
            daemon=True,
        ).start()

    def _run_alignment(self, arguments: tuple[str, ...], overwrite: bool) -> None:
        try:
            result = align_audio(
                audio_file=arguments[0],
                transcript_file=arguments[1],
                mfa_executable=arguments[2],
                dictionary_model=arguments[3],
                acoustic_model=arguments[4],
                output_directory=arguments[5],
                overwrite=overwrite,
            )
            self.messages.put(("success", str(result)))
        except AlignmentError as error:
            self.messages.put(("error", str(error)))

    def _register_silence_result(self, result: Path) -> None:
        entry = self._silence_entry
        self._silence_entry = None
        if entry is None:
            return
        item_id = str(entry.audio_path)
        current_entry = self._corpus_entries.get(item_id, entry)
        updated_entry = CorpusEntry(
            audio_path=current_entry.audio_path,
            trs_path=result,
            textgrid_path=current_entry.textgrid_path,
        )
        if item_id in self._corpus_entries:
            self._corpus_entries[item_id] = updated_entry
            if self.corpus_tree.exists(item_id):
                self.corpus_tree.set(item_id, "trs", result.name)
        if (
            self._current_entry is not None
            and self._current_entry.audio_path == updated_entry.audio_path
        ):
            self._current_entry = updated_entry
            self.visualizer.load_trs(result)
        if (
            item_id in self._corpus_entries
            and self._corpus_root is not None
            and updated_entry.audio_path.is_relative_to(self._corpus_root)
            and result.is_relative_to(self._corpus_root)
        ):
            try:
                save_corpus_catalog(
                    self._corpus_root,
                    tuple(self._corpus_entries.values()),
                    self._correction_info,
                )
            except ValueError as error:
                messagebox.showerror(
                    "Catalogue non enregistré",
                    str(error),
                    parent=self.root,
                )
        self._status.set(f"Détection terminée : annotation créée dans {result.name}.")

    def _poll_messages(self) -> None:
        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_messages)
            return

        self.busy = False
        if kind == "silence_success":
            self._register_silence_result(Path(payload))
        elif kind == "cv_ready":
            self._complete_cv_syllabifier(Path(payload))
        elif kind == "success":
            result = Path(payload)
            alignment_entry = self._alignment_entry
            self._alignment_entry = None
            if alignment_entry is not None:
                item_id = str(alignment_entry.audio_path)
                current_entry = self._corpus_entries.get(item_id, alignment_entry)
                updated_entry = CorpusEntry(
                    audio_path=current_entry.audio_path,
                    trs_path=current_entry.trs_path,
                    textgrid_path=result,
                )
                if item_id in self._corpus_entries:
                    self._corpus_entries[item_id] = updated_entry
                    if self.corpus_tree.exists(item_id):
                        self.corpus_tree.set(item_id, "textgrid", result.name)
                if (
                    self._current_entry is not None
                    and self._current_entry.audio_path == updated_entry.audio_path
                ):
                    self._current_entry = updated_entry
                    self.visualizer.load_textgrid(result)
                if (
                    item_id in self._corpus_entries
                    and self._corpus_root is not None
                    and updated_entry.audio_path.is_relative_to(self._corpus_root)
                    and result.is_relative_to(self._corpus_root)
                ):
                    try:
                        save_corpus_catalog(
                            self._corpus_root,
                            tuple(self._corpus_entries.values()),
                            self._correction_info,
                        )
                    except ValueError as error:
                        messagebox.showerror(
                            "Catalogue non enregistré",
                            str(error),
                            parent=self.root,
                        )
            else:
                self.visualizer.load_textgrid(result)
            self._status.set(f"Alignement terminé ; annotations chargées : {result.name}")
        else:
            self._alignment_entry = None
            self._silence_entry = None
            self._cv_entry = None
            self._cv_source_text = None
            self._cv_source_tiers = None
            self._cv_result = None
            if self._active_plugin == "silence_detector":
                self._status.set("Échec du Silence Detector.")
                message_title = "Échec du Silence Detector"
            elif self._active_plugin == "cv_syllabifier":
                self._status.set("Échec du CV and Syllable Segmenter.")
                message_title = "Échec du CV and Syllable Segmenter"
            else:
                self._status.set("Échec de l’alignement MFA.")
                message_title = "Échec de l’alignement MFA"
            messagebox.showerror(message_title, str(payload), parent=self.root)
        self._active_plugin = None
        self._on_plugin_selected()
        self.root.after(100, self._poll_messages)


def main() -> None:
    root = tk.Tk()
    PraalineSimpleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
player.py — TuneGen Audio Player
==================================
Plays note sequences using your own WAV files.

Expected WAV file naming
------------------------
One file per note, named exactly after the note it represents:

    sounds/
        C4.wav
        D4.wav
        E4.wav
        F4.wav
        G4.wav
        A4.wav
        B4.wav
        C5.wav

You can change the folder and the naming pattern via the constants below
or by passing arguments to the Player constructor.

Dependencies
------------
    pip install pygame

pygame.mixer is used because it supports overlapping playback and
precise per-note timing without requiring a heavy audio framework.
"""

from __future__ import annotations

import time
from pathlib import Path

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults — edit these to match your setup
# ---------------------------------------------------------------------------

DEFAULT_SOUNDS_DIR: str = "sounds"   # folder that contains your WAV files
DEFAULT_NOTE_DURATION: float = 0.5   # seconds each note is held before next
DEFAULT_NOTE_GAP: float = 0.05       # silence between notes (seconds)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    """
    Loads WAV files for every note in C_MAJOR_SCALE and plays sequences.

    Parameters
    ----------
    sounds_dir : str | Path
        Directory containing one WAV file per note (e.g. ``sounds/C4.wav``).
    note_duration : float
        Seconds to wait before starting the next note.
    note_gap : float
        Extra silence inserted between notes for articulation.
    extension : str
        File extension of your audio files (default ``".wav"``).
    """

    NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    def __init__(
        self,
        sounds_dir: str | Path = DEFAULT_SOUNDS_DIR,
        note_duration: float   = DEFAULT_NOTE_DURATION,
        note_gap: float        = DEFAULT_NOTE_GAP,
        extension: str         = ".wav",
    ) -> None:
        if not _PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for audio playback.\n"
                "Install it with:  pip install pygame"
            )

        self.sounds_dir    = Path(sounds_dir)
        self.note_duration = note_duration
        self.note_gap      = note_gap
        self.extension     = extension
        self._sounds: dict[str, pygame.mixer.Sound] = {}

        pygame.mixer.init()
        self._load_sounds()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_sounds(self) -> None:
        """Pre-load every note's WAV file into memory."""
        missing: list[str] = []

        for note in self.NOTES:
            path = self.sounds_dir / f"{note}{self.extension}"
            if path.exists():
                self._sounds[note] = pygame.mixer.Sound(str(path))
            else:
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                "The following WAV files were not found:\n"
                + "\n".join(f"  {p}" for p in missing)
                + f"\n\nMake sure your sounds folder is '{self.sounds_dir}' "
                  f"and each file is named like 'C4.wav', 'D4.wav', etc."
            )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_note(self, note: str) -> None:
        """Play a single note and block for note_duration + note_gap."""
        if note not in self._sounds:
            raise ValueError(
                f"'{note}' has no loaded sound. "
                f"Valid notes: {self.NOTES}"
            )
        self._sounds[note].play()
        time.sleep(self.note_duration)
        self._sounds[note].stop()
        time.sleep(self.note_gap)

    def play_sequence(self, notes: list[str], label: str = "") -> None:
        """
        Play a list of notes one after another.

        Parameters
        ----------
        notes : list[str]
            Note names to play (e.g. ``["C4", "E4", "G4"]``).
        label : str
            Optional label printed to stdout before playback starts.
        """
        if label:
            print(f"  ♪  {label}: {' → '.join(notes)}")
        for note in notes:
            self.play_note(note)

    def play_result(
        self,
        result,                          # TuneGenResult from engine.py
        play_input:    bool = True,
        play_safe:     bool = True,
        play_creative: bool = True,
        play_unexpected: bool = True,
        pause_between: float = 0.4,      # silence between sections
    ) -> None:
        """
        Convenience method: plays all (or selected) sections of a
        TuneGenResult in order — input → SAFE → CREATIVE → UNEXPECTED.

        Parameters
        ----------
        result : TuneGenResult
            The object returned by ``engine.generate()``.
        play_input : bool
            Whether to play the original input sequence first.
        play_safe / play_creative / play_unexpected : bool
            Toggle each continuation on or off.
        pause_between : float
            Seconds of silence inserted between sections.
        """
        sections: list[tuple[str, list[str]]] = []

        if play_input:
            sections.append(("INPUT", result.input_sequence))
        if play_safe:
            sections.append(("SAFE", result.safe))
        if play_creative:
            sections.append(("CREATIVE", result.creative))
        if play_unexpected:
            sections.append(("UNEXPECTED", result.unexpected))

        print(f"\n  Playing: {result.chord_detected} "
              f"({result.contour} contour)\n")

        for label, notes in sections:
            self.play_sequence(notes, label=label)
            time.sleep(pause_between)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release pygame mixer resources."""
        pygame.mixer.quit()

    def __enter__(self) -> "Player":
        return self

    def __exit__(self, *_) -> None:
        self.close()

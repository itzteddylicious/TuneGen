"""
player.py — TuneGen Audio Player
==================================
Plays note sequences using your own WAV files.

Expected WAV file naming
------------------------
One file per note, named exactly after the note it represents:

    sounds/
        C4.wav  D4.wav  E4.wav  F4.wav
        G4.wav  A4.wav  B4.wav  C5.wav

Dependencies
------------
    pip install pygame

How duration works
------------------
Each WAV file is 2–3 seconds long. We start playback then call stop()
after our chosen hold time — giving us full control over note length
without needing multiple files.

Duration profiles (seconds per note)
-------------------------------------
  SAFE       — 1.0s  long, legato, calm and resolved
  CREATIVE   — 0.6s  medium, flowing but varied
  UNEXPECTED — 0.25s short, staccato, punchy contrast
  INPUT      — 0.5s  neutral reference playback
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
# Duration profiles — tweak these to taste
# ---------------------------------------------------------------------------

# How long each note rings before the next one starts (seconds)
DURATION_PROFILES: dict[str, float] = {
    "INPUT":      0.50,
    "SAFE":       1.00,
    "CREATIVE":   0.60,
    "UNEXPECTED": 0.25,
}

# Silence between notes — kept short so staccato still sounds tight
GAP_PROFILES: dict[str, float] = {
    "INPUT":      0.05,
    "SAFE":       0.08,   # tiny breath between held notes
    "CREATIVE":   0.06,
    "UNEXPECTED": 0.10,   # punchy gap reinforces the staccato feel
}

DEFAULT_SOUNDS_DIR: str = "sounds"
DEFAULT_EXTENSION:  str = ".wav"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    """
    Loads WAV files for every note in the C Major scale and plays sequences.

    Parameters
    ----------
    sounds_dir : str | Path
        Directory containing one WAV per note (e.g. ``sounds/C4.wav``).
    extension : str
        File extension of your audio files (default ``".wav"``).
    """

    NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    def __init__(
        self,
        sounds_dir: str | Path = DEFAULT_SOUNDS_DIR,
        extension:  str        = DEFAULT_EXTENSION,
    ) -> None:
        if not _PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for audio playback.\n"
                "Install it with:  pip install pygame"
            )

        self.sounds_dir = Path(sounds_dir)
        self.extension  = extension
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
                + f"\n\nExpected folder: '{self.sounds_dir}'"
                  f"\nExpected naming: C4.wav, D4.wav, ... C5.wav"
            )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_note(self, note: str, mode: str = "INPUT") -> None:
        """
        Play a single note and block for the mode's hold duration + gap.

        Parameters
        ----------
        note : str
            Note name, e.g. ``"C4"``.
        mode : str
            One of ``"INPUT"``, ``"SAFE"``, ``"CREATIVE"``, ``"UNEXPECTED"``.
            Controls how long the note rings and the gap after it.
        """
        if note not in self._sounds:
            raise ValueError(
                f"'{note}' has no loaded sound. Valid notes: {self.NOTES}"
            )

        hold = DURATION_PROFILES.get(mode, DURATION_PROFILES["INPUT"])
        gap  = GAP_PROFILES.get(mode, GAP_PROFILES["INPUT"])

        self._sounds[note].play()
        time.sleep(hold)
        self._sounds[note].stop()
        time.sleep(gap)

    def play_sequence(
        self,
        notes: list[str],
        mode:  str = "INPUT",
        label: str = "",
    ) -> None:
        """
        Play a list of notes one after another using the given mode's timing.

        Parameters
        ----------
        notes : list[str]
            Note names to play.
        mode : str
            Duration profile — ``"SAFE"``, ``"CREATIVE"``, or ``"UNEXPECTED"``.
        label : str
            Optional label printed before playback.
        """
        if label:
            hold = DURATION_PROFILES.get(mode, 0.5)
            print(f"  ♪  {label:<12} ({hold}s/note) : {' → '.join(notes)}")

        for note in notes:
            self.play_note(note, mode=mode)

    def play_result(
        self,
        result,                        # TuneGenResult from engine.py
        play_input:      bool = True,
        play_safe:       bool = True,
        play_creative:   bool = True,
        play_unexpected: bool = True,
        pause_between:   float = 0.5,  # silence between sections
    ) -> None:
        """
        Play all (or selected) sections of a TuneGenResult in order:
        INPUT → SAFE → CREATIVE → UNEXPECTED.

        Each section uses its own duration profile automatically.

        Parameters
        ----------
        result : TuneGenResult
            Returned by ``engine.generate()``.
        play_input / play_safe / play_creative / play_unexpected : bool
            Toggle individual sections on or off.
        pause_between : float
            Seconds of silence between sections.
        """
        sections: list[tuple[str, list[str]]] = []

        if play_input:      sections.append(("INPUT",      result.input_sequence))
        if play_safe:       sections.append(("SAFE",       result.safe))
        if play_creative:   sections.append(("CREATIVE",   result.creative))
        if play_unexpected: sections.append(("UNEXPECTED", result.unexpected))

        print(f"\n  ► {result.chord_detected}  |  {result.contour} contour\n")

        for mode, notes in sections:
            self.play_sequence(notes, mode=mode, label=mode)
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
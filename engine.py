"""
engine.py — TuneGen Core
=========================
All music theory logic lives here:
  - Scale & chord definitions
  - Input validation
  - Chord detection
  - Contour analysis
  - Context builder
  - Three continuation generators (SAFE, CREATIVE, UNEXPECTED)
  - Result dataclass

Nothing in this file plays audio or touches the filesystem beyond
returning plain Python data structures.
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Scale & chord definitions
# ---------------------------------------------------------------------------

C_MAJOR_SCALE: list[str] = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

SCALE_INDEX: dict[str, int] = {note: i for i, note in enumerate(C_MAJOR_SCALE)}

CHORD_SETS: dict[str, frozenset[str]] = {
    "C_MAJOR": frozenset({"C4", "E4", "G4"}),
    "F_MAJOR": frozenset({"F4", "A4", "C5"}),
    "G_MAJOR": frozenset({"G4", "B4", "D4"}),
}

# Preferred resolution order per chord (root → third → fifth)
RESOLUTION_TARGETS: dict[str, list[str]] = {
    "C_MAJOR": ["C4", "E4", "G4"],
    "F_MAJOR": ["F4", "A4", "C5"],
    "G_MAJOR": ["G4", "B4", "D4"],
    "UNKNOWN": ["C4", "E4", "G4"],
}


# ---------------------------------------------------------------------------
# Result dataclass (returned by generate())
# ---------------------------------------------------------------------------

@dataclass
class TuneGenResult:
    input_sequence: list[str]
    chord_detected: str
    chord_tones:    list[str]
    contour:        str
    avg_interval:   float
    safe:           list[str]
    creative:       list[str]
    unexpected:     list[str]


# ---------------------------------------------------------------------------
# Internal context (private to engine)
# ---------------------------------------------------------------------------

@dataclass
class _MusicContext:
    input_seq:       list[str]
    chord_name:      str
    chord_tones:     frozenset[str]
    non_chord_tones: list[str]
    contour:         str
    last_note:       str
    last_index:      int
    avg_interval:    float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(index: int) -> int:
    return max(0, min(len(C_MAJOR_SCALE) - 1, index))

def _note_at(index: int) -> str:
    return C_MAJOR_SCALE[_clamp(index)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input(seq: list[str]) -> None:
    """Raise ValueError if seq is not 3 valid C Major notes."""
    if len(seq) != 3:
        raise ValueError(f"Input must be exactly 3 notes; got {len(seq)}.")
    for note in seq:
        if note not in SCALE_INDEX:
            raise ValueError(
                f"'{note}' is not in the C Major scale (C4–C5). "
                f"Valid notes: {C_MAJOR_SCALE}"
            )


# ---------------------------------------------------------------------------
# Chord detection
# ---------------------------------------------------------------------------

def detect_chord(seq: list[str]) -> tuple[str, frozenset[str]]:
    """
    Exact subset match first; fall back to partial match (≥ 2 tones).
    Returns (chord_name, chord_tone_set).
    """
    note_set = frozenset(seq)

    for chord_name, tones in CHORD_SETS.items():
        if note_set.issubset(tones) or tones.issubset(note_set):
            return chord_name, tones

    best_chord, best_count = "UNKNOWN", 0
    for chord_name, tones in CHORD_SETS.items():
        overlap = len(note_set & tones)
        if overlap > best_count:
            best_chord, best_count = chord_name, overlap

    if best_count >= 2:
        return best_chord, CHORD_SETS[best_chord]

    return "UNKNOWN", frozenset(RESOLUTION_TARGETS["UNKNOWN"])


# ---------------------------------------------------------------------------
# Contour analysis
# ---------------------------------------------------------------------------

def analyze_contour(seq: list[str]) -> tuple[str, float]:
    """Return (contour_label, mean_absolute_interval_in_scale_steps)."""
    indices = [SCALE_INDEX[n] for n in seq]
    deltas  = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    mean_leap = sum(abs(d) for d in deltas) / len(deltas)

    if all(d > 0 for d in deltas):   return "ascending",  mean_leap
    if all(d < 0 for d in deltas):   return "descending", mean_leap
    if all(d == 0 for d in deltas):  return "static",     mean_leap
    if deltas[0] > 0 and deltas[-1] < 0: return "arch",   mean_leap
    if deltas[0] < 0 and deltas[-1] > 0: return "valley", mean_leap
    return "mixed", mean_leap


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(seq: list[str]) -> _MusicContext:
    validate_input(seq)
    chord_name, chord_tones = detect_chord(seq)
    non_chord = [n for n in C_MAJOR_SCALE if n not in chord_tones]
    contour, avg_interval = analyze_contour(seq)

    return _MusicContext(
        input_seq       = seq,
        chord_name      = chord_name,
        chord_tones     = chord_tones,
        non_chord_tones = non_chord,
        contour         = contour,
        last_note       = seq[-1],
        last_index      = SCALE_INDEX[seq[-1]],
        avg_interval    = avg_interval,
    )


# ---------------------------------------------------------------------------
# Generator — SAFE
# ---------------------------------------------------------------------------

def _generate_safe(ctx: _MusicContext) -> list[str]:
    """
    Stepwise motion (±1 scale step), snaps off-chord notes to the nearest
    chord tone, always resolves to the chord root on note 3.
    """
    resolution = RESOLUTION_TARGETS[ctx.chord_name]
    direction  = 1 if ctx.contour in ("ascending", "valley") else -1
    result: list[str] = []
    cur = ctx.last_index

    for step in range(3):
        if step == 2:
            result.append(resolution[0])
            break

        candidate_idx  = _clamp(cur + direction)
        candidate      = _note_at(candidate_idx)

        if candidate not in ctx.chord_tones:
            nearest = min(
                ctx.chord_tones,
                key=lambda n: abs(SCALE_INDEX[n] - candidate_idx)
            )
            prev_note = result[-1] if result else ctx.last_note
            if nearest != prev_note:
                candidate     = nearest
                candidate_idx = SCALE_INDEX[candidate]
            # else: keep the scale tone to avoid repeating

        result.append(candidate)
        cur       = candidate_idx
        direction *= -1

    return result


# ---------------------------------------------------------------------------
# Generator — CREATIVE
# ---------------------------------------------------------------------------

def _generate_creative(ctx: _MusicContext) -> list[str]:
    """
    Contour-driven skip table (±2–3 scale steps).
    Even steps prefer chord tones; odd steps prefer non-chord tones.

    Bug fix: chord-snapping is skipped when it would repeat the previous
    note — the skip target is kept as-is instead, which is still a valid
    C Major scale tone and produces better melodic variety.
    """
    non_chord = ctx.non_chord_tones

    skips = {
        "ascending":  [+2, -1, +3],
        "descending": [-2, +1, -3],
        "arch":       [+2, -3, +2],
        "valley":     [-2, +3, -2],
        "static":     [+3, -2, +3],
        "mixed":      [+2, +3, -1],
    }
    pattern = skips.get(ctx.contour, [+2, -1, +2])

    result: list[str] = []
    cur = ctx.last_index

    for i, skip in enumerate(pattern):
        next_idx  = _clamp(cur + skip)
        next_note = _note_at(next_idx)

        # Reference note: whatever came before this step
        prev_note = result[-1] if result else ctx.last_note

        # Even steps — prefer a chord tone, but only snap if it won't repeat
        if i % 2 == 0 and next_note not in ctx.chord_tones:
            nearest_chord = min(
                ctx.chord_tones,
                key=lambda n: abs(SCALE_INDEX[n] - next_idx)
            )
            if nearest_chord != prev_note:
                next_note = nearest_chord
                next_idx  = SCALE_INDEX[next_note]
            # else: keep the skip target — it's still in scale, just not a chord tone

        # Odd steps — prefer a non-chord tone
        elif i % 2 == 1 and next_note in ctx.chord_tones and non_chord:
            next_note = min(
                non_chord,
                key=lambda n: abs(SCALE_INDEX[n] - next_idx)
            )
            next_idx = SCALE_INDEX[next_note]

        result.append(next_note)
        cur = next_idx

    return result


# ---------------------------------------------------------------------------
# Generator — UNEXPECTED
# ---------------------------------------------------------------------------

def _generate_unexpected(ctx: _MusicContext) -> list[str]:
    """
    Contrarian leap table — inverts the source contour direction.
    Every note must be a non-chord scale degree with no repetitions.

    Bug fix: the old offset walk could loop back to the same note when
    large downward leaps clamped to the scale floor repeatedly.
    Now we maintain a 'used' set and select from the remaining non-chord
    pool, favouring notes in the leap direction when possible.
    """
    non_chord_sorted = sorted(
        ctx.non_chord_tones or [C_MAJOR_SCALE[0], C_MAJOR_SCALE[-1]],
        key=lambda n: SCALE_INDEX[n]
    )

    contra = {
        "ascending":  [-3, -4, -2],
        "descending": [+3, +4, +2],
        "arch":       [-4, +3, -3],
        "valley":     [+4, -3, +3],
        "static":     [+4, -4, +3],
        "mixed":      [-3, +4, -4],
    }
    pattern = contra.get(ctx.contour, [-3, +4, -3])

    result: list[str] = []
    used:   set[str]  = set()
    cur = ctx.last_index

    for leap in pattern:
        next_idx  = _clamp(cur + leap)
        next_note = _note_at(next_idx)

        # If the ideal target is a chord tone or already used, find a replacement
        if next_note in ctx.chord_tones or next_note in used:
            # Pool: unused non-chord tones; fall back to all non-chord if exhausted
            pool = [n for n in non_chord_sorted if n not in used]
            if not pool:
                pool = non_chord_sorted

            if leap < 0:
                # Wanted to go down — pick the lowest available pool note
                # that is still below current position if possible
                below = [n for n in pool if SCALE_INDEX[n] < cur]
                next_note = below[0] if below else pool[0]
            else:
                # Wanted to go up — pick the highest available pool note
                # that is still above current position if possible
                above = [n for n in pool if SCALE_INDEX[n] > cur]
                next_note = above[-1] if above else pool[-1]

            next_idx = SCALE_INDEX[next_note]

        result.append(next_note)
        used.add(next_note)
        cur = next_idx

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(input_seq: list[str]) -> TuneGenResult:
    """
    Full pipeline: validate → detect chord → analyse contour →
    produce SAFE, CREATIVE, and UNEXPECTED continuations.

    Parameters
    ----------
    input_seq : list[str]
        Exactly 3 notes from C_MAJOR_SCALE (e.g. ["C4", "E4", "G4"]).

    Returns
    -------
    TuneGenResult
        Structured result containing all analysis data and the three
        3-note continuations.
    """
    ctx = _build_context(input_seq)

    return TuneGenResult(
        input_sequence = ctx.input_seq,
        chord_detected = ctx.chord_name,
        chord_tones    = sorted(ctx.chord_tones, key=lambda n: SCALE_INDEX[n]),
        contour        = ctx.contour,
        avg_interval   = round(ctx.avg_interval, 2),
        safe           = _generate_safe(ctx),
        creative       = _generate_creative(ctx),
        unexpected     = _generate_unexpected(ctx),
    )
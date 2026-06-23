import random

# ----------------------------
# NOTE SPACE
# ----------------------------
NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]


# ----------------------------
# HELPERS
# ----------------------------
def idx(note):
    return NOTES.index(note)


def interval(a, b):
    return idx(b) - idx(a)


# ----------------------------
# SAFE GENERATOR
# ----------------------------
def generate_safe(sequence):
    last = sequence[-1]
    last_i = idx(last)

    candidates = []

    for n in NOTES:
        i = idx(n)

        # no repetition
        if n == last:
            continue

        # small step movement only
        if abs(i - last_i) <= 1:
            candidates.append(n)

    if not candidates:
        candidates = [n for n in NOTES if n != last]

    return candidates[:3]


# ----------------------------
# CREATIVE GENERATOR
# ----------------------------
def generate_creative(sequence):
    last = sequence[-1]
    last_i = idx(last)

    candidates = []

    for n in NOTES:
        i = idx(n)

        # allow medium jumps
        if abs(i - last_i) <= 3:
            candidates.append(n)

    # avoid repetition dominance
    candidates = [n for n in candidates if n != last]

    if not candidates:
        candidates = [n for n in NOTES if n != last]

    random.shuffle(candidates)

    return candidates[:3]


# ----------------------------
# UNEXPECTED GENERATOR
# ----------------------------
def generate_unexpected(sequence):
    last = sequence[-1]
    last_i = idx(last)

    direction = 0
    if len(sequence) >= 2:
        direction = idx(sequence[-1]) - idx(sequence[-2])

    candidates = []

    for n in NOTES:
        i = idx(n)

        # break motion expectation
        if direction > 0 and i <= last_i:
            candidates.append(n)
        elif direction < 0 and i >= last_i:
            candidates.append(n)
        else:
            candidates.append(n)

    # remove repetition loops
    candidates = [n for n in candidates if n != last]

    if not candidates:
        candidates = [n for n in NOTES if n != last]

    random.shuffle(candidates)

    return candidates[:3]


# ----------------------------
# MAIN SUGGEST FUNCTION
# ----------------------------
def suggest(sequence):

    safe = generate_safe(sequence)
    creative = generate_creative(sequence)
    unexpected = generate_unexpected(sequence)

    return {
        "safe": safe,
        "creative": creative,
        "unexpected": unexpected
    }
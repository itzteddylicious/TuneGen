NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

SCALE = {"C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"}

def interval_score(last_note, candidate):
    last = NOTES.index(last_note)
    cand = NOTES.index(candidate)
    diff = abs(cand - last)

    if diff == 0:
        return 3  # repetition = stable
    elif diff <= 1:
        return 5  # stepwise = very safe
    elif diff <= 2:
        return 3  # small leap = ok
    elif diff <= 4:
        return 1  # larger leap = creative
    else:
        return -2  # big jump = tense
    
def scale_score(note):
    return 2 if note in SCALE else -3

def score(last_note, candidate):
    return interval_score(last_note, candidate) + scale_score(candidate)

def suggest(last_notes):
    last = last_notes[-1]

    scored = [(note, score(last, note)) for note in NOTES]

    safe = [n for n, s in scored if s >= 7]
    creative = [n for n, s in scored if 4 <= s < 7]
    unexpected = [n for n, s in scored if s < 4]

    return {
        "safe": safe[:3],
        "creative": creative[:3],
        "unexpected": unexpected[:3]
    }
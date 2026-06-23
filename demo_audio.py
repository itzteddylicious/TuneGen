import pygame
import time
from engine import suggest, NOTES

pygame.mixer.init()

sounds = {
    "C4": pygame.mixer.Sound("sounds/C4.wav"),
    "D4": pygame.mixer.Sound("sounds/D4.wav"),
    "E4": pygame.mixer.Sound("sounds/E4.wav"),
    "F4": pygame.mixer.Sound("sounds/F4.wav"),
    "G4": pygame.mixer.Sound("sounds/G4.wav"),
    "A4": pygame.mixer.Sound("sounds/A4.wav"),
    "B4": pygame.mixer.Sound("sounds/B4.wav"),
    "C5": pygame.mixer.Sound("sounds/C5.wav"),
}


def play(note, delay=0.35):
    sounds[note].play()
    time.sleep(delay)


def play_note_sequence(seq):
    for n in seq:
        play(n)


def generate_chain(base, category, steps=4):
    """
    Builds a full melody by repeatedly extending the sequence.
    """
    seq = base.copy()

    for _ in range(steps):
        result = suggest(seq)

        options = result[category]

        print(category, options)

        if not options:
            break

        next_note = options[0]
        seq.append(next_note)

    return seq


def play_category(label, seq):
    print(f"\n=== {label} ===")
    print(" -> ".join(seq))
    play_note_sequence(seq)
    time.sleep(1)


if __name__ == "__main__":

    user_input = input(
        "Enter exactly 3 notes separated by spaces (e.g. C4 E4 G4): "
    )

    base = [note.upper() for note in user_input.split()]

    if len(base) != 3:
        print("Please enter exactly 3 notes.")
        exit()

    for note in base:
        if note not in NOTES:
            print(f"Invalid note: {note}")
            print(f"Valid notes are: {', '.join(NOTES)}")
            exit()

    print("\nINPUT:")
    print(" -> ".join(base))
    play_note_sequence(base)

    time.sleep(1)

    safe_melody = generate_chain(base, "safe")
    creative_melody = generate_chain(base, "creative")
    unexpected_melody = generate_chain(base, "unexpected")

    print("\nGenerated Melodies:")
    print("SAFE:", " -> ".join(safe_melody))
    print("CREATIVE:", " -> ".join(creative_melody))
    print("UNEXPECTED:", " -> ".join(unexpected_melody))

    time.sleep(1)

    play_category("SAFE", safe_melody)
    play_category("CREATIVE", creative_melody)
    play_category("UNEXPECTED", unexpected_melody)
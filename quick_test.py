from engine import suggest

sequence = ["C4", "E4", "G4"]

result = suggest(sequence)

print("SAFE:", result["safe"])
print("CREATIVE:", result["creative"])
print("UNEXPECTED:", result["unexpected"])
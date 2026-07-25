import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
LEXICON_PATH = "datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
INPUT_FILE = "outputs/communication_features.csv"
OUTPUT_FILE = "outputs/emotion_features.csv"

print("Loading NRC Emotion Lexicon...")

lexicon = {}

with open(LEXICON_PATH, "r", encoding="utf-8") as file:
    for line in file:
        word, emotion, association = line.strip().split("\t")

        if association == "1":
            lexicon.setdefault(word, []).append(emotion)

print("Lexicon Loaded!")

# -----------------------------
# Load communication features
# -----------------------------
df = pd.read_csv(INPUT_FILE)

emotion_labels = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

results = []

print("Performing emotion analysis...")

for text in df["clean_text"]:

    scores = {emotion: 0 for emotion in emotion_labels}

    words = str(text).lower().split()

    for word in words:
        if word in lexicon:
            for emotion in lexicon[word]:
                if emotion in scores:
                    scores[emotion] += 1

    results.append(scores)

emotion_df = pd.DataFrame(results)

final_df = pd.concat([df, emotion_df], axis=1)

os.makedirs("outputs", exist_ok=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nEmotion Analysis Complete!\n")
print(
    final_df[
        [
            "clean_text",
            "sentiment",
            "anger",
            "fear",
            "joy",
            "sadness"
        ]
    ].head()
)

print(f"\nSaved to {OUTPUT_FILE}")
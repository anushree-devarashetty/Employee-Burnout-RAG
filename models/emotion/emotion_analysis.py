import os
import pandas as pd

# File paths
LEXICON_PATH = "datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
INPUT_FILE = "outputs/communication_features.csv"
OUTPUT_FILE = "outputs/emotion_features.csv"

# Load NRC Emotion Lexicon
print("Loading NRC Emotion Lexicon...")

lexicon = {}

with open(LEXICON_PATH, "r", encoding="utf-8") as file:
    for line in file:
        word, emotion, association = line.strip().split("\t")

        if association == "1":
            if word not in lexicon:
                lexicon[word] = []

            lexicon[word].append(emotion)

print("Lexicon Loaded!")

# Load communication features
df = pd.read_csv(INPUT_FILE)

emotions = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

# Count emotions in each sentence
results = []

for text in df["clean_text"]:

    scores = {emotion: 0 for emotion in emotions}

    words = str(text).lower().split()

    for word in words:

        if word in lexicon:

            for emotion in lexicon[word]:

                if emotion in scores:
                    scores[emotion] += 1

    results.append(scores)

emotion_df = pd.DataFrame(results)

# Merge with original data
final_df = pd.concat([df, emotion_df], axis=1)

os.makedirs("outputs", exist_ok=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nEmotion Analysis Complete!")
print(final_df.head())

print(f"\nSaved to {OUTPUT_FILE}")
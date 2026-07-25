import os
import pandas as pd

INPUT_FILE = "outputs/emotion_features.csv"
OUTPUT_FILE = "outputs/behavior_features.csv"

print("Loading emotion features...")

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Behavioral Features
# -----------------------------

df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))

df["char_count"] = df["clean_text"].apply(lambda x: len(str(x)))

df["avg_word_length"] = df["clean_text"].apply(
    lambda x: round(
        sum(len(word) for word in str(x).split()) /
        max(len(str(x).split()), 1),
        2
    )
)

df["sentence_count"] = df["message"].apply(
    lambda x: len(
        [s for s in str(x).split(".") if s.strip()]
    )
)

df["exclamation_count"] = df["message"].apply(
    lambda x: str(x).count("!")
)

df["question_count"] = df["message"].apply(
    lambda x: str(x).count("?")
)


def uppercase_ratio(text):

    text = str(text)

    letters = [c for c in text if c.isalpha()]

    if len(letters) == 0:
        return 0

    upper = sum(c.isupper() for c in letters)

    return round(upper / len(letters), 3)


df["uppercase_ratio"] = df["message"].apply(uppercase_ratio)

os.makedirs("outputs", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nBehavior Feature Extraction Complete!\n")

print(
    df[
        [
            "word_count",
            "char_count",
            "avg_word_length",
            "sentence_count",
            "exclamation_count",
            "question_count",
            "uppercase_ratio"
        ]
    ].head()
)

print(f"\nSaved to {OUTPUT_FILE}")
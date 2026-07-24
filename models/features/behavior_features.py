import os
import pandas as pd

INPUT_FILE = "outputs/emotion_features.csv"
OUTPUT_FILE = "outputs/behavior_features.csv"

# Load data
df = pd.read_csv(INPUT_FILE)

# Behavioral features
df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["clean_text"].apply(lambda x: len(str(x)))
df["avg_word_length"] = df["clean_text"].apply(
    lambda x: sum(len(word) for word in str(x).split()) / max(len(str(x).split()), 1)
)

df["exclamation_count"] = df["clean_text"].apply(lambda x: str(x).count("!"))
df["question_count"] = df["clean_text"].apply(lambda x: str(x).count("?"))

# Uppercase ratio
def uppercase_ratio(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0
    upper = sum(c.isupper() for c in letters)
    return round(upper / len(letters), 3)

df["uppercase_ratio"] = df["clean_text"].apply(uppercase_ratio)

# Save
os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nBehavioral Feature Extraction Complete!\n")
print(df.head())

print(f"\nSaved to {OUTPUT_FILE}")
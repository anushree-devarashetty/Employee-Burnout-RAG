import os
import pandas as pd

INPUT_FILE = "outputs/employee_features.csv"
OUTPUT_FILE = "outputs/burnout_scores.csv"

df = pd.read_csv(INPUT_FILE)


def calculate_score(row):
    score = 0

    # Sentiment
    if row["sentiment"] == "Negative":
        score += 2
    elif row["sentiment"] == "Positive":
        score -= 1

    # Emotions
    score += row["fear"]
    score += row["sadness"]
    score += row["anger"]

    # Positive emotions reduce burnout
    score -= row["joy"]

    # Long emails
    if row["word_count"] > 25:
        score += 1

    # Excessive punctuation
    if row["exclamation_count"] > 2:
        score += 1

    return max(score, 0)


def risk_level(score):
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Medium"
    else:
        return "High"


df["burnout_score"] = df.apply(calculate_score, axis=1)
df["risk_level"] = df["burnout_score"].apply(risk_level)

os.makedirs("outputs", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nBurnout Prediction Complete!\n")
print(df[["clean_text", "burnout_score", "risk_level"]])

print(f"\nSaved to {OUTPUT_FILE}")
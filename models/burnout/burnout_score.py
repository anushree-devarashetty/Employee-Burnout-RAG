import os
import pandas as pd

INPUT_FILE = "outputs/employee_features.csv"
OUTPUT_FILE = "outputs/burnout_scores.csv"

print("Loading employee features...")

df = pd.read_csv(INPUT_FILE)


# -----------------------------
# Burnout Score
# -----------------------------
def calculate_score(row):

    score = 0

    # Sentiment
    if row["sentiment"] == "Negative":
        score += 2
    elif row["sentiment"] == "Neutral":
        score += 1
    elif row["sentiment"] == "Positive":
        score -= 1

    # Emotions
    score += row["anger"]
    score += row["fear"]
    score += row["sadness"]

    score -= row["joy"]

    # Behavioural Features
    if row["word_count"] > 120:
        score += 1

    if row["sentence_count"] > 8:
        score += 1

    if row["exclamation_count"] > 2:
        score += 1

    if row["uppercase_ratio"] > 0.25:
        score += 1

    return max(score, 0)


# -----------------------------
# Risk Level
# -----------------------------
def risk_level(score):

    if score <= 2:
        return "Low"

    elif score <= 5:
        return "Medium"

    else:
        return "High"


# -----------------------------
# Explainability
# -----------------------------
def explanation(row):

    reasons = []

    if row["sentiment"] == "Negative":
        reasons.append("Negative sentiment")

    elif row["sentiment"] == "Neutral":
        reasons.append("Neutral sentiment")

    if row["anger"] > 0:
        reasons.append("Anger detected")

    if row["fear"] > 0:
        reasons.append("Fear detected")

    if row["sadness"] > 0:
        reasons.append("Sadness detected")

    if row["joy"] > 0:
        reasons.append("Positive emotion detected")

    if row["word_count"] > 120:
        reasons.append("Long email")

    if row["sentence_count"] > 8:
        reasons.append("Many sentences")

    if row["uppercase_ratio"] > 0.25:
        reasons.append("High uppercase usage")

    if row["exclamation_count"] > 2:
        reasons.append("Frequent exclamation marks")

    if len(reasons) == 0:
        reasons.append("No significant burnout indicators")

    return ", ".join(reasons)


# -----------------------------
# Apply
# -----------------------------
df["burnout_score"] = df.apply(calculate_score, axis=1)

df["risk_level"] = df["burnout_score"].apply(risk_level)

df["explanation"] = df.apply(explanation, axis=1)


# -----------------------------
# Save
# -----------------------------
os.makedirs("outputs", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nBurnout Prediction Complete!\n")

print(
    df[
        [
            "sentiment",
            "burnout_score",
            "risk_level",
            "explanation"
        ]
    ].head(10)
)

print(f"\nSaved to {OUTPUT_FILE}")
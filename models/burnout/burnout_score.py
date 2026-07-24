import os
import pandas as pd

INPUT_FILE = "outputs/employee_features.csv"
OUTPUT_FILE = "outputs/burnout_scores.csv"

print("Loading employee features...")

df = pd.read_csv(INPUT_FILE)


# -----------------------------
# Burnout Score Calculation
# -----------------------------
def calculate_score(row):
    score = 0

    # Sentiment
    if row["sentiment"] == "Negative":
        score += 2
    elif row["sentiment"] == "Positive":
        score -= 1

    # Emotion Scores
    score += row["anger"]
    score += row["fear"]
    score += row["sadness"]

    # Positive emotion reduces burnout
    score -= row["joy"]

    # Behavioral Features
    if row["word_count"] > 25:
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
# Explainable AI
# -----------------------------
def explanation(row):

    reasons = []

    if row["sentiment"] == "Negative":
        reasons.append("Negative sentiment detected")

    if row["anger"] > 0:
        reasons.append("Anger-related words found")

    if row["fear"] > 0:
        reasons.append("Fear-related words found")

    if row["sadness"] > 0:
        reasons.append("Sadness-related words found")

    if row["joy"] > 0:
        reasons.append("Positive emotional indicators")

    if row["word_count"] > 25:
        reasons.append("Long email")

    if row["exclamation_count"] > 2:
        reasons.append("Frequent exclamation marks")

    if row["uppercase_ratio"] > 0.25:
        reasons.append("High uppercase usage")

    if len(reasons) == 0:
        reasons.append("No significant burnout indicators")

    return ", ".join(reasons)


# -----------------------------
# Apply Functions
# -----------------------------
df["burnout_score"] = df.apply(calculate_score, axis=1)

df["risk_level"] = df["burnout_score"].apply(risk_level)

df["explanation"] = df.apply(explanation, axis=1)


# -----------------------------
# Save Results
# -----------------------------
os.makedirs("outputs", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)


# -----------------------------
# Display Results
# -----------------------------
print("\nBurnout Prediction Complete!\n")

print(
    df[
        [
            "clean_text",
            "sentiment",
            "burnout_score",
            "risk_level",
            "explanation",
        ]
    ]
)

print(f"\nResults saved to: {OUTPUT_FILE}")
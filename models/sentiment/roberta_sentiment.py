import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = "datasets/processed/enron_clean.csv"
OUTPUT_FILE = "outputs/communication_features.csv"

# -----------------------------
# Load dataset
# -----------------------------
print("Loading processed Enron dataset...")

df = pd.read_csv(INPUT_FILE)

# For development, use only first 1000 emails.
# Remove this line when you're ready to process the full dataset.
df = df.head(1000)

# -----------------------------
# Load RoBERTa
# -----------------------------
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = ["Negative", "Neutral", "Positive"]

sentiments = []
confidences = []

# -----------------------------
# Sentiment Prediction
# -----------------------------
for text in df["clean_text"]:

    text = str(text)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model(**encoded)

    scores = softmax(output.logits.detach().numpy()[0])

    index = scores.argmax()

    sentiments.append(labels[index])
    confidences.append(round(float(scores[index]), 4))

# -----------------------------
# Save Results
# -----------------------------
df["sentiment"] = sentiments
df["confidence"] = confidences

os.makedirs("outputs", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSentiment Analysis Complete!\n")
print(df[["clean_text", "sentiment", "confidence"]].head())

print(f"\nSaved to {OUTPUT_FILE}")
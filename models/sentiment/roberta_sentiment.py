import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["Negative", "Neutral", "Positive"]


def predict_sentiment(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = model(**encoded)

    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)

    sentiment = labels[scores.argmax()]
    confidence = float(scores.max())

    return sentiment, confidence


df = pd.read_csv("datasets/sample.csv")

sentiments = []
confidences = []

for text in df["clean_text"]:
    sentiment, confidence = predict_sentiment(str(text))
    sentiments.append(sentiment)
    confidences.append(confidence)

df["sentiment"] = sentiments
df["confidence"] = confidences

df.to_csv("outputs/communication_features.csv", index=False)

print(df)
print("\nSaved to outputs/communication_features.csv")
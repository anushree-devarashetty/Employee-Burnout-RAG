import os
import pandas as pd

INPUT_FILE = "outputs/behavior_features.csv"
OUTPUT_FILE = "outputs/employee_features.csv"

print("Loading behavioral features...")

df = pd.read_csv(INPUT_FILE)

# Future integration point:
# Merge Person 1's burnout dataset or stress dataset here if needed.
# For now, behavioral_features.csv already contains all previous features.

final_df = df.copy()

os.makedirs("outputs", exist_ok=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature Fusion Complete!\n")

print(final_df.head())

print(f"\nSaved to {OUTPUT_FILE}")
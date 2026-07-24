import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    if "clean_text" not in df.columns:
        raise ValueError("Dataset must contain 'clean_text' column.")

    return df

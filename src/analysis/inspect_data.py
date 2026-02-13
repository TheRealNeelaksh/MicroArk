import pandas as pd

try:
    df = pd.read_csv('data/metalog_raw.csv')
    print("Columns found:")
    for col in df.columns:
        print(col)
except Exception as e:
    print(f"Error: {e}")

import requests
import pandas as pd
import gzip
import io

url = "https://metalog.embl.de/static/download/metadata/animal_extended_wide_latest.tsv.gz"
output_path = "data/metalog_raw.csv"

print(f"Downloading from {url}...")
try:
    response = requests.get(url)
    response.raise_for_status()
    
    print("Decompressing and loading into DataFrame...")
    with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t')
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Success.")

except Exception as e:
    print(f"Error: {e}")

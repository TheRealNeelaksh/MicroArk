import requests
import os

url = "https://metalog.embl.de/api/samples/animal.csv"
output_path = "data/metalog_raw.csv"

print(f"Downloading from {url}...")
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Successfully downloaded to {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

except Exception as e:
    print(f"Error downloading file: {e}")

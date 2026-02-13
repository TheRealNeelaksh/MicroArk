import pandas as pd
import os

from pathlib import Path

def process_data():
    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / "data" / "metalog_raw.csv"
    output_file = project_root / "data" / "species_features.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading data...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Initial shape: {df.shape}")
    print(df.columns.tolist())

    # Map columns
    # host_species <-- host_tax_id
    if 'host_tax_id' in df.columns:
        df['host_species'] = df['host_tax_id']
    else:
        print("Error: 'host_tax_id' column missing.")
        return

    # country <-- location (cleaned)
    if 'location' in df.columns:
        # Extract country from location string (e.g., "USA: California" -> "USA")
        df['country'] = df['location'].astype(str).apply(lambda x: x.split(':')[0].strip() if pd.notnull(x) else "Unknown")
    else:
        print("Warning: 'location' column missing. Using 'Unknown'.")
        df['country'] = "Unknown"

    # collection_year <-- collection_date
    if 'collection_date' in df.columns:
        # Handle mixed formats or errors. Extract year.
        # Format seems to be YYYY-MM-DD usually.
        df['collection_year'] = pd.to_datetime(df['collection_date'], errors='coerce').dt.year
    else:
        print("Warning: 'collection_date' column missing. Using 0.")
        df['collection_year'] = 0

    # Ensure coordinates exist
    if 'latitude' not in df.columns:
        df['latitude'] = 0
    if 'longitude' not in df.columns:
        df['longitude'] = 0

    # Clean
    print("Cleaning data...")
    df = df.dropna(subset=["host_species"])
    
    # Filter species with < 5 samples
    species_counts = df["host_species"].value_counts()
    valid_species = species_counts[species_counts >= 5].index
    df = df[df["host_species"].isin(valid_species)]
    print(f"Retained {len(valid_species)} species with >= 5 samples. Total samples: {len(df)}")

    # Create Features
    print("Creating features...")
    grouped = df.groupby("host_species")

    features = pd.DataFrame({
        "num_samples": grouped.size(),
        "num_countries": grouped["country"].nunique(),
        "year_span": grouped["collection_year"].max() - grouped["collection_year"].min(),
        "lat_variance": grouped["latitude"].var(),
        "long_variance": grouped["longitude"].var()
    })

    features = features.fillna(0)
    features.reset_index(inplace=True)

    # Save
    features.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")
    print(features.head())

if __name__ == "__main__":
    process_data()

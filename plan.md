# 🌍 BioARK: Metagenomic Proxy for Extinction Risk
> **A Hackathon Approach to Biodiversity Early Warning Systems**

## 🚦 Project Overview
**BioARK** avoids the computational bottleneck of processing raw DNA sequences (FASTQ/QIIME). Instead, we hypothesize that **metadata signals** from the [Metalog](https://metalog.biom.gurdon.cam.ac.uk/) database—such as sampling density, geographic fragmentation, and temporal continuity—act as strong proxy indicators for species stress.

We combine these metadata features with **IUCN Red List** conservation statuses to build a **Random Forest** predictive model that flags species at risk of extinction.

---

## 🏗 Architecture
1.  **Data Source:** Metalog Metadata (Host species, Location, Time).
2.  **Proxy Logic:** Ecological instability is reflected in microbiome sampling variability and fragmentation.
3.  **Target Variable:** IUCN Red List Categories (LC, NT, VU, EN, CR).
4.  **Model:** Random Forest Classifier.
5.  **Interface:** Streamlit Dashboard.

---

## 📥 Step 1: Data Acquisition (Manual)
Since we are avoiding raw reads, we fetch metadata directly:
1.  Navigate to **Metalog** > **Animal Samples**.
2.  Click **"Explore all metagenomic samples"**.
3.  **Filter Logic:**
    * Select `Host Species` with >10 samples.
    * Ensure diversity in `Collection Year` and `Country`.
4.  **Export:** Download the metadata as `.csv`.

---

## 🧬 Step 2: Data Preprocessing & Feature Engineering
We transform raw sample rows into a **Species-Level Dataset**. This script groups samples to calculate density, spread, and temporal span.

**`data_processing.py`**
```python
import pandas as pd
import numpy as np

def build_species_dataset(input_csv):
    # Load raw Metalog export
    df = pd.read_csv(input_csv)
    
    # 1. Group by Host Species
    # We calculate features that serve as proxies for ecological health
    species_df = df.groupby('host_species').agg({
        'sample_id': 'count',              # Proxy: Population abundance/interest
        'country': 'nunique',              # Proxy: Geographic Range
        'collection_year': lambda x: x.max() - x.min(), # Proxy: Temporal resilience
        'latitude': 'std',                 # Proxy: Geographic spread (North/South)
        'longitude': 'std'                 # Proxy: Geographic spread (East/West)
    }).reset_index()

    # Rename columns for clarity
    species_df.columns = [
        'species', 'num_samples', 'num_countries', 
        'year_span', 'lat_variance', 'long_variance'
    ]

    # Fill NaN variance (for species with only 1 location) with 0
    species_df = species_df.fillna(0)
    
    # Calculate a composite 'Habitat Fragmentation Score'
    # (Inverse of spread relative to sample count)
    species_df['fragmentation_index'] = (
        species_df['num_samples'] / (species_df['num_countries'] + 1)
    )

    return species_df

# Example usage
# dataset = build_species_dataset('metalog_export.csv')
# dataset.to_csv('processed_features.csv', index=False)
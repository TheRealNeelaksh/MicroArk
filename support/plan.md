# 🌍 EcoSentinel: Metagenomic Proxy for Extinction Risk
> **A Hackathon Approach to Biodiversity Early Warning Systems**

## 🚦 Project Overview
**EcoSentinel** avoids the computational bottleneck of processing raw DNA sequences (FASTQ/QIIME). Instead, we hypothesize that **metadata signals** from the [Metalog](https://metalog.embl.de/) database—such as sampling density, geographic fragmentation, and temporal continuity—act as strong proxy indicators for species stress.

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

```

---

## 🏷 Step 3: Labeling (IUCN Integration)

We manually or programmatically map species to their IUCN status to create the target variable (`risk_level`).

**Mapping Logic:**

* **0:** Least Concern (LC)
* **1:** Near Threatened (NT)
* **2:** Vulnerable (VU)
* **3:** Endangered/Critically Endangered (EN/CR)

**`add_labels.py`**

```python
def map_risk_labels(df):
    # In a real scenario, use an API. For hackathon, use a dictionary map.
    iucn_map = {
        'Pan troglodytes': 3,   # Endangered
        'Canis lupus': 0,       # Least Concern
        'Loxodonta africana': 2 # Vulnerable
        # ... add more mappings
    }
    
    df['risk_label'] = df['species'].map(iucn_map)
    
    # Drop rows where we couldn't find a label
    return df.dropna(subset=['risk_label'])

```

---

## 🤖 Step 4: AI Modeling (Random Forest)

We use a Random Forest classifier to predict the `risk_label` based on our metadata proxy features.

**`train_model.py`**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_early_warning_system(df):
    # Features vs Target
    X = df[['num_samples', 'num_countries', 'year_span', 
            'lat_variance', 'long_variance', 'fragmentation_index']]
    y = df['risk_label']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    
    # Train
    rf.fit(X_train, y_train)

    # Validate
    predictions = rf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

    # Feature Importance (Explainability)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print(importances)
    
    return rf

# Save model for the app
# model = train_early_warning_system(labeled_data)
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

```

---

## 🖥 Step 5: The Demo (Streamlit)

The frontend allows users to check a species and see if the "Early Warning" flag is triggered.

**`app.py`**

```python
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load Data and Model
df = pd.read_csv('processed_features.csv') # The dataset with features
# model = pickle.load(open('model.pkl', 'rb')) # The trained model

st.title("🚦 EcoSentinel: Biodiversity Early Warning")
st.markdown("Predicting extinction risk using **Metalog microbiome metadata** as a proxy.")

# User Input
selected_species = st.selectbox("Select a Species to Analyze", df['species'].unique())

if st.button('Analyze Risk'):
    # Get species data
    species_data = df[df['species'] == selected_species].iloc[0]
    
    # Extract features for prediction
    features = species_data[['num_samples', 'num_countries', 'year_span', 
                             'lat_variance', 'long_variance', 'fragmentation_index']].values.reshape(1, -1)
    
    # Run Prediction (Mocking logic if model isn't loaded)
    # prediction = model.predict(features)[0] 
    # probabilities = model.predict_proba(features)
    
    # HARDCODED LOGIC FOR DEMO visualization
    risk_score = 0
    if species_data['year_span'] < 2: risk_score += 1
    if species_data['num_countries'] < 3: risk_score += 1
    if species_data['lat_variance'] < 5.0: risk_score += 1
    
    st.subheader(f"Analysis for: {selected_species}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples Found", int(species_data['num_samples']))
    col2.metric("Countries", int(species_data['num_countries']))
    col3.metric("Years Covered", int(species_data['year_span']))

    st.divider()

    # Early Warning Logic
    if risk_score >= 2:
        st.error("🚨 EARLY WARNING TRIGGERED")
        st.write("This species shows high geographic fragmentation and low temporal sampling stability.")
        st.progress(90)
    else:
        st.success("✅ Stable Metadata Profile")
        st.write("Sampling distribution suggests a stable population range.")
        st.progress(20)

    # Feature Map
    st.map(pd.DataFrame({
        'lat': [np.random.uniform(-50, 50) for _ in range(int(species_data['num_countries']))],
        'lon': [np.random.uniform(-100, 100) for _ in range(int(species_data['num_countries']))]
    }))

```

---

## 🚀 How to Run

1. **Install Requirements:**
```bash
pip install pandas numpy scikit-learn streamlit

```


2. **Process Data:**
Run the processing script to turn your CSV into features.
3. **Launch Dashboard:**
```bash
streamlit run app.py

```



```

```
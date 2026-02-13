# 🧭 PHASE 0 — Project Setup (30–60 min)

## 🎯 Goal

Create working environment.

## ✅ Do This

1. Create new folder:

```
wildlife-ai/
```

2. Create virtual environment (optional but good):

```
python -m venv venv
```

3. Install libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

4. Create structure:

```
wildlife-ai/
│
├── data/
├── notebooks/
├── model/
├── app.py
└── train.py
```

---

# 🧭 PHASE 1 — Get Data from Metalog (3–4 hrs)

## 🎯 Goal

Download metadata CSV.

### ✅ Steps

1. Go to:
   Metalog → Animal samples → Explore all samples

2. Export metadata as CSV.

3. Save as:

```
data/metalog_raw.csv
```

---

# 🧭 PHASE 2 — Data Cleaning & Feature Creation (3–5 hrs)

## 🎯 Goal

Turn sample-level data into species-level features.

---

## 🧩 Step 2.1 — Load Data

In `train.py`:

```python
import pandas as pd

df = pd.read_csv("data/metalog_raw.csv")
print(df.head())
```

---

## 🧩 Step 2.2 — Clean

Remove:

```python
df = df.dropna(subset=["host_species"])
```

Remove species with < 5 samples:

```python
species_counts = df["host_species"].value_counts()
valid_species = species_counts[species_counts >= 5].index
df = df[df["host_species"].isin(valid_species)]
```

---

## 🧩 Step 2.3 — Create Features Per Species

Group by species:

```python
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
```

Save:

```python
features.to_csv("data/species_features.csv", index=False)
```

---

# 🧭 PHASE 3 — Add IUCN Labels (2–3 hrs)

## 🎯 Goal

Add extinction risk class.

Manually create CSV:

```
data/iucn_labels.csv
```

Format:

| host_species    | risk_label |
| --------------- | ---------- |
| Panthera tigris | 4          |
| Bos taurus      | 0          |

Mapping:
LC=0, NT=1, VU=2, EN=3, CR=4

---

## Merge:

```python
labels = pd.read_csv("data/iucn_labels.csv")
final_df = features.merge(labels, on="host_species")
```

Save:

```python
final_df.to_csv("data/final_dataset.csv", index=False)
```

---

# 🧭 PHASE 4 — Train ML Model (3–4 hrs)

## 🎯 Goal

Train Random Forest.

---

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

X = final_df.drop(["host_species", "risk_label"], axis=1)
y = final_df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(model, "model/extinction_model.pkl")
```

---

# 🧭 PHASE 5 — Add Early Warning Logic (1–2 hrs)

Add simple rule:

```python
def early_warning(row):
    if row["year_span"] < 5 and row["lat_variance"] < 1:
        return 1
    return 0

final_df["early_warning_flag"] = final_df.apply(early_warning, axis=1)
```

---

# 🧭 PHASE 6 — Explainability (2 hrs)

```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.xticks(rotation=45)
plt.show()
```

Save figure for presentation.

---

# 🧭 PHASE 7 — Build Streamlit App (3–4 hrs)

In `app.py`:

```python
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/extinction_model.pkl")
data = pd.read_csv("data/final_dataset.csv")

st.title("Wildlife Extinction Risk Predictor")

species = st.selectbox("Select Species", data["host_species"])

row = data[data["host_species"] == species]
X = row.drop(["host_species", "risk_label"], axis=1)

prediction = model.predict(X)[0]
prob = model.predict_proba(X)[0].max()

st.write("Predicted Risk Level:", prediction)
st.write("Confidence:", prob)
```

Run:

```
streamlit run app.py
```

---

# 🧭 PHASE 8 — Testing & Polishing (2–3 hrs)

* Test 5 species
* Check prediction
* Screenshot app
* Prepare demo narrative

---

# 🧠 Total Time Estimate

| Phase          | Time    |
| -------------- | ------- |
| Setup          | 1 hr    |
| Data cleaning  | 4 hr    |
| Labels         | 3 hr    |
| Model          | 4 hr    |
| Explainability | 2 hr    |
| App            | 4 hr    |
| Polish         | 2 hr    |
| **Total**      | ~20 hrs |

---

# 🎯 Final Output

You now have:

* AI model
* Early warning rule
* Web app
* Real dataset
* Conservation narrative

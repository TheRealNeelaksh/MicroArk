
## 🧭 PHASE 1 — Understand & Lock Scope (1–2 Hours)

**WHAT to do:**
Decide exactly what your system predicts.

**DO THIS:**
Write this sentence clearly:

> Our system predicts extinction risk category of animal species using microbiome metadata patterns and environmental signals.

**WHY:**
If you don’t lock scope early, you’ll drown in complexity.

---

## 📥 PHASE 2 — Collect Data from Metalog (3–4 Hours)

**WHERE:**
Go to:

* 👉 Animal Samples
* 👉 Explore all metagenomic samples

### Step 2.1 — Filter Species

**WHAT:**
Choose 10–20 species that:

* Have many samples
* Have data across multiple years
* Have multiple geographic locations

**HOW:**

* Sort by host species frequency.
* Pick the most frequent ones.

### Step 2.2 — Export Metadata

Click:

* 👉 Show metadata fields
* 👉 Export CSV

Download the full dataset.

### Step 2.3 — Clean Data

Open in Excel or Python.
**Remove:**

* Empty species names
* Missing coordinates
* Very rare species (less than 5 samples)

Now group by species.
**For each species compute:**

* Number of samples
* Number of countries
* First year
* Last year
* Year span
* Geographic spread (lat/long variance)

Save as: `species_features.csv`

---

## 🏷 PHASE 3 — Add Extinction Labels (2–3 Hours)

**WHAT:**
For each selected species:

Go to: **IUCN Red List website**

Find: **Conservation category**

**Map categories:**

* LC → 0
* NT → 1
* VU → 2
* EN → 3
* CR → 4

Add this as a column in your dataset.

**Now your dataset looks like:**

| Species | Num_samples | Num_countries | Year_span | Geo_spread | Risk_label |
| --- | --- | --- | --- | --- | --- |


Save as: `final_dataset.csv`

---

## 🧠 PHASE 4 — Build ML Model (3–5 Hours)

**HOW:**

1. Open Python.
2. Load dataset.
3. **Split:**
* 70% train
* 30% test


4. **Use:** `RandomForestClassifier`
5. Train model to predict `Risk_label`.
6. **Print:**
* Accuracy
* Confusion matrix



Save model as: `extinction_model.pkl`

---

## 🔍 PHASE 5 — Add Early Warning Logic (1–2 Hours)

We add a simple rule.

**If:**

* Year span shrinking
* Geo spread low
* Sample distribution uneven

**Then:**

* Flag as “Early Warning”

This can be a simple threshold rule.
No need for complex math.

---

## 📊 PHASE 6 — Explainability (2 Hours)

**Use:**
Feature importance from Random Forest.

**Plot:**
Bar chart.

**This answers:**

* Why did the model predict high risk?

This is **VERY important** for judges.

---

## 🖥 PHASE 7 — Build Streamlit App (3–4 Hours)

Create simple interface:

User selects species from dropdown.

**App shows:**

* Risk prediction
* Risk probability
* Early warning flag
* Feature importance chart

Keep UI simple.

---

## 🎤 PHASE 8 — Pitch Preparation (2 Hours)

**Structure:**

* **Problem:** Conservation reacts too late.
* **Insight:** Microbiome and sampling patterns shift earlier.
* **Solution:** AI model predicting extinction risk.
* **Impact:** Enables proactive conservation.

---

## ⏱ Suggested Timeline (24 Hours)

* **Hour 0–2** → Scope + understand Metalog
* **Hour 2–6** → Download + clean data
* **Hour 6–9** → Feature engineering
* **Hour 9–13** → ML model
* **Hour 13–15** → Early warning rule
* **Hour 15–19** → Streamlit app
* **Hour 19–22** → Testing
* **Hour 22–24** → Pitch polish

---

## 🧠 Important Mindset

**You are NOT building:**

* A complete ecological model
* A genome analysis pipeline
* A publishable research system

**You ARE building:**

* A structured AI prototype using real data that demonstrates early-warning capability.

---

## 🚨 If You Get Stuck

If Metalog feels too complex:

**Fallback plan:**

1. Select only 10 species.
2. Use only:
* Num_samples
* Num_countries
* Year_span


3. Train model on that.

Even that is enough for hackathon.
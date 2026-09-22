# AI-Powered Household Energy Consumption Analysis and Unit Prediction

## Project Title
**AI-Powered Household Energy Consumption Analysis and Unit Prediction**
IBM SkillsBuild Data Analytics with AI Academic Internship · BharatCares × AICTE

## Project Overview

An end-to-end **data analytics and machine learning solution** that analyses
household characteristics and predicts monthly energy consumption (kWh) using
five regression models. The entire project — data loading, EDA, preprocessing,
model training, evaluation, and interactive prediction — runs from a **single
Streamlit application** (`app.py`).

Built as part of the **IBM SkillsBuild Data Analytics with AI Academic
Internship Program** conducted by **BharatCares** in association with **AICTE**.

---

## Problem Statement

Household-level energy consumption prediction is essential for smart-grid
management, energy policy design, and sustainable urban planning. Given eight
household attributes (rooms, occupants, area, appliances, location type), this
project trains a machine learning regression model to predict monthly energy
consumption in kWh.

---

## Objectives

1. Analyse household characteristics and their relationship with energy usage.
2. Perform thorough exploratory data analysis (EDA) with visualisations.
3. Engineer meaningful features from existing variables.
4. Train and compare five regression models.
5. Select the best model based on MAE, RMSE, and R².
6. Interpret feature importance / associations with predicted energy usage.
7. Provide an interactive prediction interface for any household profile.

---

## Dataset

| Property | Value |
|----------|-------|
| **File name** | `Household energy unit data.csv` |
| **Records** | 1 000 rows |
| **Target variable** | `units` — monthly energy consumption (kWh) |
| **Dataset source** | Dataset file is included in this project repository. |

### Features

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `num_rooms` | Integer | Number of rooms in the household |
| 2 | `num_people` | Integer | Total number of occupants |
| 3 | `housearea` | Float | Total house area in square feet |
| 4 | `is_ac` | Binary (0/1) | Air conditioning present (1 = Yes) |
| 5 | `is_tv` | Binary (0/1) | Television present (1 = Yes) |
| 6 | `is_flat` | Binary (0/1) | Flat/apartment type (1 = Flat) |
| 7 | `num_children` | Integer | Number of children in household |
| 8 | `is_urban` | Binary (0/1) | Urban area (1 = Urban) |

**Data quality notes:**
- No missing values, no duplicates.
- 5 rows had `num_rooms = −1` and 4 rows had `num_people = −1` (invalid);
  replaced with column medians.

---

## Technologies Used

| Category | Library / Tool |
|----------|----------------|
| Language | Python 3.9+ |
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn |
| Model serialisation | joblib |
| Web application | Streamlit |
| Statistical utilities | scipy |

---

## Project Structure

```
Household-Energy-Analytics/
│
├── app.py                          ← Single file: frontend + backend
├── Household energy unit data.csv  ← Dataset
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── ProjectReport.docx              ← Full academic project report
└── final_model.pkl                 ← Best model (auto-generated on first run)
```

---

## Installation

**Prerequisites:** Python 3.9 or later

```bash
# 1. Clone or download the project folder
# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — Interactive Streamlit Dashboard (recommended)

```bash
# From the project folder (where app.py lives):
streamlit run app.py
```

Streamlit will open the application in your default browser at
`http://localhost:8501`.

**First-run behaviour:**
The pipeline trains all five models on startup (~10–15 seconds). Results are
cached for the rest of the session. The best model is saved as
`final_model.pkl` automatically.

**Navigation:**
Use the sidebar to jump between project sections:
- Overview → Data Understanding → EDA → Outlier Analysis → Feature Engineering
- Models → Model Evaluation → Model Comparison → Feature Importance
- Predict Energy Units (interactive form)
- Insights & Conclusion

### Option B — Command-Line Training Mode

```bash
# Runs the full ML pipeline in the terminal and prints a training report
python app.py --train
```

This mode:
- Loads and cleans the dataset
- Runs all 5 regression models with 5-fold cross-validation
- Prints a full evaluation report (MAE, RMSE, R²) to the console
- Saves the best model as `final_model.pkl`
- Prints a sample prediction for a demo household

---

## Methodology

### 1. Data Cleaning
- Detected 9 rows with invalid negative counts in `num_rooms` and `num_people`.
- Replaced with column medians (conservative imputation, no rows dropped).

### 2. Exploratory Data Analysis
- Univariate distributions (histograms, count plots).
- Bivariate scatter plots and box plots for each feature vs. target.
- Correlation heatmap for multivariate analysis.

### 3. Outlier Analysis
- IQR method applied to all numeric columns.
- No rows removed; outliers represent valid edge-case households.

### 4. Feature Engineering
Three new features were added:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `people_per_room` | `num_people / num_rooms` | Occupancy density |
| `area_per_person` | `housearea / num_people` | Space per person |
| `appliance_count` | `is_ac + is_tv` | Total appliances |

### 5. Machine Learning
- **Train / test split:** 80 % / 20 % (`random_state=42`)
- **Models trained:** Linear Regression, Ridge, Decision Tree, Random Forest,
  Gradient Boosting

### 6. Evaluation
Models compared on MAE, MSE, RMSE, and R² on the hold-out test set.

---

## Results

Actual metrics computed from `Household energy unit data.csv`
(800-sample training set, 200-sample test set, `random_state=42`):

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 11.163 | 13.833 | 0.7532 |
| Ridge Regression | 11.153 | 13.836 | 0.7531 |
| Decision Tree | 11.029 | 14.647 | 0.7233 |
| Random Forest | 10.553 | 13.612 | 0.7611 |
| **Gradient Boosting** | **10.186** | **12.708** | **0.7917** |

**Best model: Gradient Boosting** — explains **79.2 %** of energy consumption
variance; average prediction error ≈ **10.2 kWh/month**.

---

## Key Insights

1. **Air conditioning** is the strongest binary predictor of high energy use.
2. **House area** has the clearest positive linear association with `units`.
3. **Occupant count** (`num_people`, `num_children`) moderately correlates
   with energy consumption.
4. **Ensemble models** (Random Forest, Gradient Boosting) outperform linear
   models due to non-linear feature interactions.
5. The engineered `people_per_room` feature contributes additional signal.

---

## Future Scope

- Add **weather / season** as external regressors.
- Use **LSTM / time-series models** for monthly forecasting.
- Integrate **SHAP** for per-prediction explanations.
- Deploy as a **REST API** (FastAPI) for smart-meter integration.
- Extend dataset with **appliance-level** consumption data.

---

## Author

| Field | Value |
|-------|-------|
| **Name** | [Your Name] |
| **Institution** | [Your College / University] |
| **Program** | IBM SkillsBuild Data Analytics with AI Academic Internship |
| **Organisation** | BharatCares in association with AICTE |
| **Email** | [your.email@example.com] |

---

*Replace all placeholder values in `[ ]` brackets before final submission.*

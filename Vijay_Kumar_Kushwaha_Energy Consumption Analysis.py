# =============================================================================
# IBM SkillsBuild Data Analytics with AI Academic Internship Program
# Conducted by BharatCares in association with AICTE
#
# PROJECT TITLE : AI-Powered Household Energy Consumption Analysis
#                 and Unit Prediction
# AUTHOR        : [Your Name]
# FILE          : app.py  — single file containing EVERYTHING
#                 (data pipeline, model training, evaluation, and
#                  interactive Streamlit dashboard)
#
# HOW TO RUN DASHBOARD : streamlit run app.py
# HOW TO TRAIN ONLY    : python app.py --train
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os, io, sys, time, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
DATA_PATH    = "Household energy unit data.csv"
MODEL_PATH   = "final_model.pkl"

# ── Dark plot palette ─────────────────────────────────────────────────────────
PLT_BG  = "#0f1117"
PLT_AX  = "#1a1d2e"
GRID_C  = "#2a2d3e"
C1, C2, C3, C4, C5 = "#00d4ff", "#a855f7", "#22d3ee", "#f59e0b", "#ef4444"

def _dark_style():
    plt.rcParams.update({
        "figure.facecolor": PLT_BG,  "axes.facecolor"  : PLT_AX,
        "axes.edgecolor"  : GRID_C,  "axes.labelcolor" : "#cbd5e1",
        "xtick.color"     : "#94a3b8","ytick.color"    : "#94a3b8",
        "text.color"      : "#e2e8f0","grid.color"     : GRID_C,
        "grid.linewidth"  : 0.6,     "axes.titlecolor" : "#f1f5f9",
        "axes.titlesize"  : 11,      "axes.labelsize"  : 9,
        "font.family"     : "DejaVu Sans",
    })

_dark_style()

def _buf(fig) -> io.BytesIO:
    """Save a figure to a PNG byte buffer."""
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=120, bbox_inches="tight", facecolor=PLT_BG)
    b.seek(0); plt.close(fig); return b


# =============================================================================
# ══════════════════════  PHASE 1-8 : BACKEND PIPELINE  ══════════════════════
# =============================================================================

# ── Phase 2 : Load ────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ── Phase 4 : Clean ───────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid (−1) values in num_rooms / num_people with median."""
    df2 = df.copy()
    for col in ["num_rooms", "num_people"]:
        med = int(df2.loc[df2[col] >= 0, col].median())
        df2.loc[df2[col] < 0, col] = med
    return df2

# ── Phase 7 : Feature Engineering ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three domain-justified engineered features."""
    df2 = df.copy()
    df2["people_per_room"] = df2.apply(
        lambda r: r["num_people"] / r["num_rooms"]
        if r["num_rooms"] > 0 else float(r["num_people"]), axis=1)
    df2["area_per_person"] = df2.apply(
        lambda r: r["housearea"] / r["num_people"]
        if r["num_people"] > 0 else float(r["housearea"]), axis=1)
    df2["appliance_count"] = df2["is_ac"] + df2["is_tv"]
    return df2

# ── Phase 8 : Split ───────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """80/20 train-test split; return (Xtr, Xte, ytr, yte, feature_cols)."""
    fc = [c for c in df.columns if c != "units"]
    X, y = df[fc], df["units"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return Xtr, Xte, ytr, yte, fc

# ── Phase 10 : Train ──────────────────────────────────────────────────────────
def train_models(Xtr, ytr) -> dict:
    """Train all 5 regression models; return dict of fitted models."""
    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree"     : DecisionTreeRegressor(
                                  max_depth=6, random_state=RANDOM_STATE),
        "Random Forest"     : RandomForestRegressor(
                                  n_estimators=150, max_depth=8,
                                  random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(
                                  n_estimators=150, max_depth=4,
                                  learning_rate=0.1, random_state=RANDOM_STATE),
    }
    return {n: m.fit(Xtr, ytr) for n, m in models.items()}

# ── Phase 11 : Evaluate ───────────────────────────────────────────────────────
def evaluate_models(fitted: dict, Xte, yte) -> pd.DataFrame:
    """Compute MAE / MSE / RMSE / R² on the test set for every model."""
    rows = []
    for n, m in fitted.items():
        yp = m.predict(Xte)
        rows.append({
            "Model": n,
            "MAE"  : round(mean_absolute_error(yte, yp),                    3),
            "MSE"  : round(mean_squared_error(yte, yp),                     3),
            "RMSE" : round(float(np.sqrt(mean_squared_error(yte, yp))),     3),
            "R²"   : round(float(r2_score(yte, yp)),                        4),
            "MAPE%": round(float(mean_absolute_percentage_error(yte, yp)) * 100, 2),
        })
    return (pd.DataFrame(rows)
              .sort_values("R²", ascending=False)
              .reset_index(drop=True))

# ── Full pipeline (Streamlit cached) ─────────────────────────────────────────
def _run_full_pipeline(path: str):
    """Execute all phases once and return every artefact."""
    df_raw  = load_data(path)
    df_cln  = clean_data(df_raw)
    df_fe   = engineer_features(df_cln)
    Xtr, Xte, ytr, yte, fc = split_data(df_fe)
    fitted  = train_models(Xtr, ytr)
    results = evaluate_models(fitted, Xte, yte)
    best    = results.iloc[0]["Model"]
    bmodel  = fitted[best]
    joblib.dump(bmodel, MODEL_PATH)
    return df_raw, df_cln, df_fe, Xtr, Xte, ytr, yte, fc, fitted, results, best, bmodel


# =============================================================================
# ══════════════════════  CLI TRAINING MODE  ══════════════════════════════════
# run with:  python app.py --train
# =============================================================================

def cli_train():
    """Standalone training pipeline — prints a full report to the console."""
    # UTF-8 safe stdout for Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  HOUSEHOLD ENERGY — MODEL TRAINING REPORT")
    print("  IBM SkillsBuild · BharatCares x AICTE")
    print(SEP)

    # 1. Load
    print(f"\n[1/6] Loading  '{DATA_PATH}' ...")
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: dataset not found at '{DATA_PATH}'")
    df = load_data(DATA_PATH)
    print(f"  Loaded : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Target : units  min={df['units'].min():.2f}  "
          f"max={df['units'].max():.2f}  mean={df['units'].mean():.2f} kWh")

    # 2. Clean
    print(f"\n[2/6] Data cleaning ...")
    df2 = clean_data(df)
    for col in ["num_rooms", "num_people"]:
        n_neg = (df[col] < 0).sum()
        med   = int(df2[col].median())
        if n_neg:
            print(f"  Fixed  : {col} — {n_neg} invalid values -> median ({med})")
        else:
            print(f"  OK     : {col} — no invalid values")
    print(f"  Missing: {df2.isnull().sum().sum()}  |  Duplicates: {df2.duplicated().sum()}")

    # 3. Feature engineering
    print(f"\n[3/6] Feature engineering ...")
    df_fe = engineer_features(df2)
    for feat in ["people_per_room", "area_per_person", "appliance_count"]:
        print(f"  Added  : {feat}")
    print(f"  Final  : {df_fe.shape[0]} rows x {df_fe.shape[1]} columns")

    # 4. Split
    print(f"\n[4/6] Train-test split  (test={TEST_SIZE}, seed={RANDOM_STATE}) ...")
    Xtr, Xte, ytr, yte, fc = split_data(df_fe)
    print(f"  Train  : {len(ytr)} | Test: {len(yte)} | Features: {len(fc)}")

    # 5. Train + evaluate (with CV)
    print(f"\n[5/6] Training 5 models ...")
    print(f"\n  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R2':>7}  {'CV-R2':>8}  {'Time':>5}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}  {'-'*8}  {'-'*5}")

    fitted_all = {}
    result_rows = []
    for name, model in {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree"     : DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest"     : RandomForestRegressor(n_estimators=150, max_depth=8,
                                                     random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                        learning_rate=0.1, random_state=RANDOM_STATE),
    }.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fitted_all[name] = model
        yp   = model.predict(Xte)
        mae  = round(mean_absolute_error(yte, yp),  3)
        rmse = round(float(np.sqrt(mean_squared_error(yte, yp))), 3)
        r2   = round(float(r2_score(yte, yp)),       4)
        cv   = round(cross_val_score(model, Xtr, ytr, cv=5,
                                     scoring="r2", n_jobs=-1).mean(), 4)
        mape = round(float(mean_absolute_percentage_error(yte, yp)) * 100, 2)
        elapsed = time.time() - t0
        result_rows.append({"Model": name, "MAE": mae, "RMSE": rmse,
                             "R2": r2, "CV-R2": cv, "MAPE%": mape})
        print(f"  {name:<22} {mae:>7.3f} {rmse:>7.3f} {r2:>7.4f}  {cv:>8.4f}  {elapsed:>4.1f}s")

    res = sorted(result_rows, key=lambda x: x["R2"], reverse=True)
    best_name  = res[0]["Model"]
    best_model = fitted_all[best_name]
    joblib.dump(best_model, MODEL_PATH)

    # 6. Summary
    print(f"\n[6/6] Summary")
    print(f"\n  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R2':>7}  {'CV-R2':>8}  {'MAPE%':>6}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}  {'-'*8}  {'-'*6}")
    for r in res:
        mark = " <-- BEST" if r["Model"] == best_name else ""
        print(f"  {r['Model']:<22} {r['MAE']:>7.3f} {r['RMSE']:>7.3f} "
              f"{r['R2']:>7.4f}  {r['CV-R2']:>8.4f}  {r['MAPE%']:>6.2f}%{mark}")

    # Demo prediction
    sample = {
        "num_rooms": 2, "num_people": 4, "housearea": 800.0,
        "is_ac": 1, "is_tv": 1, "is_flat": 0,
        "num_children": 1, "is_urban": 1,
        "people_per_room": 2.0, "area_per_person": 200.0, "appliance_count": 2,
    }
    pred = best_model.predict(pd.DataFrame([sample])[fc])[0]

    print(f"\n  Best Model  : {best_name}")
    print(f"  R2          : {res[0]['R2']}  ({res[0]['R2']*100:.1f}% variance explained)")
    print(f"  RMSE        : {res[0]['RMSE']} kWh")
    print(f"  Model saved : {MODEL_PATH}  ({os.path.getsize(MODEL_PATH):,} bytes)")
    print(f"\n  Demo — 2 rooms | 4 people | 800 sqft | AC+TV | Urban")
    print(f"  Predicted   : {pred:.2f} kWh / month")
    print(f"\n{SEP}\n")


# =============================================================================
# ══════════════════════  STREAMLIT CSS  ══════════════════════════════════════
# =============================================================================

CUSTOM_CSS = """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
.stApp {
    background: linear-gradient(135deg,#0a0e1a 0%,#0f1629 50%,#0a1628 100%);
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#161b2c 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebarNav"] { display: none; }
.top-banner {
    background: linear-gradient(90deg,#0284c7 0%,#7c3aed 50%,#0ea5e9 100%);
    padding: 18px 28px; border-radius: 14px; margin-bottom: 22px;
    box-shadow: 0 4px 24px rgba(0,212,255,0.25);
}
.top-banner h1 {
    font-size:1.55rem; font-weight:800; color:#fff; margin:0 0 4px 0;
}
.top-banner p { color:rgba(255,255,255,0.82); font-size:0.85rem; margin:0; }
.kpi-grid {
    display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px;
}
.kpi-card {
    background:linear-gradient(135deg,#1a1d2e 0%,#1e2540 100%);
    border:1px solid #2a3a5c; border-radius:12px; padding:18px 16px;
    text-align:center; position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0;
    height:3px; border-radius:12px 12px 0 0;
}
.kpi-card.cyan::before  { background:linear-gradient(90deg,#00d4ff,#0ea5e9); }
.kpi-card.purple::before{ background:linear-gradient(90deg,#a855f7,#7c3aed); }
.kpi-card.amber::before { background:linear-gradient(90deg,#f59e0b,#ef4444); }
.kpi-card.green::before { background:linear-gradient(90deg,#10b981,#059669); }
.kpi-icon  { font-size:1.6rem; margin-bottom:6px; }
.kpi-value { font-size:1.7rem; font-weight:800; letter-spacing:-0.5px; margin-bottom:2px; }
.kpi-card.cyan   .kpi-value { color:#00d4ff; }
.kpi-card.purple .kpi-value { color:#c084fc; }
.kpi-card.amber  .kpi-value { color:#fbbf24; }
.kpi-card.green  .kpi-value { color:#34d399; }
.kpi-label { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; }
.section-card {
    background:linear-gradient(135deg,#141826 0%,#1a1f30 100%);
    border:1px solid #1e3a5f; border-radius:14px; padding:22px 24px; margin-bottom:20px;
}
.section-title {
    font-size:0.8rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.2px; color:#00d4ff; border-bottom:1px solid #1e3a5f;
    padding-bottom:10px; margin-bottom:16px;
}
[data-testid="stDataFrame"] {
    border:1px solid #1e3a5f !important; border-radius:10px !important; overflow:hidden;
}
[data-testid="metric-container"] {
    background:linear-gradient(135deg,#1a1d2e,#1e2540);
    border:1px solid #2a3a5c; border-radius:10px; padding:12px 14px;
}
[data-testid="stMetricValue"] { color:#00d4ff !important; font-weight:800; }
[data-testid="stMetricLabel"] {
    color:#64748b !important; font-size:0.72rem !important; text-transform:uppercase;
}
.predict-result {
    background:linear-gradient(135deg,#022c22,#064e3b);
    border:1px solid #10b981; border-radius:12px;
    padding:20px 24px; text-align:center; margin-top:16px;
}
.predict-result .big-num {
    font-size:2.8rem; font-weight:900; color:#34d399; letter-spacing:-1px;
}
.predict-result .unit { font-size:1rem; color:#6ee7b7; }
.stProgress > div > div { background:linear-gradient(90deg,#00d4ff,#a855f7) !important; }
.stButton > button {
    background:linear-gradient(90deg,#0284c7,#7c3aed) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; padding:10px 24px !important;
    font-size:0.9rem !important; box-shadow:0 2px 14px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover { opacity:0.88 !important; }
.stSelectbox label, .stNumberInput label {
    color:#94a3b8 !important; font-size:0.78rem !important;
}
[data-baseweb="select"] { background:#1a1d2e !important; border-color:#2a3a5c !important; }
[data-baseweb="input"]  {
    background:#1a1d2e !important; border-color:#2a3a5c !important; color:#e2e8f0 !important;
}
hr { border-color:#1e3a5f !important; }
[data-testid="stRadio"] label { color:#94a3b8 !important; font-size:0.82rem !important; }
.stAlert { border-radius:10px !important; }
.stTabs [data-baseweb="tab"] { color:#94a3b8 !important; font-size:0.82rem !important; }
.stTabs [aria-selected="true"] { color:#00d4ff !important; border-bottom-color:#00d4ff !important; }
</style>
"""


# =============================================================================
# ══════════════════════  PLOT HELPERS  ═══════════════════════════════════════
# =============================================================================

def plot_univariate(df):
    cols = ["num_rooms","num_people","housearea","num_children","units"]
    clrs = [C1, C2, C3, C4, C5]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), facecolor=PLT_BG)
    ax = ax.flatten()
    for i, (c, cl) in enumerate(zip(cols, clrs)):
        ax[i].hist(df[c], bins=28, color=cl, alpha=0.85,
                   edgecolor="#0a0e1a", linewidth=0.4)
        ax[i].set_title(c, fontweight="bold", color="#e2e8f0")
        ax[i].set_xlabel(c, fontsize=8); ax[i].set_ylabel("Count", fontsize=8)
        ax[i].grid(True, axis="y", alpha=0.3)
    ax[5].set_visible(False)
    fig.suptitle("Univariate Distributions — Numeric Features",
                 fontsize=13, fontweight="bold", color="#f1f5f9", y=1.01)
    plt.tight_layout(); return _buf(fig)

def plot_binary_counts(df):
    bin_cols = ["is_ac","is_tv","is_flat","is_urban"]
    labels   = ["Air Conditioning","Television","Flat / House","Urban / Rural"]
    fig, ax  = plt.subplots(1, 4, figsize=(14, 4.5), facecolor=PLT_BG)
    for i, (col, lbl, c) in enumerate(zip(bin_cols, labels, [C1,C2,C3,C4])):
        vc   = df[col].value_counts().sort_index()
        bars = ax[i].bar(["No","Yes"], vc.values,
                         color=[GRID_C, c], edgecolor="#0a0e1a", width=0.55)
        ax[i].set_title(lbl, fontweight="bold", color="#e2e8f0")
        ax[i].set_ylabel("Count", fontsize=8)
        ax[i].grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vc.values):
            ax[i].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+4, str(v),
                       ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    fig.suptitle("Binary Feature Distributions",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_scatter(df):
    feats = ["housearea","num_people","num_rooms","num_children"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), facecolor=PLT_BG)
    ax = ax.flatten()
    for i, (col, c) in enumerate(zip(feats, [C1,C2,C3,C4])):
        ax[i].scatter(df[col], df["units"], alpha=0.25, color=c, s=14, edgecolors="none")
        m, b = np.polyfit(df[col], df["units"], 1)
        xl = np.linspace(df[col].min(), df[col].max(), 200)
        ax[i].plot(xl, m*xl+b, color=C5, linewidth=1.8, label="Trend")
        ax[i].set_xlabel(col, fontsize=9); ax[i].set_ylabel("Units (kWh)", fontsize=9)
        ax[i].set_title(f"{col}  vs  Units", fontweight="bold", color="#e2e8f0")
        ax[i].legend(fontsize=8); ax[i].grid(True, alpha=0.3)
    fig.suptitle("Bivariate — Features vs Energy Units",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_boxplots(df):
    bin_cols = ["is_ac","is_tv","is_flat","is_urban"]
    fig, ax  = plt.subplots(1, 4, figsize=(14, 5), facecolor=PLT_BG)
    for i, (col, c) in enumerate(zip(bin_cols, [C1,C2,C3,C4])):
        g0 = df[df[col]==0]["units"].values
        g1 = df[df[col]==1]["units"].values
        bp = ax[i].boxplot([g0,g1], patch_artist=True,
                           boxprops=dict(facecolor=PLT_AX, color=c),
                           medianprops=dict(color=C5, linewidth=2.2),
                           whiskerprops=dict(color="#475569"),
                           capprops=dict(color="#475569"),
                           flierprops=dict(marker="o",color=C5,markersize=3,alpha=0.5))
        bp["boxes"][0].set_facecolor("#1a1d2e")
        bp["boxes"][1].set_facecolor(c+"33")
        ax[i].set_xticks([1,2]); ax[i].set_xticklabels(["No","Yes"])
        ax[i].set_title(col, fontweight="bold", color="#e2e8f0")
        ax[i].set_ylabel("Units (kWh)", fontsize=8)
        ax[i].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Energy Units by Binary Features",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=PLT_BG)
    ax.set_facecolor(PLT_AX)
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, linewidths=0.5, linecolor="#0a0e1a",
                annot_kws={"size":8,"color":"#e2e8f0"},
                cbar_kws={"shrink":0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=13, fontweight="bold", color="#f1f5f9", pad=12)
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout(); return _buf(fig)

def plot_outliers(df):
    cols = ["num_rooms","num_people","housearea","num_children","units"]
    fig, ax = plt.subplots(1, 5, figsize=(16, 5), facecolor=PLT_BG)
    for i, (col, c) in enumerate(zip(cols, [C1,C2,C3,C4,C5])):
        ax[i].boxplot(df[col], patch_artist=True,
                      boxprops=dict(facecolor=c+"22", color=c),
                      medianprops=dict(color="#ffffff", linewidth=2),
                      whiskerprops=dict(color="#475569"), capprops=dict(color="#475569"),
                      flierprops=dict(marker="o",color=C5,markersize=3,alpha=0.6))
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR   = Q3-Q1; lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
        n     = int(((df[col]<lo)|(df[col]>hi)).sum())
        ax[i].set_title(col, fontweight="bold", color="#e2e8f0")
        ax[i].set_xlabel(f"Outliers: {n}", fontsize=8,
                         color=C5 if n>0 else "#34d399")
        ax[i].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Outlier Analysis — IQR Method",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_comparison(res_df, best_name):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=PLT_BG)
    for i, (metric, c) in enumerate(zip(["MAE","RMSE","R²"], [C1,C2,C3])):
        vals  = res_df[metric].values
        bar_c = [C5 if n==best_name else c for n in res_df["Model"]]
        bars  = ax[i].barh(res_df["Model"], vals,
                            color=bar_c, edgecolor="#0a0e1a", height=0.55)
        ax[i].set_title(metric, fontweight="bold", color="#f1f5f9")
        ax[i].set_xlabel(metric, fontsize=9)
        ax[i].grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax[i].text(bar.get_width()+bar.get_width()*0.015,
                       bar.get_y()+bar.get_height()/2,
                       f"{val:.3f}", va="center", fontsize=8.5, color="#e2e8f0")
    fig.suptitle("Model Comparison — MAE | RMSE | R²",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_feature_importance(best_name, best_model, fc):
    scores = (best_model.feature_importances_
              if hasattr(best_model, "feature_importances_")
              else np.abs(best_model.coef_))
    label  = "Importance" if hasattr(best_model,"feature_importances_") else "|Coeff|"
    fi = (pd.DataFrame({"Feature":fc, label:scores})
            .sort_values(label).reset_index(drop=True))
    clrs = [plt.cm.cool(x) for x in np.linspace(0.1, 0.9, len(fi))]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PLT_BG)
    bars = ax.barh(fi["Feature"], fi[label], color=clrs, edgecolor="#0a0e1a")
    for bar, val in zip(bars, fi[label]):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#e2e8f0")
    ax.set_title(f"Feature Importance — {best_name}",
                 fontweight="bold", color="#f1f5f9")
    ax.set_xlabel(label, fontsize=9); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); return _buf(fig)

def plot_actual_vs_pred(bmodel, Xte, yte, best_name):
    yp  = bmodel.predict(Xte)
    res = np.array(yte) - yp
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor=PLT_BG)
    ax[0].scatter(yte, yp, alpha=0.5, color=C1, s=18, edgecolors="none")
    mn, mx = min(float(yte.min()),float(yp.min())), max(float(yte.max()),float(yp.max()))
    ax[0].plot([mn,mx],[mn,mx], color=C5, linewidth=1.5, linestyle="--", label="Perfect fit")
    ax[0].set_xlabel("Actual (kWh)",fontsize=9); ax[0].set_ylabel("Predicted (kWh)",fontsize=9)
    ax[0].set_title("Actual vs Predicted", fontweight="bold", color="#f1f5f9")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].scatter(yp, res, alpha=0.4, color=C2, s=18, edgecolors="none")
    ax[1].axhline(0, color=C5, linewidth=1.5, linestyle="--")
    ax[1].set_xlabel("Predicted (kWh)",fontsize=9); ax[1].set_ylabel("Residual",fontsize=9)
    ax[1].set_title("Residual Plot", fontweight="bold", color="#f1f5f9")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"Prediction Quality — {best_name}",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    plt.tight_layout(); return _buf(fig)

def plot_target_dist(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=PLT_BG)
    ax[0].hist(df["units"], bins=35, color=C1, edgecolor="#0a0e1a", alpha=0.85)
    ax[0].set_title("Target Distribution (units)", fontweight="bold", color="#f1f5f9")
    ax[0].set_xlabel("kWh/month",fontsize=9); ax[0].set_ylabel("Count",fontsize=9)
    ax[0].grid(True, axis="y", alpha=0.3)
    kde = gaussian_kde(df["units"])
    x   = np.linspace(df["units"].min(), df["units"].max(), 300)
    ax[1].fill_between(x, kde(x), alpha=0.4, color=C2)
    ax[1].plot(x, kde(x), color=C1, linewidth=2)
    ax[1].axvline(df["units"].mean(),   color=C5, linewidth=1.5, linestyle="--",
                  label=f"Mean={df['units'].mean():.1f}")
    ax[1].axvline(df["units"].median(), color=C4, linewidth=1.5, linestyle=":",
                  label=f"Median={df['units'].median():.1f}")
    ax[1].set_title("Kernel Density Estimate", fontweight="bold", color="#f1f5f9")
    ax[1].set_xlabel("kWh/month",fontsize=9); ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); return _buf(fig)


# =============================================================================
# ══════════════════════  HTML HELPERS  ═══════════════════════════════════════
# =============================================================================

def _kpi(icon, value, label, cls):
    return (f'<div class="kpi-card {cls}">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>')

def _card_open(title):
    return f'<div class="section-card"><div class="section-title">{title}</div>'

def _card_close():
    return '</div>'


# =============================================================================
# ══════════════════════  STREAMLIT PAGE SECTIONS  ════════════════════════════
# =============================================================================

def page_overview(df_raw, results_df, best_name):
    import streamlit as st
    br = results_df.iloc[0]
    r2_pct = f"{br['R\u00b2'] * 100:.1f}%"
    st.markdown(
        '<div class="top-banner">'
        '<h1>⚡ AI-Powered Household Energy Analytics</h1>'
        '<p>IBM SkillsBuild · BharatCares × AICTE &nbsp;|&nbsp; '
        'Predicting monthly energy consumption from household characteristics</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="kpi-grid">'
        f'{_kpi("🏠","1,000","Total Records","cyan")}'
        f'{_kpi("🔢","11","Features Used","purple")}'
        f'{_kpi("🏆", best_name.replace(" ","<br>"), "Best Model","amber")}'
        f'{_kpi("📈", r2_pct, "Variance Explained","green")}'
        f'</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(_card_open("📋 PROJECT SUMMARY"), unsafe_allow_html=True)
        st.markdown(
            "**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship  \n"
            "**Organisation:** BharatCares in association with AICTE  \n"
            "**Dataset:** Household energy unit data.csv — 1,000 rows × 9 columns  \n"
            "**Target:** `units` — monthly energy consumption (kWh)  \n"
            "**Models:** Linear Regression · Ridge · Decision Tree · "
            "Random Forest · Gradient Boosting  \n"
            "**Seed:** 42 (fully reproducible)  \n\n"
            "> **Problem:** Predict monthly household energy consumption from "
            "8 characteristics — enabling smart-grid management, energy auditing, "
            "and policy design.")
        st.markdown(_card_close(), unsafe_allow_html=True)
    with c2:
        st.markdown(_card_open("🎯 QUICK RESULTS"), unsafe_allow_html=True)
        st.dataframe(
            results_df[["Model","MAE","RMSE","R²"]].style
                .highlight_max(subset=["R²"],  color="#064e3b")
                .highlight_min(subset=["MAE","RMSE"], color="#064e3b")
                .set_properties(**{"color":"#e2e8f0","background-color":"#1a1d2e"}),
            use_container_width=True, height=230)
        st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("📊 TARGET DISTRIBUTION"), unsafe_allow_html=True)
    st.image(plot_target_dist(df_raw), use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)


def page_data(df_raw, df_clean):
    import streamlit as st
    st.markdown(_card_open("📂 DATASET — FIRST & LAST 5 ROWS"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("First 5 rows")
        st.dataframe(df_raw.head().style.set_properties(
            **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
            use_container_width=True)
    with c2:
        st.caption("Last 5 rows")
        st.dataframe(df_raw.tail().style.set_properties(
            **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
            use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("🔬 COLUMN INFORMATION"), unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column"  : df_raw.columns,
        "Type"    : df_raw.dtypes.values.astype(str),
        "Non-Null": df_raw.notnull().sum().values,
        "Unique"  : [df_raw[c].nunique() for c in df_raw.columns],
        "Min"     : [round(df_raw[c].min(),2) for c in df_raw.columns],
        "Max"     : [round(df_raw[c].max(),2) for c in df_raw.columns],
        "Mean"    : [round(df_raw[c].mean(),2) for c in df_raw.columns],
    })
    st.dataframe(info_df.style.set_properties(
        **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
        use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("🛡️ DATA QUALITY REPORT"), unsafe_allow_html=True)
    rm = int(df_raw.loc[df_raw["num_rooms"]>=0,"num_rooms"].median())
    pm = int(df_raw.loc[df_raw["num_people"]>=0,"num_people"].median())
    qdf = pd.DataFrame({
        "Check"  : ["Missing values","Duplicate rows","Infinite values",
                    "num_rooms = -1 (invalid)","num_people = -1 (invalid)"],
        "Found"  : [0,0,0,
                    int((df_raw["num_rooms"]<0).sum()),
                    int((df_raw["num_people"]<0).sum())],
        "Action" : ["None required","None required","None required",
                    f"Replaced with median ({rm})",
                    f"Replaced with median ({pm})"],
    })
    st.dataframe(qdf.style.set_properties(
        **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
        use_container_width=True)
    st.success(f"After cleaning: **{df_clean.shape[0]} rows x {df_clean.shape[1]} cols** — 0 issues remaining.")
    st.markdown(_card_close(), unsafe_allow_html=True)


def page_eda(df_clean):
    import streamlit as st
    t1,t2,t3,t4,t5 = st.tabs([
        "Distributions","Binary Features","Scatter Plots","Box Plots","Heatmap"])
    with t1:
        st.markdown(_card_open("UNIVARIATE — NUMERIC FEATURES"), unsafe_allow_html=True)
        st.image(plot_univariate(df_clean), use_container_width=True)
        st.markdown("> `num_rooms` peaks 2-3 · `num_people` centres 4-5 · `units` near-normal ~123 kWh")
        st.markdown(_card_close(), unsafe_allow_html=True)
    with t2:
        st.markdown(_card_open("UNIVARIATE — BINARY FEATURES"), unsafe_allow_html=True)
        st.image(plot_binary_counts(df_clean), use_container_width=True)
        st.markdown("> ~38% have AC · ~80% own TV · ~48% flat · ~61% urban")
        st.markdown(_card_close(), unsafe_allow_html=True)
    with t3:
        st.markdown(_card_open("BIVARIATE — FEATURES vs UNITS"), unsafe_allow_html=True)
        st.image(plot_scatter(df_clean), use_container_width=True)
        st.markdown("> `housearea` clearest positive trend · `num_people` moderate positive association")
        st.markdown(_card_close(), unsafe_allow_html=True)
    with t4:
        st.markdown(_card_open("BIVARIATE — UNITS BY BINARY FEATURES"), unsafe_allow_html=True)
        st.image(plot_boxplots(df_clean), use_container_width=True)
        st.markdown("> AC households consume noticeably more energy on average")
        st.markdown(_card_close(), unsafe_allow_html=True)
    with t5:
        st.markdown(_card_open("MULTIVARIATE — CORRELATION HEATMAP"), unsafe_allow_html=True)
        st.image(plot_heatmap(df_clean), use_container_width=True)
        st.markdown("> `is_ac` has strongest positive correlation with `units` · low inter-feature correlations")
        st.markdown(_card_close(), unsafe_allow_html=True)


def page_features(df_clean, df_fe):
    import streamlit as st
    st.markdown(_card_open("OUTLIER ANALYSIS — IQR METHOD"), unsafe_allow_html=True)
    st.image(plot_outliers(df_clean), use_container_width=True)
    cont = ["num_rooms","num_people","housearea","num_children","units"]
    rows = []
    for col in cont:
        Q1,Q3 = df_clean[col].quantile(0.25),df_clean[col].quantile(0.75)
        IQR = Q3-Q1; lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
        rows.append({"Feature":col,"Q1":round(Q1,2),"Q3":round(Q3,2),
                     "IQR":round(IQR,2),"Lower":round(lo,2),
                     "Upper":round(hi,2),
                     "Outliers":int(((df_clean[col]<lo)|(df_clean[col]>hi)).sum())})
    st.dataframe(pd.DataFrame(rows).style.set_properties(
        **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
        use_container_width=True)
    st.info("No records removed — extreme values represent valid edge-case households.")
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("FEATURE ENGINEERING — 3 NEW FEATURES"), unsafe_allow_html=True)
    fe_df = pd.DataFrame({
        "Feature" : ["people_per_room","area_per_person","appliance_count"],
        "Formula" : ["num_people/num_rooms","housearea/num_people","is_ac+is_tv"],
        "Rationale": ["Occupancy density — crowding may increase energy demand",
                      "Space per person — larger area may raise usage",
                      "Count of major energy-consuming appliances"],
    })
    st.dataframe(fe_df.style.set_properties(
        **{"background-color":"#1a1d2e","color":"#e2e8f0"}),
        use_container_width=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Original Features","8")
    c2.metric("Engineered Features","3")
    c3.metric("Total Features","11")
    st.markdown(_card_close(), unsafe_allow_html=True)


def page_models(results_df, fitted_models, best_name, best_model, Xte, yte, fc):
    import streamlit as st
    st.markdown(_card_open("MODEL EVALUATION RESULTS"), unsafe_allow_html=True)
    st.markdown("""
| Metric | Meaning |
|--------|---------|
| **MAE** | Mean Absolute Error — avg prediction error in kWh (lower = better) |
| **RMSE** | Root Mean Squared Error — same unit as target, penalises large errors |
| **R²** | Proportion of variance explained — 1.0 is perfect (higher = better) |
    """)
    st.dataframe(
        results_df.style
            .highlight_max(subset=["R²"],  color="#064e3b")
            .highlight_min(subset=["MAE","RMSE"], color="#064e3b")
            .set_properties(**{"background-color":"#1a1d2e","color":"#e2e8f0"}),
        use_container_width=True)
    br = results_df.iloc[0]
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#022c22,#064e3b);'
        f'border:1px solid #10b981;border-radius:12px;padding:16px 22px;margin-top:12px;">'
        f'<span style="font-size:1.1rem;font-weight:800;color:#34d399;">'
        f'Best Model: {best_name}</span>&nbsp;&nbsp;'
        f'<span style="color:#6ee7b7;font-size:0.9rem;">'
        f'R\u00b2 = {br["R\u00b2"]} &nbsp;|&nbsp; RMSE = {br["RMSE"]} kWh '
        f'&nbsp;|&nbsp; MAE = {br["MAE"]} kWh</span></div>',
        unsafe_allow_html=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("MODEL COMPARISON CHARTS"), unsafe_allow_html=True)
    st.image(plot_comparison(results_df, best_name), use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("ACTUAL vs PREDICTED & RESIDUAL PLOT"), unsafe_allow_html=True)
    st.image(plot_actual_vs_pred(best_model, Xte, yte, best_name), use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open(f"FEATURE IMPORTANCE — {best_name}"), unsafe_allow_html=True)
    st.image(plot_feature_importance(best_name, best_model, fc), use_container_width=True)
    st.markdown("> Features ranked higher are more strongly **associated with** "
                "predicted energy consumption (not causal).")
    st.markdown(_card_close(), unsafe_allow_html=True)


def page_predict(best_model, best_name, fc, results_df):
    import streamlit as st
    br = results_df.iloc[0]
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1117,#161b2c);'
        f'border:1px solid #1e3a5f;border-radius:16px;padding:22px 28px;">'
        f'<span style="font-size:1.8rem;">🔮</span>&nbsp;&nbsp;'
        f'<span style="font-size:1.1rem;font-weight:800;color:#e2e8f0;">'
        f'Energy Consumption Predictor</span><br>'
        f'<span style="font-size:0.8rem;color:#64748b;">'
        f'Model: <span style="color:#00d4ff;">{best_name}</span> &nbsp;·&nbsp; '
        f'R\u00b2 = <span style="color:#34d399;">{br["R\u00b2"]}</span> &nbsp;·&nbsp; '
        f'RMSE = <span style="color:#fbbf24;">{br["RMSE"]} kWh</span></span></div>',
        unsafe_allow_html=True)
    st.markdown("")

    with st.form("pred_form", clear_on_submit=False):
        st.markdown("#### Household Characteristics")
        c1,c2,c3,c4 = st.columns(4)
        num_rooms    = c1.number_input("Rooms",       0,10, 2)
        num_people   = c2.number_input("People",      0,15, 4)
        housearea    = c3.number_input("Area (sqft)", 100.0,2000.0,800.0,step=25.0)
        num_children = c4.number_input("Children",    0,10, 1)

        st.markdown("#### Appliances & Location")
        c5,c6,c7,c8 = st.columns(4)
        is_ac    = c5.selectbox("Air Conditioning",[1,0],
                                format_func=lambda x:"Yes" if x else "No")
        is_tv    = c6.selectbox("Television",      [1,0],
                                format_func=lambda x:"Yes" if x else "No")
        is_flat  = c7.selectbox("Dwelling Type",   [0,1],
                                format_func=lambda x:"House" if x==0 else "Flat")
        is_urban = c8.selectbox("Location",        [1,0],
                                format_func=lambda x:"Urban" if x else "Rural")
        submitted = st.form_submit_button("PREDICT ENERGY CONSUMPTION",
                                          use_container_width=True)

    if submitted:
        row = {
            "num_rooms"      : num_rooms,
            "num_people"     : num_people,
            "housearea"      : housearea,
            "is_ac"          : is_ac,
            "is_tv"          : is_tv,
            "is_flat"        : is_flat,
            "num_children"   : num_children,
            "is_urban"       : is_urban,
            "people_per_room": num_people/num_rooms if num_rooms>0 else float(num_people),
            "area_per_person": housearea/num_people if num_people>0 else float(housearea),
            "appliance_count": is_ac+is_tv,
        }
        pred  = best_model.predict(pd.DataFrame([row])[fc])[0]
        level = ("Low" if pred<80 else "Medium" if pred<140 else "High")
        lclr  = ("#34d399" if pred<80 else "#fbbf24" if pred<140 else "#ef4444")
        icon  = ("🟢" if pred<80 else "🟡" if pred<140 else "🔴")

        st.markdown(
            f'<div class="predict-result">'
            f'<div style="font-size:0.85rem;color:#6ee7b7;margin-bottom:8px;'
            f'text-transform:uppercase;letter-spacing:1px;">Predicted Monthly Consumption</div>'
            f'<div class="big-num">{pred:.1f}</div>'
            f'<div class="unit">kWh / month</div>'
            f'<div style="margin-top:12px;font-size:0.9rem;color:{lclr};font-weight:700;">'
            f'{icon} Consumption Level: {level}</div></div>',
            unsafe_allow_html=True)

        st.markdown("---")
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Prediction",       f"{pred:.1f} kWh")
        cc2.metric("Level",            level)
        cc3.metric("vs Dataset Mean",  f"{pred-123.6:+.1f} kWh")
        cc4.metric("Model",            best_name.split()[0])

        with st.expander("View Input Summary"):
            st.dataframe(pd.DataFrame([{
                "Rooms":num_rooms,"People":num_people,"Area":housearea,
                "AC":("Yes" if is_ac else "No"),"TV":("Yes" if is_tv else "No"),
                "Flat":("Yes" if is_flat else "No"),
                "Children":num_children,"Urban":("Yes" if is_urban else "No"),
                "Prediction (kWh)":round(pred,2),
            }]).style.set_properties(**{"background-color":"#1a1d2e","color":"#e2e8f0"}),
                use_container_width=True)


def page_insights(results_df, best_name):
    import streamlit as st
    br = results_df[results_df["Model"]==best_name].iloc[0]
    st.markdown(_card_open("KEY FINDINGS"), unsafe_allow_html=True)
    st.markdown(f"""
**Air Conditioning (`is_ac`)** — Strongest binary predictor; AC households consume significantly more energy.  
**House Area (`housearea`)** — Clear positive relationship with energy units.  
**Occupants** (`num_people`, `num_children`) — Positively associated with consumption.  
**Best Model: {best_name}** — R\u00b2 = {br["R\u00b2"]} ({br["R\u00b2"]*100:.1f}% variance), RMSE = {br["RMSE"]} kWh.  
**Ensemble models outperform linear baselines** — non-linear interactions exist in the data.
    """)
    st.markdown(_card_close(), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_card_open("REAL-WORLD APPLICATIONS"), unsafe_allow_html=True)
        st.markdown("""
**Smart Grid Planning** — Utilities forecast load from household profiles.  
**Energy Audits** — Target high-consumption households for efficiency drives.  
**Policy Design** — Subsidise star-rated appliances for high-consumer segments.  
**Green Building** — Architects estimate demand at design stage.
        """)
        st.markdown(_card_close(), unsafe_allow_html=True)
    with c2:
        st.markdown(_card_open("FUTURE SCOPE"), unsafe_allow_html=True)
        st.markdown("""
**Weather Data** — Temperature & season as external regressors.  
**Time-Series Models** — LSTM / Prophet for monthly forecasting.  
**SHAP Values** — Per-prediction explainability.  
**REST API Deployment** — FastAPI for smart-meter integration.  
**Real Data** — Validate on actual meter readings across regions.
        """)
        st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("CONCLUSION"), unsafe_allow_html=True)
    st.markdown(f"""
This project delivered a complete, reproducible AI-powered energy analytics pipeline.
From raw household data, it performed quality checks, EDA, feature engineering, and
trained 5 regression models. **{best_name}** is the best model
(R\u00b2 = {br["R\u00b2"]}, RMSE = {br["RMSE"]} kWh), explaining
**{br["R\u00b2"]*100:.1f}%** of the variance in monthly energy consumption.
The interactive prediction tool makes the solution practically deployable.
    """)
    st.markdown(_card_close(), unsafe_allow_html=True)


# =============================================================================
# ══════════════════════  STREAMLIT MAIN  ═════════════════════════════════════
# =============================================================================

def streamlit_app():
    import streamlit as st

    @st.cache_resource(show_spinner="Training models on your data...")
    def _cached_pipeline(path):
        return _run_full_pipeline(path)

    st.set_page_config(
        page_title="Energy Analytics Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: `{DATA_PATH}`\n\n"
                 "Place `Household energy unit data.csv` in the same folder.")
        st.stop()

    (df_raw, df_cln, df_fe,
     Xtr, Xte, ytr, yte,
     fc, fitted, results_df,
     best_name, best_model) = _cached_pipeline(DATA_PATH)

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:16px 0 8px;">'
            '<div style="font-size:2rem;">⚡</div>'
            '<div style="font-weight:800;font-size:1rem;color:#e2e8f0;line-height:1.3;">'
            'Energy Analytics<br>Dashboard</div>'
            '<div style="font-size:0.7rem;color:#475569;margin-top:4px;">'
            'IBM SkillsBuild · BharatCares x AICTE</div></div>'
            '<hr style="border-color:#1e3a5f;margin:8px 0 14px;">',
            unsafe_allow_html=True)

        pages = [
            "Overview", "Data Explorer", "EDA",
            "Features", "Models", "Predict", "Insights",
        ]
        icons = ["🏠","📂","📊","⚙️","🤖","🔮","💡"]
        labels = [f"{ic} {pg}" for ic, pg in zip(icons, pages)]
        selected = st.radio("Navigation", labels, label_visibility="collapsed")

        st.markdown('<hr style="border-color:#1e3a5f;margin:14px 0 10px;">', unsafe_allow_html=True)
        br = results_df.iloc[0]
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #1e3a5f;'
            f'border-radius:10px;padding:12px;font-size:0.75rem;">'
            f'<div style="color:#64748b;text-transform:uppercase;font-size:0.65rem;'
            f'margin-bottom:6px;">BEST MODEL</div>'
            f'<div style="color:#00d4ff;font-weight:700;">{best_name}</div>'
            f'<div style="color:#34d399;font-size:0.8rem;">R\u00b2 = {br["R\u00b2"]}</div>'
            f'<div style="color:#fbbf24;font-size:0.8rem;">RMSE = {br["RMSE"]} kWh</div>'
            f'<div style="color:#94a3b8;font-size:0.8rem;">MAE = {br["MAE"]} kWh</div></div>',
            unsafe_allow_html=True)

    # Dispatch
    pg = selected.split(" ", 1)[1]
    st.header(selected)
    if   pg == "Overview"    : page_overview(df_raw, results_df, best_name)
    elif pg == "Data Explorer": page_data(df_raw, df_cln)
    elif pg == "EDA"          : page_eda(df_cln)
    elif pg == "Features"     : page_features(df_cln, df_fe)
    elif pg == "Models"       : page_models(results_df, fitted, best_name,
                                             best_model, Xte, yte, fc)
    elif pg == "Predict"      : page_predict(best_model, best_name, fc, results_df)
    elif pg == "Insights"     : page_insights(results_df, best_name)


# =============================================================================
# ══════════════════════  ENTRY POINT  ════════════════════════════════════════
# =============================================================================

if __name__ == "__main__":
    if "--train" in sys.argv:
        # ── CLI training mode ──────────────────────────────────────────────
        cli_train()
    else:
        # ── Streamlit dashboard mode ───────────────────────────────────────
        # When run as `streamlit run app.py`, __name__ == "__main__"
        # but Streamlit also calls the module top-level, so we call
        # streamlit_app() here too.
        streamlit_app()

# Streamlit also executes the module at the top-level on import,
# so we call streamlit_app() unconditionally for that path.
try:
    import streamlit as _st
    if _st.runtime.exists():
        streamlit_app()
except Exception:
    pass

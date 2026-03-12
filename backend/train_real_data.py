"""
train_real_data.py
------------------
Trains all UrbanPulse AI ML models using the three real uploaded datasets:

  1. traffic.csv               – Hourly vehicle counts at 4 junctions (2015-2017)
  2. open-meteo-*.xlsx         – Hourly weather at lat 21.97N 78.98E (2000-2009)
  3. public_emdat_project.csv  – Global disaster events with deaths/affected (2000-2024)

Pipeline:
  A) Feature Engineering  – build one merged hourly row per timestamp
  B) Label Engineering    – define flood_risk, congestion, power_outage, disaster_severity
  C) Train 4 models       – RandomForest + GradientBoosting with SMOTE for imbalance
  D) Save .pkl artifacts  – loaded by iutf_ml_model.py at server startup
  E) Patch graph edges    – compute empirical edge probabilities from real data
"""

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

import pickle

# ============================================================
# PATHS — adjust if your files are elsewhere
# ============================================================
TRAFFIC_CSV  = "/tmp/data/traffic.csv"
WEATHER_XLSX = "/mnt/user-data/uploads/open-meteo-21_97N78_98E696m.xlsx"
EMDAT_CSV    = "/tmp/data/public_emdat_project.csv"
OUT_DIR      = Path("/tmp/models")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("  UrbanPulse AI — Real Dataset Training Pipeline")
print("=" * 65)


# ============================================================
# STEP 1 — Load & clean each dataset
# ============================================================
print("\n[1/5] Loading datasets...")

# ── Traffic ──────────────────────────────────────────────────
df_t = pd.read_csv(TRAFFIC_CSV)
df_t["DateTime"] = pd.to_datetime(df_t["DateTime"])
# Pivot: one column per junction
df_traffic = df_t.pivot_table(
    index="DateTime", columns="Junction", values="Vehicles", aggfunc="mean"
).reset_index()
df_traffic.columns = ["time", "vehicles_j1", "vehicles_j2", "vehicles_j3", "vehicles_j4"]
df_traffic["time"] = pd.to_datetime(df_traffic["time"]).dt.floor("h")
df_traffic["total_vehicles"] = df_traffic[["vehicles_j1","vehicles_j2","vehicles_j3","vehicles_j4"]].sum(axis=1)

# Normalise vehicle counts
vmax = df_traffic["total_vehicles"].max()
df_traffic["flow_norm"] = df_traffic["total_vehicles"] / vmax

# Congestion label: top 30% of vehicle counts → congested
thresh_cong = df_traffic["total_vehicles"].quantile(0.70)
df_traffic["congestion_label"] = (df_traffic["total_vehicles"] >= thresh_cong).astype(int)

print(f"  Traffic:  {df_traffic.shape[0]:,} hourly rows  "
      f"(congestion rate={df_traffic['congestion_label'].mean():.1%})")

# ── Weather ───────────────────────────────────────────────────
df_w = pd.read_excel(WEATHER_XLSX, skiprows=3, header=0)
df_w = df_w.dropna(axis=1, how="all")
df_w = df_w[df_w["time"] != "time"].copy()
df_w["time"] = pd.to_datetime(df_w["time"], errors="coerce")
df_w = df_w.dropna(subset=["time"])
df_w["time"] = df_w["time"].dt.floor("h")

# Select key weather features
weather_cols = [
    "time",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "rain (mm)",
    "weather_code (wmo code)",
    "cloud_cover (%)",
    "wind_speed_10m (km/h)",
    "wind_gusts_10m (km/h)",
    "soil_moisture_0_to_7cm (m³/m³)",
    "vapour_pressure_deficit (kPa)",
    "surface_pressure (hPa)",
]
df_weather = df_w[[c for c in weather_cols if c in df_w.columns]].copy()

# Rename for cleanliness
df_weather.columns = [
    c.split(" (")[0].strip().replace(" ", "_").replace("/", "_per_").lower()
    for c in df_weather.columns
]
df_weather.rename(columns={"time": "time"}, inplace=True)

# Numeric conversion
for col in df_weather.columns:
    if col != "time":
        df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce")

# Normalise precipitation
p_max = df_weather["precipitation"].max()
df_weather["rainfall_intensity"] = df_weather["precipitation"] / p_max

# Flood label: precipitation > 5 mm/h  (heavy rain threshold for this region)
df_weather["flood_risk_label"] = (df_weather["precipitation"] > 5.0).astype(int)

# Heatwave label: temperature > 40°C
df_weather["heatwave_label"] = (df_weather["temperature_2m"] > 40.0).astype(int)

print(f"  Weather:  {df_weather.shape[0]:,} hourly rows  "
      f"(flood_risk rate={df_weather['flood_risk_label'].mean():.1%},  "
      f"heatwave rate={df_weather['heatwave_label'].mean():.1%})")

# ── EM-DAT ───────────────────────────────────────────────────
df_e = pd.read_csv(EMDAT_CSV, encoding="latin1")

# Filter India events (dataset location is MP, India)
df_india = df_e[df_e["ISO"] == "IND"].copy()

# Build annual disaster severity index
df_india["year"] = df_india["Start Year"].astype(int)
df_india["Total Deaths"]   = pd.to_numeric(df_india["Total Deaths"],  errors="coerce").fillna(0)
df_india["Total Affected"] = pd.to_numeric(df_india["Total Affected"], errors="coerce").fillna(0)

annual = df_india.groupby("year").agg(
    num_disasters   = ("DisNo.", "count"),
    total_deaths    = ("Total Deaths", "sum"),
    total_affected  = ("Total Affected", "sum"),
).reset_index()

# Disaster severity score (normalised 0-1)
annual["severity_raw"] = (
    0.5 * (annual["total_deaths"]   / annual["total_deaths"].max()) +
    0.3 * (annual["total_affected"] / annual["total_affected"].max()) +
    0.2 * (annual["num_disasters"]  / annual["num_disasters"].max())
)
# High disaster year label: top 40%
thresh_sev = annual["severity_raw"].quantile(0.60)
annual["high_disaster_year"] = (annual["severity_raw"] >= thresh_sev).astype(int)

# Disaster type flags (for edge probability computation)
disaster_type_counts = df_india.groupby(["year", "Disaster Type"]).size().unstack(fill_value=0)
annual = annual.merge(disaster_type_counts, on="year", how="left").fillna(0)

print(f"  EM-DAT:   {df_india.shape[0]} India events across {annual.shape[0]} years  "
      f"(high-disaster-year rate={annual['high_disaster_year'].mean():.1%})")


# ============================================================
# STEP 2 — Feature Engineering
# ============================================================
print("\n[2/5] Engineering features...")

# ── Weather × Traffic merge ───────────────────────────────────
# Traffic is 2015-2017, Weather is 2000-2009 — no overlap by date
# Strategy: use hour-of-day + month pattern matching to align them
# i.e. weather seasonal profile (month, hour) merged onto traffic

# Build weather seasonal profile (avg per month+hour)
df_weather["month"] = df_weather["time"].dt.month
df_weather["hour"]  = df_weather["time"].dt.hour

weather_profile = df_weather.groupby(["month", "hour"]).agg(
    precipitation_avg = ("precipitation", "mean"),
    precipitation_p90 = ("precipitation", lambda x: x.quantile(0.90)),
    temperature_avg   = ("temperature_2m", "mean"),
    humidity_avg      = ("relative_humidity_2m", "mean"),
    wind_avg          = ("wind_speed_10m", "mean"),
    soil_moist_avg    = ("soil_moisture_0_to_7cm", "mean"),
    cloud_avg         = ("cloud_cover", "mean"),
    flood_risk_rate   = ("flood_risk_label", "mean"),
    heatwave_rate     = ("heatwave_label", "mean"),
).reset_index()

# Add temporal features to traffic
df_traffic["month"] = df_traffic["time"].dt.month
df_traffic["hour"]  = df_traffic["time"].dt.hour
df_traffic["day_of_week"] = df_traffic["time"].dt.dayofweek
df_traffic["is_peak_hour"] = df_traffic["hour"].isin([8,9,17,18,19]).astype(int)
df_traffic["is_weekend"]   = (df_traffic["day_of_week"] >= 5).astype(int)

# Merge seasonal weather profile onto traffic
df_merged = df_traffic.merge(weather_profile, on=["month", "hour"], how="left")

# Add annual disaster context (from EM-DAT)
df_merged["year"] = df_traffic["time"].dt.year
df_merged = df_merged.merge(
    annual[["year", "severity_raw", "high_disaster_year", "num_disasters"]],
    on="year", how="left"
)
df_merged["severity_raw"]      = df_merged["severity_raw"].fillna(annual["severity_raw"].mean())
df_merged["high_disaster_year"] = df_merged["high_disaster_year"].fillna(0).astype(int)
df_merged["num_disasters"]      = df_merged["num_disasters"].fillna(annual["num_disasters"].mean())

# Normalise disaster context
df_merged["disaster_context"] = df_merged["severity_raw"]

# Build final normalised columns
df_merged["rainfall_intensity"] = df_merged["precipitation_avg"] / p_max
df_merged["speed_norm"]         = 1.0 - df_merged["flow_norm"]   # inverse proxy
df_merged["occupancy"]          = df_merged["flow_norm"] * (1 + 0.3 * df_merged["flood_risk_rate"])

# Flood risk label: high seasonal rain rate OR detected precipitation
df_merged["flood_risk_label"] = (
    (df_merged["flood_risk_rate"] > 0.05) |
    (df_merged["precipitation_p90"] > 5.0)
).astype(int)

# Power outage label: high wind + high temp + high demand (vehicle flow proxy)
df_merged["power_outage_label"] = (
    (df_merged["wind_avg"] > df_merged["wind_avg"].quantile(0.75)) &
    (df_merged["temperature_avg"] > df_merged["temperature_avg"].quantile(0.65))
).astype(int)

print(f"  Merged dataset: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} cols")
print(f"  Label rates:")
print(f"    congestion_label:    {df_merged['congestion_label'].mean():.1%}")
print(f"    flood_risk_label:    {df_merged['flood_risk_label'].mean():.1%}")
print(f"    power_outage_label:  {df_merged['power_outage_label'].mean():.1%}")
print(f"    high_disaster_year:  {df_merged['high_disaster_year'].mean():.1%}")


# ============================================================
# STEP 3 — Model Training
# ============================================================
print("\n[3/5] Training models...")

# Shared feature set (matches iutf_ml_model.py interface)
FEATURES = [
    "rainfall_intensity",   # from weather
    "flow_norm",            # from traffic
    "occupancy",            # derived
    "speed_norm",           # derived
    "temperature_avg",      # weather
    "humidity_avg",         # weather
    "wind_avg",             # weather
    "soil_moist_avg",       # weather
    "cloud_avg",            # weather
    "disaster_context",     # from EM-DAT
    "is_peak_hour",         # temporal
    "is_weekend",           # temporal
    "month",                # seasonal
]

MODELS_CONFIG = [
    ("congestion",    "congestion_label",    RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ("flood",         "flood_risk_label",    GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)),
    ("power_outage",  "power_outage_label",  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ("disaster_sev",  "high_disaster_year",  GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)),
]

results   = {}
pipelines = {}

df_clean = df_merged[FEATURES + ["congestion_label","flood_risk_label","power_outage_label","high_disaster_year"]].dropna()
print(f"  Clean rows for training: {df_clean.shape[0]:,}")

for model_name, label_col, clf in MODELS_CONFIG:
    print(f"\n  ── {model_name.upper()} ──")

    X = df_clean[FEATURES].values.astype(np.float32)
    y = df_clean[label_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class imbalance
    pos_rate = y_train.mean()
    if 0.05 < pos_rate < 0.45:
        try:
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"    SMOTE: {len(y_train):,} → {len(y_train_res):,} samples  (pos_rate {pos_rate:.1%} → {y_train_res.mean():.1%})")
        except Exception as e:
            print(f"    SMOTE skipped ({e})")
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train
        print(f"    No SMOTE needed (pos_rate={pos_rate:.1%})")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])
    pipeline.fit(X_train_res, y_train_res)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"    Accuracy : {acc:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    {classification_report(y_test, y_pred, target_names=['No','Yes'], digits=3)}")

    results[model_name]   = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4), "label": label_col}
    pipelines[model_name] = pipeline

    # Save model
    pkl_path = OUT_DIR / f"{model_name}_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"    Saved → {pkl_path}")


# ============================================================
# STEP 4 — Feature Importance
# ============================================================
print("\n[4/5] Feature importance (congestion model)...")
rf_clf = pipelines["congestion"].named_steps["clf"]
if hasattr(rf_clf, "feature_importances_"):
    importances = rf_clf.feature_importances_
    fi = pd.DataFrame({"feature": FEATURES, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 60)
        print(f"  {row['feature']:30s} {row['importance']:.4f}  {bar}")


# ============================================================
# STEP 5 — Empirical edge probabilities from real data
# ============================================================
print("\n[5/5] Computing empirical graph edge probabilities...")

# P(Congestion | Heavy Rain) from weather × traffic merged profile
rain_heavy = df_merged[df_merged["precipitation_avg"] > 3.0]
p_cong_given_rain = rain_heavy["congestion_label"].mean() if len(rain_heavy) > 0 else 0.7

# P(Road Flooding | Heavy Rain)
p_flood_given_rain = rain_heavy["flood_risk_label"].mean() if len(rain_heavy) > 0 else 0.55

# P(Drain Overload | Heavy Rain): use p90 precipitation proxy
heavy_p90 = df_merged[df_merged["precipitation_p90"] > 8.0]
p_drain = heavy_p90["flood_risk_label"].mean() if len(heavy_p90) > 0 else 0.65

# P(Emergency Delay | Congestion): high congestion + high disaster context
cong_rows = df_merged[df_merged["congestion_label"] == 1]
p_emergency = (
    cong_rows[cong_rows["disaster_context"] > cong_rows["disaster_context"].quantile(0.5)].shape[0]
    / max(len(cong_rows), 1)
)

# P(Power Outage | Storm): high wind + high cloud
storm_rows = df_merged[
    (df_merged["wind_avg"] > df_merged["wind_avg"].quantile(0.8)) &
    (df_merged["cloud_avg"] > 60)
]
p_power_storm = storm_rows["power_outage_label"].mean() if len(storm_rows) > 0 else 0.6

# P(Heatwave | Extreme Temp): from EM-DAT India
india_extreme = df_india[df_india["Disaster Type"] == "Extreme temperature"]
p_heatwave = min(len(india_extreme) / max(len(df_india), 1) * 5, 0.9)

# P(Flood | Heavy Rain) — EM-DAT validated
india_floods = df_india[df_india["Disaster Type"] == "Flood"]
p_flood_emdat = len(india_floods) / max(len(df_india), 1)

edge_probabilities = {
    ("Heavy Rain",          "Drain Overload"):           round(float(p_drain), 4),
    ("Heavy Rain",          "Road Flooding"):            round(float(p_flood_given_rain), 4),
    ("Drain Overload",      "Flooding"):                 round(float(min(p_flood_given_rain * 1.15, 0.95)), 4),
    ("Flooding",            "Traffic Congestion"):       round(float(p_cong_given_rain), 4),
    ("Road Flooding",       "Traffic Congestion"):       round(float(p_cong_given_rain * 0.95), 4),
    ("Traffic Congestion",  "Emergency Vehicle Delay"):  round(float(p_emergency), 4),
    ("Power Grid Failure",  "Traffic Signal Failure"):   round(float(p_power_storm * 1.1), 4),
    ("Traffic Signal Failure", "Traffic Congestion"):   round(float(p_power_storm), 4),
    ("Heatwave",            "Power Grid Failure"):       round(float(p_heatwave), 4),
    ("Flooding",            "Power Grid Failure"):       round(float(p_flood_emdat * 0.8), 4),
}

print("\n  Edge probability table:")
print(f"  {'Source':35s} {'Target':30s} {'Probability':>12}")
print("  " + "-" * 80)
for (src, dst), prob in edge_probabilities.items():
    print(f"  {src:35s} {dst:30s} {prob:>12.4f}")

# Save edge probabilities as JSON
edge_probs_serializable = {f"{k[0]}|||{k[1]}": v for k, v in edge_probabilities.items()}
with open(OUT_DIR / "edge_probabilities.json", "w") as f:
    json.dump(edge_probs_serializable, f, indent=2)

# Save summary metrics
with open(OUT_DIR / "training_summary.json", "w") as f:
    json.dump({
        "models":            results,
        "features":          FEATURES,
        "training_rows":     int(df_clean.shape[0]),
        "data_sources": {
            "traffic":  {"rows": int(df_traffic.shape[0]), "junctions": 4, "years": "2015-2017"},
            "weather":  {"rows": int(df_weather.shape[0]), "location": "21.97N 78.98E (MP, India)", "years": "2000-2009"},
            "emdat":    {"rows": int(df_india.shape[0]),   "country": "India", "years": "2000-2024"},
        },
    }, f, indent=2)

print(f"\n  Saved edge_probabilities.json → {OUT_DIR}")
print(f"  Saved training_summary.json   → {OUT_DIR}")

print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print(f"  Models saved to: {OUT_DIR}")
for name, res in results.items():
    print(f"  {name:15s}  acc={res['accuracy']:.4f}  auc={res['roc_auc']:.4f}")
print("=" * 65)

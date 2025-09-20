import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/isoc_ci_it_h.tsv")
MODEL_PATH = Path("models/broadband_predictor_linear.json")

st.set_page_config(page_title="AI ICT Insights", layout="wide")
st.title("AI ICT Insights: Broadband Adoption Predictor")

# -----------------------------
# Helpers: portable model
# -----------------------------
def load_portable_model(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def portable_predict(row_geo, row_year, row_value, row_value_prev, row_value_prev2,
                     row_value_prev3, row_roll_mean3, row_growth_1y, artifact) -> float:
    geo_levels = artifact["meta"]["features"]["categorical_levels"]["geo"]
    geo_vec = [1.0 if g == row_geo else 0.0 for g in geo_levels]

    num_order = artifact["meta"]["features"]["numeric"]
    nums = {
        "year": row_year,
        "value": row_value,
        "value_prev": row_value_prev,
        "value_prev2": row_value_prev2,
        "value_prev3": row_value_prev3,
        "roll_mean3": row_roll_mean3,
        "growth_1y": row_growth_1y
    }
    x_num = np.array([nums[k] for k in num_order], dtype=float)

    imp_stats = np.array(artifact["preprocessing"]["imputer_statistics"], dtype=float)
    nan_mask = np.isnan(x_num)
    x_num[nan_mask] = imp_stats[nan_mask]

    mean = np.array(artifact["preprocessing"]["scaler_mean"], dtype=float)
    scale = np.array(artifact["preprocessing"]["scaler_scale"], dtype=float)
    x_num_std = (x_num - mean) / scale

    x = np.concatenate([np.array(geo_vec, dtype=float), x_num_std])
    coef = np.array(artifact["model"]["coef"], dtype=float)
    intercept = float(artifact["model"]["intercept"])
    return float(np.dot(x, coef) + intercept)

# -----------------------------
# Data loading and cleaning
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_tsv(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    return df

def to_num(x):
    if pd.isna(x):
        return np.nan
    x = x.strip()
    if x == ":":
        return np.nan
    num = ""
    dot = False
    for ch in x:
        if ch.isdigit():
            num += ch
        elif ch == "." and not dot:
            num += ch
            dot = True
        else:
            break
    return float(num) if num else np.nan

@st.cache_data(show_spinner=False)
def load_long_df(tsv_path: Path) -> pd.DataFrame:
    raw = load_raw_tsv(tsv_path)
    first_col = raw.columns[0]
    year_cols = [c for c in raw.columns if c.strip().isdigit()]

    parts = raw[first_col].str.split(",", expand=True)
    parts.columns = ["indic_is", "unit", "hhtyp", "geo"]
    df = pd.concat([parts, raw[year_cols]], axis=1)

    for c in year_cols:
        df[c] = df[c].apply(to_num)

    slice_df = df[
        (df["indic_is"] == "H_BBFIX") &
        (df["unit"] == "PC_HH") &
        (df["hhtyp"] == "A1")
    ].copy()

    long_df = slice_df.melt(
        id_vars=["indic_is", "unit", "hhtyp", "geo"],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    long_df["year"] = long_df["year"].str.strip().astype(int)
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    return long_df[["geo", "year", "value"]].sort_values(["geo", "year"])

def build_features_for_geo(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("year").copy()
    g["value_prev"]  = g["value"].shift(1)
    g["value_prev2"] = g["value"].shift(2)
    g["value_prev3"] = g["value"].shift(3)
    g["roll_mean3"]  = g["value"].rolling(3).mean()
    g["growth_1y"]   = g["value"] - g["value_prev"]
    return g

# -----------------------------
# Load artifacts
# -----------------------------
if not DATA_PATH.exists():
    st.error(f"Missing data file at {DATA_PATH}. Add isoc_ci_it_h.tsv to data/ and redeploy.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing model file at {MODEL_PATH}. Add broadband_predictor_linear.json to models/ and redeploy.")
    st.stop()

artifact = load_portable_model(MODEL_PATH)
long_df = load_long_df(DATA_PATH)

# -----------------------------
# Sidebar controls
# -----------------------------
all_countries = sorted(long_df["geo"].unique().tolist())
default_country = all_countries[0] if all_countries else None
country = st.sidebar.selectbox("Country", options=all_countries, index=all_countries.index(default_country) if default_country else 0)
st.sidebar.write("Model:", artifact["meta"]["model_type"])
st.sidebar.write("Validation MAE:", f"{artifact['meta']['metrics']['val_mae']:.2f}")
st.sidebar.write("Validation R2:", f"{artifact['meta']['metrics']['val_r2']:.3f}")

# -----------------------------
# Main view: history and prediction
# -----------------------------
geo_df = long_df[long_df["geo"] == country].copy()
if geo_df.empty:
    st.warning("No data for the selected country.")
    st.stop()

feat_geo = build_features_for_geo(geo_df)

# latest row that has at least previous value
latest_row = feat_geo.dropna(subset=["value_prev"]).tail(1)
if latest_row.empty:
    st.warning("Not enough history to form features for this country.")
    st.dataframe(geo_df)
    st.stop()

r = latest_row.iloc[0]
pred_next = portable_predict(
    row_geo=country,
    row_year=float(r["year"]),
    row_value=float(r["value"]),
    row_value_prev=float(r["value_prev"]),
    row_value_prev2=float(r["value_prev2"]) if not math.isnan(r["value_prev2"]) else float(r["value_prev"]),
    row_value_prev3=float(r["value_prev3"]) if not math.isnan(r["value_prev3"]) else float(r["value_prev"]),
    row_roll_mean3=float(r["roll_mean3"]) if not math.isnan(r["roll_mean3"]) else float(r["value"]),
    row_growth_1y=float(r["growth_1y"]) if not math.isnan(r["growth_1y"]) else 0.0,
    artifact=artifact
)

next_year = int(r["year"]) + 1

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader(f"History for {country}")
    chart_df = geo_df.rename(columns={"value": "broadband_%"}).set_index("year")[["broadband_%"]]
    st.line_chart(chart_df)

with col2:
    st.subheader("Prediction")
    st.metric(label=f"Predicted broadband adoption in {next_year}", value=f"{pred_next:.2f}%")
    st.caption("Target: fixed broadband, % of households (A1).")

st.divider()

st.subheader("Latest features used for prediction")
feat_view = latest_row[["geo","year","value","value_prev","value_prev2","value_prev3","roll_mean3","growth_1y"]].copy()
feat_view.columns = ["geo","year","value_t","value_t-1","value_t-2","value_t-3","rolling_mean_3","growth_1y"]
st.dataframe(feat_view.reset_index(drop=True))

st.divider()

with st.expander("About this model"):
    st.write("This app loads a portable JSON model trained in Colab. The model is a small ridge regression over one-hot encoded country and standardized numeric features. Missing numeric inputs are imputed with stored medians before scaling. No scikit-learn is required at runtime.")
    st.json(artifact["meta"], expanded=False)

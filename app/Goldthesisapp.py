# st.py
# Streamlit sederhana – Prediksi Harga Emas (ΔMA3 ARIMAX, 5 tahun)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
from datetime import datetime, timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 0. Konfigurasi dasar
# ==============================
st.set_page_config(
    page_title="Prediksi Harga Emas Harian",
    layout="wide"
)

# >>> GANTI DENGAN API KEY-MU SENDIRI <<<
FRED_API_KEY = "4c6126e875c9d02dfb10781331c01a2d"  # pakai key yang sama kaya notebook

# ==============================
# 1. Fungsi ambil data
# ==============================

def get_gold_daily_stooq():
    """Ambil data XAUUSD daily dari Stooq."""
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns=str.lower)   # date, open, high, low, close, volume
    df = df[["date", "open", "high", "low", "close"]].sort_values("date")

    # filter 5 tahun terakhir biar konsisten
    start_date = datetime.now() - timedelta(days=5 * 365)
    df = df[df["date"] >= pd.Timestamp(start_date.date())].reset_index(drop=True)
    return df


def get_fred_series(series_id, api_key, frequency="d", start_years=5):
    """Ambil data FRED (misal DXY)."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "frequency": frequency,
        "observation_start": (datetime.now() - timedelta(days=start_years * 365)).strftime("%Y-%m-%d"),
        "observation_end": datetime.now().strftime("%Y-%m-%d"),
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "observations" not in data:
        st.error(f"FRED API error untuk {series_id}: {data}")
        return None

    rows = []
    for obs in data["observations"]:
        value = obs["value"]
        rows.append(
            {
                "date": obs["date"],
                "value": None if value == "." else float(value),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().reset_index(drop=True)
    return df


def eval_model(y_true, y_pred):
    """Balikin MAE, RMSE, R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ==============================
# 2. Pipeline data + model
# ==============================

@st.cache_data(show_spinner=True)
def load_and_fit_model():
    # --- ambil data ---
    gold_df = get_gold_daily_stooq()
    dxy_df = get_fred_series("DTWEXBGS", "4c6126e875c9d02dfb10781331c01a2d", "d", start_years=5)

    # merge
    df = pd.merge(gold_df, dxy_df, on="date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.rename(columns={"value": "dxy"})

    # pakai date sebagai index
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")

    # bersihkan hanya kolom fitur (dxy)
    df["dxy"] = df["dxy"].ffill().bfill()

    # buang index duplikat kalau ada
    df = df[~df.index.duplicated(keep="first")]

    # simpan versi untuk plotting harga original
    df_close_only = df[["close"]].copy()

    # ---------- Feature engineering ΔMA3 ----------
    df = df.sort_index()
    df["ma3"] = df["close"].rolling(3).mean()

    # target ΔMA3 = MA3_{t+1} - MA3_t
    df["target"] = df["ma3"].shift(-1) - df["ma3"]

    feature_cols = ["close", "ma3", "dxy"]
    lags = [1, 3, 7, 14]

    for col in feature_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # drop NaN dari rolling + lag
    df = df.dropna()

    y = df["target"]
    X = df.drop(columns=["target", "ma3", "close", "dxy"])

    # ---------- Train-Test split time series ----------
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Naive baseline: ΔMA3 = 0
    naive_train = np.zeros_like(y_train)
    naive_test = np.zeros_like(y_test)

    # ---------- ARIMAX ----------
    arimax_model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=(1, 0, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    arimax_res = arimax_model.fit(disp=False)

    # in-sample
    y_pred_train = arimax_res.fittedvalues.loc[y_train.index]

    # out-of-sample
    y_pred_test = arimax_res.forecast(steps=len(y_test), exog=X_test)
    y_pred_test.index = y_test.index

    # ---------- Evaluasi ----------
    mae_naive_train, rmse_naive_train, r2_naive_train = eval_model(y_train, naive_train)
    mae_naive_test, rmse_naive_test, r2_naive_test = eval_model(y_test, naive_test)

    mae_ar_train, rmse_ar_train, r2_ar_train = eval_model(y_train, y_pred_train)
    mae_ar_test, rmse_ar_test, r2_ar_test = eval_model(y_test, y_pred_test)

    metrics = {
        "Naive (Train)": (mae_naive_train, rmse_naive_train, r2_naive_train),
        "Naive (Test)": (mae_naive_test, rmse_naive_test, r2_naive_test),
        "ARIMAX (Train)": (mae_ar_train, rmse_ar_train, r2_ar_train),
        "ARIMAX (Test)": (mae_ar_test, rmse_ar_test, r2_ar_test),
    }

    # ---------- Rekonstruksi harga besok ----------
    ma3_today = df["ma3"].iloc[-1]
    delta_pred = y_pred_test.iloc[-1]
    ma3_pred_tomorrow = ma3_today + delta_pred

    close_t = df["close"].iloc[-1]
    close_tm1 = df["close"].iloc[-2]
    close_pred_tomorrow = 3 * ma3_pred_tomorrow - close_t - close_tm1

    # DataFrame untuk plot Actual vs Predicted (titik terakhir beda)
    df_pred = df[["close", "ma3"]].copy()
    df_pred["ma3_pred"] = df_pred["ma3"]
    df_pred["close_pred"] = df_pred["close"]

    df_pred.iloc[-1, df_pred.columns.get_loc("ma3_pred")] = ma3_pred_tomorrow
    df_pred.iloc[-1, df_pred.columns.get_loc("close_pred")] = close_pred_tomorrow

    return df_close_only, df_pred, metrics, close_t, close_pred_tomorrow


# ==============================
# 3. UI Streamlit
# ==============================

st.title("📈 Prediksi Harga Emas Harian – ΔMA3 ARIMAX (5 Tahun Terakhir)")
st.caption("Data XAUUSD dari Stooq + DXY dari FRED (5 tahun, tanpa input manual).")

with st.spinner("Mengambil data dan melatih model..."):
    df_close, df_pred, metrics, last_close, close_pred_tomorrow = load_and_fit_model()

# --- Section 1: Harga Emas 5 Tahun Terakhir ---
st.subheader("📊 Harga Emas Harian – 5 Tahun Terakhir")
st.line_chart(df_close)

# --- Section 2: Actual vs Predicted ---
st.subheader("🔮 Actual vs Predicted Close (titik terakhir = prediksi besok)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_pred.index, df_pred["close"], label="Actual Close")
ax.plot(df_pred.index, df_pred["close_pred"], linestyle="--", label="Predicted Close (Reconstructed)")
ax.legend()
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (USD)")
ax.grid(True)
st.pyplot(fig)

# --- Section 3: Ringkasan Prediksi Besok ---
st.subheader("📌 Ringkasan Prediksi Besok (ΔMA3 Scenario)")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Harga Close Terakhir",
        f"{last_close:,.2f} USD",
    )
with col2:
    st.metric(
        "Prediksi Harga Close Besok",
        f"{close_pred_tomorrow:,.2f} USD",
        delta=f"{(close_pred_tomorrow - last_close):+.2f} USD",
    )

# --- Section 4: Metrik Model ---
st.subheader("📉 Evaluasi Model pada Target ΔMA3")

metrics_df = (
    pd.DataFrame(metrics, index=["MAE", "RMSE", "R²"])
    .T
    .style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"})
)

st.dataframe(metrics_df, use_container_width=True)

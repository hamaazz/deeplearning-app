import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense

# ================== Utility ================== #
def load_example_monthly(periods=72, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=periods, freq="M")
    trend = np.linspace(80000, 150000, periods)
    season = 20000 * np.sin(np.arange(periods) * 2 * np.pi / 12)
    noise = rng.normal(0, 5000, periods)
    return pd.DataFrame({"Posting Date": idx, "Quantity": trend + season + noise})


def ensure_datetime(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col])


def train_test_split_ts(df, test_size):
    n = len(df)
    n_test = max(1, int(n * test_size))
    return df.iloc[:-n_test], df.iloc[-n_test:]


def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def preprocess_series_monthly(series):
    s = series.astype(float).copy()

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    lower = max(Q1 - 3 * IQR, 0)
    s_clip = s.clip(lower=lower, upper=upper)

    s_log = np.log1p(s_clip)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(s_log.values.reshape(-1, 1)).flatten()

    return scaled, scaler


# ================== UI ================== #
st.set_page_config(page_title="Monthly DL Forecasting", layout="wide")
st.title("📈 Deep Learning Forecasting Bulanan (LSTM / BiLSTM / GRU / BiGRU)")
st.subheader("Kerja Prakter Industri - 23524032 - 23524037")

with st.sidebar:
    st.header("⚙️ Pengaturan Data")
    source = st.radio("Sumber data", ["Upload CSV", "Generate Sample Dataset"], index=1)
    date_col = st.text_input("Kolom tanggal (bulanan)", "Posting Date")
    y_col = st.text_input("Kolom target", "Quantity")

    test_size = st.slider("Proporsi Test", 0.1, 0.4, 0.2)
    window = st.slider("Window Size (lookback, bulan)", 3, 24, 12)

    # Horizon dibatasi 3 / 6 / 9 / 12 bulan
    horizon = st.sidebar.slider(
        "Horizon Forecast (bulan)",
        min_value=3,
        max_value=12,
        step=3,
        value=3
    )

    model_name = st.selectbox(
        "Model DL",
        ["LSTM", "BiLSTM", "GRU", "BiGRU"],
    )

# ================== Load Data ================== #
st.markdown("### 1) Data Bulanan")

if source == "Upload CSV":
    file = st.file_uploader("Upload file CSV (format: Posting Date, Quantity)", type=["csv"])
    if file is None:
        st.stop()
    df_raw = pd.read_csv(file)
else:
    df_raw = load_example_monthly()

if date_col not in df_raw.columns or y_col not in df_raw.columns:
    st.error(f"Kolom '{date_col}' atau '{y_col}' tidak ditemukan di file.")
    st.stop()

df = ensure_datetime(df_raw, date_col).sort_values(date_col)
df = df[[date_col, y_col]].rename(columns={date_col: "date", y_col: "y"})

df = df.set_index("date").asfreq("M")
df["y"] = df["y"].interpolate()

st.write("Preview data bulanan:")
st.dataframe(df.head(), use_container_width=True)
st.plotly_chart(px.line(df.reset_index(), x="date", y="y"), use_container_width=True)

# ============ EXECUTE BUTTON ============ #
run_process = st.button("🚀 EXECUTE FORECASTING")

if not run_process:
    st.stop()

if len(df) <= window + 5:
    st.error("Data terlalu pendek.")
    st.stop()

# ================== Train/Test Split ================== #
train_df, test_df = train_test_split_ts(df, test_size)
y_test = test_df["y"].values

# ================== Preprocessing ================== #
scaled_all, scaler = preprocess_series_monthly(df["y"])
X_all, y_all = create_sequences(scaled_all, window)

n_test = len(test_df)
X_train = X_all[:-n_test]
y_train_seq = y_all[:-n_test]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# ================== Build Model ================== #
model = Sequential()

if model_name == "LSTM":
    model.add(LSTM(64, input_shape=(window, 1)))
elif model_name == "BiLSTM":
    model.add(Bidirectional(LSTM(64), input_shape=(window, 1)))
elif model_name == "GRU":
    model.add(GRU(64, input_shape=(window, 1)))
elif model_name == "BiGRU":
    model.add(Bidirectional(GRU(64), input_shape=(window, 1)))

model.add(Dense(32, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

with st.spinner("Training model ..."):
    history = model.fit(
        X_train,
        y_train_seq,
        epochs=50,
        batch_size=8,
        verbose=0,
        validation_split=0.1,
    )

st.success("Training selesai!")

# ================== STABLE TEST PREDICTION ================== #
st.markdown("### 4) Evaluasi Model (Test)")

pred_list = []
scaled_full = scaled_all.copy()

start_pos = len(scaled_full) - (window + len(test_df))

for i in range(len(test_df)):
    seq = scaled_full[start_pos + i : start_pos + i + window]
    seq = np.array(seq).reshape(1, window, 1)

    next_scaled = model.predict(seq, verbose=0)[0, 0]
    pred_list.append(next_scaled)

pred_scaled = np.array(pred_list).reshape(-1, 1)
pred_log = scaler.inverse_transform(pred_scaled)
pred = np.expm1(pred_log).flatten()

test_df["pred"] = np.round(pred).astype(int)

# Metrics
mae = mean_absolute_error(test_df["y"], test_df["pred"])
rmse = sqrt(mean_squared_error(test_df["y"], test_df["pred"]))
mape_val = mape(test_df["y"], test_df["pred"])

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("MAPE (%)", f"{mape_val:.2f}")

st.plotly_chart(
    px.line(test_df.reset_index(), x="date", y=["y", "pred"], title="Actual vs Pred"),
    use_container_width=True,
)

# ================== Forecast Forward ================== #
st.markdown("### 5) Forecast Bulanan")

last_seq = scaled_all[-window:].reshape(1, window, 1)
future_scaled = []

for _ in range(horizon):
    next_pred = model.predict(last_seq, verbose=0)[0, 0]
    future_scaled.append(next_pred)
    last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

future_scaled = np.array(future_scaled).reshape(-1, 1)
future_log = scaler.inverse_transform(future_scaled)
future = np.expm1(future_log).flatten()
future = np.round(future).astype(int)

future_index = pd.date_range(df.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
fcst_df = pd.DataFrame({"date": future_index, "forecast": future}).set_index("date")

st.plotly_chart(
    px.line(
        pd.concat(
            [
                df[["y"]].iloc[-24:].rename(columns={"y": "Actual"}),
                fcst_df.rename(columns={"forecast": "Forecast"}),
            ],
            axis=1,
        ).reset_index(),
        x="date",
        y=["Actual", "Forecast"],
        title="Forecast Bulanan",
    ),
    use_container_width=True,
)

st.dataframe(fcst_df)

# ================== KPI TRIWULAN ================== #
st.markdown("### ⭐ KPI Forecast Triwulanan")

fcst_df_q = fcst_df.copy()
fcst_df_q["quarter"] = fcst_df_q.index.to_period("Q")

kpi_df = fcst_df_q.groupby("quarter")["forecast"].sum().reset_index()

cols = st.columns(len(kpi_df))
for i, row in kpi_df.iterrows():
    cols[i].metric(f"Triwulan {row['quarter']}", f"{int(row['forecast']):,}")

st.dataframe(kpi_df)

# ================== Download ================== #
st.download_button(
    "💾 Download Forecast CSV",
    fcst_df.reset_index().to_csv(index=False).encode("utf-8"),
    "forecast_monthly_dl.csv",
    "text/csv",
)

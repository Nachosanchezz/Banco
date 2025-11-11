# app.py — StockSage (versión estética)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import streamlit as st
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# Config & tema visual
# =========================
st.set_page_config(page_title="StockSage", layout="wide")
pio.templates.default = "plotly_white"     # Tema limpio para todas las figuras

# --- CSS ligero para “look&feel” ---
st.markdown("""
<style>
/* Tipografía y anchuras */
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
section.main > div { padding-top: 1rem; }

/* Título principal */
.hero-title { font-size: 32px; font-weight: 800; margin-bottom: 4px; }
.hero-sub   { font-size: 16px; color: #616771; margin-bottom: 1.25rem; }

/* Tarjetas KPI */
.kpi { background: #ffffff; border: 1px solid #ECEFF4; border-radius: 14px; padding: 14px 16px; }
.kpi h4{ font-size: 12px; font-weight: 600; color:#6b7280; margin:0 0 6px 0; }
.kpi .v { font-size: 20px; font-weight: 800; }
.kpi .d { font-size: 12px; color:#6b7280; }

/* Tabs con un poco de espacio */
.stTabs [role="tablist"] { gap: .5rem }
.stTabs [role="tab"] { padding: .5rem .9rem }

</style>
""", unsafe_allow_html=True)

LOOKBACK = 60
FEATURE_COLS = [
    "Open","High","Low","Volume",
    "Range","Return","MA_7","MA_30","Volatility_30","Body",
    "evento_negativo"
]
TARGET_COL = "Close"

# =========================
# Modelo LSTM (tu clase)
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)      # (batch, seq_len, hidden)
        out = out[:, -1, :]        # último paso temporal
        return self.head(out)      # (batch, 1)

@st.cache_resource
def load_model(pt_path, n_features):
    model = LSTMRegressor(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def load_scalers(x_path, y_path):
    scaler_X = joblib.load(x_path)
    scaler_y = joblib.load(y_path)
    return scaler_X, scaler_y

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def create_sequences(mat_X, lb):
    xs = []
    for i in range(len(mat_X) - lb):
        xs.append(mat_X[i:i+lb])
    return np.array(xs)

def recursive_forecast(df_raw, model, scaler_X, scaler_y, n_days=7):
    window_raw = df_raw.tail(LOOKBACK).copy()
    buf = df_raw.copy()
    fut_dates, fut_preds = [], []

    for _ in range(n_days):
        w_scaled = window_raw.copy()
        cont_cols = [c for c in FEATURE_COLS if c not in ["evento_negativo","Return"]]
        w_scaled[cont_cols] = scaler_X.transform(window_raw[cont_cols])

        X_seq = w_scaled[FEATURE_COLS].astype(float).values
        xb = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_hat_scaled = model(xb).cpu().numpy().reshape(-1,1)
        y_hat = scaler_y.inverse_transform(y_hat_scaled).reshape(-1)[0]

        next_date = pd.to_datetime(buf["Date"].iloc[-1]) + BDay(1)
        fut_dates.append(next_date)
        fut_preds.append(y_hat)

        last_row = buf.iloc[-1]
        new_row = last_row.copy()
        new_row["Date"]  = next_date
        new_row["Open"]  = y_hat
        new_row["High"]  = max(y_hat, last_row["High"])
        new_row["Low"]   = min(y_hat, last_row["Low"])
        new_row["Close"] = y_hat
        new_row["Volume"] = last_row["Volume"]
        new_row["evento_negativo"] = 0

        buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)
        buf["Return"]        = buf["Close"].pct_change()
        buf["MA_7"]          = buf["Close"].rolling(7).mean()
        buf["MA_30"]         = buf["Close"].rolling(30).mean()
        buf["Volatility_30"] = buf["Return"].rolling(30).std()
        buf["Range"]         = buf["High"] - buf["Low"]
        buf["Body"]          = (buf["Close"] - buf["Open"]).abs()

        window_raw = buf.tail(LOOKBACK).copy()

    return pd.DataFrame({"Date": fut_dates, "Forecast": fut_preds})

# =========================
# Utilidades
# =========================
def estimate_sigma(y_real_tail, y_pred_tail):
    resid = np.array(y_real_tail) - np.array(y_pred_tail)
    if len(resid) < 2:
        return 0.0
    use = resid[-50:] if len(resid) > 50 else resid
    return float(np.std(use))

def quick_backtest(dates, real, pred):
    dates = np.array(dates); real = np.array(real); pred = np.array(pred)
    if len(real) < 3:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan,
                "hit": np.nan, "equity": np.array([]), "equity_dates": np.array([])}
    signal = np.where(pred[1:] > real[:-1], 1, -1)
    ret = (real[1:] - real[:-1]) / np.where(real[:-1]==0, 1e-9, real[:-1])
    strat = signal * ret
    hit = np.mean(np.sign(pred[1:] - real[:-1]) == np.sign(real[1:] - real[:-1]))
    rmse = float(np.sqrt(np.mean((pred-real)**2)))
    mae  = float(np.mean(np.abs(pred-real)))
    mape = float(np.mean(np.abs((pred-real)/np.where(real==0,1e-9,real))))*100
    eq = (1+strat).cumprod()
    return {"rmse": rmse, "mae": mae, "mape": mape, "hit": hit,
            "equity": eq, "equity_dates": dates[1:]}

def scenario_first_forecast(df_raw, model, scaler_X, scaler_y, shock_pct):
    tmp = df_raw.copy()
    tmp.loc[tmp.index[-1], "Close"] *= (1+shock_pct)
    tmp.loc[tmp.index[-1], "Open"]  *= (1+shock_pct)
    tmp["Return"]        = tmp["Close"].pct_change()
    tmp["MA_7"]          = tmp["Close"].rolling(7).mean()
    tmp["MA_30"]         = tmp["Close"].rolling(30).mean()
    tmp["Volatility_30"] = tmp["Return"].rolling(30).std()
    tmp["Range"]         = tmp["High"] - tmp["Low"]
    tmp["Body"]          = (tmp["Close"] - tmp["Open"]).abs()

    window_raw = tmp.tail(LOOKBACK).copy()
    cont_cols = [c for c in FEATURE_COLS if c not in ["evento_negativo","Return"]]
    w_scaled = window_raw.copy()
    w_scaled[cont_cols] = scaler_X.transform(window_raw[cont_cols])

    X_seq = w_scaled[FEATURE_COLS].astype(float).values
    xb = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_scaled = model(xb).cpu().numpy().reshape(-1,1)
    return float(scaler_y.inverse_transform(y_scaled).reshape(-1)[0])

# =========================
# Header + Sidebar
# =========================
st.markdown('<div class="hero-title">StockSage</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predicción de BBVA y Santander a 7 días hábiles con LSTM. '
            '<b>Uso educativo, no es recomendación de inversión.</b></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Controles")
    ticker = st.selectbox("Activo", ["BBVA", "Santander"], index=0)
    horizon = st.slider("Horizonte (días hábiles)", 3, 20, 7)
    n_hist = st.slider("Histórico a mostrar (días)", 60, 200, 120)
    st.caption("Consejo: usa 120–160 días para un gráfico más suave.")

# Rutas por activo (ajusta a tu estructura)
if ticker == "BBVA":
    CSV = "data/bbva_data_actualizado.csv"
    PT  = "models/lstm_bbva3.pt"
    SX  = "models/scaler_X_bbva3.pkl"
    SY  = "models/scaler_y_bbva3.pkl"
else:
    CSV = "data/santander_data_actualizado.csv"
    PT  = "models/lstm_santander2.pt"
    SX  = "models/scaler_X_santander2.pkl"
    SY  = "models/scaler_y_santander2.pkl"

# =========================
# Carga y predicción
# =========================
df = load_data(CSV)
scaler_X, scaler_y = load_scalers(SX, SY)

df_scaled = df.copy()
cont_cols = [c for c in FEATURE_COLS if c not in ["evento_negativo","Return"]]
df_scaled[cont_cols] = scaler_X.transform(df_scaled[cont_cols])
X_all = df_scaled[FEATURE_COLS].astype(float).values
dates_all = df["Date"].values
X_seq = create_sequences(X_all, LOOKBACK)

model = load_model(PT, n_features=X_seq.shape[-1])
with torch.no_grad():
    y_pred_scaled = model(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy().reshape(-1,1)
y_pred_on_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
y_real_aligned = df[TARGET_COL].values[LOOKBACK:]
dates_aligned  = pd.to_datetime(dates_all[LOOKBACK:])

y_real_tail = y_real_aligned[-n_hist:]
y_pred_tail = y_pred_on_real[-n_hist:]
dates_tail  = dates_aligned[-n_hist:]

df_future = recursive_forecast(df.copy(), model, scaler_X, scaler_y, n_days=horizon)

# =========================
# KPI “tarjetas”
# =========================
last_price = float(df[TARGET_COL].iloc[-1])
pred_1d = float(df_future["Forecast"].iloc[0])
sigma = estimate_sigma(y_real_tail, y_pred_tail)
ci95 = 1.96 * sigma
low, high = pred_1d - ci95, pred_1d + ci95
prob_up = 0.5 if sigma == 0 else float(1 - 0.5*np.exp(-abs(pred_1d-last_price)/(sigma+1e-9)))

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><h4>Cierre último</h4><div class="v">{last_price:,.2f} €</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><h4>Predicción 1ᵉʳ día hábil</h4><div class="v">{pred_1d:,.2f} €</div>'
            f'<div class="d">Δ {pred_1d-last_price:+.2f} €</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><h4>Prob. subida (heurística)</h4><div class="v">{prob_up*100:,.0f}%</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><h4>Bandas 95% aprox.</h4><div class="v">{low:,.2f} · {high:,.2f} €</div>'
            f'<div class="d">σ ≈ {sigma:,.4f}</div></div>', unsafe_allow_html=True)

st.markdown("")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Gráfico", "Señal", "Backtest exprés", "Análisis de sensibilidad"])

# --- Tab 1: Gráfico con banda de confianza en forecast
with tab1:
    fig = go.Figure()

    # Histórico y pred sobre reales
    fig.add_trace(go.Scatter(
        x=dates_tail, y=y_real_tail, name="Real",
        mode="lines", line=dict(width=2, color="#1f2a44")
    ))
    fig.add_trace(go.Scatter(
        x=dates_tail, y=y_pred_tail, name="Pred sobre reales (LSTM)",
        mode="lines", line=dict(width=2, dash="dash", color="#2f6df6")
    ))

    # SOLO el forecast (SIN banda 95% ni límites)
    fig.add_trace(go.Scatter(
        x=df_future["Date"], y=df_future["Forecast"],
        name=f"Forecast +{horizon} hábiles",
        mode="lines+markers",
        line=dict(color="#2f6df6", width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=480,
        yaxis_title="€",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Señal (repite KPIs con explicación)
with tab2:
    st.write("**Lectura rápida de la señal**")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"""
        - La predicción para el **próximo día hábil** es **{pred_1d:,.2f} €**  
        - Cambio respecto al cierre: **{pred_1d-last_price:+.2f} €**  
        - **Bandas 95%** aproximadas: **[{low:,.2f}, {high:,.2f}] €**  
        - Probabilidad **heurística** de que suba: **{prob_up*100:,.0f}%**
        """)
        st.caption("Nota: la probabilidad es un *proxy* basado en la desviación de errores recientes; no es una probabilidad estadística calibrada.")
    with c2:
        st.info("Consejo: observa si el forecast queda fuera de la banda para detectar posibles sorpresas/rupturas.")

# --- Tab 3: Backtest exprés
with tab3:
    res_bt = quick_backtest(dates_tail, y_real_tail, y_pred_tail)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{res_bt['rmse']:.3f}")
    m2.metric("MAE",  f"{res_bt['mae']:.3f}")
    m3.metric("MAPE", f"{res_bt['mape']:.2f}%")
    m4.metric("Hit-rate", f"{res_bt['hit']*100:.0f}%" if not np.isnan(res_bt['hit']) else "—")

    if res_bt["equity"].size > 0:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=res_bt["equity_dates"], y=res_bt["equity"],
                                    mode="lines", name="Índice estrategia",
                                    line=dict(color="#1f2a44", width=2)))
        fig_eq.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=300,
                             yaxis_title="Índice (base=1)", xaxis_title=None)
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No hay suficientes datos para el backtest en el tramo mostrado.")

# --- Tab 4: ¿Y si…? (sensibilidad)
with tab4:
    shock = st.slider("Shock del último cierre (%)", -3, 3, 0)
    if shock != 0:
        scen_pred = scenario_first_forecast(df, model, scaler_X, scaler_y, shock/100)
        delta_scen = scen_pred - pred_1d
        st.success(f"Con un shock de {shock:+d}%, la predicción del próximo día sería **{scen_pred:.2f} €** "
                   f"({delta_scen:+.2f} € vs. predicción base).")
    else:
        st.caption("Mueve el deslizador para simular una sorpresa de mercado (±3%).")

st.divider()
st.caption("La inversión conlleva riesgos. Uso educativo. No constituye asesoramiento financiero.")

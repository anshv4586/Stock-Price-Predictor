import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import feedparser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockLens · Peer Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Design System ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --bg-void:      #080B11;
    --bg-base:      #0D1117;
    --bg-raised:    #111820;
    --bg-float:     #161D28;
    --border-dim:   rgba(255,255,255,0.055);
    --border-mid:   rgba(255,255,255,0.10);
    --border-hi:    rgba(255,255,255,0.18);

    --ink-primary:  #F0F4FF;
    --ink-secondary:#8B98B8;
    --ink-muted:    #4A556A;

    --accent-cyan:  #00D4FF;
    --accent-teal:  #00FFCC;
    --accent-rose:  #FF4D6D;
    --accent-amber: #FFB347;
    --accent-lime:  #A3FF5C;

    --glow-cyan:    0 0 20px rgba(0,212,255,0.25);
    --glow-teal:    0 0 20px rgba(0,255,204,0.20);
    --glow-rose:    0 0 20px rgba(255,77,109,0.25);

    --radius-sm:    8px;
    --radius-md:    14px;
    --radius-lg:    20px;
    --radius-xl:    28px;

    --font-display: 'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Global background ── */
.stApp {
    background: var(--bg-void);
    font-family: var(--font-display);
    color: var(--ink-primary);
}

/* ── Hide default chrome ── */
.stApp > header { display: none !important; }
#MainMenu, footer, .viewerBadge_container__1QSob { visibility: hidden; }

/* ── Main block padding ── */
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1600px !important;
}

/* ── Hero Header ── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 0.5rem;
}
.hero-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: var(--glow-cyan);
    flex-shrink: 0;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, var(--ink-primary) 0%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.hero-tagline {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--ink-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-mid), transparent);
    margin: 1.8rem 0;
}

/* ── Cards / Containers ── */
div[data-testid="stContainer"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.4rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}
div[data-testid="stContainer"]:hover {
    border-color: var(--border-mid) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-base) !important;
    border-right: 1px solid var(--border-dim) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem !important;
}
.sidebar-brand {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--ink-muted);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-size: 0.65rem;
    font-family: var(--font-mono);
    letter-spacing: 0.14em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-dim);
}

/* ── Sidebar inputs ── */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stDateInput input {
    background: var(--bg-float) !important;
    border-color: var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stMultiSelect > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Labels & captions ── */
label[data-testid="stWidgetLabel"] p,
.stSidebar label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--ink-secondary) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-float);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem !important;
    transition: all 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 16px rgba(0,212,255,0.10);
}
div[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--ink-muted) !important;
}
div[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--ink-primary) !important;
    letter-spacing: -0.03em !important;
}
div[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}

/* ── Section headings ── */
h2, h3 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: var(--ink-primary) !important;
}
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1rem !important; }

/* ── Dataframes ── */
.stDataFrame {
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}
.stDataFrame iframe { border-radius: var(--radius-md) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-float) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    border: 1px solid var(--border-dim) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-raised) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border-mid) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.2rem !important;
}

/* ── Alerts / status ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-width: 1px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}
div[data-baseweb="notification"] {
    background: rgba(0,255,204,0.06) !important;
    border-color: rgba(0,255,204,0.25) !important;
}

/* ── Buttons ── */
.stDownloadButton button, .stButton button {
    background: var(--bg-float) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover, .stButton button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    color: var(--accent-cyan) !important;
}

/* ── Toggle ── */
.stToggle span { font-family: var(--font-mono) !important; font-size: 0.75rem !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-float) !important;
    border-color: var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
details summary {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--ink-secondary) !important;
    letter-spacing: 0.06em !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ── Stat row chip ── */
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-float);
    border: 1px solid var(--border-dim);
    border-radius: 99px;
    padding: 4px 12px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--ink-secondary);
    white-space: nowrap;
}
.chip .dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.chip .dot-green  { background: var(--accent-teal); box-shadow: 0 0 6px var(--accent-teal); }
.chip .dot-red    { background: var(--accent-rose); box-shadow: 0 0 6px var(--accent-rose); }
.chip .dot-amber  { background: var(--accent-amber); box-shadow: 0 0 6px var(--accent-amber); }
.chip .val { color: var(--ink-primary); font-weight: 400; }

/* ── Peer mini chart label ── */
.peer-delta {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 4px 0 2px;
    color: var(--ink-muted);
}
.peer-delta .up   { color: var(--accent-teal); font-weight: 600; }
.peer-delta .down { color: var(--accent-rose);  font-weight: 600; }

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-dim);
}

/* ── Market snapshot card ── */
.snapshot-ticker {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--accent-cyan);
    line-height: 1;
}
.snapshot-price {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    color: var(--ink-primary);
}
.snapshot-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-muted);
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STOCKS = [
    "AAPL","ABBV","ACN","ADBE","ADP","AMD","AMGN","AMT","AMZN","APD",
    "AVGO","AXP","BA","BK","BKNG","BMY","BRK-B","BSX","C","CAT",
    "CI","CL","CMCSA","COST","CRM","CSCO","CVX","DE","DHR","DIS",
    "DUK","ELV","EOG","EQR","FDX","GD","GE","GILD","GOOG","GOOGL",
    "HD","HON","HUM","IBM","ICE","INTC","ISRG","JNJ","JPM","KO",
    "LIN","LLY","LMT","LOW","MA","MCD","MDLZ","META","MMC","MO",
    "MRK","MSFT","NEE","NFLX","NKE","NOW","NVDA","ORCL","PEP","PFE",
    "PG","PLD","PM","PSA","REGN","RTX","SBUX","SCHW","SLB","SO",
    "SPGI","T","TJX","TMO","TSLA","TXN","UNH","UNP","UPS","V","VZ",
    "WFC","WM","WMT","XOM"
]
DEFAULT_STOCKS = ["AAPL","MSFT","GOOGL","NVDA","AMZN","TSLA","META"]

# Refined color palette – thin lines need vivid, distinct hues
PALETTE = [
    "#00D4FF",  # cyan
    "#00FFCC",  # teal
    "#FF4D6D",  # rose
    "#FFB347",  # amber
    "#A3FF5C",  # lime
    "#BF7FFF",  # violet
    "#FF6B9D",  # pink
    "#4DFFB4",  # mint
    "#FFD700",  # gold
    "#6EC3FF",  # sky
]

PLOT_CFG = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8B98B8", size=11),
    margin=dict(l=8, r=8, t=44, b=8),
    xaxis=dict(showgrid=False, zeroline=False, showline=False,
               tickfont=dict(size=10), color="#4A556A"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
               zeroline=False, showline=False,
               tickfont=dict(size=10), color="#4A556A"),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#161D28",
        bordercolor="#2A3547",
        font=dict(family="DM Mono, monospace", size=11, color="#F0F4FF")
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
        font=dict(size=10),
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1
    )
)

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
def stocks_to_str(s): return ",".join(s)

if "tickers_input" not in st.session_state:
    raw = st.query_params.get("stocks", stocks_to_str(DEFAULT_STOCKS))
    if isinstance(raw, list): raw = raw[0] if raw else stocks_to_str(DEFAULT_STOCKS)
    st.session_state.tickers_input = raw.split(",")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ StockLens v2.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Search</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Filter directory…", label_visibility="collapsed")
    if search:
        hits = [s for s in STOCKS if search.upper() in s]
        if hits:
            st.caption(f"→ {', '.join(hits[:8])}" + ("…" if len(hits) > 8 else ""))

    st.markdown('<div class="sidebar-section">Portfolio</div>', unsafe_allow_html=True)
    selected_new = st.multiselect(
        "Tickers",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        default=st.session_state.tickers_input,
        placeholder="Add tickers…",
        key="ticker_multiselect",
        label_visibility="collapsed"
    )
    if selected_new != st.session_state.tickers_input:
        st.session_state.tickers_input = selected_new

    if st.session_state.tickers_input:
        st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)

    st.markdown('<div class="sidebar-section">Date Range</div>', unsafe_allow_html=True)
    start_date = st.date_input("From", value=pd.to_datetime("2024-01-01"), label_visibility="collapsed")
    end_date   = st.date_input("To",   label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)
    use_log_returns = st.toggle("Log Returns", False)
    dark_mode = st.toggle("Dark Mode", True)
    plot_template = "plotly_dark" if dark_mode else "plotly_white"

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#4A556A;line-height:1.6;">'
        'Data via Yahoo Finance<br>Prices auto-adjusted<br>Refreshed hourly'
        '</div>',
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
  <div class="hero-logo">◈</div>
  <div>
    <div class="hero-title">StockLens</div>
    <div class="hero-tagline">Peer Analysis · Real-time Intelligence</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(ttl="1h", show_spinner=False)
def load_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start, end=end,
                           group_by='ticker', auto_adjust=True,
                           threads=True, progress=False)
        df = pd.DataFrame()
        if len(tickers) == 1:
            t = tickers[0]
            if isinstance(data.columns, pd.MultiIndex):
                df[t] = data.xs('Close', level=1, axis=1).iloc[:, 0]
            else:
                df[t] = data['Close']
        else:
            for t in tickers:
                try:
                    if t in data.columns.levels[0]:
                        df[t] = data[t]['Close']
                except: continue
        return df.dropna()
    except:
        return None

if not st.session_state.tickers_input:
    st.warning("Select at least one ticker from the sidebar.")
    st.stop()

with st.spinner(""):
    sorted_tickers = sorted(st.session_state.tickers_input)
    raw_df = load_data(sorted_tickers, start_date, end_date)

if raw_df is None or raw_df.empty:
    st.error("Unable to fetch data. Check your connection or ticker symbols.")
    st.stop()

# ── Derived data ──
if use_log_returns:
    norm_df = np.log(raw_df / raw_df.iloc[0])
    y_label  = "Log Return"
    val_fmt  = ".3f"
else:
    norm_df = raw_df / raw_df.iloc[0]
    y_label  = "Norm. Price"
    val_fmt  = ".3f"

final_vals   = norm_df.iloc[-1].sort_values(ascending=False)
best_ticker  = final_vals.index[0]
worst_ticker = final_vals.index[-1]
best_val     = final_vals.iloc[0]
worst_val    = final_vals.iloc[-1]

returns      = raw_df.pct_change().dropna()
volatility   = returns.std().mean() * np.sqrt(252)
sharpe_ratio = (returns.mean() / returns.std()).mean() * np.sqrt(252)
avg_perf     = (final_vals.mean() - 1) * 100
latest_price = raw_df[best_ticker].iloc[-1]

best_pct  = f"{(best_val  - 1)*100:+.1f}%"
worst_pct = f"{(worst_val - 1)*100:+.1f}%"

# -----------------------------------------------------------------------------
# KPI Row (full width, compact chips + metric grid)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">Market Snapshot</div>', unsafe_allow_html=True)

col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
with col_a:
    st.metric("Avg Return",    f"{avg_perf:.2f}%")
with col_b:
    st.metric("Ann. Volatility", f"{volatility:.3f}")
with col_c:
    st.metric("Sharpe Ratio",  f"{sharpe_ratio:.2f}")
with col_d:
    st.metric("Top Performer", best_ticker,  delta=best_pct)
with col_e:
    st.metric("Laggard",       worst_ticker, delta=worst_pct,  delta_color="inverse")
with col_f:
    st.metric(f"{best_ticker} Price",  f"${latest_price:,.2f}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Chart — full width, thin crisp lines
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">Price Comparison</div>', unsafe_allow_html=True)

with st.container(border=True):
    fig_main = go.Figure()

    for i, col in enumerate(norm_df.columns):
        c = PALETTE[i % len(PALETTE)]
        r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        is_best  = col == best_ticker

        fig_main.add_trace(go.Scatter(
            x=norm_df.index,
            y=norm_df[col],
            mode='lines',
            name=col,
            line=dict(
                width=1.2 if not is_best else 1.8,
                color=c,
                shape='spline'
            ),
            opacity=1.0 if is_best else 0.72,
            hovertemplate=f"<b>{col}</b>  %{{y:{val_fmt}}}<extra></extra>"
        ))

    fig_main.update_layout(
        **PLOT_CFG,
        height=400,
        title=dict(
            text=f"Normalized Performance · {start_date} → {end_date}",
            font=dict(size=12, color="#4A556A", family="DM Mono, monospace"),
            x=0
        ),
        yaxis_title=y_label,
        transition_duration=400
    )
    st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Ranking + Bar Chart
# -----------------------------------------------------------------------------
# ------------------------------------------------------------------
# Ranking + Bar Chart
# ------------------------------------------------------------------
st.markdown('<div class="section-label">Rankings & Distribution</div>', unsafe_allow_html=True)

col_rank, col_bar = st.columns([1, 1.8], gap="medium")

with col_rank:
    with st.container(border=True):
        st.markdown("**Leaderboard**")
        styled = final_vals.rename("Perf.").to_frame()
        st.dataframe(
            styled.style
                .background_gradient(cmap="Blues", vmin=styled["Perf."].min())
                .format("{:.3f}"),
            use_container_width=True,
            height=340
        )

with col_bar:
    with st.container(border=True):
        st.markdown("**Return Distribution**")
        sv = norm_df.iloc[-1].sort_values(ascending=False).reset_index()
        sv.columns = ["Stock", "Perf"]
        sv["color"] = [PALETTE[i % len(PALETTE)] for i in range(len(sv))]

        fig_bar = go.Figure()
        for _, row in sv.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["Stock"]],
                y=[row["Perf"]],
                marker_color=row["color"],
                marker_line_width=0,
                opacity=0.85,
                showlegend=False,
                hovertemplate=f"<b>{row['Stock']}</b> {row['Perf']:{val_fmt}}<extra></extra>"
            ))

        # ✅ FIXED PART (IMPORTANT)
        fig_bar.update_layout(
            **PLOT_CFG,
            height=340,
            bargap=0.25,
            title=dict(
                text="Performance vs Peers",
                font=dict(size=12, color="#4A556A", family="DM Mono"),
                x=0
            ),
            yaxis_title=y_label   # ✅ Correct key
        )

        # ✅ Handle x-axis separately (NO DUPLICATE ERROR)
        fig_bar.update_xaxes(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10),
            color="#4A556A"
        )

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# AI Insights — compact row
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">AI Insights</div>', unsafe_allow_html=True)

col_ins, col_sig = st.columns([1.2, 1], gap="medium")

with col_ins:
    with st.container(border=True):
        st.markdown("**Signal Summary**")

        momentum = best_val > 1
        sharpe_ok = sharpe_ratio > 1
        top3 = final_vals.head(3)

        if momentum:
            st.success(f"📈 Upward momentum detected  ·  Leader: **{best_ticker}** ({best_pct})")
        else:
            st.error(f"📉 Market under pressure  ·  Leader: **{best_ticker}** ({best_pct})")

        if sharpe_ok:
            st.success(f"🛡 Healthy risk-adjusted returns  ·  Sharpe {sharpe_ratio:.2f}")
        else:
            st.warning(f"⚠️ Elevated volatility profile  ·  Sharpe {sharpe_ratio:.2f}")

        st.markdown("**Top 3 Picks**")
        for t in top3.index:
            pct = (top3[t]-1)*100
            st.markdown(
                f'<span class="chip"><span class="dot dot-green"></span>'
                f'{t} <span class="val">{pct:+.1f}%</span></span> ',
                unsafe_allow_html=True
            )

        # Fundamentals
        try:
            info = yf.Ticker(best_ticker).info
            pe   = info.get('trailingPE', None)
            mc   = info.get('marketCap', None)
            pe_s = f"{pe:.1f}×" if pe else "N/A"
            mc_s = f"${mc/1e9:.1f}B" if mc else "N/A"
            st.markdown(f"""
            <br><div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#8B98B8;">
            <b style="color:#F0F4FF">{best_ticker}</b> &nbsp;·&nbsp; P/E {pe_s} &nbsp;·&nbsp; Mkt Cap {mc_s}
            </div>""", unsafe_allow_html=True)
        except: pass

with col_sig:
    with st.container(border=True):
        st.markdown("**Trading Signals**")

        signals = []
        for ticker in raw_df.columns:
            price  = raw_df[ticker].iloc[-1]
            sma_20 = raw_df[ticker].rolling(20).mean().iloc[-1]
            if pd.isna(sma_20):
                sig = "WAIT"
            elif price > sma_20:
                sig = "BUY"
            else:
                sig = "SELL"
            signals.append({"Stock": ticker, "Price": round(price,2),
                             "SMA(20)": round(sma_20,2) if not pd.isna(sma_20) else None,
                             "Signal": sig})

        sig_df = pd.DataFrame(signals)
        filt   = st.selectbox("Filter", ["All","BUY","SELL"], label_visibility="collapsed")
        if filt != "All":
            sig_df = sig_df[sig_df["Signal"] == filt]

        def color_sig(val):
            if val == "BUY":  return "color:#00FFCC;font-weight:600"
            if val == "SELL": return "color:#FF4D6D;font-weight:600"
            return "color:#4A556A"

        sty = sig_df.style.applymap(color_sig, subset=["Signal"]).format({"Price": "${:.2f}"})
        st.dataframe(sty, use_container_width=True, height=260)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Peer Performance Grid
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">Peer Performance Grid</div>', unsafe_allow_html=True)

if len(st.session_state.tickers_input) < 2:
    st.info("Select 2 or more tickers to see peer comparison.")
else:
    cols = st.columns(4)
    for i, ticker in enumerate(sorted_tickers):
        if ticker not in norm_df.columns: continue
        peers    = [t for t in norm_df.columns if t != ticker]
        if not peers: continue

        peer_avg = norm_df[peers].mean(axis=1)
        stock_ln = norm_df[ticker]
        delta    = stock_ln - peer_avg
        is_out   = delta.iloc[-1] >= 0
        lc       = "#00FFCC" if is_out else "#FF4D6D"
        final_d  = delta.iloc[-1]

        with cols[i % 4]:
            with st.container(border=True):
                fig_m = go.Figure()
                # Peer avg — very faint dotted
                fig_m.add_trace(go.Scatter(
                    x=norm_df.index, y=peer_avg,
                    mode='lines', name='Peer',
                    line=dict(color='rgba(255,255,255,0.12)', dash='dot', width=0.8),
                    showlegend=False, hoverinfo='skip'
                ))
                # Stock — thin crisp
                fig_m.add_trace(go.Scatter(
                    x=norm_df.index, y=stock_ln,
                    mode='lines', name=ticker,
                    line=dict(color=lc, width=1.2, shape='spline'),
                    showlegend=False,
                    hovertemplate=f'%{{y:{val_fmt}}}<extra>{ticker}</extra>'
                ))
                fig_m.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=28,b=0),
                    height=160,
                    title=dict(text=ticker,
                               font=dict(family="Syne",size=13,color="#F0F4FF"),
                               x=0.5, xanchor='center'),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="#161D28",bordercolor="#2A3547",
                                    font=dict(family="DM Mono",size=10))
                )
                st.plotly_chart(fig_m, use_container_width=True,
                                config={"displayModeBar": False})

                if use_log_returns:
                    d_str = f"{final_d:+.3f}"
                else:
                    d_str = f"{final_d*100:+.1f}%"

                arrow = "▲" if is_out else "▼"
                cls   = "up"  if is_out else "down"
                st.markdown(
                    f'<div class="peer-delta">vs peers '
                    f'<span class="{cls}">{arrow} {d_str}</span></div>',
                    unsafe_allow_html=True
                )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Raw Data Expander
# -----------------------------------------------------------------------------
with st.expander("📂 Raw Data & Correlation Matrix"):
    t1, t2 = st.tabs(["Price History", "Correlation"])

    with t1:
        st.dataframe(raw_df.style.highlight_max(axis=0), use_container_width=True)
        st.download_button(
            "⬇ Download CSV",
            raw_df.to_csv().encode(),
            "stocklens_data.csv",
            "text/csv"
        )

    with t2:
        corr    = raw_df.pct_change().corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto"
        )
        fig_corr.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono,monospace", size=10, color="#8B98B8"),
            margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_colorbar=dict(tickfont=dict(size=9))
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

        high_c = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
        pairs  = high_c.stack().sort_values(ascending=False).head(5)
        st.caption("Top correlations: " + "  ·  ".join(
            f"{a}/{b} {v:.2f}" for (a,b),v in pairs.items()
        ))

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Advanced Analysis Tabs
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">Advanced Analytics</div>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "Efficient Frontier","Price Prediction","Risk Metrics",
    "Moving Averages","Sentiment","Technicals","LSTM"
])

# Helper – apply base layout
def base_layout(fig, h=400, title=""):
    cfg = {**PLOT_CFG, "height": h}
    if title:
        cfg["title"] = dict(text=title, font=dict(size=12, color="#4A556A",
                            family="DM Mono,monospace"), x=0)
    fig.update_layout(**cfg)
    return fig

# ── Tab 1: Portfolio Optimization ──────────────────────────────────────────
with tab1:
    st.markdown("**Efficient Frontier** — maximize return per unit of risk")
    if len(st.session_state.tickers_input) >= 2:
        rets  = raw_df.pct_change().dropna()
        mu    = rets.mean()
        sigma = rets.cov()
        N     = 5000
        res   = np.zeros((3, N))
        wts   = []
        for i in range(N):
            w = np.random.random(len(mu)); w /= w.sum(); wts.append(w)
            r = np.sum(w*mu)*252
            s = np.sqrt(w @ sigma.values @ w)*np.sqrt(252)
            res[:,i] = r, s, r/s

        fig_ef = go.Figure(go.Scatter(
            x=res[1], y=res[0],
            mode='markers',
            marker=dict(color=res[2], colorscale="Viridis",
                        size=3, opacity=0.65,
                        colorbar=dict(title="Sharpe", thickness=8,
                                      tickfont=dict(size=9))),
            hovertemplate="Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>"
        ))
        mx = np.argmax(res[2])
        fig_ef.add_trace(go.Scatter(
            x=[res[1,mx]], y=[res[0,mx]],
            mode='markers+text',
            marker=dict(color="#00D4FF", size=10, symbol="star",
                        line=dict(color="#fff",width=1)),
            text=["  Optimal"], textfont=dict(size=10, color="#00D4FF"),
            showlegend=False,
            hovertemplate=f"Return: {res[0,mx]:.3f}<br>Risk: {res[1,mx]:.3f}<br>Sharpe: {res[2,mx]:.3f}<extra>Optimal</extra>"
        ))
        base_layout(fig_ef, 400, "Efficient Frontier")
        fig_ef.update_layout(xaxis_title="Annualised Risk", yaxis_title="Annualised Return")
        st.plotly_chart(fig_ef, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
             background:var(--bg-float);border:1px solid var(--border-dim);
             border-radius:10px;padding:0.9rem 1.2rem;color:#8B98B8;display:inline-block;">
        ⭐ Optimal &nbsp;|&nbsp; Return <b style="color:#00FFCC">{res[0,mx]:.3f}</b>
        &nbsp;·&nbsp; Risk <b style="color:#FF4D6D">{res[1,mx]:.3f}</b>
        &nbsp;·&nbsp; Sharpe <b style="color:#00D4FF">{res[2,mx]:.3f}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Optimal Allocation**")
        alloc = pd.DataFrame({"Stock": raw_df.columns,
                               "Weight %": (wts[mx]*100).round(2)}
                             ).sort_values("Weight %", ascending=False)
        st.dataframe(alloc, use_container_width=True, height=220)
    else:
        st.info("Select 2+ tickers.")

# ── Tab 2: Price Prediction ─────────────────────────────────────────────────
with tab2:
    sel = st.selectbox("Stock", raw_df.columns, key="pred_sel")
    data_p = raw_df[[sel]].dropna()
    data_p['d'] = np.arange(len(data_p))
    mdl = LinearRegression().fit(data_p[['d']], data_p[sel])
    fd  = np.arange(len(data_p), len(data_p)+30).reshape(-1,1)
    pr  = mdl.predict(fd)
    std = np.std(pr)
    fut = pd.date_range(start=raw_df.index[-1]+pd.Timedelta(days=1), periods=30)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=raw_df.index, y=raw_df[sel], name="Historical",
        line=dict(color="#00D4FF", width=1.2, shape='spline')))
    fig_p.add_trace(go.Scatter(
        x=fut, y=(pr+std).flatten(), name="Upper",
        line=dict(color="rgba(255,179,71,0.4)", dash='dot', width=0.8),
        showlegend=False))
    fig_p.add_trace(go.Scatter(
        x=fut, y=(pr-std).flatten(), name="Confidence Band",
        line=dict(color="rgba(255,179,71,0.4)", dash='dot', width=0.8),
        fill='tonexty', fillcolor='rgba(255,179,71,0.06)'))
    fig_p.add_trace(go.Scatter(
        x=fut, y=pr.flatten(), name="Forecast",
        line=dict(color="#FFB347", dash='dash', width=1.4)))

    base_layout(fig_p, 420, f"{sel} — Linear Regression Forecast (30d)")
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

# ── Tab 3: Risk Metrics ─────────────────────────────────────────────────────
with tab3:
    vol_s   = returns.std() * np.sqrt(252)
    sharpe_s = (returns.mean() / returns.std()) * np.sqrt(252)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Annualised Volatility**")
        st.dataframe(vol_s.rename("Volatility").to_frame()
                     .style.background_gradient(cmap="Reds").format("{:.4f}"),
                     use_container_width=True)
    with c2:
        st.markdown("**Sharpe Ratio**")
        st.dataframe(sharpe_s.rename("Sharpe").to_frame()
                     .style.background_gradient(cmap="Greens").format("{:.3f}"),
                     use_container_width=True)

    drawdown = (raw_df / raw_df.cummax()) - 1
    fig_dd = go.Figure()
    for i, col in enumerate(drawdown.columns):
        c = PALETTE[i % len(PALETTE)]
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown[col], name=col,
            line=dict(color=c, width=1.0), fill='tozeroy',
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.04)"
        ))
    base_layout(fig_dd, 340, "Drawdown Over Time")
    fig_dd.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar":False})

    st.markdown("**Momentum Backtest**")
    bt = st.selectbox("Stock", raw_df.columns, key="bt")
    bt_r = raw_df[bt].pct_change(); bt_r[bt_r < 0] = 0
    cum  = (1 + bt_r).cumprod()
    fig_bt = go.Figure(go.Scatter(x=cum.index, y=cum,
                                   line=dict(color=PALETTE[0], width=1.2)))
    base_layout(fig_bt, 300, f"Momentum Backtest · {bt}")
    st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar":False})

# ── Tab 4: Moving Averages ───────────────────────────────────────────────────
with tab4:
    ma_tk = st.selectbox("Stock", raw_df.columns, key="ma_tk")
    sma20 = raw_df[ma_tk].rolling(20).mean()
    ema20 = raw_df[ma_tk].ewm(span=20).mean()
    sma50 = raw_df[ma_tk].rolling(50).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=raw_df[ma_tk],
        name="Price", line=dict(color="#00D4FF",width=1.2,shape='spline')))
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=sma20,
        name="SMA 20", line=dict(color="#FFB347",dash='dash',width=0.9)))
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=ema20,
        name="EMA 20", line=dict(color="#00FFCC",dash='dot',width=0.9)))
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=sma50,
        name="SMA 50", line=dict(color="#BF7FFF",dash='dashdot',width=0.9)))
    base_layout(fig_ma, 400, f"{ma_tk} · Moving Averages")
    st.plotly_chart(fig_ma, use_container_width=True, config={"displayModeBar":False})

# ── Tab 5: Sentiment ────────────────────────────────────────────────────────
with tab5:
    sent_tk = st.selectbox("Stock", raw_df.columns, key="sent_tk")
    feed    = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={sent_tk}")

    if feed.entries:
        scores = []
        for e in feed.entries[:5]:
            sc = TextBlob(e.title).sentiment.polarity
            scores.append(sc)
            if sc > 0.1:   st.success(f"🟢 {e.title}")
            elif sc < -0.1: st.error(f"🔴 {e.title}")
            else:            st.info(f"⚪ {e.title}")

        avg_s = np.mean(scores)
        st.markdown(f"**Overall:** {'Positive 📈' if avg_s>0.1 else 'Negative 📉' if avg_s<-0.1 else 'Neutral 😐'}")
        fig_s = go.Figure(go.Bar(
            x=[f"#{i+1}" for i in range(len(scores))], y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0,"#FF4D6D"],[0.5,"#4A556A"],[1,"#00FFCC"]],
                line_width=0
            )
        ))
        base_layout(fig_s, 280, f"Sentiment · {sent_tk}")
        fig_s.update_layout(yaxis_title="Polarity", showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
    else:
        st.error("No RSS entries found.")

# ── Tab 6: Technical Indicators ─────────────────────────────────────────────
with tab6:
    tech_tk = st.selectbox("Stock", raw_df.columns, key="tech_tk")

    def calc_rsi(s, p=14):
        d = s.diff(); g = d.where(d>0,0).rolling(p).mean()
        l = (-d.where(d<0,0)).rolling(p).mean(); rs = g/l
        return 100 - (100/(1+rs))

    def calc_macd(s):
        e12 = s.ewm(span=12).mean(); e26 = s.ewm(span=26).mean()
        m = e12-e26; sig = m.ewm(span=9).mean(); return m, sig

    rsi_s        = calc_rsi(raw_df[tech_tk])
    macd_s, sig_s = calc_macd(raw_df[tech_tk])

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=raw_df.index, y=rsi_s,
        line=dict(color="#00D4FF",width=1.2), name="RSI",
        fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'))
    fig_rsi.add_hline(y=70, line=dict(dash='dash',color='rgba(255,77,109,0.5)',width=0.8))
    fig_rsi.add_hline(y=30, line=dict(dash='dash',color='rgba(0,255,204,0.5)',width=0.8))
    base_layout(fig_rsi, 260, f"{tech_tk} · RSI (14)")
    st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar":False})

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=raw_df.index, y=macd_s,
        line=dict(color="#00D4FF",width=1.2), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=raw_df.index, y=sig_s,
        line=dict(color="#FFB347",dash='dash',width=0.9), name="Signal"))
    hist = macd_s - sig_s
    fig_macd.add_trace(go.Bar(x=raw_df.index, y=hist, name="Histogram",
        marker=dict(color=hist, colorscale=[[0,"#FF4D6D"],[1,"#00FFCC"]],
                    line_width=0), opacity=0.5, showlegend=False))
    base_layout(fig_macd, 280, f"{tech_tk} · MACD")
    st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar":False})

    p_now = raw_df[tech_tk].iloc[-1]
    sma_  = raw_df[tech_tk].rolling(20).mean().iloc[-1]
    rsi_  = rsi_s.iloc[-1]

    if p_now > sma_ and rsi_ < 70:
        st.success(f"🟢 STRONG BUY · {tech_tk}  —  Price above SMA20, RSI healthy at {rsi_:.1f}")
    elif p_now < sma_ and rsi_ > 30:
        st.error(f"🔴 STRONG SELL · {tech_tk}  —  Price below SMA20, RSI at {rsi_:.1f}")
    else:
        st.info(f"⚪ HOLD · {tech_tk}  —  Mixed signals, RSI {rsi_:.1f}")

# ── Tab 7: LSTM ──────────────────────────────────────────────────────────────
with tab7:
    st.markdown("**LSTM Deep Learning Forecast** — 30-day ahead prediction")
    lstm_tk = st.selectbox("Stock", raw_df.columns, key="lstm_tk")

    def lstm_predict(df, ticker, days=30):
        data  = df[[ticker]].dropna()
        sc    = MinMaxScaler(); sd = sc.fit_transform(data)
        X, y  = [], []
        for i in range(60, len(sd)):
            X.append(sd[i-60:i]); y.append(sd[i])
        X, y = np.array(X), np.array(y)
        mdl = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
            LSTM(50), Dense(1)
        ])
        mdl.compile(optimizer='adam', loss='mean_squared_error')
        mdl.fit(X, y, epochs=5, batch_size=16, verbose=0)
        lw   = sd[-60:]
        preds = []
        for _ in range(days):
            p = mdl.predict(lw.reshape(1,60,1), verbose=0)
            preds.append(p[0][0]); lw = np.append(lw[1:], p, axis=0)
        return sc.inverse_transform(np.array(preds).reshape(-1,1))

    with st.spinner("Training LSTM model…"):
        preds = lstm_predict(raw_df, lstm_tk)

    fut = pd.date_range(start=raw_df.index[-1]+pd.Timedelta(days=1), periods=30)
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(
        x=raw_df.index, y=raw_df[lstm_tk], name="Historical",
        line=dict(color="#00D4FF", width=1.2, shape='spline')))
    fig_l.add_trace(go.Scatter(
        x=fut, y=preds.flatten(), name="LSTM Forecast",
        line=dict(color="#A3FF5C", dash='dash', width=1.4)))
    base_layout(fig_l, 420, f"{lstm_tk} · LSTM 30-Day Forecast")
    st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar":False})

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#4A556A;
     text-align:center;padding:0.5rem 0 1rem;">
StockLens · Data via Yahoo Finance · Not financial advice
</div>
""", unsafe_allow_html=True)
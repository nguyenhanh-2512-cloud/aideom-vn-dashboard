
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


st.set_page_config(
    page_title="VN AIDEOM-VN | Decision Optimization Dashboard",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
PLOT_TEMPLATE = "plotly_dark"


# =========================
# Global style
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(244,63,94,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(20,184,166,0.12), transparent 30%),
            linear-gradient(135deg, #08111f 0%, #0b1020 48%, #111827 100%);
        color: #E5E7EB;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    h1, h2, h3 { letter-spacing: -0.03em; }
    .hero {
        padding: 30px 32px;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.70));
        box-shadow: 0 20px 60px rgba(0,0,0,.24);
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 46px;
        line-height: 1.02;
        font-weight: 850;
        color: #F8FAFC;
        margin-bottom: 10px;
    }
    .hero-sub {
        color: #CBD5E1;
        font-size: 18px;
        max-width: 980px;
    }
    .section-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 28px rgba(0,0,0,.16);
        margin-bottom: 16px;
    }
    .kpi-card {
        padding: 16px 16px;
        border-radius: 18px;
        background: linear-gradient(160deg, rgba(30,41,59,0.82), rgba(15,23,42,0.86));
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 116px;
    }
    .kpi-label { color: #94A3B8; font-size: 13px; margin-bottom: 6px; }
    .kpi-value { color: #FB7185; font-size: 29px; font-weight: 850; letter-spacing: -0.04em; }
    .kpi-note { color: #10B981; font-size: 12px; margin-top: 4px; }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(16,185,129,0.16);
        color: #A7F3D0;
        border: 1px solid rgba(16,185,129,0.28);
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
    }
    .muted { color: #94A3B8; }
    .warning-box {
        padding: 14px 16px;
        border-left: 4px solid #F59E0B;
        background: rgba(245,158,11,0.09);
        border-radius: 14px;
        margin: 12px 0;
    }
    .success-box {
        padding: 14px 16px;
        border-left: 4px solid #10B981;
        background: rgba(16,185,129,0.09);
        border-radius: 14px;
        margin: 12px 0;
    }
    div[data-testid="stMetricValue"] { color: #FB7185; }
    div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Data and helpers
# =========================
@st.cache_data
def load_macro():
    return pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv", encoding="utf-8-sig")


@st.cache_data
def load_regions():
    return pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv", encoding="utf-8-sig")


@st.cache_data
def load_sectors():
    return pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv", encoding="utf-8-sig")


def hero(title, subtitle, tags=None):
    tags = tags or []
    tag_html = "".join([f"<span class='pill'>{t}</span>" for t in tags])
    st.markdown(
        f"""
        <div class="hero">
            <div>{tag_html}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def section(title, body=None):
    st.markdown(f"### {title}")
    if body:
        st.markdown(f"<div class='muted'>{body}</div>", unsafe_allow_html=True)


def safe_mape(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), 1e-9))) * 100


def minmax(s):
    s = pd.Series(s, dtype=float)
    if abs(s.max() - s.min()) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def reverse_minmax(s):
    s = pd.Series(s, dtype=float)
    if abs(s.max() - s.min()) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s.max() - s) / (s.max() - s.min())


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x - np.min(x)
    if np.mean(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x)) / (n * np.sum(x)) - (n + 1) / n


def plot_bar(df, x, y, title, color=None, text=None):
    fig = px.bar(df, x=x, y=y, color=color, text=text, template=PLOT_TEMPLATE, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=54, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def plot_line(df, x, y, title, color=None):
    fig = px.line(df, x=x, y=y, markers=True, color=color, template=PLOT_TEMPLATE, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=54, b=10))
    return fig


def cobb_douglas_arrays():
    macro = load_macro()
    Y = macro["GDP_trillion_VND"].values.astype(float)
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
    D = macro["digital_economy_share_GDP_pct"].values.astype(float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)
    return macro["year"].values, Y, K, L, D, AI, H


def compute_tfp(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07):
    years, Y, K, L, D, AI, H = cobb_douglas_arrays()
    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    return years, Y, K, L, D, AI, H, A


def region_beta_matrix():
    regions = [
        "Trung du miền núi phía Bắc",
        "Đồng bằng sông Hồng",
        "Bắc Trung Bộ + DH Trung Bộ",
        "Tây Nguyên",
        "Đông Nam Bộ",
        "Đồng bằng sông Cửu Long",
    ]
    items = ["I - Hạ tầng số", "D - CĐS DN", "AI", "H - Nhân lực số"]
    beta = np.array(
        [
            [1.15, 0.85, 0.55, 1.30],
            [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15],
            [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00],
            [1.10, 0.85, 0.65, 1.25],
        ],
        dtype=float,
    )
    D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)
    return regions, items, beta, D0


def topsis_score(df, criteria, weights, is_benefit):
    X = df[criteria].values.astype(float)
    denom = np.sqrt((X**2).sum(axis=0))
    R = X / np.where(denom == 0, 1, denom)
    V = R * np.asarray(weights)
    is_benefit = np.asarray(is_benefit)
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    return S_neg / np.maximum(S_star + S_neg, 1e-12)


def entropy_weights_positive(X):
    X = np.asarray(X, dtype=float)
    X = X - X.min(axis=0) + 1e-9
    P = X / np.maximum(X.sum(axis=0), 1e-12)
    k = 1.0 / np.log(len(X))
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / np.maximum(d.sum(), 1e-12)


def scenario_shares():
    return {
        "S1 - Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
        "S2 - Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
        "S3 - AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
        "S4 - Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
        "S5 - Tối ưu cân bằng": np.array([0.34, 0.26, 0.18, 0.22]),
    }


def simulate_dynamic(shares, start=2026, end=2035, invest_rate=0.22, shock_2028=0.0):
    years, Y0s, K0s, L0s, D0s, AI0s, H0s, A0s = compute_tfp()
    K = K0s[-1] * 1.06
    L = L0s[-1] * 1.01
    D = D0s[-1] + 0.8
    AI = AI0s[-1] + 6
    H = H0s[-1] + 0.8
    A = A0s[-1] * 1.012
    rows = []
    for year in range(start, end + 1):
        Y = A * (K**0.33) * (L**0.42) * (D**0.10) * (AI**0.08) * (H**0.07)
        if year == 2028:
            Y *= 1 - shock_2028
        invest = Y * invest_rate
        C = Y - invest
        rows.append([year, Y, C, K, D, AI, H, invest])
        K = (1 - 0.05) * K + shares[0] * invest
        D = max(1, (1 - 0.12) * D + shares[1] * invest / 240)
        AI = max(1, (1 - 0.15) * AI + shares[2] * invest / 135)
        H = max(1, H + 0.8 * shares[3] * invest / 520 - 0.02 * H)
        L = L * 1.006
        A = A * (1 + 0.00008 * D + 0.00004 * AI + 0.00006 * H)
    return pd.DataFrame(rows, columns=["Năm", "Y_GDP", "C_tiêu_dùng", "K", "D", "AI", "H", "Đầu_tư"])


# =========================
# Pages
# =========================
def page_home():
    # -----------------------------------------------------
    # CSS riêng cho trang chủ
    # -----------------------------------------------------
    st.html(
        """
        <style>
        /* ---------- Khung tổng ---------- */
        .aideom-home {
            width: 100%;
            margin: 0 auto;
            padding: 0.2rem 0 2rem 0;
        }

        .aideom-home * {
            box-sizing: border-box;
        }

        /* ---------- Hero ---------- */
        .aideom-hero {
            position: relative;
            overflow: hidden;
            padding: 2.15rem 2.2rem 2rem 2.2rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            background:
                radial-gradient(
                    circle at 88% 12%,
                    rgba(14, 165, 233, 0.16),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 12% 88%,
                    rgba(34, 197, 94, 0.10),
                    transparent 28%
                ),
                linear-gradient(
                    145deg,
                    rgba(15, 23, 42, 0.98),
                    rgba(17, 24, 39, 0.96)
                );
            box-shadow: 0 22px 60px rgba(2, 6, 23, 0.28);
            margin-bottom: 1.35rem;
        }

        .aideom-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(
                    rgba(148, 163, 184, 0.035) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(148, 163, 184, 0.035) 1px,
                    transparent 1px
                );
            background-size: 28px 28px;
            pointer-events: none;
        }

        .aideom-hero-content {
            position: relative;
            z-index: 2;
        }

        .aideom-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.38rem 0.72rem;
            margin-bottom: 1rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 197, 94, 0.34);
            background: rgba(6, 78, 59, 0.28);
            color: #86efac;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .aideom-eyebrow-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.10);
        }

        .aideom-title {
            margin: 0;
            font-size: clamp(2.15rem, 4vw, 3.5rem);
            line-height: 1.02;
            letter-spacing: -0.045em;
            font-weight: 900;
            color: #f8fafc;
        }

        .aideom-title-accent {
            background: linear-gradient(
                90deg,
                #f8fafc 0%,
                #93c5fd 48%,
                #5eead4 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .aideom-subtitle {
            margin: 0.85rem 0 0 0;
            color: #cbd5e1;
            font-size: clamp(1rem, 1.5vw, 1.24rem);
            line-height: 1.65;
            font-weight: 650;
            font-style: italic;
        }

        .aideom-description {
            max-width: 980px;
            margin: 0.75rem 0 0 0;
            color: #94a3b8;
            font-size: 0.97rem;
            line-height: 1.72;
        }

        .aideom-hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.25rem;
        }

        .aideom-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.42rem 0.72rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.58);
            color: #dbeafe;
            font-size: 0.79rem;
            font-weight: 650;
        }

        /* ---------- KPI ---------- */
        .aideom-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.7rem 0;
        }

        .aideom-kpi-card {
            min-height: 142px;
            padding: 1.05rem 1rem 0.95rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            background:
                linear-gradient(
                    180deg,
                    rgba(30, 41, 59, 0.88),
                    rgba(15, 23, 42, 0.88)
                );
            box-shadow: 0 12px 30px rgba(2, 6, 23, 0.18);
            transition:
                transform 160ms ease,
                border-color 160ms ease,
                box-shadow 160ms ease;
        }

        .aideom-kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.38);
            box-shadow: 0 18px 34px rgba(2, 6, 23, 0.28);
        }

        .aideom-kpi-label {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 650;
            margin-bottom: 0.6rem;
        }

        .aideom-kpi-value {
            color: #fb7185;
            font-size: clamp(1.45rem, 2.1vw, 2.15rem);
            line-height: 1.1;
            font-weight: 900;
            letter-spacing: -0.03em;
        }

        .aideom-kpi-note {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            margin-top: 0.72rem;
            padding: 0.28rem 0.48rem;
            border-radius: 999px;
            background: rgba(5, 150, 105, 0.15);
            color: #6ee7b7;
            font-size: 0.75rem;
            font-weight: 700;
        }

        /* ---------- Section ---------- */
        .aideom-section-head {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.8rem 0 0.95rem 0;
        }

        .aideom-section-title {
            margin: 0;
            color: #f8fafc;
            font-size: clamp(1.35rem, 2vw, 1.85rem);
            line-height: 1.2;
            font-weight: 850;
            letter-spacing: -0.025em;
        }

        .aideom-section-subtitle {
            margin: 0.28rem 0 0 0;
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .aideom-section-badge {
            flex: 0 0 auto;
            padding: 0.38rem 0.65rem;
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 999px;
            background: rgba(14, 116, 144, 0.12);
            color: #7dd3fc;
            font-size: 0.76rem;
            font-weight: 700;
        }

        /* ---------- Level cards ---------- */
        .aideom-levels {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .aideom-level-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.19);
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.72);
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.14);
        }

        .aideom-level-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }

        .aideom-level-name {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 800;
        }

        .aideom-level-dot {
            width: 11px;
            height: 11px;
            border-radius: 999px;
            box-shadow: 0 0 0 5px rgba(255, 255, 255, 0.035);
        }

        .level-green { background: #22c55e; }
        .level-yellow { background: #facc15; }
        .level-orange { background: #fb923c; }
        .level-purple { background: #a78bfa; }

        .aideom-level-count {
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .aideom-task-row {
            display: grid;
            grid-template-columns: 58px minmax(0, 1fr);
            gap: 0.75rem;
            align-items: start;
            padding: 0.72rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.11);
        }

        .aideom-task-row:first-of-type {
            border-top: none;
        }

        .aideom-task-code {
            color: #e2e8f0;
            font-weight: 850;
            font-size: 0.84rem;
        }

        .aideom-task-name {
            color: #cbd5e1;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .aideom-task-tech {
            display: block;
            margin-top: 0.22rem;
            color: #64748b;
            font-size: 0.73rem;
            line-height: 1.4;
        }

        /* ---------- Features ---------- */
        .aideom-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
        }

        .aideom-feature-card {
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 17px;
            background:
                linear-gradient(
                    160deg,
                    rgba(30, 41, 59, 0.68),
                    rgba(15, 23, 42, 0.72)
                );
        }

        .aideom-feature-icon {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            margin-bottom: 0.75rem;
            border-radius: 12px;
            background: rgba(14, 165, 233, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.22);
            font-size: 1.05rem;
        }

        .aideom-feature-title {
            color: #f8fafc;
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .aideom-feature-text {
            color: #94a3b8;
            font-size: 0.80rem;
            line-height: 1.55;
        }

        /* ---------- Data strip ---------- */
        .aideom-data-strip {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }

        .aideom-data-card {
            padding: 1rem;
            border-radius: 17px;
            border: 1px solid rgba(148, 163, 184, 0.17);
            background: rgba(15, 23, 42, 0.64);
        }

        .aideom-data-title {
            color: #e2e8f0;
            font-size: 0.85rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .aideom-data-text {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.62;
        }

        .aideom-status-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }

        .aideom-status {
            padding: 0.3rem 0.5rem;
            border-radius: 999px;
            font-size: 0.71rem;
            font-weight: 700;
            border: 1px solid rgba(34, 197, 94, 0.20);
            background: rgba(22, 101, 52, 0.14);
            color: #86efac;
        }

        /* ---------- Footer ---------- */
        .aideom-footer {
            margin-top: 1.6rem;
            padding: 0.95rem 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.15);
            color: #64748b;
            font-size: 0.74rem;
            line-height: 1.55;
            text-align: center;
        }

        /* ---------- Responsive ---------- */
        @media (max-width: 1100px) {
            .aideom-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .aideom-levels {
                grid-template-columns: 1fr;
            }

            .aideom-feature-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .aideom-data-strip {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 720px) {
            .aideom-hero {
                padding: 1.45rem 1.1rem 1.35rem 1.1rem;
                border-radius: 18px;
            }

            .aideom-kpi-grid {
                grid-template-columns: 1fr;
            }

            .aideom-feature-grid {
                grid-template-columns: 1fr;
            }

            .aideom-section-head {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """
    )

    # -----------------------------------------------------
    # Nội dung trang chủ
    # -----------------------------------------------------
    st.html(
        """
        <div class="aideom-home">

            <section class="aideom-hero">
                <div class="aideom-hero-content">

                    <div class="aideom-eyebrow">
                        <span class="aideom-eyebrow-dot"></span>
                        Decision Intelligence Platform
                    </div>

                    <h1 class="aideom-title">
                        VN <span class="aideom-title-accent">AIDEOM-VN</span>
                    </h1>

                    <p class="aideom-subtitle">
                        AI-Driven Decision Optimization Model for Vietnam
                    </p>

                    <p class="aideom-description">
                        Nền tảng mô hình hóa và hỗ trợ ra quyết định phát triển kinh tế Việt Nam
                        trong kỷ nguyên AI. Hệ thống tích hợp 12 bài toán từ dự báo tăng trưởng,
                        phân bổ ngân sách, xếp hạng ưu tiên, tối ưu đa mục tiêu, bất định
                        đến học tăng cường và dashboard chính sách.
                    </p>

                    <div class="aideom-hero-tags">
                        <span class="aideom-tag">● Streamlit</span>
                        <span class="aideom-tag">● Python</span>
                        <span class="aideom-tag">● Optimization</span>
                        <span class="aideom-tag">● AI & Digital Economy</span>
                        <span class="aideom-tag">● Vietnam 2020–2035</span>
                    </div>

                </div>
            </section>

            <section class="aideom-kpi-grid">

                <article class="aideom-kpi-card">
                    <div class="aideom-kpi-label">GDP Việt Nam 2025</div>
                    <div class="aideom-kpi-value">514,0 tỷ USD</div>
                    <div class="aideom-kpi-note">↗ 8,02% so với 2024</div>
                </article>

                <article class="aideom-kpi-card">
                    <div class="aideom-kpi-label">Kinh tế số / GDP</div>
                    <div class="aideom-kpi-value">≈ 19,5%</div>
                    <div class="aideom-kpi-note">↗ 1,2 điểm phần trăm</div>
                </article>

                <article class="aideom-kpi-card">
                    <div class="aideom-kpi-label">FDI giải ngân 2025</div>
                    <div class="aideom-kpi-value">27,6 tỷ USD</div>
                    <div class="aideom-kpi-note">↗ 8,9% cùng kỳ</div>
                </article>

                <article class="aideom-kpi-card">
                    <div class="aideom-kpi-label">GDP bình quân đầu người</div>
                    <div class="aideom-kpi-value">5.026 USD</div>
                    <div class="aideom-kpi-note">↗ 6,9% theo giá hiện hành</div>
                </article>

            </section>

            <div class="aideom-section-head">
                <div>
                    <h2 class="aideom-section-title">📚 12 bài toán theo 4 cấp độ</h2>
                    <p class="aideom-section-subtitle">
                        Lộ trình từ mô hình nền tảng đến hệ thống hỗ trợ quyết định tích hợp.
                    </p>
                </div>
                <div class="aideom-section-badge">12 models · 4 levels</div>
            </div>

            <section class="aideom-levels">

                <article class="aideom-level-card">
                    <div class="aideom-level-head">
                        <div class="aideom-level-name">
                            <span class="aideom-level-dot level-green"></span>
                            CẤP ĐỘ DỄ — Làm quen mô hình
                        </div>
                        <div class="aideom-level-count">Bài 1–3</div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 1</div>
                        <div class="aideom-task-name">
                            Hàm sản xuất Cobb–Douglas mở rộng và Growth Accounting
                            <span class="aideom-task-tech">
                                TFP · dự báo GDP 2030 · phân rã tăng trưởng
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 2</div>
                        <div class="aideom-task-name">
                            LP phân bổ ngân sách số cho bốn hạng mục
                            <span class="aideom-task-tech">
                                SciPy · shadow price · phân tích độ nhạy
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 3</div>
                        <div class="aideom-task-name">
                            Chỉ số ưu tiên cho 10 ngành Việt Nam
                            <span class="aideom-task-tech">
                                Min-max · weighted scoring · policy sensitivity
                            </span>
                        </div>
                    </div>
                </article>

                <article class="aideom-level-card">
                    <div class="aideom-level-head">
                        <div class="aideom-level-name">
                            <span class="aideom-level-dot level-yellow"></span>
                            CẤP ĐỘ TRUNG BÌNH — Tối ưu cổ điển
                        </div>
                        <div class="aideom-level-count">Bài 4–6</div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 4</div>
                        <div class="aideom-task-name">
                            LP phân bổ ngân sách số ngành–vùng
                            <span class="aideom-task-tech">
                                24 biến · fairness · PuLP · CVXPY
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 5</div>
                        <div class="aideom-task-name">
                            MIP lựa chọn 15 dự án chuyển đổi số
                            <span class="aideom-task-tech">
                                Binary · knapsack · prerequisite · CBC
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 6</div>
                        <div class="aideom-task-name">
                            TOPSIS xếp hạng sáu vùng kinh tế
                            <span class="aideom-task-tech">
                                Expert weight · Entropy · AHP · sensitivity
                            </span>
                        </div>
                    </div>
                </article>

                <article class="aideom-level-card">
                    <div class="aideom-level-head">
                        <div class="aideom-level-name">
                            <span class="aideom-level-dot level-orange"></span>
                            CẤP ĐỘ NÂNG CAO — Đa mục tiêu và động
                        </div>
                        <div class="aideom-level-count">Bài 7–9</div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 7</div>
                        <div class="aideom-task-name">
                            Pareto đa mục tiêu và khung NSGA-II
                            <span class="aideom-task-tech">
                                Growth · inclusion · emission · data risk
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 8</div>
                        <div class="aideom-task-name">
                            Tối ưu động liên thời gian 2026–2035
                            <span class="aideom-task-tech">
                                SLSQP · welfare · shock 2028 · front-load
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 9</div>
                        <div class="aideom-task-name">
                            Tác động AI tới thị trường lao động
                            <span class="aideom-task-tech">
                                NetJob · retraining · vulnerable groups · Sankey
                            </span>
                        </div>
                    </div>
                </article>

                <article class="aideom-level-card">
                    <div class="aideom-level-head">
                        <div class="aideom-level-name">
                            <span class="aideom-level-dot level-purple"></span>
                            CẤP ĐỘ CHUYÊN SÂU — Bất định và thích nghi
                        </div>
                        <div class="aideom-level-count">Bài 10–12</div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 10</div>
                        <div class="aideom-task-name">
                            Quy hoạch ngẫu nhiên hai giai đoạn
                            <span class="aideom-task-tech">
                                SP · EV · VSS · EVPI · robust optimization
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 11</div>
                        <div class="aideom-task-name">
                            Q-learning cho chính sách kinh tế thích nghi
                            <span class="aideom-task-tech">
                                MDP · tabular Q-learning · DQN extension
                            </span>
                        </div>
                    </div>

                    <div class="aideom-task-row">
                        <div class="aideom-task-code">Bài 12</div>
                        <div class="aideom-task-name">
                            AIDEOM-VN tích hợp
                            <span class="aideom-task-tech">
                                6 modules · 5 scenarios · KPI · alerts · recommendations
                            </span>
                        </div>
                    </div>
                </article>

            </section>

            <div class="aideom-section-head">
                <div>
                    <h2 class="aideom-section-title">🧠 Năng lực của hệ thống</h2>
                    <p class="aideom-section-subtitle">
                        Một dashboard, nhiều lớp phân tích và công cụ ra quyết định.
                    </p>
                </div>
                <div class="aideom-section-badge">Decision support stack</div>
            </div>

            <section class="aideom-feature-grid">

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">📈</div>
                    <div class="aideom-feature-title">Dự báo & mô phỏng</div>
                    <div class="aideom-feature-text">
                        Dự báo GDP, TFP, quỹ đạo vốn, số hóa, AI và nhân lực theo thời gian.
                    </div>
                </article>

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">⚙️</div>
                    <div class="aideom-feature-title">Tối ưu hóa</div>
                    <div class="aideom-feature-text">
                        LP, MIP, stochastic programming, dynamic optimization và robust optimization.
                    </div>
                </article>

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">🏆</div>
                    <div class="aideom-feature-title">Xếp hạng đa tiêu chí</div>
                    <div class="aideom-feature-text">
                        TOPSIS, Entropy, AHP, chỉ số Priority và phân tích độ nhạy trọng số.
                    </div>
                </article>

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">🌐</div>
                    <div class="aideom-feature-title">Đa mục tiêu</div>
                    <div class="aideom-feature-text">
                        Pareto, đánh đổi tăng trưởng–bao trùm–môi trường–an ninh dữ liệu.
                    </div>
                </article>

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">🤖</div>
                    <div class="aideom-feature-title">Chính sách thích nghi</div>
                    <div class="aideom-feature-text">
                        Q-learning và khung DQN cho cơ cấu đầu tư theo trạng thái kinh tế.
                    </div>
                </article>

                <article class="aideom-feature-card">
                    <div class="aideom-feature-icon">🛡️</div>
                    <div class="aideom-feature-title">Cảnh báo rủi ro</div>
                    <div class="aideom-feature-text">
                        Theo dõi cyber risk, phát thải, việc làm, công bằng vùng và năng lực hấp thụ.
                    </div>
                </article>

            </section>

            <section class="aideom-data-strip">

                <article class="aideom-data-card">
                    <div class="aideom-data-title">📁 Hệ dữ liệu sử dụng</div>
                    <div class="aideom-data-text">
                        Dữ liệu vĩ mô, ngành và vùng Việt Nam giai đoạn 2020–2025;
                        tham số mô phỏng được cấu trúc cho các bài toán chính sách
                        giai đoạn 2026–2035.
                    </div>

                    <div class="aideom-status-list">
                        <span class="aideom-status">Macro</span>
                        <span class="aideom-status">10 sectors</span>
                        <span class="aideom-status">6 regions</span>
                        <span class="aideom-status">Scenarios</span>
                        <span class="aideom-status">Policy parameters</span>
                    </div>
                </article>

                <article class="aideom-data-card">
                    <div class="aideom-data-title">✅ Trạng thái hệ thống</div>
                    <div class="aideom-data-text">
                        12/12 trang mô hình đã được cấu trúc thành dashboard tương tác.
                        Chọn bài ở thanh điều hướng bên trái để xem mô hình, kết quả,
                        biểu đồ và diễn giải chính sách.
                    </div>

                    <div class="aideom-status-list">
                        <span class="aideom-status">Online</span>
                        <span class="aideom-status">Interactive</span>
                        <span class="aideom-status">Downloadable results</span>
                    </div>
                </article>

            </section>

            <footer class="aideom-footer">
                VN AIDEOM-VN · AI-Driven Decision Optimization Model for Vietnam<br>
                Python · Streamlit · SciPy · PuLP · CVXPY · Plotly
            </footer>

        </div>
        """
    )

def page_1():
    hero(
        "Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa",
        "Trình bày đầy đủ các mục 1.1-1.5: bối cảnh, mô hình, dữ liệu, bốn yêu cầu lập trình và thảo luận chính sách.",
        ["1.1-1.5", "Growth accounting", "TFP", "MAPE", "GDP 2030"],
    )

    # =====================================================
    # 1.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown("## 1.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        GDP Việt Nam tăng từ **8.044,4 nghìn tỷ VND năm 2020** lên
        **12.847,6 nghìn tỷ VND năm 2025**. Trong cùng giai đoạn, tỷ trọng
        kinh tế số tăng từ **12,0%** lên **19,5% GDP**, năng lực AI được đại diện
        bởi số doanh nghiệp công nghệ số tăng từ **55,6 nghìn** lên **80,1 nghìn**,
        và tỷ lệ lao động qua đào tạo tăng từ **24,1%** lên **29,2%**.

        Bài toán cần kiểm tra liệu hàm sản xuất Cobb-Douglas mở rộng có thể tái hiện
        hợp lý sản lượng thực tế hay không, đồng thời xác định mức đóng góp của vốn,
        lao động, số hóa, AI, nhân lực số và TFP vào tăng trưởng.
        """
    )

    macro = load_macro().sort_values("year").reset_index(drop=True)
    context_table = macro.loc[
        macro["year"].isin([2024, 2025]),
        [
            "year",
            "GDP_trillion_VND",
            "GDP_growth_pct",
            "digital_economy_share_GDP_pct",
            "labor_productivity_million_VND",
        ],
    ].rename(
        columns={
            "year": "Năm",
            "GDP_trillion_VND": "GDP (nghìn tỷ VND)",
            "GDP_growth_pct": "Tăng trưởng GDP (%)",
            "digital_economy_share_GDP_pct": "Kinh tế số/GDP (%)",
            "labor_productivity_million_VND": "NSLĐ (triệu VND/người)",
        }
    )
    st.dataframe(context_table, use_container_width=True, hide_index=True)

    # =====================================================
    # 1.2. Mô hình toán học
    # =====================================================
    st.markdown("## 1.2. Mô hình toán học")
    st.latex(r"Y_t=A_tK_t^{\alpha}L_t^{\beta}D_t^{\gamma}AI_t^{\delta}H_t^{\theta}")
    st.latex(r"\alpha+\beta+\gamma+\delta+\theta=1")
    st.markdown(
        """
        Trong đó: **Y** là GDP; **A** là năng suất nhân tố tổng hợp (TFP);
        **K** là vốn vật chất; **L** là lao động; **D** là mức độ số hóa;
        **AI** là năng lực trí tuệ nhân tạo; **H** là vốn nhân lực số.
        Các số mũ là độ co giãn của sản lượng đối với từng đầu vào.
        """
    )
    st.latex(
        r"\Delta\ln Y_t=\Delta\ln A_t+\alpha\Delta\ln K_t+"
        r"\beta\Delta\ln L_t+\gamma\Delta\ln D_t+"
        r"\delta\Delta\ln AI_t+\theta\Delta\ln H_t"
    )

    with st.expander("Điều chỉnh hệ số đàn hồi", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        alpha = c1.slider("α - Vốn K", 0.20, 0.45, 0.33, 0.01, key="b1_alpha")
        beta = c2.slider("β - Lao động L", 0.25, 0.55, 0.42, 0.01, key="b1_beta")
        gamma = c3.slider("γ - Số hóa D", 0.02, 0.20, 0.10, 0.01, key="b1_gamma")
        delta = c4.slider("δ - AI", 0.02, 0.18, 0.08, 0.01, key="b1_delta")
        theta = 1 - alpha - beta - gamma - delta
        if theta <= 0:
            st.error("Tổng α + β + γ + δ phải nhỏ hơn 1 để θ dương.")
            st.stop()
        st.success(
            f"θ = {theta:.2f}; tổng hệ số = "
            f"{alpha + beta + gamma + delta + theta:.2f} (lợi suất không đổi theo quy mô)."
        )

    # =====================================================
    # 1.3. Dữ liệu Việt Nam 2020-2025
    # =====================================================
    st.markdown("## 1.3. Dữ liệu Việt Nam 2020-2025")
    years, Y, K, L, D, AI, H, A = compute_tfp(alpha, beta, gamma, delta, theta)
    input_df = pd.DataFrame(
        {
            "Năm": years,
            "Y - GDP (nghìn tỷ VND)": Y,
            "K - Vốn tích lũy": K,
            "L - Lao động (triệu)": L,
            "D - Kinh tế số/GDP (%)": D,
            "AI - DN số (nghìn)": AI,
            "H - Lao động qua đào tạo (%)": H,
        }
    )
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    st.caption(
        "Hệ số mặc định theo đề: α=0,33; β=0,42; γ=0,10; δ=0,08; θ=0,07. "
        "Các biến K, L, AI và H được dùng theo bảng dữ liệu trong đề bài."
    )

    # Các kết quả dùng chung cho 1.4 và 1.5
    A_mean = A.mean()
    Y_hat = A_mean * K**alpha * L**beta * D**gamma * AI**delta * H**theta
    mape = safe_mape(Y, Y_hat)

    annual_contributions = {
        "K - Vốn": alpha * np.diff(np.log(K)),
        "L - Lao động": beta * np.diff(np.log(L)),
        "D - Số hóa": gamma * np.diff(np.log(D)),
        "AI - Năng lực AI": delta * np.diff(np.log(AI)),
        "H - Nhân lực số": theta * np.diff(np.log(H)),
        "TFP": np.diff(np.log(A)),
    }
    avg_log_growth = np.diff(np.log(Y)).mean()
    contrib_df = pd.DataFrame(
        {
            "Yếu tố": list(annual_contributions.keys()),
            "Đóng góp bình quân (% log/năm)": [
                100 * values.mean() for values in annual_contributions.values()
            ],
            "Tỷ trọng trong tăng trưởng (%)": [
                100 * values.mean() / avg_log_growth
                for values in annual_contributions.values()
            ],
        }
    )

    # Kịch bản 2030: đề yêu cầu cả K và L tăng 6%/năm
    n_years = 5
    K2030 = K[-1] * (1.06**n_years)
    L2030 = L[-1] * (1.06**n_years)
    D2030 = 30.0
    AI2030 = 100.0
    H2030 = 35.0
    A2030 = A[-1] * (1.012**n_years)
    Y2030 = (
        A2030
        * K2030**alpha
        * L2030**beta
        * D2030**gamma
        * AI2030**delta
        * H2030**theta
    )
    growth_2025_2030 = (Y2030 / Y[-1] - 1) * 100

    # =====================================================
    # 1.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 1.4. Yêu cầu lập trình")
    tab141, tab142, tab143, tab144 = st.tabs(
        ["1.4.1 - TFP", "1.4.2 - Dự báo & MAPE", "1.4.3 - Phân rã", "1.4.4 - GDP 2030"]
    )

    with tab141:
        st.markdown("### Câu 1.4.1. Ước lượng TFP Aₜ")
        tfp_df = pd.DataFrame({"Năm": years, "TFP A_t": A})
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.dataframe(tfp_df, use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(
                plot_line(tfp_df, "Năm", "TFP A_t", "TFP Aₜ giai đoạn 2020-2025"),
                use_container_width=True,
            )
        tfp_cagr = ((A[-1] / A[0]) ** (1 / (len(A) - 1)) - 1) * 100
        tfp_direction = "tăng" if A[-1] > A[0] else "giảm" if A[-1] < A[0] else "ổn định"
        st.info(
            f"TFP có xu hướng **{tfp_direction}** từ {A[0]:.3f} năm 2020 "
            f"lên {A[-1]:.3f} năm 2025; tốc độ thay đổi bình quân khoảng "
            f"**{tfp_cagr:.2f}%/năm**."
        )
        with st.expander("Xem mã Python cho câu 1.4.1"):
            st.code(
                """A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
tfp_df = pd.DataFrame({'year': years, 'TFP_A_t': A})
fig = px.line(tfp_df, x='year', y='TFP_A_t', markers=True)""",
                language="python",
            )

    with tab142:
        st.markdown("### Câu 1.4.2. GDP dự báo và MAPE")
        forecast_df = pd.DataFrame(
            {
                "Năm": years,
                "GDP thực tế": Y,
                "GDP dự báo": Y_hat,
                "Sai số tuyệt đối (%)": np.abs((Y - Y_hat) / Y) * 100,
            }
        )
        kpi_cards(
            [
                ("A trung bình", f"{A_mean:.3f}", "trung bình 2020-2025"),
                ("MAPE", f"{mape:.2f}%", "sai số phần trăm tuyệt đối bình quân"),
                ("Sai số nhỏ nhất", f"{forecast_df['Sai số tuyệt đối (%)'].min():.2f}%", "theo năm"),
                ("Sai số lớn nhất", f"{forecast_df['Sai số tuyệt đối (%)'].max():.2f}%", "theo năm"),
            ]
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        fig = px.line(
            forecast_df,
            x="Năm",
            y=["GDP thực tế", "GDP dự báo"],
            markers=True,
            template=PLOT_TEMPLATE,
            title="So sánh GDP thực tế và GDP dự báo",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"MAPE bằng **{mape:.2f}%**. Chỉ số này càng thấp thì mức khớp trong mẫu càng tốt; "
            "tuy nhiên đây là phép hiệu chỉnh trên chính giai đoạn 2020-2025, chưa phải kiểm định dự báo ngoài mẫu."
        )
        with st.expander("Xem mã Python cho câu 1.4.2"):
            st.code(
                """A_mean = A.mean()
Y_hat = A_mean * K**alpha * L**beta * D**gamma * AI**delta * H**theta
MAPE = np.mean(np.abs((Y - Y_hat) / Y)) * 100""",
                language="python",
            )

    with tab143:
        st.markdown("### Câu 1.4.3. Phân rã tăng trưởng 2020-2025")
        st.metric("Tăng trưởng GDP bình quân theo log", f"{100 * avg_log_growth:.2f}%/năm")
        st.dataframe(
            contrib_df.style.format(
                {
                    "Đóng góp bình quân (% log/năm)": "{:.3f}",
                    "Tỷ trọng trong tăng trưởng (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            plot_bar(
                contrib_df,
                "Yếu tố",
                "Tỷ trọng trong tăng trưởng (%)",
                "Tỷ trọng đóng góp vào tăng trưởng GDP",
                text="Tỷ trọng trong tăng trưởng (%)",
            ),
            use_container_width=True,
        )
        st.caption(
            "Tỷ trọng có thể âm hoặc vượt 100% khi một yếu tố kéo giảm tăng trưởng và các yếu tố khác bù đắp."
        )
        with st.expander("Xem mã Python cho câu 1.4.3"):
            st.code(
                """growth_Y = np.diff(np.log(Y))
contribution_K = alpha * np.diff(np.log(K))
contribution_L = beta * np.diff(np.log(L))
contribution_D = gamma * np.diff(np.log(D))
contribution_AI = delta * np.diff(np.log(AI))
contribution_H = theta * np.diff(np.log(H))
contribution_TFP = np.diff(np.log(A))""",
                language="python",
            )

    with tab144:
        st.markdown("### Câu 1.4.4. Mô phỏng GDP Việt Nam năm 2030")
        scenario_df = pd.DataFrame(
            {
                "Biến": ["K", "L", "D", "AI", "H", "TFP A"],
                "Năm 2025": [K[-1], L[-1], D[-1], AI[-1], H[-1], A[-1]],
                "Kịch bản 2030": [K2030, L2030, D2030, AI2030, H2030, A2030],
                "Giả định": [
                    "Tăng 6%/năm",
                    "Tăng 6%/năm",
                    "Đạt 30% GDP",
                    "100 nghìn DN",
                    "Đạt 35%",
                    "Tăng 1,2%/năm",
                ],
            }
        )
        kpi_cards(
            [
                ("GDP 2025", f"{Y[-1]:,.1f}", "nghìn tỷ VND"),
                ("GDP 2030 mô phỏng", f"{Y2030:,.1f}", "nghìn tỷ VND"),
                ("Tăng 2025-2030", f"{growth_2025_2030:.1f}%", "theo kịch bản đề bài"),
                ("Tăng trưởng quy đổi", f"{((Y2030 / Y[-1]) ** (1/5) - 1) * 100:.2f}%/năm", "CAGR GDP"),
            ]
        )
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        st.warning(
            "Kịch bản 1.4.4 là mô phỏng có điều kiện. Kết quả phụ thuộc mạnh vào giả định cả vốn K và lao động L "
            "đều tăng 6%/năm; đây không phải dự báo chính thức."
        )
        with st.expander("Xem mã Python cho câu 1.4.4"):
            st.code(
                """K2030 = K[-1] * 1.06**5
L2030 = L[-1] * 1.06**5
D2030, AI2030, H2030 = 30, 100, 35
A2030 = A[-1] * 1.012**5
Y2030 = A2030 * K2030**alpha * L2030**beta * D2030**gamma * AI2030**delta * H2030**theta""",
                language="python",
            )

    # File kết quả cho người dùng tải từ web
    result_export = forecast_df.merge(tfp_df, on="Năm")
    st.download_button(
        "Tải kết quả Bài 1 dạng CSV",
        data=result_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai1_cobb_douglas_ket_qua.csv",
        mime="text/csv",
        key="download_bai1",
    )

    # =====================================================
    # 1.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown("## 1.5. Câu hỏi thảo luận chính sách")

    cumulative_new = {
        "số hóa D": gamma * np.log(D[-1] / D[0]),
        "năng lực AI": delta * np.log(AI[-1] / AI[0]),
        "nhân lực số H": theta * np.log(H[-1] / H[0]),
    }
    strongest_new = max(cumulative_new, key=cumulative_new.get)
    digital_pp_required = (30 - D[-1]) / 5
    digital_cagr_required = ((30 / D[-1]) ** (1 / 5) - 1) * 100

    with st.expander("a) TFP tăng hay giảm và phản ánh điều gì?", expanded=True):
        st.markdown(
            f"TFP **{tfp_direction}** trong giai đoạn 2020-2025, từ **{A[0]:.3f}** "
            f"lên **{A[-1]:.3f}**, tương ứng khoảng **{tfp_cagr:.2f}%/năm**. "
            "Nếu TFP tăng, chất lượng tăng trưởng được cải thiện vì một phần sản lượng tăng thêm đến từ hiệu quả, "
            "công nghệ và tổ chức sản xuất, không chỉ từ mở rộng vốn và lao động. Tuy nhiên, kết luận này phụ thuộc "
            "vào cách đo D, AI, H và bộ hệ số đàn hồi giả định."
        )

    with st.expander("b) Trong D, AI và H, yếu tố nào đóng góp nhiều nhất?", expanded=True):
        comparison_df = pd.DataFrame(
            {
                "Yếu tố mới": list(cumulative_new.keys()),
                "Đóng góp log tích lũy 2020-2025": list(cumulative_new.values()),
            }
        ).sort_values("Đóng góp log tích lũy 2020-2025", ascending=False)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"Theo bộ hệ số và dữ liệu hiện tại, **{strongest_new}** đóng góp lớn nhất trong ba yếu tố mới. "
            "Kết quả được quyết định đồng thời bởi tốc độ tăng của biến và độ co giãn gán cho biến đó; vì vậy "
            "không nên diễn giải đây là quan hệ nhân quả đã được kiểm định."
        )

    with st.expander("c) Mục tiêu kinh tế số đạt 30% GDP năm 2030 có khả thi không?", expanded=True):
        st.markdown(
            f"Từ mức **{D[-1]:.1f}% năm 2025** lên **30% năm 2030**, tỷ trọng kinh tế số phải tăng trung bình "
            f"khoảng **{digital_pp_required:.2f} điểm phần trăm/năm**, tương đương tăng **{digital_cagr_required:.2f}%/năm** "
            "theo mức chỉ số. Mô hình cho thấy mục tiêu có thể được mô phỏng, nhưng để đánh giá khả thi cần bổ sung "
            "các ràng buộc về ngân sách, năng lực nhân lực, hạ tầng dữ liệu, an ninh mạng, khả năng hấp thụ của doanh nghiệp, "
            "chênh lệch vùng và độ trễ chính sách."
        )


def page_2():
    hero(
        "Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số",
        "Trình bày đầy đủ các mục 2.1-2.5: bối cảnh, mô hình LP, diễn giải hệ số, bốn yêu cầu lập trình và thảo luận chính sách.",
        ["2.1-2.5", "Linear Programming", "SciPy", "PuLP", "Shadow price"],
    )

    # =====================================================
    # Hàm giải dùng chung
    # =====================================================
    item_names = [
        "x₁ - Hạ tầng số",
        "x₂ - AI và dữ liệu",
        "x₃ - Nhân lực số",
        "x₄ - R&D công nghệ",
    ]
    impact = np.array([0.85, 1.20, 0.95, 1.35], dtype=float)

    def solve_with_scipy(budget=100.0, min_human=20.0):
        # Tối đa hóa Z tương đương tối thiểu hóa -Z
        c = -impact
        A_ub = np.array(
            [
                [1.00, 1.00, 1.00, 1.00],      # tổng ngân sách
                [-1.00, 0.00, 0.00, 0.00],     # x1 >= 25
                [0.00, -1.00, 0.00, 0.00],     # x2 >= 15
                [0.00, 0.00, -1.00, 0.00],     # x3 >= min_human
                [0.00, 0.00, 0.00, -1.00],     # x4 >= 10
                [0.35, -0.65, 0.35, -0.65],    # x2+x4 >= 35% tổng
            ],
            dtype=float,
        )
        b_ub = np.array(
            [budget, -25.0, -15.0, -float(min_human), -10.0, 0.0],
            dtype=float,
        )
        return linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0, None)] * 4,
            method="highs",
        )

    def build_allocation_table(x):
        out = pd.DataFrame(
            {
                "Hạng mục": item_names,
                "Phân bổ tối ưu (nghìn tỷ VND)": x,
                "Hệ số tác động": impact,
            }
        )
        out["GDP kỳ vọng tăng thêm"] = (
            out["Phân bổ tối ưu (nghìn tỷ VND)"] * out["Hệ số tác động"]
        )
        return out

    # Nghiệm chuẩn của đề bài
    base_res = solve_with_scipy(budget=100.0, min_human=20.0)
    if not base_res.success:
        st.error("Bài toán chuẩn B=100 không khả thi. Hãy kiểm tra lại các ràng buộc.")
        return

    base_x = base_res.x
    base_z = -base_res.fun
    base_table = build_allocation_table(base_x)
    strategic_share = 100 * (base_x[1] + base_x[3]) / max(base_x.sum(), 1e-12)

    # =====================================================
    # 2.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown("## 2.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        Theo Chương trình Chuyển đổi số quốc gia, Việt Nam đặt mục tiêu nâng tỷ trọng
        kinh tế số và thúc đẩy đồng thời hạ tầng số, dữ liệu, trí tuệ nhân tạo, nhân lực
        số và nghiên cứu phát triển. Giả sử Chính phủ có **100 nghìn tỷ VND** để phân bổ
        cho bốn hạng mục:

        - **x₁:** hạ tầng số;
        - **x₂:** AI và dữ liệu;
        - **x₃:** nhân lực số;
        - **x₄:** R&D công nghệ.

        Bài toán cần xác định cơ cấu phân bổ làm **tăng GDP kỳ vọng lớn nhất**, nhưng vẫn
        bảo đảm mức đầu tư tối thiểu và tỷ trọng tối thiểu dành cho công nghệ chiến lược.
        """
    )

    context_df = pd.DataFrame(
        {
            "Hạng mục": item_names,
            "Mức tối thiểu": [25, 15, 20, 10],
            "Hệ số GDP kỳ vọng": impact,
            "Ý nghĩa": [
                "Nền tảng kết nối, trung tâm dữ liệu và hạ tầng số",
                "AI, dữ liệu lớn và năng lực xử lý số",
                "Đào tạo kỹ năng số và kỹ sư AI",
                "Nghiên cứu, đổi mới sáng tạo và công nghệ lõi",
            ],
        }
    )
    st.dataframe(context_df, use_container_width=True, hide_index=True)

    # =====================================================
    # 2.2. Mô hình toán học
    # =====================================================
    st.markdown("## 2.2. Mô hình toán học")
    st.markdown("### Biến quyết định")
    st.latex(
        r"x_1=\text{đầu tư hạ tầng số},\quad "
        r"x_2=\text{đầu tư AI và dữ liệu},\quad "
        r"x_3=\text{đầu tư nhân lực số},\quad "
        r"x_4=\text{đầu tư R\&D}"
    )

    st.markdown("### Hàm mục tiêu")
    st.latex(r"\max Z=0.85x_1+1.20x_2+0.95x_3+1.35x_4")

    st.markdown("### Các ràng buộc")
    st.latex(r"x_1+x_2+x_3+x_4\leq 100")
    st.latex(r"x_1\geq 25,\quad x_2\geq 15,\quad x_3\geq 20,\quad x_4\geq 10")
    st.latex(
        r"x_2+x_4\geq 0.35(x_1+x_2+x_3+x_4)"
    )
    st.latex(r"x_1,x_2,x_3,x_4\geq 0")

    st.info(
        "Ràng buộc công nghệ chiến lược có thể chuyển về dạng chuẩn của linprog: "
        "0,35x₁ - 0,65x₂ + 0,35x₃ - 0,65x₄ ≤ 0."
    )

    # =====================================================
    # 2.3. Diễn giải hệ số mục tiêu
    # =====================================================
    st.markdown("## 2.3. Diễn giải hệ số mục tiêu")
    coefficient_df = pd.DataFrame(
        {
            "Biến": ["x₁", "x₂", "x₃", "x₄"],
            "Hạng mục": [
                "Hạ tầng số",
                "AI và dữ liệu",
                "Nhân lực số",
                "R&D công nghệ",
            ],
            "Hệ số": impact,
            "Diễn giải": [
                "Một đơn vị đầu tư tạo 0,85 đơn vị GDP kỳ vọng",
                "Một đơn vị đầu tư tạo 1,20 đơn vị GDP kỳ vọng",
                "Một đơn vị đầu tư tạo 0,95 đơn vị GDP kỳ vọng",
                "Một đơn vị đầu tư tạo 1,35 đơn vị GDP kỳ vọng",
            ],
        }
    )
    st.dataframe(coefficient_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        Trong mô hình giả định, **R&D có hệ số cao nhất (1,35)**, tiếp theo là
        **AI và dữ liệu (1,20)**, **nhân lực số (0,95)** và **hạ tầng số (0,85)**.
        Đây là hệ số tác động kỳ vọng dùng cho bài tập, không phải ước lượng nhân quả
        đã được kiểm định từ dữ liệu vi mô.
        """
    )

    # =====================================================
    # 2.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 2.4. Yêu cầu lập trình")
    tab241, tab242, tab243, tab244 = st.tabs(
        [
            "2.4.1 - SciPy",
            "2.4.2 - PuLP & dual",
            "2.4.3 - Độ nhạy ngân sách",
            "2.4.4 - Ưu tiên nhân lực",
        ]
    )

    with tab241:
        st.markdown("### Câu 2.4.1. Giải bằng scipy.optimize.linprog")
        kpi_cards(
            [
                ("Giá trị tối ưu Z*", f"{base_z:,.2f}", "nghìn tỷ VND GDP kỳ vọng"),
                ("Ngân sách sử dụng", f"{base_x.sum():,.2f}", "trên tổng 100"),
                ("AI + R&D", f"{strategic_share:.1f}%", "yêu cầu tối thiểu 35%"),
                (
                    "Hạng mục nhận nhiều nhất",
                    base_table.sort_values(
                        "Phân bổ tối ưu (nghìn tỷ VND)", ascending=False
                    ).iloc[0]["Hạng mục"],
                    "theo nghiệm tối ưu",
                ),
            ]
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                base_table.style.format(
                    {
                        "Phân bổ tối ưu (nghìn tỷ VND)": "{:.2f}",
                        "Hệ số tác động": "{:.2f}",
                        "GDP kỳ vọng tăng thêm": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.plotly_chart(
                plot_bar(
                    base_table,
                    "Hạng mục",
                    "Phân bổ tối ưu (nghìn tỷ VND)",
                    "Phân bổ tối ưu theo SciPy",
                    text="Phân bổ tối ưu (nghìn tỷ VND)",
                ),
                use_container_width=True,
            )

        residual_df = pd.DataFrame(
            {
                "Ràng buộc": [
                    "Ngân sách tổng",
                    "x₁ tối thiểu",
                    "x₂ tối thiểu",
                    "x₃ tối thiểu",
                    "x₄ tối thiểu",
                    "Tỷ trọng AI + R&D",
                ],
                "Slack theo SciPy": base_res.ineqlin.residual,
                "Đang chặt?": np.isclose(
                    base_res.ineqlin.residual, 0.0, atol=1e-7
                ),
            }
        )
        st.dataframe(residual_df, use_container_width=True, hide_index=True)

        with st.expander("Xem mã Python cho câu 2.4.1"):
            st.code(
                """from scipy.optimize import linprog

c = [-0.85, -1.20, -0.95, -1.35]
A_ub = [
    [1, 1, 1, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1],
    [0.35, -0.65, 0.35, -0.65],
]
b_ub = [100, -25, -15, -20, -10, 0]

res = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(0, None)] * 4,
    method="highs",
)
Z_star = -res.fun
x_star = res.x""",
                language="python",
            )

    with tab242:
        st.markdown("### Câu 2.4.2. Giải lại bằng PuLP và đọc giá đối ngẫu")
        try:
            import pulp

            model = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
            x1 = pulp.LpVariable("x1_Infrastructure", lowBound=0)
            x2 = pulp.LpVariable("x2_AI_Data", lowBound=0)
            x3 = pulp.LpVariable("x3_Digital_Human", lowBound=0)
            x4 = pulp.LpVariable("x4_RD", lowBound=0)

            model += (
                0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4,
                "Expected_GDP_Gain",
            )
            model += x1 + x2 + x3 + x4 <= 100, "C1_Total_Budget"
            model += x1 >= 25, "C2_Min_Infrastructure"
            model += x2 >= 15, "C3_Min_AI_Data"
            model += x3 >= 20, "C4_Min_Digital_Human"
            model += x4 >= 10, "C5_Min_RD"
            model += (
                x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4),
                "C6_Strategic_Technology",
            )

            solver = pulp.PULP_CBC_CMD(msg=False)
            model.solve(solver)

            pulp_status = pulp.LpStatus[model.status]
            pulp_x = np.array(
                [x1.value(), x2.value(), x3.value(), x4.value()],
                dtype=float,
            )
            pulp_z = float(pulp.value(model.objective))

            pulp_table = build_allocation_table(pulp_x)
            compare_df = pd.DataFrame(
                {
                    "Hạng mục": item_names,
                    "SciPy": base_x,
                    "PuLP": pulp_x,
                    "Chênh lệch tuyệt đối": np.abs(base_x - pulp_x),
                }
            )

            dual_rows = []
            for name, constraint in model.constraints.items():
                dual_rows.append(
                    {
                        "Ràng buộc": name,
                        "Giá đối ngẫu (pi)": getattr(constraint, "pi", np.nan),
                        "Slack": getattr(constraint, "slack", np.nan),
                    }
                )
            dual_df = pd.DataFrame(dual_rows)

            budget_dual_series = dual_df.loc[
                dual_df["Ràng buộc"] == "C1_Total_Budget",
                "Giá đối ngẫu (pi)",
            ]
            budget_dual = (
                float(budget_dual_series.iloc[0])
                if len(budget_dual_series)
                else np.nan
            )

            kpi_cards(
                [
                    ("Trạng thái PuLP", pulp_status, "CBC solver"),
                    ("Z* PuLP", f"{pulp_z:,.2f}", "so sánh với SciPy"),
                    (
                        "Sai lệch nghiệm",
                        f"{np.max(np.abs(base_x - pulp_x)):.6f}",
                        "max |SciPy - PuLP|",
                    ),
                    (
                        "Shadow price ngân sách",
                        f"{budget_dual:.2f}"
                        if np.isfinite(budget_dual)
                        else "Không đọc được",
                        "GDP kỳ vọng / 1 đơn vị ngân sách",
                    ),
                ]
            )

            st.markdown("#### So sánh nghiệm SciPy và PuLP")
            st.dataframe(
                compare_df.style.format(
                    {
                        "SciPy": "{:.4f}",
                        "PuLP": "{:.4f}",
                        "Chênh lệch tuyệt đối": "{:.6f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Giá đối ngẫu và slack")
            st.dataframe(
                dual_df.style.format(
                    {
                        "Giá đối ngẫu (pi)": "{:.4f}",
                        "Slack": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            if np.isfinite(budget_dual):
                st.info(
                    f"Giá đối ngẫu của ràng buộc ngân sách tổng là khoảng "
                    f"**{budget_dual:.2f}**. Trong vùng độ nhạy hiện tại, tăng thêm "
                    f"1 nghìn tỷ VND ngân sách làm Z* tăng xấp xỉ "
                    f"**{budget_dual:.2f} nghìn tỷ VND**."
                )
            else:
                st.warning(
                    "CBC trong môi trường hiện tại không trả về thuộc tính dual. "
                    "Bạn vẫn có thể dùng marginal của HiGHS ở bảng SciPy hoặc cài CBC đầy đủ."
                )

            with st.expander("Xem mã Python cho câu 2.4.2"):
                st.code(
                    """import pulp

m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
x1 = pulp.LpVariable("x1", lowBound=0)
x2 = pulp.LpVariable("x2", lowBound=0)
x3 = pulp.LpVariable("x3", lowBound=0)
x4 = pulp.LpVariable("x4", lowBound=0)

m += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4
m += x1 + x2 + x3 + x4 <= 100, "Total_Budget"
m += x1 >= 25, "Min_I"
m += x2 >= 15, "Min_AI"
m += x3 >= 20, "Min_H"
m += x4 >= 10, "Min_RD"
m += x2 + x4 >= 0.35*(x1+x2+x3+x4), "Strategic"

m.solve(pulp.PULP_CBC_CMD(msg=False))

for name, constraint in m.constraints.items():
    print(name, constraint.pi, constraint.slack)""",
                    language="python",
                )

        except ModuleNotFoundError:
            st.error(
                "Chưa cài thư viện PuLP. Mở requirements.txt, thêm dòng `pulp>=2.7`, "
                "lưu file, sau đó chạy `pip install -r requirements.txt`."
            )

    with tab243:
        st.markdown("### Câu 2.4.3. Phân tích độ nhạy ngân sách B")
        budget_values = [100.0, 120.0, 140.0]
        sensitivity_rows = []

        for budget in budget_values:
            res_b = solve_with_scipy(budget=budget, min_human=20.0)
            if res_b.success:
                x_b = res_b.x
                sensitivity_rows.append(
                    {
                        "Ngân sách B": budget,
                        "Z*": -res_b.fun,
                        "x₁": x_b[0],
                        "x₂": x_b[1],
                        "x₃": x_b[2],
                        "x₄": x_b[3],
                        "AI + R&D (%)": 100
                        * (x_b[1] + x_b[3])
                        / max(x_b.sum(), 1e-12),
                    }
                )

        sensitivity_df = pd.DataFrame(sensitivity_rows)
        sensitivity_df["ΔZ so với B=100"] = (
            sensitivity_df["Z*"] - sensitivity_df.loc[0, "Z*"]
        )
        sensitivity_df["Z tăng / 20 ngân sách"] = sensitivity_df["Z*"].diff()

        st.dataframe(
            sensitivity_df.style.format(
                {
                    "Ngân sách B": "{:.0f}",
                    "Z*": "{:.2f}",
                    "x₁": "{:.2f}",
                    "x₂": "{:.2f}",
                    "x₃": "{:.2f}",
                    "x₄": "{:.2f}",
                    "AI + R&D (%)": "{:.1f}",
                    "ΔZ so với B=100": "{:.2f}",
                    "Z tăng / 20 ngân sách": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            plot_line(
                sensitivity_df,
                "Ngân sách B",
                "Z*",
                "Đường cong độ nhạy Z*(B)",
            ),
            use_container_width=True,
        )

        allocation_long = sensitivity_df.melt(
            id_vars=["Ngân sách B"],
            value_vars=["x₁", "x₂", "x₃", "x₄"],
            var_name="Biến",
            value_name="Phân bổ",
        )
        fig_alloc = px.bar(
            allocation_long,
            x="Ngân sách B",
            y="Phân bổ",
            color="Biến",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="Cơ cấu phân bổ khi ngân sách tăng",
        )
        fig_alloc.update_layout(height=430, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig_alloc, use_container_width=True)

        marginal_growth = (
            sensitivity_df.loc[1, "Z*"] - sensitivity_df.loc[0, "Z*"]
        ) / (
            sensitivity_df.loc[1, "Ngân sách B"]
            - sensitivity_df.loc[0, "Ngân sách B"]
        )
        st.info(
            f"Trong khoảng B=100-140, Z* tăng gần tuyến tính. Mỗi 1 nghìn tỷ VND "
            f"ngân sách tăng thêm làm Z* tăng khoảng **{marginal_growth:.2f}** "
            "nghìn tỷ VND theo hệ số mô hình."
        )

        with st.expander("Xem mã Python cho câu 2.4.3"):
            st.code(
                """rows = []
for B in [100, 120, 140]:
    res = solve_with_scipy(budget=B, min_human=20)
    rows.append({
        "B": B,
        "Z_star": -res.fun,
        "x1": res.x[0],
        "x2": res.x[1],
        "x3": res.x[2],
        "x4": res.x[3],
    })
sensitivity_df = pd.DataFrame(rows)""",
                language="python",
            )

    with tab244:
        st.markdown("### Câu 2.4.4. Tăng sàn nhân lực số lên x₃ ≥ 30")
        human_res = solve_with_scipy(budget=100.0, min_human=30.0)

        if not human_res.success:
            st.error(
                "Bài toán không khả thi khi x₃ ≥ 30 với ngân sách B=100."
            )
        else:
            human_x = human_res.x
            human_z = -human_res.fun
            human_table = build_allocation_table(human_x)

            comparison_human = pd.DataFrame(
                {
                    "Hạng mục": item_names,
                    "Mô hình gốc x₃≥20": base_x,
                    "Ưu tiên nhân lực x₃≥30": human_x,
                    "Thay đổi": human_x - base_x,
                }
            )

            kpi_cards(
                [
                    ("Trạng thái", "Khả thi", "x₃ ≥ 30"),
                    ("Z* mới", f"{human_z:,.2f}", "nghìn tỷ VND"),
                    ("Z* thay đổi", f"{human_z - base_z:,.2f}", "so với mô hình gốc"),
                    (
                        "Tỷ lệ giảm Z*",
                        f"{100 * (base_z - human_z) / base_z:.2f}%",
                        "chi phí ưu tiên nhân lực",
                    ),
                ]
            )

            st.dataframe(
                comparison_human.style.format(
                    {
                        "Mô hình gốc x₃≥20": "{:.2f}",
                        "Ưu tiên nhân lực x₃≥30": "{:.2f}",
                        "Thay đổi": "{:+.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            compare_long = comparison_human.melt(
                id_vars="Hạng mục",
                value_vars=[
                    "Mô hình gốc x₃≥20",
                    "Ưu tiên nhân lực x₃≥30",
                ],
                var_name="Kịch bản",
                value_name="Phân bổ",
            )
            fig_compare = px.bar(
                compare_long,
                x="Hạng mục",
                y="Phân bổ",
                color="Kịch bản",
                barmode="group",
                template=PLOT_TEMPLATE,
                title="So sánh phân bổ trước và sau khi tăng sàn nhân lực",
            )
            fig_compare.update_layout(
                height=430, margin=dict(l=10, r=10, t=54, b=10)
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            st.info(
                f"Bài toán vẫn khả thi. Khi buộc x₃ tăng thêm 10 đơn vị, một phần "
                f"ngân sách phải dịch chuyển khỏi hạng mục có lợi ích biên cao hơn. "
                f"Do đó Z* giảm từ **{base_z:.2f}** xuống **{human_z:.2f}**, "
                f"tức giảm **{base_z - human_z:.2f}**."
            )

            with st.expander("Xem mã Python cho câu 2.4.4"):
                st.code(
                    """# Thay ràng buộc x3 >= 20 bằng x3 >= 30
res_priority_H = solve_with_scipy(
    budget=100,
    min_human=30,
)
print(res_priority_H.success)
print(-res_priority_H.fun)
print(res_priority_H.x)""",
                    language="python",
                )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_df = base_table.copy()
    export_df["Kịch bản"] = "B=100, x3>=20"
    st.download_button(
        "Tải kết quả Bài 2 dạng CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai2_lp_phan_bo_ngan_sach.csv",
        mime="text/csv",
        key="download_bai2",
    )

    # =====================================================
    # 2.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown("## 2.5. Câu hỏi thảo luận chính sách")

    # Shadow price từ SciPy/HiGHS cho bài toán tối thiểu hóa -Z
    scipy_budget_shadow = -float(base_res.ineqlin.marginals[0])

    with st.expander(
        "a) Ngân sách tăng thêm 1 đơn vị thì GDP kỳ vọng tăng bao nhiêu?",
        expanded=True,
    ):
        st.markdown(
            f"Giá đối ngẫu của ràng buộc ngân sách tổng theo nghiệm HiGHS là khoảng "
            f"**{scipy_budget_shadow:.2f}**. Nghĩa là, trong phạm vi độ nhạy hiện tại, "
            f"tăng thêm **1 nghìn tỷ VND** ngân sách có thể làm giá trị mục tiêu tăng "
            f"xấp xỉ **{scipy_budget_shadow:.2f} nghìn tỷ VND**. Đây là lợi ích biên "
            "theo mô hình, không nên coi là cận trên chắc chắn của chi phí cơ hội vốn công "
            "vì mô hình chưa phản ánh thuế, nợ công, độ trễ, rủi ro triển khai và hiệu ứng lấn át."
        )

    with st.expander(
        "b) Vì sao R&D có hệ số cao nhất nhưng mức tối thiểu thấp nhất?",
        expanded=True,
    ):
        st.markdown(
            "R&D có lợi ích lan tỏa dài hạn nhưng thường đi kèm độ trễ lớn, rủi ro thất bại, "
            "khó hấp thụ nếu thiếu nhân lực và hạ tầng, đồng thời kết quả khó đo lường trong ngắn hạn. "
            "Vì vậy, mô hình đặt sàn R&D thấp hơn để duy trì tính linh hoạt ngân sách. Tuy nhiên, "
            "do hệ số mục tiêu của R&D cao nhất, nghiệm tối ưu vẫn có xu hướng phân bổ phần ngân sách "
            "còn lại vào R&D sau khi đáp ứng các mức tối thiểu."
        )

    with st.expander(
        "c) Tỷ lệ 35% cho AI + R&D có khả thi trong thực tiễn không?",
        expanded=True,
    ):
        st.markdown(
            f"Trong nghiệm chuẩn, AI + R&D chiếm khoảng **{strategic_share:.1f}%** tổng ngân sách, "
            "cao hơn mức tối thiểu 35%, nên ràng buộc này khả thi về mặt toán học. Trong thực tiễn, "
            "khả năng thực hiện còn phụ thuộc vào cạnh tranh ngân sách với giao thông, y tế, giáo dục "
            "và an sinh xã hội; năng lực giải ngân; nguồn nhân lực; chất lượng dự án; cũng như cơ chế "
            "giám sát hiệu quả đầu tư. Vì vậy cần bổ sung các ràng buộc về trần giải ngân, tiến độ, "
            "rủi ro và năng lực hấp thụ."
        )


def _b3_column_map(df):
    """
    Xác định tên cột dữ liệu hiện có trong file sectors.
    Nếu chưa có cột năng suất lao động, dùng tỷ trọng GDP làm biến đại diện.
    """
    productivity_col = (
        "labor_productivity_million_VND"
        if "labor_productivity_million_VND" in df.columns
        else "gdp_share_2024_pct"
    )

    return {
        "growth": "growth_rate_2024_pct",
        "productivity": productivity_col,
        "spillover": "spillover_coef_0_1",
        "export": "export_billion_USD",
        "employment": "labor_million",
        "ai": "ai_readiness_0_100",
        "risk": "automation_risk_pct",
    }


def _b3_prepare_data():
    """
    Đọc dữ liệu ngành và chuẩn hóa 7 tiêu chí về [0,1].
    Risk là tiêu chí chi phí nên được trừ trong hàm Priority.
    """
    df = load_sectors().copy()
    cmap = _b3_column_map(df)

    norm = pd.DataFrame(index=df.index)

    norm[cmap["growth"]] = minmax(df[cmap["growth"]])
    norm[cmap["productivity"]] = minmax(df[cmap["productivity"]])
    norm[cmap["spillover"]] = minmax(df[cmap["spillover"]])
    norm[cmap["export"]] = minmax(df[cmap["export"]])
    norm[cmap["employment"]] = minmax(df[cmap["employment"]])
    norm[cmap["ai"]] = minmax(df[cmap["ai"]])
    norm[cmap["risk"]] = minmax(df[cmap["risk"]])

    return df, cmap, norm


def _b3_priority_score(norm, cmap, weights):
    """
    Tính:
    Priority = a1*Growth + a2*Productivity + a3*Spillover
             + a4*Export + a5*Employment + a6*AI - a7*Risk
    """
    weights = np.asarray(weights, dtype=float)

    positive_matrix = np.column_stack(
        [
            norm[cmap["growth"]].to_numpy(dtype=float),
            norm[cmap["productivity"]].to_numpy(dtype=float),
            norm[cmap["spillover"]].to_numpy(dtype=float),
            norm[cmap["export"]].to_numpy(dtype=float),
            norm[cmap["employment"]].to_numpy(dtype=float),
            norm[cmap["ai"]].to_numpy(dtype=float),
        ]
    )

    risk = norm[cmap["risk"]].to_numpy(dtype=float)

    return positive_matrix @ weights[:6] - weights[6] * risk


def page_3():
    hero(
        "Bài 3 — Tính chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam",
        "Trình bày đầy đủ các mục 3.1-3.5: bối cảnh, mô hình, dữ liệu, chuẩn hóa, xếp hạng, độ nhạy trọng số và thảo luận chính sách.",
        ["3.1-3.5", "Min-max", "MCDM", "AI readiness", "Policy weights"],
    )

    df, cmap, norm = _b3_prepare_data()
    sector_col = "sector_name_vi"

    # =====================================================
    # 3.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown("## 3.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        Việt Nam cần xác định ngành nào nên được ưu tiên chuyển đổi số và ứng dụng AI
        trước để tạo hiệu ứng lan tỏa lớn nhất. Nếu chỉ dựa vào một chỉ tiêu như tốc độ
        tăng trưởng hoặc năng suất lao động thì kết quả có thể bị thiên lệch.

        Vì vậy, Bài 3 xây dựng một **chỉ số ưu tiên tổng hợp Priorityᵢ** dựa trên:
        tăng trưởng, năng suất, hiệu ứng lan tỏa, xuất khẩu, việc làm, mức sẵn sàng AI
        và rủi ro tự động hóa.
        """
    )

    # =====================================================
    # 3.2. Mô hình toán học
    # =====================================================
    st.markdown("## 3.2. Mô hình toán học")

    st.latex(
        r"Priority_i="
        r"a_1Growth_i+a_2Productivity_i+a_3Spillover_i"
        r"+a_4Export_i+a_5Employment_i+a_6AIReadiness_i"
        r"-a_7Risk_i"
    )

    st.markdown("### Chuẩn hóa min-max")
    st.latex(
        r"\widetilde{x}_i="
        r"\frac{x_i-\min(x)}{\max(x)-\min(x)}"
    )

    st.markdown(
        """
        Sáu tiêu chí đầu là **tiêu chí lợi ích**: giá trị càng cao càng tốt.
        Rủi ro tự động hóa là **tiêu chí chi phí**, do đó được trừ khỏi tổng điểm.

        Các biến được chuẩn hóa về thang từ 0 đến 1 trước khi tính chỉ số Priority.
        """
    )

    # =====================================================
    # 3.3. Dữ liệu 10 ngành Việt Nam năm 2024
    # =====================================================
    st.markdown("## 3.3. Dữ liệu 10 ngành Việt Nam năm 2024")

    display_cols = [
        sector_col,
        cmap["growth"],
        cmap["productivity"],
        cmap["spillover"],
        cmap["export"],
        cmap["employment"],
        cmap["ai"],
        cmap["risk"],
    ]

    productivity_label = (
        "Năng suất lao động"
        if cmap["productivity"] == "labor_productivity_million_VND"
        else "Tỷ trọng GDP đại diện năng suất"
    )

    rename_map = {
        sector_col: "Ngành",
        cmap["growth"]: "Tăng trưởng (%)",
        cmap["productivity"]: productivity_label,
        cmap["spillover"]: "Hiệu ứng lan tỏa",
        cmap["export"]: "Xuất khẩu (tỷ USD)",
        cmap["employment"]: "Việc làm (triệu người)",
        cmap["ai"]: "AI readiness",
        cmap["risk"]: "Rủi ro tự động hóa (%)",
    }

    st.dataframe(
        df[display_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Nếu file dữ liệu chưa có cột năng suất lao động, hệ thống sử dụng tỷ trọng GDP "
        "làm biến đại diện để trang vẫn hoạt động ổn định."
    )

    # Bộ trọng số mặc định theo đề
    default_weights = np.array(
        [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15],
        dtype=float,
    )

    default_score = _b3_priority_score(
        norm,
        cmap,
        default_weights,
    )

    default_result = pd.DataFrame(
        {
            "Ngành": df[sector_col],
            "Priority": default_score,
        }
    ).sort_values(
        "Priority",
        ascending=False,
    )

    default_result["Xếp hạng"] = np.arange(
        1,
        len(default_result) + 1,
    )

    # =====================================================
    # 3.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 3.4. Yêu cầu lập trình")

    tab341, tab342, tab343, tab344 = st.tabs(
        [
            "3.4.1 - Chuẩn hóa",
            "3.4.2 - Priority mặc định",
            "3.4.3 - Độ nhạy AI",
            "3.4.4 - Hai định hướng",
        ]
    )

    # -----------------------------------------------------
    # 3.4.1
    # -----------------------------------------------------
    with tab341:
        st.markdown(
            "### Câu 3.4.1. Chuẩn hóa min-max toàn bộ 7 tiêu chí"
        )

        normalized_table = pd.DataFrame(
            {
                "Ngành": df[sector_col],
                "Growth_norm": norm[cmap["growth"]],
                "Productivity_norm": norm[cmap["productivity"]],
                "Spillover_norm": norm[cmap["spillover"]],
                "Export_norm": norm[cmap["export"]],
                "Employment_norm": norm[cmap["employment"]],
                "AI_norm": norm[cmap["ai"]],
                "Risk_norm": norm[cmap["risk"]],
            }
        )

        format_dict = {
            column: "{:.4f}"
            for column in normalized_table.columns
            if column != "Ngành"
        }

        st.dataframe(
            normalized_table.style.format(format_dict),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Sau chuẩn hóa, giá trị nhỏ nhất của mỗi tiêu chí bằng 0 và giá trị lớn nhất bằng 1."
        )

        with st.expander("Xem mã Python cho câu 3.4.1"):
            st.code(
                """def minmax_normalize(series):
    return (series - series.min()) / (
        series.max() - series.min()
    )

normalized = df[criteria].apply(
    minmax_normalize
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 3.4.2
    # -----------------------------------------------------
    with tab342:
        st.markdown(
            "### Câu 3.4.2. Tính Priorityᵢ và xếp hạng 10 ngành"
        )

        top3_names = ", ".join(
            default_result.head(3)["Ngành"].tolist()
        )

        kpi_cards(
            [
                (
                    "Ngành xếp thứ nhất",
                    default_result.iloc[0]["Ngành"],
                    f"Priority = {default_result.iloc[0]['Priority']:.3f}",
                ),
                (
                    "Top 3",
                    top3_names,
                    "theo bộ trọng số mặc định",
                ),
                (
                    "AI readiness cao nhất",
                    df.loc[
                        df[cmap["ai"]].idxmax(),
                        sector_col,
                    ],
                    "theo dữ liệu ngành",
                ),
                (
                    "Rủi ro cao nhất",
                    df.loc[
                        df[cmap["risk"]].idxmax(),
                        sector_col,
                    ],
                    "cần ưu tiên đào tạo lại",
                ),
            ]
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.dataframe(
                default_result.style.format(
                    {"Priority": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.plotly_chart(
                plot_bar(
                    default_result,
                    "Ngành",
                    "Priority",
                    "Xếp hạng Priority theo ngành",
                    text="Priority",
                ),
                use_container_width=True,
            )

        st.caption(
            "Tổng trọng số trong đề lớn hơn 1. Điều này không làm thay đổi thứ hạng "
            "nếu tất cả trọng số được nhân hoặc chia cùng một tỷ lệ."
        )

        with st.expander("Xem mã Python cho câu 3.4.2"):
            st.code(
                """weights = np.array([
    0.15, 0.15, 0.20,
    0.15, 0.10, 0.20, 0.15
])

priority = (
    weights[0] * growth_norm
    + weights[1] * productivity_norm
    + weights[2] * spillover_norm
    + weights[3] * export_norm
    + weights[4] * employment_norm
    + weights[5] * ai_norm
    - weights[6] * risk_norm
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 3.4.3
    # -----------------------------------------------------
    with tab343:
        st.markdown(
            "### Câu 3.4.3. Phân tích độ nhạy trọng số AI readiness"
        )

        ai_weight_values = np.arange(
            0.05,
            0.401,
            0.05,
        )

        # Trọng số còn lại, trừ AI
        other_base = np.array(
            [0.15, 0.15, 0.20, 0.15, 0.10, 0.15],
            dtype=float,
        )

        sensitivity_rows = []
        top3_sets = []

        for ai_weight in ai_weight_values:
            remaining_weight = 1 - ai_weight

            scaled_other = (
                other_base
                / other_base.sum()
                * remaining_weight
            )

            weights = np.array(
                [
                    scaled_other[0],
                    scaled_other[1],
                    scaled_other[2],
                    scaled_other[3],
                    scaled_other[4],
                    ai_weight,
                    scaled_other[5],
                ]
            )

            score = _b3_priority_score(
                norm,
                cmap,
                weights,
            )

            temp_result = pd.DataFrame(
                {
                    "Ngành": df[sector_col],
                    "Điểm": score,
                }
            ).sort_values(
                "Điểm",
                ascending=False,
            )

            top3_sets.append(
                (
                    ai_weight,
                    tuple(
                        temp_result.head(3)["Ngành"]
                    ),
                )
            )

            for rank, (_, row) in enumerate(
                temp_result.iterrows(),
                start=1,
            ):
                sensitivity_rows.append(
                    [
                        ai_weight,
                        row["Ngành"],
                        rank,
                        row["Điểm"],
                    ]
                )

        sensitivity_df = pd.DataFrame(
            sensitivity_rows,
            columns=[
                "Trọng số AI",
                "Ngành",
                "Xếp hạng",
                "Điểm",
            ],
        )

        fig_line = px.line(
            sensitivity_df,
            x="Trọng số AI",
            y="Xếp hạng",
            color="Ngành",
            markers=True,
            template=PLOT_TEMPLATE,
            title="Thứ hạng ngành khi trọng số AI thay đổi",
        )

        fig_line.update_yaxes(
            autorange="reversed"
        )

        fig_line.update_layout(
            height=540,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True,
        )

        rank_pivot = sensitivity_df.pivot(
            index="Ngành",
            columns="Trọng số AI",
            values="Xếp hạng",
        )

        fig_heatmap = px.imshow(
            rank_pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            template=PLOT_TEMPLATE,
            title="Heatmap độ nhạy thứ hạng",
        )

        fig_heatmap.update_layout(
            height=560,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
        )

        unique_top3_count = len(
            set(
                top3
                for _, top3 in top3_sets
            )
        )

        st.info(
            f"Trong dải trọng số AI từ 0,05 đến 0,40 có "
            f"**{unique_top3_count} cấu hình top-3 khác nhau**. "
            "Nếu số này lớn hơn 1, thứ hạng chưa hoàn toàn ổn định."
        )

        with st.expander("Xem mã Python cho câu 3.4.3"):
            st.code(
                """for w_ai in np.arange(
    0.05, 0.401, 0.05
):
    # phân bổ lại trọng số còn lại
    # tính lại Priority
    # xếp hạng và lưu kết quả
    pass""",
                language="python",
            )

    # -----------------------------------------------------
    # 3.4.4
    # -----------------------------------------------------
    with tab344:
        st.markdown(
            "### Câu 3.4.4. So sánh hai định hướng chính sách"
        )

        growth_weights = np.array(
            [
                0.24,
                0.22,
                0.12,
                0.20,
                0.06,
                0.10,
                0.06,
            ]
        )

        inclusive_weights = np.array(
            [
                0.08,
                0.08,
                0.22,
                0.05,
                0.22,
                0.12,
                0.23,
            ]
        )

        growth_score = _b3_priority_score(
            norm,
            cmap,
            growth_weights,
        )

        inclusive_score = _b3_priority_score(
            norm,
            cmap,
            inclusive_weights,
        )

        comparison = pd.DataFrame(
            {
                "Ngành": df[sector_col],
                "Điểm tăng trưởng": growth_score,
                "Điểm bao trùm": inclusive_score,
            }
        )

        comparison["Hạng tăng trưởng"] = (
            comparison["Điểm tăng trưởng"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        comparison["Hạng bao trùm"] = (
            comparison["Điểm bao trùm"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        comparison["Thay đổi hạng"] = (
            comparison["Hạng tăng trưởng"]
            - comparison["Hạng bao trùm"]
        )

        st.dataframe(
            comparison.sort_values(
                "Hạng tăng trưởng"
            ).style.format(
                {
                    "Điểm tăng trưởng": "{:.4f}",
                    "Điểm bao trùm": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        comparison_long = comparison.melt(
            id_vars="Ngành",
            value_vars=[
                "Điểm tăng trưởng",
                "Điểm bao trùm",
            ],
            var_name="Định hướng",
            value_name="Điểm",
        )

        fig_compare = px.bar(
            comparison_long,
            x="Ngành",
            y="Điểm",
            color="Định hướng",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="So sánh định hướng tăng trưởng và bao trùm",
        )

        fig_compare.update_layout(
            height=480,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True,
        )

        growth_top3 = ", ".join(
            comparison.sort_values(
                "Hạng tăng trưởng"
            ).head(3)["Ngành"]
        )

        inclusive_top3 = ", ".join(
            comparison.sort_values(
                "Hạng bao trùm"
            ).head(3)["Ngành"]
        )

        st.success(
            f"Top-3 theo định hướng tăng trưởng: **{growth_top3}**."
        )

        st.info(
            f"Top-3 theo định hướng bao trùm: **{inclusive_top3}**."
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    st.download_button(
        "Tải kết quả Bài 3 dạng CSV",
        data=default_result.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai3_priority_10_nganh.csv",
        mime="text/csv",
        key="download_bai3",
    )

    # =====================================================
    # 3.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 3.5. Câu hỏi thảo luận chính sách"
    )

    top3_names = ", ".join(
        default_result.head(3)["Ngành"].tolist()
    )

    with st.expander(
        "a) Ba ngành nào nên được ưu tiên chuyển đổi số và AI trước?",
        expanded=True,
    ):
        st.markdown(
            f"Theo bộ trọng số mặc định, ba ngành có chỉ số Priority cao nhất là "
            f"**{top3_names}**. Đây là các ngành đạt sự kết hợp tương đối tốt giữa "
            "tăng trưởng, lan tỏa, xuất khẩu, việc làm và mức sẵn sàng AI."
        )

    with st.expander(
        "b) Vì sao Khai khoáng có năng suất cao nhưng có thể không thuộc nhóm ưu tiên?",
        expanded=True,
    ):
        st.markdown(
            "Khai khoáng có thể có năng suất lao động cao nhưng quy mô việc làm nhỏ, "
            "tốc độ tăng trưởng thấp hoặc âm, hiệu ứng lan tỏa số hạn chế và rủi ro "
            "tự động hóa tương đối lớn. Chỉ số tổng hợp vì vậy không đồng nhất năng suất "
            "cao với mức ưu tiên chính sách cao."
        )

    with st.expander(
        "c) Trọng số nên do chuyên gia, chính trị hay tham vấn công khai quyết định?",
        expanded=True,
    ):
        st.markdown(
            "Nên sử dụng quy trình kết hợp. Chuyên gia kỹ thuật xây dựng cơ sở định lượng; "
            "cơ quan hoạch định chính sách xác định mục tiêu phát triển và giới hạn ngân sách; "
            "tham vấn công khai giúp tăng tính minh bạch, trách nhiệm giải trình và tính chính danh."
        )

def _b4_solve_scipy(
    fairness=True,
    total_budget=50000.0,
    region_floor=5000.0,
    region_cap=12000.0,
    human_floor=12000.0,
    gamma=0.002,
    lam=0.70,
):
    """
    Giải LP phân bổ ngân sách số cho 6 vùng × 4 hạng mục.

    Biến:
    - 24 biến x[r,j], theo thứ tự I, D, AI, H.
    - 1 biến M khi bật công bằng vùng.

    Công bằng:
        D0[r] + gamma*x_D[r] <= M
        D0[r] + gamma*x_D[r] >= lam*M
    """
    regions, items, beta, D0 = region_beta_matrix()

    n_x = 24
    m_index = 24
    n_var = 25

    c = np.zeros(n_var, dtype=float)
    c[:n_x] = -beta.reshape(-1)

    A_ub = []
    b_ub = []

    # C1. Tổng ngân sách
    row = np.zeros(n_var, dtype=float)
    row[:n_x] = 1.0
    A_ub.append(row)
    b_ub.append(float(total_budget))

    # C2-C3. Sàn và trần từng vùng
    for r in range(6):
        row = np.zeros(n_var, dtype=float)
        row[r * 4 : r * 4 + 4] = -1.0
        A_ub.append(row)
        b_ub.append(-float(region_floor))

        row = np.zeros(n_var, dtype=float)
        row[r * 4 : r * 4 + 4] = 1.0
        A_ub.append(row)
        b_ub.append(float(region_cap))

    # C4. Sàn nhân lực số
    row = np.zeros(n_var, dtype=float)
    for r in range(6):
        row[r * 4 + 3] = -1.0
    A_ub.append(row)
    b_ub.append(-float(human_floor))

    # C5. Công bằng vùng
    if fairness:
        for r in range(6):
            row = np.zeros(n_var, dtype=float)
            row[r * 4 + 1] = float(gamma)
            row[m_index] = -1.0
            A_ub.append(row)
            b_ub.append(-float(D0[r]))

        for r in range(6):
            row = np.zeros(n_var, dtype=float)
            row[r * 4 + 1] = -float(gamma)
            row[m_index] = float(lam)
            A_ub.append(row)
            b_ub.append(float(D0[r]))

    bounds = [(0, None)] * n_var

    return linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )


def _b4_find_max_lambda(
    total_budget=50000.0,
    region_floor=5000.0,
    region_cap=12000.0,
    human_floor=12000.0,
    gamma=0.002,
    iterations=45,
):
    """
    Tìm lambda lớn nhất còn khả thi bằng tìm kiếm nhị phân.
    """
    low = 0.0
    high = 1.0

    for _ in range(int(iterations)):
        mid = (low + high) / 2.0

        result = _b4_solve_scipy(
            fairness=True,
            total_budget=total_budget,
            region_floor=region_floor,
            region_cap=region_cap,
            human_floor=human_floor,
            gamma=gamma,
            lam=mid,
        )

        if result.success:
            low = mid
        else:
            high = mid

    return low


def _b4_allocation_table(result, regions, items):
    """
    Chuyển nghiệm 24 biến thành bảng 6 vùng × 4 hạng mục.
    """
    X = np.asarray(
        result.x[:24],
        dtype=float,
    ).reshape(6, 4)

    table = pd.DataFrame(
        X,
        columns=items,
        index=regions,
    )

    table["Tổng vùng"] = table.sum(axis=1)

    return X, table


def _b4_validate_solution(
    X,
    D0,
    lam,
    gamma=0.002,
    total_budget=50000.0,
    region_floor=5000.0,
    region_cap=12000.0,
    human_floor=12000.0,
):
    """
    Kiểm tra nghiệm sau tối ưu và trả về bảng điều kiện.
    """
    region_total = X.sum(axis=1)
    digital_after = D0 + gamma * X[:, 1]
    max_digital = digital_after.max()

    checks = pd.DataFrame(
        {
            "Điều kiện": [
                "Tổng ngân sách ≤ giới hạn",
                "Mọi vùng ≥ sàn",
                "Mọi vùng ≤ trần",
                "Tổng nhân lực số ≥ sàn",
                "Công bằng vùng",
                "Không âm",
            ],
            "Giá trị kiểm tra": [
                X.sum(),
                region_total.min(),
                region_total.max(),
                X[:, 3].sum(),
                (digital_after / max_digital).min(),
                X.min(),
            ],
            "Ngưỡng": [
                total_budget,
                region_floor,
                region_cap,
                human_floor,
                lam,
                0.0,
            ],
            "Đạt": [
                X.sum() <= total_budget + 1e-6,
                region_total.min() >= region_floor - 1e-6,
                region_total.max() <= region_cap + 1e-6,
                X[:, 3].sum() >= human_floor - 1e-6,
                (digital_after / max_digital).min() >= lam - 1e-6,
                X.min() >= -1e-6,
            ],
        }
    )

    return checks, digital_after


def page_4():
    hero(
        "Bài 4 — LP phân bổ ngân sách số theo vùng",
        "Giải đúng mô hình 24 biến, kiểm tra tính khả thi của λ=0,70, so sánh SciPy–PuLP–CVXPY và lượng hóa chi phí công bằng, chi phí trần vùng.",
        ["4.1-4.5", "LP", "Feasibility", "PuLP", "CVXPY", "Fairness"],
    )

    regions, items, beta, D0 = region_beta_matrix()

    total_budget = 50000.0
    region_floor = 5000.0
    region_cap = 12000.0
    human_floor = 12000.0
    gamma = 0.002
    lambda_original = 0.70

    # =====================================================
    # 4.1. Bối cảnh
    # =====================================================
    st.markdown("## 4.1. Bối cảnh Việt Nam")

    st.markdown(
        """
        Chính phủ phân bổ **50.000 tỷ VND** cho 6 vùng và 4 hạng mục:
        hạ tầng số (I), chuyển đổi số doanh nghiệp (D), AI và nhân lực số (H).

        Mục tiêu là tối đa hóa GDP gain kỳ vọng, đồng thời kiểm soát tập trung ngân sách,
        bảo đảm đầu tư nhân lực và thu hẹp chênh lệch Digital Index giữa các vùng.
        """
    )

    # =====================================================
    # 4.2. Mô hình
    # =====================================================
    st.markdown("## 4.2. Mô hình toán học")

    st.latex(
        r"\max Z=\sum_{r=1}^{6}\sum_{j\in\{I,D,AI,H\}}\beta_{j,r}x_{j,r}"
    )
    st.latex(
        r"\sum_r\sum_jx_{j,r}\leq 50{,}000"
    )
    st.latex(
        r"5{,}000\leq\sum_jx_{j,r}\leq12{,}000,\quad\forall r"
    )
    st.latex(
        r"\sum_rx_{H,r}\geq12{,}000"
    )
    st.latex(
        r"D_r+\gamma x_{D,r}\geq\lambda M,\quad"
        r"D_r+\gamma x_{D,r}\leq M"
    )
    st.latex(
        r"\gamma=0.002,\quad\lambda=0.70,\quad x_{j,r}\geq0"
    )

    # =====================================================
    # 4.3. Dữ liệu
    # =====================================================
    st.markdown("## 4.3. Hệ số tác động và Digital Index ban đầu")

    parameter_table = pd.DataFrame(
        beta,
        columns=items,
    )
    parameter_table.insert(0, "Vùng", regions)
    parameter_table["Digital Index ban đầu"] = D0

    st.dataframe(
        parameter_table,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # Chẩn đoán tính khả thi λ = 0,70
    # =====================================================
    st.markdown("## 4.4. Yêu cầu lập trình")

    original_result = _b4_solve_scipy(
        fairness=True,
        total_budget=total_budget,
        region_floor=region_floor,
        region_cap=region_cap,
        human_floor=human_floor,
        gamma=gamma,
        lam=lambda_original,
    )

    lambda_max = _b4_find_max_lambda(
        total_budget=total_budget,
        region_floor=region_floor,
        region_cap=region_cap,
        human_floor=human_floor,
        gamma=gamma,
    )

    M_min = float(D0.max())
    required_index = lambda_original * M_min
    required_x_d = np.maximum(
        0.0,
        (required_index - D0) / gamma,
    )

    feasibility_table = pd.DataFrame(
        {
            "Vùng": regions,
            "D₀": D0,
            "D tối thiểu khi λ=0,70": required_index,
            "x_D tối thiểu": required_x_d,
            "Trần ngân sách vùng": region_cap,
            "Vượt trần": required_x_d > region_cap + 1e-9,
        }
    )

    if original_result.success:
        st.success(
            "Mô hình gốc với λ=0,70 khả thi."
        )
    else:
        st.error(
            "Mô hình gốc với λ=0,70 không khả thi. Đây là kết quả toán học, "
            "không phải lỗi Streamlit hoặc lỗi solver."
        )

        st.dataframe(
            feasibility_table.style.format(
                {
                    "D₀": "{:.2f}",
                    "D tối thiểu khi λ=0,70": "{:.2f}",
                    "x_D tối thiểu": "{:,.0f}",
                    "Trần ngân sách vùng": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        tay_nguyen_required = float(required_x_d[3])
        extra_cap = max(0.0, tay_nguyen_required - region_cap)
        gamma_required = (
            (required_index - D0[3]) / region_cap
        )

        st.warning(
            f"Tây Nguyên cần x_D ≈ **{tay_nguyen_required:,.0f}** tỷ VND, "
            f"nhưng trần vùng chỉ **{region_cap:,.0f}** tỷ VND. "
            f"Thiếu ít nhất **{extra_cap:,.0f}** tỷ VND. "
            f"Với trần 12.000, γ phải tăng từ 0,002 lên khoảng "
            f"**{gamma_required:.6f}**, hoặc λ phải giảm."
        )

    kpi_cards(
        [
            (
                "λ theo đề",
                f"{lambda_original:.3f}",
                "không khả thi với dữ liệu gốc",
            ),
            (
                "λ lớn nhất khả thi",
                f"{lambda_max:.4f}",
                "tìm bằng LP + binary search",
            ),
            (
                "Ngưỡng trực tiếp",
                f"{56/82:.4f}",
                "Tây Nguyên tối đa 56 / mức cao nhất 82",
            ),
            (
                "Chênh lệch λ",
                f"{lambda_original-lambda_max:.4f}",
                "mức cần điều chỉnh",
            ),
        ]
    )

    # Chọn lambda để tiếp tục sinh nghiệm và hoàn thành các câu còn lại.
    default_lambda = min(0.68, float(lambda_max) - 1e-5)

    lambda_used = st.slider(
        "λ dùng để sinh nghiệm khả thi và so sánh solver",
        min_value=0.50,
        max_value=float(round(lambda_max, 4)),
        value=float(round(default_lambda, 4)),
        step=0.001,
        key="b4_lambda_used",
    )

    fair_result = _b4_solve_scipy(
        fairness=True,
        total_budget=total_budget,
        region_floor=region_floor,
        region_cap=region_cap,
        human_floor=human_floor,
        gamma=gamma,
        lam=lambda_used,
    )

    if not fair_result.success:
        st.error(
            "λ đang chọn vẫn không khả thi do sai số làm tròn. "
            "Hãy giảm λ xuống 0,001."
        )
        return

    X_fair, allocation_fair = _b4_allocation_table(
        fair_result,
        regions,
        items,
    )
    z_fair = -float(fair_result.fun)

    nofair_result = _b4_solve_scipy(
        fairness=False,
        total_budget=total_budget,
        region_floor=region_floor,
        region_cap=region_cap,
        human_floor=human_floor,
        gamma=gamma,
        lam=lambda_used,
    )

    X_nofair, allocation_nofair = _b4_allocation_table(
        nofair_result,
        regions,
        items,
    )
    z_nofair = -float(nofair_result.fun)

    no_cap_result = _b4_solve_scipy(
        fairness=True,
        total_budget=total_budget,
        region_floor=region_floor,
        region_cap=total_budget,
        human_floor=human_floor,
        gamma=gamma,
        lam=lambda_used,
    )

    X_no_cap, allocation_no_cap = _b4_allocation_table(
        no_cap_result,
        regions,
        items,
    )
    z_no_cap = -float(no_cap_result.fun)

    tab441, tab442, tab443, tab444 = st.tabs(
        [
            "4.4.1 - SciPy & PuLP",
            "4.4.2 - CVXPY",
            "4.4.3 - Heatmap & kiểm định",
            "4.4.4 - Chi phí công bằng",
        ]
    )

    # -----------------------------------------------------
    # 4.4.1
    # -----------------------------------------------------
    with tab441:
        st.markdown("### Câu 4.4.1. Giải mô hình bằng SciPy và PuLP")

        kpi_cards(
            [
                (
                    "Z* SciPy",
                    f"{z_fair:,.2f}",
                    f"λ={lambda_used:.3f}",
                ),
                (
                    "Tổng ngân sách",
                    f"{X_fair.sum():,.0f}",
                    "tỷ VND",
                ),
                (
                    "Tổng nhân lực số",
                    f"{X_fair[:, 3].sum():,.0f}",
                    "tỷ VND",
                ),
                (
                    "Digital ratio thấp nhất",
                    f"{((D0+gamma*X_fair[:,1])/(D0+gamma*X_fair[:,1]).max()).min():.4f}",
                    f"phải ≥ {lambda_used:.3f}",
                ),
            ]
        )

        st.dataframe(
            allocation_fair.reset_index().rename(
                columns={"index": "Vùng"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        try:
            import pulp

            model = pulp.LpProblem(
                "VN_Regional_Digital_Budget",
                pulp.LpMaximize,
            )

            x = pulp.LpVariable.dicts(
                "x",
                (range(6), range(4)),
                lowBound=0,
            )
            M = pulp.LpVariable(
                "M",
                lowBound=0,
            )

            model += pulp.lpSum(
                beta[r, j] * x[r][j]
                for r in range(6)
                for j in range(4)
            )

            model += pulp.lpSum(
                x[r][j]
                for r in range(6)
                for j in range(4)
            ) <= total_budget

            for r in range(6):
                model += pulp.lpSum(
                    x[r][j] for j in range(4)
                ) >= region_floor

                model += pulp.lpSum(
                    x[r][j] for j in range(4)
                ) <= region_cap

            model += pulp.lpSum(
                x[r][3] for r in range(6)
            ) >= human_floor

            for r in range(6):
                model += (
                    D0[r] + gamma * x[r][1] <= M
                )
                model += (
                    D0[r] + gamma * x[r][1]
                    >= lambda_used * M
                )

            model.solve(
                pulp.PULP_CBC_CMD(msg=False)
            )

            pulp_status = pulp.LpStatus[
                model.status
            ]

            if pulp_status == "Optimal":
                X_pulp = np.array(
                    [
                        [
                            x[r][j].value()
                            for j in range(4)
                        ]
                        for r in range(6)
                    ],
                    dtype=float,
                )

                z_pulp = float(
                    pulp.value(model.objective)
                )

                comparison = pd.DataFrame(
                    {
                        "Chỉ tiêu": [
                            "Z*",
                            "Tổng ngân sách",
                            "Sai lệch phân bổ lớn nhất",
                        ],
                        "SciPy": [
                            z_fair,
                            X_fair.sum(),
                            0.0,
                        ],
                        "PuLP": [
                            z_pulp,
                            X_pulp.sum(),
                            np.max(
                                np.abs(
                                    X_pulp - X_fair
                                )
                            ),
                        ],
                    }
                )

                st.markdown("#### Đối chiếu SciPy và PuLP")
                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    "Nếu Z* giống nhau nhưng phân bổ khác nhẹ, bài toán có thể có nhiều nghiệm tối ưu."
                )
            else:
                st.warning(
                    f"PuLP trả về trạng thái: {pulp_status}."
                )

        except ModuleNotFoundError:
            st.warning(
                "Chưa cài PuLP. Thêm `pulp>=2.7` vào requirements.txt."
            )

    # -----------------------------------------------------
    # 4.4.2
    # -----------------------------------------------------
    with tab442:
        st.markdown("### Câu 4.4.2. Giải lại bằng CVXPY")

        try:
            import cvxpy as cp

            X = cp.Variable(
                (6, 4),
                nonneg=True,
            )
            M = cp.Variable(
                nonneg=True,
            )

            constraints = [
                cp.sum(X) <= total_budget,
                cp.sum(X[:, 3]) >= human_floor,
            ]

            for r in range(6):
                constraints.extend(
                    [
                        cp.sum(X[r, :]) >= region_floor,
                        cp.sum(X[r, :]) <= region_cap,
                        D0[r] + gamma * X[r, 1] <= M,
                        D0[r] + gamma * X[r, 1]
                        >= lambda_used * M,
                    ]
                )

            problem = cp.Problem(
                cp.Maximize(
                    cp.sum(
                        cp.multiply(
                            beta,
                            X,
                        )
                    )
                ),
                constraints,
            )

            installed = cp.installed_solvers()

            if "CLARABEL" in installed:
                problem.solve(
                    solver=cp.CLARABEL,
                    verbose=False,
                )
            elif "SCS" in installed:
                problem.solve(
                    solver=cp.SCS,
                    verbose=False,
                )
            else:
                problem.solve(
                    verbose=False,
                )

            if X.value is None:
                st.error(
                    "CVXPY không trả về nghiệm."
                )
            else:
                X_cvx = np.asarray(
                    X.value,
                    dtype=float,
                )
                z_cvx = float(
                    problem.value
                )

                comparison = pd.DataFrame(
                    {
                        "Chỉ tiêu": [
                            "Z*",
                            "Tổng ngân sách",
                            "Sai lệch phân bổ lớn nhất",
                        ],
                        "SciPy": [
                            z_fair,
                            X_fair.sum(),
                            0.0,
                        ],
                        "CVXPY": [
                            z_cvx,
                            X_cvx.sum(),
                            np.max(
                                np.abs(
                                    X_cvx - X_fair
                                )
                            ),
                        ],
                    }
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

                st.success(
                    f"CVXPY status: {problem.status}."
                )

        except ModuleNotFoundError:
            st.warning(
                "Chưa cài CVXPY. Thêm `cvxpy>=1.4` vào requirements.txt."
            )
        except Exception as exc:
            st.warning(
                f"CVXPY gặp lỗi: {exc}"
            )

    # -----------------------------------------------------
    # 4.4.3
    # -----------------------------------------------------
    with tab443:
        st.markdown("### Câu 4.4.3. Heatmap và kiểm tra ràng buộc")

        heatmap_df = pd.DataFrame(
            X_fair,
            columns=items,
            index=regions,
        )

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdPu",
            template=PLOT_TEMPLATE,
            title="Phân bổ tối ưu 6 vùng × 4 hạng mục",
        )
        fig_heatmap.update_layout(
            height=560,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
        )

        checks, digital_after = _b4_validate_solution(
            X_fair,
            D0,
            lam=lambda_used,
            gamma=gamma,
            total_budget=total_budget,
            region_floor=region_floor,
            region_cap=region_cap,
            human_floor=human_floor,
        )

        st.markdown("#### Kiểm tra nghiệm sau tối ưu")
        st.dataframe(
            checks,
            use_container_width=True,
            hide_index=True,
        )

        digital_table = pd.DataFrame(
            {
                "Vùng": regions,
                "D ban đầu": D0,
                "Đầu tư D": X_fair[:, 1],
                "D sau đầu tư": digital_after,
                "Tỷ lệ so với mức cao nhất": (
                    digital_after
                    / digital_after.max()
                ),
            }
        )

        st.dataframe(
            digital_table.style.format(
                {
                    "D ban đầu": "{:.2f}",
                    "Đầu tư D": "{:,.0f}",
                    "D sau đầu tư": "{:.2f}",
                    "Tỷ lệ so với mức cao nhất": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if bool(checks["Đạt"].all()):
            st.success(
                "Nghiệm vượt qua toàn bộ kiểm tra ràng buộc."
            )
        else:
            st.error(
                "Có ít nhất một ràng buộc chưa đạt."
            )

    # -----------------------------------------------------
    # 4.4.4
    # -----------------------------------------------------
    with tab444:
        st.markdown("### Câu 4.4.4. Chi phí kinh tế của công bằng vùng")

        fairness_cost = z_nofair - z_fair
        fairness_rate = (
            100 * fairness_cost / z_nofair
            if abs(z_nofair) > 1e-12
            else 0.0
        )

        cap_cost = z_no_cap - z_fair
        cap_rate = (
            100 * cap_cost / z_no_cap
            if abs(z_no_cap) > 1e-12
            else 0.0
        )

        kpi_cards(
            [
                (
                    "Z* có công bằng",
                    f"{z_fair:,.2f}",
                    f"λ={lambda_used:.3f}",
                ),
                (
                    "Z* bỏ C5",
                    f"{z_nofair:,.2f}",
                    "không công bằng",
                ),
                (
                    "Chi phí công bằng",
                    f"{fairness_cost:,.2f}",
                    f"{fairness_rate:.3f}% Z*",
                ),
                (
                    "Chi phí trần vùng C3",
                    f"{cap_cost:,.2f}",
                    f"{cap_rate:.3f}% Z*",
                ),
            ]
        )

        region_compare = pd.DataFrame(
            {
                "Vùng": regions,
                "Có công bằng": X_fair.sum(axis=1),
                "Bỏ C5": X_nofair.sum(axis=1),
                "Bỏ trần C3": X_no_cap.sum(axis=1),
            }
        )

        st.dataframe(
            region_compare,
            use_container_width=True,
            hide_index=True,
        )

        compare_long = region_compare.melt(
            id_vars="Vùng",
            var_name="Mô hình",
            value_name="Ngân sách",
        )

        fig_compare = px.bar(
            compare_long,
            x="Vùng",
            y="Ngân sách",
            color="Mô hình",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="So sánh ngân sách vùng giữa ba mô hình",
        )
        fig_compare.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True,
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_df = allocation_fair.reset_index().rename(
        columns={"index": "Vùng"}
    )

    st.download_button(
        "Tải kết quả Bài 4 dạng CSV",
        data=export_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="bai4_lp_vung_ket_qua.csv",
        mime="text/csv",
        key="download_bai4_fixed",
    )

    # =====================================================
    # 4.5. Thảo luận chính sách
    # =====================================================
    st.markdown("## 4.5. Câu hỏi thảo luận chính sách")

    region_totals_nofair = X_nofair.sum(axis=1)
    dominant_region = regions[
        int(np.argmax(region_totals_nofair))
    ]

    with st.expander(
        "a) Nếu bỏ công bằng, vốn chảy về đâu và hậu quả là gì?",
        expanded=True,
    ):
        st.markdown(
            f"Khi bỏ C5, vùng nhận ngân sách lớn nhất là **{dominant_region}**. "
            "Nguyên nhân là vùng này có tổ hợp hệ số tác động cao, đặc biệt ở D và AI. "
            "Tuy nhiên, tập trung kéo dài có thể mở rộng khoảng cách số, làm giảm cơ hội "
            "tiếp cận dịch vụ số và năng lực hấp thụ công nghệ của các vùng yếu hơn."
        )

    with st.expander(
        "b) Trần vùng C3 làm giảm Z* bao nhiêu?",
        expanded=True,
    ):
        st.markdown(
            f"So với mô hình bỏ trần vùng, C3 làm Z* giảm khoảng "
            f"**{cap_cost:,.2f}**, tương đương **{cap_rate:.3f}%**. "
            "Đây có thể được xem là chi phí hiệu quả của phân quyền và chống tập trung. "
            "Mức giảm có chấp nhận được hay không phụ thuộc vào mục tiêu cân bằng vùng."
        )

    with st.expander(
        "c) Tây Nguyên nên ưu tiên AI hay H và I?",
        expanded=True,
    ):
        tay_nguyen = pd.Series(
            X_fair[3],
            index=items,
        ).sort_values(
            ascending=False
        )

        st.markdown(
            f"Trong nghiệm đang chọn, hạng mục lớn nhất tại Tây Nguyên là "
            f"**{tay_nguyen.index[0]}** với khoảng **{tay_nguyen.iloc[0]:,.0f} tỷ VND**. "
            "Do hệ số AI của Tây Nguyên thấp nhưng hệ số H và I cao, mô hình thường ưu tiên "
            "nhân lực, hạ tầng và khoản D cần thiết để đáp ứng công bằng trước khi mở rộng AI."
        )

    with st.expander(
        "d) Vì sao không được tự ý đổi λ=0,70 thành 0,68 mà không giải thích?",
        expanded=True,
    ):
        st.markdown(
            f"λ=0,70 là tham số của đề nhưng không khả thi với bộ dữ liệu và trần 12.000. "
            f"Do đó, bài làm tốt phải **báo cáo bất khả thi**, chỉ ra nguyên nhân, rồi mới "
            f"thực hiện phân tích độ nhạy. Giá trị lớn nhất khả thi hiện tại là khoảng "
            f"**{lambda_max:.4f}**; λ={lambda_used:.3f} chỉ là kịch bản hiệu chỉnh để "
            "tiếp tục so sánh solver và phân tích chính sách."
        )


def _b5_project_table():
    """Trả về dữ liệu 15 dự án chuyển đổi số cấp quốc gia."""
    rows = [
        {
            "Mã": "P01",
            "Dự án": "Trung tâm dữ liệu quốc gia",
            "Trụ cột": "Hạ tầng",
            "Chi phí": 18000.0,
            "Lợi ích": 28000.0,
            "Rủi ro": 0.12,
            "Tiên quyết": "",
            "Bao trùm": 0,
        },
        {
            "Mã": "P02",
            "Dự án": "Nền tảng điện toán đám mây Chính phủ",
            "Trụ cột": "Hạ tầng",
            "Chi phí": 12000.0,
            "Lợi ích": 19500.0,
            "Rủi ro": 0.10,
            "Tiên quyết": "P01",
            "Bao trùm": 0,
        },
        {
            "Mã": "P03",
            "Dự án": "Hạ tầng 5G và băng rộng vùng khó khăn",
            "Trụ cột": "Hạ tầng",
            "Chi phí": 15000.0,
            "Lợi ích": 22500.0,
            "Rủi ro": 0.15,
            "Tiên quyết": "",
            "Bao trùm": 1,
        },
        {
            "Mã": "P04",
            "Dự án": "Nền tảng chuyển đổi số cho SME",
            "Trụ cột": "Ứng dụng",
            "Chi phí": 9000.0,
            "Lợi ích": 17000.0,
            "Rủi ro": 0.18,
            "Tiên quyết": "",
            "Bao trùm": 1,
        },
        {
            "Mã": "P05",
            "Dự án": "Trung tâm tính toán AI quốc gia",
            "Trụ cột": "AI",
            "Chi phí": 20000.0,
            "Lợi ích": 34500.0,
            "Rủi ro": 0.28,
            "Tiên quyết": "P01",
            "Bao trùm": 0,
        },
        {
            "Mã": "P06",
            "Dự án": "Mô hình ngôn ngữ tiếng Việt",
            "Trụ cột": "AI",
            "Chi phí": 8000.0,
            "Lợi ích": 16000.0,
            "Rủi ro": 0.25,
            "Tiên quyết": "P05",
            "Bao trùm": 0,
        },
        {
            "Mã": "P07",
            "Dự án": "Trung tâm điều hành an ninh mạng quốc gia",
            "Trụ cột": "An toàn số",
            "Chi phí": 7000.0,
            "Lợi ích": 13500.0,
            "Rủi ro": 0.08,
            "Tiên quyết": "",
            "Bao trùm": 0,
        },
        {
            "Mã": "P08",
            "Dự án": "Đào tạo kỹ năng số cho một triệu lao động",
            "Trụ cột": "Nhân lực",
            "Chi phí": 11000.0,
            "Lợi ích": 20500.0,
            "Rủi ro": 0.10,
            "Tiên quyết": "",
            "Bao trùm": 1,
        },
        {
            "Mã": "P09",
            "Dự án": "Chương trình nhân lực bán dẫn và AI",
            "Trụ cột": "Nhân lực",
            "Chi phí": 10000.0,
            "Lợi ích": 21000.0,
            "Rủi ro": 0.17,
            "Tiên quyết": "",
            "Bao trùm": 0,
        },
        {
            "Mã": "P10",
            "Dự án": "Nông nghiệp thông minh và truy xuất nguồn gốc",
            "Trụ cột": "Ứng dụng",
            "Chi phí": 6000.0,
            "Lợi ích": 12500.0,
            "Rủi ro": 0.20,
            "Tiên quyết": "P03",
            "Bao trùm": 1,
        },
        {
            "Mã": "P11",
            "Dự án": "Logistics và cảng biển thông minh",
            "Trụ cột": "Ứng dụng",
            "Chi phí": 8500.0,
            "Lợi ích": 15750.0,
            "Rủi ro": 0.16,
            "Tiên quyết": "P02",
            "Bao trùm": 0,
        },
        {
            "Mã": "P12",
            "Dự án": "Y tế số và hồ sơ sức khỏe liên thông",
            "Trụ cột": "Ứng dụng",
            "Chi phí": 9500.0,
            "Lợi ích": 17750.0,
            "Rủi ro": 0.14,
            "Tiên quyết": "P02",
            "Bao trùm": 1,
        },
        {
            "Mã": "P13",
            "Dự án": "Trung tâm dữ liệu xanh",
            "Trụ cột": "Hạ tầng",
            "Chi phí": 6500.0,
            "Lợi ích": 10800.0,
            "Rủi ro": 0.09,
            "Tiên quyết": "P01",
            "Bao trùm": 0,
        },
        {
            "Mã": "P14",
            "Dự án": "Nền tảng dữ liệu mở quốc gia",
            "Trụ cột": "Quản trị số",
            "Chi phí": 5500.0,
            "Lợi ích": 12800.0,
            "Rủi ro": 0.11,
            "Tiên quyết": "P02",
            "Bao trùm": 1,
        },
        {
            "Mã": "P15",
            "Dự án": "Định danh số và xác thực điện tử 2.0",
            "Trụ cột": "Quản trị số",
            "Chi phí": 7500.0,
            "Lợi ích": 15200.0,
            "Rủi ro": 0.07,
            "Tiên quyết": "",
            "Bao trùm": 1,
        },
    ]

    df = pd.DataFrame(rows)
    df["Lợi ích kỳ vọng"] = df["Lợi ích"] * (1.0 - df["Rủi ro"])
    df["B/C gộp"] = df["Lợi ích"] / df["Chi phí"]
    df["B/C kỳ vọng"] = df["Lợi ích kỳ vọng"] / df["Chi phí"]
    return df


def _b5_check_vector(selection, df, budget, max_projects=10):
    """Kiểm tra toàn bộ ràng buộc với một vector nhị phân."""
    selection = np.asarray(selection, dtype=int)
    code_to_index = {
        code: i
        for i, code in enumerate(df["Mã"].tolist())
    }

    total_cost = float(
        np.dot(selection, df["Chi phí"].to_numpy(dtype=float))
    )
    selected_count = int(selection.sum())

    checks = []

    checks.append(total_cost <= budget + 1e-7)
    checks.append(selected_count <= int(max_projects))

    # P07 là lớp an toàn số bắt buộc.
    checks.append(selection[code_to_index["P07"]] == 1)

    # Ít nhất một dự án nhân lực.
    human_mask = (
        df["Trụ cột"].to_numpy() == "Nhân lực"
    ).astype(int)
    checks.append(int(np.dot(selection, human_mask)) >= 1)

    # Ít nhất hai dự án có tính bao trùm.
    inclusive = df["Bao trùm"].to_numpy(dtype=int)
    checks.append(int(np.dot(selection, inclusive)) >= 2)

    # Không quá bốn dự án hạ tầng.
    infrastructure = (
        df["Trụ cột"].to_numpy() == "Hạ tầng"
    ).astype(int)
    checks.append(int(np.dot(selection, infrastructure)) <= 4)

    # Quan hệ tiên quyết.
    for i, prerequisite in enumerate(df["Tiên quyết"].tolist()):
        if prerequisite:
            j = code_to_index[prerequisite]
            checks.append(selection[i] <= selection[j])

    # Nếu chọn trung tâm AI P05 thì phải có ít nhất một dự án nhân lực.
    checks.append(
        selection[code_to_index["P05"]]
        <= int(np.dot(selection, human_mask))
    )

    return bool(all(checks))


def _b5_solve_enumeration(
    budget=80000.0,
    risk_adjusted=False,
    max_projects=10,
):
    """Fallback vét cạn cho 15 dự án, tối đa 2^15 tổ hợp."""
    df = _b5_project_table()

    objective_values = (
        df["Lợi ích kỳ vọng"].to_numpy(dtype=float)
        if risk_adjusted
        else df["Lợi ích"].to_numpy(dtype=float)
    )

    best_objective = -np.inf
    best_selection = None

    for mask in range(1 << len(df)):
        selection = np.array(
            [(mask >> i) & 1 for i in range(len(df))],
            dtype=int,
        )

        if not _b5_check_vector(
            selection,
            df,
            budget=budget,
            max_projects=max_projects,
        ):
            continue

        objective = float(
            np.dot(selection, objective_values)
        )

        if objective > best_objective + 1e-9:
            best_objective = objective
            best_selection = selection.copy()

    if best_selection is None:
        return {
            "success": False,
            "status": "Infeasible",
            "solver": "Enumeration",
            "selection": None,
            "objective": np.nan,
        }

    return {
        "success": True,
        "status": "Optimal",
        "solver": "Enumeration fallback",
        "selection": best_selection,
        "objective": float(best_objective),
    }


def _b5_solve_mip(
    budget=80000.0,
    risk_adjusted=False,
    max_projects=10,
):
    """Giải MIP bằng PuLP/CBC; tự dùng fallback nếu CBC không chạy."""
    df = _b5_project_table()

    try:
        import pulp
    except ModuleNotFoundError:
        return _b5_solve_enumeration(
            budget=budget,
            risk_adjusted=risk_adjusted,
            max_projects=max_projects,
        )

    objective_column = (
        "Lợi ích kỳ vọng"
        if risk_adjusted
        else "Lợi ích"
    )

    model = pulp.LpProblem(
        "Vietnam_National_Digital_Projects",
        pulp.LpMaximize,
    )

    codes = df["Mã"].tolist()
    y = {
        code: pulp.LpVariable(
            f"select_{code}",
            cat="Binary",
        )
        for code in codes
    }

    model += pulp.lpSum(
        float(df.loc[i, objective_column]) * y[codes[i]]
        for i in range(len(df))
    ), "Total_Benefit"

    model += pulp.lpSum(
        float(df.loc[i, "Chi phí"]) * y[codes[i]]
        for i in range(len(df))
    ) <= float(budget), "Budget"

    model += pulp.lpSum(
        y[code] for code in codes
    ) <= int(max_projects), "Implementation_Capacity"

    # An toàn số bắt buộc.
    model += y["P07"] == 1, "Mandatory_Cybersecurity"

    # Ít nhất một dự án nhân lực.
    human_codes = df.loc[
        df["Trụ cột"] == "Nhân lực",
        "Mã",
    ].tolist()
    model += pulp.lpSum(
        y[code] for code in human_codes
    ) >= 1, "Human_Capital_Minimum"

    # Ít nhất hai dự án bao trùm.
    inclusive_codes = df.loc[
        df["Bao trùm"] == 1,
        "Mã",
    ].tolist()
    model += pulp.lpSum(
        y[code] for code in inclusive_codes
    ) >= 2, "Inclusion_Minimum"

    # Không quá bốn dự án hạ tầng.
    infrastructure_codes = df.loc[
        df["Trụ cột"] == "Hạ tầng",
        "Mã",
    ].tolist()
    model += pulp.lpSum(
        y[code] for code in infrastructure_codes
    ) <= 4, "Infrastructure_Cap"

    # Quan hệ tiên quyết.
    for _, row in df.iterrows():
        if row["Tiên quyết"]:
            model += (
                y[row["Mã"]]
                <= y[row["Tiên quyết"]]
            ), f"Prerequisite_{row['Mã']}"

    # Nếu P05 được chọn thì phải có ít nhất một dự án nhân lực.
    model += (
        y["P05"]
        <= pulp.lpSum(y[code] for code in human_codes)
    ), "AI_Requires_Human_Capital"

    try:
        model.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                timeLimit=30,
            )
        )
    except Exception:
        return _b5_solve_enumeration(
            budget=budget,
            risk_adjusted=risk_adjusted,
            max_projects=max_projects,
        )

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "solver": "PuLP/CBC",
            "selection": None,
            "objective": np.nan,
        }

    selection = np.array(
        [
            int(round(y[code].value() or 0))
            for code in codes
        ],
        dtype=int,
    )

    return {
        "success": True,
        "status": status,
        "solver": "PuLP/CBC",
        "selection": selection,
        "objective": float(
            pulp.value(model.objective)
        ),
    }


def _b5_result_table(result, budget):
    """Tạo bảng đầu ra và KPI của một nghiệm."""
    df = _b5_project_table().copy()

    if not result["success"]:
        return df.iloc[0:0].copy(), {}

    df["Chọn"] = result["selection"]
    selected = df[df["Chọn"] == 1].copy()

    total_cost = float(selected["Chi phí"].sum())
    gross_benefit = float(selected["Lợi ích"].sum())
    expected_benefit = float(selected["Lợi ích kỳ vọng"].sum())

    metrics = {
        "solver": result["solver"],
        "status": result["status"],
        "n_projects": int(len(selected)),
        "total_cost": total_cost,
        "unused_budget": float(budget - total_cost),
        "gross_benefit": gross_benefit,
        "expected_benefit": expected_benefit,
        "portfolio_risk": (
            float(
                np.average(
                    selected["Rủi ro"],
                    weights=selected["Chi phí"],
                )
            )
            if total_cost > 0
            else 0.0
        ),
    }

    return selected, metrics


def _b5_validation_table(result, budget, max_projects=10):
    """Lập bảng xác nhận các ràng buộc chính."""
    df = _b5_project_table()
    selection = result["selection"]

    if selection is None:
        return pd.DataFrame(
            {
                "Ràng buộc": ["Mô hình khả thi"],
                "Giá trị": [result["status"]],
                "Đạt": [False],
            }
        )

    selected_df = df[selection == 1]
    selected_codes = set(selected_df["Mã"].tolist())

    prerequisite_ok = True
    for _, row in selected_df.iterrows():
        if row["Tiên quyết"] and row["Tiên quyết"] not in selected_codes:
            prerequisite_ok = False

    checks = [
        {
            "Ràng buộc": "Tổng chi phí ≤ ngân sách",
            "Giá trị": f"{selected_df['Chi phí'].sum():,.0f} / {budget:,.0f}",
            "Đạt": selected_df["Chi phí"].sum() <= budget + 1e-7,
        },
        {
            "Ràng buộc": "Số dự án ≤ năng lực triển khai",
            "Giá trị": f"{len(selected_df)} / {max_projects}",
            "Đạt": len(selected_df) <= max_projects,
        },
        {
            "Ràng buộc": "P07 an toàn số bắt buộc",
            "Giá trị": "P07" in selected_codes,
            "Đạt": "P07" in selected_codes,
        },
        {
            "Ràng buộc": "Có ít nhất một dự án nhân lực",
            "Giá trị": int(
                (selected_df["Trụ cột"] == "Nhân lực").sum()
            ),
            "Đạt": (
                selected_df["Trụ cột"] == "Nhân lực"
            ).sum() >= 1,
        },
        {
            "Ràng buộc": "Có ít nhất hai dự án bao trùm",
            "Giá trị": int(selected_df["Bao trùm"].sum()),
            "Đạt": selected_df["Bao trùm"].sum() >= 2,
        },
        {
            "Ràng buộc": "Tất cả quan hệ tiên quyết được thỏa",
            "Giá trị": prerequisite_ok,
            "Đạt": prerequisite_ok,
        },
    ]

    return pd.DataFrame(checks)


def page_5():
    hero(
        "Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số quốc gia",
        "Giải mô hình nhị phân bằng PuLP/CBC, kiểm tra quan hệ tiên quyết, so sánh ngân sách 80.000–100.000 tỷ VND và tối ưu lợi ích kỳ vọng có rủi ro.",
        ["5.1-5.5", "MIP", "PuLP/CBC", "Knapsack", "Risk-adjusted"],
    )

    df = _b5_project_table()

    # =====================================================
    # 5.1. Bối cảnh
    # =====================================================
    st.markdown("## 5.1. Bối cảnh Việt Nam")

    st.markdown(
        """
        Chính phủ phải lựa chọn một danh mục từ **15 dự án chuyển đổi số** trong điều kiện
        ngân sách và năng lực triển khai có hạn. Các dự án không độc lập hoàn toàn:
        một số dự án AI, dữ liệu và ứng dụng chỉ có thể triển khai sau khi hạ tầng nền tảng
        tương ứng đã được lựa chọn.

        Mô hình cần đồng thời bảo đảm an toàn số, nhân lực và tính bao trùm, thay vì chỉ chọn
        các dự án có lợi ích gộp cao nhất.
        """
    )

    # =====================================================
    # 5.2. Mô hình toán học
    # =====================================================
    st.markdown("## 5.2. Mô hình toán học")

    st.latex(
        r"y_i="
        r"\begin{cases}"
        r"1,&\text{nếu chọn dự án }i\\"
        r"0,&\text{ngược lại}"
        r"\end{cases}"
    )

    st.latex(
        r"\max Z=\sum_{i=1}^{15}B_i y_i"
    )

    st.latex(
        r"\sum_{i=1}^{15}C_i y_i\leq Budget"
    )

    st.latex(
        r"y_i\leq y_j"
        r"\quad\text{khi dự án }i\text{ cần dự án }j"
    )

    st.latex(
        r"\sum_i y_i\leq10,\quad y_i\in\{0,1\}"
    )

    st.markdown(
        "Trong mô hình rủi ro, lợi ích được thay bằng "
        r"$E(B_i)=B_i(1-p_i)$."
    )

    # =====================================================
    # 5.3. Dữ liệu
    # =====================================================
    st.markdown("## 5.3. Dữ liệu 15 dự án")

    display_df = df.copy()
    display_df["Rủi ro (%)"] = 100 * display_df["Rủi ro"]

    st.dataframe(
        display_df[
            [
                "Mã",
                "Dự án",
                "Trụ cột",
                "Chi phí",
                "Lợi ích",
                "Lợi ích kỳ vọng",
                "Rủi ro (%)",
                "Tiên quyết",
                "Bao trùm",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Đơn vị chi phí và lợi ích: tỷ VND. Bộ số liệu dùng cho mô phỏng bài tập; "
        "cần phân biệt với số liệu dự toán chính thức."
    )

    # =====================================================
    # 5.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 5.4. Yêu cầu lập trình")

    tab541, tab542, tab543, tab544 = st.tabs(
        [
            "5.4.1 - MIP cơ sở",
            "5.4.2 - Ngân sách 100.000",
            "5.4.3 - Lợi ích có rủi ro",
            "5.4.4 - Độ nhạy & kiểm định",
        ]
    )

    # -----------------------------------------------------
    # 5.4.1
    # -----------------------------------------------------
    with tab541:
        st.markdown(
            "### Câu 5.4.1. Giải MIP với ngân sách 80.000 tỷ VND"
        )

        base_budget = 80000.0
        base_result = _b5_solve_mip(
            budget=base_budget,
            risk_adjusted=False,
            max_projects=10,
        )

        if not base_result["success"]:
            st.error(
                f"Mô hình không có nghiệm tối ưu. Status: {base_result['status']}"
            )
        else:
            selected, metrics = _b5_result_table(
                base_result,
                base_budget,
            )

            kpi_cards(
                [
                    (
                        "Solver",
                        metrics["solver"],
                        metrics["status"],
                    ),
                    (
                        "Số dự án",
                        str(metrics["n_projects"]),
                        "tối đa 10",
                    ),
                    (
                        "Tổng chi phí",
                        f"{metrics['total_cost']:,.0f}",
                        "tỷ VND",
                    ),
                    (
                        "Tổng lợi ích",
                        f"{metrics['gross_benefit']:,.0f}",
                        f"còn {metrics['unused_budget']:,.0f} ngân sách",
                    ),
                ]
            )

            st.dataframe(
                selected[
                    [
                        "Mã",
                        "Dự án",
                        "Trụ cột",
                        "Chi phí",
                        "Lợi ích",
                        "Tiên quyết",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            selected_plot = selected.sort_values(
                "Lợi ích",
                ascending=True,
            )

            fig = px.bar(
                selected_plot,
                x="Lợi ích",
                y="Dự án",
                orientation="h",
                color="Trụ cột",
                text="Lợi ích",
                template=PLOT_TEMPLATE,
                title="Lợi ích của danh mục dự án được chọn",
            )
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=54, b=10),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.markdown("#### Kiểm tra ràng buộc")
            st.dataframe(
                _b5_validation_table(
                    base_result,
                    base_budget,
                    max_projects=10,
                ),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Xem mã PuLP/CBC rút gọn"):
                st.code(
                    """model = pulp.LpProblem(
    "Digital_Project_Selection",
    pulp.LpMaximize
)

y = {
    code: pulp.LpVariable(
        f"select_{code}",
        cat="Binary"
    )
    for code in codes
}

model += pulp.lpSum(
    benefit[i] * y[codes[i]]
    for i in range(15)
)

model += pulp.lpSum(
    cost[i] * y[codes[i]]
    for i in range(15)
) <= 80000

# Ví dụ quan hệ tiên quyết
model += y["P05"] <= y["P01"]
model += y["P06"] <= y["P05"]

model.solve(
    pulp.PULP_CBC_CMD(msg=False)
)""",
                    language="python",
                )

    # -----------------------------------------------------
    # 5.4.2
    # -----------------------------------------------------
    with tab542:
        st.markdown(
            "### Câu 5.4.2. Tăng ngân sách lên 100.000 tỷ VND"
        )

        result_80 = _b5_solve_mip(
            budget=80000.0,
            risk_adjusted=False,
            max_projects=10,
        )
        result_100 = _b5_solve_mip(
            budget=100000.0,
            risk_adjusted=False,
            max_projects=10,
        )

        selected_80, metrics_80 = _b5_result_table(
            result_80,
            80000.0,
        )
        selected_100, metrics_100 = _b5_result_table(
            result_100,
            100000.0,
        )

        comparison = pd.DataFrame(
            {
                "Chỉ tiêu": [
                    "Số dự án",
                    "Tổng chi phí",
                    "Tổng lợi ích",
                    "Lợi ích kỳ vọng",
                ],
                "Ngân sách 80.000": [
                    metrics_80["n_projects"],
                    metrics_80["total_cost"],
                    metrics_80["gross_benefit"],
                    metrics_80["expected_benefit"],
                ],
                "Ngân sách 100.000": [
                    metrics_100["n_projects"],
                    metrics_100["total_cost"],
                    metrics_100["gross_benefit"],
                    metrics_100["expected_benefit"],
                ],
            }
        )

        kpi_cards(
            [
                (
                    "Lợi ích tăng thêm",
                    f"{metrics_100['gross_benefit']-metrics_80['gross_benefit']:,.0f}",
                    "khi tăng ngân sách 20.000",
                ),
                (
                    "Chi phí tăng thêm",
                    f"{metrics_100['total_cost']-metrics_80['total_cost']:,.0f}",
                    "tỷ VND",
                ),
                (
                    "Dự án tăng thêm",
                    str(
                        metrics_100["n_projects"]
                        - metrics_80["n_projects"]
                    ),
                    "số lượng",
                ),
                (
                    "Lợi ích biên/chi phí",
                    f"{(metrics_100['gross_benefit']-metrics_80['gross_benefit'])/20000:.3f}",
                    "trên 1 đồng ngân sách bổ sung",
                ),
            ]
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        codes_80 = set(selected_80["Mã"].tolist())
        codes_100 = set(selected_100["Mã"].tolist())

        portfolio_change = pd.DataFrame(
            {
                "Nhóm": [
                    "Giữ lại",
                    "Bổ sung ở 100.000",
                    "Bị thay thế",
                ],
                "Mã dự án": [
                    ", ".join(sorted(codes_80 & codes_100)),
                    ", ".join(sorted(codes_100 - codes_80)) or "Không có",
                    ", ".join(sorted(codes_80 - codes_100)) or "Không có",
                ],
            }
        )

        st.dataframe(
            portfolio_change,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 5.4.3
    # -----------------------------------------------------
    with tab543:
        st.markdown(
            "### Câu 5.4.3. Tối ưu lợi ích kỳ vọng có rủi ro"
        )

        deterministic_result = _b5_solve_mip(
            budget=80000.0,
            risk_adjusted=False,
            max_projects=10,
        )
        risk_result = _b5_solve_mip(
            budget=80000.0,
            risk_adjusted=True,
            max_projects=10,
        )

        deterministic_selected, deterministic_metrics = _b5_result_table(
            deterministic_result,
            80000.0,
        )
        risk_selected, risk_metrics = _b5_result_table(
            risk_result,
            80000.0,
        )

        risk_compare = pd.DataFrame(
            {
                "Chỉ tiêu": [
                    "Tổng chi phí",
                    "Lợi ích gộp",
                    "Lợi ích kỳ vọng",
                    "Rủi ro danh mục",
                    "Số dự án",
                ],
                "Tối ưu lợi ích gộp": [
                    deterministic_metrics["total_cost"],
                    deterministic_metrics["gross_benefit"],
                    deterministic_metrics["expected_benefit"],
                    deterministic_metrics["portfolio_risk"],
                    deterministic_metrics["n_projects"],
                ],
                "Tối ưu lợi ích kỳ vọng": [
                    risk_metrics["total_cost"],
                    risk_metrics["gross_benefit"],
                    risk_metrics["expected_benefit"],
                    risk_metrics["portfolio_risk"],
                    risk_metrics["n_projects"],
                ],
            }
        )

        kpi_cards(
            [
                (
                    "Expected benefit mới",
                    f"{risk_metrics['expected_benefit']:,.0f}",
                    "tỷ VND",
                ),
                (
                    "Rủi ro danh mục",
                    f"{100*risk_metrics['portfolio_risk']:.2f}%",
                    "gia quyền theo chi phí",
                ),
                (
                    "Thay đổi expected benefit",
                    f"{risk_metrics['expected_benefit']-deterministic_metrics['expected_benefit']:+,.0f}",
                    "so với tối ưu lợi ích gộp",
                ),
                (
                    "Thay đổi rủi ro",
                    f"{100*(risk_metrics['portfolio_risk']-deterministic_metrics['portfolio_risk']):+.2f} điểm %",
                    "risk-adjusted trừ deterministic",
                ),
            ]
        )

        st.dataframe(
            risk_compare,
            use_container_width=True,
            hide_index=True,
        )

        deterministic_codes = set(
            deterministic_selected["Mã"].tolist()
        )
        risk_codes = set(
            risk_selected["Mã"].tolist()
        )

        st.markdown("#### Thay đổi danh mục do điều chỉnh rủi ro")

        st.dataframe(
            pd.DataFrame(
                {
                    "Nhóm": [
                        "Cùng được chọn",
                        "Chỉ mô hình gộp chọn",
                        "Chỉ mô hình rủi ro chọn",
                    ],
                    "Dự án": [
                        ", ".join(
                            sorted(
                                deterministic_codes & risk_codes
                            )
                        ),
                        ", ".join(
                            sorted(
                                deterministic_codes - risk_codes
                            )
                        ) or "Không có",
                        ", ".join(
                            sorted(
                                risk_codes - deterministic_codes
                            )
                        ) or "Không có",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 5.4.4
    # -----------------------------------------------------
    with tab544:
        st.markdown(
            "### Câu 5.4.4. Phân tích độ nhạy ngân sách và tính ổn định danh mục"
        )

        budgets = np.arange(
            60000.0,
            100001.0,
            5000.0,
        )

        rows = []
        selection_frequency = {
            code: 0
            for code in df["Mã"].tolist()
        }

        for budget in budgets:
            result = _b5_solve_mip(
                budget=float(budget),
                risk_adjusted=True,
                max_projects=10,
            )

            selected, metrics = _b5_result_table(
                result,
                float(budget),
            )

            selected_codes = set(
                selected["Mã"].tolist()
            )

            for code in selected_codes:
                selection_frequency[code] += 1

            rows.append(
                [
                    budget,
                    metrics["n_projects"],
                    metrics["total_cost"],
                    metrics["gross_benefit"],
                    metrics["expected_benefit"],
                    metrics["portfolio_risk"],
                    ", ".join(sorted(selected_codes)),
                ]
            )

        sensitivity = pd.DataFrame(
            rows,
            columns=[
                "Ngân sách",
                "Số dự án",
                "Tổng chi phí",
                "Lợi ích gộp",
                "Lợi ích kỳ vọng",
                "Rủi ro danh mục",
                "Danh mục",
            ],
        )

        fig_sensitivity = px.line(
            sensitivity,
            x="Ngân sách",
            y=[
                "Lợi ích gộp",
                "Lợi ích kỳ vọng",
            ],
            markers=True,
            template=PLOT_TEMPLATE,
            title="Độ nhạy lợi ích theo ngân sách",
        )
        fig_sensitivity.update_layout(
            height=470,
            margin=dict(l=10, r=10, t=54, b=10),
        )
        st.plotly_chart(
            fig_sensitivity,
            use_container_width=True,
        )

        frequency_df = pd.DataFrame(
            {
                "Mã": list(selection_frequency.keys()),
                "Số lần được chọn": list(
                    selection_frequency.values()
                ),
            }
        ).merge(
            df[["Mã", "Dự án"]],
            on="Mã",
            how="left",
        )

        frequency_df["Tần suất (%)"] = (
            100
            * frequency_df["Số lần được chọn"]
            / len(budgets)
        )

        frequency_df = frequency_df.sort_values(
            ["Số lần được chọn", "Mã"],
            ascending=[False, True],
        )

        st.markdown("#### Tần suất dự án xuất hiện trong nghiệm tối ưu")
        st.dataframe(
            frequency_df,
            use_container_width=True,
            hide_index=True,
        )

        stable_projects = frequency_df.loc[
            frequency_df["Số lần được chọn"] == len(budgets),
            "Mã",
        ].tolist()

        st.success(
            "Các dự án xuất hiện ở mọi mức ngân sách: "
            + (
                ", ".join(stable_projects)
                if stable_projects
                else "Không có"
            )
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_result = _b5_solve_mip(
        budget=80000.0,
        risk_adjusted=True,
        max_projects=10,
    )
    export_selected, _ = _b5_result_table(
        export_result,
        80000.0,
    )

    st.download_button(
        "Tải danh mục tối ưu Bài 5",
        data=export_selected.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="bai5_danh_muc_du_an_toi_uu.csv",
        mime="text/csv",
        key="download_bai5_fixed",
    )

    # =====================================================
    # 5.5. Thảo luận chính sách
    # =====================================================
    st.markdown("## 5.5. Câu hỏi thảo luận chính sách")

    policy_result = _b5_solve_mip(
        budget=80000.0,
        risk_adjusted=True,
        max_projects=10,
    )
    policy_selected, policy_metrics = _b5_result_table(
        policy_result,
        80000.0,
    )

    with st.expander(
        "a) Vì sao không nên chỉ xếp hạng dự án theo B/C rồi chọn lần lượt?",
        expanded=True,
    ):
        st.markdown(
            "Cách chọn tuần tự theo B/C có thể vi phạm quan hệ tiên quyết, bỏ qua yêu cầu "
            "an toàn số, nhân lực và tính bao trùm. MIP đánh giá đồng thời toàn bộ tổ hợp, "
            "nên phù hợp hơn khi các dự án có quan hệ bổ trợ hoặc phụ thuộc."
        )

    with st.expander(
        "b) Ngân sách tăng lên 100.000 tỷ có luôn làm danh mục tốt hơn tương ứng không?",
        expanded=True,
    ):
        st.markdown(
            "Lợi ích không nhất thiết tăng tỷ lệ thuận với ngân sách. Sau khi các dự án "
            "có hiệu quả cao đã được chọn, ngân sách bổ sung có thể chỉ tài trợ các dự án "
            "có lợi ích biên thấp hơn hoặc bị giới hạn bởi năng lực triển khai tối đa 10 dự án."
        )

    with st.expander(
        "c) Tại sao mô hình có rủi ro có thể loại một dự án lợi ích gộp cao?",
        expanded=True,
    ):
        st.markdown(
            "Dự án có lợi ích danh nghĩa cao nhưng xác suất thất bại lớn có thể có lợi ích "
            "kỳ vọng thấp hơn một dự án quy mô nhỏ nhưng ổn định. Điều chỉnh rủi ro giúp "
            "danh mục tránh phụ thuộc quá lớn vào một số dự án công nghệ có độ bất định cao."
        )

    with st.expander(
        "d) Danh mục khuyến nghị hiện tại là gì?",
        expanded=True,
    ):
        selected_codes = ", ".join(
            policy_selected["Mã"].tolist()
        )

        st.markdown(
            f"Với ngân sách 80.000 tỷ VND và mục tiêu lợi ích kỳ vọng, danh mục mô phỏng "
            f"chọn **{policy_metrics['n_projects']} dự án**: **{selected_codes}**. "
            f"Tổng chi phí khoảng **{policy_metrics['total_cost']:,.0f} tỷ VND**, "
            f"lợi ích kỳ vọng khoảng **{policy_metrics['expected_benefit']:,.0f} tỷ VND**. "
            "Kết quả là đầu vào hỗ trợ quyết định, không thay thế thẩm định kỹ thuật, pháp lý "
            "và đánh giá tác động xã hội của từng dự án."
        )


def _b6_prepare_data():
    """Chuẩn bị dữ liệu, tiêu chí, loại tiêu chí và trọng số chuyên gia."""
    df = load_regions().copy()

    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]

    is_benefit = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    expert_weights = np.array(
        [
            0.10,
            0.10,
            0.15,
            0.20,
            0.15,
            0.15,
            0.05,
            0.10,
        ],
        dtype=float,
    )

    return df, criteria, is_benefit, expert_weights


def _b6_labels():
    """Nhãn tiếng Việt của các tiêu chí."""
    return {
        "grdp_per_capita_million_VND": "GRDP/người",
        "fdi_registered_billion_USD": "FDI đăng ký",
        "digital_index_0_100": "Digital Index",
        "ai_readiness_0_100": "AI Readiness",
        "trained_labor_pct": "Lao động qua đào tạo",
        "rd_intensity_pct": "R&D/GDP",
        "internet_penetration_pct": "Internet",
        "gini_coef": "Gini",
    }


def _b6_entropy_weights(df, criteria, is_benefit):
    """
    Tính trọng số Entropy.

    Trước khi tính Entropy, toàn bộ tiêu chí được chuyển về hướng
    giá trị lớn hơn là tốt hơn và chuẩn hóa min-max.
    """
    transformed = pd.DataFrame(index=df.index)

    for criterion, benefit in zip(criteria, is_benefit):
        if benefit:
            transformed[criterion] = minmax(df[criterion])
        else:
            transformed[criterion] = reverse_minmax(df[criterion])

    weights = entropy_weights_positive(
        transformed.to_numpy(dtype=float)
    )

    return weights, transformed


def _b6_rank_result(df, scores, score_name):
    """Tạo bảng điểm và thứ hạng."""
    result = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            score_name: np.asarray(scores, dtype=float),
        }
    )

    result["Xếp hạng"] = (
        result[score_name]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result.sort_values(
        ["Xếp hạng", score_name],
        ascending=[True, False],
    ).reset_index(drop=True)


def _b6_ahp_pairwise_matrix():
    """
    Ma trận so sánh cặp AHP minh họa theo thang Saaty.

    Thứ tự:
    GRDP, FDI, Digital, AI, Lao động, R&D, Internet, Gini.

    Ma trận được nhập trực tiếp từ các đánh giá cặp, không suy ra
    bằng phép chia bộ trọng số chuyên gia.
    """
    n = 8
    matrix = np.ones((n, n), dtype=float)

    judgments = {
        (0, 1): 1,
        (0, 2): 1 / 2,
        (0, 3): 1 / 3,
        (0, 4): 1 / 2,
        (0, 5): 1 / 2,
        (0, 6): 2,
        (0, 7): 1,

        (1, 2): 1 / 2,
        (1, 3): 1 / 3,
        (1, 4): 1 / 2,
        (1, 5): 1 / 2,
        (1, 6): 2,
        (1, 7): 1,

        (2, 3): 1 / 2,
        (2, 4): 1,
        (2, 5): 1,
        (2, 6): 3,
        (2, 7): 2,

        (3, 4): 2,
        (3, 5): 2,
        (3, 6): 4,
        (3, 7): 3,

        (4, 5): 1,
        (4, 6): 3,
        (4, 7): 2,

        (5, 6): 3,
        (5, 7): 2,

        (6, 7): 1 / 2,
    }

    for (i, j), value in judgments.items():
        matrix[i, j] = float(value)
        matrix[j, i] = 1.0 / float(value)

    return matrix


def _b6_ahp_weights(pairwise_matrix):
    """
    Tính vector trọng số AHP và tỷ lệ nhất quán CR.
    """
    pairwise_matrix = np.asarray(
        pairwise_matrix,
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eig(
        pairwise_matrix
    )

    principal_index = int(
        np.argmax(eigenvalues.real)
    )

    lambda_max = float(
        eigenvalues[principal_index].real
    )

    principal_vector = np.abs(
        eigenvectors[:, principal_index].real
    )

    weights = (
        principal_vector
        / principal_vector.sum()
    )

    n = pairwise_matrix.shape[0]

    consistency_index = (
        (lambda_max - n) / (n - 1)
        if n > 1
        else 0.0
    )

    random_index_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    random_index = random_index_table.get(
        n,
        1.49,
    )

    consistency_ratio = (
        consistency_index / random_index
        if random_index > 0
        else 0.0
    )

    return (
        weights,
        lambda_max,
        consistency_index,
        consistency_ratio,
    )


def _b6_rank_stability_table(
    df,
    expert_score,
    entropy_score,
    ahp_score,
):
    """So sánh thứ hạng của ba phương pháp."""
    comparison = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            "TOPSIS chuyên gia": expert_score,
            "TOPSIS Entropy": entropy_score,
            "TOPSIS AHP": ahp_score,
        }
    )

    comparison["Hạng chuyên gia"] = (
        comparison["TOPSIS chuyên gia"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Hạng Entropy"] = (
        comparison["TOPSIS Entropy"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Hạng AHP"] = (
        comparison["TOPSIS AHP"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Biên độ hạng"] = (
        comparison[
            [
                "Hạng chuyên gia",
                "Hạng Entropy",
                "Hạng AHP",
            ]
        ].max(axis=1)
        - comparison[
            [
                "Hạng chuyên gia",
                "Hạng Entropy",
                "Hạng AHP",
            ]
        ].min(axis=1)
    )

    return comparison.sort_values(
        "Hạng chuyên gia"
    ).reset_index(drop=True)


def page_6():
    hero(
        "Bài 6 — TOPSIS xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI",
        "Xếp hạng đa tiêu chí bằng TOPSIS, so sánh trọng số chuyên gia, Entropy và AHP; kiểm tra độ nhạy của AI Readiness và độ ổn định thứ hạng.",
        ["6.1-6.5", "TOPSIS", "Entropy", "AHP", "Sensitivity"],
    )

    (
        df,
        criteria,
        is_benefit,
        expert_weights,
    ) = _b6_prepare_data()

    labels = _b6_labels()

    # =====================================================
    # 6.1. Bối cảnh
    # =====================================================
    st.markdown("## 6.1. Bối cảnh Việt Nam")

    st.markdown(
        """
        Nguồn lực dành cho trung tâm AI, hạ tầng dữ liệu, nghiên cứu và đào tạo
        không thể phân bổ đồng đều cho mọi vùng. Sáu vùng kinh tế có khác biệt lớn
        về quy mô kinh tế, FDI, hạ tầng số, nhân lực, R&D và mức độ bất bình đẳng.

        Bài 6 sử dụng TOPSIS để xác định vùng gần nhất với phương án lý tưởng.
        Kết quả được kiểm tra bằng ba hệ trọng số: chuyên gia, Entropy và AHP.
        """
    )

    # =====================================================
    # 6.2. Mô hình
    # =====================================================
    st.markdown("## 6.2. Mô hình TOPSIS")

    st.markdown("### Bước 1. Chuẩn hóa vector")
    st.latex(
        r"r_{ij}="
        r"\frac{x_{ij}}"
        r"{\sqrt{\sum_{i=1}^{m}x_{ij}^{2}}}"
    )

    st.markdown("### Bước 2. Ma trận có trọng số")
    st.latex(r"v_{ij}=w_jr_{ij}")

    st.markdown("### Bước 3. Nghiệm lý tưởng")
    st.latex(
        r"A^*=\{v_1^*,...,v_n^*\},"
        r"\qquad"
        r"A^-=\{v_1^-,...,v_n^-\}"
    )

    st.markdown("### Bước 4. Khoảng cách Euclid")
    st.latex(
        r"S_i^*="
        r"\sqrt{\sum_j(v_{ij}-v_j^*)^2},"
        r"\qquad"
        r"S_i^-="
        r"\sqrt{\sum_j(v_{ij}-v_j^-)^2}"
    )

    st.markdown("### Bước 5. Hệ số gần lý tưởng")
    st.latex(
        r"C_i^*="
        r"\frac{S_i^-}{S_i^*+S_i^-}"
    )

    st.info(
        "C* càng lớn thì vùng càng gần phương án lý tưởng và được ưu tiên cao hơn."
    )

    # =====================================================
    # 6.3. Dữ liệu
    # =====================================================
    st.markdown("## 6.3. Dữ liệu và tiêu chí")

    data_display = df[
        ["region_name_vi"] + criteria
    ].rename(
        columns={
            "region_name_vi": "Vùng",
            **labels,
        }
    )

    st.dataframe(
        data_display,
        use_container_width=True,
        hide_index=True,
    )

    criteria_table = pd.DataFrame(
        {
            "Tiêu chí": [
                labels[c]
                for c in criteria
            ],
            "Loại": [
                "Lợi ích" if flag else "Chi phí"
                for flag in is_benefit
            ],
            "Trọng số chuyên gia": expert_weights,
        }
    )

    st.dataframe(
        criteria_table.style.format(
            {"Trọng số chuyên gia": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Kết quả dùng chung.
    expert_score = topsis_score(
        df,
        criteria,
        expert_weights,
        is_benefit,
    )

    entropy_weights, entropy_matrix = _b6_entropy_weights(
        df,
        criteria,
        is_benefit,
    )

    entropy_score = topsis_score(
        df,
        criteria,
        entropy_weights,
        is_benefit,
    )

    pairwise_matrix = _b6_ahp_pairwise_matrix()

    (
        ahp_weights,
        ahp_lambda_max,
        ahp_ci,
        ahp_cr,
    ) = _b6_ahp_weights(
        pairwise_matrix
    )

    ahp_score = topsis_score(
        df,
        criteria,
        ahp_weights,
        is_benefit,
    )

    expert_result = _b6_rank_result(
        df,
        expert_score,
        "TOPSIS chuyên gia",
    )

    entropy_result = _b6_rank_result(
        df,
        entropy_score,
        "TOPSIS Entropy",
    )

    ahp_result = _b6_rank_result(
        df,
        ahp_score,
        "TOPSIS AHP",
    )

    # =====================================================
    # 6.4. Lập trình
    # =====================================================
    st.markdown("## 6.4. Yêu cầu lập trình")

    tab641, tab642, tab643, tab644 = st.tabs(
        [
            "6.4.1 - TOPSIS chuyên gia",
            "6.4.2 - Entropy",
            "6.4.3 - Độ nhạy AI",
            "6.4.4 - AHP & ổn định hạng",
        ]
    )

    # -----------------------------------------------------
    # 6.4.1
    # -----------------------------------------------------
    with tab641:
        st.markdown(
            "### Câu 6.4.1. TOPSIS với trọng số chuyên gia"
        )

        kpi_cards(
            [
                (
                    "Vùng dẫn đầu",
                    expert_result.iloc[0]["Vùng"],
                    f"C*={expert_result.iloc[0]['TOPSIS chuyên gia']:.4f}",
                ),
                (
                    "Vùng thứ hai",
                    expert_result.iloc[1]["Vùng"],
                    f"C*={expert_result.iloc[1]['TOPSIS chuyên gia']:.4f}",
                ),
                (
                    "Vùng thứ ba",
                    expert_result.iloc[2]["Vùng"],
                    f"C*={expert_result.iloc[2]['TOPSIS chuyên gia']:.4f}",
                ),
                (
                    "Tổng trọng số",
                    f"{expert_weights.sum():.2f}",
                    "phải bằng 1",
                ),
            ]
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.dataframe(
                expert_result.style.format(
                    {"TOPSIS chuyên gia": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.plotly_chart(
                plot_bar(
                    expert_result,
                    "Vùng",
                    "TOPSIS chuyên gia",
                    "TOPSIS với trọng số chuyên gia",
                    text="TOPSIS chuyên gia",
                ),
                use_container_width=True,
            )

        with st.expander(
            "Xem mã TOPSIS rút gọn"
        ):
            st.code(
                """score = topsis_score(
    df,
    criteria,
    expert_weights,
    is_benefit
)

result = pd.DataFrame({
    "Vùng": df["region_name_vi"],
    "C_star": score
}).sort_values(
    "C_star",
    ascending=False
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 6.4.2
    # -----------------------------------------------------
    with tab642:
        st.markdown(
            "### Câu 6.4.2. Trọng số Entropy và TOPSIS"
        )

        weight_compare = pd.DataFrame(
            {
                "Tiêu chí": [
                    labels[c]
                    for c in criteria
                ],
                "Chuyên gia": expert_weights,
                "Entropy": entropy_weights,
                "Chênh lệch": (
                    entropy_weights
                    - expert_weights
                ),
            }
        )

        st.dataframe(
            weight_compare.style.format(
                {
                    "Chuyên gia": "{:.4f}",
                    "Entropy": "{:.4f}",
                    "Chênh lệch": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        rank_compare = pd.DataFrame(
            {
                "Vùng": df["region_name_vi"],
                "Điểm chuyên gia": expert_score,
                "Điểm Entropy": entropy_score,
            }
        )

        rank_compare["Hạng chuyên gia"] = (
            rank_compare["Điểm chuyên gia"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        rank_compare["Hạng Entropy"] = (
            rank_compare["Điểm Entropy"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        rank_compare["Thay đổi hạng"] = (
            rank_compare["Hạng chuyên gia"]
            - rank_compare["Hạng Entropy"]
        )

        st.dataframe(
            rank_compare.sort_values(
                "Hạng chuyên gia"
            ).style.format(
                {
                    "Điểm chuyên gia": "{:.4f}",
                    "Điểm Entropy": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        compare_long = rank_compare.melt(
            id_vars="Vùng",
            value_vars=[
                "Điểm chuyên gia",
                "Điểm Entropy",
            ],
            var_name="Phương pháp",
            value_name="Điểm",
        )

        fig = px.bar(
            compare_long,
            x="Vùng",
            y="Điểm",
            color="Phương pháp",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="So sánh TOPSIS chuyên gia và Entropy",
        )
        fig.update_layout(
            height=480,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        changed = rank_compare.loc[
            rank_compare["Thay đổi hạng"] != 0,
            "Vùng",
        ].tolist()

        st.info(
            "Các vùng thay đổi thứ hạng: "
            + (
                ", ".join(changed)
                if changed
                else "không có"
            )
            + "."
        )

    # -----------------------------------------------------
    # 6.4.3
    # -----------------------------------------------------
    with tab643:
        st.markdown(
            "### Câu 6.4.3. Độ nhạy trọng số AI Readiness"
        )

        ai_values = np.arange(
            0.10,
            0.401,
            0.05,
        )

        rows = []

        for ai_weight in ai_values:
            weights = expert_weights.copy()

            other_mask = np.ones(
                len(weights),
                dtype=bool,
            )
            other_mask[3] = False

            weights[other_mask] = (
                weights[other_mask]
                / weights[other_mask].sum()
                * (1.0 - ai_weight)
            )
            weights[3] = ai_weight

            score = topsis_score(
                df,
                criteria,
                weights,
                is_benefit,
            )

            temp = pd.DataFrame(
                {
                    "Vùng": df["region_name_vi"],
                    "Điểm": score,
                }
            )

            temp["Hạng"] = (
                temp["Điểm"]
                .rank(
                    ascending=False,
                    method="min",
                )
                .astype(int)
            )

            for _, row in temp.iterrows():
                rows.append(
                    [
                        ai_weight,
                        row["Vùng"],
                        row["Điểm"],
                        row["Hạng"],
                    ]
                )

        sensitivity = pd.DataFrame(
            rows,
            columns=[
                "Trọng số AI",
                "Vùng",
                "Điểm",
                "Hạng",
            ],
        )

        fig_rank = px.line(
            sensitivity,
            x="Trọng số AI",
            y="Hạng",
            color="Vùng",
            markers=True,
            template=PLOT_TEMPLATE,
            title="Độ nhạy thứ hạng theo trọng số AI",
        )
        fig_rank.update_yaxes(
            autorange="reversed"
        )
        fig_rank.update_layout(
            height=540,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_rank,
            use_container_width=True,
        )

        rank_pivot = sensitivity.pivot(
            index="Vùng",
            columns="Trọng số AI",
            values="Hạng",
        )

        fig_heatmap = px.imshow(
            rank_pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            template=PLOT_TEMPLATE,
            title="Heatmap độ nhạy thứ hạng",
        )
        fig_heatmap.update_layout(
            height=510,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
        )

        leaders = (
            sensitivity[
                sensitivity["Hạng"] == 1
            ]["Vùng"]
            .value_counts()
            .rename_axis("Vùng")
            .reset_index(
                name="Số lần đứng đầu"
            )
        )

        st.dataframe(
            leaders,
            use_container_width=True,
            hide_index=True,
        )

        if len(leaders) == 1:
            st.success(
                "Vị trí dẫn đầu ổn định trong toàn bộ dải trọng số AI."
            )
        else:
            st.warning(
                "Vị trí dẫn đầu thay đổi khi trọng số AI thay đổi; "
                "kết luận chính sách nhạy với ưu tiên của người ra quyết định."
            )

    # -----------------------------------------------------
    # 6.4.4
    # -----------------------------------------------------
    with tab644:
        st.markdown(
            "### Câu 6.4.4. AHP và độ ổn định thứ hạng"
        )

        ahp_weight_table = pd.DataFrame(
            {
                "Tiêu chí": [
                    labels[c]
                    for c in criteria
                ],
                "Trọng số AHP": ahp_weights,
            }
        )

        kpi_cards(
            [
                (
                    "λmax",
                    f"{ahp_lambda_max:.4f}",
                    "giá trị riêng lớn nhất",
                ),
                (
                    "CI",
                    f"{ahp_ci:.4f}",
                    "Consistency Index",
                ),
                (
                    "CR",
                    f"{ahp_cr:.4f}",
                    "đạt nếu <0,10",
                ),
                (
                    "Vùng dẫn đầu AHP",
                    ahp_result.iloc[0]["Vùng"],
                    f"C*={ahp_result.iloc[0]['TOPSIS AHP']:.4f}",
                ),
            ]
        )

        c1, c2 = st.columns([0.8, 1.2])

        with c1:
            st.dataframe(
                ahp_weight_table.style.format(
                    {"Trọng số AHP": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.dataframe(
                ahp_result.style.format(
                    {"TOPSIS AHP": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if ahp_cr < 0.10:
            st.success(
                f"Ma trận AHP nhất quán ở mức chấp nhận được: CR={ahp_cr:.4f}<0,10."
            )
        else:
            st.error(
                f"CR={ahp_cr:.4f}≥0,10. Cần rà soát lại đánh giá cặp."
            )

        pairwise_df = pd.DataFrame(
            pairwise_matrix,
            index=[
                labels[c]
                for c in criteria
            ],
            columns=[
                labels[c]
                for c in criteria
            ],
        )

        with st.expander(
            "Xem ma trận so sánh cặp AHP"
        ):
            st.dataframe(
                pairwise_df.style.format(
                    "{:.3f}"
                ),
                use_container_width=True,
            )

        stability = _b6_rank_stability_table(
            df,
            expert_score,
            entropy_score,
            ahp_score,
        )

        st.markdown(
            "#### So sánh độ ổn định giữa ba phương pháp"
        )

        st.dataframe(
            stability.style.format(
                {
                    "TOPSIS chuyên gia": "{:.4f}",
                    "TOPSIS Entropy": "{:.4f}",
                    "TOPSIS AHP": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        rank_corr = (
            stability[
                [
                    "Hạng chuyên gia",
                    "Hạng Entropy",
                    "Hạng AHP",
                ]
            ]
            .corr(
                method="spearman"
            )
        )

        st.markdown(
            "#### Tương quan Spearman giữa các thứ hạng"
        )

        st.dataframe(
            rank_corr.style.format(
                "{:.3f}"
            ),
            use_container_width=True,
        )

        stable_regions = stability.loc[
            stability["Biên độ hạng"] <= 1,
            "Vùng",
        ].tolist()

        st.info(
            "Các vùng có thứ hạng tương đối ổn định, biên độ không quá 1 bậc: "
            + (
                ", ".join(stable_regions)
                if stable_regions
                else "không có"
            )
            + "."
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_result = _b6_rank_stability_table(
        df,
        expert_score,
        entropy_score,
        ahp_score,
    )

    st.download_button(
        "Tải kết quả Bài 6 dạng CSV",
        data=export_result.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="bai6_topsis_6_vung.csv",
        mime="text/csv",
        key="download_bai6_fixed",
    )

    # =====================================================
    # 6.5. Thảo luận
    # =====================================================
    st.markdown(
        "## 6.5. Câu hỏi thảo luận chính sách"
    )

    top_three = expert_result.head(
        3
    )["Vùng"].tolist()

    with st.expander(
        "a) Vùng dẫn đầu có nên được đặt trung tâm AI quốc gia đầu tiên không?",
        expanded=True,
    ):
        st.markdown(
            f"Vùng dẫn đầu theo trọng số chuyên gia là "
            f"**{expert_result.iloc[0]['Vùng']}**. Đây là ứng viên mạnh về quy mô "
            "kinh tế, hạ tầng số, AI và nhân lực. Tuy nhiên, quyết định địa điểm còn "
            "phải xét nguồn điện, quỹ đất, an ninh dữ liệu, liên kết đại học-doanh nghiệp "
            "và yêu cầu cân bằng vùng."
        )

    with st.expander(
        "b) Vì sao Entropy có thể cho thứ hạng khác chuyên gia?",
        expanded=True,
    ):
        st.markdown(
            "Entropy trao trọng số lớn cho tiêu chí có khả năng phân biệt các vùng mạnh. "
            "Trong khi đó, trọng số chuyên gia phản ánh ưu tiên chính sách. Vì vậy, "
            "hai cách tiếp cận có thể khác nhau dù cùng sử dụng một bộ dữ liệu."
        )

    with st.expander(
        "c) Nếu AI Readiness và Internet tương quan cao thì xử lý thế nào?",
        expanded=True,
    ):
        st.markdown(
            "Tương quan cao có thể gây đếm trùng thông tin. Có thể loại bớt một tiêu chí, "
            "dùng PCA, điều chỉnh trọng số theo tương quan hoặc sử dụng CRITIC. "
            "Kết quả nên được kiểm tra lại sau khi loại tiêu chí trùng lặp."
        )

    with st.expander(
        "d) Nếu xây dựng ba trung tâm AI, nên ưu tiên vùng nào?",
        expanded=True,
    ):
        st.markdown(
            f"Theo TOPSIS chuyên gia, ba vùng dẫn đầu là "
            f"**{', '.join(top_three)}**. Mô hình chỉ cung cấp xếp hạng định lượng; "
            "cần kết hợp thẩm định hạ tầng năng lượng, an ninh, vốn nhân lực và tác động lan tỏa."
        )

    with st.expander(
        "e) Kết luận có ổn định giữa các phương pháp không?",
        expanded=True,
    ):
        max_rank_range = int(
            export_result["Biên độ hạng"].max()
        )

        st.markdown(
            f"Biên độ thay đổi thứ hạng lớn nhất giữa chuyên gia, Entropy và AHP là "
            f"**{max_rank_range} bậc**. Nếu biên độ nhỏ, kết luận tương đối vững; "
            "nếu biên độ lớn, báo cáo phải trình bày kết quả như một khoảng ưu tiên "
            "thay vì khẳng định một thứ hạng duy nhất."
        )


def _b7_decode(decision_vector):
    return np.asarray(decision_vector, dtype=float).reshape(6, 4)


def _b7_objectives(decision_vector):
    """
    Trả về bốn mục tiêu theo dạng tự nhiên:
    Growth càng cao càng tốt;
    Inequality, Emission, DataRisk càng thấp càng tốt.
    """
    regions, items, beta, d0 = region_beta_matrix()
    x = _b7_decode(decision_vector)

    region_gain = (beta * x).sum(axis=1)
    growth = float(region_gain.sum())

    digital_after = d0 + 0.002 * x[:, 1]
    inequality = float(gini(digital_after))

    emission_coeff = np.array([0.42, 0.14, 0.31, 0.08], dtype=float)
    emission = float((x * emission_coeff).sum())

    data_risk_coeff = np.array([0.12, 0.24, 0.58, 0.10], dtype=float)
    data_risk = float((x * data_risk_coeff).sum())

    fairness_ratio = float(
        digital_after.min() / max(digital_after.max(), 1e-12)
    )

    return {
        "Growth": growth,
        "Inequality": inequality,
        "Emission": emission,
        "DataRisk": data_risk,
        "FairnessRatio": fairness_ratio,
    }


def _b7_constraint_values(
    decision_vector,
    budget=50000.0,
    region_floor=5000.0,
    region_cap=12000.0,
    human_floor=12000.0,
    fairness_lambda=0.68,
):
    """
    Pymoo quy ước G <= 0 là khả thi.
    """
    x = _b7_decode(decision_vector)
    _, _, _, d0 = region_beta_matrix()

    region_total = x.sum(axis=1)
    digital_after = d0 + 0.002 * x[:, 1]

    constraints = [
        x.sum() - budget,
        human_floor - x[:, 3].sum(),
    ]

    constraints.extend(region_floor - region_total)
    constraints.extend(region_total - region_cap)

    max_digital = digital_after.max()
    constraints.extend(
        fairness_lambda * max_digital - digital_after
    )

    return np.asarray(constraints, dtype=float)


@st.cache_data(show_spinner=False)
def _b7_run_nsga2(
    population_size=100,
    generations=200,
    seed=42,
    fairness_lambda=0.68,
):
    """
    Chạy NSGA-II thật bằng pymoo.
    """
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thiếu pymoo. Hãy cài pymoo>=0.6.1."
        ) from exc

    class RegionalParetoProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=20,
                xl=np.zeros(24, dtype=float),
                xu=np.full(24, 12000.0, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            obj = _b7_objectives(x)

            # Pymoo luôn MINIMIZE.
            out["F"] = np.array(
                [
                    -obj["Growth"],
                    obj["Inequality"],
                    obj["Emission"],
                    obj["DataRisk"],
                ],
                dtype=float,
            )

            out["G"] = _b7_constraint_values(
                x,
                fairness_lambda=fairness_lambda,
            )

    algorithm = NSGA2(
        pop_size=int(population_size),
        sampling=FloatRandomSampling(),
        crossover=SBX(
            prob=0.90,
            eta=15,
        ),
        mutation=PM(
            eta=20,
        ),
        eliminate_duplicates=True,
    )

    result = minimize(
        RegionalParetoProblem(),
        algorithm,
        get_termination(
            "n_gen",
            int(generations),
        ),
        seed=int(seed),
        save_history=False,
        verbose=False,
    )

    if result.X is None or result.F is None:
        return pd.DataFrame(), np.empty((0, 24))

    x_values = np.atleast_2d(
        np.asarray(result.X, dtype=float)
    )
    f_values = np.atleast_2d(
        np.asarray(result.F, dtype=float)
    )

    objective_df = pd.DataFrame(
        {
            "Growth": -f_values[:, 0],
            "Inequality": f_values[:, 1],
            "Emission": f_values[:, 2],
            "DataRisk": f_values[:, 3],
        }
    )

    objective_df["FairnessRatio"] = [
        _b7_objectives(x)["FairnessRatio"]
        for x in x_values
    ]

    objective_df["SolutionID"] = np.arange(
        1,
        len(objective_df) + 1,
    )

    return objective_df, x_values


def _b7_topsis_compromise(
    pareto_df,
    weights,
):
    """
    Chọn nghiệm thỏa hiệp bằng TOPSIS thật.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)

    criteria = [
        "Growth",
        "Inequality",
        "Emission",
        "DataRisk",
    ]
    benefit_flags = [
        True,
        False,
        False,
        False,
    ]

    scores = topsis_score(
        pareto_df,
        criteria,
        weights,
        benefit_flags,
    )

    best_position = int(
        np.argmax(scores)
    )

    result = pareto_df.copy()
    result["TOPSIS"] = scores
    result["Rank"] = (
        result["TOPSIS"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result, best_position


def _b7_solution_table(
    decision_vector,
):
    regions, items, _, _ = region_beta_matrix()
    x = _b7_decode(decision_vector)

    table = pd.DataFrame(
        x,
        columns=items,
    )
    table.insert(
        0,
        "Vùng",
        regions,
    )
    table["Tổng vùng"] = x.sum(axis=1)

    return table


def page_7():
    hero(
        "Bài 7 — Tối ưu đa mục tiêu Pareto bằng NSGA-II",
        "Chạy NSGA-II thật với 24 biến và 4 mục tiêu; xây dựng tập Pareto, chọn nghiệm thỏa hiệp bằng TOPSIS và lượng hóa chi phí cơ hội chính sách.",
        ["7.1-7.5", "NSGA-II", "Pareto", "TOPSIS", "4 objectives"],
    )

    st.markdown("## 7.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        Phân bổ ngân sách chuyển đổi số theo vùng tạo ra nhiều đánh đổi:
        tăng trưởng cao có thể làm gia tăng tập trung, phát thải từ hạ tầng số
        hoặc rủi ro dữ liệu. Vì vậy không tồn tại một nghiệm tối ưu duy nhất;
        nhà hoạch định cần quan sát tập nghiệm Pareto và chọn phương án thỏa hiệp.
        """
    )

    st.markdown("## 7.2. Mô hình đa mục tiêu")
    st.latex(
        r"\max f_1(x)=\sum_{r,j}\beta_{rj}x_{rj}"
    )
    st.latex(
        r"\min f_2(x)=Gini(D_r+\gamma x_{D,r})"
    )
    st.latex(
        r"\min f_3(x)=\sum_{r,j}e_jx_{rj}"
    )
    st.latex(
        r"\min f_4(x)=\sum_{r,j}\rho_jx_{rj}"
    )

    st.markdown(
        """
        Ràng buộc giữ nguyên logic Bài 4: tổng ngân sách 50.000,
        mỗi vùng từ 5.000 đến 12.000, nhân lực tối thiểu 12.000,
        và công bằng Digital Index. Do λ=0,70 không khả thi với dữ liệu gốc,
        NSGA-II sử dụng λ=0,68 và báo rõ đây là kịch bản hiệu chỉnh.
        """
    )

    st.markdown("## 7.3. Cấu hình NSGA-II")

    c1, c2, c3 = st.columns(3)

    population_size = c1.select_slider(
        "Population size",
        options=[40, 60, 80, 100, 120],
        value=100,
        key="b7_population_size",
    )

    generations = c2.select_slider(
        "Số thế hệ",
        options=[50, 100, 150, 200],
        value=200,
        key="b7_generations",
    )

    seed = c3.number_input(
        "Random seed",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
        key="b7_seed",
    )

    with st.spinner(
        "Đang chạy NSGA-II và xây dựng tập Pareto..."
    ):
        pareto_df, decision_matrix = _b7_run_nsga2(
            population_size=int(population_size),
            generations=int(generations),
            seed=int(seed),
            fairness_lambda=0.68,
        )

    if pareto_df.empty:
        st.error(
            "NSGA-II không tạo được nghiệm khả thi. "
            "Hãy kiểm tra pymoo hoặc giảm yêu cầu công bằng."
        )
        return

    kpi_cards(
        [
            (
                "Số nghiệm Pareto",
                f"{len(pareto_df):,}",
                "NSGA-II không trội",
            ),
            (
                "Growth lớn nhất",
                f"{pareto_df['Growth'].max():,.1f}",
                "mục tiêu 1",
            ),
            (
                "Inequality thấp nhất",
                f"{pareto_df['Inequality'].min():.4f}",
                "mục tiêu 2",
            ),
            (
                "Fairness thấp nhất",
                f"{pareto_df['FairnessRatio'].min():.4f}",
                "phải ≥ 0,68",
            ),
        ]
    )

    st.markdown("## 7.4. Kết quả lập trình")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "7.4.1 - Tập Pareto",
            "7.4.2 - TOPSIS thỏa hiệp",
            "7.4.3 - Phân bổ nghiệm chọn",
            "7.4.4 - Chi phí cơ hội",
        ]
    )

    with tab1:
        fig = px.scatter_3d(
            pareto_df,
            x="Growth",
            y="Emission",
            z="DataRisk",
            color="Inequality",
            hover_data=[
                "SolutionID",
                "FairnessRatio",
            ],
            template=PLOT_TEMPLATE,
            title="Tập Pareto NSGA-II",
        )
        fig.update_layout(
            height=620,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        parallel = px.parallel_coordinates(
            pareto_df[
                [
                    "Growth",
                    "Inequality",
                    "Emission",
                    "DataRisk",
                    "FairnessRatio",
                ]
            ],
            color="Growth",
            title="Quan hệ đánh đổi giữa bốn mục tiêu",
        )
        parallel.update_layout(
            height=540,
        )
        st.plotly_chart(
            parallel,
            use_container_width=True,
        )

        st.dataframe(
            pareto_df.sort_values(
                "Growth",
                ascending=False,
            ).head(30),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.markdown(
            "### Chọn trọng số TOPSIS"
        )

        cols = st.columns(4)
        labels = [
            "Growth",
            "Inequality",
            "Emission",
            "Data risk",
        ]
        defaults = [
            0.40,
            0.25,
            0.20,
            0.15,
        ]

        weights = np.array(
            [
                col.slider(
                    label,
                    0.05,
                    0.70,
                    float(default),
                    0.05,
                    key=f"b7_weight_{i}",
                )
                for i, (
                    col,
                    label,
                    default,
                ) in enumerate(
                    zip(
                        cols,
                        labels,
                        defaults,
                    )
                )
            ],
            dtype=float,
        )

        ranked_df, best_position = (
            _b7_topsis_compromise(
                pareto_df,
                weights,
            )
        )

        best_row = ranked_df.iloc[
            best_position
        ]

        kpi_cards(
            [
                (
                    "Solution ID",
                    str(
                        int(
                            best_row["SolutionID"]
                        )
                    ),
                    "nghiệm TOPSIS",
                ),
                (
                    "Growth",
                    f"{best_row['Growth']:,.1f}",
                    "càng cao càng tốt",
                ),
                (
                    "Inequality",
                    f"{best_row['Inequality']:.4f}",
                    "càng thấp càng tốt",
                ),
                (
                    "TOPSIS",
                    f"{best_row['TOPSIS']:.4f}",
                    "hệ số gần lý tưởng",
                ),
            ]
        )

        top_ranked = ranked_df.sort_values(
            "Rank"
        ).head(15)

        st.dataframe(
            top_ranked[
                [
                    "SolutionID",
                    "Growth",
                    "Inequality",
                    "Emission",
                    "DataRisk",
                    "FairnessRatio",
                    "TOPSIS",
                    "Rank",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        ranked_df, best_position = (
            _b7_topsis_compromise(
                pareto_df,
                np.array(
                    [0.40, 0.25, 0.20, 0.15],
                    dtype=float,
                ),
            )
        )

        chosen_vector = decision_matrix[
            best_position
        ]

        allocation = _b7_solution_table(
            chosen_vector
        )

        st.dataframe(
            allocation,
            use_container_width=True,
            hide_index=True,
        )

        long_df = allocation.melt(
            id_vars=[
                "Vùng",
                "Tổng vùng",
            ],
            value_vars=[
                col
                for col in allocation.columns
                if col not in [
                    "Vùng",
                    "Tổng vùng",
                ]
            ],
            var_name="Hạng mục",
            value_name="Ngân sách",
        )

        fig = px.bar(
            long_df,
            x="Vùng",
            y="Ngân sách",
            color="Hạng mục",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="Phân bổ của nghiệm thỏa hiệp",
        )
        fig.update_layout(
            height=520,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        constraints = _b7_constraint_values(
            chosen_vector,
            fairness_lambda=0.68,
        )

        st.success(
            "Nghiệm thỏa toàn bộ ràng buộc."
            if np.max(constraints) <= 1e-5
            else "Có ràng buộc chưa đạt do sai số thuật toán."
        )

    with tab4:
        max_growth_row = pareto_df.loc[
            pareto_df["Growth"].idxmax()
        ]
        min_inequality_row = pareto_df.loc[
            pareto_df["Inequality"].idxmin()
        ]
        min_emission_row = pareto_df.loc[
            pareto_df["Emission"].idxmin()
        ]
        min_risk_row = pareto_df.loc[
            pareto_df["DataRisk"].idxmin()
        ]

        ranked_df, best_position = (
            _b7_topsis_compromise(
                pareto_df,
                np.array(
                    [0.40, 0.25, 0.20, 0.15],
                    dtype=float,
                ),
            )
        )
        compromise = ranked_df.iloc[
            best_position
        ]

        opportunity = pd.DataFrame(
            {
                "Phương án": [
                    "Tăng trưởng cực đại",
                    "Bất bình đẳng thấp nhất",
                    "Phát thải thấp nhất",
                    "Rủi ro dữ liệu thấp nhất",
                    "Nghiệm thỏa hiệp",
                ],
                "Growth": [
                    max_growth_row["Growth"],
                    min_inequality_row["Growth"],
                    min_emission_row["Growth"],
                    min_risk_row["Growth"],
                    compromise["Growth"],
                ],
                "Inequality": [
                    max_growth_row["Inequality"],
                    min_inequality_row["Inequality"],
                    min_emission_row["Inequality"],
                    min_risk_row["Inequality"],
                    compromise["Inequality"],
                ],
                "Emission": [
                    max_growth_row["Emission"],
                    min_inequality_row["Emission"],
                    min_emission_row["Emission"],
                    min_risk_row["Emission"],
                    compromise["Emission"],
                ],
                "DataRisk": [
                    max_growth_row["DataRisk"],
                    min_inequality_row["DataRisk"],
                    min_emission_row["DataRisk"],
                    min_risk_row["DataRisk"],
                    compromise["DataRisk"],
                ],
            }
        )

        opportunity[
            "Chi phí tăng trưởng so với cực đại"
        ] = (
            max_growth_row["Growth"]
            - opportunity["Growth"]
        )

        st.dataframe(
            opportunity,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            opportunity,
            x="Phương án",
            y="Chi phí tăng trưởng so với cực đại",
            template=PLOT_TEMPLATE,
            title="Chi phí cơ hội khi ưu tiên mục tiêu khác",
        )
        fig.update_layout(
            height=450,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.download_button(
        "Tải tập Pareto Bài 7",
        data=pareto_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai7_pareto_nsga2.csv",
        mime="text/csv",
        key="download_bai7_nsga2",
    )

    st.markdown("## 7.5. Thảo luận chính sách")

    with st.expander(
        "a) Vì sao không chọn trực tiếp nghiệm tăng trưởng cao nhất?",
        expanded=True,
    ):
        st.markdown(
            "Nghiệm tăng trưởng cao nhất có thể phải đánh đổi bằng bất bình đẳng, "
            "phát thải hoặc rủi ro dữ liệu lớn. Tập Pareto buộc nhà hoạch định "
            "trình bày công khai các đánh đổi thay vì che giấu chúng trong một điểm tổng hợp."
        )

    with st.expander(
        "b) TOPSIS có thay thế quyết định chính trị không?",
        expanded=True,
    ):
        st.markdown(
            "Không. TOPSIS chỉ chuyển hệ trọng số chính sách thành một nghiệm thỏa hiệp. "
            "Kết quả phải được kiểm tra độ nhạy và tham vấn các bên chịu tác động."
        )

    with st.expander(
        "c) Vì sao dùng λ=0,68?",
        expanded=True,
    ):
        st.markdown(
            "λ=0,70 không khả thi với Digital Index và trần vùng hiện tại. "
            "Bài 7 dùng λ=0,68 sau khi Bài 4 đã chứng minh ngưỡng khả thi, "
            "không phải tự ý thay đổi tham số mà không giải thích."
        )


def _b8_initial_state():
    """
    Lấy trạng thái cuối năm 2025 từ hàm compute_tfp().

    Thứ tự đầu ra của compute_tfp():
    years, Y, K, L, D, AI, H, A.
    """
    (
        years,
        Y,
        K,
        L,
        D,
        AI,
        H,
        A,
    ) = compute_tfp()

    return {
        "year": int(years[-1]),
        "Y": float(Y[-1]),
        "K": float(K[-1]),
        "L": float(L[-1]),
        "D": float(D[-1]),
        "AI": float(AI[-1]),
        "H": float(H[-1]),
        "A": float(A[-1]),
    }


def _b8_parameters():
    """Bộ tham số động học dùng trong mô phỏng."""
    return {
        "start_year": 2026,
        "end_year": 2035,
        "investment_rate": 0.22,
        "delta_K": 0.05,
        "delta_D": 0.12,
        "delta_AI": 0.15,
        "mu_H": 0.02,
        "theta_H": 0.80,
        "scale_D": 240.0,
        "scale_AI": 135.0,
        "scale_H": 520.0,
        "labor_growth": 0.006,
        "tfp_base_growth": 0.0,
        "share_lower_bound": 0.02,
        "share_upper_bound": 0.85,
    }


def _b8_simulate(
    shares_matrix,
    invest_rates=None,
    shock_2028=0.0,
    rho=0.97,
):
    """
    Mô phỏng động giai đoạn 2026-2035.

    shares_matrix:
        Ma trận 10x4, mỗi hàng là tỷ trọng đầu tư vào K, D, AI, H.

    invest_rates:
        Tỷ lệ đầu tư trên GDP từng năm. Nếu None, dùng 22%/năm.

    shock_2028:
        Mức giảm GDP năm 2028. Ví dụ 0.08 tương ứng giảm 8%.

    rho:
        Hệ số chiết khấu phúc lợi.
    """
    initial = _b8_initial_state()
    parameters = _b8_parameters()

    years = np.arange(
        parameters["start_year"],
        parameters["end_year"] + 1,
    )
    periods = len(years)

    shares_matrix = np.asarray(
        shares_matrix,
        dtype=float,
    )

    if shares_matrix.shape != (periods, 4):
        raise ValueError(
            "shares_matrix phải có kích thước 10x4."
        )

    if not np.isfinite(shares_matrix).all():
        raise ValueError(
            "shares_matrix chứa NaN hoặc giá trị vô hạn."
        )

    row_sum = shares_matrix.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sum <= 0):
        raise ValueError(
            "Mỗi năm phải có tổng tỷ trọng đầu tư lớn hơn 0."
        )

    shares_matrix = (
        shares_matrix
        / row_sum
    )

    if invest_rates is None:
        invest_rates = np.full(
            periods,
            parameters["investment_rate"],
            dtype=float,
        )
    else:
        invest_rates = np.asarray(
            invest_rates,
            dtype=float,
        )

    if invest_rates.shape != (periods,):
        raise ValueError(
            "invest_rates phải có đúng 10 phần tử."
        )

    if (
        np.any(invest_rates <= 0)
        or np.any(invest_rates >= 1)
    ):
        raise ValueError(
            "Mỗi tỷ lệ đầu tư phải nằm trong khoảng (0,1)."
        )

    if not 0 < float(rho) <= 1:
        raise ValueError(
            "Hệ số chiết khấu rho phải nằm trong khoảng (0,1]."
        )

    if not 0 <= float(shock_2028) < 1:
        raise ValueError(
            "Cú sốc 2028 phải nằm trong khoảng [0,1)."
        )

    # Trạng thái đầu năm 2026.
    K = initial["K"] * 1.06
    L = initial["L"] * (
        1 + parameters["labor_growth"]
    )
    D = initial["D"] + 0.80
    AI = initial["AI"] + 6.00
    H = initial["H"] + 0.80
    A = initial["A"] * 1.012

    welfare = 0.0
    rows = []

    for period, year in enumerate(years):
        gdp_before_shock = (
            A
            * K**0.33
            * L**0.42
            * D**0.10
            * AI**0.08
            * H**0.07
        )

        shock_factor = (
            1.0 - float(shock_2028)
            if year == 2028
            else 1.0
        )

        gdp = max(
            gdp_before_shock * shock_factor,
            1e-9,
        )

        investment_rate = float(
            invest_rates[period]
        )

        total_investment = (
            investment_rate
            * gdp
        )

        consumption = max(
            gdp - total_investment,
            1e-9,
        )

        period_utility = (
            float(rho) ** period
            * np.log(consumption)
        )

        welfare += period_utility

        shares = shares_matrix[period]

        investment_k = (
            shares[0]
            * total_investment
        )
        investment_d = (
            shares[1]
            * total_investment
        )
        investment_ai = (
            shares[2]
            * total_investment
        )
        investment_h = (
            shares[3]
            * total_investment
        )

        rows.append(
            [
                year,
                gdp,
                gdp_before_shock,
                consumption,
                total_investment,
                investment_rate,
                K,
                L,
                D,
                AI,
                H,
                A,
                investment_k,
                investment_d,
                investment_ai,
                investment_h,
                shares[0],
                shares[1],
                shares[2],
                shares[3],
                period_utility,
                welfare,
            ]
        )

        # Phương trình chuyển trạng thái.
        K = (
            (1 - parameters["delta_K"]) * K
            + investment_k
        )

        D = max(
            1e-6,
            (1 - parameters["delta_D"]) * D
            + investment_d / parameters["scale_D"],
        )

        AI = max(
            1e-6,
            (1 - parameters["delta_AI"]) * AI
            + investment_ai / parameters["scale_AI"],
        )

        H = max(
            1e-6,
            H
            + parameters["theta_H"]
            * investment_h
            / parameters["scale_H"]
            - parameters["mu_H"] * H,
        )

        # TFP tăng nội sinh theo số hóa, AI và nhân lực.
        A = (
            A
            * (
                1
                + parameters["tfp_base_growth"]
                + 0.00008 * D
                + 0.00004 * AI
                + 0.00006 * H
            )
        )

        L *= (
            1
            + parameters["labor_growth"]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Năm",
            "GDP",
            "GDP trước cú sốc",
            "Tiêu dùng",
            "Tổng đầu tư",
            "Tỷ lệ đầu tư",
            "K",
            "L",
            "D",
            "AI",
            "H",
            "A",
            "I_K",
            "I_D",
            "I_AI",
            "I_H",
            "Share_K",
            "Share_D",
            "Share_AI",
            "Share_H",
            "Phúc lợi kỳ",
            "Welfare_lũy_kế",
        ],
    )


def _b8_validation_table(
    shares_matrix,
    simulation,
):
    """Kiểm tra tính hợp lệ của nghiệm sau tối ưu."""
    parameters = _b8_parameters()

    shares_matrix = np.asarray(
        shares_matrix,
        dtype=float,
    )

    row_sums = shares_matrix.sum(
        axis=1
    )

    checks = [
        {
            "Kiểm tra": "Đúng 10 năm 2026-2035",
            "Giá trị": (
                f"{int(simulation['Năm'].min())}-"
                f"{int(simulation['Năm'].max())}"
            ),
            "Đạt": (
                len(simulation) == 10
                and simulation["Năm"].min() == 2026
                and simulation["Năm"].max() == 2035
            ),
        },
        {
            "Kiểm tra": "Tổng tỷ trọng mỗi năm bằng 1",
            "Giá trị": (
                f"max sai lệch = "
                f"{np.max(np.abs(row_sums - 1.0)):.2e}"
            ),
            "Đạt": bool(
                np.allclose(
                    row_sums,
                    1.0,
                    atol=1e-6,
                )
            ),
        },
        {
            "Kiểm tra": "Tỷ trọng không dưới cận 0,02",
            "Giá trị": f"{shares_matrix.min():.6f}",
            "Đạt": bool(
                shares_matrix.min()
                >= parameters["share_lower_bound"]
                - 1e-6
            ),
        },
        {
            "Kiểm tra": "Tỷ trọng không vượt cận 0,85",
            "Giá trị": f"{shares_matrix.max():.6f}",
            "Đạt": bool(
                shares_matrix.max()
                <= parameters["share_upper_bound"]
                + 1e-6
            ),
        },
        {
            "Kiểm tra": "GDP dương và hữu hạn",
            "Giá trị": f"min = {simulation['GDP'].min():,.3f}",
            "Đạt": bool(
                np.isfinite(
                    simulation["GDP"]
                ).all()
                and (
                    simulation["GDP"] > 0
                ).all()
            ),
        },
        {
            "Kiểm tra": "Tiêu dùng dương và hữu hạn",
            "Giá trị": (
                f"min = "
                f"{simulation['Tiêu dùng'].min():,.3f}"
            ),
            "Đạt": bool(
                np.isfinite(
                    simulation["Tiêu dùng"]
                ).all()
                and (
                    simulation["Tiêu dùng"] > 0
                ).all()
            ),
        },
        {
            "Kiểm tra": "Các biến trạng thái dương",
            "Giá trị": (
                f"min = "
                f"{simulation[['K','L','D','AI','H','A']].min().min():.6f}"
            ),
            "Đạt": bool(
                (
                    simulation[
                        [
                            "K",
                            "L",
                            "D",
                            "AI",
                            "H",
                            "A",
                        ]
                    ] > 0
                ).all().all()
            ),
        },
    ]

    return pd.DataFrame(checks)


@st.cache_data(show_spinner=False)
def _b8_optimize_shares(
    rho=0.97,
    shock_2028=0.0,
):
    """
    Tối ưu 40 biến tỷ trọng bằng SLSQP.

    Mỗi năm:
        share_K + share_D + share_AI + share_H = 1.
    """
    from scipy.optimize import minimize

    parameters = _b8_parameters()
    periods = 10

    initial_share = np.array(
        [0.34, 0.26, 0.18, 0.22],
        dtype=float,
    )

    x0 = np.tile(
        initial_share,
        periods,
    )

    investment_rates = np.full(
        periods,
        parameters["investment_rate"],
        dtype=float,
    )

    def objective(flat_shares):
        shares = flat_shares.reshape(
            periods,
            4,
        )

        simulation = _b8_simulate(
            shares_matrix=shares,
            invest_rates=investment_rates,
            shock_2028=shock_2028,
            rho=rho,
        )

        welfare = float(
            simulation.iloc[-1][
                "Welfare_lũy_kế"
            ]
        )

        if not np.isfinite(welfare):
            return 1e12

        return -welfare

    constraints = [
        {
            "type": "eq",
            "fun": (
                lambda flat_shares, period=period:
                np.sum(
                    flat_shares.reshape(
                        periods,
                        4,
                    )[period]
                )
                - 1.0
            ),
        }
        for period in range(periods)
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (
                parameters["share_lower_bound"],
                parameters["share_upper_bound"],
            )
        ]
        * (
            periods * 4
        ),
        constraints=constraints,
        options={
            "maxiter": 500,
            "ftol": 1e-9,
            "disp": False,
        },
    )

    candidate = (
        np.asarray(
            result.x,
            dtype=float,
        )
        if result.x is not None
        else x0.copy()
    )

    optimized_shares = candidate.reshape(
        periods,
        4,
    )

    row_sums = optimized_shares.sum(
        axis=1,
        keepdims=True,
    )

    optimized_shares = (
        optimized_shares
        / np.where(
            row_sums == 0,
            1.0,
            row_sums,
        )
    )

    simulation = _b8_simulate(
        shares_matrix=optimized_shares,
        invest_rates=investment_rates,
        shock_2028=shock_2028,
        rho=rho,
    )

    max_equality_violation = float(
        np.max(
            np.abs(
                optimized_shares.sum(axis=1)
                - 1.0
            )
        )
    )

    success = bool(
        result.success
        and np.isfinite(
            simulation[
                "Welfare_lũy_kế"
            ]
        ).all()
        and max_equality_violation <= 1e-5
    )

    solver_info = {
        "success": success,
        "message": str(result.message),
        "status": int(result.status),
        "iterations": int(
            getattr(result, "nit", 0)
        ),
        "objective": float(
            -result.fun
        )
        if np.isfinite(result.fun)
        else np.nan,
        "max_equality_violation": (
            max_equality_violation
        ),
    }

    return (
        optimized_shares,
        simulation,
        solver_info,
    )


def _b8_strategy_comparison(
    rho=0.97,
):
    """So sánh đầu tư trải đều và front-load với cùng tổng tỷ lệ đầu tư."""
    fixed_shares = np.tile(
        np.array(
            [0.34, 0.26, 0.18, 0.22],
            dtype=float,
        ),
        (10, 1),
    )

    equal_rates = np.full(
        10,
        0.22,
        dtype=float,
    )

    front_load_rates = np.array(
        [
            0.28,
            0.27,
            0.26,
            0.24,
            0.22,
            0.21,
            0.19,
            0.18,
            0.18,
            0.17,
        ],
        dtype=float,
    )

    equal_simulation = _b8_simulate(
        shares_matrix=fixed_shares,
        invest_rates=equal_rates,
        shock_2028=0.0,
        rho=rho,
    )

    front_simulation = _b8_simulate(
        shares_matrix=fixed_shares,
        invest_rates=front_load_rates,
        shock_2028=0.0,
        rho=rho,
    )

    comparison = pd.DataFrame(
        {
            "Chiến lược": [
                "Trải đều",
                "Front-load",
            ],
            "Tổng tỷ lệ đầu tư 10 năm": [
                equal_rates.sum(),
                front_load_rates.sum(),
            ],
            "Welfare": [
                equal_simulation.iloc[-1][
                    "Welfare_lũy_kế"
                ],
                front_simulation.iloc[-1][
                    "Welfare_lũy_kế"
                ],
            ],
            "GDP 2035": [
                equal_simulation.iloc[-1]["GDP"],
                front_simulation.iloc[-1]["GDP"],
            ],
            "Tiêu dùng 2035": [
                equal_simulation.iloc[-1][
                    "Tiêu dùng"
                ],
                front_simulation.iloc[-1][
                    "Tiêu dùng"
                ],
            ],
        }
    )

    return (
        comparison,
        equal_simulation,
        front_simulation,
        equal_rates,
        front_load_rates,
    )


def page_8():
    hero(
        "Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
        "Mô hình hóa động học K-D-AI-H, tối ưu 40 biến tỷ trọng bằng SLSQP, kiểm định nghiệm, phân tích cú sốc 2028 và so sánh đầu tư trải đều với front-load.",
        [
            "8.1-8.5",
            "Dynamic optimization",
            "SLSQP",
            "Welfare",
            "Shock 2028",
        ],
    )

    # =====================================================
    # 8.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown(
        "## 8.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        Đầu tư số có tác động tích lũy và độ trễ. Đầu tư mạnh vào AI ở hiện tại
        có thể chưa tạo lợi ích nếu thiếu nhân lực số; ngược lại, đầu tư vào nhân lực
        và hạ tầng tạo nền tảng hấp thụ công nghệ trong dài hạn.

        Bài 8 xây dựng mô hình tối ưu động giai đoạn **2026-2035** nhằm lựa chọn
        quỹ đạo phân bổ đầu tư vào vốn vật chất, chuyển đổi số, AI và nhân lực số,
        sao cho tổng phúc lợi tiêu dùng có chiết khấu đạt mức cao nhất.
        """
    )

    # =====================================================
    # 8.2. Mô hình toán học
    # =====================================================
    st.markdown(
        "## 8.2. Mô hình toán học"
    )

    st.markdown(
        "### Hàm mục tiêu liên thời gian"
    )

    st.latex(
        r"\max\ "
        r"\sum_{t=2026}^{2035}"
        r"\rho^{t-2026}\ln(C_t)"
    )

    st.markdown("### Hàm sản xuất")

    st.latex(
        r"Y_t="
        r"A_tK_t^{0.33}L_t^{0.42}"
        r"D_t^{0.10}AI_t^{0.08}H_t^{0.07}"
    )

    st.markdown(
        "### Phương trình chuyển trạng thái"
    )

    st.latex(
        r"K_{t+1}=(1-\delta_K)K_t+I_{K,t}"
    )
    st.latex(
        r"D_{t+1}=(1-\delta_D)D_t+I_{D,t}/s_D"
    )
    st.latex(
        r"AI_{t+1}=(1-\delta_{AI})AI_t+I_{AI,t}/s_{AI}"
    )
    st.latex(
        r"H_{t+1}=H_t+\theta_HI_{H,t}/s_H-\mu_HH_t"
    )
    st.latex(
        r"C_t=Y_t-I_t,\qquad "
        r"I_{j,t}=s_{j,t}I_t"
    )
    st.latex(
        r"\sum_js_{j,t}=1,\qquad "
        r"0.02\leq s_{j,t}\leq0.85"
    )

    st.info(
        "SLSQP tối ưu 40 biến: bốn tỷ trọng đầu tư cho mỗi năm trong 10 năm."
    )

    # =====================================================
    # 8.3. Dữ liệu và tham số
    # =====================================================
    st.markdown(
        "## 8.3. Dữ liệu đầu vào và tham số"
    )

    initial = _b8_initial_state()
    parameters = _b8_parameters()

    initial_table = pd.DataFrame(
        {
            "Biến": [
                "GDP Y",
                "Vốn K",
                "Lao động L",
                "Số hóa D",
                "Năng lực AI",
                "Nhân lực H",
                "TFP A",
            ],
            "Giá trị cuối năm 2025": [
                initial["Y"],
                initial["K"],
                initial["L"],
                initial["D"],
                initial["AI"],
                initial["H"],
                initial["A"],
            ],
        }
    )

    parameter_table = pd.DataFrame(
        {
            "Tham số": [
                "Tỷ lệ đầu tư/GDP",
                "Khấu hao K",
                "Khấu hao D",
                "Khấu hao AI",
                "Suy giảm H",
                "Hiệu quả đầu tư H",
                "Hệ số quy đổi D",
                "Hệ số quy đổi AI",
                "Hệ số quy đổi H",
                "Tăng lao động",
                "Cận dưới tỷ trọng",
                "Cận trên tỷ trọng",
            ],
            "Ký hiệu": [
                "i",
                "delta_K",
                "delta_D",
                "delta_AI",
                "mu_H",
                "theta_H",
                "s_D",
                "s_AI",
                "s_H",
                "g_L",
                "lb",
                "ub",
            ],
            "Giá trị": [
                parameters["investment_rate"],
                parameters["delta_K"],
                parameters["delta_D"],
                parameters["delta_AI"],
                parameters["mu_H"],
                parameters["theta_H"],
                parameters["scale_D"],
                parameters["scale_AI"],
                parameters["scale_H"],
                parameters["labor_growth"],
                parameters["share_lower_bound"],
                parameters["share_upper_bound"],
            ],
        }
    )

    c1, c2 = st.columns(2)

    with c1:
        st.dataframe(
            initial_table,
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.dataframe(
            parameter_table,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Trạng thái năm 2025 được lấy từ dữ liệu vĩ mô và hàm Cobb-Douglas của Bài 1. "
        "Các tham số động học là giả định mô phỏng, cần được nêu rõ trong báo cáo."
    )

    # =====================================================
    # 8.4. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 8.4. Yêu cầu lập trình"
    )

    rho = st.slider(
        "Hệ số chiết khấu phúc lợi rho",
        min_value=0.90,
        max_value=1.00,
        value=0.97,
        step=0.01,
        key="b8_rho_fixed",
    )

    with st.spinner(
        "Đang tối ưu quỹ đạo đầu tư 2026-2035..."
    ):
        (
            optimal_shares,
            optimal_simulation,
            solver_info,
        ) = _b8_optimize_shares(
            rho=float(rho),
            shock_2028=0.0,
        )

    tab841, tab842, tab843, tab844 = st.tabs(
        [
            "8.4.1 - Tối ưu SLSQP",
            "8.4.2 - Quỹ đạo động",
            "8.4.3 - Cú sốc 2028",
            "8.4.4 - Front-load & độ nhạy",
        ]
    )

    # -----------------------------------------------------
    # 8.4.1
    # -----------------------------------------------------
    with tab841:
        st.markdown(
            "### Câu 8.4.1. Giải bài toán phi tuyến bằng SLSQP"
        )

        shares_table = pd.DataFrame(
            optimal_shares,
            columns=[
                "Share_K",
                "Share_D",
                "Share_AI",
                "Share_H",
            ],
        )

        shares_table.insert(
            0,
            "Năm",
            np.arange(2026, 2036),
        )

        kpi_cards(
            [
                (
                    "Trạng thái solver",
                    (
                        "Thành công"
                        if solver_info["success"]
                        else "Cảnh báo"
                    ),
                    solver_info["message"][:45],
                ),
                (
                    "Số vòng lặp",
                    str(
                        solver_info["iterations"]
                    ),
                    "SLSQP",
                ),
                (
                    "Welfare tối ưu",
                    f"{optimal_simulation.iloc[-1]['Welfare_lũy_kế']:.4f}",
                    f"rho={rho:.2f}",
                ),
                (
                    "GDP năm 2035",
                    f"{optimal_simulation.iloc[-1]['GDP']:,.1f}",
                    "nghìn tỷ VND",
                ),
            ]
        )

        st.dataframe(
            shares_table.style.format(
                {
                    "Share_K": "{:.4f}",
                    "Share_D": "{:.4f}",
                    "Share_AI": "{:.4f}",
                    "Share_H": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        shares_long = shares_table.melt(
            id_vars="Năm",
            value_vars=[
                "Share_K",
                "Share_D",
                "Share_AI",
                "Share_H",
            ],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )

        fig_shares = px.area(
            shares_long,
            x="Năm",
            y="Tỷ trọng",
            color="Hạng mục",
            template=PLOT_TEMPLATE,
            title="Quỹ đạo tỷ trọng đầu tư tối ưu",
        )
        fig_shares.update_layout(
            height=480,
            yaxis_title="Tỷ trọng đầu tư",
            xaxis_title="Năm",
        )

        st.plotly_chart(
            fig_shares,
            use_container_width=True,
        )

        st.markdown(
            "#### Kiểm định nghiệm sau tối ưu"
        )

        validation = _b8_validation_table(
            optimal_shares,
            optimal_simulation,
        )

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True,
        )

        if bool(validation["Đạt"].all()):
            st.success(
                "Nghiệm vượt qua toàn bộ kiểm tra số và ràng buộc."
            )
        else:
            st.error(
                "Có ít nhất một kiểm tra chưa đạt; chưa nên sử dụng kết quả làm bản final."
            )

    # -----------------------------------------------------
    # 8.4.2
    # -----------------------------------------------------
    with tab842:
        st.markdown(
            "### Câu 8.4.2. Vẽ quỹ đạo tối ưu K, D, AI, H, Y và C"
        )

        c1, c2 = st.columns(2)

        with c1:
            fig_output = px.line(
                optimal_simulation,
                x="Năm",
                y=[
                    "GDP",
                    "Tiêu dùng",
                    "Tổng đầu tư",
                ],
                markers=True,
                template=PLOT_TEMPLATE,
                title="GDP, tiêu dùng và đầu tư",
            )
            fig_output.update_layout(
                height=470,
                xaxis_title="Năm",
                yaxis_title="Nghìn tỷ VND",
            )
            st.plotly_chart(
                fig_output,
                use_container_width=True,
            )

        with c2:
            state_long = optimal_simulation.melt(
                id_vars="Năm",
                value_vars=[
                    "K",
                    "D",
                    "AI",
                    "H",
                ],
                var_name="Biến trạng thái",
                value_name="Giá trị",
            )

            fig_state = px.line(
                state_long,
                x="Năm",
                y="Giá trị",
                color="Biến trạng thái",
                markers=True,
                template=PLOT_TEMPLATE,
                title="Quỹ đạo K, D, AI và H",
            )
            fig_state.update_layout(
                height=470,
                xaxis_title="Năm",
                yaxis_title="Chỉ số/giá trị mô phỏng",
            )
            st.plotly_chart(
                fig_state,
                use_container_width=True,
            )

        st.dataframe(
            optimal_simulation,
            use_container_width=True,
            hide_index=True,
        )

        gdp_cagr = (
            (
                optimal_simulation.iloc[-1]["GDP"]
                / optimal_simulation.iloc[0]["GDP"]
            )
            ** (1 / 9)
            - 1
        ) * 100

        st.info(
            f"GDP mô phỏng tăng bình quân khoảng **{gdp_cagr:.2f}%/năm** "
            "trong giai đoạn 2026-2035 theo quỹ đạo tối ưu."
        )

    # -----------------------------------------------------
    # 8.4.3
    # -----------------------------------------------------
    with tab843:
        st.markdown(
            "### Câu 8.4.3. Cú sốc GDP năm 2028 giảm 8% và tối ưu lại"
        )

        with st.spinner(
            "Đang tối ưu lại sau cú sốc 2028..."
        ):
            (
                shock_shares,
                shock_simulation,
                shock_solver_info,
            ) = _b8_optimize_shares(
                rho=float(rho),
                shock_2028=0.08,
            )

        comparison_shock = pd.DataFrame(
            {
                "Năm": optimal_simulation["Năm"],
                "GDP cơ sở": optimal_simulation["GDP"],
                "GDP sau cú sốc": shock_simulation["GDP"],
                "Tiêu dùng cơ sở": optimal_simulation["Tiêu dùng"],
                "Tiêu dùng sau cú sốc": shock_simulation["Tiêu dùng"],
            }
        )

        welfare_loss = (
            optimal_simulation.iloc[-1][
                "Welfare_lũy_kế"
            ]
            - shock_simulation.iloc[-1][
                "Welfare_lũy_kế"
            ]
        )

        gdp_2035_change = (
            shock_simulation.iloc[-1]["GDP"]
            - optimal_simulation.iloc[-1]["GDP"]
        )

        kpi_cards(
            [
                (
                    "Solver cú sốc",
                    (
                        "Thành công"
                        if shock_solver_info["success"]
                        else "Cảnh báo"
                    ),
                    shock_solver_info["message"][:45],
                ),
                (
                    "Welfare mất đi",
                    f"{welfare_loss:.4f}",
                    "so với cơ sở",
                ),
                (
                    "GDP 2035 thay đổi",
                    f"{gdp_2035_change:+,.1f}",
                    "nghìn tỷ VND",
                ),
                (
                    "Cú sốc năm 2028",
                    "-8%",
                    "GDP trong năm",
                ),
            ]
        )

        fig_shock = px.line(
            comparison_shock,
            x="Năm",
            y=[
                "GDP cơ sở",
                "GDP sau cú sốc",
            ],
            markers=True,
            template=PLOT_TEMPLATE,
            title="Quỹ đạo GDP trước và sau cú sốc 2028",
        )
        fig_shock.update_layout(
            height=480,
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
        )
        st.plotly_chart(
            fig_shock,
            use_container_width=True,
        )

        share_change = pd.DataFrame(
            {
                "Năm": np.arange(2026, 2036),
                "Delta_Share_K": (
                    shock_shares[:, 0]
                    - optimal_shares[:, 0]
                ),
                "Delta_Share_D": (
                    shock_shares[:, 1]
                    - optimal_shares[:, 1]
                ),
                "Delta_Share_AI": (
                    shock_shares[:, 2]
                    - optimal_shares[:, 2]
                ),
                "Delta_Share_H": (
                    shock_shares[:, 3]
                    - optimal_shares[:, 3]
                ),
            }
        )

        st.markdown(
            "#### Thay đổi cơ cấu đầu tư sau cú sốc"
        )

        st.dataframe(
            share_change.style.format(
                {
                    "Delta_Share_K": "{:+.4f}",
                    "Delta_Share_D": "{:+.4f}",
                    "Delta_Share_AI": "{:+.4f}",
                    "Delta_Share_H": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 8.4.4
    # -----------------------------------------------------
    with tab844:
        st.markdown(
            "### Câu 8.4.4. So sánh trải đều, front-load và độ nhạy rho"
        )

        (
            strategy_comparison,
            equal_simulation,
            front_simulation,
            equal_rates,
            front_load_rates,
        ) = _b8_strategy_comparison(
            rho=float(rho)
        )

        st.dataframe(
            strategy_comparison,
            use_container_width=True,
            hide_index=True,
        )

        strategy_long = pd.concat(
            [
                equal_simulation[
                    ["Năm", "GDP"]
                ].assign(
                    Chiến_lược="Trải đều"
                ),
                front_simulation[
                    ["Năm", "GDP"]
                ].assign(
                    Chiến_lược="Front-load"
                ),
            ],
            ignore_index=True,
        )

        fig_strategy = px.line(
            strategy_long,
            x="Năm",
            y="GDP",
            color="Chiến_lược",
            markers=True,
            template=PLOT_TEMPLATE,
            title="GDP theo hai chiến lược đầu tư",
        )
        fig_strategy.update_layout(
            height=470,
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
        )
        st.plotly_chart(
            fig_strategy,
            use_container_width=True,
        )

        rates_table = pd.DataFrame(
            {
                "Năm": np.arange(2026, 2036),
                "Trải đều": equal_rates,
                "Front-load": front_load_rates,
            }
        )

        st.dataframe(
            rates_table,
            use_container_width=True,
            hide_index=True,
        )

        sensitivity_rows = []

        for rho_test in [
            0.90,
            0.95,
            0.97,
            0.99,
            1.00,
        ]:
            (
                comparison_test,
                _,
                _,
                _,
                _,
            ) = _b8_strategy_comparison(
                rho=rho_test
            )

            equal_welfare = float(
                comparison_test.loc[
                    comparison_test[
                        "Chiến lược"
                    ] == "Trải đều",
                    "Welfare",
                ].iloc[0]
            )

            front_welfare = float(
                comparison_test.loc[
                    comparison_test[
                        "Chiến lược"
                    ] == "Front-load",
                    "Welfare",
                ].iloc[0]
            )

            sensitivity_rows.append(
                {
                    "rho": rho_test,
                    "Welfare trải đều": equal_welfare,
                    "Welfare front-load": front_welfare,
                    "Chênh lệch front-load": (
                        front_welfare
                        - equal_welfare
                    ),
                    "Chiến lược tốt hơn": (
                        "Front-load"
                        if front_welfare
                        > equal_welfare
                        else "Trải đều"
                    ),
                }
            )

        sensitivity_df = pd.DataFrame(
            sensitivity_rows
        )

        st.markdown(
            "#### Độ nhạy theo hệ số chiết khấu"
        )

        st.dataframe(
            sensitivity_df,
            use_container_width=True,
            hide_index=True,
        )

        better_strategy = (
            strategy_comparison.sort_values(
                "Welfare",
                ascending=False,
            ).iloc[0]["Chiến lược"]
        )

        st.success(
            f"Với rho={rho:.2f}, chiến lược có phúc lợi cao hơn là "
            f"**{better_strategy}**."
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    shares_export = pd.DataFrame(
        optimal_shares,
        columns=[
            "Share_K",
            "Share_D",
            "Share_AI",
            "Share_H",
        ],
    )
    shares_export.insert(
        0,
        "Năm",
        np.arange(2026, 2036),
    )

    export_df = optimal_simulation.merge(
        shares_export,
        on=[
            "Năm",
            "Share_K",
            "Share_D",
            "Share_AI",
            "Share_H",
        ],
        how="left",
    )

    st.download_button(
        "Tải kết quả Bài 8 dạng CSV",
        data=export_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai8_toi_uu_dong_2026_2035.csv",
        mime="text/csv",
        key="download_bai8_fixed",
    )

    # =====================================================
    # 8.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 8.5. Câu hỏi thảo luận chính sách"
    )

    first_three_average = (
        optimal_shares[:3].mean(axis=0)
    )
    last_three_average = (
        optimal_shares[-3:].mean(axis=0)
    )

    component_names = [
        "vốn K",
        "số hóa D",
        "AI",
        "nhân lực H",
    ]

    first_priority = component_names[
        int(
            np.argmax(
                first_three_average
            )
        )
    ]

    last_priority = component_names[
        int(
            np.argmax(
                last_three_average
            )
        )
    ]

    with st.expander(
        "a) Nên ưu tiên hạng mục nào trong ba năm đầu?",
        expanded=True,
    ):
        st.markdown(
            f"Theo nghiệm tối ưu, hạng mục có tỷ trọng bình quân lớn nhất "
            f"trong ba năm đầu là **{first_priority}**. Kết quả phản ánh lợi ích "
            "tích lũy và độ trễ của đầu tư, nhưng vẫn phụ thuộc vào bộ tham số mô phỏng."
        )

    with st.expander(
        "b) Cơ cấu đầu tư thay đổi thế nào về cuối kỳ?",
        expanded=True,
    ):
        st.markdown(
            f"Trong ba năm cuối, hạng mục có tỷ trọng bình quân lớn nhất là "
            f"**{last_priority}**. Sự thay đổi giữa đầu kỳ và cuối kỳ cho thấy "
            "mô hình phân biệt đầu tư nền tảng với đầu tư khai thác thành quả."
        )

    with st.expander(
        "c) Cú sốc 2028 có làm thay đổi chính sách tối ưu không?",
        expanded=True,
    ):
        average_policy_change = float(
            np.mean(
                np.abs(
                    shock_shares
                    - optimal_shares
                )
            )
        )

        st.markdown(
            f"Cú sốc làm thay đổi tỷ trọng đầu tư bình quân tuyệt đối khoảng "
            f"**{average_policy_change:.4f}**. Nếu thay đổi nhỏ, chính sách tương đối "
            "ổn định; nếu thay đổi lớn, cần chuẩn bị cơ chế tái phân bổ ngân sách linh hoạt."
        )

    with st.expander(
        "d) Front-load có luôn tốt hơn trải đều không?",
        expanded=True,
    ):
        st.markdown(
            "Không. Front-load có thể tạo năng lực sớm và nâng sản lượng tương lai, "
            "nhưng làm giảm tiêu dùng hiện tại. Kết luận phụ thuộc vào hệ số chiết khấu, "
            "hiệu quả chuyển đổi đầu tư và tốc độ khấu hao của từng loại tài sản."
        )

    with st.expander(
        "e) Giới hạn lớn nhất của mô hình là gì?",
        expanded=True,
    ):
        st.markdown(
            "Mô hình sử dụng tham số giả định, chưa có chi phí điều chỉnh, độ trễ dự án, "
            "trần hấp thụ từng năm, nợ công, phân phối thu nhập và bất định tham số. "
            "Do đó kết quả là công cụ mô phỏng và hỗ trợ quyết định, không phải dự báo chính thức."
        )


def _b9_parameters():
    """Các tham số chung của Bài 9."""
    return {
        "total_budget": 100000.0,
        "ai_budget_share": 0.55,
        "training_budget_share": 0.45,
        "minimum_training_share": 0.30,
        "maximum_sector_share": 0.22,
        "displacement_cap_million": 0.55,
        "ai_scale": 12000.0,
        "training_scale": 9000.0,
        "job_creation_weight": 1.00,
        "productivity_weight": 0.35,
        "displacement_penalty": 1.20,
        "training_effectiveness": 0.62,
    }


def _b9_prepare_data():
    """
    Chuẩn bị dữ liệu 10 ngành và các hệ số mô phỏng.
    """
    df = load_sectors().copy()

    required_columns = [
        "sector_name_vi",
        "growth_rate_2024_pct",
        "gdp_share_2024_pct",
        "labor_million",
        "ai_readiness_0_100",
        "automation_risk_pct",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột dữ liệu Bài 9: "
            + ", ".join(missing)
        )

    df = df[required_columns].copy()

    df["AI_readiness"] = (
        df["ai_readiness_0_100"] / 100.0
    )

    df["Automation_risk"] = (
        df["automation_risk_pct"] / 100.0
    )

    df["Growth_norm"] = minmax(
        df["growth_rate_2024_pct"]
    )

    df["GDP_share_norm"] = minmax(
        df["gdp_share_2024_pct"]
    )

    df["Labor_norm"] = minmax(
        df["labor_million"]
    )

    # Hệ số tạo việc làm khi đầu tư AI.
    df["AI_job_creation_coef"] = (
        0.010
        + 0.035 * df["AI_readiness"]
        + 0.018 * df["Growth_norm"]
    )

    # Hệ số việc làm bị thay thế bởi AI.
    df["AI_displacement_coef"] = (
        0.010
        + 0.055 * df["Automation_risk"]
        - 0.012 * df["AI_readiness"]
    ).clip(lower=0.004)

    # Hệ số đào tạo lại.
    df["Training_recovery_coef"] = (
        0.025
        + 0.050 * df["Automation_risk"]
        + 0.020 * df["Labor_norm"]
    )

    # Hệ số năng suất.
    df["Productivity_coef"] = (
        0.020
        + 0.055 * df["AI_readiness"]
        + 0.020 * df["GDP_share_norm"]
    )

    return df


def _b9_job_metrics(
    df,
    ai_investment,
    training_investment,
):
    """
    Tính tác động việc làm từ nghiệm đầu tư.
    """
    params = _b9_parameters()

    ai_investment = np.asarray(
        ai_investment,
        dtype=float,
    )

    training_investment = np.asarray(
        training_investment,
        dtype=float,
    )

    ai_intensity = (
        ai_investment
        / params["ai_scale"]
    )

    training_intensity = (
        training_investment
        / params["training_scale"]
    )

    labor = df["labor_million"].to_numpy(
        dtype=float
    )

    jobs_created = (
        labor
        * df[
            "AI_job_creation_coef"
        ].to_numpy(dtype=float)
        * ai_intensity
    )

    jobs_displaced_raw = (
        labor
        * df[
            "AI_displacement_coef"
        ].to_numpy(dtype=float)
        * ai_intensity
    )

    retraining_capacity = (
        labor
        * df[
            "Training_recovery_coef"
        ].to_numpy(dtype=float)
        * training_intensity
        * params["training_effectiveness"]
    )

    jobs_retrained = np.minimum(
        jobs_displaced_raw,
        retraining_capacity,
    )

    unrecovered_displacement = (
        jobs_displaced_raw
        - jobs_retrained
    )

    net_jobs = (
        jobs_created
        - unrecovered_displacement
    )

    productivity_gain = (
        df[
            "Productivity_coef"
        ].to_numpy(dtype=float)
        * ai_intensity
    )

    return {
        "jobs_created": jobs_created,
        "jobs_displaced_raw": jobs_displaced_raw,
        "jobs_retrained": jobs_retrained,
        "unrecovered_displacement": unrecovered_displacement,
        "net_jobs": net_jobs,
        "productivity_gain": productivity_gain,
    }


def _b9_solve_cvxpy(
    total_budget=100000.0,
    training_budget_share=0.45,
    displacement_cap_million=0.55,
):
    """
    Giải mô hình phân bổ AI và đào tạo bằng CVXPY.

    Mục tiêu tuyến tính hóa:
    tối đa hóa việc làm tạo mới + năng suất
    - việc làm bị thay thế chưa được đào tạo lại.
    """
    import cvxpy as cp

    df = _b9_prepare_data()
    params = _b9_parameters()

    n = len(df)

    ai_budget = (
        float(total_budget)
        * (
            1.0
            - float(training_budget_share)
        )
    )

    training_budget = (
        float(total_budget)
        * float(training_budget_share)
    )

    ai = cp.Variable(
        n,
        nonneg=True,
        name="AI_investment",
    )

    training = cp.Variable(
        n,
        nonneg=True,
        name="Training_investment",
    )

    displaced = cp.Variable(
        n,
        nonneg=True,
        name="Displaced_jobs",
    )

    retrained = cp.Variable(
        n,
        nonneg=True,
        name="Retrained_jobs",
    )

    labor = df[
        "labor_million"
    ].to_numpy(dtype=float)

    ai_creation_coef = (
        labor
        * df[
            "AI_job_creation_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    ai_displacement_coef = (
        labor
        * df[
            "AI_displacement_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    training_recovery_coef = (
        labor
        * df[
            "Training_recovery_coef"
        ].to_numpy(dtype=float)
        * params["training_effectiveness"]
        / params["training_scale"]
    )

    productivity_coef = (
        df[
            "Productivity_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    max_ai_sector = (
        ai_budget
        * params["maximum_sector_share"]
    )

    max_training_sector = (
        training_budget
        * params["maximum_sector_share"]
    )

    constraints = [
        cp.sum(ai) <= ai_budget,
        cp.sum(training) <= training_budget,
        cp.sum(training)
        >= (
            params[
                "minimum_training_share"
            ]
            * float(total_budget)
        ),
        ai <= max_ai_sector,
        training <= max_training_sector,
        displaced
        == cp.multiply(
            ai_displacement_coef,
            ai,
        ),
        retrained
        <= cp.multiply(
            training_recovery_coef,
            training,
        ),
        retrained <= displaced,
        cp.sum(
            displaced - retrained
        )
        <= float(
            displacement_cap_million
        ),
    ]

    jobs_created = cp.multiply(
        ai_creation_coef,
        ai,
    )

    unrecovered = (
        displaced - retrained
    )

    productivity = cp.multiply(
        productivity_coef,
        ai,
    )

    objective = cp.Maximize(
        params["job_creation_weight"]
        * cp.sum(
            jobs_created
        )
        + params["productivity_weight"]
        * cp.sum(
            productivity
        )
        - params["displacement_penalty"]
        * cp.sum(
            unrecovered
        )
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    installed = cp.installed_solvers()

    if "CLARABEL" in installed:
        problem.solve(
            solver=cp.CLARABEL,
            verbose=False,
        )
    elif "SCIPY" in installed:
        problem.solve(
            solver=cp.SCIPY,
            verbose=False,
        )
    elif "SCS" in installed:
        problem.solve(
            solver=cp.SCS,
            verbose=False,
        )
    else:
        problem.solve(
            verbose=False,
        )

    success_status = {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }

    if problem.status not in success_status:
        return {
            "success": False,
            "status": str(problem.status),
            "solver": "CVXPY",
            "ai": None,
            "training": None,
            "objective": np.nan,
        }

    ai_value = np.maximum(
        np.asarray(
            ai.value,
            dtype=float,
        ).reshape(-1),
        0.0,
    )

    training_value = np.maximum(
        np.asarray(
            training.value,
            dtype=float,
        ).reshape(-1),
        0.0,
    )

    return {
        "success": True,
        "status": str(problem.status),
        "solver": "CVXPY",
        "ai": ai_value,
        "training": training_value,
        "objective": float(problem.value),
    }


def _b9_solve_scipy(
    total_budget=100000.0,
    training_budget_share=0.45,
    displacement_cap_million=0.55,
):
    """
    Phương án dự phòng bằng scipy.optimize.linprog.
    """
    df = _b9_prepare_data()
    params = _b9_parameters()

    n = len(df)

    ai_budget = (
        float(total_budget)
        * (
            1.0
            - float(training_budget_share)
        )
    )

    training_budget = (
        float(total_budget)
        * float(training_budget_share)
    )

    labor = df[
        "labor_million"
    ].to_numpy(dtype=float)

    create_coef = (
        labor
        * df[
            "AI_job_creation_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    displace_coef = (
        labor
        * df[
            "AI_displacement_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    recover_coef = (
        labor
        * df[
            "Training_recovery_coef"
        ].to_numpy(dtype=float)
        * params["training_effectiveness"]
        / params["training_scale"]
    )

    productivity_coef = (
        df[
            "Productivity_coef"
        ].to_numpy(dtype=float)
        / params["ai_scale"]
    )

    # Biến: ai[n], training[n], retrained[n]
    n_var = 3 * n

    c = np.zeros(
        n_var,
        dtype=float,
    )

    c[:n] = -(
        params["job_creation_weight"]
        * create_coef
        + params[
            "productivity_weight"
        ]
        * productivity_coef
        - params[
            "displacement_penalty"
        ]
        * displace_coef
    )

    c[2 * n:] = -params[
        "displacement_penalty"
    ]

    A_ub = []
    b_ub = []

    # Tổng ngân sách AI.
    row = np.zeros(
        n_var,
        dtype=float,
    )
    row[:n] = 1.0
    A_ub.append(row)
    b_ub.append(ai_budget)

    # Tổng ngân sách đào tạo.
    row = np.zeros(
        n_var,
        dtype=float,
    )
    row[n:2 * n] = 1.0
    A_ub.append(row)
    b_ub.append(training_budget)

    # Sàn đào tạo.
    row = np.zeros(
        n_var,
        dtype=float,
    )
    row[n:2 * n] = -1.0
    A_ub.append(row)
    b_ub.append(
        -params[
            "minimum_training_share"
        ]
        * float(total_budget)
    )

    # retrained_i <= recovery_coef_i * training_i
    for i in range(n):
        row = np.zeros(
            n_var,
            dtype=float,
        )
        row[2 * n + i] = 1.0
        row[n + i] = -recover_coef[i]
        A_ub.append(row)
        b_ub.append(0.0)

    # retrained_i <= displaced_i
    for i in range(n):
        row = np.zeros(
            n_var,
            dtype=float,
        )
        row[2 * n + i] = 1.0
        row[i] = -displace_coef[i]
        A_ub.append(row)
        b_ub.append(0.0)

    # Tổng displacement chưa phục hồi <= cap.
    row = np.zeros(
        n_var,
        dtype=float,
    )
    row[:n] = displace_coef
    row[2 * n:] = -1.0
    A_ub.append(row)
    b_ub.append(
        float(
            displacement_cap_million
        )
    )

    bounds = []

    for _ in range(n):
        bounds.append(
            (
                0.0,
                ai_budget
                * params[
                    "maximum_sector_share"
                ],
            )
        )

    for _ in range(n):
        bounds.append(
            (
                0.0,
                training_budget
                * params[
                    "maximum_sector_share"
                ],
            )
        )

    for _ in range(n):
        bounds.append(
            (
                0.0,
                None,
            )
        )

    result = linprog(
        c,
        A_ub=np.asarray(
            A_ub,
            dtype=float,
        ),
        b_ub=np.asarray(
            b_ub,
            dtype=float,
        ),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return {
            "success": False,
            "status": str(result.message),
            "solver": "SciPy/HiGHS",
            "ai": None,
            "training": None,
            "objective": np.nan,
        }

    return {
        "success": True,
        "status": str(result.message),
        "solver": "SciPy/HiGHS fallback",
        "ai": np.asarray(
            result.x[:n],
            dtype=float,
        ),
        "training": np.asarray(
            result.x[n:2 * n],
            dtype=float,
        ),
        "objective": float(
            -result.fun
        ),
    }


@st.cache_data(show_spinner=False)
def _b9_solve(
    total_budget=100000.0,
    training_budget_share=0.45,
    displacement_cap_million=0.55,
):
    """
    Ưu tiên CVXPY; nếu lỗi thư viện thì dùng SciPy.
    """
    try:
        result = _b9_solve_cvxpy(
            total_budget=total_budget,
            training_budget_share=training_budget_share,
            displacement_cap_million=displacement_cap_million,
        )

        if result["success"]:
            return result

    except Exception:
        pass

    return _b9_solve_scipy(
        total_budget=total_budget,
        training_budget_share=training_budget_share,
        displacement_cap_million=displacement_cap_million,
    )


def _b9_result_table(
    solve_result,
):
    """Tạo bảng đầu ra đầy đủ theo ngành."""
    df = _b9_prepare_data().copy()

    if not solve_result["success"]:
        return pd.DataFrame(), {}

    metrics = _b9_job_metrics(
        df,
        solve_result["ai"],
        solve_result["training"],
    )

    result = pd.DataFrame(
        {
            "Ngành": df[
                "sector_name_vi"
            ],
            "Lao động (triệu)": df[
                "labor_million"
            ],
            "AI Readiness": df[
                "AI_readiness"
            ],
            "Rủi ro tự động hóa": df[
                "Automation_risk"
            ],
            "Đầu tư AI": solve_result[
                "ai"
            ],
            "Đầu tư đào tạo": solve_result[
                "training"
            ],
            "Việc làm tạo mới": metrics[
                "jobs_created"
            ],
            "Việc làm bị thay thế": metrics[
                "jobs_displaced_raw"
            ],
            "Lao động đào tạo lại": metrics[
                "jobs_retrained"
            ],
            "Mất việc chưa phục hồi": metrics[
                "unrecovered_displacement"
            ],
            "Việc làm ròng": metrics[
                "net_jobs"
            ],
            "Tăng năng suất": metrics[
                "productivity_gain"
            ],
        }
    )

    summary = {
        "total_ai": float(
            result[
                "Đầu tư AI"
            ].sum()
        ),
        "total_training": float(
            result[
                "Đầu tư đào tạo"
            ].sum()
        ),
        "jobs_created": float(
            result[
                "Việc làm tạo mới"
            ].sum()
        ),
        "jobs_displaced": float(
            result[
                "Việc làm bị thay thế"
            ].sum()
        ),
        "jobs_retrained": float(
            result[
                "Lao động đào tạo lại"
            ].sum()
        ),
        "unrecovered": float(
            result[
                "Mất việc chưa phục hồi"
            ].sum()
        ),
        "net_jobs": float(
            result[
                "Việc làm ròng"
            ].sum()
        ),
        "productivity": float(
            result[
                "Tăng năng suất"
            ].sum()
        ),
    }

    return (
        result.sort_values(
            "Việc làm ròng",
            ascending=False,
        ).reset_index(drop=True),
        summary,
    )


def _b9_validation_table(
    result_table,
    summary,
    total_budget,
    training_budget_share,
    displacement_cap_million,
):
    """Kiểm tra ràng buộc sau tối ưu."""
    params = _b9_parameters()

    ai_budget = (
        float(total_budget)
        * (
            1.0
            - float(training_budget_share)
        )
    )

    training_budget = (
        float(total_budget)
        * float(training_budget_share)
    )

    checks = [
        {
            "Ràng buộc": "Tổng đầu tư AI không vượt ngân sách",
            "Giá trị": (
                f"{summary['total_ai']:,.2f} / "
                f"{ai_budget:,.2f}"
            ),
            "Đạt": (
                summary["total_ai"]
                <= ai_budget
                + 1e-5
            ),
        },
        {
            "Ràng buộc": "Tổng đào tạo không vượt ngân sách",
            "Giá trị": (
                f"{summary['total_training']:,.2f} / "
                f"{training_budget:,.2f}"
            ),
            "Đạt": (
                summary["total_training"]
                <= training_budget
                + 1e-5
            ),
        },
        {
            "Ràng buộc": "Đào tạo đạt mức tối thiểu",
            "Giá trị": (
                f"{summary['total_training']:,.2f} / "
                f"{params['minimum_training_share'] * total_budget:,.2f}"
            ),
            "Đạt": (
                summary["total_training"]
                >= params[
                    "minimum_training_share"
                ]
                * total_budget
                - 1e-5
            ),
        },
        {
            "Ràng buộc": "Mất việc chưa phục hồi không vượt trần",
            "Giá trị": (
                f"{summary['unrecovered']:.4f} / "
                f"{displacement_cap_million:.4f} triệu"
            ),
            "Đạt": (
                summary["unrecovered"]
                <= displacement_cap_million
                + 1e-5
            ),
        },
        {
            "Ràng buộc": "Đầu tư không âm",
            "Giá trị": float(
                result_table[
                    [
                        "Đầu tư AI",
                        "Đầu tư đào tạo",
                    ]
                ].min().min()
            ),
            "Đạt": bool(
                (
                    result_table[
                        [
                            "Đầu tư AI",
                            "Đầu tư đào tạo",
                        ]
                    ]
                    >= -1e-7
                ).all().all()
            ),
        },
        {
            "Ràng buộc": "Đào tạo lại không vượt số bị thay thế",
            "Giá trị": float(
                (
                    result_table[
                        "Lao động đào tạo lại"
                    ]
                    - result_table[
                        "Việc làm bị thay thế"
                    ]
                ).max()
            ),
            "Đạt": bool(
                (
                    result_table[
                        "Lao động đào tạo lại"
                    ]
                    <= result_table[
                        "Việc làm bị thay thế"
                    ]
                    + 1e-7
                ).all()
            ),
        },
    ]

    return pd.DataFrame(
        checks
    )


def _b9_sankey_figure(
    summary,
):
    """Tạo Sankey từ lao động hiện tại đến các trạng thái sau AI."""
    import plotly.graph_objects as go

    labels = [
        "Việc làm chịu tác động AI",
        "Việc làm tạo mới",
        "Việc làm bị thay thế",
        "Được đào tạo lại",
        "Mất việc chưa phục hồi",
        "Việc làm ròng",
    ]

    source = [
        0,
        0,
        2,
        2,
        1,
        3,
    ]

    target = [
        1,
        2,
        3,
        4,
        5,
        5,
    ]

    values = [
        max(
            summary["jobs_created"],
            1e-8,
        ),
        max(
            summary["jobs_displaced"],
            1e-8,
        ),
        max(
            summary["jobs_retrained"],
            1e-8,
        ),
        max(
            summary["unrecovered"],
            1e-8,
        ),
        max(
            summary["jobs_created"],
            1e-8,
        ),
        max(
            summary["jobs_retrained"],
            1e-8,
        ),
    ]

    figure = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    label=labels,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                ),
            )
        ]
    )

    figure.update_layout(
        title="Luồng tác động việc làm của AI và đào tạo lại",
        height=520,
        template=PLOT_TEMPLATE,
    )

    return figure


def page_9():
    hero(
        "Bài 9 — Phân bổ đầu tư AI và đào tạo lại lao động",
        "Tối ưu đồng thời đầu tư AI và đào tạo theo 10 ngành bằng CVXPY; kiểm soát mất việc, lượng hóa việc làm ròng, phân tích ngưỡng đào tạo và trực quan hóa dòng lao động.",
        [
            "9.1-9.5",
            "CVXPY",
            "AI & labor",
            "Reskilling",
            "Sankey",
        ],
    )

    params = _b9_parameters()
    df = _b9_prepare_data()

    # =====================================================
    # 9.1. Bối cảnh
    # =====================================================
    st.markdown(
        "## 9.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        AI có thể đồng thời tạo việc làm mới, nâng năng suất và thay thế một phần lao động.
        Tác động không đồng đều giữa các ngành vì khác nhau về rủi ro tự động hóa,
        năng lực AI, quy mô lao động và tốc độ tăng trưởng.

        Bài 9 lựa chọn mức đầu tư AI và đào tạo lại theo ngành nhằm tối đa hóa
        việc làm ròng và năng suất, đồng thời giới hạn số lao động mất việc
        chưa được chuyển đổi kỹ năng.
        """
    )

    # =====================================================
    # 9.2. Mô hình toán học
    # =====================================================
    st.markdown(
        "## 9.2. Mô hình toán học"
    )

    st.latex(
        r"\max\ "
        r"\sum_i"
        r"\left("
        r"J_i^{create}"
        r"+\omega P_i"
        r"-\lambda U_i"
        r"\right)"
    )

    st.latex(
        r"J_i^{create}=a_iAI_i"
    )

    st.latex(
        r"J_i^{displace}=d_iAI_i"
    )

    st.latex(
        r"J_i^{retrain}\leq h_iT_i,"
        r"\qquad"
        r"J_i^{retrain}\leq J_i^{displace}"
    )

    st.latex(
        r"U_i="
        r"J_i^{displace}"
        r"-J_i^{retrain}"
    )

    st.latex(
        r"\sum_iAI_i\leq B_{AI},"
        r"\qquad"
        r"\sum_iT_i\leq B_T"
    )

    st.latex(
        r"\sum_iU_i\leq \overline{U}"
    )

    st.info(
        "Mô hình dùng CVXPY làm solver chính. Nếu CVXPY không khả dụng, trang tự dùng SciPy/HiGHS."
    )

    # =====================================================
    # 9.3. Dữ liệu
    # =====================================================
    st.markdown(
        "## 9.3. Dữ liệu 10 ngành"
    )

    data_display = df[
        [
            "sector_name_vi",
            "labor_million",
            "growth_rate_2024_pct",
            "gdp_share_2024_pct",
            "ai_readiness_0_100",
            "automation_risk_pct",
            "AI_job_creation_coef",
            "AI_displacement_coef",
            "Training_recovery_coef",
            "Productivity_coef",
        ]
    ].rename(
        columns={
            "sector_name_vi": "Ngành",
            "labor_million": "Lao động (triệu)",
            "growth_rate_2024_pct": "Tăng trưởng (%)",
            "gdp_share_2024_pct": "Tỷ trọng GDP (%)",
            "ai_readiness_0_100": "AI Readiness",
            "automation_risk_pct": "Rủi ro tự động hóa (%)",
            "AI_job_creation_coef": "Hệ số tạo việc",
            "AI_displacement_coef": "Hệ số thay thế",
            "Training_recovery_coef": "Hệ số đào tạo",
            "Productivity_coef": "Hệ số năng suất",
        }
    )

    st.dataframe(
        data_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Dữ liệu ngành được lấy từ CSV của dự án; các hệ số tác động là giả định mô phỏng và phải được nêu rõ trong báo cáo."
    )

    # =====================================================
    # 9.4. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 9.4. Yêu cầu lập trình"
    )

    c1, c2, c3 = st.columns(3)

    total_budget = c1.slider(
        "Tổng ngân sách (tỷ VND)",
        min_value=60000,
        max_value=140000,
        value=100000,
        step=5000,
        key="b9_total_budget",
    )

    training_budget_share = c2.slider(
        "Tỷ trọng ngân sách đào tạo",
        min_value=0.30,
        max_value=0.65,
        value=0.45,
        step=0.05,
        key="b9_training_share",
    )

    displacement_cap = c3.slider(
        "Trần mất việc chưa phục hồi (triệu)",
        min_value=0.10,
        max_value=1.00,
        value=0.55,
        step=0.05,
        key="b9_displacement_cap",
    )

    with st.spinner(
        "Đang tối ưu phân bổ AI và đào tạo..."
    ):
        solve_result = _b9_solve(
            total_budget=float(
                total_budget
            ),
            training_budget_share=float(
                training_budget_share
            ),
            displacement_cap_million=float(
                displacement_cap
            ),
        )

    if not solve_result["success"]:
        st.error(
            f"Mô hình không giải được. Status: {solve_result['status']}"
        )
        return

    result_table, summary = (
        _b9_result_table(
            solve_result
        )
    )

    tab941, tab942, tab943, tab944 = st.tabs(
        [
            "9.4.1 - Nghiệm tối ưu",
            "9.4.2 - Việc làm & năng suất",
            "9.4.3 - Ngưỡng đào tạo",
            "9.4.4 - Sankey & kiểm định",
        ]
    )

    # -----------------------------------------------------
    # 9.4.1
    # -----------------------------------------------------
    with tab941:
        st.markdown(
            "### Câu 9.4.1. Phân bổ ngân sách AI và đào tạo"
        )

        kpi_cards(
            [
                (
                    "Solver",
                    solve_result["solver"],
                    solve_result["status"],
                ),
                (
                    "Đầu tư AI",
                    f"{summary['total_ai']:,.0f}",
                    "tỷ VND",
                ),
                (
                    "Đầu tư đào tạo",
                    f"{summary['total_training']:,.0f}",
                    "tỷ VND",
                ),
                (
                    "Mất việc chưa phục hồi",
                    f"{summary['unrecovered']:.4f} triệu",
                    f"trần={displacement_cap:.2f}",
                ),
            ]
        )

        st.dataframe(
            result_table[
                [
                    "Ngành",
                    "Đầu tư AI",
                    "Đầu tư đào tạo",
                    "AI Readiness",
                    "Rủi ro tự động hóa",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        allocation_long = result_table.melt(
            id_vars="Ngành",
            value_vars=[
                "Đầu tư AI",
                "Đầu tư đào tạo",
            ],
            var_name="Khoản đầu tư",
            value_name="Ngân sách",
        )

        fig = px.bar(
            allocation_long,
            x="Ngành",
            y="Ngân sách",
            color="Khoản đầu tư",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="Phân bổ AI và đào tạo theo ngành",
        )
        fig.update_layout(
            height=520,
            xaxis_tickangle=-25,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 9.4.2
    # -----------------------------------------------------
    with tab942:
        st.markdown(
            "### Câu 9.4.2. Tác động đến việc làm và năng suất"
        )

        kpi_cards(
            [
                (
                    "Việc làm tạo mới",
                    f"{summary['jobs_created']:.4f} triệu",
                    "từ đầu tư AI",
                ),
                (
                    "Việc làm bị thay thế",
                    f"{summary['jobs_displaced']:.4f} triệu",
                    "trước đào tạo",
                ),
                (
                    "Được đào tạo lại",
                    f"{summary['jobs_retrained']:.4f} triệu",
                    "chuyển đổi kỹ năng",
                ),
                (
                    "Việc làm ròng",
                    f"{summary['net_jobs']:+.4f} triệu",
                    "sau đào tạo",
                ),
            ]
        )

        st.dataframe(
            result_table[
                [
                    "Ngành",
                    "Việc làm tạo mới",
                    "Việc làm bị thay thế",
                    "Lao động đào tạo lại",
                    "Mất việc chưa phục hồi",
                    "Việc làm ròng",
                    "Tăng năng suất",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig_net = px.bar(
            result_table.sort_values(
                "Việc làm ròng"
            ),
            x="Việc làm ròng",
            y="Ngành",
            orientation="h",
            template=PLOT_TEMPLATE,
            title="Việc làm ròng theo ngành",
        )
        fig_net.update_layout(
            height=560,
        )

        st.plotly_chart(
            fig_net,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 9.4.3
    # -----------------------------------------------------
    with tab943:
        st.markdown(
            "### Câu 9.4.3. Phân tích ngưỡng ngân sách đào tạo"
        )

        training_shares = np.arange(
            0.30,
            0.651,
            0.05,
        )

        rows = []

        for share in training_shares:
            scenario_result = _b9_solve(
                total_budget=float(
                    total_budget
                ),
                training_budget_share=float(
                    round(
                        share,
                        2,
                    )
                ),
                displacement_cap_million=float(
                    displacement_cap
                ),
            )

            if not scenario_result["success"]:
                rows.append(
                    {
                        "Tỷ trọng đào tạo": share,
                        "Khả thi": False,
                        "Việc làm ròng": np.nan,
                        "Mất việc chưa phục hồi": np.nan,
                        "Đầu tư AI": np.nan,
                        "Đầu tư đào tạo": np.nan,
                    }
                )
                continue

            _, scenario_summary = (
                _b9_result_table(
                    scenario_result
                )
            )

            rows.append(
                {
                    "Tỷ trọng đào tạo": share,
                    "Khả thi": True,
                    "Việc làm ròng": scenario_summary[
                        "net_jobs"
                    ],
                    "Mất việc chưa phục hồi": scenario_summary[
                        "unrecovered"
                    ],
                    "Đầu tư AI": scenario_summary[
                        "total_ai"
                    ],
                    "Đầu tư đào tạo": scenario_summary[
                        "total_training"
                    ],
                }
            )

        sensitivity = pd.DataFrame(
            rows
        )

        st.dataframe(
            sensitivity,
            use_container_width=True,
            hide_index=True,
        )

        fig_sensitivity = px.line(
            sensitivity[
                sensitivity["Khả thi"]
            ],
            x="Tỷ trọng đào tạo",
            y=[
                "Việc làm ròng",
                "Mất việc chưa phục hồi",
            ],
            markers=True,
            template=PLOT_TEMPLATE,
            title="Độ nhạy việc làm theo tỷ trọng đào tạo",
        )
        fig_sensitivity.update_layout(
            height=470,
        )

        st.plotly_chart(
            fig_sensitivity,
            use_container_width=True,
        )

        feasible_rows = sensitivity[
            sensitivity["Khả thi"]
        ]

        if not feasible_rows.empty:
            best_row = feasible_rows.loc[
                feasible_rows[
                    "Việc làm ròng"
                ].idxmax()
            ]

            st.success(
                f"Trong dải thử nghiệm, tỷ trọng đào tạo cho việc làm ròng cao nhất là "
                f"**{best_row['Tỷ trọng đào tạo']:.2f}**, với việc làm ròng "
                f"**{best_row['Việc làm ròng']:+.4f} triệu**."
            )

    # -----------------------------------------------------
    # 9.4.4
    # -----------------------------------------------------
    with tab944:
        st.markdown(
            "### Câu 9.4.4. Sankey và kiểm định nghiệm"
        )

        st.plotly_chart(
            _b9_sankey_figure(
                summary
            ),
            use_container_width=True,
        )

        validation = _b9_validation_table(
            result_table=result_table,
            summary=summary,
            total_budget=float(
                total_budget
            ),
            training_budget_share=float(
                training_budget_share
            ),
            displacement_cap_million=float(
                displacement_cap
            ),
        )

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True,
        )

        if bool(validation["Đạt"].all()):
            st.success(
                "Nghiệm vượt qua toàn bộ kiểm tra ràng buộc."
            )
        else:
            st.error(
                "Có ít nhất một ràng buộc chưa đạt."
            )

    # =====================================================
    # Tải kết quả
    # =====================================================
    st.download_button(
        "Tải kết quả Bài 9 dạng CSV",
        data=result_table.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai9_ai_training_labor.csv",
        mime="text/csv",
        key="download_bai9_fixed",
    )

    # =====================================================
    # 9.5. Thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 9.5. Câu hỏi thảo luận chính sách"
    )

    top_ai_sector = result_table.loc[
        result_table[
            "Đầu tư AI"
        ].idxmax(),
        "Ngành",
    ]

    top_training_sector = result_table.loc[
        result_table[
            "Đầu tư đào tạo"
        ].idxmax(),
        "Ngành",
    ]

    most_vulnerable_sector = result_table.loc[
        result_table[
            "Mất việc chưa phục hồi"
        ].idxmax(),
        "Ngành",
    ]

    with st.expander(
        "a) Ngành nào được ưu tiên đầu tư AI?",
        expanded=True,
    ):
        st.markdown(
            f"Ngành nhận đầu tư AI lớn nhất là **{top_ai_sector}**. "
            "Mô hình ưu tiên ngành có mức sẵn sàng AI, khả năng tạo việc làm "
            "và tác động năng suất tương đối cao."
        )

    with st.expander(
        "b) Ngành nào cần ưu tiên đào tạo lại?",
        expanded=True,
    ):
        st.markdown(
            f"Ngành nhận ngân sách đào tạo lớn nhất là **{top_training_sector}**. "
            "Đây thường là ngành có quy mô lao động hoặc rủi ro tự động hóa cao, "
            "nên cần chuyển đổi kỹ năng song song với triển khai AI."
        )

    with st.expander(
        "c) Ngành nào dễ bị tổn thương nhất?",
        expanded=True,
    ):
        st.markdown(
            f"Ngành có số việc làm chưa phục hồi cao nhất là **{most_vulnerable_sector}**. "
            "Cơ quan quản lý nên kết hợp đào tạo, bảo hiểm thất nghiệp, hỗ trợ dịch chuyển "
            "việc làm và yêu cầu doanh nghiệp có kế hoạch chuyển đổi lao động."
        )

    with st.expander(
        "d) Có nên tối đa hóa đầu tư AI và giảm đào tạo không?",
        expanded=True,
    ):
        st.markdown(
            "Không. Đầu tư AI cao có thể tăng năng suất nhưng đồng thời làm tăng việc làm "
            "bị thay thế. Nếu đào tạo thấp, số lao động mất việc chưa phục hồi có thể vượt "
            "ngưỡng chính sách và làm giảm việc làm ròng."
        )

    with st.expander(
        "e) Giới hạn của kết quả là gì?",
        expanded=True,
    ):
        st.markdown(
            "Các hệ số tạo việc, thay thế và đào tạo là giả định mô phỏng; chưa có dữ liệu "
            "vi mô theo nghề, độ tuổi, kỹ năng và địa phương. Vì vậy kết quả nên dùng để "
            "so sánh kịch bản, không được coi là dự báo việc làm chính thức."
        )


def _b10_data():
    categories = [
        "K - Hạ tầng",
        "D - Chuyển đổi số",
        "AI",
        "H - Nhân lực",
    ]

    scenarios = [
        "Thấp",
        "Cơ sở",
        "Cao",
    ]

    probability = {
        "Thấp": 0.25,
        "Cơ sở": 0.50,
        "Cao": 0.25,
    }

    demand = {
        "Thấp": 72.0,
        "Cơ sở": 92.0,
        "Cao": 116.0,
    }

    service_value = {
        "Thấp": 980.0,
        "Cơ sở": 1080.0,
        "Cao": 1180.0,
    }

    shortage_penalty = {
        "Thấp": 1350.0,
        "Cơ sở": 1550.0,
        "Cao": 1850.0,
    }

    recourse_cost = {
        "K - Hạ tầng": 1.28,
        "D - Chuyển đổi số": 1.22,
        "AI": 1.35,
        "H - Nhân lực": 1.18,
    }

    productivity = {
        ("K - Hạ tầng", "Thấp"): 0.00150,
        ("D - Chuyển đổi số", "Thấp"): 0.00180,
        ("AI", "Thấp"): 0.00120,
        ("H - Nhân lực", "Thấp"): 0.00170,

        ("K - Hạ tầng", "Cơ sở"): 0.00165,
        ("D - Chuyển đổi số", "Cơ sở"): 0.00210,
        ("AI", "Cơ sở"): 0.00175,
        ("H - Nhân lực", "Cơ sở"): 0.00195,

        ("K - Hạ tầng", "Cao"): 0.00175,
        ("D - Chuyển đổi số", "Cao"): 0.00235,
        ("AI", "Cao"): 0.00245,
        ("H - Nhân lực", "Cao"): 0.00210,
    }

    return {
        "categories": categories,
        "scenarios": scenarios,
        "probability": probability,
        "demand": demand,
        "service_value": service_value,
        "shortage_penalty": shortage_penalty,
        "recourse_cost": recourse_cost,
        "productivity": productivity,
        "first_stage_budget": 50000.0,
        "recourse_budget": 9000.0,
    }


def _b10_solver():
    """
    Tìm solver cho Pyomo.
    Ưu tiên CBC đi kèm PuLP, sau đó GLPK/HiGHS.
    """
    import pyomo.environ as pyo

    candidates = []

    try:
        import pulp

        cbc_path = pulp.PULP_CBC_CMD(
            msg=False
        ).path

        if cbc_path:
            candidates.append(
                (
                    "cbc",
                    cbc_path,
                )
            )
    except Exception:
        pass

    candidates.extend(
        [
            ("appsi_highs", None),
            ("highs", None),
            ("glpk", None),
            ("cbc", None),
        ]
    )

    for solver_name, executable in candidates:
        try:
            solver = pyo.SolverFactory(
                solver_name,
                executable=executable,
            )

            if solver is not None and solver.available(
                exception_flag=False
            ):
                return solver, solver_name
        except Exception:
            continue

    raise RuntimeError(
        "Không tìm thấy solver Pyomo. "
        "Hãy cài PuLP/CBC hoặc HiGHS."
    )


def _b10_build_model(
    scenario_subset=None,
    fixed_x=None,
    expected_value=False,
    robust=False,
):
    import pyomo.environ as pyo

    data = _b10_data()
    categories = data["categories"]
    all_scenarios = data["scenarios"]

    scenarios = (
        list(scenario_subset)
        if scenario_subset is not None
        else all_scenarios
    )

    model = pyo.ConcreteModel(
        name="AIDEOM_Stochastic_Program"
    )

    model.J = pyo.Set(
        initialize=categories,
        ordered=True,
    )
    model.S = pyo.Set(
        initialize=scenarios,
        ordered=True,
    )

    if expected_value:
        expected_demand = sum(
            data["probability"][s]
            * data["demand"][s]
            for s in all_scenarios
        )

        expected_value_per_service = sum(
            data["probability"][s]
            * data["service_value"][s]
            for s in all_scenarios
        )

        expected_penalty = sum(
            data["probability"][s]
            * data["shortage_penalty"][s]
            for s in all_scenarios
        )

        expected_productivity = {
            j: sum(
                data["probability"][s]
                * data["productivity"][(j, s)]
                for s in all_scenarios
            )
            for j in categories
        }

        scenario_name = "Expected"
        model.S = pyo.Set(
            initialize=[scenario_name],
            ordered=True,
        )
        scenarios = [scenario_name]

        probability = {
            scenario_name: 1.0,
        }
        demand = {
            scenario_name: expected_demand,
        }
        service_value = {
            scenario_name: expected_value_per_service,
        }
        shortage_penalty = {
            scenario_name: expected_penalty,
        }
        productivity = {
            (j, scenario_name): expected_productivity[j]
            for j in categories
        }
    else:
        probability = {
            s: (
                1.0
                if len(scenarios) == 1
                else data["probability"][s]
            )
            for s in scenarios
        }

        if len(scenarios) > 1:
            total_probability = sum(
                probability.values()
            )
            probability = {
                s: probability[s]
                / total_probability
                for s in scenarios
            }

        demand = {
            s: data["demand"][s]
            for s in scenarios
        }
        service_value = {
            s: data["service_value"][s]
            for s in scenarios
        }
        shortage_penalty = {
            s: data["shortage_penalty"][s]
            for s in scenarios
        }
        productivity = {
            (j, s): data["productivity"][(j, s)]
            for j in categories
            for s in scenarios
        }

    model.x = pyo.Var(
        model.J,
        domain=pyo.NonNegativeReals,
    )

    model.r = pyo.Var(
        model.J,
        model.S,
        domain=pyo.NonNegativeReals,
    )

    model.served = pyo.Var(
        model.S,
        domain=pyo.NonNegativeReals,
    )

    model.shortage = pyo.Var(
        model.S,
        domain=pyo.NonNegativeReals,
    )

    if fixed_x is not None:
        for j in categories:
            model.x[j].fix(
                float(fixed_x[j])
            )

    model.first_stage_budget = pyo.Constraint(
        expr=sum(
            model.x[j]
            for j in model.J
        )
        <= data["first_stage_budget"]
    )

    model.human_floor = pyo.Constraint(
        expr=model.x[
            "H - Nhân lực"
        ]
        >= 8000.0
    )

    model.digital_floor = pyo.Constraint(
        expr=model.x[
            "D - Chuyển đổi số"
        ]
        >= 7000.0
    )

    def recourse_budget_rule(
        current_model,
        scenario,
    ):
        return sum(
            current_model.r[j, scenario]
            for j in current_model.J
        ) <= data["recourse_budget"]

    model.recourse_budget = pyo.Constraint(
        model.S,
        rule=recourse_budget_rule,
    )

    def served_capacity_rule(
        current_model,
        scenario,
    ):
        return current_model.served[
            scenario
        ] <= sum(
            productivity[(j, scenario)]
            * (
                current_model.x[j]
                + current_model.r[j, scenario]
            )
            for j in current_model.J
        )

    model.served_capacity = pyo.Constraint(
        model.S,
        rule=served_capacity_rule,
    )

    def served_demand_rule(
        current_model,
        scenario,
    ):
        return (
            current_model.served[scenario]
            <= demand[scenario]
        )

    model.served_demand = pyo.Constraint(
        model.S,
        rule=served_demand_rule,
    )

    def shortage_rule(
        current_model,
        scenario,
    ):
        return (
            current_model.shortage[scenario]
            >= demand[scenario]
            - current_model.served[scenario]
        )

    model.shortage_balance = pyo.Constraint(
        model.S,
        rule=shortage_rule,
    )

    def scenario_profit_expression(
        current_model,
        scenario,
    ):
        return (
            service_value[scenario]
            * current_model.served[scenario]
            - shortage_penalty[scenario]
            * current_model.shortage[scenario]
            - sum(
                data["recourse_cost"][j]
                * current_model.r[j, scenario]
                for j in current_model.J
            )
            - 0.20
            * sum(
                current_model.x[j]
                for j in current_model.J
            )
        )

    model.scenario_profit = pyo.Expression(
        model.S,
        rule=scenario_profit_expression,
    )

    if robust:
        model.z = pyo.Var(
            domain=pyo.Reals
        )

        def robust_rule(
            current_model,
            scenario,
        ):
            return (
                current_model.z
                <= current_model.scenario_profit[
                    scenario
                ]
            )

        model.robust_constraints = pyo.Constraint(
            model.S,
            rule=robust_rule,
        )

        model.objective = pyo.Objective(
            expr=model.z,
            sense=pyo.maximize,
        )
    else:
        model.objective = pyo.Objective(
            expr=sum(
                probability[s]
                * model.scenario_profit[s]
                for s in model.S
            ),
            sense=pyo.maximize,
        )

    model._aideom_meta = {
        "probability": probability,
        "demand": demand,
        "service_value": service_value,
        "shortage_penalty": shortage_penalty,
        "productivity": productivity,
        "robust": robust,
    }

    return model


def _b10_solve_model(model):
    import pyomo.environ as pyo

    solver, solver_name = _b10_solver()

    result = solver.solve(
        model,
        tee=False,
    )

    termination = str(
        result.solver.termination_condition
    ).lower()

    success = (
        "optimal" in termination
    )

    if not success:
        return {
            "success": False,
            "status": termination,
            "solver": solver_name,
            "model": model,
        }

    objective = float(
        pyo.value(
            model.objective
        )
    )

    return {
        "success": True,
        "status": termination,
        "solver": solver_name,
        "objective": objective,
        "model": model,
    }


def _b10_extract_solution(
    solve_result,
):
    import pyomo.environ as pyo

    model = solve_result["model"]
    data = _b10_data()

    x = {
        j: float(
            pyo.value(
                model.x[j]
            )
        )
        for j in model.J
    }

    scenario_rows = []

    for s in model.S:
        scenario_rows.append(
            {
                "Kịch bản": str(s),
                "Served": float(
                    pyo.value(
                        model.served[s]
                    )
                ),
                "Shortage": float(
                    pyo.value(
                        model.shortage[s]
                    )
                ),
                "Recourse": float(
                    sum(
                        pyo.value(
                            model.r[j, s]
                        )
                        for j in model.J
                    )
                ),
                "Profit": float(
                    pyo.value(
                        model.scenario_profit[s]
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            {
                "Hạng mục": list(x.keys()),
                "Đầu tư giai đoạn 1": list(x.values()),
            }
        ),
        pd.DataFrame(
            scenario_rows
        ),
        x,
    )


@st.cache_data(show_spinner=False)
def _b10_full_analysis():
    """
    Tính SP, EV/EEV, WS và Robust.
    """
    data = _b10_data()

    sp_result = _b10_solve_model(
        _b10_build_model()
    )

    if not sp_result["success"]:
        raise RuntimeError(
            f"SP không tối ưu: {sp_result['status']}"
        )

    sp_x_table, sp_scenario_table, sp_x = (
        _b10_extract_solution(
            sp_result
        )
    )

    ev_result = _b10_solve_model(
        _b10_build_model(
            expected_value=True
        )
    )

    if not ev_result["success"]:
        raise RuntimeError(
            f"EV không tối ưu: {ev_result['status']}"
        )

    _, _, ev_x = _b10_extract_solution(
        ev_result
    )

    eev_result = _b10_solve_model(
        _b10_build_model(
            fixed_x=ev_x
        )
    )

    if not eev_result["success"]:
        raise RuntimeError(
            f"EEV không tối ưu: {eev_result['status']}"
        )

    ws_values = []
    ws_tables = []

    for scenario in data["scenarios"]:
        scenario_result = _b10_solve_model(
            _b10_build_model(
                scenario_subset=[
                    scenario
                ]
            )
        )

        if not scenario_result["success"]:
            raise RuntimeError(
                f"WS {scenario} không tối ưu."
            )

        x_table, scenario_table, _ = (
            _b10_extract_solution(
                scenario_result
            )
        )

        ws_values.append(
            data["probability"][scenario]
            * scenario_result["objective"]
        )

        scenario_table[
            "Kịch bản WS"
        ] = scenario
        ws_tables.append(
            scenario_table
        )

    wait_and_see = float(
        sum(ws_values)
    )

    robust_result = _b10_solve_model(
        _b10_build_model(
            robust=True
        )
    )

    if not robust_result["success"]:
        raise RuntimeError(
            f"Robust không tối ưu: {robust_result['status']}"
        )

    robust_x_table, robust_scenario_table, _ = (
        _b10_extract_solution(
            robust_result
        )
    )

    stochastic_solution = float(
        sp_result["objective"]
    )
    expected_value_result = float(
        eev_result["objective"]
    )

    vss = (
        stochastic_solution
        - expected_value_result
    )

    evpi = (
        wait_and_see
        - stochastic_solution
    )

    return {
        "sp_result": sp_result,
        "sp_x_table": sp_x_table,
        "sp_scenario_table": sp_scenario_table,
        "ev_result": ev_result,
        "eev_result": eev_result,
        "wait_and_see": wait_and_see,
        "vss": vss,
        "evpi": evpi,
        "robust_result": robust_result,
        "robust_x_table": robust_x_table,
        "robust_scenario_table": robust_scenario_table,
    }


def page_10():
    hero(
        "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
        "Xây dựng mô hình Pyomo với quyết định đầu tư trước bất định và quyết định bù đắp theo kịch bản; tính SP, EV, EEV, VSS, EVPI và phương án robust.",
        ["10.1-10.5", "Pyomo", "Two-stage SP", "VSS", "EVPI", "Robust"],
    )

    st.markdown("## 10.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        Chính sách đầu tư số phải được quyết định trước khi biết chắc mức cầu,
        hiệu quả hấp thụ và giá trị kinh tế của công nghệ. Sau khi kịch bản xảy ra,
        Chính phủ có thể bổ sung một phần ngân sách khẩn cấp nhưng với chi phí cao hơn.
        """
    )

    st.markdown("## 10.2. Mô hình toán học")
    st.latex(
        r"\max\ -0.2\sum_jx_j"
        r"+\sum_s p_s"
        r"\left[v_s y_s-q_su_s-\sum_jc_j^Rr_{js}\right]"
    )
    st.latex(
        r"\sum_jx_j\leq50{,}000"
    )
    st.latex(
        r"\sum_jr_{js}\leq9{,}000,\quad\forall s"
    )
    st.latex(
        r"y_s\leq\sum_j a_{js}(x_j+r_{js}),"
        r"\quad y_s\leq d_s"
    )
    st.latex(
        r"u_s\geq d_s-y_s"
    )

    st.markdown("## 10.3. Dữ liệu kịch bản")

    data = _b10_data()

    scenario_table = pd.DataFrame(
        {
            "Kịch bản": data["scenarios"],
            "Xác suất": [
                data["probability"][s]
                for s in data["scenarios"]
            ],
            "Nhu cầu": [
                data["demand"][s]
                for s in data["scenarios"]
            ],
            "Giá trị phục vụ": [
                data["service_value"][s]
                for s in data["scenarios"]
            ],
            "Phạt thiếu hụt": [
                data["shortage_penalty"][s]
                for s in data["scenarios"]
            ],
        }
    )

    st.dataframe(
        scenario_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## 10.4. Kết quả lập trình")

    try:
        with st.spinner(
            "Đang giải SP, EV, EEV, WS và Robust bằng Pyomo..."
        ):
            analysis = _b10_full_analysis()
    except Exception as exc:
        st.error(
            f"Không thể giải Bài 10: {exc}"
        )
        st.info(
            "Kiểm tra requirements.txt có `pyomo>=6.8` "
            "và môi trường có CBC/HiGHS."
        )
        return

    sp = analysis["sp_result"]
    eev = analysis["eev_result"]
    robust = analysis["robust_result"]

    kpi_cards(
        [
            (
                "RP/SP",
                f"{sp['objective']:,.1f}",
                f"solver={sp['solver']}",
            ),
            (
                "EEV",
                f"{eev['objective']:,.1f}",
                "dùng quyết định EV",
            ),
            (
                "VSS",
                f"{analysis['vss']:,.1f}",
                "SP - EEV",
            ),
            (
                "EVPI",
                f"{analysis['evpi']:,.1f}",
                "WS - SP",
            ),
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "10.4.1 - SP",
            "10.4.2 - EV, EEV, VSS",
            "10.4.3 - EVPI",
            "10.4.4 - Robust",
        ]
    )

    with tab1:
        st.markdown(
            "### Quyết định giai đoạn 1"
        )
        st.dataframe(
            analysis["sp_x_table"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "### Quyết định bù đắp theo kịch bản"
        )
        st.dataframe(
            analysis[
                "sp_scenario_table"
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            analysis["sp_x_table"],
            x="Hạng mục",
            y="Đầu tư giai đoạn 1",
            template=PLOT_TEMPLATE,
            title="Phân bổ first-stage của SP",
        )
        fig.update_layout(
            height=430,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab2:
        comparison = pd.DataFrame(
            {
                "Chỉ tiêu": [
                    "Stochastic solution (RP)",
                    "EEV",
                    "VSS",
                ],
                "Giá trị": [
                    sp["objective"],
                    eev["objective"],
                    analysis["vss"],
                ],
            }
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        if analysis["vss"] >= -1e-6:
            st.success(
                "VSS không âm: mô hình hóa bất định có giá trị kinh tế."
            )
        else:
            st.warning(
                "VSS âm do sai số số học hoặc cấu hình solver; cần kiểm tra."
            )

    with tab3:
        ws_sp = pd.DataFrame(
            {
                "Chỉ tiêu": [
                    "Wait-and-see",
                    "Stochastic solution",
                    "EVPI",
                ],
                "Giá trị": [
                    analysis["wait_and_see"],
                    sp["objective"],
                    analysis["evpi"],
                ],
            }
        )

        st.dataframe(
            ws_sp,
            use_container_width=True,
            hide_index=True,
        )

        if analysis["evpi"] >= -1e-6:
            st.success(
                "EVPI không âm: thông tin hoàn hảo có giá trị."
            )
        else:
            st.warning(
                "EVPI âm do sai số số học hoặc mô hình chưa nhất quán."
            )

    with tab4:
        st.dataframe(
            analysis[
                "robust_x_table"
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.dataframe(
            analysis[
                "robust_scenario_table"
            ],
            use_container_width=True,
            hide_index=True,
        )

        robust_compare = pd.DataFrame(
            {
                "Mô hình": [
                    "Stochastic expected value",
                    "Robust worst-case",
                ],
                "Giá trị mục tiêu": [
                    sp["objective"],
                    robust["objective"],
                ],
            }
        )

        st.dataframe(
            robust_compare,
            use_container_width=True,
            hide_index=True,
        )

    export = analysis[
        "sp_scenario_table"
    ].copy()

    st.download_button(
        "Tải kết quả Bài 10",
        data=export.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai10_stochastic_pyomo.csv",
        mime="text/csv",
        key="download_bai10_pyomo",
    )

    st.markdown("## 10.5. Thảo luận chính sách")

    with st.expander(
        "a) VSS có ý nghĩa gì?",
        expanded=True,
    ):
        st.markdown(
            f"VSS = **{analysis['vss']:,.1f}** đo lợi ích của việc "
            "ra quyết định first-stage bằng mô hình ngẫu nhiên thay vì "
            "thay bất định bằng giá trị trung bình."
        )

    with st.expander(
        "b) EVPI có ý nghĩa gì?",
        expanded=True,
    ):
        st.markdown(
            f"EVPI = **{analysis['evpi']:,.1f}** là mức tối đa đáng trả "
            "cho thông tin hoàn hảo về kịch bản tương lai."
        )

    with st.expander(
        "c) Khi nào chọn robust thay vì SP?",
        expanded=True,
    ):
        st.markdown(
            "Robust phù hợp khi cơ quan quản lý ưu tiên bảo vệ kết quả tệ nhất "
            "và không tin cậy hoàn toàn vào xác suất kịch bản. SP phù hợp hơn "
            "khi xác suất có cơ sở và mục tiêu là tối đa hóa kết quả kỳ vọng."
        )


def _b11_actions():
    """
    Mỗi hành động biểu diễn một cơ cấu chính sách:
    [tăng GDP, tăng số hóa, giảm thất nghiệp, tăng rủi ro].
    """
    return [
        (
            "Hạ tầng số",
            np.array(
                [0.055, 0.045, 0.018, 0.012],
                dtype=float,
            ),
        ),
        (
            "Chuyển đổi số SME",
            np.array(
                [0.045, 0.065, 0.026, 0.018],
                dtype=float,
            ),
        ),
        (
            "AI dẫn dắt",
            np.array(
                [0.072, 0.072, 0.010, 0.050],
                dtype=float,
            ),
        ),
        (
            "Đào tạo lại",
            np.array(
                [0.030, 0.038, 0.062, -0.018],
                dtype=float,
            ),
        ),
        (
            "Cân bằng",
            np.array(
                [0.050, 0.052, 0.040, 0.008],
                dtype=float,
            ),
        ),
    ]


def _b11_transition(
    state,
    action,
    rng,
):
    """
    State = [GDP, Digital, Unemployment, Risk], đều chuẩn hóa [0,1].
    """
    state = np.asarray(
        state,
        dtype=float,
    )

    _, effect = _b11_actions()[
        int(action)
    ]

    gdp, digital, unemployment, risk = state

    macro_shock = float(
        rng.normal(
            0.0,
            0.018,
        )
    )

    technology_shock = float(
        rng.normal(
            0.0,
            0.014,
        )
    )

    labor_shock = float(
        rng.normal(
            0.0,
            0.012,
        )
    )

    risk_shock = float(
        rng.normal(
            0.0,
            0.010,
        )
    )

    next_gdp = (
        0.82 * gdp
        + effect[0]
        + 0.12 * digital
        - 0.05 * unemployment
        + macro_shock
    )

    next_digital = (
        0.88 * digital
        + effect[1]
        + technology_shock
    )

    next_unemployment = (
        0.90 * unemployment
        - effect[2]
        + 0.025 * risk
        + labor_shock
    )

    next_risk = (
        0.86 * risk
        + effect[3]
        + 0.030 * digital
        + risk_shock
    )

    next_state = np.clip(
        np.array(
            [
                next_gdp,
                next_digital,
                next_unemployment,
                next_risk,
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )

    reward = float(
        2.20 * next_state[0]
        + 1.65 * next_state[1]
        - 2.00 * next_state[2]
        - 1.55 * next_state[3]
        - 0.04
        * abs(
            int(action) - 4
        )
    )

    return next_state, reward


def _b11_discretize(
    state,
    bins=5,
):
    state = np.clip(
        np.asarray(
            state,
            dtype=float,
        ),
        0.0,
        1.0,
    )

    indexes = np.floor(
        state * bins
    ).astype(int)

    indexes = np.clip(
        indexes,
        0,
        bins - 1,
    )

    return tuple(
        indexes.tolist()
    )


def _b11_random_initial_state(
    rng,
):
    return np.array(
        [
            rng.uniform(0.25, 0.65),
            rng.uniform(0.20, 0.65),
            rng.uniform(0.25, 0.70),
            rng.uniform(0.15, 0.55),
        ],
        dtype=np.float32,
    )


@st.cache_data(show_spinner=False)
def _b11_train_tabular_q(
    episodes=10000,
    bins=5,
    horizon=20,
    alpha=0.12,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=42,
):
    rng = np.random.default_rng(
        int(seed)
    )

    n_actions = len(
        _b11_actions()
    )

    q_table = np.zeros(
        (
            bins,
            bins,
            bins,
            bins,
            n_actions,
        ),
        dtype=float,
    )

    episode_rewards = []
    epsilon_values = []

    for episode in range(
        int(episodes)
    ):
        state = _b11_random_initial_state(
            rng
        )

        epsilon = max(
            epsilon_end,
            epsilon_start
            * (
                1.0
                - episode
                / max(
                    episodes * 0.85,
                    1.0,
                )
            ),
        )

        total_reward = 0.0

        for _ in range(
            int(horizon)
        ):
            discrete_state = (
                _b11_discretize(
                    state,
                    bins=bins,
                )
            )

            if rng.random() < epsilon:
                action = int(
                    rng.integers(
                        n_actions
                    )
                )
            else:
                action = int(
                    np.argmax(
                        q_table[
                            discrete_state
                        ]
                    )
                )

            next_state, reward = (
                _b11_transition(
                    state,
                    action,
                    rng,
                )
            )

            next_discrete = (
                _b11_discretize(
                    next_state,
                    bins=bins,
                )
            )

            best_next = float(
                np.max(
                    q_table[
                        next_discrete
                    ]
                )
            )

            current_q = q_table[
                discrete_state
                + (action,)
            ]

            q_table[
                discrete_state
                + (action,)
            ] = (
                current_q
                + alpha
                * (
                    reward
                    + gamma * best_next
                    - current_q
                )
            )

            state = next_state
            total_reward += reward

        episode_rewards.append(
            total_reward
        )
        epsilon_values.append(
            epsilon
        )

    history = pd.DataFrame(
        {
            "Episode": np.arange(
                1,
                episodes + 1,
            ),
            "Reward": episode_rewards,
            "Epsilon": epsilon_values,
        }
    )

    history[
        "Reward MA100"
    ] = (
        history["Reward"]
        .rolling(
            100,
            min_periods=1,
        )
        .mean()
    )

    return q_table, history


def _b11_evaluate_policy(
    policy_function,
    episodes=200,
    horizon=20,
    seed=123,
):
    rng = np.random.default_rng(
        int(seed)
    )

    rewards = []
    action_counts = np.zeros(
        len(_b11_actions()),
        dtype=int,
    )

    for _ in range(
        int(episodes)
    ):
        state = _b11_random_initial_state(
            rng
        )
        total_reward = 0.0

        for _ in range(
            int(horizon)
        ):
            action = int(
                policy_function(
                    state
                )
            )

            action_counts[action] += 1

            state, reward = (
                _b11_transition(
                    state,
                    action,
                    rng,
                )
            )

            total_reward += reward

        rewards.append(
            total_reward
        )

    return {
        "mean": float(
            np.mean(rewards)
        ),
        "std": float(
            np.std(rewards)
        ),
        "min": float(
            np.min(rewards)
        ),
        "max": float(
            np.max(rewards)
        ),
        "action_counts": action_counts,
        "rewards": np.asarray(
            rewards,
            dtype=float,
        ),
    }


def _b11_q_policy(
    q_table,
    bins=5,
):
    def policy(state):
        discrete_state = (
            _b11_discretize(
                state,
                bins=bins,
            )
        )

        return int(
            np.argmax(
                q_table[
                    discrete_state
                ]
            )
        )

    return policy


def _b11_rule_policy(
    state,
):
    gdp, digital, unemployment, risk = state

    if unemployment >= 0.55:
        return 3

    if risk >= 0.55:
        return 4

    if digital <= 0.35:
        return 1

    if gdp <= 0.35 and risk < 0.45:
        return 2

    return 4


def _b11_make_env(
    seed=42,
    horizon=20,
):
    import gymnasium as gym
    from gymnasium import spaces

    class AIDEOMPolicyEnv(
        gym.Env
    ):
        metadata = {
            "render_modes": [],
        }

        def __init__(self):
            super().__init__()

            self.observation_space = (
                spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            )

            self.action_space = (
                spaces.Discrete(
                    len(
                        _b11_actions()
                    )
                )
            )

            self.horizon = int(
                horizon
            )

            self._seed = int(
                seed
            )

            self.rng = (
                np.random.default_rng(
                    self._seed
                )
            )

            self.state = None
            self.step_count = 0
            self.episode_reward = 0.0

        def reset(
            self,
            *,
            seed=None,
            options=None,
        ):
            super().reset(
                seed=seed
            )

            if seed is not None:
                self.rng = (
                    np.random.default_rng(
                        int(seed)
                    )
                )

            self.state = (
                _b11_random_initial_state(
                    self.rng
                )
            )

            self.step_count = 0
            self.episode_reward = 0.0

            return (
                self.state.copy(),
                {},
            )

        def step(
            self,
            action,
        ):
            next_state, reward = (
                _b11_transition(
                    self.state,
                    int(action),
                    self.rng,
                )
            )

            self.state = next_state
            self.step_count += 1
            self.episode_reward += reward

            terminated = False
            truncated = (
                self.step_count
                >= self.horizon
            )

            info = {}

            if truncated:
                info[
                    "episode_reward"
                ] = float(
                    self.episode_reward
                )

            return (
                self.state.copy(),
                float(reward),
                terminated,
                truncated,
                info,
            )

    return AIDEOMPolicyEnv()


@st.cache_resource(show_spinner=False)
def _b11_train_dqn(
    timesteps=10000,
    seed=42,
):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeRewardCallback(
        BaseCallback
    ):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []

        def _on_step(self):
            infos = self.locals.get(
                "infos",
                [],
            )

            for info in infos:
                if (
                    isinstance(
                        info,
                        dict,
                    )
                    and "episode" in info
                ):
                    self.episode_rewards.append(
                        float(
                            info[
                                "episode"
                            ]["r"]
                        )
                    )

            return True

    def make_training_env():
        return Monitor(
            _b11_make_env(
                seed=int(seed),
                horizon=20,
            )
        )

    vector_env = DummyVecEnv(
        [
            make_training_env
        ]
    )

    callback = (
        EpisodeRewardCallback()
    )

    model = DQN(
        policy="MlpPolicy",
        env=vector_env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.40,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[
                64,
                64,
            ]
        ),
        seed=int(seed),
        verbose=0,
        device="cpu",
    )

    model.learn(
        total_timesteps=int(
            timesteps
        ),
        callback=callback,
        progress_bar=False,
    )

    history = pd.DataFrame(
        {
            "Episode": np.arange(
                1,
                len(
                    callback.episode_rewards
                )
                + 1,
            ),
            "Reward": callback.episode_rewards,
        }
    )

    if not history.empty:
        history[
            "Reward MA20"
        ] = (
            history["Reward"]
            .rolling(
                20,
                min_periods=1,
            )
            .mean()
        )

    return model, history


def _b11_dqn_policy(
    model,
):
    def policy(state):
        action, _ = model.predict(
            np.asarray(
                state,
                dtype=np.float32,
            ),
            deterministic=True,
        )

        return int(
            np.asarray(
                action
            ).item()
        )

    return policy


def page_11():
    hero(
        "Bài 11 — Học tăng cường cho chính sách kinh tế thích nghi",
        "Xây dựng môi trường Gymnasium đúng chuẩn, huấn luyện Q-learning bảng và DQN thật; đánh giá ngoài mẫu so với chính sách quy tắc.",
        ["11.1-11.4", "Gymnasium", "Q-learning", "DQN", "Stable-Baselines3"],
    )

    st.markdown("## 11.1. Bối cảnh Việt Nam")
    st.markdown(
        """
        Cơ cấu đầu tư số phù hợp phụ thuộc vào trạng thái kinh tế hiện tại.
        Khi thất nghiệp cao, đào tạo lại có thể quan trọng hơn AI; khi năng lực số thấp,
        nền tảng số cho doanh nghiệp có thể tạo tác động lớn hơn đầu tư công nghệ biên.
        Học tăng cường mô hình hóa chuỗi quyết định thích nghi theo trạng thái.
        """
    )

    st.markdown("## 11.2. Môi trường MDP")
    st.latex(
        r"s_t=(GDP_t,Digital_t,Unemployment_t,Risk_t)"
    )
    st.latex(
        r"a_t\in\{Infrastructure,SME,AI,Training,Balanced\}"
    )
    st.latex(
        r"R_t=2.2GDP_t+1.65Digital_t"
        r"-2.0Unemployment_t-1.55Risk_t"
    )

    action_table = pd.DataFrame(
        {
            "Action ID": np.arange(
                len(
                    _b11_actions()
                )
            ),
            "Chính sách": [
                item[0]
                for item in _b11_actions()
            ],
        }
    )

    st.dataframe(
        action_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## 11.3. Yêu cầu lập trình")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "11.3.1 - Gymnasium Env",
            "11.3.2 - Q-learning",
            "11.3.3 - Chính sách học được",
            "11.3.4 - Đánh giá ngoài mẫu",
            "11.3.5 - DQN thật",
        ]
    )

    q_table, q_history = (
        _b11_train_tabular_q(
            episodes=10000,
            bins=5,
            horizon=20,
            seed=42,
        )
    )

    with tab1:
        try:
            import gymnasium
            env = _b11_make_env(
                seed=42,
                horizon=20,
            )

            observation, info = (
                env.reset(
                    seed=42
                )
            )

            next_observation, reward, terminated, truncated, info = (
                env.step(4)
            )

            st.success(
                "Môi trường kế thừa gymnasium.Env và chạy đúng API reset/step."
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Thuộc tính": [
                            "Observation space",
                            "Action space",
                            "Observation mẫu",
                            "Reward mẫu",
                            "terminated",
                            "truncated",
                        ],
                        "Giá trị": [
                            str(
                                env.observation_space
                            ),
                            str(
                                env.action_space
                            ),
                            np.round(
                                observation,
                                4,
                            ).tolist(),
                            round(
                                reward,
                                4,
                            ),
                            terminated,
                            truncated,
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(
                f"Gymnasium chưa chạy được: {exc}"
            )

    with tab2:
        kpi_cards(
            [
                (
                    "Episodes",
                    "10.000",
                    "Q-learning bảng",
                ),
                (
                    "Reward đầu",
                    f"{q_history['Reward MA100'].iloc[99]:.2f}",
                    "MA100",
                ),
                (
                    "Reward cuối",
                    f"{q_history['Reward MA100'].iloc[-1]:.2f}",
                    "MA100",
                ),
                (
                    "Epsilon cuối",
                    f"{q_history['Epsilon'].iloc[-1]:.3f}",
                    "exploration",
                ),
            ]
        )

        fig = px.line(
            q_history,
            x="Episode",
            y="Reward MA100",
            template=PLOT_TEMPLATE,
            title="Learning curve của Q-learning",
        )
        fig.update_layout(
            height=480,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab3:
        representative_states = pd.DataFrame(
            [
                [
                    "Tăng trưởng thấp, số hóa thấp, thất nghiệp cao",
                    0.25,
                    0.25,
                    0.70,
                    0.35,
                ],
                [
                    "Tăng trưởng trung bình, số hóa thấp",
                    0.50,
                    0.30,
                    0.45,
                    0.30,
                ],
                [
                    "Tăng trưởng cao, số hóa cao, rủi ro thấp",
                    0.75,
                    0.75,
                    0.25,
                    0.20,
                ],
                [
                    "Số hóa cao nhưng rủi ro cao",
                    0.60,
                    0.80,
                    0.35,
                    0.70,
                ],
            ],
            columns=[
                "Trạng thái",
                "GDP",
                "Digital",
                "Unemployment",
                "Risk",
            ],
        )

        q_policy = _b11_q_policy(
            q_table,
            bins=5,
        )

        representative_states[
            "Action ID"
        ] = representative_states.apply(
            lambda row: q_policy(
                np.array(
                    [
                        row["GDP"],
                        row["Digital"],
                        row["Unemployment"],
                        row["Risk"],
                    ],
                    dtype=float,
                )
            ),
            axis=1,
        )

        action_names = [
            item[0]
            for item in _b11_actions()
        ]

        representative_states[
            "Chính sách"
        ] = representative_states[
            "Action ID"
        ].map(
            {
                i: action_names[i]
                for i in range(
                    len(
                        action_names
                    )
                )
            }
        )

        st.dataframe(
            representative_states,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Tải chính sách đại diện Bài 11",
            data=representative_states.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
            file_name="bai11_qlearning_policy.csv",
            mime="text/csv",
            key="download_bai11_policy",
        )

    with tab4:
        q_eval = _b11_evaluate_policy(
            _b11_q_policy(
                q_table,
                bins=5,
            ),
            episodes=300,
            seed=777,
        )

        rule_eval = _b11_evaluate_policy(
            _b11_rule_policy,
            episodes=300,
            seed=777,
        )

        fixed_results = []

        for action_id, (
            action_name,
            _,
        ) in enumerate(
            _b11_actions()
        ):
            evaluation = (
                _b11_evaluate_policy(
                    lambda state, a=action_id: a,
                    episodes=300,
                    seed=777,
                )
            )

            fixed_results.append(
                [
                    action_name,
                    evaluation["mean"],
                    evaluation["std"],
                ]
            )

        comparison = pd.DataFrame(
            [
                [
                    "Q-learning",
                    q_eval["mean"],
                    q_eval["std"],
                ],
                [
                    "Rule-based",
                    rule_eval["mean"],
                    rule_eval["std"],
                ],
            ]
            + fixed_results,
            columns=[
                "Chính sách",
                "Reward trung bình",
                "Độ lệch chuẩn",
            ],
        ).sort_values(
            "Reward trung bình",
            ascending=False,
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            comparison,
            x="Chính sách",
            y="Reward trung bình",
            error_y="Độ lệch chuẩn",
            template=PLOT_TEMPLATE,
            title="Đánh giá ngoài mẫu 300 episodes",
        )
        fig.update_layout(
            height=470,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab5:
        dependency_table = pd.DataFrame(
            {
                "Thư viện": [
                    "gymnasium",
                    "stable-baselines3",
                ],
                "Yêu cầu": [
                    "1.2.3",
                    "2.8.0",
                ],
            }
        )

        st.dataframe(
            dependency_table,
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)

        dqn_timesteps = c1.select_slider(
            "Total timesteps",
            options=[
                5000,
                10000,
                20000,
                50000,
            ],
            value=10000,
            key="b11_dqn_timesteps",
        )

        dqn_seed = c2.number_input(
            "DQN seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            key="b11_dqn_seed",
        )

        if st.button(
            "Huấn luyện DQN",
            type="primary",
            key="b11_train_dqn_button",
        ):
            st.session_state[
                "b11_dqn_configuration"
            ] = (
                int(
                    dqn_timesteps
                ),
                int(
                    dqn_seed
                ),
            )

        configuration = st.session_state.get(
            "b11_dqn_configuration"
        )

        if configuration is None:
            st.info(
                "Nhấn “Huấn luyện DQN”. Chọn 50.000 timesteps cho lần chạy cuối; "
                "10.000 phù hợp để kiểm tra nhanh."
            )
        else:
            selected_timesteps, selected_seed = (
                configuration
            )

            try:
                with st.spinner(
                    "Đang huấn luyện DQN trên CPU..."
                ):
                    model, dqn_history = (
                        _b11_train_dqn(
                            timesteps=selected_timesteps,
                            seed=selected_seed,
                        )
                    )

                dqn_eval = (
                    _b11_evaluate_policy(
                        _b11_dqn_policy(
                            model
                        ),
                        episodes=300,
                        seed=888,
                    )
                )

                q_eval = (
                    _b11_evaluate_policy(
                        _b11_q_policy(
                            q_table,
                            bins=5,
                        ),
                        episodes=300,
                        seed=888,
                    )
                )

                rule_eval = (
                    _b11_evaluate_policy(
                        _b11_rule_policy,
                        episodes=300,
                        seed=888,
                    )
                )

                kpi_cards(
                    [
                        (
                            "DQN reward",
                            f"{dqn_eval['mean']:.2f}",
                            f"std={dqn_eval['std']:.2f}",
                        ),
                        (
                            "Q-learning reward",
                            f"{q_eval['mean']:.2f}",
                            f"std={q_eval['std']:.2f}",
                        ),
                        (
                            "Rule reward",
                            f"{rule_eval['mean']:.2f}",
                            f"std={rule_eval['std']:.2f}",
                        ),
                        (
                            "Timesteps",
                            f"{selected_timesteps:,}",
                            "Stable-Baselines3",
                        ),
                    ]
                )

                if not dqn_history.empty:
                    fig = px.line(
                        dqn_history,
                        x="Episode",
                        y="Reward MA20",
                        template=PLOT_TEMPLATE,
                        title="Learning curve DQN",
                    )
                    fig.update_layout(
                        height=470,
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                comparison = pd.DataFrame(
                    {
                        "Phương pháp": [
                            "DQN",
                            "Q-learning",
                            "Rule-based",
                        ],
                        "Reward trung bình": [
                            dqn_eval["mean"],
                            q_eval["mean"],
                            rule_eval["mean"],
                        ],
                        "Độ lệch chuẩn": [
                            dqn_eval["std"],
                            q_eval["std"],
                            rule_eval["std"],
                        ],
                    }
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.error(
                    f"DQN chưa huấn luyện được: {exc}"
                )

    st.markdown("## 11.4. Thảo luận chính sách")

    with st.expander(
        "a) Khi thất nghiệp cao, mô hình nên phản ứng thế nào?",
        expanded=True,
    ):
        st.markdown(
            "Reward phạt thất nghiệp mạnh, nên Q-learning và DQN thường ưu tiên "
            "đào tạo lại hoặc phương án cân bằng thay vì AI thuần túy."
        )

    with st.expander(
        "b) Vì sao phải đánh giá ngoài mẫu?",
        expanded=True,
    ):
        st.markdown(
            "Reward trong quá trình học có thể cao do khai thác đúng các chuỗi trạng thái "
            "đã gặp. Đánh giá bằng seed mới kiểm tra khả năng khái quát của chính sách."
        )

    with st.expander(
        "c) Có nên để DQN tự quyết định chính sách thật không?",
        expanded=True,
    ):
        st.markdown(
            "Không. DQN là công cụ mô phỏng và khuyến nghị. Quyết định thật cần giới hạn "
            "an toàn, giải thích mô hình, kiểm định dữ liệu và phê duyệt của con người."
        )


def _b12_scenarios():
    return {
        "S1 - Truyền thống": np.array(
            [0.70, 0.10, 0.10, 0.10],
            dtype=float,
        ),
        "S2 - Số hóa nhanh": np.array(
            [0.25, 0.45, 0.15, 0.15],
            dtype=float,
        ),
        "S3 - AI dẫn dắt": np.array(
            [0.20, 0.20, 0.45, 0.15],
            dtype=float,
        ),
        "S4 - Bao trùm số": np.array(
            [0.30, 0.20, 0.10, 0.40],
            dtype=float,
        ),
        "S5 - Tối ưu cân bằng": np.array(
            [0.34, 0.26, 0.18, 0.22],
            dtype=float,
        ),
    }


def _b12_module_1_forecast(
    shares,
):
    simulation = simulate_dynamic(
        shares=shares,
        start=2026,
        end=2030,
        invest_rate=0.22,
        shock_2028=0.03,
    )

    first = simulation.iloc[0]
    last = simulation.iloc[-1]

    return {
        "GDP_2030": float(
            last["Y_GDP"]
        ),
        "Consumption_2030": float(
            last["C_tiêu_dùng"]
        ),
        "GDP_Growth_2026_2030": float(
            100
            * (
                last["Y_GDP"]
                / first["Y_GDP"]
                - 1
            )
        ),
        "Digital_2030": float(
            last["D"]
        ),
        "AI_2030": float(
            last["AI"]
        ),
        "Human_2030": float(
            last["H"]
        ),
        "trajectory": simulation,
    }


def _b12_module_2_region_readiness():
    df = load_regions().copy()

    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]

    weights = np.array(
        [
            0.10,
            0.10,
            0.15,
            0.20,
            0.15,
            0.15,
            0.05,
            0.10,
        ],
        dtype=float,
    )

    benefit_flags = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    score = topsis_score(
        df,
        criteria,
        weights,
        benefit_flags,
    )

    result = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            "ReadinessScore": score,
        }
    )

    result["ReadinessRank"] = (
        result["ReadinessScore"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result.sort_values(
        "ReadinessRank"
    ).reset_index(drop=True)


def _b12_module_3_allocation():
    """
    Ưu tiên dùng solver Bài 4. Nếu Bài 4 chưa được thay,
    dùng một LP tích hợp tối giản để dashboard vẫn chạy.
    """
    regions, items, beta, d0 = region_beta_matrix()

    solver_function = globals().get(
        "_b4_solve_scipy"
    )

    if callable(
        solver_function
    ):
        result = solver_function(
            fairness=True,
            total_budget=50000.0,
            region_floor=5000.0,
            region_cap=12000.0,
            human_floor=12000.0,
            gamma=0.002,
            lam=0.68,
        )

        if result.success:
            x = np.asarray(
                result.x[:24],
                dtype=float,
            ).reshape(6, 4)

            objective = float(
                -result.fun
            )

            source = "Bài 4 - SciPy LP"
        else:
            x = None
    else:
        x = None

    if x is None:
        n = 24
        c = -beta.reshape(-1)

        a_ub = []
        b_ub = []

        row = np.ones(
            n,
            dtype=float,
        )
        a_ub.append(
            row
        )
        b_ub.append(
            50000.0
        )

        for region_index in range(6):
            row = np.zeros(
                n,
                dtype=float,
            )
            row[
                region_index * 4:
                region_index * 4 + 4
            ] = -1.0
            a_ub.append(
                row
            )
            b_ub.append(
                -5000.0
            )

            row = np.zeros(
                n,
                dtype=float,
            )
            row[
                region_index * 4:
                region_index * 4 + 4
            ] = 1.0
            a_ub.append(
                row
            )
            b_ub.append(
                12000.0
            )

        row = np.zeros(
            n,
            dtype=float,
        )
        for region_index in range(6):
            row[
                region_index * 4 + 3
            ] = -1.0

        a_ub.append(
            row
        )
        b_ub.append(
            -12000.0
        )

        result = linprog(
            c,
            A_ub=np.asarray(
                a_ub,
                dtype=float,
            ),
            b_ub=np.asarray(
                b_ub,
                dtype=float,
            ),
            bounds=[
                (0.0, None)
            ]
            * n,
            method="highs",
        )

        if not result.success:
            raise RuntimeError(
                "M3 không giải được LP."
            )

        x = np.asarray(
            result.x,
            dtype=float,
        ).reshape(6, 4)

        objective = float(
            -result.fun
        )

        source = "LP tích hợp dự phòng"

    allocation = pd.DataFrame(
        x,
        columns=items,
    )

    allocation.insert(
        0,
        "Vùng",
        regions,
    )

    allocation["Tổng vùng"] = (
        x.sum(axis=1)
    )

    allocation[
        "Digital sau đầu tư"
    ] = (
        d0
        + 0.002
        * x[:, 1]
    )

    return {
        "table": allocation,
        "objective": objective,
        "source": source,
        "total_budget": float(
            x.sum()
        ),
        "human_budget": float(
            x[:, 3].sum()
        ),
    }


def _b12_module_4_labor(
    shares,
):
    sectors = load_sectors().copy()

    ai_share = float(
        shares[2]
    )
    human_share = float(
        shares[3]
    )
    digital_share = float(
        shares[1]
    )

    readiness = (
        sectors[
            "ai_readiness_0_100"
        ]
        / 100.0
    )

    automation = (
        sectors[
            "automation_risk_pct"
        ]
        / 100.0
    )

    labor = sectors[
        "labor_million"
    ]

    jobs_created = (
        labor
        * (
            0.018
            + 0.055
            * digital_share
            + 0.050
            * ai_share
            * readiness
            + 0.048
            * human_share
        )
    )

    jobs_displaced = (
        labor
        * automation
        * (
            0.018
            + 0.065
            * ai_share
        )
    )

    jobs_retrained = (
        labor
        * (
            0.018
            + 0.090
            * human_share
        )
    )

    net_jobs = (
        jobs_created
        + 0.55
        * jobs_retrained
        - jobs_displaced
    )

    result = pd.DataFrame(
        {
            "Ngành": sectors[
                "sector_name_vi"
            ],
            "JobsCreated_million": jobs_created,
            "JobsDisplaced_million": jobs_displaced,
            "JobsRetrained_million": jobs_retrained,
            "NetJobs_million": net_jobs,
        }
    )

    return {
        "table": result.sort_values(
            "NetJobs_million",
            ascending=False,
        ).reset_index(drop=True),
        "net_jobs_total": float(
            net_jobs.sum()
        ),
        "displaced_total": float(
            jobs_displaced.sum()
        ),
        "retrained_total": float(
            jobs_retrained.sum()
        ),
    }


def _b12_module_5_risk(
    shares,
):
    k_share = float(
        shares[0]
    )
    d_share = float(
        shares[1]
    )
    ai_share = float(
        shares[2]
    )
    h_share = float(
        shares[3]
    )

    cyber_risk = float(
        np.clip(
            100
            * (
                0.48
                * ai_share
                + 0.25
                * d_share
                - 0.20
                * h_share
            ),
            0.0,
            100.0,
        )
    )

    emission_risk = float(
        np.clip(
            100
            * (
                0.34
                * k_share
                + 0.38
                * ai_share
                + 0.08
                * d_share
            ),
            0.0,
            100.0,
        )
    )

    inclusion_score = float(
        np.clip(
            100
            * (
                0.58
                * h_share
                + 0.30
                * d_share
                + 0.12
                * k_share
            ),
            0.0,
            100.0,
        )
    )

    concentration_risk = float(
        100
        * np.max(
            shares
        )
    )

    return {
        "CyberRisk": cyber_risk,
        "EmissionRisk": emission_risk,
        "InclusionScore": inclusion_score,
        "ConcentrationRisk": concentration_risk,
    }


@st.cache_data(show_spinner=False)
def _b12_run_pipeline():
    scenarios = _b12_scenarios()

    readiness = (
        _b12_module_2_region_readiness()
    )

    allocation = (
        _b12_module_3_allocation()
    )

    rows = []
    trajectories = {}
    labor_results = {}

    for scenario_name, shares in scenarios.items():
        forecast = (
            _b12_module_1_forecast(
                shares
            )
        )

        labor = (
            _b12_module_4_labor(
                shares
            )
        )

        risk = (
            _b12_module_5_risk(
                shares
            )
        )

        trajectories[
            scenario_name
        ] = forecast[
            "trajectory"
        ]

        labor_results[
            scenario_name
        ] = labor[
            "table"
        ]

        rows.append(
            {
                "Kịch bản": scenario_name,
                "Share_K": float(
                    shares[0]
                ),
                "Share_D": float(
                    shares[1]
                ),
                "Share_AI": float(
                    shares[2]
                ),
                "Share_H": float(
                    shares[3]
                ),
                "GDP_2030": forecast[
                    "GDP_2030"
                ],
                "Consumption_2030": forecast[
                    "Consumption_2030"
                ],
                "GDP_Growth_2026_2030": forecast[
                    "GDP_Growth_2026_2030"
                ],
                "Digital_2030": forecast[
                    "Digital_2030"
                ],
                "AI_2030": forecast[
                    "AI_2030"
                ],
                "Human_2030": forecast[
                    "Human_2030"
                ],
                "NetJobs_million": labor[
                    "net_jobs_total"
                ],
                "Displaced_million": labor[
                    "displaced_total"
                ],
                "Retrained_million": labor[
                    "retrained_total"
                ],
                **risk,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "GDP_norm"
    ] = minmax(
        result[
            "GDP_2030"
        ]
    )

    result[
        "Jobs_norm"
    ] = minmax(
        result[
            "NetJobs_million"
        ]
    )

    result[
        "Inclusion_norm"
    ] = minmax(
        result[
            "InclusionScore"
        ]
    )

    result[
        "Cyber_norm"
    ] = minmax(
        result[
            "CyberRisk"
        ]
    )

    result[
        "Emission_norm"
    ] = minmax(
        result[
            "EmissionRisk"
        ]
    )

    result[
        "IntegratedScore"
    ] = (
        0.35
        * result[
            "GDP_norm"
        ]
        + 0.20
        * result[
            "Jobs_norm"
        ]
        + 0.20
        * result[
            "Inclusion_norm"
        ]
        + 0.15
        * (
            1
            - result[
                "Cyber_norm"
            ]
        )
        + 0.10
        * (
            1
            - result[
                "Emission_norm"
            ]
        )
    )

    result[
        "IntegratedRank"
    ] = (
        result[
            "IntegratedScore"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return {
        "scenarios": result.sort_values(
            "IntegratedRank"
        ).reset_index(drop=True),
        "trajectories": trajectories,
        "readiness": readiness,
        "allocation": allocation,
        "labor_results": labor_results,
    }


def _b12_validation(
    pipeline,
):
    scenarios = pipeline[
        "scenarios"
    ]

    allocation = pipeline[
        "allocation"
    ]

    readiness = pipeline[
        "readiness"
    ]

    checks = [
        {
            "Kiểm tra": "5 kịch bản được tạo",
            "Đạt": len(
                scenarios
            ) == 5,
            "Giá trị": len(
                scenarios
            ),
        },
        {
            "Kiểm tra": "Tỷ trọng mỗi kịch bản bằng 1",
            "Đạt": bool(
                np.allclose(
                    scenarios[
                        [
                            "Share_K",
                            "Share_D",
                            "Share_AI",
                            "Share_H",
                        ]
                    ].sum(axis=1),
                    1.0,
                )
            ),
            "Giá trị": scenarios[
                [
                    "Share_K",
                    "Share_D",
                    "Share_AI",
                    "Share_H",
                ]
            ].sum(axis=1).round(
                6
            ).tolist(),
        },
        {
            "Kiểm tra": "GDP 2030 hữu hạn và dương",
            "Đạt": bool(
                np.isfinite(
                    scenarios[
                        "GDP_2030"
                    ]
                ).all()
                and (
                    scenarios[
                        "GDP_2030"
                    ]
                    > 0
                ).all()
            ),
            "Giá trị": float(
                scenarios[
                    "GDP_2030"
                ].min()
            ),
        },
        {
            "Kiểm tra": "Ngân sách M3 không vượt 50.000",
            "Đạt": (
                allocation[
                    "total_budget"
                ]
                <= 50000.0
                + 1e-6
            ),
            "Giá trị": allocation[
                "total_budget"
            ],
        },
        {
            "Kiểm tra": "Nhân lực M3 tối thiểu 12.000",
            "Đạt": (
                allocation[
                    "human_budget"
                ]
                >= 12000.0
                - 1e-6
            ),
            "Giá trị": allocation[
                "human_budget"
            ],
        },
        {
            "Kiểm tra": "M2 xếp hạng đủ 6 vùng",
            "Đạt": (
                len(
                    readiness
                )
                == 6
                and set(
                    readiness[
                        "ReadinessRank"
                    ]
                )
                == set(
                    range(
                        1,
                        7,
                    )
                )
            ),
            "Giá trị": readiness[
                "ReadinessRank"
            ].tolist(),
        },
        {
            "Kiểm tra": "IntegratedScore nằm trong [0,1]",
            "Đạt": bool(
                scenarios[
                    "IntegratedScore"
                ].between(
                    0.0,
                    1.0,
                ).all()
            ),
            "Giá trị": [
                float(
                    scenarios[
                        "IntegratedScore"
                    ].min()
                ),
                float(
                    scenarios[
                        "IntegratedScore"
                    ].max()
                ),
            ],
        },
    ]

    return pd.DataFrame(
        checks
    )


def page_12():
    hero(
        "Bài 12 — Hệ thống hỗ trợ quyết định tích hợp AIDEOM-VN",
        "Kết nối thật sáu module từ dự báo, xếp hạng vùng, phân bổ ngân sách, lao động và rủi ro đến dashboard so sánh năm kịch bản và kiểm định pipeline.",
        ["12.1-12.6", "Integrated pipeline", "6 modules", "5 scenarios", "Validation"],
    )

    with st.spinner(
        "Đang chạy pipeline M1-M6..."
    ):
        pipeline = (
            _b12_run_pipeline()
        )

    scenarios = pipeline[
        "scenarios"
    ]

    st.markdown("## 12.1. Kiến trúc sáu module")

    architecture = pd.DataFrame(
        [
            [
                "M1",
                "Dự báo kinh tế",
                "compute_tfp + simulate_dynamic",
                "GDP, D, AI, H năm 2030",
            ],
            [
                "M2",
                "Sẵn sàng vùng",
                "TOPSIS từ dữ liệu 6 vùng",
                "Điểm và thứ hạng vùng",
            ],
            [
                "M3",
                "Phân bổ ngân sách",
                "LP Bài 4 hoặc LP dự phòng",
                "Ma trận 6 vùng × 4 hạng mục",
            ],
            [
                "M4",
                "Lao động và AI",
                "Dữ liệu 10 ngành",
                "Việc làm tạo mới, mất đi, đào tạo lại",
            ],
            [
                "M5",
                "Rủi ro",
                "Cơ cấu K-D-AI-H",
                "Cyber, phát thải, bao trùm, tập trung",
            ],
            [
                "M6",
                "Dashboard",
                "Đầu ra M1-M5",
                "KPI, xếp hạng, cảnh báo, tải CSV",
            ],
        ],
        columns=[
            "Module",
            "Chức năng",
            "Nguồn đầu vào",
            "Đầu ra",
        ],
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## 12.2. Năm kịch bản")

    st.dataframe(
        scenarios[
            [
                "Kịch bản",
                "Share_K",
                "Share_D",
                "Share_AI",
                "Share_H",
                "GDP_2030",
                "NetJobs_million",
                "InclusionScore",
                "CyberRisk",
                "EmissionRisk",
                "IntegratedScore",
                "IntegratedRank",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    best = scenarios.iloc[0]

    kpi_cards(
        [
            (
                "Kịch bản số 1",
                best[
                    "Kịch bản"
                ],
                f"Điểm={best['IntegratedScore']:.4f}",
            ),
            (
                "GDP 2030",
                f"{best['GDP_2030']:,.1f}",
                "kịch bản dẫn đầu",
            ),
            (
                "Net jobs",
                f"{best['NetJobs_million']:.3f} triệu",
                "M4",
            ),
            (
                "Bao trùm",
                f"{best['InclusionScore']:.2f}",
                "M5",
            ),
        ]
    )

    st.markdown("## 12.3. Dashboard tích hợp")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "M1 - Dự báo",
            "M2 - Vùng",
            "M3 - Phân bổ",
            "M4 - Lao động",
            "M5 - Rủi ro",
            "M6 - Kiểm định",
        ]
    )

    with tab1:
        selected = st.multiselect(
            "Chọn kịch bản",
            options=list(
                pipeline[
                    "trajectories"
                ].keys()
            ),
            default=list(
                pipeline[
                    "trajectories"
                ].keys()
            ),
            key="b12_forecast_scenarios",
        )

        rows = []

        for scenario_name in selected:
            temp = pipeline[
                "trajectories"
            ][scenario_name].copy()

            temp[
                "Kịch bản"
            ] = scenario_name

            rows.append(
                temp
            )

        if rows:
            forecast_df = pd.concat(
                rows,
                ignore_index=True,
            )

            fig = px.line(
                forecast_df,
                x="Năm",
                y="Y_GDP",
                color="Kịch bản",
                markers=True,
                template=PLOT_TEMPLATE,
                title="Quỹ đạo GDP 2026-2030",
            )
            fig.update_layout(
                height=500,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with tab2:
        readiness = pipeline[
            "readiness"
        ]

        st.dataframe(
            readiness,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            readiness.sort_values(
                "ReadinessScore"
            ),
            x="ReadinessScore",
            y="Vùng",
            orientation="h",
            template=PLOT_TEMPLATE,
            title="M2 - Sẵn sàng AI và số hóa theo vùng",
        )
        fig.update_layout(
            height=480,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab3:
        allocation = pipeline[
            "allocation"
        ]

        st.caption(
            f"Nguồn solver: {allocation['source']} | "
            f"Z*={allocation['objective']:,.2f}"
        )

        st.dataframe(
            allocation[
                "table"
            ],
            use_container_width=True,
            hide_index=True,
        )

        long_df = allocation[
            "table"
        ].melt(
            id_vars=[
                "Vùng",
                "Tổng vùng",
                "Digital sau đầu tư",
            ],
            value_vars=[
                "I - Hạ tầng số",
                "D - CĐS DN",
                "AI",
                "H - Nhân lực số",
            ],
            var_name="Hạng mục",
            value_name="Ngân sách",
        )

        fig = px.bar(
            long_df,
            x="Vùng",
            y="Ngân sách",
            color="Hạng mục",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="M3 - Phân bổ ngân sách vùng",
        )
        fig.update_layout(
            height=520,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab4:
        selected_labor_scenario = (
            st.selectbox(
                "Kịch bản lao động",
                options=list(
                    pipeline[
                        "labor_results"
                    ].keys()
                ),
                index=4,
                key="b12_labor_scenario",
            )
        )

        labor_df = pipeline[
            "labor_results"
        ][selected_labor_scenario]

        st.dataframe(
            labor_df,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            labor_df.sort_values(
                "NetJobs_million"
            ),
            x="NetJobs_million",
            y="Ngành",
            orientation="h",
            template=PLOT_TEMPLATE,
            title="M4 - Việc làm ròng theo ngành",
        )
        fig.update_layout(
            height=560,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab5:
        risk_long = scenarios.melt(
            id_vars="Kịch bản",
            value_vars=[
                "CyberRisk",
                "EmissionRisk",
                "InclusionScore",
                "ConcentrationRisk",
            ],
            var_name="KPI",
            value_name="Điểm",
        )

        fig = px.bar(
            risk_long,
            x="Kịch bản",
            y="Điểm",
            color="KPI",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="M5 - Rủi ro và bao trùm",
        )
        fig.update_layout(
            height=500,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        warnings = []

        for _, row in scenarios.iterrows():
            if (
                row["Share_AI"]
                >= 0.40
                and row[
                    "Share_H"
                ]
                < 0.20
            ):
                warnings.append(
                    f"{row['Kịch bản']}: AI cao nhưng nhân lực thấp."
                )

            if row[
                "CyberRisk"
            ] > scenarios[
                "CyberRisk"
            ].median():
                warnings.append(
                    f"{row['Kịch bản']}: cyber risk trên trung vị."
                )

            if row[
                "InclusionScore"
            ] < scenarios[
                "InclusionScore"
            ].median():
                warnings.append(
                    f"{row['Kịch bản']}: bao trùm dưới trung vị."
                )

        for warning in warnings:
            st.markdown(
                f"<div class='warning-box'>{warning}</div>",
                unsafe_allow_html=True,
            )

    with tab6:
        validation = (
            _b12_validation(
                pipeline
            )
        )

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True,
        )

        passed = int(
            validation[
                "Đạt"
            ].sum()
        )

        kpi_cards(
            [
                (
                    "Số kiểm định đạt",
                    f"{passed}/{len(validation)}",
                    "pipeline self-test",
                ),
                (
                    "M1-M5",
                    "Đã kết nối",
                    "M6 nhận đầu ra thật",
                ),
                (
                    "Số vùng",
                    str(
                        len(
                            pipeline[
                                "readiness"
                            ]
                        )
                    ),
                    "M2",
                ),
                (
                    "Ngân sách M3",
                    f"{pipeline['allocation']['total_budget']:,.0f}",
                    "không vượt 50.000",
                ),
            ]
        )

        if bool(
            validation[
                "Đạt"
            ].all()
        ):
            st.success(
                "Pipeline vượt qua toàn bộ kiểm định nội bộ."
            )
        else:
            st.error(
                "Có kiểm định chưa đạt; không nên coi kết quả là final."
            )

    st.markdown("## 12.4. Sản phẩm bàn giao")

    deliverables = pd.DataFrame(
        [
            [
                "Dashboard",
                "12 trang Streamlit",
                "Hoàn thành",
            ],
            [
                "Mã nguồn",
                "app.py + requirements.txt + data",
                "Hoàn thành",
            ],
            [
                "Kiểm định",
                "Self-test pipeline M1-M6",
                "Hoàn thành trong Bài 12",
            ],
            [
                "Báo cáo",
                "Word/PDF 15-25 trang",
                "Cần nộp kèm",
            ],
            [
                "Slide",
                "Khoảng 15 slide",
                "Cần nộp kèm",
            ],
            [
                "Video",
                "Demo 3-5 phút",
                "Cần nộp kèm",
            ],
        ],
        columns=[
            "Sản phẩm",
            "Nội dung",
            "Trạng thái",
        ],
    )

    st.dataframe(
        deliverables,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## 12.5. Tiêu chí đánh giá")

    rubric = pd.DataFrame(
        [
            [
                "Mô hình toán",
                20,
            ],
            [
                "Chất lượng code",
                20,
            ],
            [
                "Dữ liệu Việt Nam",
                15,
            ],
            [
                "Phân tích chính sách",
                20,
            ],
            [
                "Dashboard",
                15,
            ],
            [
                "Báo cáo và thuyết trình",
                10,
            ],
        ],
        columns=[
            "Tiêu chí",
            "Trọng số (%)",
        ],
    )

    st.dataframe(
        rubric,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## 12.6. Hướng mở rộng")
    st.markdown(
        """
        - Tách M1-M5 thành package Python độc lập trong thư mục `src/`.
        - Thêm unit test bằng pytest cho LP, TOPSIS, SP và RL.
        - Kết nối API dữ liệu chính thức thay vì chỉ dùng CSV tĩnh.
        - Huấn luyện DQN offline, lưu model và kiểm định nhiều seed.
        - Bổ sung use case chuyên sâu cho Đồng bằng sông Cửu Long hoặc bán dẫn.
        """
    )

    st.download_button(
        "Tải bảng tổng hợp năm kịch bản",
        data=scenarios.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai12_pipeline_5_kich_ban.csv",
        mime="text/csv",
        key="download_bai12_pipeline",
    )


PAGES = {
    "🏠 Trang chủ": page_home,
    "🌱 Bài 1 — Cobb-Douglas + AI": page_1,
    "💰 Bài 2 — LP ngân sách số": page_2,
    "📊 Bài 3 — Priority 10 ngành": page_3,
    "🗺️ Bài 4 — LP ngành-vùng": page_4,
    "🎯 Bài 5 — MIP 15 dự án": page_5,
    "🏆 Bài 6 — TOPSIS 6 vùng": page_6,
    "🌐 Bài 7 — Pareto đa mục tiêu": page_7,
    "⏳ Bài 8 — Động 2026-2035": page_8,
    "👷 Bài 9 — Lao động & AI": page_9,
    "🎲 Bài 10 — Stochastic SP": page_10,
    "🤖 Bài 11 — Q-learning RL": page_11,
    "🇻🇳 Bài 12 — AIDEOM tích hợp": page_12,
}


st.sidebar.markdown("## 🇻🇳 VN AIDEOM-VN")
st.sidebar.caption("Decision Optimization Dashboard")
choice = st.sidebar.radio("Điều hướng 13 trang", list(PAGES.keys()), label_visibility="visible")
st.sidebar.markdown("---")
st.sidebar.caption("Dữ liệu: macro, sectors, regions 2020-2025. Mỗi trang có bảng kết quả, biểu đồ và diễn giải mô hình.")
PAGES[choice]()

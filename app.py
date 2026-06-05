
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
    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Nội dung trang chủ
    # -----------------------------------------------------
    st.markdown(
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
        """,
        unsafe_allow_html=True,
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

def _b4_solve_scipy(fairness=True, total_budget=50000.0):
    """
    Giải bài toán LP phân bổ ngân sách số cho 6 vùng × 4 hạng mục.

    Biến:
    - 24 biến x[r,j]
    - 1 biến M dùng cho ràng buộc công bằng vùng

    fairness=True:
        bật ràng buộc mọi vùng phải đạt ít nhất 70% mức Digital Index cao nhất.
    """
    regions, items, beta, D0 = region_beta_matrix()

    n_x = 24
    m_index = 24
    n_var = 25

    # scipy.optimize.linprog là bài toán tối thiểu hóa,
    # nên đổi max Z thành min(-Z)
    c = np.zeros(n_var, dtype=float)
    c[:n_x] = -beta.reshape(-1)

    A_ub = []
    b_ub = []

    # C1. Tổng ngân sách toàn quốc không vượt 50.000
    row = np.zeros(n_var, dtype=float)
    row[:n_x] = 1.0
    A_ub.append(row)
    b_ub.append(float(total_budget))

    # C2-C3. Mỗi vùng nhận từ 5.000 đến 12.000
    for r in range(6):
        # Tổng vùng >= 5.000
        row = np.zeros(n_var, dtype=float)
        row[r * 4 : r * 4 + 4] = -1.0
        A_ub.append(row)
        b_ub.append(-5000.0)

        # Tổng vùng <= 12.000
        row = np.zeros(n_var, dtype=float)
        row[r * 4 : r * 4 + 4] = 1.0
        A_ub.append(row)
        b_ub.append(12000.0)

    # C4. Tổng đầu tư nhân lực số toàn quốc >= 12.000
    row = np.zeros(n_var, dtype=float)
    for r in range(6):
        row[r * 4 + 3] = -1.0
    A_ub.append(row)
    b_ub.append(-12000.0)

    # C5. Công bằng vùng:
    # D_r + gamma*x_D,r >= lambda*M
    # D_r + gamma*x_D,r <= M
    if fairness:
        gamma = 0.002
        lam = 0.7

        # M phải lớn hơn hoặc bằng Digital Index sau đầu tư của từng vùng
        for r in range(6):
            row = np.zeros(n_var, dtype=float)
            row[r * 4 + 1] = gamma
            row[m_index] = -1.0
            A_ub.append(row)
            b_ub.append(-float(D0[r]))

        # Mỗi vùng phải đạt ít nhất 70% mức cao nhất M
        for r in range(6):
            row = np.zeros(n_var, dtype=float)
            row[r * 4 + 1] = -gamma
            row[m_index] = lam
            A_ub.append(row)
            b_ub.append(float(D0[r]))

    bounds = [(0, None)] * n_var

    result = linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )

    return result, regions, items, beta, D0


def _b4_allocation_table(result, regions, items):
    """
    Chuyển nghiệm 24 biến thành ma trận 6 vùng × 4 hạng mục.
    """
    X = result.x[:24].reshape(6, 4)

    allocation = pd.DataFrame(
        X,
        columns=items,
        index=regions,
    )

    allocation["Tổng vùng"] = allocation.sum(axis=1)

    return X, allocation


def page_4():
    hero(
        "Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng",
        "Trình bày đầy đủ các mục 4.1-4.5: mô hình 24 biến, sàn-trần vùng, nhân lực số, công bằng vùng, so sánh PuLP-CVXPY và chi phí của công bằng.",
        ["4.1-4.5", "Regional LP", "PuLP", "CVXPY", "Fairness"],
    )

    regions, items, beta, D0 = region_beta_matrix()

    # =====================================================
    # 4.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown("## 4.1. Bối cảnh Việt Nam")

    st.markdown(
        """
        Việt Nam có sáu vùng kinh tế - xã hội với mức độ phát triển và sẵn sàng số
        khác nhau. Giả sử Chính phủ có **50.000 tỷ VND** để phân bổ cho bốn hạng mục:

        - **I:** hạ tầng số;
        - **D:** chuyển đổi số doanh nghiệp;
        - **AI:** trí tuệ nhân tạo;
        - **H:** nhân lực số.

        Bài toán cần tối đa hóa mức tăng GDP kỳ vọng nhưng vẫn phải:
        bảo đảm ngân sách tối thiểu cho từng vùng, tránh tập trung quá mức,
        dành đủ nguồn lực cho nhân lực số và duy trì công bằng vùng miền.
        """
    )

    # =====================================================
    # 4.2. Mô hình toán học
    # =====================================================
    st.markdown("## 4.2. Mô hình toán học đầy đủ")

    st.markdown("### Biến quyết định")
    st.latex(
        r"x_{j,r}="
        r"\text{ngân sách phân bổ cho hạng mục }j"
        r"\text{ tại vùng }r"
    )

    st.markdown("### Hàm mục tiêu")
    st.latex(
        r"\max Z="
        r"\sum_{r=1}^{6}"
        r"\sum_{j\in\{I,D,AI,H\}}"
        r"\beta_{j,r}x_{j,r}"
    )

    st.markdown("### Các ràng buộc")

    st.latex(
        r"\sum_{r=1}^{6}\sum_jx_{j,r}"
        r"\leq 50{,}000"
    )

    st.latex(
        r"5{,}000"
        r"\leq\sum_jx_{j,r}"
        r"\leq12{,}000,\quad \forall r"
    )

    st.latex(
        r"\sum_{r=1}^{6}x_{H,r}"
        r"\geq12{,}000"
    )

    st.latex(
        r"D_r+\gamma x_{D,r}"
        r"\geq\lambda\max_r"
        r"(D_r+\gamma x_{D,r})"
    )

    st.latex(
        r"\gamma=0.002,\quad"
        r"\lambda=0.7"
    )

    st.latex(
        r"x_{j,r}\geq0"
    )

    st.info(
        "Ràng buộc công bằng yêu cầu Digital Index sau đầu tư của mỗi vùng "
        "đạt ít nhất 70% mức cao nhất trong sáu vùng."
    )

    # =====================================================
    # 4.3. Bảng hệ số tác động biên
    # =====================================================
    st.markdown("## 4.3. Bảng hệ số tác động biên βⱼ,ᵣ")

    beta_table = pd.DataFrame(
        beta,
        columns=items,
    )

    beta_table.insert(
        0,
        "Vùng",
        regions,
    )

    beta_table["Digital Index ban đầu"] = D0

    st.dataframe(
        beta_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Hệ số β càng lớn thì một đơn vị ngân sách đầu tư vào hạng mục đó "
        "tại vùng tương ứng tạo mức tăng GDP kỳ vọng càng cao."
    )

    # Giải mô hình chuẩn và mô hình không có công bằng
    fair_result, _, _, _, _ = _b4_solve_scipy(
        fairness=True,
        total_budget=50000.0,
    )

    nofair_result, _, _, _, _ = _b4_solve_scipy(
        fairness=False,
        total_budget=50000.0,
    )

    if not fair_result.success:
        st.error(
            "Mô hình có ràng buộc công bằng vùng không khả thi. "
            "Hãy kiểm tra lại các hệ số và giới hạn ngân sách."
        )
        return

    X_fair, allocation_fair = _b4_allocation_table(
        fair_result,
        regions,
        items,
    )

    z_fair = -float(fair_result.fun)

    if nofair_result.success:
        X_nofair, allocation_nofair = _b4_allocation_table(
            nofair_result,
            regions,
            items,
        )

        z_nofair = -float(nofair_result.fun)
        fairness_cost = z_nofair - z_fair
    else:
        X_nofair = None
        allocation_nofair = None
        z_nofair = np.nan
        fairness_cost = np.nan

    # =====================================================
    # 4.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 4.4. Yêu cầu lập trình")

    tab441, tab442, tab443, tab444 = st.tabs(
        [
            "4.4.1 - PuLP",
            "4.4.2 - CVXPY",
            "4.4.3 - Heatmap",
            "4.4.4 - Bỏ công bằng",
        ]
    )

    # -----------------------------------------------------
    # 4.4.1
    # -----------------------------------------------------
    with tab441:
        st.markdown(
            "### Câu 4.4.1. Cài đặt mô hình bằng PuLP"
        )

        try:
            import pulp

            model = pulp.LpProblem(
                "VN_Digital_Regional_Budget",
                pulp.LpMaximize,
            )

            x = pulp.LpVariable.dicts(
                "x",
                (range(6), range(4)),
                lowBound=0,
            )

            M = pulp.LpVariable(
                "Digital_Index_Max",
                lowBound=0,
            )

            model += pulp.lpSum(
                beta[r, j] * x[r][j]
                for r in range(6)
                for j in range(4)
            ), "Expected_GDP_Gain"

            model += (
                pulp.lpSum(
                    x[r][j]
                    for r in range(6)
                    for j in range(4)
                )
                <= 50000
            ), "C1_Total_Budget"

            for r in range(6):
                model += (
                    pulp.lpSum(
                        x[r][j]
                        for j in range(4)
                    )
                    >= 5000
                ), f"C2_Min_Region_{r+1}"

                model += (
                    pulp.lpSum(
                        x[r][j]
                        for j in range(4)
                    )
                    <= 12000
                ), f"C3_Max_Region_{r+1}"

            model += (
                pulp.lpSum(
                    x[r][3]
                    for r in range(6)
                )
                >= 12000
            ), "C4_Digital_Human"

            gamma = 0.002
            lam = 0.7

            for r in range(6):
                model += (
                    D0[r] + gamma * x[r][1]
                    <= M
                ), f"C5a_Max_Index_{r+1}"

                model += (
                    D0[r] + gamma * x[r][1]
                    >= lam * M
                ), f"C5b_Fairness_{r+1}"

            solver = pulp.PULP_CBC_CMD(
                msg=False
            )

            model.solve(
                solver
            )

            pulp_status = pulp.LpStatus[
                model.status
            ]

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
                pulp.value(
                    model.objective
                )
            )

            pulp_table = pd.DataFrame(
                X_pulp,
                columns=items,
            )

            pulp_table.insert(
                0,
                "Vùng",
                regions,
            )

            pulp_table["Tổng vùng"] = (
                X_pulp.sum(axis=1)
            )

            kpi_cards(
                [
                    (
                        "Trạng thái",
                        pulp_status,
                        "CBC solver",
                    ),
                    (
                        "Z* PuLP",
                        f"{z_pulp:,.2f}",
                        "GDP gain kỳ vọng",
                    ),
                    (
                        "Tổng ngân sách",
                        f"{X_pulp.sum():,.0f}",
                        "tỷ VND",
                    ),
                    (
                        "Nhân lực số",
                        f"{X_pulp[:, 3].sum():,.0f}",
                        "tỷ VND",
                    ),
                ]
            )

            st.dataframe(
                pulp_table,
                use_container_width=True,
                hide_index=True,
            )

            comparison_solver = pd.DataFrame(
                {
                    "Chỉ tiêu": [
                        "Giá trị mục tiêu",
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

            st.markdown(
                "#### So sánh nghiệm PuLP và SciPy"
            )

            st.dataframe(
                comparison_solver,
                use_container_width=True,
                hide_index=True,
            )

        except ModuleNotFoundError:
            st.warning(
                "Môi trường chưa cài PuLP. Trang vẫn hiển thị nghiệm SciPy. "
                "Hãy thêm `pulp>=2.7` vào requirements.txt."
            )

            st.dataframe(
                allocation_fair
                .reset_index()
                .rename(
                    columns={"index": "Vùng"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "Xem mã PuLP rút gọn"
        ):
            st.code(
                """model = pulp.LpProblem(
    "RegionalBudget",
    pulp.LpMaximize
)

x = pulp.LpVariable.dicts(
    "x",
    (range(6), range(4)),
    lowBound=0
)

model += pulp.lpSum(
    beta[r,j] * x[r][j]
    for r in range(6)
    for j in range(4)
)

model.solve(
    pulp.PULP_CBC_CMD(msg=False)
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 4.4.2
    # -----------------------------------------------------
    with tab442:
        st.markdown(
            "### Câu 4.4.2. Cài đặt lại bằng CVXPY và so sánh"
        )

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
                cp.sum(X) <= 50000,
                cp.sum(X[:, 3]) >= 12000,
            ]

            gamma = 0.002
            lam = 0.7

            for r in range(6):
                constraints.extend(
                    [
                        cp.sum(X[r, :]) >= 5000,
                        cp.sum(X[r, :]) <= 12000,
                        D0[r] + gamma * X[r, 1] <= M,
                        D0[r] + gamma * X[r, 1] >= lam * M,
                    ]
                )

            objective = cp.Maximize(
                cp.sum(
                    cp.multiply(
                        beta,
                        X,
                    )
                )
            )

            problem = cp.Problem(
                objective,
                constraints,
            )

            problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
            )

            if X.value is None:
                problem.solve(
                    solver=cp.SCS,
                    verbose=False,
                )

            if X.value is None:
                st.error(
                    "CVXPY không trả về nghiệm."
                )
            else:
                X_cvxpy = np.asarray(
                    X.value,
                    dtype=float,
                )

                z_cvxpy = float(
                    problem.value
                )

                cvxpy_table = pd.DataFrame(
                    X_cvxpy,
                    columns=items,
                )

                cvxpy_table.insert(
                    0,
                    "Vùng",
                    regions,
                )

                cvxpy_table["Tổng vùng"] = (
                    X_cvxpy.sum(axis=1)
                )

                solver_compare = pd.DataFrame(
                    {
                        "Chỉ tiêu": [
                            "Z*",
                            "Tổng phân bổ",
                            "Sai lệch phân bổ lớn nhất",
                        ],
                        "SciPy": [
                            z_fair,
                            X_fair.sum(),
                            0.0,
                        ],
                        "CVXPY": [
                            z_cvxpy,
                            X_cvxpy.sum(),
                            np.max(
                                np.abs(
                                    X_cvxpy - X_fair
                                )
                            ),
                        ],
                    }
                )

                kpi_cards(
                    [
                        (
                            "Z* CVXPY",
                            f"{z_cvxpy:,.2f}",
                            "GDP gain",
                        ),
                        (
                            "Z* SciPy",
                            f"{z_fair:,.2f}",
                            "GDP gain",
                        ),
                        (
                            "Chênh lệch Z*",
                            f"{abs(z_cvxpy-z_fair):,.6f}",
                            "giữa hai solver",
                        ),
                        (
                            "Tổng phân bổ",
                            f"{X_cvxpy.sum():,.0f}",
                            "tỷ VND",
                        ),
                    ]
                )

                st.dataframe(
                    solver_compare,
                    use_container_width=True,
                    hide_index=True,
                )

                st.dataframe(
                    cvxpy_table,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    "PuLP, CVXPY và SciPy có thể tạo cơ cấu phân bổ hơi khác "
                    "khi tồn tại nhiều nghiệm tối ưu, nhưng giá trị mục tiêu phải gần nhau."
                )

        except ModuleNotFoundError:
            st.warning(
                "Chưa cài CVXPY. Thêm `cvxpy>=1.4` vào requirements.txt."
            )

        except Exception as exc:
            st.warning(
                f"CVXPY hoặc solver gặp lỗi: {exc}"
            )

    # -----------------------------------------------------
    # 4.4.3
    # -----------------------------------------------------
    with tab443:
        st.markdown(
            "### Câu 4.4.3. Heatmap phân bổ tối ưu 6 vùng × 4 hạng mục"
        )

        heatmap_data = pd.DataFrame(
            X_fair,
            columns=items,
            index=regions,
        )

        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdPu",
            template=PLOT_TEMPLATE,
            title="Heatmap phân bổ ngân sách tối ưu",
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

        region_total = allocation_fair[
            "Tổng vùng"
        ].sort_values(
            ascending=False
        )

        item_total = allocation_fair[
            items
        ].sum().sort_values(
            ascending=False
        )

        kpi_cards(
            [
                (
                    "Vùng nhận nhiều nhất",
                    region_total.index[0],
                    f"{region_total.iloc[0]:,.0f} tỷ VND",
                ),
                (
                    "Vùng nhận ít nhất",
                    region_total.index[-1],
                    f"{region_total.iloc[-1]:,.0f} tỷ VND",
                ),
                (
                    "Hạng mục lớn nhất",
                    item_total.index[0],
                    f"{item_total.iloc[0]:,.0f} tỷ VND",
                ),
                (
                    "Tổng nhân lực số",
                    f"{allocation_fair[items[3]].sum():,.0f}",
                    "tỷ VND",
                ),
            ]
        )

        st.dataframe(
            allocation_fair
            .reset_index()
            .rename(
                columns={"index": "Vùng"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 4.4.4
    # -----------------------------------------------------
    with tab444:
        st.markdown(
            "### Câu 4.4.4. So sánh mô hình có và không có công bằng vùng"
        )

        if (
            nofair_result.success
            and allocation_nofair is not None
        ):
            region_compare = pd.DataFrame(
                {
                    "Vùng": regions,
                    "Có công bằng": X_fair.sum(
                        axis=1
                    ),
                    "Không công bằng": X_nofair.sum(
                        axis=1
                    ),
                }
            )

            region_compare["Thay đổi"] = (
                region_compare["Có công bằng"]
                - region_compare["Không công bằng"]
            )

            fairness_rate = (
                100 * fairness_cost / z_nofair
                if z_nofair != 0
                else 0
            )

            kpi_cards(
                [
                    (
                        "Z* có công bằng",
                        f"{z_fair:,.2f}",
                        "GDP gain",
                    ),
                    (
                        "Z* không công bằng",
                        f"{z_nofair:,.2f}",
                        "GDP gain",
                    ),
                    (
                        "Chi phí công bằng",
                        f"{fairness_cost:,.2f}",
                        "GDP gain bị giảm",
                    ),
                    (
                        "Tỷ lệ chi phí",
                        f"{fairness_rate:.3f}%",
                        "so với mô hình không C5",
                    ),
                ]
            )

            st.dataframe(
                region_compare,
                use_container_width=True,
                hide_index=True,
            )

            compare_long = region_compare.melt(
                id_vars="Vùng",
                value_vars=[
                    "Có công bằng",
                    "Không công bằng",
                ],
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
                title="Tổng ngân sách từng vùng trước và sau ràng buộc công bằng",
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

            st.info(
                "Chi phí công bằng là phần giá trị mục tiêu phải đánh đổi "
                "để ngăn ngân sách tập trung quá mức vào các vùng có lợi ích biên cao."
            )

        else:
            st.warning(
                "Không giải được mô hình bỏ ràng buộc công bằng."
            )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_table = (
        allocation_fair
        .reset_index()
        .rename(
            columns={"index": "Vùng"}
        )
    )

    st.download_button(
        "Tải kết quả Bài 4 dạng CSV",
        data=export_table.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai4_lp_nganh_vung.csv",
        mime="text/csv",
        key="download_bai4",
    )

    # =====================================================
    # 4.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 4.5. Câu hỏi thảo luận chính sách"
    )

    with st.expander(
        "a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào?",
        expanded=True,
    ):
        st.markdown(
            "Vốn có xu hướng chảy về các vùng có hệ số tác động biên cao, "
            "đặc biệt nơi có mức sẵn sàng số và AI tốt. Điều này làm tăng hiệu quả "
            "ngắn hạn nhưng có thể làm sâu sắc thêm khoảng cách vùng."
        )

    with st.expander(
        "b) Trần ngân sách vùng có phải là chính sách phân quyền không?",
        expanded=True,
    ):
        st.markdown(
            "Trần 12.000 tỷ VND ngăn một hoặc hai vùng hấp thụ phần lớn ngân sách. "
            "Có thể xem đây là cơ chế chống tập trung và hỗ trợ phân quyền. "
            "Chi phí hiệu quả được đo bằng phần Z* giảm so với mô hình không có giới hạn."
        )

    with st.expander(
        "c) Tây Nguyên nên đầu tư AI ngay hay ưu tiên H và I trước?",
        expanded=True,
    ):
        st.markdown(
            "Do hệ số AI của Tây Nguyên tương đối thấp trong khi hệ số nhân lực và "
            "hạ tầng cao hơn, mô hình thường ưu tiên H và I trước. Đây là trình tự "
            "xây dựng năng lực hấp thụ: tạo nền tảng rồi mới mở rộng AI."
        )


def project_table():
    rows = [
        ("P1", "Trung tâm dữ liệu quốc gia Hòa Lạc", "Hạ tầng", 12000, 21500, 8500, 3500),
        ("P2", "Trung tâm dữ liệu quốc gia phía Nam", "Hạ tầng", 11500, 20800, 7500, 4000),
        ("P3", "Hệ thống 5G phủ sóng toàn quốc", "Hạ tầng", 18000, 32500, 12000, 6000),
        ("P4", "Hệ thống định danh điện tử VNeID 2.0", "Chính phủ số", 4500, 9200, 3500, 1000),
        ("P5", "Cổng dịch vụ công quốc gia v3", "Chính phủ số", 3200, 6800, 2500, 700),
        ("P6", "Y tế số quốc gia", "Y tế số", 5800, 11400, 4000, 1800),
        ("P7", "Giáo dục số K-12 toàn quốc", "Giáo dục", 6500, 12200, 4500, 2000),
        ("P8", "Trung tâm AI quốc gia + supercomputing", "AI", 15000, 28500, 9000, 6000),
        ("P9", "Sandbox tài chính số", "Tài chính số", 2500, 5800, 1800, 700),
        ("P10", "Logistics thông minh + cảng biển số", "Logistics", 7200, 13800, 5000, 2200),
        ("P11", "Nông nghiệp số ĐBSCL", "Nông nghiệp", 4800, 8500, 3500, 1300),
        ("P12", "Đào tạo 50.000 kỹ sư AI/bán dẫn", "Nhân lực", 8500, 16200, 5500, 3000),
        ("P13", "Khu CN bán dẫn Bắc Ninh - Bắc Giang", "Bán dẫn", 20000, 35000, 13000, 7000),
        ("P14", "An ninh mạng quốc gia SOC", "An ninh", 3800, 7500, 2800, 1000),
        ("P15", "Open Data + dữ liệu mở quốc gia", "Dữ liệu", 1500, 3800, 1200, 300),
    ]
    return pd.DataFrame(rows, columns=["Mã", "Tên dự án", "Lĩnh vực", "Chi phí", "NPV", "Năm 1-2", "Năm 3-5"])


def solve_mip_bruteforce(budget=80000, require_p1p2=False):
    df = project_table()
    best = None
    n = len(df)
    for mask in range(1 << n):
        y = np.array([(mask >> i) & 1 for i in range(n)])
        if y.sum() < 7 or y.sum() > 11:
            continue
        cost = (df["Chi phí"].values * y).sum()
        cost12 = (df["Năm 1-2"].values * y).sum()
        benefit = (df["NPV"].values * y).sum()
        if cost > budget or cost12 > 40000:
            continue
        if y[0] + y[1] > 1:
            continue
        if require_p1p2 and (y[0] + y[1] < 2):
            continue
        if y[7] > y[11]:
            continue
        if y[12] > y[11]:
            continue
        if y[3] + y[4] < 1:
            continue
        if y[13] < 1:
            continue
        if best is None or benefit > best[0]:
            best = (benefit, cost, cost12, y)
    return best


def page_5():
    hero(
        "Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số",
        "Bài toán knapsack tổng quát hóa với biến nhị phân, ràng buộc loại trừ, tiên quyết, ngân sách đa năm và số lượng dự án.",
        ["MIP", "Binary selection", "Knapsack"],
    )
    budget = st.slider("Ngân sách tổng 5 năm", 70000, 105000, 80000, 5000)
    require_p1p2 = st.checkbox("Thử kịch bản bắt buộc cả P1 và P2 nhưng vẫn giữ ràng buộc loại trừ")
    best = solve_mip_bruteforce(budget, require_p1p2=require_p1p2)
    df = project_table()

    if best is None:
        st.error("Không khả thi: yêu cầu chọn cả P1 và P2 xung đột trực tiếp với ràng buộc y₁ + y₂ ≤ 1.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    benefit, cost, cost12, y = best
    selected = df[y == 1].copy()
    selected["NPV/Chi phí"] = selected["NPV"] / selected["Chi phí"]

    kpi_cards(
        [
            ("Tổng NPV", f"{benefit:,.0f}", "tỷ VND"),
            ("Tổng chi phí", f"{cost:,.0f}", f"ngân sách {budget:,.0f}"),
            ("Số dự án", f"{int(y.sum())}", "ràng buộc 7-11"),
            ("NPV/Chi phí", f"{benefit / cost:.2f}", "hiệu quả danh mục"),
        ]
    )
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.dataframe(selected, use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(selected, x="Mã", y="NPV", color="Lĩnh vực", template=PLOT_TEMPLATE, title="Lợi ích NPV của dự án được chọn", hover_name="Tên dự án")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig, use_container_width=True)


def _b6_prepare_data():
    """
    Chuẩn bị dữ liệu, danh sách tiêu chí, loại tiêu chí
    và bộ trọng số chuyên gia cho Bài 6.
    """
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

    # 7 tiêu chí lợi ích và 1 tiêu chí chi phí
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
            0.10,  # GRDP/người
            0.10,  # FDI
            0.15,  # Digital Index
            0.20,  # AI readiness
            0.15,  # Lao động qua đào tạo
            0.15,  # R&D intensity
            0.05,  # Internet
            0.10,  # Gini
        ],
        dtype=float,
    )

    return df, criteria, is_benefit, expert_weights


def _b6_entropy_weights(df, criteria):
    """
    Tính trọng số Entropy sau khi chuyển tiêu chí Gini
    về hướng càng lớn càng tốt.
    """
    X = df[criteria].astype(float).copy()

    for col in criteria[:-1]:
        X[col] = minmax(X[col])

    # Gini là tiêu chí chi phí
    X[criteria[-1]] = reverse_minmax(X[criteria[-1]])

    return entropy_weights_positive(
        X.to_numpy(dtype=float)
    )


def _b6_rank_result(df, score, score_name):
    """
    Tạo bảng xếp hạng TOPSIS.
    """
    result = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            score_name: score,
        }
    ).sort_values(
        score_name,
        ascending=False,
    )

    result["Xếp hạng"] = np.arange(
        1,
        len(result) + 1,
    )

    return result


def page_6():
    hero(
        "Bài 6 — TOPSIS xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI",
        "Trình bày đầy đủ các mục 6.1-6.5: lý thuyết TOPSIS, trọng số chuyên gia, Entropy, độ nhạy AI readiness và AHP.",
        ["6.1-6.5", "TOPSIS", "Entropy", "AHP", "Regional AI"],
    )

    df, criteria, is_benefit, expert_weights = _b6_prepare_data()

    criterion_labels = {
        "grdp_per_capita_million_VND": "GRDP/người",
        "fdi_registered_billion_USD": "FDI đăng ký",
        "digital_index_0_100": "Digital Index",
        "ai_readiness_0_100": "AI Readiness",
        "trained_labor_pct": "Lao động qua đào tạo",
        "rd_intensity_pct": "R&D/GDP",
        "internet_penetration_pct": "Internet",
        "gini_coef": "Gini",
    }

    # =====================================================
    # 6.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown("## 6.1. Bối cảnh Việt Nam")

    st.markdown(
        """
        Việt Nam đặt mục tiêu phát triển các trung tâm nghiên cứu, đào tạo và ứng dụng AI,
        nhưng nguồn lực công có giới hạn. Sáu vùng kinh tế - xã hội có mức độ phát triển,
        hạ tầng số, nhân lực và năng lực đổi mới sáng tạo khác nhau.

        Bài 6 sử dụng **TOPSIS** để xếp hạng mức độ ưu tiên đầu tư AI cho từng vùng.
        Phương án tốt là vùng có khoảng cách gần nhất với nghiệm lý tưởng tích cực
        và xa nhất với nghiệm lý tưởng tiêu cực.
        """
    )

    # =====================================================
    # 6.2. Lý thuyết TOPSIS
    # =====================================================
    st.markdown("## 6.2. Lý thuyết TOPSIS")

    st.markdown("### Bước 1. Chuẩn hóa vector")
    st.latex(
        r"r_{ij}="
        r"\frac{x_{ij}}"
        r"{\sqrt{\sum_{i=1}^{m}x_{ij}^{2}}}"
    )

    st.markdown("### Bước 2. Ma trận chuẩn hóa có trọng số")
    st.latex(
        r"v_{ij}=w_jr_{ij}"
    )

    st.markdown("### Bước 3. Nghiệm lý tưởng")
    st.latex(
        r"A^*=\{v_1^*,v_2^*,...,v_n^*\},"
        r"\quad"
        r"A^-=\{v_1^-,v_2^-,...,v_n^-\}"
    )

    st.markdown("### Bước 4. Khoảng cách đến nghiệm lý tưởng")
    st.latex(
        r"S_i^*="
        r"\sqrt{\sum_j(v_{ij}-v_j^*)^2}"
    )

    st.latex(
        r"S_i^-="
        r"\sqrt{\sum_j(v_{ij}-v_j^-)^2}"
    )

    st.markdown("### Bước 5. Hệ số gần lý tưởng")
    st.latex(
        r"C_i^*="
        r"\frac{S_i^-}"
        r"{S_i^*+S_i^-}"
    )

    st.info(
        "Vùng có C* càng lớn thì càng gần phương án lý tưởng và được xếp hạng ưu tiên cao hơn."
    )

    # =====================================================
    # 6.3. Dữ liệu 6 vùng
    # =====================================================
    st.markdown("## 6.3. Dữ liệu 6 vùng kinh tế - xã hội")

    data_display = df[
        ["region_name_vi"] + criteria
    ].rename(
        columns={
            "region_name_vi": "Vùng",
            **criterion_labels,
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
                criterion_labels[col]
                for col in criteria
            ],
            "Loại tiêu chí": [
                "Lợi ích" if flag else "Chi phí"
                for flag in is_benefit
            ],
            "Trọng số chuyên gia": expert_weights,
        }
    )

    st.markdown("### Bộ trọng số chuyên gia")

    st.dataframe(
        criteria_table.style.format(
            {"Trọng số chuyên gia": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Kết quả dùng chung
    expert_score = topsis_score(
        df,
        criteria,
        expert_weights,
        is_benefit,
    )

    entropy_weights = _b6_entropy_weights(
        df,
        criteria,
    )

    entropy_score = topsis_score(
        df,
        criteria,
        entropy_weights,
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

    # =====================================================
    # 6.4. Yêu cầu lập trình
    # =====================================================
    st.markdown("## 6.4. Yêu cầu lập trình")

    tab641, tab642, tab643, tab644 = st.tabs(
        [
            "6.4.1 - TOPSIS expert",
            "6.4.2 - Entropy",
            "6.4.3 - Độ nhạy AI",
            "6.4.4 - AHP",
        ]
    )

    # -----------------------------------------------------
    # 6.4.1
    # -----------------------------------------------------
    with tab641:
        st.markdown(
            "### Câu 6.4.1. Xếp hạng TOPSIS với trọng số chuyên gia"
        )

        kpi_cards(
            [
                (
                    "Vùng dẫn đầu",
                    expert_result.iloc[0]["Vùng"],
                    f"C* = {expert_result.iloc[0]['TOPSIS chuyên gia']:.3f}",
                ),
                (
                    "Vùng thứ hai",
                    expert_result.iloc[1]["Vùng"],
                    f"C* = {expert_result.iloc[1]['TOPSIS chuyên gia']:.3f}",
                ),
                (
                    "Vùng thứ ba",
                    expert_result.iloc[2]["Vùng"],
                    f"C* = {expert_result.iloc[2]['TOPSIS chuyên gia']:.3f}",
                ),
                (
                    "Số tiêu chí",
                    "8",
                    "7 lợi ích, 1 chi phí",
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

        with st.expander("Xem mã Python cho câu 6.4.1"):
            st.code(
                """score = topsis_score(
    df,
    criteria,
    expert_weights,
    is_benefit
)

result = pd.DataFrame({
    "region": df["region_name_vi"],
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
            "### Câu 6.4.2. Tính trọng số Entropy và chạy lại TOPSIS"
        )

        weight_compare = pd.DataFrame(
            {
                "Tiêu chí": [
                    criterion_labels[col]
                    for col in criteria
                ],
                "Trọng số chuyên gia": expert_weights,
                "Trọng số Entropy": entropy_weights,
            }
        )

        rank_compare = pd.DataFrame(
            {
                "Vùng": df["region_name_vi"],
                "TOPSIS chuyên gia": expert_score,
                "TOPSIS Entropy": entropy_score,
            }
        )

        rank_compare["Hạng chuyên gia"] = (
            rank_compare["TOPSIS chuyên gia"]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        rank_compare["Hạng Entropy"] = (
            rank_compare["TOPSIS Entropy"]
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

        c1, c2 = st.columns([0.9, 1.1])

        with c1:
            st.markdown("#### So sánh trọng số")

            st.dataframe(
                weight_compare.style.format(
                    {
                        "Trọng số chuyên gia": "{:.4f}",
                        "Trọng số Entropy": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.markdown("#### So sánh thứ hạng")

            st.dataframe(
                rank_compare.sort_values(
                    "Hạng chuyên gia"
                ).style.format(
                    {
                        "TOPSIS chuyên gia": "{:.4f}",
                        "TOPSIS Entropy": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        compare_long = rank_compare.melt(
            id_vars="Vùng",
            value_vars=[
                "TOPSIS chuyên gia",
                "TOPSIS Entropy",
            ],
            var_name="Phương pháp",
            value_name="Điểm",
        )

        fig_compare = px.bar(
            compare_long,
            x="Vùng",
            y="Điểm",
            color="Phương pháp",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="So sánh TOPSIS chuyên gia và Entropy",
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

        changed_regions = rank_compare.loc[
            rank_compare["Thay đổi hạng"] != 0,
            "Vùng",
        ].tolist()

        st.info(
            "Các vùng thay đổi thứ hạng khi dùng Entropy: "
            + (
                ", ".join(changed_regions)
                if changed_regions
                else "không có"
            )
            + "."
        )

        with st.expander("Xem mã Python cho câu 6.4.2"):
            st.code(
                """entropy_w = entropy_weights_positive(
    normalized_matrix
)

entropy_score = topsis_score(
    df,
    criteria,
    entropy_w,
    is_benefit
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 6.4.3
    # -----------------------------------------------------
    with tab643:
        st.markdown(
            "### Câu 6.4.3. Độ nhạy khi trọng số AI readiness thay đổi"
        )

        sensitivity_rows = []

        ai_weight_values = np.arange(
            0.10,
            0.401,
            0.05,
        )

        for ai_weight in ai_weight_values:
            new_weights = expert_weights.copy()

            remaining_weight = 1 - ai_weight

            other_mask = np.ones(
                len(new_weights),
                dtype=bool,
            )

            # AI readiness là tiêu chí thứ 4, index = 3
            other_mask[3] = False

            new_weights[other_mask] = (
                new_weights[other_mask]
                / new_weights[other_mask].sum()
                * remaining_weight
            )

            new_weights[3] = ai_weight

            score = topsis_score(
                df,
                criteria,
                new_weights,
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
                sensitivity_rows.append(
                    [
                        ai_weight,
                        row["Vùng"],
                        row["Điểm"],
                        row["Hạng"],
                    ]
                )

        sensitivity_df = pd.DataFrame(
            sensitivity_rows,
            columns=[
                "Trọng số AI",
                "Vùng",
                "Điểm",
                "Hạng",
            ],
        )

        fig_line = px.line(
            sensitivity_df,
            x="Trọng số AI",
            y="Hạng",
            color="Vùng",
            markers=True,
            template=PLOT_TEMPLATE,
            title="Độ nhạy thứ hạng theo trọng số AI readiness",
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

        rank_heatmap = sensitivity_df.pivot(
            index="Vùng",
            columns="Trọng số AI",
            values="Hạng",
        )

        fig_heatmap = px.imshow(
            rank_heatmap,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            template=PLOT_TEMPLATE,
            title="Heatmap thứ hạng theo trọng số AI",
        )

        fig_heatmap.update_layout(
            height=500,
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

        first_place_count = (
            sensitivity_df[
                sensitivity_df["Hạng"] == 1
            ]["Vùng"]
            .value_counts()
            .reset_index()
        )

        first_place_count.columns = [
            "Vùng",
            "Số lần đứng đầu",
        ]

        st.markdown(
            "#### Mức ổn định của vị trí dẫn đầu"
        )

        st.dataframe(
            first_place_count,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 6.4.4
    # -----------------------------------------------------
    with tab644:
        st.markdown(
            "### Câu 6.4.4. Dùng AHP để xác định trọng số rồi chạy TOPSIS"
        )

        # Tạo ma trận so sánh cặp nhất quán từ vector trọng số chuyên gia.
        # A_ij = w_i / w_j
        pairwise_matrix = (
            expert_weights[:, None]
            / expert_weights[None, :]
        )

        eigenvalues, eigenvectors = np.linalg.eig(
            pairwise_matrix
        )

        principal_index = int(
            np.argmax(
                eigenvalues.real
            )
        )

        ahp_weights = np.abs(
            eigenvectors[
                :,
                principal_index,
            ].real
        )

        ahp_weights = (
            ahp_weights
            / ahp_weights.sum()
        )

        lambda_max = float(
            eigenvalues[
                principal_index
            ].real
        )

        n = len(expert_weights)

        consistency_index = (
            (lambda_max - n)
            / (n - 1)
        )

        # Random Index cho n=8
        random_index = 1.41

        consistency_ratio = (
            consistency_index
            / random_index
            if random_index != 0
            else 0
        )

        ahp_score = topsis_score(
            df,
            criteria,
            ahp_weights,
            is_benefit,
        )

        ahp_result = _b6_rank_result(
            df,
            ahp_score,
            "TOPSIS-AHP",
        )

        ahp_weight_table = pd.DataFrame(
            {
                "Tiêu chí": [
                    criterion_labels[col]
                    for col in criteria
                ],
                "Trọng số AHP": ahp_weights,
            }
        )

        kpi_cards(
            [
                (
                    "λmax",
                    f"{lambda_max:.4f}",
                    "giá trị riêng lớn nhất",
                ),
                (
                    "CI",
                    f"{consistency_index:.4f}",
                    "Consistency Index",
                ),
                (
                    "CR",
                    f"{consistency_ratio:.4f}",
                    "CR < 0,10 là phù hợp",
                ),
                (
                    "Vùng dẫn đầu",
                    ahp_result.iloc[0]["Vùng"],
                    f"C* = {ahp_result.iloc[0]['TOPSIS-AHP']:.3f}",
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
                    {"TOPSIS-AHP": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        pairwise_df = pd.DataFrame(
            pairwise_matrix,
            index=[
                criterion_labels[col]
                for col in criteria
            ],
            columns=[
                criterion_labels[col]
                for col in criteria
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

        st.info(
            "Ma trận AHP trong dashboard được tạo từ bộ trọng số chuyên gia nên nhất quán hoàn toàn. "
            "Khi khảo sát chuyên gia thực tế, cần nhập trực tiếp đánh giá cặp theo thang Saaty 1-9."
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_result = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            "TOPSIS chuyên gia": expert_score,
            "TOPSIS Entropy": entropy_score,
        }
    )

    export_result["Hạng chuyên gia"] = (
        export_result[
            "TOPSIS chuyên gia"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    export_result["Hạng Entropy"] = (
        export_result[
            "TOPSIS Entropy"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    st.download_button(
        "Tải kết quả Bài 6 dạng CSV",
        data=export_result.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai6_topsis_6_vung.csv",
        mime="text/csv",
        key="download_bai6",
    )

    # =====================================================
    # 6.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown("## 6.5. Câu hỏi thảo luận chính sách")

    top_three = expert_result.head(3)["Vùng"].tolist()

    with st.expander(
        "a) Vùng nào dẫn đầu? Có nên đặt trung tâm AI quốc gia đầu tiên tại đó không?",
        expanded=True,
    ):
        st.markdown(
            f"Vùng dẫn đầu theo bộ trọng số chuyên gia là "
            f"**{expert_result.iloc[0]['Vùng']}**. "
            "Đây là ứng viên mạnh về kinh tế, hạ tầng số và năng lực AI. "
            "Tuy nhiên, quyết định cuối cùng còn phải xét đất đai, năng lượng, "
            "an ninh dữ liệu, liên kết đại học-doanh nghiệp và cân bằng vùng."
        )

    with st.expander(
        "b) Vì sao trọng số Entropy có thể làm thay đổi thứ hạng?",
        expanded=True,
    ):
        st.markdown(
            "Entropy trao trọng số lớn hơn cho tiêu chí có mức độ phân hóa dữ liệu cao. "
            "Do đó, vùng nổi bật ở các tiêu chí biến thiên mạnh có thể tăng hạng, "
            "dù chuyên gia không gán trọng số cao nhất cho tiêu chí đó."
        )

    with st.expander(
        "c) Tương quan cao giữa AI readiness và Internet ảnh hưởng thế nào?",
        expanded=True,
    ):
        st.markdown(
            "Tương quan cao có thể gây đếm trùng thông tin, làm một nhóm năng lực số "
            "được phản ánh nhiều lần. Có thể xử lý bằng PCA, loại bớt một tiêu chí, "
            "điều chỉnh trọng số theo tương quan hoặc sử dụng phương pháp CRITIC."
        )

    with st.expander(
        "d) Nếu xây dựng ba trung tâm AI quốc gia, nên chọn ba vùng nào?",
        expanded=True,
    ):
        st.markdown(
            f"Theo kết quả TOPSIS chuyên gia, ba vùng dẫn đầu là "
            f"**{', '.join(top_three)}**. "
            "Khi triển khai thực tế nên bổ sung các tiêu chí về năng lượng, "
            "an ninh, khả năng kết nối nghiên cứu và tác động lan tỏa sang vùng yếu hơn."
        )


def _b7_non_dominated_mask(cost_matrix):
    """
    Trả về mask của các nghiệm không bị trội.

    cost_matrix phải được đưa về dạng tất cả mục tiêu đều là MIN.
    Ví dụ Growth là mục tiêu MAX nên sử dụng -Growth.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n = len(cost_matrix)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        dominated_by_other = np.any(
            np.all(
                cost_matrix <= cost_matrix[i],
                axis=1,
            )
            & np.any(
                cost_matrix < cost_matrix[i],
                axis=1,
            )
        )

        if dominated_by_other:
            is_pareto[i] = False

    return is_pareto


def _b7_parameters():
    """
    Tham số cho bốn mục tiêu của Bài 7.
    """
    regions, items, beta, D0 = region_beta_matrix()

    emission_coef = np.array(
        [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
        dtype=float,
    )

    data_risk_coef = np.array(
        [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
        dtype=float,
    )

    human_protection_coef = np.array(
        [0.32, 0.28, 0.30, 0.35, 0.25, 0.30],
        dtype=float,
    )

    return (
        regions,
        items,
        beta,
        D0,
        emission_coef,
        data_risk_coef,
        human_protection_coef,
    )


def _b7_objectives(
    X,
    beta,
    emission_coef,
    data_risk_coef,
    human_protection_coef,
):
    """
    Tính bốn mục tiêu:
    1. Growth: tối đa hóa
    2. Inequality: tối thiểu hóa
    3. Emission: tối thiểu hóa
    4. DataRisk: tối thiểu hóa
    """
    region_budget = X.sum(axis=1)

    growth = float(
        np.sum(
            beta * X
        )
    )

    inequality = float(
        np.mean(
            np.abs(
                region_budget
                - region_budget.mean()
            )
        )
    )

    emission = float(
        np.sum(
            emission_coef
            * (
                X[:, 0]
                + X[:, 2]
            )
        )
    )

    data_risk = float(
        np.sum(
            data_risk_coef
            * X[:, 2]
        )
        - np.sum(
            human_protection_coef
            * X[:, 3]
        )
    )

    return (
        growth,
        inequality,
        emission,
        data_risk,
    )


@st.cache_data
def _b7_sample_pareto(
    n_samples=2200,
    seed=42,
    fairness_lambda=0.68,
):
    """
    Tạo nghiệm khả thi theo phương pháp xây dựng trực tiếp thay vì
    lấy mẫu ngẫu nhiên rồi loại bỏ gần như toàn bộ nghiệm.

    Nguyên nhân lỗi cũ:
    - fairness_lambda=0.68 rất sát biên khả thi;
    - Tây Nguyên và Trung du miền núi phía Bắc cần mức đầu tư D rất lớn;
    - lấy mẫu Dirichlet ngẫu nhiên hầu như không tạo được nghiệm thỏa đồng thời
      ràng buộc công bằng, sàn-trần vùng và H >= 12.000.

    Hàm mới chủ động:
    1. Tính mức D tối thiểu để đạt công bằng;
    2. Sinh ngân sách vùng trong [5.000, 12.000];
    3. Phân bổ D, H, I, AI sao cho mọi ràng buộc đều thỏa;
    4. Lọc tập nghiệm Pareto.
    """

    rng = np.random.default_rng(seed)

    (
        regions,
        items,
        beta,
        D0,
        emission_coef,
        data_risk_coef,
        human_protection_coef,
    ) = _b7_parameters()

    def allocate_bounded(
        total,
        lower,
        upper,
    ):
        """
        Phân bổ một tổng cố định trong khoảng lower <= x <= upper.
        """
        lower = np.asarray(
            lower,
            dtype=float,
        )

        upper = np.asarray(
            upper,
            dtype=float,
        )

        x = lower.copy()

        remaining = float(
            total - x.sum()
        )

        if remaining < -1e-7:
            return None

        if total > upper.sum() + 1e-7:
            return None

        for _ in range(100):
            if remaining <= 1e-8:
                break

            capacity = upper - x

            active = np.where(
                capacity > 1e-10
            )[0]

            if len(active) == 0:
                break

            weights = rng.dirichlet(
                np.ones(
                    len(active)
                )
            )

            proposed = (
                remaining
                * weights
            )

            added = np.minimum(
                proposed,
                capacity[active],
            )

            x[active] += added

            remaining -= float(
                added.sum()
            )

        if remaining > 1e-5:
            return None

        return x

    accepted_rows = []
    accepted_matrices = []

    # Mức Digital Index cao nhất ban đầu là 82.
    # Giữ M=82 giúp ràng buộc fairness_lambda=0.68 khả thi.
    target_max_index = float(
        np.max(D0)
    )

    gamma = 0.002

    # Mức đầu tư D tối thiểu để từng vùng đạt lambda*M
    digital_lower = np.maximum(
        0.0,
        (
            fairness_lambda
            * target_max_index
            - D0
        )
        / gamma,
    )

    # Mức đầu tư D tối đa để không vượt quá M
    digital_upper_global = np.maximum(
        digital_lower,
        (
            target_max_index
            - D0
        )
        / gamma,
    )

    # Ngân sách tối thiểu vùng phải đủ chứa D tối thiểu
    region_lower = np.maximum(
        5000.0,
        digital_lower,
    )

    region_upper = np.full(
        6,
        12000.0,
    )

    if region_lower.sum() > 50000:
        empty = pd.DataFrame(
            columns=[
                "Growth",
                "Inequality",
                "Emission",
                "DataRisk",
                "FairnessRatio",
            ]
        )
        return empty, []

    attempts = 0
    max_attempts = n_samples * 20

    while (
        len(accepted_rows) < n_samples
        and attempts < max_attempts
    ):
        attempts += 1

        # -------------------------------------------------
        # Bước 1. Sinh tổng ngân sách từng vùng
        # -------------------------------------------------
        region_budget = allocate_bounded(
            total=50000.0,
            lower=region_lower,
            upper=region_upper,
        )

        if region_budget is None:
            continue

        # -------------------------------------------------
        # Bước 2. Phân bổ chuyển đổi số D
        # -------------------------------------------------
        digital_upper = np.minimum(
            digital_upper_global,
            region_budget,
        )

        min_digital_total = float(
            digital_lower.sum()
        )

        # Phải chừa tối thiểu 12.000 cho nhân lực H
        max_digital_total = min(
            float(
                digital_upper.sum()
            ),
            50000.0 - 12000.0,
        )

        if (
            max_digital_total
            < min_digital_total
        ):
            continue

        digital_total = (
            min_digital_total
            + rng.beta(
                1.5,
                3.0,
            )
            * (
                max_digital_total
                - min_digital_total
            )
        )

        x_digital = allocate_bounded(
            total=digital_total,
            lower=digital_lower,
            upper=digital_upper,
        )

        if x_digital is None:
            continue

        remaining_after_digital = (
            region_budget
            - x_digital
        )

        # -------------------------------------------------
        # Bước 3. Phân bổ nhân lực H
        # -------------------------------------------------
        max_human_total = float(
            remaining_after_digital.sum()
        )

        if max_human_total < 12000.0:
            continue

        human_total = (
            12000.0
            + rng.beta(
                1.5,
                3.0,
            )
            * (
                max_human_total
                - 12000.0
            )
        )

        x_human = allocate_bounded(
            total=human_total,
            lower=np.zeros(6),
            upper=remaining_after_digital,
        )

        if x_human is None:
            continue

        # -------------------------------------------------
        # Bước 4. Phần còn lại chia cho I và AI
        # -------------------------------------------------
        remaining_for_i_ai = (
            remaining_after_digital
            - x_human
        )

        infrastructure_share = rng.beta(
            1.4,
            1.4,
            size=6,
        )

        x_infrastructure = (
            remaining_for_i_ai
            * infrastructure_share
        )

        x_ai = (
            remaining_for_i_ai
            - x_infrastructure
        )

        X = np.column_stack(
            [
                x_infrastructure,
                x_digital,
                x_ai,
                x_human,
            ]
        )

        # -------------------------------------------------
        # Bước 5. Kiểm tra toàn bộ ràng buộc
        # -------------------------------------------------
        regional_totals = X.sum(
            axis=1
        )

        digital_after = (
            D0
            + gamma
            * X[:, 1]
        )

        fairness_ratio = float(
            digital_after.min()
            / max(
                digital_after.max(),
                1e-12,
            )
        )

        feasible = (
            abs(
                X.sum()
                - 50000.0
            )
            <= 1e-4
            and np.all(
                regional_totals
                >= 5000.0 - 1e-5
            )
            and np.all(
                regional_totals
                <= 12000.0 + 1e-5
            )
            and X[:, 3].sum()
            >= 12000.0 - 1e-5
            and fairness_ratio
            >= fairness_lambda - 1e-8
        )

        if not feasible:
            continue

        (
            growth,
            inequality,
            emission,
            data_risk,
        ) = _b7_objectives(
            X,
            beta,
            emission_coef,
            data_risk_coef,
            human_protection_coef,
        )

        accepted_rows.append(
            [
                growth,
                inequality,
                emission,
                data_risk,
                fairness_ratio,
            ]
        )

        accepted_matrices.append(
            X
        )

    if not accepted_rows:
        empty = pd.DataFrame(
            columns=[
                "Growth",
                "Inequality",
                "Emission",
                "DataRisk",
                "FairnessRatio",
            ]
        )
        return empty, []

    objective_df = pd.DataFrame(
        accepted_rows,
        columns=[
            "Growth",
            "Inequality",
            "Emission",
            "DataRisk",
            "FairnessRatio",
        ],
    )

    # Chuyển tất cả mục tiêu về dạng MIN để lọc Pareto
    cost_matrix = np.column_stack(
        [
            -objective_df["Growth"],
            objective_df["Inequality"],
            objective_df["Emission"],
            objective_df["DataRisk"],
        ]
    )

    pareto_mask = _b7_non_dominated_mask(
        cost_matrix
    )

    pareto_df = (
        objective_df.loc[
            pareto_mask
        ]
        .reset_index(
            drop=True
        )
    )

    pareto_matrices = [
        accepted_matrices[i]
        for i, is_pareto
        in enumerate(
            pareto_mask
        )
        if is_pareto
    ]

    pareto_df[
        "Growth_norm"
    ] = minmax(
        pareto_df["Growth"]
    )

    pareto_df[
        "Inequality_norm"
    ] = minmax(
        pareto_df["Inequality"]
    )

    pareto_df[
        "Emission_norm"
    ] = minmax(
        pareto_df["Emission"]
    )

    pareto_df[
        "DataRisk_norm"
    ] = minmax(
        pareto_df["DataRisk"]
    )

    return (
        pareto_df,
        pareto_matrices,
    )



def _b7_compromise_score(
    pareto_df,
    weights,
):
    """
    Tính điểm thỏa hiệp:
    Growth càng cao càng tốt;
    ba mục tiêu còn lại càng thấp càng tốt.
    """
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = (
        weights
        / max(
            weights.sum(),
            1e-12,
        )
    )

    return (
        weights[0]
        * pareto_df[
            "Growth_norm"
        ]
        + weights[1]
        * (
            1
            - pareto_df[
                "Inequality_norm"
            ]
        )
        + weights[2]
        * (
            1
            - pareto_df[
                "Emission_norm"
            ]
        )
        + weights[3]
        * (
            1
            - pareto_df[
                "DataRisk_norm"
            ]
        )
    )


def page_7():
    hero(
        "Bài 7 — Tối ưu đa mục tiêu Pareto với khung NSGA-II",
        "Trình bày đầy đủ các mục 7.1-7.5: tăng trưởng, bao trùm, phát thải, rủi ro dữ liệu, tập Pareto, nghiệm thỏa hiệp và chi phí cơ hội.",
        ["7.1-7.5", "Pareto", "NSGA-II", "4 objectives", "Compromise solution"],
    )

    (
        regions,
        items,
        beta,
        D0,
        emission_coef,
        data_risk_coef,
        human_protection_coef,
    ) = _b7_parameters()

    # =====================================================
    # 7.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown(
        "## 7.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        Chính sách kinh tế số không chỉ tối đa hóa tăng trưởng. Nhà hoạch định còn phải
        cân bằng giữa **bao trùm vùng miền**, **phát thải** và **an ninh dữ liệu**.

        Các mục tiêu này xung đột với nhau nên không tồn tại một nghiệm tối ưu tuyệt đối.
        Thay vào đó, mô hình tạo ra **tập nghiệm Pareto**: không thể cải thiện một mục tiêu
        mà không làm ít nhất một mục tiêu khác xấu đi.
        """
    )

    # =====================================================
    # 7.2. Mô hình toán học
    # =====================================================
    st.markdown(
        "## 7.2. Mô hình toán học đa mục tiêu"
    )

    st.markdown(
        "### Mục tiêu 1 — Tối đa hóa tăng trưởng"
    )

    st.latex(
        r"\max f_1(x)="
        r"\sum_r\sum_j"
        r"\beta_{j,r}x_{j,r}"
    )

    st.markdown(
        "### Mục tiêu 2 — Tối thiểu hóa bất bình đẳng vùng"
    )

    st.latex(
        r"\min f_2(x)=G(x)"
    )

    st.markdown(
        "### Mục tiêu 3 — Tối thiểu hóa phát thải"
    )

    st.latex(
        r"\min f_3(x)="
        r"\sum_re_r"
        r"(x_{I,r}+x_{AI,r})"
    )

    st.markdown(
        "### Mục tiêu 4 — Tối thiểu hóa rủi ro dữ liệu"
    )

    st.latex(
        r"\min f_4(x)="
        r"\sum_r\rho_rx_{AI,r}"
        r"-\sum_r\sigma_rx_{H,r}"
    )

    st.markdown(
        "### Các ràng buộc"
    )

    st.latex(
        r"\sum_r\sum_jx_{j,r}"
        r"\leq50{,}000"
    )

    st.latex(
        r"5{,}000"
        r"\leq\sum_jx_{j,r}"
        r"\leq12{,}000"
    )

    st.latex(
        r"\sum_rx_{H,r}"
        r"\geq12{,}000"
    )

    st.latex(
        r"x_{j,r}\geq0"
    )

    st.warning(
        "Với bộ D₀ hiện tại, γ=0,002 và trần 12.000 tỷ/vùng, "
        "ngưỡng công bằng λ=0,70 không tạo được nghiệm khả thi cho Tây Nguyên. "
        "Dashboard dùng λ=0,68 để minh họa tập Pareto khả thi; báo cáo nên nêu rõ điều chỉnh này."
    )

    # =====================================================
    # 7.3. Bảng tham số
    # =====================================================
    st.markdown(
        "## 7.3. Bảng tham số bổ sung"
    )

    parameter_table = pd.DataFrame(
        {
            "Vùng": regions,
            "eᵣ - Phát thải": emission_coef,
            "ρᵣ - Rủi ro dữ liệu": data_risk_coef,
            "σᵣ - Bảo vệ bởi nhân lực": human_protection_coef,
            "Digital Index ban đầu": D0,
        }
    )

    st.dataframe(
        parameter_table,
        use_container_width=True,
        hide_index=True,
    )

    n_samples = st.slider(
        "Số nghiệm khả thi dùng để xấp xỉ Pareto",
        min_value=800,
        max_value=4000,
        value=2200,
        step=400,
        key="b7_samples",
    )

    pareto_df, pareto_matrices = _b7_sample_pareto(
        n_samples=n_samples,
        seed=42,
        fairness_lambda=0.68,
    )

    if pareto_df.empty:
        st.error(
            "Không tạo được tập nghiệm Pareto. "
            "Hãy giảm số mẫu hoặc kiểm tra lại ràng buộc."
        )
        return

    # Trọng số thỏa hiệp mặc định
    default_weights = np.array(
        [0.40, 0.25, 0.20, 0.15],
        dtype=float,
    )

    pareto_df = pareto_df.copy()

    pareto_df[
        "CompromiseScore"
    ] = _b7_compromise_score(
        pareto_df,
        default_weights,
    )

    # =====================================================
    # 7.4. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 7.4. Yêu cầu lập trình"
    )

    tab741, tab742, tab743, tab744 = st.tabs(
        [
            "7.4.1 - Pareto/NSGA-II",
            "7.4.2 - Trực quan",
            "7.4.3 - Nghiệm thỏa hiệp",
            "7.4.4 - Chi phí cơ hội",
        ]
    )

    # -----------------------------------------------------
    # 7.4.1
    # -----------------------------------------------------
    with tab741:
        st.markdown(
            "### Câu 7.4.1. Xây dựng tập nghiệm Pareto"
        )

        kpi_cards(
            [
                (
                    "Số nghiệm Pareto",
                    f"{len(pareto_df)}",
                    "nghiệm không bị trội",
                ),
                (
                    "Growth lớn nhất",
                    f"{pareto_df['Growth'].max():,.0f}",
                    "GDP gain",
                ),
                (
                    "Inequality nhỏ nhất",
                    f"{pareto_df['Inequality'].min():,.0f}",
                    "độ lệch vùng",
                ),
                (
                    "Fairness thấp nhất",
                    f"{pareto_df['FairnessRatio'].min():.3f}",
                    "yêu cầu ≥0,68",
                ),
            ]
        )

        st.dataframe(
            pareto_df.sort_values(
                "CompromiseScore",
                ascending=False,
            ).head(20),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Dashboard dùng lấy mẫu khả thi và lọc nondominated để chạy ổn định trên Streamlit. "
            "Khung mã NSGA-II bằng pymoo được cung cấp bên dưới để đáp ứng yêu cầu phương pháp."
        )

        with st.expander(
            "Xem khung mã pymoo/NSGA-II"
        ):
            st.code(
                """from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class VietnamDigitalProblem(
    ElementwiseProblem
):
    def __init__(self):
        super().__init__(
            n_var=24,
            n_obj=4,
            n_ieq_constr=constraints_count,
            xl=np.zeros(24),
            xu=np.ones(24) * 12000
        )

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs
    ):
        X = x.reshape(6, 4)

        growth = (
            beta * X
        ).sum()

        inequality = np.abs(
            X.sum(axis=1)
            - X.sum(axis=1).mean()
        ).mean()

        emission = (
            e * (
                X[:, 0]
                + X[:, 2]
            )
        ).sum()

        data_risk = (
            rho * X[:, 2]
        ).sum() - (
            sigma * X[:, 3]
        ).sum()

        out["F"] = [
            -growth,
            inequality,
            emission,
            data_risk
        ]

        out["G"] = constraint_vector

algorithm = NSGA2(
    pop_size=100
)

result = minimize(
    VietnamDigitalProblem(),
    algorithm,
    ("n_gen", 200),
    seed=42,
    verbose=False
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 7.4.2
    # -----------------------------------------------------
    with tab742:
        st.markdown(
            "### Câu 7.4.2. Vẽ đường biên Pareto"
        )

        fig_3d = px.scatter_3d(
            pareto_df,
            x="Growth",
            y="Inequality",
            z="Emission",
            color="DataRisk",
            size="CompromiseScore",
            template=PLOT_TEMPLATE,
            title="Đường biên Pareto 3D",
            hover_data=[
                "FairnessRatio",
                "CompromiseScore",
            ],
        )

        fig_3d.update_layout(
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=54,
                b=0,
            ),
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True,
        )

        parallel_data = pareto_df[
            [
                "Growth",
                "Inequality",
                "Emission",
                "DataRisk",
                "FairnessRatio",
                "CompromiseScore",
            ]
        ].copy()

        fig_parallel = px.parallel_coordinates(
            parallel_data,
            dimensions=[
                "Growth",
                "Inequality",
                "Emission",
                "DataRisk",
                "FairnessRatio",
            ],
            color="CompromiseScore",
            title="Parallel coordinates của tập Pareto",
        )

        fig_parallel.update_layout(
            height=560,
        )

        st.plotly_chart(
            fig_parallel,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 7.4.3
    # -----------------------------------------------------
    with tab743:
        st.markdown(
            "### Câu 7.4.3. Chọn nghiệm thỏa hiệp bằng trọng số chính sách"
        )

        c1, c2, c3, c4 = st.columns(4)

        w_growth = c1.slider(
            "Growth",
            0.0,
            1.0,
            0.40,
            0.05,
            key="b7_w_growth",
        )

        w_inclusion = c2.slider(
            "Bao trùm",
            0.0,
            1.0,
            0.25,
            0.05,
            key="b7_w_inclusion",
        )

        w_emission = c3.slider(
            "Môi trường",
            0.0,
            1.0,
            0.20,
            0.05,
            key="b7_w_emission",
        )

        w_data = c4.slider(
            "An ninh dữ liệu",
            0.0,
            1.0,
            0.15,
            0.05,
            key="b7_w_data",
        )

        compromise_weights = np.array(
            [
                w_growth,
                w_inclusion,
                w_emission,
                w_data,
            ]
        )

        if compromise_weights.sum() <= 0:
            st.error(
                "Tổng trọng số phải lớn hơn 0."
            )
            return

        compromise_score = _b7_compromise_score(
            pareto_df,
            compromise_weights,
        )

        best_position = int(
            np.argmax(
                compromise_score.to_numpy()
            )
        )

        best_row = pareto_df.iloc[
            best_position
        ].copy()

        best_matrix = pareto_matrices[
            best_position
        ]

        allocation_table = pd.DataFrame(
            best_matrix,
            columns=items,
        )

        allocation_table.insert(
            0,
            "Vùng",
            regions,
        )

        allocation_table[
            "Tổng vùng"
        ] = best_matrix.sum(
            axis=1
        )

        kpi_cards(
            [
                (
                    "Growth",
                    f"{best_row['Growth']:,.0f}",
                    "mục tiêu tối đa hóa",
                ),
                (
                    "Inequality",
                    f"{best_row['Inequality']:,.0f}",
                    "mục tiêu tối thiểu hóa",
                ),
                (
                    "Emission",
                    f"{best_row['Emission']:,.0f}",
                    "mục tiêu tối thiểu hóa",
                ),
                (
                    "Data risk",
                    f"{best_row['DataRisk']:,.0f}",
                    "mục tiêu tối thiểu hóa",
                ),
            ]
        )

        st.dataframe(
            allocation_table,
            use_container_width=True,
            hide_index=True,
        )

        heatmap = px.imshow(
            pd.DataFrame(
                best_matrix,
                index=regions,
                columns=items,
            ),
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdPu",
            template=PLOT_TEMPLATE,
            title="Phân bổ của nghiệm thỏa hiệp",
        )

        heatmap.update_layout(
            height=560,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True,
        )

        st.info(
            f"Tỷ lệ công bằng của nghiệm thỏa hiệp là "
            f"**{best_row['FairnessRatio']:.3f}**."
        )

    # -----------------------------------------------------
    # 7.4.4
    # -----------------------------------------------------
    with tab744:
        st.markdown(
            "### Câu 7.4.4. Chi phí cơ hội của nghiệm tăng trưởng cao nhất"
        )

        growth_position = int(
            pareto_df[
                "Growth"
            ].to_numpy().argmax()
        )

        default_best_position = int(
            pareto_df[
                "CompromiseScore"
            ].to_numpy().argmax()
        )

        growth_row = pareto_df.iloc[
            growth_position
        ]

        compromise_row = pareto_df.iloc[
            default_best_position
        ]

        comparison = pd.DataFrame(
            {
                "Nghiệm": [
                    "Tăng trưởng cao nhất",
                    "Thỏa hiệp 0,40-0,25-0,20-0,15",
                ],
                "Growth": [
                    growth_row["Growth"],
                    compromise_row["Growth"],
                ],
                "Inequality": [
                    growth_row["Inequality"],
                    compromise_row["Inequality"],
                ],
                "Emission": [
                    growth_row["Emission"],
                    compromise_row["Emission"],
                ],
                "DataRisk": [
                    growth_row["DataRisk"],
                    compromise_row["DataRisk"],
                ],
                "FairnessRatio": [
                    growth_row["FairnessRatio"],
                    compromise_row["FairnessRatio"],
                ],
            }
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        growth_gain = (
            growth_row["Growth"]
            - compromise_row["Growth"]
        )

        inequality_change = (
            growth_row["Inequality"]
            - compromise_row["Inequality"]
        )

        emission_change = (
            growth_row["Emission"]
            - compromise_row["Emission"]
        )

        data_risk_change = (
            growth_row["DataRisk"]
            - compromise_row["DataRisk"]
        )

        kpi_cards(
            [
                (
                    "Growth tăng thêm",
                    f"{growth_gain:,.0f}",
                    "so với nghiệm thỏa hiệp",
                ),
                (
                    "Inequality thay đổi",
                    f"{inequality_change:+,.0f}",
                    "dương là xấu hơn",
                ),
                (
                    "Emission thay đổi",
                    f"{emission_change:+,.0f}",
                    "dương là xấu hơn",
                ),
                (
                    "Data risk thay đổi",
                    f"{data_risk_change:+,.0f}",
                    "dương là xấu hơn",
                ),
            ]
        )

        long_compare = comparison.melt(
            id_vars="Nghiệm",
            value_vars=[
                "Growth",
                "Inequality",
                "Emission",
                "DataRisk",
            ],
            var_name="Mục tiêu",
            value_name="Giá trị",
        )

        fig_compare = px.bar(
            long_compare,
            x="Mục tiêu",
            y="Giá trị",
            color="Nghiệm",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="Đánh đổi giữa tăng trưởng và các mục tiêu còn lại",
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

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_columns = [
        "Growth",
        "Inequality",
        "Emission",
        "DataRisk",
        "FairnessRatio",
        "CompromiseScore",
    ]

    st.download_button(
        "Tải tập Pareto Bài 7 dạng CSV",
        data=pareto_df[
            export_columns
        ].to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai7_pareto_front.csv",
        mime="text/csv",
        key="download_bai7",
    )

    # =====================================================
    # 7.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 7.5. Câu hỏi thảo luận chính sách"
    )

    with st.expander(
        "a) Đánh đổi tăng trưởng và bao trùm có rõ ràng không?",
        expanded=True,
    ):
        correlation = pareto_df[
            [
                "Growth",
                "Inequality",
            ]
        ].corr().iloc[
            0,
            1,
        ]

        st.markdown(
            f"Hệ số tương quan trong tập Pareto giữa Growth và Inequality là "
            f"**{correlation:.3f}**. Nếu hệ số dương và đủ lớn, tăng trưởng cao hơn "
            "thường đi kèm mức độ tập trung ngân sách vùng lớn hơn."
        )

    with st.expander(
        "b) Bộ trọng số 0,40-0,25-0,20-0,15 có phù hợp không?",
        expanded=True,
    ):
        st.markdown(
            "Bộ trọng số này ưu tiên tăng trưởng nhưng vẫn dành 60% tổng trọng số cho "
            "bao trùm, môi trường và an ninh dữ liệu. Có thể tăng trọng số môi trường "
            "để phản ánh cam kết COP26 hoặc tăng trọng số dữ liệu khi mở rộng AI quy mô lớn."
        )

    with st.expander(
        "c) NSGA-II có thay thế quyết định chính trị không?",
        expanded=True,
    ):
        st.markdown(
            "Không. NSGA-II chỉ tạo tập phương án và lượng hóa đánh đổi. "
            "Lựa chọn cuối cùng vẫn cần tham vấn xã hội, đánh giá pháp lý, "
            "trách nhiệm giải trình và quyết định của cơ quan có thẩm quyền."
        )


def _b8_initial_state():
    """
    Lấy trạng thái cuối năm 2025 từ hàm compute_tfp() đã có trong app.py.
    Thứ tự trả về dự kiến:
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


def _b8_simulate(
    shares_matrix,
    invest_rates=None,
    shock_2028=0.0,
    rho=0.97,
):
    """
    Mô phỏng động giai đoạn 2026-2035.

    shares_matrix:
        Ma trận 10×4, mỗi hàng là tỷ trọng đầu tư vào K, D, AI, H.

    invest_rates:
        Tỷ lệ đầu tư trên GDP từng năm. Nếu None, dùng 22%/năm.

    shock_2028:
        Mức giảm GDP năm 2028. Ví dụ 0.08 tương ứng giảm 8%.

    rho:
        Hệ số chiết khấu phúc lợi.
    """
    initial = _b8_initial_state()

    years = np.arange(
        2026,
        2036,
    )

    T = len(years)

    shares_matrix = np.asarray(
        shares_matrix,
        dtype=float,
    )

    if shares_matrix.shape != (T, 4):
        raise ValueError(
            "shares_matrix phải có kích thước 10×4."
        )

    row_sum = shares_matrix.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sum <= 0
    ):
        raise ValueError(
            "Mỗi năm phải có tổng tỷ trọng đầu tư lớn hơn 0."
        )

    shares_matrix = (
        shares_matrix
        / row_sum
    )

    if invest_rates is None:
        invest_rates = np.full(
            T,
            0.22,
            dtype=float,
        )
    else:
        invest_rates = np.asarray(
            invest_rates,
            dtype=float,
        )

    if len(invest_rates) != T:
        raise ValueError(
            "invest_rates phải có 10 phần tử."
        )

    # Trạng thái đầu năm 2026
    K = initial["K"] * 1.06
    L = initial["L"] * 1.006
    D = initial["D"] + 0.80
    AI = initial["AI"] + 6.00
    H = initial["H"] + 0.80
    A = initial["A"] * 1.012

    # Tham số động học
    delta_K = 0.05
    delta_D = 0.12
    delta_AI = 0.15
    mu_H = 0.02
    theta_H = 0.80

    # Hệ số quy đổi đầu tư thành biến trạng thái
    scale_D = 240.0
    scale_AI = 135.0
    scale_H = 520.0

    welfare = 0.0
    rows = []

    for t, year in enumerate(
        years
    ):
        Y = (
            A
            * K**0.33
            * L**0.42
            * D**0.10
            * AI**0.08
            * H**0.07
        )

        if year == 2028:
            Y *= (
                1.0
                - float(shock_2028)
            )

        investment_rate = float(
            invest_rates[t]
        )

        total_investment = (
            investment_rate
            * Y
        )

        consumption = max(
            Y - total_investment,
            1e-9,
        )

        welfare += (
            rho**t
            * np.log(
                consumption
            )
        )

        shares = shares_matrix[t]

        I_K = (
            shares[0]
            * total_investment
        )

        I_D = (
            shares[1]
            * total_investment
        )

        I_AI = (
            shares[2]
            * total_investment
        )

        I_H = (
            shares[3]
            * total_investment
        )

        rows.append(
            [
                year,
                Y,
                consumption,
                total_investment,
                investment_rate,
                K,
                D,
                AI,
                H,
                A,
                I_K,
                I_D,
                I_AI,
                I_H,
                shares[0],
                shares[1],
                shares[2],
                shares[3],
                welfare,
            ]
        )

        # Phương trình chuyển trạng thái
        K = (
            (1 - delta_K) * K
            + I_K
        )

        D = max(
            1e-6,
            (1 - delta_D) * D
            + I_D / scale_D,
        )

        AI = max(
            1e-6,
            (1 - delta_AI) * AI
            + I_AI / scale_AI,
        )

        H = max(
            1e-6,
            H
            + theta_H * I_H / scale_H
            - mu_H * H,
        )

        # TFP tăng nội sinh theo D, AI và H
        A = (
            A
            * (
                1
                + 0.00008 * D
                + 0.00004 * AI
                + 0.00006 * H
            )
        )

        L *= 1.006

    return pd.DataFrame(
        rows,
        columns=[
            "Năm",
            "GDP",
            "Tiêu dùng",
            "Tổng đầu tư",
            "Tỷ lệ đầu tư",
            "K",
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
            "Welfare_lũy_kế",
        ],
    )


@st.cache_data
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

    T = 10

    initial_share = np.array(
        [0.34, 0.26, 0.18, 0.22],
        dtype=float,
    )

    x0 = np.tile(
        initial_share,
        T,
    )

    def objective(
        flat_shares
    ):
        shares = flat_shares.reshape(
            T,
            4,
        )

        simulation = _b8_simulate(
            shares_matrix=shares,
            invest_rates=np.full(
                T,
                0.22,
            ),
            shock_2028=shock_2028,
            rho=rho,
        )

        return -float(
            simulation.iloc[-1][
                "Welfare_lũy_kế"
            ]
        )

    constraints = []

    for t in range(T):
        constraints.append(
            {
                "type": "eq",
                "fun": (
                    lambda flat_shares, t=t:
                    np.sum(
                        flat_shares.reshape(
                            T,
                            4,
                        )[t]
                    )
                    - 1.0
                ),
            }
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (0.02, 0.85)
        ]
        * (
            T * 4
        ),
        constraints=constraints,
        options={
            "maxiter": 300,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    optimized_shares = result.x.reshape(
        T,
        4,
    )

    optimized_shares = (
        optimized_shares
        / optimized_shares.sum(
            axis=1,
            keepdims=True,
        )
    )

    simulation = _b8_simulate(
        shares_matrix=optimized_shares,
        invest_rates=np.full(
            T,
            0.22,
        ),
        shock_2028=shock_2028,
        rho=rho,
    )

    return (
        optimized_shares,
        simulation,
        bool(result.success),
        str(result.message),
    )


def page_8():
    hero(
        "Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
        "Trình bày đầy đủ các mục 8.1-8.4: động học K-D-AI-H, quỹ đạo tối ưu, cú sốc 2028 và so sánh đầu tư trải đều với front-load.",
        ["8.1-8.4", "Dynamic optimization", "SLSQP", "Welfare", "2026-2035"],
    )

    # =====================================================
    # 8.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown(
        "## 8.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        Đầu tư số có tác động tích lũy và độ trễ. Đầu tư mạnh vào AI hiện tại có thể
        chưa tạo lợi ích nếu thiếu nhân lực số; ngược lại, đầu tư vào nhân lực và hạ tầng
        tạo nền tảng hấp thụ công nghệ trong dài hạn.

        Bài 8 xây dựng mô hình tối ưu động giai đoạn **2026-2035** nhằm lựa chọn quỹ đạo
        đầu tư vào vốn vật chất, chuyển đổi số, AI và nhân lực số để tối đa hóa tổng phúc lợi.
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
        r"\max"
        r"\sum_{t=2026}^{2035}"
        r"\rho^{t-2026}"
        r"\ln(C_t)"
    )

    st.markdown(
        "### Hàm sản xuất"
    )

    st.latex(
        r"Y_t="
        r"A_t"
        r"K_t^{0.33}"
        r"L_t^{0.42}"
        r"D_t^{0.10}"
        r"AI_t^{0.08}"
        r"H_t^{0.07}"
    )

    st.markdown(
        "### Phương trình chuyển trạng thái"
    )

    st.latex(
        r"K_{t+1}="
        r"(1-\delta_K)K_t"
        r"+I_{K,t}"
    )

    st.latex(
        r"D_{t+1}="
        r"(1-\delta_D)D_t"
        r"+I_{D,t}"
    )

    st.latex(
        r"AI_{t+1}="
        r"(1-\delta_{AI})AI_t"
        r"+I_{AI,t}"
    )

    st.latex(
        r"H_{t+1}="
        r"H_t+\theta_HI_{H,t}"
        r"-\mu H_t"
    )

    st.markdown(
        "### Ràng buộc nguồn lực"
    )

    st.latex(
        r"C_t"
        r"+I_{K,t}"
        r"+I_{D,t}"
        r"+I_{AI,t}"
        r"+I_{H,t}"
        r"\leq Y_t"
    )

    st.latex(
        r"I_{K,t},I_{D,t},I_{AI,t},I_{H,t}\geq0"
    )

    rho = st.slider(
        "Hệ số chiết khấu ρ",
        min_value=0.90,
        max_value=0.99,
        value=0.97,
        step=0.01,
        key="b8_rho",
    )

    (
        optimal_shares,
        optimal_simulation,
        solver_success,
        solver_message,
    ) = _b8_optimize_shares(
        rho=rho,
        shock_2028=0.0,
    )

    # =====================================================
    # 8.3. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 8.3. Yêu cầu lập trình"
    )

    tab831, tab832, tab833, tab834 = st.tabs(
        [
            "8.3.1 - Tối ưu SLSQP",
            "8.3.2 - Quỹ đạo",
            "8.3.3 - Cú sốc 2028",
            "8.3.4 - Hai chiến lược",
        ]
    )

    # -----------------------------------------------------
    # 8.3.1
    # -----------------------------------------------------
    with tab831:
        st.markdown(
            "### Câu 8.3.1. Giải bài toán phi tuyến bằng SLSQP"
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
            np.arange(
                2026,
                2036,
            ),
        )

        kpi_cards(
            [
                (
                    "Trạng thái solver",
                    (
                        "Thành công"
                        if solver_success
                        else "Cảnh báo"
                    ),
                    solver_message[:45],
                ),
                (
                    "Welfare",
                    f"{optimal_simulation.iloc[-1]['Welfare_lũy_kế']:.3f}",
                    f"ρ = {rho:.2f}",
                ),
                (
                    "GDP năm 2035",
                    f"{optimal_simulation.iloc[-1]['GDP']:,.0f}",
                    "kết quả mô phỏng",
                ),
                (
                    "Tiêu dùng 2035",
                    f"{optimal_simulation.iloc[-1]['Tiêu dùng']:,.0f}",
                    "kết quả mô phỏng",
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
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_shares,
            use_container_width=True,
        )

        with st.expander(
            "Xem mã SLSQP rút gọn"
        ):
            st.code(
                """from scipy.optimize import minimize

def objective(flat_shares):
    shares = flat_shares.reshape(10, 4)
    simulation = simulate(shares)
    welfare = simulation.iloc[-1][
        "Welfare_lũy_kế"
    ]
    return -welfare

constraints = [
    {
        "type": "eq",
        "fun": lambda x, t=t:
        x.reshape(10,4)[t].sum() - 1
    }
    for t in range(10)
]

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=[(0.02,0.85)] * 40,
    constraints=constraints
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 8.3.2
    # -----------------------------------------------------
    with tab832:
        st.markdown(
            "### Câu 8.3.2. Vẽ quỹ đạo tối ưu K, D, AI, H, Y và C"
        )

        c1, c2 = st.columns(
            2
        )

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
                height=460,
                margin=dict(
                    l=10,
                    r=10,
                    t=54,
                    b=10,
                ),
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
                height=460,
                margin=dict(
                    l=10,
                    r=10,
                    t=54,
                    b=10,
                ),
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

    # -----------------------------------------------------
    # 8.3.3
    # -----------------------------------------------------
    with tab833:
        st.markdown(
            "### Câu 8.3.3. Cú sốc GDP năm 2028 giảm 8% và tối ưu lại"
        )

        (
            shock_shares,
            shock_simulation,
            shock_success,
            shock_message,
        ) = _b8_optimize_shares(
            rho=rho,
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
                        if shock_success
                        else "Cảnh báo"
                    ),
                    shock_message[:45],
                ),
                (
                    "Welfare mất đi",
                    f"{welfare_loss:.4f}",
                    "so với kịch bản cơ sở",
                ),
                (
                    "GDP 2035 thay đổi",
                    f"{gdp_2035_change:+,.0f}",
                    "sau tối ưu lại",
                ),
                (
                    "Cú sốc năm 2028",
                    "-8%",
                    "GDP",
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
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_shock,
            use_container_width=True,
        )

        share_change = pd.DataFrame(
            {
                "Năm": np.arange(
                    2026,
                    2036,
                ),
                "ΔShare_K": (
                    shock_shares[:, 0]
                    - optimal_shares[:, 0]
                ),
                "ΔShare_D": (
                    shock_shares[:, 1]
                    - optimal_shares[:, 1]
                ),
                "ΔShare_AI": (
                    shock_shares[:, 2]
                    - optimal_shares[:, 2]
                ),
                "ΔShare_H": (
                    shock_shares[:, 3]
                    - optimal_shares[:, 3]
                ),
            }
        )

        st.markdown(
            "#### Chính sách tối ưu thay đổi sau cú sốc"
        )

        st.dataframe(
            share_change.style.format(
                {
                    "ΔShare_K": "{:+.4f}",
                    "ΔShare_D": "{:+.4f}",
                    "ΔShare_AI": "{:+.4f}",
                    "ΔShare_H": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 8.3.4
    # -----------------------------------------------------
    with tab834:
        st.markdown(
            "### Câu 8.3.4. So sánh đầu tư trải đều và front-load"
        )

        fixed_shares = np.tile(
            np.array(
                [0.34, 0.26, 0.18, 0.22]
            ),
            (
                10,
                1,
            ),
        )

        # Tổng bằng 2,20 cho cả hai chiến lược
        equal_rates = np.full(
            10,
            0.22,
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

        strategy_comparison = pd.DataFrame(
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
                    equal_simulation.iloc[-1]["Tiêu dùng"],
                    front_simulation.iloc[-1]["Tiêu dùng"],
                ],
            }
        )

        st.dataframe(
            strategy_comparison,
            use_container_width=True,
            hide_index=True,
        )

        strategy_long = pd.concat(
            [
                equal_simulation[
                    [
                        "Năm",
                        "GDP",
                    ]
                ].assign(
                    Chiến_lược="Trải đều"
                ),
                front_simulation[
                    [
                        "Năm",
                        "GDP",
                    ]
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
            height=480,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_strategy,
            use_container_width=True,
        )

        better_strategy = (
            strategy_comparison.sort_values(
                "Welfare",
                ascending=False,
            ).iloc[0]["Chiến lược"]
        )

        st.success(
            f"Chiến lược có phúc lợi cao hơn trong mô phỏng là **{better_strategy}**."
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    st.download_button(
        "Tải quỹ đạo tối ưu Bài 8",
        data=optimal_simulation.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai8_quy_dao_toi_uu_2026_2035.csv",
        mime="text/csv",
        key="download_bai8",
    )

    # =====================================================
    # 8.4. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 8.4. Câu hỏi thảo luận chính sách"
    )

    first_three_average = optimal_shares[
        :3
    ].mean(
        axis=0
    )

    last_three_average = optimal_shares[
        -3:
    ].mean(
        axis=0
    )

    front_loaded = bool(
        (
            first_three_average[
                1:
            ].sum()
            > last_three_average[
                1:
            ].sum()
        )
    )

    with st.expander(
        "a) Quỹ đạo đầu tư tối ưu có front-loaded không?",
        expanded=True,
    ):
        st.markdown(
            (
                "Kết quả cho thấy đầu tư số, AI và nhân lực trong ba năm đầu "
                "cao hơn ba năm cuối, nên quỹ đạo có xu hướng **front-loaded**."
                if front_loaded
                else
                "Kết quả không cho thấy xu hướng front-loaded rõ ràng; "
                "tỷ trọng đầu tư số, AI và nhân lực khá ổn định hoặc tăng về cuối kỳ."
            )
        )

        st.markdown(
            "Front-load có thể tận dụng hiệu ứng tích lũy TFP, nhưng làm giảm tiêu dùng "
            "ở giai đoạn đầu và đòi hỏi năng lực giải ngân cao."
        )

    with st.expander(
        "b) Nên đầu tư AI trước nhân lực hay đồng thời?",
        expanded=True,
    ):
        st.markdown(
            "AI có tốc độ khấu hao nhanh và phụ thuộc vào kỹ năng lao động. "
            "Vì vậy, nên đầu tư nhân lực trước một bước hoặc đồng thời với AI. "
            "Đầu tư AI mạnh khi năng lực hấp thụ thấp có thể tạo tài sản công nghệ "
            "nhưng không chuyển hóa thành năng suất."
        )

    with st.expander(
        "c) Khi ρ giảm từ 0,97 xuống 0,90, chính sách thay đổi thế nào?",
        expanded=True,
    ):
        st.markdown(
            "ρ thấp làm lợi ích tương lai có giá trị hiện tại nhỏ hơn. "
            "Mô hình vì vậy có xu hướng ưu tiên tiêu dùng hiện tại, giảm đầu tư "
            "có thời gian hoàn vốn dài và có thể chuyển ngân sách từ nhân lực, "
            "R&D hoặc nền tảng số sang vốn tạo sản lượng ngắn hạn."
        )


def _b9_parameters():
    """
    Bộ tham số minh họa tác động AI và đào tạo lại tại 8 ngành.

    Đơn vị:
    - Lao động: triệu người
    - x_AI, x_H: tỷ VND
    - Các hệ số việc làm: số việc làm trên một tỷ VND đầu tư
    """
    return {
        "Ngành": [
            "Nông-Lâm-Thủy sản",
            "Công nghiệp chế biến, chế tạo",
            "Xây dựng",
            "Bán buôn và bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
        ],
        "Lao động (triệu)": np.array(
            [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
            dtype=float,
        ),
        "Risk": np.array(
            [18, 42, 25, 38, 52, 35, 28, 22],
            dtype=float,
        ) / 100.0,
        "a1": np.array(
            [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
            dtype=float,
        ),
        "b1": np.array(
            [45, 28, 35, 32, 22, 30, 20, 55],
            dtype=float,
        ),
        "c1": np.array(
            [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
            dtype=float,
        ),
        "d1": np.array(
            [50, 32, 42, 38, 26, 36, 24, 62],
            dtype=float,
        ),
    }


def _b9_solve(
    add_5pct_limit=False,
    budget=30000.0,
    cap_per_sector=6000.0,
):
    """
    Giải LP phân bổ đầu tư AI và đào tạo lại cho 8 ngành.

    Biến:
    - x_AI_i: đầu tư AI ngành i
    - x_H_i: đầu tư đào tạo lại/người lao động ngành i

    Mục tiêu:
        Max tổng NetJob.

    Ràng buộc:
    - Tổng đầu tư <= budget
    - Tổng AI >= 30% ngân sách
    - Tổng đào tạo >= 30% ngân sách
    - Mỗi ngành không nhận quá cap_per_sector
    - NetJob_i >= 0
    - DisplacedJob_i <= RetrainingCapacity_i
    - Kịch bản 9.4.4: mất việc <= 5% lao động ngành
    """
    p = _b9_parameters()
    n_sector = len(p["Ngành"])

    # Việc làm ròng:
    # NetJob = a1*x_AI + b1*x_H - c1*risk*x_AI
    ai_net_coefficient = (
        p["a1"]
        - p["c1"] * p["Risk"]
    )

    human_coefficient = p["b1"]

    # linprog tối thiểu hóa nên đổi dấu hàm mục tiêu
    c = -np.r_[
        ai_net_coefficient,
        human_coefficient,
    ]

    A_ub = []
    b_ub = []

    # C1. Tổng ngân sách
    row = np.ones(
        2 * n_sector,
        dtype=float,
    )
    A_ub.append(row)
    b_ub.append(float(budget))

    # C2. Tổng đầu tư AI >= 30% ngân sách
    row = np.zeros(
        2 * n_sector,
        dtype=float,
    )
    row[:n_sector] = -1.0
    A_ub.append(row)
    b_ub.append(-0.30 * budget)

    # C3. Tổng đầu tư đào tạo >= 30% ngân sách
    row = np.zeros(
        2 * n_sector,
        dtype=float,
    )
    row[n_sector:] = -1.0
    A_ub.append(row)
    b_ub.append(-0.30 * budget)

    for i in range(n_sector):
        # C4. NetJob_i >= 0
        row = np.zeros(
            2 * n_sector,
            dtype=float,
        )
        row[i] = -ai_net_coefficient[i]
        row[n_sector + i] = -human_coefficient[i]
        A_ub.append(row)
        b_ub.append(0.0)

        # C5. DisplacedJob_i <= RetrainingCapacity_i
        row = np.zeros(
            2 * n_sector,
            dtype=float,
        )
        row[i] = (
            p["c1"][i]
            * p["Risk"][i]
        )
        row[n_sector + i] = -p["d1"][i]
        A_ub.append(row)
        b_ub.append(0.0)

        # C6. Trần đầu tư mỗi ngành
        row = np.zeros(
            2 * n_sector,
            dtype=float,
        )
        row[i] = 1.0
        row[n_sector + i] = 1.0
        A_ub.append(row)
        b_ub.append(float(cap_per_sector))

        # C7. Mất việc tối đa 5% lao động ngành
        if add_5pct_limit:
            row = np.zeros(
                2 * n_sector,
                dtype=float,
            )
            row[i] = (
                p["c1"][i]
                * p["Risk"][i]
            )
            A_ub.append(row)
            b_ub.append(
                0.05
                * p["Lao động (triệu)"][i]
                * 1_000_000
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
        bounds=[
            (0, None)
        ] * (
            2 * n_sector
        ),
        method="highs",
    )

    return result, p


def _b9_result_table(result, p):
    """
    Chuyển nghiệm LP thành bảng kết quả ngành.
    """
    n_sector = len(p["Ngành"])

    x_ai = result.x[:n_sector]
    x_h = result.x[n_sector:]

    new_job = (
        p["a1"]
        * x_ai
    )

    upgrade_job = (
        p["b1"]
        * x_h
    )

    displaced_job = (
        p["c1"]
        * p["Risk"]
        * x_ai
    )

    retraining_capacity = (
        p["d1"]
        * x_h
    )

    net_job = (
        new_job
        + upgrade_job
        - displaced_job
    )

    table = pd.DataFrame(
        {
            "Ngành": p["Ngành"],
            "x_AI": x_ai,
            "x_H": x_h,
            "NewJob": new_job,
            "UpgradeJob": upgrade_job,
            "DisplacedJob": displaced_job,
            "RetrainingCapacity": retraining_capacity,
            "NetJob": net_job,
        }
    )

    return table


def page_9():
    hero(
        "Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
        "Trình bày đầy đủ các mục 9.1-9.5: việc làm ròng, đầu tư đào tạo lại, ngưỡng nhân lực, nhóm dễ tổn thương và giới hạn mất việc.",
        ["9.1-9.5", "NetJob", "Retraining", "Labor LP", "Sankey"],
    )

    base_result, parameters = _b9_solve(
        add_5pct_limit=False,
        budget=30000.0,
        cap_per_sector=6000.0,
    )

    if not base_result.success:
        st.error(
            "Mô hình lao động cơ sở không khả thi. "
            "Hãy kiểm tra lại ngân sách và các ràng buộc."
        )
        return

    base_table = _b9_result_table(
        base_result,
        parameters,
    )

    # =====================================================
    # 9.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown(
        "## 9.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        AI vừa tạo ra việc làm mới vừa thay thế một phần nhiệm vụ hiện hữu.
        Nếu đầu tư AI nhanh hơn năng lực đào tạo lại, thất nghiệp cơ cấu có thể tăng,
        đặc biệt ở các ngành có tỷ lệ tự động hóa cao.

        Bài 9 phân bổ **30.000 tỷ VND** giữa đầu tư AI và đào tạo lại lao động tại
        tám ngành, nhằm tối đa hóa việc làm ròng nhưng vẫn bảo đảm tốc độ tự động hóa
        không vượt quá năng lực chuyển đổi kỹ năng.
        """
    )

    # =====================================================
    # 9.2. Mô hình toán học
    # =====================================================
    st.markdown(
        "## 9.2. Mô hình toán học"
    )

    st.markdown(
        "### Việc làm mới, việc làm nâng cấp và việc làm bị thay thế"
    )

    st.latex(
        r"NewJob_i="
        r"a_{1i}x_{AI,i}"
    )

    st.latex(
        r"UpgradeJob_i="
        r"b_{1i}x_{H,i}"
    )

    st.latex(
        r"DisplacedJob_i="
        r"c_{1i}Risk_i x_{AI,i}"
    )

    st.latex(
        r"RetrainingCapacity_i="
        r"d_{1i}x_{H,i}"
    )

    st.markdown(
        "### Việc làm ròng"
    )

    st.latex(
        r"NetJob_i="
        r"NewJob_i"
        r"+UpgradeJob_i"
        r"-DisplacedJob_i"
    )

    st.markdown(
        "### Hàm mục tiêu"
    )

    st.latex(
        r"\max"
        r"\sum_iNetJob_i"
    )

    st.markdown(
        "### Các ràng buộc"
    )

    st.latex(
        r"\sum_i"
        r"(x_{AI,i}+x_{H,i})"
        r"\leq30{,}000"
    )

    st.latex(
        r"\sum_ix_{AI,i}"
        r"\geq0.30\times30{,}000"
    )

    st.latex(
        r"\sum_ix_{H,i}"
        r"\geq0.30\times30{,}000"
    )

    st.latex(
        r"NetJob_i\geq0"
    )

    st.latex(
        r"DisplacedJob_i"
        r"\leq RetrainingCapacity_i"
    )

    st.latex(
        r"x_{AI,i},x_{H,i}\geq0"
    )

    # =====================================================
    # 9.3. Tham số 8 ngành
    # =====================================================
    st.markdown(
        "## 9.3. Tham số 8 ngành"
    )

    parameter_table = pd.DataFrame(
        {
            "Ngành": parameters["Ngành"],
            "Lao động (triệu)": parameters["Lao động (triệu)"],
            "Risk (%)": parameters["Risk"] * 100,
            "a1 - NewJob": parameters["a1"],
            "b1 - UpgradeJob": parameters["b1"],
            "c1 - Displaced": parameters["c1"],
            "d1 - Retraining": parameters["d1"],
        }
    )

    st.dataframe(
        parameter_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Các hệ số được sử dụng như bộ tham số mô phỏng cho bài tập, "
        "không phải kết quả ước lượng nhân quả từ dữ liệu vi mô."
    )

    # =====================================================
    # 9.4. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 9.4. Yêu cầu lập trình"
    )

    tab941, tab942, tab943, tab944 = st.tabs(
        [
            "9.4.1 - Phân bổ tối ưu",
            "9.4.2 - Ngưỡng đào tạo",
            "9.4.3 - Nhóm dễ tổn thương",
            "9.4.4 - Giới hạn 5%",
        ]
    )

    # -----------------------------------------------------
    # 9.4.1
    # -----------------------------------------------------
    with tab941:
        st.markdown(
            "### Câu 9.4.1. Giải LP và xác định phân bổ AI-đào tạo tối ưu"
        )

        total_net_job = float(
            base_table["NetJob"].sum()
        )

        total_ai = float(
            base_table["x_AI"].sum()
        )

        total_human = float(
            base_table["x_H"].sum()
        )

        top_sector = base_table.loc[
            base_table["NetJob"].idxmax(),
            "Ngành",
        ]

        kpi_cards(
            [
                (
                    "Tổng NetJob",
                    f"{total_net_job:,.0f}",
                    "việc làm ròng",
                ),
                (
                    "Tổng đầu tư AI",
                    f"{total_ai:,.0f}",
                    "tỷ VND",
                ),
                (
                    "Tổng đào tạo",
                    f"{total_human:,.0f}",
                    "tỷ VND",
                ),
                (
                    "NetJob cao nhất",
                    top_sector,
                    "theo nghiệm tối ưu",
                ),
            ]
        )

        st.dataframe(
            base_table.style.format(
                {
                    "x_AI": "{:.2f}",
                    "x_H": "{:.2f}",
                    "NewJob": "{:.0f}",
                    "UpgradeJob": "{:.0f}",
                    "DisplacedJob": "{:.0f}",
                    "RetrainingCapacity": "{:.0f}",
                    "NetJob": "{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        job_long = base_table.melt(
            id_vars="Ngành",
            value_vars=[
                "NewJob",
                "UpgradeJob",
                "DisplacedJob",
                "NetJob",
            ],
            var_name="Thành phần",
            value_name="Số việc làm",
        )

        fig_jobs = px.bar(
            job_long,
            x="Ngành",
            y="Số việc làm",
            color="Thành phần",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="Tác động việc làm theo ngành",
        )

        fig_jobs.update_layout(
            height=520,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_jobs,
            use_container_width=True,
        )

        investment_long = base_table.melt(
            id_vars="Ngành",
            value_vars=[
                "x_AI",
                "x_H",
            ],
            var_name="Loại đầu tư",
            value_name="Ngân sách",
        )

        fig_investment = px.bar(
            investment_long,
            x="Ngành",
            y="Ngân sách",
            color="Loại đầu tư",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="Phân bổ ngân sách AI và đào tạo lại",
        )

        fig_investment.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_investment,
            use_container_width=True,
        )

        with st.expander(
            "Xem mã LP rút gọn"
        ):
            st.code(
                """# Biến:
# x_AI[0:8], x_H[0:8]

c = -np.r_[
    a1 - c1*risk,
    b1
]

result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(0,None)] * 16,
    method="highs"
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 9.4.2
    # -----------------------------------------------------
    with tab942:
        st.markdown(
            "### Câu 9.4.2. Tính ngưỡng đào tạo lại tối thiểu"
        )

        selected_sector = st.selectbox(
            "Chọn ngành",
            options=parameters["Ngành"],
            index=1,
            key="b9_threshold_sector",
        )

        sector_index = parameters[
            "Ngành"
        ].index(
            selected_sector
        )

        assumed_ai = st.slider(
            "Giả định đầu tư AI của ngành (tỷ VND)",
            min_value=0,
            max_value=6000,
            value=3000,
            step=250,
            key="b9_assumed_ai",
        )

        net_job_threshold = max(
            0.0,
            (
                (
                    parameters["c1"][sector_index]
                    * parameters["Risk"][sector_index]
                    - parameters["a1"][sector_index]
                )
                / parameters["b1"][sector_index]
            )
            * assumed_ai,
        )

        retraining_threshold = (
            parameters["c1"][sector_index]
            * parameters["Risk"][sector_index]
            / parameters["d1"][sector_index]
        ) * assumed_ai

        final_threshold = max(
            net_job_threshold,
            retraining_threshold,
        )

        threshold_table = pd.DataFrame(
            {
                "Điều kiện": [
                    "NetJob ≥ 0",
                    "Displaced ≤ Retraining",
                    "Ngưỡng cuối cùng",
                ],
                "x_H tối thiểu": [
                    net_job_threshold,
                    retraining_threshold,
                    final_threshold,
                ],
            }
        )

        kpi_cards(
            [
                (
                    "Ngành",
                    selected_sector,
                    "phân tích ngưỡng",
                ),
                (
                    "x_AI giả định",
                    f"{assumed_ai:,.0f}",
                    "tỷ VND",
                ),
                (
                    "x_H tối thiểu",
                    f"{final_threshold:,.2f}",
                    "tỷ VND",
                ),
                (
                    "Risk",
                    f"{100*parameters['Risk'][sector_index]:.1f}%",
                    "rủi ro tự động hóa",
                ),
            ]
        )

        st.dataframe(
            threshold_table.style.format(
                {"x_H tối thiểu": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            f"Với x_AI = {assumed_ai:,.0f} tỷ VND tại ngành "
            f"**{selected_sector}**, cần x_H ít nhất khoảng "
            f"**{final_threshold:,.2f} tỷ VND** để đồng thời không tạo mất việc ròng "
            "và đủ năng lực đào tạo lại."
        )

    # -----------------------------------------------------
    # 9.4.3
    # -----------------------------------------------------
    with tab943:
        st.markdown(
            "### Câu 9.4.3. Trực quan nhóm lao động dễ tổn thương"
        )

        vulnerable_indices = [
            0,  # Nông nghiệp
            1,  # Chế biến chế tạo
            3,  # Bán buôn bán lẻ
        ]

        vulnerable_table = base_table.iloc[
            vulnerable_indices
        ].copy()

        st.dataframe(
            vulnerable_table,
            use_container_width=True,
            hide_index=True,
        )

        labels = [
            "Đầu tư AI",
            "Đào tạo lại",
        ]

        labels += vulnerable_table[
            "Ngành"
        ].tolist()

        labels += [
            "Việc làm mới",
            "Việc làm nâng cấp",
            "Việc làm bị thay thế",
        ]

        source = []
        target = []
        value = []

        first_sector_node = 2
        new_job_node = first_sector_node + len(
            vulnerable_indices
        )
        upgrade_node = new_job_node + 1
        displaced_node = new_job_node + 2

        for position, row in vulnerable_table.reset_index(
            drop=True
        ).iterrows():
            sector_node = (
                first_sector_node
                + position
            )

            # Luồng ngân sách vào ngành
            source.extend(
                [0, 1]
            )

            target.extend(
                [
                    sector_node,
                    sector_node,
                ]
            )

            value.extend(
                [
                    max(
                        float(row["x_AI"]),
                        0.001,
                    ),
                    max(
                        float(row["x_H"]),
                        0.001,
                    ),
                ]
            )

            # Luồng kết quả việc làm
            source.extend(
                [
                    sector_node,
                    sector_node,
                    sector_node,
                ]
            )

            target.extend(
                [
                    new_job_node,
                    upgrade_node,
                    displaced_node,
                ]
            )

            value.extend(
                [
                    max(
                        float(row["NewJob"]),
                        0.001,
                    ),
                    max(
                        float(row["UpgradeJob"]),
                        0.001,
                    ),
                    max(
                        float(row["DisplacedJob"]),
                        0.001,
                    ),
                ]
            )

        sankey_figure = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        label=labels,
                        pad=15,
                        thickness=18,
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                    ),
                )
            ]
        )

        sankey_figure.update_layout(
            title=(
                "Luồng đầu tư và dịch chuyển việc làm "
                "của nhóm dễ tổn thương"
            ),
            height=620,
        )

        st.plotly_chart(
            sankey_figure,
            use_container_width=True,
        )

        vulnerability_score = pd.DataFrame(
            {
                "Ngành": parameters["Ngành"],
                "Risk (%)": parameters["Risk"] * 100,
                "Lao động (triệu)": parameters["Lao động (triệu)"],
            }
        )

        vulnerability_score[
            "Chỉ số dễ tổn thương"
        ] = (
            vulnerability_score["Risk (%)"]
            * vulnerability_score[
                "Lao động (triệu)"
            ]
        )

        vulnerability_score = (
            vulnerability_score.sort_values(
                "Chỉ số dễ tổn thương",
                ascending=False,
            )
        )

        st.markdown(
            "#### Xếp hạng mức dễ tổn thương"
        )

        st.dataframe(
            vulnerability_score,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # 9.4.4
    # -----------------------------------------------------
    with tab944:
        st.markdown(
            "### Câu 9.4.4. Thêm giới hạn mất việc tối đa 5% lao động ngành"
        )

        limited_result, limited_parameters = _b9_solve(
            add_5pct_limit=True,
            budget=30000.0,
            cap_per_sector=6000.0,
        )

        if not limited_result.success:
            st.error(
                "Bài toán không khả thi sau khi thêm giới hạn mất việc 5%."
            )
        else:
            limited_table = _b9_result_table(
                limited_result,
                limited_parameters,
            )

            displaced_limit = (
                0.05
                * limited_parameters[
                    "Lao động (triệu)"
                ]
                * 1_000_000
            )

            limited_table[
                "Giới hạn mất việc 5%"
            ] = displaced_limit

            limited_table[
                "Sử dụng giới hạn (%)"
            ] = (
                100
                * limited_table[
                    "DisplacedJob"
                ]
                / np.maximum(
                    displaced_limit,
                    1e-12,
                )
            )

            comparison_5pct = pd.DataFrame(
                {
                    "Chỉ tiêu": [
                        "Tổng NetJob",
                        "Tổng x_AI",
                        "Tổng x_H",
                        "Tổng DisplacedJob",
                    ],
                    "Mô hình cơ sở": [
                        base_table["NetJob"].sum(),
                        base_table["x_AI"].sum(),
                        base_table["x_H"].sum(),
                        base_table["DisplacedJob"].sum(),
                    ],
                    "Giới hạn 5%": [
                        limited_table["NetJob"].sum(),
                        limited_table["x_AI"].sum(),
                        limited_table["x_H"].sum(),
                        limited_table["DisplacedJob"].sum(),
                    ],
                }
            )

            kpi_cards(
                [
                    (
                        "Trạng thái",
                        "Khả thi",
                        "giới hạn 5%",
                    ),
                    (
                        "NetJob mới",
                        f"{limited_table['NetJob'].sum():,.0f}",
                        "việc làm ròng",
                    ),
                    (
                        "x_AI mới",
                        f"{limited_table['x_AI'].sum():,.0f}",
                        "tỷ VND",
                    ),
                    (
                        "x_H mới",
                        f"{limited_table['x_H'].sum():,.0f}",
                        "tỷ VND",
                    ),
                ]
            )

            st.dataframe(
                comparison_5pct,
                use_container_width=True,
                hide_index=True,
            )

            st.dataframe(
                limited_table.style.format(
                    {
                        "DisplacedJob": "{:.0f}",
                        "Giới hạn mất việc 5%": "{:.0f}",
                        "Sử dụng giới hạn (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.success(
                "Bài toán vẫn khả thi sau khi giới hạn mất việc ở mức tối đa "
                "5% lực lượng lao động của từng ngành."
            )

    # =====================================================
    # Tải kết quả
    # =====================================================
    st.download_button(
        "Tải kết quả Bài 9 dạng CSV",
        data=base_table.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai9_tac_dong_ai_lao_dong.csv",
        mime="text/csv",
        key="download_bai9",
    )

    # =====================================================
    # 9.5. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 9.5. Câu hỏi thảo luận chính sách"
    )

    max_training_sector = base_table.loc[
        base_table["x_H"].idxmax(),
        "Ngành",
    ]

    with st.expander(
        "a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?",
        expanded=True,
    ):
        st.markdown(
            f"Ngành nhận đầu tư đào tạo lớn nhất trong nghiệm là "
            f"**{max_training_sector}**. Kết quả phụ thuộc vào rủi ro tự động hóa, "
            "hệ số dịch chuyển việc làm và hiệu quả đào tạo lại."
        )

    with st.expander(
        "b) Chiến lược phù hợp cho Tài chính-Ngân hàng là gì?",
        expanded=True,
    ):
        st.markdown(
            "Nên kết hợp đầu tư AI với tái thiết kế công việc, đào tạo kỹ năng dữ liệu, "
            "quản trị mô hình và chuyển lao động sang tư vấn, kiểm soát rủi ro, tuân thủ "
            "và chăm sóc khách hàng giá trị cao."
        )

    with st.expander(
        "c) Có nên đầu tư AI vào nông nghiệp không?",
        expanded=True,
    ):
        st.markdown(
            "Có thể đầu tư có chọn lọc vào truy xuất nguồn gốc, dự báo mùa vụ, quản lý "
            "đầu vào và logistics. Tuy nhiên nên ưu tiên công nghệ bổ trợ lao động thay vì "
            "thay thế hàng loạt, đồng thời tăng đào tạo và dịch vụ hỗ trợ nông hộ."
        )

    with st.expander(
        "d) Ràng buộc nào thể hiện tốc độ tự động hóa không vượt đào tạo lại?",
        expanded=True,
    ):
        st.markdown(
            "`DisplacedJobᵢ ≤ RetrainingCapacityᵢ` là ràng buộc trực tiếp. "
            "Có thể bổ sung trợ cấp chuyển việc, sàn thu nhập, giới hạn mất việc theo vùng "
            "và yêu cầu hoàn thành đào tạo trước khi doanh nghiệp triển khai tự động hóa."
        )


def _b10_parameters():
    """
    Tham số bài toán quy hoạch ngẫu nhiên hai giai đoạn.

    x_j:
        Quyết định đầu tư nền tảng trước khi biết kịch bản.

    y_j^s:
        Ngân sách điều chỉnh sau khi kịch bản s xảy ra.

    Đơn vị ngân sách: tỷ VND.
    """
    items = [
        "I - Hạ tầng số",
        "D - Chuyển đổi số",
        "AI - Trí tuệ nhân tạo",
        "H - Nhân lực số",
    ]

    scenarios = [
        "s1 - Lạc quan",
        "s2 - Cơ sở",
        "s3 - Bi quan",
        "s4 - Khủng hoảng",
    ]

    probabilities = np.array(
        [0.30, 0.45, 0.20, 0.05],
        dtype=float,
    )

    world_growth = np.array(
        [3.5, 2.8, 1.5, 0.2],
        dtype=float,
    )

    fdi_vietnam = np.array(
        [32, 27, 20, 12],
        dtype=float,
    )

    export_growth = np.array(
        [12, 8, 3, -5],
        dtype=float,
    )

    # Hệ số tác động của đầu tư nền tảng
    beta_first_stage = np.array(
        [1.00, 1.10, 1.25, 0.95],
        dtype=float,
    )

    # Hệ số tác động của ngân sách điều chỉnh theo kịch bản
    beta_recourse = np.array(
        [
            [1.25, 1.35, 1.55, 1.05],
            [1.00, 1.10, 1.25, 0.95],
            [0.75, 0.85, 0.90, 1.00],
            [0.40, 0.50, 0.55, 1.10],
        ],
        dtype=float,
    )

    return {
        "items": items,
        "scenarios": scenarios,
        "probabilities": probabilities,
        "world_growth": world_growth,
        "fdi_vietnam": fdi_vietnam,
        "export_growth": export_growth,
        "beta_first_stage": beta_first_stage,
        "beta_recourse": beta_recourse,
    }


def _b10_solve_stochastic_scipy(
    fixed_x=None,
):
    """
    Giải mô hình stochastic programming hai giai đoạn bằng SciPy/HiGHS.

    Biến:
    - x[0:4]: first-stage
    - y[s,j]: recourse, tổng 16 biến

    Tổng số biến: 20.
    """
    p = _b10_parameters()

    probability = p["probabilities"]
    beta_x = p["beta_first_stage"]
    beta_y = p["beta_recourse"]

    n_items = 4
    n_scenarios = 4
    n_variables = 20

    # linprog tối thiểu hóa nên đổi dấu hàm mục tiêu
    c = np.zeros(
        n_variables,
        dtype=float,
    )

    c[:n_items] = -beta_x

    for s in range(n_scenarios):
        start = n_items + s * n_items
        end = start + n_items

        c[start:end] = (
            -probability[s]
            * beta_y[s]
        )

    A_ub = []
    b_ub = []

    # C1. Ngân sách first-stage <= 65.000
    row = np.zeros(
        n_variables,
        dtype=float,
    )
    row[:n_items] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    # C2-C3. Mỗi kịch bản có recourse <=15.000
    # và y_AI^s <= 0.5*x_H
    for s in range(n_scenarios):
        start = n_items + s * n_items

        row = np.zeros(
            n_variables,
            dtype=float,
        )
        row[start:start + n_items] = 1.0
        A_ub.append(row)
        b_ub.append(15000.0)

        row = np.zeros(
            n_variables,
            dtype=float,
        )

        # y_AI là phần tử thứ 3 của mỗi vector y
        row[start + 2] = 1.0

        # x_H là phần tử thứ 4 của vector x
        row[3] = -0.5

        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = None
    b_eq = None

    if fixed_x is not None:
        fixed_x = np.asarray(
            fixed_x,
            dtype=float,
        )

        A_eq = np.zeros(
            (4, n_variables),
            dtype=float,
        )

        for j in range(4):
            A_eq[j, j] = 1.0

        b_eq = fixed_x

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
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[
            (0, None)
        ] * n_variables,
        method="highs",
    )

    return result


def _b10_solve_stochastic_pulp():
    """
    Giải cùng mô hình bằng PuLP/CBC để đúng yêu cầu Pyomo/PuLP.
    Nếu PuLP chưa cài, trả về None.
    """
    try:
        import pulp
    except ModuleNotFoundError:
        return None

    p = _b10_parameters()

    items = p["items"]
    scenarios = p["scenarios"]
    probability = p["probabilities"]
    beta_x = p["beta_first_stage"]
    beta_y = p["beta_recourse"]

    model = pulp.LpProblem(
        "Vietnam_Two_Stage_Stochastic_Programming",
        pulp.LpMaximize,
    )

    x = {
        j: pulp.LpVariable(
            f"x_{j}",
            lowBound=0,
        )
        for j in range(4)
    }

    y = {
        (s, j): pulp.LpVariable(
            f"y_{s}_{j}",
            lowBound=0,
        )
        for s in range(4)
        for j in range(4)
    }

    model += (
        pulp.lpSum(
            beta_x[j] * x[j]
            for j in range(4)
        )
        + pulp.lpSum(
            probability[s]
            * beta_y[s, j]
            * y[(s, j)]
            for s in range(4)
            for j in range(4)
        )
    ), "Expected_Benefit"

    model += (
        pulp.lpSum(
            x[j]
            for j in range(4)
        )
        <= 65000
    ), "C1_First_Stage_Budget"

    for s in range(4):
        model += (
            pulp.lpSum(
                y[(s, j)]
                for j in range(4)
            )
            <= 15000
        ), f"C2_Recourse_Budget_{s+1}"

        model += (
            y[(s, 2)]
            <= 0.5 * x[3]
        ), f"C3_AI_Depends_On_H_{s+1}"

    try:
        model.solve(
            pulp.PULP_CBC_CMD(
                msg=False
            )
        )
    except Exception:
        return None

    status = pulp.LpStatus[
        model.status
    ]

    if status != "Optimal":
        return {
            "status": status,
            "objective": np.nan,
            "x": None,
            "y": None,
        }

    x_value = np.array(
        [
            x[j].value()
            for j in range(4)
        ],
        dtype=float,
    )

    y_value = np.array(
        [
            [
                y[(s, j)].value()
                for j in range(4)
            ]
            for s in range(4)
        ],
        dtype=float,
    )

    return {
        "status": status,
        "objective": float(
            pulp.value(
                model.objective
            )
        ),
        "x": x_value,
        "y": y_value,
        "items": items,
        "scenarios": scenarios,
    }


def _b10_solve_expected_value():
    """
    Giải bài toán Expected Value (EV).

    Dùng hệ số recourse trung bình:
        beta_bar = sum_s p_s*beta_s

    Biến:
    - x[4]
    - y_bar[4]
    """
    p = _b10_parameters()

    probability = p["probabilities"]
    beta_x = p["beta_first_stage"]
    beta_y_bar = np.average(
        p["beta_recourse"],
        axis=0,
        weights=probability,
    )

    # x[4] + y[4]
    c = -np.r_[
        beta_x,
        beta_y_bar,
    ]

    A_ub = []
    b_ub = []

    # First-stage budget
    row = np.zeros(
        8,
        dtype=float,
    )
    row[:4] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    # Average recourse budget
    row = np.zeros(
        8,
        dtype=float,
    )
    row[4:] = 1.0
    A_ub.append(row)
    b_ub.append(15000.0)

    # y_AI <= 0.5*x_H
    row = np.zeros(
        8,
        dtype=float,
    )
    row[6] = 1.0
    row[3] = -0.5
    A_ub.append(row)
    b_ub.append(0.0)

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
        bounds=[
            (0, None)
        ] * 8,
        method="highs",
    )

    return result


def _b10_solve_deterministic_scenario(
    scenario_index,
):
    """
    Wait-and-see: giải riêng một kịch bản khi đã biết chắc kịch bản xảy ra.

    Biến:
    - x[4]
    - y[4]
    """
    p = _b10_parameters()

    beta_x = p["beta_first_stage"]
    beta_y = p["beta_recourse"][
        scenario_index
    ]

    c = -np.r_[
        beta_x,
        beta_y,
    ]

    A_ub = []
    b_ub = []

    row = np.zeros(
        8,
        dtype=float,
    )
    row[:4] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    row = np.zeros(
        8,
        dtype=float,
    )
    row[4:] = 1.0
    A_ub.append(row)
    b_ub.append(15000.0)

    row = np.zeros(
        8,
        dtype=float,
    )
    row[6] = 1.0
    row[3] = -0.5
    A_ub.append(row)
    b_ub.append(0.0)

    return linprog(
        c,
        A_ub=np.asarray(
            A_ub,
            dtype=float,
        ),
        b_ub=np.asarray(
            b_ub,
            dtype=float,
        ),
        bounds=[
            (0, None)
        ] * 8,
        method="highs",
    )


def _b10_calculate_metrics(
    stochastic_result,
    ev_result,
):
    """
    Tính:
    - RP: Recourse Problem
    - EEV: Expected result of using EV solution
    - VSS = RP - EEV
    - WS: Wait-and-see
    - EVPI = WS - RP
    """
    p = _b10_parameters()

    probability = p["probabilities"]

    rp = -float(
        stochastic_result.fun
    )

    x_ev = ev_result.x[:4]

    # Giữ x_EV cố định và tối ưu recourse theo từng kịch bản thật
    eev_result = _b10_solve_stochastic_scipy(
        fixed_x=x_ev
    )

    eev = (
        -float(
            eev_result.fun
        )
        if eev_result.success
        else np.nan
    )

    wait_and_see_values = []
    wait_and_see_x = []
    wait_and_see_y = []

    for s in range(4):
        scenario_result = (
            _b10_solve_deterministic_scenario(
                s
            )
        )

        wait_and_see_values.append(
            -float(
                scenario_result.fun
            )
        )

        wait_and_see_x.append(
            scenario_result.x[:4]
        )

        wait_and_see_y.append(
            scenario_result.x[4:]
        )

    ws = float(
        np.dot(
            probability,
            np.asarray(
                wait_and_see_values
            ),
        )
    )

    vss = rp - eev
    evpi = ws - rp

    return {
        "RP": rp,
        "EEV": eev,
        "VSS": vss,
        "WS": ws,
        "EVPI": evpi,
        "x_EV": x_ev,
        "EEV_result": eev_result,
        "WS_values": wait_and_see_values,
        "WS_x": np.asarray(
            wait_and_see_x
        ),
        "WS_y": np.asarray(
            wait_and_see_y
        ),
    }


def _b10_solve_robust():
    """
    Robust optimization dạng max-min.

    Biến:
    - x[4]
    - y[s,j] = 16
    - z = lợi ích nhỏ nhất giữa bốn kịch bản

    Mục tiêu:
        max z
    """
    p = _b10_parameters()

    beta_x = p["beta_first_stage"]
    beta_y = p["beta_recourse"]

    # Tổng 21 biến
    n_variables = 21
    z_index = 20

    c = np.zeros(
        n_variables,
        dtype=float,
    )

    # Max z tương đương Min -z
    c[z_index] = -1.0

    A_ub = []
    b_ub = []

    # First-stage budget
    row = np.zeros(
        n_variables,
        dtype=float,
    )
    row[:4] = 1.0
    A_ub.append(row)
    b_ub.append(65000.0)

    for s in range(4):
        start = 4 + s * 4

        # Recourse budget
        row = np.zeros(
            n_variables,
            dtype=float,
        )
        row[start:start + 4] = 1.0
        A_ub.append(row)
        b_ub.append(15000.0)

        # y_AI^s <= 0.5*x_H
        row = np.zeros(
            n_variables,
            dtype=float,
        )
        row[start + 2] = 1.0
        row[3] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)

        # z <= beta*x + beta_s*y_s
        row = np.zeros(
            n_variables,
            dtype=float,
        )
        row[:4] = -beta_x
        row[start:start + 4] = -beta_y[s]
        row[z_index] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

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
        bounds=(
            [(0, None)] * 20
            + [(None, None)]
        ),
        method="highs",
    )

    return result


def page_10():
    hero(
        "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
        "Trình bày đầy đủ các mục 10.1-10.6: first-stage, recourse, EV, SP, VSS, EVPI và robust optimization.",
        ["10.1-10.6", "Stochastic LP", "VSS", "EVPI", "Robust"],
    )

    p = _b10_parameters()

    items = p["items"]
    scenarios = p["scenarios"]
    probabilities = p["probabilities"]

    stochastic_result = (
        _b10_solve_stochastic_scipy()
    )

    if not stochastic_result.success:
        st.error(
            "Mô hình stochastic programming không khả thi."
        )
        return

    x_sp = stochastic_result.x[:4]

    y_sp = stochastic_result.x[
        4:
    ].reshape(
        4,
        4,
    )

    rp = -float(
        stochastic_result.fun
    )

    # =====================================================
    # 10.1. Bối cảnh Việt Nam
    # =====================================================
    st.markdown(
        "## 10.1. Bối cảnh Việt Nam"
    )

    st.markdown(
        """
        Việt Nam có độ mở thương mại cao nên hiệu quả đầu tư số chịu ảnh hưởng mạnh
        từ tăng trưởng thế giới, dòng vốn FDI và xuất khẩu.

        Chính phủ phải quyết định **65.000 tỷ VND đầu tư nền tảng** trước khi biết
        kịch bản kinh tế nào xảy ra. Sau đó, mỗi kịch bản cho phép sử dụng thêm
        tối đa **15.000 tỷ VND ngân sách điều chỉnh**.
        """
    )

    # =====================================================
    # 10.2. Cấu trúc kịch bản
    # =====================================================
    st.markdown(
        "## 10.2. Cấu trúc kịch bản"
    )

    scenario_table = pd.DataFrame(
        {
            "Kịch bản": scenarios,
            "Tăng trưởng thế giới (%)": p[
                "world_growth"
            ],
            "FDI Việt Nam (tỷ USD)": p[
                "fdi_vietnam"
            ],
            "Xuất khẩu tăng (%)": p[
                "export_growth"
            ],
            "Xác suất": probabilities,
        }
    )

    st.dataframe(
        scenario_table.style.format(
            {
                "Xác suất": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 10.3. Mô hình toán học
    # =====================================================
    st.markdown(
        "## 10.3. Mô hình toán học"
    )

    st.markdown(
        "### Biến quyết định giai đoạn 1"
    )

    st.latex(
        r"x_j="
        r"\text{đầu tư nền tảng trước khi biết kịch bản}"
    )

    st.markdown(
        "### Biến điều chỉnh giai đoạn 2"
    )

    st.latex(
        r"y_j^s="
        r"\text{ngân sách điều chỉnh sau khi kịch bản }s\text{ xảy ra}"
    )

    st.markdown(
        "### Hàm mục tiêu kỳ vọng"
    )

    st.latex(
        r"\max"
        r"\left["
        r"\sum_j\beta_jx_j"
        r"+\sum_sp_s"
        r"\sum_j\beta_j^sy_j^s"
        r"\right]"
    )

    st.markdown(
        "### Các ràng buộc"
    )

    st.latex(
        r"\sum_jx_j"
        r"\leq65{,}000"
    )

    st.latex(
        r"\sum_jy_j^s"
        r"\leq15{,}000,\quad\forall s"
    )

    st.latex(
        r"y_{AI}^s"
        r"\leq0.5x_H,\quad\forall s"
    )

    st.latex(
        r"x_j,y_j^s\geq0"
    )

    # =====================================================
    # 10.4. Hệ số theo kịch bản
    # =====================================================
    st.markdown(
        "## 10.4. Hệ số tác động theo kịch bản"
    )

    coefficient_table = pd.DataFrame(
        p["beta_recourse"],
        columns=items,
    )

    coefficient_table.insert(
        0,
        "Kịch bản",
        scenarios,
    )

    coefficient_table[
        "Xác suất"
    ] = probabilities

    st.dataframe(
        coefficient_table.style.format(
            {"Xác suất": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    base_coefficient_table = pd.DataFrame(
        {
            "Hạng mục": items,
            "β giai đoạn 1": p[
                "beta_first_stage"
            ],
        }
    )

    st.dataframe(
        base_coefficient_table,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 10.5. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 10.5. Yêu cầu lập trình"
    )

    tab1051, tab1052, tab1053, tab1054 = st.tabs(
        [
            "10.5.1 - SP PuLP/SciPy",
            "10.5.2 - EV & deterministic",
            "10.5.3 - VSS & EVPI",
            "10.5.4 - Robust",
        ]
    )

    # -----------------------------------------------------
    # 10.5.1
    # -----------------------------------------------------
    with tab1051:
        st.markdown(
            "### Câu 10.5.1. Cài đặt mô hình stochastic programming"
        )

        first_stage_table = pd.DataFrame(
            {
                "Hạng mục": items,
                "First-stage x": x_sp,
            }
        )

        recourse_table = pd.DataFrame(
            y_sp,
            columns=items,
        )

        recourse_table.insert(
            0,
            "Kịch bản",
            scenarios,
        )

        recourse_table[
            "Tổng recourse"
        ] = y_sp.sum(
            axis=1
        )

        kpi_cards(
            [
                (
                    "RP/SP objective",
                    f"{rp:,.2f}",
                    "lợi ích kỳ vọng",
                ),
                (
                    "Tổng first-stage",
                    f"{x_sp.sum():,.0f}",
                    "≤ 65.000",
                ),
                (
                    "x_H nền tảng",
                    f"{x_sp[3]:,.0f}",
                    "hỗ trợ AI recourse",
                ),
                (
                    "Hạng mục first-stage lớn nhất",
                    items[
                        int(
                            np.argmax(
                                x_sp
                            )
                        )
                    ],
                    f"{x_sp.max():,.0f}",
                ),
            ]
        )

        c1, c2 = st.columns(
            [0.8, 1.2]
        )

        with c1:
            st.markdown(
                "#### First-stage"
            )

            st.dataframe(
                first_stage_table,
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.markdown(
                "#### Recourse theo kịch bản"
            )

            st.dataframe(
                recourse_table,
                use_container_width=True,
                hide_index=True,
            )

        recourse_long = recourse_table.melt(
            id_vars=[
                "Kịch bản",
                "Tổng recourse",
            ],
            value_vars=items,
            var_name="Hạng mục",
            value_name="Ngân sách",
        )

        fig_recourse = px.bar(
            recourse_long,
            x="Kịch bản",
            y="Ngân sách",
            color="Hạng mục",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="Ngân sách điều chỉnh theo bốn kịch bản",
        )

        fig_recourse.update_layout(
            height=480,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_recourse,
            use_container_width=True,
        )

        pulp_result = (
            _b10_solve_stochastic_pulp()
        )

        if (
            pulp_result is not None
            and pulp_result["x"] is not None
        ):
            solver_comparison = pd.DataFrame(
                {
                    "Chỉ tiêu": [
                        "Objective",
                        "Tổng first-stage",
                        "Sai lệch x lớn nhất",
                        "Sai lệch y lớn nhất",
                    ],
                    "SciPy": [
                        rp,
                        x_sp.sum(),
                        0.0,
                        0.0,
                    ],
                    "PuLP": [
                        pulp_result[
                            "objective"
                        ],
                        pulp_result[
                            "x"
                        ].sum(),
                        np.max(
                            np.abs(
                                pulp_result[
                                    "x"
                                ]
                                - x_sp
                            )
                        ),
                        np.max(
                            np.abs(
                                pulp_result[
                                    "y"
                                ]
                                - y_sp
                            )
                        ),
                    ],
                }
            )

            st.markdown(
                "#### So sánh PuLP và SciPy"
            )

            st.dataframe(
                solver_comparison,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "PuLP/CBC chưa chạy được. Trang vẫn dùng SciPy/HiGHS. "
                "Kiểm tra `pulp>=2.7` trong requirements.txt."
            )

        with st.expander(
            "Xem mã PuLP rút gọn"
        ):
            st.code(
                """model = pulp.LpProblem(
    "TwoStageSP",
    pulp.LpMaximize
)

x = pulp.LpVariable.dicts(
    "x",
    range(4),
    lowBound=0
)

y = pulp.LpVariable.dicts(
    "y",
    (range(4), range(4)),
    lowBound=0
)

model += (
    pulp.lpSum(beta[j]*x[j] for j in range(4))
    + pulp.lpSum(
        p[s]*beta_s[s,j]*y[s][j]
        for s in range(4)
        for j in range(4)
    )
)

model.solve(
    pulp.PULP_CBC_CMD(msg=False)
)""",
                language="python",
            )

    # -----------------------------------------------------
    # 10.5.2
    # -----------------------------------------------------
    with tab1052:
        st.markdown(
            "### Câu 10.5.2. So sánh bốn lời giải xác định và lời giải EV"
        )

        deterministic_rows = []

        for s in range(4):
            scenario_result = (
                _b10_solve_deterministic_scenario(
                    s
                )
            )

            deterministic_rows.append(
                [
                    scenarios[s],
                    -float(
                        scenario_result.fun
                    ),
                    *scenario_result.x[:4],
                    *scenario_result.x[4:],
                ]
            )

        deterministic_table = pd.DataFrame(
            deterministic_rows,
            columns=[
                "Kịch bản",
                "Objective",
                "x_I",
                "x_D",
                "x_AI",
                "x_H",
                "y_I",
                "y_D",
                "y_AI",
                "y_H",
            ],
        )

        ev_result = (
            _b10_solve_expected_value()
        )

        if not ev_result.success:
            st.error(
                "Không giải được bài toán Expected Value."
            )
            return

        x_ev = ev_result.x[:4]
        y_ev = ev_result.x[4:]

        ev_table = pd.DataFrame(
            {
                "Hạng mục": items,
                "x_EV": x_ev,
                "y_EV trung bình": y_ev,
                "x_SP": x_sp,
                "Chênh lệch x_SP-x_EV": (
                    x_sp - x_ev
                ),
            }
        )

        st.markdown(
            "#### Bốn lời giải xác định"
        )

        st.dataframe(
            deterministic_table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### So sánh EV và SP"
        )

        st.dataframe(
            ev_table.style.format(
                {
                    "x_EV": "{:.2f}",
                    "y_EV trung bình": "{:.2f}",
                    "x_SP": "{:.2f}",
                    "Chênh lệch x_SP-x_EV": "{:+.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        comparison_long = ev_table.melt(
            id_vars="Hạng mục",
            value_vars=[
                "x_EV",
                "x_SP",
            ],
            var_name="Phương pháp",
            value_name="First-stage",
        )

        fig_ev_sp = px.bar(
            comparison_long,
            x="Hạng mục",
            y="First-stage",
            color="Phương pháp",
            barmode="group",
            template=PLOT_TEMPLATE,
            title="So sánh first-stage của EV và SP",
        )

        fig_ev_sp.update_layout(
            height=460,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_ev_sp,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 10.5.3
    # -----------------------------------------------------
    with tab1053:
        st.markdown(
            "### Câu 10.5.3. Tính VSS và EVPI"
        )

        ev_result = (
            _b10_solve_expected_value()
        )

        metrics = _b10_calculate_metrics(
            stochastic_result,
            ev_result,
        )

        metrics_table = pd.DataFrame(
            {
                "Chỉ tiêu": [
                    "RP - Recourse Problem",
                    "EEV - Expected result of EV",
                    "VSS = RP - EEV",
                    "WS - Wait and See",
                    "EVPI = WS - RP",
                ],
                "Giá trị": [
                    metrics["RP"],
                    metrics["EEV"],
                    metrics["VSS"],
                    metrics["WS"],
                    metrics["EVPI"],
                ],
            }
        )

        kpi_cards(
            [
                (
                    "VSS",
                    f"{metrics['VSS']:,.2f}",
                    "RP - EEV",
                ),
                (
                    "EVPI",
                    f"{metrics['EVPI']:,.2f}",
                    "WS - RP",
                ),
                (
                    "EEV",
                    f"{metrics['EEV']:,.2f}",
                    "dùng quyết định EV",
                ),
                (
                    "WS",
                    f"{metrics['WS']:,.2f}",
                    "thông tin hoàn hảo",
                ),
            ]
        )

        st.dataframe(
            metrics_table.style.format(
                {"Giá trị": "{:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        wait_and_see_table = pd.DataFrame(
            {
                "Kịch bản": scenarios,
                "Xác suất": probabilities,
                "Objective khi biết trước": metrics[
                    "WS_values"
                ],
                "Đóng góp vào WS": (
                    probabilities
                    * np.asarray(
                        metrics[
                            "WS_values"
                        ]
                    )
                ),
            }
        )

        st.dataframe(
            wait_and_see_table,
            use_container_width=True,
            hide_index=True,
        )

        if metrics["VSS"] >= -1e-6:
            st.success(
                "VSS không âm: việc mô hình hóa bất định tạo giá trị "
                "so với chỉ sử dụng kịch bản trung bình."
            )
        else:
            st.warning(
                "VSS âm do sai số số học hoặc thiết lập EV chưa nhất quán; "
                "cần kiểm tra lại mô hình."
            )

        st.info(
            "EVPI là mức tối đa hợp lý mà nhà hoạch định có thể trả "
            "để có thông tin hoàn hảo trước khi ra quyết định."
        )

    # -----------------------------------------------------
    # 10.5.4
    # -----------------------------------------------------
    with tab1054:
        st.markdown(
            "### Câu 10.5.4. Robust optimization dạng max-min"
        )

        robust_result = (
            _b10_solve_robust()
        )

        if not robust_result.success:
            st.error(
                "Mô hình robust không khả thi."
            )
        else:
            x_robust = robust_result.x[:4]

            y_robust = robust_result.x[
                4:20
            ].reshape(
                4,
                4,
            )

            worst_case_value = float(
                robust_result.x[20]
            )

            robust_first_stage = pd.DataFrame(
                {
                    "Hạng mục": items,
                    "SP kỳ vọng": x_sp,
                    "Robust": x_robust,
                    "Thay đổi": (
                        x_robust - x_sp
                    ),
                }
            )

            robust_scenario_values = []

            for s in range(4):
                scenario_value = float(
                    p[
                        "beta_first_stage"
                    ] @ x_robust
                    + p[
                        "beta_recourse"
                    ][s] @ y_robust[s]
                )

                robust_scenario_values.append(
                    scenario_value
                )

            robust_scenario_table = pd.DataFrame(
                {
                    "Kịch bản": scenarios,
                    "Lợi ích robust": robust_scenario_values,
                    "Recourse": y_robust.sum(
                        axis=1
                    ),
                }
            )

            kpi_cards(
                [
                    (
                        "Worst-case objective",
                        f"{worst_case_value:,.2f}",
                        "max-min",
                    ),
                    (
                        "Robust ưu tiên",
                        items[
                            int(
                                np.argmax(
                                    x_robust
                                )
                            )
                        ],
                        f"{x_robust.max():,.0f}",
                    ),
                    (
                        "Tổng first-stage",
                        f"{x_robust.sum():,.0f}",
                        "≤ 65.000",
                    ),
                    (
                        "Khác SP",
                        f"{np.abs(x_robust-x_sp).sum():,.0f}",
                        "khoảng cách L1",
                    ),
                ]
            )

            st.dataframe(
                robust_first_stage.style.format(
                    {
                        "SP kỳ vọng": "{:.2f}",
                        "Robust": "{:.2f}",
                        "Thay đổi": "{:+.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.dataframe(
                robust_scenario_table,
                use_container_width=True,
                hide_index=True,
            )

            robust_long = robust_first_stage.melt(
                id_vars="Hạng mục",
                value_vars=[
                    "SP kỳ vọng",
                    "Robust",
                ],
                var_name="Mô hình",
                value_name="First-stage",
            )

            fig_robust = px.bar(
                robust_long,
                x="Hạng mục",
                y="First-stage",
                color="Mô hình",
                barmode="group",
                template=PLOT_TEMPLATE,
                title="So sánh SP kỳ vọng và robust max-min",
            )

            fig_robust.update_layout(
                height=470,
                margin=dict(
                    l=10,
                    r=10,
                    t=54,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig_robust,
                use_container_width=True,
            )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_first_stage = pd.DataFrame(
        {
            "Hạng mục": items,
            "First-stage_SP": x_sp,
        }
    )

    st.download_button(
        "Tải kết quả first-stage Bài 10",
        data=export_first_stage.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai10_first_stage_stochastic.csv",
        mime="text/csv",
        key="download_bai10",
    )

    # =====================================================
    # 10.6. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 10.6. Câu hỏi thảo luận chính sách"
    )

    with st.expander(
        "a) SP đầu tư H nhiều hơn hay ít hơn lời giải xác định? Vì sao?",
        expanded=True,
    ):
        st.markdown(
            "Nhân lực số có vai trò như một tài sản linh hoạt: H vừa tạo lợi ích trực tiếp "
            "vừa mở rộng khả năng bổ sung AI trong giai đoạn recourse. Vì vậy, SP có thể duy trì "
            "x_H cao hơn lời giải chỉ tối ưu cho một kịch bản thuận lợi."
        )

    with st.expander(
        "b) VSS dương nói lên điều gì?",
        expanded=True,
    ):
        st.markdown(
            "VSS dương chứng minh việc sử dụng phân phối xác suất và cơ chế điều chỉnh "
            "tạo ra quyết định tốt hơn so với chỉ tối ưu theo kịch bản trung bình."
        )

    with st.expander(
        "c) Việt Nam có đang dưới đầu tư vào nhân lực số không?",
        expanded=True,
    ):
        st.markdown(
            "Nếu SP và robust đều phân bổ nhiều hơn cho H so với EV hoặc lời giải lạc quan, "
            "đó là bằng chứng trong mô hình rằng nhân lực số có giá trị chống chịu trước cú sốc, "
            "ngoài tác động tăng trưởng trực tiếp."
        )


_B11_ACTIONS = {
    0: (
        "a0 - Truyền thống",
        np.array(
            [0.70, 0.10, 0.10, 0.10],
            dtype=float,
        ),
    ),
    1: (
        "a1 - Cân bằng",
        np.array(
            [0.40, 0.25, 0.15, 0.20],
            dtype=float,
        ),
    ),
    2: (
        "a2 - Số hóa nhanh",
        np.array(
            [0.25, 0.45, 0.15, 0.15],
            dtype=float,
        ),
    ),
    3: (
        "a3 - AI dẫn dắt",
        np.array(
            [0.20, 0.20, 0.45, 0.15],
            dtype=float,
        ),
    ),
    4: (
        "a4 - Bao trùm",
        np.array(
            [0.30, 0.20, 0.10, 0.40],
            dtype=float,
        ),
    ),
}


_B11_STATE_LABELS = {
    0: "Thấp",
    1: "Trung bình",
    2: "Cao",
}


def _b11_action_table():
    """
    Bảng năm hành động phân bổ ngân sách.
    Thứ tự tỷ trọng: K, D, AI, H.
    """
    rows = []

    for action_id, (
        action_name,
        shares,
    ) in _B11_ACTIONS.items():
        rows.append(
            [
                action_id,
                action_name,
                shares[0],
                shares[1],
                shares[2],
                shares[3],
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Mã",
            "Hành động",
            "K",
            "D",
            "AI",
            "H",
        ],
    )


def _b11_transition(
    state,
    action,
    rng,
):
    """
    Hàm chuyển trạng thái mô phỏng nền kinh tế.

    State:
    - g: tăng trưởng GDP, 0-2
    - d: mức độ số hóa, 0-2
    - ai: năng lực AI, 0-2
    - u: rủi ro thất nghiệp, 0-2

    Action:
    - 0 đến 4, tương ứng năm cơ cấu ngân sách.
    """
    g, d, ai, u = np.asarray(
        state,
        dtype=int,
    )

    shares = _B11_ACTIONS[
        int(action)
    ][1]

    share_k = float(
        shares[0]
    )
    share_d = float(
        shares[1]
    )
    share_ai = float(
        shares[2]
    )
    share_h = float(
        shares[3]
    )

    # Tăng trưởng kỳ vọng từ cơ cấu đầu tư và trạng thái hiện tại
    growth_effect = (
        0.18 * share_k
        + 0.34 * share_d
        + 0.50 * share_ai
        + 0.28 * share_h
        + 0.025 * g
        + 0.020 * d
        + 0.018 * ai
        + rng.normal(
            0.0,
            0.018,
        )
    )

    # Dương nghĩa là rủi ro thất nghiệp tăng
    unemployment_effect = (
        0.46 * share_ai
        - 0.62 * share_h
        - 0.12 * share_d
        + 0.025 * ai
        - 0.015 * d
        + rng.normal(
            0.0,
            0.015,
        )
    )

    cyber_risk = max(
        0.0,
        0.52 * share_ai
        + 0.22 * share_d
        - 0.28 * share_h
        + 0.025 * ai,
    )

    emission_risk = max(
        0.0,
        0.38 * share_k
        + 0.30 * share_ai
        - 0.10 * share_d
        - 0.06 * share_h,
    )

    inclusion_gain = (
        0.55 * share_h
        + 0.20 * share_d
        - 0.18 * share_ai
    )

    reward = (
        0.42 * growth_effect
        - 0.24 * max(
            unemployment_effect,
            0.0,
        )
        - 0.17 * cyber_risk
        - 0.12 * emission_risk
        + 0.05 * inclusion_gain
    )

    next_g = int(
        np.clip(
            g
            + int(
                growth_effect > 0.31
            )
            - int(
                growth_effect < 0.19
            ),
            0,
            2,
        )
    )

    next_d = int(
        np.clip(
            d
            + int(
                share_d >= 0.30
            )
            - int(
                share_d <= 0.10
            ),
            0,
            2,
        )
    )

    next_ai = int(
        np.clip(
            ai
            + int(
                share_ai >= 0.30
            )
            - int(
                share_ai <= 0.10
            ),
            0,
            2,
        )
    )

    next_u = int(
        np.clip(
            u
            + int(
                unemployment_effect > 0.08
            )
            - int(
                unemployment_effect < -0.05
            ),
            0,
            2,
        )
    )

    next_state = np.array(
        [
            next_g,
            next_d,
            next_ai,
            next_u,
        ],
        dtype=int,
    )

    components = {
        "growth_effect": float(
            growth_effect
        ),
        "unemployment_effect": float(
            unemployment_effect
        ),
        "cyber_risk": float(
            cyber_risk
        ),
        "emission_risk": float(
            emission_risk
        ),
        "inclusion_gain": float(
            inclusion_gain
        ),
    }

    return (
        next_state,
        float(
            reward
        ),
        components,
    )


class _B11SimpleEnv:
    """
    Môi trường MDP tối giản, không bắt buộc cài Gymnasium.
    """

    def __init__(
        self,
        horizon=10,
        seed=42,
    ):
        self.horizon = int(
            horizon
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.state = None
        self.t = 0

    def reset(
        self,
        initial_state=None,
    ):
        if initial_state is None:
            initial_state = np.array(
                [1, 1, 0, 1],
                dtype=int,
            )

        self.state = np.asarray(
            initial_state,
            dtype=int,
        ).copy()

        self.t = 0

        return self.state.copy()

    def step(
        self,
        action,
    ):
        (
            next_state,
            reward,
            components,
        ) = _b11_transition(
            self.state,
            action,
            self.rng,
        )

        self.state = next_state
        self.t += 1

        done = (
            self.t >= self.horizon
        )

        return (
            next_state.copy(),
            reward,
            done,
            components,
        )


@st.cache_data
def _b11_train_q_learning(
    episodes=10000,
    learning_rate=0.10,
    discount_factor=0.95,
    seed=42,
):
    """
    Huấn luyện Q-learning tabular.

    Q có kích thước:
        3 × 3 × 3 × 3 × 5
    tương ứng 81 trạng thái và 5 hành động.
    """
    rng = np.random.default_rng(
        seed
    )

    q_table = np.zeros(
        (
            3,
            3,
            3,
            3,
            5,
        ),
        dtype=float,
    )

    reward_history = []
    epsilon_history = []

    for episode in range(
        int(
            episodes
        )
    ):
        state = np.array(
            [1, 1, 0, 1],
            dtype=int,
        )

        total_reward = 0.0

        epsilon = max(
            0.05,
            1.0
            - episode
            / max(
                episodes * 0.65,
                1,
            ),
        )

        for _ in range(
            10
        ):
            state_key = tuple(
                state
            )

            if rng.random() < epsilon:
                action = int(
                    rng.integers(
                        0,
                        5,
                    )
                )
            else:
                action = int(
                    np.argmax(
                        q_table[
                            state_key
                        ]
                    )
                )

            (
                next_state,
                reward,
                _,
            ) = _b11_transition(
                state,
                action,
                rng,
            )

            next_state_key = tuple(
                next_state
            )

            old_value = q_table[
                state_key
                + (
                    action,
                )
            ]

            target = (
                reward
                + discount_factor
                * np.max(
                    q_table[
                        next_state_key
                    ]
                )
            )

            q_table[
                state_key
                + (
                    action,
                )
            ] = (
                old_value
                + learning_rate
                * (
                    target
                    - old_value
                )
            )

            state = next_state
            total_reward += reward

        reward_history.append(
            total_reward
        )

        epsilon_history.append(
            epsilon
        )

    learning_curve = pd.DataFrame(
        {
            "Episode": np.arange(
                1,
                episodes + 1,
            ),
            "Reward": reward_history,
            "Epsilon": epsilon_history,
        }
    )

    rolling_window = min(
        200,
        max(
            20,
            episodes // 20,
        ),
    )

    learning_curve[
        "Reward_MA"
    ] = (
        learning_curve[
            "Reward"
        ]
        .rolling(
            rolling_window,
            min_periods=1,
        )
        .mean()
    )

    return (
        q_table,
        learning_curve,
    )


def _b11_evaluate_policy(
    q_table=None,
    fixed_action=None,
    random_policy=False,
    episodes=500,
    seed=7,
):
    """
    Đánh giá chính sách trên các mô phỏng ngoài mẫu.
    """
    rng = np.random.default_rng(
        seed
    )

    total_rewards = []
    ending_states = []

    for _ in range(
        int(
            episodes
        )
    ):
        state = np.array(
            [1, 1, 0, 1],
            dtype=int,
        )

        episode_reward = 0.0

        for _ in range(
            10
        ):
            if random_policy:
                action = int(
                    rng.integers(
                        0,
                        5,
                    )
                )
            elif fixed_action is not None:
                action = int(
                    fixed_action
                )
            else:
                action = int(
                    np.argmax(
                        q_table[
                            tuple(
                                state
                            )
                        ]
                    )
                )

            (
                state,
                reward,
                _,
            ) = _b11_transition(
                state,
                action,
                rng,
            )

            episode_reward += reward

        total_rewards.append(
            episode_reward
        )

        ending_states.append(
            state.copy()
        )

    ending_states = np.asarray(
        ending_states,
        dtype=float,
    )

    return {
        "mean_reward": float(
            np.mean(
                total_rewards
            )
        ),
        "std_reward": float(
            np.std(
                total_rewards
            )
        ),
        "mean_final_g": float(
            ending_states[:, 0].mean()
        ),
        "mean_final_d": float(
            ending_states[:, 1].mean()
        ),
        "mean_final_ai": float(
            ending_states[:, 2].mean()
        ),
        "mean_final_u": float(
            ending_states[:, 3].mean()
        ),
    }


def _b11_representative_policy(
    q_table,
):
    """
    Trích xuất chính sách tại một số trạng thái đại diện.
    """
    representative_states = {
        "Việt Nam 2026": (
            1,
            1,
            0,
            1,
        ),
        "GDP thấp, số hóa thấp, thất nghiệp cao": (
            0,
            0,
            0,
            2,
        ),
        "GDP cao, AI cao, thất nghiệp thấp": (
            2,
            2,
            2,
            0,
        ),
        "Số hóa cao nhưng AI thấp": (
            1,
            2,
            0,
            1,
        ),
        "Rủi ro thất nghiệp cao": (
            1,
            1,
            1,
            2,
        ),
        "GDP thấp nhưng năng lực AI cao": (
            0,
            1,
            2,
            1,
        ),
    }

    rows = []

    for state_name, state in (
        representative_states.items()
    ):
        q_values = q_table[
            state
        ]

        best_action = int(
            np.argmax(
                q_values
            )
        )

        rows.append(
            [
                state_name,
                str(
                    state
                ),
                _B11_ACTIONS[
                    best_action
                ][0],
                float(
                    q_values[
                        best_action
                    ]
                ),
                float(
                    np.max(
                        q_values
                    )
                    - np.partition(
                        q_values,
                        -2,
                    )[-2]
                ),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Trạng thái",
            "Mã hóa",
            "Hành động π*",
            "Q-value",
            "Khoảng cách Q tốt nhất-thứ hai",
        ],
    )


def page_11():
    hero(
        "Bài 11 — Học tăng cường Q-learning cho chính sách kinh tế thích nghi",
        "Trình bày đầy đủ các mục 11.1-11.4: môi trường MDP, Q-learning 10.000 episodes, chính sách π*, so sánh rule-based và mở rộng DQN.",
        ["11.1-11.4", "Q-learning", "MDP", "Adaptive policy", "DQN"],
    )

    # =====================================================
    # 11.1. Bối cảnh
    # =====================================================
    st.markdown(
        "## 11.1. Bối cảnh"
    )

    st.markdown(
        """
        Các mô hình LP trước đó giả định tham số đã biết và quyết định được đưa ra một lần.
        Trong thực tế, Chính phủ cần điều chỉnh cơ cấu đầu tư khi tăng trưởng, mức độ số hóa,
        năng lực AI và rủi ro thất nghiệp thay đổi.

        Bài 11 mô hình hóa nền kinh tế như một **Markov Decision Process (MDP)**.
        Tác nhân học cách lựa chọn cơ cấu ngân sách theo trạng thái kinh tế để tối đa hóa
        tổng phần thưởng trong 10 giai đoạn.
        """
    )

    # =====================================================
    # 11.2. Mô hình MDP
    # =====================================================
    st.markdown(
        "## 11.2. Mô hình Markov Decision Process"
    )

    st.markdown(
        "### Không gian trạng thái"
    )

    state_table = pd.DataFrame(
        {
            "Thành phần": [
                "g - Tăng trưởng GDP",
                "d - Mức độ số hóa",
                "ai - Năng lực AI",
                "u - Rủi ro thất nghiệp",
            ],
            "Mức 0": [
                "Thấp",
                "Thấp",
                "Thấp",
                "Thấp",
            ],
            "Mức 1": [
                "Trung bình",
                "Trung bình",
                "Trung bình",
                "Trung bình",
            ],
            "Mức 2": [
                "Cao",
                "Cao",
                "Cao",
                "Cao",
            ],
        }
    )

    st.dataframe(
        state_table,
        use_container_width=True,
        hide_index=True,
    )

    st.latex(
        r"s_t="
        r"(g_t,d_t,ai_t,u_t),"
        r"\quad"
        r"g_t,d_t,ai_t,u_t\in\{0,1,2\}"
    )

    st.markdown(
        "### Không gian hành động"
    )

    action_table = (
        _b11_action_table()
    )

    st.dataframe(
        action_table.style.format(
            {
                "K": "{:.0%}",
                "D": "{:.0%}",
                "AI": "{:.0%}",
                "H": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Hàm phần thưởng"
    )

    st.latex(
        r"R_t="
        r"0.42\Delta GDP_t"
        r"-0.24\max(\Delta U_t,0)"
        r"-0.17CyberRisk_t"
        r"-0.12Emission_t"
        r"+0.05Inclusion_t"
    )

    st.markdown(
        "### Phương trình cập nhật Q-learning"
    )

    st.latex(
        r"Q(s_t,a_t)"
        r"\leftarrow"
        r"Q(s_t,a_t)"
        r"+\alpha"
        r"\left["
        r"R_t+\gamma\max_aQ(s_{t+1},a)"
        r"-Q(s_t,a_t)"
        r"\right]"
    )

    # Tham số huấn luyện
    c1, c2, c3 = st.columns(
        3
    )

    episodes = c1.select_slider(
        "Số episode",
        options=[
            1000,
            2000,
            4000,
            6000,
            8000,
            10000,
        ],
        value=10000,
        key="b11_episodes",
    )

    learning_rate = c2.slider(
        "Learning rate α",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.05,
        key="b11_alpha",
    )

    discount_factor = c3.slider(
        "Discount factor γ",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        key="b11_gamma",
    )

    (
        q_table,
        learning_curve,
    ) = _b11_train_q_learning(
        episodes=episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        seed=42,
    )

    # =====================================================
    # 11.3. Yêu cầu lập trình
    # =====================================================
    st.markdown(
        "## 11.3. Yêu cầu lập trình"
    )

    (
        tab1131,
        tab1132,
        tab1133,
        tab1134,
        tab1135,
    ) = st.tabs(
        [
            "11.3.1 - Environment",
            "11.3.2 - Q-learning",
            "11.3.3 - Chính sách π*",
            "11.3.4 - So sánh",
            "11.3.5 - DQN",
        ]
    )

    # -----------------------------------------------------
    # 11.3.1
    # -----------------------------------------------------
    with tab1131:
        st.markdown(
            "### Câu 11.3.1. Xây dựng môi trường MDP"
        )

        demo_env = _B11SimpleEnv(
            horizon=10,
            seed=123,
        )

        initial_state = demo_env.reset(
            initial_state=np.array(
                [1, 1, 0, 1],
                dtype=int,
            )
        )

        demo_rows = []

        demo_state = (
            initial_state.copy()
        )

        for period in range(
            1,
            6,
        ):
            demo_action = int(
                np.argmax(
                    q_table[
                        tuple(
                            demo_state
                        )
                    ]
                )
            )

            (
                next_state,
                reward,
                done,
                components,
            ) = demo_env.step(
                demo_action
            )

            demo_rows.append(
                [
                    period,
                    str(
                        tuple(
                            demo_state
                        )
                    ),
                    _B11_ACTIONS[
                        demo_action
                    ][0],
                    reward,
                    components[
                        "growth_effect"
                    ],
                    components[
                        "unemployment_effect"
                    ],
                    components[
                        "cyber_risk"
                    ],
                    components[
                        "emission_risk"
                    ],
                    str(
                        tuple(
                            next_state
                        )
                    ),
                ]
            )

            demo_state = (
                next_state.copy()
            )

            if done:
                break

        demo_table = pd.DataFrame(
            demo_rows,
            columns=[
                "Giai đoạn",
                "State",
                "Action",
                "Reward",
                "Growth effect",
                "Unemployment effect",
                "Cyber risk",
                "Emission risk",
                "Next state",
            ],
        )

        st.dataframe(
            demo_table.style.format(
                {
                    "Reward": "{:.4f}",
                    "Growth effect": "{:.4f}",
                    "Unemployment effect": "{:.4f}",
                    "Cyber risk": "{:.4f}",
                    "Emission risk": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Môi trường trong dashboard được viết trực tiếp bằng Python nên không bắt buộc cài Gymnasium."
        )

        with st.expander(
            "Xem khung môi trường Gymnasium"
        ):
            st.code(
                """import gymnasium as gym
from gymnasium import spaces

class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete(
            [3, 3, 3, 3]
        )
        self.horizon = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1, 1, 0, 1])
        self.t = 0
        return self.state, {}

    def step(self, action):
        next_state, reward, info = transition(
            self.state,
            action
        )
        self.state = next_state
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False
        return (
            self.state,
            reward,
            terminated,
            truncated,
            info
        )""",
                language="python",
            )

    # -----------------------------------------------------
    # 11.3.2
    # -----------------------------------------------------
    with tab1132:
        st.markdown(
            "### Câu 11.3.2. Huấn luyện Q-learning"
        )

        early_reward = float(
            learning_curve[
                "Reward_MA"
            ]
            .head(
                min(
                    500,
                    len(
                        learning_curve
                    ),
                )
            )
            .mean()
        )

        final_reward = float(
            learning_curve[
                "Reward_MA"
            ]
            .tail(
                min(
                    500,
                    len(
                        learning_curve
                    ),
                )
            )
            .mean()
        )

        kpi_cards(
            [
                (
                    "Episodes",
                    f"{episodes:,}",
                    "10 bước/episode",
                ),
                (
                    "Learning rate",
                    f"{learning_rate:.2f}",
                    "α",
                ),
                (
                    "Discount",
                    f"{discount_factor:.2f}",
                    "γ",
                ),
                (
                    "Reward cải thiện",
                    f"{final_reward-early_reward:+.4f}",
                    "cuối kỳ so với đầu kỳ",
                ),
            ]
        )

        fig_learning = px.line(
            learning_curve,
            x="Episode",
            y="Reward_MA",
            template=PLOT_TEMPLATE,
            title="Learning curve — Reward trung bình trượt",
        )

        fig_learning.update_layout(
            height=480,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_learning,
            use_container_width=True,
        )

        epsilon_sample = (
            learning_curve.iloc[
                ::max(
                    episodes // 20,
                    1,
                )
            ][
                [
                    "Episode",
                    "Epsilon",
                ]
            ]
        )

        fig_epsilon = px.line(
            epsilon_sample,
            x="Episode",
            y="Epsilon",
            markers=True,
            template=PLOT_TEMPLATE,
            title="Lịch giảm epsilon",
        )

        fig_epsilon.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            fig_epsilon,
            use_container_width=True,
        )

        with st.expander(
            "Xem mã Q-learning rút gọn"
        ):
            st.code(
                """Q = np.zeros((3,3,3,3,5))

for episode in range(10000):
    state = np.array([1,1,0,1])
    epsilon = max(
        0.05,
        1 - episode/(10000*0.65)
    )

    for t in range(10):
        if np.random.rand() < epsilon:
            action = np.random.randint(5)
        else:
            action = np.argmax(Q[tuple(state)])

        next_state, reward = env_step(
            state,
            action
        )

        old = Q[tuple(state)+(action,)]
        target = reward + 0.95*np.max(
            Q[tuple(next_state)]
        )

        Q[tuple(state)+(action,)] = (
            old + 0.10*(target-old)
        )

        state = next_state""",
                language="python",
            )

    # -----------------------------------------------------
    # 11.3.3
    # -----------------------------------------------------
    with tab1133:
        st.markdown(
            "### Câu 11.3.3. Trích xuất chính sách tối ưu π*(s)"
        )

        policy_table = (
            _b11_representative_policy(
                q_table
            )
        )

        st.dataframe(
            policy_table.style.format(
                {
                    "Q-value": "{:.4f}",
                    "Khoảng cách Q tốt nhất-thứ hai": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Khuyến nghị theo trạng thái do người dùng chọn"
        )

        s1, s2, s3, s4 = st.columns(
            4
        )

        selected_g = s1.selectbox(
            "GDP growth",
            options=[
                0,
                1,
                2,
            ],
            format_func=lambda x: _B11_STATE_LABELS[
                x
            ],
            index=1,
            key="b11_state_g",
        )

        selected_d = s2.selectbox(
            "Digital index",
            options=[
                0,
                1,
                2,
            ],
            format_func=lambda x: _B11_STATE_LABELS[
                x
            ],
            index=1,
            key="b11_state_d",
        )

        selected_ai = s3.selectbox(
            "AI capacity",
            options=[
                0,
                1,
                2,
            ],
            format_func=lambda x: _B11_STATE_LABELS[
                x
            ],
            index=0,
            key="b11_state_ai",
        )

        selected_u = s4.selectbox(
            "Unemployment risk",
            options=[
                0,
                1,
                2,
            ],
            format_func=lambda x: _B11_STATE_LABELS[
                x
            ],
            index=1,
            key="b11_state_u",
        )

        selected_state = (
            selected_g,
            selected_d,
            selected_ai,
            selected_u,
        )

        selected_q = q_table[
            selected_state
        ]

        recommended_action = int(
            np.argmax(
                selected_q
            )
        )

        recommended_name = (
            _B11_ACTIONS[
                recommended_action
            ][0]
        )

        recommended_shares = (
            _B11_ACTIONS[
                recommended_action
            ][1]
        )

        kpi_cards(
            [
                (
                    "Hành động π*",
                    recommended_name,
                    f"state={selected_state}",
                ),
                (
                    "K",
                    f"{recommended_shares[0]:.0%}",
                    "vốn vật chất",
                ),
                (
                    "D",
                    f"{recommended_shares[1]:.0%}",
                    "chuyển đổi số",
                ),
                (
                    "AI và H",
                    f"{recommended_shares[2]:.0%} / {recommended_shares[3]:.0%}",
                    "AI / nhân lực",
                ),
            ]
        )

        q_value_table = pd.DataFrame(
            {
                "Hành động": [
                    _B11_ACTIONS[
                        action
                    ][0]
                    for action in range(
                        5
                    )
                ],
                "Q-value": selected_q,
            }
        ).sort_values(
            "Q-value",
            ascending=False,
        )

        st.dataframe(
            q_value_table.style.format(
                {"Q-value": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            plot_bar(
                q_value_table,
                "Hành động",
                "Q-value",
                "Q-value theo hành động tại trạng thái đã chọn",
                text="Q-value",
            ),
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 11.3.4
    # -----------------------------------------------------
    with tab1134:
        st.markdown(
            "### Câu 11.3.4. So sánh Q-learning với chính sách rule-based"
        )

        q_result = (
            _b11_evaluate_policy(
                q_table=q_table,
                episodes=500,
                seed=7,
            )
        )

        balanced_result = (
            _b11_evaluate_policy(
                fixed_action=1,
                episodes=500,
                seed=7,
            )
        )

        ai_result = (
            _b11_evaluate_policy(
                fixed_action=3,
                episodes=500,
                seed=7,
            )
        )

        inclusive_result = (
            _b11_evaluate_policy(
                fixed_action=4,
                episodes=500,
                seed=7,
            )
        )

        random_result = (
            _b11_evaluate_policy(
                random_policy=True,
                episodes=500,
                seed=7,
            )
        )

        evaluation_table = pd.DataFrame(
            [
                [
                    "Q-learning π*",
                    *q_result.values(),
                ],
                [
                    "Luôn a1 - Cân bằng",
                    *balanced_result.values(),
                ],
                [
                    "Luôn a3 - AI dẫn dắt",
                    *ai_result.values(),
                ],
                [
                    "Luôn a4 - Bao trùm",
                    *inclusive_result.values(),
                ],
                [
                    "Random",
                    *random_result.values(),
                ],
            ],
            columns=[
                "Chính sách",
                "Reward trung bình",
                "Độ lệch chuẩn",
                "GDP cuối kỳ",
                "Digital cuối kỳ",
                "AI cuối kỳ",
                "Thất nghiệp cuối kỳ",
            ],
        ).sort_values(
            "Reward trung bình",
            ascending=False,
        )

        best_policy = evaluation_table.iloc[
            0
        ][
            "Chính sách"
        ]

        kpi_cards(
            [
                (
                    "Chính sách tốt nhất",
                    best_policy,
                    "reward ngoài mẫu",
                ),
                (
                    "Reward Q-learning",
                    f"{q_result['mean_reward']:.4f}",
                    "500 episodes",
                ),
                (
                    "Độ lệch chuẩn",
                    f"{q_result['std_reward']:.4f}",
                    "ổn định chính sách",
                ),
                (
                    "Thất nghiệp cuối kỳ",
                    f"{q_result['mean_final_u']:.2f}",
                    "0 thấp, 2 cao",
                ),
            ]
        )

        st.dataframe(
            evaluation_table.style.format(
                {
                    "Reward trung bình": "{:.4f}",
                    "Độ lệch chuẩn": "{:.4f}",
                    "GDP cuối kỳ": "{:.2f}",
                    "Digital cuối kỳ": "{:.2f}",
                    "AI cuối kỳ": "{:.2f}",
                    "Thất nghiệp cuối kỳ": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            plot_bar(
                evaluation_table,
                "Chính sách",
                "Reward trung bình",
                "So sánh reward tích lũy ngoài mẫu",
                text="Reward trung bình",
            ),
            use_container_width=True,
        )

        st.info(
            "Q-learning chỉ được coi là tốt hơn khi reward ngoài mẫu cao hơn hoặc ổn định hơn các quy tắc cố định."
        )

    # -----------------------------------------------------
    # 11.3.5
    # -----------------------------------------------------
    with tab1135:
        st.markdown(
            "### Câu 11.3.5. Mở rộng bằng Deep Q-Network"
        )

        gym_available = False
        sb3_available = False

        try:
            import gymnasium  # noqa: F401
            gym_available = True
        except ModuleNotFoundError:
            gym_available = False

        try:
            import stable_baselines3  # noqa: F401
            sb3_available = True
        except ModuleNotFoundError:
            sb3_available = False

        dependency_table = pd.DataFrame(
            {
                "Thư viện": [
                    "gymnasium",
                    "stable-baselines3",
                ],
                "Trạng thái": [
                    (
                        "Đã cài"
                        if gym_available
                        else "Chưa cài"
                    ),
                    (
                        "Đã cài"
                        if sb3_available
                        else "Chưa cài"
                    ),
                ],
                "Bắt buộc để trang chạy": [
                    "Không",
                    "Không",
                ],
            }
        )

        st.dataframe(
            dependency_table,
            use_container_width=True,
            hide_index=True,
        )

        if (
            gym_available
            and sb3_available
        ):
            st.success(
                "Môi trường đã có Gymnasium và Stable-Baselines3. Có thể huấn luyện DQN."
            )
        else:
            st.warning(
                "DQN đang ở chế độ minh họa mã nguồn để tránh làm Streamlit Cloud nặng. "
                "Tabular Q-learning phía trên vẫn chạy đầy đủ."
            )

        st.code(
            """from stable_baselines3 import DQN

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,
    exploration_fraction=0.40,
    exploration_final_eps=0.05,
    policy_kwargs=dict(
        net_arch=[64, 64]
    ),
    verbose=1
)

model.learn(
    total_timesteps=50_000
)

obs, _ = env.reset()
action, _ = model.predict(
    obs,
    deterministic=True
)""",
            language="python",
        )

        st.markdown(
            """
            **Tiêu chí đánh giá DQN:**

            - reward ngoài mẫu phải cao hơn hoặc ổn định hơn Q-learning;
            - kết quả phải được kiểm tra trên nhiều seed;
            - tránh huấn luyện trực tiếp mỗi lần người dùng mở trang;
            - nên huấn luyện offline, lưu model và chỉ tải model khi triển khai web.
            """
        )

    # =====================================================
    # Tải kết quả
    # =====================================================
    policy_export = (
        _b11_representative_policy(
            q_table
        )
    )

    st.download_button(
        "Tải chính sách đại diện Bài 11",
        data=policy_export.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai11_qlearning_policy.csv",
        mime="text/csv",
        key="download_bai11",
    )

    # =====================================================
    # 11.4. Câu hỏi thảo luận chính sách
    # =====================================================
    st.markdown(
        "## 11.4. Câu hỏi thảo luận chính sách"
    )

    low_state = (
        0,
        0,
        0,
        2,
    )

    high_state = (
        2,
        2,
        2,
        0,
    )

    low_action = int(
        np.argmax(
            q_table[
                low_state
            ]
        )
    )

    high_action = int(
        np.argmax(
            q_table[
                high_state
            ]
        )
    )

    with st.expander(
        "a) Khi GDP thấp, số hóa thấp và thất nghiệp cao, chính sách chọn gì?",
        expanded=True,
    ):
        st.markdown(
            f"Chính sách học được chọn **{_B11_ACTIONS[low_action][0]}**. "
            "Kết quả cần được diễn giải theo hàm thưởng: trong trạng thái khó khăn, "
            "đào tạo lại và số hóa có thể tạo việc làm, trong khi AI dẫn dắt quá nhanh "
            "có thể làm tăng rủi ro thất nghiệp."
        )

    with st.expander(
        "b) Khi GDP cao, AI cao và thất nghiệp thấp, chính sách chọn gì?",
        expanded=True,
    ):
        st.markdown(
            f"Chính sách học được chọn **{_B11_ACTIONS[high_action][0]}**. "
            "Trong trạng thái thuận lợi, tác nhân có thể tăng tốc AI hoặc chuyển sang "
            "củng cố nền tảng và bao trùm tùy theo đánh đổi trong hàm phần thưởng."
        )

    with st.expander(
        "c) Làm sao tích hợp π* mà không thay thế quyết định chính trị?",
        expanded=True,
    ):
        st.markdown(
            "Nên sử dụng π* như hệ thống khuyến nghị: công khai giả định, kiểm định theo "
            "kịch bản, cho phép chuyên gia phản biện, lưu vết quyết định và để cơ quan có "
            "thẩm quyền chịu trách nhiệm cuối cùng. Không nên tự động thực thi chính sách "
            "chỉ dựa trên một mô hình mô phỏng."
        )


def _b12_scenario_shares():
    """
    Năm kịch bản chính sách tích hợp.

    Thứ tự tỷ trọng:
    [K - vốn vật chất, D - chuyển đổi số, AI, H - nhân lực số]
    """
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


def _b12_initial_state():
    """
    Lấy trạng thái cuối năm 2025 từ compute_tfp() đã có trong app.py.
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


def _b12_simulate_scenario(
    shares,
    start_year=2026,
    end_year=2030,
    investment_rate=0.22,
):
    """
    Mô phỏng một kịch bản tích hợp từ 2026 đến 2030.
    """
    state = _b12_initial_state()

    shares = np.asarray(
        shares,
        dtype=float,
    )

    shares = (
        shares
        / max(
            shares.sum(),
            1e-12,
        )
    )

    years = np.arange(
        start_year,
        end_year + 1,
    )

    K = state["K"] * 1.06
    L = state["L"] * 1.006
    D = state["D"] + 0.80
    AI = state["AI"] + 6.00
    H = state["H"] + 0.80
    A = state["A"] * 1.012

    delta_K = 0.05
    delta_D = 0.12
    delta_AI = 0.15
    mu_H = 0.02
    theta_H = 0.80

    rows = []

    for year in years:
        GDP = (
            A
            * K**0.33
            * L**0.42
            * D**0.10
            * AI**0.08
            * H**0.07
        )

        total_investment = (
            investment_rate
            * GDP
        )

        consumption = (
            GDP
            - total_investment
        )

        I_K = (
            shares[0]
            * total_investment
        )

        I_D = (
            shares[1]
            * total_investment
        )

        I_AI = (
            shares[2]
            * total_investment
        )

        I_H = (
            shares[3]
            * total_investment
        )

        cyber_risk = max(
            0.0,
            100
            * (
                0.55 * shares[2]
                + 0.20 * shares[1]
                - 0.25 * shares[3]
            ),
        )

        emission_risk = max(
            0.0,
            100
            * (
                0.36 * shares[0]
                + 0.40 * shares[2]
                + 0.12 * shares[3]
            ),
        )

        inclusion_score = max(
            0.0,
            min(
                100.0,
                100
                * (
                    shares[3]
                    + 0.50 * shares[1]
                ),
            ),
        )

        rows.append(
            [
                year,
                GDP,
                consumption,
                total_investment,
                K,
                D,
                AI,
                H,
                A,
                I_K,
                I_D,
                I_AI,
                I_H,
                cyber_risk,
                emission_risk,
                inclusion_score,
            ]
        )

        K = (
            (1 - delta_K) * K
            + I_K
        )

        D = max(
            1e-6,
            (1 - delta_D) * D
            + I_D / 240.0,
        )

        AI = max(
            1e-6,
            (1 - delta_AI) * AI
            + I_AI / 135.0,
        )

        H = max(
            1e-6,
            H
            + theta_H * I_H / 520.0
            - mu_H * H,
        )

        A = (
            A
            * (
                1
                + 0.00008 * D
                + 0.00004 * AI
                + 0.00006 * H
            )
        )

        L *= 1.006

    return pd.DataFrame(
        rows,
        columns=[
            "Năm",
            "GDP",
            "Tiêu dùng",
            "Tổng đầu tư",
            "K",
            "D",
            "AI",
            "H",
            "A",
            "I_K",
            "I_D",
            "I_AI",
            "I_H",
            "CyberRisk",
            "EmissionRisk",
            "InclusionScore",
        ],
    )


@st.cache_data
def _b12_build_results():
    """
    Chạy toàn bộ năm kịch bản và tạo bảng KPI năm 2030.
    """
    scenarios = _b12_scenario_shares()

    result_rows = []
    simulation_dict = {}

    for scenario_name, shares in scenarios.items():
        simulation = _b12_simulate_scenario(
            shares=shares,
            start_year=2026,
            end_year=2030,
            investment_rate=0.22,
        )

        simulation_dict[
            scenario_name
        ] = simulation

        last = simulation.iloc[-1]

        growth_2026_2030 = (
            100
            * (
                last["GDP"]
                / simulation.iloc[0]["GDP"]
                - 1
            )
        )

        result_rows.append(
            [
                scenario_name,
                float(last["GDP"]),
                float(last["Tiêu dùng"]),
                float(last["K"]),
                float(last["D"]),
                float(last["AI"]),
                float(last["H"]),
                float(last["CyberRisk"]),
                float(last["EmissionRisk"]),
                float(last["InclusionScore"]),
                float(growth_2026_2030),
                float(shares[0]),
                float(shares[1]),
                float(shares[2]),
                float(shares[3]),
            ]
        )

    result_df = pd.DataFrame(
        result_rows,
        columns=[
            "Kịch bản",
            "GDP_2030",
            "Tiêu dùng_2030",
            "K_2030",
            "D_2030",
            "AI_2030",
            "H_2030",
            "CyberRisk",
            "EmissionRisk",
            "InclusionScore",
            "Tăng trưởng 2026-2030 (%)",
            "Share_K",
            "Share_D",
            "Share_AI",
            "Share_H",
        ],
    )

    # Chuẩn hóa để tính điểm cân bằng tích hợp
    result_df[
        "GDP_norm"
    ] = minmax(
        result_df[
            "GDP_2030"
        ]
    )

    result_df[
        "Inclusion_norm"
    ] = minmax(
        result_df[
            "InclusionScore"
        ]
    )

    result_df[
        "Cyber_norm"
    ] = minmax(
        result_df[
            "CyberRisk"
        ]
    )

    result_df[
        "Emission_norm"
    ] = minmax(
        result_df[
            "EmissionRisk"
        ]
    )

    result_df[
        "Điểm tích hợp"
    ] = (
        0.40
        * result_df[
            "GDP_norm"
        ]
        + 0.25
        * result_df[
            "Inclusion_norm"
        ]
        + 0.20
        * (
            1
            - result_df[
                "Cyber_norm"
            ]
        )
        + 0.15
        * (
            1
            - result_df[
                "Emission_norm"
            ]
        )
    )

    result_df[
        "Xếp hạng tích hợp"
    ] = (
        result_df[
            "Điểm tích hợp"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return (
        result_df,
        simulation_dict,
    )


def _b12_policy_recommendations(
    result_df,
):
    """
    Tạo danh sách cảnh báo và khuyến nghị từ kết quả kịch bản.
    """
    warnings = []

    for _, row in result_df.iterrows():
        scenario = row[
            "Kịch bản"
        ]

        if (
            row["Share_AI"] >= 0.40
            and row["Share_H"] < 0.20
        ):
            warnings.append(
                {
                    "Kịch bản": scenario,
                    "Loại": "Rủi ro nhân lực",
                    "Mức": "Cao",
                    "Khuyến nghị": (
                        "AI cao nhưng tỷ trọng nhân lực thấp; "
                        "cần tăng đào tạo, quản trị dữ liệu và chuyển đổi việc làm."
                    ),
                }
            )

        if row["Share_K"] >= 0.65:
            warnings.append(
                {
                    "Kịch bản": scenario,
                    "Loại": "Rủi ro chậm số hóa",
                    "Mức": "Cao",
                    "Khuyến nghị": (
                        "Phụ thuộc vốn vật chất; có nguy cơ bỏ lỡ tăng năng suất "
                        "từ chuyển đổi số và AI."
                    ),
                }
            )

        if row["CyberRisk"] > result_df[
            "CyberRisk"
        ].median():
            warnings.append(
                {
                    "Kịch bản": scenario,
                    "Loại": "An ninh dữ liệu",
                    "Mức": "Trung bình",
                    "Khuyến nghị": (
                        "Tăng đầu tư SOC, tiêu chuẩn dữ liệu, kiểm toán thuật toán "
                        "và năng lực ứng phó sự cố."
                    ),
                }
            )

        if row["EmissionRisk"] > result_df[
            "EmissionRisk"
        ].median():
            warnings.append(
                {
                    "Kịch bản": scenario,
                    "Loại": "Môi trường",
                    "Mức": "Trung bình",
                    "Khuyến nghị": (
                        "Bổ sung năng lượng sạch, trung tâm dữ liệu xanh "
                        "và tiêu chuẩn hiệu quả năng lượng."
                    ),
                }
            )

        if row["InclusionScore"] < result_df[
            "InclusionScore"
        ].median():
            warnings.append(
                {
                    "Kịch bản": scenario,
                    "Loại": "Bao trùm số",
                    "Mức": "Trung bình",
                    "Khuyến nghị": (
                        "Tăng đầu tư kỹ năng số, hỗ trợ vùng yếu "
                        "và tiếp cận công nghệ cho doanh nghiệp nhỏ."
                    ),
                }
            )

    return pd.DataFrame(
        warnings
    )


def page_12():
    hero(
        "Bài 12 — Đồ án tích hợp hệ thống hỗ trợ quyết định AIDEOM-VN",
        "Tích hợp các mô hình thành dashboard hỗ trợ quyết định với 6 module, 5 kịch bản, KPI 2030, cảnh báo rủi ro và khuyến nghị chính sách.",
        ["12.1-12.6", "Integrated dashboard", "5 scenarios", "KPI 2030", "Decision support"],
    )

    (
        result_df,
        simulations,
    ) = _b12_build_results()

    # =====================================================
    # 12.1. Yêu cầu chức năng
    # =====================================================
    st.markdown(
        "## 12.1. Yêu cầu chức năng: 6 module tích hợp"
    )

    module_table = pd.DataFrame(
        [
            [
                "M1",
                "Dự báo kinh tế",
                "Dữ liệu vĩ mô 2020-2025",
                "GDP, TFP, lao động 2026-2030",
                "Cobb-Douglas",
            ],
            [
                "M2",
                "Đánh giá sẵn sàng số",
                "Dữ liệu ngành và vùng",
                "Digital Index, AI Readiness",
                "TOPSIS + Entropy",
            ],
            [
                "M3",
                "Tối ưu phân bổ",
                "Ngân sách, hệ số tác động",
                "Phân bổ ngành-vùng-thời gian",
                "LP + MIP + Dynamic",
            ],
            [
                "M4",
                "Mô phỏng lao động",
                "Đầu tư AI và đào tạo",
                "NetJob theo ngành",
                "LP lao động",
            ],
            [
                "M5",
                "Đánh giá rủi ro",
                "Kịch bản và tham số rủi ro",
                "Cyber, phát thải, bất định",
                "Pareto + Stochastic",
            ],
            [
                "M6",
                "Dashboard tích hợp",
                "Đầu ra M1-M5",
                "KPI, cảnh báo, khuyến nghị",
                "Streamlit + Plotly",
            ],
        ],
        columns=[
            "Module",
            "Tên module",
            "Đầu vào",
            "Đầu ra",
            "Kỹ thuật",
        ],
    )

    st.dataframe(
        module_table,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 12.2. Năm kịch bản
    # =====================================================
    st.markdown(
        "## 12.2. Năm kịch bản chính sách"
    )

    scenario_table = pd.DataFrame(
        [
            [
                "S1 - Truyền thống",
                "70% K + 10% D + 10% AI + 10% H",
                "Ưu tiên vốn vật chất và hạ tầng truyền thống",
            ],
            [
                "S2 - Số hóa nhanh",
                "25% K + 45% D + 15% AI + 15% H",
                "Tăng mạnh chuyển đổi số doanh nghiệp và dịch vụ công",
            ],
            [
                "S3 - AI dẫn dắt",
                "20% K + 20% D + 45% AI + 15% H",
                "Ưu tiên AI, dữ liệu và năng lực tính toán",
            ],
            [
                "S4 - Bao trùm số",
                "30% K + 20% D + 10% AI + 40% H",
                "Ưu tiên nhân lực, kỹ năng và thu hẹp khoảng cách số",
            ],
            [
                "S5 - Tối ưu cân bằng",
                "34% K + 26% D + 18% AI + 22% H",
                "Cân bằng tăng trưởng, số hóa, AI và nhân lực",
            ],
        ],
        columns=[
            "Kịch bản",
            "Cơ cấu phân bổ",
            "Định hướng",
        ],
    )

    st.dataframe(
        scenario_table,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 12.3. Dashboard tích hợp
    # =====================================================
    st.markdown(
        "## 12.3. Dashboard tích hợp"
    )

    (
        tab_overview,
        tab_allocation,
        tab_scenarios,
        tab_risk,
        tab_custom,
    ) = st.tabs(
        [
            "Tổng quan",
            "Phân bổ",
            "So sánh kịch bản",
            "Cảnh báo rủi ro",
            "Kịch bản tùy chỉnh",
        ]
    )

    # -----------------------------------------------------
    # Tổng quan
    # -----------------------------------------------------
    with tab_overview:
        best_gdp = result_df.loc[
            result_df[
                "GDP_2030"
            ].idxmax()
        ]

        best_inclusion = result_df.loc[
            result_df[
                "InclusionScore"
            ].idxmax()
        ]

        lowest_cyber = result_df.loc[
            result_df[
                "CyberRisk"
            ].idxmin()
        ]

        best_integrated = result_df.sort_values(
            "Xếp hạng tích hợp"
        ).iloc[0]

        kpi_cards(
            [
                (
                    "GDP 2030 cao nhất",
                    best_gdp[
                        "Kịch bản"
                    ],
                    f"{best_gdp['GDP_2030']:,.0f}",
                ),
                (
                    "Bao trùm cao nhất",
                    best_inclusion[
                        "Kịch bản"
                    ],
                    f"{best_inclusion['InclusionScore']:.1f}",
                ),
                (
                    "Cyber risk thấp nhất",
                    lowest_cyber[
                        "Kịch bản"
                    ],
                    f"{lowest_cyber['CyberRisk']:.1f}",
                ),
                (
                    "Xếp hạng tích hợp số 1",
                    best_integrated[
                        "Kịch bản"
                    ],
                    f"Điểm={best_integrated['Điểm tích hợp']:.3f}",
                ),
            ]
        )

        display_columns = [
            "Kịch bản",
            "GDP_2030",
            "Tiêu dùng_2030",
            "Tăng trưởng 2026-2030 (%)",
            "CyberRisk",
            "EmissionRisk",
            "InclusionScore",
            "Điểm tích hợp",
            "Xếp hạng tích hợp",
        ]

        st.dataframe(
            result_df[
                display_columns
            ]
            .sort_values(
                "Xếp hạng tích hợp"
            )
            .style.format(
                {
                    "GDP_2030": "{:,.0f}",
                    "Tiêu dùng_2030": "{:,.0f}",
                    "Tăng trưởng 2026-2030 (%)": "{:.2f}",
                    "CyberRisk": "{:.2f}",
                    "EmissionRisk": "{:.2f}",
                    "InclusionScore": "{:.2f}",
                    "Điểm tích hợp": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        ranking_figure = px.bar(
            result_df.sort_values(
                "Điểm tích hợp",
                ascending=False,
            ),
            x="Kịch bản",
            y="Điểm tích hợp",
            color="Kịch bản",
            text="Điểm tích hợp",
            template=PLOT_TEMPLATE,
            title="Xếp hạng tích hợp năm kịch bản",
        )

        ranking_figure.update_layout(
            height=470,
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            ranking_figure,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Phân bổ
    # -----------------------------------------------------
    with tab_allocation:
        allocation_long = result_df.melt(
            id_vars="Kịch bản",
            value_vars=[
                "Share_K",
                "Share_D",
                "Share_AI",
                "Share_H",
            ],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )

        allocation_figure = px.bar(
            allocation_long,
            x="Kịch bản",
            y="Tỷ trọng",
            color="Hạng mục",
            barmode="stack",
            template=PLOT_TEMPLATE,
            title="Cơ cấu phân bổ của năm kịch bản",
        )

        allocation_figure.update_yaxes(
            tickformat=".0%"
        )

        allocation_figure.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=54,
                b=10,
            ),
        )

        st.plotly_chart(
            allocation_figure,
            use_container_width=True,
        )

        allocation_table = result_df[
            [
                "Kịch bản",
                "Share_K",
                "Share_D",
                "Share_AI",
                "Share_H",
            ]
        ].copy()

        st.dataframe(
            allocation_table.style.format(
                {
                    "Share_K": "{:.0%}",
                    "Share_D": "{:.0%}",
                    "Share_AI": "{:.0%}",
                    "Share_H": "{:.0%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # So sánh kịch bản
    # -----------------------------------------------------
    with tab_scenarios:
        selected_scenarios = st.multiselect(
            "Chọn kịch bản để so sánh",
            options=list(
                simulations.keys()
            ),
            default=list(
                simulations.keys()
            ),
            key="b12_compare_scenarios",
        )

        if not selected_scenarios:
            st.warning(
                "Hãy chọn ít nhất một kịch bản."
            )
        else:
            GDP_rows = []

            for scenario in selected_scenarios:
                temp = simulations[
                    scenario
                ][
                    [
                        "Năm",
                        "GDP",
                        "Tiêu dùng",
                        "D",
                        "AI",
                        "H",
                    ]
                ].copy()

                temp[
                    "Kịch bản"
                ] = scenario

                GDP_rows.append(
                    temp
                )

            compare_df = pd.concat(
                GDP_rows,
                ignore_index=True,
            )

            GDP_figure = px.line(
                compare_df,
                x="Năm",
                y="GDP",
                color="Kịch bản",
                markers=True,
                template=PLOT_TEMPLATE,
                title="Quỹ đạo GDP 2026-2030",
            )

            GDP_figure.update_layout(
                height=480,
                margin=dict(
                    l=10,
                    r=10,
                    t=54,
                    b=10,
                ),
            )

            st.plotly_chart(
                GDP_figure,
                use_container_width=True,
            )

            c1, c2 = st.columns(
                2
            )

            with c1:
                GDP_bar = px.bar(
                    result_df[
                        result_df[
                            "Kịch bản"
                        ].isin(
                            selected_scenarios
                        )
                    ],
                    x="Kịch bản",
                    y="GDP_2030",
                    color="Kịch bản",
                    template=PLOT_TEMPLATE,
                    title="GDP năm 2030",
                )

                GDP_bar.update_layout(
                    height=430,
                    showlegend=False,
                )

                st.plotly_chart(
                    GDP_bar,
                    use_container_width=True,
                )

            with c2:
                risk_long = result_df[
                    result_df[
                        "Kịch bản"
                    ].isin(
                        selected_scenarios
                    )
                ].melt(
                    id_vars="Kịch bản",
                    value_vars=[
                        "CyberRisk",
                        "EmissionRisk",
                        "InclusionScore",
                    ],
                    var_name="KPI",
                    value_name="Điểm",
                )

                risk_figure = px.bar(
                    risk_long,
                    x="Kịch bản",
                    y="Điểm",
                    color="KPI",
                    barmode="group",
                    template=PLOT_TEMPLATE,
                    title="Rủi ro và bao trùm",
                )

                risk_figure.update_layout(
                    height=430,
                )

                st.plotly_chart(
                    risk_figure,
                    use_container_width=True,
                )

    # -----------------------------------------------------
    # Cảnh báo rủi ro
    # -----------------------------------------------------
    with tab_risk:
        warning_df = (
            _b12_policy_recommendations(
                result_df
            )
        )

        if warning_df.empty:
            st.success(
                "Không phát hiện cảnh báo theo các ngưỡng hiện tại."
            )
        else:
            st.dataframe(
                warning_df,
                use_container_width=True,
                hide_index=True,
            )

            warning_count = (
                warning_df.groupby(
                    [
                        "Kịch bản",
                        "Mức",
                    ]
                )
                .size()
                .reset_index(
                    name="Số cảnh báo"
                )
            )

            warning_figure = px.bar(
                warning_count,
                x="Kịch bản",
                y="Số cảnh báo",
                color="Mức",
                barmode="stack",
                template=PLOT_TEMPLATE,
                title="Số cảnh báo theo kịch bản",
            )

            warning_figure.update_layout(
                height=450,
                margin=dict(
                    l=10,
                    r=10,
                    t=54,
                    b=10,
                ),
            )

            st.plotly_chart(
                warning_figure,
                use_container_width=True,
            )

            for _, warning in warning_df.iterrows():
                st.markdown(
                    (
                        "<div class='warning-box'>"
                        f"<b>{warning['Kịch bản']} — {warning['Loại']} "
                        f"({warning['Mức']})</b><br>"
                        f"{warning['Khuyến nghị']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------
    # Kịch bản tùy chỉnh
    # -----------------------------------------------------
    with tab_custom:
        st.markdown(
            "### Xây dựng kịch bản phân bổ riêng"
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        custom_k = c1.slider(
            "K",
            min_value=0.0,
            max_value=1.0,
            value=0.34,
            step=0.01,
            key="b12_custom_k",
        )

        custom_d = c2.slider(
            "D",
            min_value=0.0,
            max_value=1.0,
            value=0.26,
            step=0.01,
            key="b12_custom_d",
        )

        custom_ai = c3.slider(
            "AI",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
            key="b12_custom_ai",
        )

        custom_h = c4.slider(
            "H",
            min_value=0.0,
            max_value=1.0,
            value=0.22,
            step=0.01,
            key="b12_custom_h",
        )

        raw_shares = np.array(
            [
                custom_k,
                custom_d,
                custom_ai,
                custom_h,
            ],
            dtype=float,
        )

        if raw_shares.sum() <= 0:
            st.error(
                "Tổng tỷ trọng phải lớn hơn 0."
            )
        else:
            normalized_shares = (
                raw_shares
                / raw_shares.sum()
            )

            custom_simulation = (
                _b12_simulate_scenario(
                    normalized_shares,
                    start_year=2026,
                    end_year=2030,
                    investment_rate=0.22,
                )
            )

            custom_last = (
                custom_simulation.iloc[-1]
            )

            kpi_cards(
                [
                    (
                        "GDP 2030",
                        f"{custom_last['GDP']:,.0f}",
                        "kịch bản tùy chỉnh",
                    ),
                    (
                        "Tiêu dùng 2030",
                        f"{custom_last['Tiêu dùng']:,.0f}",
                        "kịch bản tùy chỉnh",
                    ),
                    (
                        "Cyber risk",
                        f"{custom_last['CyberRisk']:.2f}",
                        "càng thấp càng tốt",
                    ),
                    (
                        "Bao trùm",
                        f"{custom_last['InclusionScore']:.2f}",
                        "càng cao càng tốt",
                    ),
                ]
            )

            normalized_table = pd.DataFrame(
                {
                    "Hạng mục": [
                        "K",
                        "D",
                        "AI",
                        "H",
                    ],
                    "Tỷ trọng chuẩn hóa": normalized_shares,
                }
            )

            st.dataframe(
                normalized_table.style.format(
                    {
                        "Tỷ trọng chuẩn hóa": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            custom_figure = px.line(
                custom_simulation,
                x="Năm",
                y=[
                    "GDP",
                    "Tiêu dùng",
                ],
                markers=True,
                template=PLOT_TEMPLATE,
                title="Quỹ đạo kịch bản tùy chỉnh",
            )

            custom_figure.update_layout(
                height=450,
            )

            st.plotly_chart(
                custom_figure,
                use_container_width=True,
            )

    # =====================================================
    # 12.4. Sản phẩm bàn giao
    # =====================================================
    st.markdown(
        "## 12.4. Sản phẩm bàn giao"
    )

    deliverable_table = pd.DataFrame(
        [
            [
                "Mã nguồn Python",
                "app.py, requirements.txt, dữ liệu và README",
                "GitHub",
            ],
            [
                "Dashboard",
                "12 menu bài tập và các mô hình tích hợp",
                "Streamlit Cloud",
            ],
            [
                "Báo cáo",
                "15-25 trang, tối thiểu 5 hình và 4 bảng",
                "Word/PDF",
            ],
            [
                "Slide",
                "Khoảng 15 slide, thuyết minh 20 phút",
                "PowerPoint",
            ],
            [
                "Video demo",
                "3-5 phút giới thiệu chức năng chính",
                "MP4 hoặc liên kết",
            ],
        ],
        columns=[
            "Sản phẩm",
            "Nội dung",
            "Định dạng",
        ],
    )

    st.dataframe(
        deliverable_table,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 12.5. Tiêu chí đánh giá
    # =====================================================
    st.markdown(
        "## 12.5. Tiêu chí đánh giá"
    )

    rubric_table = pd.DataFrame(
        [
            [
                "Mô hình toán học",
                20,
                "Đúng biến, mục tiêu, ràng buộc và diễn giải",
            ],
            [
                "Chất lượng mã nguồn",
                20,
                "Cấu trúc rõ, chạy ổn định, có kiểm tra lỗi",
            ],
            [
                "Dữ liệu Việt Nam",
                15,
                "Có dữ liệu và bối cảnh phù hợp",
            ],
            [
                "Phân tích chính sách",
                20,
                "Giải thích kết quả và đánh đổi",
            ],
            [
                "Trực quan và dashboard",
                15,
                "Bảng, biểu đồ, tương tác và tải kết quả",
            ],
            [
                "Báo cáo và thuyết trình",
                10,
                "Trình bày logic, đúng thời lượng",
            ],
        ],
        columns=[
            "Hạng mục",
            "Trọng số (%)",
            "Yêu cầu",
        ],
    )

    st.dataframe(
        rubric_table,
        use_container_width=True,
        hide_index=True,
    )

    rubric_figure = px.pie(
        rubric_table,
        names="Hạng mục",
        values="Trọng số (%)",
        template=PLOT_TEMPLATE,
        title="Cơ cấu tiêu chí đánh giá",
    )

    rubric_figure.update_layout(
        height=480,
    )

    st.plotly_chart(
        rubric_figure,
        use_container_width=True,
    )

    # =====================================================
    # 12.6. Hướng mở rộng
    # =====================================================
    st.markdown(
        "## 12.6. Hướng mở rộng"
    )

    st.markdown(
        """
        - Phát triển use case chuyên sâu cho Đồng bằng sông Cửu Long, chế biến chế tạo hoặc bán dẫn.
        - Mở rộng mô hình sang CGE/DSGE có yếu tố AI và kinh tế số.
        - Tích hợp dữ liệu theo tháng hoặc quý từ nguồn chính thức.
        - Huấn luyện mô hình RL offline và triển khai chính sách dưới dạng hệ thống khuyến nghị.
        - Xây dựng multi-agent RL cho các bộ, ngành và địa phương.
        - Bổ sung API AI để giải thích kết quả, nhưng vẫn giữ cơ chế kiểm tra và phê duyệt của con người.
        """
    )

    # =====================================================
    # Tải kết quả
    # =====================================================
    export_columns = [
        "Kịch bản",
        "GDP_2030",
        "Tiêu dùng_2030",
        "D_2030",
        "AI_2030",
        "H_2030",
        "CyberRisk",
        "EmissionRisk",
        "InclusionScore",
        "Tăng trưởng 2026-2030 (%)",
        "Điểm tích hợp",
        "Xếp hạng tích hợp",
    ]

    st.download_button(
        "Tải bảng tổng hợp năm kịch bản",
        data=result_df[
            export_columns
        ].to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="bai12_aideom_5_kich_ban.csv",
        mime="text/csv",
        key="download_bai12",
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

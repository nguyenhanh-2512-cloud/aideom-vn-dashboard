
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
    macro = load_macro()
    sectors = load_sectors()
    regions = load_regions()

    hero(
        "VN AIDEOM-VN",
        "AI-Driven Decision Optimization Model for Vietnam. Dashboard này gom 12 bài mô hình ra quyết định thành một web trực quan: từ Cobb-Douglas, LP, MIP, TOPSIS đến Pareto, stochastic programming và Q-learning.",
        ["Streamlit", "Python", "Optimization", "Vietnam 2020-2025"],
    )

    kpi_cards(
        [
            ("GDP 2025", f"{macro.loc[macro.year == 2025, 'GDP_billion_USD'].iloc[0]:,.1f} tỷ USD", "từ bộ dữ liệu macro"),
            ("Kinh tế số / GDP", f"{macro.loc[macro.year == 2025, 'digital_economy_share_GDP_pct'].iloc[0]:.1f}%", "+1,2 điểm so với 2024"),
            ("FDI giải ngân 2025", f"{macro.loc[macro.year == 2025, 'FDI_disbursed_billion_USD'].iloc[0]:.1f} tỷ USD", "macro indicator"),
            ("Số vùng phân tích", f"{len(regions)} vùng", "TOPSIS và LP vùng"),
        ]
    )

    section("Bản đồ 12 bài theo 4 cấp độ")
    overview = pd.DataFrame(
        [
            ["Dễ", "Bài 1", "Cobb-Douglas mở rộng + AI", "TFP, MAPE, Growth accounting"],
            ["Dễ", "Bài 2", "LP ngân sách số 4 hạng mục", "Phân bổ tối ưu, shadow price, sensitivity"],
            ["Dễ", "Bài 3", "Priority 10 ngành", "Min-max, trọng số chính sách, top ngành"],
            ["Trung bình", "Bài 4", "LP ngành-vùng", "24 biến, ràng buộc công bằng vùng"],
            ["Trung bình", "Bài 5", "MIP 15 dự án", "Knapsack, precedence, ngân sách đa năm"],
            ["Trung bình", "Bài 6", "TOPSIS 6 vùng", "Expert weight, entropy weight"],
            ["Khá khó", "Bài 7", "Pareto đa mục tiêu", "Growth, inclusion, emission, data risk"],
            ["Khá khó", "Bài 8", "Tối ưu động 2026-2035", "Quỹ đạo K-D-AI-H-Y-C"],
            ["Khá khó", "Bài 9", "Lao động và AI", "NetJob, retraining capacity"],
            ["Khó", "Bài 10", "Stochastic programming", "First-stage, recourse, VSS/EVPI"],
            ["Khó", "Bài 11", "Q-learning", "Chính sách thích nghi theo trạng thái"],
            ["Khó", "Bài 12", "AIDEOM tích hợp", "So sánh 5 kịch bản chính sách"],
        ],
        columns=["Cấp độ", "Trang", "Mô hình", "Kết quả chính"],
    )
    st.dataframe(overview, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.line(
            macro,
            x="year",
            y=["GDP_trillion_VND", "exports_billion_USD", "FDI_disbursed_billion_USD"],
            markers=True,
            template=PLOT_TEMPLATE,
            title="Tổng quan dữ liệu vĩ mô Việt Nam 2020-2025",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            regions,
            x="digital_index_0_100",
            y="ai_readiness_0_100",
            size="grdp_trillion_VND",
            color="region_name_vi",
            template=PLOT_TEMPLATE,
            title="Sẵn sàng số và AI theo vùng",
            hover_name="region_name_vi",
        )
        fig2.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        <div class="success-box">
        <b>Gợi ý khi nộp bài:</b> Trang chủ nên đóng vai trò executive dashboard. Các trang sau đi vào kết quả từng mô hình, có bảng số, biểu đồ và diễn giải chính sách ngắn.
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
        "Bài 2 — LP phân bổ ngân sách số",
        "Tối đa hóa GDP kỳ vọng từ bốn hạng mục đầu tư: hạ tầng số, AI và dữ liệu, nhân lực số, R&D công nghệ.",
        ["Linear Programming", "scipy.optimize", "Sensitivity"],
    )

    c1, c2 = st.columns(2)
    B = c1.slider("Ngân sách tổng B (nghìn tỷ VND)", 80, 160, 100, 10)
    min_human = c2.slider("Sàn nhân lực số x₃", 20, 50, 20, 5)

    c = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ]
    b_ub = [B, -25, -15, -min_human, -10, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    if not res.success:
        st.error("Bài toán không khả thi với tham số hiện tại. Giảm sàn nhân lực hoặc tăng ngân sách.")
        return

    names = ["Hạ tầng số I", "AI và dữ liệu", "Nhân lực số H", "R&D công nghệ"]
    out = pd.DataFrame({"Hạng mục": names, "Phân bổ tối ưu": res.x, "Hệ số GDP": [0.85, 1.20, 0.95, 1.35]})
    out["GDP gain"] = out["Phân bổ tối ưu"] * out["Hệ số GDP"]
    Z = -res.fun

    kpi_cards(
        [
            ("Z* tối ưu", f"{Z:,.2f}", "nghìn tỷ VND GDP gain"),
            ("AI + R&D", f"{out.loc[out['Hạng mục'].isin(['AI và dữ liệu','R&D công nghệ']), 'Phân bổ tối ưu'].sum() / out['Phân bổ tối ưu'].sum() * 100:.1f}%", "tỷ trọng công nghệ chiến lược"),
            ("Ngân sách sử dụng", f"{out['Phân bổ tối ưu'].sum():,.1f}", "nghìn tỷ VND"),
            ("Hạng mục lớn nhất", out.sort_values("Phân bổ tối ưu", ascending=False).iloc[0]["Hạng mục"], "allocation"),
        ]
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(out, use_container_width=True, hide_index=True)
    with c2:
        st.plotly_chart(plot_bar(out, "Hạng mục", "Phân bổ tối ưu", "Phân bổ tối ưu theo hạng mục", text="Phân bổ tối ưu"), use_container_width=True)

    sens_rows = []
    for b in [100, 120, 140, B]:
        r = linprog(c, A_ub=A_ub, b_ub=[b, -25, -15, -min_human, -10, 0], bounds=[(0, None)] * 4, method="highs")
        if r.success:
            sens_rows.append([b, -r.fun])
    sens = pd.DataFrame(sens_rows, columns=["B", "Z*"])
    st.plotly_chart(plot_line(sens.sort_values("B"), "B", "Z*", "Độ nhạy ngân sách tổng: Z*(B)"), use_container_width=True)


def page_3():
    hero(
        "Bài 3 — Chỉ số Priority cho 10 ngành",
        "Chuẩn hóa min-max, gán trọng số chính sách và xếp hạng 10 ngành Việt Nam theo mức độ ưu tiên chuyển đổi số và AI.",
        ["MCDM", "Min-max", "Policy weights"],
    )
    df = load_sectors().copy()
    cols_good = [
        "growth_rate_2024_pct",
        "gdp_share_2024_pct",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
    ]
    labels = ["Tăng trưởng", "Quy mô GDP", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI readiness", "Giảm rủi ro"]
    default_w = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15], dtype=float)
    st.markdown("#### Điều chỉnh trọng số")
    cols = st.columns(7)
    weights = []
    for col, label, val in zip(cols, labels, default_w):
        weights.append(col.slider(label, 0.00, 0.40, float(val), 0.01))
    weights = np.array(weights)
    weights = weights / max(weights.sum(), 1e-9)

    X = pd.DataFrame({c: minmax(df[c]) for c in cols_good})
    X["risk_reversed"] = reverse_minmax(df["automation_risk_pct"])
    df["Priority"] = X.values @ weights
    result = df[["sector_name_vi", "Priority", "growth_rate_2024_pct", "export_billion_USD", "ai_readiness_0_100", "automation_risk_pct"]].sort_values("Priority", ascending=False)
    result["Rank"] = np.arange(1, len(result) + 1)

    top3 = ", ".join(result.head(3)["sector_name_vi"].tolist())
    kpi_cards(
        [
            ("Top 1", result.iloc[0]["sector_name_vi"], f"Priority={result.iloc[0]['Priority']:.3f}"),
            ("Top 3", top3, "ưu tiên chính sách"),
            ("AI readiness cao nhất", df.loc[df["ai_readiness_0_100"].idxmax(), "sector_name_vi"], "dữ liệu ngành 2024"),
            ("Rủi ro tự động hóa cao nhất", df.loc[df["automation_risk_pct"].idxmax(), "sector_name_vi"], "cần đào tạo lại"),
        ]
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(result, use_container_width=True, hide_index=True)
    with c2:
        st.plotly_chart(plot_bar(result, "sector_name_vi", "Priority", "Xếp hạng Priority theo ngành", text="Priority"), use_container_width=True)

    ai_vals = np.arange(0.05, 0.41, 0.05)
    rows = []
    base_without_ai = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.15], dtype=float)
    for w_ai in ai_vals:
        w_other = base_without_ai / base_without_ai.sum() * (1 - w_ai)
        w = np.array([w_other[0], w_other[1], w_other[2], w_other[3], w_other[4], w_ai, w_other[5]])
        score = X.values @ w
        rank_df = pd.DataFrame({"sector_name_vi": df["sector_name_vi"], "score": score}).sort_values("score", ascending=False)
        for rank, sector in enumerate(rank_df["sector_name_vi"], start=1):
            rows.append([w_ai, sector, rank])
    sens = pd.DataFrame(rows, columns=["Trọng số AI", "Ngành", "Rank"])
    top_sectors = result.head(6)["sector_name_vi"].tolist()
    fig = px.line(sens[sens["Ngành"].isin(top_sectors)], x="Trọng số AI", y="Rank", color="Ngành", markers=True, template=PLOT_TEMPLATE, title="Độ nhạy thứ hạng khi tăng trọng số AI")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def solve_region_lp(fairness=True):
    regions, items, beta, D0 = region_beta_matrix()
    n_x = 24
    M_idx = 24
    n = 25
    c = np.zeros(n)
    c[:n_x] = -beta.reshape(-1)
    A_ub, b_ub = [], []

    row = np.zeros(n)
    row[:n_x] = 1
    A_ub.append(row)
    b_ub.append(50000)

    for r in range(6):
        row = np.zeros(n)
        row[r * 4 : r * 4 + 4] = 1
        A_ub.append(row)
        b_ub.append(12000)
        row = np.zeros(n)
        row[r * 4 : r * 4 + 4] = -1
        A_ub.append(row)
        b_ub.append(-5000)

    row = np.zeros(n)
    for r in range(6):
        row[r * 4 + 3] = -1
    A_ub.append(row)
    b_ub.append(-12000)

    if fairness:
        gamma = 0.002
        lam = 0.7
        for r in range(6):
            row = np.zeros(n)
            row[r * 4 + 1] = gamma
            row[M_idx] = -1
            A_ub.append(row)
            b_ub.append(-D0[r])
        for r in range(6):
            row = np.zeros(n)
            row[r * 4 + 1] = -gamma
            row[M_idx] = lam
            A_ub.append(row)
            b_ub.append(D0[r])

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * n, method="highs")
    return res, regions, items, beta


def page_4():
    hero(
        "Bài 4 — LP phân bổ ngân sách số theo ngành-vùng",
        "Giải bài toán phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục, đồng thời kiểm tra chi phí kinh tế của ràng buộc công bằng vùng.",
        ["24 variables", "Regional fairness", "LP"],
    )
    fairness = st.toggle("Bật ràng buộc công bằng vùng C5", value=True)
    res, regions, items, beta = solve_region_lp(fairness=fairness)
    if not res.success:
        st.error("Bài toán không khả thi với cấu hình này.")
        return
    X = res.x[:24].reshape(6, 4)
    alloc = pd.DataFrame(X, columns=items, index=regions)
    alloc["Tổng vùng"] = alloc.sum(axis=1)
    Z = -res.fun
    nofair, _, _, _ = solve_region_lp(fairness=False)
    gap = (-nofair.fun) - Z if nofair.success else np.nan

    kpi_cards(
        [
            ("Z* GDP gain", f"{Z:,.0f}", "tỷ VND quy đổi theo hệ số"),
            ("Vùng nhận nhiều nhất", alloc["Tổng vùng"].idxmax(), f"{alloc['Tổng vùng'].max():,.0f} tỷ"),
            ("Hạng mục H toàn quốc", f"{alloc['H - Nhân lực số'].sum():,.0f}", "sàn 12.000 tỷ"),
            ("Chi phí công bằng", f"{gap:,.0f}", "so với mô hình bỏ C5"),
        ]
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(alloc.reset_index().rename(columns={"index": "Vùng"}), use_container_width=True, hide_index=True)
    with c2:
        fig = px.imshow(
            pd.DataFrame(X, columns=items, index=regions),
            text_auto=".0f",
            color_continuous_scale="RdPu",
            template=PLOT_TEMPLATE,
            title="Heatmap phân bổ ngân sách tối ưu",
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig, use_container_width=True)


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


def page_6():
    hero(
        "Bài 6 — TOPSIS xếp hạng 6 vùng",
        "Đánh giá mức độ ưu tiên đầu tư AI theo GRDP/người, FDI, Digital Index, AI readiness, lao động đào tạo, R&D, Internet và Gini.",
        ["TOPSIS", "Entropy weight", "MCDM"],
    )
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
    is_benefit = [True, True, True, True, True, True, True, False]
    expert_w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    df["TOPSIS_expert"] = topsis_score(df, criteria, expert_w, is_benefit)

    X_entropy = df[criteria].copy()
    X_entropy["gini_coef"] = reverse_minmax(X_entropy["gini_coef"])
    for c in criteria[:-1]:
        X_entropy[c] = minmax(X_entropy[c])
    ew = entropy_weights_positive(X_entropy.values)
    df["TOPSIS_entropy"] = topsis_score(df, criteria, ew, is_benefit)

    result = df[["region_name_vi", "TOPSIS_expert", "TOPSIS_entropy", "digital_index_0_100", "ai_readiness_0_100"]].sort_values("TOPSIS_expert", ascending=False)
    result["Rank expert"] = np.arange(1, len(result) + 1)

    kpi_cards(
        [
            ("Dẫn đầu expert", result.iloc[0]["region_name_vi"], f"C*={result.iloc[0]['TOPSIS_expert']:.3f}"),
            ("Dẫn đầu entropy", df.sort_values("TOPSIS_entropy", ascending=False).iloc[0]["region_name_vi"], "trọng số khách quan"),
            ("AI readiness cao nhất", df.loc[df["ai_readiness_0_100"].idxmax(), "region_name_vi"], "theo vùng"),
            ("Digital Index cao nhất", df.loc[df["digital_index_0_100"].idxmax(), "region_name_vi"], "theo vùng"),
        ]
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(result, use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(result, x="region_name_vi", y=["TOPSIS_expert", "TOPSIS_entropy"], barmode="group", template=PLOT_TEMPLATE, title="So sánh TOPSIS expert vs entropy")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=54, b=10))
        st.plotly_chart(fig, use_container_width=True)


def pareto_front(points):
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto


@st.cache_data
def sample_pareto(n_samples=1800, seed=42):
    rng = np.random.default_rng(seed)
    regions, items, beta, D0 = region_beta_matrix()
    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    rows, matrices = [], []
    attempts = 0
    while len(rows) < n_samples and attempts < n_samples * 80:
        attempts += 1
        region_budget = rng.dirichlet(np.ones(6)) * 50000
        if np.any(region_budget < 5000) or np.any(region_budget > 12000):
            continue
        X = np.vstack([rng.dirichlet(np.ones(4)) * b for b in region_budget])
        if X[:, 3].sum() < 12000:
            continue
        growth = (beta * X).sum()
        ineq = np.abs(region_budget - region_budget.mean()).mean()
        emission = (e * (X[:, 0] + X[:, 2])).sum()
        risk = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
        rows.append([growth, ineq, emission, risk])
        matrices.append(X)
    F = np.array(rows)
    costs = np.column_stack([-F[:, 0], F[:, 1], F[:, 2], F[:, 3]])
    mask = pareto_front(costs)
    pareto = pd.DataFrame(F[mask], columns=["Growth", "Inequality", "Emission", "DataRisk"])
    mats = [matrices[i] for i, ok in enumerate(mask) if ok]
    for col in ["Growth", "Inequality", "Emission", "DataRisk"]:
        pareto[col + "_norm"] = minmax(pareto[col])
    pareto["CompromiseScore"] = (
        0.40 * pareto["Growth_norm"]
        + 0.25 * (1 - pareto["Inequality_norm"])
        + 0.20 * (1 - pareto["Emission_norm"])
        + 0.15 * (1 - pareto["DataRisk_norm"])
    )
    return pareto, mats


def page_7():
    hero(
        "Bài 7 — Tối ưu đa mục tiêu Pareto",
        "Dashboard minh họa tập nghiệm Pareto cho 4 mục tiêu: tăng trưởng, bao trùm, phát thải và rủi ro dữ liệu. Bản nộp code có thể thay phần lấy mẫu này bằng pymoo/NSGA-II.",
        ["Pareto", "Multi-objective", "Compromise solution"],
    )
    n_samples = st.slider("Số nghiệm mô phỏng", 600, 3000, 1800, 300)
    pareto, mats = sample_pareto(n_samples=n_samples)
    best_idx = pareto["CompromiseScore"].idxmax()
    best = pareto.loc[best_idx]

    kpi_cards(
        [
            ("Số nghiệm Pareto", f"{len(pareto)}", "từ tập mô phỏng khả thi"),
            ("Growth nghiệm thỏa hiệp", f"{best['Growth']:,.0f}", "GDP gain"),
            ("Inequality", f"{best['Inequality']:,.0f}", "MAD ngân sách vùng"),
            ("Compromise score", f"{best['CompromiseScore']:.3f}", "TOPSIS-style"),
        ]
    )
    fig = px.scatter_3d(
        pareto,
        x="Growth",
        y="Inequality",
        z="Emission",
        color="CompromiseScore",
        template=PLOT_TEMPLATE,
        title="Không gian Pareto: Growth - Inequality - Emission",
    )
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=54, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pareto.sort_values("CompromiseScore", ascending=False).head(10), use_container_width=True, hide_index=True)


def page_8():
    hero(
        "Bài 8 — Tối ưu động phân bổ 2026-2035",
        "Mô phỏng quỹ đạo K, D, AI, H, GDP và tiêu dùng trong 10 năm với các chiến lược phân bổ liên thời gian.",
        ["Dynamic optimization", "2026-2035", "Policy trajectory"],
    )
    shares_map = scenario_shares()
    scenario = st.selectbox("Chọn chiến lược", list(shares_map.keys()), index=4)
    invest_rate = st.slider("Tỷ lệ đầu tư / GDP", 0.15, 0.35, 0.22, 0.01)
    shock = st.slider("Cú sốc GDP năm 2028", 0.00, 0.15, 0.00, 0.01)

    sim = simulate_dynamic(shares_map[scenario], invest_rate=invest_rate, shock_2028=shock)
    kpi_cards(
        [
            ("GDP 2035", f"{sim.iloc[-1]['Y_GDP']:,.0f}", "nghìn tỷ VND mô phỏng"),
            ("Tiêu dùng 2035", f"{sim.iloc[-1]['C_tiêu_dùng']:,.0f}", "C_t"),
            ("AI capacity 2035", f"{sim.iloc[-1]['AI']:.1f}", "nghìn DN/capacity proxy"),
            ("H nhân lực 2035", f"{sim.iloc[-1]['H']:.1f}%", "vốn nhân lực số"),
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sim, x="Năm", y=["Y_GDP", "C_tiêu_dùng", "Đầu_tư"], markers=True, template=PLOT_TEMPLATE, title="Y, C và đầu tư theo thời gian")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.line(sim, x="Năm", y=["K", "D", "AI", "H"], markers=True, template=PLOT_TEMPLATE, title="Quỹ đạo trạng thái K, D, AI, H")
        fig2.update_layout(height=430)
        st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(sim, use_container_width=True, hide_index=True)


def labor_parameters():
    names = [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
    ]
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    labor = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    return names, labor, risk, a1, b1, c1, d1


def page_9():
    hero(
        "Bài 9 — Tác động AI tới thị trường lao động",
        "Tối ưu phân bổ x_AI và x_H cho 8 ngành, bảo đảm NetJob không âm và tốc độ tự động hóa không vượt quá năng lực đào tạo lại.",
        ["NetJob", "Retraining", "Labor policy"],
    )
    names, labor, risk, a1, b1, c1p, d1 = labor_parameters()
    budget = st.slider("Ngân sách lao động và AI", 20000, 40000, 30000, 2500)
    cap = st.slider("Trần ngân sách mỗi ngành", 3000, 10000, 8000, 1000)
    N = len(names)

    coeff_ai = a1 - c1p * risk
    coeff_h = b1
    c = -np.r_[coeff_ai, coeff_h]
    A_ub, b_ub = [], []

    row = np.ones(2 * N)
    A_ub.append(row)
    b_ub.append(budget)

    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = -coeff_ai[i]
        row[N + i] = -coeff_h[i]
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(2 * N)
        row[i] = c1p[i] * risk[i]
        row[N + i] = -d1[i]
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(2 * N)
        row[i] = 1
        row[N + i] = 1
        A_ub.append(row)
        b_ub.append(cap)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * (2 * N), method="highs")
    if not res.success:
        st.error("Bài toán không khả thi với tham số hiện tại.")
        return
    xAI = res.x[:N]
    xH = res.x[N:]
    new = a1 * xAI
    upgrade = b1 * xH
    displaced = c1p * risk * xAI
    retrain = d1 * xH
    net = new + upgrade - displaced
    out = pd.DataFrame({"Ngành": names, "x_AI": xAI, "x_H": xH, "NewJob": new, "Upgrade": upgrade, "Displaced": displaced, "RetrainCap": retrain, "NetJob": net})

    kpi_cards(
        [
            ("Tổng NetJob", f"{net.sum():,.0f}", "việc làm ròng"),
            ("Tổng x_AI", f"{xAI.sum():,.0f}", "tỷ VND"),
            ("Tổng x_H", f"{xH.sum():,.0f}", "tỷ VND"),
            ("Ngành NetJob cao nhất", out.loc[out["NetJob"].idxmax(), "Ngành"], "theo mô hình"),
        ]
    )
    c1_, c2_ = st.columns([1, 1])
    with c1_:
        st.dataframe(out, use_container_width=True, hide_index=True)
    with c2_:
        fig = px.bar(out, x="Ngành", y=["NewJob", "Upgrade", "Displaced"], barmode="group", template=PLOT_TEMPLATE, title="Tạo việc làm, nâng cấp và dịch chuyển")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)


def page_10():
    hero(
        "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
        "Ra quyết định first-stage trước bất định, sau đó điều chỉnh bằng recourse theo 4 kịch bản kinh tế toàn cầu.",
        ["Stochastic Programming", "First-stage", "Recourse"],
    )
    items = ["I", "D", "AI", "H"]
    scenarios = ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"]
    prob = np.array([0.30, 0.45, 0.20, 0.05])
    beta = np.array([1.00, 1.10, 1.25, 0.95])
    beta_s = np.array([
        [1.25, 1.35, 1.55, 1.05],
        [1.00, 1.10, 1.25, 0.95],
        [0.75, 0.85, 0.90, 1.00],
        [0.40, 0.50, 0.55, 1.10],
    ])
    n = 4 + 4 * 4
    c = np.zeros(n)
    c[:4] = -beta
    for s in range(4):
        c[4 + s * 4 : 4 + (s + 1) * 4] = -prob[s] * beta_s[s]

    A_ub, b_ub = [], []
    row = np.zeros(n)
    row[:4] = 1
    A_ub.append(row)
    b_ub.append(65000)
    for s in range(4):
        row = np.zeros(n)
        row[4 + s * 4 : 4 + (s + 1) * 4] = 1
        A_ub.append(row)
        b_ub.append(15000)
        row = np.zeros(n)
        row[4 + s * 4 + 2] = 1
        row[3] = -0.5
        A_ub.append(row)
        b_ub.append(0)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * n, method="highs")
    if not res.success:
        st.error("Không tìm được nghiệm tối ưu.")
        return

    x = res.x[:4]
    y = res.x[4:].reshape(4, 4)
    first = pd.DataFrame({"Hạng mục": items, "First-stage x": x})
    recourse = pd.DataFrame(y, columns=items)
    recourse.insert(0, "Kịch bản", scenarios)
    Z = -res.fun

    kpi_cards(
        [
            ("Expected objective", f"{Z:,.0f}", "lợi ích kỳ vọng"),
            ("First-stage lớn nhất", items[int(np.argmax(x))], f"{x.max():,.0f}"),
            ("Dự phòng mỗi kịch bản", "15.000", "tỷ VND"),
            ("Ràng buộc AI", "y_AI ≤ 0,5 x_H", "capacity link"),
        ]
    )
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.dataframe(first, use_container_width=True, hide_index=True)
        st.dataframe(recourse, use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(first, x="Hạng mục", y="First-stage x", template=PLOT_TEMPLATE, title="Quyết định here-and-now")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)


ACTION_LABELS = {
    0: "a0 Truyền thống",
    1: "a1 Cân bằng",
    2: "a2 Số hóa nhanh",
    3: "a3 AI dẫn dắt",
    4: "a4 Bao trùm",
}


@st.cache_data
def train_q_learning(episodes=4000, seed=7):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5))
    shares = np.array(list(scenario_shares().values()))
    shares = np.vstack([
        np.array([0.70, 0.10, 0.10, 0.10]),
        np.array([0.40, 0.25, 0.15, 0.20]),
        np.array([0.25, 0.45, 0.15, 0.15]),
        np.array([0.20, 0.20, 0.45, 0.15]),
        np.array([0.30, 0.20, 0.10, 0.40]),
    ])

    rewards = []
    for ep in range(episodes):
        s = np.array([1, 1, 0, 1])
        total = 0
        eps = max(0.05, 1.0 - ep / (episodes * 0.65))
        for t in range(10):
            if rng.random() < eps:
                a = rng.integers(0, 5)
            else:
                a = int(np.argmax(Q[tuple(s)]))
            sh = shares[a]
            g, d, ai, u = s
            dgdp = 0.20 * sh[0] + 0.35 * sh[1] + 0.50 * sh[2] + 0.25 * sh[3] + rng.normal(0, 0.03)
            dunemp = 0.50 * sh[2] - 0.62 * sh[3] - 0.10 * sh[1] + rng.normal(0, 0.02)
            cyber = 0.55 * sh[2] + 0.20 * sh[1] - 0.25 * sh[3]
            emission = 0.36 * sh[0] + 0.40 * sh[2] + 0.12 * sh[3]
            r = 0.40 * dgdp - 0.25 * dunemp - 0.20 * cyber - 0.15 * emission
            total += r

            s2 = np.array([
                np.clip(g + (dgdp > 0.22) - (dgdp < 0.08), 0, 2),
                np.clip(d + (sh[1] > 0.30) - (sh[1] < 0.15), 0, 2),
                np.clip(ai + (sh[2] > 0.30) - (sh[2] < 0.12), 0, 2),
                np.clip(u + (dunemp > 0.10) - (dunemp < -0.05), 0, 2),
            ]).astype(int)

            Q[tuple(s) + (a,)] += 0.10 * (r + 0.95 * Q[tuple(s2)].max() - Q[tuple(s) + (a,)])
            s = s2
        rewards.append(total)
    return Q, pd.DataFrame({"Episode": np.arange(episodes), "Reward": rewards})


def page_11():
    hero(
        "Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
        "Mô hình hóa nền kinh tế như MDP rời rạc 81 trạng thái, 5 hành động ngân sách và huấn luyện chính sách π*(s).",
        ["Reinforcement Learning", "MDP", "Q-learning"],
    )
    episodes = st.slider("Số episode huấn luyện", 1000, 10000, 4000, 1000)
    Q, rewards = train_q_learning(episodes=episodes)
    initial_states = {
        "VN 2026 thực tế": (1, 1, 0, 1),
        "GDP thấp, D thấp, U cao": (0, 0, 0, 2),
        "GDP cao, AI cao, U thấp": (2, 2, 2, 0),
        "Số hóa cao nhưng AI thấp": (1, 2, 0, 1),
        "Rủi ro thất nghiệp cao": (1, 1, 1, 2),
    }
    rows = []
    for name, state in initial_states.items():
        action = int(np.argmax(Q[state]))
        rows.append([name, str(state), ACTION_LABELS[action], Q[state][action]])
    policy = pd.DataFrame(rows, columns=["Trạng thái", "Mã hóa", "Hành động π*(s)", "Q-value"])

    kpi_cards(
        [
            ("Action cho VN 2026", policy.iloc[0]["Hành động π*(s)"], "policy learned"),
            ("Reward cuối", f"{rewards['Reward'].tail(200).mean():.3f}", "trung bình 200 episode"),
            ("Số trạng thái", "81", "3⁴"),
            ("Số hành động", "5", "a0-a4"),
        ]
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(policy, use_container_width=True, hide_index=True)
    with c2:
        smooth = rewards.copy()
        smooth["Reward_MA"] = smooth["Reward"].rolling(100, min_periods=1).mean()
        st.plotly_chart(plot_line(smooth, "Episode", "Reward_MA", "Learning curve: reward trung bình trượt"), use_container_width=True)


def page_12():
    hero(
        "Bài 12 — AIDEOM-VN tích hợp",
        "Trang tổng hợp 5 kịch bản chính sách S1-S5, chạy pipeline mô phỏng đến 2030 và tạo bảng KPI để phục vụ báo cáo, slide và demo.",
        ["Integrated dashboard", "5 scenarios", "Decision support"],
    )
    rows = []
    sims = {}
    for name, sh in scenario_shares().items():
        sim = simulate_dynamic(sh, end=2030)
        sims[name] = sim
        last = sim.iloc[-1]
        rows.append([name, last["Y_GDP"], last["C_tiêu_dùng"], last["K"], last["D"], last["AI"], last["H"], sh[0], sh[1], sh[2], sh[3]])
    df = pd.DataFrame(rows, columns=["Kịch bản", "GDP_2030", "C_2030", "K_2030", "D_2030", "AI_2030", "H_2030", "Share_K", "Share_D", "Share_AI", "Share_H"])
    best = df.loc[df["GDP_2030"].idxmax()]

    kpi_cards(
        [
            ("GDP 2030 cao nhất", best["Kịch bản"], f"{best['GDP_2030']:,.0f} nghìn tỷ VND"),
            ("AI capacity cao nhất", df.loc[df["AI_2030"].idxmax(), "Kịch bản"], "theo mô phỏng"),
            ("H nhân lực cao nhất", df.loc[df["H_2030"].idxmax(), "Kịch bản"], "bao trùm số"),
            ("Số kịch bản", "5", "S1-S5"),
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(df, x="Kịch bản", y="GDP_2030", color="Kịch bản", template=PLOT_TEMPLATE, title="So sánh GDP 2030 theo kịch bản")
        fig.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        melted = df.melt(id_vars="Kịch bản", value_vars=["Share_K", "Share_D", "Share_AI", "Share_H"], var_name="Hạng mục", value_name="Tỷ trọng")
        fig2 = px.bar(melted, x="Kịch bản", y="Tỷ trọng", color="Hạng mục", template=PLOT_TEMPLATE, title="Cấu trúc phân bổ của 5 kịch bản")
        fig2.update_layout(height=430)
        st.plotly_chart(fig2, use_container_width=True)

    warnings = []
    for _, row in df.iterrows():
        if row["Share_AI"] >= 0.40 and row["Share_H"] < 0.20:
            warnings.append(f"{row['Kịch bản']}: AI cao nhưng H thấp, cần bổ sung đào tạo lại và quản trị rủi ro dữ liệu.")
        if row["Share_K"] >= 0.65:
            warnings.append(f"{row['Kịch bản']}: phụ thuộc lớn vào vốn vật chất, dễ bỏ lỡ hiệu ứng năng suất số.")
        if row["Share_H"] >= 0.35:
            warnings.append(f"{row['Kịch bản']}: phù hợp mục tiêu bao trùm nhưng cần kiểm tra độ trễ tăng trưởng.")
    st.markdown("#### Cảnh báo chính sách")
    for w in warnings:
        st.markdown(f"<div class='warning-box'>{w}</div>", unsafe_allow_html=True)


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

"""M1 - Macro forecasting for the AIDEOM-VN project.

This module implements the Cobb-Douglas extension used in Exercise 1 and
Module M1 of Exercise 12.  It is intentionally independent from Streamlit so it
can be unit-tested with pytest and reused by the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_ALPHA = 0.33
DEFAULT_BETA = 0.42
DEFAULT_GAMMA = 0.10
DEFAULT_DELTA = 0.08
DEFAULT_THETA = 0.07


@dataclass(frozen=True)
class CobbDouglasParams:
    """Elasticities for the extended Cobb-Douglas production function."""

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    gamma: float = DEFAULT_GAMMA
    delta: float = DEFAULT_DELTA
    theta: float = DEFAULT_THETA

    def validate(self) -> None:
        """Raise ValueError if elasticities are invalid."""
        values = [self.alpha, self.beta, self.gamma, self.delta, self.theta]
        if any(v <= 0 for v in values):
            raise ValueError("All Cobb-Douglas elasticities must be positive.")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Elasticities must sum to 1 for constant returns to scale.")


DEFAULT_MACRO = pd.DataFrame(
    {
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "GDP_growth_pct": [2.91, 2.58, 8.02, 5.05, 7.09, 8.02],
        "digital_economy_share_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "labor_productivity_million_VND": [150.1, 172.8, 188.1, 199.3, 221.9, 245.0],
    }
)

DEFAULT_K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
DEFAULT_L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
DEFAULT_AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
DEFAULT_H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)


def load_macro(data_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """Load vietnam_macro_2020_2025.csv, falling back to built-in teaching data."""
    if data_dir is not None:
        path = Path(data_dir) / "vietnam_macro_2020_2025.csv"
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig").sort_values("year").reset_index(drop=True)
    return DEFAULT_MACRO.copy()


def input_arrays(data_dir: Optional[Path | str] = None) -> Dict[str, np.ndarray]:
    """Return arrays used by the production-function model."""
    macro = load_macro(data_dir)
    return {
        "year": macro["year"].to_numpy(dtype=int),
        "Y": macro["GDP_trillion_VND"].to_numpy(dtype=float),
        "K": DEFAULT_K.copy(),
        "L": DEFAULT_L.copy(),
        "D": macro["digital_economy_share_GDP_pct"].to_numpy(dtype=float),
        "AI": DEFAULT_AI.copy(),
        "H": DEFAULT_H.copy(),
    }


def production(K: np.ndarray | float, L: np.ndarray | float, D: np.ndarray | float,
               AI: np.ndarray | float, H: np.ndarray | float, A: np.ndarray | float,
               params: CobbDouglasParams = CobbDouglasParams()) -> np.ndarray:
    """Compute Y = A K^alpha L^beta D^gamma AI^delta H^theta."""
    params.validate()
    return (
        np.asarray(A, dtype=float)
        * np.asarray(K, dtype=float) ** params.alpha
        * np.asarray(L, dtype=float) ** params.beta
        * np.asarray(D, dtype=float) ** params.gamma
        * np.asarray(AI, dtype=float) ** params.delta
        * np.asarray(H, dtype=float) ** params.theta
    )


def compute_tfp(data_dir: Optional[Path | str] = None,
                params: CobbDouglasParams = CobbDouglasParams()) -> pd.DataFrame:
    """Estimate TFP A_t by inverting the extended Cobb-Douglas function."""
    params.validate()
    arr = input_arrays(data_dir)
    denominator = production(arr["K"], arr["L"], arr["D"], arr["AI"], arr["H"], 1.0, params)
    A = arr["Y"] / denominator
    return pd.DataFrame(
        {
            "year": arr["year"],
            "Y": arr["Y"],
            "K": arr["K"],
            "L": arr["L"],
            "D": arr["D"],
            "AI": arr["AI"],
            "H": arr["H"],
            "TFP_A": A,
        }
    )


def safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean absolute percentage error in percent."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))) * 100)


def in_sample_forecast(data_dir: Optional[Path | str] = None,
                       params: CobbDouglasParams = CobbDouglasParams()) -> pd.DataFrame:
    """Forecast in-sample GDP using average TFP over 2020-2025."""
    df = compute_tfp(data_dir, params)
    A_mean = float(df["TFP_A"].mean())
    df["Y_hat"] = production(df["K"], df["L"], df["D"], df["AI"], df["H"], A_mean, params)
    df["abs_error_pct"] = np.abs((df["Y"] - df["Y_hat"]) / df["Y"]) * 100
    df.attrs["MAPE"] = safe_mape(df["Y"], df["Y_hat"])
    df.attrs["A_mean"] = A_mean
    return df


def growth_accounting(data_dir: Optional[Path | str] = None,
                      params: CobbDouglasParams = CobbDouglasParams()) -> pd.DataFrame:
    """Decompose average GDP growth into K, L, D, AI, H and TFP contributions."""
    df = compute_tfp(data_dir, params)
    log_growth_y = np.diff(np.log(df["Y"].to_numpy(dtype=float)))
    avg_growth = float(log_growth_y.mean())
    pieces = {
        "K - Vốn": params.alpha * np.diff(np.log(df["K"])),
        "L - Lao động": params.beta * np.diff(np.log(df["L"])),
        "D - Số hóa": params.gamma * np.diff(np.log(df["D"])),
        "AI - Năng lực AI": params.delta * np.diff(np.log(df["AI"])),
        "H - Nhân lực số": params.theta * np.diff(np.log(df["H"])),
        "TFP": np.diff(np.log(df["TFP_A"])),
    }
    return pd.DataFrame(
        {
            "factor": list(pieces.keys()),
            "avg_log_contribution_pct": [float(v.mean() * 100) for v in pieces.values()],
            "share_of_growth_pct": [float(v.mean() / avg_growth * 100) for v in pieces.values()],
        }
    )


def forecast_2030(data_dir: Optional[Path | str] = None,
                  params: CobbDouglasParams = CobbDouglasParams(),
                  digital_share_2030: float = 30.0,
                  ai_firms_2030: float = 100.0,
                  trained_labor_2030: float = 35.0,
                  k_growth: float = 0.06,
                  l_growth: float = 0.06,
                  tfp_growth: float = 0.012) -> Dict[str, float]:
    """Simulate the 2030 scenario required by Exercise 1.4.4."""
    df = compute_tfp(data_dir, params)
    last = df.iloc[-1]
    years = 5
    K2030 = float(last["K"] * (1 + k_growth) ** years)
    L2030 = float(last["L"] * (1 + l_growth) ** years)
    A2030 = float(last["TFP_A"] * (1 + tfp_growth) ** years)
    Y2030 = float(production(K2030, L2030, digital_share_2030, ai_firms_2030, trained_labor_2030, A2030, params))
    return {
        "GDP_2025": float(last["Y"]),
        "GDP_2030": Y2030,
        "K_2030": K2030,
        "L_2030": L2030,
        "D_2030": digital_share_2030,
        "AI_2030": ai_firms_2030,
        "H_2030": trained_labor_2030,
        "TFP_2030": A2030,
        "GDP_growth_2025_2030_pct": (Y2030 / float(last["Y"]) - 1.0) * 100,
        "GDP_cagr_pct": ((Y2030 / float(last["Y"])) ** (1 / years) - 1.0) * 100,
    }


def simulate_policy_path(shares: np.ndarray, start: int = 2026, end: int = 2030,
                         invest_rate: float = 0.22,
                         data_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """Simulate a simple GDP path for one policy allocation share vector [K,D,AI,H]."""
    shares = np.asarray(shares, dtype=float)
    if shares.shape != (4,):
        raise ValueError("shares must have four elements: K, D, AI, H.")
    shares = shares / shares.sum()

    df = compute_tfp(data_dir)
    last = df.iloc[-1]
    K = float(last["K"] * 1.06)
    L = float(last["L"] * 1.01)
    D = float(last["D"] + 0.8)
    AI = float(last["AI"] + 6.0)
    H = float(last["H"] + 0.8)
    A = float(last["TFP_A"] * 1.012)

    rows = []
    for year in range(start, end + 1):
        Y = float(production(K, L, D, AI, H, A))
        invest = Y * invest_rate
        rows.append({"year": year, "GDP": Y, "K": K, "D": D, "AI": AI, "H": H, "investment": invest})
        K = (1 - 0.05) * K + shares[0] * invest
        D = max(1.0, (1 - 0.12) * D + shares[1] * invest / 240)
        AI = max(1.0, (1 - 0.15) * AI + shares[2] * invest / 135)
        H = max(1.0, H + 0.8 * shares[3] * invest / 520 - 0.02 * H)
        L *= 1.006
        A *= 1 + 0.00008 * D + 0.00004 * AI + 0.00006 * H
    return pd.DataFrame(rows)


def run_m1(data_dir: Optional[Path | str] = None) -> Dict[str, object]:
    """Run all M1 outputs used in Exercise 12."""
    forecast = in_sample_forecast(data_dir)
    return {
        "tfp": compute_tfp(data_dir),
        "forecast": forecast,
        "growth_accounting": growth_accounting(data_dir),
        "scenario_2030": forecast_2030(data_dir),
        "mape": forecast.attrs["MAPE"],
    }

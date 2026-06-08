"""M6 - Integrated AIDEOM-VN pipeline.

This module links M1-M5 into a compact 5-scenario dashboard output for Exercise
12.  It does not replace the Streamlit app; instead, it provides reusable,
unit-testable computation functions for the app.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .m1_forecast import simulate_policy_path
from .m2_readiness import run_m2
from .m3_allocation import run_m3
from .m4_labor import solve_labor_lp, LabourLPConfig
from .m5_risk import risk_score_from_shares, run_m5

SCENARIO_SHARES = {
    "S1 - Truyền thống": np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    "S2 - Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    "S3 - AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    "S4 - Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
    "S5 - Tối ưu cân bằng": np.array([0.34, 0.26, 0.18, 0.22], dtype=float),
}


def _minmax(values: pd.Series, reverse: bool = False) -> pd.Series:
    values = pd.Series(values, dtype=float)
    if values.max() - values.min() < 1e-12:
        return pd.Series(np.ones(len(values)), index=values.index)
    out = (values - values.min()) / (values.max() - values.min())
    return 1 - out if reverse else out


def _labor_proxy(shares: np.ndarray) -> Dict[str, float]:
    """Map scenario shares into a conservative labour KPI.

    The full M4 LP is available in m4_labor.  This scenario-level proxy keeps the
    dashboard responsive and makes scenarios comparable.
    """
    ai_share = float(shares[2])
    h_share = float(shares[3])
    # Calibrated around the 30,000 billion VND Exercise 9 budget.
    net_jobs = 1_200_000 + 2_300_000 * h_share + 1_600_000 * ai_share - 700_000 * max(ai_share - h_share, 0)
    displaced = 180_000 + 1_150_000 * ai_share - 330_000 * h_share
    retraining = 280_000 + 1_900_000 * h_share
    return {
        "NetJobs": float(max(net_jobs, 0)),
        "DisplacedJobs": float(max(displaced, 0)),
        "RetrainingCapacity": float(max(retraining, 0)),
        "RetrainingSafe": bool(retraining + 1e-9 >= displaced),
    }


def run_scenarios(data_dir: Optional[str] = None, end_year: int = 2030) -> pd.DataFrame:
    """Run 5 AIDEOM-VN scenarios and return KPI table."""
    rows = []
    for name, shares in SCENARIO_SHARES.items():
        path = simulate_policy_path(shares, end=end_year, data_dir=data_dir)
        last = path.iloc[-1]
        labour = _labor_proxy(shares)
        risk = risk_score_from_shares(shares)
        rows.append(
            {
                "scenario": name,
                "K_share": float(shares[0]),
                "D_share": float(shares[1]),
                "AI_share": float(shares[2]),
                "H_share": float(shares[3]),
                "GDP_2030": float(last["GDP"]),
                "DigitalIndex_2030": float(last["D"]),
                "AICapacity_2030": float(last["AI"]),
                "HumanCapital_2030": float(last["H"]),
                "NetJobs": labour["NetJobs"],
                "DisplacedJobs": labour["DisplacedJobs"],
                "RetrainingCapacity": labour["RetrainingCapacity"],
                "RetrainingSafe": labour["RetrainingSafe"],
                **risk,
            }
        )
    df = pd.DataFrame(rows)
    df["score_growth"] = _minmax(df["GDP_2030"])
    df["score_jobs"] = _minmax(df["NetJobs"])
    df["score_risk"] = _minmax(df["overall_risk"], reverse=True)
    df["integrated_score"] = 0.45 * df["score_growth"] + 0.30 * df["score_jobs"] + 0.25 * df["score_risk"]
    df["rank"] = df["integrated_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def run_aideom_pipeline(data_dir: Optional[str] = None) -> Dict[str, object]:
    """Run M1-M5 summaries and the integrated M6 scenario table."""
    scenarios = run_scenarios(data_dir)
    return {
        "M2_readiness": run_m2(data_dir),
        "M3_allocation": run_m3(),
        "M4_labor": solve_labor_lp(LabourLPConfig()),
        "M5_risk": run_m5(),
        "M6_scenarios": scenarios,
        "best_scenario": scenarios.iloc[0].to_dict(),
    }


def minimum_submission_checks(data_dir: Optional[str] = None) -> Dict[str, bool]:
    """Return the core checks required for Exercise 12 code/dashboard."""
    scenarios = run_scenarios(data_dir)
    m3 = run_m3()
    return {
        "has_five_scenarios": set(scenarios["scenario"]) == set(SCENARIO_SHARES),
        "scenario_shares_sum_to_one": all(abs(v.sum() - 1) < 1e-9 for v in SCENARIO_SHARES.values()),
        "s1_s3_s5_have_positive_gdp2030": bool((scenarios.loc[scenarios["scenario"].isin(["S1 - Truyền thống", "S3 - AI dẫn dắt", "S5 - Tối ưu cân bằng"]), "GDP_2030"] > 0).all()),
        "m3_budget_not_exceed_50000": m3["total_budget"] <= 50_000 + 1e-7,
        "m3_human_floor_met": m3["human_budget"] >= 12_000 - 1e-7,
        "scores_between_0_and_1": bool(scenarios["integrated_score"].between(0, 1).all()),
    }

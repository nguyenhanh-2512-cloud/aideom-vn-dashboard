"""M4 - Labour-market impact simulation for AI policy.

This module implements Exercise 9 and Module M4 of Exercise 12.  The LP follows
NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i and the key safety constraint
DisplacedJob_i <= RetrainingCapacity_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

SECTORS_8 = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
]
LABOR_MILLION = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15], dtype=float)
RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22], dtype=float) / 100
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5], dtype=float)
A2 = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0], dtype=float)
B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], dtype=float)
C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5], dtype=float)
D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], dtype=float)


@dataclass(frozen=True)
class LabourLPConfig:
    """Configuration for the labour LP."""

    budget: float = 30_000.0
    use_displacement_cap: bool = False
    max_displacement_labor_share: float = 0.05


def labour_parameters() -> pd.DataFrame:
    """Return the 8-sector parameter table from the assignment."""
    return pd.DataFrame(
        {
            "sector": SECTORS_8,
            "labor_million": LABOR_MILLION,
            "risk": RISK,
            "a1_new_ai_jobs_per_billion": A1,
            "a2_new_digital_jobs_per_billion": A2,
            "b1_upgrade_jobs_per_billion": B1,
            "c1_displaced_jobs_per_billion": C1,
            "d1_retraining_capacity_per_billion": D1,
        }
    )


def solve_labor_lp(config: LabourLPConfig = LabourLPConfig()) -> Dict[str, object]:
    """Solve Exercise 9 LP for x_AI and x_H over eight sectors."""
    n = len(SECTORS_8)
    # Variable order: x_AI[0:n], x_H[0:n]
    net_ai_coef = A1 - C1 * RISK
    net_h_coef = B1
    c = -np.r_[net_ai_coef, net_h_coef]

    A_ub = []
    b_ub = []

    row = np.ones(2 * n)
    A_ub.append(row)
    b_ub.append(config.budget)

    # NetJob_i >= 0 -> -net_ai*xAI - b1*xH <= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -net_ai_coef[i]
        row[n + i] = -B1[i]
        A_ub.append(row)
        b_ub.append(0.0)

    # Displaced_i <= RetrainingCapacity_i
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = C1[i] * RISK[i]
        row[n + i] = -D1[i]
        A_ub.append(row)
        b_ub.append(0.0)

    if config.use_displacement_cap:
        for i in range(n):
            row = np.zeros(2 * n)
            row[i] = C1[i] * RISK[i]
            A_ub.append(row)
            b_ub.append(config.max_displacement_labor_share * LABOR_MILLION[i] * 1_000_000)

    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * (2 * n), method="highs")
    if not res.success:
        raise RuntimeError(f"Labour LP failed: {res.message}")

    x_ai = res.x[:n]
    x_h = res.x[n:]
    new_job = A1 * x_ai
    upgrade_job = B1 * x_h
    displaced = C1 * RISK * x_ai
    retraining_capacity = D1 * x_h
    net_job = new_job + upgrade_job - displaced

    result = labour_parameters()
    result["x_AI"] = x_ai
    result["x_H"] = x_h
    result["NewJob"] = new_job
    result["UpgradeJob"] = upgrade_job
    result["DisplacedJob"] = displaced
    result["RetrainingCapacity"] = retraining_capacity
    result["NetJob"] = net_job
    result["NoNetLoss"] = result["NetJob"] >= -1e-7
    result["RetrainingSafe"] = result["DisplacedJob"] <= result["RetrainingCapacity"] + 1e-7

    return {
        "allocation": result,
        "total_net_jobs": float(result["NetJob"].sum()),
        "total_budget": float(x_ai.sum() + x_h.sum()),
        "solver_status": res.message,
        "raw_result": res,
    }


def manufacturing_retraining_threshold(x_ai: float) -> float:
    """Minimum x_H for manufacturing so NetJob_2 >= 0 and retraining capacity is safe."""
    i = SECTORS_8.index("CN chế biến chế tạo")
    # NetJob >= 0
    required_by_netjob = max(0.0, (C1[i] * RISK[i] - A1[i]) * x_ai / B1[i])
    # Displaced <= RetrainingCapacity
    required_by_capacity = C1[i] * RISK[i] * x_ai / D1[i]
    return float(max(required_by_netjob, required_by_capacity))


def run_m4() -> Dict[str, object]:
    """Run all M4 outputs used in Exercise 12."""
    base = solve_labor_lp()
    capped = solve_labor_lp(LabourLPConfig(use_displacement_cap=True))
    base["with_displacement_cap"] = capped
    return base

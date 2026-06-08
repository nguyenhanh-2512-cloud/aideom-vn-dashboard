"""M3 - Regional budget allocation LP for AIDEOM-VN.

This module implements Exercise 4 and Module M3 of Exercise 12 with the exact
regional floor, ceiling, human-capital floor and fairness constraints described
in the assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = ["I - Hạ tầng số", "D - CĐS DN", "AI", "H - Nhân lực số"]
BETA = np.array(
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


@dataclass(frozen=True)
class RegionalLPConfig:
    """Configuration for Exercise 4 / Module M3 LP."""

    total_budget: float = 50_000.0
    region_floor: float = 5_000.0
    region_ceiling: float = 12_000.0
    human_floor: float = 12_000.0
    gamma: float = 0.002
    fairness_lambda: float = 0.70
    use_fairness: bool = True
    allow_elastic_fairness: bool = True
    fairness_slack_penalty: float = 1_000.0


def _build_and_solve(config: RegionalLPConfig, elastic: bool = False) -> Dict[str, object]:
    """Internal LP builder.  If elastic=True, lower fairness constraints get slack."""
    n_regions = len(REGIONS)
    n_items = len(ITEMS)
    n_x = n_regions * n_items
    has_m = config.use_fairness
    n_slack = n_regions if (has_m and elastic) else 0
    m_idx = n_x if has_m else None
    slack_start = n_x + (1 if has_m else 0)
    n_var = n_x + (1 if has_m else 0) + n_slack

    c = np.zeros(n_var)
    c[:n_x] = -BETA.reshape(-1)
    if n_slack:
        c[slack_start:] = config.fairness_slack_penalty

    A_ub = []
    b_ub = []

    row = np.zeros(n_var)
    row[:n_x] = 1
    A_ub.append(row)
    b_ub.append(config.total_budget)

    for r in range(n_regions):
        idx = slice(r * n_items, (r + 1) * n_items)
        row = np.zeros(n_var)
        row[idx] = -1
        A_ub.append(row)
        b_ub.append(-config.region_floor)

        row = np.zeros(n_var)
        row[idx] = 1
        A_ub.append(row)
        b_ub.append(config.region_ceiling)

    row = np.zeros(n_var)
    for r in range(n_regions):
        row[r * n_items + ITEMS.index("H")] = -1
    A_ub.append(row)
    b_ub.append(-config.human_floor)

    if has_m:
        d_idx = ITEMS.index("D")
        for r in range(n_regions):
            # D0_r + gamma*x_Dr <= M
            row = np.zeros(n_var)
            row[r * n_items + d_idx] = config.gamma
            row[m_idx] = -1
            A_ub.append(row)
            b_ub.append(-D0[r])

        for r in range(n_regions):
            # D0_r + gamma*x_Dr + slack_r >= lambda*M
            row = np.zeros(n_var)
            row[r * n_items + d_idx] = -config.gamma
            row[m_idx] = config.fairness_lambda
            if elastic:
                row[slack_start + r] = -1
            A_ub.append(row)
            b_ub.append(D0[r])

    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * n_var, method="highs")
    if not res.success:
        raise RuntimeError(f"Regional LP failed: {res.message}")

    allocation = res.x[:n_x].reshape(n_regions, n_items)
    gross_objective = float((BETA * allocation).sum())
    alloc_df = pd.DataFrame(allocation, index=REGION_NAMES, columns=ITEM_NAMES)
    alloc_df.index.name = "region"
    region_totals = alloc_df.sum(axis=1).rename("total_budget").reset_index()
    item_totals = alloc_df.sum(axis=0).rename("total_budget").reset_index().rename(columns={"index": "item"})
    fairness_slack = res.x[slack_start:] if n_slack else np.zeros(n_regions)
    return {
        "allocation": alloc_df,
        "region_totals": region_totals,
        "item_totals": item_totals,
        "objective": gross_objective,
        "total_budget": float(allocation.sum()),
        "human_budget": float(allocation[:, ITEMS.index("H")].sum()),
        "fairness_slack": pd.DataFrame({"region": REGION_NAMES, "slack_index_points": fairness_slack}),
        "uses_elastic_fairness": bool(elastic),
        "solver_status": res.message,
        "raw_result": res,
    }


def solve_regional_allocation(config: RegionalLPConfig = RegionalLPConfig()) -> Dict[str, object]:
    """Solve the regional LP.

    The assignment's strict C5 parameters (gamma=0.002, lambda=0.7) can be
    infeasible because Tây Nguyên starts from D0=32 and cannot reach 70% of the
    maximum digital index under a 12,000 ceiling.  Therefore the default first
    attempts the strict model and, only if infeasible, uses an elastic fairness
    slack while reporting the slack explicitly.
    """
    try:
        return _build_and_solve(config, elastic=False)
    except RuntimeError:
        if not (config.use_fairness and config.allow_elastic_fairness):
            raise
        return _build_and_solve(config, elastic=True)

def fairness_cost(config: RegionalLPConfig = RegionalLPConfig()) -> Dict[str, float]:
    """Compute GDP-gain cost of the fairness constraint relative to no fairness."""
    with_fair = solve_regional_allocation(config)
    no_fair_cfg = RegionalLPConfig(
        total_budget=config.total_budget,
        region_floor=config.region_floor,
        region_ceiling=config.region_ceiling,
        human_floor=config.human_floor,
        gamma=config.gamma,
        fairness_lambda=config.fairness_lambda,
        use_fairness=False,
    )
    without_fair = solve_regional_allocation(no_fair_cfg)
    cost = float(without_fair["objective"] - with_fair["objective"])
    return {
        "objective_with_fairness": float(with_fair["objective"]),
        "objective_without_fairness": float(without_fair["objective"]),
        "fairness_cost": cost,
        "fairness_cost_pct": cost / float(without_fair["objective"]) * 100,
    }


def run_m3() -> Dict[str, object]:
    """Run all M3 outputs used in Exercise 12."""
    out = solve_regional_allocation()
    out["fairness_cost"] = fairness_cost()
    return out

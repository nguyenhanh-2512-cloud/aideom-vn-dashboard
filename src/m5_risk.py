"""M5 - Risk and two-stage stochastic programming for AIDEOM-VN.

This module implements the scenario structure of Exercise 10 and also provides
simple risk KPIs for Module M5 of Exercise 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ITEMS = ["I", "D", "AI", "H"]
BETA_BASE = np.array([1.00, 1.10, 1.25, 0.95], dtype=float)
SCENARIOS = ["s1 - Lạc quan", "s2 - Cơ sở", "s3 - Bi quan", "s4 - Khủng hoảng"]
PROB = np.array([0.30, 0.45, 0.20, 0.05], dtype=float)
BETA_S = np.array(
    [
        [1.25, 1.35, 1.55, 1.05],
        [1.00, 1.10, 1.25, 0.95],
        [0.75, 0.85, 0.90, 1.00],
        [0.40, 0.50, 0.55, 1.10],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SPConfig:
    """Configuration for the two-stage stochastic program."""

    first_stage_budget: float = 65_000.0
    second_stage_budget: float = 15_000.0
    ai_human_link: float = 0.5


def solve_two_stage_sp(config: SPConfig = SPConfig()) -> Dict[str, object]:
    """Solve the two-stage stochastic programming model with scipy linprog."""
    n_items = len(ITEMS)
    n_s = len(SCENARIOS)
    # Variable order: x[4], y[s,j] flattened scenario-major
    n_var = n_items + n_s * n_items
    c = np.zeros(n_var)
    c[:n_items] = -BETA_BASE
    for s in range(n_s):
        start = n_items + s * n_items
        c[start:start + n_items] = -PROB[s] * BETA_S[s]

    A_ub = []
    b_ub = []

    row = np.zeros(n_var)
    row[:n_items] = 1
    A_ub.append(row)
    b_ub.append(config.first_stage_budget)

    for s in range(n_s):
        start = n_items + s * n_items
        row = np.zeros(n_var)
        row[start:start + n_items] = 1
        A_ub.append(row)
        b_ub.append(config.second_stage_budget)

        row = np.zeros(n_var)
        row[start + ITEMS.index("AI")] = 1
        row[ITEMS.index("H")] = -config.ai_human_link
        A_ub.append(row)
        b_ub.append(0.0)

    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * n_var, method="highs")
    if not res.success:
        raise RuntimeError(f"Two-stage SP failed: {res.message}")

    x = res.x[:n_items]
    y = res.x[n_items:].reshape(n_s, n_items)
    x_df = pd.DataFrame({"item": ITEMS, "first_stage_x": x, "base_beta": BETA_BASE})
    y_df = pd.DataFrame(y, columns=ITEMS)
    y_df.insert(0, "scenario", SCENARIOS)
    y_df.insert(1, "probability", PROB)
    return {
        "first_stage": x_df,
        "second_stage": y_df,
        "objective": float(-res.fun),
        "solver_status": res.message,
        "raw_result": res,
    }


def solve_expected_value(config: SPConfig = SPConfig()) -> Dict[str, object]:
    """Solve a deterministic expected-value version using expected second-stage beta."""
    expected_beta = PROB @ BETA_S
    n_items = len(ITEMS)
    n_var = 2 * n_items
    c = -np.r_[BETA_BASE, expected_beta]
    A_ub = []
    b_ub = []
    row = np.zeros(n_var)
    row[:n_items] = 1
    A_ub.append(row)
    b_ub.append(config.first_stage_budget)
    row = np.zeros(n_var)
    row[n_items:] = 1
    A_ub.append(row)
    b_ub.append(config.second_stage_budget)
    row = np.zeros(n_var)
    row[n_items + ITEMS.index("AI")] = 1
    row[ITEMS.index("H")] = -config.ai_human_link
    A_ub.append(row)
    b_ub.append(0.0)
    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * n_var, method="highs")
    if not res.success:
        raise RuntimeError(f"Expected-value LP failed: {res.message}")
    return {"x": res.x[:n_items], "y_expected": res.x[n_items:], "objective": float(-res.fun)}


def evaluate_fixed_first_stage(x: Sequence[float], config: SPConfig = SPConfig()) -> float:
    """Evaluate expected objective after fixing first-stage x and optimizing recourse."""
    x = np.asarray(x, dtype=float)
    if x.shape != (len(ITEMS),):
        raise ValueError("x must have four elements")
    total = float(BETA_BASE @ x)
    for s in range(len(SCENARIOS)):
        c = -BETA_S[s]
        A_ub = [np.ones(len(ITEMS))]
        b_ub = [config.second_stage_budget]
        row = np.zeros(len(ITEMS))
        row[ITEMS.index("AI")] = 1
        A_ub.append(row)
        b_ub.append(config.ai_human_link * x[ITEMS.index("H")])
        res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * len(ITEMS), method="highs")
        if not res.success:
            raise RuntimeError(f"Recourse evaluation failed: {res.message}")
        total += float(PROB[s] * (-res.fun))
    return total


def perfect_information_value(config: SPConfig = SPConfig()) -> float:
    """Expected objective when the scenario is known before choosing x."""
    values = []
    for s in range(len(SCENARIOS)):
        # Single-scenario version: x + y_s with the same budgets and AI-human link.
        n_items = len(ITEMS)
        c = -np.r_[BETA_BASE, BETA_S[s]]
        A_ub = []
        b_ub = []
        row = np.zeros(2 * n_items)
        row[:n_items] = 1
        A_ub.append(row)
        b_ub.append(config.first_stage_budget)
        row = np.zeros(2 * n_items)
        row[n_items:] = 1
        A_ub.append(row)
        b_ub.append(config.second_stage_budget)
        row = np.zeros(2 * n_items)
        row[n_items + ITEMS.index("AI")] = 1
        row[ITEMS.index("H")] = -config.ai_human_link
        A_ub.append(row)
        b_ub.append(0.0)
        res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * (2 * n_items), method="highs")
        if not res.success:
            raise RuntimeError(f"Perfect-information LP failed: {res.message}")
        values.append(float(-res.fun))
    return float(PROB @ np.asarray(values))


def stochastic_metrics(config: SPConfig = SPConfig()) -> Dict[str, float]:
    """Compute SP objective, EV objective, VSS and EVPI."""
    sp = solve_two_stage_sp(config)
    ev = solve_expected_value(config)
    ev_evaluated = evaluate_fixed_first_stage(ev["x"], config)
    pi = perfect_information_value(config)
    sp_obj = float(sp["objective"])
    return {
        "SP": sp_obj,
        "EV_evaluated": float(ev_evaluated),
        "VSS": float(sp_obj - ev_evaluated),
        "Perfect_information": float(pi),
        "EVPI": float(pi - sp_obj),
    }


def solve_robust_regret(config: SPConfig = SPConfig()) -> Dict[str, object]:
    """A compact robust proxy: maximize the worst scenario-specific recourse-adjusted objective."""
    n_items = len(ITEMS)
    n_s = len(SCENARIOS)
    # x[4], y[s,j], z. Maximize z -> minimize -z.
    z_idx = n_items + n_s * n_items
    n_var = z_idx + 1
    c = np.zeros(n_var)
    c[z_idx] = -1
    A_ub = []
    b_ub = []
    row = np.zeros(n_var)
    row[:n_items] = 1
    A_ub.append(row)
    b_ub.append(config.first_stage_budget)
    for s in range(n_s):
        start = n_items + s * n_items
        row = np.zeros(n_var)
        row[start:start + n_items] = 1
        A_ub.append(row)
        b_ub.append(config.second_stage_budget)
        row = np.zeros(n_var)
        row[start + ITEMS.index("AI")] = 1
        row[ITEMS.index("H")] = -config.ai_human_link
        A_ub.append(row)
        b_ub.append(0.0)
        # z <= beta_base*x + beta_s*y_s -> z - ... <= 0
        row = np.zeros(n_var)
        row[z_idx] = 1
        row[:n_items] = -BETA_BASE
        row[start:start + n_items] = -BETA_S[s]
        A_ub.append(row)
        b_ub.append(0.0)
    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * n_var, method="highs")
    if not res.success:
        raise RuntimeError(f"Robust LP failed: {res.message}")
    return {"objective_worst_case": float(res.x[z_idx]), "x": res.x[:n_items], "raw_result": res}


def risk_score_from_shares(shares: Sequence[float]) -> Dict[str, float]:
    """Compute simple normalized risk KPIs from [K,D,AI,H] shares."""
    shares = np.asarray(shares, dtype=float)
    shares = shares / np.maximum(shares.sum(), 1e-12)
    cyber = float(np.clip(0.25 + 0.90 * shares[2] - 0.35 * shares[3], 0, 1))
    emission = float(np.clip(0.20 + 0.55 * shares[0] + 0.35 * shares[2] - 0.15 * shares[1], 0, 1))
    dependency = float(np.clip(0.25 + 0.50 * shares[2] - 0.20 * shares[3], 0, 1))
    overall = float(0.4 * cyber + 0.35 * emission + 0.25 * dependency)
    return {"cyber_risk": cyber, "emission_risk": emission, "dependency_risk": dependency, "overall_risk": overall}


def run_m5() -> Dict[str, object]:
    """Run all M5 outputs used in Exercise 12."""
    return {
        "stochastic_solution": solve_two_stage_sp(),
        "metrics": stochastic_metrics(),
        "robust_solution": solve_robust_regret(),
    }

"""M2 - Regional digital and AI readiness assessment.

This module implements TOPSIS and entropy weights for Exercise 6 and Module M2
of Exercise 12.  It contains fallback teaching data so tests can run even when
CSV files are not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]
IS_BENEFIT = np.array([True, True, True, True, True, True, True, False], dtype=bool)
EXPERT_WEIGHTS = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10], dtype=float)

DEFAULT_REGIONS = pd.DataFrame(
    {
        "region_name_vi": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index_0_100": [38, 78, 55, 32, 82, 48],
        "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
        "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
        "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    }
)


def load_regions(data_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """Load vietnam_regions_2024.csv, falling back to built-in teaching data."""
    if data_dir is not None:
        path = Path(data_dir) / "vietnam_regions_2024.csv"
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig")
    return DEFAULT_REGIONS.copy()


def _normalise_weights(weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return weights / total


def topsis_score(df: pd.DataFrame, criteria: Sequence[str] = CRITERIA,
                 weights: Sequence[float] = EXPERT_WEIGHTS,
                 is_benefit: Sequence[bool] = IS_BENEFIT) -> np.ndarray:
    """Compute TOPSIS closeness coefficient C_i* for each region."""
    X = df[list(criteria)].to_numpy(dtype=float)
    weights = _normalise_weights(weights)
    is_benefit = np.asarray(is_benefit, dtype=bool)
    denom = np.sqrt((X ** 2).sum(axis=0))
    R = X / np.where(denom == 0, 1, denom)
    V = R * weights
    ideal = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    s_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    return s_neg / np.maximum(s_pos + s_neg, 1e-12)


def entropy_weights(df: pd.DataFrame, criteria: Sequence[str] = CRITERIA,
                    is_benefit: Sequence[bool] = IS_BENEFIT) -> np.ndarray:
    """Objective entropy weights after making all criteria benefit-oriented."""
    X = df[list(criteria)].to_numpy(dtype=float)
    is_benefit = np.asarray(is_benefit, dtype=bool)
    for j, benefit in enumerate(is_benefit):
        col = X[:, j]
        if benefit:
            X[:, j] = col - col.min()
        else:
            X[:, j] = col.max() - col
    X = X + 1e-9
    P = X / np.maximum(X.sum(axis=0), 1e-12)
    k = 1.0 / np.log(len(df))
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1.0 - E
    return d / np.maximum(d.sum(), 1e-12)


def rank_regions(data_dir: Optional[Path | str] = None,
                 weights: Sequence[float] = EXPERT_WEIGHTS) -> pd.DataFrame:
    """Return TOPSIS ranking using expert weights."""
    df = load_regions(data_dir)
    out = df[["region_name_vi"] + CRITERIA].copy()
    out["TOPSIS_score"] = topsis_score(out, CRITERIA, weights, IS_BENEFIT)
    out = out.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def rank_regions_entropy(data_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """Return TOPSIS ranking using entropy-derived weights."""
    df = load_regions(data_dir)
    w = entropy_weights(df)
    out = df[["region_name_vi"] + CRITERIA].copy()
    out["TOPSIS_score"] = topsis_score(out, CRITERIA, w, IS_BENEFIT)
    out = out.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.attrs["entropy_weights"] = w
    return out


def ai_weight_sensitivity(data_dir: Optional[Path | str] = None,
                          values: Sequence[float] = tuple(np.arange(0.10, 0.401, 0.05))) -> pd.DataFrame:
    """Rank regions while varying the AI-readiness weight."""
    rows = []
    base_other = EXPERT_WEIGHTS.copy()
    ai_idx = CRITERIA.index("ai_readiness_0_100")
    base_other[ai_idx] = 0
    for w_ai in values:
        weights = base_other / base_other.sum() * (1 - w_ai)
        weights[ai_idx] = w_ai
        ranking = rank_regions(data_dir, weights)
        for _, row in ranking.iterrows():
            rows.append({"ai_weight": float(w_ai), "region": row["region_name_vi"], "rank": int(row["rank"]), "score": float(row["TOPSIS_score"])})
    return pd.DataFrame(rows)


def run_m2(data_dir: Optional[Path | str] = None) -> Dict[str, object]:
    """Run all M2 outputs used in Exercise 12."""
    expert = rank_regions(data_dir)
    entropy = rank_regions_entropy(data_dir)
    return {
        "expert_ranking": expert,
        "entropy_ranking": entropy,
        "entropy_weights": entropy.attrs["entropy_weights"],
        "sensitivity": ai_weight_sensitivity(data_dir),
        "top3_expert": expert.head(3)["region_name_vi"].tolist(),
    }

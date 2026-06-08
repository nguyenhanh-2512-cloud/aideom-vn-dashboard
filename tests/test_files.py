"""Unit tests for the VN AIDEOM-VN project.

These tests are intentionally lightweight enough for GitHub Codespaces and
Streamlit Cloud, but they verify the main deliverables required by the final
project: project structure, reusable src modules, M1-M6 outputs, and the
S1/S3/S5 scenario checks for 2030.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_SRC_MODULES = [
    "src.model_catalog",
    "src.m1_forecast",
    "src.m2_readiness",
    "src.m3_allocation",
    "src.m4_labor",
    "src.m5_risk",
    "src.m6_pipeline",
]


REQUIRED_DATA_FILES = [
    "vietnam_macro_2020_2025.csv",
    "vietnam_regions_2024.csv",
    "vietnam_sectors_2024.csv",
]


def test_project_files_exist() -> None:
    """Core repository files and folders must exist."""
    assert (ROOT / "app.py").exists(), "Missing app.py"
    assert (ROOT / "requirements.txt").exists(), "Missing requirements.txt"
    assert (ROOT / "README.md").exists(), "Missing README.md"
    assert (ROOT / "src").is_dir(), "Missing src/ folder"
    assert (ROOT / "tests").is_dir(), "Missing tests/ folder"
    assert DATA_DIR.is_dir(), "Missing data/ folder"

    for filename in REQUIRED_DATA_FILES:
        assert (DATA_DIR / filename).exists(), f"Missing data/{filename}"


def test_src_modules_are_importable() -> None:
    """All modular M1-M6 source files should be importable."""
    for module_name in REQUIRED_SRC_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, f"Could not import {module_name}"


def test_model_catalog_covers_all_12_exercises() -> None:
    """The model catalog should document all 12 exercises and 5 scenarios."""
    from src.model_catalog import EXERCISES, SCENARIO_DESCRIPTIONS, list_exercises

    exercises = list_exercises()
    assert len(exercises) == 12
    assert set(EXERCISES.keys()) == set(range(1, 13))
    assert len(SCENARIO_DESCRIPTIONS) == 5
    assert all(spec.title and spec.method and spec.core_outputs for spec in exercises)


def test_m1_forecast_outputs_are_valid() -> None:
    """M1 should calculate TFP, in-sample forecast, growth accounting and GDP 2030."""
    from src.m1_forecast import run_m1

    result = run_m1(DATA_DIR)
    assert {"tfp", "forecast", "growth_accounting", "scenario_2030", "mape"}.issubset(result)

    forecast = result["forecast"]
    growth_accounting = result["growth_accounting"]
    scenario_2030 = result["scenario_2030"]

    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 6
    assert {"year", "Y", "TFP_A", "Y_hat", "abs_error_pct"}.issubset(forecast.columns)
    assert np.isfinite(forecast["TFP_A"]).all()
    assert np.isfinite(forecast["Y_hat"]).all()
    assert math.isfinite(float(result["mape"]))
    assert float(result["mape"]) >= 0

    assert isinstance(growth_accounting, pd.DataFrame)
    assert len(growth_accounting) >= 6
    assert {"factor", "avg_log_contribution_pct", "share_of_growth_pct"}.issubset(
        growth_accounting.columns
    )

    assert scenario_2030["GDP_2030"] > scenario_2030["GDP_2025"]
    assert scenario_2030["D_2030"] == 30.0
    assert scenario_2030["AI_2030"] == 100.0
    assert scenario_2030["H_2030"] == 35.0


def test_m2_readiness_outputs_are_valid() -> None:
    """M2 should rank exactly 6 regions with TOPSIS and entropy weights."""
    from src.m2_readiness import run_m2

    result = run_m2(DATA_DIR)
    expert = result["expert_ranking"]
    entropy = result["entropy_ranking"]
    weights = result["entropy_weights"]
    sensitivity = result["sensitivity"]

    assert isinstance(expert, pd.DataFrame)
    assert isinstance(entropy, pd.DataFrame)
    assert len(expert) == 6
    assert len(entropy) == 6
    assert expert["TOPSIS_score"].between(0, 1).all()
    assert entropy["TOPSIS_score"].between(0, 1).all()
    assert sorted(expert["rank"].tolist()) == list(range(1, 7))
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert len(weights) == 8
    assert isinstance(sensitivity, pd.DataFrame)
    assert sensitivity["AI_weight" if "AI_weight" in sensitivity.columns else "ai_weight"].nunique() >= 4
    assert len(result["top3_expert"]) == 3


def test_m3_allocation_respects_budget_and_human_floor() -> None:
    """M3 regional allocation should satisfy the key project constraints."""
    from src.m3_allocation import run_m3

    result = run_m3()
    allocation = result["allocation"]

    assert isinstance(allocation, pd.DataFrame)
    assert allocation.shape == (6, 4)
    assert result["solver_status"]
    assert result["total_budget"] <= 50000 + 1e-5
    assert result["human_budget"] >= 12000 - 1e-5
    assert result["objective"] > 0
    assert (allocation.to_numpy() >= -1e-7).all()

    region_totals = result["region_totals"]
    assert len(region_totals) == 6
    assert region_totals["budget" if "budget" in region_totals.columns else "total_budget"].between(5000 - 1e-5, 12000 + 1e-5).all()


def test_m4_labor_outputs_are_valid() -> None:
    """M4 should solve the 8-sector NetJob and retraining model."""
    from src.m4_labor import manufacturing_retraining_threshold, run_m4

    result = run_m4()
    allocation = result["allocation"]

    assert isinstance(allocation, pd.DataFrame)
    assert len(allocation) == 8
    assert result["total_budget"] <= 30000 + 1e-5
    assert np.isfinite(allocation["NetJob"]).all()
    assert np.isfinite(allocation["DisplacedJob"]).all()
    assert (allocation["NetJob"] >= -1e-5).all()
    assert (allocation["RetrainingCapacity"] + 1e-5 >= allocation["DisplacedJob"]).all()
    assert allocation["NoNetLoss"].all()
    assert allocation["RetrainingSafe"].all()

    threshold = manufacturing_retraining_threshold(x_ai=1000)
    assert threshold >= 0


def test_m5_stochastic_programming_outputs_are_valid() -> None:
    """M5 should solve the 4-scenario two-stage stochastic programming model."""
    from src.m5_risk import PROB, run_m5, risk_score_from_shares

    assert np.isclose(float(PROB.sum()), 1.0)

    result = run_m5()
    sp = result["stochastic_solution"]
    metrics = result["metrics"]
    robust = result["robust_solution"]

    assert sp["objective"] > 0
    assert sp["first_stage"].shape[0] == 4
    assert sp["second_stage"].shape[0] == 4
    assert set(["s1 - Lạc quan", "s2 - Cơ sở", "s3 - Bi quan", "s4 - Khủng hoảng"]).issubset(
        set(sp["second_stage"]["scenario"])
    )
    assert metrics["SP"] > 0
    assert metrics["Perfect_information"] >= metrics["SP"] - 1e-5
    assert metrics["EVPI"] >= -1e-5
    assert "objective_worst_case" in robust

    risk = risk_score_from_shares([0.25, 0.45, 0.15, 0.15])
    assert 0 <= risk["overall_risk"] <= 1
    assert 0 <= risk["cyber_risk"] <= 1
    assert 0 <= risk["emission_risk"] <= 1


def test_m6_pipeline_covers_five_scenarios_and_internal_checks() -> None:
    """M6 should integrate M1-M5 and produce the required S1-S5 scenario table."""
    from src.m6_pipeline import minimum_submission_checks, run_aideom_pipeline

    result = run_aideom_pipeline(DATA_DIR)
    scenarios = result["M6_scenarios"]
    checks = minimum_submission_checks(DATA_DIR)

    required_scenarios = {
        "S1 - Truyền thống",
        "S2 - Số hóa nhanh",
        "S3 - AI dẫn dắt",
        "S4 - Bao trùm số",
        "S5 - Tối ưu cân bằng",
    }

    assert isinstance(scenarios, pd.DataFrame)
    assert len(scenarios) == 5
    assert required_scenarios == set(scenarios["scenario"])
    assert scenarios["GDP_2030"].gt(0).all()
    assert scenarios["integrated_score"].between(0, 1).all()
    assert sorted(scenarios["rank"].tolist()) == [1, 2, 3, 4, 5]

    for scenario in ["S1 - Truyền thống", "S3 - AI dẫn dắt", "S5 - Tối ưu cân bằng"]:
        value = scenarios.loc[scenarios["scenario"] == scenario, "GDP_2030"].iloc[0]
        assert value > 0

    assert all(checks.values()), f"Submission checks failed: {checks}"
    assert result["best_scenario"]["scenario"] in required_scenarios

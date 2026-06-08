"""Support modules for VN AIDEOM-VN Streamlit project.

The src package separates reusable modelling code from app.py so the project can
be tested with pytest and reviewed more easily on GitHub.
"""

from .model_catalog import (
    ExerciseSpec,
    EXERCISES,
    SCENARIO_DESCRIPTIONS,
    FORMULA_LIBRARY,
    get_exercise,
    list_exercises,
    exercises_by_level,
    dashboard_minimum_check,
    policy_notes,
    report_outline,
)
from .m1_forecast import run_m1, compute_tfp, forecast_2030, simulate_policy_path
from .m2_readiness import run_m2, rank_regions, rank_regions_entropy
from .m3_allocation import run_m3, solve_regional_allocation, fairness_cost
from .m4_labor import run_m4, solve_labor_lp, manufacturing_retraining_threshold
from .m5_risk import run_m5, solve_two_stage_sp, stochastic_metrics
from .m6_pipeline import run_aideom_pipeline, run_scenarios, minimum_submission_checks

__all__ = [
    "ExerciseSpec",
    "EXERCISES",
    "SCENARIO_DESCRIPTIONS",
    "FORMULA_LIBRARY",
    "get_exercise",
    "list_exercises",
    "exercises_by_level",
    "dashboard_minimum_check",
    "policy_notes",
    "report_outline",
    "run_m1",
    "compute_tfp",
    "forecast_2030",
    "simulate_policy_path",
    "run_m2",
    "rank_regions",
    "rank_regions_entropy",
    "run_m3",
    "solve_regional_allocation",
    "fairness_cost",
    "run_m4",
    "solve_labor_lp",
    "manufacturing_retraining_threshold",
    "run_m5",
    "solve_two_stage_sp",
    "stochastic_metrics",
    "run_aideom_pipeline",
    "run_scenarios",
    "minimum_submission_checks",
]

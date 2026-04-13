"""Water Consumption Impact (WCI) Index calculation package."""

from .wci_calculator import (
    compute_wci,
    classify_risk,
    compute_household_equiv,
    compute_community_share,
    project_growth,
    compute_policy_levers,
    run_all,
)

__version__ = "1.0.0"
__author__ = "Water-AI Triad"

__all__ = [
    "compute_wci",
    "classify_risk",
    "compute_household_equiv",
    "compute_community_share",
    "project_growth",
    "compute_policy_levers",
    "run_all",
]

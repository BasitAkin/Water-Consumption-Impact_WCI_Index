"""
Water Consumption Impact (WCI) Index Calculator
================================================

Reproducible implementation of the Water Consumption Impact (WCI) framework
presented in:

    Akinade, B. A., Amanambu, A. C., & Frame, J. M. (2026).
    AI Data Centers and the Water Use Feedback Loop.
    Earth-Science Reviews (submitted).

The WCI quantifies the fraction of a host public water system's peak-day
delivery capacity that is permanently consumed by a single data-centre
facility's evaporative cooling demand:

    WCI = C_peak / K = (W / K) x r x PF          [Eq. 1]

A three-factor decomposition maps each term to a distinct policy lever:
    - W/K  (demand scale)        -> Siting
    - r    (consumptive ratio)   -> Cooling technology
    - PF   (peak amplification)  -> Demand management

Equations
---------
    Eq 1:  WCI     = (W/K) x r x PF
    Eq 2:  C_avg   = r x W
    Eq 3:  C_peak  = C_avg x PF
    Eq 4:  HH_eq   = (C_avg x 365.25) / H_avg
    Eq 5:  Share   = C_avg / (pop x h_pc)
    Eq 6:  WCI(t)  = WCI_0 x (1 + g)^t
    Eq 7:  WCI_sit = (alpha x W/K) x r x PF
    Eq 8:  WCI_tec = (W/K) x min(r, r_target) x PF
    Eq 9:  WCI_mgt = (W/K) x r x min(PF, PF_target)

Variable definitions follow the paper's notation exactly:
    r  = consumptive ratio = C / W  (fraction of withdrawal lost to evaporation)
    W  = total water withdrawal (ML/d)
    K  = host utility permitted delivery capacity (ML/d)
    PF = peak-day to average-day consumption ratio

License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MGD_TO_ML_D = 3.78541            # 1 MGD = 3.78541 ML/d
H_AVG = 0.4146                   # ML/household/year  (USGS 2015 Circular 1441)
H_PC = 0.000303                  # ML/person/day      (USGS 2015 Circular 1441)

RISK_TIERS = [
    ("OVERLOAD", 1.00, float("inf")),
    ("SEVERE",   0.50, 1.00),
    ("HIGH",     0.25, 0.50),
    ("MODERATE",  0.10, 0.25),
    ("LOW",      0.05, 0.10),
    ("MINIMAL",  0.00, 0.05),
]

# Default policy-lever targets (Section 6.3 of the paper)
ALPHA_DEFAULT = 0.75       # siting: relocate to utility 33 % larger
R_TARGET_DEFAULT = 0.25    # technology: closed-loop / dry-cooling hybrid
PF_TARGET_DEFAULT = 2.0    # management: well-managed demand shifting


# ---------------------------------------------------------------------------
# Core WCI calculations  (Eqs. 1-3)
# ---------------------------------------------------------------------------
def compute_wci(W_ML_d: float, r: float, K_ML_d: float, PF: float) -> Dict[str, float]:
    """Return WCI and intermediate metrics for a single site.

    Parameters
    ----------
    W_ML_d : float  -- facility water withdrawal (ML/d)
    r      : float  -- consumptive ratio, C/W (0-1)
    K_ML_d : float  -- host utility capacity (ML/d)
    PF     : float  -- peaking factor (>= 1)

    Returns
    -------
    dict with keys: WCI, C_avg, C_peak, W_over_K
    """
    W_over_K = W_ML_d / K_ML_d
    C_avg = r * W_ML_d                    # Eq. 2
    C_peak = C_avg * PF                   # Eq. 3
    WCI = W_over_K * r * PF               # Eq. 1
    return {"WCI": WCI, "C_avg": C_avg, "C_peak": C_peak, "W_over_K": W_over_K}


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
def classify_risk(wci: float) -> str:
    """Map a WCI value to its six-tier risk class."""
    for tier, lo, hi in RISK_TIERS:
        if wci > lo:
            return tier
    return "MINIMAL"


# ---------------------------------------------------------------------------
# Community equivalence  (Eqs. 4-5)
# ---------------------------------------------------------------------------
def compute_household_equiv(C_avg_ML_d: float) -> float:
    """Eq 4: household equivalents = (C_avg x 365.25) / H_avg."""
    return (C_avg_ML_d * 365.25) / H_AVG


def compute_community_share(C_avg_ML_d: float, population: int) -> Optional[float]:
    """Eq 5: DC share of residential water = C_avg / (pop x h_pc)."""
    if population <= 0:
        return None
    return C_avg_ML_d / (population * H_PC)


# ---------------------------------------------------------------------------
# Growth projections  (Eq. 6)
# ---------------------------------------------------------------------------
def project_growth(wci_base: float, g: float, years: int = 11) -> Dict[int, float]:
    """Eq 6: WCI(t) = WCI_0 x (1+g)^t, for t = 0 .. years."""
    return {t: wci_base * (1 + g) ** t for t in range(years + 1)}


def breach_year(wci_base: float, g: float, base_year: int = 2024,
                threshold: float = 1.0, max_years: int = 50) -> Optional[int]:
    """Return the calendar year WCI first exceeds *threshold*, or None."""
    if wci_base >= threshold:
        return base_year
    if wci_base <= 0 or g <= 0:
        return None
    import math
    t = math.ceil(math.log(threshold / wci_base) / math.log(1 + g))
    return base_year + t if t <= max_years else None


# ---------------------------------------------------------------------------
# Policy-lever sensitivity  (Eqs. 7-9)
# ---------------------------------------------------------------------------
def compute_policy_levers(
    W_over_K: float, r: float, PF: float,
    alpha: float = ALPHA_DEFAULT,
    r_target: float = R_TARGET_DEFAULT,
    PF_target: float = PF_TARGET_DEFAULT,
) -> Dict[str, float]:
    """Return new WCI and fractional reduction for each lever applied alone."""
    baseline = W_over_K * r * PF

    wci_sit = (alpha * W_over_K) * r * PF                       # Eq. 7
    wci_tec = W_over_K * min(r, r_target) * PF                  # Eq. 8
    wci_mgt = W_over_K * r * min(PF, PF_target)                 # Eq. 9

    def pct(new):
        return (baseline - new) / baseline if baseline > 0 else 0.0

    return {
        "baseline": baseline,
        "WCI_siting": wci_sit,  "delta_siting": pct(wci_sit),
        "WCI_tech":   wci_tec,  "delta_tech":   pct(wci_tec),
        "WCI_mgmt":   wci_mgt,  "delta_mgmt":   pct(wci_mgt),
    }


# ---------------------------------------------------------------------------
# Dominant-factor identification
# ---------------------------------------------------------------------------
def dominant_factor(W_over_K: float, r: float, PF: float) -> str:
    """Identify which of the three WCI factors is furthest from its target."""
    gaps = {
        "siting (W/K)":     W_over_K / ALPHA_DEFAULT if ALPHA_DEFAULT else 0,
        "technology (r)":   r / R_TARGET_DEFAULT if R_TARGET_DEFAULT else 0,
        "management (PF)":  PF / PF_TARGET_DEFAULT if PF_TARGET_DEFAULT else 0,
    }
    return max(gaps, key=gaps.get)


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------
def run_all(input_csv: str) -> pd.DataFrame:
    """Read site_inputs.csv and compute every WCI metric.

    Expected CSV columns:
        site, state, operator, W_ML_d, r, K_MGD, K_ML_d, PF, population
    Optional: W_ML_yr, W_MGD, PUE, source, notes
    """
    df = pd.read_csv(input_csv)

    # Accept either K_ML_d directly or compute from K_MGD
    if "K_ML_d" not in df.columns and "K_MGD" in df.columns:
        df["K_ML_d"] = df["K_MGD"] * MGD_TO_ML_D

    records = []
    for _, row in df.iterrows():
        W   = row["W_ML_d"]
        r   = row["r"]
        K   = row["K_ML_d"]
        PF  = row["PF"]
        pop = row.get("population", 0)

        core = compute_wci(W, r, K, PF)
        risk = classify_risk(core["WCI"])
        hh   = compute_household_equiv(core["C_avg"])
        share = compute_community_share(core["C_avg"], pop)
        lev  = compute_policy_levers(core["W_over_K"], r, PF)
        dom  = dominant_factor(core["W_over_K"], r, PF)

        # Growth projections (moderate 13 %)
        proj_mod = project_growth(core["WCI"], 0.13, 11)
        proj_hi  = project_growth(core["WCI"], 0.24, 11)
        breach_mod = breach_year(core["WCI"], 0.13)
        breach_hi  = breach_year(core["WCI"], 0.24)

        rec = {
            "site": row.get("site", ""),
            "state": row.get("state", ""),
            "operator": row.get("operator", ""),
            "W_ML_d": W, "r": r, "K_ML_d": K, "PF": PF,
            "W_over_K": core["W_over_K"],
            "C_avg_ML_d": core["C_avg"],
            "C_peak_ML_d": core["C_peak"],
            "WCI": core["WCI"],
            "WCI_pct": core["WCI"] * 100,
            "risk_class": risk,
            "dominant_factor": dom,
            "HH_equiv": hh,
            "population": pop,
            "DC_community_pct": share * 100 if share else None,
            # Policy levers
            "WCI_siting": lev["WCI_siting"],
            "WCI_tech": lev["WCI_tech"],
            "WCI_mgmt": lev["WCI_mgmt"],
            "delta_siting_pct": lev["delta_siting"] * 100,
            "delta_tech_pct": lev["delta_tech"] * 100,
            "delta_mgmt_pct": lev["delta_mgmt"] * 100,
            # Growth
            "breach_year_13pct": breach_mod,
            "breach_year_24pct": breach_hi,
        }
        # Add year-by-year moderate projections
        for t in range(12):
            rec[f"WCI_{2024+t}_mod"] = proj_mod[t]
        for t in range(12):
            rec[f"WCI_{2024+t}_hi"] = proj_hi[t]

        records.append(rec)

    return pd.DataFrame(records).sort_values("WCI", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).parent
    data_csv = script_dir.parent / "data" / "site_inputs.csv"
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if not data_csv.exists():
        print(f"Error: {data_csv} not found.")
        sys.exit(1)

    print(f"Reading {data_csv} ...")
    results = run_all(str(data_csv))

    # Console summary
    cols = ["site","state","WCI","WCI_pct","risk_class","C_avg_ML_d",
            "C_peak_ML_d","HH_equiv","breach_year_13pct"]
    print("\n" + "=" * 100)
    print("WATER CONSUMPTION IMPACT (WCI) INDEX -- TEN-SITE ASSESSMENT")
    print("=" * 100)
    print(results[cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 100)

    out = results_dir / "wci_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")
    return results


if __name__ == "__main__":
    main()

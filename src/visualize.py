"""
Publication-quality figures for the WCI Index.

Generates four figures matching the manuscript (Section 6):
    Fig 3a: WCI by site (horizontal bar)
    Fig 4a: Three-factor decomposition (stacked bar)
    Fig 5a: Growth trajectories (line plot)
    Fig 5b: Policy-lever reductions (grouped bar)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import math

RISK_COLORS = {
    "OVERLOAD": "#d32f2f", "SEVERE": "#e65100", "HIGH": "#f9a825",
    "MODERATE": "#43a047", "LOW": "#1e88e5", "MINIMAL": "#90caf9",
}

SITE_ABBREV = {
    "Lebanon": "Lebanon, IN", "Council Bluffs": "Council Bluffs, IA",
    "Mayes County": "Mayes Co., OK", "The Dalles": "The Dalles, OR",
    "Douglas County": "Douglas Co., GA", "Wisconsin Site": "Wisconsin Site, WI",
    "Botetourt County": "Botetourt Co., VA", "Midlothian": "Midlothian, TX",
    "Memphis": "Memphis, TN", "Henderson": "Henderson, NV",
}

def _label(site):
    return SITE_ABBREV.get(site, site)


def plot_wci_bar(df, out_dir="results"):
    df = df.sort_values("WCI", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [RISK_COLORS.get(r, "#999") for r in df["risk_class"]]
    ax.barh([_label(s) for s in df["site"]], df["WCI_pct"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("WCI (%)", fontsize=11)
    ax.set_title("Water Consumption Impact by Site", fontsize=13, fontweight="bold")
    ax.axvline(100, color="#d32f2f", ls="--", lw=1, label="100% capacity")
    ax.axvline(50, color="#e65100", ls=":", lw=0.8)
    ax.axvline(25, color="#f9a825", ls=":", lw=0.8)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    p = Path(out_dir) / "fig_wci_bar.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return str(p)


def plot_three_factor(df, out_dir="results"):
    df = df.sort_values("WCI", ascending=False).head(8)
    log_wk = [abs(math.log10(v)) if v > 0 else 0 for v in df["W_over_K"]]
    log_r  = [abs(math.log10(v)) if v > 0 else 0 for v in df["r"]]
    log_pf = [abs(math.log10(v)) if v > 0 else 0 for v in df["PF"]]
    totals = [a+b+c for a,b,c in zip(log_wk, log_r, log_pf)]
    frac_wk = [a/t*100 if t else 0 for a,t in zip(log_wk, totals)]
    frac_r  = [a/t*100 if t else 0 for a,t in zip(log_r, totals)]
    frac_pf = [a/t*100 if t else 0 for a,t in zip(log_pf, totals)]

    labels = [_label(s) for s in df["site"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, frac_wk, label="W/K (siting)", color="#1565c0")
    ax.bar(x, frac_r, bottom=frac_wk, label="r (technology)", color="#43a047")
    ax.bar(x, frac_pf, bottom=[a+b for a,b in zip(frac_wk, frac_r)], label="PF (management)", color="#e65100")
    ax.set_ylabel("Relative contribution (%)", fontsize=11)
    ax.set_title("Three-Factor Decomposition of WCI", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9); plt.tight_layout()
    p = Path(out_dir) / "fig_three_factor.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return str(p)


def plot_growth_trajectories(df, out_dir="results"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    years = list(range(2024, 2036))
    cols = [f"WCI_{y}_mod" for y in years]
    styles = ["-o", "-s", "-^", "-D", "-v", "-<", "->", "-p"]
    top = df.head(8)
    for i, (_, row) in enumerate(top.iterrows()):
        vals = [row[c] * 100 for c in cols if c in row.index]
        ax.plot(years[:len(vals)], vals, styles[i % len(styles)], label=_label(row["site"]),
                markersize=4, linewidth=1.5)
    ax.axhline(100, color="#d32f2f", ls="--", lw=1.2, label="100% capacity")
    ax.set_xlabel("Year", fontsize=11); ax.set_ylabel("WCI (%)", fontsize=11)
    ax.set_title("WCI Growth Projections (13% CAGR)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_xlim(2024, 2035); ax.tick_params(labelsize=9); plt.tight_layout()
    p = Path(out_dir) / "fig_growth_trajectories.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return str(p)


def plot_policy_levers(df, out_dir="results"):
    top = df.sort_values("WCI", ascending=False).head(8)
    labels = [_label(s) for s in top["site"]]
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, top["delta_siting_pct"], w, label="Siting", color="#1565c0")
    ax.bar(x,     top["delta_tech_pct"],   w, label="Technology", color="#43a047")
    ax.bar(x + w, top["delta_mgmt_pct"],   w, label="Management", color="#e65100")
    ax.set_ylabel("WCI reduction (%)", fontsize=11)
    ax.set_title("Policy Lever Effectiveness", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9); plt.tight_layout()
    p = Path(out_dir) / "fig_policy_levers.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return str(p)


def create_all_figures(df, out_dir="results"):
    Path(out_dir).mkdir(exist_ok=True)
    paths = [plot_wci_bar(df, out_dir), plot_three_factor(df, out_dir),
             plot_growth_trajectories(df, out_dir), plot_policy_levers(df, out_dir)]
    print(f"Saved {len(paths)} figures to {out_dir}/")
    return paths


if __name__ == "__main__":
    from wci_calculator import run_all
    df = run_all(str(Path(__file__).parent.parent / "data" / "site_inputs.csv"))
    create_all_figures(df, str(Path(__file__).parent.parent / "results"))

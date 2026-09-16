"""
Per-pathway summary figure: 2×2 panel layout combining carbon price analysis
(top row) and cost–GHG positioning (bottom row) for a single PtX pathway.

Layout:
  (a) Break-even carbon price       |  (b) Additional fuel cost due to SAF mandate / Cost gap
  (c) Cost–GHG scatter  2025        |  (d) Cost–GHG scatter  2050
      + epsilon-constraint Pareto       + epsilon-constraint Pareto
        frontiers for CN / AU / US        frontiers for CN / AU / US
        (loaded from the cache             (same cache, 2050 tech)
         produced by plot_ptx_pareto_frontier.py)

The per-country Pareto overlays are optional; if no cache is found they
are silently skipped and the bottom panels fall back to scatter only.

Usage:
    python plot_ptx_pathway_summary_figure.py --pathway ftsaf
    python plot_ptx_pathway_summary_figure.py --pathway meoh_to_saf
    python plot_ptx_pathway_summary_figure.py --pathway meoh
    python plot_ptx_pathway_summary_figure.py  # generates all three
    python plot_ptx_pathway_summary_figure.py --pathway meoh --result-version global

    # pick which cached Pareto scenario to overlay (default: hybrid)
    python plot_ptx_pathway_summary_figure.py --pathway ftsaf \
        --pareto-scenario grid_connected
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

from config import (
    FILE_PATH_GLOBAL_RESULTS_PTX,
    FILE_PATH_GLOBAL_RESULTS_PTX_GRID,
    NAME_FUTURE_DB,
    NAME_REF_DB,
    T_DAY_MEOH,
    T_DAY_SAF,
    LOCATIONS,
)

try:
    import numpy.core.numeric as _np_core_num
    sys.modules.setdefault("numpy._core.numeric", _np_core_num)
except (AttributeError, ModuleNotFoundError):
    pass


# ── Pathway definitions ────────────────────────────────────────────────────────
PATHWAYS = {
    "meoh": {
        "name":           "Methanol",
        "unit":           "MeOH",
        "is_saf":         False,
        "case_study_results_file": Path("results") / "case_studies_ptx_meoh.pkl",
        "cost_col":       "euro_tMeOH",
        "ghg_col":        "tCO2_tMeOH",
        "fossil_cost":    450.0,    # EUR/t  cradle-to-gate grey methanol
        "fossil_ghg":     0.70,     # tCO2/t cradle-to-gate
        "fossil_label":   "Fossil methanol (2025)",
        "color":          "#2b6cb0",
        "annual_prod":    T_DAY_MEOH * 365,
    },
    "meoh_to_saf": {
        "name":           "SAF (MtJ)",
        "unit":           "SAF",
        "is_saf":         True,
        "case_study_results_file": Path("results") / "case_studies_ptx_meoh_to_saf.pkl",
        "cost_col":       "euro_tSAF",
        "ghg_col":        "tCO2_tSAF",
        "fossil_cost":    750.0,    # EUR/t  WTW fossil jet midpoint
        "fossil_ghg":     3.90,     # tCO2/t WTW fossil jet
        "fossil_label":   "Fossil jet fuel (2025)",
        "color":          "#dd6b20",
        "annual_prod":    T_DAY_SAF * 365,
    },
    "ftsaf": {
        "name":           "SAF (FT)",
        "unit":           "SAF",
        "is_saf":         True,
        "case_study_results_file": Path("results") / "case_studies_ptx_ftsaf.pkl",
        "cost_col":       "euro_tSAF",
        "ghg_col":        "tCO2_tSAF",
        "fossil_cost":    750.0,
        "fossil_ghg":     3.90,
        "fossil_label":   "Fossil jet fuel (2025)",
        "color":          "#2f855a",
        "annual_prod":    T_DAY_SAF * 365,
    },
}

SCENARIO_ORDER  = ["grid_connected", "hybrid", "hybrid-green", "off_grid"]
SCENARIO_SHORT  = {"grid_connected": "Grid", "hybrid": "Hybrid",
                   "hybrid-green": "H-green", "off_grid": "Off-grid"}
SCENARIO_COLORS = {"grid_connected": "#1b9e77", "hybrid": "#d95f02",
                   "hybrid-green":   "#66a61e", "off_grid": "#7570b3"}
SCENARIO_MARKERS = {"grid_connected": "o", "hybrid": "s",
                    "hybrid-green":   "^", "off_grid": "D"}

# ── Epsilon-constraint Pareto overlay (CN/AU/US) ──────────────────────────────
# Headline production regions whose epsilon-constraint Pareto frontiers
# (produced by plot_ptx_pareto_frontier.py) will be overlaid on panels c and d.
PARETO_COUNTRIES = ["CN", "AU", "US"]
PARETO_COUNTRY_NAMES = {"CN": "China", "AU": "Australia", "US": "United States"}
PARETO_COUNTRY_COLORS = {
    "CN": "#bd0026",   # dark red
    "AU": "#253494",   # dark blue
    "US": "#810f7c",   # dark purple
}
PARETO_COUNTRY_MARKERS = {"CN": "o", "AU": "s", "US": "^"}
PARETO_CACHE_DIR = Path("results")
# Pathways for which the ε-constraint frontier overlay is currently disabled
PARETO_DISABLED_PATHWAYS = {"ftsaf"}

DB_LABELS = {NAME_REF_DB: "2025", NAME_FUTURE_DB: "2050"}
DB_PLOT_STYLES = {
    NAME_REF_DB: {"marker": "o", "alpha": 0.65, "legend": "2025 tech"},
    NAME_FUTURE_DB: {"marker": "s", "alpha": 0.92, "legend": "2050 tech"},
}
COUNTRY_ORDER = [loc[0] for loc in LOCATIONS]

# ReFuelEU Aviation mandate schedule
REF_SAF_FRAC   = np.array([0.020, 0.060, 0.060, 0.200, 0.340, 0.420, 0.700])
REF_EFUEL_FRAC = np.array([0.000, 0.007, 0.012, 0.050, 0.100, 0.150, 0.350])

# EU carbon price reference lines (year, value, color, linestyle)
# CO₂ price projections from Sitarz et al. (2024, Nature Energy), "EU carbon prices
# signal high policy credibility and farsighted actors", doi: 10.1038/s41560-024-01505-x.
# Scenario: EU climate target compatible (EU ETS sectors).
# USD values from Odenweller & Ueckerdt (2025, Nature Energy), Extended Data Table 3,
# converted to EUR at 0.93 EUR/USD (ECB 2024 annual average).
# 2025: observed EU carbon price (EU ETS spot price average, Ember, 2025).
ETS_MILESTONES = [
    (2025,  65.0, "#f59e0b", ":"),
    (2030, 139.0, "#b45309", "--"),
    (2035, 179.0, "#92400e", "-."),
    (2040, 229.0, "#78350f", (0, (3, 1, 1, 1))),
    (2050, 379.0, "#451a03", "-"),
]
BP_CAP = 550.0
FOSSIL_KEROSENE_PRICE = 1420.0   # €/t  – used as 100 % reference on secondary axes

# HEFA SAF reference
HEFA_COST_LOW, HEFA_COST_HIGH = 2000.0, 3000.0
HEFA_COST_MID = (HEFA_COST_LOW + HEFA_COST_HIGH) / 2
HEFA_GHG      = 1.73   # tCO2e/tSAF

# Fossil jet cost band
FOSSIL_JET_LOW, FOSSIL_JET_HIGH = 670.0, 830.0


plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         11.5,
    "axes.labelsize":    12.5,
    "axes.titlesize":    13.0,
    "xtick.labelsize":   10.8,
    "ytick.labelsize":   10.8,
    "legend.fontsize":   10.5,
    "axes.linewidth":    0.85,
    "axes.axisbelow":    True,
    "savefig.dpi":       300,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def _apply_saf_dac_wtw_correction(df, ghg_col, annual_prod):
    """Remove DAC atmospheric CO2 uptake credit to obtain WTW GHG intensity."""
    if "an_ghg_dac" in df.columns:
        ap_col = "annual_production_t" if "annual_production_t" in df.columns else None
        denom = df[ap_col].astype(float) if ap_col else annual_prod
        correction = (-df["an_ghg_dac"].astype(float) / denom).clip(lower=0)
        df = df.copy()
        df[ghg_col] = df[ghg_col].astype(float) + correction
    return df


def _eps_non_dominated(df, x_col="ghg", y_col="cost"):
    pts = df.sort_values([x_col, y_col]).reset_index(drop=True)
    keep, best = [], float("inf")
    for _, r in pts.iterrows():
        if r[y_col] <= best + 1e-6:
            keep.append(True)
            best = r[y_col]
        else:
            keep.append(False)
    return pts[keep].reset_index(drop=True)


def load_pareto_frontiers(pw_key, db_name, scenario="hybrid",
                          cache_dir=PARETO_CACHE_DIR):
    """Load cached epsilon-constraint Pareto frontiers produced by
    plot_ptx_pareto_frontier.py and return {iso2 -> non-dominated DataFrame}.

    Returns ``{}`` if no cache is available, letting the caller skip the
    overlay silently.
    """
    db_label = "reference" if db_name == NAME_REF_DB else "future"
    pkl = Path(cache_dir) / f"pareto_frontier_{pw_key}_{db_label}_{scenario}.pkl"
    if not pkl.exists():
        return {}
    try:
        df = pd.read_pickle(pkl)
    except Exception as exc:
        warnings.warn(f"Could not read Pareto cache {pkl}: {exc}")
        return {}

    if "model_status" in df.columns:
        df = df[df["model_status"].astype(str).str.contains("OPTIMAL", case=False)]
    df = df.dropna(subset=["cost", "ghg"])

    fronts = {}
    for iso2 in PARETO_COUNTRIES:
        sub = df[df["iso2"].astype(str) == iso2][["ghg", "cost"]]
        if sub.empty:
            continue
        fronts[iso2] = _eps_non_dominated(sub)
    return fronts


def _read_results_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    with path.open("rb") as f:
        return pickle.load(f)


def _ordered_unique(values, preferred_order):
    seen = set()
    ordered = []
    values = [str(v) for v in values if pd.notna(v)]
    for item in preferred_order:
        if item in values and item not in seen:
            ordered.append(item)
            seen.add(item)
    for item in values:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _db_offsets(n, width=0.22):
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def _db_style(db_name):
    style = DB_PLOT_STYLES.get(db_name, {})
    return {
        "marker": style.get("marker", "o"),
        "alpha": style.get("alpha", 0.85),
        "legend": style.get("legend", DB_LABELS.get(db_name, str(db_name))),
    }


def _available_db_names(df):
    return _ordered_unique(df["db_name"].astype(str).tolist(), [NAME_REF_DB, NAME_FUTURE_DB])


def load_pathway_data(pw_key, result_version, global_file, global_grid_file):
    pw = PATHWAYS[pw_key]
    if result_version == "case_study":
        df = _read_results_table(pw["case_study_results_file"]).reset_index().copy()
    else:
        frames = []
        for src_path in [global_grid_file, global_file]:
            src = _read_results_table(src_path)
            frames.append(src.reset_index().copy())
        df = pd.concat(frames, ignore_index=True, sort=False)
        df = df[df["ptx_pathway"].astype(str) == pw_key].copy()
        df["db_name"] = df.get("db_name", NAME_REF_DB)
        if "model_status" in df.columns:
            df = df[df["model_status"].astype(str) == "OPTIMAL"].copy()

    df["cost"] = df[pw["cost_col"]].astype(float)
    df["ghg"]  = df[pw["ghg_col"]].astype(float)

    if pw["is_saf"]:
        df = _apply_saf_dac_wtw_correction(df, pw["ghg_col"], pw["annual_prod"])
        df["ghg"] = df[pw["ghg_col"]].astype(float)

    df["fossil_cost"] = pw["fossil_cost"]
    df["fossil_ghg"]  = pw["fossil_ghg"]
    df["scenario"] = pd.Categorical(
        df["scenario"].astype(str),
        categories=_ordered_unique(df["scenario"].astype(str).tolist(), SCENARIO_ORDER),
        ordered=True,
    )
    if "country" in df.columns:
        df["country"] = pd.Categorical(
            df["country"].astype(str),
            categories=_ordered_unique(df["country"].astype(str).tolist(), COUNTRY_ORDER),
            ordered=True,
        )
    return df


# ── Strip-plot helper ──────────────────────────────────────────────────────────

def _strip(ax, x, vals, color, seed=0, width=0.28, marker="o", alpha=0.80):
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return
    rng = np.random.default_rng(seed)
    xs  = x + rng.uniform(-width * 0.4, width * 0.4, finite.size)
    ax.scatter(xs, finite, color=color, s=22, alpha=alpha,
               marker=marker, edgecolors="white", lw=0.3, zorder=5, clip_on=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = float(np.nanmedian(finite))
    ax.hlines(med, x - width * 0.5, x + width * 0.5,
              color=color, lw=2.2, zorder=6)
    if finite.size >= 4:
        q1, q3 = np.nanpercentile(finite, [25, 75])
        ax.add_patch(FancyBboxPatch(
            (x - width * 0.45, q1), width * 0.9, q3 - q1,
            boxstyle="square,pad=0",
            facecolor=color, alpha=0.18,
            edgecolor=color, lw=0.7, zorder=4,
        ))


def _spine_style(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.35")


# ── Panel (a): Break-even carbon price ────────────────────────────────────────

def _panel_breakeven(ax, data, pw, db_names):
    grp_w, gap, strip_x = 1.0, 0.55, 0.22
    xtick_pos, xtick_lab = [], []
    scens = [s for s in SCENARIO_ORDER if s in data["scenario"].cat.categories
             and data[data["scenario"] == s].shape[0] > 0]
    db_offsets = _db_offsets(len(db_names), width=strip_x)

    for si, scen in enumerate(scens):
        x_ctr = si * (grp_w + gap)
        col   = SCENARIO_COLORS[scen]
        xtick_pos.append(x_ctr)
        xtick_lab.append(SCENARIO_SHORT[scen])

        for db, db_offset in zip(db_names, db_offsets):
            style = _db_style(db)
            xpos = x_ctr + db_offset
            sub = data[(data["db_name"] == db) & (data["scenario"] == scen)]
            if sub.empty:
                continue
            d_cost = sub["cost"].values  - pw["fossil_cost"]
            d_ghg  = pw["fossil_ghg"]   - sub["ghg"].values
            with np.errstate(invalid="ignore", divide="ignore"):
                bp = np.where(
                    d_cost <= 0,  0.0,
                    np.where(d_ghg > 1e-9, np.clip(d_cost / d_ghg, 0, BP_CAP), np.nan)
                )
            _strip(ax, xpos, bp, col, seed=si * 10 + db_names.index(db),
                   marker=style["marker"], alpha=style["alpha"])

    # ETS milestones – labels placed inside the axes using a blended transform
    trans_ets = blended_transform_factory(ax.transAxes, ax.transData)
    for yr, val, col_e, ls in ETS_MILESTONES:
        if val <= BP_CAP * 1.05:
            ax.axhline(val, color=col_e, lw=0.9, ls=ls, zorder=2)
            ax.text(0.975, val, f"{yr}: {val:.0f}",
                    transform=trans_ets,
                    va="center", ha="right", fontsize=8.0, color=col_e, zorder=7,
                    bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=0.8))

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab)
    ax.set_xlim(-0.55, (len(scens) - 1) * (grp_w + gap) + 0.65)
    ax.set_ylim(bottom=0, top=BP_CAP * 1.08)
    ax.set_ylabel(r"Break-even CP  [€ t$^{-1}$ CO$_2$]", labelpad=4)
    ax.yaxis.grid(True, color="0.91", lw=0.5, zorder=0)
    _spine_style(ax)

    # Secondary axis: % of fossil kerosene price
    ax2_a = ax.twinx()
    _ylim_a = ax.get_ylim()
    ax2_a.set_ylim(_ylim_a[0] / FOSSIL_KEROSENE_PRICE * 100,
                   _ylim_a[1] / FOSSIL_KEROSENE_PRICE * 100)
    ax2_a.set_ylabel(
        f"% of fossil jet fuel price Apr 2026({FOSSIL_KEROSENE_PRICE:.0f} €/t)",
        labelpad=4,
    )
    ax2_a.tick_params(axis="y", labelsize=10.0)
    _spine_style(ax2_a)

    # Mini legend
    ax.legend(
        handles=[
            Line2D([0],[0], marker=_db_style(db)["marker"], ls="None", ms=5,
                   color="0.4", alpha=_db_style(db)["alpha"], label=_db_style(db)["legend"])
            for db in db_names
        ],
        loc="upper left", frameon=False, fontsize=9.5,
        handletextpad=0.4, borderpad=0.3,
    )


# ── Panel (b): Additional fuel cost due to SAF mandate / Cost gap ─────────────

def _panel_right(ax, data, pw, db_names):
    grp_w, gap, strip_x = 1.0, 0.55, 0.22
    xtick_pos, xtick_lab = [], []
    scens = [s for s in SCENARIO_ORDER if s in data["scenario"].cat.categories
             and data[data["scenario"] == s].shape[0] > 0]
    y_max = 0.0
    db_offsets = _db_offsets(len(db_names), width=strip_x)

    for si, scen in enumerate(scens):
        x_ctr = si * (grp_w + gap)
        col   = SCENARIO_COLORS[scen]
        xtick_pos.append(x_ctr)
        xtick_lab.append(SCENARIO_SHORT[scen])

        d25 = data[(data["db_name"] == NAME_REF_DB)    & (data["scenario"] == scen)]
        d50 = data[(data["db_name"] == NAME_FUTURE_DB) & (data["scenario"] == scen)]

        if pw["is_saf"]:
            y25  = REF_SAF_FRAC[0]    * np.maximum(0, d25["cost"].values - pw["fossil_cost"])
            y50  = REF_SAF_FRAC[-1]   * np.maximum(0, d50["cost"].values - pw["fossil_cost"])
            ye50 = REF_EFUEL_FRAC[-1] * np.maximum(0, d50["cost"].values - pw["fossil_cost"])
        else:
            y25  = np.maximum(0, d25["cost"].values - pw["fossil_cost"])
            y50  = np.maximum(0, d50["cost"].values - pw["fossil_cost"])
            ye50 = np.array([])

        for db_idx, (db, db_offset) in enumerate(zip(db_names, db_offsets)):
            style = _db_style(db)
            sub = data[(data["db_name"] == db) & (data["scenario"] == scen)]
            if pw["is_saf"]:
                frac = REF_SAF_FRAC[0] if db == NAME_REF_DB else REF_SAF_FRAC[-1]
                yvals = frac * np.maximum(0, sub["cost"].values - pw["fossil_cost"])
            else:
                yvals = np.maximum(0, sub["cost"].values - pw["fossil_cost"])
            _strip(ax, x_ctr + db_offset, yvals, col, seed=si * 10 + db_idx,
                   marker=style["marker"], alpha=style["alpha"])
            y_max = max(y_max, float(np.nanmax(yvals)) if yvals.size else 0)

        if pw["is_saf"] and NAME_FUTURE_DB in db_names and ye50.size:
            rng = np.random.default_rng(si * 10 + 2)
            xe  = x_ctr + strip_x + rng.uniform(-0.12, 0.12, ye50.size)
            ax.scatter(xe, ye50, color=col, s=20, alpha=0.55, marker="^",
                       edgecolors=col, lw=0.6, zorder=5, facecolors="none")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ax.hlines(float(np.nanmedian(ye50)),
                          x_ctr + strip_x - 0.18, x_ctr + strip_x + 0.18,
                          color=col, lw=1.2, ls="--", zorder=6)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab)
    ax.set_xlim(-0.55, (len(scens) - 1) * (grp_w + gap) + 0.65)
    ax.set_ylim(bottom=0, top=max(y_max * 1.20, 50))
    ax.yaxis.grid(True, color="0.91", lw=0.5, zorder=0)
    _spine_style(ax)

    # Secondary axis: % of fossil kerosene price
    ax2_b = ax.twinx()
    _ylim_b = ax.get_ylim()
    ax2_b.set_ylim(_ylim_b[0] / FOSSIL_KEROSENE_PRICE * 100,
                   _ylim_b[1] / FOSSIL_KEROSENE_PRICE * 100)
    ax2_b.set_ylabel(
        f"% of fossil jet fuel price ({FOSSIL_KEROSENE_PRICE:.0f} €/t)",
        labelpad=4,
    )
    ax2_b.tick_params(axis="y", labelsize=10.0)
    _spine_style(ax2_b)

    if pw["is_saf"]:
        ax.set_ylabel(r"Additional fuel cost due to SAF mandate  [€ t$^{-1}$ total fuel]", labelpad=4)
        if NAME_FUTURE_DB in db_names:
            mandate_text = (
                f"2025 snapshot:  {REF_SAF_FRAC[0]*100:.0f}% SAF mandate"
                f"  ({REF_EFUEL_FRAC[0]*100:.0f}% e-fuel)\n"
                f"2050 snapshot:  {REF_SAF_FRAC[-1]*100:.0f}% SAF mandate"
                f"  ({REF_EFUEL_FRAC[-1]*100:.0f}% e-fuel)"
            )
        else:
            mandate_text = (
                f"Available data snapshot:  {REF_SAF_FRAC[0]*100:.0f}% SAF mandate"
                f"  ({REF_EFUEL_FRAC[0]*100:.0f}% e-fuel)"
            )
        ax.text(0.03, 0.97, mandate_text,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.8, color="0.35",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.75",
                          boxstyle="round,pad=0.3"))
        handles = [
            Line2D([0],[0], marker=_db_style(db)["marker"], ls="None", ms=5, color="0.4",
                   alpha=_db_style(db)["alpha"],
                   label=f"{_db_style(db)['legend']} (solid)")
            for db in db_names
        ]
        if NAME_FUTURE_DB in db_names:
            handles.append(
                Line2D([0],[0], marker="^", ls="None", ms=5, color="0.4", mfc="none",
                       alpha=0.55, label="2050 e-fuel only (hollow)")
            )
    else:
        ax.set_ylabel(r"Cost gap above fossil  [€ t$^{-1}$ MeOH]", labelpad=4)
        ax.text(0.97, 0.97, "max(0, PtX cost \u2212 grey MeOH cost)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.8, color="0.45", style="italic")
        handles = [
            Line2D([0],[0], marker=_db_style(db)["marker"], ls="None", ms=5, color="0.4",
                   alpha=_db_style(db)["alpha"], label=_db_style(db)["legend"])
            for db in db_names
        ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=9.0, handletextpad=0.4, borderpad=0.3)


# ── Panels (c) and (d): Cost–GHG scatter ─────────────────────────────────────

def _pareto_front(df, x_col, y_col):
    if df.empty:
        return df.copy()
    pts = df.sort_values([x_col, y_col]).reset_index(drop=True)
    best_cost = float("inf")
    keep = []
    for _, row in pts.iterrows():
        if row[y_col] <= best_cost + 1e-9:
            keep.append(True)
            best_cost = row[y_col]
        else:
            keep.append(False)
    return pts[keep]


def _panel_cost_ghg(ax, data, pw, db_name, label_side="right",
                    pareto_frontiers=None):
    """label_side='right' puts marker labels upper-right (panel c / 2025);
       label_side='left'  puts marker labels upper-left  (panel d / 2050).

    ``pareto_frontiers`` is an optional ``{iso2: DataFrame}`` mapping produced
    by :func:`load_pareto_frontiers`; when supplied, the per-country
    epsilon-constraint Pareto frontiers are overlaid on top of the scatter.
    """
    subset = data[data["db_name"] == db_name].copy()
    unit   = pw["unit"]
    is_saf = pw["is_saf"]

    if subset.empty:
        ax.set_visible(False)
        return

    x_col, y_col = "ghg", "cost"
    subset[y_col] = subset[y_col] / 1e3          # €/t → k€/t
    valid_x = subset[x_col].dropna()
    valid_y = subset[y_col].dropna()
    if valid_x.empty or valid_y.empty:
        ax.set_visible(False)
        return

    foss_ghg  = pw["fossil_ghg"]
    foss_cost = pw["fossil_cost"] / 1e3          # €/t → k€/t

    pareto_frontiers = pareto_frontiers or {}
    frontier_x_values = [f["ghg"] for f in pareto_frontiers.values() if not f.empty]
    frontier_y_values = [f["cost"] / 1e3 for f in pareto_frontiers.values() if not f.empty]

    def _extend(values, extra):
        if not extra:
            return values
        return pd.concat([values] + extra, ignore_index=True)

    x_all = _extend(valid_x, frontier_x_values)
    y_all = _extend(valid_y, frontier_y_values)

    xpad = max((x_all.max() - x_all.min()) * 0.12, 0.08)
    ypad = max((y_all.max() - y_all.min()) * 0.12, 0.03)
    xlim = (min(x_all.min() - xpad, foss_ghg  - xpad),
            max(x_all.max() + xpad, foss_ghg  + xpad))
    ylim = (0.0,
            max(y_all.max() + ypad, foss_cost + ypad))

    # Quadrant shading
    ax.fill_betweenx([ylim[0], foss_cost], xlim[0], foss_ghg,
                     color="#c6f6d5", alpha=0.35, zorder=0, lw=0)
    ax.fill_betweenx([foss_cost, ylim[1]], foss_ghg, xlim[1],
                     color="#fed7d7", alpha=0.35, zorder=0, lw=0)

    # Fossil reference band + lines
    if is_saf:
        ax.axhspan(FOSSIL_JET_LOW / 1e3, FOSSIL_JET_HIGH / 1e3,
                   color="0.30", alpha=0.10, zorder=1, lw=0)
    ax.axhline(foss_cost, color="0.38", lw=0.85, ls="--", zorder=1)
    ax.axvline(foss_ghg,  color="0.38", lw=0.85, ls="--", zorder=1)

    # HEFA SAF reference (SAF pathways only)
    hefa_low_k  = HEFA_COST_LOW  / 1e3
    hefa_high_k = HEFA_COST_HIGH / 1e3
    hefa_mid_k  = HEFA_COST_MID  / 1e3
    if is_saf:
        ax.axhspan(hefa_low_k, hefa_high_k,
                   color="#276749", alpha=0.10, zorder=1, lw=0)
        ax.axhline(hefa_mid_k, color="#276749", lw=0.85, ls="--", zorder=1)
        ax.axvline(HEFA_GHG, color="#276749", lw=0.85, ls="--", zorder=1)
        ax.scatter([HEFA_GHG], [hefa_mid_k], s=48, color="#276749",
                   marker="o", zorder=4)
        txt_x = HEFA_GHG + xpad * 0.12 if label_side == "right" else HEFA_GHG - xpad * 0.12
        ax.text(txt_x, hefa_mid_k + ypad * 0.25,
                "HEFA SAF (2025)", fontsize=8.8, color="#276749",
                va="bottom", ha="left" if label_side == "right" else "right", zorder=7)

    # Scatter per scenario
    for scen in SCENARIO_ORDER:
        sc = subset[subset["scenario"] == scen]
        if sc.empty:
            continue
        ax.scatter(sc[x_col], sc[y_col],
                   s=40, alpha=0.75,
                   color=SCENARIO_COLORS[scen],
                   marker=SCENARIO_MARKERS[scen],
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=SCENARIO_SHORT[scen])

    # Epsilon-constraint Pareto frontiers for the three headline regions
    # (CN / AU / US) produced by plot_ptx_pareto_frontier.py.
    for iso2 in PARETO_COUNTRIES:
        front_eps = pareto_frontiers.get(iso2) if pareto_frontiers else None
        if front_eps is None or front_eps.empty:
            continue
        color  = PARETO_COUNTRY_COLORS[iso2]
        marker = PARETO_COUNTRY_MARKERS[iso2]
        label  = f"{iso2} \u03b5-frontier"
        cost_k = front_eps["cost"] / 1e3
        if len(front_eps) >= 2:
            ax.plot(front_eps["ghg"], cost_k,
                    drawstyle="steps-post",
                    color=color, lw=1.7, alpha=0.95,
                    zorder=5.5, label=label)
        ax.scatter(front_eps["ghg"], cost_k,
                   s=36, color=color, marker=marker,
                   edgecolors="white", linewidths=0.5,
                   zorder=6, label=None if len(front_eps) >= 2 else label)

    # Fossil reference point
    ax.scatter([foss_ghg], [foss_cost], s=60, color="0.2", marker="*", zorder=4)
    # label: upper-right in c (2025), upper-left in d (2050)
    ftxt_x = foss_ghg + xpad * 0.12 if label_side == "right" else foss_ghg - xpad * 0.12
    ax.text(ftxt_x, foss_cost + ypad * 0.25, pw["fossil_label"],
            fontsize=8.0, color="0.3",
            va="bottom", ha="left" if label_side == "right" else "right", zorder=7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(rf"GHG intensity  [t CO$_2$ eq t$^{{-1}}$ {unit}]", labelpad=3)
    ax.set_ylabel(rf"Production cost  [k€ t$^{{-1}}$ {unit}]", labelpad=3)
    ax.yaxis.grid(True, color="0.91", lw=0.5, zorder=0)
    _spine_style(ax)


# ── Master 2×2 figure builder ─────────────────────────────────────────────────

def _draw_missing_panel(ax, title, message):
    ax.set_title(title, loc="left", weight="semibold", pad=8, fontsize=13.0)
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_pathway_figure(pw_key, out_dir, result_version, global_file, global_grid_file):
    pw   = PATHWAYS[pw_key]
    data = load_pathway_data(pw_key, result_version, global_file, global_grid_file)
    db_names = _available_db_names(data)
    source_label = "Case-study results" if result_version == "case_study" else "Global results"

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.0, 10.5),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07, right=0.97, bottom=0.06, top=0.90,
        wspace=0.28, hspace=0.18,
    )

    def _add_panel_label(ax, letter):
        ax.text(-0.08, 1.04, letter,
                transform=ax.transAxes,
                fontsize=18, fontweight="bold",
                va="bottom", ha="right", zorder=10,
                clip_on=False)

    # ── Row 0: carbon price metrics ───────────────────────────────────────────
    ax_a, ax_b = axes[0, 0], axes[0, 1]

    _panel_breakeven(ax_a, data, pw, db_names)
    ax_a.set_title(f"{pw['name']}  \u00b7  Break-even carbon price",
                   loc="left", weight="semibold", pad=8, fontsize=13.0)
    _add_panel_label(ax_a, "a")

    _panel_right(ax_b, data, pw, db_names)
    right_title = (
        f"{pw['name']}  \u00b7  Additional fuel cost due to SAF mandate (ReFuelEU)"
        if pw["is_saf"] else
        f"{pw['name']}  \u00b7  Cost gap above fossil methanol"
    )
    ax_b.set_title(right_title, loc="left", weight="semibold", pad=8, fontsize=13.0)
    _add_panel_label(ax_b, "b")

    # ── Row 1: cost–GHG positioning + epsilon-constraint Pareto overlays ─────
    # Pre-load CN/AU/US epsilon-constraint frontiers (cached per DB).  When no
    # cache is present yet the panels fall back to the original scatter-only
    # behaviour.
    pareto_scenario = getattr(make_pathway_figure, "_pareto_scenario", "hybrid")
    frontiers_by_db = {
        db: ({} if pw_key in PARETO_DISABLED_PATHWAYS
             else load_pareto_frontiers(pw_key, db, scenario=pareto_scenario))
        for db in db_names
    }
    any_frontiers = any(v for v in frontiers_by_db.values())

    bottom_specs = [("c", "right"), ("d", "left")]
    for col_idx, (letter, side) in enumerate(bottom_specs):
        ax = axes[1, col_idx]
        if col_idx < len(db_names):
            db_name = db_names[col_idx]
            fronts = frontiers_by_db.get(db_name, {})
            _panel_cost_ghg(ax, data, pw, db_name, label_side=side,
                            pareto_frontiers=fronts)
            yr = DB_LABELS.get(db_name, str(db_name))
            title_extra = f"  \u00b7  \u03b5-constraint frontier" if fronts else ""
            ax.set_title(f"{pw['name']}  \u00b7  {yr}{title_extra}",
                         loc="left", weight="semibold", pad=8, fontsize=13.0)
        else:
            _draw_missing_panel(
                ax,
                f"{pw['name']}  \u00b7  Unavailable",
                "No additional technology-year results\nwere found for this source.",
            )
        _add_panel_label(ax, letter)

    fig.text(0.07, 0.935, source_label, ha="left", va="center",
             fontsize=10.0, color="0.38")

    # ── Scenario legend for bottom panels ─────────────────────────────────────
    scen_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKERS[s], ls="None", ms=6,
               color=SCENARIO_COLORS[s], label=SCENARIO_SHORT[s])
        for s in SCENARIO_ORDER
    ]

    if any_frontiers:
        scen_handles += [
            Line2D([0], [0],
                   color=PARETO_COUNTRY_COLORS[iso2],
                   marker=PARETO_COUNTRY_MARKERS[iso2],
                   ms=6, lw=1.7,
                   label=f"{PARETO_COUNTRY_NAMES[iso2]} \u03b5-frontier")
            for iso2 in PARETO_COUNTRIES
        ]

    fig.legend(
        handles=scen_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=min(len(scen_handles), 5),
        frameon=False,
        fontsize=10.0,
    )

    suffix = "" if result_version == "case_study" else "_global"
    out_path = Path(out_dir) / f"figure_pathway_summary_{pw_key}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-pathway 2x2 summary figure: carbon price + cost-GHG positioning."
    )
    p.add_argument(
        "--pathway",
        choices=list(PATHWAYS) + ["all"],
        default="all",
        help="Pathway to plot, or 'all' to generate one figure per pathway.",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path("figs") / "pathway_summary"),
        help="Output directory.",
    )
    p.add_argument(
        "--result-version",
        choices=["case_study", "global"],
        default="case_study",
        help="Use case-study outputs or currently available global results.",
    )
    p.add_argument(
        "--global-file",
        default=str(FILE_PATH_GLOBAL_RESULTS_PTX),
        help="Hybrid/off-grid global results file (.pkl or .parquet).",
    )
    p.add_argument(
        "--global-grid-file",
        default=str(FILE_PATH_GLOBAL_RESULTS_PTX_GRID),
        help="Grid-connected global results file (.pkl or .parquet).",
    )
    p.add_argument(
        "--pareto-scenario",
        default="hybrid",
        choices=["hybrid", "grid_connected", "off_grid"],
        help="Which cached epsilon-constraint sweep (scenario) to overlay on "
             "panels c/d.  The cache is produced by plot_ptx_pareto_frontier.py.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pathways = list(PATHWAYS) if args.pathway == "all" else [args.pathway]
    make_pathway_figure._pareto_scenario = args.pareto_scenario
    for pw_key in pathways:
        make_pathway_figure(
            pw_key,
            args.out_dir,
            result_version=args.result_version,
            global_file=args.global_file,
            global_grid_file=args.global_grid_file,
        )


if __name__ == "__main__":
    main()

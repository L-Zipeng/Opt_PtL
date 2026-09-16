import argparse
import math
import pickle
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import NAME_FUTURE_DB, NAME_REF_DB, T_DAY_MEOH, T_DAY_SAF
from mapping import color_dict_cost, name_dict_fig


SCENARIO_ORDER = ["grid_connected", "hybrid", "hybrid-green", "off_grid"]
DB_LABELS = {
    NAME_REF_DB: "2025",
    NAME_FUTURE_DB: "2050",
}

PTX_PATHWAY_INFO = {
    "meoh": {
        "name": "Methanol",
        "unit": "MeOH",
        "product_column": "tCO2_tMeOH",
        "cost_column": "euro_tMeOH",
        "annual_production": T_DAY_MEOH * 365,
        "conventional_cost": 450,
        "conventional_ghg": 0.7,
        "benchmark_bands": [
            {
                "low": 500,
                "high": 700,
                "label": "Fossil methanol reference range",
                "color": "#d8d8d8",
                "alpha": 0.45,
            },
            {
                "low": 800,
                "high": 1200,
                "label": "Biomethanol reference range",
                "color": "#cfe8cf",
                "alpha": 0.45,
            },
        ],
        "reference_lines": [],
    },
    "meoh_to_saf": {
        "name": "SAF (MtJ)",
        "unit": "SAF",
        "product_column": "tCO2_tSAF",
        "cost_column": "euro_tSAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_cost": 700,
        "conventional_ghg": 3.9,   # WTW fossil jet fuel (DAC uptake credit removed from PtX GHG)
        "benchmark_bands": [
            {
                "low": 670,
                "high": 830,
                "label": "Fossil jet fuel reference range",
                "color": "#d8d8d8",
                "alpha": 0.45,
            },
            {
                "low": 2000,
                "high": 3000,
                "label": "HEFA SAF reference range",
                "color": "#f4d7a6",
                "alpha": 0.42,
            },
        ],
        "reference_lines": [
            {
                "value": 1420,
                "label": "10 Apr 2026 jet fuel price (IATA)",
                "color": "#3f3f46",
                "linestyle": "--",
                "linewidth": 1.0,
            }
        ],
    },
    "ftsaf": {
        "name": "SAF (FT)",
        "unit": "SAF",
        "product_column": "tCO2_tSAF",
        "cost_column": "euro_tSAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_cost": 700,
        "conventional_ghg": 3.9,   # WTW fossil jet fuel (DAC uptake credit removed from PtX GHG)
        "benchmark_bands": [
            {
                "low": 670,
                "high": 830,
                "label": "Fossil jet fuel reference range",
                "color": "#d8d8d8",
                "alpha": 0.45,
            },
            {
                "low": 2000,
                "high": 3000,
                "label": "HEFA SAF reference range",
                "color": "#f4d7a6",
                "alpha": 0.42,
            },
        ],
        "reference_lines": [
            {
                "value": 1420,
                "label": "10 Apr 2026 jet fuel price (IATA)",
                "color": "#3f3f46",
                "linestyle": "--",
                "linewidth": 1.0,
            }
        ],
    },
}

COST_UNIT_SCALE = 1000.0  # benchmark band values are in €/t; divide by 1000 → k€/t

# Rename legacy country labels in loaded results (e.g. after correcting config.py)
COUNTRY_RENAMES = {
    "India (Bintan)": "Indonesia (Bintan)",
}

PREFERRED_COST_ORDER = [
    "an_costs_op_co2",
    "an_costs_op_grid_abs",
    "an_costs_op_grid_inj",
    "an_costs_capex_pv",
    "an_costs_capex_wind_on",
    "an_costs_capex_bat_en",
    "an_costs_capex_bat_p",
    "an_costs_capex_grid_ins",
    "an_costs_capex_electrolyzer",
    "an_costs_capex_h2_ves",
    "an_costs_capex_co2_ves",
    "an_costs_capex_meoh",
    "an_costs_capex_mtj",
    "an_costs_capex_ftsaf",
    "an_costs_capex_dac",
    "an_costs_capex_hp",
    "an_costs_rep",
    "an_costs_om",
]

LABEL_OVERRIDES = {
    "an_costs_op_co2": "CO$_2$ tax",
    "an_costs_op_grid_abs": "Operation - grid power absorption",
    "an_costs_op_grid_inj": "Operation - grid power injection",
    "an_costs_capex_pv": "Investment - PV system",
    "an_costs_capex_wind_on": "Investment - onshore wind",
    "an_costs_capex_bat_en": "Investment - battery energy system",
    "an_costs_capex_bat_p": "Investment - battery power system",
    "an_costs_capex_grid_ins": "Investment - grid connection",
    "an_costs_capex_electrolyzer": "Investment - electrolyzer",
    "an_costs_capex_h2_ves": "Investment - H$_2$ storage",
    "an_costs_capex_co2_ves": "Investment - CO$_2$ storage",
    "an_costs_capex_dac": "Investment - DAC",
    "an_costs_capex_hp": "Investment - heat pump",
    "an_costs_capex_meoh": "Investment - methanol synthesis",
    "an_costs_capex_mtj": "Investment - MeOH-to-jet",
    "an_costs_capex_ftsaf": "Investment - FT-SAF",
    "an_costs_rep": "Replacement costs",
    "an_costs_om": "Operation & maintenance",
}

COST_COLOR_OVERRIDES = {
    "an_costs_op_co2": "#7A7A7A",
    "an_costs_op_grid_abs": "#9dc7dd",
    "an_costs_op_grid_inj": "#2A9D55",
    "an_costs_capex_pv": "#EA8C23",
    "an_costs_capex_wind_on": "#428BBA",
    "an_costs_capex_bat_en": "#FFD34D",
    "an_costs_capex_bat_p": "#F2C230",
    "an_costs_capex_grid_ins": "#4E4E4E",
    "an_costs_capex_electrolyzer": "#2CBDB5",
    "an_costs_capex_h2_ves": "#D7301F",
    "an_costs_capex_co2_ves": "#969696",
    "an_costs_capex_meoh": "#57AE47",
    "an_costs_capex_mtj": "#7B3294",
    "an_costs_capex_ftsaf": "#7B3294",
    "an_costs_capex_dac": "#5F5F5F",
    "an_costs_capex_hp": "#4CC9B0",
    "an_costs_rep": "#FEE08B",
    "an_costs_om": "#6CC0A8",
}

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.3,
        "axes.titlesize": 11.0,
        "axes.labelsize": 8.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "patch.linewidth": 0.0,
        "axes.grid": False,
        "axes.axisbelow": True,
        "mathtext.default": "regular",
    }
)


def cost_color(column_name):
    if column_name in COST_COLOR_OVERRIDES:
        return COST_COLOR_OVERRIDES[column_name]
    return color_dict_cost.get(column_name, "#9E9E9E")


def draw_reference_text(
    ax, y_value, label, *, color, zorder, y_offset_points=0.0, alpha=1.0
):
    ax.annotate(
        label,
        xy=(-0.38, y_value),
        xycoords="data",
        xytext=(0, y_offset_points),
        textcoords="offset points",
        fontsize=7.1,
        color=color,
        ha="left",
        va="center",
        weight="semibold",
        alpha=alpha,
        zorder=zorder,
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.0),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PtX cost figures with GHG markers."
    )
    parser.add_argument(
        "--pathway",
        choices=sorted(PTX_PATHWAY_INFO),
        default="meoh_to_saf",
        help="PtX pathway to plot.",
    )
    parser.add_argument(
        "--db",
        choices=["reference", "future", "both"],
        default="both",
        help="Database snapshot to plot.",
    )
    parser.add_argument(
        "--outdir",
        default="figs/cost_figs",
        help="Output directory for PNG files.",
    )
    return parser.parse_args()


def resolve_dbs(db_arg):
    if db_arg == "reference":
        return [NAME_REF_DB]
    if db_arg == "future":
        return [NAME_FUTURE_DB]
    return [NAME_REF_DB, NAME_FUTURE_DB]


def _apply_saf_dac_wtw_correction(df, ghg_col, annual_production):
    """Remove the DAC atmospheric-CO2 uptake credit to convert SAF GHG to a WTW basis.

    an_ghg_dac is negative (CO2 captured from atmosphere); adding its absolute
    value back aligns PtX SAF with the WTW fossil jet comparator (3.9 tCO2/tSAF).
    """
    if "an_ghg_dac" in df.columns:
        correction = (-df["an_ghg_dac"].astype(float) / annual_production).clip(lower=0)
        df = df.copy()
        df[ghg_col] = df[ghg_col].astype(float) + correction
    return df


def load_cost_results(pathway):
    pkl_path = Path("results") / f"case_studies_ptx_{pathway}.pkl"
    with pkl_path.open("rb") as fh:
        df = pickle.load(fh)
    info = PTX_PATHWAY_INFO.get(pathway, {})
    if info.get("unit") == "SAF":
        df = _apply_saf_dac_wtw_correction(
            df, info["product_column"], info["annual_production"]
        )
    return df


ZERO_COST_TOL = 1e-9


def globally_zero_an_cost_columns(results_dir=None):
    """
    Return an_costs_* column names that are identically zero in every row of
    every existing case_studies_ptx_<pathway>.pkl (all pathways in PTX_PATHWAY_INFO).
    """
    if results_dir is None:
        results_dir = Path("results")
    global_max_abs = {}
    for pathway in sorted(PTX_PATHWAY_INFO):
        pkl_path = results_dir / f"case_studies_ptx_{pathway}.pkl"
        if not pkl_path.is_file():
            continue
        with pkl_path.open("rb") as fh:
            df = pickle.load(fh)
        for col in df.columns:
            if not str(col).startswith("an_costs_"):
                continue
            mx = float(df[col].astype(float).abs().max())
            global_max_abs[col] = max(global_max_abs.get(col, 0.0), mx)
    return frozenset(
        col for col, mx in global_max_abs.items() if mx < ZERO_COST_TOL
    )


def ordered_columns(columns):
    preferred = [c for c in PREFERRED_COST_ORDER if c in columns]
    remaining = [c for c in columns if c not in preferred]
    return preferred + remaining


def cleaned_label(column_name):
    label = LABEL_OVERRIDES.get(column_name, name_dict_fig.get(column_name, column_name))
    return label.replace("â€“", "-").replace("–", "-")


def legend_columns(n_labels):
    if n_labels <= 6:
        return 3
    if n_labels <= 12:
        return 4
    return 5


def segments_from_index(idx_vals):
    scenario_labels = [cs.split(" | ", 1)[0] for cs in idx_vals]
    segments = []
    start_idx = 0
    for i in range(1, len(scenario_labels)):
        if scenario_labels[i] != scenario_labels[i - 1]:
            segments.append((scenario_labels[start_idx], start_idx, i - 1))
            start_idx = i
    segments.append((scenario_labels[start_idx], start_idx, len(scenario_labels) - 1))
    return segments


def compute_shared_axis_ranges(pathway, df_init, dbs):
    info = PTX_PATHWAY_INFO[pathway]
    ghg_col = info["product_column"]
    annual_production = info["annual_production"]
    conventional_cost = info["conventional_cost"] / COST_UNIT_SCALE
    conventional_ghg = info["conventional_ghg"]
    benchmark_high = (
        max(b["high"] for b in info.get("benchmark_bands", [])) / COST_UNIT_SCALE
        if info.get("benchmark_bands")
        else conventional_cost
    )

    cost_max = 0.0
    ghg_min = 0.0
    ghg_max = float(conventional_ghg)

    for db in dbs:
        if "db_name" not in df_init.index.names:
            continue
        if db not in df_init.index.get_level_values("db_name"):
            continue

        df_cost = df_init.xs(db, level="db_name").copy()
        cost_columns = [col for col in df_cost.columns if col.startswith("an_costs_")]
        if not cost_columns:
            continue

        cost_totals = (
            df_cost[cost_columns].sum(axis=1).divide(annual_production)
        )
        cost_max = max(cost_max, float(cost_totals.max()))

        ghg_values = df_cost[ghg_col].astype(float)
        ghg_min = min(ghg_min, float(ghg_values.min()))
        ghg_max = max(ghg_max, float(ghg_values.max()))

    cost_ylim = (0, max(cost_max * 1.12, benchmark_high * 1.12, conventional_cost * 1.20))
    ghg_pad = max((ghg_max - ghg_min) * 0.12, 0.6)
    ghg_ylim = (ghg_min - ghg_pad * 0.25, ghg_max + ghg_pad)
    return cost_ylim, ghg_ylim


def plot_costs(
    pathway,
    db,
    df_cost,
    out_dir,
    cost_ylim=None,
    ghg_ylim=None,
    skip_an_costs=frozenset(),
):
    info = PTX_PATHWAY_INFO[pathway]
    product_unit = info["unit"]
    product_name = info["name"]
    ghg_col = info["product_column"]
    annual_production = info["annual_production"]
    conventional_cost = info["conventional_cost"] / COST_UNIT_SCALE

    cost_columns = [
        col
        for col in df_cost.columns
        if col.startswith("an_costs_") and col not in skip_an_costs
    ]
    if not cost_columns:
        return None

    df_plot = df_cost.copy()
    df_plot[cost_columns] = df_plot[cost_columns].divide(annual_production)
    df_plot = df_plot.reset_index()
    if "country" in df_plot.columns:
        df_plot["country"] = df_plot["country"].replace(COUNTRY_RENAMES)
    df_plot["scenario"] = pd.Categorical(
        df_plot["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    df_plot = df_plot.sort_values(by=["scenario", "country"]).reset_index(drop=True)
    df_plot["country_scenario"] = df_plot["scenario"].astype(str) + " | " + df_plot["country"]

    ordered_cost_columns = ordered_columns(cost_columns)
    pivot_cost = (
        df_plot.set_index("country_scenario")[ordered_cost_columns]
        .fillna(0)
    )

    colors_for_plot = [cost_color(col) for col in pivot_cost.columns]
    totals = pivot_cost.sum(axis=1)
    max_total = float(totals.max())
    if cost_ylim is None:
        max_y = max(max_total * 1.12, conventional_cost * 1.85)
        cost_ylim = (0, max_y)
    else:
        max_y = float(cost_ylim[1])

    ghg_values = df_plot[ghg_col].astype(float)
    if ghg_ylim is None:
        ghg_min = float(min(0.0, ghg_values.min()))
        ghg_max = float(max(PTX_PATHWAY_INFO[pathway]["conventional_ghg"], ghg_values.max()))
        ghg_pad = max((ghg_max - ghg_min) * 0.12, 0.6)
        ghg_ylim = (ghg_min - ghg_pad * 0.25, ghg_max + ghg_pad)

    legend_ncol = legend_columns(len(pivot_cost.columns))
    legend_rows = math.ceil(len(pivot_cost.columns) / legend_ncol)

    fig_width = min(max(10.0, 0.24 * len(pivot_cost) + 3.4), 14.0)
    fig_height = 5.35 + 0.22 * (legend_rows - 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)

    top_margin = 0.82 - 0.028 * (legend_rows - 1)
    axes_top = max(0.73, top_margin)
    fig.subplots_adjust(
        left=0.070,
        right=0.925,
        bottom=0.215,
        top=axes_top,
    )

    pivot_cost.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.60,
        zorder=3,
        color=colors_for_plot,
        edgecolor="none",
        linewidth=0,
        legend=False,
    )

    for patch in ax.patches:
        patch.set_antialiased(False)

    ax.set_ylabel(f"{product_name} production costs [k€ t$^{{-1}}$ {product_unit}]")
    ax.set_xlabel("")

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="x", rotation=90, labelsize=7.2, length=2.2, colors="black", pad=2)
    ax.tick_params(axis="y", labelsize=7.6, colors="0.08")
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_xlim(-0.45, len(pivot_cost) - 0.55)
    ax.set_ylim(*cost_ylim)
    ax.margins(x=0.0)

    benchmark_bands = info.get("benchmark_bands", [])
    for idx, band in enumerate(benchmark_bands):
        band_low = band["low"] / COST_UNIT_SCALE
        band_high = band["high"] / COST_UNIT_SCALE
        ax.axhspan(
            band_low,
            band_high,
            color=band.get("color", "#d8d8d8"),
            alpha=band.get("alpha", 0.45),
            zorder=1,
        )
        if db == NAME_REF_DB:
            draw_reference_text(
                ax,
                (band_low + band_high) / 2,
                band["label"],
                color="0.15",
                zorder=11 + idx,
            )

    for idx, ref in enumerate(info.get("reference_lines", [])):
        y_val = ref["value"] / COST_UNIT_SCALE
        ax.axhline(
            y=y_val,
            color=ref.get("color", "#3f3f46"),
            linestyle=ref.get("linestyle", "--"),
            linewidth=ref.get("linewidth", 1.0),
            zorder=10 + idx,
        )
        if db == NAME_REF_DB:
            draw_reference_text(
                ax,
                y_val,
                ref["label"],
                color=ref.get("color", "#3f3f46"),
                zorder=12 + idx,
                y_offset_points=5.0 if "IATA" in ref["label"] else 0.0,
                alpha=0.7 if "IATA" in ref["label"] else 1.0,
            )

    handles, labels = ax.get_legend_handles_labels()
    # Legend just above the axes (not pinned to the figure top).
    legend_gap_above_axes = 0.006
    legend_anchor_y = axes_top + legend_gap_above_axes
    legend_height_est = 0.028 * legend_rows + 0.022
    title_gap_above_legend = 0.018
    fig.legend(
        handles,
        [cleaned_label(lbl) for lbl in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=legend_ncol,
        frameon=False,
        handlelength=2.1,
        handleheight=1.0,
        handletextpad=0.52,
        labelspacing=0.40,
        columnspacing=1.30,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f"{product_name} cost breakdown | {DB_LABELS.get(db, db)}",
        fontsize=12.0,
        weight="semibold",
        y=legend_anchor_y + legend_height_est + title_gap_above_legend,
    )

    segments = segments_from_index(pivot_cost.index)
    for s_idx, (scen, s_start, s_end) in enumerate(segments):
        if s_idx < len(segments) - 1:
            ax.vlines(
                x=s_end + 0.5,
                ymin=0,
                ymax=max_y,
                colors="0.70",
                lw=0.8,
                zorder=2,
            )

        ax.text(
            (s_start + s_end) / 2,
            max_y * 0.965,
            scen.capitalize().replace("_", " "),
            fontsize=8.4,
            color="0.12",
            ha="center",
            va="top",
            weight="semibold",
            zorder=8,
        )

    label_offset = max(max_y * 0.012, 0.075)
    for i, val in enumerate(totals):
        ax.text(
            i,
            float(val) + label_offset,
            f"{val:,.2f}",
            fontsize=6.8,
            ha="center",
            va="bottom",
            color="black",
            weight="semibold",
            clip_on=False,
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.80),
        )

    country_labels = [cs.split(" | ", 1)[1] for cs in pivot_cost.index]
    ax.set_xticklabels(country_labels, rotation=90, ha="center")

    ax2 = ax.twinx()
    for side in ["right", "top"]:
        ax2.spines[side].set_linewidth(0.85)
        ax2.spines[side].set_color("black")

    ax2.scatter(
        x=range(len(df_plot)),
        y=ghg_values,
        marker="D",
        s=28,
        facecolors="white",
        edgecolors="black",
        linewidths=0.7,
        zorder=13,
    )
    ax2.set_ylim(*ghg_ylim)
    ax2.set_ylabel(f"$\\diamond$ Climate change impacts [t CO$_2$ eq t$^{{-1}}$ {product_unit}]")
    ax2.tick_params(axis="y", labelsize=7.6, colors="black")
    ax2.grid(False)

    stem = f"ptx_{pathway}_{db}_costs"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    return png_path


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_init = load_cost_results(args.pathway)
    saved_paths = []
    dbs = resolve_dbs(args.db)
    skip_an_costs = globally_zero_an_cost_columns(Path("results"))
    if skip_an_costs:
        print(
            "Omitting cost components (all zero across all pathway case studies): "
            + ", ".join(sorted(skip_an_costs))
        )
    cost_ylim, ghg_ylim = compute_shared_axis_ranges(args.pathway, df_init, dbs)

    for db in dbs:
        if "db_name" not in df_init.index.names:
            continue
        if db not in df_init.index.get_level_values("db_name"):
            continue

        df_cost = df_init.xs(db, level="db_name").copy()
        print(f"{args.pathway} | {db} | costs")
        saved_path = plot_costs(
            args.pathway,
            db,
            df_cost,
            out_dir,
            cost_ylim=cost_ylim,
            ghg_ylim=ghg_ylim,
            skip_an_costs=skip_an_costs,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

    if saved_paths:
        print(f"Saved {len(saved_paths)} figure(s) to {out_dir}")
    else:
        print("No figures were generated.")


if __name__ == "__main__":
    main()

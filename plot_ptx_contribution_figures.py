import argparse
import math
import pickle
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import NAME_FUTURE_DB, NAME_REF_DB, T_DAY_MEOH, T_DAY_SAF
from mapping import dict_units
from plot_ptx_cost_figures import COST_COLOR_OVERRIDES


SHOW_INTERNAL_TITLE = False
HYBRID_GREEN_CAP = 1.4
EXCLUDE_DAC_FROM_CLIMATE = True
SCENARIO_ORDER = ["grid_connected", "hybrid", "hybrid-green", "off_grid"]

PTX_PATHWAY_INFO = {
    "meoh": {
        "unit": "MeOH",
        "annual_production": T_DAY_MEOH * 365,
        "conventional_ghg": 0.7,
    },
    "meoh_to_saf": {
        "unit": "SAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_ghg": 3.9,   # WTW fossil jet fuel; EXCLUDE_DAC_FROM_CLIMATE removes the DAC uptake column
    },
    "ftsaf": {
        "unit": "SAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_ghg": 3.9,   # WTW fossil jet fuel; EXCLUDE_DAC_FROM_CLIMATE removes the DAC uptake column
    },
}

# Map LCA contributor labels to plot_ptx_cost_figures.COST_COLOR_OVERRIDES keys (cross-figure consistency).
CONTRIBUTOR_TO_COST_COLOR_KEY = {
    "Operation, electricity": "an_costs_op_grid_abs",
    "Construction, grid electricity network": "an_costs_capex_grid_ins",
    "Construction, PV": "an_costs_capex_pv",
    "Construction, wind onshore": "an_costs_capex_wind_on",
    "Construction, electrolyzer": "an_costs_capex_electrolyzer",
    "Construction, battery": "an_costs_capex_bat_en",
    "Construction, FT-SAF": "an_costs_capex_ftsaf",
    "Construction, MTJ": "an_costs_capex_mtj",
    "Construction, methanol synthesis": "an_costs_capex_meoh",
    "Construction, DAC": "an_costs_capex_dac",
    "Construction, H2 storage": "an_costs_capex_h2_ves",
    "Construction, CO2 compression": "an_costs_capex_co2_ves",
    "Construction, RWGS": "an_costs_capex_hp",
}


def lca_contributor_color(contributor_name):
    key = CONTRIBUTOR_TO_COST_COLOR_KEY.get(contributor_name)
    if key is not None and key in COST_COLOR_OVERRIDES:
        return COST_COLOR_OVERRIDES[key]
    return "#9E9E9E"

PREFERRED_ORDER = [
    "Construction, DAC",
    "Construction, FT-SAF",
    "Construction, PV",
    "Construction, electrolyzer",
    "Construction, grid electricity network",
    "Construction, wind onshore",
    "Construction, battery",
    "Construction, MTJ",
    "Construction, methanol synthesis",
    "Construction, RWGS",
    "Construction, H2 storage",
    "Construction, CO2 compression",
    "Operation, electricity",
]

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.3,
        "axes.titlesize": 9.8,
        "axes.labelsize": 8.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.4,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PtX LCA contribution figures as PNG files."
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
        "--category",
        default=None,
        help="Optional single impact category to plot, for example 'climate change'.",
    )
    parser.add_argument(
        "--outdir",
        default="figs/contri_figs",
        help="Output directory for PNG files.",
    )
    return parser.parse_args()


def get_unit(category, unit_dict=dict_units):
    for key, unit in unit_dict.items():
        if category.lower() == key[1].lower():
            return unit
    return None


def clean_category_name(category):
    return category.replace("_", " ").replace(":", "").strip()


def ordered_contributors(columns):
    preferred = [c for c in PREFERRED_ORDER if c in columns]
    remaining = [c for c in columns if c not in preferred]
    return preferred + remaining


def legend_columns(n_labels):
    if n_labels <= 5:
        return n_labels
    if n_labels <= 8:
        return 4
    return 5


def format_total_label(val):
    return f"{val:.1f}"


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


def rebalance_hybrid_green_rows(pivot_df, total_cap=HYBRID_GREEN_CAP):
    plot_df = pivot_df.copy()
    hg_idx = [idx for idx in plot_df.index if str(idx).startswith("hybrid-green |")]
    for idx in hg_idx:
        row = plot_df.loc[idx].astype(float)
        net_total = float(row.sum())
        if not np.isfinite(net_total) or net_total <= 0:
            plot_df.loc[idx, :] = 0.0
            continue

        positives = row.clip(lower=0)
        positive_total = float(positives.sum())
        if positive_total <= 0:
            plot_df.loc[idx, :] = 0.0
            continue

        target_total = min(net_total, total_cap)
        plot_df.loc[idx, :] = positives * (target_total / positive_total)
    return plot_df


def load_lca_results(pathway):
    pkl_path = Path("results") / f"case_studies_lca_ptx_{pathway}.pkl"
    with pkl_path.open("rb") as fh:
        return pickle.load(fh)


def resolve_dbs(db_arg):
    if db_arg == "reference":
        return [NAME_REF_DB]
    if db_arg == "future":
        return [NAME_FUTURE_DB]
    return [NAME_REF_DB, NAME_FUTURE_DB]


def build_pivot(data_cat, annual_production):
    grouped = (
        data_cat.groupby(["scenario", "country", "contributor"])["impact_value"]
        .sum()
        .reset_index()
    )

    grouped["scenario"] = pd.Categorical(
        grouped["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    grouped = grouped.sort_values(by=["scenario", "country"])
    grouped["impact_value"] = grouped["impact_value"].divide(annual_production * 1e3)
    grouped["country_scenario"] = (
        grouped["scenario"].astype(str) + " | " + grouped["country"]
    )
    ordered_index = grouped["country_scenario"].drop_duplicates().tolist()

    pivot = grouped.pivot_table(
        index="country_scenario",
        columns="contributor",
        values="impact_value",
        fill_value=0,
    ).reindex(ordered_index)

    if pivot.empty:
        return grouped, pivot

    pivot = pivot.reindex(columns=ordered_contributors(list(pivot.columns)), fill_value=0)
    return grouped, pivot


def _prepare_climate_pivot(pivot):
    pivot_prepared = pivot.copy()
    if EXCLUDE_DAC_FROM_CLIMATE and "Construction, DAC" in pivot_prepared.columns:
        pivot_prepared = pivot_prepared.drop(columns=["Construction, DAC"])
    return pivot_prepared


def _compute_plot_max_y(category, pivot):
    pivot_plot = pivot.copy()
    if category == "climate change":
        pivot_plot = _prepare_climate_pivot(pivot_plot)
        # Plot only positives for hybrid-green so the visible stack matches the 1.4 cap.
        pivot_plot = rebalance_hybrid_green_rows(pivot_plot)

    totals = pivot_plot.sum(axis=1)
    raw_totals = pivot.sum(axis=1)
    max_total = float(max(totals.max(), raw_totals.max()))
    return max_total * (1.14 if category == "climate change" else 1.10)


def compute_shared_max_y(df_init, dbs, category, annual_production):
    max_y_values = []
    for db in dbs:
        if "db_name" in df_init.index.names and db not in df_init.index.get_level_values("db_name"):
            continue
        df = df_init.xs(db, level="db_name").copy()
        if category not in df.index.get_level_values("category"):
            continue
        data_cat = df.xs(category, level="category")
        _, pivot = build_pivot(data_cat, annual_production)
        if pivot.empty:
            continue
        max_y_values.append(_compute_plot_max_y(category, pivot))
    return max(max_y_values) if max_y_values else None


def plot_category(pathway, db, category, pivot, product_unit, conventional_ghg, out_dir, max_y_override=None):
    pivot_plot = pivot.copy()
    if category == "climate change":
        pivot_plot = _prepare_climate_pivot(pivot_plot)
        pivot = _prepare_climate_pivot(pivot)
        # Plot only positives for hybrid-green so the visible stack matches the 1.4 cap.
        pivot_plot = rebalance_hybrid_green_rows(pivot_plot)

    colors_for_plot = [lca_contributor_color(c) for c in pivot_plot.columns]
    totals = pivot_plot.sum(axis=1)
    raw_totals = pivot.sum(axis=1)
    max_total = float(max(totals.max(), raw_totals.max()))
    max_y = max_y_override if max_y_override is not None else max_total * (1.14 if category == "climate change" else 1.10)

    n_labels = len(pivot_plot.columns)
    legend_ncol = legend_columns(n_labels)
    legend_rows = math.ceil(n_labels / legend_ncol)

    fig_width = min(max(8.0, 0.225 * len(pivot_plot) + 2.5), 11.5)
    base_height = 4.2 if category == "climate change" else 4.0
    fig_height = base_height + 0.16 * (legend_rows - 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)

    top_margin = 0.84 - 0.040 * (legend_rows - 1)
    if SHOW_INTERNAL_TITLE:
        top_margin -= 0.03

    fig.subplots_adjust(
        left=0.072,
        right=0.995,
        bottom=0.225,
        top=max(0.74, top_margin),
    )

    pivot_plot.plot(
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

    if SHOW_INTERNAL_TITLE:
        ax.set_title(
            f"Contribution analysis on {clean_category_name(category)}",
            pad=8,
            weight="semibold",
        )
    else:
        ax.set_title("")

    if category == "climate change":
        ax.set_ylabel(
            f"Climate change impacts [t CO$_2$ eq t$^{{-1}}$ {product_unit}]"
        )
    else:
        impact_unit = f"{get_unit(category)} kg$^{{-1}}$ {product_unit}"
        ax.set_ylabel(
            f"Impacts on {clean_category_name(category).capitalize()}\n"
            f"[{impact_unit}]"
        )
    ax.set_xlabel("")

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="x", rotation=90, labelsize=7.2, length=2.2, colors="black", pad=2)
    ax.tick_params(axis="y", labelsize=7.6, colors="0.08")
    ax.yaxis.grid(True, color="0.90", linewidth=0.55)
    ax.xaxis.grid(False)

    country_labels = [cs.split(" | ", 1)[1] for cs in pivot_plot.index]
    ax.set_xticklabels(country_labels, rotation=90, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=legend_ncol,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.42,
        columnspacing=0.95,
        borderaxespad=0.0,
    )

    segments = segments_from_index(pivot_plot.index)
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
            max_y * 0.98,
            scen.capitalize().replace("_", " "),
            fontsize=8.0,
            color="0.28",
            ha="center",
            va="top",
            weight="semibold",
            zorder=8,
        )

    if category == "climate change":
        for scen, s_start, s_end in segments:
            if scen == "hybrid-green":
                ax.hlines(
                    HYBRID_GREEN_CAP,
                    xmin=s_start - 0.35,
                    xmax=s_end + 0.35,
                    colors="0.45",
                    linewidth=0.75,
                    linestyles=(0, (2, 2)),
                    zorder=2,
                )

        label_offset = max(max_y * 0.010, 0.035)
        for i, val in enumerate(totals):
            ax.text(
                i,
                float(val) + label_offset,
                format_total_label(float(val)),
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
                weight="semibold",
                clip_on=False,
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78),
            )

        ax.axhline(
            y=conventional_ghg,
            color="0.25",
            linewidth=0.95,
            linestyle=(0, (4, 2)),
            zorder=2,
        )

        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.995,
            conventional_ghg + max_y * 0.05,
            f"Fossil Jet Fuel ({conventional_ghg:.2f} t CO$_2$ eq t$^{{-1}}$ fuel)",
            transform=trans,
            fontsize=7.2,
            color="0.20",
            ha="right",
            va="bottom",
            style="italic",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.88),
        )

    ax.set_ylim(0, max_y)
    ax.set_xlim(-0.45, len(pivot_plot) - 0.55)
    ax.margins(x=0.0)

    # Zoom inset: reference (2025) only — 2050 bars are already on a compact scale without extra whitespace.
    if (
        category == "climate change"
        and db != NAME_FUTURE_DB
        and len(segments) >= 2
    ):
        start_zoom = segments[-2][1]
        end_zoom = segments[-1][2]
        x1 = start_zoom - 0.35
        x2 = end_zoom + 0.35

        selected_totals = totals.iloc[start_zoom : end_zoom + 1]
        selected_max = float(selected_totals.max())
        y2 = max(conventional_ghg * 1.10, selected_max * 1.10, 3.4)

        x_left_main, x_right_main = ax.get_xlim()
        x_span = x_right_main - x_left_main
        inset_left = max(0.02, (x1 - x_left_main) / x_span)
        inset_right = min(0.995, (x2 - x_left_main) / x_span)
        inset_width = max(0.24, inset_right - inset_left)
        if inset_left + inset_width > 0.995:
            inset_width = 0.995 - inset_left

        ax_inset = inset_axes(
            ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(inset_left, 0.36, inset_width, 0.38),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )

        pivot_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax_inset,
            width=0.60,
            legend=False,
            color=colors_for_plot,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )

        for patch in ax_inset.patches:
            patch.set_antialiased(False)

        ax_inset.set_xlim(x1 - 0.15, x2 + 0.15)
        ax_inset.set_ylim(0, y2)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.tick_params(axis="x", which="both", labelbottom=False, bottom=False, length=0)
        ax_inset.tick_params(axis="y", labelsize=6.0, colors="0.15", length=1.8)

        if y2 <= 4.2:
            ax_inset.set_yticks([0, 1, 2, 3, 4])
        else:
            ax_inset.set_yticks(np.arange(0, np.ceil(y2) + 0.1, 1.0))

        ax_inset.yaxis.grid(True, color="0.93", linewidth=0.45)
        ax_inset.xaxis.grid(False)
        ax_inset.set_facecolor("white")
        ax_inset.patch.set_alpha(0.98)

        for side in ["left", "right", "top", "bottom"]:
            ax_inset.spines[side].set_visible(True)
            ax_inset.spines[side].set_linewidth(0.8)
            ax_inset.spines[side].set_color("black")

        if conventional_ghg < y2:
            ax_inset.axhline(
                y=conventional_ghg,
                color="0.30",
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                zorder=2,
            )

        hg_seg = next((seg for seg in segments if seg[0] == "hybrid-green"), None)
        if hg_seg is not None:
            _, hg_start, hg_end = hg_seg
            if not (hg_end < start_zoom or hg_start > end_zoom):
                ax_inset.hlines(
                    HYBRID_GREEN_CAP,
                    xmin=hg_start - 0.35,
                    xmax=hg_end + 0.35,
                    colors="0.35",
                    linewidth=0.8,
                    linestyles=(0, (2, 2)),
                    zorder=2,
                )

        inset_offset = max(y2 * 0.018, 0.04)
        for j, val in enumerate(selected_totals):
            x_pos = start_zoom + j
            ax_inset.text(
                x_pos,
                float(val) + inset_offset,
                format_total_label(float(val)),
                fontsize=5.7,
                ha="center",
                va="bottom",
                color="black",
                weight="semibold",
                clip_on=False,
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.74),
            )

    stem = f"ptx_{pathway}_{db}_{category.replace(' ', '_').replace('/', '_').replace(':', '')}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    return png_path


def main():
    args = parse_args()

    pathway_info = PTX_PATHWAY_INFO[args.pathway]
    product_unit = pathway_info["unit"]
    annual_production = pathway_info["annual_production"]
    conventional_ghg = pathway_info["conventional_ghg"]
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_init = load_lca_results(args.pathway).rename(
        columns={"multi_energy_system": "impact_value"}
    )
    _country_renames = {"India (Bintan)": "Indonesia (Bintan)"}
    if "country" in df_init.columns:
        df_init["country"] = df_init["country"].replace(_country_renames)
    elif "country" in df_init.index.names:
        new_idx = df_init.index.to_frame(index=False)
        new_idx["country"] = new_idx["country"].replace(_country_renames)
        df_init.index = pd.MultiIndex.from_frame(new_idx)

    dbs = resolve_dbs(args.db)
    saved_paths = []

    for db in dbs:
        if "db_name" in df_init.index.names and db not in df_init.index.get_level_values("db_name"):
            continue

        df = df_init.xs(db, level="db_name").copy()
        grouped_categories = list(df.groupby("category"))

        for category, data_cat in grouped_categories:
            if args.category and category != args.category:
                continue

            print(f"{args.pathway} | {db} | {category}")
            _, pivot = build_pivot(data_cat, annual_production)
            if pivot.empty:
                continue

            max_y_override = None
            if category == "climate change" and len(dbs) > 1:
                max_y_override = compute_shared_max_y(df_init, dbs, category, annual_production)

            saved_path = plot_category(
                args.pathway,
                db,
                category,
                pivot,
                product_unit,
                conventional_ghg,
                out_dir,
                max_y_override=max_y_override,
            )
            saved_paths.append(saved_path)

    if saved_paths:
        print(f"Saved {len(saved_paths)} figure(s) to {out_dir}")
    else:
        print("No figures were generated.")


if __name__ == "__main__":
    main()

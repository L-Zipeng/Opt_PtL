import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import FILE_PATH_SENS_ANALYSIS_PTX, LOCATIONS_SENS, PTX_PATHWAYS


SCENARIO_ORDER = ["grid_connected", "hybrid", "off_grid"]
SCENARIO_LABELS = {
    "grid_connected": "Grid connected",
    "hybrid": "Hybrid",
    "off_grid": "Off-grid",
}

PATHWAY_INFO = {
    "meoh": {
        "name": "Methanol",
        "cost_column": "euro_tMeOH",
    },
    "meoh_to_saf": {
        "name": "SAF (MtJ)",
        "cost_column": "euro_tSAF",
    },
    "ftsaf": {
        "name": "SAF (FT)",
        "cost_column": "euro_tSAF",
    },
}

IRRELEVANT_SENSITIVITIES_BY_PATHWAY = {
    "meoh": {"mtj_capex", "ftsaf_capex"},
    "meoh_to_saf": {"ftsaf_capex"},
    "ftsaf": {"meoh_capex", "mtj_capex"},
}

PUBLISHED_SENSITIVITIES_BY_PATHWAY = {
    "meoh": {
        "electr_eff",
        "power_prices",
        "electr_capex",
        "dr",
        "cf_wind",
        "wind_on_capex",
        "cf_pv",
        "pv_capex",
        "dac_capex",
        "meoh_capex",
    },
    "meoh_to_saf": {
        "electr_eff",
        "power_prices",
        "electr_capex",
        "dr",
        "cf_wind",
        "wind_on_capex",
        "cf_pv",
        "pv_capex",
        "meoh_capex",
        "mtj_capex",
    },
    "ftsaf": {
        "electr_eff",
        "power_prices",
        "electr_capex",
        "dr",
        "cf_wind",
        "wind_on_capex",
        "cf_pv",
        "pv_capex",
        "dac_capex",
        "ftsaf_capex",
    },
}

# Match 8_create_figures.ipynb §3 (sensitivity case studies) label wording.
SENS_LABELS_MAP = {
    "power_prices": "Grid power price",
    "dr": "WACC",
    "dac_capex": "CAPEX – DAC",
    "meoh_capex": "CAPEX – MeOH reactor",
    "mtj_capex": "CAPEX – MtJ reactor",
    "ftsaf_capex": "CAPEX – FT-SAF reactor",
    "pv_capex": "CAPEX – Solar PV",
    "wind_on_capex": "CAPEX – Onshore wind",
    "electr_eff": "Electrolyzer efficiency",
    "electr_capex": "CAPEX – Electrolyzer",
    "cf_wind": "Capacity factor – Onshore wind",
    "cf_pv": "Capacity factor – Solar PV",
}

# Distinct qualitative hues (one per parameter row). Paired encoding uses same hue:
# −20% = lighter fill; +20% = stronger fill + thin edge (no hatch — different from notebook overlay).
PUBLICATION_SENS_COLORS = [
    "#3D5A80",
    "#EE6C4D",
    "#3A7CA5",
    "#98C1D9",
    "#293241",
    "#E0A458",
    "#7B9E89",
    "#B8B8D1",
    "#5C677D",
    "#9B6B9E",
    "#4C956C",
    "#D4A373",
    "#718355",
    "#E07A5F",
    "#81B29A",
    "#6D6875",
]

FIXED_SENSITIVITY_COLOR_ORDER = [
    "Electrolyzer efficiency",
    "Grid power price",
    "WACC",
    "CAPEX – Electrolyzer",
    "Capacity factor – Onshore wind",
    "CAPEX – Onshore wind",
    "Capacity factor – Solar PV",
    "CAPEX – Solar PV",
    "CAPEX – MeOH reactor",
    "CAPEX – MtJ reactor",
    "CAPEX – FT-SAF reactor",
    "CAPEX – DAC",
]

SENSITIVITY_FIXED_COLORS = {
    label: PUBLICATION_SENS_COLORS[i % len(PUBLICATION_SENS_COLORS)]
    for i, label in enumerate(FIXED_SENSITIVITY_COLOR_ORDER)
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.titlesize": 11.2,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.7,
        "axes.linewidth": 0.65,
        "axes.grid": False,
        "patch.linewidth": 0.0,
        "mathtext.default": "regular",
    }
)

# Alternating column panel backgrounds (readability without heavy borders).
PANEL_FACE_ALT = "#f5f6f8"
PANEL_FACE_MAIN = "#ffffff"


def clean_country_name(country):
    country = str(country)
    if " (" in country:
        return country.split(" (", 1)[0]
    return country


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate standalone publication-style PtX sensitivity figures."
    )
    parser.add_argument("--pathway", choices=sorted(PTX_PATHWAYS), default="ftsaf")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=[clean_country_name(c) for c, _, _, _ in LOCATIONS_SENS],
        help="Country names to include. Defaults to sensitivity case-study countries.",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path("figs") / "sensitivity_figs"),
        help="Output directory.",
    )
    return parser.parse_args()


def load_sensitivity_results():
    path = Path(FILE_PATH_SENS_ANALYSIS_PTX)
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity results file not found: {path}")

    df = pd.read_pickle(path)
    if df.empty:
        raise ValueError(f"Sensitivity results file is empty: {path}")

    if "ptx_pathway" not in df.index.names:
        if "ptx_pathway" in df.columns:
            df = df.set_index(["country", "iso2", "scenario", "ptx_pathway", "sensitivity", "factor"])
        else:
            raise ValueError("Sensitivity results do not contain 'ptx_pathway'.")

    return df.reset_index()


def prepare_plot_table(pathway, countries):
    info = PATHWAY_INFO[pathway]
    countries = [clean_country_name(c) for c in countries]
    sens_df = load_sensitivity_results()
    sens_df = sens_df[sens_df["ptx_pathway"] == pathway].copy()
    sens_df = sens_df[
        ~sens_df["sensitivity"].isin(IRRELEVANT_SENSITIVITIES_BY_PATHWAY.get(pathway, set()))
    ].copy()
    sens_df["country"] = sens_df["country"].map(clean_country_name)
    sens_df = sens_df[sens_df["country"].isin(countries)].copy()
    sens_df = sens_df[
        sens_df["sensitivity"].isin(PUBLISHED_SENSITIVITIES_BY_PATHWAY.get(pathway, set()))
    ].copy()

    if sens_df.empty:
        raise ValueError(f"No sensitivity rows found for pathway={pathway} and countries={countries}")

    df = sens_df.copy()
    cost_col = info["cost_column"]
    pair_cols = ["country", "iso2", "scenario", "ptx_pathway", "sensitivity"]
    df["pair_reference_cost"] = df.groupby(pair_cols)[cost_col].transform("mean")
    df["delta_pct"] = (df[cost_col] / df["pair_reference_cost"] - 1.0) * 100.0
    df["sensitivity_pretty"] = df["sensitivity"].map(SENS_LABELS_MAP).fillna(df["sensitivity"])
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values(["scenario", "country", "sensitivity_pretty", "factor"]).reset_index(drop=True)

    return df


def global_sensitivity_order(df):
    summary = (
        df.groupby("sensitivity_pretty")["delta_pct"]
        .apply(lambda s: float(np.nanmax(np.abs(s))))
        .rename("max_abs_delta")
        .reset_index()
        .sort_values("max_abs_delta", ascending=True)
    )
    return summary["sensitivity_pretty"].tolist()


def build_case_table(df, country, scenario, sens_order):
    sub = df[(df["country"] == country) & (df["scenario"] == scenario)].copy()
    if sub.empty:
        return pd.DataFrame(index=sens_order)

    table = (
        sub.pivot_table(
            index="sensitivity_pretty",
            columns="factor",
            values="delta_pct",
            aggfunc="first",
        )
        .reindex(sens_order)
    )
    # Normalize column keys to 0.8 / 1.2 (float noise from merges)
    col_map = {}
    for c in table.columns:
        try:
            fv = float(c)
        except (TypeError, ValueError):
            continue
        if abs(fv - 0.8) < 0.02:
            col_map[c] = 0.8
        elif abs(fv - 1.2) < 0.02:
            col_map[c] = 1.2
    if col_map:
        table = table.rename(columns=col_map)
    return table


def compute_xlim(df):
    max_abs = float(np.nanmax(np.abs(df["delta_pct"])))
    max_abs = max(max_abs, 5.0)
    return (-1.12 * max_abs, 1.12 * max_abs)


def plot_sensitivity_figure(pathway, countries, outdir):
    info = PATHWAY_INFO[pathway]
    df = prepare_plot_table(pathway, countries)
    sens_order = global_sensitivity_order(df)

    scenarios_present = [s for s in SCENARIO_ORDER if s in df["scenario"].dropna().unique()]
    countries_present = [c for c in countries if c in df["country"].unique()]
    if not scenarios_present or not countries_present:
        raise ValueError("No matching sensitivity cases to plot.")

    xlim = compute_xlim(df)
    n_rows = len(countries_present)
    n_cols = len(scenarios_present)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.10 * n_cols + 0.25, 0.21 * len(sens_order) * n_rows + 1.35),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)
    # Compact publication layout: tighter panels, less empty space.
    fig.subplots_adjust(left=0.165, right=0.992, bottom=0.115, top=0.965, wspace=0.08, hspace=0.14)

    y = np.arange(len(sens_order))
    y_off = 0.20
    bar_h = 0.34
    row_colors = [
        SENSITIVITY_FIXED_COLORS.get(
            label,
            PUBLICATION_SENS_COLORS[i % len(PUBLICATION_SENS_COLORS)],
        )
        for i, label in enumerate(sens_order)
    ]

    for r, country in enumerate(countries_present):
        for c, scenario in enumerate(scenarios_present):
            ax = axes[r, c]
            ax.set_facecolor(PANEL_FACE_ALT if c % 2 == 0 else PANEL_FACE_MAIN)

            table = build_case_table(df, country, scenario, sens_order)

            low = table.get(0.8, pd.Series(index=sens_order, dtype=float)).fillna(0.0).to_numpy()
            high = table.get(1.2, pd.Series(index=sens_order, dtype=float)).fillna(0.0).to_numpy()

            ax.axvline(
                0,
                color="#9aa5b1",
                lw=0.9,
                ls=(0, (3.5, 3)),
                zorder=1,
            )
            # Paired bars per parameter (offset): lighter = −20%, stronger = +20%
            ax.barh(
                y - y_off,
                low,
                height=bar_h,
                color=row_colors,
                alpha=0.42,
                edgecolor="none",
                zorder=3,
                linewidth=0,
            )
            ax.barh(
                y + y_off,
                high,
                height=bar_h,
                color=row_colors,
                alpha=0.88,
                edgecolor="0.28",
                linewidth=0.45,
                zorder=4,
            )

            ax.set_xlim(*xlim)
            ax.xaxis.grid(True, color="#e4e6ea", linewidth=0.5, linestyle="-", zorder=0)
            ax.yaxis.grid(False)
            ax.tick_params(axis="x", length=2.2, colors="0.2")
            ax.tick_params(axis="y", length=0, colors="0.15")

            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            for side in ["left", "bottom"]:
                ax.spines[side].set_linewidth(0.65)
                ax.spines[side].set_color("#c5cad3")

            if r == 0:
                ax.set_title(
                    SCENARIO_LABELS[scenario],
                    fontsize=11.0,
                    weight="semibold",
                    pad=6,
                    color="0.15",
                )
            if c == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(sens_order)
                ax.set_ylabel(
                    country,
                    rotation=90,
                    fontsize=10.8,
                    weight="semibold",
                    labelpad=14,
                    color="0.12",
                )
            else:
                ax.set_yticks(y)
                ax.tick_params(labelleft=False)

            if r == n_rows - 1:
                ax.set_xlabel(
                    r"Cost change vs.\ pair midpoint [$\Delta$%]",
                    fontsize=8.7,
                    color="0.25",
                    labelpad=4,
                )

    _demo = PUBLICATION_SENS_COLORS[0]
    legend_handles = [
        Patch(
            facecolor=_demo,
            edgecolor="none",
            alpha=0.42,
            label=r"$-$20% parameter change",
        ),
        Patch(
            facecolor=_demo,
            edgecolor="0.28",
            alpha=0.88,
            linewidth=0.45,
            label=r"+20% parameter change",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#dde1e8",
        facecolor="#fafbfc",
        handlelength=1.75,
        handletextpad=0.5,
        columnspacing=1.85,
        borderpad=0.45,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"sensitivity_cost_response_{pathway}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    out_path = plot_sensitivity_figure(args.pathway, args.countries, outdir)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

import argparse
import pickle
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from config import NAME_FUTURE_DB, NAME_REF_DB, T_DAY_MEOH, T_DAY_SAF
from mapping import column_dict_categories


SCENARIO_ORDER = ["grid_connected", "hybrid", "hybrid-green", "off_grid"]
SCENARIO_LABELS = {
    "grid_connected": "Grid connected",
    "hybrid": "Hybrid",
    "hybrid-green": "Hybrid-green",
    "off_grid": "Off-grid",
}
SCENARIO_COLORS = {
    "grid_connected": "#355c7d",
    "hybrid": "#e07a2d",
    "hybrid-green": "#4b8f5d",
    "off_grid": "#2a9d8f",
}
DB_LABELS = {
    NAME_REF_DB: "2025",
    NAME_FUTURE_DB: "2050",
}

PATHWAY_INFO = {
    "meoh": {
        "name": "Methanol",
        "unit": "MeOH",
        "cost_col": "euro_tMeOH",
        "cost_file": Path("results") / "case_studies_ptx_meoh.pkl",
        "lca_file": Path("results") / "case_studies_lca_ptx_meoh.pkl",
        "annual_production": T_DAY_MEOH * 365,
    },
    "meoh_to_saf": {
        "name": "SAF (MtJ)",
        "unit": "SAF",
        "cost_col": "euro_tSAF",
        "cost_file": Path("results") / "case_studies_ptx_meoh_to_saf.pkl",
        "lca_file": Path("results") / "case_studies_lca_ptx_meoh_to_saf.pkl",
        "annual_production": T_DAY_SAF * 365,
    },
    "ftsaf": {
        "name": "SAF (FT)",
        "unit": "SAF",
        "cost_col": "euro_tSAF",
        "cost_file": Path("results") / "case_studies_ptx_ftsaf_merged.pkl",
        "lca_file": Path("results") / "case_studies_lca_ptx_ftsaf_merged.pkl",
        "annual_production": T_DAY_SAF * 365,
    },
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.edgecolor": "#4a4a4a",
        "axes.linewidth": 1.0,
        "grid.color": "#d9dde3",
        "grid.linewidth": 0.8,
        "mathtext.default": "regular",
    }
)


def radar_factory(num_vars, frame="polygon"):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="none")
            raise ValueError(f"unknown frame {frame}")

        def draw(self, renderer):
            if frame == "polygon":
                for gl in self.yaxis.get_gridlines():
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            spine = Spine(axes=self, spine_type="circle", path=MplPath.unit_regular_polygon(num_vars))
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta


def clean_country_name(country):
    return re.sub(r" \(.*\)", "", str(country)).strip()


def ordered_category_labels(columns):
    ordered = []
    for raw, pretty in column_dict_categories.items():
        if pretty in columns:
            ordered.append(pretty)
    ordered.extend([c for c in columns if c not in ordered])
    return ordered


def load_pathway_frames(pathway):
    info = PATHWAY_INFO[pathway]
    with info["cost_file"].open("rb") as f:
        cost_df = pickle.load(f)
    with info["lca_file"].open("rb") as f:
        lca_df = pickle.load(f)
    return info, cost_df, lca_df


def prepare_plot_data(pathway, db_name, countries):
    info, cost_df, lca_df = load_pathway_frames(pathway)

    lca = lca_df.rename(columns={"multi_energy_system": "impact_value"}).reset_index().copy()
    lca = lca[lca["db_name"] == db_name].copy()
    lca["country_clean"] = lca["country"].map(clean_country_name)
    lca = lca[lca["country_clean"].isin(countries)].copy()

    grouped = (
        lca.groupby(["country_clean", "scenario", "category"], as_index=False)["impact_value"]
        .sum()
    )
    grouped["category_short"] = grouped["category"].map(column_dict_categories).fillna(grouped["category"])

    pivot = grouped.pivot_table(
        index=["country_clean", "scenario"],
        columns="category_short",
        values="impact_value",
        aggfunc="sum",
        fill_value=0.0,
    )
    if pivot.empty:
        raise ValueError(f"No LCA data found for pathway={pathway}, db={db_name}, countries={countries}")

    ordered_cols = ordered_category_labels(list(pivot.columns))
    pivot = pivot.reindex(columns=ordered_cols)

    category_max = pivot.max(axis=0).replace(0, np.nan)
    normalized = pivot.divide(category_max, axis=1).fillna(0.0).clip(lower=0.0, upper=1.0)

    cost = cost_df.reset_index().copy()
    cost = cost[cost["db_name"] == db_name].copy()
    cost["country_clean"] = cost["country"].map(clean_country_name)
    cost = cost[cost["country_clean"].isin(countries)].copy()
    cost_lookup = cost.set_index(["country_clean", "scenario"])[info["cost_col"]]

    countries_present = [c for c in countries if c in normalized.index.get_level_values("country_clean")]
    scenarios_present = [s for s in SCENARIO_ORDER if s in normalized.index.get_level_values("scenario")]

    return info, normalized, cost_lookup, countries_present, scenarios_present


def style_radar_axis(ax, theta, labels, show_rlabels):
    ax.set_ylim(0, 1.0)
    ax.set_rgrids(
        [0.2, 0.4, 0.6, 0.8],
        labels=["0.2", "0.4", "0.6", "0.8"] if show_rlabels else ["", "", "", ""],
        angle=90,
    )
    ax.set_varlabels(labels)
    ax.grid(True, color="#d9dde3", linewidth=0.8)
    ax.spines["polar"].set_color("#8e98a3")
    ax.spines["polar"].set_linewidth(1.0)
    for label in ax.get_xticklabels():
        label.set_fontsize(8.0)
        label.set_color("#2b2b2b")
    for label in ax.get_yticklabels():
        label.set_fontsize(7.2)
        label.set_color("#6b7280")


def plot_spider_figure(pathway, db_name, countries, out_dir):
    info, normalized, cost_lookup, countries_present, scenarios_present = prepare_plot_data(pathway, db_name, countries)
    theta = radar_factory(len(normalized.columns), frame="polygon")

    n_rows = len(countries_present)
    n_cols = len(scenarios_present)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.7 * n_cols + 1.2, 2.55 * n_rows + 1.0),
        subplot_kw={"projection": "radar"},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.91, wspace=0.16, hspace=0.34)

    for r, country in enumerate(countries_present):
        for c, scenario in enumerate(scenarios_present):
            ax = axes[r][c]
            key = (country, scenario)
            if key not in normalized.index:
                ax.set_visible(False)
                continue

            values = normalized.loc[key].to_numpy(dtype=float)
            color = SCENARIO_COLORS[scenario]

            style_radar_axis(ax, theta, list(normalized.columns), show_rlabels=(c == 0))
            ax.plot(theta, values, color=color, linewidth=2.0, zorder=3)
            ax.fill(theta, values, color=color, alpha=0.18, zorder=2)

            cost_value = float(cost_lookup.loc[key]) if key in cost_lookup.index else np.nan
            cost_text = "n/a" if not np.isfinite(cost_value) else f"{cost_value:,.0f} € t$^{{-1}}$"

            ax.text(
                0.5,
                1.16,
                cost_text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.2,
                color="#1f2937",
                clip_on=False,
            )

            if r == 0:
                ax.set_title(
                    SCENARIO_LABELS[scenario],
                    fontsize=10.2,
                    weight="semibold",
                    y=1.28,
                    pad=2,
                )

            if c == 0:
                ax.text(
                    -0.34,
                    0.50,
                    country,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.0,
                    weight="semibold",
                    color="#1f2937",
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spider_tradeoff_{pathway}_{DB_LABELS[db_name]}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone publication-style spider/radar plots for PtX case-study LCA trade-offs.")
    parser.add_argument("--pathway", choices=sorted(PATHWAY_INFO), default="meoh_to_saf")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["Australia", "China", "USA"],
        help="Clean country names to include, e.g. Australia China USA",
    )
    parser.add_argument(
        "--db",
        choices=["reference", "future", "both"],
        default="both",
        help="Database year to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("figs") / "spider_figs"),
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    dbs = []
    if args.db in ("reference", "both"):
        dbs.append(NAME_REF_DB)
    if args.db in ("future", "both"):
        dbs.append(NAME_FUTURE_DB)

    saved = []
    for db_name in dbs:
        saved.append(plot_spider_figure(args.pathway, db_name, args.countries, out_dir))

    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

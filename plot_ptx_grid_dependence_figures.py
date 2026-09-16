from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pycountry


FIG_DIR = Path("figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRID_GHG_FILE = Path("input_data/ghg_factors.json")

MEOH_COLOR = "#2B6CB0"
MTJ_COLOR = "#E46A1A"
FT_COLOR = "#2E8B57"
LOW_CARBON_SHADE = "#D9EAD7"
COMPARATOR_COLOR = "#3A3A3A"
LIFECYCLE_BAND = "#CFCFCF"
MTJ_SHADE = "#F7D8BF"
FT_SHADE = "#DCEFE3"


COUNTRY_LABELS = [
    "Norway",
    "Switzerland",
    "Iceland",
    "Denmark",
    "Spain",
    "United Kingdom",
    "Germany",
    "United States",
    "World",
]

LOW_CARBON_REGION_MAX = 0.10
X_MAX = 0.80

METHANOL_CONFIG = {
    "title": "Methanol pathway",
    "product_unit": "product",
    "line_color": MEOH_COLOR,
    "current": {"label": "MeOH 2025", "i0": 0.75, "e": 11.4, "linestyle": "-"},
    "future": {"label": "MeOH 2050", "i0": 0.47, "e": 10.0, "linestyle": (0, (6, 4))},
    "comparator": {
        "line_y": 1.375,
        "line_text": r"Methanol line (1.375 t CO$_2$ eq t$^{-1}$)",
        "band_low": 2.05,
        "band_high": 2.20,
        "band_text": r"Fossil methanol life cycle CO$_2$ eq intensity 2.05–2.20 t CO$_2$ eq t$^{-1}$",
    },
    "y_max": 10.9,
    "output": FIG_DIR / "grid_dependence_methanol.png",
}

SAF_CONFIG = {
    "title": "SAF pathways",
    "product_unit": "product",
    "pathways": [
        {
            "legend_label": "MeOH-to-SAF (MtJ)",
            "color": MTJ_COLOR,
            "current": {"label": "MtJ 2025", "i0": 1.88, "e": 25.9, "linestyle": "-"},
            "future": {"label": "MtJ 2050", "i0": 1.21, "e": 23.0, "linestyle": (0, (6, 4))},
        },
        {
            "legend_label": "FT-SAF",
            "color": FT_COLOR,
            "current": {"label": "FT-SAF 2025", "i0": 0.45, "e": 27.0, "linestyle": "-"},
            "future": {"label": "FT-SAF 2050", "i0": 0.27, "e": 24.0, "linestyle": (0, (6, 4))},
        },
    ],
    "comparator": {
        "reference_y": 3.90,
        "reference_text": r"Fossil jet fuel life cycle CO$_2$ eq intensity (3.9 t CO$_2$ eq t$^{-1}$)",
        "green_target_y": 1.53,
        "green_target_text": r"–60% reduction target (1.56 t CO$_2$ eq t$^{-1}$)",
    },
    "y_max": 24.8,
    "output": FIG_DIR / "grid_dependence_saf.png",
}


def _country_to_iso2(country_name: str) -> str:
    special_cases = {
        "United Kingdom": "GB",
        "United States": "US",
        "World": "GLO",
        "Europe": "EU",
        "Norway": "NO",
        "Switzerland": "CH",
        "Iceland": "IS",
        "Denmark": "DK",
        "Spain": "ES",
        "Germany": "DE",
    }
    if country_name in special_cases:
        return special_cases[country_name]

    country = pycountry.countries.get(name=country_name)
    if country is None:
        country = pycountry.countries.get(common_name=country_name)
    if country is None:
        raise ValueError(f"Country '{country_name}' not found in pycountry.")
    return country.alpha_2


def _load_grid_ghg_positions():
    with GRID_GHG_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    positions = {}
    for name in COUNTRY_LABELS:
        iso2 = _country_to_iso2(name)
        entry = data.get(iso2)
        if entry is None and iso2 != "GLO":
            entry = data.get("GLO")
        if entry is None:
            raise KeyError(f"No grid GHG factor found for '{name}' / '{iso2}'.")
        positions[name] = float(entry["lca_impact_climate change"])
    return positions


def _line_values(i0, e, x):
    return i0 + e * x


def _angle_for_segment(ax, x1, y1, x2, y2):
    p1 = ax.transData.transform((x1, y1))
    p2 = ax.transData.transform((x2, y2))
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def _base_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.color": "#D4D4D4",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "mathtext.default": "regular",
        }
    )


def _format_axes(ax, y_max, title):
    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel(r"GHG intensity of grid electricity [2025 grid factors] [t CO$_2$ eq MWh$^{-1}$]")
    ax.set_ylabel(r"Climate change impacts [t CO$_2$ eq t$^{-1}$ product]")
    ax.set_title(title, pad=14)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(top=True, right=True)


def _draw_country_guides(ax, y_max, country_positions):
    for name, x in country_positions.items():
        ax.axvline(x, color="#B8B8B8", linestyle="--", linewidth=1.4, zorder=1)
        x_text = x - 0.004
        if name in {"Norway"}:
            x_text = x - 0.010
        ax.text(
            x_text,
            y_max * 0.98,
            name,
            rotation=90,
            va="top",
            ha="center",
            color="#444444",
            fontsize=10.5,
            zorder=5,
        )


def _draw_low_carbon_region(ax, y_max):
    ax.axvspan(0.0, LOW_CARBON_REGION_MAX, color=LOW_CARBON_SHADE, alpha=0.45, zorder=0)
    ax.text(
        0.03,
        y_max * 0.76,
        "Low-carbon\nelectricity",
        color="#1B6B27",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        zorder=6,
    )


def _draw_methanol(ax, cfg, country_positions):
    x = np.linspace(0.0, X_MAX, 400)
    current = cfg["current"]
    future = cfg["future"]
    comp = cfg["comparator"]
    threshold_label_y = 6.5

    y_current = _line_values(current["i0"], current["e"], x)
    y_future = _line_values(future["i0"], future["e"], x)

    ax.fill_between(x, y_future, y_current, color=MTJ_SHADE, alpha=0.45, zorder=2)
    ax.plot(x, y_current, color=MEOH_COLOR, linewidth=2.8, linestyle=current["linestyle"], zorder=4)
    ax.plot(x, y_future, color=MEOH_COLOR, linewidth=2.6, linestyle=future["linestyle"], zorder=4)

    ax.axhspan(comp["band_low"], comp["band_high"], color=LIFECYCLE_BAND, alpha=0.35, zorder=1)
    ax.axhline(comp["line_y"], color=COMPARATOR_COLOR, linewidth=1.8, linestyle="--", zorder=3)

    x_curr_thr = (comp["line_y"] - current["i0"]) / current["e"]
    x_fut_thr = (comp["line_y"] - future["i0"]) / future["e"]
    ax.vlines(x_curr_thr, 0, threshold_label_y, color=MEOH_COLOR, linewidth=1.6, zorder=5)
    ax.vlines(x_fut_thr, 0, threshold_label_y, color=MEOH_COLOR, linewidth=1.6, linestyles=(0, (6, 4)), zorder=5)
    ax.text(x_curr_thr - 0.003, threshold_label_y + 0.08, current["label"], rotation=90, va="bottom", color=MEOH_COLOR, fontsize=11.0, ha="center")
    ax.text(x_fut_thr - 0.003, threshold_label_y + 0.08, future["label"], rotation=90, va="bottom", color=MEOH_COLOR, fontsize=11.0, ha="center")

    angle_current = _angle_for_segment(ax, 0.48, _line_values(current["i0"], current["e"], 0.48), 0.60, _line_values(current["i0"], current["e"], 0.60))
    angle_future = _angle_for_segment(ax, 0.62, _line_values(future["i0"], future["e"], 0.62), 0.74, _line_values(future["i0"], future["e"], 0.74))
    x_curr_anchor, x_curr_label, y_curr_offset = 0.56, 0.48, 0.85
    x_fut_anchor, x_fut_label, y_fut_offset = 0.56, 0.48, -0.80
    y_curr_anchor = _line_values(current["i0"], current["e"], x_curr_anchor)
    y_fut_anchor = _line_values(future["i0"], future["e"], x_fut_anchor)
    y_curr_text = _line_values(current["i0"], current["e"], x_curr_label) + y_curr_offset
    y_fut_text = _line_values(future["i0"], future["e"], x_fut_label) + y_fut_offset
    ax.annotate(
        current["label"] + rf" ($E$={current['e']:.0f} kWh/kg)",
        xy=(x_curr_anchor, y_curr_anchor),
        xytext=(x_curr_label, y_curr_text),
        color=MEOH_COLOR,
        fontsize=13,
        rotation=angle_current,
        rotation_mode="anchor",
        ha="left",
        va="center",
        zorder=6,
        bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
        arrowprops=dict(arrowstyle="-", color=MEOH_COLOR, lw=1.0, shrinkA=0, shrinkB=0),
    )
    ax.annotate(
        future["label"] + rf" ($E$={future['e']:.0f} kWh/kg)",
        xy=(x_fut_anchor, y_fut_anchor),
        xytext=(x_fut_label, y_fut_text),
        color=MEOH_COLOR,
        fontsize=13,
        rotation=angle_future,
        rotation_mode="anchor",
        ha="left",
        va="center",
        zorder=6,
        bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
        arrowprops=dict(arrowstyle="-", color=MEOH_COLOR, lw=1.0, shrinkA=0, shrinkB=0),
    )

    ax.text(X_MAX - 0.01, comp["band_high"] + 0.18, comp["band_text"], color="#4A4A4A", fontsize=11.5, ha="right")
    ax.text(X_MAX - 0.01, comp["line_y"] + 0.14, comp["line_text"], color=COMPARATOR_COLOR, fontsize=11.5, style="italic", ha="right")

    handles = [
        mlines.Line2D([], [], color=MEOH_COLOR, linewidth=2.8, label="MeOH"),
        mlines.Line2D([], [], color="black", linewidth=2.2, linestyle="-", label="2025 demand"),
        mlines.Line2D([], [], color="black", linewidth=2.2, linestyle=(0, (6, 4)), label="2050 demand"),
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle="--", label="Comparator line"),
    ]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, handlelength=2.6, columnspacing=1.6, handletextpad=0.6)


def _draw_saf(ax, cfg, country_positions):
    x = np.linspace(0.0, X_MAX, 400)
    comp = cfg["comparator"]
    threshold_label_y = 10.0

    for pathway in cfg["pathways"]:
        color = pathway["color"]
        current = pathway["current"]
        future = pathway["future"]
        shade = MTJ_SHADE if "MtJ" in current["label"] else FT_SHADE

        y_current = _line_values(current["i0"], current["e"], x)
        y_future = _line_values(future["i0"], future["e"], x)

        ax.fill_between(x, y_future, y_current, color=shade, alpha=0.45, zorder=2)
        ax.plot(x, y_current, color=color, linewidth=2.8, linestyle=current["linestyle"], zorder=4)
        ax.plot(x, y_future, color=color, linewidth=2.6, linestyle=future["linestyle"], zorder=4)

        x_curr_thr = (comp["reference_y"] - current["i0"]) / current["e"]
        x_fut_thr = (comp["reference_y"] - future["i0"]) / future["e"]
        ax.vlines(x_curr_thr, 0, threshold_label_y, color=color, linewidth=1.6, zorder=5)
        ax.vlines(x_fut_thr, 0, threshold_label_y, color=color, linewidth=1.6, linestyles=(0, (6, 4)), zorder=5)
        ax.text(
            x_curr_thr - 0.002,
            threshold_label_y + 0.10,
            current["label"],
            rotation=90,
            va="bottom",
            color=color,
            fontsize=11.5,
            ha="center",
        )
        ax.text(
            x_fut_thr - 0.002,
            threshold_label_y + 0.10,
            future["label"],
            rotation=90,
            va="bottom",
            color=color,
            fontsize=11.5,
            ha="center",
        )

        angle_current = _angle_for_segment(ax, 0.46, _line_values(current["i0"], current["e"], 0.46), 0.62, _line_values(current["i0"], current["e"], 0.62))
        angle_future = _angle_for_segment(ax, 0.58, _line_values(future["i0"], future["e"], 0.58), 0.74, _line_values(future["i0"], future["e"], 0.74))

        if "MtJ" in current["label"]:
            x_curr_anchor, x_curr_label, y_curr_offset = 0.50, 0.39, 1.55
            x_fut_anchor, x_fut_label, y_fut_offset = 0.50, 0.39, -1.35
        else:
            x_curr_anchor, x_curr_label, y_curr_offset = 0.61, 0.58, 1.35
            x_fut_anchor, x_fut_label, y_fut_offset = 0.61, 0.58, -1.15

        y_curr_anchor = _line_values(current["i0"], current["e"], x_curr_anchor)
        y_fut_anchor = _line_values(future["i0"], future["e"], x_fut_anchor)
        y_curr_text = _line_values(current["i0"], current["e"], x_curr_label) + y_curr_offset
        y_fut_text = _line_values(future["i0"], future["e"], x_fut_label) + y_fut_offset

        ax.annotate(
            current["label"] + rf" ($E$={current['e']:.0f} kWh/kg)",
            xy=(x_curr_anchor, y_curr_anchor),
            xytext=(x_curr_label, y_curr_text),
            color=color,
            fontsize=13,
            rotation=angle_current,
            rotation_mode="anchor",
            ha="left",
            va="center",
            zorder=6,
            bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=0),
        )
        ax.annotate(
            future["label"] + rf" ($E$={future['e']:.0f} kWh/kg)",
            xy=(x_fut_anchor, y_fut_anchor),
            xytext=(x_fut_label, y_fut_text),
            color=color,
            fontsize=13,
            rotation=angle_future,
            rotation_mode="anchor",
            ha="left",
            va="center",
            zorder=6,
            bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=0),
        )

    ax.axhline(comp["reference_y"], color=COMPARATOR_COLOR, linewidth=1.8, linestyle="--", zorder=3)
    ax.axhline(comp["green_target_y"], color="#1E7D22", linewidth=1.4, linestyle=(0, (3, 2)), zorder=3)

    ax.text(
        0.48,
        comp["reference_y"] + 0.18,
        comp["reference_text"],
        color=COMPARATOR_COLOR,
        fontsize=12,
        ha="left",
        bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
    )
    ax.text(
        0.48,
        comp["green_target_y"] + 0.10,
        comp["green_target_text"],
        color="#1E7D22",
        fontsize=12,
        ha="left",
        bbox=dict(facecolor="none", edgecolor="none", pad=0.6),
    )

    handles = [
        mlines.Line2D([], [], color=MTJ_COLOR, linewidth=2.8, label="MeOH-to-SAF (MtJ)"),
        mlines.Line2D([], [], color=FT_COLOR, linewidth=2.8, label="FT-SAF"),
        mlines.Line2D([], [], color="black", linewidth=2.2, linestyle="-", label="2025 demand"),
        mlines.Line2D([], [], color="black", linewidth=2.2, linestyle=(0, (6, 4)), label="2050 demand"),
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle="--", label="Fossil lifecycle"),
        mlines.Line2D([], [], color="#1E7D22", linewidth=1.4, linestyle=(0, (3, 2)), label="−60% target"),
    ]
    ax.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False, handlelength=2.6, columnspacing=1.4, handletextpad=0.6)


def plot_methanol(country_positions):
    _base_style()
    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    _format_axes(ax, METHANOL_CONFIG["y_max"], METHANOL_CONFIG["title"])
    _draw_country_guides(ax, METHANOL_CONFIG["y_max"], country_positions)
    _draw_methanol(ax, METHANOL_CONFIG, country_positions)
    fig.tight_layout()
    fig.savefig(METHANOL_CONFIG["output"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_saf(country_positions):
    _base_style()
    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    _format_axes(ax, SAF_CONFIG["y_max"], SAF_CONFIG["title"])
    _draw_country_guides(ax, SAF_CONFIG["y_max"], country_positions)
    _draw_saf(ax, SAF_CONFIG, country_positions)
    fig.tight_layout()
    fig.savefig(SAF_CONFIG["output"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    country_positions = _load_grid_ghg_positions()
    plot_methanol(country_positions)
    plot_saf(country_positions)
    print(f"Saved: {METHANOL_CONFIG['output']}")
    print(f"Saved: {SAF_CONFIG['output']}")


if __name__ == "__main__":
    main()

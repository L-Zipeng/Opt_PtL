"""Global PtX geographic driver analysis — LightGBM surrogate + SHAP attribution.

Method:
  For each (scenario, outcome) combination a LightGBM surrogate is trained on the
  existing pixel-level MILP results.  SHAP (SHapley Additive exPlanations) values
  are then computed via TreeExplainer.  SHAP decomposes every prediction into
  signed, additive, unit-consistent contributions from each geographic driver,
  correctly accounting for driver co-correlations and non-linear interactions.

Produces two publication figures:
  1. figure_global_<pathway>_shap.png  – 2×2 SHAP beeswarm (cost + GHG, both scenarios)
  2. figure_dac_climate_dependence.png – DAC demand in climate space + outcome response
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import numpy.core.numeric
from matplotlib.lines import Line2D

from config import DEG_RES

sys.modules["numpy._core.numeric"] = numpy.core.numeric

# ── File paths ─────────────────────────────────────────────────────────────────
DEFAULT_GLOBAL_RESULTS_FILE = Path("results") / "global_results_ptx_1.pkl"
PROCESSED_DS_FILE           = Path("processed_data") / f"ds_processed_res_{DEG_RES}.pkl"

PATHWAY_INFO = {
    "meoh": {
        "name": "MeOH",
        "title": "Global MeOH",
        "slug": "meoh",
        "cost_col": "euro_tMeOH",
        "ghg_col": "tCO2_tMeOH",
    },
    "meoh_to_saf": {
        "name": "SAF (MtJ)",
        "title": "Global SAF (MtJ)",
        "slug": "meoh_to_saf",
        "cost_col": "euro_tSAF",
        "ghg_col": "tCO2_tSAF",
    },
    "ftsaf": {
        "name": "SAF (FT)",
        "title": "Global SAF (FT)",
        "slug": "ftsaf",
        "cost_col": "euro_tSAF",
        "ghg_col": "tCO2_tSAF",
    },
}

PATHWAY_ALIASES = {
    "mtj": "meoh_to_saf",
    "saf_mtj": "meoh_to_saf",
    "meoh-to-saf": "meoh_to_saf",
    "ft": "ftsaf",
    "ft-saf": "ftsaf",
    "saf_ft": "ftsaf",
}

SCENARIO_ORDER = ["hybrid", "off_grid", "grid_connected"]
SCENARIO_LABELS = {
    "hybrid": "Hybrid",
    "off_grid": "Off-grid",
    "grid_connected": "Grid-connected",
}
SCENARIO_COLORS = {
    "hybrid": "#4878d0",
    "off_grid": "#ee854a",
    "grid_connected": "#6acc64",
}
SCENARIO_MARKERS = {"hybrid": "o", "off_grid": "s", "grid_connected": "^"}

# Drivers available per scenario (off-grid has no grid connection)
SCENARIO_DRIVERS = {
    "hybrid": [
        "cost_grid_mean",
        "ghg_grid_mean",
        "cf_solar",
        "cf_wind",
        "dac_el_kWh_per_kgCO2",
        "dac_heat_kWh_per_kgCO2",
    ],
    "off_grid": [
        "cf_solar",
        "cf_wind",
        "dac_el_kWh_per_kgCO2",
        "dac_heat_kWh_per_kgCO2",
    ],
    "grid_connected": [
        "cost_grid_mean",
        "ghg_grid_mean",
        "dac_el_kWh_per_kgCO2",
        "dac_heat_kWh_per_kgCO2",
    ],
}

DRIVER_LABELS = {
    "cf_solar":               "Solar CF",
    "cf_wind":                "Wind CF",
    "cost_grid_mean":         "Grid electricity price",
    "ghg_grid_mean":          "Grid carbon intensity",
    "dac_el_kWh_per_kgCO2":  "DAC electricity demand",
    "dac_heat_kWh_per_kgCO2":"DAC heat demand",
}

PIXEL_DRIVER_COLUMNS = [
    "cost_grid_mean",
    "ghg_grid_mean",
    "cf_solar",
    "cf_wind",
    "temp_air_c",
    "rel_humidity",
    "dac_el_kWh_per_kgCO2",
    "dac_heat_kWh_per_kgCO2",
]

OUTCOME_LABELS = {
    "euro_tMeOH": "Production cost [€ t$^{-1}$ MeOH]",
    "tCO2_tMeOH": "Climate impact [t CO$_2$ eq t$^{-1}$ MeOH]",
    "euro_tSAF": "Production cost [€ t$^{-1}$ SAF]",
    "tCO2_tSAF": "Climate impact [t CO$_2$ eq t$^{-1}$ SAF]",
}
OUTCOME_UNITS = {
    "euro_tMeOH": "€ t$^{-1}$ MeOH",
    "tCO2_tMeOH": "t CO$_2$ eq t$^{-1}$ MeOH",
    "euro_tSAF": "€ t$^{-1}$ SAF",
    "tCO2_tSAF": "t CO$_2$ eq t$^{-1}$ SAF",
}
OUTCOME_SHORT = {
    "euro_tMeOH": "Cost",
    "tCO2_tMeOH": "Climate impact",
    "euro_tSAF": "Cost",
    "tCO2_tSAF": "Climate impact",
}

# ── RC params ──────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.6,
        "savefig.dpi": 500,
    }
)

# Multi-step colormap: white → amber → red → dark crimson (for heatmap)
from matplotlib.colors import LinearSegmentedColormap as _LSC
HEAT_CMAP = _LSC.from_list(
    "heat9",
    ["#ffffff", "#fff3cd", "#fde68a", "#fbbf24", "#f59e0b",
     "#ef4444", "#dc2626", "#991b1b", "#4c0519"],
    N=256,
)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Global PtX SHAP driver analysis figures."
    )
    pathway_choices = sorted(set(PATHWAY_INFO) | set(PATHWAY_ALIASES) | {"all", "saf"})
    p.add_argument(
        "--pathway",
        choices=pathway_choices,
        default="meoh",
        help=(
            "Pathway to analyse. Use 'mtj' for meoh_to_saf, 'ftsaf' for FT-SAF, "
            "'saf' for both SAF pathways, or 'all' for all pathways."
        ),
    )
    p.add_argument(
        "--pathways",
        nargs="+",
        choices=pathway_choices,
        default=None,
        help="Optional list of pathways to analyse in one run.",
    )
    p.add_argument(
        "--input-file",
        default=None,
        help=(
            "Path to global PtX results (.pkl or .parquet). Defaults to the newest "
            "non-partial results/global_results_ptx_*.pkl/.parquet file."
        ),
    )
    p.add_argument(
        "--grid-input-file",
        default=None,
        help=(
            "Optional grid-connected global PtX results file. Defaults to the newest "
            "results/global_results_ptx_grid_* file and is appended to --input-file."
        ),
    )
    p.add_argument(
        "--no-grid-connected",
        action="store_true",
        help="Do not append separate grid-connected results.",
    )
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def resolve_pathways(args):
    requested = args.pathways if args.pathways else [args.pathway]
    resolved = []
    for key in requested:
        if key == "all":
            expanded = list(PATHWAY_INFO)
        elif key == "saf":
            expanded = ["meoh_to_saf", "ftsaf"]
        else:
            expanded = [PATHWAY_ALIASES.get(key, key)]
        for item in expanded:
            if item not in resolved:
                resolved.append(item)
    return resolved


def find_latest_global_results_file():
    candidates = [
        p for p in Path("results").glob("global_results_ptx_*.*")
        if p.suffix in {".pkl", ".parquet"}
        and ".partial" not in p.name
        and "_grid_" not in p.name
    ]
    if not candidates:
        return DEFAULT_GLOBAL_RESULTS_FILE

    def sort_key(path):
        # Prefer pickle on exact timestamp ties because it avoids optional parquet engines.
        suffix_rank = 1 if path.suffix == ".pkl" else 0
        return (path.stat().st_mtime, suffix_rank)

    return max(candidates, key=sort_key)


def find_latest_global_grid_results_file():
    candidates = [
        p for p in Path("results").glob("global_results_ptx_grid_*.*")
        if p.suffix in {".pkl", ".parquet"} and ".partial" not in p.name
    ]
    if not candidates:
        return None

    def sort_key(path):
        suffix_rank = 1 if path.suffix == ".pkl" else 0
        return (path.stat().st_mtime, suffix_rank)

    return max(candidates, key=sort_key)


def _looks_like_parquet(path):
    with Path(path).open("rb") as fh:
        return fh.read(4) == b"PAR1"


def _read_results_table(path):
    path = Path(path)
    if path.suffix == ".parquet" or _looks_like_parquet(path):
        return pd.read_parquet(path)
    with path.open("rb") as fh:
        return pickle.load(fh)


def _pathway_row_count(path, pathway):
    try:
        df = _read_results_table(path).reset_index()
    except Exception as exc:
        print(f"  Could not inspect {path}: {exc}")
        return 0
    required = {"ptx_pathway", "scenario", "model_status"}
    if not required.issubset(df.columns):
        return 0
    mask = (
        (df["ptx_pathway"] == pathway)
        & (df["model_status"] == "OPTIMAL")
        & (df["scenario"].isin(SCENARIO_ORDER))
    )
    return int(mask.sum())


def find_latest_pathway_results_file(pathway):
    candidates = [
        p for p in Path("results").glob("global_results_ptx_*.*")
        if p.suffix in {".pkl", ".parquet"} and ".partial" not in p.name
    ]
    candidates += [
        p for p in Path("results").glob("job_*")
        if p.is_file()
    ]

    viable = []
    for path in candidates:
        n = _pathway_row_count(path, pathway)
        if n > 0:
            viable.append((path.stat().st_mtime, n, path))
    if not viable:
        return None
    return max(viable, key=lambda item: item[0])[2]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_global_results(pathway, input_file):
    input_path = Path(input_file)
    df = _read_results_table(input_path)
    df = df.reset_index()
    df = df[(df["ptx_pathway"] == pathway) & (df["model_status"] == "OPTIMAL")]
    return df[df["scenario"].isin(SCENARIO_ORDER)].copy()


def load_cached_pixel_driver_data(pathway):
    slug = PATHWAY_INFO[pathway]["slug"]
    preferred = [
        Path("figs") / f"global_{slug}_driver_analysis" / f"global_{slug}_pixel_data.csv",
        Path("figs") / "global_meoh_driver_analysis" / "global_meoh_pixel_data.csv",
        Path("figs") / "global_meoh_to_saf_driver_analysis" / "global_meoh_to_saf_pixel_data.csv",
    ]
    candidates = preferred + sorted(Path("figs").glob("global_*_driver_analysis/global_*_pixel_data.csv"))
    seen = set()
    frames = []
    used_paths = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [c for c in ["lat", "lon", *PIXEL_DRIVER_COLUMNS] if c in cols]
        missing = {"lat", "lon", "cf_solar", "cf_wind", "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2"} - set(usecols)
        if missing:
            continue
        frames.append(pd.read_csv(path, usecols=usecols).dropna(subset=["lat", "lon"]))
        used_paths.append(path)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["lat", "lon"], keep="first")
    print(
        "  Loaded cached pixel drivers from "
        + ", ".join(str(p) for p in used_paths[:3])
        + (" ..." if len(used_paths) > 3 else "")
    )
    return df


def load_pixel_driver_data():
    with PROCESSED_DS_FILE.open("rb") as fh:
        ds = pickle.load(fh)
    vars_needed = [
        "cf_solar", "cf_wind", "temp_air_c", "rel_humidity",
        "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2",
    ]
    arrays = []
    for name in vars_needed:
        arr = ds[name]
        if "time" in arr.dims:
            arr = arr.mean("time", skipna=True)
        arrays.append(arr.rename(name))
    ds2 = arrays[0].to_dataset(name=arrays[0].name)
    for arr in arrays[1:]:
        ds2[arr.name] = arr
    return ds2.to_dataframe().reset_index()


def _merge_missing_driver_columns(base, drivers):
    if drivers is None:
        return base
    missing = [
        col for col in PIXEL_DRIVER_COLUMNS
        if col not in base.columns or base[col].isna().all()
    ]
    merge_cols = ["lat", "lon", *[c for c in missing if c in drivers.columns]]
    if len(merge_cols) <= 2:
        return base
    return base.merge(drivers[merge_cols], on=["lat", "lon"], how="left")


def build_analysis_table(pathway, input_file):
    results = load_global_results(pathway, input_file)
    merged = _merge_missing_driver_columns(results, load_cached_pixel_driver_data(pathway))
    missing = [
        col for col in PIXEL_DRIVER_COLUMNS
        if col not in merged.columns or merged[col].isna().all()
    ]
    if missing:
        print(
            "  Cached drivers incomplete; loading processed dataset "
            f"{PROCESSED_DS_FILE}."
        )
        merged = _merge_missing_driver_columns(merged, load_pixel_driver_data())
    return merged


def build_analysis_table_from_files(pathway, input_files):
    frames = []
    for input_file in input_files:
        df = build_analysis_table(pathway, input_file)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_cols = [c for c in ["ptx_pathway", "scenario", "lat", "lon", "db_name"] if c in merged.columns]
    if dedupe_cols:
        merged = merged.drop_duplicates(dedupe_cols, keep="first")
    return merged


def pathway_outcomes(pathway):
    info = PATHWAY_INFO[pathway]
    return [info["cost_col"], info["ghg_col"]]


def available_scenarios(shap_results, pathway):
    outcomes = pathway_outcomes(pathway)
    return [
        scenario for scenario in SCENARIO_ORDER
        if any((scenario, outcome) in shap_results for outcome in outcomes)
    ]


def available_result_cols(shap_results, pathway):
    outcomes = pathway_outcomes(pathway)
    return [
        (scenario, outcome)
        for scenario in available_scenarios(shap_results, pathway)
        for outcome in outcomes
        if (scenario, outcome) in shap_results
    ]


def _grid_span(index, n_items, total_cols):
    start = int(round(index * total_cols / n_items))
    stop = int(round((index + 1) * total_cols / n_items))
    return slice(start, stop)


# ── Analysis utilities ─────────────────────────────────────────────────────────
def _winsorize(s, lo=0.01, hi=0.99):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.clip(s.quantile(lo), s.quantile(hi))


def _qbin(values, n=14):
    r = pd.Series(values).rank(method="first")
    return pd.qcut(r, q=min(n, len(r)), labels=False, duplicates="drop")


def build_response_curves(merged, pairs, n_bins=14):
    """Binned median ± p10–p90 response curves (used by DAC figure)."""
    rows = []
    for driver, outcome, scenario in pairs:
        df = merged.loc[
            merged["scenario"] == scenario, [driver, outcome]
        ].dropna().copy()
        if len(df) < 30:
            continue
        df[driver] = _winsorize(df[driver])
        df[outcome] = _winsorize(df[outcome])
        df["bin"] = _qbin(df[driver], n=n_bins)
        g = (
            df.groupby("bin", as_index=False)
            .agg(
                xmed=(driver, "median"),
                ymed=(outcome, "median"),
                yp10=(outcome, lambda s: s.quantile(0.10)),
                yp90=(outcome, lambda s: s.quantile(0.90)),
            )
            .sort_values("xmed")
        )
        g["driver"] = driver
        g["outcome"] = outcome
        g["scenario"] = scenario
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── SHAP surrogate analysis ────────────────────────────────────────────────────
def build_shap_results(merged, pathway):
    """
    For each (scenario, outcome):
      1. Train LightGBM surrogate on 80 % of pixel data
      2. Report out-of-sample R² (test set)
      3. Refit on full data and compute SHAP values via TreeExplainer

    Returns dict keyed by (scenario, outcome) with:
      shap_values    – ndarray (n, n_features)  [same units as outcome]
      feature_values – ndarray (n, n_features)  [original scale, winsorized]
      feature_names  – list of human-readable driver labels
      feature_keys   – list of column keys
      r2_test        – out-of-sample R²
      n              – number of samples used
    """
    try:
        import lightgbm as lgb
        import shap as shap_lib
    except ImportError:
        raise ImportError("Run: pip install lightgbm shap")

    from sklearn.model_selection import train_test_split

    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    results = {}
    for scenario in SCENARIO_ORDER:
        features = SCENARIO_DRIVERS[scenario]
        sub = merged[merged["scenario"] == scenario].copy()

        for outcome in pathway_outcomes(pathway):
            df = sub[features + [outcome]].dropna().copy()
            if len(df) < 50:
                print(f"  SKIP {scenario}|{outcome}: too few rows ({len(df)})")
                continue

            # Winsorize all columns
            X = pd.DataFrame(
                {col: _winsorize(df[col]).values for col in features},
                index=df.index,
            )
            y = _winsorize(df[outcome])

            # ── Out-of-sample R² ──────────────────────────────────────────────
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            model_test = lgb.LGBMRegressor(**lgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_test.fit(X_tr, y_tr)
            yhat_te = model_test.predict(X_te)
            ss_res = float(np.sum((y_te.values - yhat_te) ** 2))
            ss_tot = float(np.sum((y_te.values - y_te.mean()) ** 2))
            r2_test = max(0.0, 1.0 - ss_res / ss_tot)

            # ── Full-data model for SHAP ──────────────────────────────────────
            model_full = lgb.LGBMRegressor(**lgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_full.fit(X, y)

            explainer = shap_lib.TreeExplainer(model_full)
            shap_vals = explainer.shap_values(X)          # (n, n_features)

            results[(scenario, outcome)] = {
                "shap_values":    np.array(shap_vals),
                "feature_values": X.values,
                "feature_names":  [DRIVER_LABELS[f] for f in features],
                "feature_keys":   features,
                "r2_test":        r2_test,
                "n":              len(df),
            }
            print(
                f"  {SCENARIO_LABELS[scenario]:8s} | {OUTCOME_SHORT[outcome]:14s}: "
                f"test R² = {r2_test:.3f}   N = {len(df):,}"
            )

    return results


# ── SHAP beeswarm plot helpers ─────────────────────────────────────────────────
def _beeswarm_y(x_vals, y_center=0.0, max_half=0.40):
    """
    Density-aware y-jitter.  Points with similar SHAP values are binned
    and spread symmetrically within ±max_half of y_center.
    """
    n = len(x_vals)
    if n == 0:
        return np.array([])

    n_bins = min(60, max(20, n // 25))
    xlo, xhi = np.percentile(x_vals, [1, 99])
    if xhi - xlo < 1e-10:
        return np.full(n, y_center)

    bins = np.linspace(xlo - 1e-8, xhi + 1e-8, n_bins + 1)
    bin_ids = np.clip(np.digitize(x_vals, bins) - 1, 0, n_bins - 1)

    y_out = np.full(n, float(y_center))
    rng = np.random.default_rng(0)
    for bid in range(n_bins):
        idx = np.where(bin_ids == bid)[0]
        cnt = len(idx)
        if cnt <= 1:
            continue
        offsets = np.linspace(-max_half, max_half, cnt)
        # slight positional noise so points don't form perfectly regular rows
        step = 2 * max_half / max(cnt - 1, 1)
        offsets += rng.uniform(-step * 0.15, step * 0.15, cnt)
        offsets = np.clip(offsets, -max_half, max_half)
        rng.shuffle(offsets)
        y_out[idx] = y_center + offsets
    return y_out


def _draw_beeswarm(ax, shap_vals, feat_vals, feat_names, r2, n, title, unit):
    """
    Draw one SHAP beeswarm panel on ax.
    Dots: x = SHAP value, y = feature index (jittered), color = feature value.
    Features are sorted bottom-to-top by ascending mean |SHAP| (most important at top).
    """
    n_feat = shap_vals.shape[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)          # ascending → top of y-axis = most important
    y_ticks = np.arange(n_feat)

    cmap = plt.cm.RdBu_r

    for yi, fi in enumerate(order):
        sv = shap_vals[:, fi]
        fv = feat_vals[:, fi]

        # Normalize feature value to [0, 1] independently per feature (standard SHAP)
        p5, p95 = np.nanpercentile(fv, [5, 95])
        fv_norm = np.clip((fv - p5) / max(p95 - p5, 1e-10), 0.0, 1.0)

        y_pos = _beeswarm_y(sv, y_center=float(yi))

        ax.scatter(
            sv, y_pos,
            c=fv_norm, cmap=cmap,
            s=4.0, alpha=0.55, vmin=0, vmax=1,
            rasterized=True, linewidths=0,
        )

    ax.axvline(0, color="0.22", linewidth=1.1, zorder=6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([feat_names[fi] for fi in order])
    ax.set_ylim(-0.65, n_feat - 0.35)
    ax.set_xlabel(f"SHAP value [{unit}]")
    ax.set_title(title, weight="semibold", loc="left", pad=5)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # R² and N annotation
    ax.text(
        0.98, 0.02,
        f"Test R$^2$ = {r2:.3f}   N = {n:,}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7.2, color="0.28",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.78", lw=0.5, alpha=0.92),
    )


# ── Figure 1: SHAP beeswarm ────────────────────────────────────────────────────
def plot_shap_analysis(shap_results, out_dir, pathway):
    """
    2 rows × 2 cols SHAP beeswarm:
    (a) Hybrid    | Cost          (b) Off-grid | Cost
    (c) Hybrid    | Climate       (d) Off-grid | Climate

    Colorbar on right: feature value (blue = low, red = high).
    """
    layout = [
        (0, 0, "hybrid",   PATHWAY_INFO[pathway]["cost_col"], "a."),
        (0, 1, "off_grid", PATHWAY_INFO[pathway]["cost_col"], "b."),
        (1, 0, "hybrid",   PATHWAY_INFO[pathway]["ghg_col"], "c."),
        (1, 1, "off_grid", PATHWAY_INFO[pathway]["ghg_col"], "d."),
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.0, 9.0),
        gridspec_kw={"hspace": 0.26, "wspace": 0.42},
    )
    fig.subplots_adjust(left=0.21, right=0.88, bottom=0.07, top=0.92)

    for r, c, scenario, outcome, letter in layout:
        ax = axes[r, c]
        key = (scenario, outcome)
        if key not in shap_results:
            ax.set_visible(False)
            continue

        res = shap_results[key]
        title = (
            f"{letter}  {SCENARIO_LABELS[scenario]} — "
            f"{OUTCOME_SHORT[outcome]} drivers"
        )
        _draw_beeswarm(
            ax,
            res["shap_values"],
            res["feature_values"],
            res["feature_names"],
            res["r2_test"],
            res["n"],
            title,
            OUTCOME_UNITS[outcome],
        )

    # ── Shared feature-value colorbar ─────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.905, 0.20, 0.013, 0.58])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("Feature value", fontsize=8.2, labelpad=7)

    fig.suptitle(
        f"Geographic drivers of {PATHWAY_INFO[pathway]['title']} performance — SHAP attribution",
        fontsize=10.2, weight="semibold", y=0.968,
    )

    out_path = out_dir / f"figure_global_{PATHWAY_INFO[pathway]['slug']}_shap.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Figure 2: SHAP importance heatmap + dependence plots ──────────────────────
def plot_shap_detail(shap_results, out_dir, pathway):
    """
    Two-row figure inspired by Geslin et al. (Nature Energy, 2025):

    Row 1 (a) – Importance heatmap (drivers × scenario-outcome, row-normalised).
                 Cell values = mean |SHAP| in outcome units.  Mirrors Fig 3b/4a.
    Row 2 (b–d) – SHAP dependence plots for the top-3 globally important drivers.
                  x = feature value, y = SHAP value (EUR/t or tCO₂/t), colour =
                  interaction feature.  Black line = binned median.
                  ρ = Pearson corr(feature, SHAP).  Mirrors Fig 3d–f.
    """
    # ── Driver display order and result columns ────────────────────────────────
    driver_order = [
        "cost_grid_mean", "ghg_grid_mean",
        "cf_solar", "cf_wind",
        "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2",
    ]
    result_cols = [
        ("hybrid",   PATHWAY_INFO[pathway]["cost_col"]),
        ("hybrid",   PATHWAY_INFO[pathway]["ghg_col"]),
        ("off_grid", PATHWAY_INFO[pathway]["cost_col"]),
        ("off_grid", PATHWAY_INFO[pathway]["ghg_col"]),
    ]
    col_labels = [
        "Hybrid\nCost", "Hybrid\nClimate",
        "Off-grid\nCost", "Off-grid\nClimate",
    ]
    n_d, n_c = len(driver_order), len(result_cols)

    # ── Build raw mean |SHAP| matrix (drivers × outcome-cols) ─────────────────
    raw_mat = np.zeros((n_d, n_c))
    for ci, key in enumerate(result_cols):
        if key not in shap_results:
            continue
        res = shap_results[key]
        mean_abs = np.abs(res["shap_values"]).mean(axis=0)
        for fi, fk in enumerate(res["feature_keys"]):
            if fk in driver_order:
                raw_mat[driver_order.index(fk), ci] = float(mean_abs[fi])

    # Row-normalise (highlight relative importance within each driver row)
    row_max = raw_mat.max(axis=1, keepdims=True)
    row_max[row_max < 1e-10] = 1.0
    norm_mat = raw_mat / row_max

    # Global importance rank (mean across outcome columns, ignoring zeros)
    with np.errstate(invalid="ignore"):
        global_mean = np.where(raw_mat > 0, raw_mat, np.nan)
    global_mean = np.nanmean(global_mean, axis=1)
    global_mean = np.nan_to_num(global_mean, nan=0.0)
    top3_idx = np.argsort(global_mean)[::-1][:3]
    top3_drivers = [driver_order[i] for i in top3_idx]

    # Sort rows descending by global importance for the heatmap
    row_order = np.argsort(global_mean)[::-1]

    # ── Figure layout: 2 rows, top = heatmap (full width), bottom = 3 panels ──
    fig = plt.figure(figsize=(12.0, 8.8))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.80, 1.0],
        hspace=0.30, wspace=0.38,
        left=0.13, right=0.97, bottom=0.07, top=0.92,
    )
    ax_heat = fig.add_subplot(gs[0, :])
    dep_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]

    # ── Panel a.: importance heatmap ──────────────────────────────────────────
    im = ax_heat.imshow(
        norm_mat[row_order, :],
        cmap=HEAT_CMAP, aspect="auto", vmin=0, vmax=1,
        interpolation="nearest",
    )
    for ri, orig_ri in enumerate(row_order):
        for ci in range(n_c):
            v = raw_mat[orig_ri, ci]
            nv = norm_mat[orig_ri, ci]
            txt_color = "white" if nv > 0.60 else "0.15"
            if v > 0.05:
                ax_heat.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                             fontsize=8.0, color=txt_color)
            else:
                ax_heat.text(ci, ri, "—", ha="center", va="center",
                             fontsize=8.0, color="0.55")

    ax_heat.set_xticks(range(n_c))
    ax_heat.set_xticklabels(col_labels, fontsize=8.8)
    ax_heat.set_yticks(range(n_d))
    ax_heat.set_yticklabels(
        [DRIVER_LABELS[driver_order[i]] for i in row_order], fontsize=8.8
    )
    ax_heat.set_title(
        "a.  Mean |SHAP| driver importance  [cell values in outcome units; "
        "colour = row-normalised relative importance]",
        weight="semibold", loc="left", pad=6, fontsize=9.0,
    )

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.020, pad=0.01)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=7.0)
    cb.set_label("Relative\nimportance", fontsize=7.5, labelpad=4)

    # ── Panels (b–d): SHAP dependence for top-3 global drivers ───────────────
    dep_letters = ["b.", "c.", "d."]
    for driver, ax, letter in zip(top3_drivers, dep_axes, dep_letters):

        # Pick scenario × outcome where this driver has highest mean |SHAP|
        best_key, best_fi = None, None
        best_val = -1.0
        for key in result_cols:
            if key not in shap_results:
                continue
            res = shap_results[key]
            if driver not in res["feature_keys"]:
                continue
            fi = res["feature_keys"].index(driver)
            v = float(np.abs(res["shap_values"][:, fi]).mean())
            if v > best_val:
                best_val, best_key, best_fi = v, key, fi

        if best_key is None:
            ax.set_visible(False)
            continue

        res = shap_results[best_key]
        scenario, outcome = best_key
        fi = best_fi

        # Interaction feature: 2nd most important for this scenario-outcome
        mean_abs = np.abs(res["shap_values"]).mean(axis=0)
        rank = np.argsort(mean_abs)[::-1]
        int_fi = next((f for f in rank if f != fi), rank[0])

        x_vals = res["feature_values"][:, fi]
        y_vals = res["shap_values"][:, fi]
        c_vals = res["feature_values"][:, int_fi]

        # Normalise interaction feature colour [0, 1]
        cp5, cp95 = np.percentile(c_vals, [5, 95])
        c_norm = np.clip((c_vals - cp5) / max(cp95 - cp5, 1e-10), 0.0, 1.0)

        ax.scatter(
            x_vals, y_vals,
            c=c_norm, cmap="RdBu_r",
            s=5, alpha=0.55, vmin=0, vmax=1,
            rasterized=True, linewidths=0,
        )

        # Binned median trend line (black)
        tmp = pd.DataFrame({"x": x_vals, "y": y_vals})
        tmp["bin"] = _qbin(tmp["x"], n=12)
        g = (tmp.groupby("bin", as_index=False)
               .agg(xm=("x", "median"), ym=("y", "median"))
               .sort_values("xm"))
        ax.plot(g["xm"], g["ym"], color="0.12", linewidth=2.0, zorder=5,
                label="Binned median")

        ax.axhline(0, color="0.40", linewidth=0.8, linestyle="--", zorder=4)

        # Pearson ρ between feature value and SHAP value
        rho_dep = float(pd.Series(x_vals).corr(pd.Series(y_vals)))

        ax.set_xlabel(DRIVER_LABELS[driver])
        ax.set_ylabel(f"SHAP value [{OUTCOME_UNITS[outcome]}]")
        ax.set_title(
            f"{letter}  {SCENARIO_LABELS[scenario]} | {DRIVER_LABELS[driver]}",
            weight="semibold", loc="left", pad=5,
        )
        ax.text(
            0.97, 0.05, f"ρ = {rho_dep:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.8, color="0.12",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.78",
                      lw=0.5, alpha=0.92),
        )
        ax.text(
            0.03, 0.97,
            f"Color: {res['feature_names'][int_fi]}\n(blue = low,  red = high)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color="0.40", style="italic",
        )
        ax.grid(True)
        ax.legend(frameon=False, fontsize=7.8, loc="upper right")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"SHAP importance summary and feature-dependence plots — {PATHWAY_INFO[pathway]['title']}",
        fontsize=10.2, weight="semibold", y=0.975,
    )

    out_path = out_dir / f"figure_global_{PATHWAY_INFO[pathway]['slug']}_shap_detail.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Figure 3: DAC climate-space dependence ────────────────────────────────────
def plot_dac_analysis(merged, out_dir, pathway):
    """
    2 rows × 2 cols:
    (a) hexbin: T vs RH → DAC electricity demand
    (b) hexbin: T vs RH → DAC heat demand
    (c) product cost vs DAC electricity demand (both scenarios, binned response)
    (d) product GHG  vs DAC electricity demand (both scenarios, binned response)
    """
    response_df = build_response_curves(
        merged,
        [
            ("dac_el_kWh_per_kgCO2", out, sc)
            for out in pathway_outcomes(pathway)
            for sc in SCENARIO_ORDER
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    fig.subplots_adjust(
        left=0.10, right=0.96, bottom=0.08, top=0.95, hspace=0.38, wspace=0.38
    )

    hex_df = (
        merged[["lat", "lon", "temp_air_c", "rel_humidity",
                "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2"]]
        .drop_duplicates(["lat", "lon"])
        .dropna()
    )
    hkw = dict(gridsize=32, mincnt=2, linewidths=0.1)

    # (a) DAC electricity demand
    ax = axes[0, 0]
    hb = ax.hexbin(hex_df["temp_air_c"], hex_df["rel_humidity"],
                   C=hex_df["dac_el_kWh_per_kgCO2"],
                   reduce_C_function=np.mean, cmap="YlOrBr", **hkw)
    cb = fig.colorbar(hb, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("kWh kg$^{-1}$ CO$_2$", fontsize=7.8)
    cb.ax.tick_params(labelsize=7.2)
    ax.set_xlabel("Mean air temperature [°C]")
    ax.set_ylabel("Relative humidity [%]")
    ax.set_title("a.  DAC electricity demand", weight="semibold", loc="left", pad=5)

    # b. DAC heat demand
    ax = axes[0, 1]
    hb2 = ax.hexbin(hex_df["temp_air_c"], hex_df["rel_humidity"],
                    C=hex_df["dac_heat_kWh_per_kgCO2"],
                    reduce_C_function=np.mean, cmap="Blues", **hkw)
    cb2 = fig.colorbar(hb2, ax=ax, fraction=0.040, pad=0.02)
    cb2.set_label("kWh kg$^{-1}$ CO$_2$", fontsize=7.8)
    cb2.ax.tick_params(labelsize=7.2)
    ax.set_xlabel("Mean air temperature [°C]")
    ax.set_ylabel("Relative humidity [%]")
    ax.set_title("b.  DAC heat demand", weight="semibold", loc="left", pad=5)

    # (c) and (d) response curves
    for col, outcome in enumerate(pathway_outcomes(pathway)):
        ax = axes[1, col]
        sub = response_df[response_df["outcome"] == outcome]
        for scenario in SCENARIO_ORDER:
            s = sub[(sub["scenario"] == scenario) &
                    (sub["driver"] == "dac_el_kWh_per_kgCO2")]
            if s.empty:
                continue
            c = SCENARIO_COLORS[scenario]
            ax.fill_between(s["xmed"], s["yp10"], s["yp90"],
                            color=c, alpha=0.14, linewidth=0)
            ax.plot(s["xmed"], s["ymed"], color=c, linewidth=1.9,
                    marker=SCENARIO_MARKERS[scenario], markersize=4.2,
                    label=SCENARIO_LABELS[scenario])
        ax.set_xlabel("DAC electricity demand [kWh kg$^{-1}$ CO$_2$]")
        ax.set_ylabel(OUTCOME_LABELS[outcome])
        ax.set_title(
            f"{'cd'[col]}.  {OUTCOME_SHORT[outcome]} vs. DAC electricity demand",
            weight="semibold", loc="left", pad=5,
        )
        ax.grid(True)

    axes[1, 1].legend(frameon=False, fontsize=8.2, loc="lower right")

    out_path = out_dir / "figure_dac_climate_dependence.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Export tables ──────────────────────────────────────────────────────────────
def export_tables(merged, shap_results, out_dir, pathway):
    # Pixel-level data
    keep = [c for c in [
        "country", "iso2", "scenario", "lat", "lon",
        PATHWAY_INFO[pathway]["cost_col"], PATHWAY_INFO[pathway]["ghg_col"],
        "cost_grid_mean", "ghg_grid_mean",
        "cf_solar", "cf_wind", "temp_air_c", "rel_humidity",
        "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2",
    ] if c in merged.columns]
    merged[keep].to_csv(
        out_dir / f"global_{PATHWAY_INFO[pathway]['slug']}_pixel_data.csv", index=False
    )

    # SHAP importance summary (mean |SHAP| per driver)
    summary_rows = []
    for (scenario, outcome), res in shap_results.items():
        mean_abs = np.abs(res["shap_values"]).mean(axis=0)
        for fi, fname in enumerate(res["feature_keys"]):
            summary_rows.append({
                "scenario": scenario,
                "outcome": outcome,
                "driver": fname,
                "driver_label": DRIVER_LABELS[fname],
                "mean_abs_shap": float(mean_abs[fi]),
                "r2_test": res["r2_test"],
                "n": res["n"],
            })
    pd.DataFrame(summary_rows).sort_values(
        ["outcome", "scenario", "mean_abs_shap"], ascending=[True, True, False]
    ).to_csv(
        out_dir / f"global_{PATHWAY_INFO[pathway]['slug']}_shap_importance.csv",
        index=False,
    )
    print("  Exported tables.")


# ── Integrated figure: beeswarms + heatmap + dependence ───────────────────────
def plot_integrated_shap_figure(shap_results, out_dir, pathway):
    """
    Single publication figure integrating all SHAP results available for the
    pathway. The layout adapts to hybrid/off-grid or grid-connected-only data.
    """
    import matplotlib.gridspec as gridspec

    scenarios = available_scenarios(shap_results, pathway)
    outcomes = pathway_outcomes(pathway)
    result_cols = available_result_cols(shap_results, pathway)
    if not scenarios or not result_cols:
        print("  No available scenario/outcome combinations for integrated figure.")
        return

    # ── Build heatmap data ────────────────────────────────────────────────────
    driver_order = [
        "cost_grid_mean", "ghg_grid_mean",
        "cf_solar", "cf_wind",
        "dac_el_kWh_per_kgCO2", "dac_heat_kWh_per_kgCO2",
    ]
    col_labels = [
        f"{SCENARIO_LABELS[scenario]}\n{OUTCOME_SHORT[outcome]}"
        for scenario, outcome in result_cols
    ]
    n_d, n_c = len(driver_order), len(result_cols)

    raw_mat = np.zeros((n_d, n_c))
    for ci, key in enumerate(result_cols):
        if key not in shap_results:
            continue
        res = shap_results[key]
        mean_abs = np.abs(res["shap_values"]).mean(axis=0)
        for fi, fk in enumerate(res["feature_keys"]):
            if fk in driver_order:
                raw_mat[driver_order.index(fk), ci] = float(mean_abs[fi])

    row_max = raw_mat.max(axis=1, keepdims=True)
    row_max[row_max < 1e-10] = 1.0
    norm_mat = raw_mat / row_max

    with np.errstate(invalid="ignore"):
        global_mean = np.where(raw_mat > 0, raw_mat, np.nan)
    global_mean = np.nanmean(global_mean, axis=1)
    global_mean = np.nan_to_num(global_mean, nan=0.0)
    top3_drivers = [driver_order[i] for i in np.argsort(global_mean)[::-1][:3]]
    row_order    = np.argsort(global_mean)[::-1]

    # ── Figure layout ─────────────────────────────────────────────────────────
    total_cols = max(3, len(scenarios))
    fig_width = max(9.0, 5.1 * len(scenarios))
    fig = plt.figure(figsize=(fig_width, 15.5))
    gs = gridspec.GridSpec(
        4, total_cols,
        height_ratios=[1.15, 1.15, 0.56, 1.05],
        hspace=0.34, wspace=0.50,
        left=0.14, right=0.91, bottom=0.05, top=0.95,
    )

    # ── Rows 1–2: SHAP beeswarms ──────────────────────────────────────────────
    letters = list("abcdefghijklmnopqrstuvwxyz")
    letter_i = 0
    for row, outcome in enumerate(outcomes):
        for si, scenario in enumerate(scenarios):
            col_span = _grid_span(si, len(scenarios), total_cols)
            ax = fig.add_subplot(gs[row, col_span])
            key = (scenario, outcome)
            if key not in shap_results:
                ax.set_visible(False)
                continue
            res = shap_results[key]
            letter = f"{letters[letter_i]}."
            letter_i += 1
            _draw_beeswarm(
                ax,
                res["shap_values"], res["feature_values"],
                res["feature_names"], res["r2_test"], res["n"],
                f"{letter}  {SCENARIO_LABELS[scenario]} — {OUTCOME_SHORT[outcome]}",
                OUTCOME_UNITS[outcome],
            )

    # Shared beeswarm colorbar (right edge, spanning rows 1–2)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax_bee = fig.add_axes([0.926, 0.552, 0.011, 0.370])
    cb_bee = fig.colorbar(sm, cax=cbar_ax_bee)
    cb_bee.set_ticks([0, 0.5, 1])
    cb_bee.set_ticklabels(["Low", "Mid", "High"])
    cb_bee.ax.tick_params(labelsize=7.2)
    cb_bee.set_label("Feature value", fontsize=8.0, labelpad=6)

    # ── Row 3: Importance heatmap ─────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[2, :])
    im = ax_heat.imshow(
        norm_mat[row_order, :],
        cmap=HEAT_CMAP, aspect="auto", vmin=0, vmax=1,
        interpolation="nearest",
    )
    for ri, orig_ri in enumerate(row_order):
        for ci in range(n_c):
            v  = raw_mat[orig_ri, ci]
            nv = norm_mat[orig_ri, ci]
            txt_color = "white" if nv > 0.60 else "0.15"
            ax_heat.text(
                ci, ri,
                f"{v:.2f}" if v > 0.05 else "—",
                ha="center", va="center",
                fontsize=7.8, color=txt_color if v > 0.05 else "0.55",
            )
    ax_heat.set_xticks(range(n_c))
    ax_heat.set_xticklabels(col_labels, fontsize=8.5)
    ax_heat.set_yticks(range(n_d))
    ax_heat.set_yticklabels(
        [DRIVER_LABELS[driver_order[i]] for i in row_order], fontsize=8.5
    )
    ax_heat.set_title(
        f"{letters[letter_i]}.  Mean |SHAP| driver importance  "
        "[cell values in outcome units; colour = row-normalised relative importance]",
        weight="semibold", loc="left", pad=5, fontsize=8.8,
    )
    letter_i += 1
    cb_heat = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.01)
    cb_heat.set_ticks([0, 0.5, 1])
    cb_heat.set_ticklabels(["Low", "Mid", "High"])
    cb_heat.ax.tick_params(labelsize=7.0)
    cb_heat.set_label("Relative\nimportance", fontsize=7.5, labelpad=4)

    # ── Row 4: SHAP dependence plots ──────────────────────────────────────────
    dep_axes = [
        fig.add_subplot(gs[3, _grid_span(j, 3, total_cols)])
        for j in range(3)
    ]
    for driver, ax in zip(top3_drivers, dep_axes):
        best_key, best_fi, best_val = None, None, -1.0
        for key in result_cols:
            if key not in shap_results:
                continue
            res = shap_results[key]
            if driver not in res["feature_keys"]:
                continue
            fi = res["feature_keys"].index(driver)
            v  = float(np.abs(res["shap_values"][:, fi]).mean())
            if v > best_val:
                best_val, best_key, best_fi = v, key, fi

        if best_key is None:
            ax.set_visible(False)
            continue

        res = shap_results[best_key]
        scenario, outcome = best_key
        fi = best_fi

        mean_abs = np.abs(res["shap_values"]).mean(axis=0)
        rank  = np.argsort(mean_abs)[::-1]
        int_fi = next((f for f in rank if f != fi), rank[0])

        x_vals = res["feature_values"][:, fi]
        y_vals = res["shap_values"][:, fi]
        c_vals = res["feature_values"][:, int_fi]
        cp5, cp95 = np.percentile(c_vals, [5, 95])
        c_norm = np.clip((c_vals - cp5) / max(cp95 - cp5, 1e-10), 0.0, 1.0)

        ax.scatter(
            x_vals, y_vals,
            c=c_norm, cmap="RdBu_r",
            s=5, alpha=0.55, vmin=0, vmax=1,
            rasterized=True, linewidths=0,
        )
        tmp = pd.DataFrame({"x": x_vals, "y": y_vals})
        tmp["bin"] = _qbin(tmp["x"], n=12)
        g = (tmp.groupby("bin", as_index=False)
               .agg(xm=("x", "median"), ym=("y", "median"))
               .sort_values("xm"))
        ax.plot(g["xm"], g["ym"], color="0.12", linewidth=2.0, zorder=5,
                label="Binned median")
        ax.axhline(0, color="0.40", linewidth=0.8, linestyle="--", zorder=4)

        rho_dep = float(pd.Series(x_vals).corr(pd.Series(y_vals)))
        letter = f"{letters[letter_i]}."
        letter_i += 1
        ax.set_xlabel(DRIVER_LABELS[driver])
        ax.set_ylabel(f"SHAP value [{OUTCOME_UNITS[outcome]}]")
        ax.set_title(
            f"{letter}  {SCENARIO_LABELS[scenario]} | {DRIVER_LABELS[driver]}",
            weight="semibold", loc="left", pad=5,
        )
        ax.text(
            0.97, 0.05, f"ρ = {rho_dep:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.12",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.78",
                      lw=0.5, alpha=0.92),
        )
        ax.text(
            0.03, 0.97,
            f"Color: {res['feature_names'][int_fi]}\n(blue = low,  red = high)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color="0.40", style="italic",
        )
        ax.grid(True)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"Geographic drivers of {PATHWAY_INFO[pathway]['title']} production — SHAP attribution",
        fontsize=10.5, weight="semibold", y=0.975,
    )

    out_path = out_dir / f"figure_global_{PATHWAY_INFO[pathway]['slug']}_shap_integrated.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    pathways = resolve_pathways(args)
    input_file = Path(args.input_file) if args.input_file else None
    grid_input_file = (
        Path(args.grid_input_file)
        if args.grid_input_file
        else find_latest_global_grid_results_file()
    )
    if input_file:
        print(f"Using global results file: {input_file}")
    else:
        print("Using newest local result file containing each requested pathway.")
    if grid_input_file is not None and not args.no_grid_connected:
        print(f"Grid-connected results file: {grid_input_file}")
    print(f"Pathways: {', '.join(pathways)}")

    for pathway in pathways:
        out_dir = Path(
            args.outdir or f"figs/global_{PATHWAY_INFO[pathway]['slug']}_driver_analysis"
        )
        if len(pathways) > 1 and args.outdir:
            out_dir = out_dir / PATHWAY_INFO[pathway]["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {PATHWAY_INFO[pathway]['title']} ===")
        print("Loading data...")
        source_files = []
        primary_file = input_file or find_latest_pathway_results_file(pathway)
        if primary_file is not None:
            source_files.append(primary_file)
        if (
            grid_input_file is not None
            and not args.no_grid_connected
            and grid_input_file not in source_files
        ):
            source_files.append(grid_input_file)
        if not source_files:
            print(f"  No local global rows found for {pathway}; skipping.")
            continue
        merged = build_analysis_table_from_files(pathway, source_files)
        sc_counts = merged["scenario"].value_counts().to_dict()
        print(f"  Sources: {', '.join(str(p) for p in source_files)}")
        print(f"  {len(merged):,} OPTIMAL rows  {sc_counts}")
        if merged.empty:
            print(f"  No global rows found for {pathway}; skipping.")
            continue

        print("Training LightGBM surrogates and computing SHAP values...")
        shap_results = build_shap_results(merged, pathway)
        if not shap_results:
            print(f"  No SHAP results generated for {pathway}; skipping figures.")
            continue

        print("Generating figures...")
        plot_integrated_shap_figure(shap_results, out_dir, pathway)
        plot_dac_analysis(merged, out_dir, pathway)

        export_tables(merged, shap_results, out_dir, pathway)
        print(f"Done — outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

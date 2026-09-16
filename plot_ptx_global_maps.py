from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import Polygon, box

from config import (
    DEG_RES,
    FILE_PATH_GLOBAL_RESULTS_PTX,
    FILE_PATH_GLOBAL_RESULTS_PTX_GRID,
    T_DAY_MEOH,
    T_DAY_SAF,
)

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import geopandas as gpd
except ModuleNotFoundError as exc:
    missing = exc.name or "required mapping dependency"
    raise SystemExit(
        f"Missing dependency: {missing}. Install geopandas/cartopy in the plotting environment to run this script."
    ) from exc


CONVENTIONAL_COST_MEOH = 450
CONVENTIONAL_COST_SAF = 700
CONVENTIONAL_GHG_MEOH = 0.7
CONVENTIONAL_GHG_SAF = 3.9   # WTW fossil jet fuel (DAC uptake credit removed from PtX GHG)

PTX_PATHWAY_INFO = {
    "meoh": {
        "name": "Methanol",
        "unit": "MeOH",
        "product_column": "tCO2_tMeOH",
        "cost_column": "euro_tMeOH",
        "annual_production": T_DAY_MEOH * 365,
        "conventional_cost": CONVENTIONAL_COST_MEOH,
        "conventional_ghg": CONVENTIONAL_GHG_MEOH,
    },
    "meoh_to_saf": {
        "name": "SAF (MtJ)",
        "unit": "SAF",
        "product_column": "tCO2_tSAF",
        "cost_column": "euro_tSAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_cost": CONVENTIONAL_COST_SAF,
        "conventional_ghg": CONVENTIONAL_GHG_SAF,
    },
    "ftsaf": {
        "name": "SAF (FT)",
        "unit": "SAF",
        "product_column": "tCO2_tSAF",
        "cost_column": "euro_tSAF",
        "annual_production": T_DAY_SAF * 365,
        "conventional_cost": CONVENTIONAL_COST_SAF,
        "conventional_ghg": CONVENTIONAL_GHG_SAF,
    },
}

SPECIAL_CASES = {
    "United Kingdom": {"ISO_A2": "GB", "ISO_A3": "GBR"},
    "World": {"ISO_A2": "WORLD", "ISO_A3": "WLD"},
    "Europe": {"ISO_A2": "EU", "ISO_A3": "EUR"},
    "Norway": {"ISO_A2": "NO", "ISO_A3": "NOR"},
    "France": {"ISO_A2": "FR", "ISO_A3": "FRA"},
    "Kosovo": {"ISO_A2": "XK", "ISO_A3": "XKX"},
    "Somaliland": {"ISO_A2": "SO", "ISO_A3": "SOM"},
}

FILE_SHAPE = Path("input_data") / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
DEFAULT_OUTPUT_DIR = Path("figs") / "global_maps"
SCENARIO_TITLES = {
    "grid_connected": "Grid-connected",
    "hybrid": "Hybrid",
    "off_grid": "Off-grid",
}

COLORBAR_RGB = [
    (87, 48, 6),
    (131, 76, 8),
    (173, 115, 35),
    (208, 163, 87),
    (230, 208, 151),
    (245, 234, 204),
    (206, 237, 231),
    (151, 214, 205),
    (90, 178, 168),
    (35, 134, 126),
    (1, 96, 87),
    (0, 60, 46),
]
PUBLICATION_COLORS = [tuple(channel / 255 for channel in rgb) for rgb in COLORBAR_RGB]
SCIENCE_CMAP = LinearSegmentedColormap.from_list(
    "science_like_continuous",
    list(reversed(PUBLICATION_COLORS)),
    N=256,
)
LAND_EDGE_COLOR = "#9aa1a6"
COAST_COLOR = "#8b9095"
OCEAN_MASK_COLOR = "#fbfbf8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PtX global maps for grid-connected, hybrid, and off-grid scenarios.",
    )
    parser.add_argument(
        "--pathway",
        choices=sorted(PTX_PATHWAY_INFO),
        default="meoh_to_saf",
        help="PtX pathway to plot.",
    )
    parser.add_argument(
        "--cost-max",
        type=float,
        default=None,
        help="Upper bound for the shared cost color scale. Default uses the 95th percentile.",
    )
    parser.add_argument(
        "--ghg-max",
        type=float,
        default=None,
        help="Upper bound for the shared GHG color scale. Default uses the 95th percentile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the exported figure.",
    )
    parser.add_argument(
        "--spatial-input",
        type=Path,
        default=None,
        help="Optional custom pickle/parquet file for hybrid/off-grid spatial results.",
    )
    parser.add_argument(
        "--grid-input",
        type=Path,
        default=None,
        help="Optional custom pickle file for grid-connected results.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(FILE_SHAPE)[["geometry", "NAME_EN", "ISO_A2", "ISO_A3", "CONTINENT"]].copy()
    world["ISO_A2"] = world["ISO_A2"].replace("-99", np.nan)
    world["ISO_A3"] = world["ISO_A3"].replace("-99", np.nan)

    for name, codes in SPECIAL_CASES.items():
        mask = world["NAME_EN"] == name
        world.loc[mask, "ISO_A2"] = codes["ISO_A2"]
        world.loc[mask, "ISO_A3"] = codes["ISO_A3"]

    return world


def _apply_saf_dac_wtw_correction(df: pd.DataFrame, ghg_col: str, annual_production: float) -> pd.DataFrame:
    """Remove the DAC atmospheric-CO2 uptake credit to convert SAF GHG to a WTW basis.

    an_ghg_dac is negative (CO2 captured from atmosphere); adding its absolute
    value back aligns PtX SAF GHG with the WTW fossil jet comparator (3.9 tCO2/tSAF).
    """
    if "an_ghg_dac" in df.columns:
        correction = (-df["an_ghg_dac"].astype(float) / annual_production).clip(lower=0)
        df = df.copy()
        df[ghg_col] = df[ghg_col].astype(float) + correction
    return df


def load_grid_connected(pathway: str, grid_input: Path | None = None) -> pd.DataFrame:
    grid_path = Path(grid_input) if grid_input is not None else Path(FILE_PATH_GLOBAL_RESULTS_PTX_GRID)
    with open(grid_path, "rb") as file:
        df = pickle.load(file)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {grid_path}, got {type(df)!r}")

    if "ptx_pathway" in df.index.names:
        df = df.xs(pathway, level="ptx_pathway")

    df = df.reset_index()
    info = PTX_PATHWAY_INFO.get(pathway, {})
    if info.get("conventional_ghg") == CONVENTIONAL_GHG_SAF:  # SAF pathway
        df = _apply_saf_dac_wtw_correction(df, info["product_column"], info["annual_production"])
    return df


def load_spatial_results(pathway: str, spatial_input: Path | None = None) -> pd.DataFrame:
    if spatial_input is not None:
        input_path = Path(spatial_input)
        pickle_path = input_path
        parquet_path = input_path
    else:
        pickle_path = Path(FILE_PATH_GLOBAL_RESULTS_PTX)
        parquet_path = Path(f"results/global_results_ptx_{DEG_RES}.parquet")

    df = None

    # Prefer the pickle because the parquet export in this workspace is truncated.
    if pickle_path is not None and pickle_path.exists():
        import numpy.core.numeric

        sys.modules["numpy._core.numeric"] = numpy.core.numeric
        try:
            with open(pickle_path, "rb") as file:
                candidate = pickle.load(file)
            if isinstance(candidate, pd.DataFrame):
                df = candidate
        except Exception:
            df = None

    if df is None and parquet_path is not None and parquet_path.exists():
        df = pd.read_parquet(parquet_path)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in global spatial results, got {type(df)!r}")

    if "ptx_pathway" in df.index.names:
        pathways = df.index.get_level_values("ptx_pathway").unique().tolist()
        if pathway in pathways:
            df = df.xs(pathway, level="ptx_pathway")
        else:
            raise KeyError(f"Pathway {pathway!r} not found in spatial results. Available: {pathways}")

    df = df.reset_index()
    info = PTX_PATHWAY_INFO.get(pathway, {})
    if info.get("conventional_ghg") == CONVENTIONAL_GHG_SAF:  # SAF pathway
        df = _apply_saf_dac_wtw_correction(df, info["product_column"], info["annual_production"])
    return df


def infer_color_limits(
    pathway: str,
    grid_connected_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> tuple[float, float]:
    info = PTX_PATHWAY_INFO[pathway]
    cost_col = info["cost_column"]
    ghg_col = info["product_column"]

    cost_series = pd.concat(
        [
            grid_connected_df[cost_col].dropna(),
            spatial_df[cost_col].dropna(),
        ]
    )
    ghg_series = pd.concat(
        [
            grid_connected_df[ghg_col].dropna(),
            spatial_df[ghg_col].dropna(),
        ]
    )

    cost_max = float(cost_series.quantile(0.95))
    ghg_max = float(ghg_series.quantile(0.95))
    return cost_max, ghg_max


def infer_lat_extent(
    grid_connected_map: gpd.GeoDataFrame,
    spatial_df: pd.DataFrame,
    cost_col: str,
    ghg_col: str,
) -> tuple[float, float]:
    half = DEG_RES / 2.0
    lat_values = []

    if not spatial_df.empty and {"lat", cost_col, ghg_col}.issubset(spatial_df.columns):
        spatial_mask = spatial_df[cost_col].notna() | spatial_df[ghg_col].notna()
        if spatial_mask.any():
            lat_values.extend((spatial_df.loc[spatial_mask, "lat"] - half).tolist())
            lat_values.extend((spatial_df.loc[spatial_mask, "lat"] + half).tolist())

    country_mask = grid_connected_map[cost_col].notna() | grid_connected_map[ghg_col].notna()
    if country_mask.any():
        _, miny, _, maxy = grid_connected_map.loc[country_mask].total_bounds
        lat_values.extend([miny, maxy])

    if not lat_values:
        return (-58.0, 82.0)

    lat_min = max(-58.0, float(min(lat_values)) - 4.0)
    lat_max = min(84.0, float(max(lat_values)) + 4.0)
    return lat_min, lat_max


def build_grid_connected_map(world: gpd.GeoDataFrame, grid_connected_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = pd.merge(
        world,
        grid_connected_df,
        left_on="ISO_A2",
        right_on="iso2",
        how="left",
    )
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=world.crs)
    gdf_equal_area = gdf.to_crs(epsg=6933)
    gdf["area_km2"] = gdf_equal_area.geometry.area / 1e6
    return gdf


def build_spatial_grid(spatial_df: pd.DataFrame, world: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    df = spatial_df.copy()
    half = DEG_RES / 2.0

    def make_square(row: pd.Series) -> Polygon:
        return Polygon(
            [
                (row["lon"] - half, row["lat"] - half),
                (row["lon"] + half, row["lat"] - half),
                (row["lon"] + half, row["lat"] + half),
                (row["lon"] - half, row["lat"] + half),
            ]
        )

    df["geometry"] = df.apply(make_square, axis=1)
    df["area_km2"] = (DEG_RES * 111.0) ** 2 * np.abs(np.cos(np.deg2rad(df["lat"])))
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    land_union = world.union_all()
    full_box = box(-180.0, -90.0, 180.0, 90.0)
    ocean_geom = full_box.difference(land_union)
    gdf_ocean = gpd.GeoDataFrame(geometry=[ocean_geom], crs="EPSG:4326")

    return gdf, gdf_ocean


def add_europe_inset(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    cmap,
    norm,
    projection: ccrs.CRS,
    gdf_ocean: gpd.GeoDataFrame | None = None,
) -> None:
    europe_ax = inset_axes(
        ax,
        width="30%",
        height="42%",
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.01, 1, 1),
        bbox_transform=ax.transAxes,
        axes_class=cartopy.mpl.geoaxes.GeoAxes,
        axes_kwargs=dict(projection=projection),
    )
    europe_ax.set_extent([-10, 28, 32, 70], crs=projection)
    europe_ax.set_facecolor(OCEAN_MASK_COLOR)
    europe_ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.28)
    europe_ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.30)
    europe_ax.set_title("Europe", fontsize=8.6, loc="left", weight="semibold", pad=2)

    if gdf_ocean is not None:
        gdf_ocean.plot(ax=europe_ax, facecolor=OCEAN_MASK_COLOR, edgecolor=None, zorder=12, transform=projection)

    gdf.plot(
        ax=europe_ax,
        column=value_col,
        cmap=cmap,
        linewidth=0.0 if gdf_ocean is not None else 0.18,
        edgecolor="none" if gdf_ocean is not None else LAND_EDGE_COLOR,
        norm=norm,
        missing_kwds={"color": OCEAN_MASK_COLOR},
        transform=projection,
        aspect=None,
    )

    europe_ax.set_xticks([])
    europe_ax.set_yticks([])


def plot_country_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    cmap,
    norm,
    ax: plt.Axes,
    lat_extent: tuple[float, float],
) -> None:
    projection = ccrs.PlateCarree()

    ax.set_extent([-180, 180, lat_extent[0], lat_extent[1]], crs=projection)
    ax.set_facecolor(OCEAN_MASK_COLOR)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35)

    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        linewidth=0.25,
        edgecolor=LAND_EDGE_COLOR,
        norm=norm,
        missing_kwds={"color": OCEAN_MASK_COLOR},
        transform=projection,
    )

    add_europe_inset(ax, gdf, value_col, cmap, norm, projection)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["geo"].set_visible(False)


def plot_spatial_grid_prepared(
    gdf: gpd.GeoDataFrame,
    gdf_ocean: gpd.GeoDataFrame,
    world: gpd.GeoDataFrame,
    value_col: str,
    cmap,
    norm,
    config: str,
    ax: plt.Axes,
    lat_extent: tuple[float, float],
) -> None:
    projection = ccrs.PlateCarree()
    gdf_config = gdf[gdf["scenario"] == config].copy()

    ax.set_extent([-180, 180, lat_extent[0], lat_extent[1]], crs=projection)
    ax.set_facecolor(OCEAN_MASK_COLOR)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35)

    world.boundary.plot(ax=ax, color=OCEAN_MASK_COLOR, linewidth=0.25, zorder=4, transform=projection)

    gdf_config.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        linewidth=0.0,
        edgecolor="none",
        norm=norm,
        zorder=2,
        transform=projection,
        aspect=None,
    )

    gdf_ocean.plot(ax=ax, facecolor=OCEAN_MASK_COLOR, edgecolor=None, zorder=3, transform=projection)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.35, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["geo"].set_visible(False)

    add_europe_inset(ax, gdf_config, value_col, cmap, norm, projection, gdf_ocean=gdf_ocean)


def format_shared_colorbar(cbar, label: str) -> None:
    cbar.set_label(label, fontsize=11.0)
    cbar.ax.tick_params(labelsize=9.2, length=2.8, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.update_ticks()


def make_plot_norm(vmin: float, vmax: float):
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def infer_subplot_limits(
    info: dict,
    grid_connected_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, float]]:
    cost_col = info["cost_column"]
    ghg_col = info["product_column"]
    limits: dict[tuple[str, str], tuple[float, float]] = {}

    grid_cost = grid_connected_df[cost_col].dropna()
    grid_ghg = grid_connected_df[ghg_col].dropna()
    limits[("grid_connected", "cost")] = (info["conventional_cost"], float(grid_cost.quantile(0.95)))
    limits[("grid_connected", "ghg")] = (0.0, float(grid_ghg.quantile(0.95)))

    for config in ["hybrid", "off_grid"]:
        subset = spatial_df[spatial_df["scenario"] == config]
        cost_vals = subset[cost_col].dropna()
        ghg_vals = subset[ghg_col].dropna()
        limits[(config, "cost")] = (info["conventional_cost"], float(cost_vals.quantile(0.95)))
        limits[(config, "ghg")] = (0.0, float(ghg_vals.quantile(0.95)))

    return limits


def render_map_figure(
    pathway: str,
    world: gpd.GeoDataFrame,
    grid_connected_map: gpd.GeoDataFrame,
    spatial_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    gdf_ocean: gpd.GeoDataFrame,
    cost_limits: dict[str, tuple[float, float]],
    ghg_limits: dict[str, tuple[float, float]],
    output_dir: Path,
    dpi: int,
    use_shared_colorbars: bool,
) -> Path:
    info = PTX_PATHWAY_INFO[pathway]
    cost_col = info["cost_column"]
    ghg_col = info["product_column"]

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.2,
            "axes.titlesize": 15,
            "axes.labelsize": 11.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
        }
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.2, 11.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0)

    cost_label = rf"€ t$^{{-1}}$ {info['unit']}"
    ghg_label = rf"t CO$_2$ eq t$^{{-1}}$ {info['unit']}"
    lat_extent = infer_lat_extent(grid_connected_map, spatial_df, cost_col, ghg_col)

    if use_shared_colorbars:
        cost_norm = make_plot_norm(cost_limits["shared"][0], cost_limits["shared"][1])
        ghg_norm = make_plot_norm(ghg_limits["shared"][0], ghg_limits["shared"][1])
    else:
        cost_norm = None
        ghg_norm = None

    plot_country_map(
        grid_connected_map,
        value_col=cost_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(cost_limits["grid_connected"][0], cost_limits["grid_connected"][1]) if not use_shared_colorbars else cost_norm,
        ax=axes[0, 0],
        lat_extent=lat_extent,
    )
    plot_country_map(
        grid_connected_map,
        value_col=ghg_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(ghg_limits["grid_connected"][0], ghg_limits["grid_connected"][1]) if not use_shared_colorbars else ghg_norm,
        ax=axes[0, 1],
        lat_extent=lat_extent,
    )
    plot_spatial_grid_prepared(
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        world=world,
        value_col=cost_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(cost_limits["hybrid"][0], cost_limits["hybrid"][1]) if not use_shared_colorbars else cost_norm,
        config="hybrid",
        ax=axes[1, 0],
        lat_extent=lat_extent,
    )
    plot_spatial_grid_prepared(
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        world=world,
        value_col=ghg_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(ghg_limits["hybrid"][0], ghg_limits["hybrid"][1]) if not use_shared_colorbars else ghg_norm,
        config="hybrid",
        ax=axes[1, 1],
        lat_extent=lat_extent,
    )
    plot_spatial_grid_prepared(
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        world=world,
        value_col=cost_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(cost_limits["off_grid"][0], cost_limits["off_grid"][1]) if not use_shared_colorbars else cost_norm,
        config="off_grid",
        ax=axes[2, 0],
        lat_extent=lat_extent,
    )
    plot_spatial_grid_prepared(
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        world=world,
        value_col=ghg_col,
        cmap=SCIENCE_CMAP,
        norm=make_plot_norm(ghg_limits["off_grid"][0], ghg_limits["off_grid"][1]) if not use_shared_colorbars else ghg_norm,
        config="off_grid",
        ax=axes[2, 1],
        lat_extent=lat_extent,
    )
    panel_labels = [["a", "b"], ["c", "d"], ["e", "f"]]
    _configs = ["grid_connected", "hybrid", "off_grid"]
    for r, config in enumerate(_configs):
        axes[r, 0].set_title(
            f"Cost – Scenario: {SCENARIO_TITLES[config]}",
            fontsize=12.5,
            weight="semibold",
            pad=6,
        )
        axes[r, 1].set_title(
            f"GHG emissions – Scenario: {SCENARIO_TITLES[config]}",
            fontsize=12.5,
            weight="semibold",
            pad=6,
        )

    for r in range(3):
        for c in range(2):
            axes[r, c].text(
                0.0,
                1.01,
                panel_labels[r][c],
                transform=axes[r, c].transAxes,
                va="bottom",
                ha="left",
                fontsize=13.5,
                fontweight="semibold",
                color="#2f3337",
                clip_on=False,
            )
    title_suffix = "shared color scale" if use_shared_colorbars else "subplot-specific color scales"
    fig.suptitle(f"{info['name']} global supply-system performance ({title_suffix})", y=1.04, fontsize=15.6, weight="semibold")

    if use_shared_colorbars:
        sm_cost = plt.cm.ScalarMappable(norm=cost_norm, cmap=SCIENCE_CMAP)
        sm_ghg = plt.cm.ScalarMappable(norm=ghg_norm, cmap=SCIENCE_CMAP)
        cbar_cost = fig.colorbar(sm_cost, ax=axes[:, 0], orientation="horizontal", fraction=0.028, pad=0.062, aspect=55, extend="max")
        cbar_ghg = fig.colorbar(sm_ghg, ax=axes[:, 1], orientation="horizontal", fraction=0.028, pad=0.062, aspect=55, extend="max")
        format_shared_colorbar(cbar_cost, cost_label)
        format_shared_colorbar(cbar_ghg, ghg_label)
    else:
        for r, config in enumerate(["grid_connected", "hybrid", "off_grid"]):
            sm_cost = plt.cm.ScalarMappable(norm=make_plot_norm(cost_limits[config][0], cost_limits[config][1]), cmap=SCIENCE_CMAP)
            sm_ghg = plt.cm.ScalarMappable(norm=make_plot_norm(ghg_limits[config][0], ghg_limits[config][1]), cmap=SCIENCE_CMAP)
            cbar_cost = fig.colorbar(sm_cost, ax=axes[r, 0], orientation="horizontal", fraction=0.03, pad=0.05, aspect=36, extend="max")
            cbar_ghg = fig.colorbar(sm_ghg, ax=axes[r, 1], orientation="horizontal", fraction=0.03, pad=0.05, aspect=36, extend="max")
            format_shared_colorbar(cbar_cost, cost_label)
            format_shared_colorbar(cbar_ghg, ghg_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "shared_scale" if use_shared_colorbars else "subplot_scale"
    out_path = output_dir / f"ptx_{pathway}_global_results_{suffix}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figures(
    pathway: str,
    cost_max: float,
    ghg_max: float,
    output_dir: Path,
    dpi: int,
    spatial_input: Path | None = None,
    grid_input: Path | None = None,
) -> list[Path]:
    info = PTX_PATHWAY_INFO[pathway]
    world = load_world()
    grid_connected_df = load_grid_connected(pathway, grid_input=grid_input)
    spatial_df = load_spatial_results(pathway, spatial_input=spatial_input)

    if cost_max is None or ghg_max is None:
        auto_cost_max, auto_ghg_max = infer_color_limits(pathway, grid_connected_df, spatial_df)
        if cost_max is None:
            cost_max = auto_cost_max
        if ghg_max is None:
            ghg_max = auto_ghg_max

    grid_connected_map = build_grid_connected_map(world, grid_connected_df)
    gdf, gdf_ocean = build_spatial_grid(spatial_df, world)
    subplot_limits = infer_subplot_limits(info, grid_connected_df, spatial_df)

    cost_limits = {
        "shared": (info["conventional_cost"], cost_max),
        "grid_connected": subplot_limits[("grid_connected", "cost")],
        "hybrid": subplot_limits[("hybrid", "cost")],
        "off_grid": subplot_limits[("off_grid", "cost")],
    }
    ghg_limits = {
        "shared": (0.0, ghg_max),
        "grid_connected": subplot_limits[("grid_connected", "ghg")],
        "hybrid": subplot_limits[("hybrid", "ghg")],
        "off_grid": subplot_limits[("off_grid", "ghg")],
    }

    out_shared = render_map_figure(
        pathway=pathway,
        world=world,
        grid_connected_map=grid_connected_map,
        spatial_df=spatial_df,
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        cost_limits=cost_limits,
        ghg_limits=ghg_limits,
        output_dir=output_dir,
        dpi=dpi,
        use_shared_colorbars=True,
    )
    out_subplot = render_map_figure(
        pathway=pathway,
        world=world,
        grid_connected_map=grid_connected_map,
        spatial_df=spatial_df,
        gdf=gdf,
        gdf_ocean=gdf_ocean,
        cost_limits=cost_limits,
        ghg_limits=ghg_limits,
        output_dir=output_dir,
        dpi=dpi,
        use_shared_colorbars=False,
    )
    return [out_shared, out_subplot]


def main() -> None:
    args = parse_args()
    out_paths = make_figures(
        pathway=args.pathway,
        cost_max=args.cost_max,
        ghg_max=args.ghg_max,
        output_dir=args.output_dir,
        dpi=args.dpi,
        spatial_input=args.spatial_input,
        grid_input=args.grid_input,
    )
    for out_path in out_paths:
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

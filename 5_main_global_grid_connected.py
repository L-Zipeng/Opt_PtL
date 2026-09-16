# Grid-connected PtX optimization: ONE representative pixel per country (ISO2).
# single representative cell per ISO2 with valid metadata.
# Grid-connected assumes access to national grid electricity
# pixel location is less critical than the country.

import pandas as pd 
import json
import concurrent.futures
import pickle
import bw2data
import sys
import time
import numpy as np

# import own Python files, vars, mappings, and functions
from config import (NAME_REF_DB, COST_DATA_PTX, OUTPUT_FILE_XARRAY,
                    PROJECT_NAME, FILE_PATH_GLOBAL_RESULTS_PTX_GRID,
                    OUT_JSON_GHG_PTX, T_DAY_MEOH, T_DAY_SAF,
                    RUN_PTX, PTX_PATHWAYS)

import energy_data_processor as ep
import opt_ptx_functions as opt_ptx

# Set BW project
bw2data.projects.set_current(PROJECT_NAME)

COST_DICT_PTX = COST_DATA_PTX[NAME_REF_DB].to_dict() if COST_DATA_PTX is not None else None

def _load_dataset():
    # Load once in the main process to avoid Windows spawn reloading in workers.
    with open(OUTPUT_FILE_XARRAY, "rb") as f:
        return pickle.load(f)

# Read GHG info for PtX system components
try:
    with open(OUT_JSON_GHG_PTX, 'r') as file:
        DICT_GHG_IMPACTS_PTX = json.load(file)
except FileNotFoundError:
    DICT_GHG_IMPACTS_PTX = None

# ---- user-defined parameters ----
scenarios = {
    "grid_connected": {
        "autonomous_elect": False,
        "no_renewables": True,
        "consider_down_times": False,
        "w_cost": 1.0,
        "w_env": 0.0,
    }
}

# Toggle to run PtX cases in this script (config-driven)

def run_single_case_ptx(args, cost_dict=COST_DICT_PTX, dict_ghg_impacts=DICT_GHG_IMPACTS_PTX):
    """Run PtX optimization cases in parallel."""
    country, iso2, lat, lon, dac_el, dac_heat, scenario_name, pathway, kwargs = args

    if cost_dict is None or dict_ghg_impacts is None:
        return None

    cost_dict = cost_dict.copy()
    dict_ghg_impacts = dict_ghg_impacts.copy()

    dict_limits = ep.get_max_caps_regions()
    cost_dict["dr"] = ep.get_latest_avg_wacc(iso2)
    power_prices = ep.get_elect_prices(iso2)

    df_data = pd.DataFrame(
        data={
            "pv_MW_array": 0,
            "wind_MW_array_on": 0,
            "grid_abs_price": power_prices,
            "rev_inj": 0,
        },
        index=pd.date_range("1/1/{} 00:00".format(2023), periods=8760, freq="h"),
    )

    df_data["ghg_impact"] = ep.get_activity_env_elect_from_dict(iso2, db=NAME_REF_DB)
    df_data["ghg_impact_cons"] = 0
    if dac_el is not None:
        df_data["dac_el_kWh_per_kgCO2"] = np.asarray(dac_el)
    if dac_heat is not None:
        df_data["dac_heat_kWh_per_kgCO2"] = np.asarray(dac_heat)

    size_product_system = (T_DAY_MEOH if pathway == "meoh" else T_DAY_SAF) * 365
    raw_kwargs = dict(kwargs or {})
    w_cost = float(raw_kwargs.pop("w_cost", 1.0))
    w_env = float(raw_kwargs.pop("w_env", 0.0))
    calc_all_lca_impacts = bool(raw_kwargs.pop("calc_all_lca_impacts", False))
    export_alias = raw_kwargs.pop("export_alias", scenario_name)
    loc_elect = raw_kwargs.pop("loc_elect", iso2)
    allowed_kwargs = {
        "grid_inj",
        "credit_env_export",
        "autonomous_elect",
        "no_renewables",
        "heuristics",
        "h2_price",
        "euro_ton_co2",
        "eps_ghg_constraint",
        "hybrid_green",
        "ghg_baseline_tco2_per_tprod",
        "ghg_reduction",
        "consider_down_times",
        "sec_db",
        "export_results",
        "logger",
        "iis_path",
    }
    safe_kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed_kwargs}
    if pathway == "meoh":
        totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh(
            df_data,
            w_cost,
            w_env,
            cost_dict,
            dict_ghg_impacts,
            dict_limits,
            calc_all_lca_impacts=calc_all_lca_impacts,
            export_alias=export_alias,
            loc_elect=loc_elect,
            size_product_system=size_product_system,
            **safe_kwargs,
        )
    elif pathway == "meoh_to_saf":
        totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh_to_saf(
            df_data,
            w_cost,
            w_env,
            cost_dict,
            dict_ghg_impacts,
            dict_limits,
            calc_all_lca_impacts=calc_all_lca_impacts,
            export_alias=export_alias,
            loc_elect=loc_elect,
            size_product_system=size_product_system,
            **safe_kwargs,
        )
    elif pathway == "ftsaf":
        totals_cost_min, __, __ = opt_ptx.opt_dac_pem_ftsaf(
            df_data,
            w_cost,
            w_env,
            cost_dict,
            dict_ghg_impacts,
            dict_limits,
            calc_all_lca_impacts=calc_all_lca_impacts,
            export_alias=export_alias,
            loc_elect=loc_elect,
            size_product_system=size_product_system,
            **safe_kwargs,
        )
    else:
        raise ValueError(f"Unknown PtX pathway: {pathway}")

    if totals_cost_min is None:
        return None

    totals_cost_min["country"] = country
    totals_cost_min["iso2"] = iso2
    totals_cost_min["lat"] = lat
    totals_cost_min["lon"] = lon
    totals_cost_min["scenario"] = scenario_name
    totals_cost_min["ptx_pathway"] = pathway
    return totals_cost_min

# --- replace the job-building loop in main() with this unique-iso2 logic ---
def main_ptx():
    start_time = time.time()

    DS = _load_dataset()
    iso2_representative = {}
    for i_lat in range(len(DS.lat)):
        for i_lon in range(len(DS.lon)):
            cell = DS.isel(lat=i_lat, lon=i_lon)

            iso2 = cell.ISO_A2.item()
            if iso2 is None or (isinstance(iso2, float) and np.isnan(iso2)):
                continue
            if str(iso2).strip().upper() in ["NAN", "-99", ""]:
                continue
            if (
                ("dac_el_kWh_per_kgCO2" in cell)
                and ("dac_heat_kWh_per_kgCO2" in cell)
                and (not np.any(np.isnan(cell.dac_el_kWh_per_kgCO2)))
                and (not np.any(np.isnan(cell.dac_heat_kWh_per_kgCO2)))
            ):
                if iso2 not in iso2_representative:
                    iso2_representative[iso2] = {
                        "name": cell.NAME_EN.item(),
                        "lat": float(cell.lat.item()),
                        "lon": float(cell.lon.item()),
                        "dac_el": cell.dac_el_kWh_per_kgCO2.values,
                        "dac_heat": cell.dac_heat_kWh_per_kgCO2.values,
                    }

    jobs = []
    for iso2, info in iso2_representative.items():
        for scenario_name, kwargs in scenarios.items():
            for pathway in PTX_PATHWAYS:
                jobs.append(
                    (
                        info["name"],
                        iso2,
                        info["lat"],
                        info["lon"],
                        info["dac_el"],
                        info["dac_heat"],
                        scenario_name,
                        pathway,
                        kwargs,
                    )
                )

    print(f"Prepared {len(jobs)} PtX jobs (one representative per unique ISO2).")
    total_jobs = len(jobs)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        for i, res in enumerate(executor.map(run_single_case_ptx, jobs), 1):
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            avg_time_per_job = elapsed_time / i
            remaining_minutes = avg_time_per_job * (total_jobs - i) / 60

            sys.stdout.write(
                f"\rFinished {i}/{total_jobs} | Elapsed: {elapsed_minutes:.0f} min | "
                f"Remaining: {remaining_minutes:.0f} min, {remaining_minutes/60:.1f} hours"
            )
            sys.stdout.flush()

            if res is not None:
                results.append(res)

    totals_cost_all = pd.concat(results, ignore_index=True).set_index(['country', 'iso2', 'scenario', 'ptx_pathway'])
    totals_cost_all.to_pickle(FILE_PATH_GLOBAL_RESULTS_PTX_GRID)
    return totals_cost_all

if __name__ == "__main__":
    if RUN_PTX:
        totals_cost_all = main_ptx()
        print("PtX job done, results shape:", totals_cost_all.shape)

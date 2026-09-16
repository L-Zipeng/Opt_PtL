import pandas as pd
import json
import concurrent.futures
import bw2data
import sys
import time
import pickle
import numpy as np
import argparse
import os
from pathlib import Path

# Compatibility shim for libraries that still reference removed NumPy aliases.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from config import (NAME_REF_DB, COST_DATA_PTX, PROJECT_NAME,
                   LOCATIONS_SENS, FILE_PATH_SENS_ANALYSIS_PTX,
                   RUN_PTX, PTX_PATHWAYS, T_DAY_MEOH, T_DAY_SAF, OUT_JSON_GHG_PTX,
                   OUTPUT_FILE_XARRAY)

import opt_ptx_functions as opt_ptx
import energy_data_processor as ep

bw2data.projects.set_current(PROJECT_NAME)

COST_DICT_PTX = COST_DATA_PTX[NAME_REF_DB].to_dict() if COST_DATA_PTX is not None else None

# Shared worker cache (set by executor initializer)
SITE_CACHE = {}

try:
    with open(OUT_JSON_GHG_PTX, 'r') as file:
        DICT_GHG_IMPACTS_PTX = json.load(file)
except FileNotFoundError:
    DICT_GHG_IMPACTS_PTX = None

scenarios = {
    "grid_connected": {"threads": 1, "time_limit": 1800},
    "hybrid":         {"threads": 1, "time_limit": 900},
    "off_grid":       {"threads": 1, "time_limit": 900},
}

SKIP_SENSITIVITIES_BY_SCENARIO = {
    # Renewable CF perturbations have no effect when renewables are disabled.
    "grid_connected": {"cf_pv", "cf_wind"},
}

search_mapping_for_sensitivity = {
    'power_prices': "Electricity grid price",
    "dr": "Discount rate",
    "dac_capex": "Capex DAC-unit",
    "hp_cop": "Heat pump COP",
    "meoh_capex": "Capex MeOH-unit",
    "mtj_capex": "Capex MTJ-unit",
    "ftsaf_capex": "Capex FT-SAF unit",
    "pv_capex": "Capex solar PV",
    "bat_en_capex": "Capex battery (energy)",
    "wind_on_capex": "Capex onshore wind",
    "electr_eff": "Electrolyzer efficiency",
    "electr_capex": "Electrolyzer capex",
    'cf_wind': "Capacity factor wind",
    'cf_pv': "Capacity factor solar",
}

factors_run = [0.2, -0.2]
RESULT_INDEX = ['country', 'iso2', 'scenario', 'ptx_pathway', 'sensitivity', 'factor']


def _empty_results_frame():
    """Return an empty result frame with the expected multi-index."""
    return pd.DataFrame(columns=RESULT_INDEX).set_index(RESULT_INDEX)


def _load_existing_results(path):
    """Load existing sensitivity results if present; otherwise return an empty frame."""
    if not Path(path).is_file():
        return _empty_results_frame()
    existing = pd.read_pickle(path)
    if not isinstance(existing, pd.DataFrame):
        raise TypeError(f"Existing results at {path} are not a DataFrame: {type(existing)!r}")
    if list(existing.index.names) != RESULT_INDEX:
        if set(RESULT_INDEX).issubset(existing.columns):
            existing = existing.set_index(RESULT_INDEX)
        elif existing.empty:
            existing = _empty_results_frame()
        else:
            raise ValueError(
                f"Existing results at {path} have unexpected index names: {existing.index.names}"
            )
    return existing


def _merge_results(existing, new_results):
    """Append new successful cases and keep the newest entry for duplicate case keys."""
    if existing.empty:
        return new_results.sort_index()
    if new_results.empty:
        return existing.sort_index()
    merged = pd.concat([existing, new_results])
    merged = merged[~merged.index.duplicated(keep='last')]
    return merged.sort_index()


def _load_existing_failures(path):
    """Load existing failed-case log if present; otherwise return an empty DataFrame."""
    if not Path(path).is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_failures(existing, new_failures):
    """Append failure rows and keep the newest record for each failed case key."""
    if existing.empty:
        merged = new_failures.copy()
    elif new_failures.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_failures], ignore_index=True)

    if merged.empty:
        return merged

    dedupe_cols = [
        col for col in ["country", "iso2", "scenario", "ptx_pathway", "sensitivity", "factor_change"]
        if col in merged.columns
    ]
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    return merged.sort_values(dedupe_cols).reset_index(drop=True) if dedupe_cols else merged

def _init_worker(site_cache):
    """Initialize per-process shared cache for pixel-level renewable and DAC profiles."""
    global SITE_CACHE
    SITE_CACHE = site_cache


def _build_site_cache():
    """Load DS once and pre-extract renewable and DAC profiles for sensitivity locations."""
    with open(OUTPUT_FILE_XARRAY, "rb") as f:
        ds = pickle.load(f)

    cache = {}
    for country, iso2, lat, lon in LOCATIONS_SENS:
        key = (country, iso2, lat, lon)
        cell = ds.sel(lat=lat, lon=lon, method="nearest")
        cache[key] = {
            "cf_pv": np.asarray(cell.cf_solar.values, dtype=float).copy(),
            "cf_wind": np.asarray(cell.cf_wind.values, dtype=float).copy(),
            "dac_el": np.asarray(cell.dac_el_kWh_per_kgCO2.values, dtype=float).copy(),
            "dac_heat": np.asarray(cell.dac_heat_kWh_per_kgCO2.values, dtype=float).copy(),
        }
    return cache


def run_single_case_ptx(args, cost_dict=COST_DICT_PTX, dict_ghg_impacts=DICT_GHG_IMPACTS_PTX):
    """Run one PtX sensitivity case in parallel."""
    (
        country,
        iso2,
        lat,
        lon,
        scenario_name,
        pathway,
        sens_factor,
        factor_change,
        kwargs,
        run_overrides,
    ) = args
    case_meta = {
        "country": country,
        "iso2": iso2,
        "scenario": scenario_name,
        "ptx_pathway": pathway,
        "sensitivity": sens_factor,
        "factor_change": factor_change,
    }

    if cost_dict is None or dict_ghg_impacts is None:
        return {"ok": False, "reason": "missing_inputs", **case_meta}

    try:
        factor = 1 + factor_change
        cost_dict = cost_dict.copy()
        dict_ghg_impacts = dict_ghg_impacts.copy()

        dict_limits = ep.get_max_caps_regions()
        cost_dict['dr'] = ep.get_latest_avg_wacc(iso2) * factor if sens_factor == 'dr' else ep.get_latest_avg_wacc(iso2)
        power_prices = ep.get_elect_prices(iso2) * factor if sens_factor == 'power_prices' else ep.get_elect_prices(iso2)

        # Use worker-local precomputed hourly renewable and DAC profiles (offline-safe).
        key = (country, iso2, lat, lon)
        if key not in SITE_CACHE:
            raise KeyError(f"Missing site cache entry for {key}")
        site_data = SITE_CACHE[key]
        cf_pv = np.asarray(site_data["cf_pv"], dtype=float).copy()
        cf_wind = np.asarray(site_data["cf_wind"], dtype=float).copy()
        dac_el = np.asarray(site_data["dac_el"], dtype=float).copy()
        dac_heat = np.asarray(site_data["dac_heat"], dtype=float).copy()

        if sens_factor == 'cf_pv':
            cf_pv *= factor
        elif sens_factor == 'cf_wind':
            cf_wind *= factor
        elif sens_factor == 'hp_cop':
            if 'hp_cop' in cost_dict:
                cost_dict['hp_cop'] *= factor
        else:
            for k in cost_dict:
                if sens_factor == 'om' and ('_om' in str(k)):
                    cost_dict[k] *= factor
                elif sens_factor == k:
                    cost_dict[k] *= factor

        df_data = pd.DataFrame(
            data={
                'pv_MW_array': cf_pv,
                'wind_MW_array_on': cf_wind,
                "grid_abs_price": power_prices,
                "rev_inj": 0,
            },
            index=pd.date_range('1/1/{} 00:00'.format(2023), periods=8760, freq='h'),
        )

        df_data['ghg_impact'] = ep.get_activity_env_elect_from_dict(iso2, db=NAME_REF_DB)
        df_data['ghg_impact_cons'] = 0
        df_data["dac_el_kWh_per_kgCO2"] = dac_el
        df_data["dac_heat_kWh_per_kgCO2"] = dac_heat

        # Apply scenario-specific constraints (must match notebook 6)
        if scenario_name == "grid_connected":
            df_data["pv_MW_array"] = 0
            df_data["wind_MW_array_on"] = 0
        elif scenario_name == "off_grid":
            dict_limits["max_grid_cap"] = 0
            df_data["grid_abs_price"] = 0
        size_product_system = (T_DAY_MEOH if pathway == "meoh" else T_DAY_SAF) * 365

        opt_kwargs = {k: v for k, v in kwargs.items() if k in {
            "hybrid_green", "ghg_reduction", "ghg_baseline_tco2_per_tprod",
            "consider_down_times", "export_results", "threads", "time_limit",
            "mip_gap", "int_feas_tol", "heuristics", "iis_path",
        }}
        opt_kwargs.update({k: v for k, v in run_overrides.items() if v is not None})

        if pathway == "meoh":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh(
                df_data, 1, 0, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=False, export_alias=scenario_name,
                loc_elect=iso2, size_product_system=size_product_system,
                **opt_kwargs
            )
        elif pathway == "meoh_to_saf":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh_to_saf(
                df_data, 1, 0, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=False, export_alias=scenario_name,
                loc_elect=iso2, size_product_system=size_product_system,
                **opt_kwargs
            )
        elif pathway == "ftsaf":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_ftsaf(
                df_data, 1, 0, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=False, export_alias=scenario_name,
                loc_elect=iso2, size_product_system=size_product_system,
                **opt_kwargs
            )
        else:
            raise ValueError(f"Unknown PtX pathway: {pathway}")

        if totals_cost_min is None:
            print(
                f"\nSkipped (no feasible solution found): "
                f"{country}, {scenario_name}, {pathway}, {sens_factor}, {factor_change}"
            )
            return {"ok": False, "reason": "no_feasible_solution", **case_meta}
        if isinstance(totals_cost_min, pd.Series):
            totals_cost_min = totals_cost_min.to_frame().T
        elif isinstance(totals_cost_min, dict):
            totals_cost_min = pd.DataFrame([totals_cost_min])
        elif not isinstance(totals_cost_min, pd.DataFrame):
            raise TypeError(f"Unexpected result type: {type(totals_cost_min)!r}")

        if totals_cost_min.empty:
            return {"ok": False, "reason": "empty_result", **case_meta}

        totals_cost_min['country'] = country
        totals_cost_min['iso2'] = iso2
        totals_cost_min['scenario'] = scenario_name
        totals_cost_min['ptx_pathway'] = pathway
        totals_cost_min['sensitivity'] = sens_factor
        totals_cost_min['factor'] = factor
        return {"ok": True, "data": totals_cost_min}
    except Exception as exc:
        print(f"\nCase failed: {country}, {scenario_name}, {pathway}, {sens_factor}, {factor_change}: {exc}")
        return {"ok": False, "reason": str(exc), **case_meta}


def main_ptx(
    filter_scenario=None,
    filter_pathway=None,
    filter_sens=None,
    *,
    max_workers=5,
    solver_threads=None,
    time_limit_scale=1.0,
    solver_mip_gap=None,
    solver_int_feas_tol=None,
    solver_heuristics=False,
    rerun_existing=False,
):
    start_time = time.time()
    site_cache = _build_site_cache()
    existing_results = _load_existing_results(FILE_PATH_SENS_ANALYSIS_PTX)

    active_scenarios = {
        k: v for k, v in scenarios.items()
        if filter_scenario is None or k == filter_scenario
    }
    active_pathways = [
        p for p in PTX_PATHWAYS
        if filter_pathway is None or p == filter_pathway
    ]
    active_sens = [
        s for s in search_mapping_for_sensitivity.keys()
        if filter_sens is None or s == filter_sens
    ]

    if not active_scenarios:
        raise ValueError(f"Unknown scenario '{filter_scenario}'. Choose from: {list(scenarios)}")
    if not active_pathways:
        raise ValueError(f"Unknown pathway '{filter_pathway}'. Choose from: {PTX_PATHWAYS}")
    if not active_sens:
        raise ValueError(f"Unknown sensitivity factor '{filter_sens}'. Choose from: {list(search_mapping_for_sensitivity)}")

    run_overrides = {
        "threads": solver_threads,
        "mip_gap": solver_mip_gap,
        "int_feas_tol": solver_int_feas_tol,
        "heuristics": True if solver_heuristics else None,
        "iis_path": None,
    }

    jobs = [
        (
            country,
            iso2,
            lat,
            lon,
            scenario_name,
            pathway,
            sens_factor,
            factor_change,
            {
                **kwargs,
                "time_limit": int(kwargs["time_limit"] * time_limit_scale),
            },
            run_overrides,
        )
        for country, iso2, lat, lon in LOCATIONS_SENS
        for scenario_name, kwargs in active_scenarios.items()
        for pathway in active_pathways
        for sens_factor in active_sens
        for factor_change in factors_run
        if sens_factor not in SKIP_SENSITIVITIES_BY_SCENARIO.get(scenario_name, set())
    ]

    if not rerun_existing and not existing_results.empty:
        existing_keys = set(existing_results.index.tolist())
        jobs = [
            job for job in jobs
            if (
                job[0],
                job[1],
                job[4],
                job[5],
                job[6],
                1 + job[7],
            ) not in existing_keys
        ]

    print(f"Running {len(jobs)} total PtX sensitivity cases...")
    total_jobs = len(jobs)

    if total_jobs == 0:
        print(
            "No pending sensitivity jobs. Existing results already cover the selected "
            "scenario/pathway/sensitivity filters."
        )
        return existing_results

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(site_cache,)
    ) as executor:
        results = []
        failures = []
        for i, res in enumerate(executor.map(run_single_case_ptx, jobs), 1):
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            avg_time_per_job = elapsed_time / i
            remaining_minutes = avg_time_per_job * (total_jobs - i) / 60

            sys.stdout.write(
                f"\rFinished {i}/{total_jobs} | Elapsed: {elapsed_minutes:.2f} min | "
                f"Remaining: {remaining_minutes:.2f} min"
            )
            sys.stdout.flush()

            if res and res.get("ok") and isinstance(res.get("data"), pd.DataFrame):
                if not res["data"].empty:
                    results.append(res["data"])
            else:
                failures.append(res if isinstance(res, dict) else {"ok": False, "reason": "unknown_worker_result"})

    success_count = len(results)
    fail_count = len(failures)
    print(f"\nCompleted sensitivity batch: {success_count} successful, {fail_count} failed/infeasible")

    fail_path = FILE_PATH_SENS_ANALYSIS_PTX.replace(".pkl", "_failed_cases.csv")
    existing_failures = _load_existing_failures(fail_path)
    if failures:
        merged_failures = _merge_failures(existing_failures, pd.DataFrame(failures))
        merged_failures.to_csv(fail_path, index=False)
        print(f"Failure log written to: {fail_path}")
    elif not existing_failures.empty:
        print(f"No new failures. Keeping existing failure log at: {fail_path}")

    if not results:
        if existing_results.empty:
            totals_cost_all = _empty_results_frame()
            totals_cost_all.to_pickle(FILE_PATH_SENS_ANALYSIS_PTX)
            print(
                "No successful sensitivity cases. Saved empty results file. "
                "Check infeasibilities and the failure log."
            )
            return totals_cost_all

        print(
            "No new successful sensitivity cases. Preserving existing results at "
            f"{FILE_PATH_SENS_ANALYSIS_PTX}."
        )
        return existing_results

    new_results = pd.concat(results, ignore_index=True).set_index(RESULT_INDEX)
    totals_cost_all = _merge_results(existing_results, new_results)
    totals_cost_all.to_pickle(FILE_PATH_SENS_ANALYSIS_PTX)
    print(
        f"Saved merged sensitivity results to {FILE_PATH_SENS_ANALYSIS_PTX} "
        f"({len(existing_results)} existing + {len(new_results)} new -> {len(totals_cost_all)} unique rows)."
    )
    return totals_cost_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PtX sensitivity analysis")
    parser.add_argument(
        "--scenario", default=None,
        choices=list(scenarios.keys()),
        help="Run only this scenario (default: all)"
    )
    parser.add_argument(
        "--pathway", default=None,
        choices=PTX_PATHWAYS,
        help="Run only this pathway (default: all)"
    )
    parser.add_argument(
        "--sens", default=None,
        choices=list(search_mapping_for_sensitivity.keys()),
        help="Run only this sensitivity factor (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, default=min(5, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: min(5, cpu_count))"
    )
    parser.add_argument(
        "--solver-threads", type=int, default=1,
        help="Gurobi threads per optimization job (default: 1)"
    )
    parser.add_argument(
        "--time-limit-scale", type=float, default=1.0,
        help="Multiply per-scenario time limits by this factor (default: 1.0)"
    )
    parser.add_argument(
        "--mip-gap", type=float, default=None,
        help="Override solver MIP gap for all jobs (example: 0.1)"
    )
    parser.add_argument(
        "--int-feas-tol", type=float, default=None,
        help="Override solver integer feasibility tolerance for all jobs"
    )
    parser.add_argument(
        "--heuristics", action="store_true",
        help="Enable Gurobi heuristic search for all jobs"
    )
    parser.add_argument(
        "--rerun-existing", action="store_true",
        help="Rerun jobs even if they already exist in the saved results"
    )
    args = parser.parse_args()

    if RUN_PTX:
        totals_cost_all = main_ptx(
            filter_scenario=args.scenario,
            filter_pathway=args.pathway,
            filter_sens=args.sens,
            max_workers=args.workers,
            solver_threads=args.solver_threads,
            time_limit_scale=args.time_limit_scale,
            solver_mip_gap=args.mip_gap,
            solver_int_feas_tol=args.int_feas_tol,
            solver_heuristics=args.heuristics,
            rerun_existing=args.rerun_existing,
        )
        print("\nPtX sensitivity done, results shape:", totals_cost_all.shape)

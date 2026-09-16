#!/usr/bin/env python3
"""
Euler-optimized PtX global optimization script.
Incorporates best practices from NH3 Euler runs.
"""

import os
import sys
import time
import json
import traceback
import logging
import multiprocessing as mp
import concurrent.futures
import pickle
import pandas as pd
import numpy as np
import pyarrow  # For parquet support

# Import own modules
from config import (
    NAME_REF_DB, COST_DATA_PTX, OUTPUT_FILE_XARRAY,
    FILE_PATH_GLOBAL_RESULTS_PTX, OUT_JSON_GHG_PTX,
    T_DAY_MEOH, T_DAY_SAF, PTX_PATHWAYS
)
import energy_data_processor as ep
import opt_ptx_functions as opt_ptx

# ==============================================================================
# EULER-SPECIFIC OPTIMIZATIONS
# ==============================================================================

# 1. Logging setup (writes to scratch if available)
log_filename = os.getenv('LOG_FILE', f"output_{os.getenv('SLURM_JOB_ID', 'local')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 2. Multiprocessing start method (fork for memory efficiency on Linux)
if sys.platform.startswith("linux"):
    try:
        mp.set_start_method("fork", force=True)
        logger.info("Using 'fork' start method for shared memory efficiency.")
    except RuntimeError:
        pass  # already set

# 3. Limit internal threads per process (avoid oversubscription)
for var in [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "GRB_NUM_THREADS"
]:
    os.environ[var] = "1"

# 4. Shared globals for large datasets (forked processes inherit these)
_GLOBAL_DS = None
_GLOBAL_COST_DICT = None
_GLOBAL_GHG_DICT = None

def _set_shared_globals(ds, cost_dict, ghg_dict):
    global _GLOBAL_DS, _GLOBAL_COST_DICT, _GLOBAL_GHG_DICT
    _GLOBAL_DS = ds
    _GLOBAL_COST_DICT = cost_dict
    _GLOBAL_GHG_DICT = ghg_dict

# ==============================================================================
# LOAD DATA ONCE (child processes inherit via fork)
# ==============================================================================

logger.info("Loading dataset and parameters...")
with open(OUTPUT_FILE_XARRAY, "rb") as f:
    DS = pickle.load(f)

COST_DICT_PTX = COST_DATA_PTX[NAME_REF_DB].to_dict() if COST_DATA_PTX is not None else None

try:
    with open(OUT_JSON_GHG_PTX, 'r') as file:
        DICT_GHG_IMPACTS_PTX = json.load(file)
except FileNotFoundError:
    logger.error(f"GHG impacts file not found: {OUT_JSON_GHG_PTX}")
    DICT_GHG_IMPACTS_PTX = None

_set_shared_globals(DS, COST_DICT_PTX, DICT_GHG_IMPACTS_PTX)
logger.info(f"Dataset loaded: {len(DS.lat)} × {len(DS.lon)} = {len(DS.lat) * len(DS.lon)} pixels")

# ==============================================================================
# SCENARIO DEFINITIONS
# ==============================================================================

scenarios = {
    "grid_connected": {
        "autonomous_elect": False,
        "no_renewables": True,
        "consider_down_times": False,
        "w_cost": 1.0,
        "w_env": 0.0,
    },
    "hybrid": {
        "autonomous_elect": False,
        "no_renewables": False,
        "consider_down_times": False,
        "w_cost": 1.0,
        "w_env": 0.0,
    },
    "off_grid": {
        "autonomous_elect": True,
        "no_renewables": False,
        "consider_down_times": False,
        "w_cost": 1.0,
        "w_env": 0.0,
    },
}


TEST_DAILY_AVG = False  # Set True for test runs
MAX_PTX_PIXELS = int(os.getenv("MAX_PTX_PIXELS", "0"))  # 0 = no limit
SAVE_INTERVAL = int(os.getenv("PTX_SAVE_INTERVAL", "500"))
ACTIVE_PTX_PATHWAYS = list(PTX_PATHWAYS)


def _parse_csv_env(name):
    raw = os.getenv(name, "")
    return [v.strip() for v in raw.split(",") if v.strip()]


def _apply_env_test_overrides():
    global scenarios, ACTIVE_PTX_PATHWAYS

    selected_scenarios = _parse_csv_env("PTX_TEST_SCENARIOS")
    if selected_scenarios:
        valid = set(scenarios.keys())
        invalid = [s for s in selected_scenarios if s not in valid]
        if invalid:
            raise ValueError(f"Invalid PTX_TEST_SCENARIOS: {invalid}. Allowed: {sorted(valid)}")
        scenarios = {s: scenarios[s] for s in selected_scenarios}

    selected_pathways = _parse_csv_env("PTX_TEST_PATHWAYS")
    if selected_pathways:
        valid = set(PTX_PATHWAYS)
        invalid = [p for p in selected_pathways if p not in valid]
        if invalid:
            raise ValueError(f"Invalid PTX_TEST_PATHWAYS: {invalid}. Allowed: {sorted(valid)}")
        ACTIVE_PTX_PATHWAYS = selected_pathways

    logger.info("Active scenarios: %s", list(scenarios.keys()))
    logger.info("Active pathways: %s", ACTIVE_PTX_PATHWAYS)


_apply_env_test_overrides()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _build_df_data(cell, power_prices, iso2):
    df_data = pd.DataFrame(
        data={
            "pv_MW_array": cell.cf_solar.values,
            "wind_MW_array_on": cell.cf_wind.values,
            "grid_abs_price": power_prices,
            "rev_inj": 0,
        },
        index=pd.date_range("1/1/2023 00:00", periods=8760, freq="h"),
    )
    df_data["ghg_impact"] = ep.get_activity_env_elect_from_dict(iso2, db=NAME_REF_DB)
    df_data["ghg_impact_cons"] = 0
    return df_data

def _to_daily_avg(df_data):
    df_daily = df_data.resample("D").mean()
    df_daily.index = pd.date_range("1/1/2023 00:00", periods=len(df_daily), freq="D")
    return df_daily


def _cell_has_required_ptx_inputs(cell):
    return (
        (not np.any(np.isnan(cell.cf_solar)))
        and (not np.any(np.isnan(cell.cf_wind)))
        and ("dac_el_kWh_per_kgCO2" in cell)
        and ("dac_heat_kWh_per_kgCO2" in cell)
        and (not np.any(np.isnan(cell.dac_el_kWh_per_kgCO2)))
        and (not np.any(np.isnan(cell.dac_heat_kWh_per_kgCO2)))
        and (not pd.isna(cell.ISO_A2.item()))
    )


def _cell_selected_for_ptx(cell):
    if not _cell_has_required_ptx_inputs(cell):
        return False

    if "ptx_thinned_ok" in cell:
        try:
            return bool(cell.ptx_thinned_ok.item())
        except Exception:
            return False

    return True

# ==============================================================================
# OPTIMIZATION WORKER
# ==============================================================================

def run_single_case_ptx(args):
    """Run PtX optimization cases in parallel (uses global shared data)."""
    country, iso2, lat, lon, scenario_name, pathway, kwargs = args

    global _GLOBAL_DS, _GLOBAL_COST_DICT, _GLOBAL_GHG_DICT
    
    if _GLOBAL_COST_DICT is None or _GLOBAL_GHG_DICT is None:
        return ([iso2, lat, lon, scenario_name, pathway], None)

    cost_dict = _GLOBAL_COST_DICT.copy()
    dict_ghg_impacts = _GLOBAL_GHG_DICT.copy()

    try:
        dict_limits = ep.get_max_caps_regions()
        cost_dict["dr"] = ep.get_latest_avg_wacc(iso2)
        power_prices = ep.get_elect_prices(iso2)

        cell = _GLOBAL_DS.sel(lat=lat, lon=lon, method="nearest")
        dac_el = cell.dac_el_kWh_per_kgCO2.values if "dac_el_kWh_per_kgCO2" in cell else None
        dac_heat = cell.dac_heat_kWh_per_kgCO2.values if "dac_heat_kWh_per_kgCO2" in cell else None

        df_data = _build_df_data(cell, power_prices, iso2)
        if TEST_DAILY_AVG:
            df_data = _to_daily_avg(df_data)
        if dac_el is not None:
            df_data["dac_el_kWh_per_kgCO2"] = dac_el
        if dac_heat is not None:
            df_data["dac_heat_kWh_per_kgCO2"] = dac_heat

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
            "battery_binary_fallback",
            "battery_overlap_tol",
            "battery_overlap_energy_tol",
            "battery_cycle_penalty",
            "sec_db",
            "export_results",
            "logger",
            "iis_path",
        }
        safe_kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed_kwargs}
        
        if pathway == "meoh":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh(
                df_data, w_cost, w_env, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=calc_all_lca_impacts, export_alias=export_alias,
                loc_elect=loc_elect, size_product_system=size_product_system,
                **safe_kwargs,
            )
        elif pathway == "meoh_to_saf":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_meoh_to_saf(
                df_data, w_cost, w_env, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=calc_all_lca_impacts, export_alias=export_alias,
                loc_elect=loc_elect, size_product_system=size_product_system,
                **safe_kwargs,
            )
        elif pathway == "ftsaf":
            totals_cost_min, __, __ = opt_ptx.opt_dac_pem_ftsaf(
                df_data, w_cost, w_env, cost_dict, dict_ghg_impacts, dict_limits,
                calc_all_lca_impacts=calc_all_lca_impacts, export_alias=export_alias,
                loc_elect=loc_elect, size_product_system=size_product_system,
                **safe_kwargs,
            )
        else:
            raise ValueError(f"Unknown PtX pathway: {pathway}")

        totals_cost_min["country"] = country
        totals_cost_min["iso2"] = iso2
        totals_cost_min["scenario"] = scenario_name
        totals_cost_min["lat"] = lat
        totals_cost_min["lon"] = lon
        totals_cost_min["ptx_pathway"] = pathway
        
        return ([iso2, lat, lon, scenario_name, pathway], totals_cost_min)

    except Exception as e:
        # Write failed cases to scratch if available
        scratch = os.environ.get("SCRATCHDIR") or os.environ.get("SLURM_TMPDIR") or "."
        failed_log = os.path.join(scratch, "failed_cases.log")
        error_info = {
            "case": [iso2, lat, lon, scenario_name, pathway],
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            with open(failed_log, "a") as f:
                f.write(json.dumps(error_info) + "\n")
        except Exception:
            # Fallback to cwd
            with open("failed_cases.log", "a") as f:
                f.write(json.dumps(error_info) + "\n")

        return ([iso2, lat, lon, scenario_name, pathway], None)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main_ptx():
    start_time = time.time()
    jobs = []
    max_pixels = MAX_PTX_PIXELS
    if max_pixels > 0:
        logger.info("Limiting to first %s valid pixels (MAX_PTX_PIXELS).", max_pixels)

    logger.info("Building job list...")
    if "ptx_thinned_ok" in DS:
        logger.info("Using ptx_thinned_ok mask for PtX pixel selection.")
    else:
        logger.warning("ptx_thinned_ok not found in dataset. Falling back to all valid PtX pixels.")
    pixel_count = 0
    stop_early = False
    for i_lat in range(len(DS.lat)):
        if stop_early:
            break
        for i_lon in range(len(DS.lon)):
            if max_pixels > 0 and pixel_count >= max_pixels:
                stop_early = True
                break
            cell = DS.isel(lat=i_lat, lon=i_lon)
            
            if _cell_selected_for_ptx(cell):
                pixel_count += 1
                name = cell.NAME_EN.item()
                iso2 = cell.ISO_A2.item()
                lat_val = cell.lat.item()
                lon_val = cell.lon.item()
                for scenario_name, kwargs in scenarios.items():
                    for pathway in ACTIVE_PTX_PATHWAYS:
                        jobs.append((name, iso2, lat_val, lon_val, scenario_name, pathway, kwargs))

    total_jobs = len(jobs)
    logger.info(f"Running {total_jobs} PtX optimizations...")
    
    if total_jobs == 0:
        raise RuntimeError("No PtX jobs created. Check cf_solar/cf_wind and DAC inputs.")

    # Prepare output paths early so periodic checkpoints survive wall-time timeouts.
    scratchdir = os.getenv("SCRATCHDIR") or os.getenv("SLURM_TMPDIR")
    default_out_path = (
        os.path.join(scratchdir, "results", os.path.basename(FILE_PATH_GLOBAL_RESULTS_PTX))
        if scratchdir
        else FILE_PATH_GLOBAL_RESULTS_PTX
    )
    out_path_pkl = os.getenv("PTX_OUTPUT_FILE", default_out_path)
    out_dir = os.path.dirname(out_path_pkl) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path_parquet = out_path_pkl.replace(".pkl", ".parquet")
    partial_path_pkl = out_path_pkl.replace(".pkl", ".partial.pkl")
    partial_path_parquet = out_path_pkl.replace(".pkl", ".partial.parquet")

    # SLURM-aware CPU detection
    slurm_ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    env_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    slurm_cpus = int(env_cpus) if env_cpus is not None else os.cpu_count() or 1

    # Avoid nested multiprocessing if using multiple SLURM tasks
    if slurm_ntasks > 1:
        logger.warning(
            f"SLURM_NTASKS={slurm_ntasks} > 1 → disabling internal pool to avoid nested parallelism."
        )
        max_workers = 1
    else:
        # Use available CPUs (cap at 48 to be safe)
        max_workers = min(slurm_cpus, 48)

    logger.info(f"Using {max_workers} worker(s) [SLURM_CPUS_PER_TASK={slurm_cpus}, SLURM_NTASKS={slurm_ntasks}]")

    results = []
    failed_cases = 0
    failed_examples = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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

                case_data, df = res
                if df is None:
                    failed_cases += 1
                    if len(failed_examples) < 5:
                        failed_examples.append(case_data)
                    logger.warning(f"\nCase failed: {case_data}")
                else:
                    results.append(df)

                # Periodic checkpoint save to avoid losing all progress on timeout.
                if (i % SAVE_INTERVAL == 0) or (i == total_jobs):
                    if results:
                        try:
                            partial_df = pd.concat(results, ignore_index=True).set_index(
                                ['country', 'iso2', 'scenario', 'lat', 'lon', 'ptx_pathway']
                            )
                            partial_df.to_pickle(partial_path_pkl)
                            partial_df.to_parquet(partial_path_parquet, index=True)
                            logger.info(
                                "Saved partial results at %s/%s to %s and %s",
                                i,
                                total_jobs,
                                partial_path_pkl,
                                partial_path_parquet,
                            )
                        except Exception as exc:
                            logger.warning("Failed to save partial results at %s/%s: %s", i, total_jobs, exc)

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt — saving partial results...")

    total_success = len(results)
    logger.info(
        "Completed PtX cases: %s success, %s failed",
        total_success,
        failed_cases,
    )
    if failed_examples:
        logger.info("Example failed cases: %s", failed_examples)

    if total_success == 0:
        logger.error("All PtX cases failed or no results returned. Check failed_cases.log.")
        empty_cols = [
            "country",
            "iso2",
            "scenario",
            "lat",
            "lon",
            "ptx_pathway",
        ]
        totals_cost_all = pd.DataFrame(columns=empty_cols).set_index(empty_cols)
    else:
        totals_cost_all = pd.concat(results, ignore_index=True).set_index(
            ['country', 'iso2', 'scenario', 'lat', 'lon', 'ptx_pathway']
        )
    
    # Save as both pickle and parquet
    totals_cost_all.to_pickle(out_path_pkl)
    totals_cost_all.to_parquet(out_path_parquet, index=True)
    
    logger.info(f"Results saved: {out_path_pkl} and {out_path_parquet}")
    logger.info(f"Results shape: {totals_cost_all.shape}")
    
    return totals_cost_all

if __name__ == "__main__":
    totals_cost_all = main_ptx()
    logger.info(f"Job done, results shape: {totals_cost_all.shape}")

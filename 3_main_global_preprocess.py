# In this Python script, we get the prepared xarray with certain resolutions
# And then, we fetch weather data, and calculate hourly capacity factors of
# onshore wind and solar PV using solarPV GIS.

import concurrent.futures
import sys
import time
import pickle
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import json
from io import StringIO

# import own Python files, vars, mappings, and functions
from config import (
    OUTPUT_FILE_XARRAY,
    OUTPUT_FILE_XARRAY_INIT,
    LOCATIONS,
    LOCATIONS_SENS,
    N_DEMAND_THRESHOLD,
    DAC_EL_BASE,
    DAC_EL_TEMP_COEF,
    DAC_EL_REF_TEMP,
    DAC_EL_RH_COEF,
    DAC_EL_REF_RH,
    DAC_TABLE_ELECTRICITY_PATH,
    DAC_TABLE_HEAT_PATH,
    DAC_RH_DEFAULT,
    PVGIS_CACHE_DIR,
)
import calculate_renewable_yield as crp

# NOTE: Dataset is loaded inside main() to avoid Windows multiprocessing memory issues
# Global reference will be set when needed
ds = None

def _load_dac_table(path: str):
    df = pd.read_csv(path)
    rh_vals = df["RH"].astype(float).to_numpy()
    temp_vals = np.array([float(c) for c in df.columns if c != "RH"])
    table = df[[str(int(t)) for t in temp_vals]].to_numpy(dtype=float)
    # Input tables are kWh/t CO2; convert to kWh/kg CO2 for the model.
    table = table / 1000.0
    return rh_vals, temp_vals, table

def _interp_dac_table(temp_c: np.ndarray, rh_pct: np.ndarray, rh_vals, temp_vals, table):
    # Clip to table bounds for stable interpolation
    temp_c = np.clip(temp_c, temp_vals.min(), temp_vals.max())
    rh_pct = np.clip(rh_pct, rh_vals.min(), rh_vals.max())

    # Interpolate along temperature for each RH row
    temp_interp = np.vstack([np.interp(temp_c, temp_vals, row) for row in table])

    # Interpolate along RH for each time step
    out = np.empty_like(temp_c, dtype=float)
    for i in range(temp_c.size):
        out[i] = np.interp(rh_pct[i], rh_vals, temp_interp[:, i])
    return np.clip(out, 0, None)

_DAC_RH_ELEC, _DAC_T_ELEC, _DAC_TABLE_ELEC = _load_dac_table(DAC_TABLE_ELECTRICITY_PATH)
_DAC_RH_HEAT, _DAC_T_HEAT, _DAC_TABLE_HEAT = _load_dac_table(DAC_TABLE_HEAT_PATH)

def _extract_relative_humidity(weather_data: pd.DataFrame, temp_air_c: np.ndarray):
    for col in ("rel_humidity", "relative_humidity", "rh"):
        if col in weather_data:
            rh = pd.to_numeric(weather_data[col], errors="coerce").to_numpy(dtype=float)
            break
    else:
        rh = None

    if rh is None or np.isnan(rh).all():
        rh = np.full_like(temp_air_c, DAC_RH_DEFAULT, dtype=float)

    if rh.shape[0] != temp_air_c.shape[0]:
        rh = np.resize(rh, temp_air_c.shape[0])
        rh = np.where(np.isnan(rh), DAC_RH_DEFAULT, rh)

    return np.clip(rh, 0, 100)

def _load_cached_weather(lat_val: float, lon_val: float):
    cache_path = Path(PVGIS_CACHE_DIR) / f"pvgis_{lat_val:.1f}_{lon_val:.1f}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r") as f:
            data = json.load(f)
        weather_data = pd.read_json(StringIO(data["weather"]), orient="split")
        # Ensure DateTimeIndex for alignment with hourly inputs
        weather_data.index = pd.to_datetime(weather_data.index)
        return weather_data
    except Exception:
        return None

def _save_cached_weather(lat_val: float, lon_val: float, weather_data: pd.DataFrame):
    if not isinstance(weather_data, pd.DataFrame) or weather_data.empty:
        return

    cache_path = Path(PVGIS_CACHE_DIR) / f"pvgis_{lat_val:.1f}_{lon_val:.1f}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weather": weather_data.to_json(orient="split", date_format="iso")
    }
    with cache_path.open("w") as f:
        json.dump(payload, f)

def _load_or_fetch_weather(data_processor, lat_val: float, lon_val: float):
    weather_data = _load_cached_weather(lat_val, lon_val)
    if weather_data is not None:
        return weather_data

    weather_data, __ = data_processor.get_weather_data()
    if isinstance(weather_data, pd.DataFrame) and not weather_data.empty:
        _save_cached_weather(lat_val, lon_val, weather_data)
        return weather_data
    return None

def _required_case_study_jobs(ds):
    required_locations = list(LOCATIONS)
    required_locations.extend(LOCATIONS_SENS)

    jobs = set()
    for _, _, lat_s, lon_s in required_locations:
        lat_near = float(ds.lat.values[np.argmin(np.abs(ds.lat.values - lat_s))])
        lon_near = float(ds.lon.values[np.argmin(np.abs(ds.lon.values - lon_s))])
        jobs.add((lat_near, lon_near))
    return jobs

def process_cell(lat_val, lon_val):
    """Process one lat/lon cell. (PtX: processes ALL pixels, no nitrogen demand filtering)"""
    # PtX MODIFICATION: Process all pixels globally for general PtX production
    # (Original NH3 version filtered by nitrogen demand)
    data_processor = crp.RenewableEnergyProcessor(lat_val, lon_val)
    weather_data = _load_or_fetch_weather(data_processor, lat_val, lon_val)
    if weather_data is None:
        # Skip only when neither cache nor live PVGIS fetch is available.
        return None

    cf_pv, cf_wind = data_processor.get_renewable_profiles(weather_data)

    # Store weather fields needed for DAC energy scaling (PVGIS RH if available).
    temp_air_c = np.array(weather_data["temp_air"]).astype(float)
    rel_humidity = _extract_relative_humidity(weather_data, temp_air_c)

    # DAC electricity/heat intensity (kWh/kg CO2) via table interpolation
    dac_el = _interp_dac_table(temp_air_c, rel_humidity, _DAC_RH_ELEC, _DAC_T_ELEC, _DAC_TABLE_ELEC)
    dac_heat = _interp_dac_table(temp_air_c, rel_humidity, _DAC_RH_HEAT, _DAC_T_HEAT, _DAC_TABLE_HEAT)

    return lat_val, lon_val, cf_pv, cf_wind, temp_air_c, rel_humidity, dac_el, dac_heat

def process_cell_wrapper(args):
    lat_val, lon_val = args
    try:
        return process_cell(lat_val, lon_val)
    except Exception as e:
        print(f"Skipping cell {lat_val, lon_val} due to error: {e}")
        return None

def main():
    global ds  # Declare global to modify the module-level variable
    
    # Load dataset here to avoid Windows multiprocessing memory issues
    print(f"Loading dataset from {OUTPUT_FILE_XARRAY_INIT}...")
    with open(OUTPUT_FILE_XARRAY_INIT, "rb") as f:
        ds = pickle.load(f)
    print(f"Dataset loaded: {len(ds.lat)} lat × {len(ds.lon)} lon = {len(ds.lat) * len(ds.lon)} total pixels")
    
    start_time = time.time()
    
    # Prepare jobs — thinned pixels + all explicit case-study locations.
    mask_var = "ptx_thinned_ok" if "ptx_thinned_ok" in ds else "ptx_pixel_ok"
    if mask_var in ds:
        mask = ds[mask_var].values
        jobs_set = {
            (float(ds.lat.values[i]), float(ds.lon.values[j]))
            for i in range(len(ds.lat))
            for j in range(len(ds.lon))
            if mask[i, j]
        }
    else:
        jobs_set = {(float(lat), float(lon)) for lat in ds.lat.values for lon in ds.lon.values}
        print("WARNING: no pixel mask found — processing all pixels.")

    jobs_set.update(_required_case_study_jobs(ds))

    jobs = sorted(jobs_set)
    print(f"Processing {len(jobs)} pixels ({mask_var} mask + case-study locations).")
    results = []

    # Parallel processing
    # NOTE: Adjust max_workers based on your system. On Windows with limited RAM, use fewer workers.
    # Recommended: 4-6 workers for 16GB RAM, 8-12 workers for 32GB+ RAM
    max_workers = 6  # Reduced from 9 to avoid memory issues on Windows
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, res in enumerate(executor.map(process_cell_wrapper, jobs), 1):
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            avg_time_per_job = elapsed_time / i
            remaining_minutes = avg_time_per_job * (len(jobs) - i) / 60

            sys.stdout.write(
                f"\rProcessed {i}/{len(jobs)} | Elapsed: {elapsed_minutes:.0f} min | "
                f"Remaining: {remaining_minutes:.0f} min, {remaining_minutes/60:.1f} hours"
            )
            sys.stdout.flush()

            if res is not None:
                results.append(res)
    
    print(f"\nSuccessfully processed {len(results)} pixels (skipped {len(jobs) - len(results)} due to errors)")

    print("\n\nAll cells processed, updating dataset...")
    
    # Initialize data variables for capacity factors if they don't exist
    if "cf_solar" not in ds.data_vars:
        ds["cf_solar"] = xr.DataArray(
            np.full((len(ds.lat), len(ds.lon), 8760), np.nan),
            dims=["lat", "lon", "time"],
            coords={"lat": ds.lat, "lon": ds.lon, "time": range(8760)}
        )
    if "cf_wind" not in ds.data_vars:
        ds["cf_wind"] = xr.DataArray(
            np.full((len(ds.lat), len(ds.lon), 8760), np.nan),
            dims=["lat", "lon", "time"],
            coords={"lat": ds.lat, "lon": ds.lon, "time": range(8760)}
        )

    # Ensure weather variables exist with the same shape as the capacity factor arrays
    if "temp_air_c" not in ds.data_vars:
        ds["temp_air_c"] = xr.full_like(ds.cf_solar, np.nan)
    if "rel_humidity" not in ds.data_vars:
        ds["rel_humidity"] = xr.full_like(ds.cf_solar, np.nan)
    if "dac_el_kWh_per_kgCO2" not in ds.data_vars:
        ds["dac_el_kWh_per_kgCO2"] = xr.full_like(ds.cf_solar, np.nan)
    if "dac_heat_kWh_per_kgCO2" not in ds.data_vars:
        ds["dac_heat_kWh_per_kgCO2"] = xr.full_like(ds.cf_solar, np.nan)

    # Update the dataset once
    for lat_val, lon_val, cf_pv, cf_wind, temp_air_c, rel_humidity, dac_el, dac_heat in results:
        ds.cf_solar.loc[dict(lat=lat_val, lon=lon_val)] = cf_pv
        ds.cf_wind.loc[dict(lat=lat_val, lon=lon_val)] = cf_wind
        ds.temp_air_c.loc[dict(lat=lat_val, lon=lon_val)] = temp_air_c
        ds.rel_humidity.loc[dict(lat=lat_val, lon=lon_val)] = rel_humidity
        ds.dac_el_kWh_per_kgCO2.loc[dict(lat=lat_val, lon=lon_val)] = dac_el
        ds.dac_heat_kWh_per_kgCO2.loc[dict(lat=lat_val, lon=lon_val)] = dac_heat

    # Save (serialize) the xarray dataset
    with open(OUTPUT_FILE_XARRAY, "wb") as f:
        pickle.dump(ds, f)

    print(f"Dataset saved to {OUTPUT_FILE_XARRAY}")

if __name__ == "__main__":
    main()

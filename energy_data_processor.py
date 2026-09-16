import numpy as np
import pandas as pd
import json
import pycountry

from config import (
    MAX_CAP_TECHS, MAX_GRID_CAP, NAME_REF_DB
    )

#import dict with retail prices
FILE_PATH_PP = r"input_data/gpp_2025_country_pages_numeric.json"

# Load JSON into Python dictionary
with open(FILE_PATH_PP, "r", encoding="utf-8") as f:
    RETAIL_PRICES_UPDATE= json.load(f)['numeric_map']

# DISCOUNT RATES from Steffen et al. https://www.nature.com/articles/s41597-025-05912-x#Sec9
DF_DR = pd.read_csv(r"input_data/SteffenEtAl2025_WACC_database.csv")
DF_DR["Value"] = DF_DR["Value"].replace('%', '', regex=True).astype(float)

# Load the dictionary once (e.g., at the start of your script)
with open("input_data/ghg_factors.json", "r") as f:
    GHG_FACTORS_GRID = json.load(f)
    
with open("input_data/ghg_factors_future.json", "r") as f:
    GHG_FACTORS_GRID_FUTURE = json.load(f)
    
def get_activity_env_elect_from_dict(iso2, db=None):
    """
    Fetch GHG impact from precomputed dictionary (ghg_factors.json),
    only if the database matches the requested 'db'. Falls back to global average.
    """
    if db!=NAME_REF_DB and db!= None:
        # take future prospective db
        ghg_factors_grid = GHG_FACTORS_GRID_FUTURE
        data = ghg_factors_grid.get(iso2)
    else:
        ghg_factors_grid = GHG_FACTORS_GRID
        data = ghg_factors_grid.get(iso2)

    if data is None:
        print(f"No entry found for country code '{iso2}' in ghg_factors.json, using Global average")
        data = ghg_factors_grid.get('GLO')
        if data is None:
            raise KeyError("Global average ('GLO') entry not found in ghg_factors.json")

    if db is not None and data.get("db") != db:
        raise KeyError(f"Database mismatch for '{iso2}': requested '{db}', found '{data.get('db')}'")

    ghg_impact = data.get("lca_impact_climate change")
    if ghg_impact is None:
        raise KeyError(f"'lca_impact_climate change' not found for '{iso2}' entry in ghg_factors.json")

    return ghg_impact

def get_max_caps_regions(st_max = MAX_CAP_TECHS,
                         min_cap=0):
    
    """
    Get maximum capacity constraints for a specified region.

    Parameters:
        st_max (int): Maximum capacity for technologies (in MW).
        min_cap (int): Minimum capacity for certain technologies (in MW).

    Returns:
        dict: A dictionary containing the maximum capacity constraints for various technologies.

    Storage bound rationale (from empirical analysis of global + case-study results):
      max_bat:         max observed ~37 MW across all global runs; 60 MW gives 60% headroom.
      max_h2_storage:  H2 storage is never selected by the optimizer (cap_h2_ves = 0 in 100%
                       of results); tight bound reduces LP variable space without affecting
                       the optimal solution.
      max_co2_storage: CO2 storage is the primary seasonal buffer. FT-SAF at 30 t/day
                       requires up to ~10,500 t CO2 (≈3 months of DAC output); old bound
                       of st_max*20=3000 was binding and caused suboptimal solutions for
                       ftsaf. Fixed at 15,000 t to cover all pathways and plant sizes.
    """
    dict_reg_limits = {
        "max_pv": st_max,
        "max_wind_on": st_max,
        "max_grid_cap": MAX_GRID_CAP,
        "st_max": st_max,
        'min_cap': min_cap,
        'max_bat': 60,                 # MW; was st_max*20=3000, empirical max ~37 MW
        'max_h2_storage': 50,          # MWh_H2; was st_max*20=3000, always 0 in practice
        'max_co2_storage': 15000,      # t CO2; was st_max*20=3000, ftsaf needs up to ~10500
        'max_electrolyzer': st_max,
        'min_electrolyzer': min_cap,
        'max_hp': st_max,
    }
    
    return dict_reg_limits

def get_latest_avg_wacc(assess_country, df=DF_DR):
    """
    Retrieve the latest available average WACC (Weighted Average Cost of Capital)
    for a given country. If missing, falls back to a regional or global average.

    Source of regional classification: World Bank standard regions.

    Args:
        assess_country (str): ISO 2- or 3-letter country code, or full name.
        df (pd.DataFrame): DataFrame containing WACC data, with columns:
            - 'Country code'
            - 'Variable'
            - 'Financing year'
            - 'Value'

    Returns:
        "wacc": float (decimal, e.g., 0.067), 
    """
    # --- region lookup table (expanded, simplified World Bank mapping) ---
    region_lookup = {
        # OECD (priority over geography)
        "OECD": [
            "DE","FR","IT","ES","NL","BE","SE","NO","FI","DK","GB","IE","CH","AT","PT","IS","LU","GR",
            "CA","US","JP","KR","NZ","AU","IL","CL"
        ],

        # Europe & Central Asia (non-OECD)
        "Europe & Central Asia": [
            "PL","CZ","SK","HU","RO","BG","TR","KZ","UZ","RU","BY","UA","RS","AL","BA","MK","GE","AM",
            "AZ","KG","TJ","TM","MD","XK"
        ],

        # East Asia & Pacific (non-OECD)
        "East Asia & Pacific": [
            "CN","VN","TH","MY","PH","ID","LA","KH","MM","MN","PG","FJ","SB","VU","TO","WS"
        ],

        # Latin America & Caribbean (non-OECD)
        "Latin America & Caribbean": [
            "BR","AR","CO","PE","MX","VE","EC","PA","DO","HN","BO","PY","NI","BZ",
            "CU","GT","CR","SV","UY","JM","TT","BB","SR"
        ],

        # Middle East & North Africa
        "Middle East & North Africa": [
            "IR","IQ","SY","JO","LB","SA","AE","OM","YE","DZ","MA","TN","EG","LY",
            "QA","KW","BH","PS","SD"
        ],

        # Sub-Saharan Africa
        "Sub-Saharan Africa": [
            "NG","GH","ET","KE","TZ","ZA","MZ","AO","ZM","ZW","BF","ML","NE","TG","SN","CM",
            "CI","UG","RW","BI","CD","CG","MG","BW","NA","SL","GN","TD","SO","SS","GA","GQ"
        ],

        # South Asia
        "South Asia": ["IN","PK","BD","LK","NP","AF","BT","MV"]
    }

    # --- regional fallback defaults (Weighted Average Cost of Capital, WACC) ---
    region_defaults = {
        "OECD": 0.065,
        "Europe & Central Asia": 0.065,
        "East Asia & Pacific": 0.08,
        "Latin America & Caribbean": 0.09,
        "Middle East & North Africa": 0.10,
        "Sub-Saharan Africa": 0.11,
        "South Asia": 0.10,
        "default": 0.07
    }

    # --- Try direct country data ---
    df_country = df[(df["Country code"] == assess_country) &
                    (df["Variable"].str.contains("WACC", case=False))]

    if not df_country.empty:
        latest_year = int(df_country["Financing year"].max())
        df_latest = df_country[df_country["Financing year"] == latest_year]
        avg_wacc = df_latest["Value"].mean() / 100
        return avg_wacc

    # --- Identify region ---
    region_found = None
    for region, countries in region_lookup.items():
        if assess_country in countries:
            region_found = region
            break

    # --- Regional fallback only ---
    if region_found:
        avg_wacc = region_defaults.get(region_found, region_defaults["default"])
    else:
        avg_wacc = region_defaults["default"]

    return avg_wacc

def get_elect_prices(loc, factor_lower_night=0.2, h_night_start=19, h_night_end=7, min_power_price=0.001):
    """
    Generate synthetic hourly electricity prices based on average retail electricity prices.
    Night hours are assumed to be 'factor_lower_night' cheaper than day hours.
    Returns: da_prices (8760,)- in €/kWh.
    """
    try:
        avg_price = RETAIL_PRICES_UPDATE[loc]
    except KeyError:
        print(f"Location '{loc}' not in retail price dictionary, take the average.")
        avg_price = round(sum(RETAIL_PRICES_UPDATE.values()) / len(RETAIL_PRICES_UPDATE),2)
    
    # This avoids unrealistic low or zero power prices
    if avg_price < min_power_price:
        print(f"Location '{loc}' has unrealistic power price: '{round(avg_price,4)}', take the average.")
        avg_price = round(sum(RETAIL_PRICES_UPDATE.values()) / len(RETAIL_PRICES_UPDATE),2)

    # Define day–night pattern (e.g., 20% cheaper at night by default)
    daily_prices = np.array([
        avg_price * (1 - factor_lower_night)
        if (h >= h_night_start or h < h_night_end)
        else avg_price * (1 + factor_lower_night)
        for h in range(24)
    ])

    # Repeat for 365 days → 8760 hours
    da_prices = np.tile(daily_prices, 365)
    #print(f"Generated electricity price profile for {loc} ({year}):")
    #print(f"  Avg. price (hourly mean): {da_prices.mean():.4f} €/kWh")
    #print(f"  Base retail price: {avg_price:.3f} €/kWh")

    return da_prices

def country_to_iso2(country_name: str) -> str:
    """
    Convert a country name to its ISO2 code using pycountry.
    
    Parameters
    ----------
    country_name : str
        Full country name (e.g., "Norway").
        
    Returns
    -------
    iso2 : str
        ISO2 country code (e.g., "NO").
        
    Raises
    ------
    ValueError
        If the country name cannot be found.
    """
    # Handle special cases that pycountry may not recognize
    special_cases = {
        'United Kingdom': 'GB',
        'World': 'WORLD',
        'Europe': 'EU',
        'Norway': 'NO',
        'France': 'FR',
        'Somaliland': 'SO', #assume for simplicity Somalia.
        'Brunei': 'BN',
        'Czech Republic': 'CZ',
        'Russia': 'RU',
        'South Korea': 'KR',
        
    }
    
    if country_name in special_cases:
        return special_cases[country_name]
    
    country = pycountry.countries.get(name=country_name)
    if not country:
        # Try searching by common_name (some countries like "Russia")
        country = pycountry.countries.get(common_name=country_name)
    
    if not country:
        raise ValueError(f"Country '{country_name}' not found in pycountry database.")
    
    return country.alpha_2

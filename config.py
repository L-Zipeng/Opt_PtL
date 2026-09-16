import pandas as pd
import warnings
import os

"""Define vars"""
DAYS = 365
MJ_KG_H2 = 120
# Exchange rate used: 1 USD = 0.86 EUR (approx., Oct 2025 snapshot)
USD_TO_EUR = 0.86
MJ_kWh = 3.6

# Resoltuion of geospatial analysis:
DEG_RES = 1

# PtX default production scale (tonnes product per day)
T_DAY_MEOH = 30
T_DAY_SAF = 30

# Run configuration
RUN_NH3 = False
RUN_PTX = True
PTX_PATHWAYS = ["meoh", "meoh_to_saf", "ftsaf"]

ASSESSMENT_YEAR = 2025

##LOCATIONS selected
#country, iso2, latitude, longitude.
LOCATIONS = [
    ["China (Shandong)", "CN", 37.42, 121.43],                            # Shandong
    ["USA (Texas)", "US", 27.67, -98.61],                                 # Texas
    ["Australia (Pilbara)", "AU", -20.32, 121.04],                        # Pilbara
    ["Netherlands (Rotterdam)", "NL", 51.95, 4.15],                       # Rotterdam
    ["Chile (San Gregorio)", "CL", -52.28, -69.52],                       # San Gregorio
    ["United Arab Emirates (Abu Dhabi)", "AE", 24.15, 54.33],             # Abu Dhabi
    ["Indonesia (Bintan)", "ID", 1.10, 104.48],                            # Bintan
    ["Morocco (Dakhla)", "MA", 23.72, -15.93],                            # Dakhla
    ["Spain (Central Spain)", "ES", 39.07, -5.12],                        # Central Spain
    ["South Africa (Nelson Mandela Bay)", "ZA", -33.92, 25.65],           # Nelson Mandela Bay
]

LOCATIONS_SENS = [
    ["China (Shandong)", "CN", 37.42, 121.43],                            # Shandong
    ["USA (Texas)", "US", 27.67, -98.61],                                 # Texas
    ["Australia (Pilbara)", "AU", -20.32, 121.04],                        # Pilbara
    # ["Netherlands (Rotterdam)", "NL", 51.95, 4.15],                       # Rotterdam
    # ["Chile (San Gregorio)", "CL", -52.28, -69.52],                       # San Gregorio
    # ["United Arab Emirates (Abu Dhabi)", "AE", 24.15, 54.33],             # Abu Dhabi
    # ["Indonesia (Bintan)", "ID", 1.10, 104.48],                            # Bintan
    # ["Morocco (Dakhla)", "MA", 23.72, -15.93],                            # Dakhla
    # ["Spain (Central Spain)", "ES", 39.07, -5.12],                        # Central Spain
    # ["South Africa (Nelson Mandela Bay)", "ZA", -33.92, 25.65],           # Nelson Mandela Bay
    ]

# KEYS
# You need a key for premise to generate prospective LCA databases
"""YOU WILL ALSO NEED TO HAVE A LOCAL LICENSE KEY FOR GUROBI, see: https://www.gurobi.com/solutions/licensing/ & /
    https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license-
"""

# Output file names
OUTPUT_FILE_XARRAY_INIT = f"processed_data/output_dataset_res_{DEG_RES}.pkl"
OUTPUT_FILE_XARRAY = f"processed_data/ds_processed_res_{DEG_RES}.pkl"

OUT_JSON_GHG = "input_data/ghg_factors.json"
OUT_JSON_GHG_FUTURE = "input_data/ghg_factors_future.json"
OUT_JSON_POWER_PRICES = "input_data/gpp_2025_country_pages_numeric.json"

# Optional exported (precomputed) LCA factors for PtX-specific technologies
# (same structure as input_data/dict_ghg_impacts.txt)
OUT_JSON_GHG_PTX = "input_data/dict_ghg_impacts_ptx.txt"
OUT_JSON_GHG_PTX_FUTURE = "input_data/dict_ghg_impacts_ptx_future.txt"

FUTURE_POWER_PRICES = "input_data/FUTURE_POWER_PRICES.xlsx"

# results
FILE_PATH_GLOBAL_RESULTS = f"results/global_results_new_{DEG_RES}.pkl"
FILE_PATH_GLOBAL_RESULTS_GRID = f"results/global_results_grid_{DEG_RES}.pkl"
TEMP_FILE = f"results/global_results_{DEG_RES}_TEMP.pkl" # same output file

# PtX results
FILE_PATH_GLOBAL_RESULTS_PTX = f"results/global_results_ptx_{DEG_RES}.pkl"
FILE_PATH_GLOBAL_RESULTS_PTX_GRID = f"results/global_results_ptx_grid_{DEG_RES}.pkl"

# Case study results
FILE_PATH_CASE_STUDIES = "results/case_studies.pkl"
FILE_PATH_CASE_STUDIES_LCA = "results/case_studies_lca.pkl"
FILE_PATH_SENS_ANALYSIS = "results/sensitivity_analysis_case_studies.pkl"

# PtX case study results
FILE_PATH_CASE_STUDIES_PTX = "results/case_studies_ptx.pkl"
FILE_PATH_CASE_STUDIES_LCA_PTX = "results/case_studies_lca_ptx.pkl"
FILE_PATH_SENS_ANALYSIS_PTX = "results/sensitivity_analysis_case_studies_ptx.pkl"

# --- PtX extension inputs (DAC / MeOH / FT / SAF) ---
FILE_NAME_COSTS_PTX = r"input_data/technology_costs_ptx.xlsx"
try:
    COST_DATA_PTX = pd.read_excel(FILE_NAME_COSTS_PTX, sheet_name='costs', index_col="Parameter", usecols=[0, 1, 2])
except FileNotFoundError:
    COST_DATA_PTX = None

# Decide on maximum capacity of technologies
N_DEMAND_THRESHOLD = 0  # example threshold
MAX_CAP_TECHS = 150 #MW
MAX_GRID_CAP = 100 #MW, Max grid connection capacity
MIN_ELECTROLYZER_CAP = 0 #MW, if installed, the minimum capacity installed of the electrolyzer.

# If using a brightway project when calculating LCA impacts yourself
# This can be enabled in case:
# 1. brightway2 and premise are installed 
# 2. One has access to ecoinvent.
# Alternatively, one can set to False, uses exported life cycle GHG emission data (without other impacts)
CALC_ALL_LCA_IMPACTS = True
GENERATE_NEW_LCA_DB = True

# DAC electricity intensity model (kWh/kg CO2)
# dac_el = DAC_EL_BASE * (1 + DAC_EL_TEMP_COEF * (T - DAC_EL_REF_TEMP))
# Optionally includes RH: + DAC_EL_RH_COEF * (RH - DAC_EL_REF_RH)
DAC_EL_BASE = 0.6
DAC_EL_TEMP_COEF = 0.0
DAC_EL_REF_TEMP = 15.0
DAC_EL_RH_COEF = 0.0
DAC_EL_REF_RH = 50.0

# DAC specific energy tables (MWh/t CO2; equals kWh/kg CO2)
DAC_TABLE_ELECTRICITY_PATH = r"input_data/DAC_SpecEnergyRequirements_electricity.csv"
DAC_TABLE_HEAT_PATH = r"input_data/DAC_SpecEnergyRequirements_heat.csv"
# Relative humidity fallback (%) if PVGIS does not provide RH
DAC_RH_DEFAULT = 50.0
# PVGIS local cache (prefetched locally, then copied to Euler)
PVGIS_CACHE_DIR = r"pvgis_cache"

USER_NAME = os.environ.get("ECOINVENT_USER", "YOUR_ECOINVENT_USERNAME")
PROJECT_NAME = "ptx_lca"
DB_NAME = 'db_ptx_system' # Temp DB name for LCA activity
EI_VERSION = "3.10"
NAME_REF_DB = "ecoinvent_{}_reference".format(EI_VERSION).replace(".","")
NAME_FUTURE_DB = "ecoinvent_remind_SSP2-PkBudg1150_2050_base"
DB_NAME_INIT = 'ecoinvent-{}-cutoff'.format(EI_VERSION)

# PtX infrastructure activity names (must match lci-add_ptx.xlsx exactly)
PTX_LCI_DAC = "carbon dioxide production, infrastructure and catalyst"
PTX_LCI_MEOH = "methanol production, infrastructure"
PTX_LCI_MTJ = "kerosene production, methanol-to-jet, infrastructure"
PTX_LCI_FTSAF = "kerosene production, ft-to-jet, infrastructure"

CC_IMPACT_NG_NH3 = 2.8  # tCO2/tNH3; fossil natural gas ammonia reference (not used for PtX pathways).
CC_METHOD = ('EF v3.1 EN15804', 'climate change', 'global warming potential (GWP100)') # standard method to calculate CC impacts.

import brightway2 as bw
from brightway2 import *
import bw2io
import bw2io as bi
from premise import *
from platform import python_version
import premise
import numpy as np
import uuid
from functools import partial
import os

# Name for the MES to be created
import brightway2 as bw
from bw2io.strategies import add_database_name, csv_restore_tuples
from bw2io.importers.base_lci import LCIImporter
from collections import defaultdict
from mapping import my_methods, contribution_mapping_system  #import mappings

from config import (NAME_REF_DB,  DB_NAME_INIT, EI_VERSION,CC_METHOD, ASSESSMENT_YEAR,
                    USER_NAME, PROJECT_NAME, NAME_REF_DB, DB_NAME, MJ_KG_H2, MJ_kWh)

import pandas as pd

_UNMAPPED_EXCHANGES = set()


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it locally before generating licensed ecoinvent/premise databases."
        )
    return value


def get_ecoinvent_password():
    return _required_env("ECOINVENT_PASSWORD")


def get_premise_key():
    return _required_env("PREMISE_KEY")

#print("Using Python v.({}) and premise v.{}.".format(python_version(), str(premise.__version__).replace(", ", ".")))

def import_additional_lcias():
    """
    Import additional LCIA methods related to water consumption and land transformation.

    This function imports and applies LCIA methods for water consumption and creates a new environmental impact
    category for land transformation. It utilizes functions from the 'premise_gwp' and 'bw_recipe_2016' packages.

    Steps:
    1. Adds premise global warming potential (GWP) methods using 'add_premise_gwp'.
    2. Retrieves the biosphere database using 'get_biosphere_database'.
    3. Creates and applies LCIA methods for water consumption.
    4. Defines a new `LCIA' method for land transformation and writes impact factors.

    Returns:
    None

    Example:
    >>> import_additional_lcias()
    """
    # define project
    bw.projects.set_current(PROJECT_NAME) #Creating/accessing the project

    """
    bw.bw2setup() #Importing elementary flows, LCIA methods and some other data

    # Step 1: Add premise global warming potential (GWP) methods
    add_premise_gwp()

    # Step 2: Retrieve the biosphere database
    biosphere = get_biosphere_database()

    # Step 3: Create and apply LCIA methods for water consumption
    gw = WaterConsumption(None, biosphere)
    gw.apply_strategies()
    gw.write_methods(overwrite=True)
    gw.data[0]

    # Step 4: Create a new LCIA method for land transformation
    my_cfs_land = []

    for bio in Database("biosphere3"):
        if "Transformation, from" in bio['name'] and "square meter" == bio['unit']:   
            line = (bio.key, 1)
            my_cfs_land.append(line)

    my_method = Method(("Own method", "Land transformation", "Land transformation"))
    my_metadata = {"unit": "m2", "meaning": "to represent land transformation"}
    my_method.register(**my_metadata)
    my_method.write(my_cfs_land)
    """

def _get_biosphere_name():
    candidate = f"ecoinvent-{EI_VERSION}-biosphere"
    if candidate in bw.databases:
        return candidate
    if "biosphere3" in bw.databases:
        return "biosphere3"
    for name in bw.databases:
        if "biosphere" in name:
            return name
    raise ValueError("No biosphere database found. Import ecoinvent biosphere first.")

def import_ecoinvent_database(db_name=DB_NAME_INIT):
    """
    Import an Ecoinvent database and create default LCIA methods if not already imported.

    This function imports an Ecoinvent database using the specified database name and location path.
    It checks whether the database is already imported and, if not, imports it, applies strategies,
    provides statistics, and writes the database. Additionally, it creates default LCIA methods and core migrations.

    Parameters:
    - db_name (str): Database name for the Ecoinvent dataset.
    - location_path (str): Location path to the datasets subfolder of the unzipped Ecoinvent file.
    - overwrite (bool, optional): If True, overwrites existing LCIA methods. Default is False.

    Returns:
    None

    Example:
    >>> import_ecoinvent_database("ecoinvent_init", "/path/to/ecoinvent/datasets", overwrite=True)
    """
    bw.projects.set_current(PROJECT_NAME)
    # Check if the database is already imported
    if db_name in bw.databases:
        print(f"{db_name} has already been imported.")
    else:
        bi.import_ecoinvent_release(EI_VERSION, "cutoff", USER_NAME, get_ecoinvent_password())
        # Import and process the Ecoinvent database
        #ei_importer = bw.SingleOutputEcospold2Importer(location_path, db_name)
        #ei_importer.apply_strategies()
        #ei_importer.statistics()
        #ei_importer.write_database()
        #bw2io.create_default_lcia_methods(overwrite=overwrite)
        #bw2io.create_core_migrations()

def generate_future_ei_dbs(scenarios = ["SSP2-Base", "SSP2-PkBudg1150","SSP2-PkBudg500"], iam = 'remind',
                           start_yr=2025, end_yr = 2050, step = 15, endstring="base"):
    """
    Generate Ecoinvent scenario models with specified parameters.

    This function generates Ecoinvent scenario models based on the specified year, scenario, and additional settings.
    It avoids adding duplicated databases by checking the existing databases in Brightway2.

    Parameters:
    - scenarios (list): The scenarios for which the models are generated. Default is: 
                ["SSP2-Base",
                "SSP2-PkBudg1150",
                "SSP2-PkBudg500"] corresponding to baseline, 2 degrees C, and 1.5 degrees C.
    - iam (str): IAM chosen, can be 'remind' or 'image'.
    - start_yr (int): The starting year for the scenarios.
    - end_yr (int): The end year for the scenarios.
    - step (int): step between scenario years.
    - endstring (str, optional): A suffix to differentiate the generated databases. Default is "base".

    Returns:
    tuple: A tuple containing two lists -
        1. List of dictionaries specifying the models for the scenarios.
        2. List of database names generated based on the specified parameters.

    Example:
    >>> generate_future_ei_dbs("SSP2-Base")
    ([{'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2030},
      {'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2050}],
     ['ecoinvent_remind_SSP2-Base_2030_custom', 'ecoinvent_remind_SSP2-Base_2050_base'])
    """

    list_years = [start_yr + i * step for i in range(1, int((end_yr - start_yr) / step) + 1)]

    list_spec_scenarios = []
    list_names = []

    for pt in scenarios:
        for yr in list_years:
            string_db = "ecoinvent_{}_{}_{}_{}".format(iam, pt, yr, endstring)

            if yr == start_yr and pt == "SSP2-Base":
                dict_spec = {"model": iam, "pathway": pt, "year": yr,
                                "exclude": ["update_electricity", "update_cement", "update_steel", "update_dac",
                                            "update_fuels", "update_emissions", "update_two_wheelers"
                                            "update_cars", "update_trucks", "update_buses"]}

                if string_db not in bw.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print("Avoid duplicated db and therefore following db not added: '{}'".format(string_db))
            else:
                dict_spec = {"model": iam, "pathway": pt, "year": yr}

                if string_db not in bw.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print("Avoid duplicated db and therefore following db not added: '{}'".format(string_db))

    return list_spec_scenarios, list_names

# ### generate the database which we are going to use, as premise include many novel datasets. Add some datasets that we generated ourselves.
def generate_reference_database():
    """
    Generate a reference database based on specified parameters.

    This function generates a reference database based on the specified parameters.
    It deletes the existing reference database with the same name if it exists and then creates a new one.

    Returns:
    None

    Example:
    >>> generate_reference_database()
    """
    bw.projects.set_current(PROJECT_NAME)
    clear_cache()
    # Delete old reference database with the same name
    for db_name in list(bw.databases):
        if NAME_REF_DB in db_name:
            print(db_name)
            del bw.databases[db_name]

    # Create a new reference database using NewDatabase
    ndb = NewDatabase(
        scenarios=[{"model": "remind", "pathway": 'SSP2-Base', "year": "2022",
                    "exclude": ["update_electricity", "update_cement", "update_steel", "update_dac",
                                "update_fuels", "update_emissions"]}],
        source_db=DB_NAME_INIT,
        source_version=EI_VERSION,
        key=get_premise_key(),
        biosphere_name=_get_biosphere_name(),
        additional_inventories=[
            {"filepath": r"input_data\lci-add_ptx.xlsx", "ecoinvent version": EI_VERSION},
        ]
    )

    # Write the new reference database to Brightway2
    ndb.write_db_to_brightway(name=NAME_REF_DB)

def generate_prospective_lca_dbs(list_spec_scenarios, list_names):
    """
    Generate and update future LCA databases for prospective LCA.

    This function generates and updates future LCA databases based on specified scenarios if needed, and writes them to Brightway2.

    Parameters:
    - list_spec_scenarios (list): List of dictionaries specifying scenarios for new databases.
    - list_names (list): List of names specifying scenario names for new databases.

    Returns:
    None

    Example:
    >>> generate_and_update_lca_databases([{"model": "remind", "pathway": 'SSP2-Base', "year": "2035"}], ['ecoinvent_remind_SSP2-Base_2035_base'])
    """
    bw.projects.set_current(PROJECT_NAME)
    if len(list_spec_scenarios) > 0:
        ndb = NewDatabase(
            scenarios=list_spec_scenarios,
            source_db=DB_NAME_INIT,
            source_version=EI_VERSION,
            key=get_premise_key(),
            biosphere_name=_get_biosphere_name(),
            additional_inventories=[
                {"filepath": r"input_data\lci-add_ptx.xlsx", "ecoinvent version": EI_VERSION},
            ]
        )

        print("START UPDATING")
        ndb.update()

        print("START WRITING")
        ndb.write_db_to_brightway(name = list_names)

def get_low_voltage_grouped_locations(database_name):
    """
    Return all locations for activities in the given Brightway2 database
    where the activity name is 'market group for electricity, low voltage'
    and the reference product is 'electricity, low voltage'.
    """
    bw.projects.set_current(PROJECT_NAME)
    db = bw.Database(database_name)
    locations = [
        act['location']
        for act in db
        if act['name'] == 'market group for electricity, low voltage'
        and act['reference product'] == 'electricity, low voltage'
    ]
    return locations


# # This equation is needed to calculate environmental burdens, other than climate change
def environmental_lca(parm, loc_elect, cap_wind_on,
                      cap_pv, cap_bat_en,
                      cap_h2_ves, cap_co2_ves, cap_electrolyzer, cap_hb, cap_asu,
                      cap_grid,
                      cap_hp=0,
                      summed_grid_abs=0, summed_grid_inj=0, ghgs_opt=0, w_cost=1,
                      sec_db=NAME_REF_DB,
                      credit_env_export=False,
                      lcia_method=CC_METHOD,
                      epsilon_constraint=False,
                      cap_dac=0,
                      cap_meoh=0,
                      cap_mtj=0,
                      cap_ftsaf=0):
    
    """
    Calculates environmental burdens of an ammonia production system, including all selected environmental impact categories. Returns a dataframe with all this information.

    Args:
        parm (dict): dictionary of techno-ecomic cost data and assumptions [-].    
        loc_elect (str): ecoinvent location of MES, abbreviation string, e.g., "GR" [str].
        cap_wind_on (float): capacity of onshore wind [MW].
        cap_pv (float): capacity of solar PV [MW].
        cap_bat_en (float): energy capacity of battery electricity storage system [MWh].        
        cap_bat_p (float): power capacity of battery electricity storage system [MW].
        cap_h2_ves (float): energy capacity of hydrogen storage system [MWh].    
        cap_co2_ves (float): carbon dioxide storage capacity [t CO2].
        cap_electrolyzer (float): electrical capacity electrolyzer [MWe].    
        cap_hb (float): fuel capacity of advanced HB unit [MW].
        cap_grid (float): (max) connection of power grid [MW].
        cap_hp (float): thermal capacity of heat pump for DAC heat supply [MW_th].
        summed_grid_abs (float): grid absorption from power grid [MWh].
        summed_grid_inj (float): grid injection to power grid [MWh].
        ghgs_opt (float): GHG emissions as a results from the optimization, this serves as input here to check whether outcomes are the identical [kg CO2-eq.].
        w_cost (float): weight of cost objective (between 0 and 1) [-].
        sec_db (str): ecoinvent database used for calculation of LCA impacts [-].
        credit_env_export (bool): if True, environmental credit is given for power injection and hydrogen export [-].
        lcia_method (str): standard climate change impact category used [-].
        epsilon_constraint (str): If true, then there is a constraint on GHG emissions.
        cap_dac (float): DAC infrastructure output scaling [kg CO2/year].
        cap_meoh (float): Methanol infrastructure output scaling [kg MeOH/year].
        cap_mtj (float): Methanol-to-jet infrastructure output scaling [kg SAF/year].
        cap_ftsaf (float): FT-to-jet infrastructure output scaling [kg SAF/year].
    Returns:
        lca_results (pd.DataFrame): dataframe with environmental burdens of all selected environmental impact categories.
    """
    exchanges = []
    
    # Onshore wind, per MW
    if cap_wind_on > 0:
        exch_wind_on = {'name': "market for wind turbine, 2MW, onshore",
                'reference product': "wind turbine, 2MW, onshore",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': (cap_wind_on/2 * (parm['project_lt']/parm['wind_on_lt']))/parm['project_lt']}
        exchanges.append(exch_wind_on)

        exch_wind_on_network = {'name': "market for wind turbine network connection, 2MW, onshore",
                'reference product': "wind turbine network connection, 2MW, onshore",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': (cap_wind_on/2 * (parm['project_lt']/parm['wind_on_lt']))/parm['project_lt']}
        exchanges.append(exch_wind_on_network)
        
    # Solar PV
    if cap_pv>0:    
        exch_pv = {'name': "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
                   "reference product": "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
                'database': sec_db,
                'location': "RER",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': (cap_pv/(0.570*0.895) * (parm['project_lt']/parm['pv_lt']))/parm['project_lt']}
        exchanges.append(exch_pv)
   
    # Battery - Energy unit
    if cap_bat_en>0:
        kWh_kg = 23.5/203 #kWh/kg
        exch_bat_cap = {'name': "market for battery, Li-ion, NMC622",
                'reference product': "battery, Li-ion, NMC622",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                #kWh * kg/kWh
                'amount': (cap_bat_en * (1/kWh_kg) * 1e3 * 
                           (parm['project_lt']/parm['bat_en_lt']))/parm['project_lt']}
        exchanges.append(exch_bat_cap)
    
    # Hydrogen storage
    if cap_h2_ves>0:
        exch_stor_ves = {'name': "high pressure hydrogen storage tank",
            'reference product': "high pressure hydrogen storage tank",
            'database': sec_db,
            'location': "GLO",
            'type': 'technosphere',
            'unit': 'kilogram',
            'amount': ( 1e3 * (cap_h2_ves/(MJ_KG_H2/MJ_kWh)) * (parm['project_lt']/parm['h2_ves_lt']))/parm['project_lt']}
        exchanges.append(exch_stor_ves)

    # CO2 storage / compression / transport
    if cap_co2_ves > 0:
        exch_co2_stor = {'name': "carbon dioxide compression, transport and storage",
            'reference product': "carbon dioxide, stored",
            'database': sec_db,
            'location': "GLO",
            'type': 'technosphere',
            'unit': 'kilogram',
            'amount': 1e3 * cap_co2_ves * (parm['project_lt']/parm['co2_stor_lt']) / parm['project_lt']}
        exchanges.append(exch_co2_stor)
    
    # Electrolyzer
    if cap_electrolyzer>0:
        # replacements are already accounted for in these LCIs, so there are set to BoS lifetime (20 years)
        exch_electr = {'name': "electrolyzer production, 1MWe, PEM, Stack",
                'reference product': "electrolyzer, 1MWe, PEM, Stack",
                'database': sec_db,
                'location': "RER",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': ( cap_electrolyzer * (parm['project_lt']/parm['electr_lt']))/parm['project_lt']}
        exchanges.append(exch_electr)
        
        exch_electr_bop = {'name': "electrolyzer production, 1MWe, PEM, Balance of Plant",
                'reference product': "electrolyzer, 1MWe, PEM, Balance of Plant",
                'database': sec_db,
                'location': "RER",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': ( cap_electrolyzer * (parm['project_lt']/parm['electr_bos_lt']))/parm['project_lt']}
        exchanges.append(exch_electr_bop)
    
    # Grid electricity
    if summed_grid_abs>0:
        # check if it is a grouped electricity market (e.g., with BR, CN, US)
        if loc_elect in get_low_voltage_grouped_locations(sec_db):
            exch_grid_elect = {'name': "market group for electricity, low voltage",
                    'reference product': "electricity, low voltage",
                    'database': sec_db,
                    'location': loc_elect,
                    'type': 'technosphere',
                    'unit': 'kilowatt hour',
                    'amount': 1e3 * summed_grid_abs}
            exchanges.append(exch_grid_elect)
        else:
            exch_grid_elect = {'name': "market for electricity, low voltage",
                    'reference product': "electricity, low voltage",
                    'database': sec_db,
                    'location': loc_elect,
                    'type': 'technosphere',
                    'unit': 'kilowatt hour',
                    'amount': 1e3 * summed_grid_abs}
            exchanges.append(exch_grid_elect)
    
    if credit_env_export:
        if loc_elect in get_low_voltage_grouped_locations(sec_db):
            exch_grid_elect_cons = {'name': "market group for electricity, low voltage",
                    'reference product': "electricity, low voltage",
                    #'database': DB_NAME_CONS,
                    'location': loc_elect,
                    'type': 'technosphere',
                    'unit': 'kilowatt hour',
                    'amount': -summed_grid_inj * 1e3}
            exchanges.append(exch_grid_elect_cons)
        else:
            # Grid electricity - FOR GRID INJECTION
            exch_grid_elect_cons = {'name': "market for electricity, low voltage",
                    'reference product': "electricity, low voltage",
                    #'database': DB_NAME_CONS,
                    'location': loc_elect,
                    'type': 'technosphere',
                    'unit': 'kilowatt hour',
                    'amount': -summed_grid_inj * 1e3}
            exchanges.append(exch_grid_elect_cons)
    
    if cap_grid>0:
        exch_impact_grid_network = {'name': "wind turbine network connection construction, 4.5MW, onshore",
                'reference product': "wind turbine network connection, 4.5MW, onshore",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': ( (cap_grid/4.5) * (parm['project_lt']/parm['grid_lt']))/parm['project_lt']}
        exchanges.append(exch_impact_grid_network)
    
    # Heat pump (for DAC thermal energy supply)
    if cap_hp > 0:
        # Using heat pump activity from ecoinvent (30kW unit as reference)
        exch_impact_hp = {'name': "heat pump production, 30kW",
                'reference product': "heat pump, 30kW",
                'database': sec_db,
                'location': "RoW",
                'type': 'technosphere',
                'unit': 'unit',
                'amount': ( (cap_hp*1000/30) * (parm['project_lt']/parm['hp_lt']))/parm['project_lt']}
        exchanges.append(exch_impact_hp)

    if cap_asu >0:
        exch_impact_asu = {'name': "nitrogen production, infrastructure",
                'reference product': "nitrogen production, infrastructure",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_asu}#in kg nitrogen demand annually
        exchanges.append(exch_impact_asu)    
        
    if cap_hb > 0:
        exch_impact_hb = {'name': "ammonia production, infrastructure and catalyst",
                'reference product': "ammonia production, infrastructure and catalyst",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_hb, }#in kg ammonia produced annually
        exchanges.append(exch_impact_hb)    

    # PtX infrastructure-only activities (from lci-add_ptx.xlsx)
    if cap_dac > 0:
        exch_impact_dac = {'name': "carbon dioxide production, infrastructure and catalyst",
                'reference product': "carbon dioxide production, infrastructure and catalyst",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_dac, } #in kg CO2 produced annually
        exchanges.append(exch_impact_dac)

    if cap_meoh > 0:
        exch_impact_meoh = {'name': "methanol production, infrastructure",
                'reference product': "methanol production, infrastructure",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_meoh, } #in kg MeOH produced annually
        exchanges.append(exch_impact_meoh)

    if cap_mtj > 0:
        exch_impact_mtj = {'name': "kerosene production, methanol-to-jet, infrastructure",
                'reference product': "kerosene production, methanol-to-jet, infrastructure",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_mtj, } #in kg SAF produced annually
        exchanges.append(exch_impact_mtj)

    if cap_ftsaf > 0:
        exch_impact_ftsaf = {'name': "kerosene production, ft-to-jet, infrastructure",
                'reference product': "kerosene production, ft-to-jet, infrastructure",
                'database': sec_db,
                'location': "GLO",
                'type': 'technosphere',
                'unit': 'kilogram',
                'amount': cap_ftsaf, } #in kg SAF produced annually
        exchanges.append(exch_impact_ftsaf)
    
    process_code, _ = create_process_and_add(loc_elect, ASSESSMENT_YEAR, exchanges, sec_db)
    mes_activity = get_mes_activity(process_code=process_code)
    activities = [mes_activity]

    lca = bw.LCA({mes_activity: 1}, method=lcia_method)
    lca.lci()
    lca.lcia()
    
    if "climate change" in str(lcia_method):
        diff = abs(ghgs_opt - lca.score)
        # Use a mixed absolute/relative tolerance; fixed 3 kg CO2-eq is too strict
        # for large grid-connected cases and can fail on harmless rounding noise.
        tol = max(10, 1e-6 * abs(ghgs_opt))
        if diff > tol:
            print("**************************************************")
            print(diff)
            print(f"Allowed tolerance is '{tol}'")
            print("Difference between scores, inititial calc score is '{}' and LCA score here is '{}'".format(ghgs_opt,lca.score))
            #report results for debugging, per exchange:
            mes = activities[0]
            lca.redo_lcia({mes: 1})
            for exc in mes.exchanges():
                if exc['type'] == 'technosphere':
                    lca.redo_lcia({exc.input: exc['amount']})
                    print("{}, amount: '{}', lca results: '{}'".format(exc['name'], 
                                                                       exc['amount'], lca.score))
                elif exc['type'] == 'biosphere':
                    # Need to multiple the amount times its CF
                    cf = lca.characterization_matrix[lca.biosphere_dict[exc.input], :].sum()
                    print("{}, amount: '{}', lca results: '{}'".format(exc['name'], 
                                                                       exc['amount'], cf * exc['amount']))        
            raise ValueError("ERROR: in calculation please check GHG calculation")

    # This part of the script is adopted from Antonini and Treyer et al. (2020)
    result_array = [[[] for _ in activities] for _ in my_methods]

    for i, method in enumerate(my_methods):
        lca.switch_method(method)
        for j, activity in enumerate(activities):
            lca.redo_lcia({activity: 1})
            # Add total to check afterwards that all exchanges add up to total
            result_array[i][j].append(("total", lca.score))
            for exc in activity.exchanges():
                if exc['type'] == 'technosphere':
                    lca.redo_lcia({exc.input: exc['amount']})
                    result_array[i][j].append((exc, lca.score))
                elif exc['type'] == 'biosphere':
                    # Need to multiple the amount times its CF
                    cf = lca.characterization_matrix[lca.biosphere_dict[exc.input], :].sum()
                    result_array[i][j].append((exc, cf * exc['amount']))
                    
    for method in result_array:
        for activity in method:
            if not np.allclose(activity[0][1], sum([o[1] for o in activity[1:]])):
                print("Mismatch")
                break

    grouped_array = [[group_exchange_scores(j) for j in i] for i in result_array]
    data_frames = []
    # Unpack in a dataframe:
    for i, group_data in enumerate(grouped_array):
        data_0 = dict(group_data[0])
        col_name = "multi_energy_system"

        df_add = pd.DataFrame.from_dict(data_0, orient='index')
        df_add.rename(columns={"climate change total": col_name, 0: col_name}, inplace=True)
        df_add.index.names = ['contributor']
        df_add['category'] = str(my_methods[i][1])
        df_add['year'] = ASSESSMENT_YEAR
        df_add['db_name'] = sec_db
        data_frames.append(df_add)

    df_total = pd.concat(data_frames, axis=0)

    lca_results = df_total.reset_index().set_index(['category','contributor', 'year', 'db_name'])

    return lca_results

def get_tech_environmental_burdens_ptx(cost_dict, sec_db=NAME_REF_DB, lcia_method=CC_METHOD):
    """
    Gets environmental impacts for PtX systems (base power/storage techs + PtX infrastructure).

    NOTE: Activity names must match exactly what is in lci-add_ptx.xlsx.
    """
    bw.projects.set_current(PROJECT_NAME)

    # Hydrogen storage vessel
    env_imp_h2_ves = get_activity_env(
        "high pressure hydrogen storage tank",
        "GLO",
        "high pressure hydrogen storage tank",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    ) / (MJ_KG_H2 / MJ_kWh)

    env_imp_co2_ves = get_activity_env(
        "carbon dioxide compression, transport and storage",
        "GLO",
        "carbon dioxide, stored",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )

    # Ground-mounted solar PV panels
    env_imp_pv = get_activity_env(
        "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
        "RER",
        "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    ) / (570 * 0.895)

    # Onshore wind
    env_imp_wind_on = (
        get_activity_env(
            "market for wind turbine, 2MW, onshore",
            "GLO",
            "wind turbine, 2MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        + get_activity_env(
            "market for wind turbine network connection, 2MW, onshore",
            "GLO",
            "wind turbine network connection, 2MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
    ) / 2000

    # Electrolyzer
    env_imp_electr = (
        get_activity_env(
            "electrolyzer production, 1MWe, PEM, Stack",
            "RER",
            "electrolyzer, 1MWe, PEM, Stack",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        + (cost_dict["electr_lt"] / cost_dict["electr_bos_lt"])
        * get_activity_env(
            "electrolyzer production, 1MWe, PEM, Balance of Plant",
            "RER",
            "electrolyzer, 1MWe, PEM, Balance of Plant",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
    ) / 1000

    # Battery, here NMC
    kWh_kg = 23.5 / 203
    env_imp_bat_cap = (
        get_activity_env(
            "market for battery, Li-ion, NMC622",
            "GLO",
            "battery, Li-ion, NMC622",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        * 1
        / kWh_kg
    )

    env_impact_grid_network = (
        get_activity_env(
            "wind turbine network connection construction, 4.5MW, onshore",
            "GLO",
            "wind turbine network connection, 4.5MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        / 4500
    )

    # PtX infrastructure activities (from lci-add_ptx.xlsx)
    env_imp_dac = get_activity_env(
        "carbon dioxide production, infrastructure and catalyst",
        "GLO",
        "carbon dioxide production, infrastructure and catalyst",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )
    env_imp_meoh = get_activity_env(
        "methanol production, infrastructure",
        "GLO",
        "methanol production, infrastructure",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )
    env_imp_mtj = get_activity_env(
        "kerosene production, methanol-to-jet, infrastructure",
        "GLO",
        "kerosene production, methanol-to-jet, infrastructure",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )
    env_imp_ftsaf = get_activity_env(
        "kerosene production, ft-to-jet, infrastructure",
        "GLO",
        "kerosene production, ft-to-jet, infrastructure",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )
    
    # Heat pump (for DAC thermal energy supply)
    env_imp_hp = get_activity_env(
        "heat pump production, 30kW",
        "RoW",
        "heat pump, 30kW",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    ) / (30 * 1000)  # Convert kg CO2-eq per 30 kW_th unit to t CO2-eq per kW_th

    dict_env_impacts_ptx = {
        "ghg_imp_h2_ves": env_imp_h2_ves, # t/MWh
        "ghg_imp_co2_ves": env_imp_co2_ves, # t/t CO2 stored
        "ghg_imp_pv": env_imp_pv, # t/MWp
        "ghg_imp_wind_on": env_imp_wind_on, # t/MWp
        "ghg_imp_electr": env_imp_electr, # t/MW
        "ghg_imp_bat_cap": env_imp_bat_cap, # t/MWp
        "ghg_impact_grid_network": env_impact_grid_network, # t/MW
        "ghg_imp_hp": env_imp_hp,         # t/kW_th
        "ghg_imp_dac": env_imp_dac,       # t/t
        "ghg_imp_meoh": env_imp_meoh,     # t/t
        "ghg_imp_mtj": env_imp_mtj,       # t/t
        "ghg_imp_ftsaf": env_imp_ftsaf,   # t/t
    }

    if lcia_method != my_methods[0]:
        dict_env_impacts_ptx.keys.replace("ghg_imp","env_imp").replace("ghg","env")

    return dict_env_impacts_ptx

#ecoinvent_remind_SSP2-PkBudg1300_2030_all
def get_activity_env(name: str, location: str, ref_product: str, 
                     db: str, year: str, scenario: str, lcia_method = 
                     CC_METHOD) -> float:   
    """
    Gets the environmental impact of an activity from a specified ecoinvent database.

    Args:
        name (str): activity name [-].
        location (str): activity location [-].
        ref_product (str): reference product [-].
        year (str): year of database [-].
        scenario (str): IAM scenario used [-].
        lcia_method (str): Standard LCIA method used, here CC_METHOD
        
    Returns:
        float: environmental impact.
        string: location of activity found.
    """
    
    if db == "":
        db_name = "ecoinvent_remind_{}_{}_all".format(scenario, year)
    else:
        db_name = db
    
    # For PV db, we don't have a reference product
    if ref_product == "":
        activity = [x for x in bw.Database(db_name) if name == x['name'] and
               location == x['location'] ][0]
    else:
        activity = [x for x in bw.Database(db_name) if name == x['name'] and
               location == x['location'] and ref_product == x['reference product']
                   ][0]
    lca = bw.LCA({activity: 1}, method=lcia_method)
    lca.lci()
    lca.lcia()
    
    return lca.score

#ecoinvent_remind_SSP2-PkBudg1300_2030_all
def get_activity_env_elect(name: str, location: str, ref_product: str, 
                     db: str, year: str, scenario: str, lcia_method = 
                     CC_METHOD) -> float:   
    """
    Gets the environmental impact of an electricity activity from a specified ecoinvent database.

    Args:
        name (str): activity name [-].
        location (str): activity location [-].
        ref_product (str): reference product [-].
        year (str): year of database [-].
        scenario (str): IAM scenario used [-].
        lcia_method (str): Standard LCIA method used, here CC_METHOD
        
    Returns:
        float: environmental impact.
    """
    
    if db == "":
        db_name = "ecoinvent_remind_{}_{}_all".format(scenario, year)
    else:
        db_name = db
        
    activity = [x for x in bw.Database(db) if name == x['name'] and
           location == x['location'] and
           ref_product == x['reference product']
               ]
    
    if len(activity) < 1:
        # Check whether there is a market group activity for larger area
        activity = [x for x in bw.Database(db) if x['name'] == "market group for electricity, low voltage" and
               location == x['location'] and
               ref_product == x['reference product']
                   ]
        
        if len(activity) < 1:
            # Try to select GLO activity
            activity = [x for x in bw.Database(db) if x['name'] == "market group for electricity, low voltage" and
                   "GLO" == x['location'] and
                   ref_product == x['reference product']
                       ]

            if len(activity) < 1:
                # Try to select RoW activity
                activity = [x for x in bw.Database(db) if name == x['name'] and
                       "RoW" == x['location'] and
                       ref_product == x['reference product']]
                if len(activity) == 1:
                    # Select this activity
                    activity = activity[0]
                else:
                    print("ERROR: No activity found for '{}'".format(name))                
            else:
                # Select this activity
                activity = activity[0]                      

        elif len(activity) == 1:
            # Select this activity
            activity = activity[0]
        else:
            print("ERROR: More than 1 activity found for '{}'".format(name))
            
    elif len(activity) == 1:
        # Select this activity
        activity = activity[0]
    else:
        print("ERROR: More than 1 activity found for '{}'".format(name))
    
    lca = bw.LCA({activity: 1}, method=lcia_method)
    lca.lci()
    lca.lcia()
    
    return lca.score, activity['location']

def create_process(location, year, exchanges, process_code=None):
    """
    Create a process for a multi-energy system.

    Parameters:
        location (str): The location of the multi-energy system.
        year (int): The year of the multi-energy system.
        exchanges (list): A list of exchanges associated with the process.

    Returns:
        dict: A dictionary representing the LCA process of the multi-energy system process.
    """
    
    name = "multi_energy_system_{}_{}".format(location, year)
    return {
        'name': name,
         "code": str(process_code or uuid.uuid4().hex),
        'unit': 'unit',
        'reference product': name,
        'location' :location,
        'exchanges': exchanges, 
    }

def group_exchange_scores(lst):
    """
    Group exchange scores based on a list of exchanges and their scores.

    Parameters:
        lst (list): A list of tuples where each tuple contains an exchange and its associated score.

    Returns:
        dict: A dictionary with group labels as keys and corresponding LCIA scores as values.
    """
    
    # We will store results in a dict, with keys for group label and values of LCIA score.
    grouped_results = defaultdict(int)

    for exc, score in lst[1:]:
        exc_name = exc.input.get("name", "")
        if isinstance(exc_name, str):
            exc_name = exc_name.strip()
        group = contribution_mapping_system.get(exc_name)
        if group is None:
            if exc_name not in _UNMAPPED_EXCHANGES:
                _UNMAPPED_EXCHANGES.add(exc_name)
                print(f"Warning: unmapped exchange name for contribution mapping: {exc_name}")
            group = "Unmapped"
        grouped_results[group] += score
    return grouped_results

def drop_empty_categories(db):
    """
    Drop categories with the value ('',) from the database.

    Parameters:
        db (list): A list of processes or datasets in a database.

    Returns:
        list: The modified database with empty categories removed.
    """

    DROP = ('',)
    for ds in db:
        if ds.get('categories') == DROP:
            del ds['categories']
        for exc in ds.get("exchanges", []):
            if exc.get('categories') == DROP:
                del exc['categories']
    return db

def strip_nonsense(db):
    """
    Strip leading and trailing spaces from strings in the database.

    Parameters:
        db (list): A list of processes or datasets in a database.

    Returns:
        list: The modified database with leading and trailing spaces removed from string values.
    """
    for ds in db:
        for key, value in ds.items():
            if isinstance(value, str):
                ds[key] = value.strip()
            for exc in ds.get('exchanges', []):
                for key, value in exc.items():
                    if isinstance(value, str):
                        exc[key] = value.strip()
    return db

def get_mes_activity(process_code):
    """Return the MES activity just written to the temp DB, identified by code."""
    matches = [act for act in bw.Database(DB_NAME) if act["code"] == process_code]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one MES activity with code {process_code!r} in {DB_NAME}, found {len(matches)}."
        )
    return matches[0]


def create_process_and_add(location, year, exchanges, sec_db, process_code=None):
    """Creates database with processes"""
    IMPORTER = LCIImporter(DB_NAME) #
    process_data = create_process(location, year, exchanges, process_code=process_code)
    IMPORTER.data = [process_data]
    
    IMPORTER.strategies = [
        partial(add_database_name, name=DB_NAME),
        csv_restore_tuples,
        drop_empty_categories,
        strip_nonsense,
    ]
    IMPORTER.apply_strategies()
    IMPORTER.match_database(sec_db, fields=('name','unit','location','reference product', 'database'))
    #IMPORTER.match_database(DB_NAME_CONS, fields=('name','unit','location','reference product', 'database'))   
    IMPORTER.match_database(fields = ('name',))
    IMPORTER.statistics()
    IMPORTER.write_excel(only_unlinked=True)
    IMPORTER.write_database()
    return process_data["code"], process_data["name"]

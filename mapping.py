from config import CC_METHOD

my_methods = [
    CC_METHOD,
    #("Own method", "Land transformation", "Land transformation"),
    ('EF v3.1', 'acidification', 'accumulated exceedance (AE)'),
    ('EF v3.1', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe)'),
    ('EF v3.1', 'ecotoxicity: freshwater, inorganics', 'comparative toxic unit for ecosystems (CTUe)'),
    ('EF v3.1', 'ecotoxicity: freshwater, organics', 'comparative toxic unit for ecosystems (CTUe)'),
    ('EF v3.1', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels'),
    ('EF v3.1', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)'),
    ('EF v3.1', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)'),
    ('EF v3.1', 'eutrophication: terrestrial', 'accumulated exceedance (AE)'),
    ('EF v3.1', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'human toxicity: carcinogenic, inorganics', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'human toxicity: carcinogenic, organics', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'human toxicity: non-carcinogenic, inorganics', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'human toxicity: non-carcinogenic, organics', 'comparative toxic unit for human (CTUh)'),
    ('EF v3.1', 'ionising radiation: human health', 'human exposure efficiency relative to u235'),
    ('EF v3.1', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)'),
    ('EF v3.1', 'ozone depletion', 'ozone depletion potential (ODP)'),
    ('EF v3.1', 'particulate matter formation', 'impact on human health'),
    ('EF v3.1', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase'),
    ('EF v3.1', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'),
    ]

name_dict_fig = {
    "an_costs_op_grid_abs": "Operation – grid power absorption",
    "an_costs_om": "Operation & Maintenance",
    "an_costs_rep": "Replacement costs",
    "an_costs_op_co2": "CO$_{2}$ tax",
    "an_costs_op_grid_inj": "Operation – grid power injection",
    "an_costs_capex_pv": "Investment – PV system",
    "an_costs_capex_bat_en": "Investment – battery energy system",
    "an_costs_capex_bat_p": "Investment – battery power system",
    "an_costs_capex_grid": "Investment – electricity grid",
    "an_costs_op_export_h2": "Export – hydrogen",
    "an_costs_capex_wind_on": "Investment – onshore wind",
    "an_costs_capex_electrolyzer": "Investment – electrolyzer",
    "an_costs_capex_h2_ves": "Investment – H$_{2}$ storage",
    "an_costs_capex_co2_ves": "Investment - CO2 storage",
    "an_costs_capex_dac": "Investment - DAC",
    "an_costs_capex_hp": "Investment - heat pump",
    "an_costs_capex_meoh": "Investment - methanol synthesis",
    "an_costs_capex_mtj": "Investment - MeOH-to-jet",
    "an_costs_capex_ftsaf": "Investment - FT-SAF",
    'an_ghg_op_grid_abs': "Operation – grid power absorption", 
    'an_ghg_op_grid_inj': "Operation – grid power injection",
    'an_ghg_pv': "Construction – PV system",
    'an_ghg_bat_en': "Construction – battery energy system",
    'an_ghg_bat_p': "Construction – battery power system", 
    'an_ghg_grid_ins': "Construction – electricity grid",
    'an_ghg_wind_on': "Construction – onshore wind",
    'an_ghg_electrolyzer': "Construction – electrolyzer",
    'an_ghg_h2_ves': "Construction – H$_{2}$ storage",

    'an_costs_capex_asu': "Investment – Air separation unit",
    'an_costs_capex_hb': "Investment – Haber-Bosch", 
}

cap_mapping = {
    'cap_wind_on': 'Onshore wind',
    'cap_pv': 'Solar PV',
    'cap_bat_en': 'Battery energy storage',
    'cap_bat_p': 'Battery power',
    'cap_h2_ves': 'Hydrogen storage',
    'cap_electrolyzer': 'Electrolyzer',
    'p_grid_connection': 'Power grid connection',
    'ratio_pv_curtailed': 'Curtailment – solar PV',
    'ratio_wind_curtailed_off': 'Curtailment – offshore wind',
    'ratio_wind_curtailed_on': 'Curtailment – onshore wind',
    'total_costs': 'Annual costs [M€/a]',
}

full_names_mapping_dict = {
    'max_pv': 'max. solar PV',
    'max_wind_on': 'max. wind onshore',
    'max_grid_cap': 'max. power grid capacity',
    'st_max': 'max. solar thermal',
    'share_bev': 'share of battery electric vehicles',
    'max_h2_export': 'max. hydrogen export',
    'min_cap': 'overall min. capacity',
    'min_cap_ind': 'min. industrial technology capacity',
    'min_cap_res': 'min. residential technology capacity',
    'max_bat': 'max. battery capacity',
    'max_electrolyzer': 'max. electrolyzer capacity',
    'min_electrolyzer': 'min. electrolyzer capacity',
}

column_dict_categories = {
    'Land transformation': "LT",
    'acidification': "AC",
    'climate change': "CC",
    'ecotoxicity: freshwater': "ET$_{F}$",
    'ecotoxicity: freshwater, inorganics': "ET$_{FI}$",
    'ecotoxicity: freshwater, organics': "ET$_{FO}$",
    'energy resources: non-renewable': "ER",
    'eutrophication: freshwater': "EF$_{F}$",
    'eutrophication: marine': "EF$_{M}$",
    'eutrophication: terrestrial': "EF$_{T}$",
    'human toxicity: carcinogenic': "HT$_{C}$",
    'human toxicity: carcinogenic, inorganics': "HT$_{CI}$",
    'human toxicity: carcinogenic, organics': "HT$_{CO}$",
    'human toxicity: non-carcinogenic': "HT$_{NC}$",
    'human toxicity: non-carcinogenic, inorganics': "HT$_{NCI}$",
    'human toxicity: non-carcinogenic, organics': "HT$_{NCO}$",
    'ionising radiation: human health': "IR",
    'material resources: metals/minerals': "MM",
    'ozone depletion': "OD",
    'particulate matter formation': "PM",
    'photochemical oxidant formation: human health': "PF",
    'water use': "WU",
}

color_op_grid_abs     = "#364390"  # strong blue
color_op_grid_inj     = "#1A9850"  # dark green → electricity injection (benefit/export)
color_an_rep          = "#FEE08B"  # soft yellow → replacement/maintenance
color_an_om           = "#66C2A5"  # teal → operation & maintenance
color_prod_grid       = "#4D4D4D"  # neutral gray → grid electricity capex
color_prod_on_wind    = "#3288BD"  # sky blue → wind power
color_prod_pv         = "#E58524"  # warm orange → solar PV
color_prod_bat        = "#FFD92F"  # bright yellow → battery (energy storage)
color_exp_h2          = "#1B5583"  # deep blue → hydrogen export
color_prod_h2_stor    = "#D73027"  # lavender → hydrogen storage
color_prod_elect      = "#19C7B8"  # dark cyan → electrolyzer
color_co2             = "#717171"  # neutral dark gray → CO₂ operation
color_asu             = "#762A83"  # purple → air separation unit (chemical process)
color_hb              = "#4DAF4A"  # medium green → Haber–Bosch (ammonia synthesis)

color_dict_cost = {
                 'an_costs_op_grid_abs': color_op_grid_abs,
                 'an_costs_capex_grid_ins':color_prod_grid,
                 'an_costs_rep':color_an_rep,
                 'an_costs_om': color_an_om,
                 'an_costs_op_grid_inj': color_op_grid_inj,
                 'an_costs_op_export_h2': color_exp_h2,
                 'an_costs_capex_pv': color_prod_pv,
                 'an_costs_capex_bat_en': color_prod_bat,
                 'an_costs_capex_bat_p': color_prod_bat,
                 'an_costs_capex_grid':color_prod_grid,
                 'an_costs_capex_wind_on':color_prod_on_wind,
                 'an_costs_capex_electrolyzer': color_prod_elect,
                 'an_costs_capex_h2_ves': color_prod_h2_stor,
                 'an_costs_capex_co2_ves': color_co2,
                 'an_costs_capex_dac': color_co2,
                 'an_costs_capex_hp': color_prod_elect,
                 'an_costs_capex_meoh': color_hb,
                 'an_costs_capex_mtj': color_asu,
                 'an_costs_capex_ftsaf': color_asu,
                 'an_costs_op_co2': color_co2,
                'an_costs_capex_asu': color_asu,
                'an_costs_capex_hb': color_hb,    
                  }

color_dict_env = {
                 'Operation, Electricity': color_op_grid_abs,
                 'Construction, grid electricity network':color_prod_grid,
                 'Construction, Wind onshore':color_prod_on_wind,
                 'Construction, PV': color_prod_pv,
                 'Construction, battery': color_prod_bat,
                 'Construction, hydrogen storage': color_prod_h2_stor,
                 'Construction, electrolyzer': color_prod_elect,
                 'Construction, wind onshore':color_prod_on_wind,
                 'Operation, electricity': color_op_grid_abs,
                 'Credit, hydrogen': color_exp_h2,
                 "Credit, electricity injection": color_op_grid_inj,
                 'Haber-Bosch, infrastructure and catalyst': color_hb,
                 'Nitrogen, ASU': color_asu,
                 'Construction, DAC': color_co2,
                 'Construction, methanol synthesis': color_hb,
                 'Construction, MTJ': color_asu,
                'an_ghg_op_grid_abs': color_op_grid_abs,
                'an_ghg_op_grid_inj': color_op_grid_inj,
                'an_ghg_op_h2': color_exp_h2,
                'an_ghg_pv': color_prod_pv,
                'an_ghg_bat_en': color_prod_bat,
                'an_ghg_bat_p': color_prod_bat,
                'an_ghg_grid_ins':color_prod_grid,
                'an_ghg_wind_on':color_prod_on_wind,
                'an_ghg_electrolyzer': color_prod_elect,
                'an_ghg_h2_ves': color_prod_h2_stor,
    
             }

hex_op_ng = ''
hex_op_grid_abs = ""
hex_op_grid_inj = ""
hex_op_chp = ""
hex_an_rep = "" 
hex_an_om = ""
hex_prod_grid = "/"
hex_prod_on_wind = "/"
hex_prod_pv = "/"
hex_prod_bat = "/"
hex_exp_h2 = ""
hex_prod_h2_stor = "/"
hex_prod_elect = "/"
hex_co2 = ''

hex_prod_asu = '/'
hex_prod_hb = '/'

hex_dict_cost = {
                'an_costs_op_grid_abs': hex_op_grid_abs,
                'an_costs_capex_grid_ins':hex_prod_grid,
                'an_costs_rep':hex_an_rep,
                'an_costs_om': hex_an_om,
                'an_costs_op_grid_inj': hex_op_grid_inj,
                'an_costs_op_export_h2': hex_exp_h2,
                'an_costs_capex_pv': hex_prod_pv,
                'an_costs_capex_bat_en': hex_prod_bat,
                'an_costs_capex_bat_p': hex_prod_bat,
                'an_costs_capex_grid':hex_prod_grid,
                'an_costs_capex_wind_on':hex_prod_on_wind,
                'an_costs_capex_electrolyzer': hex_prod_elect,
                'an_costs_capex_h2_ves': hex_prod_h2_stor,
                'an_costs_capex_co2_ves': hex_co2,
                'an_costs_capex_dac': hex_co2,
                'an_costs_capex_hp': hex_prod_elect,
                'an_costs_capex_meoh': hex_prod_hb,
                'an_costs_capex_mtj': hex_prod_asu,
                'an_costs_capex_ftsaf': hex_prod_asu,
                'an_costs_op_co2': hex_co2,
                'an_costs_capex_asu': hex_prod_asu,
                'an_costs_capex_hb': hex_prod_hb,               
                  }

hex_dict_env = {
                'Operation, Electricity': hex_op_grid_abs,
                'Construction, grid electricity network':hex_prod_grid,
                'Construction, Wind onshore':hex_prod_on_wind,
                'Construction, PV': hex_prod_pv,
                'Construction, battery': hex_prod_bat,
                'Construction, hydrogen storage': hex_prod_h2_stor,
                'Construction, electrolyzer': hex_prod_elect,
                'Construction, wind onshore':hex_prod_on_wind,
                'Operation, electricity': hex_op_grid_abs,
                'Credit, hydrogen': hex_exp_h2,
                "Credit, electricity injection": hex_op_grid_inj,
                'an_ghg_op_grid_abs': hex_op_grid_abs,
                'an_ghg_op_grid_inj': hex_op_grid_inj,
                'an_ghg_op_h2': hex_exp_h2,
                'an_ghg_pv': hex_prod_pv,
                'an_ghg_bat_en': hex_prod_bat,
                'an_ghg_bat_p': hex_prod_bat,
                'an_ghg_grid_ins':hex_prod_grid,
                'an_ghg_wind_on':hex_prod_on_wind,
                'an_ghg_electrolyzer': hex_prod_elect,
                'an_ghg_h2_ves': hex_prod_h2_stor,
             }

contribution_mapping_system = {
    "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground":"Construction, PV",
    "li-ion (NMC)":"Construction, battery",    
    "battery management system, kWh":"Construction, battery",   
    "energy management system, kWh":"Construction, battery",        
    "power conditioning system, container system":"Construction, battery",       
    'market for wind power plant, 2MW, offshore, fixed parts':"Construction, wind offshore",     
    'market for wind power plant, 2MW, offshore, moving parts':"Construction, wind offshore",
    "market for wind turbine, 2MW, onshore":"Construction, wind onshore",
    "market for wind turbine network connection, 2MW, onshore":"Construction, wind onshore",
    "wind turbine network connection construction, 4.5MW, onshore":"Construction, grid electricity network",
    'electrolyzer production, 1MWe, PEM, Stack':"Construction, electrolyzer", 
    'electrolyzer production, 1MWe, PEM, Balance of Plant':"Construction, electrolyzer", 
    'market for electricity, low voltage':"Operation, electricity",    
    'market group for electricity, low voltage':"Operation, electricity",  
    'market for electricity, low voltage, Crete':"Operation, electricity",    
    "market for electricity, low voltage, grid injection":"Credit, electricity injection", 
    "high pressure hydrogen storage tank": "Construction, hydrogen storage",
    "hydrogen production, steam reforming": "Credit, hydrogen",
    "nitrogen production, infrastructure": "Nitrogen, ASU",
    "ammonia production, infrastructure and catalyst": "Haber-Bosch, infrastructure and catalyst",
    "carbon dioxide production, infrastructure and catalyst": "Construction, DAC",
    "methanol production, infrastructure": "Construction, methanol synthesis",
    "methanol-to-jet production, infrastructure": "Construction, MTJ",
    "kerosene production, methanol-to-jet, infrastructure": "Construction, MTJ",  # Actual name in ecoinvent
    "kerosene production, ft-to-jet, infrastructure": "Construction, FT-SAF",
    "heat pump production, 30kW": "Construction, heat pump",
    "market for battery, Li-ion, NMC622":"Construction, battery",
    }

dict_units = {
    CC_METHOD: "kg CO$_{2}$-eq.",  
    ("Own method", "Land transformation", "Land transformation"): "m$^{2}$ land", 
    ('EF v3.1', 'acidification', 'accumulated exceedance (AE)'): "mol H+ -eq.", 
    ('EF v3.1', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe)'): "CTUe",  
    ('EF v3.1', 'ecotoxicity: freshwater, inorganics', 'comparative toxic unit for ecosystems (CTUe)'): "CTUe",  
    ('EF v3.1', 'ecotoxicity: freshwater, organics', 'comparative toxic unit for ecosystems (CTUe)'): "CTUe",  
    ('EF v3.1', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels'): "MJ, net calorific value",  
    ('EF v3.1', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)'): "kg P-eq.",
    ('EF v3.1', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)'): "kg N-eq.",  
    ('EF v3.1', 'eutrophication: terrestrial', 'accumulated exceedance (AE)'): "mol N-eq." ,
    ('EF v3.1', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'human toxicity: carcinogenic, inorganics', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'human toxicity: carcinogenic, organics', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'human toxicity: non-carcinogenic, inorganics', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'human toxicity: non-carcinogenic, organics', 'comparative toxic unit for human (CTUh)'): "CTUh",  
    ('EF v3.1', 'ionising radiation: human health', 'human exposure efficiency relative to u235'): "kBq U$_{235}$-eq.",  
    ('EF v3.1', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)'): "kg Sb-eq.",  
    ('EF v3.1', 'ozone depletion', 'ozone depletion potential (ODP)'): "kg CFC-11-eq.",  
    ('EF v3.1', 'particulate matter formation', 'impact on human health'): "disease incidence",  
    ('EF v3.1', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase'): "kg NMVOC-eq.",  
    ('EF v3.1', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'): "m$^{3}$ world eq. deprived",  
             }

dict_color_n = {'p_grid_abs':'brown',
                 'p_battdis': "crimson",
                  'pv_total': 'yellow',
                 'wind_total_on': 'lightblue',
                 'p_battch': "crimson",
                 'p_grid_inj': '#355E3B',
                 'f_elect': 'darkorange',
                
                
                 'p_elect': 'blue',
                 'p_h2_ves_dis':'red',
                 'p_h2_ves':'darkred',
                 'export_h2': 'darkblue',  
               }

# Define a replacement dictionary
replacement_dict = {
    "p_grid_abs": "Grid - import",
    "p_grid_inj": "Grid - export",
    "p_battdis": "Battery - discharge",
    "p_battch": "Battery - charge",
    "pv_total": "Generation - solar PV",
    "wind_total_on": "Generation - On. wind",
    "f_elect": "Consumption - electrolyzer",
    "p_elect": "Conversion - electrolyzer",
    "export_h2": "H$_2$ export",
    "p_h2_ves_dis": "H$_2$ storage - disch.",
    "p_h2_ves": "H$_2$ storage - ch.",
    }

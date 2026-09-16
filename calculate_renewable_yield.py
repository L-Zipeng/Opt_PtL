import pvlib
from pvlib import pvsystem
import datetime as dt
import windpowerlib as wp
from windpowerlib import ModelChain, WindTurbine
import warnings
import numpy as np
import pandas as pd

def pv_profile_generator_tmy(weather_data, latitude: float, longitude: float,
                             module_spec="Sharp_NDQ235F4__2013_", type_pv = "open_rack"):
    """
    Calculates PV generation per kWp.

    Args:
        weather_data (dataframe): dataframe with weather information for one year [pd.Dataframe].
        latitude (float): latitude, in decimal degrees, between -90 and 90, north is positive (ISO 19115) [degrees].
        longitude (float): longitude, in decimal degrees, between -180 and 180, east is positive (ISO 19115) [degrees].
        module_spec (str): type of solar module to be modelled [str] default is Sharp_NDQ235F4__2013_. The PV module modelled, here standard the Sharp, since parameters are available and multi-Si
        type_pv (str): str, default is roof_mounted. How the module will be installed, here assumed to be roof mounted. However, can be modified in the future
    Returns:
        total_profile_tmy (dataframe): the PV profile per kWp panel modelled, for a TMY.
    """
    
    # Assuming that panels are oriented to the south with angle of 35 deg to surface
    sa = 180 #Surface azimuth angle, 180 (south), 270(west), 90(east)
    st = 35 #angle the roof makes with the surface, assumed to be 35 degrees
    
    # If TMY, delete first substrings from rows to avoid inconsitencies
    # Safety checks
    if not isinstance(weather_data, (pd.DataFrame,)):
        warnings.warn(f"Weather data invalid for ({latitude}, {longitude}) — skipping this cell.")
        return np.nan  # or return None, depending on what your pipeline expects

    if 'temp_air' not in weather_data.columns:
        warnings.warn(f"Missing 'temp_air' column for ({latitude}, {longitude}).")
        return np.nan
    
    temperature = weather_data['temp_air'].astype(float)
    windspeed = weather_data['wind_speed'].astype(float)
    ghi = weather_data['ghi']
    dni = weather_data['dni']
    dhi = weather_data['dhi']
    pressure = weather_data['pressure'].astype(float)
    
    #solar details for location
    sp = pvlib.solarposition.get_solarposition(weather_data.index, float(latitude), float(longitude), pressure=pressure)
    irradiance = pvlib.irradiance.get_total_irradiance(st, sa, sp.apparent_zenith, sp.azimuth, dni, ghi, dhi)
    
    # Get db with modules, choose the one in the function and get specs
    modules = pvsystem.retrieve_sam('SandiaMod').T
    modules_multi_si = modules[modules['Material']=="mc-Si"]
    multi_si = modules_multi_si[modules_multi_si.index.str.contains(module_spec)].T
    multi_si = multi_si[module_spec]
    
    #relative airmass and aoi
    airmass = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith)
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass, pressure=pressure)

    #aoi and celltemp
    aoi = pvlib.irradiance.aoi(st, sa, sp.zenith, sp.azimuth)

    if type_pv == "roof_mounted":
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global, temperature,windspeed, **temperature_model_parameters)
    elif type_pv == "open_rack":
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global, temperature,windspeed, **temperature_model_parameters)
    else:
        raise ValueError("Invalid type of PV panel installation")
        
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(irradiance.poa_direct, irradiance.poa_diffuse, 
                                                                    airmass_absolute, aoi, multi_si)
    
    #Multi-Si
    dc = pvlib.pvsystem.sapm(effective_irradiance, celltemp, multi_si)

    #Multi-Si specifications
    Imp_multiSi = multi_si.Impo #8.1243
    Vmp_multiSi = multi_si.Vmpo #29.1988
    kWp = (Imp_multiSi * Vmp_multiSi) / 1000
    
    #use pv Watts model to convert dc to ac power (inverter model)
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

    #convert to kW and fill NaN values
    ac_mod = ac.fillna(0) / 1000
    ac_mod[ac_mod < 0] = 0

    # Generate profile per 1 kWp
    total_profile_tmy = (1/kWp) * ac_mod
    #print("Calculated PV profile with a capacity factor of '{}'".format(round(total_profile_tmy.mean(),4)))
    
    return total_profile_tmy

def wind_profile_generator_tmy(weather_data, turbine_spec="E-126/4200", hub_height = 135, 
                               offshore=False, assessment_year=2025):
    """
    Calculates wind generation per kWp.

    Args:
        weather_data (dataframe): dataframe with weather information for one year [pd.Dataframe].
        latitude (float): latitude, in decimal degrees, between -90 and 90, north is positive (ISO 19115) [degrees].
        longitude (float): longitude, in decimal degrees, between -180 and 180, east is positive (ISO 19115) [degrees].
        turbine_spec (str): type of wind turbine to be modelled [str] default is E-126/4200.
        hub_height (float): float, default is 135 meter.
        roughness_length (float): standard roughtness length applied.
        offshore (bool): if True, the location is offshore.
    Returns:
        total_profile (dataframe): the wind profile per kWp turbine modelled, for a TMY.
    """
    
    df = pd.DataFrame(data=[], 
                  index = pd.date_range('1/1/{} 00:00'.format(assessment_year), periods=8760, freq='h'), 
                 columns=np.arange(0,5))
    df.columns = [
        ['pressure','temperature','wind_speed','roughness_length','temperature'],
        [0, 2, 10, 0, 10]]
    df.columns.names = ['variable_name', 'height']

    """
        DataFrame with time series for wind speed `wind_speed` in m/s,
        temperature `temperature` in K, roughness length `roughness_length`
        in m, and pressure `pressure` in Pa.
        The columns of the DataFrame are a MultiIndex where the first level
        contains the variable name (e.g. wind_speed) and the second level
        contains the height at which it applies (e.g. 10, if it was
        measured at a height of 10 m).
    """
    # Replace new data in df
    df['temperature',2] = weather_data.temp_air + 273.15
    df['temperature',10] = wp.temperature.linear_gradient(weather_data['temp_air']+ 273.15, 2, 10)
    df['wind_speed',10] = weather_data.wind_speed
    df['roughness_length', 0] = get_roughness_length(overseas=offshore)
    df['pressure', 0] = weather_data.pressure
    
    # Forward fill in case empty or NaN values
    df.ffill(inplace=True)

    # specification of wind turbine where power curve is provided in the
    # oedb turbine library
    enercon_e126 = {
            'turbine_type': turbine_spec,  # turbine type as in oedb turbine library
            'hub_height': hub_height  # in m
        }
    # initialize WindTurbine object
    e126 = WindTurbine(**enercon_e126)
    
    # own specifications for ModelChain setup
    modelchain_data = {
        'wind_speed_model': 'logarithmic',      # 'logarithmic' (default),
                                                # 'hellman' or
                                                # 'interpolation_extrapolation'
        'density_model': 'ideal_gas',           # 'barometric' (default), 'ideal_gas'
                                                #  or 'interpolation_extrapolation'
        'temperature_model': 'linear_gradient', # 'linear_gradient' (def.) or
                                                # 'interpolation_extrapolation'
        'power_output_model':
            'power_curve',                      # 'power_curve' (default) or
                                                # 'power_coefficient_curve'
        'density_correction': True,             # False (default) or True
        'obstacle_height': 0,                   # default: 0
        'hellman_exp': None}                    # None (default) or None

    # initialize ModelChain with own specifications and use run_model method to
    # calculate power output
    mc_e126 = ModelChain(e126, **modelchain_data).run_model(df)
    
    # write power output time series to WindTurbine object
    e126.power_output = mc_e126.power_output
    
    # Calculcate power output per kW, and modify export file
    total_profile = e126.power_output/e126.nominal_power
    
    # Sometimes it can happen that the pwoer output is slightly higher than the nomial power, i.e. make sure that this is not allowed;
    total_profile[total_profile > 1] = 1 

    #print("Calculated wind profile with a capacity factor of '{}'".format(round(total_profile.mean(),3)))
    return total_profile

class RenewableEnergyProcessor:
    """
    Class to handle data processing for energy optimization.

    Attributes:
        lat (float): self.lat of the location.
        lon (float): self.lon of the location.
    """
    
    def __init__(self, lat, lon):
        """
        Initializes the EnergyDataProcessor class.

        Parameters:
            lat (float): self.lat of the location.
            lon (float): self.lon of the location.
        """
        self.lat = lat
        self.lon = lon

    def get_weather_data(self, start_year=2007, end_year=2016):
        """
        Gets location-specific weather data from PVGIS.

        Returns
        -------
        data : pandas.DataFrame or np.nan
            Typical meteorological year data with temperature, radiation, and wind.
        elevation : float or np.nan
            Elevation of the location [m].
        """

        try:
            all_data = pvlib.iotools.get_pvgis_tmy(
                float(self.lat), float(self.lon),
                startyear=start_year, endyear=end_year
            )
        except Exception as e:
            warnings.warn(f"PVGIS data fetch failed for ({self.lat}, {self.lon}): {e}")
            return np.nan, np.nan

        # PVGIS tuple layout varies by pvlib version; the weather dataframe is first,
        # while metadata/elevation can appear in later positions.
        if not isinstance(all_data, (tuple, list)) or len(all_data) < 1:
            warnings.warn(f"Unexpected PVGIS output format for ({self.lat}, {self.lon}). Got type {type(all_data)}")
            return np.nan, np.nan

        try:
            data = all_data[0]
            elevation = np.nan
            for meta in all_data[1:]:
                if isinstance(meta, dict):
                    if "location" in meta and isinstance(meta["location"], dict):
                        elevation = meta["location"].get("elevation", elevation)
                    elif "elevation" in meta:
                        elevation = meta.get("elevation", elevation)
        except Exception as e:
            warnings.warn(f"Could not parse PVGIS data for ({self.lat}, {self.lon}): {e}")
            return np.nan, np.nan

        # Ensure DataFrame
        if not isinstance(data, pd.DataFrame):
            warnings.warn(f"PVGIS returned invalid weather data (not a DataFrame) for ({self.lat}, {self.lon}).")
            return np.nan, np.nan

        # Check required columns
        required_cols = {'temp_air', 'ghi', 'dni', 'dhi', 'wind_speed'}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            warnings.warn(f"Missing columns {missing_cols} in PVGIS data for ({self.lat}, {self.lon}).")
            return np.nan, np.nan

        # ---- Normalization and cleaning ----
        data = data.reset_index()
        data['time(UTC)'] = data['time(UTC)'].astype(str)
        data['time(UTC)'] = data['time(UTC)'].str[:-9]
        data['time(UTC)'] = data['time(UTC)'].apply(lambda x: str(2025) + x[4:])
        data['time(UTC)'] = data['time(UTC)'].apply(
            lambda x: dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M')
        )
        data = data.set_index('time(UTC)')

        for col in ['ghi', 'dni', 'dhi']:
            data[col] = data[col].fillna(0)

        data.loc[data['wind_speed'] < 0, 'wind_speed'] = 0

        # Warn if data contains NaNs or is unexpectedly small
        if data.empty or data.isna().all().all():
            warnings.warn(f"Empty or invalid weather dataset for ({self.lat}, {self.lon}).")
            return np.nan, np.nan

        return data, elevation

    def get_renewable_profiles(self, weather_data):
        """
        Calculates renewable electricity generation profiles for solar PV, onshore wind, and offshore wind for TMY per kWp.

        Args:
            weather_data (dataframe): dataframe with weather information for one year [pd.Dataframe].
            self.lat (float): self.lat, in decimal degrees, between -90 and 90, north is positive (ISO 19115) [degrees].
            self.lon (float): self.lon, in decimal degrees, between -180 and 180, east is positive (ISO 19115) [degrees].
        Returns:
            pv_MW (np.array): the pv profile per kWp module modelled [kWp].
            wind_MW_on (np.array): the wind profile for on-shore per kWp module modelled [kWp].
            wind_MW_off (np.array): the wind profile for off-shore per kWp module modelled [kWp] [kWp].
        """

        #PV
        pv_MW_start = pv_profile_generator_tmy(weather_data, self.lat, self.lon)
        pv_MW = np.array(pv_MW_start)

        # WIND
        wind_MW_start_on = wind_profile_generator_tmy(weather_data, turbine_spec="V90/2000", hub_height = 80)
        wind_MW_on = np.array(wind_MW_start_on)
        return pv_MW, wind_MW_on

    def process_data(self):
        # Get weather data and elevation
        weather_data, __ = self.get_weather_data()

        # Get renewable profiles
        pv_MW_array, wind_MW_array_on = self.get_renewable_profiles(weather_data)
        
        return pv_MW_array, wind_MW_array_on

def get_roughness_length(overseas=False):
    """
    Determines roughness length, now simply assumed to be 0.05 for overseases and 0.15 onshore. A tif file can be used instead...
    Args:
        latitude (float): latitude [degrees].
        longitude (float): longitude [degrees].
        overseas (bool): whether location is offshore.
    Returns:
        float: rouhgness length.
    """
    if overseas:
        #print("WARNING: roughness length set to 0.05")
        rix = 0.05
    else:
        #print("WARNING: roughness length set to 0.15")
        rix = 0.15
    return rix
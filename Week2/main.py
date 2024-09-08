# Part 1 - Data consolidation
#%%
from importlib import reload
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import pandas as pd

#%%
location = Location(latitude=55.901, longitude=-3.083, altitude=66, tz='GMT')
temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
# PVGIS (c) European Union, 2001-2024
PVgis_df = pd.read_csv("D:\\_desktop\\_repos\\IESW 2024\\offline_data\\PVGIS_output.csv")

'''
path to wheater data: "D:\_desktop\_repos\IESW 2024\offline_data\PVGIS_output.csv" PVGIS_2.csv
META DATA 
Latitude (decimal degrees):	55.901
Longitude (decimal degrees):	-3.083
Elevation (m):	66
Radiation database:	PVGIS-SARAH2


Slope: 41 deg. (optimum)
Azimuth: 0 deg. (optimum)
Nominal power of the PV system (c-Si) (kWp):	1.0
System losses (%):	14.0


P: PV system power (W)
Gb(i): Beam (direct) irradiance on the inclined plane (plane of the array) (W/m2)
Gd(i): Diffuse irradiance on the inclined plane (plane of the array) (W/m2)
Gr(i): Reflected irradiance on the inclined plane (plane of the array) (W/m2)
H_sun: Sun height (degree)
T2m: 2-m air temperature (degree Celsius)
WS10m: 10-m total wind speed (m/s)
Int: 1 means solar radiation values are reconstructed
'''


#PVgis_df = pd.read_csv("D:\\_desktop\\_repos\\IESW 2024\\offline_data\\PVGIS_2.csv")
#Convert Time to datetime (original foramt = 20170101:0011)
PVgis_df["time"] = pd.to_datetime(PVgis_df["time"], format="%Y%m%d:%H%M")
PVgis_df.set_index("time", inplace=True)



#%% setting up modelchain
import pv_fnc as pv_fnc 
name = 'Canadian_Solar_CS5P_220M___2009_'
module = pv_fnc.find_modules_by_name(name)
name_inv = 'ABB__PVI_3_0_OUTD_S_US__208V_'  #ABBPVI 3 9 ouro s us 208V
inverter = pv_fnc.find_inverters_by_name(name_inv)

'''
    ======   ============================================================
    Column   Description
    ======   ============================================================
    Paco     AC power rating of the inverter. [W]
    Pdco     DC power input that results in Paco output at reference
            voltage Vdco. [W]
    Vdco     DC voltage at which the AC power rating is achieved
            with Pdco power input. [V]
    Pso      DC power required to start the inversion process, or
            self-consumption by inverter, strongly influences inverter
            efficiency at low power levels. [W]
    C0       Parameter defining the curvature (parabolic) of the
            relationship between AC power and DC power at the reference
            operating condition. [1/W]
    C1       Empirical coefficient allowing ``Pdco`` to vary linearly
            with DC voltage input. [1/V]
    C2       Empirical coefficient allowing ``Pso`` to vary linearly with
            DC voltage input. [1/V]
    C3       Empirical coefficient allowing ``C0`` to vary linearly with
            DC voltage input. [1/V]
    Pnt      AC power consumed by the inverter at night (night tare). [W]
    ======   ============================================================

'''
inverter_criteria = {
    'Paco': [1500, 3000],  # select kWp range
}
inverter = pv_fnc.find_inverters_by_criteria(inverter_criteria)



#%%
modules_per_string = 6
strings_per_inverter = 1

system = PVSystem( surface_azimuth=180,
                   surface_tilt=45,
                   module_parameters=module,
                   inverter_parameters=inverter,
                   temperature_model_parameters=temperature_parameters,
                   modules_per_string=modules_per_string,
                   strings_per_inverter=strings_per_inverter)

modelchain = ModelChain(system, location)

# %%

PVgis_df = PVgis_df.rename(columns={
    "T2m": "temp_air",
    "WS10m": "wind_speed",
    "Gb(i)": "ghi",
    "Gd(i)": "dni",
    "Gr(i)": "dhi"
})

# Select only the required columns
weather = PVgis_df[["temp_air", "wind_speed", "ghi", "dni", "dhi"]]



#%%

import matplotlib.pyplot as plt
modelchain.run_model(weather)

#resampel
weather = weather.resample("h").mean()
start_date = weather.index.min()
end_date = weather.index.max()
modelchain.results.ac.plot( figsize=(15, 5), title="AC Power (W)")

#results df
results_df = modelchain.results.ac     

plt.show()

## add kwp calculation based on clear sky model
# Run a clear sky model to simulate one day
#location = Location(latitude=49.20384548387552, longitude=16.65858167781466, altitude=66, tz='GMT') # 49.20384548387552, 16.65858167781466  BRNO  \ og: location = Location(latitude=55.901, longitude=-3.083, altitude=66, tz='GMT')}
times = pd.date_range(start=start_date, end=end_date, freq='h', tz=location.tz)
weather = location.get_clearsky(times)  # Use the times attribute of ModelChain for the index
modelchain.run_model(weather)

# kWp calculation
kwp = modelchain.results.ac.max() / 1000  # Convert W to kW to get kWp
print("Estimated kW peak (kWp):", kwp)
modelchain.results.ac.plot( figsize=(15, 5), title="AC Power (W)")
plt.show()


#%%

# Usage

home_id = 100

# load metadata:
home_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\home.csv"
data = pd.read_csv(home_path)

# load apliances:
apliances_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\appliance.csv"
apliances = pd.read_csv(apliances_path)

# load other apliances 
other_apliances_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\other_appliance.csv"
other_apliances = pd.read_csv(other_apliances_path)

base_path = r"D:\_desktop\_repos\IESW 2024\offline_data\sensordata"      #<---------- replace with your path

import os
def get_files_with_home_and_electric(base_path):
    files_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if "electric" in file and f"home{home_id}" in file:
                files_list.append(os.path.join(root, file))
    return files_list
# Get the list of files
files = get_files_with_home_and_electric(base_path)

# Print the list of files
for file in files:
    print(file)

# from tyhe files get file with string: combined.csv
def get_combined_file(files):
    for file in files:
        if "combined" in file:
            return file
    return None

headers = ["datetime", f"mains_combined_{home_id}"]

# Read the data without a header and assign custom headers
df_main = pd.read_csv(get_combined_file(files), header=None, names=headers)  # original data in W

# Display the DataFrame
print(df_main.head())
df_main["datetime"] = pd.to_datetime(df_main["datetime"])
df_main_resampled =  df_main.set_index("datetime").resample("h").sum() /3600 /1000   # convert to kWh


#%%
# Assuming df_main_resampled and results_df are your DataFrames

# Step 1: Ensure both DataFrames are sorted by datetime
df_main_resampled = df_main_resampled.sort_index()

results_df = results_df.rename("results")
results_df_kwh = results_df.copy() /1000
results_df_kwh = results_df_kwh.sort_index()
# Step 2: Use merge_asof to merge them based on the nearest time, within a given tolerance (e.g., 1 hour)
merged_df = pd.merge_asof(df_main_resampled, 
                          results_df_kwh, 
                          left_index=True, 
                          right_index=True, 
                          tolerance=pd.Timedelta('1H'), # Adjust tolerance as needed
                          direction='nearest') # Use 'nearest', 'backward', or 'forward' for your needs
#merged_df.head()

merged_df.plot()
# merged_df will contain both datasets aligned by nearest timestamps

def pv_cogs_per_kwpeak(capacity_kw, pricing='MSP'):
    """
    Calculate the COGS for a PV system based on capacity in kWpeak.
    
    :param capacity_kw: System capacity in kilowatts peak (kWp).
    :param pricing: 'MSP' for Minimum Sustainable Price, 'MMP' for Modeled Market Price.
    :return: Total cost in USD.
    """
    if pricing == 'MSP':
        cost_per_kw = 2340  # USD per kWp from the MSP benchmark for residential systems without ESS.
    else:
        cost_per_kw = 2680  # USD per kWp from the MMP benchmark for residential systems without ESS.
        
    return capacity_kw * cost_per_kw

def battery_cogs_per_kwh(capacity_kwh, pricing='MSP'):
    """
    Calculate the COGS for a battery storage system based on capacity in kWh.
    
    :param capacity_kwh: Battery capacity in kilowatt-hours (kWh).
    :param pricing: 'MSP' for Minimum Sustainable Price, 'MMP' for Modeled Market Price.
    :return: Total cost in USD.
    """
    if pricing == 'MSP':
        cost_per_kwh = 3880  # USD per kWh from the MSP benchmark for residential systems with ESS.
    else:
        cost_per_kwh = 4700  # USD per kWh from the MMP benchmark for residential systems with ESS.
        
    return capacity_kwh * cost_per_kwh



#%%
import plotly.express as px
import plotly.graph_objects as go

# Define battery parameters
battery_capacity_kwh = 10  # Battery capacity in kWh
battery_efficiency = 0.9   # 90% efficient for charging/discharging
soc = 0                    # Initial state of charge (kWh)

# Create new columns for battery state, grid usage, and energy sources
merged_df['battery_soc'] = 0  # State of charge (kWh)
merged_df['grid_usage'] = 0   # Power drawn from the grid (kWh)
merged_df['solar_consumed'] = 0  # Power directly consumed from solar (kWh)
merged_df['battery_consumed'] = 0  # Power consumed from battery (kWh)
merged_df['grid_consumed'] = 0  # Power consumed from grid (kWh)

# Iterate through each hour to simulate battery charging/discharging
for i in range(1, len(merged_df)):
    generation = merged_df['results'].iloc[i]     # Solar generation (kWh)
    consumption = merged_df[f'mains_combined_{home_id}'].iloc[i]  # Household consumption (kWh)
    
    # Net power surplus or deficit
    net_power = generation - consumption
    
    # If surplus (generation > consumption), charge the battery
    if net_power > 0:
        # Directly consume solar power, any excess goes to charging the battery
        merged_df['solar_consumed'].iloc[i] = consumption  # All consumption met by solar
        charge_amount = min(net_power * battery_efficiency, battery_capacity_kwh - soc)
        soc += charge_amount
        merged_df['battery_soc'].iloc[i] = soc
        merged_df['grid_consumed'].iloc[i] = 0  # No grid usage
        merged_df['battery_consumed'].iloc[i] = 0  # No battery consumption

    # If deficit (generation < consumption), discharge the battery or use grid
    elif net_power < 0:
        solar_contribution = generation
        remaining_demand = abs(net_power)
        
        # Try to cover the deficit with the battery
        discharge_amount = min(remaining_demand / battery_efficiency, soc)
        soc -= discharge_amount
        merged_df['battery_soc'].iloc[i] = soc
        merged_df['battery_consumed'].iloc[i] = discharge_amount
        
        # If battery can't cover the deficit, the grid must supply the rest
        grid_demand = remaining_demand - discharge_amount
        merged_df['grid_consumed'].iloc[i] = grid_demand
        
        # Assign consumed solar power
        merged_df['solar_consumed'].iloc[i] = solar_contribution

    # Ensure SOC remains within battery limits
    soc = min(max(soc, 0), battery_capacity_kwh)

# Plot the results using Plotly
fig = go.Figure()

# Add traces for each series
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['results'], mode='lines', name='Solar Generation'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df[f'mains_combined_{home_id}'], mode='lines', name=f'Household Consumption'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['battery_soc'], mode='lines', name='Battery SOC'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['grid_usage'], mode='lines', name='Grid Usage'))

# Add traces for consumed sources
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['solar_consumed'], mode='lines', name='Solar Consumed'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['battery_consumed'], mode='lines', name='Battery Consumed'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['grid_consumed'], mode='lines', name='Grid Consumed'))

# Update layout
fig.update_layout(
    title='Solar Generation, Consumption, Battery SOC, and Grid Usage',
    xaxis_title='Time',
    yaxis_title='kWh',
    legend_title='Legend',
    template='plotly_white',
    width=1000,
    height=600
)

# Show the plot
fig.show()

#%%
# Example usage:
grid_OPEX = 40 # cent per kwg ct/kWh 
pv_CAPEX = pv_cogs_per_kwpeak(kwp)
battery_CAPEX = battery_cogs_per_kwh(battery_capacity_kwh)
# Find the peak kilowatt output (kWp)
# kWp_real = merged_df["results"].max() 


#calculationg the costs = total grid + negative solar prices (aka inverots idle)
total_grid_usage = merged_df['grid_consumed'].sum() 

total_OPEX = total_grid_usage * grid_OPEX
total_CAPEX = pv_CAPEX + battery_CAPEX
total_raw_usage = merged_df[f'mains_combined_{home_id}'].sum()

saving_OPEX = total_raw_usage * grid_OPEX - total_OPEX
renuewable_coverage = 1 -   total_grid_usage / total_raw_usage 

roi = total_CAPEX / saving_OPEX
















# %%

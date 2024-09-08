#%%

import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pv_fnc as pv_fnc  # Assuming pv_fnc is defined somewhere that handles PV specific functions

#%% PATHS
base_path = r"D:\_desktop\_repos\IESW 2024\offline_data\sensordata"      #<---------- replace with your path
PVgis_df = pd.read_csv("D:\\_desktop\\_repos\\IESW 2024\\offline_data\\PVGIS_output.csv")
home_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\home.csv"
apliances_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\appliance.csv"
other_apliances_path = r"D:\_desktop\_repos\IESW 2024\offline_data\metadata\other_appliance.csv"
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


#%% Loading the wheat data

location = Location(latitude=55.901, longitude=-3.083, altitude=66, tz='GMT')
temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
PVgis_df["time"] = pd.to_datetime(PVgis_df["time"], format="%Y%m%d:%H%M")
PVgis_df.set_index("time", inplace=True)

PVgis_df = PVgis_df.rename(columns={
    "T2m": "temp_air",
    "WS10m": "wind_speed",
    "Gb(i)": "ghi",
    "Gd(i)": "dni",
    "Gr(i)": "dhi"
})

# Select only the required columns
weather = PVgis_df[["temp_air", "wind_speed", "ghi", "dni", "dhi"]]

#%% PV SIMULATION 
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

modules_per_string = 12
strings_per_inverter = 2

system = PVSystem( surface_azimuth=180,
                   surface_tilt=45,
                   module_parameters=module,
                   inverter_parameters=inverter,
                   temperature_model_parameters=temperature_parameters,
                   modules_per_string=modules_per_string,
                   strings_per_inverter=strings_per_inverter)

modelchain = ModelChain(system, location)

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

#%% Consumptuion data 
home_id = 100

data = pd.read_csv(home_path)
apliances = pd.read_csv(apliances_path)
other_apliances = pd.read_csv(other_apliances_path)

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
# %% Merging the results
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




#%% Financial analysis

def pv_cogs_per_kwpeak(capacity_kw):
    # Adjust the cost per kWp as needed, choose between 1000 and 1500
    cost_per_kw = 1250  # Example: midpoint cost
    return capacity_kw * cost_per_kw

def battery_cogs_per_kwh(capacity_kwh):
    # Adjust the cost per kWh as needed, choose between 800 and 1000
    cost_per_kwh = 900  # Example: midpoint cost
    return capacity_kwh * cost_per_kwh


import plotly.express as px
import plotly.graph_objects as go

# Define battery parameters
battery_capacity_kwh = 12  # Battery capacity in kWh
battery_efficiency = 0.9   # 90% efficient for charging/discharging
soc = 0                    # Initial state of charge (kWh)

# Create new columns for battery state, grid usage, and energy sources
merged_df['battery_soc'] = 0  # State of charge (kWh)
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
        charge_amount = min(net_power * battery_efficiency, battery_capacity_kwh - soc) # ensure SOC doesn't exceed capacity
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

# Create a Plotly figure
fig = go.Figure()

# Add traces for each series
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['results'], mode='lines', name='Solar Generation'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df[f'mains_combined_{home_id}'], mode='lines', name=f'Household Consumption'))
fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['battery_soc'], mode='lines', name='Battery SOC'))

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
)

# Render the plot in the default web browser
pyo.plot(fig, filename='plot.html')

# %% Final ROI calculation
# Logic Falw ..... We dont have 1 year worth of data.. sometimes more somethimes less... 
# soooo we gotta have to add soem workaround for that...   
# ngl its a bit too late and I am tired... so I will just leave it here...

# Example usage:
grid_OPEX = 0.40 # eur... per kwg eur/kWh 
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
renewable_coverage = 1 -   total_grid_usage / total_raw_usage 

roi = total_CAPEX / saving_OPEX

'''
The average electricity consumption for a single person living in a house varies depending on factors like household size, energy efficiency, and lifestyle. However, in Germany and many European countries, the typical annual electricity consumption for a single person living in a house can be estimated as follows:

Average Electricity Consumption:
Single person in a house: Around 1,500 to 2,500 kWh per year.

Breakdown:
Lighting & Small Appliances: 500-800 kWh
Refrigeration: 300-400 kWh
Washing & Cleaning: 300-500 kWh
Electronics (TV, computer, etc.): 200-400 kWh

chat gpt btw... pretty cool we got
'''

# Function to format numbers with space as thousands separator
def format_currency(value):
    return f"${value:,.0f}".replace(",", " ")

# Print the formatted values
print(f'''
total_CAPEX: {format_currency(round(total_CAPEX))}
total_OPEX: {format_currency(round(total_OPEX))}
total_raw_usage: {round(total_raw_usage)} kWh
saving_OPEX: {round(saving_OPEX)} kWh
renewable_coverage: {round(renewable_coverage, 3)} %
roi: {round(roi, 3)} years
''')
# %%
# Case A: No Subsidies (Base case)
def calc_case_A(total_CAPEX, saving_OPEX):
    roi_A = total_CAPEX / saving_OPEX
    return roi_A

# Case B: 50% CAPEX Subsidies
def calc_case_B(total_CAPEX, saving_OPEX):
    capex_subsidized = total_CAPEX * 0.5  # 50% of CAPEX subsidized
    roi_B = capex_subsidized / saving_OPEX
    return roi_B

# Case C: Borrowing with Interest
def calc_case_C(total_CAPEX, saving_OPEX, loan_term_years, interest_rate):
    # Loan parameters
    loan_amount = total_CAPEX  # Borrowing full CAPEX
    monthly_rate = interest_rate / 12 / 100  # Convert annual rate to monthly decimal
    num_payments = loan_term_years * 12  # Total loan payments in months
    
    # Calculate monthly loan payment using annuity formula
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    # Total loan cost (principal + interest)
    total_loan_cost = monthly_payment * num_payments
    roi_C = total_loan_cost / saving_OPEX
    
    return total_loan_cost, roi_C

# Ecological Analysis: CO2 savings
def ecological_analysis(total_raw_usage, total_grid_usage, co2_factor, co2_certificate_price):
    energy_saved_kwh = total_raw_usage - total_grid_usage
    co2_saved_kg = energy_saved_kwh * co2_factor  # CO2 saved in kg
    co2_saved_tons = co2_saved_kg / 1000  # Convert to tons
    co2_value = co2_saved_tons * co2_certificate_price  # Monetary value of CO2 saved
    
    return co2_saved_tons, co2_value

# Parameters for case C (borrowing)
loan_term_years = 10  # Example loan term of 10 years
interest_rate = 4.0   # Example interest rate of 4%

# Calculate the ROI for each case
roi_A = calc_case_A(total_CAPEX, saving_OPEX)
roi_B = calc_case_B(total_CAPEX, saving_OPEX)
total_loan_cost_C, roi_C = calc_case_C(total_CAPEX, saving_OPEX, loan_term_years, interest_rate)

# Ecological Analysis: CO2 savings
co2_factor = 0.4  # Average CO2 emissions factor in Germany (kg CO2 per kWh)
co2_certificate_price = 90  # Cost of CO2 certificates in € per ton


co2_saved_tons, co2_value = ecological_analysis(total_raw_usage, total_grid_usage, co2_factor, co2_certificate_price)

# Print the formatted values
print(f'''
Case A: No Subsidies
total_CAPEX: {format_currency(round(total_CAPEX))}
ROI: {round(roi_A, 3)} years

Case B: 50% CAPEX Subsidies
Subsidized CAPEX: {format_currency(round(total_CAPEX * 0.5))}
ROI: {round(roi_B, 3)} years

Case C: Borrowing with {interest_rate}% Interest over {loan_term_years} years
Total Loan Cost: {format_currency(round(total_loan_cost_C))}
ROI: {round(roi_C, 3)} years

Ecological Impact:
CO2 Saved: {round(co2_saved_tons, 3)} tons
Monetary Value of CO2 Savings: {format_currency(round(co2_value))}

total_OPEX: {format_currency(round(total_OPEX))}
total_raw_usage: {round(total_raw_usage)} kWh
saving_OPEX: {round(saving_OPEX)} kWh
renewable_coverage: {round(renewable_coverage, 3)} %
''')

# %%
'''

total_CAPEX: $14 550
total_OPEX: $136
total_raw_usage: 2498 kWh
saving_OPEX: 863 kWh
renewable_coverage: 0.864 %
roi: 16.852 years

'''
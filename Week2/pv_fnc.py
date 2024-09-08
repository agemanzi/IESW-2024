

from pvlib.pvsystem import retrieve_sam
import pvlib


def find_modules_by_criteria(criteria):
    '''
    ==========================================================================
    Parameter             Description                                      Unit
    ==========================================================================
    Vintage               Year the module specifications were recorded     None
    Area                  Surface area of the module                       m²
    Material              Type of semiconductor material used              None
    Cells_in_Series       Number of cells connected in series              None
    Parallel_Strings      Number of strings of cells connected in parallel None
    Isco                  Short Circuit Current                            A
    Voco                  Open Circuit Voltage                             V
    Impo                  Current at maximum power point                   A
    Vmpo                  Voltage at maximum power point                   V
    Aisc                  Short-circuit current temperature coefficient    1/°C
    Aimp                  Maximum power current temperature coefficient    1/°C
    C0                    Coefficient relating Imp to irradiance           None
    C1                    Coefficient relating Imp to irradiance           None
    Bvoco                 Open circuit voltage temperature coefficient     V/°C
    Mbvoc                 Modifier for Bvoco to irradiance                 V/°C
    Bvmpo                 Maximum power voltage temperature coefficient   V/°C
    Mbvmp                 Modifier for Bvmpo to irradiance                 V/°C
    N                     Diode ideality factor                           None
    C2                    Coefficient relating Vmp to irradiance           None
    C3                    Coefficient relating Vmp to irradiance           1/V
    A0                    Air mass coefficient                             None
    A1                    Air mass coefficient                             None
    A2                    Air mass coefficient                             None
    A3                    Air mass coefficient                             None
    A4                    Air mass coefficient                             None
    B0                    Incidence angle modifier coefficient             None
    B1                    Incidence angle modifier coefficient             None
    B2                    Incidence angle modifier coefficient             None
    B3                    Incidence angle modifier coefficient             None
    B4                    Incidence angle modifier coefficient             None
    B5                    Incidence angle modifier coefficient             None
    DTC                   Temperature coefficient for power output         °C
    FD                    Fraction of diffuse irradiance utilized          None
    A                     Modifier for temperature dependence              None
    B                     Modifier for temperature dependence              None
    C4                    Coefficient relating Ix to G                     None
    C5                    Coefficient relating Ix to G                     None
    IXO                   Parameter for performance under different irr.   A
    IXXO                  Parameter for performance under different irr.   A
    C6                    Coefficient relating Ixx to G                    None
    C7                    Coefficient relating Ixx to G                    None
    Notes                 Additional notes                                 None
    ==========================================================================
    '''

    # Retrieve module data from the Sandia database
    modules = retrieve_sam('SandiaMod')
    
    # Filter modules based on criteria
    for key, value_range in criteria.items():
        if key in modules.columns:
            modules = modules[modules[key].between(*value_range)]
        else:
            print(f"Warning: {key} is not a valid parameter for modules. It will be ignored.")
    
    return modules


def find_inverters_by_criteria(criteria):       
    '''
    ======   ============================================================
    Column   Description
    ======   ============================================================
    Paco     AC power rating of the inverter. [W] <--------- main kWp parameter
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
    # Filter inverters based on criteria
    
    inverters = retrieve_sam('cecinverter').copy().transpose()
    filtered_inverters = inverters.copy()  # Start with a copy of all inverters
    #convert the columns to floats to be able to filter them
    for col in filtered_inverters.columns:
        try:
            filtered_inverters[col] = filtered_inverters[col].astype(float)
        except:
            print(f"Warning: {col} is not a valid parameter for inverters. It will be ignored.")
            
    for key, value_range in criteria.items():
        if key in filtered_inverters.columns:
            # Filter each column based on the parameter range, keeping columns that meet criteria
            map_inverters = filtered_inverters[key].between(value_range[0], value_range[1], inclusive="both")
            inverters_results = inverters[map_inverters].copy().transpose().iloc[:,0]
        else:
            print(f"Warning: {key} is not a valid parameter for inverters. It will be ignored.")
    return inverters_results


def find_modules_by_name(name):
    # Retrieve module data from the Sandia database
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    
    # Filter modules based on criteria
    modules = sandia_modules[name]
    
    return modules

def find_inverters_by_name(name):
    # Retrieve inverter data from the CEC database
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    
    # Filter inverters based on criteria
    inverters = cec_inverters[name]
    
    return inverters    
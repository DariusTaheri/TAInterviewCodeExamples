#=== Libraries ===

import pandas as pd


# === Functions ===
def GetDispatchandLMP(GasNeeded,GasPrice):
    DispatchList = []
    CumulativeGenSum=0
    for y,row in EIA860df.iterrows():
        CumulativeGenSum += row['Nameplate Capacity (MW)']
        DispatchList.append(row['Plant Name'])
        
        if CumulativeGenSum >= GasNeeded:
            MarginalCost = row['Heat Rate (MMBtu/Mwh)'] * GasPrice
            break

    DispatchList = DispatchList[::-1]
    return MarginalCost, DispatchList 




# === Running Script ===
if __name__ == '__main__':
    '''
    === ASSUMPTIONS ===
    - Removed plants in WY since no WY load and generation data provided
    - Removed plants in MISO and SWPP because despite being located in MT and ID their dispatch would be based on the MISO and SWPP system (import column)
    - Kept BYUI Central Energy Facility because despite being a non-PacNW BA (PACE) the plant is located in ID which is included in load data.
    '''
    
    '''=== Global settings/variables ==='''
    #Plant and Location Identifiers
    CoalList  =['BIT','LIG','SUB','WC']
    NatGasList =['NG']
    PaCNWStateList =['ID','MT','OR','WA']
    PacNWBAList = ['AVA','AVRN','BPAT','GRID','IPCO','NWMT','PACE','PACW','PGE','PSEI','SCL']

    OperatingList = ['(OP) Operating','(SB) Standby/Backup: available for service but not normally used']


    '''=== Main Code ==='''
    #Creating DataFrames
    HourlyDatadf = pd.read_excel('midc stack test data.xlsx',sheet_name='Hourly Data')
    EIA860df = pd.read_excel('midc stack test data.xlsx',sheet_name='EIA 860')
    EIA923df = pd.read_excel('midc stack test data.xlsx',sheet_name='EIA923')
   
    #Changing EIA923 Column Names to match EIA860 column names 
    EIA923df = EIA923df.rename(columns={"Operator Name": "Entity Name",
                                        "Plant Id": "Plant ID",
                                        "Reported\nPrime Mover": "Prime Mover Code",
                                        "Reported\nFuel Type Code": "Energy Source Code",
                                        "Elec Fuel Consumption\nMMBtu": "Elec Fuel Consumption (MMBtu)",
                                        "Net Generation\n(Megawatthours)": "Net Generation (MWh)"})

    
    #Removing plants not considered NG
    EIA860df = EIA860df[EIA860df['Energy Source Code'].isin(NatGasList)]

    #Keeping plants that are still operating
    EIA860df = EIA860df[EIA860df['Status'].isin(OperatingList)]
    
    #Keeping plants in PacNW region (excluding plants in WY region)
    EIA860df = EIA860df[EIA860df['Plant State'].isin(PaCNWStateList)]

    #Keeping plants in PacNW Balancing Authorities (excluding plants in MISO and SWPP region)
    EIA860df = EIA860df[EIA860df['Balancing Authority Code'].isin(PacNWBAList)]

    #Resetting index
    EIA860df= EIA860df.reset_index(drop=True, inplace=False)

    #Merging EIA923 Generation Data with EIA 860 sheet based on entity name, plant id, prime mover code, and energy source code
    EIA860df = EIA860df.merge(EIA923df, on=['Plant ID','Plant Name','Energy Source Code','Prime Mover Code'], how='left')
            
    #Grouping plants to adjust total heat rates from combining combustion turbine (CT) and steam part (CA) of the combined cycle. 
    EIA860df = EIA860df.groupby(['Plant ID','Plant Name','Unit Code'], as_index=False,dropna=False).agg({
        'Technology':'first',
        'Status':'first',
        'Nameplate Capacity (MW)':'sum',
        'Net Summer Capacity (MW)':'sum',
        'Net Winter Capacity (MW)':'sum',
        'Elec Fuel Consumption (MMBtu)':'sum',
        'Net Generation (MWh)':'sum'
        })

    #Calculating Plant Heat Rates & removing plants with no data
    EIA860df['Heat Rate (MMBtu/Mwh)'] = round(EIA860df['Elec Fuel Consumption (MMBtu)']/EIA860df['Net Generation (MWh)'],2)
    EIA860df = EIA860df[EIA860df['Heat Rate (MMBtu/Mwh)'].notna()]


    #Sorting by status, then heat rates
    EIA860df['Status'] = pd.Categorical(EIA860df['Status'], categories=OperatingList, ordered=True)
    EIA860df = EIA860df.sort_values(by=['Status','Heat Rate (MMBtu/Mwh)'])
    EIA860df= EIA860df.reset_index(drop=True, inplace=False)

    #==== Modelling Dispatched Plants and Power Price Estimate ====
    HourlyDatadf['PacNW Gen excl. Gas'] = HourlyDatadf['PNW Coal'] + HourlyDatadf['PNW Hydro'] + HourlyDatadf['Wind'] + HourlyDatadf['Solar'] + HourlyDatadf['Imports'] + HourlyDatadf['Columbia'] 
    HourlyDatadf['Gas Dispatch Needed'] = HourlyDatadf['PacNW Demand'] - HourlyDatadf['PacNW Gen excl. Gas']

    HourlyDatadf['Estimated Marginal Price'] = None
    HourlyDatadf['Dispatched Plants (Descending)'] = None
    for i, row in HourlyDatadf.iterrows():
        GasNeeded = row['Gas Dispatch Needed']
        GasPrice = row['Gas Price']

        HourlyDatadf.at[i,'Estimated Marginal Price'], HourlyDatadf.at[i,'Dispatched Plants (Descending)'] = GetDispatchandLMP(GasNeeded,GasPrice)
        

    
    
    EIA860df.to_excel("Gas Generation Stack.xlsx")
    HourlyDatadf.to_excel("Hourly Data.xlsx")
    
    
    
    
    

   






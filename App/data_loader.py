import pandas as pd

class DataLoader:
    def __init__(self, emd_path, om_path):
        self.emd_path = emd_path
        self.om_path = om_path

    def load_data(self):
        emd_df = pd.read_csv(self.emd_path)
        emd_df['Timestamp'] = pd.to_datetime(emd_df['Timestamp'], errors='coerce')
        emd_df = emd_df.drop(columns=['Renewable_Investment_Costs', 'Fossil_Fuel_Costs', 'Electricity_Export_Prices', 'Market_Elasticity', 'Subsidies', 'Energy_Production_By_Solar', 'Energy_Production_By_Wind', 'Energy_Production_By_Coal', 'Energy_Storage_Capacity', 'GHG_Emissions', 'Renewable_Penetration_Rate', 'Regulatory_Policies', 'Energy_Access_Data', 'LCOE', 'ROI', 'Net_Present_Value', 'Population_Growth', 'Optimal_Energy_Mix', 'Electricity_Price_Forecast', 'Project_Risk_Analysis', 'Investment_Feasibility'])

        om_df = pd.read_csv(self.om_path)
        om_df['Timestamp'] = pd.to_datetime(om_df['time'], errors='coerce')
        om_df = om_df.drop(columns=['time'])

        df = pd.merge(emd_df, om_df, how='left', on='Timestamp')
        return df
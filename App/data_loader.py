import pandas as pd

class DataLoader:
    def __init__(self, pgc_df, om_path):
        self.pgc_df = pgc_df
        self.om_path = om_path

    def load_data(self):
        pgc_df = pd.read_csv(self.pgc_df)
        pgc_df['Timestamp'] = pd.to_datetime(pgc_df['date_id'], errors='coerce')
        pgc_df = pgc_df.drop(columns=['date_id', 'date_id.1'])

        om_df = pd.read_csv(self.om_path)
        om_df['Timestamp'] = pd.to_datetime(om_df['time'], errors='coerce')
        om_df = om_df.drop(columns=['time'])

        df = pd.merge(pgc_df, om_df, how='left', on='Timestamp')
        return df
import pandas as pd
import numpy as np
import gsw
import sys
from datetime import timedelta
from src.exception import CustomException
from src.logger import logging
from src.components.data_loading import DataLoader

def calculate_month(file_number):
    hours = int(file_number)
    start_date = pd.Timestamp('2000-01-01 00:00:00')
    target_datetime = start_date + timedelta(hours=hours)
    return target_datetime.month

def preprocess_data(df, file_number):
    try:
        # Convert file_number to month
        month = calculate_month(file_number)
        df['month'] = month

        # Define missing value and tolerance
        missing_value = 9.969210e+36
        tolerance = 1e+30

        # Filter out missing values
        mask = np.isclose(df['Salinity'], missing_value, atol=tolerance) | np.isclose(df['Temperature'], missing_value, atol=tolerance)
        df_filtered = df[~mask]

        # Calculate pressure and sound speed
        df_filtered['pressure'] = df_filtered.apply(lambda x: gsw.conversions.p_from_z(-x['Depth'], x['Latitude']), axis=1)
        df_filtered['c'] = df_filtered.apply(lambda x: gsw.sound_speed(x['Temperature'], x['Salinity'], x['pressure']), axis=1)

        # Map surface salinity (SSS) and temperature (SST)
        surface_df = df_filtered[df_filtered['Depth'] == 0.0]
        surface_df = surface_df.drop_duplicates(subset=['Latitude', 'Longitude'])

        sss_mapping = surface_df.set_index(['Latitude', 'Longitude'])['Salinity'].to_dict()
        sst_mapping = surface_df.set_index(['Latitude', 'Longitude'])['Temperature'].to_dict()

        df_filtered['SSS'] = df_filtered.apply(lambda row: sss_mapping.get((row['Latitude'], row['Longitude'])), axis=1)
        df_filtered['SST'] = df_filtered.apply(lambda row: sst_mapping.get((row['Latitude'], row['Longitude'])), axis=1)

        logging.info("Data preprocessing completed successfully.")
        return df_filtered
    except Exception as e:
        logging.error("An error occurred during data preprocessing.", exc_info=True)
        raise CustomException(e, sys)
    
# if __name__ == "__main__":
#     file_number = '2759'  # Example file number

#     # Initialize DataLoader and load & merge datasets
#     data_loader = DataLoader(file_number=file_number)
#     combined_df = data_loader.load_and_merge_datasets()

#     # Now preprocess the loaded and combined data
#     preprocessed_df = preprocess_data(combined_df, file_number)

#     # Display the head of the preprocessed DataFrame to check the results
#     print(preprocessed_df.head())

import pandas as pd
import numpy as np
import math
import os
import joblib
from src.config import ConFig
from src.components.data_loading import DataLoader
from src.components.preprocessing import preprocess_data,calculate_month
from src.components.model import KNNModel
from src.logger import logging

def compute_features(df):
    # Convert 'month' column to integer if not already
    df['month'] = df['month'].astype(int)

    # Calculate 'lat_norm'
    df['lat_norm'] = df['Latitude'] / 90

    # Calculate 'lon_sin' and 'lon_cos'
    df["lon_norm"] = 2 * math.pi * df["Longitude"] / 360
    df["lon_sin"] = np.sin(df["lon_norm"])
    df["lon_cos"] = np.cos(df["lon_norm"])

    # Calculate 'mon_cos' and 'mon_sin'
    df["mon_norm"] = 2 * math.pi * df["month"] / 12
    df["mon_cos"] = np.cos(df["mon_norm"])
    df["mon_sin"] = np.sin(df["mon_norm"])

    # Select relevant columns for prediction
    return df[['lat_norm', 'lon_sin', 'lon_cos', 'SSS', 'SST', 'mon_cos', 'mon_sin', 'Depth', 'c']]

def make_predictions(df):
    # Assuming df is preprocessed and contains all necessary columns for prediction
    X = df[['lat_norm', 'lon_sin', 'lon_cos', 'SSS', 'SST', 'mon_cos', 'mon_sin', 'Depth']].values
    knn_model = KNNModel()
    predictions = knn_model.predict(X)
    df['predicted_c'] = predictions
    return df

if __name__ == "__main__":
    file_number = '2759'  # Example file number

    # Initialize DataLoader and load & merge datasets
    data_loader = DataLoader(file_number=file_number)
    combined_df = data_loader.load_and_merge_datasets()

    df_filtered= preprocess_data(combined_df, file_number)

    # Compute features
    df_with_features = compute_features(df_filtered)

    # Make predictions
    final_df = make_predictions(df_with_features)
    logging.info("Predicted successfully.")

    # Save the final DataFrame
    output_file_path = os.path.join(ConFig.DATA_DIR, f'final_data_{file_number}.pkl')
    final_df.to_pickle(output_file_path)
    
    logging.info("Final DataFrame saved successfully.")

    # Display the head of the final DataFrame to check the results
    print(final_df.head())

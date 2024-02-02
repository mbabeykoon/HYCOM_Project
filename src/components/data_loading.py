import xarray as xr
import pandas as pd
import numpy as np
import sys
import scipy.io

# sys.path.append('d:/Nik/Hy/src')
from src.config import ConFig
from src.exception import CustomException
from src.logger import logging



class DataLoader:
    def __init__(self, file_number):
        self.file_number = file_number
        self.salt_file_path = ConFig.SALT_FILE_TEMPLATE.format(file_number)
        self.temp_file_path = ConFig.TEMP_FILE_TEMPLATE.format(file_number)
        self.grid_mat_file = ConFig.GRID_MAT_FILE
        self.ssp_mat_file = ConFig.SSP_MAT_FILE_TEMPLATE.format(file_number)

    def load_salt_data(self):
        try:
            logging.info(f"Loading salt data from {self.salt_file_path}")
            ds_s = xr.open_dataset(self.salt_file_path)
            df_s = ds_s.to_dataframe().reset_index()
            ds_s.close()
            logging.info("Salt data loaded successfully")
            return df_s
        except Exception as e:
            logging.error("Error loading salt data", exc_info=True)
            raise CustomException(e,sys)

        
            

    def load_temp_data(self):
        try:
            logging.info(f"Loading temperature data from {self.temp_file_path}")
            ds_t = xr.open_dataset(self.temp_file_path)
            df_t = ds_t.to_dataframe().reset_index()
            ds_t.close()
            logging.info("Temperature data loaded successfully")
            return df_t
        except Exception as e:
            logging.error("Error loading temperature data", exc_info=True)
            raise CustomException(e,sys)

   
    def load_ssp_data(self):
        try:
            logging.info(f"Loading SSP data from {self.ssp_mat_file}")
            # Load SSP data
            mat_contents_ssp = scipy.io.loadmat(self.ssp_mat_file)
            ssp = mat_contents_ssp['ssp']

            # Load grid data to ensure lon, lat, and depth are defined
            mat_contents_grid = scipy.io.loadmat(self.grid_mat_file)
            depth = mat_contents_grid['depth'].flatten()
            lat = mat_contents_grid['lat'].flatten()
            lon = mat_contents_grid['lon'].flatten()

            # Now that lon, lat, and depth are defined, create the meshgrid
            lon_grid, lat_grid, depth_grid = np.meshgrid(lon, lat, depth, indexing='ij')

            # Reshape ssp correctly
            ssp_reshaped = ssp.reshape(-1, ssp.shape[-1])  # Assuming ssp.shape[2] is correct, but safer to use -1

            # Create DataFrame
            df_ssp = pd.DataFrame({
                'Depth': depth_grid.flatten(),
                'Latitude': lat_grid.flatten(),
                'Longitude': lon_grid.flatten(),
                'SSP': ssp_reshaped.flatten()
            })

            logging.info("SSP data loaded successfully")
            return df_ssp
        except Exception as e:
            logging.error("Error loading SSP data", exc_info=True)
            raise CustomException(e, sys)


    def load_and_merge_datasets(self):
        try:
            df_s = self.load_salt_data()
            df_t = self.load_temp_data()
            df_st = pd.merge(df_s, df_t, on=['Dates', 'Depth', 'Latitude', 'Longitude'], how='inner')
            df_ssp = self.load_ssp_data()
            df_combined = pd.merge(df_st, df_ssp, on=['Depth', 'Latitude', 'Longitude'], how='inner')
            logging.info("Datasets merged successfully")
            return df_combined
        except Exception as e:
            logging.error("Error merging datasets", exc_info=True)
            raise CustomException(e,sys)
        

#to code check 
# if __name__ == "__main__":
#     file_number = '2759'
#     data_loader = DataLoader(file_number)  # Create an instance of DataLoader
#     try:
#         df_salt = data_loader.load_salt_data()
#         df_temp = data_loader.load_temp_data()
#         df_ssp = data_loader.load_ssp_data()
#         # Optionally, if you want to merge datasets as well
#         df_combined = data_loader.load_and_merge_datasets()
#         print(df_combined.head())
#     except Exception as ex:
#         logging.error(f"Data loading failed: {ex}")

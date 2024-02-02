import xarray as xr
import pandas as pd
import numpy as np
import sys
import scipy.io

# sys.path.append('d:/Nik/Hy/src')
from src.config import Config
from exception import CustomException
from logger import logging



class DataLoader:
    def __init__(self, file_number):
        self.file_number = file_number
        self.salt_file_path = Config.SALT_FILE_TEMPLATE.format(file_number)
        self.temp_file_path = Config.TEMP_FILE_TEMPLATE.format(file_number)
        self.grid_mat_file = Config.GRID_MAT_FILE
        self.ssp_mat_file = Config.SSP_MAT_FILE_TEMPLATE.format(file_number)

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
            mat_contents = scipy.io.loadmat(self.ssp_mat_file)
            ssp = mat_contents['ssp'].reshape(-1, ssp.shape[2])
            lon_grid, lat_grid, depth_grid = np.meshgrid(lon, lat, depth, indexing='ij')
            df_ssp = pd.DataFrame({
                'Depth': depth_grid.flatten(),
                'Latitude': lat_grid.flatten(),
                'Longitude': lon_grid.flatten(),
                'SSP': ssp.flatten()
            })
            logging.info("SSP data loaded successfully")
            return df_ssp
        except Exception as e:
            logging.error("Error loading SSP data", exc_info=True)
            raise CustomException(e,sys)

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


if __name__ == "__main__":
    file_number = '2759'
    try:
        df_salt = load_salt_data(file_number)
        df_temp = load_temp_data(file_number)
        df_ssp = load_ssp_data(file_number)
    except Exception as ex:
        logging.error(f"Data loading failed: {ex}")
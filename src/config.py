import os

class ConFig:
    DATA_DIR = 'artifacts/data'
    MODEL_DIR = 'artifacts/models'
    SALT_FILE_TEMPLATE = os.path.join(DATA_DIR, 'salt', 'Salt_{}_.nc')
    TEMP_FILE_TEMPLATE = os.path.join(DATA_DIR, 'temp', 'Temp_{}_.nc')
    GRID_MAT_FILE = os.path.join(DATA_DIR, 'HYCOM_Grid.mat')
    SSP_MAT_FILE_TEMPLATE = os.path.join(DATA_DIR, 'ssp', 'SSP_.{}.mat')
    MODEL_PATH = os.path.join(MODEL_DIR, 'knn_final.joblib')

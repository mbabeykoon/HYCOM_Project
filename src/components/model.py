import joblib
import numpy as np
import sys
from src.config import ConFig
from src.exception import CustomException
from src.logger import logging

class KNNModel:
    def __init__(self):
        try:
            self.model = joblib.load(ConFig.MODEL_PATH)
            logging.info("Model loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Model file not found at {ConFig.MODEL_PATH}.", exc_info=True)
            raise CustomException(e, sys)
        except Exception as e:
            logging.error("Error loading the model.", exc_info=True)
            raise CustomException(e, sys)
        
    def predict(self, X):
        try:
            if self.model is not None:
                # Assuming X is a numpy array or similar; add checks or conversion if needed
                predictions = self.model.predict(X)
                logging.info(f"Predictions made successfully for input of shape {X.shape}.")
                return predictions
            else:
                logging.error("Model is not loaded.")
                raise CustomException("Model is not loaded.", sys)
        except Exception as e:
            logging.error("Error making predictions.", exc_info=True)
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        # Initialize the KNNModel
        knn_model = KNNModel()
        

        X_sample = np.array([[0.522126,	-0.446360,	0.894853,	32.8405,	9.1031,	-0.500000,	-8.660254e-01,	631.732014]])  # Example input, modify as necessary
        
        # Make predictions using the loaded model
        predictions = knn_model.predict(X_sample)
        
        # Print the predictions to verify
        print("Predictions:", predictions)
        
    except CustomException as e:
        logging.error(f"An error occurred: {e}")


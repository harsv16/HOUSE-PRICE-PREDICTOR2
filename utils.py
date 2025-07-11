import pandas as pd
import pickle

def load_model(path='model.pkl'):
    """
    Load the trained machine learning model from the given path.
    
    Parameters:
        path (str): Path to the pickle (.pkl) file containing the model.
    
    Returns:
        model: Loaded machine learning model.
    """
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Make sure 'model.pkl' exists.")

def preprocess_input(location, area, bedrooms, bathrooms, age):
    """
    Convert user inputs into a DataFrame suitable for prediction.
    
    Parameters:
        location (str): Location of the house.
        area (int): Area in square feet.
        bedrooms (int): Number of bedrooms.
        bathrooms (int): Number of bathrooms.
        age (int): Age of the house in years.
    
    Returns:
        pd.DataFrame: A DataFrame containing the input in correct format.
    """
    data = {
        'Location': [location],
        'Area (sqft)': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age]
    }
    return pd.DataFrame(data)

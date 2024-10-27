import json
import numpy as np
import pandas as pd
import pickle
import os

from logger import logging
from customexcept import CustomException

def load_config(filename):
    """
    Read config.json

    Args: 
    filename (str): name of the key 

    Returns : 
    dict : items in the key
    """
    logging.info("Reading config.json")
    with open('src/config.json', 'r') as f:
        config = json.load(f)  
        
    return config[filename]

def save_model(model,name):
    """
    Save a machine learning model to a .pkl file.

    Args: 
    object: The machine learning model to be saved
    filename (str): The name of the .pkl file.

    Example:
    >>> save_model(model, filename)
    """
    try:
        # Read config and set path
        config = load_config("models")
        path = config["model_path"]

        # Ensure there is an existing path
        logging.info("Saving models...")
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        # Write model to path
        with open(f"{dir_path}/{name}.pkl", 'wb') as file:
            pickle.dump(model,file)
            print("Model have been saved in models")

    except Exception as e :
            logging.error(f"Fail to save model : {e}")

def load_model(name : str): 
    """
    Load a machine learning model from a .pkl file.

    Args: 
    filename (str): The path to the .pkl file.

    Returns:
    object: The machine learning model loaded from the .pkl file.

    Example:
    >>> load_model(filename)
    """
    try:
        # Read config and set path
        config = load_config("models")
        path = config["model_path"]

        # Ensure there is an existing path 
        logging.info("Loading models...")
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        
        # load .pkl file to model
        with open(f"{dir_path}/{name}.pkl", 'rb') as file:
            model = pickle.load(file)
            print("Model had been loaded")

        return model

    except Exception as e :
            logging.error(f"Fail to load model : {e}")
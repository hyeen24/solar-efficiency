import os
import sys
import pandas as pd
import requests
import sqlite3
from dataclasses import dataclass
from utils import load_config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from logger import logging
from customexcept import CustomException

@dataclass
class DataIngestionConfig:
    config = load_config('data_ingestion')
    db_path: str = os.path.join(config['db_path'])
    raw_weather_data_path: str = os.path.join(config["raw_weather_data_csv"])
    raw_air_quality_data_path: str = os.path.join(config["raw_air_quality_data_csv"])
    weather_url = config["weather_url"]
    air_quality_url = config["air_quality_url"]
    weather_db_path: str = os.path.join(config["weather_db_path"])
    air_quality_db_path: str = os.path.join(config["air_quality_db_path"])



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def reading_from_db(self):
        """
        This function read sent requests to database url with the path stated in DataIngestionConfig.

        Saving the response content into data folder.
        """
        logging.info("Reading from database ....")
        try: 

            # Request
            weather_response = requests.get(self.ingestion_config.weather_url)
            air_quality_response = requests.get(self.ingestion_config.air_quality_url)


            logging.info("Saving db into data path...")
            db_path= os.path.dirname(self.ingestion_config.db_path)
            os.makedirs(db_path, exist_ok=True)

            # Save db
            with open(self.ingestion_config.weather_db_path, 'wb') as f:
                f.write(weather_response.content)

            with open(self.ingestion_config.air_quality_db_path, 'wb') as f:
                f.write(air_quality_response.content)

            logging.info("Data is saved")

            return 
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def reading_dataframe_from_path(self):
        logging.info("Reading dataframe....")
        try: 
            # Connect to the weather database
            weather_conn = sqlite3.connect(self.ingestion_config.weather_db_path)
            air_quality_conn = sqlite3.connect(self.ingestion_config.air_quality_db_path)

            # Define a query
            weather_query = "SELECT * FROM weather" 
            air_quality_query = "SELECT * FROM air_quality" 

            # Execute the query and load the results into a Pandas DataFrame
            weather_df = pd.read_sql_query(weather_query, weather_conn)
            air_df = pd.read_sql_query(air_quality_query, air_quality_conn)

            # Close the connection
            weather_conn.close()
            air_quality_conn.close()

            os.makedirs(self.ingestion_config.db_path+"/raw",exist_ok=True)
            # Saving dataframe into csv
            weather_df.to_csv(self.ingestion_config.raw_weather_data_path,index=False)
            air_df.to_csv(self.ingestion_config.raw_air_quality_data_path,index=False)
            logging.info("Data retrieved and saved into csv")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.reading_from_db()
    obj.reading_dataframe_from_path()
    

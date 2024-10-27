import os
import json
import sys
from dataclasses import dataclass
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from logger import logging
from customexcept import CustomException
from utils import load_config

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

@dataclass
class DataTransformationConfig:
    config = load_config("data_transformation")
    raw_weather_data_path: str = os.path.join(config['raw_weather_data_path'])
    raw_air_quality_data_path: str = os.path.join(config['raw_air_quality_data_path'])
    merged_data_path: str = os.path.join(config['merged_data_path'])
    final_data_path: str = os.path.join(config['final_data_path'])
    clean_data_path: str = os.path.join(config['clean_data_path'])


class DataTransformation:
    def __init__(self):
        self.transform_config = DataTransformationConfig()

    def merge_rows(group):
        merged = group.iloc[0].copy()
        for i in range(1, len(group)):
            for col in group.columns:
                if pd.isna(merged[col]) or merged[col] in [None, '-', '--']:
                    merged[col] = group.iloc[i][col] if not pd.isna(group.iloc[i][col]) and group.iloc[i][col] not in [None, '-', '--'] else merged[col]
        return merged


    def drop_duplicates_merge(self):
        """
        This function drop duplicates for weather.csv and air_quality.csv

        return 
            pd.DataFrame
        """
        try:
            logging.info("Reading csv files....")
            weather_df = pd.read_csv(self.transform_config.raw_weather_data_path)
            air_df = pd.read_csv(self.transform_config.raw_air_quality_data_path)

            # Drop duplicate data
            weather_df = weather_df.drop_duplicates()
            air_df = air_df.drop_duplicates()
            logging.info("Drop duplicates for both dataframe.")

            # Getting dataframe for duplicated row for air_quality data
            duplicates = air_df[air_df.duplicated('data_ref',keep=False)]

            # Merging row
            merged_rows = duplicates.groupby('data_ref').apply(DataTransformation.merge_rows,include_groups=False).reset_index(drop=True)
            logging.info("Merge air duplicated rows.")

            # Merging psi and pm25 into same data frame
            non_duplicates = air_df[~air_df['data_ref'].isin(duplicates['data_ref'])]
            final_air_df = pd.concat([non_duplicates, merged_rows], ignore_index=True)  

            os.makedirs(self.transform_config.clean_data_path,exist_ok=True)
            df = pd.merge(weather_df, final_air_df, on='date', how='inner')
            df = df.iloc[:,2:]

            df.to_csv(self.transform_config.merged_data_path)
            logging.info("Dataframe merged saved successfully.")

            return df
        
        except Exception as e:
            logging.error(f"Unable to drop duplicates and merged data : {e}")
            raise CustomException(e,sys)
        
    def transforming_dtype(self):
        """
        This function transformed dataframe dtype 

        return 
            pd.DataFrame
        """
        logging.info("Transforming data type.....")
        try:
            # Read dataframe
            df = pd.read_csv(self.transform_config.merged_data_path)

            # Convert date to datetime format
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            logging.info("date dtype was transformed")

            columns = ['Daily Rainfall Total (mm)','Highest 30 Min Rainfall (mm)','Highest 60 Min Rainfall (mm)','Highest 120 Min Rainfall (mm)','Min Temperature (deg C)','Maximum Temperature (deg C)','Min Wind Speed (km/h)',
           'Max Wind Speed (km/h)','pm25_north','pm25_south','pm25_east','pm25_west','pm25_central','psi_north','psi_south','psi_east','psi_west','psi_central']
            
            # Transforming object to numeric
            for column in columns:
                df[column] = df[column].apply(pd.to_numeric, errors='coerce')

            df = df.iloc[:,1:]
            logging.info("Dtype was successfully transformed.")
            return df
        

        except Exception as e:
            logging.error(f"Unable to transform dtype of dataframe : {e}")
            raise CustomException(e, sys)

    def cleaning_formatting_value(self):
        """
            This function will replace missing values in data and saved the data

        """
        logging.info("Replacing missing values")
        try:
            # Read dataframe
            df = self.transforming_dtype()

            df = df.drop(columns=['data_ref_y'])
            # Assigning feature with na value 
            missing_value_features = [feature for feature in df if df[feature].isna().sum() > 0]

            # Replace missing values with median value
            for feature in missing_value_features:
              df[feature] = df[feature].fillna(df[feature].median())
            logging.info("missing values was replaced.")

            # Mapping dictionary
            dew_mapping = {
                'High': 'High',
                'Very High': 'Very High',
                'Very Low': 'Very Low',
                'Moderate': 'Moderate',
                'Low': 'Low',
                'High Level': 'High',
                'H': 'High',
                'HIGH': 'High',
                'high': 'High',
                'very high': 'Very High',
                'Extreme': 'Extreme',
                'VERY HIGH': 'Very High',
                'VH': 'Very High',
                'M': 'Moderate',
                'MODERATE': 'Moderate',
                'LOW': 'Low',
                'Normal': 'Moderate',  # Assuming 'Normal' is akin to 'Moderate'
                'Minimal': 'Very Low',  # Assuming 'Minimal' is akin to 'Very Low'
                'VL': 'Very Low',
                'very low': 'Very Low',
                'low': 'Low',
                'moderate': 'Moderate',
                'Below Average': 'Low',  # Assuming 'Below Average' is akin to 'Low'
                'L': 'Low',
                'VERY LOW': 'Very Low'
            }

            # Mapping dictionary for wind direction
            wind_direction_mapping = {
                'SW': 'Southwest',
                'SW.': 'Southwest',
                'southwest': 'Southwest',
                'SOUTHEAST': 'Southeast',
                'southeast': 'Southeast',
                'SE': 'Southeast',
                'SE.': 'Southeast',
                'NORTHEAST': 'Northeast',
                'northeast': 'Northeast',
                'NE': 'Northeast',
                'NE.': 'Northeast',
                'NORTHWEST': 'Northwest',
                'northwest': 'Northwest',
                'NW': 'Northwest',
                'NW.': 'Northwest',
                'Northward': 'North',
                'NORTH': 'North',
                'north': 'North',
                'N': 'North',
                'N.': 'North',
                'Southward': 'South',
                'SOUTH': 'South',
                'south': 'South',
                'S': 'South',
                'S.': 'South',
                'WEST': 'West',
                'west': 'West',
                'W': 'West',
                'W.': 'West',
                'EAST': 'East',
                'east': 'East',
                'E': 'East',
                'E.': 'East'
            }

            # Mapping wind directioin
            df['Wind Direction'] = df['Wind Direction'].replace(wind_direction_mapping)

            # Mapping dew point data
            df['Dew Point Category'] = df['Dew Point Category'].replace(dew_mapping)
            logging.info("Data in wind direction and dew point catergory was formatted.")

            # Average wind speed 
            df['average_wind_speed'] = (df['Min Wind Speed (km/h)'] + df['Max Wind Speed (km/h)']) /2

            # Average temperature
            df['average_temperature'] = (df['Min Temperature (deg C)']+ df['Maximum Temperature (deg C)']) /2

            # Average psi
            df['psi_average'] = (df['psi_north'] + df['psi_central'] + df['psi_east'] + df['psi_south'] + df['psi_west']) /5

            # Average pm25
            df['pm25_average'] = (df['pm25_central'] + df['pm25_north'] + df['pm25_south'] + df['pm25_east'] + df['pm25_west']) / 5

            logging.info("Average feature was created for psi, pm25, wind speed and temperature")

            # Assign columns to drop
            columns_to_drop = ['pm25_north','pm25_south','pm25_east','pm25_west','pm25_central','psi_central',
                            'psi_north','psi_south','psi_east','psi_west',
                            'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)','Highest 120 Min Rainfall (mm)',
                            'Min Wind Speed (km/h)', 'Max Wind Speed (km/h)',
                            'Min Temperature (deg C)', 'Maximum Temperature (deg C)']
            
            logging.info("Similar feature is dropped to prevent noise")

            # normalize the data
            num_feature = ['Air Pressure (hPa)','Wet Bulb Temperature (deg F)','average_wind_speed','Daily Rainfall Total (mm)',
                              'Sunshine Duration (hrs)','Cloud Cover (%)','Relative Humidity (%)','average_temperature','psi_average','pm25_average']

            for feature in num_feature:
                if 0 in df[feature]:
                    pass
                else:
                    df[feature] = np.log(df[feature])

            # Drop Columns
            df = df.drop(columns=columns_to_drop)

            # Removing Outliers
            mask = np.ones(len(df), dtype=bool)

            for feature in num_feature:
                z_scores = np.abs(stats.zscore(df[feature]))
                feature_mask = z_scores < 3
                mask &= feature_mask

            df_filtered = df[mask]

            df_filtered.to_csv(self.transform_config.final_data_path)

            logging.info("Cleaning and formatting completed.")
            return df 

        except Exception as e:
            logging.error(f"Unable to clean missing value : {e}")
            raise CustomException(e, sys)


if __name__ == "__main__" :
    # Create data transformation object
    obj = DataTransformation()
    obj.drop_duplicates_merge()
    obj.transforming_dtype()
    obj.cleaning_formatting_value()

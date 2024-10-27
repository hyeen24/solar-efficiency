import os
import sys
from logger import logging
from customexcept import CustomException
from dataclasses import dataclass
import pandas as pd
from utils import load_config

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

@dataclass
class DataPreprocessingConfig:
    config = load_config('data_preprocessing')
    clean_data_path: str = os.path.join(config['clean_data.csv'])

class DataPreprocessing:
    def __init__(self):
        self.preprocess_config = DataPreprocessingConfig()
    
    def preprocessing_pipeline(self):
        """
        This function create and return pipeline for preprocessing
        """
        logging.info("Creating preprocessing pipeline")
        try:

            # Categorical feature
            ohe_feature = ['Wind Direction']
            ode_feature = ['Dew Point Category']
                
            std_feature = ['Air Pressure (hPa)','Wet Bulb Temperature (deg F)','average_wind_speed']
            minmax_feature = ['Daily Rainfall Total (mm)','Sunshine Duration (hrs)','Cloud Cover (%)','Relative Humidity (%)','average_temperature','psi_average','pm25_average']

            # Norminal Encoder
            ohe_pipeline = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy="most_frequent")),
                ('OneHotEncoder',OneHotEncoder())
            ])

            # Ordinal Encoder
            ode_pipeline = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy="most_frequent")),
                ('OrdinalEncoder',OrdinalEncoder())
            ])

            # Standard Scaler
            num_standard = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy="median")),
                ('StandardScaler',StandardScaler())
            ])

            # Min Max Scaler
            num_minmax = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy="median")),
                ('MinMaxScaler',MinMaxScaler())
            ])

            # Column Transformer Pipe
            column_transformer = ColumnTransformer([
                ('nominal transformer', ohe_pipeline, ohe_feature),
                ('rank transformer', ode_pipeline, ode_feature),
                ('numerical standard', num_standard,std_feature),
                ('numerical minmax', num_minmax,minmax_feature)

            ])

            # Pipeline for preprocessing
            pipeline = Pipeline(steps=[
                ('preprocessing', column_transformer)
            ])
            logging.info("Pipeline created")
            return pipeline
        
        except Exception as e:
            logging.error(f"Fail to create pipeline : {e}")
            raise CustomException(e, sys)
    
    def scale_data(self):
        """
        This function scale the data and split the data

        return
            X_train, X_test, y_train, y_test

        """
        logging.info("Scaling data.....")
        try: 

            # Read data frame
            df = pd.read_csv(self.preprocess_config.clean_data_path)
            classifier = 'Daily Solar Panel Efficiency'
            
            X = df.drop(columns=classifier)
            y = df[classifier]

            # Pipeline
            pipeline = self.preprocessing_pipeline()

            # # Data preprocessing
            X = pipeline.fit_transform(X)

            # Spliting dataframe with train test split
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            # Oversample in training data
            smote = SMOTE(random_state=22)
            X_train, y_train = smote.fit_resample(X_train,y_train)

            logging.info("Data have been scaled")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Fail to scale data : {e}")
            raise CustomException(e, sys)
        
    
# if __name__ == "__main__":
#     obj = DataPreprocessing()
#     scale_data = obj.scale_data()
#     print(scale_data)
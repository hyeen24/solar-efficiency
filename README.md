## Problem Statement
Maximising solar power generation involves strategically managing operational and manpower costs,
 particularly by leveraging forecasted weather data to optimise operational planning. During
 high-efficiency days, which are influenced by favourable weather conditions, increasing battery capacity
 allows for the storage of excess power generated, ensuring a steady supply even during less optimal
 periods. Investing in advanced energy storage solutions and grid integration technologies is crucial.
 Conversely, low-efficiency days provide an ideal window for scheduled maintenance. By taking some
 panel arrays offline for cleaning and repairs during these periods, downtime is minimised, and overall
 system performance is maintained. Efficient management of these aspects ensures maximum power
 generation and reliability of solar power systems.
 
 As an Artificial Intelligence Engineer at XXXX, your primary responsibility is to develop models that
 classify solar panel efficiency as ‘Low’, ‘Medium’, or ‘High’. By leveraging historical same day forecasted
 weather data, implement predictive algorithms that identify and learn from patterns associated
 with varying efficiency levels.
 

#### Overview of Directory Structure

```bash
    ├── src
    │   ├── data_ingestion.py
    │   ├── data_preprocessing.py
    │   ├── data_transformation.py
    │   ├── model_training.py
    │   ├── logger.py
    │   ├── config.json
    │   ├── utils.py
    │   ├── customexcept.py
    │   └── __init__.py
    ├── data
    |   ├── raw
    |   |   ├── air_quality.csv
    |   |   └── weather.csv
    |   ├── clean
    |   |   ├── merged_data.csv
    |   |   └── final_data.csv
    │   ├── weather.db
    │   ├── air_quality.db
    ├── run.sh
    ├── eda.ipynb
    ├── model.ipynb
    ├── requirements.txt
    └── README.md
```

## Instruction

1. Clone the repository:
```bash
git clone https://github.com/hyeen24/aiap-18-pang-hong-yeen-736Z.git
```
2. Navigate to project directory
3. Installing dependencies
4. run.sh
5. Model parameters modification can be done at config.json
6. Use model_training.py for training the models

## Pipeline Overview

![Team document (1)](https://github.com/user-attachments/assets/82fd0e3f-7332-4828-a7a3-408fcd91c985)

## Key findings from EDA
- Air quality have duplicated data entry with pm25 on 1 entry while psi on the other entry.
- Both data have duplicated entry
- data type of numerical measure are in string
- Dew Point Category and Wind Direction have inconsitent data
- It is quite a unbalance data with majority of the efficiency labeled as medium
- Rainfall,pm25,psi catergories have similar distribution and high correlationship
- There huge amount of outliers
  
## Features Engineering
- Custom feature is created for average Wind speed , Temperature, psi, pm25
- To prevent noise, I have dropped psi , pm 25 and min / max of temperature and wind speed
- Performed log transformation to reduce skewed data and manage outliers (performance slightly increased)
- Removing of outliers using Zscore after log transformation
- Non Gaussian feature will be transformed by Min max scaler while Air pressure , wet bulb temperature and average wind speed will be done by standard scaler
- Dew Point Catergory will be transformed by OrdinalEncorder as it is non nominal data
  
## Models
- KNeighborsClassifer
- SVM
- RandomForestClassifier
- DecisionTreeClassifier

Decided to go with Random Forest as it provide a better overal performance be it presision recall or accuracy.

Uses GridsearchCV to perform hyperparameters tunining.

## Deployment
Model can be deployed in cloud server to predict the class based on data provided. However, prediction pipeline and training pipeline should be modified to train model continously with new data.



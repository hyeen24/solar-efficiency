#!/bin/bash
# set -e

# Creating enviroment
# python -m venv venv
# source .\venv\scripts\acttivate
# pip install --upgrade pip

# Run data ingestion
echo "Ingesting data.py ....."
python src/data_ingestion.py
echo "Data ingestion completed."

# Run data transformation
echo "Transforming data....."
python src/data_transformation.py
echo "Data transformation completed."

# Run model training
# echo "Building model ....."
# python src/model_training.py

# echo "Do you want to make a prediction? (Y/N)"
# read response

# response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
# if [ "$response" = "yes" ]; then
#     # Prompt the user for input
#     echo "Please enter the input data for prediction:"
#     read input_data
# else
#     echo "Prediction not requested. Exiting."
# fi

# read -p "Press any key to exit..."
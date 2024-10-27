import os
import sys
from dataclasses import dataclass
from utils import load_config, save_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay 
from data_preprocessing import DataPreprocessing
import matplotlib.pyplot as plt

from logger import logging
from customexcept import CustomException

@dataclass
class ModelTrainerConfig:
    config = load_config('model_training')

class ModelTrainer:
    def __init__(self):
        self.modeltrainer_config = ModelTrainerConfig()

    def model_selection(self):
        """
        This function display a selection menu of models
        """
        # Menu for models selection
        print("1. KNeighbors Classfier")
        print("2. SVM")
        print("3. Decision Tree")
        print("4. Random Forest [Default]")
        while True:
            
            model_num = input("Please select your model [Default press Enter]: ") 
            print(model_num)

            if model_num == "1":
                model_name = "KNeighbors Classifer"
                model = KNeighborsClassifier()
                return model, model_name
            
            elif model_num == "2":
                model_name = "SVM"
                model = SVC()
                return model, model_name
            
            elif model_num == "3":
                model_name = "Decision Tree"
                model = DecisionTreeClassifier()
                return model, model_name
                
            
            elif model_num == "" or model_num == "4":
                model_name = "Random Forest"
                model = RandomForestClassifier()
                return model, model_name
            
            else:
                print("Invalid selection. Please try again.")

    def train_model(self):
        """
        This function is reponsible for training the model
        """
        logging.info("Training Model.....")
        try:
            # Model selection
            model, model_name = self.model_selection()
            
            # Parameter settings in config.json
            model.set_params(**self.modeltrainer_config.config[model_name]['params'])   
            print("Using model : ", model_name , model)
            
            # Scaling and splitting data
            dataprocessing = DataPreprocessing()
            X_train, X_test, y_train, y_test = dataprocessing.scale_data()

            # Train model
            model.fit(X_train, y_train) 

            # Make predictions
            y_test_pred = model.predict(X_test)
           
            # Evaluate Train and Test dataset
            report = classification_report(y_test,y_test_pred)
            # Generate the confusion matrix
            cm = confusion_matrix(y_test, y_test_pred, labels=['Low', 'Medium', 'High'])

            # Print the confusion matrix
            print("Confusion Matrix:\n", cm)

            # Visualize the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
            disp.plot(cmap=plt.cm.Blues)
            plt.show()

            print(report)
            accuracy_line = [line for line in report.split('\n') if 'accuracy' in line]
            accuracy = float(accuracy_line[0].split()[1])
            print(f'Accuracy: {accuracy}')

            return model

        except Exception as e:
            logging.error(f"Unable to train model : {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Create a model trainer object
    obj = ModelTrainer()
    
    while True:
        model = obj.train_model()

        # Check if user want to save model
        response = input("Do you wish to save the model? Enter y to save : ")
        if response.lower() == "y" or response.lower() == "yes" :
            name = input("Enter your model name")
            save_model(model,name)
        
        # Await input from user
        response = input("Do you wish to make changes? : (y/N)")
        if response.lower() == "n" or response.lower() == "no" :
            break


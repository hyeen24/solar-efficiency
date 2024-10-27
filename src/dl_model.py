import os
import sys
import json
from logger import logging
from customexcept import CustomException
import tensorflow as tf
from dataclasses import dataclass
from utils import load_config
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import DataPreprocessing
import matplotlib.pyplot as plt

@dataclass
class DeepLearningModelConfig:
    config = load_config("deep_learning_model")
    model_path:str = os.path.join(config['models_path'])


class DeepLearningModel:
    def __init__(self):
        self.dl_config = DeepLearningModelConfig()

    def model_architect(self, X_train):
        nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(len(X_train[0]),)),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(3,activation='softmax')
        ])

        return nn_model

    def train_model(self):
        try:
            # Scaling and splitting data
            dataprocessing = DataPreprocessing()
            X_train, X_test, y_train, y_test = dataprocessing.scale_data()
           
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

            model = self.model_architect(X_train)
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=100, batch_size = 32, validation_split=0.2)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f'Test accuracy: {test_acc} , Test loss : {test_loss}')

            # Plot the training and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            return history
        
        except Exception as e :
            logging.error(f"Fail to train model : {e}")
            raise CustomException(e, sys)



if __name__ == "__main__":
    obj = DeepLearningModel()
    history = obj.train_model()
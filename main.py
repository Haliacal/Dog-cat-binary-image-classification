import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
from IPython.display import display

class IndianActorClassification:

    def __init__(self, width, height, epochs, batch_size):

        self.imge_width = width
        self.image_height = height
        self.batch_size = batch_size

        # Import data
        dataset_dir =  pathlib.Path("dataset/Bollywood Actor Images")
        label_dir = pathlib.Path("dataset/List of Actors.txt")

        train_ds = keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset='training',
            seed=7,
            image_size = (self.image_height,self.image_width),
            batch_size=batch_size
        )

        val_ds = keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset='validation',
            seed=7,
            image_size = (self.image_height,self.image_width),
            batch_size=batch_size
        )

    def summary(self):
        pass


    def train(self): 
        pass       
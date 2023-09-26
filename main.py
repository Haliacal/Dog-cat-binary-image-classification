import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib


class IndianActorClassification:

    def __init__(self, width, height, epochs, batch_size):

        self.imge_width = width
        self.image_height = height
        self.batch_size = batch_size

        # Datset and label paths
        self.dataset_dir =  pathlib.Path("dataset/Bollywood Actor Images")
        self.label_dir = pathlib.Path("dataset/List of Actors.txt")
        
        

        # Augumentation function (Helps with overfitting)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal",
            input_shape=(self.image_height,self.image_width,3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        self.train_ds = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=0.2,
            subset='training',
            seed=7,
            image_size = (self.image_height,self.image_width),
            batch_size=self.batch_size
        )

        
        self.val_ds = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=0.2,
            subset='validation',
            seed=7,
            image_size = (self.image_height,self.image_width),
            batch_size=self.batch_size
        )

        class_names = self.train_ds.class_names
        num_classes = len(class_names)

        self.model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(516, activation='relu'),
            layers.Dense(num_classes, name="outputs")
        ])

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  

    def summary(self):
        self.model.summary()

    def train(self): 
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        results = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        return results

    def plotLoss(self, results):
        acc = results.history['accuracy']
        val_acc = results.history['val_accuracy']

        loss = results.history['loss']
        val_loss = results.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

def main():
    classModel = IndianActorClassification()
    print(classModel.summary())
    results = classModel.train()
    classModel.plotLoss(results)


    

if __name__ == "__main__":
    main()
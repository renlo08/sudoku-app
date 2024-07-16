import logging

import keras
import numpy as np


class SudokuOCRClassifier:

    model = None
    image_width = 28  # number of pixels
    image_height = 28  # number of pixels
    image_depth = 1  # image are in gray
    classes = 10  # sudoku numbers are going from 1..9 or empty

    def __init__(self):
        self.optimizer = None
        self.model = None

    def set_optimizer(self):
        self.optimizer = keras.optimizers.SGD(learning_rate=.01, momentum=0.9, nesterov=True)

    def build_model(self):
        model = keras.models.Sequential(name='SudokuNet')

        # Input layer
        input_shape = (self.image_width, self.image_height, self.image_depth)
        model.add(keras.layers.Input(shape=input_shape))

        # first set of CONV => RELU => POOL layers
        model.add(keras.layers.Conv2D(32, (5, 5), padding="same"))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(keras.layers.Conv2D(32, (3, 3), padding="same"))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # first set of FC => RELU layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        # second set of FC => RELU layers
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        # softmax classifier
        model.add(keras.layers.Dense(self.classes))
        model.add(keras.layers.Activation("softmax"))
        self.model = model

    def load_weights(self, weights: str):
        if self.model is None:
            logging.info("Model is not built. Build the model first.")
        self.model.load_weights(weights)

    @classmethod
    def setup_classifier(cls, weights_file: str):
        instance = cls()
        instance.set_optimizer()
        instance.build_model()
        instance.model.compile(optimizer=instance.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        instance.load_weights(weights_file)
        return instance

    def predict(self, rois: np.ndarray):
        return self.model.predict(rois).argmax(axis=1)

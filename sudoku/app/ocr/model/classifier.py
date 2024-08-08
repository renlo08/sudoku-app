import logging
import os
from typing import Generator
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning,
                        module='keras.src.trainers.data_adapters.py_dataset_adapter')

import keras
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import numpy as np
from tensorflow.python.keras import callbacks

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Create a custom logger
logger = logging.getLogger(__name__)
# Set the level of logger
logger.setLevel(logging.INFO)
# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
console_format = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
# Add handlers to the logger
logger.addHandler(console_handler)


class SudokuOCRClassifier:
    model = None
    image_width = 32  # number of pixels
    image_height = 32  # number of pixels
    image_depth = 1  # image are in gray
    classes = 10  # sudoku numbers are going from 1..9 or empty
    epochs = 100
    batch_size = 32

    def __init__(self, **kwargs):
        self.optimizer = None
        self.model = None
        self.history = {}
        self.weights_file = kwargs.get('weights_file', 'model/sudoku_ocr_classifier.weights.h5')
        self.state = 'init'

    def set_optimizer(self):
        logger.info('Setting optimizer')
        initial_learning_rate = 0.01
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )

        self.optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    def build_model(self):
        logger.info("Building model structure...")
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
        self.state = 'build'
        logger.info(f"Model structure built:\n{model.summary()}")

    def load_weights(self, weights: str):
        if self.model is None:
            logger.info("Model is not built. Build the model first.")
        self.model.load_weights(weights)

    def save_weights(self, file_path: str):
        if self.state != 'trained':
            logger.info("Model is not trained. Train the model first.")
        self.model.save_weights(file_path)

    @classmethod
    def prepare(cls, load_weights: bool = False, **kwargs):
        logger.info("Preparing classifier...")
        instance = cls(**kwargs)
        instance.set_optimizer()
        instance.build_model()
        logger.info("Compiling the model...")
        instance.model.compile(optimizer=instance.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if load_weights:
            logger.info("Loading weights...")

            instance.load_weights(instance.weights_file)
            instance.state = 'trained'
        logger.info("Classifier is ready to use.")
        return instance

    def predict(self, rois: np.ndarray):
        logger.info(f"Run inference for {rois.shape[0]} digits.")
        probs = self.model.predict(rois, verbose=1)
        max_prob = np.max(probs, axis=1)
        return np.where(max_prob > 0.6, probs.argmax(axis=1), 0)

    def predict_generator(self, generator: Generator):
        return self.model.predict(generator, verbose=0)

    def setup_callbacks(self):
        return [callbacks.EarlyStopping(monitor='val_loss',
                                        patience=15,
                                        restore_best_weights=True)]

    def train(self, train_data: ImageDataGenerator,
              val_data: ImageDataGenerator):
        """ Train the model """

        devices = tf.config.list_physical_devices()
        print("\nDevices: ", devices)

        if gpus := tf.config.list_physical_devices('GPU'):
            details = tf.config.experimental.get_device_details(gpus[0])
            logger.info(f"GPU details: {details}")
        logger.info("Training classifier...")

        callbacks = self.setup_callbacks()

        # Train the classifier
        self.history = self.model.fit(
            train_data,
            steps_per_epoch=train_data.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=val_data,
            validation_steps=val_data.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1)
        self.state = 'trained'
        logger.info("Training complete.")

    def evaluate_performance(self, test_data: ImageDataGenerator):
        logger.info("Evaluating classifier performance...")
        inference = self.predict_generator(test_data)
        inference = np.argmax(inference, axis=1)
        klass = [str(i) for i in range(self.classes)]  # classes should be string list
        report = metrics.classification_report(test_data.classes, inference, target_names=klass)
        logger.info(f"classification report:\n{report}")

    def save(self):
        self.model.save_weights(self.weights_file)

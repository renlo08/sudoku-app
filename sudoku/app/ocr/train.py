import os
from typing import Tuple

from tensorflow.keras import preprocessing

from app.ocr.model import classifier


def invert_color(image):
    return 1 - image


def prepare_train_val_test_set(batch_size: int, image_size: Tuple[int, int]):
    """ Prepare the train, validation and test set """

    # Create generator with augmented image for training
    train_gen = preprocessing.image.ImageDataGenerator(
        preprocessing_function=invert_color,
        rescale=1. / 255,
        rotation_range=2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=2,
        brightness_range=[0.5, 1.1],
        validation_split=0.25)

    # Create generator with dataset image for validation
    val_test_gen = preprocessing.image.ImageDataGenerator(
        preprocessing_function=invert_color,
        rescale=1. / 255,
        validation_split=0.333)

    data_dir = os.path.join(os.getcwd(), 'data')

    # Adjusting splits so that distribution becomes 60% train, 20% validation, 20% test in total data
    train_generator = train_gen.flow_from_directory(
        batch_size=batch_size,
        directory='./data',
        color_mode="grayscale",
        shuffle=True,
        target_size=image_size,
        class_mode='categorical',
        seed=42,
        subset='training')

    validation_generator = val_test_gen.flow_from_directory(
        batch_size=batch_size,
        directory='./data',
        color_mode="grayscale",
        shuffle=True,
        target_size=image_size,
        class_mode='categorical',
        seed=42,
        subset='training')  # Use 'training' to take 2/3 of the remaining 40% (after training set)

    test_generator = val_test_gen.flow_from_directory(
        batch_size=batch_size,
        directory='./data',
        color_mode="grayscale",
        shuffle=True,
        target_size=image_size,
        class_mode='categorical',
        seed=42,
        subset='validation')  # Use 'validation' to take 1/3 of the remaining 40% (after training set)

    return train_generator, validation_generator, test_generator


if __name__ == '__main__':
    # Collect relevant parameters
    batch_size = classifier.SudokuOCRClassifier.batch_size
    image_size = classifier.SudokuOCRClassifier.image_width, classifier.SudokuOCRClassifier.image_height

    # Generate the train, validation and test set
    train, validation, test = prepare_train_val_test_set(batch_size=batch_size, image_size=image_size)

    # Prepare a classifier
    classifier = classifier.SudokuOCRClassifier.prepare(load_weights=False)

    # Train the model
    classifier.train(train, validation)

    # Save the model weights
    classifier.save()

    # Evaluate the performance
    classifier.evaluate_performance(test_data=test)

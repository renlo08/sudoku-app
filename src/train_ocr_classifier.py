import pathlib
import tensorflow as tf
from model.SudokuNet import SudokuNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from Sudoku.search_sudoku_in_image import get_app_dir

# dataset from https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

OCR_DATASET_PATH = get_app_dir() / 'res' / 'data'
MODEL_WEIGHT_PATH = get_app_dir() / 'res' / 'ocr_model_weights.h5'
MOMENTUM = .9
EPOCHS = 50
BS = 32
IMG_HEIGHT = 28
IMG_WIDTH = 28

def invert_colors(img):
    """ invert the color of the image """
    return 1 - img

def create_train_generator(shuffle=True):
    """ 
    Create a generator containing the training set
    and validation set.
    """

    augmented_image_gen = ImageDataGenerator(
        preprocessing_function=invert_colors,
        rescale = 1/255.0,
        rotation_range=2,
        width_shift_range=.1,
        height_shift_range=.1,
        zoom_range=0.1,
        shear_range=2,
        brightness_range=[0.9, 1.1],
        validation_split=0.2)

    normal_image_gen = ImageDataGenerator(
        preprocessing_function=invert_colors,
        rescale = 1/255.0,
        validation_split=0.2)

    train_data_gen = augmented_image_gen.flow_from_directory(batch_size=BS,
                                                        directory=OCR_DATASET_PATH / "training_data",
                                                        color_mode="grayscale",
                                                        shuffle=shuffle,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode="categorical",
                                                        seed=65657867,
                                                        subset='training')

    val_data_gen = normal_image_gen.flow_from_directory(batch_size=BS,
                                                        directory=OCR_DATASET_PATH / "training_data",
                                                        color_mode="grayscale",
                                                        shuffle=shuffle,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode="categorical",
                                                        seed=65657867,
                                                        subset='validation')

    return train_data_gen, val_data_gen

def create_test_generator():
    """ Create a generator containing the test set """

    test_image_gen = ImageDataGenerator(
        preprocessing_function=invert_colors,
        rescale = 1/255.0)

    test_data_gen = test_image_gen.flow_from_directory(batch_size=BS,
                                                        directory=OCR_DATASET_PATH / "testing_data",
                                                        color_mode="grayscale",
                                                        shuffle=False,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode="categorical")
    
    return test_data_gen

def create_callbacks():
    #Prepare call backs
    EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_WEIGHT_PATH, monitor = 'val_loss',
                                mode = 'min',save_best_only= True, save_weights_only=True)
    return [EarlyStop_callback, checkpoint]

def setup_optimizer():
    #lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    return SGD(learning_rate=0.01, momentum=MOMENTUM)

def fit_model(train_data_gen, val_data_gen):
    """ Fit SudokuNet """
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = setup_optimizer()
    model = SudokuNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # define the callbacks
    my_callback = create_callbacks()

    if pathlib.Path.exists(MODEL_WEIGHT_PATH):
        model.load_weights(MODEL_WEIGHT_PATH)
        history = None
    else:
        # train the network
        print("[INFO] training network...")
        history = model.fit(
            train_data_gen,
            steps_per_epoch=train_data_gen.samples // BS,
            epochs=EPOCHS,
            validation_data=val_data_gen,
            validation_steps=val_data_gen.samples // BS,
            callbacks = my_callback)

    return history, model

def load_classifier_model():
    print("[INFO] loading model...")
    opt = setup_optimizer()
    model = SudokuNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.load_weights(MODEL_WEIGHT_PATH)
    return model

def evaluate_model_performance(generator, model):
    """ Evaluate model """
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(generator)
    labels = generator.classes
    print(classification_report(
        labels,
        predictions.argmax(axis=1),
        target_names=[str(x) for x in range(10)]))

def evaluate_overall_performance():
    """ Evaluate the overall performance of the model """

    model = load_classifier_model()

    print("[INFO] generate the training set...")
    train_gen, _ = create_train_generator(shuffle=False)
    evaluate_model_performance(train_gen, model)

    print("[INFO] generate the test set...")
    test_gen = create_test_generator()
    evaluate_model_performance(test_gen, model)

if __name__ == '__main__':
    import sys
    try:
        # train_gen, val_gen = create_train_generator()

        evaluate_overall_performance()
    except Exception as e:
        print(e)
        sys.stdout(e)

import pathlib

from model.SudokuNet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from Sudoku.search_sudoku_in_image import get_app_dir


MODEL_WEIGHT_PATH = get_app_dir() / 'res' / 'model_weights.h5'
print(MODEL_WEIGHT_PATH)
INIT_LR = 1e-3
EPOCHS = 10
BS = 128


def preprocess_data():
    """ Preprocess data"""
    # grab the MNIST dataset
    print("[INFO] accessing MNIST...")
    ((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()

    # add a channel (i.e., grayscale) dimension to the digits
    train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
    test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0

    # convert the labels from integers to vectors
    le = LabelBinarizer()

    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    return train_data, train_labels, test_data, test_labels, le


def fit_model(train_data, train_labels, test_data, test_labels):
    """ Fit SudokuNet """
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    if pathlib.Path.exists(MODEL_WEIGHT_PATH):
        model.load_weights(MODEL_WEIGHT_PATH)
        history = None
    else:
        # train the network
        print("[INFO] training network...")
        history = model.fit(train_data, train_labels,
                            validation_data=(test_data, test_labels),
                            batch_size=BS,
                            epochs=EPOCHS,
                            verbose=1)

        model.save_weights(MODEL_WEIGHT_PATH)

    return history, model


def load_classifier_model():
    opt = Adam(learning_rate=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.load_weights(MODEL_WEIGHT_PATH)
    return model


def evaluate_model_performance(data, labels, model, lab_encoder):
    """ Evaluate model """

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(data)
    print(classification_report(
        labels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in lab_encoder.classes_]))


if __name__ == '__main__':

    # collect training and test data
    train_data, train_labels, test_data, test_labels, label_encoder = preprocess_data()

    # train the model
    _, model = fit_model(train_data, train_labels, test_data, test_labels)

    # evaluate the model performance
    evaluate_model_performance(test_data, test_labels, model, label_encoder)

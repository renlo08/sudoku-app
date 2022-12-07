from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt


def load_dataset():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        # add a channel (i.e., grayscale) dimension to the digits
    train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
    test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

    return (train_data, train_labels), (test_data, test_labels)

def inspect_images_of_label(inspect_label, data):
    images, labels = data
    for img, label in zip(images, labels):
        if label == inspect_label:
            plt.imshow(img, interpolation='nearest')
            plt.show()


if __name__ == '__main__':
    import sys
    try:
        train, test = load_dataset()
        inspect_images_of_label(6, train)
    except Exception as e:
        print(e)
        sys.stdout(e)
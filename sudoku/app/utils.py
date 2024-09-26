import base64
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.utils.text import slugify
from tensorflow.keras import preprocessing

from app.ocr.model import classifier


def slugify_instance_name(instance, save=False, new_slug=None):
    if new_slug is not None:
        slug = new_slug
    else:
        slug = slugify(instance.name)
    Klass = instance.__class__
    qs = Klass.objects.filter(slug=slug).exclude(id=instance.id)
    if qs.exists():
        # auto generate new slug
        rand_int = random.randint(300_000, 500_000)
        slug = f"{slug}-{rand_int}"
        return slugify_instance_name(instance, save=save, new_slug=slug)
    instance.slug = slug
    if save:
        instance.save()
    return instance


def find_contours(image):
    """Find all the outline contours present in the image."""
    # Apply thresholding to create a binary image
    processed_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    # OpenCV findContour can require black background and white foreground, thus invert color.
    processed_image = cv2.bitwise_not(processed_image)

    # Find contours in the image
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def find_largest_quadrilateral_contour(contours):
    """
    Find the largest quadrilateral contour from a list of contours.

    :param contours: List of contours to search through.
    :return: The largest quadrilateral contour and its area.
    """
    largest_contour = np.array([])
    max_area = 0

    # Sort contours by area in descending order to prioritize larger contours.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in sorted_contours:
        # Calculate the contour area.
        area = cv2.contourArea(contour)

        # Approximate the contour to reduce the number of points.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the contour is the largest found so far and has four points.
        if area > max_area and len(approx) == 4:
            largest_contour = approx
            max_area = area

    return largest_contour, max_area


def reorder_contour_points(points):
    """
    Reorder the contour points to a consistent order: top-left, top-right, bottom-right, bottom-left.

    :param points: A numpy array of shape (4, 2) representing the contour points.
    :return: A numpy array of shape (4, 1, 2) with the reordered contour points.
    """
    points = points.reshape((4, 2))
    reordered_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(axis=1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    return reordered_points


def split_sudoku_cells(image):
    image = np.array(image)
    # Split the image of sudoku in 9 rows, then split each row horizontally in 9 cells.

    rows = np.vsplit(image, 9)
    cells = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for cell in cols:
            cells.append(cell)
    return cells


def warp_image(largest_contour, image):
    """
    Extract the biggest contour, reorder its points, and apply a perspective transform to obtain a top-down view.

    :param largest_contour: The largest contour found in the image.
    :param image: The original image.
    :return: The transformed image with a top-down view of the biggest contour.
    """
    # Reorder the contour points to a consistent order: top-left, top-right, bottom-right, bottom-left.
    largest_contour = reorder_contour_points(largest_contour)

    # Draw the reordered contour on the image copy.
    image_copy = image.copy()
    cv2.drawContours(image_copy, [largest_contour], -1, (0, 0, 255), 10)

    # Define the points for perspective transformation.
    pts_1 = np.float32(largest_contour)
    pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    # Get the perspective transformation matrix.
    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)

    # Apply the perspective transformation to get a top-down view.
    image_wrap = cv2.warpPerspective(image, matrix, (306, 306))

    # Convert the transformed image to grayscale.
    image_wrap = cv2.cvtColor(image_wrap, cv2.COLOR_BGR2GRAY)

    return image_wrap


black_image = np.zeros((450, 450, 3), dtype=np.uint8)


def crop_cell(cell):
    from PIL import Image
    data = np.array(cell, dtype=np.uint8)
    return data[2:32, 2:32]
    # return Image.fromarray(data[4:46, 4:46])


def prepare_cell_for_classification(cell):
    # Prepare the cell for classification
    roi = np.array(cv2.resize(cell, (32, 32)).astype("float") / 255.0)
    return np.expand_dims(preprocessing.image.img_to_array(roi), axis=0)


def get_predicted_board(classifier, cropped_cells):
    prepared_cells = np.concatenate(
        [prepare_cell_for_classification(cell) for cell in cropped_cells])

    predictions = classifier.predict(prepared_cells)
    return predictions.tolist()
    # return np.reshape(predictions, (9, 9)).tolist()


def contrasted_image(reshaped_image):
    return cv2.adaptiveThreshold(
        reshaped_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def to_image(numpy_img):
    return Image.fromarray(numpy_img, 'L')


def to_data_uri(pil_img):
    image_io = BytesIO()
    pil_img.save(image_io, "PNG")
    # return u'data:image/jpeg;base64,' + data64.decode('utf-8')
    return base64.b64encode(image_io.getvalue()).decode('utf-8')


def get_classification_model():
    """Return the prepared image classification model."""
    weights_file = 'app/ocr/model/sudoku_ocr_classifier.weights.h5'
    return classifier.SudokuOCRClassifier.prepare(
        load_weights=True, weights_file=weights_file
    )

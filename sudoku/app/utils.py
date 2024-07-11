import random

import cv2
import numpy as np
from django.utils.text import slugify

from app.models import Sudoku


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


def detect_contours(image: Sudoku):
    """
    Find all the outline contours present in the image.
    :param image: the original image before processing.
    :return: the processed image with the detected contours, the contour areas and the outline contours.
    """
    processed_image = image.process_image()
    contoured_image = image.convert_as_array()
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
    return contoured_image, contours, hierarchy


def main_outline_contours(contours):
    biggest = np.array([])
    max_area = 0

    # Reorder the contours in reverse order (by area) to speed up the research of main outline contour.
    ordered_contour = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in ordered_contour:
        # Calculate the contour area
        area = cv2.contourArea(c)
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Check if the contour area is bigger than the biggest until now
        # the contour must have only four approximated points typically indicating rectangular or square-like shapes
        # (or other four-sided polygons)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest, max_area


def reframe_outline_contours(points):
    """  """
    points = points.reshape((4, 2))
    reframe_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(axis=1)
    reframe_points[0] = points[np.argmin(add)]
    reframe_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reframe_points[1] = points[np.argmin(diff)]
    reframe_points[2] = points[np.argmax(diff)]
    return reframe_points


def split_sudoku_cells(image):
    # Split the image of sudoku in 9 rows, then split each row horizontally in 9 cells.
    rows = np.vsplit(image, 9)
    cells = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for cell in cols:
            cells.append(cell)
    return cells


def reshape_image(contours, image):
    image_copy = image.copy()
    biggest_contour, max_area = main_outline_contours(contours)
    if biggest_contour.size != 0:
        biggest_contour = reframe_outline_contours(biggest_contour)
        cv2.drawContours(image_copy, biggest_contour, -1, (0, 0, 255), 10)
        pts_1 = np.float32(biggest_contour)
        pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
        image_wrap = cv2.warpPerspective(image, matrix, (300, 300))
        image_wrap = cv2.cvtColor(image_wrap, cv2.COLOR_BGR2GRAY)
        return image_wrap
    raise ValueError("Cannot find a suitable contour in the image.")


black_image = np.zeros((450, 450, 3), dtype=np.uint8)


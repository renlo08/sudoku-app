import io
from os import path

import cv2
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.db import models

from app import utils


class ProcessedImage(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    data = models.JSONField(default=dict)  # Zone Of Interest

    def save(self, *args, **kwargs):
        if self.data:
            # Ensure that the Zone Of Interest is a python list (np.array)
            self.data = np.array(self.data).tolist()
        super().save(*args, **kwargs)


class BoardCell(models.Model):
    id = models.AutoField(primary_key=True)
    row = models.IntegerField()
    col = models.IntegerField()
    data = models.JSONField(default=dict)  # Zone Of Interest
    predicted_value = models.CharField(max_length=1, blank=True, null=True)

    def save(self, *args, **kwargs):
        if self.data:
            # Ensure that the Zone Of Interest is a python list (np.array)
            self.data = np.array(self.data).tolist()
        super().save(*args, **kwargs)


class SudokuBoard(models.Model):
    id = models.AutoField(primary_key=True)
    cells = models.ManyToManyField(BoardCell)
    grayscale_data = models.JSONField(default=dict, null=True, blank=True)
    countour_data = models.JSONField(default=dict, null=True, blank=True)
    reshaped_data = models.JSONField(default=dict, null=True, blank=True)
    contrasted_data = models.JSONField(default=dict, null=True, blank=True)


class Sudoku(models.Model):
    id = models.AutoField(primary_key=True)
    photo = models.ImageField(upload_to='sudoku_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    board = models.ForeignKey(SudokuBoard, on_delete=models.CASCADE, blank=True, null=True)

    def save(self, *args, **kwargs):
        # call the parent class's save.
        super().save(*args, **kwargs)

        # Open the image using its path and resize
        img = Image.open(self.photo.path)
        img.thumbnail(size=(300, 300))
        output = io.BytesIO()
        img.save(output, format=img.format, quality=90)
        output.seek(0)

        # get the filename from the path
        filename = path.basename(self.photo.name)

        self.photo = ContentFile(output.read(), filename)
        super().save(*args, **kwargs)

    def convert_as_array(self):
        self.photo.seek(0)

        # Open the image file
        img = Image.open(self.photo.path)

        # Convert PIL image to OpenCV format (numpy array)
        return np.array(img)

    def color_background_to_gray(self):
        cv_img = self.convert_as_array()

        # Convert image to gray scale using cv2
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return gray

    def process_image(self):
        image = self.color_background_to_gray()

        # Apply thresholding to create a binary image
        # _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        # OpenCV findContour can require black background and white foreground, thus invert color.
        processed_image = cv2.bitwise_not(processed_image)
        return processed_image

    def prepare_images(self):
        image = self.convert_as_array()
        gray_image = self.color_background_to_gray()
        image_with_contours, contours, hierarchy = utils.detect_contours(self)
        reshaped_image = utils.reshape_image(contours, image)
        contrasted_image = utils.contrasted_image(reshaped_image)
        return contrasted_image, reshaped_image, gray_image, image, image_with_contours

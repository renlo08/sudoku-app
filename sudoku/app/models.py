import io
from os import path
import typing

import cv2
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.db import models

from app import utils


class Sudoku(models.Model):
    id = models.AutoField(primary_key=True)
    photo = models.ImageField(upload_to='sudoku_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

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

    def process_board(self):
        """ Create the SudokuBoard instance and process the image """
        sudoku_board = SudokuBoard(sudoku=self)
        sudoku_board.save()

    def convert_as_array(self):
        self.photo.seek(0)

        # Open the image file
        img = Image.open(self.photo.path)

        # Convert PIL image to OpenCV format (numpy array)
        img_array = np.array(img)
        return img_array

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

class ProcessedData(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    data = models.JSONField(default=dict)  # Zone Of Interest

    def save(self, *args, **kwargs):
        if self.data:
            # Ensure that the Zone Of Interest is a python list (np.array)
            self.data = np.array(self.data).tolist()
        super().save(*args, **kwargs)

class SudokuBoard(models.Model):
    id = models.AutoField(primary_key=True)
    data = models.JSONField(default=dict, null=True, blank=True)
    grayscale_data = models.JSONField(default=dict, null=True, blank=True)
    countour_data = models.JSONField(default=dict, null=True, blank=True)
    reshaped_data = models.JSONField(default=dict, null=True, blank=True)
    contrasted_data = models.JSONField(default=dict, null=True, blank=True)
    sudoku = models.ForeignKey(Sudoku, on_delete=models.CASCADE, blank=True, null=True)
    
    def save(self, *args, **kwargs):
        if not self.data:
            self.data = self.convert_as_array().tolist()
        super().save(*args, **kwargs)
    
    @property
    def has_grayscale_data(self):
        return bool(self.grayscale_data)

    @property
    def has_countour_data(self):
        return bool(self.countour_data)
    
    @property
    def has_reshaped_data(self):
        return bool(self.reshaped_data)
    
    @property
    def has_contrasted_data(self):
        return bool(self.contrasted_data)
    
    def generate_data(self, name: str, process_func: typing.Callable):
        """ Get or create the original data """
        setattr(self, name, process_func())
        # Convert numpy array to list before saving
        setattr(self, name, getattr(self, name).tolist())
        self.save()

    def convert_as_array(self):
        self.sudoku.photo.seek(0)

        # Open the image file
        img = Image.open(self.sudoku.photo.path)

        # Convert PIL image to OpenCV format (numpy array)
        return np.array(img)
        

    def convert_to_gray(self):
        # Convert original data to gray scale
        return cv2.cvtColor(self.get_original_data(), cv2.COLOR_BGR2GRAY)

    def process_grayscale(self):
        self.generate_data('grayscale_data', self.convert_to_gray)

    def get_original_data(self):
        return np.array(self.data).astype(np.uint8)

    def get_grayscale_data(self):
        if not self.has_grayscale_data:
            self.process_grayscale()
        return np.array(self.grayscale_data)
    
    def process_image(self):
        image = self.convert_as_array()

        # Apply thresholding to create a binary image
        # _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        # OpenCV findContour can require black background and white foreground, thus invert color.
        processed_image = cv2.bitwise_not(processed_image)
        return processed_image



class BoardCell(models.Model):
    id = models.AutoField(primary_key=True)
    row = models.IntegerField()
    col = models.IntegerField()
    data = models.ForeignKey(ProcessedData, on_delete=models.CASCADE, blank=True, null=True)
    predicted_value = models.CharField(max_length=1, blank=True, null=True)
    board = models.ForeignKey(SudokuBoard, on_delete=models.CASCADE, blank=True, null=True)
    
    def save(self, *args, **kwargs):
        if self.data:
            # Ensure that the Zone Of Interest is a python list (np.array)
            self.data = np.array(self.data).tolist()
        super().save(*args, **kwargs)





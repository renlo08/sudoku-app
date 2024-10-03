import base64
import io
import itertools
import json
from os import path

import cv2
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.db import models
from tensorflow.keras import preprocessing

from app import utils


class Sudoku(models.Model):
    id = models.AutoField(primary_key=True)
    photo = models.ImageField(upload_to='sudoku_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def get_photo_name(self) -> str:
        return self.photo.name

    def save(self, *args, **kwargs):
        # Check if the instance is being created (i.e., it doesn't have an ID yet)
        if not self.id:
            # Open the image using the file object
            img = Image.open(self.photo)
            img.thumbnail(size=(300, 300))
            output = io.BytesIO()
            img.save(output, format=img.format, quality=90)
            output.seek(0)

            # Get the filename from the original file
            filename = path.basename(self.photo.name)

            # Replace the original photo with the resized image
            self.photo = ContentFile(output.read(), filename)

        # Call the parent class's save method
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


class SudokuBoard(models.Model):
    id = models.AutoField(primary_key=True)
    data = models.JSONField(default=dict, null=True, blank=True)
    grayscale_data = models.JSONField(default=dict, null=True, blank=True)
    contours = models.TextField(null=True, blank=True)
    warp_data = models.JSONField(default=dict, null=True, blank=True)
    contrasted_data = models.JSONField(default=dict, null=True, blank=True)
    sudoku = models.ForeignKey(
        Sudoku, on_delete=models.CASCADE, blank=True, null=True)

    def save(self, *args, **kwargs):
        self._convert_data_to_list()
        self._convert_contours_to_json()
        self._convert_warp_data_to_list()
        self._convert_contrasted_data_to_list()
        super().save(*args, **kwargs)

    def _convert_data_to_list(self):
        if not self.data:
            self.data = self.convert_as_array().tolist()

    def _convert_contours_to_json(self):
        if isinstance(self.contours, tuple):
            contours = [contour.tolist() for contour in self.contours]
            self.contours = json.dumps(contours)

    def _convert_warp_data_to_list(self):
        if isinstance(self.warp_data, np.ndarray):
            self.warp_data = self.warp_data.tolist()

    def _convert_contrasted_data_to_list(self):
        if isinstance(self.contrasted_data, np.ndarray):
            self.contrasted_data = self.contrasted_data.tolist()

    @property
    def has_grayscale_data(self):
        return bool(self.grayscale_data)

    @property
    def has_contour_data(self):
        return bool(self.contours)

    @property
    def has_warp_data(self):
        return bool(self.warp_data)

    @property
    def has_contrasted_data(self):
        return bool(self.contrasted_data)

    def convert_as_array(self):
        # Ensure the photo file is at the beginning
        self.sudoku.photo.seek(0)

        # Open the image file
        with Image.open(self.sudoku.photo) as img:
            # Convert PIL image to OpenCV format (numpy array)
            img_array = np.array(img)

        return img_array

    def convert_to_gray(self):
        # Convert original data to gray scale
        return cv2.cvtColor(self.get_original_data(), cv2.COLOR_BGR2GRAY)

    def draw_contours(self):
        # Find contours on the grayscale image and save them
        self.contours = utils.find_contours(self.get_grayscale_data())
        self.save()

    def warp_image(self):
        # Find the largest contour and warp the image
        largest_contour, _ = utils.find_largest_quadrilateral_contour(
            self.get_contour_data())
        if largest_contour.size != 0:
            self.warp_data = utils.warp_image(
                largest_contour, self.get_original_data())
            self.save()
            return
        raise ValueError("Cannot find a suitable contour in the image.")

    def constrast_image(self):
        self.contrasted_data = cv2.adaptiveThreshold(self.get_warp_data(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.save()

    def get_original_data(self):
        return np.array(self.data).astype(np.uint8)

    def get_grayscale_data(self):
        return cv2.cvtColor(self.get_original_data(), cv2.COLOR_BGR2GRAY).astype(np.uint8)

    def get_contour_data(self):
        if not self.has_contour_data:
            self.draw_contours()
        return tuple(np.array(contour)
                         for contour in json.loads(self.contours))

    def get_warp_data(self):
        if not self.has_warp_data:
            self.warp_image()
        return np.array(self.warp_data).astype(np.uint8)
    
    def get_contrasted_data(self):
        if not self.has_contrasted_data:
            self.constrast_image()
        return np.array(self.contrasted_data).astype(np.uint8)
    

class BoardCellQuerySet(models.QuerySet):
    def get_or_create_cells(self, board):
        cells = []
        for row, col in itertools.product(range(1, 10), range(1, 10)):
            cell, _ = self.get_or_create(row=row, col=col, board=board)
            cells.append(cell)
        # Ensure the cells are ordered by row and col
        return sorted(cells, key=lambda cell: (cell.row, cell.col))


class BoardCellManager(models.Manager):
    def get_queryset(self):
        return BoardCellQuerySet(self.model, using=self._db)

    def get_or_create_cells(self, board):
        return self.get_queryset().get_or_create_cells(board)
    

class BoardCell(models.Model):
    id = models.AutoField(primary_key=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    row = models.IntegerField()
    col = models.IntegerField()
    predicted_value = models.IntegerField(blank=True, null=True)
    solved_value = models.IntegerField(blank=True, null=True)
    board = models.ForeignKey(
        SudokuBoard, on_delete=models.CASCADE, blank=True, null=True)

    objects = BoardCellManager()
    
    @property
    def roi(self):
        return self.get_roi()
    
    @property
    def uri(self):
        image = Image.fromarray(self.roi)
        image_io = io.BytesIO()
        image.save(image_io, "PNG")
        return base64.b64encode(image_io.getvalue()).decode('utf-8')
    
    def get_roi(self):
        image = self.board.get_contrasted_data()

        # Resize the image so that its dimensions are divisible by 9
        new_height = (image.shape[0] // 9) * 9
        new_width = (image.shape[1] // 9) * 9

        image = cv2.resize(image, (new_width, new_height))
        cell_height = image.shape[0] // 9
        cell_width = image.shape[1] // 9
        # Adjust row and col to be zero-indexed
        row = self.row - 1
        col = self.col - 1
        start_row = row * cell_height
        end_row = (row + 1) * cell_height
        start_col = col * cell_width
        end_col = (col + 1) * cell_width

        # Ensure the ROI is within the image boundaries
        end_row = min(end_row, image.shape[0])
        end_col = min(end_col, image.shape[1])

        return image[start_row:end_row, start_col:end_col]
    
    def prepare_roi_for_classification(self):
        roi = np.array(cv2.resize(self.roi, (32, 32)).astype("float") / 255.0)
        return np.expand_dims(preprocessing.image.img_to_array(roi), axis=0)
    
    def get_prediction(self):
        model = utils.get_classification_model()
        cell = self.get_roi()

    


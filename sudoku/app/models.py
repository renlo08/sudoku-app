import io
from os import path

import cv2
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.db import models


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

    def denoised_background(self):
        gray_image = self.color_background_to_gray()

        # filter noice
        blurred = cv2.GaussianBlur(gray_image, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
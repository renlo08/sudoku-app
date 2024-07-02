import io
from os import path

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

        # Open the image using its path.
        img = Image.open(self.photo.path)

        if img.height > 600 or img.width > 600:
            output_size = (600, 600)
            img.thumbnail(output_size)
            output = io.BytesIO()
            img.save(output, format='PNG', quality=85)
            output.seek(0)

            # get the filename from the path
            filename = path.basename(self.photo.name)

            self.photo = ContentFile(output.read(), filename)

        # call the parent class's save.
        super().save(*args, **kwargs)
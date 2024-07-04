import io
from os import path

from PIL import Image
from django.core.files.base import ContentFile
from django.db import models
from django.urls import reverse


class Sudoku(models.Model):
    id = models.AutoField(primary_key=True)
    photo = models.ImageField(upload_to='sudoku_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def get_gray_scale_image_url(self):
        return reverse("app:grayed_image", kwargs={"pk": self.pk})
        # Convert PIL image to OpenCV format (numpy array)
        cv_img = np.array(image)

        # Convert image to gray scale using cv2
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Convert back to PIL Image from numpy array
        img_gray = Image.fromarray(gray)

        return img_gray

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

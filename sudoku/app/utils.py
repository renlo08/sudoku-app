import random

import cv2
from django.utils.text import slugify


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


def detect_contours(image, processed_image):
    """
    Find all the outline contours present in the image.
    :param image: the original image before processing.
    :param processed_image: the processed image after processing.
    :return: the processed image with the detected contours, the contour areas and the outline contours.
    """
    contoured_image = image.copy()
    contour, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contoured_image, contour, -1, (0, 255, 0), 3)
    return contoured_image, contour, hierarchy

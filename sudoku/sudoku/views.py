from django.shortcuts import render

from app.forms import ImageForm
from app.models import Image


def index(request):
    # Fetch the 3 latest images (you might want to handle case with no images or less than 3 images)
    latest_images = Image.objects.order_by('-id')[:3]

    context = {
        'form': ImageForm(),
        'latest_images': latest_images
    }
    if image_pk := request.session.get('image_selection'):
        image = Image.objects.get(pk=image_pk)
        context['image'] = image
    return render(request, 'index.html', context=context)

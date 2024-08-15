from django.shortcuts import render

from app.forms import ImageForm
from app.models import Sudoku


def index(request):
    # Fetch the 3 latest images (you might want to handle case with no images or less than 3 images)
    latest_images = Sudoku.objects.order_by('-id').all()
    form = request.session.pop('imageForm', ImageForm())
    context = {
        'form': form,
        'latest_images': latest_images
    }
    if pk := request.session.pop('imageID', None):
        image = Sudoku.objects.get(pk=pk)
        context['image'] = image
    return render(request, 'index.html', context=context)

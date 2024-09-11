from django.shortcuts import render

from app.forms import UploadForm
from app.models import Sudoku


def index(request):
    # Fetch the 3 latest images (you might want to handle case with no images or less than 3 images)
    stored_objects = Sudoku.objects.order_by('-id').all()
    form = request.session.pop('uploadForm', UploadForm())
    context = {
        'form': form,
        'objects': stored_objects
    }
    if pk := request.session.pop('pk', None):
        # context['object'] = Sudoku.objects.get(pk=pk)
        context['object'] = None
    return render(request, 'index.html', context=context)

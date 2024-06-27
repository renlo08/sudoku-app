from django.shortcuts import render

from app.forms import ImageForm
from app.models import Image


def upload_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = Image(image=request.FILES['file'])
            new_image.save()
    else:
        form = ImageForm()
    return render(request, 'process.html', context={'process_context': 'frwegfwef'})
    # return redirect('home', context={'form': form})
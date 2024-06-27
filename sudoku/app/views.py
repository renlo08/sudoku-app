from django.shortcuts import render, redirect

from app.forms import ImageForm
from app.models import Image


def upload_photo_view(request):
    if request.htmx:
        if request.method == 'POST':
            form = ImageForm(request.POST, request.FILES)
            if form.is_valid():
                new_image = Image(photo=request.FILES['photo'])
                new_image.save()
            return render(request, 'app/partials/upload.html', context={'form': form})
    return redirect('home')

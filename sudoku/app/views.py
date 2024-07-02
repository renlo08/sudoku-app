from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from app.forms import ImageForm
from app.models import Sudoku


def upload_photo_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = Sudoku(photo=request.FILES['photo'])
            new_image.save()  # Save the image so we can access it later
            request.session['image_selection'] = new_image.id

            return redirect('home')
        else:
            request.session.clear()  # remove all the data
            # Render the form with validation errors
            return render(request, 'app/partials/upload.html', context={'form': form})
    else:
        return redirect('home')


def upload_latest_view(request):
    if request.method == 'POST':
        img = get_object_or_404(Sudoku, pk=request.POST['imageSelect'])
        request.session['image_selection'] = img.id
        return redirect('home')

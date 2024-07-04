import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from django.shortcuts import render, redirect, get_object_or_404

from app.forms import ImageForm
from app.models import Sudoku


def upload_photo_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = Sudoku(photo=request.FILES['photo'])
            new_image.save()  # Save the image so we can access it later

            request.session['imageID'] = new_image.pk  # store the image PK in session

        else:
            request.session['imageForm'] = form
    return redirect('home')


def upload_latest_view(request):
    if request.method == 'POST':
        img = get_object_or_404(Sudoku, pk=request.POST['imageSelect'])

        request.session['imageID'] = img.pk  # store the image PK in session

        return redirect('home')


def gray_view(request, pk: int):
    # Retrieve the sudoku instance
    sudoku = get_object_or_404(Sudoku, pk=pk)

    # Open the image file
    img = Image.open(sudoku.photo.path)

    # Convert PIL image to OpenCV format (numpy array)
    cv_img = np.array(img)

    # Convert image to gray scale using cv2
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # create the plot
    fig = px.imshow(gray)
    fig.update_layout(width=250, height=250, margin=dict(l=10, r=10, b=10, t=10))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    gray_plt = fig.to_html()
    return render(request, 'app/partials/plot.html', context={'plot': gray_plt})

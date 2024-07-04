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
    if request.method == 'GET':
        img = get_object_or_404(Sudoku, pk=request.GET['latestImageId'])

        request.session['imageID'] = img.pk  # store the image PK in session

        return redirect('home')


def plot_image_view(request, pk):
    step = request.GET.get('processingStep')
    # Retrieve the sudoku instance
    sudoku = get_object_or_404(Sudoku, pk=pk)

    if step == 'gray':
        # Convert image to gray scale
        image = sudoku.color_background_to_gray()
        fig = px.imshow(image, color_continuous_scale=["black", "white"])
    elif step == 'denoised':
        # Convert image to gray scale
        image = sudoku.color_background_to_gray()
        fig = px.imshow(image, color_continuous_scale=["black", "white"])
    else:
        # get the basic image
        image = sudoku.convert_as_array()
        fig = px.imshow(image)

    # create the plot
    fig.update_layout(width=300, height=300, margin=dict(l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    gray_plt = fig.to_html()
    return render(request, 'app/partials/plot.html', context={'plot': gray_plt})

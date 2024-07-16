import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import render, redirect, get_object_or_404

from app.ai.classifier.model import SudokuOCRClassifier
from app.forms import ImageForm
from app.models import Sudoku, ProcessedImage
from app import utils


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
    plot_context = {}
    step = request.GET.get('step')
    # Retrieve the sudoku instance
    sudoku = get_object_or_404(Sudoku, pk=pk)

    # All image preparation happens inside Sudoku class
    reshaped_image, gray_image, image, image_with_contours = sudoku.prepare_images()

    request.session['reshaped_image'] = reshaped_image.tolist()

    fig = generate_fig(step, gray_image, image_with_contours, reshaped_image, image)

    fig.update_layout(width=300, height=300, margin=dict(l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    plot_context['plot'] = fig.to_html()
    return render(request, 'app/partials/plot.html', context=plot_context)


def generate_fig(step, gray_image, image_with_contours, reshaped_image, image):
    # Simplified conditional logic
    if step == 'gray':
        return px.imshow(gray_image, binary_string=True)
    elif step == 'find-contours':
        return px.imshow(image_with_contours)
    elif step == 'reshape':
        return px.imshow(reshaped_image)
    else:
        return px.imshow(image)


def fill_board_view(request):
    # Initialise the sudoku board and the classification model
    # model = SudokuOCRClassifier.setup_classifier('app/ai/classifier/model_weights.h5')
    model = SudokuOCRClassifier.setup_classifier('app/ai/classifier/ocr_model_weights.h5')
    raw_cells = utils.split_sudoku_cells(request.session.get('reshaped_image'))
    raw_cells = list(map(utils.crop_cell, raw_cells))
    board = utils.get_predicted_board(model, raw_cells)
    return render(request, 'app/partials/prepare-board.html', {'board': board})

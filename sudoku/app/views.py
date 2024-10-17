import cv2
import numpy as np
import plotly.express as px
from django.http import HttpResponse
from django.views.decorators.http import require_POST
from django.shortcuts import render, redirect, get_object_or_404

from app.forms import UploadForm
from app.models import BoardCell, Sudoku, SudokuBoard
from app import utils
from app.solve import solve_board

def get_started_view(request):
    form = UploadForm()
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            sudoku_obj = Sudoku(photo=request.FILES['photo'])
            sudoku_obj.save()
            render(request, 'app/get-started.html', {'form': form})
    return render(request, 'app/get-started.html', {'form': form})

def add_upload_details_view(request):
    context = {}
    if request.htmx and request.method =="POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            context['file_path'] = form.cleaned_data['photo'].name
        context['form'] = form
    return render(request, 'app/partials/upload-details.html', context=context)

def add_image_view(request):
    form = UploadForm()
    return render(request, 'app/upload-container.html', {'form': form})

def list_image_view(request):
    objects = Sudoku.objects.all()

    sudoku_list = list(objects)
    objects_with_neighbors = [
        (sudoku_list[i-1], sudoku_list[i], sudoku_list[(i+1) % len(sudoku_list)])
        for i in range(len(sudoku_list))
    ]
    return render(request, 'app/reload-container.html', {'objects_with_neighbors': objects_with_neighbors, 'first_object': objects.first()})

def get_next_button_view(request):
    return render(request, 'app/partials/next-button.html', {'object': obj})

def upload_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            sudoku_obj = Sudoku(photo=request.FILES['photo'])
            sudoku_obj.save()  # Save the image so we can access it later

            # store the image PK in session
            request.session['pk'] = sudoku_obj.pk

            # create the sudoku board instance and process the image
            sudoku_obj.process_board()

        else:
            request.session['uploadForm'] = form
    return redirect('home')


def reload_view(request):
    if request.method == 'GET':
        img = get_object_or_404(Sudoku, pk=request.GET['pk'])

        request.session['pk'] = img.pk  # store the image PK in session

    return redirect('home')


def display_original_view(request):
    board_obj = SudokuBoard.objects.get(sudoku__pk=request.session.get('pk'))
    data = board_obj.get_original_data()
    figure = px.imshow(data)
    figure.update_layout(width=300, height=300, margin=dict(
        l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    figure.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False)
    figure_html = figure.to_html()
    return render(request, 'board/partials/draw.html', context={'figure': figure_html})


def display_grayscale_view(request):
    board_obj = SudokuBoard.objects.get(sudoku__pk=request.session.get('pk'))
    data = board_obj.get_grayscale_data()
    figure = px.imshow(data, binary_string=True)
    figure.update_layout(width=300, height=300, margin=dict(
        l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    figure.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False)
    figure_html = figure.to_html()
    return render(request, 'board/partials/draw.html', context={'figure': figure_html})


def display_contour_view(request):
    board_obj = SudokuBoard.objects.get(sudoku__pk=request.session.get('pk'))
    # Get the contours from the board object
    contours = board_obj.get_contour_data()

    # Draw the contours on the original image
    data = board_obj.get_original_data()
    cv2.drawContours(data, contours, -1, (0, 255, 0), 2)

    # Create a plotly figure
    figure = px.imshow(data)
    figure.update_layout(width=300, height=300, margin=dict(
        l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    figure.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False)
    figure_html = figure.to_html()

    return render(request, 'board/partials/draw.html', context={'figure': figure_html})


def display_warp_view(request):
    try:
        board_obj = SudokuBoard.objects.get(
            sudoku__pk=request.session.get('pk'))
        data = board_obj.get_warp_data()
        figure = px.imshow(data)
        figure.update_layout(width=300, height=300, margin=dict(
            l=10, r=10, b=10, t=10), coloraxis_showscale=False)
        figure.update_xaxes(showticklabels=False).update_yaxes(
            showticklabels=False)
        figure_html = figure.to_html()
        return render(request, 'board/partials/draw.html', context={'figure': figure_html})
    except ValueError as e:
        return render(request, 'board/partials/draw.html', context={'error': str(e)})


def display_constrast_view(request):
    board_obj = SudokuBoard.objects.get(sudoku__pk=request.session.get('pk'))
    figure = px.imshow(board_obj.get_contrasted_data())
    figure.update_layout(width=300, height=300, margin=dict(
        l=10, r=10, b=10, t=10), coloraxis_showscale=False)
    figure.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False)
    figure_html = figure.to_html()
    return render(request, 'board/partials/draw.html', context={'figure': figure_html})

def extract_board_view(request, pk=None):
    if request.method == 'GET':
        # Get the board cells from the database
        board_obj = SudokuBoard.objects.get(
            sudoku__pk=request.session.get('pk'))
        cells = BoardCell.objects.get_or_create_cells(board=board_obj)
        # Prepare the model for classification
        model = utils.get_classification_model()

        # Extract the cells from the image
        model_input = np.concatenate(
            [cell.prepare_roi_for_classification() for cell in cells])

        # Predict the digits in the cells
        predictions = model.predict(model_input).tolist()

        # Save the predictions to the cells
        for cell, prediction in zip(cells, predictions):
            cell.predicted_value = prediction
            cell.save()
    return redirect('home')


def fill_board_view(request):
    # Get board image from session
    board_image_list = request.session.get('board-image')
    if not board_image_list:
        # handle error, for example:
        return HttpResponse('No board image provided!', status=400)

    # Initialise the sudoku board and the classification model
    model = utils.get_classification_model()

    # Prepare raw cells images
    raw_cells = utils.split_sudoku_cells(board_image_list)
    raw_cells = [utils.crop_cell(cell) for cell in raw_cells]

    # Convert raw cells to pil images and data URIs
    pil_images = [utils.to_image(cell) for cell in raw_cells]
    images_uri = [utils.to_data_uri(img) for img in pil_images]

    # Predict board values using the model
    board = utils.get_predicted_board(model, raw_cells)

    # Render the template
    return render(request, 'app/partials/prepare-board.html', {'board': board, 'cells_uri': images_uri})


def update_cell_view(request):
    # Extract the cell value from the GET parameters
    cell_value = request.GET.get('name')
    board = np.zeros((81, 1)).tolist()
    return render(request, 'app/partials/prepare-board.html', {'board': board})


def edit_board(request):
    # Get the board from table
    board = request.GET.get('board')
    return render(request, 'app/partials/edit-board.html', {'board': board})


@require_POST
def save_cell_view(request, pk=None):
    # Get the cell value from the request
    if request.htmx:

        cell_obj = BoardCell.objects.get(pk=pk)
        selected_value = request.POST.get(
            f"cellValue{cell_obj.col}-{cell_obj.row}")
        if selected_value == '':
            cell_obj.predicted_value = None
        else:
            cell_obj.predicted_value = int(selected_value)
        cell_obj.save()
    return render(request, 'board/partials/edit-board-cell.html', {'cell': cell_obj})


def solve_board_view(request, pk=None):
    if request.htmx:
        # Get the board from the database
        board_obj = SudokuBoard.objects.get(sudoku__pk=pk)
        cells = BoardCell.objects.get_or_create_cells(board=board_obj)
        # Join the 81 cells values as a single string
        board_string = ''.join(['0' if cell.predicted_value is None else str(
            cell.predicted_value) for cell in cells])
        board_string = solve_board(board_string)
        # Save the solved board to the cells
        for cell, value in zip(cells, board_string.values()):
            cell.solved_value = int(value)
            cell.save()
    return render(request, 'board/solve-board.html', {'cells': cells})

from django.shortcuts import render

from app.forms import UploadForm
from app.models import BoardCell, Sudoku, SudokuBoard


def index(request):
    # Fetch the 3 latest images (you might want to handle case with no images or less than 3 images)
    stored_objects = Sudoku.objects.order_by('-id').all()
    form = request.session.pop('uploadForm', UploadForm())
    context = {'form': form, 'items': stored_objects}
    if pk := request.session.get('pk', None):
        try:
            context['object'] = Sudoku.objects.get(pk=pk)
            board_obj = SudokuBoard.objects.get(sudoku__pk=pk)
            context['cells'] = BoardCell.objects.get_or_create_cells(board=board_obj)
        except Sudoku.DoesNotExist:
            context['object'] = None
    return render(request, 'index.html', context=context)

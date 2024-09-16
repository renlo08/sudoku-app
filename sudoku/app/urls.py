from django.urls import path

from app import views
from sudoku import utils

app_name = 'app'

urlpatterns = [
    path('upload/', views.upload_view, name='upload'),
    path('reload/', views.reload_view, name='reload'),
]

htmx_urlpatterns = [
    path('display-original/', views.display_original_view, name='display-original'),
    path('display-grayscale/', views.display_grayscale_view, name='display-grayscale'),
    path('<int:pk>/plot-image/', views.plot_image_view, name='plot-image'),
    path('fill-board/', views.fill_board_view, name='prepare-board'),
    path('update_cell', views.update_cell, name='update_cell'),
    path('edit/', views.edit_board, name='edit_board'),
]

urlpatterns = utils.arrange_urlpatterns(urlpatterns + htmx_urlpatterns)

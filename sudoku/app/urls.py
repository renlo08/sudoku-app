from django.urls import path

from app import views
from sudoku import utils

app_name = 'app'

urlpatterns = [
    path('upload/', views.upload_photo_view, name='upload_photo'),
    path('upload_latest/', views.upload_latest_view, name='upload_latest'),
]

htmx_urlpatterns = [
    path('gray/<int:pk>/', views.gray_view, name='plot_gray'),
]

urlpatterns = utils.arrange_urlpatterns(urlpatterns + htmx_urlpatterns)

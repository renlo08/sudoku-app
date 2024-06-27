from django.urls import path

from app import views
from sudoku import utils

app_name = 'app'

urlpatterns = [
]

htmx_urlpatterns = [
    path('upload/', views.upload_photo_view, name='upload_photo'),
]

urlpatterns = utils.arrange_urlpatterns(urlpatterns + htmx_urlpatterns)

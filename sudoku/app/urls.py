from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from app import views
from sudoku import utils

app_name = 'app'

urlpatterns = [
    path('',views.get_started_view, name='get-started'),
    path('upload/', views.upload_view, name='upload'),
    path('reload/', views.reload_view, name='reload'),
    path('<int:pk>/extract/', views.extract_board_view, name='extract-board'),
]

htmx_urlpatterns = [
    path('upload-details/', views.add_upload_details_view, name='hx-upload-details'),
    path('new-image/', views.add_image_view, name='hx-add-image'),
    path('list-images/', views.list_image_view, name='hx-list-images'),
    path('next-button/', views.get_next_button_view, name='hx-next-btn'),
    path('display-original/', views.display_original_view,
         name='hx-display-original'),
    path('display-grayscale/', views.display_grayscale_view,
         name='hx-display-grayscale'),
    path('display-contours/', views.display_contour_view,
         name='hx-display-contours'),
    path('display-warp/', views.display_warp_view, name='hx-display-warp'),
    path('display-constrast/', views.display_constrast_view,
         name='hx-display-constrast'),
    path('fill-board/', views.fill_board_view, name='prepare-board'),
    path('update_cell', views.update_cell_view, name='hx-update_cell'),
    path('<int:pk>/save-cell/', views.save_cell_view, name='hx-save-cell'),
    path('<int:pk>/solve/', views.solve_board_view, name='hx-solve-board'),
]

# Serving static files during the development phase
# TODO: Remove this in production: https://docs.djangoproject.com/en/5.1/howto/static-files/
static_jspatterns = static(
    settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns = utils.arrange_urlpatterns(
    urlpatterns + htmx_urlpatterns + static_jspatterns)

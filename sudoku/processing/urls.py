from django.urls import path

from processing import views


app_name = 'processing'

urlpatterns = [
    path('<int:pk>/', views.home_view, name='home'),
]

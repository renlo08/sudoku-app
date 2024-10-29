from django.shortcuts import render


def home_view(request, pk:int):
    return render(request, 'processing/index.html')

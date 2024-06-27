from django.shortcuts import render

from app.forms import ImageForm


def index(request):
    context = {'form': ImageForm()}
    return render(request, 'index.html', context=context)

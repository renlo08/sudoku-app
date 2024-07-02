import os

from django import forms

from app.models import Sudoku


class ImageForm(forms.ModelForm):
    class Meta:
        model = Sudoku
        fields = '__all__'
        labels = {
            'photo': 'Choisir la photo '
        }

    def clean_photo(self):
        photo = self.cleaned_data.get('photo')
        if photo:
            main, ext = os.path.splitext(photo.name)
            ext = ext.lstrip('.').lower()
            if ext not in ['png', 'jpg', 'jpeg']:
                raise forms.ValidationError(
                    'Format de fichier non supporté. Charger une photo au format .png, .jpg, .jpeg')
        return photo

import os

from django import forms

from app.models import Sudoku


class UploadForm(forms.ModelForm):
    class Meta:
        model = Sudoku
        fields = ['photo']
        widgets = {
            'photo': forms.ClearableFileInput(attrs={'accept': 'image/*', 'id': 'new-file'})
        }

    def clean_photo(self):
        photo = self.cleaned_data.get('photo')
        if photo:
            _, ext = os.path.splitext(photo.name)
            ext = ext.lstrip('.').lower()
            if ext not in ['png', 'jpg', 'jpeg']:
                raise forms.ValidationError(
                    f'Format non supporté. Formats autorisés: .png, .jpg, .jpeg')
        return photo

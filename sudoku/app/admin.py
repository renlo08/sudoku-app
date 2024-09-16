from django.utils.html import format_html
from django.contrib import admin

from app.models import Sudoku, SudokuBoard


class SudokuAdmin(admin.ModelAdmin):
    list_display = ('id', 'uploaded_at', 'preview')

    def preview(self, obj):
        return format_html('<img src="{}" width="50" height="50" />', obj.photo.url)

    preview.short_description = 'Preview'


admin.site.register(Sudoku, SudokuAdmin)


class BoardAdmin(admin.ModelAdmin):
    list_display = ('id', 'sudoku', 'has_grayscale_data', 
                    'has_contour_data', 'has_warp_data', 'has_contrasted_data')

    def preview(self, obj):
        return format_html('<img src="{}" width="50" height="50" />', obj.sudoku.photo.url)

    preview.short_description = 'Preview'

admin.site.register(SudokuBoard, BoardAdmin)
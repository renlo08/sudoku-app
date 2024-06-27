from django.utils.html import format_html
from django.contrib import admin

from app.models import Image


class ImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'uploaded_at', 'preview')

    def preview(self, obj):
        return format_html('<img src="{}" width="50" height="50" />', obj.photo.url)

    preview.short_description = 'Preview'


admin.site.register(Image, ImageAdmin)

# Generated by Django 4.2 on 2024-07-09 21:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0008_blurring_thresholding_sudoku_blur_params_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sudoku',
            name='blur_params',
        ),
        migrations.RemoveField(
            model_name='sudoku',
            name='threshold_params',
        ),
        migrations.DeleteModel(
            name='Blurring',
        ),
        migrations.DeleteModel(
            name='Thresholding',
        ),
    ]

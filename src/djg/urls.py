"""
hack6 URL Configuration
"""
from django.contrib import admin
from django.urls import path

from hack6_dj import views

urlpatterns = [
    path('predict/', views.Predict()),
    path('update/', views.Update()),
    path('list-db-contents/', views.List()),

    path('admin/', admin.site.urls),
]

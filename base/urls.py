from django.urls import path
from . import views

urlpatterns = [
    path('api/process-text/', views.process_text, name='process_text'),
] 
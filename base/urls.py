from django.urls import path
from . import views

urlpatterns = [
    path('', views.health_check, name='health_check'),
    path('api/process-text/', views.process_text, name='process_text'),
] 
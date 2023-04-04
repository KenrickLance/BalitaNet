from django.urls import path, re_path

from . import views

urlpatterns = [
    path('api/generate', views.generate, name='generate'),
    path('api/test', views.test, name='test'),
    re_path(r'.*', views.index, name='spa'),
]
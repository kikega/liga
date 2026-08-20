from django.urls import path

from ligas import views

app_name = "ligas"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("jornada/<int:jornada_id>/predecir/", views.jornada_predecir, name="jornada_predecir"),
]
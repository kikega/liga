"""Rutas de la aplicación de usuarios."""

from django.urls import path

from usuarios import views

app_name = "usuarios"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]

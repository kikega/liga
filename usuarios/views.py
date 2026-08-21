"""Vistas de autenticación: login y logout."""

from django.contrib import messages
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.shortcuts import redirect, render

from usuarios.forms import LoginForm


def login_view(request):
    """Vista de inicio de sesión con email."""
    if request.user.is_authenticated:
        return redirect("ligas:dashboard")

    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            messages.success(request, f"¡Bienvenido de nuevo, {user.nombre or user.email}!")
            next_url = request.GET.get("next") or "ligas:dashboard"
            return redirect(next_url)
    else:
        form = LoginForm(request)

    return render(request, "usuarios/login.html", {"form": form})


def logout_view(request):
    """Vista de cierre de sesión."""
    if request.method == "POST":
        auth_logout(request)
        messages.info(request, "Has cerrado sesión correctamente.")
    return redirect("usuarios:login")

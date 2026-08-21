"""Formularios de autenticación y gestión de usuarios."""

from django import forms
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm

FORM_INPUT = (
    "w-full border border-slate-300 rounded-xl px-4 py-3 bg-white text-sm "
    "shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition"
)


class LoginForm(AuthenticationForm):
    """Formulario de inicio de sesión con email y contraseña."""

    username = forms.EmailField(
        label="Correo electrónico",
        widget=forms.EmailInput(attrs={"class": FORM_INPUT, "placeholder": "tu@email.com", "autofocus": True}),
    )
    password = forms.CharField(
        label="Contraseña",
        strip=False,
        widget=forms.PasswordInput(attrs={"class": FORM_INPUT, "placeholder": "••••••••"}),
    )

    def clean(self):
        email = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if email and password:
            self.user_cache = authenticate(self.request, username=email, password=password)
            if self.user_cache is None:
                raise forms.ValidationError(
                    "Credenciales incorrectas. Comprueba tu correo y contraseña.",
                    code="invalid_login",
                )
            self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data

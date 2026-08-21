"""Registro del modelo Usuario en el panel de administración."""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from usuarios.models import Usuario


@admin.register(Usuario)
class UsuarioAdmin(BaseUserAdmin):
    list_display = ("email", "nombre", "is_staff", "is_active", "date_joined")
    list_filter = ("is_staff", "is_active")
    ordering = ("-date_joined",)
    search_fields = ("email", "nombre")

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Información personal", {"fields": ("nombre",)}),
        ("Permisos", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        ("Fechas importantes", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "nombre", "password1", "password2"),
            },
        ),
    )

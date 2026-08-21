"""Modelo de usuario personalizado con email como identificador único."""

from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)
from django.db import models
from django.utils import timezone


class UsuarioManager(BaseUserManager):
    """Manager para el modelo de Usuario que usa el email como identificador."""

    def create_user(self, email: str, password: str | None = None, **extra_fields) -> "Usuario":
        if not email:
            raise ValueError("El email es un campo obligatorio.")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email: str, password: str | None = None, **extra_fields) -> "Usuario":
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser debe tener is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser debe tener is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


class Usuario(AbstractBaseUser, PermissionsMixin):
    """Usuario del sistema que se autentica mediante email."""

    email = models.EmailField(unique=True, max_length=255, verbose_name="Correo electrónico")
    nombre = models.CharField(max_length=120, blank=True, verbose_name="Nombre completo")
    is_active = models.BooleanField(default=True, verbose_name="Activo")
    is_staff = models.BooleanField(default=False, verbose_name="Es administrador")
    date_joined = models.DateTimeField(default=timezone.now, verbose_name="Fecha de registro")

    objects = UsuarioManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    class Meta:
        verbose_name = "Usuario"
        verbose_name_plural = "Usuarios"
        ordering = ["-date_joined"]

    def __str__(self) -> str:
        return self.nombre or self.email

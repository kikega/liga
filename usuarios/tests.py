"""Pruebas del sistema de usuarios y autenticación con email."""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

Usuario = get_user_model()


class UsuarioModelTests(TestCase):
    def test_crear_usuario_con_email(self):
        usuario = Usuario.objects.create_user(
            email="kike@example.com", password="password123", nombre="Kike"
        )
        self.assertEqual(usuario.email, "kike@example.com")
        self.assertEqual(usuario.nombre, "Kike")
        self.assertTrue(usuario.is_active)
        self.assertFalse(usuario.is_staff)
        self.assertFalse(usuario.is_superuser)
        self.assertTrue(usuario.check_password("password123"))

    def test_crear_usuario_sin_email_lanza_error(self):
        with self.assertRaises(ValueError):
            Usuario.objects.create_user(email="", password="password123")

    def test_crear_superuser(self):
        admin = Usuario.objects.create_superuser(
            email="admin@example.com", password="adminpassword"
        )
        self.assertTrue(admin.is_staff)
        self.assertTrue(admin.is_superuser)
        self.assertTrue(admin.is_active)


class AutenticacionVistasTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = Usuario.objects.create_user(
            email="tester@example.com", password="secretpassword", nombre="Tester"
        )

    def test_login_exitoso(self):
        response = self.client.post(
            reverse("usuarios:login"),
            {"username": "tester@example.com", "password": "secretpassword"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_login_fallido_credenciales_invalidas(self):
        response = self.client.post(
            reverse("usuarios:login"),
            {"username": "tester@example.com", "password": "wrongpassword"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertIn("Credenciales incorrectas", response.content.decode())

    def test_logout(self):
        self.client.login(username="tester@example.com", password="secretpassword")
        response = self.client.post(reverse("usuarios:logout"))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_rutas_configuracion_requieren_login(self):
        # Usuario anónimo intenta acceder a configuración
        rutas_protegidas = [
            reverse("ligas:configuracion"),
            reverse("ligas:jornada_nueva"),
            reverse("ligas:equipos"),
            reverse("ligas:equipo_nuevo"),
        ]
        for ruta in rutas_protegidas:
            response = self.client.get(ruta)
            self.assertEqual(response.status_code, 302)
            self.assertIn("/auth/login/", response.url)

        # Usuario autenticado puede acceder
        self.client.login(username="tester@example.com", password="secretpassword")
        for ruta in rutas_protegidas:
            response = self.client.get(ruta)
            self.assertEqual(response.status_code, 200, f"Fallo al acceder a {ruta}")

import csv
import tempfile
from datetime import date
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import Client, TestCase

from ligas.ml.dixon_coles import DixonColesModel
from ligas.ml.elo import EloCalculator
from ligas.models import (
    PUNTOS_DERROTA,
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    Ausencia,
    Configuracion,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Partido,
    Participacion,
    Prediccion,
    Temporada,
)
from ligas.quiniela import GeneradorQuiniela, calcular_entropia
from ligas.services import clasificacion_por_division

Usuario = get_user_model()


def crear_division(nivel):
    return Division.objects.create(nombre=f"{nivel}ª División", nivel=nivel)


def crear_equipos(*nombres):
    return [Equipo.objects.create(nombre=n) for n in nombres]


def crear_temporada(nombre="2024-2025", activa=True):
    anio = int(nombre.split("-")[0])
    return Temporada.objects.create(
        nombre=nombre, inicio=date(anio, 8, 1), fin=date(anio + 1, 7, 31), activa=activa
    )


class PuntosTests(TestCase):
    def setUp(self):
        div = crear_division(1)
        self.temporada = crear_temporada()
        self.e1, self.e2 = crear_equipos("A", "B")
        Participacion.objects.create(temporada=self.temporada, equipo=self.e1, division=div)
        Participacion.objects.create(temporada=self.temporada, equipo=self.e2, division=div)
        jornada = Jornada.objects.create(temporada=self.temporada, numero=1)
        self.partido = Partido.objects.create(jornada=jornada, local=self.e1, visitante=self.e2)

    def test_victoria_local(self):
        self.partido.guardar_resultado(2, 0)
        self.assertEqual(self.partido.puntos_local(), PUNTOS_VICTORIA)
        self.assertEqual(self.partido.puntos_visitante(), PUNTOS_DERROTA)
        self.assertEqual(self.partido.resultado, "1")

    def test_empate(self):
        self.partido.guardar_resultado(1, 1)
        self.assertEqual(self.partido.puntos_local(), PUNTOS_EMPATE)
        self.assertEqual(self.partido.puntos_visitante(), PUNTOS_EMPATE)
        self.assertEqual(self.partido.resultado, "X")

    def test_victoria_visitante(self):
        self.partido.guardar_resultado(0, 3)
        self.assertEqual(self.partido.puntos_local(), PUNTOS_DERROTA)
        self.assertEqual(self.partido.puntos_visitante(), PUNTOS_VICTORIA)
        self.assertEqual(self.partido.resultado, "2")

    def test_no_jugado_sin_puntos(self):
        self.assertIsNone(self.partido.resultado)
        self.assertIsNone(self.partido.puntos_local())


class ClasificacionTests(TestCase):
    def setUp(self):
        div = crear_division(1)
        self.temporada = crear_temporada()
        self.e1, self.e2, self.e3, self.e4 = crear_equipos("Real", "Atletico", "Valencia", "Sevilla")
        for e in (self.e1, self.e2, self.e3, self.e4):
            Participacion.objects.create(temporada=self.temporada, equipo=e, division=div)
        j1 = Jornada.objects.create(temporada=self.temporada, numero=1)
        Partido.objects.create(jornada=j1, local=self.e1, visitante=self.e2, goles_local=2, goles_visitante=0)
        Partido.objects.create(jornada=j1, local=self.e3, visitante=self.e1, goles_local=0, goles_visitante=1)

    def test_orden_y_puntos(self):
        clasificacion = clasificacion_por_division(self.temporada, Division.objects.get(nivel=1))
        # Sevilla no ha jugado partidos pero debe figurar con 0 puntos
        self.assertEqual(len(clasificacion), 4)
        nombres = [f["nombre"] for f in clasificacion]
        self.assertIn("Sevilla", nombres)
        top = clasificacion[0]
        self.assertEqual(top["nombre"], "Real")
        self.assertEqual((top["pj"], top["pg"], top["pts"], top["gf"], top["gc"]), (2, 2, 6, 3, 0))


class EloCalculatorTests(TestCase):
    def setUp(self):
        self.calculator = EloCalculator(k_factor=20.0, home_advantage=60.0)
        self.e1, self.e2 = crear_equipos("LocalTeam", "VisitTeam")
        self.div = crear_division(1)
        self.temporada = crear_temporada()
        for e in (self.e1, self.e2):
            Participacion.objects.create(temporada=self.temporada, equipo=e, division=self.div)
        self.j1 = Jornada.objects.create(temporada=self.temporada, numero=1)

    def test_elo_updates_correctly(self):
        p = Partido.objects.create(
            jornada=self.j1, local=self.e1, visitante=self.e2, goles_local=3, goles_visitante=0
        )
        history = self.calculator.compute_match_ratings([p])
        self.assertIn(p.id, history)
        self.assertEqual(history[p.id]["local_elo"], 1500.0)
        self.assertEqual(history[p.id]["visit_elo"], 1500.0)

        delta_l, delta_v = self.calculator.calculate_update(1500.0, 1500.0, 3, 0)
        self.assertGreater(delta_l, 0)
        self.assertLess(delta_v, 0)


class DixonColesTests(TestCase):
    def test_exact_scores_and_1x2_sum_to_one(self):
        model = DixonColesModel()
        model.team_attack = {1: 0.2, 2: -0.1}
        model.team_defense = {1: -0.1, 2: 0.2}

        matrix = model.score_probability_matrix(1, 2)
        self.assertAlmostEqual(matrix.sum(), 1.0, places=4)

        p_l, p_e, p_v = model.predict_1x2_probabilities(1, 2)
        self.assertAlmostEqual(p_l + p_e + p_v, 1.0, places=4)
        self.assertGreater(p_l, p_v)

        pleno = model.predict_pleno_al_15(1, 2)
        self.assertIn(pleno["pleno_recomendado"], ("0-0", "1-0", "2-0", "2-1", "1-1", "M-0", "M-1", "0-1", "1-2"))


class QuinielaTests(TestCase):
    def test_generador_asigna_dobles_y_triples_por_entropia(self):
        # 14 partidos simulados con distintas incertidumbres
        partidos = []
        for i in range(1, 16):
            if i == 1:
                # Muy cierto (baja entropía)
                pl, pe, pv = 0.85, 0.10, 0.05
            elif i == 2:
                # Máxima incertidumbre (alta entropía)
                pl, pe, pv = 0.34, 0.33, 0.33
            elif i == 3:
                # Disputado
                pl, pe, pv = 0.40, 0.35, 0.25
            else:
                pl, pe, pv = 0.50, 0.30, 0.20

            e1 = Equipo(id=i * 2, nombre=f"Local {i}")
            e2 = Equipo(id=i * 2 + 1, nombre=f"Visitante {i}")
            partidos.append(
                {
                    "partido": None,
                    "local": e1,
                    "visitante": e2,
                    "prob_local": pl,
                    "prob_empate": pe,
                    "prob_visitante": pv,
                    "pleno_info": {"pleno_recomendado": "2-1", "marcador_probable": (2, 1), "marcador_probable_prob": 0.22},
                }
            )

        generador = GeneradorQuiniela(n_dobles=2, n_triples=1)
        self.assertEqual(generador.total_columnas, (2**2) * (3**1))  # 4 * 3 = 12
        self.assertEqual(generador.coste_total, 12 * 0.75)  # 9.00 €

        boleto = generador.generar_boleto(partidos)
        self.assertEqual(len(boleto["filas"]), 14)
        self.assertIsNotNone(boleto["pleno_al_15"])

        # El partido con máxima entropía debe ser TRIPLE
        partido_incierto = next(f for f in boleto["filas"] if f["numero"] == 2)
        self.assertEqual(partido_incierto["tipo_apuesta"], "TRIPLE")
        self.assertEqual(partido_incierto["signos_jugados"], ["1", "X", "2"])

        # El partido muy seguro debe ser FIJO
        partido_seguro = next(f for f in boleto["filas"] if f["numero"] == 1)
        self.assertEqual(partido_seguro["tipo_apuesta"], "FIJO")
        self.assertEqual(partido_seguro["signos_jugados"], ["1"])


class PipelineMlTests(TestCase):
    """Flujo completo: datos -> características -> entrenamiento -> predicción -> cierre."""

    def setUp(self):
        self.div1 = crear_division(1)
        self.div2 = crear_division(2)
        self.equipos = crear_equipos("A", "B", "C", "D", "E", "F")
        temporadas = [crear_temporada(nombre, activa=False) for nombre in ("2022-2023", "2023-2024")]
        temporadas.append(crear_temporada("2024-2025", activa=True))
        self.temporadas = temporadas

        n_equipos = len(self.equipos)
        for idx, temporada in enumerate(temporadas):
            for e in self.equipos:
                Participacion.objects.create(temporada=temporada, equipo=e, division=self.div1)
            for jornada_n in range(1, 6):
                jornada = Jornada.objects.create(temporada=temporada, numero=jornada_n)
                for i in range(n_equipos // 2):
                    local = self.equipos[i]
                    visitante = self.equipos[(i + jornada_n) % n_equipos]
                    if local.id == visitante.id:
                        continue
                    gl = (jornada_n * 2 + i + idx) % 4
                    gv = (jornada_n + i * 2 + idx) % 3
                    Partido.objects.create(
                        jornada=jornada, local=local, visitante=visitante, goles_local=gl, goles_visitante=gv
                    )

    def test_entrena_predice_y_cierra(self):
        from ligas.ml.predictor import entrenar_modelo, predecir_jornada

        temporada = Temporada.objects.get(nombre="2024-2025")
        partidos = list(
            Partido.objects.filter(goles_local__isnull=False)
            .select_related("jornada__temporada", "local", "visitante")
            .prefetch_related("ausencias__jugador")
        )
        self.assertGreaterEqual(len(partidos), 30)

        predictor, exactitud, brier, _, ruta = entrenar_modelo(
            partidos, output=str(Path(tempfile.gettempdir()) / "test_model.joblib")
        )
        self.assertIsNotNone(predictor.modelo)
        self.assertGreaterEqual(exactitud, 0.0)
        self.assertGreaterEqual(brier, 0.0)
        self.assertTrue(Path(ruta).exists())

        # Procesa los partidos de la temporada 2024-2025 como "próximos" (sin resultado).
        jornada = Jornada.objects.create(temporada=temporada, numero=6)
        for i in range(3):
            Partido.objects.create(jornada=jornada, local=self.equipos[i], visitante=self.equipos[i + 1])

        predicciones = predecir_jornada(jornada, path=str(Path(tempfile.gettempdir()) / "test_model.joblib"))
        self.assertEqual(len(predicciones), 3)
        for pred in predicciones:
            self.assertIn(pred.resultado_predicho, ("1", "X", "2"))
            self.assertAlmostEqual(pred.prob_local + pred.prob_empate + pred.prob_visitante, 1.0, places=4)
            self.assertGreaterEqual(pred.prob_local_pct, 0.0)
            self.assertLessEqual(pred.prob_local_pct, 100.0)
            self.assertIn(pred.confianza_nivel, ("ALTA", "MEDIA", "DISPUTADO"))

        # Cierre de jornada: registra resultados y re-entrena.
        for p in jornada.partidos.all():
            p.guardar_resultado(1, 0)
        call_command(
            "close_jornada",
            temporada="2024-2025",
            jornada=6,
            output=str(Path(tempfile.gettempdir()) / "test_model.joblib"),
        )
        jornada.refresh_from_db()
        self.assertTrue(jornada.cerrada)
        self.assertEqual(Prediccion.objects.filter(resultado_real="1").count(), 3)

        Path(ruta).unlink(missing_ok=True)
        Path(tempfile.gettempdir(), "test_model.joblib").unlink(missing_ok=True)


class VistasConfiguracionTests(TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")
        self.user = Usuario.objects.create_user(
            email="admin@test.com", password="adminpassword", is_staff=True
        )
        self.client.force_login(self.user)
        self.div = crear_division(1)
        self.temporada = crear_temporada("2024-2025")
        self.e1, self.e2, self.e3, self.e4 = crear_equipos("A", "B", "C", "D")
        for e in (self.e1, self.e2, self.e3, self.e4):
            Participacion.objects.create(temporada=self.temporada, equipo=e, division=self.div)
        cfg = Configuracion.cargar()
        cfg.temporada_actual = self.temporada
        cfg.save()

    def test_paginas_configuracion_renderizan(self):
        for url in ("/configuracion/", "/configuracion/jornada/nueva/", "/configuracion/equipos/", "/quiniela/"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, url)
        html = self.client.get("/configuracion/").content.decode()
        self.assertIn("Configuración", html)
        self.assertIn("Django Admin", html)

    def test_crear_jornada_y_anadir_partidos(self):
        response = self.client.post(
            "/configuracion/jornada/nueva/", {"temporada": self.temporada.id, "numero": ""}
        )
        self.assertEqual(response.status_code, 302)
        jornada = Jornada.objects.filter(temporada=self.temporada).order_by("-numero").first()
        self.assertIsNotNone(jornada)
        self.assertEqual(jornada.numero, 1)

        data = {
            "form-TOTAL_FORMS": "2",
            "form-INITIAL_FORMS": "0",
            "form-MIN_NUM_FORMS": "1",
            "form-MAX_NUM_FORMS": "1000",
            "form-0-local": self.e1.id,
            "form-0-visitante": self.e2.id,
            "form-0-fecha": "",
            "form-0-goles_local": "",
            "form-0-goles_visitante": "",
            "form-1-local": self.e3.id,
            "form-1-visitante": self.e4.id,
            "form-1-fecha": "",
            "form-1-goles_local": "",
            "form-1-goles_visitante": "",
        }
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/", data)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(jornada.partidos.count(), 2)

        partido = jornada.partidos.first()
        response = self.client.post(
            f"/configuracion/jornada/{jornada.id}/partido/{partido.id}/eliminar/"
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(jornada.partidos.count(), 1)

    def test_actualizar_resultados_conserva_fecha_existente(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=3)
        dt = parse_fecha_flexible("2026-08-15")
        p = Partido.objects.create(jornada=jornada, local=self.e1, visitante=self.e2, fecha=dt)
        data = {
            "accion": "actualizar_resultados",
            f"gl_{p.id}": "2",
            f"gv_{p.id}": "1",
        }
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/", data)
        self.assertEqual(response.status_code, 302)
        p.refresh_from_db()
        self.assertEqual(p.goles_local, 2)
        self.assertEqual(p.goles_visitante, 1)
        self.assertIsNotNone(p.fecha)
        self.assertEqual(p.fecha, dt)

    def test_partido_editar_individual(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=4)
        p = Partido.objects.create(jornada=jornada, local=self.e1, visitante=self.e2)
        response = self.client.get(f"/configuracion/jornada/{jornada.id}/partido/{p.id}/editar/")
        self.assertEqual(response.status_code, 200)

        data = {
            "local": self.e1.id,
            "visitante": self.e3.id,
            "fecha": "2026-08-16",
            "goles_local": "2",
            "goles_visitante": "0",
        }
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/partido/{p.id}/editar/", data)
        self.assertEqual(response.status_code, 302)
        p.refresh_from_db()
        self.assertEqual(p.visitante, self.e3)
        self.assertEqual(p.goles_local, 2)

    def test_reentrenar_modelo_web(self):
        response = self.client.post("/configuracion/modelo/reentrenar/")
        self.assertEqual(response.status_code, 302)

    def test_cerrar_jornada_web(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=2)
        Partido.objects.create(jornada=jornada, local=self.e1, visitante=self.e2, goles_local=1, goles_visitante=0)
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/cerrar/")
        self.assertEqual(response.status_code, 302)
        jornada.refresh_from_db()
        self.assertTrue(jornada.cerrada)

    def test_jornada_activar(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=5)
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/activar/")
        self.assertEqual(response.status_code, 302)
        config = Configuracion.cargar()
        self.assertEqual(config.proxima_jornada_id, jornada.id)

    def test_crear_jornada_desde_configuracion(self):
        response = self.client.post("/configuracion/", {
            "accion": "crear_jornada",
            "temporada": self.temporada.id,
            "numero": "6",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Jornada.objects.filter(temporada=self.temporada, numero=6).exists())

    def test_equipos_cambiar_division_rapido(self):
        div2 = Division.objects.get(nivel=2)
        response = self.client.post("/configuracion/equipos/", {
            "accion": "cambiar_division",
            "equipo_id": self.e1.id,
            "division_id": div2.id,
        })
        self.assertEqual(response.status_code, 302)
        p = Participacion.objects.get(temporada=self.temporada, equipo=self.e1)
        self.assertEqual(p.division, div2)

    def test_equipo_nuevo_con_division(self):
        div1 = Division.objects.get(nivel=1)
        response = self.client.post("/configuracion/equipos/nuevo/", {
            "nombre": "Nuevo Club",
            "division": div1.id,
            "color_primario": "#112233",
            "color_secundario": "#445566",
        })
        self.assertEqual(response.status_code, 302)
        nuevo = Equipo.objects.get(nombre="Nuevo Club")
        self.assertTrue(Participacion.objects.filter(temporada=self.temporada, equipo=nuevo, division=div1).exists())
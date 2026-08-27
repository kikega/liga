import csv
import tempfile
from datetime import date
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import Client, TestCase

from ligas.forms import parse_fecha_flexible
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
    Quiniela,
    CasillaQuiniela,
    Temporada,
)
from ligas.quiniela import GeneradorQuiniela, aplicar_predicciones_a_quiniela, calcular_entropia
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

    def test_crear_quiniela_y_casillas_oficiales(self):
        div1 = crear_division(1)
        div2 = crear_division(2)
        temporada = crear_temporada("2024-2025", activa=True)
        jornada = Jornada.objects.create(temporada=temporada, numero=1)
        equipos = [Equipo.objects.create(nombre=f"Eq_{i}") for i in range(30)]

        for i, eq in enumerate(equipos):
            Participacion.objects.create(temporada=temporada, equipo=eq, division=div1 if i < 15 else div2)

        partidos = []
        for i in range(15):
            p = Partido.objects.create(
                jornada=jornada,
                local=equipos[i * 2],
                visitante=equipos[i * 2 + 1],
            )
            partidos.append(p)

        quiniela = Quiniela.objects.create(
            temporada=temporada,
            jornada=jornada,
            numero=1,
            nombre="Quiniela Jornada 1",
            activa=True,
            n_dobles=2,
            n_triples=1,
        )

        for idx, p in enumerate(partidos, start=1):
            CasillaQuiniela.objects.create(
                quiniela=quiniela,
                posicion=idx,
                partido=p,
            )

        self.assertEqual(quiniela.total_casillas, 15)
        self.assertEqual(len(quiniela.casillas_1_14), 14)
        self.assertEqual(quiniela.casilla_15.posicion, 15)

        # Aplicar predicciones
        aplicar_predicciones_a_quiniela(quiniela)

        # Comprobar que las 15 casillas tienen signos
        casillas = list(quiniela.casillas.order_by("posicion"))
        for c in casillas[:14]:
            self.assertIn(c.tipo_apuesta, ("FIJO", "DOBLE", "TRIPLE"))
            self.assertTrue(len(c.signos_jugados) >= 1)
        self.assertIsNotNone(casillas[14].pronostico_pleno)

    def test_evaluar_aciertos_con_partidos_jugados(self):
        div1 = crear_division(1)
        temporada = crear_temporada("2024-2025", activa=True)
        jornada = Jornada.objects.create(temporada=temporada, numero=2)
        equipos = [Equipo.objects.create(nombre=f"Team_{i}") for i in range(30)]
        for eq in equipos:
            Participacion.objects.create(temporada=temporada, equipo=eq, division=div1)

        quiniela = Quiniela.objects.create(
            temporada=temporada,
            jornada=jornada,
            numero=2,
            nombre="Quiniela Jornada 2",
            activa=True,
            n_dobles=2,
            n_triples=1,
        )

        partidos = []
        for i in range(15):
            p = Partido.objects.create(
                jornada=jornada,
                local=equipos[i * 2],
                visitante=equipos[i * 2 + 1],
            )
            partidos.append(p)
            CasillaQuiniela.objects.create(
                quiniela=quiniela,
                posicion=i + 1,
                partido=p,
                signo_base="1",
                tipo_apuesta="FIJO",
                signos_jugados="1",
                pronostico_pleno="2-1" if i == 14 else None,
            )

        # Registrar resultados reales en algunos partidos
        partidos[0].guardar_resultado(2, 0)  # Signo '1' -> Acierto
        partidos[1].guardar_resultado(0, 1)  # Signo '2' -> Fallo
        partidos[14].guardar_resultado(2, 1)  # 2-1 -> Acierto pleno

        res = quiniela.evaluar_aciertos()
        self.assertEqual(res["partidos_jugados"], 3)
        self.assertEqual(res["aciertos_14"], 1)
        self.assertTrue(res["pleno_acierto"])

        c1 = CasillaQuiniela.objects.get(quiniela=quiniela, posicion=1)
        self.assertTrue(c1.acierto)
        self.assertEqual(c1.resultado_real, "1")

        c2 = CasillaQuiniela.objects.get(quiniela=quiniela, posicion=2)
        self.assertFalse(c2.acierto)
        self.assertEqual(c2.resultado_real, "2")

        c15 = CasillaQuiniela.objects.get(quiniela=quiniela, posicion=15)
        self.assertTrue(c15.acierto)
        self.assertEqual(c15.resultado_real, "2-1")

    def test_vistas_quinielas_flujo_completo(self):
        usuario = Usuario.objects.create_superuser("admin_q@test.com", "pass")
        client = Client()
        client.force_login(usuario)

        div1 = crear_division(1)
        div2 = crear_division(2)
        temporada = crear_temporada("2024-2025", activa=True)
        jornada = Jornada.objects.create(temporada=temporada, numero=3)
        equipos = [Equipo.objects.create(nombre=f"Club_{i}") for i in range(30)]
        for i, eq in enumerate(equipos):
            Participacion.objects.create(temporada=temporada, equipo=eq, division=div1 if i < 15 else div2)

        for i in range(15):
            Partido.objects.create(
                jornada=jornada,
                local=equipos[i * 2],
                visitante=equipos[i * 2 + 1],
            )

        # 1. Crear Quiniela por POST (ahora auto-asigna casillas y calcula ML de inmediato)
        resp = client.post("/configuracion/quinielas/", {
            "accion": "crear_quiniela",
            "temporada": temporada.id,
            "jornada": jornada.id,
            "numero": 3,
            "nombre": "Quiniela Oficial J3",
            "n_dobles": 2,
            "n_triples": 1,
            "activa": True,
        })
        self.assertEqual(resp.status_code, 302)
        q = Quiniela.objects.get(numero=3)
        self.assertTrue(q.activa)
        self.assertEqual(q.casillas.count(), 15)

        # Configuración global actualizada con la quiniela activa
        config = Configuracion.cargar()
        self.assertEqual(config.quiniela_actual, q)

        # 2. Re-autollenar casillas 1 a 15
        resp_auto = client.post(f"/configuracion/quiniela/{q.id}/", {"accion": "autollenar"})
        self.assertEqual(resp_auto.status_code, 302)
        self.assertEqual(q.casillas.count(), 15)

        # 3. Mover casilla 1 hacia abajo
        resp_bajar = client.post(f"/configuracion/quiniela/{q.id}/", {"accion": "bajar", "posicion": 1})
        self.assertEqual(resp_bajar.status_code, 302)

        # 4. Ver página pública de Quiniela
        resp_pub = client.get("/quiniela/")
        self.assertEqual(resp_pub.status_code, 200)
        self.assertContains(resp_pub, "Quiniela Oficial J3")
        self.assertContains(resp_pub, "Pleno al 15")

        # 5. Recalcular predicciones vía HTMX en la vista pública
        resp_htmx = client.post("/quiniela/", {"accion": "recalcular", "quiniela_id": q.id}, HTTP_HX_REQUEST="true")
        self.assertEqual(resp_htmx.status_code, 200)
        self.assertContains(resp_htmx, "Bloque Principal (14 Partidos)")

    def test_crear_quiniela_campos_automaticos(self):
        usuario = Usuario.objects.create_superuser("admin_auto@test.com", "pass")
        client = Client()
        client.force_login(usuario)

        div1 = crear_division(1)
        div2 = crear_division(2)
        temporada = crear_temporada("2024-2025", activa=True)
        jornada = Jornada.objects.create(temporada=temporada, numero=1)
        equipos = [Equipo.objects.create(nombre=f"TeamX_{i}") for i in range(30)]
        for i, eq in enumerate(equipos):
            Participacion.objects.create(temporada=temporada, equipo=eq, division=div1 if i < 15 else div2)

        for i in range(15):
            Partido.objects.create(
                jornada=jornada,
                local=equipos[i * 2],
                visitante=equipos[i * 2 + 1],
            )

        # Crear sin especificar numero ni nombre
        resp = client.post("/configuracion/quinielas/", {
            "accion": "crear_quiniela",
            "temporada": temporada.id,
            "jornada": jornada.id,
            "numero": "",
            "nombre": "",
            "n_dobles": 2,
            "n_triples": 1,
            "activa": True,
        })
        self.assertEqual(resp.status_code, 302)
        q = Quiniela.objects.get(temporada=temporada, numero=1)
        self.assertEqual(q.nombre, "Quiniela Jornada 1")
        self.assertEqual(q.casillas.count(), 15)

    def test_crear_quiniela_validacion_errores(self):
        usuario = Usuario.objects.create_superuser("admin_err@test.com", "pass")
        client = Client()
        client.force_login(usuario)

        temporada = crear_temporada("2024-2025", activa=True)

        # Enviar dobles + triples > 14
        resp = client.post("/configuracion/quinielas/", {
            "accion": "crear_quiniela",
            "temporada": temporada.id,
            "numero": "1",
            "nombre": "Quiniela Invalida",
            "n_dobles": 10,
            "n_triples": 5,
            "activa": True,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "La suma de dobles y triples no puede superar")

    def test_quiniela_selector_partidos_solo_de_la_jornada(self):
        usuario = Usuario.objects.create_superuser("admin_jornada@test.com", "pass")
        client = Client()
        client.force_login(usuario)

        div1 = crear_division(1)
        div2 = crear_division(2)
        temporada = crear_temporada("2024-2025", activa=True)
        j1 = Jornada.objects.create(temporada=temporada, numero=1)
        j2 = Jornada.objects.create(temporada=temporada, numero=2)

        equipos = [Equipo.objects.create(nombre=f"ClubE_{i}") for i in range(30)]
        for i, eq in enumerate(equipos):
            Participacion.objects.create(temporada=temporada, equipo=eq, division=div1 if i < 15 else div2)

        # Partidos de Jornada 1
        partidos_j1 = []
        for i in range(15):
            p = Partido.objects.create(jornada=j1, local=equipos[i * 2], visitante=equipos[i * 2 + 1])
            partidos_j1.append(p)

        # Partidos de Jornada 2
        partidos_j2 = []
        for i in range(15):
            p = Partido.objects.create(jornada=j2, local=equipos[i * 2 + 1], visitante=equipos[i * 2])
            partidos_j2.append(p)

        # Crear Quiniela para Jornada 2
        resp = client.post("/configuracion/quinielas/", {
            "accion": "crear_quiniela",
            "temporada": temporada.id,
            "jornada": j2.id,
            "numero": 2,
            "nombre": "Quiniela Jornada 2",
            "n_dobles": 2,
            "n_triples": 1,
            "activa": True,
        })
        self.assertEqual(resp.status_code, 302)
        q2 = Quiniela.objects.get(temporada=temporada, numero=2)

        # Verificar que las casillas auto-asignadas son SOLO de Jornada 2
        for casilla in q2.casillas.all():
            self.assertEqual(casilla.partido.jornada, j2)

        # Verificar que el selector de confeccionar muestra SOLO partidos de Jornada 2
        resp_conf = client.get(f"/configuracion/quiniela/{q2.id}/")
        self.assertEqual(resp_conf.status_code, 200)
        choices = resp_conf.context["partidos_choices"]

        choice_ids = set()
        for group, opts in choices:
            if group and isinstance(opts, list):
                for pid, _ in opts:
                    choice_ids.add(pid)

        p_j1_ids = {p.id for p in partidos_j1}
        p_j2_ids = {p.id for p in partidos_j2}

        # No debe haber ningún partido de Jornada 1
        self.assertEqual(choice_ids.intersection(p_j1_ids), set())
        # Todos los partidos del selector deben ser de Jornada 2
        self.assertTrue(choice_ids.issubset(p_j2_ids))


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
        self.div2 = crear_division(2)
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

    def test_jornada_cerrada_no_se_puede_activar(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=7, cerrada=True)
        config_antes = Configuracion.cargar()
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/activar/")
        self.assertEqual(response.status_code, 302)
        config_despues = Configuracion.cargar()
        self.assertNotEqual(config_despues.proxima_jornada_id, jornada.id)

        # En la página de configuración debe aparecer 'Editar Resultados' y no 'Activar'
        resp_get = self.client.get("/configuracion/")
        self.assertEqual(resp_get.status_code, 200)
        self.assertContains(resp_get, "Editar Resultados")
        self.assertContains(resp_get, "🔒 Cerrada")

    def test_cerrar_jornada_avanza_proxima_jornada(self):
        j1 = Jornada.objects.create(temporada=self.temporada, numero=8)
        j2 = Jornada.objects.create(temporada=self.temporada, numero=9)
        config = Configuracion.cargar()
        config.proxima_jornada = j1
        config.save()

        Partido.objects.create(jornada=j1, local=self.e1, visitante=self.e2, goles_local=2, goles_visitante=1)
        response = self.client.post(f"/configuracion/jornada/{j1.id}/cerrar/")
        self.assertEqual(response.status_code, 302)

        j1.refresh_from_db()
        self.assertTrue(j1.cerrada)
        config.refresh_from_db()
        self.assertEqual(config.proxima_jornada_id, j2.id)

    def test_jornada_partidos_actualizar_y_reentrenar(self):
        jornada = Jornada.objects.create(temporada=self.temporada, numero=10)
        p = Partido.objects.create(jornada=jornada, local=self.e1, visitante=self.e2)
        data = {
            "accion": "actualizar_y_reentrenar",
            f"gl_{p.id}": "3",
            f"gv_{p.id}": "0",
        }
        response = self.client.post(f"/configuracion/jornada/{jornada.id}/", data)
        self.assertEqual(response.status_code, 302)
        p.refresh_from_db()
        jornada.refresh_from_db()
        self.assertEqual(p.goles_local, 3)
        self.assertEqual(p.goles_visitante, 0)
        self.assertTrue(jornada.cerrada)

    def test_crear_jornada_desde_configuracion(self):
        response = self.client.post("/configuracion/", {
            "accion": "crear_jornada",
            "temporada": self.temporada.id,
            "numero": "6",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Jornada.objects.filter(temporada=self.temporada, numero=6).exists())

    def test_equipos_lista_get(self):
        response = self.client.get("/configuracion/equipos/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.e1.nombre)

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

    def test_equipos_desasignar_division_rapido(self):
        self.assertTrue(Participacion.objects.filter(temporada=self.temporada, equipo=self.e1).exists())
        response = self.client.post("/configuracion/equipos/", {
            "accion": "desasignar_division",
            "equipo_id": self.e1.id,
        })
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Participacion.objects.filter(temporada=self.temporada, equipo=self.e1).exists())

    def test_equipo_editar_desasignar_division(self):
        self.assertTrue(Participacion.objects.filter(temporada=self.temporada, equipo=self.e1).exists())
        response = self.client.post(f"/configuracion/equipos/{self.e1.id}/", {
            "nombre": self.e1.nombre,
            "division": "",
            "color_primario": "#112233",
            "color_secundario": "#445566",
        })
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Participacion.objects.filter(temporada=self.temporada, equipo=self.e1).exists())

    def test_equipo_nuevo_sin_division(self):
        response = self.client.post("/configuracion/equipos/nuevo/", {
            "nombre": "Club Sin Division",
            "division": "",
            "color_primario": "#112233",
            "color_secundario": "#445566",
        })
        self.assertEqual(response.status_code, 302)
        nuevo = Equipo.objects.get(nombre="Club Sin Division")
        self.assertFalse(Participacion.objects.filter(temporada=self.temporada, equipo=nuevo).exists())

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
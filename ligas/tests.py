import csv
import tempfile
from datetime import date
from pathlib import Path

from django.core.management import call_command
from django.test import TestCase

from ligas.models import (
    PUNTOS_DERROTA,
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    Ausencia,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Partido,
    Participacion,
    Prediccion,
    Temporada,
)
from ligas.services import clasificacion_por_division


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
        self.e1, self.e2, self.e3 = crear_equipos("Real", "Atletico", "Valencia")
        for e in (self.e1, self.e2, self.e3):
            Participacion.objects.create(temporada=self.temporada, equipo=e, division=div)
        j1 = Jornada.objects.create(temporada=self.temporada, numero=1)
        Partido.objects.create(jornada=j1, local=self.e1, visitante=self.e2, goles_local=2, goles_visitante=0)
        Partido.objects.create(jornada=j1, local=self.e3, visitante=self.e1, goles_local=0, goles_visitante=1)

    def test_orden_y_puntos(self):
        clasificacion = clasificacion_por_division(self.temporada, Division.objects.get(nivel=1))
        self.assertEqual([f["equipo_id"] for f in clasificacion], [self.e1.id, self.e3.id, self.e2.id])
        top = clasificacion[0]
        self.assertEqual((top["pj"], top["pg"], top["pts"], top["gf"], top["gc"]), (2, 2, 6, 3, 0))


class AscensosDescensosTests(TestCase):
    def setUp(self):
        self.div1 = crear_division(1)
        self.div2 = crear_division(2)
        self.t2024 = crear_temporada("2024-2025", activa=False)
        self.t2025 = crear_temporada("2025-2026", activa=True)
        self.j1 = Jornada.objects.create(temporada=self.t2024, numero=1)
        self.d1_equipos = crear_equipos("D1A", "D1B", "D1C", "D1D")
        self.d2_equipos = crear_equipos("D2A", "D2B", "D2C", "D2D")
        for e in self.d1_equipos:
            Participacion.objects.create(temporada=self.t2024, equipo=e, division=self.div1)
        for e in self.d2_equipos:
            Participacion.objects.create(temporada=self.t2024, equipo=e, division=self.div2)

    def _resultado(self, local, visitante, gl, gv):
        return Partido.objects.create(
            jornada=self.j1, local=local, visitante=visitante, goles_local=gl, goles_visitante=gv
        )

    def test_regla_3_descensos_y_3_ascensos(self):
        d1, d2, d3, d4 = self.d1_equipos
        e1, e2, e3, e4 = self.d2_equipos
        self._resultado(d1, d2, 1, 0)
        self._resultado(d3, d4, 1, 0)
        self._resultado(e1, e2, 2, 0)
        self._resultado(e3, e4, 2, 0)

        movimientos = self.t2024.aplicar_ascensos_descensos()

        descendidos = {e for e, _, tipo in movimientos if tipo == "descenso"}
        ascendidos = {e for e, _, tipo in movimientos if tipo == "ascenso"}

        self.assertEqual(len(descendidos), 3)
        self.assertEqual(len(ascendidos), 3)

        participaciones_2025 = {
            (p.equipo_id, p.division_id) for p in Participacion.objects.filter(temporada=self.t2025)
        }
        for e in descendidos:
            self.assertIn((e, self.div2.id), participaciones_2025)
        for e in ascendidos:
            self.assertIn((e, self.div1.id), participaciones_2025)


class ImportCsvTests(TestCase):
    def test_importa_partidos_jugadores_y_bajas(self):
        filas = [
            {
                "temporada": "2023-2024",
                "division": "1",
                "jornada": "1",
                "fecha": "2023-08-20",
                "local": "Betis",
                "visitante": "Sevilla",
                "goles_local": "2",
                "goles_visitante": "1",
                "jugadores_clave_local": "Isco;Fekir",
                "jugadores_clave_visitante": "Navas",
                "bajas_local": "Fekir",
                "bajas_visitante": "",
            },
            {
                "temporada": "2023-2024",
                "division": "1",
                "jornada": "2",
                "fecha": "2023-08-27",
                "local": "Sevilla",
                "visitante": "Betis",
                "goles_local": "",
                "goles_visitante": "",
                "jugadores_clave_local": "Navas",
                "jugadores_clave_visitante": "Isco;Fekir",
                "bajas_local": "",
                "bajas_visitante": "",
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            writer.writeheader()
            writer.writerows(filas)
            ruta = fh.name
        try:
            call_command("import_csv", csv=ruta)
        finally:
            Path(ruta).unlink()

        self.assertEqual(Equipo.objects.count(), 2)
        self.assertEqual(Partido.objects.filter(goles_local__isnull=False).count(), 1)
        self.assertEqual(Partido.objects.filter(goles_local__isnull=True).count(), 1)
        isco = Jugador.objects.get(nombre="Isco")
        self.assertTrue(isco.es_importante)
        fekir = Jugador.objects.get(nombre="Fekir")
        self.assertTrue(fekir.es_importante)
        self.assertEqual(Ausencia.objects.filter(jugador=fekir).count(), 1)

    def test_dry_run_no_escribe(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write("temporada,division,jornada,fecha,local,visitante,goles_local,goles_visitante\n")
            fh.write("2023-2024,1,1,2023-08-20,Betis,Sevilla,2,1\n")
            ruta = fh.name
        try:
            call_command("import_csv", csv=ruta, dry_run=True)
        finally:
            Path(ruta).unlink()
        self.assertEqual(Equipo.objects.count(), 0)
        self.assertEqual(Partido.objects.count(), 0)


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

        predictor, exactitud, _, ruta = entrenar_modelo(partidos, output=str(Path(tempfile.gettempdir()) / "test_model.joblib"))
        self.assertIsNotNone(predictor.modelo)
        self.assertGreaterEqual(exactitud, 0.0)
        self.assertTrue(Path(ruta).exists())

        # Procesa los partidos de la temporada 2024-2025 como "próximos" (sin resultado).
        jornada = Jornada.objects.create(temporada=temporada, numero=6)
        for i in range(3):
            Partido.objects.create(jornada=jornada, local=self.equipos[i], visitante=self.equipos[i + 1])

        predicciones = predecir_jornada(jornada, path=str(Path(tempfile.gettempdir()) / "test_model.joblib"))
        self.assertEqual(len(predicciones), 3)
        for pred in predicciones:
            self.assertIn(pred.resultado_predicho, ("1", "X", "2"))
            self.assertAlmostEqual(pred.prob_local + pred.prob_empate + pred.prob_visitante, 1.0, places=5)

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
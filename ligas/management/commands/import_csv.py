"""Ingesta de datos históricos desde CSV.

Acepta dos variantes de cabecera:

Variante A (jornada explícita):

    temporada,division,jornada,fecha,local,visitante,goles_local,goles_visitante,
    jugadores_clave_local,jugadores_clave_visitante,bajas_local,bajas_visitante

Variante B (jornada calculada desde la fecha, ignora ``hora``, usa
``resultado``/``quiniela`` como refuerzo):

    temporada,division,fecha,hora,local,visitante,goles_local,goles_visitante,resultado,quiniela

En la variante B el número de jornada se deduce de las fechas distintas de la
temporada (la 1ª fecha distinta = jornada 1, la siguiente = jornada 2, ...).

Columnas opcionales en ambas variantes:
- ``jornada``: si está presente se usa tal cual; si no, se calcula por fecha.
- ``goles_local`` / ``goles_visitante``: vacíos si el partido aún no se jugó.
- ``resultado`` / ``quiniela``: "1"/"X"/"2". Si faltan los goles, se genera un
  marcador sintético coherente (1-0, 0-0, 0-1); si hay goles, se valida la
  coherencia y se avisa en caso de discrepancia.
- ``jugadores_clave_local`` / ``jugadores_clave_visitante``: jugadores clave
  separados por ';' (marca ``es_importante``).
- ``bajas_local`` / ``bajas_visitante``: ausentes por lesión/sanción separados
  por ';' (crea registros de Ausencia).

Uso:

    python manage.py import_csv --csv datos/historico.csv [--dry-run]
"""

import csv
from datetime import date, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from ligas.models import (
    MOTIVO_AUSENCIA_CHOICES,
    Ausencia,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Participacion,
    Partido,
    Temporada,
)

FORMATOS_FECHA = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S.%f")


def normalizar_division(valor):
    texto = str(valor or "").strip().lower()
    if texto.startswith("1") or texto.startswith("primera"):
        return 1
    if texto.startswith("2") or texto.startswith("segunda"):
        return 2
    raise CommandError(f"División no reconocida: {valor!r} (usa '1' o '2').")


def parsear_fecha(valor):
    valor = (valor or "").strip()
    if not valor:
        return None
    fecha = None
    for formato in FORMATOS_FECHA:
        try:
            fecha = datetime.strptime(valor, formato)
            break
        except ValueError:
            continue
    if fecha is None:
        fecha = parse_datetime(valor)
    if fecha is None:
        return None
    if timezone.is_naive(fecha):
        fecha = timezone.make_aware(fecha, timezone.get_current_timezone())
    return fecha


def parsear_goles(valor):
    valor = (valor or "").strip()
    if not valor:
        return None
    return int(valor)


def normalizar_resultado(valor):
    texto = str(valor or "").strip().upper()
    if texto in ("1", "L", "LOCAL"):
        return "1"
    if texto in ("X", "E", "EMPATE"):
        return "X"
    if texto in ("2", "V", "VISITANTE"):
        return "2"
    return None


class Command(BaseCommand):
    help = "Importa partidos históricos desde un CSV (idempotente)."

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True, help="Ruta al archivo CSV.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Valida el CSV sin escribir en la base de datos.",
        )

    def handle(self, *args, **options):
        ruta = Path(options["csv"])
        if not ruta.exists():
            raise CommandError(f"No existe el archivo: {ruta}")

        with ruta.open(encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))

        if not filas:
            raise CommandError("El CSV está vacío o no tiene cabecera.")

        resumen = {
            "temporadas": 0,
            "equipos": 0,
            "partidos": 0,
            "ausencias": 0,
        }
        creados = {"temporadas": set(), "equipos": set()}
        self.avisos = []
        fecha_a_jornada = self._mapear_fechas_a_jornadas(filas)

        with transaction.atomic():
            for n, fila in enumerate(filas, start=2):
                self._n_fila = n
                try:
                    self._procesar_fila(fila, resumen, creados, fecha_a_jornada)
                except KeyError as exc:
                    raise CommandError(f"Fila {n}: falta la columna {exc}.") from exc
                except Exception as exc:
                    raise CommandError(f"Fila {n}: {exc}") from exc
            if options["dry_run"]:
                transaction.set_rollback(True)

        self.stdout.write(
            self.style.SUCCESS(
                f"Importación {'(dry-run) ' if options['dry_run'] else ''}correcta: "
                f"{resumen['temporadas']} temporada(s), {resumen['equipos']} equipo(s), "
                f"{resumen['partidos']} partido(s), {resumen['ausencias']} ausencia(s)."
            )
        )
        for aviso in self.avisos:
            self.stdout.write(self.style.WARNING(aviso))

    def _procesar_fila(self, fila, resumen, creados, fecha_a_jornada):
        temporada, es_nueva_t = self._get_temporada(fila["temporada"])
        division = Division.objects.get_or_create(
            nivel=normalizar_division(fila["division"]),
            defaults={"nombre": f"{normalizar_division(fila['division'])}ª División"},
        )[0]
        jornada, _ = Jornada.objects.get_or_create(
            temporada=temporada,
            numero=self._numero_jornada(fila, temporada, fecha_a_jornada),
        )
        local = self._get_equipo(fila["local"])
        visitante = self._get_equipo(fila["visitante"])

        Participacion.objects.get_or_create(temporada=temporada, equipo=local, defaults={"division": division})
        Participacion.objects.get_or_create(temporada=temporada, equipo=visitante, defaults={"division": division})

        if es_nueva_t:
            creados["temporadas"].add(temporada.pk)
        if local.pk not in creados["equipos"]:
            creados["equipos"].add(local.pk)
        if visitante.pk not in creados["equipos"]:
            creados["equipos"].add(visitante.pk)
        resumen["temporadas"] = len(creados["temporadas"])
        resumen["equipos"] = len(creados["equipos"])

        gl, gv, resultado_derivado = self._goles_fila(fila)
        partido, _ = Partido.objects.get_or_create(
            jornada=jornada,
            local=local,
            visitante=visitante,
            defaults={
                "goles_local": gl,
                "goles_visitante": gv,
                "fecha": parsear_fecha(fila.get("fecha")),
            },
        )
        if not partido.jugado and gl is not None and gv is not None:
            partido.goles_local = gl
            partido.goles_visitante = gv
            partido.save(update_fields=["goles_local", "goles_visitante"])
        resumen["partidos"] += 1

        resultado_csv = normalizar_resultado(fila.get("resultado") or fila.get("quiniela"))
        if resultado_csv and partido.jugado and resultado_csv != partido.resultado:
            self.avisos.append(
                f"Fila {self._n_fila}: {local} vs {visitante}: resultado del CSV "
                f"({resultado_csv}) no coincide con los goles ({partido.resultado})."
            )

        self._marcar_jugadores_clave(local, fila.get("jugadores_clave_local"))
        self._marcar_jugadores_clave(visitante, fila.get("jugadores_clave_visitante"))

        ausencias = self._procesar_ausencias(partido, local, fila.get("bajas_local"))
        ausencias += self._procesar_ausencias(partido, visitante, fila.get("bajas_visitante"))
        resumen["ausencias"] += ausencias

    def _numero_jornada(self, fila, temporada, fecha_a_jornada):
        explicito = (fila.get("jornada") or "").strip()
        if explicito:
            return int(explicito)
        fecha = parsear_fecha(fila.get("fecha"))
        if fecha is not None:
            semana = fecha.date().isocalendar()[:2]
            numero = fecha_a_jornada.get(temporada.nombre, {}).get(semana)
            if numero is not None:
                return numero
        ultima = (
            Jornada.objects.filter(temporada=temporada)
            .order_by("-numero")
            .values_list("numero", flat=True)
            .first()
        )
        return (ultima or 0) + 1

    @staticmethod
    def _mapear_fechas_a_jornadas(filas):
        """Asigna a cada semana ISO de una temporada su número de jornada.

        La jornada se juega normalmente en un mismo fin de semana, así que se
        agrupan las fechas por semana ISO (año, semana).
        """
        semanas = {}
        for fila in filas:
            nombre = (fila.get("temporada") or "").strip()
            if not nombre or (fila.get("jornada") or "").strip():
                continue
            fecha = parsear_fecha(fila.get("fecha"))
            if fecha is None:
                continue
            semana = fecha.date().isocalendar()[:2]
            semanas.setdefault(nombre, set()).add(semana)
        return {
            nombre: {semana: i for i, semana in enumerate(sorted(set_semanas), start=1)}
            for nombre, set_semanas in semanas.items()
        }

    def _goles_fila(self, fila):
        """Devuelve (goles_local, goles_visitante, resultado_derivado).

        Si no hay goles pero existe resultado/quiniela, genera un marcador
        sintético coherente para que el partido cuente como jugado.
        """
        gl = parsear_goles(fila.get("goles_local"))
        gv = parsear_goles(fila.get("goles_visitante"))
        if gl is not None and gv is not None:
            return gl, gv, None
        resultado = normalizar_resultado(fila.get("resultado") or fila.get("quiniela"))
        if resultado:
            sintetico = {"1": (1, 0), "X": (0, 0), "2": (0, 1)}[resultado]
            return sintetico[0], sintetico[1], resultado
        return None, None, None

    def _get_temporada(self, nombre):
        temporada, es_nueva = Temporada.objects.get_or_create(
            nombre=nombre,
            defaults=self._fechas_por_nombre(nombre),
        )
        return temporada, es_nueva

    @staticmethod
    def _fechas_por_nombre(nombre):
        anio = int(nombre.split("-")[0].strip())
        siguiente = anio + 1
        return {"inicio": date(anio, 8, 1), "fin": date(siguiente, 7, 31)}

    def _get_equipo(self, nombre):
        return Equipo.objects.get_or_create(nombre=nombre)[0]

    @staticmethod
    def _marcar_jugadores_clave(equipo, valor):
        for nombre in (valor or "").split(";"):
            nombre = nombre.strip()
            if not nombre:
                continue
            jugador, _ = Jugador.objects.get_or_create(
                equipo=equipo, nombre=nombre, defaults={"es_importante": True}
            )
            if not jugador.es_importante:
                jugador.es_importante = True
                jugador.save(update_fields=["es_importante"])

    @staticmethod
    def _procesar_ausencias(partido, equipo, valor):
        contador = 0
        for nombre in (valor or "").split(";"):
            nombre = nombre.strip()
            if not nombre:
                continue
            jugador = Jugador.objects.filter(equipo=equipo, nombre=nombre).first()
            if jugador is None:
                continue
            _, creado = Ausencia.objects.get_or_create(
                partido=partido,
                jugador=jugador,
                defaults={"motivo": MOTIVO_AUSENCIA_CHOICES[0][0]},
            )
            contador += int(creado)
        return contador
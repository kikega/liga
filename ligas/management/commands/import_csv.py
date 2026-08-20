"""Ingesta de datos históricos desde CSV.

Formato esperado del CSV (UTF-8, con cabecera). Los partidos pueden ir con o
sin resultado (para programar jornadas futuras basta dejar goles en blanco):

    temporada,division,jornada,fecha,local,visitante,goles_local,goles_visitante,
    jugadores_clave_local,jugadores_clave_visitante,bajas_local,bajas_visitante

Ejemplo:

    temporada,division,jornada,fecha,local,visitante,goles_local,goles_visitante,jugadores_clave_local,jugadores_clave_visitante,bajas_local,bajas_visitante
    2024-2025,1,1,2024-08-18,Real Madrid,Barcelona,3,1,Modric;Vinicius;Courtois,De Jong;Lewandowski,Tchouameni,
    2024-2025,1,1,2024-08-18,Atletico,Valencia,2,2,Griezmann;Oblak,Canos;Perez,,

Columnas:
- ``temporada``: nombre de la temporada, p. ej. "2024-2025".
- ``division``: "1" o "2" (también acepta "Primera"/"Segunda").
- ``jornada``: número de jornada (entero).
- ``fecha``: "YYYY-MM-DD" o "YYYY-MM-DD HH:MM[:SS]" (opcional).
- ``local`` / ``visitante``: nombres de los equipos.
- ``goles_local`` / ``goles_visitante``: enteros; vacíos si aún no se jugó.
- ``jugadores_clave_local`` / ``jugadores_clave_visitante``: jugadores clave
  separados por ';' (marca ``es_importante`` en los jugadores de cada equipo).
- ``bajas_local`` / ``bajas_visitante``: jugadores ausentes por lesión/sanción
  en ese partido separados por ';' (crea registros de Ausencia).

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

        with transaction.atomic():
            for n, fila in enumerate(filas, start=2):
                try:
                    self._procesar_fila(fila, resumen, creados)
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

    def _procesar_fila(self, fila, resumen, creados):
        temporada, es_nueva_t = self._get_temporada(fila["temporada"])
        division = Division.objects.get_or_create(
            nivel=normalizar_division(fila["division"]),
            defaults={"nombre": f"{normalizar_division(fila['division'])}ª División"},
        )[0]
        jornada, _ = Jornada.objects.get_or_create(
            temporada=temporada,
            numero=int(fila["jornada"]),
        )
        local = self._get_equipo(fila["local"])
        visitante = self._get_equipo(fila["visitante"])

        if es_nueva_t:
            creados["temporadas"].add(temporada.pk)
        if local.pk not in creados["equipos"]:
            creados["equipos"].add(local.pk)
        if visitante.pk not in creados["equipos"]:
            creados["equipos"].add(visitante.pk)
        resumen["temporadas"] = len(creados["temporadas"])
        resumen["equipos"] = len(creados["equipos"])

        partido, _ = Partido.objects.get_or_create(
            jornada=jornada,
            local=local,
            visitante=visitante,
            defaults={
                "goles_local": parsear_goles(fila.get("goles_local")),
                "goles_visitante": parsear_goles(fila.get("goles_visitante")),
                "fecha": parsear_fecha(fila.get("fecha")),
            },
        )
        if not partido.jugado:
            partido.goles_local = parsear_goles(fila.get("goles_local"))
            partido.goles_visitante = parsear_goles(fila.get("goles_visitante"))
            partido.save(update_fields=["goles_local", "goles_visitante"])
        resumen["partidos"] += 1

        self._marcar_jugadores_clave(local, fila.get("jugadores_clave_local"))
        self._marcar_jugadores_clave(visitante, fila.get("jugadores_clave_visitante"))

        ausencias = self._procesar_ausencias(partido, local, fila.get("bajas_local"))
        ausencias += self._procesar_ausencias(partido, visitante, fila.get("bajas_visitante"))
        resumen["ausencias"] += ausencias

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
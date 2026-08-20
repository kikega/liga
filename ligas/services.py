"""Lógica de negocio reutilizable: clasificaciones y utilidades de puntos."""

from django.db import models
from django.db.models import Case, Count, IntegerField, Q, Sum, Value, When

from ligas.models import (
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    RESULTADO_EMPATE,
    RESULTADO_LOCAL,
    RESULTADO_VISITANTE,
    Partido,
)


def clasificacion_por_division(temporada, division):
    """Clasificación general de una división (PTS, PJ, PG, PE, PP, GF, GC).

    Devuelve lista de dicts ordenada por puntos, diferencia de goles y goles a favor.
    Optimizado: 2 agregaciones (local + visitante) por división.
    """
    partidos = (
        Partido.objects.filter(
            jornada__temporada=temporada,
            goles_local__isnull=False,
            goles_visitante__isnull=False,
        )
        .select_related("local", "visitante")
        .filter(
            Q(local__participaciones__temporada=temporada, local__participaciones__division=division)
            | Q(visitante__participaciones__temporada=temporada, visitante__participaciones__division=division)
        )
    )

    puntos_local = Case(
        When(goles_local__gt=models.F("goles_visitante"), then=Value(PUNTOS_VICTORIA)),
        When(goles_local=models.F("goles_visitante"), then=Value(PUNTOS_EMPATE)),
        default=Value(0),
        output_field=IntegerField(),
    )
    puntos_visitante = Case(
        When(goles_visitante__gt=models.F("goles_local"), then=Value(PUNTOS_VICTORIA)),
        When(goles_visitante=models.F("goles_local"), then=Value(PUNTOS_EMPATE)),
        default=Value(0),
        output_field=IntegerField(),
    )

    def _local():
        return (
            partidos.values("local_id")
            .annotate(
                nombre=models.Max("local__nombre"),
                pj=Count("id"),
                pg=Count("id", filter=Q(goles_local__gt=models.F("goles_visitante"))),
                pe=Count("id", filter=Q(goles_local=models.F("goles_visitante"))),
                pp=Count("id", filter=Q(goles_local__lt=models.F("goles_visitante"))),
                gf=Sum("goles_local"),
                gc=Sum("goles_visitante"),
                pts=Sum(puntos_local),
            )
            .order_by()
        )

    def _visitante():
        return (
            partidos.values("visitante_id")
            .annotate(
                nombre=models.Max("visitante__nombre"),
                pj=Count("id"),
                pg=Count("id", filter=Q(goles_visitante__gt=models.F("goles_local"))),
                pe=Count("id", filter=Q(goles_visitante=models.F("goles_local"))),
                pp=Count("id", filter=Q(goles_visitante__lt=models.F("goles_local"))),
                gf=Sum("goles_visitante"),
                gc=Sum("goles_local"),
                pts=Sum(puntos_visitante),
            )
            .order_by()
        )

    filas = {}
    for fila in list(_local()) + list(_visitante()):
        equipo_id = fila.get("local_id") or fila.get("visitante_id")
        acc = filas.setdefault(
            equipo_id,
            {
                "equipo_id": equipo_id,
                "nombre": fila["nombre"],
                "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "pts": 0,
            },
        )
        for campo in ("pj", "pg", "pe", "pp", "gf", "gc", "pts"):
            acc[campo] += fila[campo] or 0

    return sorted(
        filas.values(),
        key=lambda f: (f["pts"], f["gf"] - f["gc"], f["gf"]),
        reverse=True,
    )

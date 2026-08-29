"""Lógica de negocio reutilizable: clasificaciones con zonas europeas/descenso y rachas recientes."""

from typing import Any, Dict, List
from django.db import models
from django.db.models import Case, Count, IntegerField, Q, Sum, Value, When

from ligas.models import (
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    Equipo,
    Partido,
)


def clasificacion_por_division(temporada, division) -> List[Dict[str, Any]]:
    """Clasificación general de una división (PTS, PJ, PG, PE, PP, GF, GC, DG, Racha, Zona).

    Devuelve lista de dicts ordenada por puntos, diferencia de goles, goles a favor y nombre.
    Incluye los últimos 5 resultados de cada equipo y la demarcación de puestos europeos/ascenso/descenso.
    """
    partidos = (
        Partido.objects.filter(
            jornada__temporada=temporada,
            goles_local__isnull=False,
            goles_visitante__isnull=False,
        )
        .select_related("local", "visitante", "jornada")
        .order_by("jornada__numero", "fecha", "id")
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
            partidos.filter(
                local__participaciones__temporada=temporada,
                local__participaciones__division=division,
            )
            .values("local_id")
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
            partidos.filter(
                visitante__participaciones__temporada=temporada,
                visitante__participaciones__division=division,
            )
            .values("visitante_id")
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

    from ligas.models import Participacion

    participaciones = (
        Participacion.objects.filter(temporada=temporada, division=division)
        .select_related("equipo")
    )
    filas = {
        p.equipo_id: {
            "equipo_id": p.equipo_id,
            "nombre": p.equipo.nombre,
            "equipo": p.equipo,
            "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "pts": 0,
            "racha": [],
        }
        for p in participaciones
    }

    for fila in list(_local()) + list(_visitante()):
        equipo_id = fila.get("local_id") or fila.get("visitante_id")
        if equipo_id not in filas:
            continue
        acc = filas[equipo_id]
        for campo in ("pj", "pg", "pe", "pp", "gf", "gc", "pts"):
            acc[campo] += fila[campo] or 0

    # Calcular racha reciente de los últimos 5 partidos jugados por equipo
    historial_partidos = {}
    for p in partidos:
        if p.local_id in filas:
            if p.goles_local > p.goles_visitante:
                res, badge = "V", "bg-emerald-500 text-white"
            elif p.goles_local == p.goles_visitante:
                res, badge = "E", "bg-amber-400 text-slate-900"
            else:
                res, badge = "D", "bg-rose-500 text-white"
            historial_partidos.setdefault(p.local_id, []).append(
                {"res": res, "badge": badge, "info": f"{p.local.nombre} {p.goles_local}-{p.goles_visitante} {p.visitante.nombre}"}
            )

        if p.visitante_id in filas:
            if p.goles_visitante > p.goles_local:
                res, badge = "V", "bg-emerald-500 text-white"
            elif p.goles_visitante == p.goles_local:
                res, badge = "E", "bg-amber-400 text-slate-900"
            else:
                res, badge = "D", "bg-rose-500 text-white"
            historial_partidos.setdefault(p.visitante_id, []).append(
                {"res": res, "badge": badge, "info": f"{p.visitante.nombre} {p.goles_visitante}-{p.goles_local} {p.local.nombre}"}
            )

    for eq_id, fila in filas.items():
        fila["racha"] = historial_partidos.get(eq_id, [])[-5:]
        fila["dg"] = fila["gf"] - fila["gc"]

    tabla_ordenada = sorted(
        filas.values(),
        key=lambda f: (f["pts"], f["dg"], f["gf"], f["nombre"]),
        reverse=True,
    )

    # Asignar demarcación de zona según posición y división
    total_equipos = len(tabla_ordenada)
    nivel = getattr(division, "nivel", 1)
    categoria = getattr(division, "categoria", "MASC")
    n_descensos = getattr(division, "n_descensos", 3)
    n_champions = getattr(division, "n_champions", 4)

    for pos, fila in enumerate(tabla_ordenada, start=1):
        fila["posicion"] = pos
        if categoria == "FEM":
            if pos <= n_champions:
                fila["zona_clase"] = "border-l-4 border-purple-500"
                fila["zona_badge"] = "bg-purple-50 text-purple-700 border-purple-200"
                fila["zona_tipo"] = "UWCL"
            elif pos > total_equipos - n_descensos:
                fila["zona_clase"] = "border-l-4 border-rose-500"
                fila["zona_badge"] = "bg-rose-50 text-rose-700 border-rose-200"
                fila["zona_tipo"] = "DESC"
            else:
                fila["zona_clase"] = "border-l-4 border-transparent"
                fila["zona_badge"] = ""
                fila["zona_tipo"] = ""
        elif nivel == 1:
            if pos <= 4:
                fila["zona_clase"] = "border-l-4 border-blue-500"
                fila["zona_badge"] = "bg-blue-50 text-blue-700 border-blue-200"
                fila["zona_tipo"] = "UCL"
            elif pos == 5:
                fila["zona_clase"] = "border-l-4 border-amber-500"
                fila["zona_badge"] = "bg-amber-50 text-amber-700 border-amber-200"
                fila["zona_tipo"] = "UEL"
            elif pos == 6:
                fila["zona_clase"] = "border-l-4 border-emerald-500"
                fila["zona_badge"] = "bg-emerald-50 text-emerald-700 border-emerald-200"
                fila["zona_tipo"] = "UECL"
            elif pos > total_equipos - n_descensos:
                fila["zona_clase"] = "border-l-4 border-rose-500"
                fila["zona_badge"] = "bg-rose-50 text-rose-700 border-rose-200"
                fila["zona_tipo"] = "DESC"
            else:
                fila["zona_clase"] = "border-l-4 border-transparent"
                fila["zona_badge"] = ""
                fila["zona_tipo"] = ""
        else:
            if pos <= 2:
                fila["zona_clase"] = "border-l-4 border-emerald-500"
                fila["zona_badge"] = "bg-emerald-50 text-emerald-700 border-emerald-200"
                fila["zona_tipo"] = "ASC"
            elif pos <= 6:
                fila["zona_clase"] = "border-l-4 border-blue-500"
                fila["zona_badge"] = "bg-blue-50 text-blue-700 border-blue-200"
                fila["zona_tipo"] = "PLAYOFF"
            elif pos > total_equipos - n_descensos:
                fila["zona_clase"] = "border-l-4 border-rose-500"
                fila["zona_badge"] = "bg-rose-50 text-rose-700 border-rose-200"
                fila["zona_tipo"] = "DESC"
            else:
                fila["zona_clase"] = "border-l-4 border-transparent"
                fila["zona_badge"] = ""
                fila["zona_tipo"] = ""

    return tabla_ordenada

"""Vistas: dashboard, clasificación, resultados y predicción HTMX."""

from django.shortcuts import get_object_or_404, render

from ligas.ml.predictor import predecir_jornada
from ligas.models import Division, Jornada, Temporada
from ligas.services import clasificacion_por_division


def _temporada_activa():
    return (
        Temporada.objects.filter(activa=True).order_by("-inicio").first()
        or Temporada.objects.order_by("-inicio").first()
    )


def dashboard(request):
    temporada = _temporada_activa()
    nivel = request.GET.get("division") or "1"
    try:
        division = Division.objects.get(nivel=int(nivel))
    except (Division.DoesNotExist, ValueError):
        division = Division.objects.order_by("nivel").first()

    contexto = {
        "temporada": temporada,
        "divisiones": Division.objects.order_by("nivel"),
        "division": division,
    }

    if temporada is None:
        return render(request, "ligas/dashboard.html", contexto)

    contexto["clasificacion"] = clasificacion_por_division(temporada, division) if division else []

    jornadas = list(
        Jornada.objects.filter(temporada=temporada)
        .prefetch_related("partidos__local", "partidos__visitante", "partidos__prediccion")
        .order_by("numero")
    )

    ultima_jugada = next(
        (j for j in reversed(jornadas) if any(p.jugado for p in j.partidos.all())), None
    )
    proxima = next(
        (
            j
            for j in jornadas
            if not j.cerrada and any(not p.jugado for p in j.partidos.all())
        ),
        None,
    )

    contexto["ultima_jornada"] = ultima_jugada
    contexto["proxima_jornada"] = proxima
    return render(request, "ligas/dashboard.html", contexto)


def jornada_predecir(request, jornada_id):
    """Disparador HTMX: predice la jornada y devuelve el partial con las probabilidades."""
    jornada = get_object_or_404(
        Jornada.objects.prefetch_related("partidos__local", "partidos__visitante"),
        pk=jornada_id,
    )
    try:
        predicciones = predecir_jornada(jornada)
    except FileNotFoundError as exc:
        contexto = {
            "error": "El modelo aún no está entrenado. Ejecuta `python manage.py train_model`.",
        }
        return render(request, "ligas/_predicciones.html", contexto, status=409)
    except ValueError as exc:
        return render(request, "ligas/_predicciones.html", {"error": str(exc)}, status=409)

    if not predicciones:
        contexto = {"error": "No hay partidos pendientes para predecir en esta jornada."}
        return render(request, "ligas/_predicciones.html", contexto)

    return render(request, "ligas/_predicciones.html", {"predicciones": predicciones})
"""Vistas: dashboard, clasificación, resultados, predicciones y generador de Quiniela con HTMX."""

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render

from ligas.ml.dixon_coles import DixonColesModel
from ligas.ml.predictor import Predictor, predecir_jornada
from ligas.models import (
    Configuracion,
    Division,
    Jornada,
    Prediccion,
    Quiniela,
    Temporada,
)
from ligas.quiniela import GeneradorQuiniela, aplicar_predicciones_a_quiniela
from ligas.services import clasificacion_por_division


def _temporada_activa():
    return (
        Temporada.objects.filter(activa=True).order_by("-inicio").first()
        or Temporada.objects.order_by("-inicio").first()
    )


def _division_desde_parametro(request):
    div_param = (request.GET.get("division") or "").strip()
    if div_param:
        if div_param.isdigit():
            div_por_pk = Division.objects.filter(pk=int(div_param)).first()
            if div_por_pk:
                return div_por_pk
            div_por_nivel = Division.objects.filter(nivel=int(div_param)).order_by("categoria", "nivel").first()
            if div_por_nivel:
                return div_por_nivel
        div_por_codigo = Division.objects.filter(codigo__iexact=div_param).first()
        if div_por_codigo:
            return div_por_codigo
        div_por_nombre = Division.objects.filter(nombre__icontains=div_param).first()
        if div_por_nombre:
            return div_por_nombre
    return Division.objects.order_by("categoria", "nivel").first()


def dashboard(request):
    config = Configuracion.cargar()
    temporadas = list(Temporada.objects.all())

    if not temporadas:
        return render(
            request,
            "ligas/dashboard.html",
            {"config": config, "temporadas": [], "divisiones": Division.objects.order_by("categoria", "nivel")},
        )

    temporada_id = request.GET.get("temporada")
    if temporada_id:
        temporada = get_object_or_404(Temporada, pk=temporada_id)
    else:
        temporada = config.temporada_actual or _temporada_activa() or temporadas[0]

    division = _division_desde_parametro(request)

    jornadas = list(
        Jornada.objects.filter(temporada=temporada)
        .prefetch_related(
            "partidos__local__participaciones",
            "partidos__visitante__participaciones",
            "partidos__prediccion",
        )
        .order_by("numero")
    )

    jornada_num = request.GET.get("jornada")
    if jornada_num:
        jornada_sel = next((j for j in jornadas if j.numero == int(jornada_num)), None)
    else:
        jornada_sel = next(
            (j for j in reversed(jornadas) if any(p.jugado for p in j.partidos.all())), None
        )

    grupos_resultados = []
    if jornada_sel is not None:
        for div in Division.objects.order_by("categoria", "nivel"):
            partidos_div = [
                p for p in jornada_sel.partidos.all() if p.division_temporada_id == div.id
            ]
            if partidos_div:
                grupos_resultados.append({"division": div, "partidos": partidos_div})

    proxima = None
    if config.proxima_jornada and config.proxima_jornada.temporada_id == temporada.id:
        proxima = config.proxima_jornada
    else:
        proxima = next(
            (j for j in jornadas if not j.cerrada and any(not p.jugado for p in j.partidos.all())),
            None,
        )

    grupos_proxima_jornada = []
    if proxima is not None:
        proxima_partidos = list(
            proxima.partidos.select_related("local", "visitante").prefetch_related(
                "local__participaciones", "visitante__participaciones"
            )
        )
        for div in Division.objects.order_by("categoria", "nivel"):
            partidos_div = [
                p for p in proxima_partidos if p.division_temporada_id == div.id
            ]
            if partidos_div:
                grupos_proxima_jornada.append({"division": div, "partidos": partidos_div})

    contexto = {
        "config": config,
        "temporadas": temporadas,
        "temporada": temporada,
        "divisiones": Division.objects.order_by("categoria", "nivel"),
        "division": division,
        "jornadas": jornadas,
        "jornada_sel": jornada_sel,
        "grupos_resultados": grupos_resultados,
        "proxima_jornada": proxima,
        "grupos_proxima_jornada": grupos_proxima_jornada,
        "clasificacion": clasificacion_por_division(temporada, division) if division else [],
    }
    return render(request, "ligas/dashboard.html", contexto)


def predicciones(request):
    """Página de predicciones de la próxima jornada configurada."""
    config = Configuracion.cargar()
    temporada = config.temporada_actual or _temporada_activa()
    jornada = config.proxima_jornada
    if jornada is None and temporada is not None:
        jornada = (
            Jornada.objects.filter(temporada=temporada)
            .exclude(cerrada=True)
            .order_by("numero")
            .first()
        )

    predicciones = []
    grupos_resultados = []
    if jornada:
        jornada = Jornada.objects.prefetch_related(
            "partidos__local__participaciones",
            "partidos__visitante__participaciones",
            "partidos__prediccion",
        ).get(pk=jornada.pk)
        predicciones = list(
            Prediccion.objects.filter(partido__jornada=jornada).select_related(
                "partido__local", "partido__visitante"
            )
        )
        for div in Division.objects.order_by("categoria", "nivel"):
            partidos_div = [
                p for p in jornada.partidos.all() if p.division_temporada_id == div.id
            ]
            if partidos_div:
                grupos_resultados.append({"division": div, "partidos": partidos_div})

    contexto = {
        "config": config,
        "temporada": temporada,
        "jornada": jornada,
        "grupos_resultados": grupos_resultados,
        "predicciones": predicciones,
    }
    return render(request, "ligas/predicciones.html", contexto)


@login_required
def jornada_predecir(request, jornada_id):
    """Disparador HTMX: predice la jornada y devuelve el partial con las probabilidades."""
    jornada = get_object_or_404(
        Jornada.objects.prefetch_related("partidos__local", "partidos__visitante"),
        pk=jornada_id,
    )
    try:
        predicciones_lista = predecir_jornada(jornada)
    except FileNotFoundError:
        contexto = {
            "error": "El modelo aún no está entrenado. Ejecuta `python manage.py train_model`.",
        }
        return render(request, "ligas/_predicciones.html", contexto, status=409)
    except ValueError as exc:
        return render(request, "ligas/_predicciones.html", {"error": str(exc)}, status=409)

    if not predicciones_lista:
        contexto = {"error": "No hay partidos pendientes para predecir en esta jornada."}
        return render(request, "ligas/_predicciones.html", contexto)

    return render(request, "ligas/_predicciones.html", {"predicciones": predicciones_lista})


def quiniela_view(request):
    """Simulador y generador inteligente de boletos para la Quiniela (1X2 + Pleno al 15)."""
    config = Configuracion.cargar()
    temporada = config.temporada_actual or _temporada_activa()

    # Obtener todas las quinielas de la temporada
    todas_quinielas = list(
        Quiniela.objects.filter(temporada=temporada).order_by("-numero")
        if temporada
        else Quiniela.objects.all().order_by("-numero")
    )

    # Identificar la quiniela solicitada
    quiniela_id_param = request.GET.get("quiniela_id")
    quiniela = None
    if quiniela_id_param and quiniela_id_param.isdigit():
        quiniela = Quiniela.objects.filter(pk=int(quiniela_id_param)).first()

    if not quiniela:
        quiniela = (
            config.quiniela_actual
            or Quiniela.objects.filter(activa=True).first()
            or (todas_quinielas[0] if todas_quinielas else None)
        )

    n_dobles = int(request.GET.get("dobles", quiniela.n_dobles if quiniela else 2))
    n_triples = int(request.GET.get("triples", quiniela.n_triples if quiniela else 1))

    jornada = quiniela.jornada if quiniela else config.proxima_jornada

    # Recálculo de predicciones bajo demanda (vía POST o HTMX)
    if request.method == "POST" and request.POST.get("accion") == "recalcular" and quiniela:
        aplicar_predicciones_a_quiniela(quiniela)

    if quiniela and quiniela.casillas.exists():
        generador = GeneradorQuiniela(n_dobles=n_dobles, n_triples=n_triples)
        boleto = generador.generar_desde_quiniela(quiniela)
    else:
        # Fallback si aún no hay quinielas creadas con casillas
        partidos_data = []
        if jornada:
            partidos_qs = list(
                jornada.partidos.select_related("local", "visitante", "prediccion").order_by("id")
            )
            dixon_coles = None
            try:
                predictor = Predictor.cargar()
                dixon_coles = predictor.dixon_coles
            except Exception:
                dixon_coles = None

            for p in partidos_qs:
                if p.goles_local is not None and p.goles_visitante is not None:
                    prob_l = 1.0 if p.goles_local > p.goles_visitante else 0.0
                    prob_e = 1.0 if p.goles_local == p.goles_visitante else 0.0
                    prob_v = 1.0 if p.goles_local < p.goles_visitante else 0.0
                    marcador_str = f"{p.goles_local if p.goles_local < 3 else 'M'}-{p.goles_visitante if p.goles_visitante < 3 else 'M'}"
                    pleno_info = {
                        "cat_local": {"0": 1.0 if p.goles_local == 0 else 0.0, "1": 1.0 if p.goles_local == 1 else 0.0, "2": 1.0 if p.goles_local == 2 else 0.0, "M": 1.0 if p.goles_local >= 3 else 0.0},
                        "cat_visit": {"0": 1.0 if p.goles_visitante == 0 else 0.0, "1": 1.0 if p.goles_visitante == 1 else 0.0, "2": 1.0 if p.goles_visitante == 2 else 0.0, "M": 1.0 if p.goles_visitante >= 3 else 0.0},
                        "marcador_probable": (p.goles_local, p.goles_visitante),
                        "marcador_probable_prob": 1.0,
                        "pleno_recomendado": marcador_str,
                    }
                else:
                    pred = getattr(p, "prediccion", None)
                    prob_l = pred.prob_local if pred else 0.45
                    prob_e = pred.prob_empate if pred else 0.28
                    prob_v = pred.prob_visitante if pred else 0.27

                    pleno_info = None
                    if dixon_coles is not None:
                        try:
                            pleno_info = dixon_coles.predict_pleno_al_15(p.local_id, p.visitante_id)
                        except Exception:
                            pleno_info = None

                partidos_data.append(
                    {
                        "partido": p,
                        "local": p.local,
                        "visitante": p.visitante,
                        "prob_local": prob_l,
                        "prob_empate": prob_e,
                        "prob_visitante": prob_v,
                        "pleno_info": pleno_info,
                    }
                )

        generador = GeneradorQuiniela(n_dobles=n_dobles, n_triples=n_triples)
        boleto = generador.generar_boleto(partidos_data)

    contexto = {
        "config": config,
        "temporada": temporada,
        "jornada": jornada,
        "quiniela": quiniela,
        "todas_quinielas": todas_quinielas,
        "boleto": boleto,
        "n_dobles": n_dobles,
        "n_triples": n_triples,
    }

    if request.headers.get("HX-Request") and (request.GET.get("partial") == "boleto" or request.POST.get("accion") == "recalcular"):
        return render(request, "ligas/_boleto_quiniela.html", contexto)

    return render(request, "ligas/quiniela.html", contexto)
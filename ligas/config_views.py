"""Vistas de configuración protegidas: temporada/jornada, equipos, jugadores clave y re-entrenamiento ML."""

from django import forms
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from ligas.forms import (
    ConfiguracionForm,
    EquipoForm,
    JornadaForm,
    JugadorForm,
    PartidoForm,
    PartidoRowForm,
    QuinielaForm,
    obtener_partidos_choices,
    parse_fecha_flexible,
)
from ligas.ml.predictor import entrenar_modelo
from ligas.models import (
    CasillaQuiniela,
    Configuracion,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Participacion,
    Partido,
    Quiniela,
    Temporada,
)
from ligas.quiniela import GeneradorQuiniela, aplicar_predicciones_a_quiniela

PartidoFormSet = forms.formset_factory(PartidoRowForm, extra=4, min_num=1)


@login_required
def configuracion(request):
    """Configuración general: temporada actual, próxima jornada y gestión de modelos."""
    config = Configuracion.cargar()

    if request.method == "POST" and request.POST.get("accion") == "nueva_temporada":
        temporada, creada, movimientos = Temporada.iniciar_siguiente()
        config.temporada_actual = temporada
        config.proxima_jornada = None
        config.save()
        mensaje = (
            f"Temporada {temporada} creada."
            if creada
            else f"La temporada {temporada} ya existía y se ha marcado como actual."
        )
        if movimientos:
            n_asc = sum(1 for _, _, tipo in movimientos if tipo == "ascenso")
            n_desc = sum(1 for _, _, tipo in movimientos if tipo == "descenso")
            mensaje += f" Ascensos aplicados: {n_asc}, descensos: {n_desc}."
        messages.success(request, mensaje)
        return redirect("ligas:configuracion")

    if request.method == "POST" and request.POST.get("accion") == "crear_jornada":
        jornada_form = JornadaForm(request.POST)
        if jornada_form.is_valid():
            nueva_j = jornada_form.save()
            messages.success(request, f"Jornada {nueva_j.numero} creada con éxito. Añade los partidos de 1ª y 2ª División.")
            return redirect("ligas:jornada_partidos_add", jornada_id=nueva_j.id)
    else:
        jornada_form = JornadaForm(initial={"temporada": config.temporada_actual})

    if request.method == "POST":
        form = ConfiguracionForm(request.POST, instance=config)
        if form.is_valid():
            form.save()
            messages.success(request, "Configuración guardada.")
            return redirect("ligas:configuracion")
    else:
        form = ConfiguracionForm(instance=config)

    temporada = config.temporada_actual
    jornadas = (
        Jornada.objects.filter(temporada=temporada)
        .annotate(
            total_partidos=Count("partidos", distinct=True),
            partidos_jugados=Count(
                "partidos",
                filter=Q(partidos__goles_local__isnull=False, partidos__goles_visitante__isnull=False),
                distinct=True,
            ),
            partidos_div1=Count(
                "partidos",
                filter=Q(partidos__local__participaciones__temporada=temporada, partidos__local__participaciones__division__nivel=1),
                distinct=True,
            ),
            partidos_div2=Count(
                "partidos",
                filter=Q(partidos__local__participaciones__temporada=temporada, partidos__local__participaciones__division__nivel=2),
                distinct=True,
            ),
        )
        .order_by("numero")
        if temporada
        else []
    )

    total_partidos_historicos = Partido.objects.filter(goles_local__isnull=False).count()

    return render(
        request,
        "ligas/configuracion.html",
        {
            "form": form,
            "jornada_form": jornada_form,
            "config": config,
            "temporada": temporada,
            "jornadas": jornadas,
            "total_partidos_historicos": total_partidos_historicos,
        },
    )


@login_required
def jornada_activar_view(request, jornada_id):
    """Establece rápidamente una jornada como la próxima activa para predicciones y Quiniela."""
    jornada = get_object_or_404(Jornada, pk=jornada_id)
    if request.method == "POST":
        config = Configuracion.cargar()
        config.temporada_actual = jornada.temporada
        config.proxima_jornada = jornada
        config.save()
        messages.success(request, f"¡Jornada {jornada.numero} ({jornada.temporada}) establecida como Próxima Jornada Activa!")
    return redirect("ligas:configuracion")


@login_required
def reentrenar_modelo_view(request):
    """Disparador web para re-entrenar el pipeline ML completo con un solo clic."""
    if request.method == "POST":
        partidos = list(
            Partido.objects.filter(goles_local__isnull=False)
            .select_related("jornada__temporada", "local", "visitante")
            .prefetch_related("ausencias__jugador")
        )
        if len(partidos) < 15:
            messages.warning(request, "Se necesitan al menos 15 partidos jugados para entrenar el modelo.")
            return redirect("ligas:configuracion")

        try:
            _, exactitud, brier, _, _ = entrenar_modelo(partidos)
            messages.success(
                request,
                f"¡Modelo re-entrenado con éxito sobre {len(partidos)} partidos! "
                f"Exactitud temporal: {exactitud:.1%} | Brier Score: {brier:.4f}",
            )
        except Exception as exc:
            messages.error(request, f"Error al re-entrenar el modelo: {exc}")

    return redirect("ligas:configuracion")


@login_required
def jornada_cerrar_view(request, jornada_id):
    """Cierra la jornada, evalúa aciertos de predicciones y re-entrena automáticamente el modelo."""
    jornada = get_object_or_404(Jornada, pk=jornada_id)
    if request.method == "POST":
        aciertos = total = 0
        for partido in jornada.partidos.select_related("prediccion").filter(goles_local__isnull=False):
            if hasattr(partido, "prediccion"):
                total += 1
                if partido.prediccion.actualizar_con_resultado(partido.resultado):
                    aciertos += 1

        jornada.cerrada = True
        jornada.save(update_fields=["cerrada"])

        partidos = list(
            Partido.objects.filter(goles_local__isnull=False)
            .select_related("jornada__temporada", "local", "visitante")
            .prefetch_related("ausencias__jugador")
        )
        msg = f"Jornada {jornada.numero} cerrada ({aciertos}/{total} pronósticos acertados)."
        try:
            if len(partidos) >= 15:
                _, exactitud, brier, _, _ = entrenar_modelo(partidos)
                msg += f" Modelo re-entrenado automáticamente (Exactitud: {exactitud:.1%}, Brier: {brier:.4f})."
        except Exception as exc:
            msg += f" (Aviso de re-entrenamiento: {exc})"

        messages.success(request, msg)

    return redirect("ligas:configuracion")


@login_required
def jornada_nueva(request):
    """Crea una jornada vacía para después añadirle los partidos."""
    if request.method == "POST":
        form = JornadaForm(request.POST)
        if form.is_valid():
            jornada = form.save()
            messages.success(request, f"Jornada {jornada.numero} creada. Añade los partidos.")
            return redirect("ligas:jornada_partidos_add", jornada_id=jornada.id)
    else:
        form = JornadaForm()
    return render(request, "ligas/jornada_nueva.html", {"form": form})


@login_required
def jornada_partidos_add(request, jornada_id):
    """Añade partidos a una jornada (varios a la vez) y permite actualizar resultados y fechas en masa."""
    jornada = get_object_or_404(Jornada, pk=jornada_id)
    temporada = jornada.temporada

    if request.method == "POST" and request.POST.get("accion") == "actualizar_resultados":
        actualizados = 0
        for p in jornada.partidos.all():
            gl_key = f"gl_{p.id}"
            gv_key = f"gv_{p.id}"
            fecha_key = f"fecha_{p.id}"

            campos_modificados = []
            if gl_key in request.POST or gv_key in request.POST:
                gl_val = request.POST.get(gl_key, "").strip()
                gv_val = request.POST.get(gv_key, "").strip()
                p.goles_local = int(gl_val) if gl_val != "" else None
                p.goles_visitante = int(gv_val) if gv_val != "" else None
                campos_modificados.extend(["goles_local", "goles_visitante"])

            if fecha_key in request.POST:
                fecha_val = request.POST.get(fecha_key, "").strip()
                p.fecha = parse_fecha_flexible(fecha_val)
                campos_modificados.append("fecha")

            if campos_modificados:
                p.save(update_fields=list(set(campos_modificados)))
                actualizados += 1

        messages.success(request, f"Resultados de {actualizados} partido(s) actualizados.")
        return redirect("ligas:jornada_partidos_add", jornada_id=jornada.id)

    if request.method == "POST":
        formset = PartidoFormSet(request.POST, form_kwargs={"temporada": temporada})
        if formset.is_valid():
            creados = 0
            for formulario in formset:
                datos = formulario.cleaned_data
                if not datos or not datos.get("local") or not datos.get("visitante"):
                    continue
                _, creado = Partido.objects.update_or_create(
                    jornada=jornada,
                    local=datos["local"],
                    visitante=datos["visitante"],
                    defaults={
                        "fecha": datos.get("fecha"),
                        "goles_local": datos.get("goles_local"),
                        "goles_visitante": datos.get("goles_visitante"),
                    },
                )
                creados += int(creado)
            messages.success(request, f"{creados} partido(s) guardados correctamente.")
            return redirect("ligas:jornada_partidos_add", jornada_id=jornada.id)
    else:
        formset = PartidoFormSet(form_kwargs={"temporada": temporada})

    from collections import OrderedDict
    from django.db.models import F

    partidos_qs = list(
        jornada.partidos
        .select_related("local", "visitante")
        .prefetch_related("local__participaciones__division", "visitante__participaciones__division")
        .order_by(F("fecha").asc(nulls_last=True), "id")
    )

    def agrupar_partidos(partidos_lista):
        grupos = OrderedDict()
        for p in partidos_lista:
            label = p.fecha_formateada
            if label not in grupos:
                grupos[label] = {"label": label, "partidos": []}
            grupos[label]["partidos"].append(p)
        return list(grupos.values())

    partidos_div1 = [p for p in partidos_qs if p.division_nivel == 1]
    partidos_div2 = [p for p in partidos_qs if p.division_nivel == 2]

    grupos_fecha = agrupar_partidos(partidos_qs)
    grupos_div1 = agrupar_partidos(partidos_div1)
    grupos_div2 = agrupar_partidos(partidos_div2)

    return render(
        request,
        "ligas/jornada_partidos.html",
        {
            "jornada": jornada,
            "temporada": temporada,
            "formset": formset,
            "partidos": partidos_qs,
            "grupos_fecha": grupos_fecha,
            "grupos_div1": grupos_div1,
            "grupos_div2": grupos_div2,
            "partidos_div1": partidos_div1,
            "partidos_div2": partidos_div2,
        },
    )


@login_required
def partido_editar(request, jornada_id, partido_id):
    """Edita un partido específico de la jornada (equipos, fecha, goles)."""
    jornada = get_object_or_404(Jornada, pk=jornada_id)
    partido = get_object_or_404(Partido, pk=partido_id, jornada_id=jornada_id)
    temporada = jornada.temporada

    if request.method == "POST":
        form = PartidoForm(request.POST, instance=partido, temporada=temporada)
        if form.is_valid():
            form.save()
            messages.success(request, f"Partido {partido.local.nombre} vs {partido.visitante.nombre} actualizado.")
            return redirect("ligas:jornada_partidos_add", jornada_id=jornada.id)
    else:
        form = PartidoForm(instance=partido, temporada=temporada)

    return render(
        request,
        "ligas/partido_editar.html",
        {"form": form, "partido": partido, "jornada": jornada, "temporada": temporada},
    )


@login_required
def partido_eliminar(request, jornada_id, partido_id):
    if request.method != "POST":
        return redirect("ligas:jornada_partidos_add", jornada_id=jornada_id)
    partido = get_object_or_404(Partido, pk=partido_id, jornada_id=jornada_id)
    partido.delete()
    messages.success(request, "Partido eliminado.")
    return redirect("ligas:jornada_partidos_add", jornada_id=jornada_id)


@login_required
def equipos(request):
    config = Configuracion.cargar()
    temporada = config.temporada_actual or Temporada.objects.filter(activa=True).first()

    div1 = Division.objects.filter(nivel=1).first()
    div2 = Division.objects.filter(nivel=2).first()

    # Cambio rápido o desasignación de división desde la lista de equipos
    if request.method == "POST" and request.POST.get("accion") in ("cambiar_division", "desasignar_division"):
        accion = request.POST.get("accion")
        equipo_id = request.POST.get("equipo_id")
        nueva_div_id = request.POST.get("division_id")
        if equipo_id and temporada:
            equipo = get_object_or_404(Equipo, pk=equipo_id)
            if accion == "desasignar_division" or not nueva_div_id:
                Participacion.objects.filter(temporada=temporada, equipo=equipo).delete()
                messages.success(request, f"Se ha desasignado la división de {equipo.nombre} para la temporada {temporada}.")
            else:
                div = get_object_or_404(Division, pk=nueva_div_id)
                Participacion.objects.update_or_create(
                    temporada=temporada,
                    equipo=equipo,
                    defaults={"division": div},
                )
                messages.success(request, f"{equipo.nombre} asignado a {div.nombre} para la temporada {temporada}.")
            return redirect("ligas:equipos")

    equipos_qs = list(
        Equipo.objects.annotate(
            n_clave=Count("jugadores", filter=Q(jugadores__es_importante=True, jugadores__activo=True), distinct=True)
        )
        .prefetch_related("participaciones__division")
        .order_by("nombre")
    )

    equipos_div1 = []
    equipos_div2 = []
    equipos_sin_asignar = []

    for eq in equipos_qs:
        part = next((p for p in eq.participaciones.all() if p.temporada_id == getattr(temporada, 'id', None)), None)
        if part:
            eq.division_actual = part.division
            if part.division.nivel == 1:
                equipos_div1.append(eq)
            else:
                equipos_div2.append(eq)
        else:
            eq.division_actual = None
            equipos_sin_asignar.append(eq)

    return render(
        request,
        "ligas/equipos.html",
        {
            "temporada": temporada,
            "equipos_div1": equipos_div1,
            "equipos_div2": equipos_div2,
            "equipos_sin_asignar": equipos_sin_asignar,
            "div1": div1,
            "div2": div2,
            "total_equipos": len(equipos_qs),
        },
    )


@login_required
def equipo_editar(request, equipo_id=None):
    """Crea/edita un equipo y gestiona su división en la temporada actual y jugadores clave."""
    config = Configuracion.cargar()
    temporada = config.temporada_actual or Temporada.objects.filter(activa=True).first()
    equipo = get_object_or_404(Equipo, pk=equipo_id) if equipo_id else None

    if request.method == "POST":
        form = (
            EquipoForm(request.POST, request.FILES, instance=equipo, temporada=temporada)
            if equipo
            else EquipoForm(request.POST, request.FILES, temporada=temporada)
        )
        if form.is_valid():
            equipo = form.save()
            messages.success(request, f"Ficha del equipo {equipo.nombre} guardada correctamente.")
            return redirect("ligas:equipo_editar", equipo_id=equipo.id)
    else:
        form = EquipoForm(instance=equipo, temporada=temporada) if equipo else EquipoForm(temporada=temporada)

    jugadores = (
        Jugador.objects.filter(equipo=equipo).order_by("-es_importante", "nombre") if equipo else []
    )
    return render(
        request,
        "ligas/equipo_editar.html",
        {"form": form, "equipo": equipo, "temporada": temporada, "jugadores": jugadores},
    )


@login_required
def jugador_nuevo(request, equipo_id):
    equipo = get_object_or_404(Equipo, pk=equipo_id)
    if request.method == "POST":
        form = JugadorForm(request.POST)
        if form.is_valid():
            Jugador.objects.create(
                equipo=equipo,
                nombre=form.cleaned_data["nombre"],
                es_importante=form.cleaned_data["es_importante"],
                activo=form.cleaned_data["activo"],
            )
            messages.success(request, "Jugador añadido.")
    return redirect("ligas:equipo_editar", equipo_id=equipo.id)


@login_required
def jugador_toggle(request, jugador_id):
    jugador = get_object_or_404(Jugador, pk=jugador_id)
    if request.method == "POST":
        jugador.es_importante = not jugador.es_importante
        jugador.save(update_fields=["es_importante"])
    return redirect("ligas:equipo_editar", equipo_id=jugador.equipo_id)


@login_required
def jugador_eliminar(request, jugador_id):
    jugador = get_object_or_404(Jugador, pk=jugador_id)
    equipo_id = jugador.equipo_id
    if request.method == "POST":
        jugador.delete()
        messages.success(request, "Jugador eliminado.")
    return redirect("ligas:equipo_editar", equipo_id=equipo_id)


# ==========================================
# GESTIÓN Y CONFECCIÓN DE QUINIELAS (1X2)
# ==========================================


@login_required
def quinielas_lista(request):
    """Lista de boletos de Quiniela creados y formulario para dar de alta una nueva."""
    config = Configuracion.cargar()
    temporada = config.temporada_actual or Temporada.objects.filter(activa=True).first()

    if request.method == "POST" and request.POST.get("accion") == "crear_quiniela":
        form = QuinielaForm(request.POST, temporada=temporada)
        if form.is_valid():
            nueva_q = form.save()
            messages.success(
                request,
                f"¡{nueva_q.nombre} creada con éxito! Confecciona ahora las 15 casillas.",
            )
            return redirect("ligas:quiniela_confeccionar", quiniela_id=nueva_q.id)
    else:
        form = QuinielaForm(temporada=temporada)

    quinielas = list(
        Quiniela.objects.filter(temporada=temporada)
        .prefetch_related("casillas__partido")
        .order_by("-numero")
        if temporada
        else []
    )

    return render(
        request,
        "ligas/quinielas_lista.html",
        {
            "form": form,
            "config": config,
            "temporada": temporada,
            "quinielas": quinielas,
        },
    )


@login_required
def quiniela_confeccionar(request, quiniela_id):
    """Editor interactivo para asignar los 15 partidos (1ª y 2ª Div) y ordenar las casillas de la Quiniela."""
    quiniela = get_object_or_404(
        Quiniela.objects.select_related("temporada", "jornada"),
        pk=quiniela_id,
    )
    temporada = quiniela.temporada
    jornada = quiniela.jornada

    # 1. AUTO-LLENAR CON 10 DE 1ª Y 5 DE 2ª
    if request.method == "POST" and request.POST.get("accion") == "autollenar":
        # Buscar partidos de la jornada deportiva seleccionada o de la temporada
        partidos_qs = list(
            Partido.objects.filter(jornada=jornada)
            .select_related("local", "visitante")
            .prefetch_related("local__participaciones__division", "visitante__participaciones__division")
            .order_by("id")
            if jornada
            else Partido.objects.filter(jornada__temporada=temporada).order_by("-id")[:20]
        )
        p_div1 = [p for p in partidos_qs if p.division_nivel == 1]
        p_div2 = [p for p in partidos_qs if p.division_nivel == 2]
        p_otros = [p for p in partidos_qs if p.division_nivel not in (1, 2)]

        # Asignar los primeros 10 a 1ª Div y 5 siguientes a 2ª Div
        elegidos = p_div1[:10] + p_div2[:5]
        if len(elegidos) < 15:
            # Rellenar con los que falten de div1 o div2 u otros
            restantes = [p for p in (p_div1[10:] + p_div2[5:] + p_otros) if p not in elegidos]
            elegidos += restantes[: (15 - len(elegidos))]

        # Guardar en base de datos
        for idx, partido in enumerate(elegidos[:15], start=1):
            CasillaQuiniela.objects.update_or_create(
                quiniela=quiniela,
                posicion=idx,
                defaults={"partido": partido},
            )

        aplicar_predicciones_a_quiniela(quiniela)
        messages.success(
            request,
            f"Se han auto-asignado {len(elegidos[:15])} partidos a las casillas oficiales 1 a 15 con pronósticos calculados.",
        )
        return redirect("ligas:quiniela_confeccionar", quiniela_id=quiniela.id)

    # 2. GUARDAR CASILLAS ASIGNADAS MANUALMENTE
    if request.method == "POST" and request.POST.get("accion") == "guardar_casillas":
        actualizadas = 0
        partidos_asignados = set()

        for pos in range(1, 16):
            partido_id_str = request.POST.get(f"partido_pos_{pos}", "").strip()
            if partido_id_str.isdigit():
                pid = int(partido_id_str)
                if pid not in partidos_asignados:
                    partido = Partido.objects.filter(pk=pid).first()
                    if partido:
                        CasillaQuiniela.objects.update_or_create(
                            quiniela=quiniela,
                            posicion=pos,
                            defaults={"partido": partido},
                        )
                        partidos_asignados.add(pid)
                        actualizadas += 1

        aplicar_predicciones_a_quiniela(quiniela)
        messages.success(request, f"¡Casillas actualizadas ({actualizadas}/15) y pronósticos ML recalculados!")
        return redirect("ligas:quiniela_confeccionar", quiniela_id=quiniela.id)

    # 3. MOVER CASILLA ARRIBA / ABAJO
    if request.method == "POST" and request.POST.get("accion") in ("subir", "bajar", "intercambiar_pleno"):
        accion = request.POST.get("accion")
        pos = int(request.POST.get("posicion", 0))
        target_pos = None

        if accion == "subir" and pos > 1:
            target_pos = pos - 1
        elif accion == "bajar" and pos < 15:
            target_pos = pos + 1
        elif accion == "intercambiar_pleno" and pos != 15:
            target_pos = 15

        if target_pos:
            c_origen = CasillaQuiniela.objects.filter(quiniela=quiniela, posicion=pos).first()
            c_destino = CasillaQuiniela.objects.filter(quiniela=quiniela, posicion=target_pos).first()
            if c_origen and c_destino:
                p_tmp = c_origen.partido
                c_origen.partido = c_destino.partido
                c_destino.partido = p_tmp
                c_origen.save(update_fields=["partido"])
                c_destino.save(update_fields=["partido"])
                aplicar_predicciones_a_quiniela(quiniela)
                messages.success(request, f"Casilla {pos} intercambiada con Casilla {target_pos}.")
        return redirect("ligas:quiniela_confeccionar", quiniela_id=quiniela.id)

    # 4. GENERAR PRONÓSTICOS ML MANUALMENTE
    if request.method == "POST" and request.POST.get("accion") == "generar_predicciones":
        aplicar_predicciones_a_quiniela(quiniela)
        messages.success(request, "¡Pronósticos 1X2, Dobles, Triples y Pleno al 15 calculados con éxito por la IA!")
        return redirect("ligas:quiniela_confeccionar", quiniela_id=quiniela.id)

    # 5. EVALUAR RESULTADOS REALES
    if request.method == "POST" and request.POST.get("accion") == "evaluar":
        res = quiniela.evaluar_aciertos()
        messages.success(
            request,
            f"Resultados sincronizados: {res['aciertos_14']} aciertos en bloque 1X2 "
            f"({res['partidos_jugados']} partidos jugados)."
            + (" ¡Pleno al 15 Acertado! 🎯" if res["pleno_acierto"] else ""),
        )
        return redirect("ligas:quiniela_confeccionar", quiniela_id=quiniela.id)

    # Cargar las 15 casillas organizadas
    casillas_dict = {
        c.posicion: c
        for c in quiniela.casillas.select_related("partido__local", "partido__visitante", "partido__prediccion").all()
    }
    casillas_list = []
    for pos in range(1, 16):
        c = casillas_dict.get(pos)
        casillas_list.append({"posicion": pos, "casilla": c, "partido": c.partido if c else None})

    partidos_choices = obtener_partidos_choices(temporada=temporada, jornada=jornada)

    generador = GeneradorQuiniela(n_dobles=quiniela.n_dobles, n_triples=quiniela.n_triples)
    boleto_preview = generador.generar_desde_quiniela(quiniela)

    return render(
        request,
        "ligas/quiniela_confeccionar.html",
        {
            "quiniela": quiniela,
            "temporada": temporada,
            "jornada": jornada,
            "casillas_list": casillas_list,
            "partidos_choices": partidos_choices,
            "boleto_preview": boleto_preview,
        },
    )


@login_required
def quiniela_activar(request, quiniela_id):
    """Establece esta Quiniela como la activa para la vista pública."""
    if request.method != "POST":
        return redirect("ligas:quinielas_lista")
    quiniela = get_object_or_404(Quiniela, pk=quiniela_id)
    quiniela.activa = True
    quiniela.save()
    config = Configuracion.cargar()
    config.quiniela_actual = quiniela
    config.save(update_fields=["quiniela_actual"])
    messages.success(request, f"¡'{quiniela.nombre}' marcada como la Quiniela Activa del simulador!")
    return redirect("ligas:quinielas_lista")


@login_required
def quiniela_eliminar(request, quiniela_id):
    """Elimina una Quiniela y sus casillas asociadas."""
    if request.method != "POST":
        return redirect("ligas:quinielas_lista")
    quiniela = get_object_or_404(Quiniela, pk=quiniela_id)
    nombre = quiniela.nombre
    quiniela.delete()
    messages.success(request, f"Quiniela '{nombre}' eliminada.")
    return redirect("ligas:quinielas_lista")
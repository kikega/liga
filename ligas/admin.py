from django import forms
from django.contrib import admin, messages
from django.db import models
from django.shortcuts import redirect
from django.urls import path

from ligas.models import (
    Ausencia,
    Configuracion,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Participacion,
    Partido,
    Prediccion,
    Temporada,
)


class EquipoAdminForm(forms.ModelForm):
    class Meta:
        model = Equipo
        fields = "__all__"


@admin.register(Division)
class DivisionAdmin(admin.ModelAdmin):
    list_display = ("nombre", "nivel", "n_descensos")


@admin.register(Temporada)
class TemporadaAdmin(admin.ModelAdmin):
    list_display = ("nombre", "inicio", "fin", "activa")
    actions = ["marcar_activa", "aplicar_ascensos"]

    @admin.action(description="Marcar como temporada activa")
    def marcar_activa(self, request, queryset):
        queryset.update(activa=True)
        Temporada.objects.exclude(pk__in=queryset).update(activa=False)
        self.message_user(request, "Temporadas actualizadas.", messages.SUCCESS)

    @admin.action(description="Aplicar ascensos/descensos")
    def aplicar_ascensos(self, request, queryset):
        for temporada in queryset:
            temporada.aplicar_ascensos_descensos()
        self.message_user(request, "Ascensos y descensos aplicados.", messages.SUCCESS)


class JugadorInline(admin.TabularInline):
    model = Jugador
    extra = 0
    fields = ("nombre", "es_importante", "activo")


@admin.register(Equipo)
class EquipoAdmin(admin.ModelAdmin):
    list_display = ("nombre", "preview_escudo", "color_primario", "n_jugadores_clave")
    search_fields = ("nombre",)
    form = EquipoAdminForm
    formfield_overrides = {
        models.CharField: {"widget": forms.TextInput(attrs={"type": "color"})},
    }
    inlines = [JugadorInline]

    @admin.display(description="Escudo")
    def preview_escudo(self, obj):
        if obj.escudo:
            return f'<img src="{obj.escudo.url}" height="28" style="max-height:28px">'
        return "—"

    preview_escudo.allow_tags = True

    @admin.display(description="Jugadores clave")
    def n_jugadores_clave(self, obj):
        return obj.jugadores.filter(es_importante=True, activo=True).count()


@admin.register(Jugador)
class JugadorAdmin(admin.ModelAdmin):
    list_display = ("nombre", "equipo", "es_importante", "activo")
    list_filter = ("es_importante", "activo")
    search_fields = ("nombre", "equipo__nombre")
    autocomplete_fields = ("equipo",)


@admin.register(Participacion)
class ParticipacionAdmin(admin.ModelAdmin):
    list_display = ("equipo", "temporada", "division", "posicion_final")
    list_filter = ("temporada", "division")
    autocomplete_fields = ("equipo",)


@admin.register(Jornada)
class JornadaAdmin(admin.ModelAdmin):
    list_display = ("temporada", "numero", "n_partidos", "cerrada")
    list_filter = ("temporada", "cerrada")
    actions = ["cerrar_jornada", "reabrir_jornada"]

    @admin.display(description="Partidos")
    def n_partidos(self, obj):
        return obj.partidos.count()

    @admin.action(description="Cerrar jornadas (re-entrena el modelo)")
    def cerrar_jornada(self, request, queryset):
        from django.core.management import call_command

        for jornada in queryset:
            jornada.cerrada = True
            jornada.save(update_fields=["cerrada"])
        try:
            call_command("train_model", verbosity=0)
            self.message_user(request, "Jornadas cerradas y modelo re-entrenado.", messages.SUCCESS)
        except Exception as exc:  # noqa: BLE001
            self.message_user(request, f"Jornadas cerradas; modelo no re-entrenado: {exc}", messages.WARNING)

    @admin.action(description="Reabrir jornadas (permite corregir resultados)")
    def reabrir_jornada(self, request, queryset):
        queryset.update(cerrada=False)
        self.message_user(request, "Jornadas reabiertas.", messages.SUCCESS)


class AusenciaInline(admin.TabularInline):
    model = Ausencia
    extra = 0
    autocomplete_fields = ("jugador",)


@admin.register(Partido)
class PartidoAdmin(admin.ModelAdmin):
    list_display = ("jornada", "fecha", "local", "resultado_display", "visitante", "tiene_prediccion")
    list_filter = ("jornada__temporada", "jornada")
    autocomplete_fields = ("local", "visitante")
    inlines = [AusenciaInline]

    @admin.display(description="Resultado")
    def resultado_display(self, obj):
        if not obj.jugado:
            return "—"
        return f"{obj.goles_local} - {obj.goles_visitante}"

    @admin.display(description="Predicción", boolean=True)
    def tiene_prediccion(self, obj):
        return Prediccion.objects.filter(partido=obj).exists()


@admin.register(Prediccion)
class PrediccionAdmin(admin.ModelAdmin):
    list_display = ("partido", "resultado_predicho", "prob_local", "prob_empate", "prob_visitante", "resultado_real", "acierto")
    readonly_fields = ("fecha_prediccion",)


@admin.register(Configuracion)
class ConfiguracionAdmin(admin.ModelAdmin):
    list_display = ("temporada_actual", "proxima_jornada")

    def has_add_permission(self, request):
        return not Configuracion.objects.exists()

    def change_view(self, request, object_id, form_url="", extra_context=None):
        config, _ = Configuracion.objects.get_or_create(pk=1)
        return super().change_view(request, config.pk, form_url, extra_context=extra_context)

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "",
                self.admin_site.admin_view(self._redirect_to_singleton),
                name=f"{self.model._meta.app_label}_{self.model._meta.model_name}_changelist",
            )
        ]
        return custom + urls

    def _redirect_to_singleton(self, request):
        config, _ = Configuracion.objects.get_or_create(pk=1)
        return redirect(f"../../{self.model._meta.model_name}/{config.pk}/change/")
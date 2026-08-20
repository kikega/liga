from django.contrib import admin

from ligas.models import (
    Ausencia,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Participacion,
    Partido,
    Prediccion,
    Temporada,
)


@admin.register(Division)
class DivisionAdmin(admin.ModelAdmin):
    list_display = ("nombre", "nivel", "n_descensos")


@admin.register(Temporada)
class TemporadaAdmin(admin.ModelAdmin):
    list_display = ("nombre", "inicio", "fin", "activa")


@admin.register(Equipo)
class EquipoAdmin(admin.ModelAdmin):
    search_fields = ("nombre",)


@admin.register(Participacion)
class ParticipacionAdmin(admin.ModelAdmin):
    list_display = ("equipo", "temporada", "division", "posicion_final")
    list_filter = ("temporada", "division")
    autocomplete_fields = ("equipo",)


class JugadorInline(admin.TabularInline):
    model = Jugador
    extra = 0


@admin.register(Jugador)
class JugadorAdmin(admin.ModelAdmin):
    list_display = ("nombre", "equipo", "es_importante", "activo")
    list_filter = ("es_importante", "activo")
    search_fields = ("nombre",)
    autocomplete_fields = ("equipo",)


@admin.register(Jornada)
class JornadaAdmin(admin.ModelAdmin):
    list_display = ("temporada", "numero", "cerrada")
    list_filter = ("temporada", "cerrada")


class PartidoInline(admin.TabularInline):
    model = Partido
    extra = 1
    autocomplete_fields = ("local", "visitante")


class AusenciaInline(admin.TabularInline):
    model = Ausencia
    extra = 0
    autocomplete_fields = ("jugador",)


@admin.register(Partido)
class PartidoAdmin(admin.ModelAdmin):
    list_display = ("jornada", "local", "visitante", "goles_local", "goles_visitante", "resultado")
    list_filter = ("jornada__temporada",)
    autocomplete_fields = ("local", "visitante")
    inlines = [AusenciaInline]


@admin.register(Prediccion)
class PrediccionAdmin(admin.ModelAdmin):
    list_display = ("partido", "resultado_predicho", "acierto", "fecha_prediccion")
    readonly_fields = ("fecha_prediccion",)

from django.urls import path

from ligas import config_views, views

app_name = "ligas"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("predicciones/", views.predicciones, name="predicciones"),
    path("quiniela/", views.quiniela_view, name="quiniela"),
    path("jornada/<int:jornada_id>/predecir/", views.jornada_predecir, name="jornada_predecir"),
    path("configuracion/", config_views.configuracion, name="configuracion"),
    path("configuracion/modelo/reentrenar/", config_views.reentrenar_modelo_view, name="reentrenar_modelo"),
    path("configuracion/jornada/nueva/", config_views.jornada_nueva, name="jornada_nueva"),
    path("configuracion/jornada/<int:jornada_id>/", config_views.jornada_partidos_add, name="jornada_partidos_add"),
    path("configuracion/jornada/<int:jornada_id>/activar/", config_views.jornada_activar_view, name="jornada_activar"),
    path("configuracion/jornada/<int:jornada_id>/cerrar/", config_views.jornada_cerrar_view, name="jornada_cerrar"),
    path("configuracion/jornada/<int:jornada_id>/partido/<int:partido_id>/editar/", config_views.partido_editar, name="partido_editar"),
    path("configuracion/jornada/<int:jornada_id>/partido/<int:partido_id>/eliminar/", config_views.partido_eliminar, name="partido_eliminar"),
    path("configuracion/equipos/", config_views.equipos, name="equipos"),
    path("configuracion/equipos/nuevo/", config_views.equipo_editar, name="equipo_nuevo"),
    path("configuracion/equipos/<int:equipo_id>/", config_views.equipo_editar, name="equipo_editar"),
    path("configuracion/equipos/<int:equipo_id>/jugadores/nuevo/", config_views.jugador_nuevo, name="jugador_nuevo"),
    path("configuracion/jugador/<int:jugador_id>/toggle/", config_views.jugador_toggle, name="jugador_toggle"),
    path("configuracion/jugador/<int:jugador_id>/eliminar/", config_views.jugador_eliminar, name="jugador_eliminar"),
    path("configuracion/quinielas/", config_views.quinielas_lista, name="quinielas_lista"),
    path("configuracion/quiniela/<int:quiniela_id>/", config_views.quiniela_confeccionar, name="quiniela_confeccionar"),
    path("configuracion/quiniela/<int:quiniela_id>/activar/", config_views.quiniela_activar, name="quiniela_activar"),
    path("configuracion/quiniela/<int:quiniela_id>/eliminar/", config_views.quiniela_eliminar, name="quiniela_eliminar"),
]
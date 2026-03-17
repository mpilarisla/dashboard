import json
import math
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. CARGA DE DATOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

radios = gpd.read_file(BASE_DIR / "procesados" / "radios_base.geojson")
piramide = pd.read_csv(BASE_DIR / "procesados" / "tabla_piramide.csv")
viviendas = pd.read_csv(BASE_DIR / "procesados" / "viviendas_radio_gpm.csv")
nbi = pd.read_csv(BASE_DIR / "procesados" / "nbi_radio_gpm.csv")

CAPAS_DIR = BASE_DIR / "datos" / "capas"

capas_ref = {
    "avenidas": gpd.read_file(CAPAS_DIR / "avenidas.shp"),
    "calles": gpd.read_file(CAPAS_DIR / "calles.shp"),
    "red_agua": gpd.read_file(CAPAS_DIR / "Red_Agua_2020.shp"),
    "red_cloaca": gpd.read_file(CAPAS_DIR / "Red_Cloaca_2020.shp"),
    "renabap": gpd.read_file(CAPAS_DIR / "RENABAP_2023.shp"),
    "barrios_gp": gpd.read_file(CAPAS_DIR / "barrios_gp.shp"),
    "barrios_privados": gpd.read_file(CAPAS_DIR / "barrios_privados.shp"),
}

if radios.crs is not None and radios.crs.to_epsg() != 4326:
    radios = radios.to_crs(4326)

for k in capas_ref:
    if capas_ref[k].crs is not None and capas_ref[k].crs.to_epsg() != 4326:
        capas_ref[k] = capas_ref[k].to_crs(4326)

for col in [
    "link_num", "poblacion_total", "mujeres_total", "varones_total",
    "area_m2", "area_ha", "area_km2", "dens_hab_ha", "dens_hab_km2"
]:
    if col in radios.columns:
        radios[col] = pd.to_numeric(radios[col], errors="coerce")

for col in ["radio_id", "personas", "edad_cod", "sexo_cod"]:
    if col in piramide.columns:
        piramide[col] = pd.to_numeric(piramide[col], errors="coerce")

for col in ["radio_id", "viviendas_ocupadas", "viviendas_desocupadas", "viviendas_totales"]:
    if col in viviendas.columns:
        viviendas[col] = pd.to_numeric(viviendas[col], errors="coerce")

for col in [
    "radio_id", "hogares_con_nbi", "hogares_sin_nbi", "hogares_totales",
    "hogares_nbi_hac", "hogares_nbi_viv", "hogares_nbi_san",
    "hogares_nbi_esc", "hogares_nbi_sub"
]:
    if col in nbi.columns:
        nbi[col] = pd.to_numeric(nbi[col], errors="coerce")

radios = radios.dropna(subset=["link_num", "dens_hab_ha"]).copy()
piramide = piramide.dropna(subset=["radio_id", "edad_cod", "sexo_cod", "personas"]).copy()
viviendas = viviendas.dropna(subset=["radio_id"]).copy()
nbi = nbi.dropna(subset=["radio_id"]).copy()

radios["link_num"] = radios["link_num"].astype(int)
piramide["radio_id"] = piramide["radio_id"].astype(int)
piramide["edad_cod"] = piramide["edad_cod"].astype(int)
piramide["sexo_cod"] = piramide["sexo_cod"].astype(int)
viviendas["radio_id"] = viviendas["radio_id"].astype(int)
nbi["radio_id"] = nbi["radio_id"].astype(int)

for col in ["viviendas_ocupadas", "viviendas_desocupadas", "viviendas_totales"]:
    if col in viviendas.columns:
        viviendas[col] = viviendas[col].fillna(0)

for col in [
    "hogares_con_nbi", "hogares_sin_nbi", "hogares_totales",
    "hogares_nbi_hac", "hogares_nbi_viv", "hogares_nbi_san",
    "hogares_nbi_esc", "hogares_nbi_sub"
]:
    if col in nbi.columns:
        nbi[col] = nbi[col].fillna(0)

# =========================================================
# 2. BARRIOS GP: OPCIONES PARA DROPDOWN
# =========================================================

barrios_options = []
if "soc_fomen" in capas_ref["barrios_gp"].columns:
    barrios_options = sorted([
        b for b in capas_ref["barrios_gp"]["soc_fomen"].dropna().astype(str).unique().tolist()
        if b.strip() != ""
    ])

TODOS_VALUE = "__TODOS__"

dropdown_barrios_options = [{"label": "Todos", "value": TODOS_VALUE}] + [
    {"label": b, "value": b} for b in barrios_options
]

# =========================================================
# 3. CENTRO / JSON / ORDEN EDADES
# =========================================================

centro = radios.unary_union.centroid
CENTER_LAT = float(centro.y)
CENTER_LON = float(centro.x)
DEFAULT_ZOOM = 9.2
ETIQUETA_GENERAL = "General Pueyrredon"

radios_geo = radios.copy()
radios_geo["link_num"] = radios_geo["link_num"].astype(str)
radios_json = json.loads(radios_geo.to_json())

orden_edades = (
    piramide[["edad_cod", "edad_label"]]
    .drop_duplicates()
    .sort_values("edad_cod")["edad_label"]
    .tolist()
)

# =========================================================
# 4. CLASIFICACIÓN GLOBAL DE DENSIDAD
# =========================================================

COLORES_INTERVALOS = [
    "#241335",
    "#3f2161",
    "#62359a",
    "#8b57d6",
    "#d7b8ff",
]
COLOR_MASCARA = "#0a0b0f"

def formato_num(n):
    if pd.isna(n):
        return "NA"
    return f"{n:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formato_entero_puntos(n):
    if pd.isna(n):
        return "NA"
    return f"{int(round(n)):,}".replace(",", ".")

def clasificar_densidad_global(df, n_clases=5):
    valores = df["dens_hab_ha"].dropna().astype(float).values
    cortes = np.quantile(valores, q=np.linspace(0, 1, n_clases + 1))
    cortes = np.unique(cortes)

    n_clases_reales = len(cortes) - 1
    if n_clases_reales < 2:
        out = df.copy()
        out["class_id"] = "0"
        legend = [("Todos", COLORES_INTERVALOS[-1])]
        color_map = {"0": COLORES_INTERVALOS[-1]}
        return out, legend, color_map

    colores = COLORES_INTERVALOS[:n_clases_reales]

    out = df.copy()
    out["class_num"] = pd.cut(
        out["dens_hab_ha"],
        bins=cortes,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    out["class_num"] = out["class_num"].astype(int)
    out["class_id"] = out["class_num"].astype(str)

    color_map = {str(i): colores[i] for i in range(n_clases_reales)}
    legend = []
    for i in range(n_clases_reales):
        a = cortes[i]
        b = cortes[i + 1]
        legend.append((f"{formato_num(a)} – {formato_num(b)}", colores[i]))

    return out, legend, color_map

radios_cls, INTERVAL_LEGEND, INTERVAL_COLOR_MAP = clasificar_densidad_global(radios, n_clases=5)
radios_cls["loc_id"] = radios_cls["link_num"].astype(str)

# =========================================================
# 5. ESTILO
# =========================================================

BG = "#050608"
PANEL = "#111318"
PANEL_2 = "#0d0f14"
BORDER = "#2a2e38"
TEXT = "#f5f7fa"
MUTED = "#a9b0bc"
SCALE_COLOR = "#d4d7dd"

PURPLE_1 = "#d7b8ff"
PURPLE_2 = "#b07cff"
PURPLE_3 = "#8b57d6"
PURPLE_4 = "#5e2bd4"
PURPLE_TOTAL_NBI = COLORES_INTERVALOS[-1]
WHITE_LAYER = "#ffffff"

container_style = {
    "backgroundColor": BG,
    "height": "100vh",
    "minHeight": "100vh",
    "padding": "8px 10px 8px 10px",
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "overflow": "hidden",
}

panel_style = {
    "backgroundColor": PANEL,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "boxShadow": "0 0 0 1px rgba(255,255,255,0.02) inset",
}

inner_panel_style = {
    "backgroundColor": PANEL_2,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
}

# =========================================================
# 6. AUXILIARES
# =========================================================

def subconjunto_por_radios(df, key_col, selected_radios):
    if selected_radios:
        return df[df[key_col].isin(selected_radios)].copy(), f"{len(selected_radios)} radio(s) seleccionado(s)"
    return df.copy(), ETIQUETA_GENERAL

def filtrar_piramide(selected_radios):
    df, etiqueta = subconjunto_por_radios(piramide, "radio_id", selected_radios)
    resumen = (
        df.groupby(["edad_cod", "edad_label", "sexo_cod", "sexo_label"], as_index=False)["personas"]
        .sum()
    )
    return resumen, etiqueta

def figura_piramide(selected_radios):
    df, etiqueta = filtrar_piramide(selected_radios)

    mujeres = (
        df[df["sexo_label"] == "Mujer/Femenino"][["edad_cod", "edad_label", "personas"]]
        .copy()
        .sort_values("edad_cod")
    )
    varones = (
        df[df["sexo_label"] == "Varón/Masculino"][["edad_cod", "edad_label", "personas"]]
        .copy()
        .sort_values("edad_cod")
    )

    mujeres["valor_plot"] = -mujeres["personas"]
    varones["valor_plot"] = varones["personas"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=mujeres["edad_label"],
            x=mujeres["valor_plot"],
            name="Mujeres",
            orientation="h",
            marker_color=PURPLE_2,
            customdata=[formato_entero_puntos(v) for v in mujeres["personas"]],
            hovertemplate="Edad: %{y}<br>Mujeres: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=varones["edad_label"],
            x=varones["valor_plot"],
            name="Varones",
            orientation="h",
            marker_color=PURPLE_4,
            customdata=[formato_entero_puntos(v) for v in varones["personas"]],
            hovertemplate="Edad: %{y}<br>Varones: %{customdata}<extra></extra>",
        )
    )

    max_abs = max(
        abs(mujeres["valor_plot"].min()) if not mujeres.empty else 0,
        abs(varones["valor_plot"].max()) if not varones.empty else 0,
    )
    max_abs = max_abs * 1.15 if max_abs > 0 else 10

    fig.update_layout(
        barmode="relative",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        title=dict(
            text="Estructura de población",
            x=0.02,
            xanchor="left",
            font=dict(size=16, color=TEXT),
        ),
        annotations=[
            dict(
                text=etiqueta,
                x=0.98,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="right",
                showarrow=False,
                font=dict(size=11, color=MUTED),
            )
        ],
        margin=dict(l=16, r=16, t=48, b=48),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11),
        ),
        xaxis=dict(
            title="Personas",
            tickformat=",",
            range=[-max_abs, max_abs],
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.18)",
            color=MUTED,
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=orden_edades,
            color=MUTED,
        ),
    )
    return fig

def _geometry_to_lines(geom):
    lons, lats = [], []
    if geom is None or geom.is_empty:
        return lons, lats

    gtype = geom.geom_type

    if gtype == "LineString":
        xs, ys = geom.xy
        lons.extend(list(xs) + [None])
        lats.extend(list(ys) + [None])
    elif gtype == "MultiLineString":
        for part in geom.geoms:
            xs, ys = part.xy
            lons.extend(list(xs) + [None])
            lats.extend(list(ys) + [None])
    elif gtype == "Polygon":
        return _geometry_to_lines(geom.boundary)
    elif gtype == "MultiPolygon":
        for part in geom.geoms:
            lon_part, lat_part = _geometry_to_lines(part.boundary)
            lons.extend(lon_part)
            lats.extend(lat_part)
    elif gtype == "GeometryCollection":
        for part in geom.geoms:
            lon_part, lat_part = _geometry_to_lines(part)
            lons.extend(lon_part)
            lats.extend(lat_part)

    return lons, lats

def make_layer_trace(gdf, name):
    lons, lats = [], []
    for geom in gdf.geometry:
        lon_part, lat_part = _geometry_to_lines(geom)
        lons.extend(lon_part)
        lats.extend(lat_part)

    return go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(width=1.15, color=WHITE_LAYER),
        hoverinfo="skip",
        name=name,
        showlegend=False,
    )

def normalizar_barrios_seleccionados(capas_activas, barrios_elegidos):
    capas_activas = capas_activas or []
    barrios_elegidos = barrios_elegidos or []

    if "barrios_gp" not in capas_activas:
        return []

    if TODOS_VALUE in barrios_elegidos:
        return barrios_options.copy()

    return [b for b in barrios_elegidos if b in barrios_options]

def figura_mapa(selected_radios, map_view, capas_activas, barrios_elegidos):
    current_zoom = map_view.get("zoom", DEFAULT_ZOOM)
    current_center = map_view.get("center", {"lat": CENTER_LAT, "lon": CENTER_LON})

    df = radios_cls.copy()

    if selected_radios:
        selected_set = set(selected_radios)
        df["map_class"] = np.where(
            df["link_num"].isin(selected_set),
            df["class_id"],
            "masked"
        )
        color_map = {**INTERVAL_COLOR_MAP, "masked": COLOR_MASCARA}
        category_order = sorted(INTERVAL_COLOR_MAP.keys(), key=int) + ["masked"]
    else:
        df["map_class"] = df["class_id"]
        color_map = INTERVAL_COLOR_MAP
        category_order = sorted(INTERVAL_COLOR_MAP.keys(), key=int)

    fig = px.choropleth_mapbox(
        df,
        geojson=radios_json,
        locations="loc_id",
        featureidkey="properties.link_num",
        color="map_class",
        category_orders={"map_class": category_order},
        color_discrete_map=color_map,
        mapbox_style="carto-darkmatter",
        center=current_center,
        zoom=current_zoom,
        opacity=0.90,
    )

    hover_text = [
        (
            f"Radio: {formato_entero_puntos(r['link_num'])}<br>"
            f"Población: {formato_entero_puntos(r['poblacion_total'])}<br>"
            f"Mujeres: {formato_entero_puntos(r['mujeres_total'])}<br>"
            f"Varones: {formato_entero_puntos(r['varones_total'])}"
        )
        for _, r in df.iterrows()
    ]

    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.12)",
        marker_line_width=0.28,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showscale=False,
    )

    capas_activas = capas_activas or []
    barrios_elegidos = normalizar_barrios_seleccionados(capas_activas, barrios_elegidos)

    if "avenidas" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["avenidas"], "Avenidas"))
    if "calles" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["calles"], "Calles"))
    if "red_agua" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["red_agua"], "Red Agua"))
    if "red_cloaca" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["red_cloaca"], "Red Cloaca"))
    if "renabap" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["renabap"], "RENABAP"))
    if "barrios_privados" in capas_activas:
        fig.add_trace(make_layer_trace(capas_ref["barrios_privados"], "Barrios Privados"))
    if "barrios_gp" in capas_activas and len(barrios_elegidos) > 0:
        gdf_barrios = capas_ref["barrios_gp"][
            capas_ref["barrios_gp"]["soc_fomen"].astype(str).isin(barrios_elegidos)
        ].copy()
        if not gdf_barrios.empty:
            fig.add_trace(make_layer_trace(gdf_barrios, "Barrios de General Pueyrredon"))

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            style="carto-darkmatter",
            center=current_center,
            zoom=current_zoom,
        ),
        clickmode="event",
        uirevision="keep-map-view",
        showlegend=False,
        autosize=True,
    )
    return fig

def calcular_totales(selected_radios):
    if selected_radios:
        base = radios[radios["link_num"].isin(selected_radios)].copy()
        etiqueta = f"{len(selected_radios)} radio(s) seleccionados"
    else:
        base = radios.copy()
        etiqueta = ETIQUETA_GENERAL

    return {
        "etiqueta": etiqueta,
        "poblacion": int(base["poblacion_total"].sum()),
        "mujeres": int(base["mujeres_total"].sum()),
        "varones": int(base["varones_total"].sum()),
    }

def calcular_viviendas(selected_radios):
    df, etiqueta = subconjunto_por_radios(viviendas, "radio_id", selected_radios)
    ocupadas = int(df["viviendas_ocupadas"].sum()) if "viviendas_ocupadas" in df.columns else 0
    desocupadas = int(df["viviendas_desocupadas"].sum()) if "viviendas_desocupadas" in df.columns else 0
    total = ocupadas + desocupadas

    return {
        "etiqueta": etiqueta,
        "ocupadas": ocupadas,
        "desocupadas": desocupadas,
        "total": total,
    }

def calcular_nbi(selected_radios):
    df, etiqueta = subconjunto_por_radios(nbi, "radio_id", selected_radios)
    return {
        "etiqueta": etiqueta,
        "con_nbi": int(df["hogares_con_nbi"].sum()) if "hogares_con_nbi" in df.columns else 0,
        "hac": int(df["hogares_nbi_hac"].sum()) if "hogares_nbi_hac" in df.columns else 0,
        "viv": int(df["hogares_nbi_viv"].sum()) if "hogares_nbi_viv" in df.columns else 0,
        "san": int(df["hogares_nbi_san"].sum()) if "hogares_nbi_san" in df.columns else 0,
        "esc": int(df["hogares_nbi_esc"].sum()) if "hogares_nbi_esc" in df.columns else 0,
        "sub": int(df["hogares_nbi_sub"].sum()) if "hogares_nbi_sub" in df.columns else 0,
    }

def figura_viviendas(selected_radios):
    res = calcular_viviendas(selected_radios)

    hover_text = [
        f"Ocupadas: {formato_entero_puntos(res['ocupadas'])}",
        f"Desocupadas: {formato_entero_puntos(res['desocupadas'])}",
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Ocupadas", "Desocupadas"],
                values=[res["ocupadas"], res["desocupadas"]],
                hole=0.60,
                sort=False,
                textinfo="percent",
                marker=dict(colors=[PURPLE_4, PURPLE_1], line=dict(color=PANEL, width=2)),
                texttemplate="%{percent}",
                customdata=hover_text,
                hovertemplate="%{customdata}<br>%{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        title=dict(
            text="Viviendas ocupadas<br>y desocupadas",
            x=0.04,
            xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        annotations=[
            dict(
                text=res["etiqueta"],
                x=0.96,
                y=1.06,
                xref="paper",
                yref="paper",
                xanchor="right",
                showarrow=False,
                font=dict(size=10, color=MUTED),
            ),
            dict(
                text=formato_entero_puntos(res["total"]),
                x=0.5,
                y=0.48,
                showarrow=False,
                font=dict(size=15, color=TEXT),
            ),
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(color=MUTED, size=10),
        ),
        margin=dict(l=8, r=8, t=54, b=24),
    )
    return fig

def figura_nbi(selected_radios):
    res = calcular_nbi(selected_radios)

    categorias = [
        "Con NBI",
        "Hacinamiento crítico",
        "Vivienda inadecuada",
        "Condiciones sanitarias",
        "Inasistencia escolar",
        "Capacidad de subsistencia",
    ]
    valores = [
        res["con_nbi"],
        res["hac"],
        res["viv"],
        res["san"],
        res["esc"],
        res["sub"],
    ]
    colores = [
        PURPLE_TOTAL_NBI,
        PURPLE_3,
        PURPLE_3,
        PURPLE_3,
        PURPLE_3,
        PURPLE_3,
    ]
    grupos = [
        "NBI total",
        "Indicadores",
        "Indicadores",
        "Indicadores",
        "Indicadores",
        "Indicadores",
    ]
    hover_text = [
        f"{cat}<br>Grupo: {grp}<br>Cantidad: {formato_entero_puntos(val)}"
        for cat, grp, val in zip(categorias, grupos, valores)
    ]

    fig = go.Figure(
        go.Bar(
            x=valores,
            y=categorias,
            orientation="h",
            marker_color=colores,
            text=None,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        title=dict(
            text="NBI total e indicadores",
            x=0.02,
            xanchor="left",
            font=dict(size=15, color=TEXT),
        ),
        annotations=[
            dict(
                text=res["etiqueta"],
                x=0.98,
                y=1.10,
                xref="paper",
                yref="paper",
                xanchor="right",
                showarrow=False,
                font=dict(size=11, color=MUTED),
            )
        ],
        margin=dict(l=150, r=18, t=54, b=18),
        xaxis=dict(
            title="Hogares",
            tickformat=",",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.16)",
            color=MUTED,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            automargin=True,
            ticklabelposition="outside",
            ticklabelstandoff=10,
            tickfont=dict(size=11),
            color=MUTED,
        ),
        showlegend=False,
    )
    return fig

def kpi_card(titulo, valor, subtitulo=None):
    return html.Div(
        style={
            **inner_panel_style,
            "padding": "14px 10px",
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "alignItems": "center",
            "textAlign": "center",
        },
        children=[
            html.Div(
                titulo.upper(),
                style={
                    "fontSize": "0.74rem",
                    "letterSpacing": "0.08em",
                    "color": MUTED,
                    "marginBottom": "8px",
                    "fontWeight": "600",
                },
            ),
            html.Div(
                formato_entero_puntos(valor),
                style={
                    "fontSize": "1.95rem",
                    "fontWeight": "700",
                    "lineHeight": "1.05",
                    "color": TEXT,
                },
            ),
            html.Div(
                subtitulo or "",
                style={"fontSize": "0.80rem", "color": MUTED, "marginTop": "6px"},
            ),
        ],
    )

def panel_titulo(texto):
    return html.Div(
        texto.upper(),
        style={
            "fontSize": "0.78rem",
            "letterSpacing": "0.08em",
            "color": MUTED,
            "fontWeight": "600",
            "marginBottom": "8px",
        },
    )

def nice_scale_length(meters):
    if meters <= 0:
        return 100
    exponent = math.floor(math.log10(meters))
    fraction = meters / (10 ** exponent)

    if fraction < 1.5:
        nice = 1
    elif fraction < 3.5:
        nice = 2
    elif fraction < 7.5:
        nice = 5
    else:
        nice = 10

    return nice * (10 ** exponent)

def format_distance(meters):
    if meters >= 1000:
        km = meters / 1000
        if km >= 10:
            return f"{int(round(km))} km"
        return f"{km:.1f} km".replace(".", ",")
    return f"{int(round(meters))} m"

def scale_bar_style_and_label(map_view):
    zoom = map_view.get("zoom", DEFAULT_ZOOM)
    lat = map_view.get("center", {}).get("lat", CENTER_LAT)

    meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    target_px = 110
    raw_distance = meters_per_pixel * target_px
    nice_distance = nice_scale_length(raw_distance)
    px_width = max(40, min(160, int(nice_distance / meters_per_pixel)))

    style = {
        "position": "absolute",
        "left": "18px",
        "bottom": "22px",
        "zIndex": 1000,
        "pointerEvents": "none",
        "backgroundColor": "rgba(5, 6, 8, 0.55)",
        "padding": "4px 6px 6px 6px",
        "borderRadius": "6px",
    }

    bar_style = {
        "width": f"{px_width}px",
        "height": "4px",
        "backgroundColor": SCALE_COLOR,
        "borderRadius": "2px",
        "boxShadow": "0 0 0 1px rgba(255,255,255,0.08)",
    }

    label_style = {
        "color": SCALE_COLOR,
        "fontSize": "0.78rem",
        "lineHeight": "1",
        "marginBottom": "4px",
        "fontWeight": "600",
    }

    return style, bar_style, label_style, format_distance(nice_distance)

def make_interval_legend():
    return html.Div(
        style={
            "position": "absolute",
            "right": "16px",
            "top": "16px",
            "zIndex": 1000,
            "backgroundColor": "rgba(5, 6, 8, 0.68)",
            "padding": "8px 10px",
            "borderRadius": "8px",
            "border": "1px solid rgba(255,255,255,0.08)",
            "maxWidth": "210px",
            "pointerEvents": "none",
        },
        children=[
            html.Div(
                "Densidad (hab/ha)",
                style={
                    "color": TEXT,
                    "fontSize": "0.82rem",
                    "fontWeight": "700",
                    "marginBottom": "6px",
                },
            ),
            *[
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "marginBottom": "4px",
                    },
                    children=[
                        html.Div(
                            style={
                                "width": "12px",
                                "height": "12px",
                                "backgroundColor": color,
                                "border": "1px solid rgba(255,255,255,0.12)",
                            }
                        ),
                        html.Div(
                            label,
                            style={"color": MUTED, "fontSize": "0.75rem", "lineHeight": "1.15"},
                        ),
                    ],
                )
                for label, color in INTERVAL_LEGEND
            ],
        ],
    )

def make_selection_box(selected_radios):
    if not selected_radios:
        items = [html.Div("General Pueyrredon", style={"color": TEXT, "fontSize": "0.82rem"})]
    else:
        items = [
            html.Div(
                f"Radio {formato_entero_puntos(r)}",
                style={"color": TEXT, "fontSize": "0.75rem", "lineHeight": "1.2", "marginBottom": "2px"},
            )
            for r in sorted(selected_radios)
        ]

    return html.Div(
        style={
            "maxWidth": "190px",
            "maxHeight": "90px",
            "overflowY": "auto",
            "backgroundColor": "rgba(5, 6, 8, 0.55)",
            "padding": "6px 8px",
            "borderRadius": "8px",
            "border": "1px solid rgba(255,255,255,0.08)",
        },
        children=[
            html.Div(
                "Selección",
                style={
                    "color": MUTED,
                    "fontSize": "0.7rem",
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "fontWeight": "700",
                    "marginBottom": "4px",
                },
            ),
            *items,
        ],
    )

# =========================================================
# 7. APP
# =========================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard Censo GPM"

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #react-entry-point {
                height: 100%;
                margin: 0;
                padding: 0;
                background: #050608;
                overflow: hidden;
            }
            body {
                background: #050608 !important;
            }
            .Select-control, .Select-menu-outer, .VirtualizedSelectOption {
                background-color: #111318 !important;
                color: #f5f7fa !important;
                border-color: #2a2e38 !important;
            }
            .Select-placeholder, .Select-value-label {
                color: #f5f7fa !important;
            }
            .is-focused:not(.is-open) > .Select-control {
                border-color: #8b57d6 !important;
                box-shadow: none !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = dbc.Container(
    fluid=True,
    style=container_style,
    children=[
        dcc.Store(id="selected-radios", data=[]),
        dcc.Store(id="map-view", data={"zoom": DEFAULT_ZOOM, "center": {"lat": CENTER_LAT, "lon": CENTER_LON}}),

        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "8px",
                "padding": "2px 4px",
                "flex": "0 0 auto",
            },
            children=[
                html.Div([
                    html.Div(
                        "DASHBOARD INTERACTIVO",
                        style={
                            "fontSize": "0.82rem",
                            "letterSpacing": "0.12em",
                            "color": MUTED,
                            "fontWeight": "600",
                        },
                    ),
                    html.H3(
                        "CENSO 2022 (INDEC) - GENERAL PUEYRREDON",
                        style={
                            "margin": "2px 0 0 0",
                            "color": TEXT,
                            "fontWeight": "700",
                            "fontSize": "1.35rem",
                        },
                    ),
                ]),
                html.Div(" ", style={"color": MUTED, "fontSize": "0.85rem"}),
            ],
        ),

        dbc.Row(
            className="g-3",
            style={
                "height": "calc(100vh - 84px)",
                "marginBottom": "0",
                "overflow": "hidden",
            },
            children=[
                dbc.Col(
                    md=7,
                    style={"height": "100%", "minHeight": "0"},
                    children=[
                        html.Div(
                            style={
                                "height": "100%",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                                "overflow": "hidden",
                            },
                            children=[
                                dbc.Row(
                                    className="g-3",
                                    style={"flex": "0 0 58%", "marginBottom": "0"},
                                    children=[
                                        dbc.Col(
                                            md=10,
                                            children=[
                                                html.Div(
                                                    style={**panel_style, "padding": "6px", "height": "100%"},
                                                    children=[
                                                        dcc.Graph(
                                                            id="grafico-piramide",
                                                            config={"displayModeBar": False},
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                        dbc.Col(
                                            md=2,
                                            children=[
                                                html.Div(
                                                    style={
                                                        "height": "100%",
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "gap": "10px",
                                                    },
                                                    children=[
                                                        html.Div(id="kpi-poblacion", style={"flex": "1"}),
                                                        html.Div(id="kpi-mujeres", style={"flex": "1"}),
                                                        html.Div(id="kpi-varones", style={"flex": "1"}),
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                dbc.Row(
                                    className="g-3",
                                    style={"flex": "1 1 auto", "minHeight": "0", "marginBottom": "0"},
                                    children=[
                                        dbc.Col(
                                            md=5,
                                            children=[
                                                html.Div(
                                                    style={**panel_style, "padding": "8px", "height": "100%"},
                                                    children=[
                                                        dcc.Graph(
                                                            id="grafico-nbi",
                                                            config={"displayModeBar": False},
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                        dbc.Col(
                                            md=4,
                                            children=[
                                                html.Div(
                                                    style={**panel_style, "padding": "8px", "height": "100%"},
                                                    children=[
                                                        dcc.Graph(
                                                            id="grafico-viviendas",
                                                            config={"displayModeBar": False},
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                        dbc.Col(
                                            md=3,
                                            children=[
                                                html.Div(
                                                    style={**panel_style, "padding": "12px", "height": "100%"},
                                                    children=[
                                                        panel_titulo("Capas de referencia"),

                                                        dcc.Checklist(
                                                            id="check-capas",
                                                            options=[
                                                                {"label": " Avenidas", "value": "avenidas"},
                                                                {"label": " Calles", "value": "calles"},
                                                                {"label": " Red Agua", "value": "red_agua"},
                                                                {"label": " Red Cloaca", "value": "red_cloaca"},
                                                                {"label": " RENABAP", "value": "renabap"},
                                                                {"label": " Barrios Privados", "value": "barrios_privados"},
                                                                {"label": " Barrios de General Pueyrredon", "value": "barrios_gp"},
                                                            ],
                                                            value=[],
                                                            inputStyle={"marginRight": "8px"},
                                                            labelStyle={
                                                                "display": "block",
                                                                "color": TEXT,
                                                                "marginBottom": "8px",
                                                                "fontSize": "0.86rem",
                                                            },
                                                        ),

                                                        dcc.Dropdown(
                                                            id="dropdown-barrios",
                                                            options=dropdown_barrios_options,
                                                            value=[],
                                                            multi=True,
                                                            placeholder="Seleccionar barrio(s)",
                                                            style={"marginTop": "6px", "marginBottom": "0px"},
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),

                dbc.Col(
                    md=5,
                    style={"height": "100%", "minHeight": "0"},
                    children=[
                        html.Div(
                            style={
                                **panel_style,
                                "padding": "10px 10px 6px 10px",
                                "height": "100%",
                                "display": "flex",
                                "flexDirection": "column",
                                "overflow": "hidden",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "flex-start",
                                        "gap": "8px",
                                        "marginBottom": "8px",
                                        "flex": "0 0 auto",
                                    },
                                    children=[
                                        html.Div([
                                            panel_titulo("Mapa"),
                                            html.Div(
                                                id="texto-seleccion",
                                                style={
                                                    "color": TEXT,
                                                    "fontSize": "0.95rem",
                                                    "fontWeight": "600",
                                                    "marginTop": "-6px",
                                                },
                                            ),
                                        ]),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "alignItems": "flex-end",
                                                "gap": "6px",
                                            },
                                            children=[
                                                html.Div(id="selection-box"),
                                                dbc.Button(
                                                    "Deseleccionar",
                                                    id="btn-limpiar",
                                                    size="sm",
                                                    style={
                                                        "backgroundColor": "#2b1740",
                                                        "border": f"1px solid {PURPLE_4}",
                                                        "color": TEXT,
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "position": "relative",
                                        "flex": "1 1 auto",
                                        "minHeight": "0",
                                        "height": "100%",
                                        "overflow": "hidden",
                                        "borderRadius": "8px",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="mapa-radios",
                                            config={"displayModeBar": False, "scrollZoom": True},
                                            style={
                                                "height": "100%",
                                                "width": "100%",
                                            },
                                        ),
                                        html.Div(id="scale-bar"),
                                        html.Div(id="interval-legend"),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)

# =========================================================
# 8. CALLBACKS
# =========================================================

@app.callback(
    Output("map-view", "data"),
    Input("mapa-radios", "relayoutData"),
    State("map-view", "data"),
    prevent_initial_call=True,
)
def actualizar_vista_mapa(relayout_data, map_view):
    if not relayout_data:
        return map_view

    map_view = map_view or {"zoom": DEFAULT_ZOOM, "center": {"lat": CENTER_LAT, "lon": CENTER_LON}}

    if "mapbox.zoom" in relayout_data:
        map_view["zoom"] = relayout_data["mapbox.zoom"]

    if "mapbox.center" in relayout_data:
        map_view["center"] = relayout_data["mapbox.center"]

    if "mapbox.center.lat" in relayout_data and "mapbox.center.lon" in relayout_data:
        map_view["center"] = {
            "lat": relayout_data["mapbox.center.lat"],
            "lon": relayout_data["mapbox.center.lon"],
        }

    return map_view

@app.callback(
    Output("selected-radios", "data"),
    Input("mapa-radios", "clickData"),
    Input("btn-limpiar", "n_clicks"),
    State("selected-radios", "data"),
    prevent_initial_call=True,
)
def actualizar_seleccion(click_data, n_limpiar, selected_radios):
    ctx = dash.callback_context
    if not ctx.triggered:
        return selected_radios or []

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "btn-limpiar":
        return []

    selected_radios = selected_radios or []

    if trigger == "mapa-radios" and click_data:
        pt = click_data["points"][0]
        if "location" not in pt:
            return selected_radios

        radio_click = int(pt["location"])

        if radio_click in selected_radios:
            selected_radios = [r for r in selected_radios if r != radio_click]
        else:
            selected_radios = selected_radios + [radio_click]

    return selected_radios

@app.callback(
    Output("dropdown-barrios", "value"),
    Input("check-capas", "value"),
    Input("dropdown-barrios", "value"),
    Input("btn-limpiar", "n_clicks"),
    prevent_initial_call=True,
)
def sincronizar_dropdown_barrios(capas_activas, barrios_value, n_limpiar):
    ctx = dash.callback_context
    if not ctx.triggered:
        return barrios_value or []

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    capas_activas = capas_activas or []
    barrios_value = barrios_value or []

    if trigger == "btn-limpiar":
        return []

    if "barrios_gp" not in capas_activas:
        return []

    if trigger == "check-capas":
        return [TODOS_VALUE]

    if trigger == "dropdown-barrios":
        valores = [v for v in barrios_value if v]

        if TODOS_VALUE in valores:
            return [TODOS_VALUE]

        return [v for v in valores if v in barrios_options]

    return barrios_value

@app.callback(
    Output("dropdown-barrios", "disabled"),
    Input("check-capas", "value"),
)
def habilitar_dropdown_barrios(capas_activas):
    capas_activas = capas_activas or []
    return "barrios_gp" not in capas_activas

@app.callback(
    Output("mapa-radios", "figure"),
    Output("grafico-piramide", "figure"),
    Output("grafico-viviendas", "figure"),
    Output("grafico-nbi", "figure"),
    Output("kpi-poblacion", "children"),
    Output("kpi-mujeres", "children"),
    Output("kpi-varones", "children"),
    Output("texto-seleccion", "children"),
    Output("selection-box", "children"),
    Output("interval-legend", "children"),
    Input("selected-radios", "data"),
    Input("map-view", "data"),
    Input("check-capas", "value"),
    Input("dropdown-barrios", "value"),
)
def actualizar_dashboard(selected_radios, map_view, capas_activas, barrios_elegidos):
    selected_radios = selected_radios or []
    capas_activas = capas_activas or []
    barrios_elegidos = barrios_elegidos or []
    map_view = map_view or {"zoom": DEFAULT_ZOOM, "center": {"lat": CENTER_LAT, "lon": CENTER_LON}}

    fig_mapa = figura_mapa(selected_radios, map_view, capas_activas, barrios_elegidos)
    fig_piramide = figura_piramide(selected_radios)
    fig_viviendas = figura_viviendas(selected_radios)
    fig_nbi = figura_nbi(selected_radios)
    tot = calcular_totales(selected_radios)

    texto = tot["etiqueta"] if selected_radios else ETIQUETA_GENERAL

    return (
        fig_mapa,
        fig_piramide,
        fig_viviendas,
        fig_nbi,
        kpi_card("Población total", tot["poblacion"], tot["etiqueta"]),
        kpi_card("Mujeres", tot["mujeres"]),
        kpi_card("Varones", tot["varones"]),
        texto,
        make_selection_box(selected_radios),
        make_interval_legend(),
    )

@app.callback(
    Output("scale-bar", "children"),
    Input("map-view", "data"),
)
def actualizar_escala(map_view):
    style, bar_style, label_style, label = scale_bar_style_and_label(
        map_view or {"zoom": DEFAULT_ZOOM, "center": {"lat": CENTER_LAT, "lon": CENTER_LON}}
    )
    return html.Div(
        style=style,
        children=[
            html.Div(label, style=label_style),
            html.Div(style=bar_style),
        ],
    )

# =========================================================
# 9. RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=False)
import geopandas as gpd
import pandas as pd
from pathlib import Path

# -------------------------
# 1. Leer radios
# -------------------------
radios = gpd.read_file("datos/radios-censales.shp")
radios_gpm = radios[radios["NOMDEPTO"] == "GENERAL PUEYRREDON"].copy()
radios_gpm["link_num"] = pd.to_numeric(radios_gpm["LINK"], errors="coerce")

# -------------------------
# 2. Leer tabla de edades/sexo
# -------------------------
edades = pd.read_csv("datos/edad_sexo_radio_gpm.csv")

edades["radio_id"] = pd.to_numeric(edades["radio_id"], errors="coerce")
edades["personas"] = pd.to_numeric(edades["personas"], errors="coerce")

# -------------------------
# 3. Tabla pirámide
# -------------------------
tabla_piramide = edades.copy()

# -------------------------
# 4. Resumen por radio
# -------------------------
totales_radio = (
    edades.groupby("radio_id", as_index=False)["personas"]
    .sum()
    .rename(columns={"personas": "poblacion_total"})
)

mujeres_radio = (
    edades[edades["sexo_label"] == "Mujer/Femenino"]
    .groupby("radio_id", as_index=False)["personas"]
    .sum()
    .rename(columns={"personas": "mujeres_total"})
)

varones_radio = (
    edades[edades["sexo_label"] == "Varón/Masculino"]
    .groupby("radio_id", as_index=False)["personas"]
    .sum()
    .rename(columns={"personas": "varones_total"})
)

resumen_radio = (
    totales_radio
    .merge(mujeres_radio, on="radio_id", how="left")
    .merge(varones_radio, on="radio_id", how="left")
)

# -------------------------
# 5. Unir con geometrías
# -------------------------
radios_base = radios_gpm.merge(
    resumen_radio,
    left_on="link_num",
    right_on="radio_id",
    how="left"
)

# -------------------------
# 6. Calcular área y densidad
# -------------------------
# Pasamos a una proyección métrica para medir área
radios_metric = radios_base.to_crs(3857).copy()

# m²
radios_base["area_m2"] = radios_metric.geometry.area

# hectáreas
radios_base["area_ha"] = radios_base["area_m2"] / 10_000

# km²
radios_base["area_km2"] = radios_base["area_m2"] / 1_000_000

# densidades
radios_base["dens_hab_ha"] = radios_base["poblacion_total"] / radios_base["area_ha"]
radios_base["dens_hab_km2"] = radios_base["poblacion_total"] / radios_base["area_km2"]

# -------------------------
# 7. Simplificar geometría para acelerar mapa
# -------------------------
radios_simpl = radios_base.to_crs(3857).copy()
radios_simpl["geometry"] = radios_simpl.geometry.simplify(
    tolerance=60,
    preserve_topology=True
)
radios_simpl = radios_simpl.to_crs(4326)

radios_simpl = radios_simpl[
    [
        "LINK",
        "link_num",
        "poblacion_total",
        "mujeres_total",
        "varones_total",
        "area_m2",
        "area_ha",
        "area_km2",
        "dens_hab_ha",
        "dens_hab_km2",
        "geometry",
    ]
].copy()

# -------------------------
# 8. Guardar salidas
# -------------------------
Path("procesados").mkdir(exist_ok=True)

tabla_piramide.to_csv("procesados/tabla_piramide.csv", index=False, encoding="utf-8")
resumen_radio.to_csv("procesados/resumen_radio.csv", index=False, encoding="utf-8")
radios_simpl.to_file("procesados/radios_base.geojson", driver="GeoJSON")

# -------------------------
# 9. Control rápido
# -------------------------
print("Tabla pirámide:", tabla_piramide.shape)
print("Resumen por radio:", resumen_radio.shape)
print("Radios base:", radios_simpl.shape)

print("\nChequeo de unidades:")
print(
    radios_simpl[
        ["LINK", "poblacion_total", "area_m2", "area_ha", "area_km2", "dens_hab_ha", "dens_hab_km2"]
    ].head(10)
)